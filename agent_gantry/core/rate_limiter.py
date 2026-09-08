"""
Rate limiting for tool execution.

Provides rate limiting capabilities to prevent abuse and manage resource consumption.

asyncio synchronisation primitives (``asyncio.Lock`` etc.) bind to the event
loop they are first awaited on. Constructing a single ``asyncio.Lock`` inside
``RateLimiter.__init__`` and reusing it across multiple loops — for example a
gantry built at import time and then driven from a worker-thread loop owned by
``DurableAIAgentWorker`` — produces ``RuntimeError: ... is bound to a
different event loop`` on the second loop. To stay correct in those setups we
keep one ``asyncio.Lock`` per running loop, lazily constructed via
:func:`_lock_for_running_loop`. Each loop sees its own lock and concurrent
counters remain consistent within that loop.
"""

from __future__ import annotations

import asyncio
import time
import weakref
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_gantry.schema.config import RateLimitConfig


class RateLimitExceeded(Exception):  # noqa: N818 - public API name, stable across releases
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RateLimiter:
    """
    Rate limiter for tool execution.

    Supports multiple strategies:
    - Sliding window: Track calls in a time window
    - Token bucket: Refill tokens at a rate
    - Fixed window: Fixed time buckets
    """

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        """
        Initialize the rate limiter.

        Args:
            config: Rate limiting configuration
        """
        self._config = config or RateLimitConfig()

        # Sliding window: deque of timestamps per key
        self._call_history: dict[str, deque[float]] = defaultdict(lambda: deque())

        # Token bucket: tokens and last refill time per key
        self._tokens: dict[str, float] = defaultdict(
            lambda: float(self._config.max_calls_per_minute)
        )
        self._last_refill: dict[str, float] = defaultdict(time.time)

        # Fixed window: call count and window start per key
        self._window_calls: dict[str, int] = defaultdict(int)
        self._window_start: dict[str, float] = defaultdict(time.time)

        # Concurrent execution tracking
        self._concurrent: dict[str, int] = defaultdict(int)
        # One Lock per running event loop — see module docstring. The
        # entries leak at the rate of one lock per distinct loop ever used,
        # which is a bounded constant in practice (typically 1–2 loops per
        # process). Closed loops are pruned lazily on lookup.
        self._loop_locks: dict[
            int, tuple[weakref.ref[asyncio.AbstractEventLoop], asyncio.Lock]
        ] = {}

    def _lock_for_running_loop(self) -> asyncio.Lock:
        """Return a per-running-loop ``asyncio.Lock``.

        Construction happens with a running loop so the lock binds
        correctly. Callers are inside ``async def`` methods, so this is
        guaranteed.
        """
        loop = asyncio.get_running_loop()
        key = id(loop)
        existing = self._loop_locks.get(key)
        if existing is not None:
            existing_loop_ref, existing_lock = existing
            existing_loop = existing_loop_ref()
            if existing_loop is loop:
                return existing_lock
            # id collision after a loop was GC'd — fall through and rebuild.
        lock = asyncio.Lock()
        self._loop_locks[key] = (weakref.ref(loop), lock)
        # Opportunistic cleanup of stale entries (closed loops).
        for stale_key in [k for k, (ref, _l) in self._loop_locks.items() if ref() is None]:
            self._loop_locks.pop(stale_key, None)
        return lock

    def _get_key(self, tool_name: str, namespace: str = "default") -> str:
        """Get rate limit key based on configuration."""
        if self._config.per_namespace:
            return namespace
        elif self._config.per_tool:
            return f"{namespace}.{tool_name}"
        else:
            return "global"

    async def acquire(
        self,
        tool_name: str,
        namespace: str = "default",
    ) -> None:
        """
        Acquire permission to execute a tool.

        Args:
            tool_name: Tool name
            namespace: Tool namespace

        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        if not self._config.enabled:
            return

        key = self._get_key(tool_name, namespace)

        # One critical section covering check -> strategy -> increment. These
        # were previously three separate steps with the lock released in
        # between. No overshoot was actually reachable -- today's strategy
        # checks contain no ``await`` and an uncontended ``asyncio.Lock`` takes
        # a non-yielding fast path, so nothing could interleave -- but the
        # ``max_concurrent`` guarantee rested on that remaining true. Holding
        # one lock makes it structural and costs one lock cycle instead of two.
        #
        # The strategy calls are awaited inside the lock, so a future strategy
        # that performs real I/O would serialize acquires behind it rather than
        # deadlock. That is the deliberate trade: correctness of the cap over
        # throughput. A strategy needing I/O should do it outside this section
        # and pass the result in.
        async with self._lock_for_running_loop():
            # Check concurrent limit
            if self._concurrent[key] >= self._config.max_concurrent:
                raise RateLimitExceeded(
                    f"Concurrent execution limit ({self._config.max_concurrent}) exceeded for {key}",
                    retry_after=1.0,
                )

            # Check rate limit based on strategy. A raise here propagates
            # without incrementing the concurrency counter, as before.
            if self._config.strategy == "sliding_window":
                await self._sliding_window_check(key)
            elif self._config.strategy == "token_bucket":
                await self._token_bucket_check(key)
            elif self._config.strategy == "fixed_window":
                await self._fixed_window_check(key)

            # Increment concurrent counter
            self._concurrent[key] += 1

    def would_exceed(self, tool_name: str, namespace: str = "default") -> str | None:
        """Whether an acquire would be refused right now, changing nothing.

        Read-only by construction — it records no call, consumes no token and
        prunes no history — so it can run *before* the work admission control
        is meant to protect. ``acquire`` remains the authority; this only
        short-circuits a call that is already over quota.

        Returns the reason, or ``None`` when the call would be admitted.
        """
        if not self._config.enabled:
            return None
        key = self._get_key(tool_name, namespace)

        if self._concurrent[key] >= self._config.max_concurrent:
            return (
                f"Concurrent execution limit ({self._config.max_concurrent}) exceeded for {key}"
            )

        if self._config.strategy == "sliding_window":
            history = self._call_history[key]
            now = time.time()
            hour_ago = now - 3600
            minute_ago = now - 60
            # Counted rather than pruned: pruning is a mutation, and this must
            # leave the limiter exactly as it found it.
            # ⚡ Bolt: Fast reverse iteration to count calls in last hour and minute instead of filtering whole history
            in_hour = 0
            in_minute = 0
            for stamp in reversed(history):
                if stamp >= minute_ago:
                    in_minute += 1
                    in_hour += 1
                elif stamp >= hour_ago:
                    in_hour += 1
                else:
                    break

            if in_hour >= self._config.max_calls_per_hour:
                return (
                    f"Rate limit exceeded: {in_hour}/"
                    f"{self._config.max_calls_per_hour} calls per hour"
                )
            if in_minute >= self._config.max_calls_per_minute:
                return (
                    f"Rate limit exceeded: {in_minute}/"
                    f"{self._config.max_calls_per_minute} calls per minute"
                )
            return None

        if self._config.strategy == "token_bucket":
            now = time.time()
            refill_rate = self._config.max_calls_per_minute / 60
            max_tokens = self._config.burst_size or self._config.max_calls_per_minute
            # The refill is *computed*, not stored: writing ``_tokens`` and
            # ``_last_refill`` here would make a peek indistinguishable from
            # an acquire for the next caller.
            available = min(
                max_tokens,
                self._tokens[key] + (now - self._last_refill[key]) * refill_rate,
            )
            if available < 1:
                return (
                    f"Rate limit exceeded: no tokens available "
                    f"(refills at {refill_rate:.2f}/s)"
                )
            return None

        if self._config.strategy == "fixed_window":
            now = time.time()
            if now - self._window_start[key] >= 60:
                return None  # the window is due to reset, so nothing is spent
            if self._window_calls[key] >= self._config.max_calls_per_minute:
                return (
                    f"Rate limit exceeded: {self._window_calls[key]}/"
                    f"{self._config.max_calls_per_minute} calls in window"
                )
        return None

    async def release(
        self,
        tool_name: str,
        namespace: str = "default",
    ) -> None:
        """
        Release a tool execution slot.

        Args:
            tool_name: Tool name
            namespace: Tool namespace
        """
        if not self._config.enabled:
            return

        key = self._get_key(tool_name, namespace)

        # Decrement concurrent counter
        async with self._lock_for_running_loop():
            self._concurrent[key] = max(0, self._concurrent[key] - 1)

    async def _sliding_window_check(self, key: str) -> None:
        """Check sliding window rate limit."""
        now = time.time()
        history = self._call_history[key]

        # Clean entries older than 1 hour (the max window we care about)
        hour_ago = now - 3600
        while history and history[0] < hour_ago:
            history.popleft()

        # Check hour limit first (uses full history)
        if len(history) >= self._config.max_calls_per_hour:
            retry_after = 3600 - (now - history[0])
            raise RateLimitExceeded(
                f"Rate limit exceeded: {len(history)}/{self._config.max_calls_per_hour} calls per hour",
                retry_after=retry_after,
            )

        # Check minute limit (count only entries within the last 60 seconds)
        minute_ago = now - 60

        # ⚡ Bolt: Iterate in reverse and break early instead of using generator over entire history
        recent_count = 0
        for t in reversed(history):
            if t >= minute_ago:
                recent_count += 1
            else:
                break

        if recent_count >= self._config.max_calls_per_minute:
            # Find the oldest entry within the minute window for retry_after
            for t in history:
                if t >= minute_ago:
                    retry_after = 60 - (now - t)
                    break
            else:
                retry_after = 1.0
            raise RateLimitExceeded(
                f"Rate limit exceeded: {recent_count}/{self._config.max_calls_per_minute} calls per minute",
                retry_after=retry_after,
            )

        # Record this call
        history.append(now)

    async def _token_bucket_check(self, key: str) -> None:
        """Check token bucket rate limit."""
        now = time.time()

        # Refill tokens based on elapsed time
        elapsed = now - self._last_refill[key]
        refill_rate = self._config.max_calls_per_minute / 60  # tokens per second
        new_tokens = elapsed * refill_rate
        max_tokens = self._config.burst_size or self._config.max_calls_per_minute

        self._tokens[key] = min(max_tokens, self._tokens[key] + new_tokens)
        self._last_refill[key] = now

        # Check if we have tokens
        if self._tokens[key] < 1:
            retry_after = 1 / refill_rate
            raise RateLimitExceeded(
                f"Rate limit exceeded: no tokens available (refills at {refill_rate:.2f}/s)",
                retry_after=retry_after,
            )

        # Consume a token
        self._tokens[key] -= 1

    async def _fixed_window_check(self, key: str) -> None:
        """Check fixed window rate limit."""
        now = time.time()

        # Check if we need to reset the window (every minute)
        if now - self._window_start[key] >= 60:
            self._window_calls[key] = 0
            self._window_start[key] = now

        # Check limit
        if self._window_calls[key] >= self._config.max_calls_per_minute:
            window_end = self._window_start[key] + 60
            retry_after = window_end - now
            raise RateLimitExceeded(
                f"Rate limit exceeded: {self._window_calls[key]}/{self._config.max_calls_per_minute} calls in window",
                retry_after=retry_after,
            )

        # Increment counter
        self._window_calls[key] += 1

    def get_stats(self, tool_name: str | None = None, namespace: str = "default") -> dict[str, Any]:
        """
        Get rate limiting statistics.

        Args:
            tool_name: Optional tool name to get stats for
            namespace: Tool namespace

        Returns:
            Dictionary of statistics
        """
        if tool_name:
            key = self._get_key(tool_name, namespace)

            # ⚡ Bolt: Fast reverse iteration to count calls in last minute instead of filtering whole history
            calls_last_minute = 0
            now = time.time()
            for t in reversed(self._call_history.get(key, [])):
                if now - t < 60:
                    calls_last_minute += 1
                else:
                    break

            return {
                "key": key,
                "concurrent": self._concurrent.get(key, 0),
                "calls_last_minute": calls_last_minute,
                "calls_last_hour": len(self._call_history.get(key, [])),
                "tokens": self._tokens.get(key, 0)
                if self._config.strategy == "token_bucket"
                else None,
            }
        else:
            # Global stats
            return {
                "total_keys": len(self._call_history),
                "total_concurrent": sum(self._concurrent.values()),
                "config": {
                    "strategy": self._config.strategy,
                    "max_calls_per_minute": self._config.max_calls_per_minute,
                    "max_calls_per_hour": self._config.max_calls_per_hour,
                    "max_concurrent": self._config.max_concurrent,
                },
            }

    async def reset(self, tool_name: str | None = None, namespace: str = "default") -> None:
        """
        Reset rate limit counters.

        Args:
            tool_name: Optional tool name to reset (resets all if None)
            namespace: Tool namespace
        """
        if tool_name:
            key = self._get_key(tool_name, namespace)
            self._call_history[key].clear()
            self._tokens[key] = float(self._config.max_calls_per_minute)
            self._last_refill[key] = time.time()
            self._window_calls[key] = 0
            self._window_start[key] = time.time()
            async with self._lock_for_running_loop():
                self._concurrent[key] = 0
        else:
            self._call_history.clear()
            self._tokens.clear()
            self._last_refill.clear()
            self._window_calls.clear()
            self._window_start.clear()
            async with self._lock_for_running_loop():
                self._concurrent.clear()

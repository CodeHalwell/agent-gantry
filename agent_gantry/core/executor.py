"""
Execution engine for Agent-Gantry.

Handles tool execution with retries, timeouts, circuit breakers, and health tracking.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from agent_gantry.schema.base import (
    RECONSTRUCTED_STRING_FORMATS,
    describe_path,
    json_identity_key,
    schema_declares_null,
)
from agent_gantry.schema.base import (
    check_json_constraints as _check_constraints,
)
from agent_gantry.schema.execution import (
    BatchToolCall,
    BatchToolResult,
    ExecutionStatus,
    ToolCall,
    ToolResult,
)
from agent_gantry.schema.introspection import build_argument_coercers

if TYPE_CHECKING:
    from agent_gantry.core.rate_limiter import RateLimiter
    from agent_gantry.core.registry import ToolRegistry
    from agent_gantry.core.security import SecurityPolicy
    from agent_gantry.observability.telemetry import TelemetryAdapter
    from agent_gantry.schema.tool import ToolDefinition

logger = logging.getLogger(__name__)

#: Assertions a tool's *root* schema can carry that ``_validate_arguments``'
#: own top-level walk does not implement — it reads ``required``,
#: ``properties``, ``patternProperties`` and ``additionalProperties`` and
#: nothing else. Each is applied by ``_validate_value``, which is handed a
#: schema of just these keys so ownership of the structural keywords stays
#: with the walk. Keep this to keywords ``_validate_value`` actually
#: implements: one it ignores would be listed as enforced without being so.
#: Keywords this validator actually evaluates — ``_validate_value``'s own,
#: plus the constraint family ``check_json_constraints`` implements.
_EVALUATED_KEYWORDS = frozenset(
    {
        # structure
        "type",
        "properties",
        "required",
        "additionalProperties",
        "patternProperties",
        "items",
        "prefixItems",
        # combinators
        "anyOf",
        "oneOf",
        "allOf",
        "not",
        # value equality
        "enum",
        "const",
        # check_json_constraints
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
    }
)

#: Keywords that carry no constraint, so their presence says nothing about
#: whether a schema can be evaluated. ``$defs``/``definitions`` belong here:
#: they only hold subschemas for a ``$ref`` to name, and a ``$ref`` is itself
#: unevaluated, so it is the reference that makes a schema indeterminate.
_INERT_KEYWORDS = frozenset(
    {
        "$comment",
        "$defs",
        "$id",
        "$schema",
        "default",
        "definitions",
        "deprecated",
        "description",
        "examples",
        "readOnly",
        "title",
        "writeOnly",
    }
)

#: Recursion bound for :func:`_fully_evaluable`, so a self-referential
#: imported schema cannot spin. Past it the schema is called indeterminate,
#: which is the fail-open answer.
_MAX_EVALUABLE_DEPTH = 16

#: Where a subschema lives, by the shape of the value holding it.
_EVALUABLE_MAPPINGS = ("properties", "patternProperties")
_EVALUABLE_LISTS = ("anyOf", "oneOf", "allOf", "prefixItems")
_EVALUABLE_SINGLES = ("items", "additionalProperties", "not")


def _applies_at_runtime(key: str, value: Any) -> bool:
    """Whether a constraint keyword will actually bind, not merely be known.

    Two of the keywords this validator implements decline to act on some
    values, both fail-open by design: ``pattern`` skips a regex Python's
    ``re`` cannot compile, because a legal ECMA-262 one like ``\\p{L}`` would
    otherwise fail every call; and ``format`` is enforced only for the string
    formats Gantry reconstructs, ``format`` being an annotation by default in
    JSON Schema. Under ``not`` that silence reads as "matched", so
    ``{"type": "string", "not": {"pattern": "\\p{L}"}}`` rejected ``"123"``
    as well as ``"abc"`` (PR #386 review).

    So evaluability asks whether the constraint *binds*, not whether the
    keyword is spelled correctly. Third instance of the same inversion, after
    ``$ref`` and the unevaluable ``oneOf`` branch, and the general form of it.
    """
    if key == "pattern":
        # ``check_json_constraints`` skips a non-string or empty pattern too.
        if not isinstance(value, str) or not value:
            return False
        try:
            re.compile(value)
        except re.error:
            return False
        return True
    if key == "patternProperties":
        if not isinstance(value, dict):
            return False
        for regex in value:
            if not isinstance(regex, str):
                return False
            try:
                re.compile(regex)
            except re.error:
                return False
        return True
    if key == "format":
        return isinstance(value, str) and value in RECONSTRUCTED_STRING_FORMATS
    return True


def _fully_evaluable(schema: Any, _depth: int = 0) -> bool:
    """Whether every constraint in ``schema`` is one this validator applies.

    ``_validate_value`` returns *valid* for anything it cannot interpret —
    "don't reject what we don't understand", which is right everywhere it is
    read as an assertion about the value. Under ``not`` that reading inverts:
    a branch reported as matching forbids the value, so a schema the validator
    merely fails to understand rejects **everything**. ``not: {"$ref":
    "#/$defs/forbidden"}`` — legal, and reachable because imported (MCP,
    OpenAPI) schemas are stored as given — turned its tool into one no call
    could satisfy (PR #386 review).

    So ``not`` asks this first and declines to forbid anything when the answer
    is no. Conservative in the safe direction: an unevaluated ``not`` is the
    behaviour that shipped before it was implemented at all, whereas a
    wrongly-enforced one is a tool nobody can call.

    Recursive, because a reference one level down does the same damage: the
    outer structure evaluates, the nested ``$ref`` reports valid, and the
    branch again matches more values than it should.
    """
    if isinstance(schema, bool):
        return True
    if not isinstance(schema, dict):
        return False
    if _depth > _MAX_EVALUABLE_DEPTH:
        return False
    for key, value in schema.items():
        if key in _INERT_KEYWORDS:
            continue
        if key not in _EVALUATED_KEYWORDS:
            return False
        if not _applies_at_runtime(key, value):
            return False
        if key in _EVALUABLE_MAPPINGS:
            if isinstance(value, dict) and not all(
                _fully_evaluable(sub, _depth + 1) for sub in value.values()
            ):
                return False
        elif key in _EVALUABLE_LISTS:
            if isinstance(value, list) and not all(
                _fully_evaluable(sub, _depth + 1) for sub in value
            ):
                return False
        elif key in _EVALUABLE_SINGLES and not _fully_evaluable(value, _depth + 1):
            return False
    return True



def _branch_declared_names(node: Any, _depth: int = 0) -> set[str]:
    """Property names a schema's combinator branches declare.

    ``additionalProperties`` does not look inside ``allOf`` and friends — a
    JSON Schema rule that bites here because Gantry closes an object whose
    ``additionalProperties`` is absent, stricter than the spec's open default.
    With the branches now evaluated, a schema whose root declares ``a`` and
    whose ``allOf`` branch declares ``b`` checked ``b`` against that branch and
    *then* reported it unknown, because only the root's own ``properties``
    counted as declared. Validating a key and rejecting it as undeclared in the
    same pass is incoherent whichever default is right (PR #386 review).

    ``not`` is deliberately not walked: it names the shape a value must *not*
    take, so its properties are the opposite of declared.
    """
    if not isinstance(node, dict) or _depth > _MAX_EVALUABLE_DEPTH:
        return set()
    names: set[str] = set()
    for key in ("allOf", "anyOf", "oneOf"):
        branches = node.get(key)
        if not isinstance(branches, list):
            continue
        for branch in branches:
            if not isinstance(branch, dict):
                continue
            props = branch.get("properties")
            if isinstance(props, dict):
                names.update(name for name in props if isinstance(name, str))
            names |= _branch_declared_names(branch, _depth + 1)
    return names


_ROOT_ASSERTIONS = (
    "allOf",
    "anyOf",
    "oneOf",
    "not",
    "const",
    "enum",
    "minProperties",
    "maxProperties",
)


class ArgumentReconstructionError(ValueError):
    """An argument could not be rebuilt into the type its handler declares.

    Terminal rather than advisory. The coercer exists *because* the handler is
    annotated with a type the JSON form isn't — so a handler annotated
    ``Payload`` is not "happy with the raw mapping": handing it a ``dict``
    trades a clear rejection for an ``AttributeError`` deep inside the tool, or
    for silently wrong behaviour. Reachable for invariants JSON Schema cannot
    express in the first place (a Pydantic ``field_validator``, a mapping key
    that doesn't parse), which is exactly where schema validation cannot have
    ruled the value out first.
    """


#: Memoized per-handler argument coercers (see ``_coercers_for``).
#: An ``OrderedDict`` rather than a plain one so the bound below can *evict*
#: rather than stop caching: a hard cap alone means that once N distinct
#: handlers have been seen, every handler after them re-runs the (not cheap)
#: signature inspection on every call, forever, while the first N are pinned
#: for the life of the process. A long-running gantry with MCP tool churn is
#: exactly that shape. ``functools.lru_cache`` would do this, but it cannot
#: drop one entry, and ``forget_handler`` needs to when a tool is deleted.
_COERCER_CACHE: OrderedDict[Any, dict[str, Any]] = OrderedDict()

#: How many handlers' coercers to keep. Least-recently-used beyond this.
_COERCER_CACHE_MAX = 512


def _coercers_for(handler: Callable[..., Any]) -> dict[str, Any]:
    """Cached ``build_argument_coercers`` for one handler.

    Signature inspection is not cheap and the answer is fixed for the life of
    the callable, so it is memoized. Unhashable handlers fall back to building
    it per call rather than failing.
    """
    try:
        coercers = _COERCER_CACHE[handler]
    except TypeError:  # unhashable handler
        return build_argument_coercers(handler)
    except KeyError:
        pass
    else:
        _COERCER_CACHE.move_to_end(handler)
        return coercers
    coercers = build_argument_coercers(handler)
    _COERCER_CACHE[handler] = coercers
    if len(_COERCER_CACHE) > _COERCER_CACHE_MAX:
        _COERCER_CACHE.popitem(last=False)
    return coercers


def forget_handler(handler: Callable[..., Any]) -> None:
    """Drop a handler's memoized coercers (called when its tool is deleted)."""
    try:
        _COERCER_CACHE.pop(handler, None)
    except TypeError:  # unhashable handler was never cached
        pass


def _reconstructed(handler: Callable[..., Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Rebuild JSON values into the Python types the handler's signature names.

    Arguments arrive JSON-decoded: a mapping for a Pydantic-model or dataclass
    parameter, an array for a ``set``/``tuple``, a string for a
    ``datetime``/``UUID``/``Enum``. The schema now *advertises* those types, so
    a provider sends exactly that shape — and dispatching it unchanged handed
    ``def f(p: Payload): return p.x`` a ``dict``, failing with "'dict' object
    has no attribute 'x'" on every schema-valid call.

    Only parameters whose declared type genuinely differs from its JSON form
    are touched (see :func:`build_argument_coercers`), so a handler taking
    scalars, lists or dicts receives byte-for-byte what it received before.

    A conversion failure raises :class:`ArgumentReconstructionError` rather
    than passing the original value through. Falling back looked conservative
    — validation had already run against the canonical schema — but the
    coercer exists precisely because the handler declares a type the JSON form
    isn't, so the raw value is the one thing the handler cannot take.
    """
    if not arguments:
        return arguments
    coercers = _coercers_for(handler)
    if not coercers:
        return arguments
    out: dict[str, Any] = {}
    changed = False
    for name, value in arguments.items():
        adapter = coercers.get(name)
        if adapter is None:
            out[name] = value
            continue
        # ``None`` used to short-circuit here, on the assumption that a null
        # is never a value worth rebuilding. It can be: an ``Enum`` with a
        # ``None``-valued member (``class Mode(Enum): UNSET = None``) emits
        # ``enum: [null]``, and a required call supplying null reached the
        # handler as raw ``None`` rather than ``Mode.UNSET``. The adapter
        # answers this correctly without a special case — an annotation that
        # admits ``None`` returns it unchanged, and one that doesn't was
        # already a mismatch between the schema and the handler's own type.
        try:
            rebuilt = adapter.validate_python(value)
        except Exception as exc:  # noqa: BLE001 - reported as a validation error
            raise ArgumentReconstructionError(
                f"Parameter '{name}' could not be rebuilt into the type its "
                f"handler declares: {exc}"
            ) from exc
        changed = changed or rebuilt is not value
        out[name] = rebuilt
    return out if changed else arguments


class ExecutionEngine:
    """
    Execution engine for tool calls.

    Handles:
    - Permission checks (policy + capabilities)
    - Argument validation
    - Circuit breaker logic
    - Retries, back-off, and timeouts
    - Health metric updates
    - Telemetry emission
    """

    def __init__(
        self,
        registry: ToolRegistry,
        default_timeout_ms: int = 30000,
        max_retries: int = 3,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout_s: int = 60,
        security_policy: SecurityPolicy | None = None,
        telemetry: TelemetryAdapter | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        """
        Initialize the execution engine.

        Args:
            registry: Tool registry for looking up handlers
            default_timeout_ms: Default timeout for tool execution
            max_retries: Maximum number of retry attempts
            circuit_breaker_threshold: Failures before opening circuit
            circuit_breaker_timeout_s: Seconds before attempting recovery
            security_policy: Security policy for permission checks
            telemetry: Telemetry adapter for observability
            rate_limiter: Rate limiter for controlling execution throughput
        """
        self._registry = registry
        self._default_timeout = default_timeout_ms
        self._max_retries = max_retries
        self._cb_threshold = circuit_breaker_threshold
        self._cb_timeout = circuit_breaker_timeout_s
        self._security_policy = security_policy
        self._telemetry = telemetry
        self._rate_limiter = rate_limiter
        self._a2a_executor: Any = None

    def _get_a2a_executor(self) -> Any:
        """Return the shared A2A executor, constructing it on first use."""
        if self._a2a_executor is None:
            from agent_gantry.adapters.executors.a2a_executor import A2AExecutor

            self._a2a_executor = A2AExecutor()
        return self._a2a_executor

    async def close(self) -> None:
        """Release resources held by the engine (A2A clients, etc.)."""
        if self._a2a_executor is not None:
            await self._a2a_executor.close()
            self._a2a_executor = None

    async def execute(self, call: ToolCall) -> ToolResult:
        """
        Execute a tool call.

        Args:
            call: The tool call to execute

        Returns:
            Result of the execution
        """
        trace_id = call.trace_id or self._generate_trace_id()
        span_id = self._generate_span_id()
        queued_at = datetime.now(timezone.utc)

        # Resolve the tool. Selection is namespace-aware, so execution must be
        # too: with two MCP servers exposing a same-named tool, a bare-name
        # lookup prefers ``default.<name>`` and would run a different tool than
        # the one that was selected. An explicit ``call.namespace`` (set by
        # every framework adapter) or a qualified ``tool_name`` pins it; a bare
        # name still resolves as before, since a provider tool-call payload
        # cannot express more than the name the model saw.
        tool = self._resolve_tool(call)
        if not tool:
            return ToolResult(
                tool_name=call.tool_name,
                status=ExecutionStatus.FAILURE,
                error=f"Tool '{call.tool_name}' not found",
                error_type="ToolNotFound",
                queued_at=queued_at,
                completed_at=datetime.now(timezone.utc),
                trace_id=trace_id,
                span_id=span_id,
            )

        # Normalize away explicit ``None``s for optional parameters before any
        # policy/validation sees the arguments. Strict-mode provider schemas
        # (see ``strict_json_schema``) widen optional parameters to admit
        # ``null`` — the model then legitimately sends ``null`` for "not
        # provided" — and several frameworks materialize every optional field
        # as ``None``. Dropping them lets the tool's own default apply; a
        # ``None`` for a *required* parameter is kept so the error stays clear.
        normalized = self._normalize_arguments(tool, call.arguments)
        if normalized is not call.arguments:
            call = call.model_copy(update={"arguments": normalized})

        # Check circuit breaker
        cb_result = await self._check_circuit_breaker(tool, call, queued_at, trace_id, span_id)
        if cb_result:
            return cb_result

        # A call that will come back PENDING_CONFIRMATION never reaches the
        # handler, so it must not consume rate-limit budget: ``acquire``
        # records the call in the limiter's window and ``release`` only frees
        # the concurrency counter, so counting the probe *and* the approved
        # replay that follows would charge one logical call twice — the same
        # double-count the SecurityPolicy pattern gate had, via the tool-flag
        # path and the real RateLimiter.
        #
        # Arguments are validated first so that decision is accurate: a call
        # whose arguments don't match the schema is terminal even on a
        # confirmation-gated tool — it returns a ValidationError, never a
        # pending prompt — so it must not take the probe exemption, or
        # malformed calls would be unlimited. The *result* is still returned
        # below, after the policy check, so a denial still outranks a
        # validation error exactly as before.
        # ...but a caller *already* over quota must not buy that validation
        # with a rejected request. The recursive validator walks the whole
        # payload and runs ``re.search`` against schema-supplied ``pattern``
        # and ``patternProperties`` on caller-controlled input, so putting it
        # in front of every admission check handed an over-quota caller
        # unlimited CPU — the limits stopped protecting the work they exist to
        # protect. This peek is read-only: it records nothing, consumes no
        # token and prunes no window, so the accounting below is untouched and
        # ``acquire``/``check_permission`` remain the authority. It only
        # short-circuits a call that is over quota *before* validation runs.
        #
        # ``_needs_confirmation`` is settled first, because an exhausted quota
        # means two different things depending on it — but it changes the
        # *answer*, never whether the peek runs. Skipping the ``RateLimiter``
        # half outright for a gated tool put the validator back in front of
        # the quota for exactly the calls an attacker controls: malformed
        # arguments make ``pending_confirmation`` false, so the real limiter
        # refuses the call *after* the recursive validator has already run its
        # ``re.search`` over the payload. Peeking and answering differently
        # keeps both properties.
        #
        # The policy window is a genuine refusal for a gated call:
        # ``check_permission`` runs every denial check for a pending one too,
        # deferring only the *recording*, so a probe that would be denied says
        # so before a human is asked to approve it.
        #
        # The ``RateLimiter`` is not. A gated call never reaches ``acquire``
        # (see ``if not pending_confirmation`` below), so its quota cannot
        # refuse the call — and refusing one is wrong twice over: the window
        # may well have room again by the time a human approves, and the probe
        # was never going to be charged. So an over-quota gated call answers
        # with the gate rather than a denial, and still skips validation, which
        # is the work the quota exists to protect. Nothing executes on either
        # path; the approved replay carries no gate, so it meets the limiter
        # head-on and is refused there if the window is still full.
        confirmation_gated = self._needs_confirmation(tool, call)
        admission_denial = self._admission_denial(tool)
        if admission_denial is not None and confirmation_gated and admission_denial[3]:
            # The gate is the answer here — but the policy has to have its say
            # first. ``_admission_denial`` consults ``would_exceed_rate_limit``
            # and nothing else, so it knows about neither ``allowed_domains``
            # nor any check a replacement policy adds; answering straight from
            # the peek asked a human to approve a call policy would refuse on
            # replay, and reported ``pending_confirmation`` where every other
            # path reports ``permission_denied``. Running the real check
            # restores the ordering the rest of ``execute`` already has: a
            # denial outranks a gate (PR #385 review).
            #
            # ``pending_confirmation=True`` keeps the probe off the policy's
            # own window, for the reason it always does — the approved replay
            # is the same logical call and is counted then. A *denial* is
            # terminal and ``check_permission`` charges it regardless, so the
            # flood this exempts stays bounded.
            #
            # ``arguments_valid=True`` because validation is precisely what
            # this path skips: the call is not known to be malformed, and
            # claiming otherwise would charge a terminal call that may yet be
            # approved and succeed.
            #
            # Validation still does not run, which is the point of the peek.
            # The recursive validator drives schema-supplied ``pattern`` and
            # ``patternProperties`` through ``re.search`` over caller-supplied
            # payloads; the policy's own walk uses fixed patterns and is
            # bounded by the payload, so it is not the work the quota is
            # protecting.
            policy_result = await self._check_security_policy(
                tool,
                call,
                queued_at,
                trace_id,
                span_id,
                pending_confirmation=True,
                arguments_valid=True,
            )
            if policy_result is not None:
                return policy_result
            # Records its own telemetry and re-asks ``_needs_confirmation``,
            # which is a pure function of the call and the tool and answered
            # true a few lines up. Should that ever stop holding, falling
            # through returns the quota denial below — the previous spelling
            # cleared ``admission_denial`` instead, which would have sent an
            # over-quota caller into the validator the peek exists to keep
            # them out of.
            gate_result = await self._check_confirmation_required(
                tool, call, queued_at, trace_id, span_id
            )
            if gate_result is not None:
                return gate_result
        if admission_denial is not None:
            denial_reason, denial_status, denial_type, _ = admission_denial
            result = ToolResult(
                tool_name=call.tool_name,
                status=denial_status,
                error=denial_reason,
                error_type=denial_type,
                queued_at=queued_at,
                completed_at=datetime.now(timezone.utc),
                trace_id=trace_id,
                span_id=span_id,
            )
            if self._telemetry:
                await self._telemetry.record_execution(call, result)
            return result

        val_result = await self._validate_call_arguments(
            tool, call, queued_at, trace_id, span_id
        )
        arguments_valid = val_result is None
        pending_confirmation = arguments_valid and self._needs_confirmation(tool, call)

        # Security policy check
        sp_result = await self._check_security_policy(
            tool, call, queued_at, trace_id, span_id, pending_confirmation, arguments_valid
        )
        if sp_result is not None and not (
            sp_result.status is ExecutionStatus.PENDING_CONFIRMATION and not arguments_valid
        ):
            return sp_result
        # A *denial* still outranks a validation error — it must not leak
        # "your arguments are malformed" about a tool the caller may not
        # invoke at all — so it returned above. A pending confirmation is
        # different: it defers rather than refuses, and deferring a call that
        # can never succeed puts a schema violation in front of a human to
        # approve and then fails it anyway. Falling through lets ``val_result``
        # be returned below, which makes "malformed arguments are terminal"
        # true for the policy's ``require_confirmation`` *pattern* gate as
        # well as the tool's own flag. The fall-through is deliberate rather
        # than an early return so the accounting below stays identical to a
        # malformed call against a tool with no confirmation gate at all.

        # Rate limiting check (acquires a concurrency slot on success)
        if not pending_confirmation:
            rl_result = await self._check_rate_limit(tool, call, queued_at, trace_id, span_id)
            if rl_result:
                return rl_result

        # Every path past a successful acquire must release the slot —
        # including validation failures and early returns — or the concurrent
        # counter leaks and eventually rejects all calls for this key.
        try:
            if val_result:
                # Recorded here rather than where it was built: this is the
                # first point at which it is known to be the outcome the
                # caller sees.
                if self._telemetry:
                    await self._telemetry.record_execution(call, val_result)
                return val_result

            # Check confirmation requirement *before* dispatching by any
            # mechanism. This used to sit after the special-source branch
            # below, so an A2A tool was executed remotely without the gate
            # ever being consulted — and once ``pending_confirmation`` also
            # skips the rate limiter above, a caller could set
            # ``require_confirmation=True`` to run A2A calls for free, past
            # both the per-minute and concurrency limits. Gating here makes
            # "pending confirmation means nothing ran" true for every source.
            confirm_result = await self._check_confirmation_required(
                tool, call, queued_at, trace_id, span_id
            )
            if confirm_result:
                return confirm_result

            # Check if this requires special execution (A2A, MCP, etc.)
            from agent_gantry.schema.tool import ToolSource

            if tool.source == ToolSource.A2A_AGENT:
                return await self._get_a2a_executor().execute(tool, call, None)

            # Get handler for Python functions
            handler = self._registry.get_handler(f"{tool.namespace}.{tool.name}")
            if not handler:
                return ToolResult(
                    tool_name=call.tool_name,
                    status=ExecutionStatus.FAILURE,
                    error=f"No handler found for tool '{call.tool_name}'",
                    error_type="HandlerNotFound",
                    queued_at=queued_at,
                    completed_at=datetime.now(timezone.utc),
                    trace_id=trace_id,
                    span_id=span_id,
                )

            return await self._execute_handler_with_retries(
                tool, call, handler, queued_at, trace_id, span_id
            )
        finally:
            if self._rate_limiter and not pending_confirmation:
                # Must mirror the acquire key above — and only when there was
                # an acquire to mirror.
                await self._rate_limiter.release(tool.name, tool.namespace)

    async def execute_batch(self, batch: BatchToolCall) -> BatchToolResult:
        """
        Execute multiple tool calls.

        Args:
            batch: The batch of tool calls

        Returns:
            Results of all executions
        """
        start_time = datetime.now(timezone.utc)
        results: list[ToolResult] = []

        if batch.execution_strategy == "sequential":
            for call in batch.calls:
                result = await self.execute(call)
                results.append(result)
                if batch.fail_fast and result.status != ExecutionStatus.SUCCESS:
                    break
        else:
            # Parallel execution
            tasks = [self.execute(call) for call in batch.calls]
            results = list(await asyncio.gather(*tasks))

        end_time = datetime.now(timezone.utc)
        total_time_ms = (end_time - start_time).total_seconds() * 1000

        successful = 0
        for r in results:
            if r.status == ExecutionStatus.SUCCESS:
                successful += 1
        failed = len(results) - successful

        return BatchToolResult(
            results=results,
            total_time_ms=total_time_ms,
            successful_count=successful,
            failed_count=failed,
        )

    async def _execute_with_timeout(
        self,
        handler: Callable[..., Any],
        arguments: dict[str, Any],
        timeout_ms: int,
    ) -> Any:
        """Execute a handler with a timeout.

        Sync handlers are dispatched via :func:`asyncio.to_thread`, which
        binds correctly to the *running* loop. ``asyncio.get_event_loop()``
        is deprecated in 3.10+ when there is no running loop, and in worker-
        thread contexts (e.g. ``DurableAIAgentWorker``) it can return a
        different loop from the one currently running, surfacing as cross-
        loop runtime errors when the executor is driven from a loop other
        than the one the gantry was constructed on.
        """
        timeout_s = timeout_ms / 1000
        arguments = _reconstructed(handler, arguments)

        if asyncio.iscoroutinefunction(handler):
            return await asyncio.wait_for(handler(**arguments), timeout=timeout_s)
        return await asyncio.wait_for(
            asyncio.to_thread(handler, **arguments),
            timeout=timeout_s,
        )

    async def _execute_handler_with_retries(
        self,
        tool: ToolDefinition,
        call: ToolCall,
        handler: Callable[..., Any],
        queued_at: datetime,
        trace_id: str,
        span_id: str,
    ) -> ToolResult:
        """Execute tool handler with retries."""
        from agent_gantry.core.security import PermissionDeniedError

        max_attempts = (call.retry_count or self._max_retries) + 1
        last_error: str | None = None
        last_error_type: str | None = None

        for attempt in range(1, max_attempts + 1):
            started_at = datetime.now(timezone.utc)
            try:
                result_value = await self._execute_with_timeout(
                    handler,
                    call.arguments,
                    call.timeout_ms or self._default_timeout,
                )
                completed_at = datetime.now(timezone.utc)
                await self._record_success(tool, (completed_at - started_at).total_seconds() * 1000)
                result = ToolResult(
                    tool_name=call.tool_name,
                    status=ExecutionStatus.SUCCESS,
                    result=result_value,
                    queued_at=queued_at,
                    started_at=started_at,
                    completed_at=completed_at,
                    attempt_number=attempt,
                    trace_id=trace_id,
                    span_id=span_id,
                )
                if self._telemetry:
                    await self._telemetry.record_execution(call, result)
                return result
            except ArgumentReconstructionError as e:
                # Deterministic — retrying re-runs the same rejected parse —
                # and reported as a validation failure because that is what it
                # is: the value satisfied the JSON Schema but not the handler's
                # own declared type, an invariant the schema could not express.
                #
                # Health is deliberately *not* recorded, matching the schema
                # validation path above: a rejected argument says nothing about
                # whether the tool works, and the arguments are caller-supplied.
                # Counting them opened the circuit breaker after five malformed
                # calls, so a caller could disable a perfectly healthy tool for
                # everyone — the valid call that followed came back CIRCUIT_OPEN.
                completed_at = datetime.now(timezone.utc)
                result = ToolResult(
                    tool_name=call.tool_name,
                    status=ExecutionStatus.FAILURE,
                    error=str(e),
                    error_type="ValidationError",
                    queued_at=queued_at,
                    started_at=started_at,
                    completed_at=completed_at,
                    attempt_number=attempt,
                    trace_id=trace_id,
                    span_id=span_id,
                )
                if self._telemetry:
                    await self._telemetry.record_execution(call, result)
                return result
            except PermissionDeniedError as e:
                completed_at = datetime.now(timezone.utc)
                await self._record_failure(tool)
                result = ToolResult(
                    tool_name=call.tool_name,
                    # The rate-limit path below already reports this exception as
                    # PERMISSION_DENIED; flattening it to FAILURE here made a
                    # permission failure distinguishable or not depending on
                    # which code path raised it. (Carried from PR #316, which
                    # could not merge cleanly once .jules/sentinel.md moved.)
                    status=ExecutionStatus.PERMISSION_DENIED,
                    error=str(e),
                    error_type="PermissionDeniedError",
                    queued_at=queued_at,
                    completed_at=completed_at,
                    attempt_number=attempt,
                    trace_id=trace_id,
                    span_id=span_id,
                )
                if self._telemetry:
                    await self._telemetry.record_execution(call, result)
                return result
            except asyncio.TimeoutError:
                last_error = "Execution timed out"
                last_error_type = "TimeoutError"
            except Exception as e:
                last_error = str(e)
                last_error_type = type(e).__name__

            if attempt < max_attempts:
                await asyncio.sleep(2**attempt * 0.1)

        completed_at = datetime.now(timezone.utc)
        await self._record_failure(tool)

        status = (
            ExecutionStatus.TIMEOUT
            if last_error_type == "TimeoutError"
            else ExecutionStatus.FAILURE
        )

        result = ToolResult(
            tool_name=call.tool_name,
            status=status,
            error=last_error,
            error_type=last_error_type,
            queued_at=queued_at,
            completed_at=completed_at,
            attempt_number=max_attempts,
            trace_id=trace_id,
            span_id=span_id,
        )
        if self._telemetry:
            await self._telemetry.record_execution(call, result)
        return result

    def _should_attempt_recovery(self, tool: ToolDefinition) -> bool:
        """Check if we should attempt circuit breaker recovery."""
        if not tool.health.last_failure:
            return True
        elapsed = (datetime.now(timezone.utc) - tool.health.last_failure).total_seconds()
        return elapsed >= self._cb_timeout

    async def _check_circuit_breaker(
        self,
        tool: ToolDefinition,
        call: ToolCall,
        queued_at: datetime,
        trace_id: str,
        span_id: str,
    ) -> ToolResult | None:
        """Check if circuit breaker is open."""
        if tool.health.circuit_breaker_open and not self._should_attempt_recovery(tool):
            result = ToolResult(
                tool_name=call.tool_name,
                status=ExecutionStatus.CIRCUIT_OPEN,
                error="Circuit breaker is open due to repeated failures",
                queued_at=queued_at,
                completed_at=datetime.now(timezone.utc),
                trace_id=trace_id,
                span_id=span_id,
            )
            if self._telemetry:
                await self._telemetry.record_execution(call, result)
            return result
        return None

    async def _check_security_policy(
        self,
        tool: ToolDefinition,
        call: ToolCall,
        queued_at: datetime,
        trace_id: str,
        span_id: str,
        pending_confirmation: bool = False,
        arguments_valid: bool = True,
    ) -> ToolResult | None:
        """Check security policy permissions.

        Args:
            pending_confirmation: Whether this call will stop at the
                executor's own confirmation gate — decided by the caller,
                which knows whether the arguments validated, so a malformed
                call is not mistaken for a pending one.
        """
        if self._security_policy:
            from agent_gantry.core.security import (
                ConfirmationRequiredError,
                PermissionDeniedError,
                accepts_confirmation_approved,
                accepts_keyword,
            )

            try:
                # Match policies against the *resolved* tool's bare name. A
                # qualified ``call.tool_name`` ("billing.search") would
                # otherwise be fnmatched as a different string than the same
                # tool reached via ``namespace=``, so one calling convention
                # could slip a pattern the other is caught by.
                # ``require_confirmation=False`` is the caller's explicit
                # "a human approved this call" signal (the same override
                # ``_check_confirmation_required`` honours for the tool-flag
                # gate) — it skips only the confirmation-pattern gate; every
                # denial check (rate limit, allowed domains) still runs.
                # The keywords are passed only when the policy's signature
                # declares them (see ``accepts_keyword``), so duck-typed
                # policies predating either keep working.
                #
                # ``pending_confirmation`` tells the policy this call will
                # stop at the executor's own tool-flag gate, which the policy
                # cannot see: without it, the probe for a tool gated by
                # ``requires_confirmation=True`` (rather than by a
                # ``require_confirmation`` pattern) was recorded against the
                # rate limit even though nothing ran, and the approved replay
                # was then denied for the rest of the window.
                kwargs: dict[str, Any] = {}
                if accepts_confirmation_approved(self._security_policy):
                    kwargs["confirmation_approved"] = call.require_confirmation is False
                if accepts_keyword(self._security_policy, "pending_confirmation"):
                    kwargs["pending_confirmation"] = pending_confirmation
                # ``arguments_valid`` lets the policy see that a call matching
                # one of its ``require_confirmation`` patterns is nonetheless
                # terminal, so it charges the quota rather than deferring to an
                # approved replay that will never come.
                if accepts_keyword(self._security_policy, "arguments_valid"):
                    kwargs["arguments_valid"] = arguments_valid
                self._security_policy.check_permission(tool.name, call.arguments, **kwargs)
            except ConfirmationRequiredError as e:
                result = ToolResult(
                    tool_name=call.tool_name,
                    status=ExecutionStatus.PENDING_CONFIRMATION,
                    # Relay the policy's own reason (it names the matched
                    # tool/pattern) so callers that only surface (status,
                    # error) report why nothing ran, plus the approve/resume
                    # hint (require_confirmation=False clears this gate too).
                    error=(
                        f"{e} It was not run. Re-issue the call with "
                        "require_confirmation=False once approved."
                        if str(e)
                        else "Tool requires human confirmation"
                    ),
                    queued_at=queued_at,
                    completed_at=datetime.now(timezone.utc),
                    trace_id=trace_id,
                    span_id=span_id,
                )
                # Only when this is the outcome the caller will see. When the
                # arguments already failed validation, ``execute`` discards
                # this pending result in favour of the ValidationError, and
                # recording it would report a status that never happened and
                # count one call twice.
                if self._telemetry and arguments_valid:
                    await self._telemetry.record_execution(call, result)
                return result
            except PermissionDeniedError as e:
                result = ToolResult(
                    tool_name=call.tool_name,
                    status=ExecutionStatus.PERMISSION_DENIED,
                    error=str(e),
                    error_type="PermissionDeniedError",
                    queued_at=queued_at,
                    completed_at=datetime.now(timezone.utc),
                    trace_id=trace_id,
                    span_id=span_id,
                )
                if self._telemetry:
                    await self._telemetry.record_execution(call, result)
                return result
        return None

    async def _check_rate_limit(
        self,
        tool: ToolDefinition,
        call: ToolCall,
        queued_at: datetime,
        trace_id: str,
        span_id: str,
    ) -> ToolResult | None:
        """Check rate limit and acquire execution slot."""
        if self._rate_limiter:
            from agent_gantry.core.rate_limiter import RateLimitExceeded

            try:
                # Key off the resolved tool, not the raw call: with per_tool
                # keys, a qualified ``call.tool_name`` produced
                # "billing.billing.search" while the ``namespace=`` form
                # produced "billing.search", so the same tool held two
                # independent budgets and a caller could double its allowance
                # by alternating styles.
                await self._rate_limiter.acquire(tool.name, tool.namespace)
            except RateLimitExceeded as e:
                result = ToolResult(
                    tool_name=call.tool_name,
                    status=ExecutionStatus.FAILURE,
                    error=str(e),
                    error_type="RateLimitExceeded",
                    queued_at=queued_at,
                    completed_at=datetime.now(timezone.utc),
                    trace_id=trace_id,
                    span_id=span_id,
                )
                if self._telemetry:
                    await self._telemetry.record_execution(call, result)
                return result
        return None

    def _admission_denial(
        self, tool: ToolDefinition
    ) -> tuple[str, ExecutionStatus, str, bool] | None:
        """The reason this call is already over quota, or ``None``.

        The fourth element says the ``RateLimiter`` was the one refusing, which
        the caller needs because a confirmation-gated call is not subject to it
        — see ``execute``.

        Consults both limiters read-only, and reports the result the *denying*
        limiter would have produced: the ``SecurityPolicy`` window raises
        ``PermissionDeniedError`` and the ``RateLimiter`` raises
        ``RateLimitExceeded``, so short-circuiting must not relabel one as the
        other — a caller distinguishing them would see the reason change
        depending only on how early the refusal happened.

        Deliberately conservative: it reports only what is *certainly* refused
        now, so a call it admits still goes through the real, recording checks
        unchanged.

        """
        policy = self._security_policy
        if policy is not None:
            check = getattr(policy, "would_exceed_rate_limit", None)
            if callable(check):
                # ``getattr``: a replacement policy predating this method must
                # not break, exactly as the ``check_permission`` keywords are
                # probed rather than assumed.
                reason = check()
                if reason:
                    return (
                        str(reason),
                        ExecutionStatus.PERMISSION_DENIED,
                        "PermissionDeniedError",
                        False,
                    )
        limiter = self._rate_limiter
        if limiter is not None:
            check = getattr(limiter, "would_exceed", None)
            if callable(check):
                reason = check(tool.name, tool.namespace)
                if reason:
                    return (
                        str(reason),
                        ExecutionStatus.FAILURE,
                        "RateLimitExceeded",
                        True,
                    )
        return None

    async def _validate_call_arguments(
        self,
        tool: ToolDefinition,
        call: ToolCall,
        queued_at: datetime,
        trace_id: str,
        span_id: str,
    ) -> ToolResult | None:
        """Validate arguments and return ToolResult if invalid."""
        is_valid, validation_error = await self._validate_arguments(tool, call.arguments)
        if not is_valid:
            result = ToolResult(
                tool_name=call.tool_name,
                status=ExecutionStatus.FAILURE,
                error=validation_error,
                error_type="ValidationError",
                queued_at=queued_at,
                completed_at=datetime.now(timezone.utc),
                trace_id=trace_id,
                span_id=span_id,
            )
            # Telemetry is *not* emitted here. This result is computed before
            # the security policy runs (so the rate-limit exemption decision is
            # accurate) but it does not always win: a denial outranks it, and a
            # rate-limit rejection can be returned instead. Recording here
            # produced two executions for one call, one of them an outcome that
            # was never returned. ``execute`` records whichever result it
            # actually returns.
            return result
        return None

    @staticmethod
    def _needs_confirmation(tool: ToolDefinition, call: ToolCall) -> bool:
        """Whether this call short-circuits to ``PENDING_CONFIRMATION``.

        A pure function of the call and the tool, so ``execute`` can settle it
        before deciding whether to charge the call against the rate limiter.
        ``ToolCall.require_confirmation`` overrides the tool's own flag in
        either direction; ``None`` defers to the tool.
        """
        needs_confirm = call.require_confirmation
        if needs_confirm is None:
            needs_confirm = tool.requires_confirmation
        return bool(needs_confirm)

    async def _check_confirmation_required(
        self,
        tool: ToolDefinition,
        call: ToolCall,
        queued_at: datetime,
        trace_id: str,
        span_id: str,
    ) -> ToolResult | None:
        """Check if tool requires confirmation."""
        if self._needs_confirmation(tool, call):
            result = ToolResult(
                tool_name=call.tool_name,
                status=ExecutionStatus.PENDING_CONFIRMATION,
                # Populate error so callers that only surface (status, error) —
                # e.g. framework adapters raising ToolExecutionError — report
                # why nothing ran instead of "no detail". Approve by re-issuing
                # the call with ``ToolCall(require_confirmation=False)``.
                error=(
                    f"Tool '{tool.name}' requires human confirmation before "
                    "execution; it was not run. Re-issue the call with "
                    "require_confirmation=False once approved."
                ),
                queued_at=queued_at,
                completed_at=datetime.now(timezone.utc),
                trace_id=trace_id,
                span_id=span_id,
            )
            if self._telemetry:
                await self._telemetry.record_execution(call, result)
            return result
        return None

    @staticmethod
    def _normalize_arguments(
        tool: ToolDefinition, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Drop explicit ``None``s for declared *optional* parameters.

        Strict-mode provider schemas widen optional parameters to accept
        ``null`` (``strict_json_schema``), so a model following the schema
        Gantry itself advertised sends ``null`` for "not provided"; several
        frameworks likewise materialize unset optional fields as ``None``.
        Treating those as omitted lets the handler's own default apply.

        ``None`` is preserved — never dropped — in three cases: for a
        *required* parameter and for keys the schema doesn't declare at all
        (so validation errors stay accurate), and for an optional parameter
        whose own schema *explicitly* types ``null`` as one of its allowed
        values (``"type": ["string", "null"]`` or ``"type": "null"``) — there,
        ``null`` is a distinct, meaningful value the schema itself declares,
        not merely strict-mode's "not provided" placeholder, so a caller
        that deliberately sends it must not have it silently discarded.

        Returns the original dict (identity) when nothing needs dropping.
        """
        if not arguments:
            return arguments
        schema = tool.parameters_schema or {}

        _declares_null = schema_declares_null

        def _declaring_subschema(prop: dict[str, Any], key: str) -> dict[str, Any] | None:
            """The subschema that actually carries ``key``, through combinators.

            An optional nested model is ``{"anyOf": [{"type": "object",
            "properties": {...}}, {"type": "null"}]}`` — the shape Pydantic
            emits for ``Payload | None`` — so the node handed down here often
            has no ``properties``/``items`` of its own. Reading only the top
            level left every value under such a property un-normalized, and
            its strict-mode nulls then failed validation.

            Ambiguity yields ``None`` rather than a guess: with two branches
            declaring the key (an ``allOf`` intersection, or a genuine union
            of two object shapes) there is no single ``required`` list to
            decide against, and dropping a key the other branch requires
            would be worse than leaving the value alone.
            """
            def _usable(value: Any) -> bool:
                # ``properties``/``items`` hold a schema; ``prefixItems`` holds
                # a list of them.
                return isinstance(value, (dict, list)) and bool(value)

            if _usable(prop.get(key)):
                return prop
            found: list[dict[str, Any]] = []
            for combinator in ("anyOf", "oneOf", "allOf"):
                branches = prop.get(combinator)
                if not isinstance(branches, list):
                    continue
                for branch in branches:
                    if isinstance(branch, dict) and _usable(branch.get(key)):
                        found.append(branch)
            return found[0] if len(found) == 1 else None

        def _normalize_child(value: Any, prop: Any) -> Any:
            """Recurse into an object/array value against its own subschema."""
            if not isinstance(prop, dict):
                return value
            if isinstance(value, dict):
                target = _declaring_subschema(prop, "properties")
                return _normalize_object(value, target) if target is not None else value
            if isinstance(value, list):
                # ``prefixItems`` types each position independently, and strict
                # mode widens the optional properties of a positional *object*
                # exactly as it does anywhere else — so looking only at
                # ``items`` left ``[{"nickname": null}, 1]`` intact for a
                # ``tuple[Payload, int]`` parameter. ``items`` covers the
                # positions past the prefix, matching how validation applies
                # the two.
                prefix_source = _declaring_subschema(prop, "prefixItems")
                prefix_items = prefix_source["prefixItems"] if prefix_source else []
                tail_source = _declaring_subschema(prop, "items")
                tail_schema = tail_source["items"] if tail_source else None
                if not prefix_items and tail_schema is None:
                    return value
                items: list[Any] = []
                changed = False
                for index, item in enumerate(value):
                    item_schema = (
                        prefix_items[index] if index < len(prefix_items) else tail_schema
                    )
                    normalized_item = _normalize_child(item, item_schema)
                    changed = changed or normalized_item is not item
                    items.append(normalized_item)
                if changed:
                    return items
            return value

        def _normalize_object(value: dict[str, Any], obj_schema: dict[str, Any]) -> Any:
            obj_properties = obj_schema.get("properties") or {}
            obj_required = set(obj_schema.get("required") or [])
            out: dict[str, Any] = {}
            changed = False
            for name, item in value.items():
                prop = obj_properties.get(name)
                if (
                    item is None
                    and name in obj_properties
                    and name not in obj_required
                    and not _declares_null(prop)
                ):
                    changed = True
                    continue
                normalized_item = _normalize_child(item, prop)
                if normalized_item is not item:
                    changed = True
                out[name] = normalized_item
            return out if changed else value

        # Recursive, not just top-level: strict-mode widening applies to every
        # object in the schema, so an optional *nested* property also comes
        # back as an explicit null meaning "not provided". Normalizing only
        # the top level left ``{"payload": {"nickname": null}}`` intact, and
        # validation then rejected it against the canonical non-null schema.
        return _normalize_object(arguments, schema)

    async def _validate_arguments(
        self, tool: ToolDefinition, arguments: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """
        Validate arguments against tool schema.

        Schema-aware where it matters for real provider/framework payloads:
        ``type`` may be a list (strict-mode ``["string", "null"]`` widening),
        ``enum`` membership is enforced, a truthy ``additionalProperties``
        admits undeclared keys (both top-level and nested), and an ``object``
        schema with no declared ``properties`` (e.g. a ``dict`` parameter)
        accepts any keys.

        Args:
            tool: Tool definition with parameter schema
            arguments: Arguments to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        schema = tool.parameters_schema
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        def _permits_additional(value: Any) -> bool:
            """Whether an ``additionalProperties`` value allows extra keys.

            ``True`` permits them unconstrained. A dict schema — including
            the empty schema ``{}``, which per JSON Schema validates every
            value and is thus spec-equivalent to ``true`` — also permits
            them (constrained by that schema when it declares anything; an
            empty one is a harmless no-op via ``_validate_value``'s graceful
            handling of a typeless schema, so it need not be special-cased).
            Plain ``bool(value)`` would wrongly collapse ``{}`` with
            ``False``/absent, since an empty dict is falsy in Python.
            ``False`` and an absent/``None`` key (Gantry's own default for
            schemas it emits itself, stricter than the JSON Schema spec
            default of ``true``) both forbid extras.
            """
            return value is True or isinstance(value, dict)

        allow_additional = _permits_additional(schema.get("additionalProperties"))

        def _matches_type(value: Any, expected_type: str) -> bool:
            if expected_type == "boolean":
                return isinstance(value, bool)
            if expected_type == "integer":
                return isinstance(value, int) and not isinstance(value, bool)
            if expected_type == "number":
                return isinstance(value, (int, float)) and not isinstance(value, bool)
            if expected_type == "string":
                return isinstance(value, str)
            if expected_type == "array":
                return isinstance(value, list)
            if expected_type == "object":
                return isinstance(value, dict)
            if expected_type == "null":
                return value is None
            # Unknown type keyword: don't reject what we don't understand.
            return True

        def _child(path: str, name: str) -> str:
            """The path of a property, unprefixed at the root.

            The object branch validates the tool's own argument object as well
            as nested ones now, and the root has no name to prefix with —
            ``f"{path}.{name}"`` there reported ``'.token'``.
            """
            return f"{path}.{name}" if path else name

        def _check_pattern_properties(
            value: dict[str, Any],
            val_schema: dict[str, Any],
            path: str,
            *,
            partial: bool = False,
        ) -> tuple[bool, set[str], str | None]:
            """Validate ``patternProperties`` over a mapping's keys.

            ``patternProperties`` types keys by regex rather than by name,
            which Pydantic emits for a mapping with constrained keys. Ignoring
            it meant a matching key's value was never checked — and, with
            ``additionalProperties: false``, that every key was rejected
            because none counted as declared.

            Returns ``(ok, matched_keys, error)``. The matched keys are
            *declared* for the purposes of every closed-object check, at the
            top level of a tool schema and inside a nested object alike: this
            lived only in the nested branch, so an identical construct one
            level up both rejected valid keys and skipped their validation.
            """
            patterns = val_schema.get("patternProperties")
            matched: set[str] = set()
            if not isinstance(patterns, dict) or not patterns:
                return True, matched, None
            for regex, subschema in patterns.items():
                if not isinstance(regex, str):
                    continue
                try:
                    compiled = re.compile(regex)
                except re.error as exc:
                    # Same fail-open reasoning as ``pattern``: an ECMA-only
                    # regex must not make every call fail, but the author
                    # needs to know it isn't enforced.
                    logger.warning(
                        "%s declares patternProperties %r that "
                        "Python's re cannot compile (%s); those keys are "
                        "not validated.",
                        describe_path(path),
                        regex,
                        exc,
                    )
                    continue
                for prop_name, prop_value in value.items():
                    if not compiled.search(prop_name):
                        continue
                    matched.add(prop_name)
                    if isinstance(subschema, bool) or (
                        isinstance(subschema, dict) and subschema
                    ):
                        # A ``false`` pattern schema forbids every matching
                        # key. Skipping booleans here accepted them *and*
                        # counted them as declared, so a closed object let them
                        # through as well.
                        is_valid, err = _validate_value(
                            prop_value, subschema, _child(path, prop_name), partial=partial
                        )
                        if not is_valid:
                            return False, matched, err
            return True, matched, None

        def _branch_matches(value: Any, branch: Any, path: str) -> bool:
            """Whether one combinator branch accepts ``value``.

            Handles the boolean schemas draft-06 added — ``true`` validates
            every value, ``false`` none — alongside the empty schema ``{}``,
            which means the same as ``true``.

            Always ``partial``: a branch constrains the value, it does not
            describe it. See ``_validate_value``.
            """
            if branch is True or branch == {}:
                return True
            if branch is False:
                return False
            return _validate_value(value, branch, path, partial=True)[0]

        def _validate_value(
            value: Any, val_schema: Any, path: str, *, partial: bool = False
        ) -> tuple[bool, str | None]:
            """Validate ``value`` against ``val_schema``.

            ``partial`` marks a schema that *constrains* the value rather than
            describing it — a combinator branch and everything beneath one.
            Only the closed-object default turns on it: an absent
            ``additionalProperties`` means "closed" for a tool schema (Gantry
            emits its own that way, stricter than the spec's open default) and
            that reading is a category error inside a branch, which asserts
            only about the keys it names. Reading a root ``allOf`` branch
            declaring ``b`` as a complete description rejected ``{"a": 1,
            "b": 2}`` with ``Unknown parameter: a`` — ``a`` being declared by
            the root schema, or by a sibling branch (PR #386 review).

            An *explicit* ``additionalProperties: false`` is still honoured in
            a branch: that is the schema saying so, not a default being
            inferred for it.
            """
            # A schema may be a bare boolean anywhere a schema is allowed, not
            # only in a combinator branch: ``properties: {"disabled": false}``,
            # ``patternProperties: {"^blocked_": false}``, ``items: false``.
            # Handled here, at the one funnel every subschema passes through,
            # rather than at each call site — reading ``.get`` on ``False`` let
            # an ``AttributeError`` escape ``execute()`` instead of returning a
            # validation failure.
            if val_schema is True:
                return True, None
            if val_schema is False:
                return False, f"{describe_path(path)} is forbidden by its schema"
            if not isinstance(val_schema, dict):
                # Not a schema at all — don't reject what we can't interpret.
                return True, None

            expected_type = val_schema.get("type")

            # A schema can constrain a value purely through combinators, with
            # no ``type`` of its own — ``{"anyOf": [{"type": "integer"},
            # {"type": "null"}]}`` is what Pydantic emits for ``int | None``,
            # including for fields of the nested models this PR now inlines.
            # Reading ``type`` alone would see ``None`` and wave the value
            # through unchecked, so enforce the branches here. Run regardless
            # of whether the schema also declares a ``type``: JSON Schema
            # applies these independently, so ``{"type": "integer", "allOf":
            # [{"minimum": 1}]}`` — a shape merged/imported schemas do use —
            # must honour both assertions, not just the type.
            for key in ("anyOf", "oneOf"):
                branches = val_schema.get(key)
                if not isinstance(branches, list) or not branches:
                    continue
                # ``true`` and ``false`` are schemas in their own right from
                # draft-06 on: ``true`` validates every value, ``false`` none.
                # Filtering to dicts dropped them, so ``{"anyOf": [true,
                # {"type": "integer"}]}`` — semantically "anything" — rejected
                # a string, and a ``false`` branch silently stopped counting
                # against ``oneOf``'s exclusivity.
                usable = [b for b in branches if isinstance(b, (dict, bool))]
                if not usable:
                    continue
                # A branch this validator cannot fully evaluate reports as
                # *matching*, since ``_validate_value`` returns valid for what
                # it cannot interpret. That inflates the count, and ``oneOf``
                # reads the count: the ``oneOf: [{"$ref": …}, {"$ref": …}]``
                # an imported schema commonly uses had every argument object
                # matching both branches and rejected for it, so the tool
                # could not be called at all (PR #386 review).
                #
                # The honest verdict for such a combinator is "unknown", so it
                # is not enforced. Skipping only the unevaluable branches
                # would be worse than leaving them in: it can drive the count
                # to zero and reject a value one of them may well have
                # accepted.
                if not all(_fully_evaluable(b) for b in usable):
                    logger.warning(
                        "%s declares a '%s' with branches this validator does "
                        "not evaluate; it is not enforced.",
                        describe_path(path),
                        key,
                    )
                    continue
                # An empty schema ``{}`` validates every value too, so it is
                # likewise a branch that always matches — excluding it turned
                # ``{"anyOf": [{}, {"type": "integer"}]}`` into an
                # integer-only constraint.
                matches = 0
                for b in usable:
                    if _branch_matches(value, b, path):
                        matches += 1
                if matches == 0:
                    return (
                        False,
                        f"{describe_path(path)} does not match any permitted schema",
                    )
                # ``oneOf`` means *exactly* one, unlike ``anyOf``: with
                # overlapping branches (``number``/``integer``) a value
                # matching both violates the schema.
                if key == "oneOf" and matches > 1:
                    return (
                        False,
                        f"{describe_path(path)} matches {matches} oneOf schemas; "
                        "exactly one must match",
                    )
            # ``not`` is an assertion like any other and applies whatever
            # else the schema declares, so it sits beside the combinators
            # rather than inside a ``type`` branch. It was parsed nowhere:
            # ``_SUBSCHEMA_KEYS`` in the provider transforms walks into it, so
            # the keyword survived a round-trip through them and then bound
            # nothing here — a schema that exists to *forbid* a shape accepted
            # it (PR #385 review).
            if "not" in val_schema:
                forbidden = val_schema["not"]
                # Anything this validator cannot fully evaluate is ignored
                # rather than treated as an always-matching branch: reading it
                # as one would reject every value, turning a schema that
                # merely uses a keyword we skip — a non-schema value, or a
                # ``$ref`` — into a tool nobody can call. Same fail-open
                # reasoning as an uncompilable pattern, and see
                # ``_fully_evaluable``.
                if isinstance(forbidden, (dict, bool)) and not _fully_evaluable(forbidden):
                    logger.warning(
                        "%s declares a 'not' schema using keywords this "
                        "validator does not evaluate (%r); it is not enforced.",
                        describe_path(path),
                        forbidden,
                    )
                elif isinstance(forbidden, (dict, bool)) and _branch_matches(
                    value, forbidden, path
                ):
                    return (
                        False,
                        f"{describe_path(path)} matches a schema it is required not to",
                    )

            allof = val_schema.get("allOf")
            if isinstance(allof, list):
                for branch in allof:
                    if branch is False:
                        # ``false`` validates nothing, so intersecting with it
                        # forbids every value.
                        return (
                            False,
                            f"{describe_path(path)} has an allOf branch that "
                            "permits no value",
                        )
                    if not isinstance(branch, dict) or not branch:
                        continue
                    is_valid, err = _validate_value(value, branch, path, partial=True)
                    if not is_valid:
                        return False, err

            # ``type`` may be a list (e.g. strict-mode ``["string", "null"]``):
            # the value is valid if it matches any listed type.
            if isinstance(expected_type, list):
                if not any(_matches_type(value, t) for t in expected_type):
                    return (
                        False,
                        f"{describe_path(path)} must be one of types {expected_type}",
                    )
                if value is None:
                    # ``null`` needs no structural checks, but ``enum`` and
                    # ``const`` are independent constraints: a schema typed
                    # ``["string", "null"]`` whose enum lists only
                    # ``"a"``/``"b"`` does not admit ``null``, and neither
                    # does one pinned by ``const: "fixed"``. Returning early
                    # here would skip both. (Gantry's own emission can't
                    # produce this shape — strict-mode widening deep-copies
                    # and never touches the canonical schema the executor
                    # validates against — but a hand-authored or
                    # MCP/OpenAPI-imported schema can.)
                    null_enum = val_schema.get("enum")
                    if isinstance(null_enum, list) and null_enum and None not in null_enum:
                        return False, f"{describe_path(path)} must be one of {null_enum}"
                    if "const" in val_schema and val_schema["const"] is not None:
                        return False, f"{describe_path(path)} must be {val_schema['const']!r}"
                    return True, None
                # Continue structural checks against the matching member.
                expected_type = next(
                    (t for t in expected_type if _matches_type(value, t)), None
                )

            if isinstance(expected_type, str) and not _matches_type(value, expected_type):
                article = "an" if expected_type[0] in "aoiue" else "a"
                return False, f"{describe_path(path)} must be {article} {expected_type}"

            # Membership and equality both go through JSON identity rather
            # than Python's ``==``. ``True == 1`` in Python, so a boolean
            # satisfied a numeric ``Literal[1, 1.5]`` — which emits an ``enum``
            # with no single ``type``, leaving this the only constraint — and
            # a tuple-valued member never matched the array a provider
            # actually returns.
            enum_values = val_schema.get("enum")
            if isinstance(enum_values, list) and enum_values:
                value_key = json_identity_key(value)
                if all(value_key != json_identity_key(v) for v in enum_values):
                    return False, f"{describe_path(path)} must be one of {enum_values}"

            # ``const`` is a one-value ``enum``. Pydantic emits it for a
            # single-value ``Literal``, so it appears inside the nested
            # model/TypedDict schemas introspection now inlines — checking
            # only ``enum`` let those values through unvalidated.
            if "const" in val_schema and json_identity_key(value) != json_identity_key(
                val_schema["const"]
            ):
                return False, f"{describe_path(path)} must be {val_schema['const']!r}"

            # Constraint keywords. Pydantic emits these for any constrained
            # field (``Annotated[int, Field(gt=0)]`` becomes ``type: integer``
            # plus ``exclusiveMinimum: 0``), so they arrive inside the nested
            # model and TypedDict schemas introspection now inlines — and
            # ``uniqueItems`` is emitted by Gantry itself for a ``set``
            # parameter. Checking only the type let every one of them through.
            constraint_error = _check_constraints(value, val_schema, path)
            if constraint_error is not None:
                return False, constraint_error

            # JSON Schema applies an array's or an object's keywords whenever
            # the *instance* is of that kind, whether or not the schema also
            # spells out a ``type``. Gating these branches on ``type`` alone
            # meant an imported or merged property such as ``{"properties":
            # {"token": {"type": "string"}}, "required": ["token"]}`` — legal,
            # and what an MCP server or an OpenAPI import produces — governed
            # nothing at all: ``{}`` was dispatched despite the missing key.
            # Both branches are no-ops when the schema declares none of their
            # keywords, so widening the gate cannot reject anything a typed
            # schema wouldn't.
            if expected_type == "array" or (
                expected_type is None and isinstance(value, list)
            ):
                # ``prefixItems`` types each position independently — what
                # Pydantic emits for a heterogeneous ``tuple[int, str]``, so
                # it arrives inside the nested models introspection now
                # inlines. ``items`` (if also present) covers the positions
                # past the prefix.
                prefix_items = val_schema.get("prefixItems")
                prefix_len = 0
                if isinstance(prefix_items, list) and prefix_items:
                    prefix_len = len(prefix_items)
                    for i, entry in enumerate(prefix_items):
                        if i >= len(value):
                            continue
                        if not isinstance(entry, bool) and (
                            not isinstance(entry, dict) or not entry
                        ):
                            continue
                        is_valid, err = _validate_value(
                            value[i], entry, f"{path}[{i}]", partial=partial
                        )
                        if not is_valid:
                            return False, err
                item_schema = val_schema.get("items")
                if isinstance(item_schema, bool) or (
                    isinstance(item_schema, dict) and item_schema
                ):
                    # ``items: false`` beside ``prefixItems`` is the standard
                    # way to forbid positions past the prefix — a fixed-length
                    # tuple. Skipping booleans accepted the extra elements the
                    # schema exists to refuse.
                    for i, item in enumerate(value):
                        if i < prefix_len:
                            continue
                        is_valid, err = _validate_value(
                            item, item_schema, f"{path}[{i}]", partial=partial
                        )
                        if not is_valid:
                            return False, err
            elif expected_type == "object" or (
                expected_type is None and isinstance(value, dict)
            ):
                obj_properties = val_schema.get("properties")
                obj_additional = val_schema.get("additionalProperties")

                matched_ok, pattern_matched, pattern_err = _check_pattern_properties(
                    value, val_schema, path, partial=partial
                )
                if not matched_ok:
                    return False, pattern_err

                obj_required_early = val_schema.get("required")
                if isinstance(obj_required_early, list) and obj_required_early:
                    # Checked before the no-properties shortcut below: a
                    # ``required`` name needs no matching ``properties`` entry,
                    # so ``{"type": "object", "properties": {}, "required":
                    # ["token"]}`` — valid, and what a merged or imported
                    # schema produces — was accepting ``{}`` outright.
                    for req_prop in obj_required_early:
                        if isinstance(req_prop, str) and req_prop not in value:
                            return False, f"Missing required parameter: {_child(path, req_prop)}"

                if not isinstance(obj_properties, dict) or not obj_properties:
                    # No declared properties. Three distinct schema intents
                    # share this shape and must not be conflated:
                    if obj_additional is False:
                        # Explicitly closed (``additionalProperties: false``
                        # with no declared properties) → no keys at all are
                        # permitted — a "no-argument object" schema, not a
                        # free-form one. A key a ``patternProperties`` entry
                        # matched *is* declared, so it survives the closure.
                        undeclared = [k for k in value if k not in pattern_matched]
                        if undeclared:
                            return False, f"{describe_path(path)} does not permit any properties"
                    elif isinstance(obj_additional, dict):
                        # A schema-valued additionalProperties (e.g. the
                        # ``dict[str, int]`` schemas introspection emits)
                        # constrains every value; an empty schema ``{}`` is a
                        # harmless no-op here (see ``_permits_additional``).
                        for prop_name, prop_value in value.items():
                            if prop_name in pattern_matched:
                                continue  # already checked against its pattern
                            is_valid, err = _validate_value(
                                prop_value,
                                obj_additional,
                                _child(path, prop_name),
                                partial=partial,
                            )
                            if not is_valid:
                                return False, err
                    # else: additionalProperties absent or True → free-form
                    # (a plain ``dict`` parameter) — any keys are acceptable.
                    return True, None
                obj_required = val_schema.get("required", [])

                for req_prop in obj_required:
                    if req_prop not in value:
                        return False, f"Missing required parameter: {_child(path, req_prop)}"

                branch_declared = _branch_declared_names(val_schema)
                for prop_name, prop_value in value.items():
                    if prop_name not in obj_properties:
                        if prop_name in pattern_matched:
                            continue  # already checked against its pattern
                        if prop_name in branch_declared:
                            continue  # declared by a combinator branch
                        if _permits_additional(obj_additional) or (
                            partial and obj_additional is None
                        ):
                            if isinstance(obj_additional, dict):
                                is_valid, err = _validate_value(
                                    prop_value,
                                    obj_additional,
                                    _child(path, prop_name),
                                    partial=partial,
                                )
                                if not is_valid:
                                    return False, err
                            continue
                        return False, f"Unknown parameter: {_child(path, prop_name)}"

                    is_valid, err = _validate_value(
                        prop_value,
                        obj_properties[prop_name],
                        _child(path, prop_name),
                        partial=partial,
                    )
                    if not is_valid:
                        return False, err

            return True, None

        # Check top-level required parameters
        for param in required:
            if param not in arguments:
                return False, f"Missing required parameter: {param}"

        # The walk below reads only ``required``, ``properties``,
        # ``patternProperties`` and ``additionalProperties`` off the root, so
        # every other assertion JSON Schema applies to the argument object as
        # a whole bound nothing at all — a root ``allOf``/``anyOf``/``oneOf``
        # naming further required keys, a root ``const``/``enum``, a root
        # ``not``, and ``minProperties``/``maxProperties``. Merged and
        # imported schemas (MCP, OpenAPI) put constraints there routinely
        # (PR #385 review).
        #
        # Handed to ``_validate_value`` — which implements all of them, and
        # was simply never called on the root — as a schema of *just* those
        # keywords. Passing the whole schema instead would re-walk the
        # properties the loop below already covers, at every dispatch, and
        # would resolve the root's ``additionalProperties`` under the nested
        # branch's rules: absent means free-form there, while a tool schema
        # treats it as closed, so an undeclared argument would stop being
        # reported. Selecting the keywords keeps one owner for each.
        root_assertions = {key: schema[key] for key in _ROOT_ASSERTIONS if key in schema}
        if root_assertions:
            is_valid, err = _validate_value(arguments, root_assertions, "")
            if not is_valid:
                return False, err

        # A tool schema can type its keys by regex too — the same construct
        # the nested-object branch handles, one level up. Reading only
        # ``properties`` here rejected a schema-valid key as unknown when the
        # schema was closed, and skipped the pattern's own constraint when it
        # was open.
        top_ok, top_pattern_matched, top_err = _check_pattern_properties(
            arguments, schema, ""
        )
        if not top_ok:
            return False, top_err

        # Check parameter types
        additional_schema = schema.get("additionalProperties")
        # A key a root combinator branch declares is declared: the branches are
        # evaluated above, so without this the same pass validated a key and
        # then called it unknown.
        top_branch_declared = _branch_declared_names(schema)
        for param_name, param_value in arguments.items():
            if param_name not in properties:
                if param_name in top_pattern_matched:
                    continue  # already checked against its pattern
                if param_name in top_branch_declared:
                    continue  # declared by a combinator branch
                if allow_additional:
                    if isinstance(additional_schema, dict):
                        is_valid, err = _validate_value(
                            param_value, additional_schema, param_name
                        )
                        if not is_valid:
                            return False, err
                    continue
                return False, f"Unknown parameter: {param_name}"

            is_valid, err = _validate_value(param_value, properties[param_name], param_name)
            if not is_valid:
                return False, err

        return True, None

    async def _record_success(self, tool: ToolDefinition, latency_ms: float) -> None:
        """Record a successful execution."""
        old_health = tool.health.model_copy() if self._telemetry else None

        tool.health.total_calls += 1
        tool.health.last_success = datetime.now(timezone.utc)
        tool.health.consecutive_failures = 0
        tool.health.circuit_breaker_open = False

        # Update average latency
        n = tool.health.total_calls
        tool.health.avg_latency_ms = (tool.health.avg_latency_ms * (n - 1) + latency_ms) / n

        # Update success rate
        tool.health.success_rate = (tool.health.success_rate * (n - 1) + 1) / n

        if self._telemetry and old_health:
            await self._telemetry.record_health_change(
                f"{tool.namespace}.{tool.name}", old_health, tool.health
            )

    async def _record_failure(self, tool: ToolDefinition) -> None:
        """Record a failed execution."""
        old_health = tool.health.model_copy() if self._telemetry else None

        tool.health.total_calls += 1
        tool.health.last_failure = datetime.now(timezone.utc)
        tool.health.consecutive_failures += 1

        # Update success rate
        n = tool.health.total_calls
        tool.health.success_rate = (tool.health.success_rate * (n - 1)) / n

        # Check circuit breaker
        if tool.health.consecutive_failures >= self._cb_threshold:
            tool.health.circuit_breaker_open = True

        if self._telemetry and old_health:
            await self._telemetry.record_health_change(
                f"{tool.namespace}.{tool.name}", old_health, tool.health
            )

    def _resolve_tool(self, call: ToolCall) -> ToolDefinition | None:
        """Resolve the tool a call targets, honouring its namespace.

        Precedence: an explicit ``call.namespace``, then a qualified
        ``tool_name`` ("billing.search" -- tool names themselves cannot contain
        a dot, so this is unambiguous), then a bare-name search across
        namespaces for backward compatibility. The bare-name path logs when the
        name exists in more than one namespace, because that is the case where
        it can silently run a tool other than the one that was selected.
        """
        if call.namespace:
            return self._registry.get_tool(call.tool_name, call.namespace)

        if "." in call.tool_name:
            namespace, _, bare = call.tool_name.rpartition(".")
            qualified = self._registry.get_tool(bare, namespace)
            if qualified is not None:
                return qualified
            # Fall through: a literal name containing a dot is invalid per the
            # ToolDefinition pattern, but a stale caller may still send one.

        candidates = self._registry.namespaces_for_name(call.tool_name)
        if len(candidates) > 1:
            logger.warning(
                "Tool %r is registered in %d namespaces (%s); executing by bare "
                "name resolves to one of them. Pass ToolCall(namespace=...) to "
                "target a specific tool.",
                call.tool_name,
                len(candidates),
                ", ".join(sorted(candidates)),
            )
        return self._registry.get_tool_by_name(call.tool_name)

    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID."""
        return str(uuid.uuid4())

    def _generate_span_id(self) -> str:
        """Generate a unique span ID."""
        return str(uuid.uuid4())[:16]

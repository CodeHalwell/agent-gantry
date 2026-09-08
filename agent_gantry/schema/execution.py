"""
Execution models for Agent-Gantry.

Models for tool calls, results, and batch operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_gantry.schema.base import reject_newlines


class ExecutionStatus(str, Enum):
    """Status of a tool execution."""

    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    PERMISSION_DENIED = "permission_denied"
    CIRCUIT_OPEN = "circuit_open"
    PENDING_CONFIRMATION = "pending_confirmation"
    CANCELLED = "cancelled"


class ToolCall(BaseModel):
    """Request to execute a tool."""

    tool_name: str
    #: Namespace the call targets. ``None`` means "resolve by bare name",
    #: which is what a provider tool-call payload can express -- the model only
    #: ever sees the bare name. Callers that already know which tool was
    #: selected (every framework adapter, since selection is namespace-aware)
    #: should set this, otherwise a same-named tool in another namespace can
    #: be executed instead. A qualified ``tool_name`` ("billing.search") is
    #: also accepted and takes effect when this field is unset.
    namespace: str | None = None
    arguments: dict[str, Any]

    timeout_ms: int = Field(default=30000, ge=100, le=300000)
    retry_count: int = Field(default=0, ge=0, le=5)
    require_confirmation: bool | None = None

    trace_id: str | None = None
    parent_span_id: str | None = None

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("tool_name", "trace_id", "parent_span_id")
    @classmethod
    def _validate_newlines(cls, v: str | None) -> str | None:
        return reject_newlines(v)


class ToolResult(BaseModel):
    """Result of a tool execution."""

    tool_name: str
    status: ExecutionStatus

    result: Any | None = None
    error: str | None = None
    error_type: str | None = None

    queued_at: datetime
    started_at: datetime | None = None
    completed_at: datetime

    attempt_number: int = Field(default=1)

    trace_id: str
    span_id: str

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("tool_name", "trace_id", "span_id")
    @classmethod
    def _validate_newlines(cls, v: str | None) -> str | None:
        return reject_newlines(v)

    @property
    def latency_ms(self) -> float:
        """Calculate execution latency in milliseconds."""
        if self.started_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0.0


class BatchToolCall(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    """Request to execute multiple tools."""

    calls: list[ToolCall]
    execution_strategy: Literal["parallel", "sequential", "adaptive"] = "adaptive"
    fail_fast: bool = False


class BatchToolResult(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    """Result of a batch tool execution."""

    results: list[ToolResult]
    total_time_ms: float
    successful_count: int
    failed_count: int


@dataclass(frozen=True)
class ToolCallEvent:
    """A completed tool execution, delivered to ``gantry.on_tool_call`` callbacks.

    Intentionally a ``@dataclass`` rather than a Pydantic ``BaseModel`` like the
    rest of this module: it is an ephemeral, in-process event (never serialised
    or validated), so a frozen dataclass keeps it allocation-cheap on the
    ``execute`` hot path. Don't "promote" it to a BaseModel without reason.

    Framework-agnostic: every call routed through
    :meth:`~agent_gantry.core.gantry.AgentGantry.execute` (and each call in a
    batch) emits one of these once execution finishes — successfully or not —
    regardless of which agent framework (if any) drove the call. This is the
    single seam for cross-framework logging and metrics; the convenience
    accessors mirror the most-used fields of the underlying result.
    """

    call: ToolCall
    result: ToolResult

    @property
    def tool_name(self) -> str:
        """Name of the tool that was executed.

        Prefers ``result.tool_name`` and falls back to ``call.tool_name`` only
        defensively (the result name is normally always populated; the fallback
        covers any error path that produced a result without one).
        """
        return self.result.tool_name or self.call.tool_name

    @property
    def status(self) -> ExecutionStatus:
        """Terminal status of the execution."""
        return self.result.status

    @property
    def ok(self) -> bool:
        """``True`` when the call completed successfully."""
        return self.result.status == ExecutionStatus.SUCCESS

    @property
    def latency_ms(self) -> float:
        """Execution latency in milliseconds (0.0 if not started)."""
        return self.result.latency_ms

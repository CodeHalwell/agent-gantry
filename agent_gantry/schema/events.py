"""
Event models for Agent-Gantry observability.

Structured events for retrieval, execution, and health changes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_gantry.schema.base import reject_newlines
from agent_gantry.schema.tool import ToolHealth


class RetrievalEvent(BaseModel):
    """Event emitted after a tool retrieval operation."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: str
    query: str
    candidate_count: int
    returned_count: int
    total_time_ms: float
    tool_names: list[str]
    scores: list[float]
    filters_applied: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("trace_id")
    @classmethod
    def _validate_newlines(cls, v: str | None) -> str | None:
        return reject_newlines(v)


class ExecutionEvent(BaseModel):
    """Event emitted after a tool execution."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: str
    span_id: str
    tool_name: str
    status: str
    latency_ms: float
    attempt_number: int
    error: str | None = None
    error_type: str | None = None

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("trace_id", "span_id", "tool_name")
    @classmethod
    def _validate_newlines(cls, v: str | None) -> str | None:
        return reject_newlines(v)


class HealthChangeEvent(BaseModel):
    """Event emitted when a tool's health status changes."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tool_name: str
    old_health: ToolHealth
    new_health: ToolHealth
    reason: str

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("tool_name")
    @classmethod
    def _validate_newlines(cls, v: str | None) -> str | None:
        return reject_newlines(v)


class TokenUsageEvent(BaseModel):
    """Event emitted for LLM token usage and savings."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    # Optional savings info if this was an optimized call
    baseline_prompt_tokens: int | None = None
    saved_prompt_tokens: int | None = None
    savings_pct: float | None = None
    trace_id: str | None = None

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("trace_id")
    @classmethod
    def _validate_newlines(cls, v: str | None) -> str | None:
        return reject_newlines(v)

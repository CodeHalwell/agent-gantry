"""
Query and context models for Agent-Gantry.

Models for conversation context, tool queries, and retrieval results.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_gantry.schema.base import reject_newlines
from agent_gantry.schema.tool import ToolCapability, ToolDefinition, ToolSource


class ConversationContext(BaseModel):
    """
    Conversation state for context-aware routing.

    Key idea: routing isn't just about the current query; we use:
    - conversation summary
    - recent messages
    - which tools have been tried / failed
    - user capabilities
    """

    # Immediate request
    query: str = Field(...)

    # Coarse history
    conversation_summary: str | None = Field(default=None)
    recent_messages: list[dict[str, str]] = Field(default_factory=list, max_length=10)

    # Tool usage in this conversation
    tools_already_used: list[str] = Field(default_factory=list)
    tools_failed: list[str] = Field(default_factory=list)

    # Optional high-level intent classification
    inferred_intent: str | None = Field(default=None)

    # Permission context
    user_capabilities: list[ToolCapability] = Field(
        default_factory=lambda: [cap for cap in ToolCapability]
    )
    require_confirmation_for: list[ToolCapability] = Field(default_factory=list)


class ToolQuery(BaseModel):
    """Request to find relevant tools."""

    context: ConversationContext

    limit: int = Field(default=5, ge=1, le=50)
    # NOTE: this 0.5 default is intentionally *not* mirrored by the
    # higher-level convenience APIs (``agent_gantry.integrations.frameworks``
    # adapters, ``with_semantic_tools`` / ``SemanticToolSelector`` in
    # ``agent_gantry.integrations.semantic_tools``). Those all default to
    # ``0.0`` (no filtering) because a flat absolute-similarity cutoff is a
    # silent-drop trap once queries get longer or more specific — long queries
    # dilute absolute similarity scores, so a non-zero default can silently
    # return zero tools with no error. Callers constructing a ``ToolQuery``
    # directly should set ``score_threshold`` explicitly rather than relying on
    # this default; the convenience layers set it to ``0.0`` for exactly this
    # reason. Kept at 0.5 here for backward compatibility with existing direct
    # ``ToolQuery`` callers — do not change this without a migration plan.
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # Filters
    namespaces: list[str] | None = None
    required_capabilities: list[ToolCapability] | None = None
    excluded_capabilities: list[ToolCapability] | None = None
    sources: list[ToolSource] | None = None
    exclude_deprecated: bool = True
    exclude_unhealthy: bool = True

    # Advanced
    # ``None`` means "defer to the gantry's reranker config": ``retrieve()``
    # turns it on when a reranker is configured. Set it explicitly to
    # ``True``/``False`` to force the behaviour regardless of config.
    enable_reranking: bool | None = None
    include_dependencies: bool = True
    diversity_factor: float = Field(default=0.0, ge=0.0, le=1.0)


class ScoredTool(BaseModel):
    """A tool with its relevance scores."""

    tool: ToolDefinition
    semantic_score: float = Field(ge=0.0, le=1.0)
    rerank_score: float | None = None
    context_score: float = 0.0
    health_penalty: float = 0.0

    @property
    def final_score(self) -> float:
        """Calculate the final composite score."""
        base = self.rerank_score if self.rerank_score is not None else self.semantic_score
        return max(0.0, base + self.context_score - self.health_penalty)


class RetrievalResult(BaseModel):
    """Result of a tool retrieval operation."""

    tools: list[ScoredTool]

    query_embedding_time_ms: float
    vector_search_time_ms: float
    rerank_time_ms: float | None = None
    total_time_ms: float

    candidate_count: int
    filtered_count: int

    trace_id: str

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("trace_id")
    @classmethod
    def _reject_newline_identifiers(cls, v: str | None) -> str | None:
        return reject_newlines(v)

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """
        Convert retrieved tools to OpenAI format.

        Deprecated: Use `to_dialect("openai")` instead.
        """
        return [t.tool.to_dialect("openai") for t in self.tools]

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        """
        Convert retrieved tools to Anthropic format.

        Deprecated: Use `to_dialect("anthropic")` instead.
        """
        return [t.tool.to_dialect("anthropic") for t in self.tools]

    def to_dialect(self, dialect: str = "auto", **options: Any) -> list[dict[str, Any]]:
        """
        Convert retrieved tools to provider-specific format.

        Uses the dialect registry for extensible provider support.

        Args:
            dialect: Target dialect/provider name (default: 'auto')
            **options: Provider-specific options (e.g., strict mode for OpenAI)

        Returns:
            List of provider-specific tool schemas
        """
        return [t.tool.to_dialect(dialect, **options) for t in self.tools]

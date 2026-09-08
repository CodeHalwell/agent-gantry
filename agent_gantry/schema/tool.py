"""
Tool definition models for Agent-Gantry.

Universal representation of tools, regardless of source (Python function, MCP server,
OpenAPI operation, or A2A agent skill).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_gantry.schema.base import HealthMetrics, reject_newlines


class SchemaDialect(str, Enum):
    """Schema dialects for different LLM providers."""

    OPENAI = "openai"
    OPENAI_RESPONSES = "openai_responses"  # OpenAI Responses API (newer API)
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    MISTRAL = "mistral"
    GROQ = "groq"
    AGENT_FRAMEWORK = "agent_framework"
    AUTO = "auto"


class ToolSource(str, Enum):
    """Source of the tool definition."""

    PYTHON_FUNCTION = "python_function"
    MCP_SERVER = "mcp_server"
    OPENAPI = "openapi"
    A2A_AGENT = "a2a_agent"
    MANUAL = "manual"
    # Imported from another agent framework's native tool object (LangChain
    # BaseTool, CrewAI BaseTool, LlamaIndex FunctionTool, ...) via
    # agent_gantry.integrations.importers. Distinct from PYTHON_FUNCTION
    # (@gantry.register-ed or bare Python callables) so telemetry/governance
    # can tell tools that originated outside Gantry's own registry apart from
    # those authored directly against it. Additive: existing serialized
    # ToolDefinition data using any other source value is unaffected.
    FRAMEWORK = "framework"


class ToolCapability(str, Enum):
    """Capabilities that a tool may have, used for permission checks."""

    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    DELETE_DATA = "delete_data"
    EXECUTE_CODE = "execute_code"
    NETWORK_ACCESS = "network_access"
    FILE_SYSTEM = "file_system"
    FINANCIAL = "financial"
    PII_ACCESS = "pii_access"
    EXTERNAL_API = "external_api"


class ToolCost(BaseModel):
    """Cost model for tool execution."""

    estimated_latency_ms: int = Field(default=100)
    monetary_cost: float | None = Field(default=None)
    rate_limit: int | None = Field(default=None, description="Max calls per minute")
    context_tokens: int = Field(default=0, description="Tokens added when selected")


class ToolHealth(HealthMetrics):
    """Runtime health metrics for a tool."""

    avg_latency_ms: float = Field(default=0.0)
    total_calls: int = Field(default=0)
    circuit_breaker_open: bool = Field(default=False)


class ToolDefinition(BaseModel):
    """
    Universal representation of a tool.

    Canonical internal format, regardless of original source:
    - Python function
    - MCP server tool
    - OpenAPI operation
    - A2A agent skill
    """

    # Identity
    name: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9_]*$")
    version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")
    namespace: str = Field(default="default")

    # Discovery
    description: str = Field(..., min_length=10, max_length=2000)
    extended_description: str | None = Field(default=None, max_length=10000)
    examples: list[str] = Field(default_factory=list, max_length=10)
    tags: list[str] = Field(default_factory=list)

    # Schema (canonical JSON Schema for input/output)
    parameters_schema: dict[str, Any] = Field(
        ..., description="JSON Schema for input parameters (OpenAI-style function calling)"
    )
    returns_schema: dict[str, Any] | None = Field(default=None)

    # Provenance
    source: ToolSource = Field(default=ToolSource.PYTHON_FUNCTION)
    source_uri: str | None = Field(default=None)

    # Capabilities & permissions
    capabilities: list[ToolCapability] = Field(default_factory=list)
    requires_confirmation: bool = Field(default=False)

    # Cost model
    cost: ToolCost = Field(default_factory=ToolCost)

    # Runtime (non-persisted)
    health: ToolHealth = Field(default_factory=ToolHealth, exclude=True)

    # Metadata / lifecycle
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    deprecated: bool = Field(default=False)
    deprecation_message: str | None = Field(default=None)
    superseded_by: str | None = Field(default=None)

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    @field_validator("name", "version", "namespace")
    @classmethod
    def _reject_newline_identifiers(cls, v: str | None) -> str | None:
        return reject_newlines(v)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that the tool name is not reserved."""
        reserved = {"register", "retrieve", "execute", "list", "delete"}
        if v in reserved:
            raise ValueError(f"Tool name '{v}' is reserved")
        return v

    @property
    def qualified_name(self) -> str:
        """Return namespace.name:version."""
        return f"{self.namespace}.{self.name}:{self.version}"

    @property
    def content_hash(self) -> str:
        """
        Deterministic hash for change detection and efficient syncing.

        Used to avoid re-embedding / re-indexing when nothing changed.
        """
        content = f"{self.name}:{self.version}:{self.description}:{self.parameters_schema}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dialect(self, dialect: SchemaDialect | str, **options: Any) -> dict[str, Any]:
        """
        Just-in-Time transcoding for specific LLMs or protocols.

        Uses the dialect registry for extensible provider support.

        Args:
            dialect: Target dialect (SchemaDialect enum or string name)
            **options: Provider-specific options (e.g., strict mode for OpenAI)

        Returns:
            Provider-specific tool schema dictionary
        """
        from agent_gantry.adapters.tool_spec.registry import get_adapter

        # Convert enum to string if needed
        dialect_str = dialect.value if isinstance(dialect, SchemaDialect) else dialect
        adapter = get_adapter(dialect_str)
        return adapter.to_provider_schema(self, **options)

    def to_searchable_text(self) -> str:
        """
        Convert tool metadata to searchable text for embedding.

        Combines name, namespace, description, tags, and examples into a
        single text representation optimized for semantic search.

        Returns:
            Concatenated string of tool metadata for embedding

        Example:
            >>> tool = ToolDefinition(
            ...     name="calculate_tax",
            ...     namespace="finance",
            ...     description="Calculate sales tax",
            ...     tags=["math", "money"],
            ...     examples=["tax on $100"],
            ...     parameters_schema={"type": "object", "properties": {}}
            ... )
            >>> text = tool.to_searchable_text()
            >>> "calculate_tax" in text and "finance" in text
            True
        """
        tags = " ".join(self.tags)
        examples = " ".join(self.examples)
        return f"{self.name} {self.namespace} {self.description} {tags} {examples}"


class ToolDependency(BaseModel):
    """Dependency relationship between tools."""

    tool_name: str
    dependency_type: Literal["requires", "suggests", "conflicts"]
    reason: str | None = None

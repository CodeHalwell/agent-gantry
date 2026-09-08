"""
MCP Server definition models for Agent-Gantry.

Represents MCP server metadata for semantic routing and dynamic selection.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_gantry.schema.base import HealthMetrics, reject_newlines


class MCPServerHealth(HealthMetrics):
    """Runtime health metrics for an MCP server."""

    avg_connection_time_ms: float = Field(default=0.0)
    total_connections: int = Field(default=0)
    available: bool = Field(default=True)


class MCPServerCost(BaseModel):
    """Cost model for MCP server operations."""

    estimated_connection_latency_ms: int = Field(default=500)
    estimated_tool_discovery_latency_ms: int = Field(default=1000)
    rate_limit: int | None = Field(default=None, description="Max connections per minute")


class MCPServerDefinition(BaseModel):
    """
    Represents an MCP server for semantic routing and dynamic selection.

    Similar to ToolDefinition but for MCP servers, enabling intelligent
    server selection based on query context.
    """

    # Identity
    name: str = Field(..., min_length=1, max_length=128)
    namespace: str = Field(default="default")

    # Discovery
    description: str = Field(..., min_length=10, max_length=2000)
    extended_description: str | None = Field(default=None, max_length=10000)
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorizing the server (e.g., 'filesystem', 'database', 'api')",
    )
    examples: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Example queries this server handles (max 10 examples, each should be concise)",
    )

    # Connection configuration (from MCPServerConfig)
    command: list[str] = Field(
        ...,
        min_length=1,
        description="Command to start the MCP server. "
        "WARNING: Ensure command components are from trusted sources to prevent command injection. "
        "Never pass unsanitized user input directly to this field.",
    )
    args: list[str] = Field(
        default_factory=list,
        description="Arguments for the command. "
        "WARNING: Validate and sanitize all arguments to prevent command injection.",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables for the server process. "
        "Marked as sensitive - avoid logging.",
        repr=False,
    )

    # Capabilities
    capabilities: list[str] = Field(
        default_factory=list,
        description="Server capabilities (e.g., 'read_files', 'write_files', 'execute_commands')",
    )

    # Cost model
    cost: MCPServerCost = Field(default_factory=MCPServerCost)

    # Runtime health (non-persisted)
    health: MCPServerHealth = Field(default_factory=MCPServerHealth, exclude=True)

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    deprecated: bool = Field(default=False)

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    @field_validator("name", "namespace")
    @classmethod
    def _reject_newline_identifiers(cls, v: str | None) -> str | None:
        return reject_newlines(v)

    @property
    def qualified_name(self) -> str:
        """Return namespace.name."""
        return f"{self.namespace}.{self.name}"

    @property
    def content_hash(self) -> str:
        """
        Deterministic hash for change detection.

        Used to avoid re-embedding when server metadata hasn't changed.
        """
        content = f"{self.name}:{self.description}:{','.join(self.tags)}:{','.join(self.examples)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_searchable_text(self) -> str:
        """
        Convert server metadata to searchable text for embedding.

        Returns:
            Text representation for semantic search
        """
        parts = [
            f"Server: {self.name}",
            f"Description: {self.description}",
        ]

        if self.extended_description:
            parts.append(f"Details: {self.extended_description}")

        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")

        if self.examples:
            parts.append(f"Examples: {', '.join(self.examples)}")

        if self.capabilities:
            parts.append(f"Capabilities: {', '.join(self.capabilities)}")

        return " ".join(parts)

    def to_config(self) -> dict[str, Any]:
        """
        Convert to MCPServerConfig-compatible dict.

        Returns:
            Configuration dictionary
        """
        return {
            "name": self.name,
            "command": self.command,
            "args": self.args,
            "env": self.env,
            "namespace": self.namespace,
        }

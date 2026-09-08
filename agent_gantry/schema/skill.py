"""
Skill definition models for Agent-Gantry.

Skills are procedural memories, patterns, and how-tos that are retrieved
and injected into the system prompt to guide agent behavior. Unlike tools,
skills are not directly executed but provide contextual knowledge.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_gantry.schema.base import reject_newlines


class SkillCategory(str, Enum):
    """Categories for organizing skills."""

    HOW_TO = "how_to"
    PATTERN = "pattern"
    PROCEDURE = "procedure"
    BEST_PRACTICE = "best_practice"
    TEMPLATE = "template"
    EXAMPLE = "example"
    GUIDELINE = "guideline"
    WORKFLOW = "workflow"


class Skill(BaseModel):
    """
    Procedural memory / skill definition.

    Skills are retrieved semantically and injected into the system prompt
    to provide contextual knowledge and guidance. They are not executed
    directly like tools.

    Example:
        >>> skill = Skill(
        ...     name="api_pagination",
        ...     description="How to implement cursor-based pagination for API endpoints",
        ...     content="When implementing pagination, use cursor-based pagination...",
        ...     category=SkillCategory.HOW_TO,
        ...     tags=["api", "pagination", "rest"],
        ... )
    """

    # Identity
    name: str = Field(..., min_length=1, max_length=128)
    namespace: str = Field(default="default")

    # Content
    description: str = Field(..., min_length=10, max_length=2000)
    content: str = Field(..., min_length=1, max_length=50000, description="Full skill content")
    summary: str | None = Field(
        default=None, max_length=500, description="Brief summary for quick reference"
    )

    # Classification
    category: SkillCategory = Field(default=SkillCategory.HOW_TO)
    tags: list[str] = Field(default_factory=list)
    related_tools: list[str] = Field(
        default_factory=list, description="Tool names this skill relates to"
    )

    # Source / provenance
    source: str | None = Field(default=None, description="Origin of the skill (docs, user, etc)")
    source_uri: str | None = Field(default=None, description="URI to source document")

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = Field(default=None)

    # Usage tracking
    usage_count: int = Field(default=0, description="Times this skill has been retrieved")
    last_used: datetime | None = Field(default=None)

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    @field_validator("name", "namespace")
    @classmethod
    def _validate_newlines(cls, v: str | None) -> str | None:
        return reject_newlines(v)

    @property
    def qualified_name(self) -> str:
        """Return namespace.name identifier."""
        return f"{self.namespace}.{self.name}"

    @property
    def content_hash(self) -> str:
        """
        Deterministic hash for change detection.

        This hash includes all fields that influence the embedding text, as well
        as the full content, so that any semantic change will invalidate it.
        """
        hash_input = f"{self.to_embedding_text()}::{self.content}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def to_prompt_text(self) -> str:
        """
        Format skill for system prompt injection.

        Returns:
            Formatted text suitable for system prompt
        """
        lines = [
            f"## {self.name.replace('_', ' ').title()}",
            f"*Category: {self.category.value}*",
            "",
            self.content,
        ]
        if self.related_tools:
            lines.extend(["", f"Related tools: {', '.join(self.related_tools)}"])
        return "\n".join(lines)

    def to_embedding_text(self) -> str:
        """
        Flatten skill metadata into text for embedding.

        Returns:
            Text representation for embedding
        """
        parts = [
            self.name,
            self.namespace,
            self.description,
            " ".join(self.tags),
            self.category.value,
            self.summary or "",
        ]
        return " ".join(filter(None, parts))


class SkillSearchResult(BaseModel):
    """Result from a skill search operation."""

    skill: Skill
    score: float = Field(..., ge=0.0, le=1.0)


class SkillRetrievalResult(BaseModel):
    """Aggregated result from skill retrieval."""

    skills: list[SkillSearchResult]
    total_count: int
    query_time_ms: float
    trace_id: str | None = None

    def to_prompt_injection(self, max_skills: int = 3) -> str:
        """
        Format top skills for system prompt injection.

        Args:
            max_skills: Maximum number of skills to include

        Returns:
            Formatted text for system prompt
        """
        if not self.skills:
            return ""

        lines = ["# Relevant Knowledge", ""]
        for result in self.skills[:max_skills]:
            lines.append(result.skill.to_prompt_text())
            lines.append("")

        return "\n".join(lines)

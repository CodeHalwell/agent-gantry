"""
A2A (Agent-to-Agent) protocol models.

Models for agent discovery, skill definitions, and task communication.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_gantry.schema.base import reject_newlines


class AgentSkill(BaseModel):
    """
    A skill provided by an A2A agent.

    Skills map to tool capabilities that can be executed via task requests.
    """

    id: str = Field(..., description="Unique identifier for the skill")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Detailed description of what the skill does")
    input_modes: list[str] = Field(default=["text"], description="Supported input types")
    output_modes: list[str] = Field(default=["text"], description="Supported output types")

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    @field_validator("id", "name")
    @classmethod
    def _validate_newlines(cls, v: str | None) -> str | None:
        return reject_newlines(v)


class AgentCard(BaseModel):
    """
    Agent Card following the A2A protocol specification.

    Served at /.well-known/agent.json to advertise agent capabilities.
    """

    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    url: str = Field(..., description="Base URL for the agent")
    version: str = Field(default="1.0.0", description="Agent version")

    skills: list[AgentSkill] = Field(
        default_factory=list, description="Skills provided by this agent"
    )

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    @field_validator("name")
    @classmethod
    def _validate_newlines(cls, v: str | None) -> str | None:
        return reject_newlines(v)

    authentication: dict[str, Any] | None = Field(
        default=None, description="Authentication configuration"
    )
    provider: dict[str, str] = Field(default_factory=dict, description="Provider information")


class TaskMessagePart(BaseModel):
    """A part of a task message."""

    type: str = Field(default="text", description="Message part type")
    text: str | None = Field(default=None, description="Text content if type is text")
    data: Any | None = Field(default=None, description="Additional data")


class TaskMessage(BaseModel):
    """A message in a task request."""

    role: str = Field(..., description="Role (e.g., 'user', 'assistant')")
    parts: list[TaskMessagePart] = Field(default_factory=list, description="Message parts")


class TaskRequest(BaseModel):
    """Request to execute a task on an A2A agent."""

    skill_id: str = Field(..., description="ID of the skill to execute")
    messages: list[TaskMessage] = Field(default_factory=list, description="Conversation messages")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TaskResponse(BaseModel):
    """Response from an A2A agent task execution."""

    status: str = Field(..., description="Status (success, error, etc.)")
    result: Any | None = Field(default=None, description="Task result")
    error: str | None = Field(default=None, description="Error message if failed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Response metadata")

"""
Configuration models for Agent-Gantry.

Single source of truth for all configuration options.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_gantry.schema.base import reject_newlines


class VectorStoreConfig(BaseModel):
    """Configuration for vector store backend."""

    type: Literal["memory", "qdrant", "chroma", "pgvector", "pinecone", "weaviate", "lancedb"] = (
        "memory"
    )
    url: str | None = None
    api_key: str | None = None
    collection_name: str = "agent_gantry"
    dimension: int | None = None
    db_path: str | None = Field(default=None, description="Path to local database (for LanceDB)")
    options: dict[str, Any] = Field(default_factory=dict)


class EmbedderConfig(BaseModel):
    """Configuration for embedding backend."""

    type: Literal[
        "openai", "azure", "cohere", "huggingface", "sentence_transformers", "ollama", "nomic"
    ] = "sentence_transformers"
    model: str = "all-MiniLM-L6-v2"
    api_key: str | None = None
    api_base: str | None = None
    api_version: str | None = Field(default=None, description="API version (for Azure OpenAI)")
    batch_size: int = 100
    max_retries: int = 3
    dimension: int | None = Field(
        default=None, description="Output dimension (for Matryoshka truncation)"
    )
    task_type: str | None = Field(default=None, description="Task type for embeddings (for Nomic)")


class RerankerConfig(BaseModel):
    """Configuration for reranker backend."""

    enabled: bool = False
    type: Literal["cohere", "cross_encoder", "llm"] = "cross_encoder"
    model: str | None = None
    top_k: int = 10


class LLMConfig(BaseModel):
    """Configuration for LLM-based features (intent classification, etc.)."""

    provider: Literal["openai", "anthropic", "google", "mistral", "groq"] = "openai"
    # BREAKING (2026-08-03): default changed from "gpt-4o-mini", which OpenAI
    # shuts down on 2026-10-23 — leaving the old default would break every
    # deployment relying on it at that date. Callers pinning the old model
    # must now set model="gpt-4o-mini" explicitly (until shutdown).
    model: str = "gpt-5.4-mini"
    api_key: str | None = None
    base_url: str | None = Field(default=None, description="Custom base URL (for OpenRouter, etc.)")
    max_tokens: int = 100
    temperature: float = 0.0


class RoutingConfig(BaseModel):
    """Configuration for semantic routing."""

    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "semantic": 0.6,
            "intent": 0.15,
            "conversation": 0.1,
            "health": 0.1,
            "cost": 0.05,
        }
    )
    enable_intent_classification: bool = True
    use_llm_for_intent: bool = False
    llm: LLMConfig = Field(default_factory=LLMConfig)
    enable_mmr: bool = True
    mmr_lambda: float = 0.7


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting."""

    enabled: bool = True
    strategy: Literal["sliding_window", "token_bucket", "fixed_window"] = "sliding_window"
    max_calls_per_minute: int = 60
    max_calls_per_hour: int = 1000
    max_concurrent: int = 10
    burst_size: int | None = Field(default=None, description="Burst size for token bucket strategy")
    per_tool: bool = Field(default=True, description="Rate limit per tool (vs. globally)")
    per_namespace: bool = Field(default=False, description="Rate limit per namespace")


class ExecutionConfig(BaseModel):
    """Configuration for tool execution."""

    default_timeout_ms: int = 30000
    max_retries: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_s: int = 60
    enable_sandbox: bool = False
    sandbox_type: Literal["none", "subprocess", "docker"] = "none"
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)


class TelemetryConfig(BaseModel):
    """Configuration for observability."""

    enabled: bool = True
    type: Literal["console", "opentelemetry", "datadog", "prometheus"] = "console"
    otlp_endpoint: str | None = None
    service_name: str = "agent_gantry"
    expose_prometheus: bool = False
    prometheus_port: int = 9090


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server to connect to."""

    name: str
    command: list[str]
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    namespace: str = "default"

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("name", "namespace")
    @classmethod
    def _reject_newline_identifiers(cls, v: str | None) -> str | None:
        return reject_newlines(v)


class MCPConfig(BaseModel):
    """Configuration for MCP integration."""

    servers: list[MCPServerConfig] = Field(default_factory=list)
    serve_mcp: bool = False
    mcp_mode: Literal["dynamic", "static", "hybrid"] = "dynamic"


class A2AAgentConfig(BaseModel):
    """Configuration for an A2A agent to connect to."""

    name: str
    url: str
    namespace: str = "default"

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("name", "namespace")
    @classmethod
    def _reject_newline_identifiers(cls, v: str | None) -> str | None:
        return reject_newlines(v)


class A2AConfig(BaseModel):
    """Configuration for A2A integration."""

    agents: list[A2AAgentConfig] = Field(default_factory=list)
    serve_a2a: bool = False
    a2a_port: int = 8080


class AgentGantryConfig(BaseModel):
    """
    Main configuration for Agent-Gantry.

    Can be loaded from YAML files or constructed programmatically.
    """

    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    a2a: A2AConfig = Field(default_factory=A2AConfig)

    auto_sync: bool = True
    sync_on_register: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> AgentGantryConfig:
        """
        Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file

        Returns:
            Configured AgentGantryConfig instance

        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If the YAML is invalid or doesn't match schema
        """
        from pathlib import Path as PathLib

        import yaml  # type: ignore[import-untyped]

        config_path = PathLib(path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {path}\n"
                f"Create a config file or use AgentGantryConfig() for defaults."
            )

        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}") from e

        if data is None:
            data = {}

        try:
            return cls(**data)
        except Exception as e:
            raise ValueError(f"Invalid configuration in {path}: {e}") from e

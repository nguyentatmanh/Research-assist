from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

AgentRole = Literal["retrieval", "reader", "summarizer", "fact_checker", "writer"]
Provider = Literal["demo", "gemini", "openai", "ollama", "deepseek"]


class AgentRoute(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Provider | None = None
    model: str = ""
    max_output_tokens: int | None = Field(default=None, ge=512, le=12000)
    thinking_level: Literal["minimal", "low", "medium", "high"] | None = None
    input_usd_per_million: float | None = Field(default=None, ge=0)
    output_usd_per_million: float | None = Field(default=None, ge=0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RA_",
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8",
    )
    data_dir: Path = Path("data")
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    llm_provider: Provider = "demo"
    llm_model: str = ""  # Provider-specific defaults; override to use a tuned model.
    workflow_profile: Literal["balanced", "academic"] = "balanced"
    agent_routes: dict[AgentRole, AgentRoute] = Field(default_factory=dict)
    gemini_api_key: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")
    openai_base_url: str = "https://api.openai.com/v1"
    ollama_base_url: str = "http://localhost:11434/v1"
    deepseek_api_key: SecretStr = SecretStr("")
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_thinking: bool = False
    structured_output: Literal["schema", "json"] = "schema"
    embedding_provider: Literal["none", "sentence-transformers", "openai"] = "none"
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_api_key: SecretStr = SecretStr("")
    embedding_batch_size: int = Field(default=32, ge=1, le=128)
    embedding_family: Literal["auto", "e5", "generic", "qwen3"] = "auto"
    embedding_revision: str = ""
    embedding_max_seq_length: int = Field(default=512, ge=64, le=8192)
    local_device: Literal["cpu", "cuda", "mps", "auto"] = "cpu"
    reranker_provider: Literal["auto", "none", "local", "cohere"] = "auto"
    reranker_model: str = ""  # auto preserves legacy behavior: empty disables local reranking.
    cohere_api_key: SecretStr = SecretStr("")
    cohere_base_url: str = "https://api.cohere.com/v2"
    reranker_requests_per_minute: int = Field(default=8, ge=1, le=1000)
    reranker_timeout_seconds: float = Field(default=60, ge=1, le=300)
    reranker_cache_ttl_seconds: int = Field(default=86400, ge=0, le=604800)
    reranker_max_tokens_per_doc: int = Field(default=4096, ge=128, le=32768)
    reranker_device: Literal["inherit", "cpu", "cuda", "mps", "auto"] = "inherit"
    reranker_batch_size: int = Field(default=4, ge=1, le=64)
    reranker_max_length: int = Field(default=512, ge=64, le=2048)
    reranker_failure_policy: Literal["fallback", "error"] = "fallback"
    search_cache_scopes: int = Field(default=2, ge=0, le=16)
    search_cache_mb: int = Field(default=128, ge=8, le=2048)
    pdf_ocr_enabled: bool = False
    pdf_ocr_language: str = "eng"
    pdf_ocr_dpi: int = Field(default=150, ge=72, le=300)
    pdf_ocr_max_pages: int = Field(default=20, ge=1, le=100)
    semantic_min_score: float = Field(default=0.25, ge=-1, le=1)
    llm_max_input_chars: int = Field(default=120000, ge=2000, le=500000)
    llm_timeout_seconds: float = Field(default=90, ge=1, le=300)
    run_timeout_seconds: float = Field(default=600, ge=10, le=1800)
    llm_max_output_tokens: int = Field(default=5000, ge=512, le=12000)
    # Only sent to the stable Gemini 2.5 Flash/Flash-Lite text models.
    gemini_thinking_budget: int = Field(default=1024, ge=-1, le=24576)
    gemini_thinking_level: Literal["minimal", "low", "medium", "high"] = "low"
    llm_max_calls: int = Field(default=12, ge=3, le=30)
    llm_max_total_tokens: int = Field(default=60000, ge=1000, le=500000)
    llm_requests_per_minute: int = Field(default=20, ge=1, le=1000)
    cache_enabled: bool = True
    cache_ttl_seconds: int = Field(default=86400, ge=0, le=604800)
    cache_max_entries: int = Field(default=1000, ge=10, le=10000)
    rerank_candidates: int = Field(default=20, ge=4, le=60)
    max_concurrent_runs: int = Field(default=2, ge=1, le=8)
    max_pending_runs: int = Field(default=12, ge=1, le=100)
    max_upload_mb: int = Field(default=12, ge=1, le=50)
    max_document_chars: int = Field(default=350000, ge=1000, le=1000000)
    max_pages: int = Field(default=300, ge=1, le=1000)
    max_chunks: int = Field(default=20000, ge=10, le=100000)
    chunk_chars: int = Field(default=1600, ge=300, le=2400)
    chunk_overlap: int = Field(default=180, ge=0, le=600)
    chunk_strategy: Literal["chars", "tokens"] = "chars"
    chunk_tokens: int = Field(default=384, ge=64, le=4096)
    chunk_token_overlap: int = Field(default=48, ge=0, le=512)
    # Optional password for shared lab deployments; no password needed on localhost.
    access_token: SecretStr = SecretStr("")
    allowed_hosts: list[str] = ["localhost", "127.0.0.1", "testserver"]

    @field_validator(
        "openai_base_url",
        "ollama_base_url",
        "embedding_base_url",
        "deepseek_base_url",
        "cohere_base_url",
    )
    @classmethod
    def endpoint(cls, value: str) -> str:
        url = urlsplit(value)
        if url.scheme not in {"http", "https"} or not url.hostname:
            raise ValueError("Provider URL must be http(s)")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("Provider URL must not contain credentials, query or fragment")
        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_configuration(self) -> "Settings":
        if (
            self.resolved_reranker_provider == "cohere"
            and not self.cohere_api_key.get_secret_value()
        ):
            raise ValueError("RA_COHERE_API_KEY is required for Cohere Rerank")
        if self.resolved_reranker_provider == "local" and not self.reranker_model.strip():
            raise ValueError("RA_RERANKER_MODEL is required for a local reranker")
        if self.chunk_overlap >= self.chunk_chars:
            raise ValueError("chunk_overlap must be smaller than chunk_chars")
        if self.chunk_token_overlap >= self.chunk_tokens:
            raise ValueError("chunk_token_overlap must be smaller than chunk_tokens")
        if (
            self.chunk_strategy == "tokens"
            and self.chunk_tokens + 16 > self.embedding_max_seq_length
        ):
            raise ValueError("Reserve at least 16 tokens for embedding prefixes and special tokens")
        providers = {self.llm_provider, *(r.provider for r in self.agent_routes.values())}
        if "gemini" in providers and not self.gemini_api_key.get_secret_value():
            raise ValueError("RA_GEMINI_API_KEY is required for Gemini")
        if "openai" in providers and not self.openai_api_key.get_secret_value():
            raise ValueError("RA_OPENAI_API_KEY is required (or a gateway token)")
        if "deepseek" in providers and not self.deepseek_api_key.get_secret_value():
            raise ValueError("RA_DEEPSEEK_API_KEY is required for DeepSeek")
        if self.embedding_provider == "openai":
            if not self.embedding_api_key.get_secret_value():
                raise ValueError("RA_EMBEDDING_API_KEY is required")
            if self.embedding_model == "intfloat/multilingual-e5-small":
                raise ValueError("Set RA_EMBEDDING_MODEL to your API's embedding model")
        return self

    @property
    def resolved_reranker_provider(self) -> str:
        if self.reranker_provider == "auto":
            return "local" if self.reranker_model.strip() else "none"
        return self.reranker_provider

    @property
    def resolved_reranker_model(self) -> str:
        if self.resolved_reranker_provider == "none":
            return ""
        return self.reranker_model.strip() or "rerank-v4.0-pro"

    @property
    def resolved_model(self) -> str:
        return (
            self.llm_model
            or {
                "demo": "extractive-demo",
                "gemini": "gemini-3.5-flash-lite",
                "openai": "gpt-4o-mini",
                "ollama": "qwen3:8b",
                "deepseek": "deepseek-flash",
            }[self.llm_provider]
        )

    def for_agent(self, role: AgentRole) -> "Settings":
        route = self.agent_routes.get(role, AgentRoute())
        provider = route.provider or self.llm_provider
        values: dict[str, object] = {"llm_provider": provider}
        if route.model:
            values["llm_model"] = route.model
        elif provider != self.llm_provider:
            values["llm_model"] = ""
        if route.max_output_tokens is not None:
            values["llm_max_output_tokens"] = route.max_output_tokens
        if route.thinking_level:
            values["gemini_thinking_level"] = route.thinking_level
        resolved = self.model_copy(update=values)
        if (
            resolved.llm_provider == "gemini"
            and resolved.resolved_model.startswith(("gemini-3.8", "gemini-3.7"))
            and resolved.gemini_thinking_level == "minimal"
        ):
            raise ValueError("Gemini 3.7/3.8 Flash does not support minimal thinking; use low")
        return resolved

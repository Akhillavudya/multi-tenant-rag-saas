"""Application configuration.

All environment access happens HERE. No other module calls os.getenv, so the
set of knobs this app has is exactly the set of fields below — and a typo in
.env fails loudly at startup instead of silently becoming None mid-request.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),  # works whether run from repo root or backend/
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # tolerate frontend/worker keys sharing the same .env
    )

    # --- App ---
    app_env: Literal["development", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Datastores ---
    database_url: str = "postgresql+psycopg://ragsaas:ragsaas@localhost:5432/ragsaas"
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "chunks"

    # --- Provider selection (Phase 1) ---
    llm_provider: Literal["gemini", "groq", "ollama", "anthropic", "openai"] = "gemini"
    embedding_provider: Literal["bge_local", "openai"] = "bge_local"
    rerank_provider: Literal["bge_local", "cohere", "noop"] = "bge_local"

    # --- Provider credentials (Phase 1) ---
    gemini_api_key: str | None = None
    groq_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    cohere_api_key: str | None = None

    # --- Auth (Phase 2) ---
    clerk_secret_key: str | None = None
    clerk_publishable_key: str | None = None
    clerk_jwks_url: str | None = None
    # The `iss` claim we require on every session token. Verifying it is what
    # stops a validly-signed token from a DIFFERENT Clerk instance (anyone can
    # create one for free) being accepted here — the signature would check out
    # against that instance's JWKS, so the issuer is the real tenant boundary.
    clerk_issuer: str | None = None
    # How long a fetched JWKS document is trusted before refetching. Clerk
    # rotates signing keys; an unknown `kid` forces an immediate refetch anyway,
    # so this only bounds how long a REVOKED key stays accepted.
    clerk_jwks_cache_seconds: int = 3600

    # --- Chat (Phase 3) ---
    # Where an LLM failure falls back to. `None` disables fallback entirely,
    # which is the honest setting when only one provider is configured — a
    # fallback that cannot authenticate turns one clean error into two.
    # Deliberately a separate knob from `llm_provider`: pointing both at the
    # same vendor would "fall back" onto the outage we are trying to escape.
    llm_fallback_provider: (
        Literal["gemini", "groq", "ollama", "anthropic", "openai"] | None
    ) = "groq"

    # How many earlier messages of a conversation are replayed to the model.
    # Counted in messages, not turns: 6 is three question/answer pairs. Bounded
    # because history is resent on EVERY request, so an unbounded transcript
    # grows the prompt (and the latency, and the cost) without limit.
    chat_history_messages: int = 6

    # Per-widget-key rate limit, as a token bucket (see security/rate_limit.py).
    # `per_minute` is the sustained refill rate; `burst` is how many requests
    # may arrive at once before throttling starts. Burst > 0 matters because
    # real users send a question, read, then send another — a strict
    # one-per-3-seconds limiter punishes that pattern for no benefit.
    chat_rate_limit_per_minute: int = 20
    chat_rate_limit_burst: int = 10

    # Browser origins allowed to call the API. Default "*" because the widget
    # is embedded on customers' own domains, which we do not know in advance.
    # Safe here only because authentication is a HEADER (bot key or bearer
    # token), never a cookie: a hostile page cannot make the browser attach a
    # credential it does not already have. If cookie auth is ever introduced,
    # this must become an explicit list on the same day.
    cors_origins: list[str] = ["*"]

    # --- Uploads (Phase 2) ---
    # Local disk in dev; Phase 7 swaps this for object storage on the Oracle
    # box. Kept as a setting rather than a constant so the worker and the API,
    # which are separate processes, can be pointed at the same shared path.
    upload_dir: Path = Path(".data/uploads")
    # Rejected before the bytes are written, not after. 25MB is roughly a
    # 500-page PDF; the ceiling exists so one upload cannot fill the disk.
    max_upload_bytes: int = 25 * 1024 * 1024

    # --- Billing (Phase 6) ---
    stripe_secret_key: str | None = None
    stripe_webhook_secret: str | None = None

    # --- Derived ---
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached so .env is parsed once per process, and so tests can override the
    whole config by clearing the cache rather than monkeypatching os.environ."""
    return Settings()


settings = get_settings()

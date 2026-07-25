"""Provider factory: env configuration -> live provider instances.

This is the only module that knows which vendors exist. `app/core/` asks for
`get_llm()` and receives something satisfying `LLMProvider`; it never learns
whether Gemini, Groq or Ollama answered. That indirection is what makes Hard
Rule #2 enforceable rather than aspirational — without it, every call site would
import a vendor and `LLM_PROVIDER=groq` would stop being a config change.

Three design decisions worth stating, because each is load-bearing:

1. **Registries are dicts of builder callables, not if/elif chains.** The set of
   valid providers becomes one readable table, adding a vendor is one line, and
   `available_*()` comes for free (the API layer needs it in Phase 4 to populate
   the model dropdown).
2. **Every vendor import happens INSIDE its builder.** Importing
   `sentence_transformers` pulls all of torch: seconds of startup and ~1GB of
   RSS. A module-level import here would charge that to every process — web,
   worker, CLI, test — no matter which provider is configured. This file must
   stay cheap to import.
3. **Instances are cached on the RESOLVED name, not on the call.** The local
   embedder and reranker each hold model weights in memory; building two would
   double that for no benefit. The public functions resolve `None` to the
   configured default first, so `get_embedder()` and `get_embedder("bge_local")`
   return the same object rather than two copies of ~130MB of weights.

The optional `name` argument on each getter exists for tests and for Phase 3's
Gemini->Groq fallback, which needs to build a specific second provider on demand.
It is deliberately NOT plumbed through to request handlers: per-bot model
overrides arrive in Phase 2 via the `model_config` column, and routing that
through here would let a request body choose a provider.
"""

import logging
from collections.abc import Callable
from functools import cache

from app.config import Settings, settings
from app.providers.base import (
    EmbeddingProvider,
    LLMProvider,
    ProviderAuthError,
    ProviderError,
    ProviderInvalidRequestError,
    RerankProvider,
)

logger = logging.getLogger(__name__)

_FACTORY = "factory"  # provider label for errors raised before a vendor is known


def _require(value: str | None, env_var: str, provider: str) -> str:
    """Return a non-empty credential or fail with a message naming the env var.

    Settings types credentials as `str | None` because most are unset in any
    given deployment; providers require `str`. This is where that gap closes,
    and it closes with the name of the variable to go and set — an error that
    says `GROQ_API_KEY is not set` costs nothing and saves a support round-trip.
    """
    if not value:
        raise ProviderAuthError(f"{env_var} is not set", provider=provider)
    return value


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------


def _build_gemini(cfg: Settings) -> LLMProvider:
    from app.providers.llm.gemini import GeminiLLM

    return GeminiLLM(api_key=_require(cfg.gemini_api_key, "GEMINI_API_KEY", "gemini"))


def _build_groq(cfg: Settings) -> LLMProvider:
    from app.providers.llm.groq_llm import GroqLLM

    return GroqLLM(api_key=_require(cfg.groq_api_key, "GROQ_API_KEY", "groq"))


def _build_ollama(cfg: Settings) -> LLMProvider:
    from app.providers.llm.ollama import OllamaLLM

    # No _require: ollama_base_url has a working default and no credential.
    return OllamaLLM(base_url=cfg.ollama_base_url)


def _build_anthropic(cfg: Settings) -> LLMProvider:
    from app.providers.llm.anthropic_llm import AnthropicLLM

    return AnthropicLLM(
        api_key=_require(cfg.anthropic_api_key, "ANTHROPIC_API_KEY", "anthropic")
    )


def _build_openai_llm(cfg: Settings) -> LLMProvider:
    from app.providers.llm.openai_llm import OpenAILLM

    return OpenAILLM(api_key=_require(cfg.openai_api_key, "OPENAI_API_KEY", "openai"))


_LLM_BUILDERS: dict[str, Callable[[Settings], LLMProvider]] = {
    "gemini": _build_gemini,
    "groq": _build_groq,
    "ollama": _build_ollama,
    "anthropic": _build_anthropic,
    "openai": _build_openai_llm,
}


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------


def _build_bge_embedder(cfg: Settings) -> EmbeddingProvider:
    from app.providers.embeddings.bge_local import BGELocalEmbedder

    return BGELocalEmbedder()


def _build_openai_embedder(cfg: Settings) -> EmbeddingProvider:
    from app.providers.embeddings.openai_embed import OpenAIEmbedder

    return OpenAIEmbedder(
        api_key=_require(cfg.openai_api_key, "OPENAI_API_KEY", "openai")
    )


_EMBEDDING_BUILDERS: dict[str, Callable[[Settings], EmbeddingProvider]] = {
    "bge_local": _build_bge_embedder,
    "openai": _build_openai_embedder,
}


# --------------------------------------------------------------------------
# Rerank
# --------------------------------------------------------------------------


def _build_bge_reranker(cfg: Settings) -> RerankProvider:
    from app.providers.rerank.bge_local import BGELocalReranker

    return BGELocalReranker()


def _build_noop_reranker(cfg: Settings) -> RerankProvider:
    from app.providers.rerank.noop import NoopReranker

    return NoopReranker()


def _build_cohere_reranker(cfg: Settings) -> RerankProvider:
    from app.providers.rerank.cohere_rerank import CohereReranker

    return CohereReranker(
        api_key=_require(cfg.cohere_api_key, "COHERE_API_KEY", "cohere")
    )


_RERANK_BUILDERS: dict[str, Callable[[Settings], RerankProvider]] = {
    "bge_local": _build_bge_reranker,
    "cohere": _build_cohere_reranker,
    "noop": _build_noop_reranker,
}


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def _unknown(
    kind: str, name: str, valid: dict[str, object]
) -> ProviderInvalidRequestError:
    """Error for a name with no builder.

    Config's Literal types already reject bad values from .env at startup, so
    this fires only on a direct call with a bad name — a code bug. Listing the
    valid options turns a typo into a five-second fix.
    """
    return ProviderInvalidRequestError(
        f"unknown {kind} provider {name!r}; valid: {', '.join(sorted(valid))}",
        provider=_FACTORY,
    )


# `functools.cache` memoizes RETURN VALUES ONLY — a raise stores nothing, so a
# build that failed on a missing credential is retried on the next call rather
# than poisoning the cache for the process lifetime. That is the behaviour we
# want (fix .env, no restart needed) but it is worth stating, because the
# opposite assumption is the natural one.
@cache
def _llm(name: str) -> LLMProvider:
    if (build := _LLM_BUILDERS.get(name)) is None:
        raise _unknown("LLM", name, _LLM_BUILDERS)
    logger.info("building LLM provider", extra={"provider": name})
    return build(settings)


@cache
def _embedder(name: str) -> EmbeddingProvider:
    if (build := _EMBEDDING_BUILDERS.get(name)) is None:
        raise _unknown("embedding", name, _EMBEDDING_BUILDERS)
    logger.info("building embedding provider", extra={"provider": name})
    return build(settings)


@cache
def _reranker(name: str) -> RerankProvider:
    if (build := _RERANK_BUILDERS.get(name)) is None:
        raise _unknown("rerank", name, _RERANK_BUILDERS)
    logger.info("building rerank provider", extra={"provider": name})
    return build(settings)


def get_llm(name: str | None = None) -> LLMProvider:
    """Return the configured LLM, or a specific one by name.

    Cached per resolved name for the process lifetime.
    """
    return _llm(name or settings.llm_provider)


@cache
def _llm_with_fallback(primary_name: str, secondary_name: str | None) -> LLMProvider:
    """Build the primary, optionally wrapped in failover. Cached like the rest.

    Degrades to the bare primary in three cases, each of which is a
    configuration statement rather than an error:

    - no fallback configured (`LLM_FALLBACK_PROVIDER=` empty),
    - fallback equals primary, which would retry the outage we are escaping,
    - the fallback cannot be built — almost always a missing API key.

    That last one is the interesting choice. Refusing to start because the
    *backup* has no key would take down a service whose primary works perfectly;
    logging loudly and running unprotected is the lesser harm. The log line is
    the thing that stops "unprotected" from becoming permanent.
    """
    primary = _llm(primary_name)
    if not secondary_name or secondary_name == primary_name:
        return primary

    try:
        secondary = _llm(secondary_name)
    except ProviderError as exc:
        logger.warning(
            "fallback LLM provider unavailable; running without failover",
            extra={
                "primary": primary_name,
                "fallback": secondary_name,
                "error": str(exc),
            },
        )
        return primary

    from app.providers.llm.fallback import FallbackLLM

    logger.info(
        "LLM failover enabled",
        extra={"primary": primary_name, "fallback": secondary_name},
    )
    return FallbackLLM(primary, secondary)


def get_llm_with_fallback() -> LLMProvider:
    """The LLM used to answer a chat request: configured provider plus failover.

    Separate from `get_llm()` because failover is not always wanted. Ingestion
    does not use an LLM at all, and the CLI deliberately exercises exactly the
    provider named in the env so that "swapping providers is one env var" stays
    a claim you can verify rather than one a fallback could quietly mask.
    """
    return _llm_with_fallback(settings.llm_provider, settings.llm_fallback_provider)


def get_embedder(name: str | None = None) -> EmbeddingProvider:
    """Return the configured embedder, or a specific one by name.

    Cached per resolved name: this instance owns the model weights, so callers
    must NOT construct their own. Two instances means two copies in RAM.
    """
    return _embedder(name or settings.embedding_provider)


def get_reranker(name: str | None = None) -> RerankProvider:
    """Return the configured reranker, or a specific one by name.

    Cached per resolved name; see get_embedder on why that matters.
    """
    return _reranker(name or settings.rerank_provider)


def available_llms() -> list[str]:
    """Names the factory can build. Phase 4's settings UI reads this rather than
    hardcoding a dropdown that drifts out of sync with the backend."""
    return sorted(_LLM_BUILDERS)


def available_embedders() -> list[str]:
    return sorted(_EMBEDDING_BUILDERS)


def available_rerankers() -> list[str]:
    return sorted(_RERANK_BUILDERS)


def reset_provider_cache() -> None:
    """Drop every cached instance.

    For tests that change provider settings between cases. Without it the first
    test to call a getter pins that provider for the whole session and later
    tests silently assert against the wrong object. Also releases the local
    models' weights, which matters when a suite would otherwise hold several
    hundred MB it no longer needs.
    """
    _llm.cache_clear()
    _llm_with_fallback.cache_clear()
    _embedder.cache_clear()
    _reranker.cache_clear()

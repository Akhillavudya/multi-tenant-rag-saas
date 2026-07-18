"""Provider contracts and error types.

Every AI vendor sits behind one of the three Protocols below. Business logic in
`app/core/` depends ONLY on these shapes, never on a vendor SDK — that is what
makes `LLM_PROVIDER=groq` a config change instead of a refactor.

Protocol (not ABC) on purpose: implementations satisfy these structurally, so
`llm/gemini.py` never imports this module. The dependency arrow points one way
(core -> contract), and vendor code stays a leaf of the tree.

The exceptions, by contrast, ARE imported by implementations: every provider
translates its SDK's private error types into these, so the API layer has one
small set of failures to map to HTTP codes.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class ProviderError(Exception):
    """Base for every provider failure.

    `retryable` is the important field: it is what lets Phase 3 fall back
    Gemini -> Groq automatically. A rate limit or a timeout is worth retrying
    on another provider; a malformed prompt or a bad API key is not, and
    retrying it just burns the fallback's quota too.
    """

    retryable: bool = False

    def __init__(self, message: str, *, provider: str) -> None:
        super().__init__(message)
        self.provider = provider

    def __str__(self) -> str:
        return f"[{self.provider}] {super().__str__()}"


class ProviderAuthError(ProviderError):
    """Missing/invalid credentials. Never retryable — the key is wrong."""


class ProviderRateLimitError(ProviderError):
    """Quota or requests-per-minute exceeded. Free tiers hit this often."""

    retryable = True


class ProviderTimeoutError(ProviderError):
    """Provider did not respond within the configured deadline."""

    retryable = True


class ProviderUnavailableError(ProviderError):
    """5xx, connection refused, model still loading (Ollama does this)."""

    retryable = True


class ProviderInvalidRequestError(ProviderError):
    """We sent something the provider rejected: context too long, bad model
    name, unsupported parameter. Our bug, not theirs — never retryable."""


# --------------------------------------------------------------------------
# Shared value types
# --------------------------------------------------------------------------

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    """One turn of a conversation.

    Frozen because a message is a historical fact: once it is in the transcript
    nothing should mutate it. Deliberately vendor-neutral — Gemini calls the
    assistant role "model" and nests content in `parts`; translating to that
    shape is each provider's private job, not the caller's.
    """

    role: Role
    content: str


# (index_into_the_input_list, relevance_score). Indexes rather than the text
# itself so the caller can map a result straight back to its own chunk objects
# (with document_id, page, source) without string matching.
ScoredIndex = tuple[int, float]


# --------------------------------------------------------------------------
# Contracts
# --------------------------------------------------------------------------


@runtime_checkable
class LLMProvider(Protocol):
    """Generates the final grounded answer.

    Stream-only by design: a chat UI must show tokens as they arrive, and a
    non-streaming call is trivially built from a stream (join the chunks) while
    the reverse is impossible. One method, so there is one thing to implement
    when a new vendor appears.
    """

    name: str

    async def stream(
        self,
        system: str,
        messages: list[Message],
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Yield answer text incrementally. Raises ProviderError subclasses."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors for similarity search."""

    name: str
    model: str
    dimension: int
    """Vector width (bge-small=384, openai-3-large=3072). Qdrant collections are
    created with a fixed size, so this value is written to the DB per chatbot;
    changing embedder means re-embedding, never silently mixing dimensions."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed chunks at ingestion time. Returns one vector per input,
        in input order — callers zip results back onto their chunks positionally."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single search query.

        Separate from embed_documents because asymmetric models (bge, e5)
        require a prefix like "Represent this sentence for searching..." on the
        query but not on the documents. Same model, different instruction —
        using the wrong one measurably hurts retrieval.
        """
        ...


@runtime_checkable
class RerankProvider(Protocol):
    """Second-stage relevance scoring over retrieval candidates.

    Vector search is a bi-encoder: query and document are embedded separately,
    so their comparison is approximate but precomputable and fast. A reranker is
    a cross-encoder: it reads query and document TOGETHER, which is far more
    accurate and far too slow to run over a whole collection. Hence the shape of
    the pipeline — search wide (top-40), rerank narrow (top-6).
    """

    name: str

    async def rerank(
        self,
        query: str,
        docs: list[str],
        top_k: int,
    ) -> list[ScoredIndex]:
        """Return at most `top_k` (index, score) pairs, highest score first.

        Indexes refer to positions in `docs`. Scores are comparable only within
        one call and one provider — they are for ordering, not for thresholding
        across providers.
        """
        ...

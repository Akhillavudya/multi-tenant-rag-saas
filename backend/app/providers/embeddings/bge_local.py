"""Local embedding provider: BAAI/bge-small-en-v1.5 via sentence-transformers.

Default embedder for the whole project (PLAN §2): free, offline, 384-dim, and
good enough that the retrieval quality ceiling is set by chunking rather than by
the model. Runs on CPU; ~130MB of weights are downloaded once to the HuggingFace
cache and reused.

Two things here are easy to get wrong and quietly halve retrieval quality:
normalization (see `_encode`) and the asymmetric query prefix (see QUERY_PREFIX).
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from app.providers.base import (
    ProviderError,
    ProviderInvalidRequestError,
    ProviderUnavailableError,
)

if TYPE_CHECKING:
    # Imported for types only — keeps `import app.providers` cheap, since
    # importing sentence_transformers pulls in all of torch (seconds, ~1GB RSS).
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIMENSION = 384

# BGE is an *asymmetric* model: it was trained with this instruction glued to
# queries and nothing glued to passages. Embedding both sides the same way lands
# them in slightly different regions of the space and measurably hurts recall on
# short queries. This is why the contract has two methods instead of one.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Chosen for CPU: large enough to amortise per-batch overhead, small enough that
# peak memory stays flat on a 512-token model. Tune with real documents, not
# intuition.
DEFAULT_BATCH_SIZE = 32


class BGELocalEmbedder:
    """EmbeddingProvider backed by a locally-loaded sentence-transformers model.

    Satisfies the Protocol structurally — it deliberately does not inherit from
    it, so this vendor-facing file stays a leaf of the dependency graph.
    """

    name = "bge_local"
    model = MODEL_NAME
    dimension = DIMENSION

    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None
        # Guards first-time load: without it, N concurrent requests arriving
        # before warmup each start their own model load — N copies of the
        # weights in RAM and N slow requests.
        self._load_lock = asyncio.Lock()

    # -- model lifecycle ----------------------------------------------------

    async def warmup(self) -> None:
        """Load the model ahead of first use.

        Called from FastAPI's lifespan (and by the Celery worker at boot) so the
        first real request doesn't pay the several-second load. Safe to call
        repeatedly.
        """
        await self._get_model()

    async def _get_model(self) -> "SentenceTransformer":
        if self._model is not None:
            return self._model

        async with self._load_lock:
            # Re-check inside the lock: another coroutine may have loaded it
            # while we were waiting (the classic double-checked locking shape).
            if self._model is not None:
                return self._model

            logger.info("loading embedding model", extra={"model": MODEL_NAME})
            try:
                self._model = await asyncio.to_thread(self._load_model_sync)
            except Exception as exc:
                # Almost always: no network on a cold cache, or out of disk.
                raise ProviderUnavailableError(
                    f"could not load {MODEL_NAME}: {exc}", provider=self.name
                ) from exc

            actual = self._model.get_sentence_embedding_dimension()
            if actual != DIMENSION:
                # A mismatch means the model id resolved to something else. Fail
                # loudly: silently writing 768-dim vectors into a 384-dim Qdrant
                # collection is a corruption bug that surfaces days later.
                raise ProviderError(
                    f"{MODEL_NAME} reports dimension {actual}, expected {DIMENSION}",
                    provider=self.name,
                )

            logger.info("embedding model ready", extra={"model": MODEL_NAME})
            return self._model

    def _load_model_sync(self) -> "SentenceTransformer":
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(MODEL_NAME, device="cpu")

    # -- EmbeddingProvider --------------------------------------------------

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunks at ingestion time. Output order matches input order, so
        callers can zip vectors back onto their chunk metadata positionally."""
        if not texts:
            return []
        if any(not t.strip() for t in texts):
            # An all-whitespace chunk yields a near-zero vector that matches
            # everything weakly — garbage that pollutes every future search.
            # The chunker should never emit these; if one arrives, it's a bug.
            raise ProviderInvalidRequestError(
                "cannot embed empty or whitespace-only text", provider=self.name
            )
        return await self._encode(texts)

    async def embed_query(self, text: str) -> list[float]:
        """Embed one search query, with the BGE instruction prefix applied."""
        if not text.strip():
            raise ProviderInvalidRequestError(
                "cannot embed an empty query", provider=self.name
            )
        vectors = await self._encode([QUERY_PREFIX + text])
        return vectors[0]

    # -- internals ----------------------------------------------------------

    async def _encode(self, texts: list[str]) -> list[list[float]]:
        model = await self._get_model()
        try:
            # to_thread: encode() is blocking CPU work. Awaiting it directly
            # would block the event loop for the whole batch, stalling every
            # other request in the process. torch drops the GIL during the
            # actual matrix math, so the thread genuinely runs in parallel.
            return await asyncio.to_thread(self._encode_sync, model, texts)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderUnavailableError(
                f"embedding failed: {exc}", provider=self.name
            ) from exc

    def _encode_sync(
        self, model: "SentenceTransformer", texts: list[str]
    ) -> list[list[float]]:
        # normalize_embeddings=True gives unit-length vectors, which makes
        # cosine similarity identical to the dot product. Qdrant then does the
        # cheaper operation, and scores stay in a stable 0..1-ish range instead
        # of drifting with text length.
        embeddings: Any = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # numpy float32 -> plain Python floats: the Qdrant client and JSON both
        # need native types, and np.float32 is not JSON-serialisable.
        return embeddings.tolist()

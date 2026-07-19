"""Local reranking provider: BAAI/bge-reranker-base via sentence-transformers.

Second stage of the retrieval pipeline (PLAN §1): vector search returns 40 cheap
candidates, this narrows them to the 6 that actually answer the question.

Why a second model at all: the embedder is a *bi-encoder* — query and document
are encoded independently, so similarity is a dot product between two vectors
that were computed without knowledge of each other. That is what makes it fast
enough to index millions of chunks, and also what makes it approximate. This is a
*cross-encoder*: it runs the transformer over the query and document jointly, so
attention can compare every query token against every document token. Much more
accurate, and O(candidates) forward passes — affordable on 40, absurd on 40,000.

RERANK_PROVIDER=noop turns this off, which is how PLAN §6 asks us to prove the
reranker earns its latency.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from app.providers.base import (
    ProviderError,
    ProviderInvalidRequestError,
    ProviderUnavailableError,
    ScoredIndex,
)

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# MEASURED, not assumed. bge-reranker-v2-m3 is XLM-RoBERTa-LARGE — 568M params,
# 24 layers — and on CPU it scored 7 candidates in 11.4s with the model already
# resident. That is not a cold-start cost that a long-lived server absorbs; it is
# the steady-state price of every question. bge-reranker-base is 278M / 12 layers
# and does the same 7 in 1.9s: 5.9x faster for the same pipeline.
#
# The tradeoff is real and deliberate: v2-m3's "m3" is MULTILINGUAL, and base is
# English/Chinese only. We accept that for now because a 13s answer is not a
# product. Revisit when reranking runs on a GPU, where v2-m3's cost largely
# disappears and its language coverage becomes free.
MODEL_NAME = "BAAI/bge-reranker-base"

# Truncation limit for the concatenated (query, document) pair. 512 is also
# bge-reranker-base's architectural ceiling, so this cannot be raised without
# changing model. It does not bind us: chunks are DEFAULT_MAX_TOKENS=400
# (chunking.py), leaving ~100 tokens for the query — no truncation in practice.
#
# Measured: dropping this to 256 saved only 15% (1930ms -> 1639ms), so sequence
# length is NOT the lever here; model depth is. Not worth the truncation risk.
MAX_LENGTH = 512

# Pairs per forward pass. Reranking is the slowest step in the chat path on CPU,
# so this is the knob to tune first if p95 latency disappoints.
DEFAULT_BATCH_SIZE = 16


class BGELocalReranker:
    """RerankProvider backed by a locally-loaded cross-encoder.

    Structurally satisfies the Protocol; deliberately does not inherit it.
    """

    name = "bge_local"
    model = MODEL_NAME

    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self._batch_size = batch_size
        self._model: CrossEncoder | None = None
        self._load_lock = asyncio.Lock()

    # -- model lifecycle ----------------------------------------------------

    async def warmup(self) -> None:
        """Preload the model so the first chat request doesn't pay for it."""
        await self._get_model()

    async def _get_model(self) -> "CrossEncoder":
        if self._model is not None:
            return self._model

        async with self._load_lock:
            # Double-check inside the lock; see the same pattern in the embedder.
            if self._model is not None:
                return self._model

            logger.info("loading reranker model", extra={"model": MODEL_NAME})
            try:
                self._model = await asyncio.to_thread(self._load_model_sync)
            except Exception as exc:
                raise ProviderUnavailableError(
                    f"could not load {MODEL_NAME}: {exc}", provider=self.name
                ) from exc

            logger.info("reranker model ready", extra={"model": MODEL_NAME})
            return self._model

    def _load_model_sync(self) -> "CrossEncoder":
        from sentence_transformers import CrossEncoder

        return CrossEncoder(MODEL_NAME, max_length=MAX_LENGTH, device="cpu")

    # -- RerankProvider -----------------------------------------------------

    async def rerank(
        self,
        query: str,
        docs: list[str],
        top_k: int,
    ) -> list[ScoredIndex]:
        """Score every doc against the query; return the best `top_k` as
        (original_index, score), highest first.

        Returns indexes rather than text so the caller can map straight back to
        its own chunk objects — the ones carrying document_id, page and source
        for citations — with no string matching.
        """
        if not docs:
            return []
        if top_k <= 0:
            raise ProviderInvalidRequestError(
                f"top_k must be positive, got {top_k}", provider=self.name
            )
        if not query.strip():
            raise ProviderInvalidRequestError(
                "cannot rerank against an empty query", provider=self.name
            )

        scores = await self._score(query, docs)

        # Pair each score with its ORIGINAL index before sorting — that mapping
        # is the entire contract, and it is the easiest thing in this file to
        # get wrong.
        ranked = sorted(
            enumerate(scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [(index, float(score)) for index, score in ranked[:top_k]]

    # -- internals ----------------------------------------------------------

    async def _score(self, query: str, docs: list[str]) -> list[float]:
        model = await self._get_model()
        try:
            return await asyncio.to_thread(self._score_sync, model, query, docs)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderUnavailableError(
                f"reranking failed: {exc}", provider=self.name
            ) from exc

    def _score_sync(
        self, model: "CrossEncoder", query: str, docs: list[str]
    ) -> list[float]:
        # The cross-encoder consumes (query, document) PAIRS — this pairing is
        # the whole difference from the embedder, which never sees the two texts
        # at the same time.
        pairs = [(query, doc) for doc in docs]

        # MEASURED on sentence-transformers 5.6.0: scores come back in 0..1
        # (e.g. 0.977 for a clear match, 0.000 for an unrelated chunk), because
        # predict() applies the model's configured default activation — a
        # sigmoid for this model — to the head's raw logit.
        #
        # Do not rely on that range. It is a property of the library version and
        # the model config, not of the contract; older sentence-transformers
        # returned the bare logit instead. Sort by these values, never threshold
        # on them, and never compare them across providers.
        scores: Any = model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return scores.tolist()

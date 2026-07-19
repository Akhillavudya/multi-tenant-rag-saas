"""Retrieval: a question becomes the handful of chunks worth answering from.

The two-stage pipeline from PLAN §2 — search wide, rerank narrow:

    query -> embed_query -> vector search (top-40) -> rerank -> top-6

The two stages exist because they fail in opposite directions, and each is asked
for the thing it is good at.

**Stage 1 is a bi-encoder.** Query and document are embedded separately, so their
comparison is a dot product between two vectors that never saw each other. That
is cheap and precomputable — the document side was embedded at ingestion — but
approximate, both because the comparison is crude and because HNSW itself is an
approximate index. So stage 1 is asked for RECALL: get the right chunk somewhere
in 40, never mind where.

**Stage 2 is a cross-encoder.** The reranker reads query and document TOGETHER in
one forward pass, so every query token attends to every document token. Far more
accurate, and far too slow to run over a collection — it is O(candidates) model
passes, which is exactly why the candidate list is capped. Stage 2 is asked for
PRECISION: put the right chunk at rank 1.

Why not one stage? Vector search alone puts the right chunk in the top-40 but
often at rank 11, and only 6 reach the model. Reranking alone would mean a
cross-encoder pass per chunk in the collection, which is seconds per query.

There is deliberately NO score threshold in this file. Rerank scores are
provider-private — bge returns unbounded logits, Cohere returns 0..1 — so any
cutoff hardcoded here silently changes meaning when RERANK_PROVIDER changes, and
"nothing scored above 0.5" is indistinguishable from "the reranker was swapped".
Deciding that the context does not answer the question is the *prompt's* job
(see `generation.py`), where the model can read the chunks and say so.
"""

import logging
import time
from dataclasses import dataclass

from app.core.vectorstore import SearchHit, VectorStore
from app.providers.base import EmbeddingProvider, RerankProvider
from app.providers.factory import get_embedder, get_reranker

logger = logging.getLogger(__name__)

# Wide first stage. Big enough that a chunk the bi-encoder ranked poorly still
# survives to the reranker, small enough to stay one cross-encoder batch.
DEFAULT_CANDIDATES = 40

# What actually reaches the model. Bounded by context budget and by the fact
# that irrelevant chunks actively hurt: a model given six passages, four of them
# off-topic, answers worse than one given two good ones.
DEFAULT_TOP_K = 6


@dataclass(frozen=True)
class RetrievedChunk:
    """A search hit plus how it got here.

    `vector_rank` and `rerank_score` are kept rather than discarded because they
    are the only way to answer "is the reranker earning its place?" — the Phase 1
    acceptance criterion is that RERANK_PROVIDER=noop visibly degrades answers,
    and you cannot see that without knowing what moved.
    """

    hit: SearchHit
    vector_score: float
    vector_rank: int
    rerank_score: float | None = None

    @property
    def text(self) -> str:
        return self.hit.text

    @property
    def citation(self) -> str:
        return self.hit.citation


async def retrieve(
    query: str,
    *,
    org_id: str,
    chatbot_id: str,
    store: VectorStore,
    embedder: EmbeddingProvider | None = None,
    reranker: RerankProvider | None = None,
    candidates: int = DEFAULT_CANDIDATES,
    top_k: int = DEFAULT_TOP_K,
    document_id: str | None = None,
) -> list[RetrievedChunk]:
    """Return the `top_k` most relevant chunks for `query`, best first.

    `org_id` and `chatbot_id` are required keyword-only arguments and are passed
    straight through to `VectorStore.search`, which builds the mandatory filter.
    This function has no unfiltered path because it cannot construct one.

    Returns an empty list when the chatbot has no matching chunks at all. That
    is a real answer, not an error: a bot whose documents are still processing
    genuinely has nothing to say, and the caller renders that as "I don't have
    that information" rather than treating it as a failure.
    """
    if not query.strip():
        raise ValueError("query must not be empty")
    if top_k > candidates:
        # Silently clamping would hide a config error whose only symptom is
        # quietly fewer chunks than the caller asked for.
        raise ValueError(
            f"top_k ({top_k}) cannot exceed candidates ({candidates}); "
            "the reranker can only narrow the candidate list, never extend it"
        )

    embedder = embedder or get_embedder()
    reranker = reranker or get_reranker()
    log = {"org_id": org_id, "chatbot_id": chatbot_id}

    started = time.perf_counter()
    # embed_query, not embed_documents: bge is asymmetric and needs its
    # instruction prefix on the query side. The provider applies it; this layer
    # only has to call the right method.
    query_vector = await embedder.embed_query(query)
    embed_ms = _elapsed_ms(started)

    searched = time.perf_counter()
    hits = await store.search(
        org_id=org_id,
        chatbot_id=chatbot_id,
        query_vector=query_vector,
        limit=candidates,
        document_id=document_id,
    )
    search_ms = _elapsed_ms(searched)

    if not hits:
        logger.info(
            "retrieval found no candidates", extra={**log, "query_len": len(query)}
        )
        return []

    reranked = time.perf_counter()
    results = await _rerank(query, hits, reranker=reranker, top_k=top_k)
    rerank_ms = _elapsed_ms(reranked)

    logger.info(
        "retrieval complete",
        extra={
            **log,
            "candidates": len(hits),
            "returned": len(results),
            "embed_ms": embed_ms,
            "search_ms": search_ms,
            "rerank_ms": rerank_ms,
            # How far the top result moved. 0 means the reranker agreed with the
            # vector search; a large number is the reranker earning its keep.
            "top_moved_from_rank": results[0].vector_rank if results else None,
        },
    )
    return results


async def _rerank(
    query: str,
    hits: list[SearchHit],
    *,
    reranker: RerankProvider,
    top_k: int,
) -> list[RetrievedChunk]:
    """Score candidates with the cross-encoder and return the best `top_k`.

    The reranker returns (index, score) pairs pointing into the list we gave it,
    NOT the documents themselves. That is the whole reason the contract uses
    indexes: we map straight back onto our own `SearchHit` objects, keeping
    document_id, source and page, with no string matching against text we would
    then have to trust the provider not to have altered.
    """
    documents = [hit.text for hit in hits]
    scored = await reranker.rerank(query, documents, top_k)

    results: list[RetrievedChunk] = []
    for index, score in scored:
        if not 0 <= index < len(hits):
            # A provider returning an out-of-range index is a bug we must not
            # paper over: silently skipping it would return fewer chunks than
            # requested with no indication why.
            raise IndexError(
                f"reranker {reranker.name!r} returned index {index} "
                f"for {len(hits)} documents"
            )
        hit = hits[index]
        results.append(
            RetrievedChunk(
                hit=hit,
                vector_score=hit.score,
                # Position in the vector-search results, before reranking.
                # Preserved so the reranker's contribution is measurable.
                vector_rank=index,
                rerank_score=score,
            )
        )
    return results


def _elapsed_ms(since: float) -> int:
    """Milliseconds since a `perf_counter` reading.

    perf_counter rather than time(): it is monotonic, so an NTP correction
    mid-request cannot produce a negative duration in the logs.
    """
    return int((time.perf_counter() - since) * 1000)

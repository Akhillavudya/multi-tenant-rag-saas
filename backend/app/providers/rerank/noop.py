"""Pass-through reranker: keeps the retriever's own ordering.

This is a Null Object, not dead code. Two jobs:

1. **Measurement.** PLAN §6 requires proving that reranking earns its latency.
   Setting RERANK_PROVIDER=noop disables stage 2 while leaving every other line
   of the pipeline identical, so a difference in Recall@6 is attributable to the
   reranker and nothing else. Without it you would be comparing two different
   code paths and learning little.
2. **Degradation.** If the reranker model fails to load on a constrained box, a
   config change gives a working (if less precise) system rather than an outage.

It deliberately mirrors BGELocalReranker's validation exactly. If noop accepted
inputs that bge_local rejects, swapping providers would change error behaviour as
well as ranking behaviour — and the A/B comparison would be confounded by it.
"""

import logging

from app.providers.base import ProviderInvalidRequestError, ScoredIndex

logger = logging.getLogger(__name__)


class NoopReranker:
    """RerankProvider that trusts the retriever and only truncates."""

    name = "noop"

    async def warmup(self) -> None:
        """No model to load. Present so callers can warm any provider without
        first asking which kind it is."""

    async def rerank(
        self,
        query: str,
        docs: list[str],
        top_k: int,
    ) -> list[ScoredIndex]:
        """Return the first `top_k` docs in their original order.

        `query` is accepted and ignored — that ignoring IS the behaviour under
        test. Vector search already ordered these by cosine similarity, so
        "no reranking" correctly means "keep stage 1's ranking".
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

        kept = min(top_k, len(docs))

        # Scores descend from 1.0 so the contract ("highest first") still holds
        # under any re-sort by the caller. They encode POSITION, not relevance —
        # which is honest here, since for a pass-through the retriever's ordering
        # is the only relevance signal that exists. Never compare these against
        # a real reranker's scores.
        return [(i, 1.0 - i / len(docs)) for i in range(kept)]

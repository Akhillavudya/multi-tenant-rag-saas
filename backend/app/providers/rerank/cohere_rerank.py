"""Cohere rerank provider — PREMIUM STUB, not yet implemented.

Reserves the `cohere` slot for `RERANK_PROVIDER`: a hosted cross-encoder for
clients who want stage-two reranking without paying the local bge-reranker's CPU
cost on every single query. General stub reasoning (constructs successfully, SDK
absent from pyproject, honest raise rather than a fake return) is in
llm/anthropic_llm.py and is not repeated.

Worth noticing about this one: Cohere's API returns results as
`(index, relevance_score)` pairs already — the exact shape of our `ScoredIndex`
contract. The finished body is close to a direct passthrough. That is a small
piece of evidence that returning *indexes* rather than reordered text in
`base.py` was the right call and not merely convenient for bge: two independent
implementations landed on the same shape.

The one real trap is scoring scale, which is provider-private. bge's cross-
encoder emits unbounded logits (roughly -11..+11, sigmoid-able but not
probabilities); Cohere returns scores normalised to 0..1. Both rank correctly and
neither is comparable to the other. Any code that grows a "drop anything below
0.5" filter tuned against one provider will silently discard everything, or
nothing, against the other — which is why `base.py` states that scores are for
ordering, never for thresholding across providers.

To finish this file:
1. Add `cohere>=5.13.0` to pyproject dependencies.
2. `AsyncClientV2.rerank(model="rerank-v3.5", query=..., documents=texts,
   top_n=top_k)`. Read `.results[i].index` and `.results[i].relevance_score`.
3. Results come back already sorted by score descending, but sort defensively
   anyway — the contract promises ordering and one line is cheaper than
   trusting a vendor's undocumented stability.
4. Guard the empty-docs case before calling: an empty `documents` list is a 400
   from Cohere, while our contract says it is simply an empty result.
5. Translate `cohere.errors.*` by status code, same table as groq_llm.py.
"""

from app.providers.base import ProviderAuthError, ScoredIndex

MODEL_NAME = "rerank-v3.5"


class CohereReranker:
    """RerankProvider backed by Cohere's hosted rerank API. Not yet built.

    Constructs successfully on purpose; the failure is deferred to `rerank()`.
    See llm/anthropic_llm.py for the full reasoning.
    """

    name = "cohere"
    model = MODEL_NAME

    def __init__(self, api_key: str, model: str = MODEL_NAME) -> None:
        if not api_key:
            raise ProviderAuthError("COHERE_API_KEY is not set", provider=self.name)
        self._api_key = api_key
        self.model = model

    async def rerank(
        self,
        query: str,
        docs: list[str],
        top_k: int,
    ) -> list[ScoredIndex]:
        """Not implemented. See the module docstring for the completion steps.

        No unreachable `yield` here, unlike the LLM stubs: this is a plain
        coroutine rather than an async generator, so the raise already fires at
        the same moment the finished implementation would fail.
        """
        raise NotImplementedError(
            "the Cohere reranker is a stub; set RERANK_PROVIDER=bge_local or "
            "noop, or implement app/providers/rerank/cohere_rerank.py"
        )

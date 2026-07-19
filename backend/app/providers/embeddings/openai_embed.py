"""OpenAI embedding provider — PREMIUM STUB, not yet implemented.

Reserves the `openai` slot for `EMBEDDING_PROVIDER`, the paid upgrade from the
local bge-small default. General stub reasoning (constructs successfully, SDK
deliberately absent from pyproject, honest raise rather than a fake return) is in
llm/anthropic_llm.py and is not repeated.

One thing here is NOT deferrable, unlike in the LLM stubs: `dimension`. It is
read at configuration time to size the Qdrant collection and is written to the
per-bot row in Postgres (PLAN §3 rule 1). A placeholder value would create a
collection the finished provider physically cannot write into — a 1536-dim
vector rejected by a 384-dim collection — and the error would surface during a
migration, not here. So the constant below is the real one.

Note also what this file does NOT need: a query prefix. bge is an *asymmetric*
model, trained with an instruction glued to queries and nothing glued to
passages, which is the whole reason `EmbeddingProvider` has two methods instead
of one. OpenAI's models are symmetric — same encoding on both sides — so the
finished `embed_query` here is genuinely just `embed_documents([text])[0]`. The
two-method contract costs this provider one trivial line and saves bge from a
silent recall regression; that asymmetry in cost is why the contract is drawn
where it is.

To finish this file:
1. Add `openai>=1.60.0` to pyproject dependencies.
2. `AsyncOpenAI.embeddings.create(model=..., input=[...])`. Returns
   `.data[i].embedding`; the API preserves input order, but assert
   `len(data) == len(texts)` anyway rather than trusting it silently.
3. Batch: the endpoint accepts many inputs per call but caps total tokens
   (~300k). Chunk the input list — one request per document is a rate-limit
   incident waiting to happen at ingestion scale.
4. Translate `openai.APIStatusError` by `.status_code`, same table as
   groq_llm.py.
5. `count_tokens` needs `tiktoken`, not the API — it is a local BPE tokenizer:
   `tiktoken.encoding_for_model(MODEL_NAME).encode(text)`, and unlike the BERT
   family there are NO special tokens to add, so the length is the answer.
   Cache the encoding object; building it per call is genuinely slow.
"""

from app.providers.base import ProviderAuthError

MODEL_NAME = "text-embedding-3-small"

# The real value, not a placeholder — see the module docstring. text-embedding-3
# models support Matryoshka truncation via a `dimensions` parameter, so this can
# legitimately be lowered later to trade a little accuracy for storage. Doing so
# is a re-embed, never an in-place edit of this constant.
DIMENSION = 1536


class OpenAIEmbedder:
    """EmbeddingProvider backed by OpenAI's embeddings API. Not yet built.

    Constructs successfully on purpose; the failure is deferred to the embed
    methods. See llm/anthropic_llm.py for the full reasoning.
    """

    name = "openai"
    model = MODEL_NAME
    dimension = DIMENSION

    def __init__(self, api_key: str, model: str = MODEL_NAME) -> None:
        if not api_key:
            raise ProviderAuthError("OPENAI_API_KEY is not set", provider=self.name)
        self._api_key = api_key
        self.model = model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Not implemented. See the module docstring for the completion steps."""
        raise NotImplementedError(
            "the OpenAI embedder is a stub; set EMBEDDING_PROVIDER=bge_local, "
            "or implement app/providers/embeddings/openai_embed.py"
        )

    def count_tokens(self, text: str) -> int:
        """Not implemented. See completion step 5 in the module docstring.

        Deliberately raises rather than falling back to a character estimate or
        borrowing bge's tokenizer. Both would "work" and both would be wrong:
        token counts are model-specific, so a borrowed count silently mis-sizes
        every chunk, and the resulting overflow is truncated during encoding
        with no error at all. A stub that fails loudly is worth more than one
        that returns a plausible number.

        Note this method could in fact be finished today — `tiktoken` is a local
        tokenizer and needs no API key and no network. It is left raising so the
        class fails uniformly: a provider where two methods raise and a third
        quietly works invites exactly the half-configured deployment that passes
        a smoke test and dies during ingestion.
        """
        raise NotImplementedError(
            "the OpenAI embedder is a stub; set EMBEDDING_PROVIDER=bge_local, "
            "or implement app/providers/embeddings/openai_embed.py"
        )

    async def embed_query(self, text: str) -> list[float]:
        """Not implemented. See the module docstring for the completion steps.

        No unreachable `yield` here, unlike the LLM stubs: these are plain
        coroutines rather than async generators, so the raise already fires at
        the same moment the finished implementation would fail.
        """
        raise NotImplementedError(
            "the OpenAI embedder is a stub; set EMBEDDING_PROVIDER=bge_local, "
            "or implement app/providers/embeddings/openai_embed.py"
        )

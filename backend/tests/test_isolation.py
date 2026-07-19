"""THE isolation test — cross-tenant leakage must be impossible.

PLAN §6 calls this test sacred, and PLAN §7 runs it in CI on every push. Every
other test in this repo protects a feature; this one protects the promise that
makes the product sellable at all. A failure here is not a bug report, it is a
breach notification.

The threat model is specific. One Qdrant collection holds every tenant's chunks
(see `vectorstore.VectorStore` for why collection-per-tenant was rejected), so
isolation is application logic rather than physical separation. That buys
cheap onboarding and costs us the guarantee — this file is where we buy it back.

Four ways the guarantee could break, and the test for each:

1. **A filter that is simply absent.** `test_search_is_scoped_to_org`.
2. **A filter that is present but partial** — right org, wrong bot.
   `test_search_is_scoped_to_chatbot_within_one_org`.
3. **A filter built from attacker-supplied values.** The API layer resolves
   org_id server-side from the auth token, never from the request body, so the
   attack is a forged *chatbot_id*: a real id belonging to someone else, paired
   with the attacker's own org. `test_forged_chatbot_id_from_another_org_*`.
4. **A filter that silently matches nothing OR everything** because a scope
   value was empty. `test_empty_scope_*`.

Two of the tests below assert on the *shape of the code* rather than on
behaviour, and that is deliberate. A behavioural test catches a leak only once
somebody has written the leaking call; a structural test fails the moment the
signature grows a default that would let them. Given the cost of this particular
bug, we want to fail at the earlier of the two moments.

Deliberately NO Docker. Qdrant's client runs a real, in-process implementation
when pointed at `:memory:` — same filter semantics, no service to start — so
this test is one CI dependency lighter than the rest of the stack. That matters
for a test whose entire value comes from running on every single push.

Everything here embeds through a stub, not through bge. The property under test
is "does the filter hold", which has nothing to do with vector quality, and
loading real weights would put ~130MB and several seconds between a developer
and the answer to the most important question in the codebase.
"""

import inspect
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from app.core import vectorstore as vectorstore_module
from app.core.chunking import Chunk
from app.core.retrieval import retrieve
from app.core.vectorstore import SearchHit, VectorStore
from app.providers.base import ScoredIndex

# Two tenants, and a second chatbot inside the first tenant so that org-level
# and bot-level scoping are distinguishable failures rather than one blur.
ORG_A = "org_aaaaaaaa"
ORG_B = "org_bbbbbbbb"
BOT_A1 = "bot_a1"
BOT_A2 = "bot_a2"
BOT_B1 = "bot_b1"

# The SAME sentence is ingested into every tenant on purpose. Identical text
# produces an identical vector, so if the filter were missing or partial, the
# other tenant's chunk would rank equally and come back. Seeding each tenant
# with distinctive text would make the test pass for the wrong reason — the
# vector similarity would be doing the filter's job.
SHARED_TEXT = "The refund window is thirty days from the date of purchase."

# Small enough to be fast, large enough that distinct texts are not forced into
# near-identical vectors by the stub embedder's modulo arithmetic.
STUB_DIMENSION = 16


class StubEmbedder:
    """Deterministic, dependency-free stand-in for a real embedding provider.

    Satisfies `EmbeddingProvider` structurally (the contracts are Protocols, so
    there is nothing to subclass). Same text in, same vector out, which is
    exactly the property this test needs: it lets two tenants hold vectors that
    are *identical*, making the filter the only thing that can separate them.
    """

    name = "stub"
    model = "stub"
    dimension = STUB_DIMENSION

    def _vector(self, text: str) -> list[float]:
        # A crude bag-of-characters hash. Not meaningful as a semantic
        # embedding, and it does not need to be: only determinism matters here.
        vector = [0.0] * self.dimension
        for position, character in enumerate(text):
            vector[position % self.dimension] += (ord(character) % 17) / 17.0
        # Normalize so cosine distance behaves and no vector is degenerate.
        magnitude = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / magnitude for value in vector]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


class PassthroughReranker:
    """Reranker that preserves the vector store's ordering.

    Isolation is a property of the *filter*, decided before any reranking
    happens, so a real cross-encoder here would add a model load and several
    seconds while testing nothing. Returning input order unchanged keeps
    `retrieve()` end-to-end exercisable without that cost.
    """

    name = "passthrough"

    async def rerank(
        self, query: str, docs: list[str], top_k: int
    ) -> list[ScoredIndex]:
        return [(index, 1.0 - index / 1000) for index in range(min(top_k, len(docs)))]


def _chunk(text: str, index: int = 0) -> Chunk:
    return Chunk(text=text, index=index, token_count=max(1, len(text) // 4))


@pytest.fixture
async def store() -> Iterator[VectorStore]:
    """A fresh in-memory vector store, pre-seeded with three tenants' chunks.

    Function-scoped rather than session-scoped: `test_delete_is_scoped_to_owner`
    removes points, and a shared store would let that deletion leak into
    whichever tests happened to run afterwards. An isolation suite that is not
    itself isolated is not evidence of anything.
    """
    embedder = StubEmbedder()
    store = VectorStore(dimension=embedder.dimension, location=":memory:")
    await store.ensure_collection()

    # Same text, three different scopes.
    for org_id, chatbot_id, document_id, source in (
        (ORG_A, BOT_A1, "doc_a1", "a1-handbook.pdf"),
        (ORG_A, BOT_A2, "doc_a2", "a2-handbook.pdf"),
        (ORG_B, BOT_B1, "doc_b1", "b1-handbook.pdf"),
    ):
        chunks = [_chunk(SHARED_TEXT)]
        vectors = await embedder.embed_documents([chunk.text for chunk in chunks])
        await store.upsert_chunks(
            org_id=org_id,
            chatbot_id=chatbot_id,
            document_id=document_id,
            source=source,
            chunks=chunks,
            vectors=vectors,
        )

    yield store
    await store.close()


@pytest.fixture
def embedder() -> StubEmbedder:
    return StubEmbedder()


async def _search(
    store: VectorStore,
    embedder: StubEmbedder,
    *,
    org_id: str,
    chatbot_id: str,
    query: str = SHARED_TEXT,
    limit: int = 50,
) -> list[SearchHit]:
    """Search with a limit far above the seeded corpus.

    The high limit is load-bearing: a limit of 1 would return one correct chunk
    and hide two leaking ones behind it. To prove absence, ask for everything.
    """
    return await store.search(
        org_id=org_id,
        chatbot_id=chatbot_id,
        query_vector=await embedder.embed_query(query),
        limit=limit,
    )


# --------------------------------------------------------------------------
# 1. The filter exists and holds at the org boundary
# --------------------------------------------------------------------------


async def test_search_is_scoped_to_org(
    store: VectorStore, embedder: StubEmbedder
) -> None:
    """Org A's search never returns a chunk belonging to Org B.

    Both orgs hold the identical sentence being searched for, so an unfiltered
    or org-blind search returns all three seeded chunks. Only the filter can
    make this assertion pass.
    """
    hits = await _search(store, embedder, org_id=ORG_A, chatbot_id=BOT_A1)

    assert hits, "expected Org A to find its own chunk — the fixture seeded one"
    assert all(hit.document_id == "doc_a1" for hit in hits)
    assert not any("b1" in hit.source for hit in hits)


async def test_each_org_sees_only_its_own_corpus(
    store: VectorStore, embedder: StubEmbedder
) -> None:
    """Symmetric check: B is equally walled off from A.

    Worth asserting in both directions. A filter built from a hardcoded or
    accidentally-captured org id could pass the A-side test and fail here.
    """
    a_hits = await _search(store, embedder, org_id=ORG_A, chatbot_id=BOT_A1)
    b_hits = await _search(store, embedder, org_id=ORG_B, chatbot_id=BOT_B1)

    a_documents = {hit.document_id for hit in a_hits}
    b_documents = {hit.document_id for hit in b_hits}

    assert a_documents == {"doc_a1"}
    assert b_documents == {"doc_b1"}
    assert not (a_documents & b_documents)


# --------------------------------------------------------------------------
# 2. The filter is complete, not partial
# --------------------------------------------------------------------------


async def test_search_is_scoped_to_chatbot_within_one_org(
    store: VectorStore, embedder: StubEmbedder
) -> None:
    """Two bots in the SAME org do not see each other's chunks.

    This is the half of the filter an org-only implementation would omit, and
    it would never surface as a cross-customer leak in testing — only as one
    customer's sales bot quoting their internal HR handbook.
    """
    hits = await _search(store, embedder, org_id=ORG_A, chatbot_id=BOT_A1)

    assert {hit.document_id for hit in hits} == {"doc_a1"}
    assert "doc_a2" not in {hit.document_id for hit in hits}


# --------------------------------------------------------------------------
# 3. Forged scope values
# --------------------------------------------------------------------------


async def test_forged_chatbot_id_from_another_org_returns_nothing(
    store: VectorStore, embedder: StubEmbedder
) -> None:
    """The realistic attack: a real chatbot_id, stolen, paired with own org.

    org_id is resolved server-side from the auth token and is never attacker
    controlled, so the value an attacker CAN influence is the bot id. Because
    the filter is `must` (AND, not OR), Org A presenting Org B's real bot id
    matches no point at all — B's chunks fail the org condition, A's fail the
    bot condition.
    """
    hits = await _search(store, embedder, org_id=ORG_A, chatbot_id=BOT_B1)

    assert hits == [], "AND-ing org and bot must yield the empty intersection"


async def test_forged_org_id_without_matching_bot_returns_nothing(
    store: VectorStore, embedder: StubEmbedder
) -> None:
    """The mirror case, for defence in depth.

    org_id should never reach here from user input. Asserting it anyway means
    that if a future refactor ever does plumb a body field into this argument,
    this file fails rather than the property silently weakening.
    """
    hits = await _search(store, embedder, org_id=ORG_B, chatbot_id=BOT_A1)

    assert hits == []


# --------------------------------------------------------------------------
# 4. Empty scope fails loudly instead of quietly
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("org_id", "chatbot_id"),
    [
        ("", BOT_A1),
        (ORG_A, ""),
        ("", ""),
    ],
)
async def test_empty_scope_raises_rather_than_searching(
    store: VectorStore, embedder: StubEmbedder, org_id: str, chatbot_id: str
) -> None:
    """An empty scope value must raise, not search.

    An empty string does not bypass the filter — Qdrant would look for points
    whose org_id is literally "" and find none. That is precisely the danger:
    the caller receives an empty list and reads it as "this bot has no relevant
    documents" rather than "my scoping is broken". A silent wrong answer is
    worse than a stack trace, so `_require_scope` raises.
    """
    with pytest.raises(ValueError, match="required"):
        await _search(store, embedder, org_id=org_id, chatbot_id=chatbot_id)


async def test_empty_scope_raises_on_upsert(store: VectorStore) -> None:
    """Writes are scoped too.

    A chunk written with a blank org_id is unreachable by any legitimate search
    and would sit in the collection forever. Rejecting it at write time keeps
    the collection free of points no filter can ever match.
    """
    with pytest.raises(ValueError, match="required"):
        await store.upsert_chunks(
            org_id="",
            chatbot_id=BOT_A1,
            document_id="doc_x",
            source="x.pdf",
            chunks=[_chunk("anything")],
            vectors=[[0.0] * STUB_DIMENSION],
        )


# --------------------------------------------------------------------------
# 5. Deletes are scoped as tightly as reads
# --------------------------------------------------------------------------


async def test_delete_is_scoped_to_owner(
    store: VectorStore, embedder: StubEmbedder
) -> None:
    """Org B cannot delete Org A's document by naming its id.

    Deleting on document_id alone is correct while ids are unique and becomes a
    cross-tenant destructive operation the moment one is guessable or reused.
    Destructive cross-tenant access is a worse failure than a read leak: the
    read is recoverable, the delete is not.
    """
    await store.delete_document(org_id=ORG_B, chatbot_id=BOT_B1, document_id="doc_a1")

    survivors = await _search(store, embedder, org_id=ORG_A, chatbot_id=BOT_A1)
    assert {hit.document_id for hit in survivors} == {"doc_a1"}


async def test_delete_removes_only_the_owners_own_document(
    store: VectorStore, embedder: StubEmbedder
) -> None:
    """The positive half: a correctly scoped delete does work.

    Without this, `test_delete_is_scoped_to_owner` would still pass if
    `delete_document` were broken and deleted nothing at all.
    """
    await store.delete_document(org_id=ORG_A, chatbot_id=BOT_A1, document_id="doc_a1")

    assert await _search(store, embedder, org_id=ORG_A, chatbot_id=BOT_A1) == []
    # The other bot in the same org is untouched.
    assert await _search(store, embedder, org_id=ORG_A, chatbot_id=BOT_A2) != []


# --------------------------------------------------------------------------
# 6. Isolation survives the layer above
# --------------------------------------------------------------------------


async def test_retrieve_passes_scope_through_to_the_store(
    store: VectorStore, embedder: StubEmbedder
) -> None:
    """The full retrieval pipeline is scoped, not just the raw store.

    `retrieve()` is what the chat endpoint will call in Phase 3. Testing the
    store alone would leave room for the layer above to search with one scope
    and render results from another.
    """
    results = await retrieve(
        SHARED_TEXT,
        org_id=ORG_A,
        chatbot_id=BOT_A1,
        store=store,
        embedder=embedder,
        reranker=PassthroughReranker(),
        candidates=50,
        top_k=6,
    )

    assert results
    assert all(chunk.hit.document_id == "doc_a1" for chunk in results)


async def test_retrieve_returns_empty_for_a_foreign_chatbot(
    store: VectorStore, embedder: StubEmbedder
) -> None:
    """A forged bot id yields no chunks rather than an error at the top layer.

    Empty is the correct outcome, not a 500: generation turns "no chunks" into
    the refusal sentence, so an attacker learns nothing about whether the bot
    id they guessed exists. An error here would be an existence oracle.
    """
    results = await retrieve(
        SHARED_TEXT,
        org_id=ORG_A,
        chatbot_id=BOT_B1,
        store=store,
        embedder=embedder,
        reranker=PassthroughReranker(),
        candidates=50,
        top_k=6,
    )

    assert results == []


# --------------------------------------------------------------------------
# 7. Structural guarantees — the leak must stay unwritable
# --------------------------------------------------------------------------
#
# The tests above prove the filter works today. These two prove it cannot
# quietly stop being mandatory tomorrow, by asserting on the code itself rather
# than on its output. They are cheap, and they fail at code-review time instead
# of at incident time.


@pytest.mark.parametrize("method_name", ["search", "upsert_chunks", "delete_document"])
def test_scope_arguments_are_required_and_keyword_only(method_name: str) -> None:
    """org_id and chatbot_id must be keyword-only with no default.

    Keyword-only so a caller cannot transpose them positionally — swapping org
    and bot would produce an empty result set that looks like "no documents".
    No default so scope can never be omitted; a default of `None` or `""` is
    exactly the change that would turn every test above into a formality.
    """
    signature = inspect.signature(getattr(VectorStore, method_name))

    for argument in ("org_id", "chatbot_id"):
        parameter = signature.parameters[argument]
        assert (
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
        ), f"VectorStore.{method_name}: {argument} must be keyword-only"
        assert parameter.default is inspect.Parameter.empty, (
            f"VectorStore.{method_name}: {argument} must not have a default — "
            "a default makes an unscoped call writable"
        )


def test_every_qdrant_query_passes_a_filter() -> None:
    """No call into Qdrant's query API omits `query_filter`.

    Source inspection, which is unusual and justified here: the danger is a
    *second*, unfiltered search being added later — a new `search_by_document`,
    a debug endpoint, an admin count. Such a function would have its own tests,
    all passing, and no existing behavioural test would notice it. Scanning the
    source catches the call that nobody thought to test.

    Narrow on purpose: it checks one file and one SDK method, so it fails on
    the thing it is about rather than becoming a lint rule nobody trusts.
    """
    source = Path(vectorstore_module.__file__).read_text(encoding="utf-8")

    # Each `query_points(` call site, with the argument list that follows it.
    call_sites = re.findall(r"query_points\((.*?)\n\s*\)", source, flags=re.DOTALL)

    assert call_sites, (
        "found no query_points call — if the Qdrant search API was renamed, "
        "update this test rather than deleting it"
    )
    for arguments in call_sites:
        assert "query_filter=" in arguments, (
            "a query_points call omits query_filter; every read must be scoped "
            "by org_id and chatbot_id (Hard Rule #3)"
        )

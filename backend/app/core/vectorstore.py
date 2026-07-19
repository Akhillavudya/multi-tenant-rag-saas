"""Qdrant wrapper: vectors in, filtered nearest neighbours out.

The only module that talks to the vector database. Everything above it deals in
`Chunk` and `SearchHit`; nothing above it imports `qdrant_client` or knows what
a payload index is.

**Hard Rule #3 lives here, and it is enforced by the type signature.** Every
search takes `org_id` and `chatbot_id` as REQUIRED keyword-only arguments and
builds a `must` filter from them before the query leaves this file. There is no
optional-filter parameter, no `**kwargs` passthrough, and no code path that
reaches Qdrant without both values — writing an unfiltered search is not a
mistake you can make by forgetting, it is a call that does not compile.

Phase 1 is single-tenant, so this looks like ceremony today. It is not: the
alternative is adding a filter argument in Phase 2 to a function that already
has callers, and hoping every one of them is updated. The isolation test in
Phase 2 (PLAN §6) is sacred precisely because that class of bug is invisible
until it is a data breach.

A note on Hard Rule #2, same reasoning as `extraction.py`. `qdrant_client` is
imported here, in `core/`. The rule keeps *swappable AI providers* behind the
factory so `LLM_PROVIDER=groq` stays a config change; Qdrant is infrastructure,
like Postgres and Redis, chosen once in PLAN §2 and not selected by an env var.
Wrapping it in a provider factory would be ceremony around one implementation.

Writes are idempotent: point IDs are derived deterministically from
(org, chatbot, document, chunk_index), so re-ingesting a document overwrites its
chunks in place rather than duplicating them. Ingestion is retried by Celery in
Phase 2, and random IDs would silently double a document's chunks on every
retry — inflating the index and letting the same passage occupy several slots in
the top-40 candidate list.
"""

import logging
import uuid
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient, models

from app.config import settings
from app.core.chunking import Chunk

logger = logging.getLogger(__name__)

# Namespace for deterministic point IDs. Any fixed UUID works; it exists so our
# IDs cannot collide with UUIDv5s generated elsewhere from the same strings.
# NEVER change it — every existing point ID is derived from it, so a new
# namespace orphans the entire collection.
_POINT_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

# Payload keys. Constants rather than string literals because they appear in
# three places each (write, filter, read) and a typo in the filter silently
# matches nothing — which looks exactly like "no relevant documents".
PAYLOAD_ORG_ID = "org_id"
PAYLOAD_CHATBOT_ID = "chatbot_id"
PAYLOAD_DOCUMENT_ID = "document_id"
PAYLOAD_SOURCE = "source"
PAYLOAD_TEXT = "text"
PAYLOAD_CHUNK_INDEX = "chunk_index"
PAYLOAD_PAGE_START = "page_start"
PAYLOAD_PAGE_END = "page_end"

# Fields we filter on need a payload index. Without one Qdrant still returns
# correct results, but it evaluates the condition by scanning candidates the
# HNSW graph walk produced — which degrades badly once one collection holds many
# tenants, the exact situation Phase 2 creates.
_INDEXED_FIELDS = (PAYLOAD_ORG_ID, PAYLOAD_CHATBOT_ID, PAYLOAD_DOCUMENT_ID)

# Points per upsert request. Large enough to amortise round trips, small enough
# that one request stays well inside default body limits at ~400 tokens of text
# plus a 384-float vector per point.
DEFAULT_UPSERT_BATCH = 128


class VectorStoreError(Exception):
    """Base for vector-store failures.

    Typed for the same reason `ExtractionError` is: the API layer turns "Qdrant
    is unreachable" (retry, our problem) and "this collection has the wrong
    dimension" (a migration is needed, an operator's problem) into completely
    different responses.
    """


class VectorStoreUnavailableError(VectorStoreError):
    """Qdrant is unreachable or refused the request. Usually retryable."""


class CollectionMismatchError(VectorStoreError):
    """The collection exists with a different vector size than the configured
    embedder produces. Never auto-corrected: fixing it means re-embedding every
    document, which is a deliberate operation, not a startup side effect."""


@dataclass(frozen=True)
class SearchHit:
    """One retrieval result, already flattened out of Qdrant's response shape.

    Carries the text AND its provenance because the generation step must cite
    source and page (PLAN §6). Fetching the text separately from Postgres would
    mean an N+1 query on the hot path of every single chat message.
    """

    id: str
    score: float
    text: str
    document_id: str
    source: str
    chunk_index: int
    page_start: int | None = None
    page_end: int | None = None

    @property
    def citation(self) -> str:
        """Human-readable provenance, e.g. "handbook.pdf p.12" or "notes.md"."""
        if self.page_start is None:
            return self.source
        if self.page_end is not None and self.page_end != self.page_start:
            return f"{self.source} pp.{self.page_start}-{self.page_end}"
        return f"{self.source} p.{self.page_start}"


def _require_scope(org_id: str, chatbot_id: str) -> None:
    """Reject empty tenant identifiers before they reach a filter.

    An empty string does not bypass the filter — Qdrant would match points whose
    org_id is literally "" and return nothing. That is the dangerous part: the
    caller sees an empty result and reads it as "no relevant documents" rather
    than "my scoping is broken". Failing loudly turns a silent wrong answer into
    a stack trace.
    """
    if not org_id or not chatbot_id:
        raise ValueError(
            f"org_id and chatbot_id are both required "
            f"(got org_id={org_id!r}, chatbot_id={chatbot_id!r})"
        )


def _point_id(org_id: str, chatbot_id: str, document_id: str, index: int) -> str:
    """Deterministic point ID, so re-ingestion overwrites instead of duplicating.

    UUIDv5 is a hash (SHA-1) of a namespace plus a name, so the same inputs
    always produce the same ID on any machine, in any process, forever — no
    coordination and no lookup table. Qdrant only accepts UUIDs or unsigned
    integers as point IDs, which is why this is not simply the joined string.
    """
    return str(
        uuid.uuid5(_POINT_NAMESPACE, f"{org_id}:{chatbot_id}:{document_id}:{index}")
    )


class VectorStore:
    """Async Qdrant client scoped to one collection.

    One collection holds every tenant's chunks, separated by payload filters
    rather than by collection-per-tenant. That is the deliberate choice: Qdrant
    keeps per-collection overhead (an HNSW graph, segment files) that would be
    paid thousands of times over, and creating a collection per signup makes
    onboarding a schema operation. Filters on an indexed payload field are the
    documented multi-tenancy pattern.

    The cost of that choice is that isolation becomes application logic instead
    of physical separation — which is precisely why the filter is mandatory in
    the signature and why Phase 2's isolation test exists.
    """

    def __init__(
        self,
        *,
        dimension: int,
        collection: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """`dimension` is required and has no default on purpose.

        It must come from the live embedder (`embedder.dimension`), not from a
        constant here. Hardcoding 384 would make swapping to a 1536-dim provider
        a silent corruption: vectors of the wrong width are rejected at write
        time, but only after a collection has been created that can never
        accept them.
        """
        self._dimension = dimension
        self._collection = collection or settings.qdrant_collection
        self._client = AsyncQdrantClient(
            url=url or settings.qdrant_url,
            api_key=api_key or settings.qdrant_api_key,
        )

    @property
    def collection(self) -> str:
        return self._collection

    # -- lifecycle ----------------------------------------------------------

    async def ensure_collection(self) -> None:
        """Create the collection and its payload indexes if absent; verify it
        matches if present. Idempotent — safe to call on every worker boot.

        Verification is the important half. A collection created against a
        384-dim embedder silently outlives the config change that switched to a
        1536-dim one, and the resulting error appears at the first upsert of a
        long ingestion run rather than at startup.
        """
        try:
            if await self._client.collection_exists(self._collection):
                await self._verify_dimension()
                return

            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=self._dimension,
                    # Cosine even though bge already returns unit vectors (where
                    # cosine and dot are identical): a future embedder may not
                    # normalize, and the failure would be a silent ranking
                    # regression rather than an error.
                    distance=models.Distance.COSINE,
                ),
            )
            for field in _INDEXED_FIELDS:
                await self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            logger.info(
                "created vector collection",
                extra={"collection": self._collection, "dimension": self._dimension},
            )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreUnavailableError(
                f"could not prepare collection {self._collection!r}: {exc}"
            ) from exc

    async def _verify_dimension(self) -> None:
        info = await self._client.get_collection(self._collection)
        params = info.config.params.vectors
        existing = getattr(params, "size", None)
        if existing is not None and existing != self._dimension:
            raise CollectionMismatchError(
                f"collection {self._collection!r} stores {existing}-dim vectors "
                f"but the configured embedder produces {self._dimension}-dim; "
                "changing embedder requires re-embedding into a new collection"
            )

    async def close(self) -> None:
        """Release the underlying HTTP connections. Called from FastAPI's
        lifespan shutdown and at the end of a CLI run."""
        await self._client.close()

    # -- writes -------------------------------------------------------------

    async def upsert_chunks(
        self,
        *,
        org_id: str,
        chatbot_id: str,
        document_id: str,
        source: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
        batch_size: int = DEFAULT_UPSERT_BATCH,
    ) -> int:
        """Write chunks and their vectors, returning the number of points sent.

        `chunks` and `vectors` are matched POSITIONALLY, which is why the length
        check below is an assertion of contract rather than defensiveness: a
        misalignment attaches every chunk's text to its neighbour's vector, and
        the result is a searchable index that returns confidently wrong passages
        with no error anywhere.
        """
        _require_scope(org_id, chatbot_id)
        if not chunks:
            return 0
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunk/vector length mismatch: {len(chunks)} chunks, "
                f"{len(vectors)} vectors"
            )

        points = [
            models.PointStruct(
                id=_point_id(org_id, chatbot_id, document_id, chunk.index),
                vector=vector,
                payload={
                    PAYLOAD_ORG_ID: org_id,
                    PAYLOAD_CHATBOT_ID: chatbot_id,
                    PAYLOAD_DOCUMENT_ID: document_id,
                    PAYLOAD_SOURCE: source,
                    PAYLOAD_TEXT: chunk.text,
                    PAYLOAD_CHUNK_INDEX: chunk.index,
                    PAYLOAD_PAGE_START: chunk.page_start,
                    PAYLOAD_PAGE_END: chunk.page_end,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        try:
            for start in range(0, len(points), batch_size):
                await self._client.upsert(
                    collection_name=self._collection,
                    points=points[start : start + batch_size],
                    # wait=True so a returned success means the points are
                    # queryable. Reporting a document "ready" while its vectors
                    # are still being indexed produces a bot that answers "I
                    # don't have that information" about a document the user
                    # just watched succeed.
                    wait=True,
                )
        except Exception as exc:
            raise VectorStoreUnavailableError(
                f"upsert of {len(points)} points failed: {exc}"
            ) from exc

        logger.info(
            "upserted chunks",
            extra={
                "org_id": org_id,
                "chatbot_id": chatbot_id,
                "document_id": document_id,
                "points": len(points),
            },
        )
        return len(points)

    async def delete_document(
        self,
        *,
        org_id: str,
        chatbot_id: str,
        document_id: str,
    ) -> None:
        """Remove every chunk of one document.

        Scoped by org and chatbot as well as document id — deleting by
        document_id alone would be correct today (ids are unique) and a
        cross-tenant deletion the moment ids are ever reused or guessable.
        Deletes are filtered for the same reason searches are.
        """
        _require_scope(org_id, chatbot_id)
        try:
            await self._client.delete(
                collection_name=self._collection,
                points_selector=models.FilterSelector(
                    filter=self._scope_filter(
                        org_id=org_id,
                        chatbot_id=chatbot_id,
                        document_id=document_id,
                    )
                ),
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreUnavailableError(
                f"delete of document {document_id!r} failed: {exc}"
            ) from exc

    # -- reads --------------------------------------------------------------

    async def search(
        self,
        *,
        org_id: str,
        chatbot_id: str,
        query_vector: list[float],
        limit: int,
        document_id: str | None = None,
    ) -> list[SearchHit]:
        """Nearest neighbours within one chatbot, highest score first.

        `org_id` and `chatbot_id` are keyword-only and required: THIS is the
        signature that makes Hard Rule #3 structural. There is deliberately no
        `filter` parameter for callers to pass — the filter is built here, from
        arguments they cannot omit.

        `limit` is the wide first stage (40 in PLAN's pipeline), not the number
        of chunks the model sees. The reranker narrows that to 6.
        """
        _require_scope(org_id, chatbot_id)

        try:
            response = await self._client.query_points(
                collection_name=self._collection,
                query=query_vector,
                query_filter=self._scope_filter(
                    org_id=org_id,
                    chatbot_id=chatbot_id,
                    document_id=document_id,
                ),
                limit=limit,
                # The vector is ~1.5KB per point and we never use it after
                # ranking; the payload is the whole point of the read.
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise VectorStoreUnavailableError(f"search failed: {exc}") from exc

        return [_to_hit(point) for point in response.points]

    @staticmethod
    def _scope_filter(
        *,
        org_id: str,
        chatbot_id: str,
        document_id: str | None = None,
    ) -> models.Filter:
        """Build the mandatory tenant filter.

        `must` is AND: every condition has to hold. The single most important
        line in this file is that org_id and chatbot_id are appended
        unconditionally, before any optional condition — there is no branch in
        which the list is empty.
        """
        conditions: list[models.Condition] = [
            models.FieldCondition(
                key=PAYLOAD_ORG_ID, match=models.MatchValue(value=org_id)
            ),
            models.FieldCondition(
                key=PAYLOAD_CHATBOT_ID, match=models.MatchValue(value=chatbot_id)
            ),
        ]
        if document_id is not None:
            conditions.append(
                models.FieldCondition(
                    key=PAYLOAD_DOCUMENT_ID, match=models.MatchValue(value=document_id)
                )
            )
        return models.Filter(must=conditions)


def _to_hit(point: models.ScoredPoint) -> SearchHit:
    """Flatten Qdrant's response into our own type.

    Deliberately not returning ScoredPoint: retrieval and generation would then
    depend on the Qdrant SDK's shapes, and swapping to pgvector later would
    reach into files that have nothing to do with storage.
    """
    payload = point.payload or {}
    return SearchHit(
        id=str(point.id),
        score=point.score,
        text=payload.get(PAYLOAD_TEXT, ""),
        document_id=payload.get(PAYLOAD_DOCUMENT_ID, ""),
        source=payload.get(PAYLOAD_SOURCE, "unknown"),
        chunk_index=payload.get(PAYLOAD_CHUNK_INDEX, 0),
        page_start=payload.get(PAYLOAD_PAGE_START),
        page_end=payload.get(PAYLOAD_PAGE_END),
    )

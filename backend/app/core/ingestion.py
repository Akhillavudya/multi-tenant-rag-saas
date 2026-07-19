"""Ingestion pipeline: a file on disk becomes searchable chunks.

The orchestrator. It owns no algorithm of its own — extraction, chunking,
embedding and storage each live in their own module — and its entire job is to
run them in the right order with the right arguments, which is exactly why it is
the file where the interesting failures live.

    extract() -> chunk_blocks() -> embed_documents() -> upsert_chunks()
    [pure]       [pure]            [provider]           [Qdrant]

Two decisions carry the file.

1. **The real tokenizer is passed to the chunker.** `chunk_blocks` defaults to a
   4-characters-per-token estimate so it can stay pure and testable; that default
   is for tests, never for ingestion. Here we hand it `embedder.count_tokens`, so
   chunks are sized against the model that will actually embed them. Skipping
   this is invisible in English prose and silently truncates technical documents.

2. **Embed and upsert in batches, interleaved.** Not embed-everything then
   store-everything. Memory stays flat on a 500-page PDF instead of holding
   thousands of vectors at once, and — because point ids are deterministic
   (`vectorstore._point_id`) — a crash partway through leaves valid, queryable
   points that the retry simply overwrites. Partial progress is durable rather
   than corrupt, which is what makes Celery's retry in Phase 2 safe.

This module deliberately does NOT catch the errors below it. `ExtractionError`,
`ProviderError` and `VectorStoreError` are already typed, already carry the
detail a caller needs, and wrapping them here would only bury the cause one
level deeper. Phase 2's Celery task decides what becomes a retry and what
becomes a `failed` status; that policy does not belong here.
"""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from app.core.chunking import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
    Chunk,
    chunk_blocks,
)
from app.core.extraction import extract
from app.core.vectorstore import VectorStore
from app.providers.base import EmbeddingProvider
from app.providers.factory import get_embedder

logger = logging.getLogger(__name__)

# Chunks embedded and written per round trip. Distinct from the embedder's own
# internal batch size (which is about GPU/CPU throughput): this one bounds how
# much work is lost to a crash, and how many vectors exist in memory at once.
DEFAULT_INGEST_BATCH = 64


@dataclass(frozen=True)
class IngestionResult:
    """What one ingest produced. Returned rather than logged-and-discarded so
    the CLI can print it and Phase 2 can write it to the documents row."""

    document_id: str
    source: str
    chunks: int
    points: int
    tokens: int
    pages: int | None

    def summary(self) -> str:
        pages = f", {self.pages} pages" if self.pages else ""
        return (
            f"{self.source}: {self.chunks} chunks, "
            f"{self.tokens} tokens{pages} -> {self.points} points"
        )


def content_document_id(path: Path) -> str:
    """Stable id derived from the file's bytes.

    Content-addressed, not path-addressed: the same file ingested twice — under
    a different name, or after a failed run — maps to the same id, so the
    deterministic point ids overwrite rather than duplicate. Renaming a file
    therefore does not create a second copy in the index, while editing it does
    create a new document, which is the behaviour we want in both cases.

    Phase 2 passes the database's document id instead; this is the default for
    the CLI, where there is no database yet.
    """
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:32]


async def ingest_file(
    path: Path,
    *,
    org_id: str,
    chatbot_id: str,
    store: VectorStore,
    document_id: str | None = None,
    source: str | None = None,
    embedder: EmbeddingProvider | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    batch_size: int = DEFAULT_INGEST_BATCH,
) -> IngestionResult:
    """Extract, chunk, embed and store one file. Returns what was written.

    `org_id` and `chatbot_id` are required keyword-only arguments for the same
    reason they are on `VectorStore.search` — every point must be written with
    the tenant it belongs to, or it can never be found again by a filtered
    query, and a chunk written without a tenant is worse than one not written at
    all: it is invisible to its owner and permanently present in the index.

    `store` is injected rather than constructed here so that ingestion, the API
    and the CLI share one client and one connection pool, and so tests can pass
    a fake without a running Qdrant.
    """
    embedder = embedder or get_embedder()
    source = source or path.name
    document_id = document_id or content_document_id(path)

    log = {"org_id": org_id, "chatbot_id": chatbot_id, "document_id": document_id}
    logger.info("ingestion started", extra={**log, "source": source})

    # 1. File -> page-tagged text. Raises ExtractionError subclasses; a scanned
    #    PDF fails HERE, loudly, rather than producing zero chunks silently.
    blocks = extract(path)

    # 2. Text -> sized chunks, measured with the embedding model's own
    #    tokenizer. `count_tokens` is the whole reason sections 17-19 existed.
    chunks = chunk_blocks(
        blocks,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        count_tokens=embedder.count_tokens,
    )
    if not chunks:
        # extract() already guarantees non-empty blocks, so reaching here means
        # the chunker dropped everything — a bug in sizing, not an empty file.
        raise ValueError(f"{source} produced no chunks from {len(blocks)} blocks")

    # 3. Make sure the collection exists and matches this embedder's dimension
    #    BEFORE embedding. Discovering a dimension mismatch after computing
    #    several thousand vectors wastes all of that work.
    await store.ensure_collection()

    # 4. Embed and write in interleaved batches.
    points = 0
    for batch in _batched(chunks, batch_size):
        vectors = await embedder.embed_documents([chunk.text for chunk in batch])
        points += await store.upsert_chunks(
            org_id=org_id,
            chatbot_id=chatbot_id,
            document_id=document_id,
            source=source,
            chunks=batch,
            vectors=vectors,
        )

    result = IngestionResult(
        document_id=document_id,
        source=source,
        chunks=len(chunks),
        points=points,
        tokens=sum(chunk.token_count for chunk in chunks),
        pages=_page_count(blocks),
    )
    logger.info("ingestion complete", extra={**log, "chunks": result.chunks})
    return result


async def delete_document(
    *,
    org_id: str,
    chatbot_id: str,
    document_id: str,
    store: VectorStore,
) -> None:
    """Remove a document's chunks. A thin pass-through today, kept here so
    callers have one ingestion-facing module rather than reaching into the
    vector store directly — Phase 2 adds the database row deletion alongside."""
    await store.delete_document(
        org_id=org_id, chatbot_id=chatbot_id, document_id=document_id
    )
    logger.info(
        "document deleted",
        extra={"org_id": org_id, "chatbot_id": chatbot_id, "document_id": document_id},
    )


def _batched(chunks: list[Chunk], size: int) -> list[list[Chunk]]:
    """Split into fixed-size batches, preserving order.

    Order matters beyond tidiness: `upsert_chunks` zips chunks against vectors
    positionally, so every batch must reach the embedder and the store in the
    same sequence.
    """
    if size < 1:
        raise ValueError(f"batch_size must be >= 1, got {size}")
    return [chunks[i : i + size] for i in range(0, len(chunks), size)]


def _page_count(blocks: list) -> int | None:
    """Highest page number seen, or None for formats without pagination.

    None rather than 1 for .docx and .txt: claiming a one-page document would be
    a fact we invented, and citations elsewhere already distinguish "page
    unknown" from "page 1".
    """
    pages = [block.page for block in blocks if block.page is not None]
    return max(pages) if pages else None

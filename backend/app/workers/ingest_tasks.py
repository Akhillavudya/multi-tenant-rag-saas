"""Celery tasks: ingestion and vector cleanup.

Where PLAN §6's status transitions live — PENDING -> PROCESSING -> READY|FAILED.
The row is the user's only window into work happening in another process, so
every exit path from `ingest_document` writes a terminal status. A task that
raises without doing so leaves a document spinning in the dashboard forever,
which is the single most common way async ingestion goes wrong.

**Retry classification is the interesting decision here.**

Not every failure deserves a retry, and retrying the wrong ones is worse than
not retrying at all — it delays the honest error the user needs and burns queue
capacity on work that cannot succeed.

    ProviderRateLimitError / Timeout / Unavailable  -> retry (transient)
    VectorStoreUnavailableError                     -> retry (Qdrant restarting)
    ExtractionError (scanned PDF, encrypted, empty) -> FAIL NOW, tell the user
    ProviderAuthError / InvalidRequest              -> FAIL NOW, our bug

Phase 1 built the vocabulary that makes this a lookup rather than a guess:
`ProviderError.retryable` and the typed `ExtractionError` hierarchy. This module
is the first consumer of both.

**Retries use exponential backoff with jitter.** Doubling the wait (10s, 20s,
40s...) gives a struggling dependency room to recover instead of being hammered.
The jitter matters when a provider outage fails fifty documents at once: without
it all fifty retry at the same instant, and the recovering service is knocked
over by the retry storm — the "thundering herd".

**Idempotency.** `task_acks_late` in celery_app.py means a task killed mid-run is
redelivered, so ingestion must tolerate running twice. It does: point ids are
deterministic (`vectorstore._point_id`), so a re-run overwrites its own partial
work rather than appending a second copy of every chunk.

**`asyncio.run` per task.** The core pipeline is async (Qdrant's client is), and
Celery tasks are plain sync functions. One event loop per task is the honest
mapping — the task IS the unit of work, and it has no other concurrency to
share a loop with. The cost is loop setup per task, which is microseconds
against an ingestion measured in seconds.
"""

import asyncio
import logging
import random
import uuid
from datetime import UTC
from pathlib import Path

from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.core.extraction import ExtractionError
from app.core.ingestion import IngestionResult, ingest_file
from app.core.vectorstore import VectorStore, VectorStoreError
from app.db.base import SessionLocal
from app.db.tables import Chatbot, ChatbotStatus, Document, DocumentStatus
from app.providers.base import ProviderError
from app.providers.factory import get_embedder
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

# Give up after this many attempts. Bounded because an unbounded retry on a
# permanently broken dependency is a queue that never drains and a document
# that never reaches a terminal status a user can act on.
MAX_RETRIES = 3

# First backoff, doubled each attempt: 10s, 20s, 40s.
RETRY_BASE_SECONDS = 10

# Error text stored on the row is shown to the user. Truncated because a
# provider stack trace can run to kilobytes, and the `error` column feeds a
# dashboard cell, not a log viewer.
MAX_ERROR_CHARS = 500


def _backoff_seconds(attempt: int) -> int:
    """Exponential backoff with jitter. See module docstring on thundering herd.

    Jitter is +/-25%, which is enough to spread a batch of simultaneous
    failures across a window without making the delay unpredictable to reason
    about.
    """
    base = RETRY_BASE_SECONDS * (2**attempt)
    return max(1, int(base * random.uniform(0.75, 1.25)))


def _stored_path(document: Document) -> Path:
    """Rebuild the on-disk path for a document.

    Derived from ids and the filename's suffix, exactly as `api/documents.py`
    built it. Duplicating the derivation is a genuine wart — the two must not
    drift — and it is preferred to storing the path in the database, which
    would break the moment Phase 7 moves files to object storage and every old
    row still held a local filesystem path.
    """
    suffix = Path(document.filename).suffix.lower()
    return (
        settings.upload_dir
        / str(document.org_id)
        / str(document.chatbot_id)
        / f"{document.id}{suffix}"
    )


@celery_app.task(bind=True, name="ingest_document", max_retries=MAX_RETRIES)
def ingest_document(self, document_id: str) -> dict:
    """Parse, chunk, embed and store one uploaded document.

    Receives only an id (see celery_app.py on why) and re-reads current state
    from Postgres.

    Returns a small dict for debugging. The real output is the row: status,
    chunk_count, pages, error.
    """
    session = SessionLocal()
    try:
        document = session.get(Document, uuid.UUID(document_id))
        if document is None:
            # Deleted between publish and execution. Not an error — the user
            # changed their mind, and there is nothing left to do.
            logger.info(
                "document gone before ingestion", extra={"document_id": document_id}
            )
            return {"status": "gone", "document_id": document_id}

        log = {
            "org_id": str(document.org_id),
            "chatbot_id": str(document.chatbot_id),
            "document_id": str(document.id),
        }

        if document.status is DocumentStatus.READY:
            # A redelivered message for work already finished. Doing it again
            # would be harmless (ingestion is idempotent) but pointless.
            logger.info("document already ingested", extra=log)
            return {"status": "already_ready", **log}

        document.status = DocumentStatus.PROCESSING
        # Clear any error from a previous attempt. The CHECK constraint on the
        # table forbids a non-failed row carrying an error message, so this is
        # required, not tidiness.
        document.error = None
        session.commit()

        path = _stored_path(document)
        if not path.exists():
            # The row says a file exists and it does not. Terminal: no retry
            # will conjure the bytes back.
            return _fail(session, document, log, "uploaded file is missing on disk")

        chatbot = session.get(Chatbot, document.chatbot_id)
        if chatbot is None:
            return _fail(session, document, log, "chatbot no longer exists")

        try:
            result = asyncio.run(
                _run_ingestion(
                    path=path,
                    org_id=str(document.org_id),
                    chatbot_id=str(document.chatbot_id),
                    document_id=str(document.id),
                    source=document.filename,
                    dimension=chatbot.dimension,
                )
            )
        except ExtractionError as exc:
            # The user's file is the problem: scanned, encrypted, empty,
            # corrupt. Retrying produces the identical failure, so fail now with
            # a message they can act on.
            return _fail(session, document, log, str(exc))
        except ProviderError as exc:
            if exc.retryable:
                return _retry(self, session, document, log, exc)
            # Auth or invalid-request: our misconfiguration, not transient.
            return _fail(session, document, log, f"provider error: {exc}")
        except VectorStoreError as exc:
            # Qdrant restarting or unreachable. Worth another attempt.
            return _retry(self, session, document, log, exc)
        except SoftTimeLimitExceeded:
            # Celery's soft limit fired. Terminal on purpose: a document that
            # cannot finish in 25 minutes will not finish in 25 more, and
            # retrying would occupy a worker for another full window.
            return _fail(session, document, log, "ingestion timed out")

        document.status = DocumentStatus.READY
        document.chunk_count = result.chunks
        document.pages = result.pages
        document.error = None

        # First ready document promotes a DRAFT bot. Without this a bot with
        # perfectly good documents still shows as unfinished in the dashboard.
        if chatbot.status is ChatbotStatus.DRAFT:
            chatbot.status = ChatbotStatus.READY

        session.commit()
        logger.info("ingestion succeeded", extra={**log, "chunks": result.chunks})
        return {"status": "ready", "chunks": result.chunks, **log}
    finally:
        # The worker is long-lived and this session holds a pooled connection.
        # Not closing it leaks one connection per task until the pool is empty
        # and every subsequent task blocks — a failure that shows up hours in,
        # under load, and looks like a database problem.
        session.close()


async def _run_ingestion(
    *,
    path: Path,
    org_id: str,
    chatbot_id: str,
    document_id: str,
    source: str,
    dimension: int,
) -> IngestionResult:
    """Async half of the task: own a VectorStore for the duration and close it.

    The store is built per task rather than shared at module scope because it
    owns an HTTP connection pool bound to an event loop, and each task runs its
    own loop. A module-level client would be bound to whichever loop happened to
    create it and fail on every subsequent task.

    `dimension` comes from the CHATBOT row, not from the live embedder. If
    EMBEDDING_PROVIDER has changed since this bot was created, that mismatch
    must surface as a loud CollectionMismatchError rather than as vectors
    silently written in a width nothing will ever query with.
    """
    store = VectorStore(dimension=dimension)
    try:
        return await ingest_file(
            path,
            org_id=org_id,
            chatbot_id=chatbot_id,
            document_id=document_id,
            source=source,
            store=store,
            embedder=get_embedder(),
        )
    finally:
        await store.close()


def _fail(session: Session, document: Document, log: dict, message: str) -> dict:
    """Write a terminal FAILED status with a user-readable reason."""
    document.status = DocumentStatus.FAILED
    document.error = message[:MAX_ERROR_CHARS]
    session.commit()
    logger.warning("ingestion failed", extra={**log, "error": message})
    return {"status": "failed", "error": message, **log}


def _retry(
    task, session: Session, document: Document, log: dict, exc: Exception
) -> dict:
    """Schedule another attempt, or fail terminally once retries are exhausted.

    The row is put back to PENDING between attempts rather than left in
    PROCESSING: PROCESSING means "a worker has this right now", and a document
    waiting 40 seconds for a retry does not. The distinction is what lets an
    operator tell a queued document from a stuck one.
    """
    if task.request.retries >= MAX_RETRIES:
        return _fail(
            session,
            document,
            log,
            f"failed after {MAX_RETRIES} attempts: {exc}",
        )

    document.status = DocumentStatus.PENDING
    session.commit()

    delay = _backoff_seconds(task.request.retries)
    logger.warning(
        "retrying ingestion",
        extra={
            **log,
            "attempt": task.request.retries + 1,
            "delay_s": delay,
            "error": str(exc),
        },
    )
    raise task.retry(exc=exc, countdown=delay)


# --------------------------------------------------------------------------
# Vector cleanup
# --------------------------------------------------------------------------
#
# Both tasks run AFTER their SQL rows are already gone, so they reclaim storage
# rather than close a leak — nothing can search points whose org/bot pair no
# longer resolves to a row. They still take the full scope and pass it to a
# filtered delete, because a delete built from one identifier is one guessed id
# away from being someone else's data.


@celery_app.task(bind=True, name="purge_document_vectors", max_retries=MAX_RETRIES)
def purge_document_vectors(
    self, org_id: str, chatbot_id: str, document_id: str
) -> dict:
    """Delete one document's chunks from Qdrant."""

    async def _run() -> None:
        store = VectorStore(dimension=get_embedder().dimension)
        try:
            await store.delete_document(
                org_id=org_id, chatbot_id=chatbot_id, document_id=document_id
            )
        finally:
            await store.close()

    try:
        asyncio.run(_run())
    except VectorStoreError as exc:
        raise self.retry(
            exc=exc, countdown=_backoff_seconds(self.request.retries)
        ) from exc

    logger.info(
        "purged document vectors",
        extra={"org_id": org_id, "chatbot_id": chatbot_id, "document_id": document_id},
    )
    return {"status": "purged", "document_id": document_id}


@celery_app.task(bind=True, name="purge_chatbot_vectors", max_retries=MAX_RETRIES)
def purge_chatbot_vectors(self, org_id: str, chatbot_id: str) -> dict:
    """Delete every chunk belonging to one chatbot."""

    async def _run() -> None:
        store = VectorStore(dimension=get_embedder().dimension)
        try:
            await store.delete_chatbot(org_id=org_id, chatbot_id=chatbot_id)
        finally:
            await store.close()

    try:
        asyncio.run(_run())
    except VectorStoreError as exc:
        raise self.retry(
            exc=exc, countdown=_backoff_seconds(self.request.retries)
        ) from exc

    logger.info(
        "purged chatbot vectors",
        extra={"org_id": org_id, "chatbot_id": chatbot_id},
    )
    return {"status": "purged", "chatbot_id": chatbot_id}


@celery_app.task(name="requeue_stuck_documents")
def requeue_stuck_documents(older_than_minutes: int = 45) -> dict:
    """Re-queue documents left in PROCESSING by a worker that died.

    `task_acks_late` covers a worker that crashes while Celery is watching. It
    does not cover the machine losing power, or a message expiring — in those
    cases the row stays PROCESSING with nothing working on it, and no amount of
    waiting fixes it.

    Not scheduled automatically in Phase 2 (Celery Beat arrives with the
    deployment work in Phase 7). Runnable by hand meanwhile:

        celery -A app.workers.celery_app call requeue_stuck_documents
    """
    from datetime import datetime, timedelta

    cutoff = datetime.now(UTC) - timedelta(minutes=older_than_minutes)
    session = SessionLocal()
    try:
        stuck = list(
            session.scalars(
                select(Document).where(
                    Document.status == DocumentStatus.PROCESSING,
                    Document.updated_at < cutoff,
                )
            )
        )
        for document in stuck:
            document.status = DocumentStatus.PENDING
        session.commit()

        for document in stuck:
            ingest_document.delay(document_id=str(document.id))

        if stuck:
            logger.warning("requeued stuck documents", extra={"count": len(stuck)})
        return {"requeued": len(stuck)}
    finally:
        session.close()

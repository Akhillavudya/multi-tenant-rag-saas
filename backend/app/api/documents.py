"""Document upload and lifecycle, scoped to one chatbot the caller owns.

Upload is the only endpoint in Phase 2 that accepts a file, which makes it the
only one with a whole extra category of attack surface. Three rules govern it:

**1. The client's filename never touches the filesystem.**

`document.pdf` is fine; `../../../../etc/cron.d/payload` is a filename too, and
so is `..\\..\\windows\\system32\\x.dll`. The stored path is built from the
document's own UUID plus a suffix validated against an allow-list, and the
original name is kept in the database as a display string only. This is path
traversal, and it is a file-upload bug so common it is worth stating in a
comment rather than assuming everyone remembers it.

**2. Size is enforced while streaming, not after.**

Checking `Content-Length` trusts a header the client wrote; checking the size
after reading means the disk is already full. The upload is copied in fixed
chunks with a running total that aborts the moment it passes the ceiling.

**3. Files are laid out under org/chatbot directories.**

Purely operational — it means an org's data can be located, sized, and deleted
with a single path, which matters the first time someone asks to be forgotten.
Isolation itself comes from the database and the Qdrant filter, never from the
directory layout.

Ingestion is enqueued, not performed. Parsing a 500-page PDF and embedding a
thousand chunks takes minutes; doing it in the request would hold an HTTP
connection open for the duration and lose all of it if the client disconnected.
The endpoint's job is to durably record "this file exists and needs work" and
return — which is exactly what DocumentStatus.PENDING means.
"""

import hashlib
import logging
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import func, select

from app.api.chatbots import CurrentOrg, CurrentOwner, DbSession, owned_chatbot
from app.api.schemas import DocumentOut
from app.config import settings
from app.core.extraction import SUPPORTED_SUFFIXES
from app.db.tables import Document, DocumentStatus, Organization, SourceType
from app.security.clerk import require_org

logger = logging.getLogger(__name__)

# Router-level auth, same as api/chatbots.py: an endpoint added below is
# authenticated whether or not its author remembered to ask for a Principal.
router = APIRouter(
    prefix="/api/bots/{chatbot_id}/documents",
    tags=["documents"],
    dependencies=[Depends(require_org)],
)

# 1MB. Large enough that copying a 25MB upload is 25 iterations rather than
# thousands, small enough that the peak memory of a request is bounded by this
# number and not by the file size.
_COPY_CHUNK_BYTES = 1024 * 1024


def _stored_path(
    org_id: uuid.UUID, chatbot_id: uuid.UUID, document_id: uuid.UUID, suffix: str
) -> Path:
    """Where an uploaded file lives on disk.

    Built entirely from server-generated UUIDs plus an allow-listed suffix, so
    no attacker-controlled string reaches the path. See rule 1 in the module
    docstring.
    """
    return (
        settings.upload_dir / str(org_id) / str(chatbot_id) / f"{document_id}{suffix}"
    )


def _validated_suffix(filename: str | None) -> str:
    """Return the lowercase suffix of an accepted format, or raise 400.

    Allow-list, never a deny-list: a deny-list is a promise to have thought of
    every dangerous extension, and the set of things this app can actually parse
    is four items long and known exactly.

    This checks the *name*, which a client controls, so it is a routing decision
    rather than a security one — the real defence is that the file is only ever
    opened by pypdf/python-docx and never executed. Rejecting early gives the
    user a clear error instead of a failed Celery job ten seconds later.
    """
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"unsupported file type {suffix or '(none)'}; "
                f"supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
            ),
        )
    return suffix


def _write_upload(upload: UploadFile, destination: Path) -> tuple[int, str]:
    """Stream the upload to disk, returning (bytes written, sha256 hex).

    Hashing during the copy rather than in a second pass over the file: one read
    of the bytes instead of two, and the hash is available the moment the write
    finishes.

    On overflow the partial file is removed before raising. Leaving it behind
    would let a client fill the disk with rejected uploads — the size limit
    would be enforced on paper and defeated in practice.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    written = 0

    try:
        # `upload.file` is the underlying synchronous file object. Used directly
        # because this handler is sync (see api/chatbots.py on why) and the
        # async UploadFile methods would need an event loop we are not on.
        with destination.open("wb") as sink:
            while chunk := upload.file.read(_COPY_CHUNK_BYTES):
                written += len(chunk)
                if written > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            "file exceeds the "
                            f"{settings.max_upload_bytes // (1024 * 1024)}MB limit"
                        ),
                    )
                digest.update(chunk)
                sink.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    if written == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="file is empty"
        )

    return written, digest.hexdigest()


@router.get("", response_model=list[DocumentOut])
def list_documents(
    chatbot_id: uuid.UUID, db: DbSession, principal: CurrentOrg
) -> list[Document]:
    """This bot's documents, newest first. Polled by the sources tab.

    `owned_chatbot` runs first and 404s if the bot is not the caller's, so the
    document query below can never be reached with a foreign chatbot_id.
    """
    chatbot = owned_chatbot(db, principal, chatbot_id)
    return list(
        db.scalars(
            select(Document)
            .where(
                # Both columns, though chatbot_id alone would be correct today.
                # The org filter costs nothing (it is the leading column of
                # ix_documents_scope) and means this query is still safe if a
                # future refactor ever loosens the check above.
                Document.org_id == principal.org_id,
                Document.chatbot_id == chatbot.id,
            )
            .order_by(Document.created_at.desc())
        )
    )


@router.post("", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
def upload_document(
    chatbot_id: uuid.UUID,
    db: DbSession,
    principal: CurrentOrg,
    file: Annotated[UploadFile, File()],
) -> Document:
    """Accept a file, record it as PENDING, and queue ingestion.

    Returns 200 with the existing row when the same bytes are re-uploaded to the
    same bot. That is not a convenience: chunk point ids are derived from the
    document id, so a second row for identical content would produce a full
    duplicate set of vectors, and the same passage would then occupy several
    slots in the top-40 candidate list — quietly crowding out other documents.
    """
    chatbot = owned_chatbot(db, principal, chatbot_id)
    organization = db.get(Organization, principal.org_id)
    if organization is None:
        raise HTTPException(status_code=403, detail="organization not found")

    documents_used = (
        db.scalar(
            select(func.count(Document.id)).where(Document.org_id == organization.id)
        )
        or 0
    )
    limit = organization.limits["documents"]
    if documents_used >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "plan_limit_reached",
                "message": (f"the {organization.plan} plan allows {limit} document(s)"),
            },
        )

    suffix = _validated_suffix(file.filename)

    # The id is generated here rather than by the database because it names the
    # file on disk, and we need the path before the row is flushed.
    document_id = uuid.uuid4()
    destination = _stored_path(organization.id, chatbot.id, document_id, suffix)
    _, content_hash = _write_upload(file, destination)

    existing = db.scalar(
        select(Document).where(
            Document.chatbot_id == chatbot.id,
            Document.content_hash == content_hash,
        )
    )
    if existing is not None:
        # Someone re-uploaded a file we already have. Drop the copy we just
        # wrote and hand back the original row.
        destination.unlink(missing_ok=True)
        logger.info(
            "duplicate upload ignored",
            extra={
                "org_id": str(organization.id),
                "chatbot_id": str(chatbot.id),
                "document_id": str(existing.id),
            },
        )
        return existing

    document = Document(
        id=document_id,
        org_id=organization.id,
        chatbot_id=chatbot.id,
        # Stored for display only — never used to build a path.
        filename=file.filename or f"upload{suffix}",
        source_type=SourceType.FILE,
        content_hash=content_hash,
        status=DocumentStatus.PENDING,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    # Enqueued AFTER the commit, deliberately. A worker is fast enough to pick
    # the job up before an uncommitted transaction is visible, and it would then
    # query for a document row that does not exist yet and fail. Commit first,
    # publish second: the worst case becomes a document stuck in PENDING (which
    # is visible and re-queueable) rather than a spurious failure.
    from app.workers.ingest_tasks import ingest_document

    ingest_document.delay(document_id=str(document.id))

    logger.info(
        "queued document for ingestion",
        extra={
            "org_id": str(organization.id),
            "chatbot_id": str(chatbot.id),
            "document_id": str(document.id),
        },
    )
    return document


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    chatbot_id: uuid.UUID,
    document_id: uuid.UUID,
    db: DbSession,
    principal: CurrentOwner,
) -> None:
    """Remove a document, its file, and its vectors.

    Scoped on org, bot AND document id together. Filtering on document_id alone
    would be correct while ids are unique and become a cross-tenant delete the
    moment one is guessable — the same reasoning as `VectorStore.delete_document`,
    applied on the SQL side.
    """
    chatbot = owned_chatbot(db, principal, chatbot_id)

    document = db.scalar(
        select(Document).where(
            Document.id == document_id,
            Document.org_id == principal.org_id,
            Document.chatbot_id == chatbot.id,
        )
    )
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")

    stored = _stored_path(
        document.org_id,
        document.chatbot_id,
        document.id,
        Path(document.filename).suffix.lower(),
    )
    org_id, bot_id, doc_id = (
        str(document.org_id),
        str(document.chatbot_id),
        str(document.id),
    )

    db.delete(document)
    db.commit()

    # Best-effort: the row is the source of truth, and a leftover file is
    # reclaimable disk rather than a correctness problem.
    stored.unlink(missing_ok=True)

    from app.workers.ingest_tasks import purge_document_vectors

    purge_document_vectors.delay(org_id=org_id, chatbot_id=bot_id, document_id=doc_id)

    logger.info(
        "deleted document",
        extra={"org_id": org_id, "chatbot_id": bot_id, "document_id": doc_id},
    )

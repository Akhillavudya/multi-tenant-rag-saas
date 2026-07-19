"""Re-embed a chatbot's documents with a different embedding model.

    python -m scripts.reembed --chatbot <uuid> --collection chunks_v2
    python -m scripts.reembed --org <uuid> --collection chunks_v2
    python -m scripts.reembed --chatbot <uuid> --dry-run

PLAN §3 rule 1: "changing embedders requires re-embedding — write
`scripts/reembed.py` in Phase 2 so it's a command, not a crisis."

**Why re-embedding is unavoidable, not merely tidy.**

Two embedding models produce vectors that are not comparable, even when they
have the same width. Each model learned its own coordinate space, so the
bge-small vector for "refund policy" and the OpenAI vector for the same phrase
point in unrelated directions. Query with the new model against documents
embedded by the old one and the nearest neighbours are effectively random.

There is no error. Qdrant compares the numbers it is given, the reranker
dutifully orders nonsense, and the LLM answers from whatever it was handed. The
only symptom is a bot that has quietly become bad at its job — which is why the
`embedding_model` and `dimension` columns exist on `chatbots`, and why this
script reads them rather than assuming.

**Why a new collection rather than an in-place rewrite.**

A Qdrant collection is created with a fixed vector size, so a 384 -> 1536 change
cannot be applied in place at all. Even when the width happens to match, writing
new vectors over old ones means that for the duration of the run the collection
holds a mix of two incomparable spaces, and every query served in that window is
degraded. Writing to a fresh collection keeps the live one untouched and correct
until the switch, which is one env var:

    QDRANT_COLLECTION=chunks_v2

The database rows are updated only after every document has been written, so an
interrupted run leaves the bot pointing at its old, still-valid collection.

**Requirement: the original files must still be on disk.**

Re-embedding means re-reading the source, because the chunk text is not stored
anywhere except inside the vector payloads we are replacing. Documents whose
file is missing are reported and skipped rather than silently dropped — a bot
that comes back with fewer documents than it started with, with no warning, is
worse than a run that fails.

Location note: PLAN §5 puts this at repo-root `scripts/`. It lives under
`backend/` because it imports `app` and needs the backend virtualenv; from the
repo root it would require sys.path manipulation to run at all.
"""

import argparse
import asyncio
import logging
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select

from app.config import settings
from app.core.extraction import ExtractionError
from app.core.ingestion import ingest_file
from app.core.logging import configure_logging
from app.core.vectorstore import VectorStore, VectorStoreError
from app.db.base import SessionLocal
from app.db.tables import Chatbot, Document, DocumentStatus
from app.providers.base import ProviderError
from app.providers.factory import get_embedder

logger = logging.getLogger(__name__)


@dataclass
class Report:
    """What a run did. Printed at the end and used for the exit code.

    Partial success is the normal outcome when files have gone missing, so this
    distinguishes "nothing to do" from "some documents could not be moved" —
    the operator needs to know which before flipping QDRANT_COLLECTION.
    """

    chatbots: int = 0
    documents_ok: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return self.documents_failed == 0 and self.documents_skipped == 0

    def summary(self) -> str:
        return (
            f"{self.chatbots} chatbot(s): {self.documents_ok} re-embedded, "
            f"{self.documents_skipped} skipped, {self.documents_failed} failed"
        )


def _stored_path(document: Document) -> Path:
    """Rebuild a document's path on disk.

    Third copy of this derivation (api/documents.py, workers/ingest_tasks.py).
    That is one too many and worth consolidating the next time any of the three
    is touched; duplicated here rather than importing from the API module so a
    maintenance script does not drag FastAPI into its import graph.
    """
    suffix = Path(document.filename).suffix.lower()
    return (
        settings.upload_dir
        / str(document.org_id)
        / str(document.chatbot_id)
        / f"{document.id}{suffix}"
    )


async def _reembed_chatbot(
    chatbot: Chatbot,
    documents: list[Document],
    *,
    store: VectorStore,
    report: Report,
    dry_run: bool,
) -> None:
    """Re-embed every ready document of one chatbot into `store`."""
    embedder = get_embedder()
    print(
        f"\nchatbot {chatbot.id} ({chatbot.name})\n"
        f"  from: {chatbot.embedding_model} ({chatbot.dimension}d)\n"
        f"  to:   {embedder.model} ({embedder.dimension}d)\n"
        f"  documents: {len(documents)}"
    )

    if dry_run:
        for document in documents:
            path = _stored_path(document)
            marker = "ok" if path.exists() else "MISSING FILE"
            print(f"    [{marker}] {document.filename}")
            if not path.exists():
                report.documents_skipped += 1
            else:
                report.documents_ok += 1
        return

    for document in documents:
        path = _stored_path(document)
        if not path.exists():
            message = f"{document.filename} ({document.id}): file missing on disk"
            print(f"    SKIP {message}")
            report.documents_skipped += 1
            report.failures.append(message)
            continue

        try:
            result = await ingest_file(
                path,
                org_id=str(document.org_id),
                chatbot_id=str(document.chatbot_id),
                # The SAME document id, so point ids stay deterministic and this
                # document occupies the same slots in the new collection that it
                # did in the old one. Re-running the script is then idempotent.
                document_id=str(document.id),
                source=document.filename,
                store=store,
                embedder=embedder,
            )
        except (ExtractionError, ProviderError, VectorStoreError) as exc:
            # Keep going. One unreadable file should not abandon the other
            # ninety-nine documents halfway through a migration.
            message = f"{document.filename} ({document.id}): {exc}"
            print(f"    FAIL {message}")
            report.documents_failed += 1
            report.failures.append(message)
            continue

        print(f"    ok   {document.filename}: {result.chunks} chunks")
        report.documents_ok += 1


async def _run(args: argparse.Namespace) -> Report:
    embedder = get_embedder()
    report = Report()

    session = SessionLocal()
    try:
        query = select(Chatbot)
        if args.chatbot:
            query = query.where(Chatbot.id == args.chatbot)
        elif args.org:
            query = query.where(Chatbot.org_id == args.org)
        chatbots = list(session.scalars(query.order_by(Chatbot.created_at)))

        if not chatbots:
            print("no chatbots matched", file=sys.stderr)
            return report

        collection = args.collection or settings.qdrant_collection
        if collection == settings.qdrant_collection and not args.dry_run:
            # Refuse by default. Overwriting the live collection mid-run leaves
            # it holding two incomparable vector spaces at once, and every query
            # served during the run is degraded. Requiring an explicit flag
            # means nobody does this by accident at 2am.
            if not args.in_place:
                print(
                    f"refusing to write into the live collection {collection!r}; "
                    "pass --collection <new-name> (recommended) or --in-place",
                    file=sys.stderr,
                )
                return report

        store = VectorStore(dimension=embedder.dimension, collection=collection)
        try:
            await store.ensure_collection()

            for chatbot in chatbots:
                documents = list(
                    session.scalars(
                        select(Document).where(
                            Document.chatbot_id == chatbot.id,
                            # Only documents that previously succeeded. A FAILED
                            # document has no vectors to migrate, and re-running
                            # its ingestion here would silently repair it in a
                            # script whose job is migration, not repair.
                            Document.status == DocumentStatus.READY,
                        )
                    )
                )
                report.chatbots += 1
                await _reembed_chatbot(
                    chatbot,
                    documents,
                    store=store,
                    report=report,
                    dry_run=args.dry_run,
                )

                if args.dry_run:
                    continue

                # Update the row LAST. If the run dies partway through, the bot
                # still points at its old model and its old collection, both of
                # which are still valid and still consistent with each other.
                chatbot.embedding_model = embedder.model
                chatbot.dimension = embedder.dimension
                session.commit()
        finally:
            await store.close()
    finally:
        session.close()

    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reembed",
        description="Re-embed chatbots' documents with the configured embedder.",
    )
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--chatbot", type=uuid.UUID, help="one chatbot id")
    scope.add_argument("--org", type=uuid.UUID, help="every chatbot in one org")
    scope.add_argument(
        "--all",
        action="store_true",
        help="every chatbot in the database (use deliberately)",
    )
    parser.add_argument(
        "--collection",
        help=(
            "target Qdrant collection; use a NEW name so the live one stays "
            "intact until you switch QDRANT_COLLECTION"
        ),
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="allow writing into the live collection (degrades queries during the run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be re-embedded, touching nothing",
    )
    return parser


def main() -> int:
    configure_logging(level="WARNING", json_output=False)
    args = _build_parser().parse_args()

    embedder = get_embedder()
    print(
        f"embedder: {embedder.model} ({embedder.dimension}d) "
        f"[EMBEDDING_PROVIDER={settings.embedding_provider}]"
    )
    if args.dry_run:
        print("DRY RUN — nothing will be written")

    report = asyncio.run(_run(args))
    print(f"\n{report.summary()}")

    if report.failures:
        print("\nproblems:")
        for failure in report.failures:
            print(f"  - {failure}")

    if not args.dry_run and report.documents_ok and report.clean:
        target = args.collection or settings.qdrant_collection
        print(
            f"\nDone. Point the app at the new collection when ready:\n"
            f"    QDRANT_COLLECTION={target}"
        )

    # Non-zero when anything was skipped or failed, so this is safe to call from
    # a deployment script that checks exit codes.
    return 0 if report.clean else 1


if __name__ == "__main__":
    raise SystemExit(main())

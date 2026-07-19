"""Command-line harness for the Phase 1 pipeline.

    python -m app.cli providers
    python -m app.cli ingest ./handbook.pdf
    python -m app.cli ask "how long do refunds take?"

This is how PLAN §6's acceptance criteria become checkable: ingest a PDF, get a
fast, cited, accurate streamed answer in the terminal; set LLM_PROVIDER=groq and
watch it keep working; set RERANK_PROVIDER=noop and watch answers get worse.

Deliberately a thin shell. Every decision lives in `core/` — this file parses
arguments, wires objects together, prints, and translates exceptions into exit
codes. The temptation with a test harness is to let logic accumulate in it,
which produces a CLI that works and an API that does not, because the two
gradually stop sharing a path.

There is no `--llm` flag, on purpose. The acceptance criterion is that switching
provider requires *zero code edits* — an env var and nothing else. A convenience
flag would let me demonstrate the swap without ever proving the property.

Phase 1 is single-tenant, so org and chatbot default to fixed local values. They
are still passed explicitly through every call, because the functions underneath
require them and will keep requiring them when Phase 2 makes them real.
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from app.config import settings
from app.core.extraction import ExtractionError
from app.core.generation import stream_answer
from app.core.ingestion import ingest_file
from app.core.retrieval import retrieve
from app.core.vectorstore import VectorStore, VectorStoreError
from app.providers.base import ProviderError
from app.providers.factory import (
    available_embedders,
    available_llms,
    available_rerankers,
    get_embedder,
)

# Single-tenant placeholders. Phase 2 replaces these with real ids from Clerk.
LOCAL_ORG_ID = "local_org"
LOCAL_CHATBOT_ID = "local_bot"

EXIT_OK = 0
EXIT_ERROR = 1


def _configure_output() -> None:
    """Make the console safe for document text.

    Windows terminals default to a legacy code page, and printing a chunk
    containing an en dash or a curly quote raises UnicodeEncodeError — the
    pipeline works and the CLI crashes while displaying its own success.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


async def _build_store() -> VectorStore:
    """Construct the store with the live embedder's dimension.

    Read from the embedder rather than hardcoded: this is the single value that
    makes swapping EMBEDDING_PROVIDER safe, since a collection built for the
    wrong width can never accept the vectors it will be asked to hold.
    """
    embedder = get_embedder()
    return VectorStore(dimension=embedder.dimension)


async def cmd_providers(_: argparse.Namespace) -> int:
    """Show what is configured and what is available. The fastest way to see
    that an env change took effect before spending a request finding out."""
    print("configured:")
    print(f"  LLM_PROVIDER        = {settings.llm_provider}")
    print(f"  EMBEDDING_PROVIDER  = {settings.embedding_provider}")
    print(f"  RERANK_PROVIDER     = {settings.rerank_provider}")
    print(f"  QDRANT_URL          = {settings.qdrant_url}")
    print(f"  QDRANT_COLLECTION   = {settings.qdrant_collection}")
    print("\navailable:")
    print(f"  llm        : {', '.join(available_llms())}")
    print(f"  embedding  : {', '.join(available_embedders())}")
    print(f"  rerank     : {', '.join(available_rerankers())}")
    return EXIT_OK


async def cmd_ingest(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser()
    store = await _build_store()
    try:
        started = time.perf_counter()
        result = await ingest_file(
            path,
            org_id=args.org,
            chatbot_id=args.bot,
            store=store,
        )
        elapsed = time.perf_counter() - started
        print(result.summary())
        print(f"document_id: {result.document_id}")
        print(f"took {elapsed:.1f}s")
        return EXIT_OK
    finally:
        # Always close, even on failure: an unclosed AsyncClient leaves the
        # event loop with pending connections and prints a warning on exit that
        # looks like a bug in the pipeline.
        await store.close()


async def cmd_ask(args: argparse.Namespace) -> int:
    store = await _build_store()
    try:
        started = time.perf_counter()
        chunks = await retrieve(
            args.question,
            org_id=args.org,
            chatbot_id=args.bot,
            store=store,
            candidates=args.candidates,
            top_k=args.top_k,
        )
        retrieved_at = time.perf_counter()

        if args.show_chunks:
            _print_chunks(chunks)

        first_token_at: float | None = None
        print()
        async for fragment in stream_answer(args.question, chunks):
            if first_token_at is None:
                first_token_at = time.perf_counter()
            # write+flush, not print: print adds a newline per call, and
            # without the flush the terminal buffers and the answer arrives in
            # one lump — which would hide whether streaming works at all.
            sys.stdout.write(fragment)
            sys.stdout.flush()
        print("\n")

        _print_sources(chunks)
        _print_timings(started, retrieved_at, first_token_at)
        return EXIT_OK
    finally:
        await store.close()


def _print_chunks(chunks: list) -> None:
    print(f"\nretrieved {len(chunks)} chunks:")
    for position, chunk in enumerate(chunks, start=1):
        moved = chunk.vector_rank - (position - 1)
        # A positive number means the reranker promoted this chunk past others.
        # This column is how you see the reranker earning its latency.
        arrow = (
            f"  (up {moved} from vector rank {chunk.vector_rank})" if moved > 0 else ""
        )
        preview = chunk.text[:90].replace("\n", " ")
        print(f"  {position}. [{chunk.citation}] score={chunk.rerank_score}{arrow}")
        print(f"     {preview}...")


def _print_sources(chunks: list) -> None:
    if not chunks:
        return
    seen: dict[str, None] = {}
    for chunk in chunks:
        seen.setdefault(chunk.citation, None)
    print("sources: " + "; ".join(seen))


def _print_timings(
    started: float, retrieved_at: float, first_token_at: float | None
) -> None:
    total = time.perf_counter() - started
    retrieval_ms = int((retrieved_at - started) * 1000)
    # Time to first token is the number that decides whether the app FEELS
    # fast. Total time matters far less: a four-second answer that starts in
    # 400ms reads as responsive, and the same answer delivered whole at four
    # seconds reads as broken.
    ttft = f"{int((first_token_at - started) * 1000)}ms" if first_token_at else "n/a"
    print(f"retrieval: {retrieval_ms}ms | first token: {ttft} | total: {total:.1f}s")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="Phase 1 single-tenant RAG harness.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log INFO events")
    sub = parser.add_subparsers(dest="command", required=True)

    tenant_parent = argparse.ArgumentParser(add_help=False)
    tenant_parent.add_argument("--org", default=LOCAL_ORG_ID)
    tenant_parent.add_argument("--bot", default=LOCAL_CHATBOT_ID)

    sub.add_parser("providers", help="show configured and available providers")

    ingest = sub.add_parser("ingest", parents=[tenant_parent], help="ingest a file")
    ingest.add_argument("path", help="path to a .pdf, .docx, .txt or .md file")

    ask = sub.add_parser("ask", parents=[tenant_parent], help="ask a question")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=6)
    ask.add_argument("--candidates", type=int, default=40)
    ask.add_argument(
        "--show-chunks",
        action="store_true",
        help="print the retrieved chunks and how far the reranker moved each",
    )
    return parser


_COMMANDS = {
    "providers": cmd_providers,
    "ingest": cmd_ingest,
    "ask": cmd_ask,
}


async def _run(args: argparse.Namespace) -> int:
    try:
        return await _COMMANDS[args.command](args)
    except (ExtractionError, VectorStoreError, ProviderError) as exc:
        # These are already typed and already carry the detail a user needs —
        # which file, which provider, what to do about it. Printing the message
        # is better than a traceback the user has to read past.
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


def main() -> int:
    _configure_output()
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())

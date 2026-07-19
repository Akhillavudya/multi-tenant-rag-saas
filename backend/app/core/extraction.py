"""File -> page-tagged plain text.

The front door of ingestion: PDF, DOCX, TXT and Markdown all become the same
`list[TextBlock]` that `chunking.chunk_blocks` consumes. Everything after this
point is format-blind, which is why adding .html or .pptx later touches only
this file.

Deliberately PURE: no network, no database, no model. Given the same bytes it
returns the same blocks, so it is tested with small fixture files and no
infrastructure. The I/O orchestration lives in `ingestion.py`.

A note on Hard Rule #2 (no vendor SDK outside `app/providers/`). `pypdf` and
`python-docx` are imported here, directly. The rule exists to keep *swappable AI
providers* behind the factory so `LLM_PROVIDER=groq` stays a config change;
nothing about PDF parsing is selected by an env var, and wrapping a single PDF
library in a provider factory would be ceremony around one implementation. The
rule is applied by intent: AI vendors are swappable and go behind the factory,
file-format libraries are not and do not.

The failure worth knowing about is the scanned PDF. A PDF either carries a real
text layer or is a photograph of a page. `pypdf` reads the first and returns an
EMPTY STRING for the second — no exception, no warning. Silently ingesting a
200-page document as zero chunks, and then answering "I don't have that
information" forever, is the worst possible outcome, so `_extract_pdf` detects
it and raises. OCR is explicitly out of scope for Phase 1.
"""

import logging
from pathlib import Path

from app.core.chunking import TextBlock

logger = logging.getLogger(__name__)

# Lowercase, with the dot, as `Path.suffix` returns them.
SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".pdf", ".docx", ".txt", ".md"})

# A page yielding fewer than this many non-whitespace characters is treated as
# having no text layer. Not zero: pypdf often scrapes a stray ligature or page
# number off a scanned page, so a strict `== ""` test would miss the very case
# this exists to catch.
_MIN_CHARS_PER_PAGE = 16

# Fraction of pages that must have a usable text layer for the document to be
# considered machine-readable. Below this it is almost certainly a scan. Not
# 100%: real PDFs legitimately contain full-page images, cover pages and blank
# separators, and rejecting those would be worse than accepting them.
_MIN_TEXT_PAGE_RATIO = 0.2

# Tried in order for plain-text files. utf-8 first because it is correct and
# fails loudly on non-utf8 input; cp1252 second because it is what Windows
# editors actually produce and it is the most common real-world second guess.
_TEXT_ENCODINGS = ("utf-8", "cp1252")


class ExtractionError(Exception):
    """Base for every failure in this module.

    Typed rather than bare ValueError because the API layer maps these to user
    messages: a password-protected file and an unsupported format are both
    "your upload failed" to a 500 handler, but they are completely different
    instructions to the person who uploaded it.
    """


class UnsupportedFormatError(ExtractionError):
    """A file type we do not parse. The user should convert and retry."""


class EncryptedDocumentError(ExtractionError):
    """Password-protected. The user must remove the password and re-upload."""


class NoTextLayerError(ExtractionError):
    """A scanned/image-only PDF. Needs OCR, which Phase 1 does not do."""


class EmptyDocumentError(ExtractionError):
    """Parsed successfully and contained nothing. Usually an empty template."""


class CorruptDocumentError(ExtractionError):
    """The parser could not read the file at all."""


def extract(path: Path) -> list[TextBlock]:
    """Extract page-tagged text from a supported file.

    Returns one block per page for PDFs, and a single block with `page=None`
    for formats that have no pagination. Raises `ExtractionError` subclasses —
    never returns an empty list, because "no content" is a condition the caller
    must handle explicitly rather than discover later as zero chunks.
    """
    if not path.is_file():
        raise CorruptDocumentError(f"no such file: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFormatError(
            f"cannot extract {suffix or '(no extension)'}; "
            f"supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    if suffix == ".pdf":
        blocks = _extract_pdf(path)
    elif suffix == ".docx":
        blocks = _extract_docx(path)
    else:
        blocks = _extract_plain(path)

    if not blocks:
        raise EmptyDocumentError(f"{path.name} contains no extractable text")

    logger.info(
        "extracted document",
        extra={
            "file": path.name,
            "blocks": len(blocks),
            "chars": sum(len(b.text) for b in blocks),
        },
    )
    return blocks


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------


def _extract_pdf(path: Path) -> list[TextBlock]:
    """One TextBlock per page, 1-indexed to match what a reader sees.

    Page numbers are the whole reason PDFs get per-page blocks: PLAN requires
    citations naming source AND page, and that number has to survive from here
    all the way to the answer. Getting it 0-indexed would make every citation
    off by one — correct-looking and wrong, the hardest kind of bug to notice.
    """
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(path)

        if reader.is_encrypted:
            # Many "encrypted" PDFs use an empty owner password and open fine;
            # trying costs nothing and saves the user a pointless round trip.
            if reader.decrypt("") == 0:
                raise EncryptedDocumentError(
                    f"{path.name} is password-protected; "
                    "remove the password and upload again"
                )

        blocks: list[TextBlock] = []
        pages_with_text = 0
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if len(text) >= _MIN_CHARS_PER_PAGE:
                pages_with_text += 1
            if text:
                blocks.append(TextBlock(text=text, page=number))

    except ExtractionError:
        raise
    except PdfReadError as exc:
        raise CorruptDocumentError(f"{path.name} is not a readable PDF: {exc}") from exc
    except Exception as exc:
        # pypdf raises assorted bare exceptions on malformed files rather than
        # a single typed family, so the catch-all is load-bearing, not lazy.
        raise CorruptDocumentError(f"failed to read {path.name}: {exc}") from exc

    total_pages = len(reader.pages)
    if total_pages and pages_with_text / total_pages < _MIN_TEXT_PAGE_RATIO:
        raise NoTextLayerError(
            f"{path.name} appears to be scanned or image-only "
            f"({pages_with_text}/{total_pages} pages have extractable text); "
            "OCR is not supported yet"
        )

    return blocks


# --------------------------------------------------------------------------
# DOCX
# --------------------------------------------------------------------------


def _extract_docx(path: Path) -> list[TextBlock]:
    """A single block with `page=None`.

    .docx genuinely has no page numbers: pagination is computed by the renderer
    from fonts, margins and paper size, so Word itself does not know where the
    breaks are until it lays the document out. Inventing a page number here
    would produce citations that are confidently wrong, which is worse than a
    citation that honestly says "no page".
    """
    try:
        from docx import Document
        from docx.opc.exceptions import PackageNotFoundError

        document = Document(str(path))
    except PackageNotFoundError as exc:
        raise CorruptDocumentError(
            f"{path.name} is not a valid .docx (a renamed .doc will fail here)"
        ) from exc
    except Exception as exc:
        raise CorruptDocumentError(f"failed to read {path.name}: {exc}") from exc

    parts = [p.text.strip() for p in document.paragraphs if p.text.strip()]

    # Tables are not in `document.paragraphs`, and in real business documents
    # they hold exactly the facts users ask about — pricing tiers, SLA numbers,
    # contact details. Skipping them silently drops the highest-value content.
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                # Pipe-joined so a row stays one semantic line for the chunker
                # rather than dissolving into disconnected fragments.
                parts.append(" | ".join(cells))

    if not parts:
        return []
    return [TextBlock(text="\n\n".join(parts), page=None)]


# --------------------------------------------------------------------------
# Plain text / Markdown
# --------------------------------------------------------------------------


def _extract_plain(path: Path) -> list[TextBlock]:
    """A single block with `page=None`.

    Markdown is read as-is rather than stripped to prose: headings and list
    markers are structure the chunker's separator hierarchy uses, and `##` costs
    two tokens while telling the reader — and the model — what the section is.
    """
    raw = path.read_bytes()
    if not raw.strip():
        return []

    for encoding in _TEXT_ENCODINGS:
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        # Every candidate failed. Decode lossily rather than reject the upload:
        # a document with a few mangled characters is still worth searching,
        # and the log line records that it happened.
        text = raw.decode("utf-8", errors="replace")
        logger.warning(
            "lossy decode: file matched no known encoding",
            extra={"file": path.name, "encodings_tried": list(_TEXT_ENCODINGS)},
        )

    text = text.strip()
    return [TextBlock(text=text, page=None)] if text else []

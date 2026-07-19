"""Document chunking: long text -> retrieval-sized, page-tagged pieces.

The most consequential file in the retrieval path, and the least glamorous.
Everything downstream inherits its mistakes: if a fact gets split so that the
claim lands in one chunk and its subject in another, no embedder, reranker,
prompt or model recovers it. Retrieval quality is capped here, not by the
model choice.

Two design decisions carry the file:

1. **Recursive splitting on a separator hierarchy**, not fixed-size slicing.
   Text has structure — paragraphs, then lines, then sentences — and each level
   is a better cut point than the one below it. We only descend when a piece is
   still too big, so most cuts land on paragraph boundaries and mid-word cuts
   happen only in pathological input.
2. **The token counter is INJECTED, not imported.** Counting tokens properly
   needs the embedding model's own tokenizer, which lives behind
   `app/providers/` (Hard Rule #2). Taking a `count_tokens` callable keeps this
   module pure: no torch import, no model load, instant tests, and the chunker
   stays correct if the embedder is swapped for one that tokenizes differently.
   The default is a crude character heuristic — good enough for tests, NOT for
   ingestion. See `estimate_tokens`.

Sizing: 400 tokens per chunk against bge-small's 512-token limit. The 112-token
headroom is not timidity — it is spent deliberately in `_merge_runt`, and it
absorbs the gap between the estimator and the real tokenizer. Anything the model
would truncate past 512 is text we embedded and then silently threw away.
"""

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Target size. Comfortably under bge-small's 512-token window; see module docs.
DEFAULT_MAX_TOKENS = 400

# 20% of the chunk. Enough to carry a sentence or two of context across a
# boundary; more would inflate the index without improving recall much, since
# the reranker already sees both neighbours as candidates.
DEFAULT_OVERLAP_TOKENS = 80

# Below this a chunk is a "runt": too little context to answer anything, but it
# still occupies a slot in the top-40 candidate list. Merged rather than kept.
MIN_CHUNK_TOKENS = 48

# Ordered best-to-worst cut points. The chunker descends this list only when a
# piece is still oversized, so a cut mid-word requires every level above it to
# have failed. Sentence patterns are regexes because ". " also ends "Dr. " and
# "3.14 " — imperfect, but wrong sentence splits cost far less here than the
# alternative of not trying.
_SEPARATORS: tuple[str, ...] = (
    "\n\n",  # paragraph
    "\n",  # line
    r"(?<=[.!?])\s+",  # sentence
    r"(?<=[;:])\s+",  # clause
    r"\s+",  # word
)

TokenCounter = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    """Rough token count: ~4 characters per token for English prose.

    A stopgap so this module has no hard dependency on a tokenizer. It is
    accurate enough for tests and for reasoning about behaviour, and wrong
    enough that ingestion must pass the real counter — it under-counts code,
    URLs, tables and non-English text badly, all of which tokenize far denser
    than prose. The 112 tokens of headroom below 512 exist partly to absorb
    this error when someone forgets.
    """
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class TextBlock:
    """A span of extracted text with the page it came from.

    Page is `None` for formats with no pagination (.txt, .docx). Keeping it
    optional rather than faking page 1 means a citation can honestly say "no
    page" instead of asserting something false about the source.
    """

    text: str
    page: int | None = None


@dataclass(frozen=True)
class Chunk:
    """One retrieval unit, ready to embed.

    Carries `page_start`/`page_end` rather than a single page because a chunk
    that spans a page break genuinely came from both — and PLAN requires
    citations to name the page. Frozen: once ingestion has embedded a chunk, its
    text is the thing the vector means, and mutating it would silently decouple
    the two.
    """

    text: str
    index: int
    token_count: int
    page_start: int | None = None
    page_end: int | None = None


@dataclass(frozen=True)
class _Segment:
    """Internal: an atomic piece that fits within the size limit."""

    text: str
    page: int | None
    tokens: int


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def chunk_blocks(
    blocks: Sequence[TextBlock],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    count_tokens: TokenCounter = estimate_tokens,
) -> list[Chunk]:
    """Split page-tagged blocks into overlapping chunks.

    The primary entry point — PDFs arrive as one block per page. Blocks are
    chunked as a continuous stream rather than independently, so a paragraph
    running across a page break stays whole and its chunk reports both pages.
    """
    if overlap_tokens >= max_tokens:
        # Would make the retained tail as large as the chunk itself: the packer
        # could never advance. Caught here rather than as a hang.
        raise ValueError(
            f"overlap_tokens ({overlap_tokens}) must be less than "
            f"max_tokens ({max_tokens})"
        )

    segments = _segment(blocks, max_tokens, count_tokens)
    if not segments:
        return []
    return _pack(segments, max_tokens, overlap_tokens)


def chunk_text(
    text: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    count_tokens: TokenCounter = estimate_tokens,
) -> list[Chunk]:
    """Convenience wrapper for unpaginated sources (.txt, .docx, pasted text)."""
    return chunk_blocks(
        [TextBlock(text=text)],
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        count_tokens=count_tokens,
    )


# --------------------------------------------------------------------------
# Stage 1: blocks -> atomic segments
# --------------------------------------------------------------------------


def _segment(
    blocks: Sequence[TextBlock],
    max_tokens: int,
    count_tokens: TokenCounter,
) -> list[_Segment]:
    """Break every block down until each piece fits, preserving its page."""
    segments: list[_Segment] = []
    for block in blocks:
        text = block.text.strip()
        if not text:
            # Blank pages are common in scanned PDFs and would otherwise become
            # empty chunks that match nothing and dilute the candidate list.
            continue
        for piece in _split_recursive(text, max_tokens, count_tokens, 0):
            piece = piece.strip()
            if piece:
                segments.append(
                    _Segment(text=piece, page=block.page, tokens=count_tokens(piece))
                )
    return segments


def _split_recursive(
    text: str,
    max_tokens: int,
    count_tokens: TokenCounter,
    depth: int,
) -> list[str]:
    """Split `text` until every piece fits, descending the separator hierarchy.

    Returns pieces in document order. `depth` indexes `_SEPARATORS`; each level
    is a worse cut point than the last, so we stop descending the moment a piece
    is small enough.
    """
    if count_tokens(text) <= max_tokens:
        return [text]

    if depth >= len(_SEPARATORS):
        # Out of separators: a single unbroken run longer than the limit —
        # minified JS, a base64 blob, a language without spaces. Hard-cut it.
        return _split_by_length(text, max_tokens, count_tokens)

    parts = [p for p in re.split(_SEPARATORS[depth], text) if p and p.strip()]
    if len(parts) <= 1:
        # This separator does not occur; try the next one without recursing on
        # an unchanged string, which would otherwise loop.
        return _split_recursive(text, max_tokens, count_tokens, depth + 1)

    out: list[str] = []
    for part in parts:
        out.extend(_split_recursive(part, max_tokens, count_tokens, depth + 1))
    return out


def _split_by_length(
    text: str,
    max_tokens: int,
    count_tokens: TokenCounter,
) -> list[str]:
    """Last-resort character cut for text with no usable separator.

    Uses the measured token/char ratio of *this* text rather than a constant,
    because the inputs that reach here are exactly the ones that tokenize
    unusually — a base64 blob is far denser than prose.
    """
    tokens = count_tokens(text)
    chars_per_token = max(1, len(text) // max(1, tokens))
    window = max(1, max_tokens * chars_per_token)
    return [text[i : i + window] for i in range(0, len(text), window)]


# --------------------------------------------------------------------------
# Stage 2: segments -> overlapping chunks
# --------------------------------------------------------------------------


def _pack(
    segments: list[_Segment],
    max_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Greedily fill chunks, carrying an overlapping tail into the next.

    Greedy rather than optimal: an optimal packing (minimum chunks, or most
    balanced sizes) would reorder or rebalance across boundaries, and document
    order is exactly what gives a chunk its coherence. Filling to the limit in
    order is both the simplest and the most faithful strategy.
    """
    chunks: list[Chunk] = []
    current: list[_Segment] = []
    current_tokens = 0

    for seg in segments:
        if current and current_tokens + seg.tokens > max_tokens:
            chunks.append(_emit(current, len(chunks)))
            current = _overlap_tail(current, overlap_tokens)
            current_tokens = sum(s.tokens for s in current)
        current.append(seg)
        current_tokens += seg.tokens

    if current:
        chunks.append(_emit(current, len(chunks)))

    return _merge_runt(chunks)


def _overlap_tail(segments: list[_Segment], overlap_tokens: int) -> list[_Segment]:
    """Return the trailing segments to repeat at the start of the next chunk.

    Never returns the whole list: if it did, the next chunk would begin already
    holding everything the last one did and the packer would stop advancing —
    an infinite loop rather than a wrong answer.
    """
    if overlap_tokens <= 0 or len(segments) <= 1:
        return []

    tail: list[_Segment] = []
    total = 0
    # Walk backwards, stopping one short of consuming the entire chunk.
    for seg in reversed(segments[1:]):
        if total + seg.tokens > overlap_tokens:
            break
        tail.insert(0, seg)
        total += seg.tokens
    return tail


def _emit(segments: list[_Segment], index: int) -> Chunk:
    """Join segments into a Chunk, recording the page span they cover."""
    pages = [s.page for s in segments if s.page is not None]
    return Chunk(
        text=" ".join(s.text for s in segments),
        index=index,
        token_count=sum(s.tokens for s in segments),
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
    )


def _merge_runt(chunks: list[Chunk]) -> list[Chunk]:
    """Fold an undersized final chunk back into its predecessor.

    A document rarely divides evenly, so the last chunk is often a fragment —
    "...and that concludes section 4." It carries no answerable content but
    still competes for one of the 40 candidate slots at search time.

    Merging can push the result slightly over `max_tokens`. That is the headroom
    below 512 being spent on purpose: a chunk at 430 tokens is fine, an orphan
    at 12 tokens is permanently useless. Only the *final* chunk is treated this
    way — a runt anywhere else means the separator hierarchy misbehaved, which
    is a bug worth seeing rather than smoothing over.
    """
    if len(chunks) < 2 or chunks[-1].token_count >= MIN_CHUNK_TOKENS:
        return chunks

    runt = chunks.pop()
    prev = chunks.pop()
    pages = [
        p
        for p in (prev.page_start, prev.page_end, runt.page_start, runt.page_end)
        if p is not None
    ]
    chunks.append(
        Chunk(
            text=f"{prev.text} {runt.text}",
            index=prev.index,
            token_count=prev.token_count + runt.token_count,
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
        )
    )
    return chunks

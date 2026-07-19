"""Generation: retrieved chunks become a streamed, cited answer.

The last stage. Formats the chunks into a context block, renders the grounded
prompt around it, and streams the model's reply.

    chunks -> build_context() -> render_grounded_answer() -> llm.stream()

Two behaviours here are worth more than the code that implements them.

**No chunks means no LLM call.** If retrieval found nothing, this yields the
refusal string directly. That saves a paid round-trip on a request that cannot
succeed, and — more importantly — it *guarantees* the exact refusal sentence
rather than hoping the model reproduces it. `prompts.is_refusal` can then be
trusted for the one case we know for certain is a refusal.

**Excerpt headers are the literal citation string.** Each chunk is introduced by
`[handbook.pdf p.12]`, exactly the form the prompt asks the model to emit. The
model copies rather than composes, which is the difference between a parseable
citation and a plausible-looking invention. Numbered headers like `[1]` were
avoided for precisely this reason: the model would cite `[1]`, and a number is
meaningless to the person reading the answer.
"""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from app.core.chunking import TokenCounter, estimate_tokens
from app.core.retrieval import RetrievedChunk
from app.prompts import REFUSAL, render_grounded_answer
from app.providers.base import LLMProvider, Message
from app.providers.factory import get_llm

logger = logging.getLogger(__name__)

# Ceiling for the answer itself. Generous enough for a multi-part answer with
# citations, small enough that a runaway generation cannot bill unbounded.
DEFAULT_MAX_ANSWER_TOKENS = 800

# Ceiling for the context block. Six chunks of ~400 tokens is ~2400, so this
# leaves headroom; it exists to bound the pathological case rather than to trim
# normal traffic. Well inside every provider's window, deliberately — the
# cheapest context window in the fleet is the one that has to fit.
DEFAULT_MAX_CONTEXT_TOKENS = 3000

_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class ContextBlock:
    """The formatted context plus what was actually included.

    `used` is not the same as the chunks passed in — the budget may drop the
    tail — and the citations shown to a user must reflect what the model
    actually saw. Returning both together makes it impossible to cite a chunk
    that was trimmed.
    """

    text: str
    used: list[RetrievedChunk]
    dropped: int

    @property
    def citations(self) -> list[str]:
        """Unique citations, in the order the model encountered them."""
        seen: dict[str, None] = {}
        for chunk in self.used:
            seen.setdefault(chunk.citation, None)
        return list(seen)


def build_context(
    chunks: list[RetrievedChunk],
    *,
    max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    count_tokens: TokenCounter = estimate_tokens,
) -> ContextBlock:
    """Format chunks into the context block the prompt expects.

    Order is preserved from retrieval — best first. That matters because models
    attend most strongly to the beginning and end of a long context and least to
    the middle ("lost in the middle"), so the reranker's top result should not
    be buried. It also means a truncated context loses the *least* relevant
    chunks, which is the only acceptable direction to lose them in.

    The token count is APPROXIMATE, and knowingly so. `LLMProvider` has no
    `count_tokens`: unlike the embedder, there is no cheap local tokenizer to
    ask — Gemini's is a network call, and each vendor's differs. So this budget
    uses the character heuristic and is set conservatively low, because the cost
    of a slightly-too-small context is a marginally worse answer while the cost
    of a too-large one is a hard provider error mid-stream.
    """
    used: list[RetrievedChunk] = []
    parts: list[str] = []
    total = 0
    seen_text: set[str] = set()

    for chunk in chunks:
        text = chunk.text.strip()
        if not text:
            continue
        if text in seen_text:
            # Overlapping chunks can be byte-identical when a short document is
            # split at a boundary. Sending the same passage twice spends budget
            # and reads to the model as corroboration it has not actually got.
            continue

        entry = f"[{chunk.citation}]\n{text}"
        cost = count_tokens(entry)
        if total + cost > max_tokens and used:
            # `and used`: the first chunk is admitted even if it alone exceeds
            # the budget. An empty context is a guaranteed refusal, which is a
            # worse outcome than an oversized one the provider may still accept.
            break

        seen_text.add(text)
        used.append(chunk)
        parts.append(entry)
        total += cost

    return ContextBlock(
        text=_SEPARATOR.join(parts),
        used=used,
        dropped=len(chunks) - len(used),
    )


async def stream_answer(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    llm: LLMProvider | None = None,
    history: list[Message] | None = None,
    max_tokens: int = DEFAULT_MAX_ANSWER_TOKENS,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
) -> AsyncIterator[str]:
    """Stream a grounded answer to `question` from `chunks`.

    Yields text fragments as they arrive. Streaming rather than returning a
    string because perceived latency is dominated by time-to-first-token: a
    four-second answer that starts in 400ms feels responsive, and the same
    answer delivered whole at four seconds feels broken.

    `history` is accepted but not yet used for retrieval — Phase 3 adds query
    rewriting so that a follow-up like "what about refunds?" is expanded with
    the earlier turns before it is embedded. Passing it through now keeps the
    signature stable.
    """
    if not question.strip():
        raise ValueError("question must not be empty")

    if not chunks:
        # Nothing was retrieved. Answer without a provider call: it cannot
        # succeed, and this guarantees the exact refusal string.
        logger.info("no context; returning refusal without an LLM call")
        yield REFUSAL
        return

    llm = llm or get_llm()
    context = build_context(chunks, max_tokens=max_context_tokens)
    if not context.used:
        yield REFUSAL
        return

    system = render_grounded_answer(context.text)
    messages = [*(history or []), Message(role="user", content=question)]

    logger.info(
        "generating answer",
        extra={
            "provider": llm.name,
            "chunks_used": len(context.used),
            "chunks_dropped": context.dropped,
            "citations": context.citations,
        },
    )

    async for fragment in llm.stream(
        system=system, messages=messages, max_tokens=max_tokens
    ):
        yield fragment


async def answer(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    llm: LLMProvider | None = None,
    max_tokens: int = DEFAULT_MAX_ANSWER_TOKENS,
) -> str:
    """Non-streaming convenience wrapper: join the stream.

    Exists because tests and scripts want a string, and because the contract is
    stream-only by design — a whole answer is trivially built from a stream
    while the reverse is impossible.
    """
    return "".join(
        [
            fragment
            async for fragment in stream_answer(
                question, chunks, llm=llm, max_tokens=max_tokens
            )
        ]
    )

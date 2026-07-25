"""One chat turn: question in, streamed grounded answer out.

The orchestration layer PLAN §2 describes as "the thing that must be fast":

    question -> (rewrite for follow-ups) -> retrieve -> rerank -> stream -> log

**This module never touches the database.** Not because layering is a virtue in
itself, but because every DB call here would be a blocking call inside an async
streaming path, and because the pipeline is then testable with two stubs and no
schema. Conversations, messages, usage counters and auth all live in
`api/chat.py`, which owns the request. This file owns the answer.

**It yields events, not strings.** A caller needs more than tokens: it needs to
know when the first one arrived (time-to-first-token is the number that decides
whether the product feels fast), what was cited, and how long the whole turn
took — none of which fit in a stream of text fragments. So `run_turn` yields
typed events and finishes with a `TurnCompleted` carrying everything the caller
must persist. The SSE encoder in `api/chat.py` is then a dumb translator, which
is exactly what a transport layer should be.

**Errors are raised, not yielded.** `ProviderError` propagates to the caller,
which is holding the reservation that has to be refunded and the HTTP response
that has to carry an error frame. Swallowing it here would leave both decisions
to a caller that no longer knows anything went wrong.
"""

import logging
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from app.config import settings
from app.core.chunking import estimate_tokens
from app.core.generation import DEFAULT_MAX_ANSWER_TOKENS, stream_answer
from app.core.retrieval import (
    DEFAULT_CANDIDATES,
    DEFAULT_TOP_K,
    RetrievedChunk,
    retrieve,
)
from app.core.vectorstore import VectorStore
from app.prompts import is_refusal
from app.providers.base import EmbeddingProvider, LLMProvider, Message, RerankProvider
from app.providers.factory import available_llms, get_llm, get_llm_with_fallback

logger = logging.getLogger(__name__)

# A follow-up shorter than this many words is assumed to depend on the previous
# turn. Tuned by what actually happens in chat: "how much does it cost?" is 5
# words and meaningless alone, while anything long enough to exceed this
# threshold almost always carries its own subject.
FOLLOW_UP_MAX_WORDS = 8

# Hard ceiling on one question. Not a validation nicety: the question is
# embedded (bge truncates past 512 tokens, silently) and replayed to the model,
# so an unbounded one is both a cost and a silent-truncation bug.
MAX_QUESTION_CHARS = 4_000


@dataclass(frozen=True)
class Citation:
    """One source the answer drew on, in the shape the client renders.

    Carries `document_id` alongside the display string so Phase 4's dashboard
    can link a citation to the document it came from, and so a source that has
    since been deleted is identifiable rather than merely unclickable.
    """

    citation: str
    document_id: str
    source: str
    page_start: int | None = None
    page_end: int | None = None
    score: float | None = None

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk) -> "Citation":
        return cls(
            citation=chunk.citation,
            document_id=chunk.hit.document_id,
            source=chunk.hit.source,
            page_start=chunk.hit.page_start,
            page_end=chunk.hit.page_end,
            score=chunk.rerank_score,
        )

    def as_dict(self) -> dict:
        return {
            "citation": self.citation,
            "document_id": self.document_id,
            "source": self.source,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "score": self.score,
        }


@dataclass(frozen=True)
class TokenChunk:
    """A fragment of the answer as it arrives."""

    text: str


@dataclass
class TurnCompleted:
    """The finished turn: everything the caller has to persist or report.

    Not frozen — `api/chat.py` fills in nothing, but the timing fields are
    assembled progressively as the stream runs, and a frozen object would mean
    rebuilding it per token.

    `tokens_in`/`tokens_out` are ESTIMATES and the docstring says so where the
    numbers are produced (`_estimate_prompt_tokens`). Nothing billable depends
    on them: plan caps count messages.
    """

    answer: str
    citations: list[Citation] = field(default_factory=list)
    provider: str = ""
    chunks_retrieved: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    ttft_ms: int | None = None
    refused: bool = False

    def citation_dicts(self) -> list[dict]:
        return [citation.as_dict() for citation in self.citations]


ChatEvent = TokenChunk | TurnCompleted


def build_retrieval_query(question: str, history: Sequence[Message]) -> str:
    """Expand a short follow-up with the previous question before embedding it.

    "And what about refunds?" is a perfectly clear question to a human reading
    the transcript and an almost content-free one to an embedding model: the
    subject lives in the previous turn, so the vector lands near documents about
    the word "refund" in general rather than near the topic being discussed.
    Prefixing the last user question puts the subject back.

    A deliberate heuristic, not a rewrite. The textbook answer is to ask the LLM
    to rewrite the question given the history, which is better — and costs an
    extra model round trip on the latency-critical path of every follow-up, on a
    free tier with a request quota. This costs nothing, helps the common case,
    and its failure mode is a slightly noisier query rather than a wrong one.
    Upgrading to LLM rewriting later changes only this function.

    Long questions are left alone: anything past the word threshold carries its
    own subject, and gluing the previous question onto it would dilute the very
    terms the retriever should be matching.
    """
    question = question.strip()
    if not history or len(question.split()) > FOLLOW_UP_MAX_WORDS:
        return question

    previous = next(
        (message.content for message in reversed(history) if message.role == "user"),
        None,
    )
    if not previous:
        return question

    return f"{previous.strip()} {question}"


def resolve_llm(model_overrides: dict | None) -> LLMProvider:
    """Pick the model for this bot: its override, else the configured default.

    PLAN §3 rule 4 — "a future premium client can run Claude while everyone else
    runs Gemini, same codebase". The override is a column on `chatbots`, written
    only through the authenticated PATCH endpoint by that org's own owner.

    Validated against the factory's registry rather than passed through, and
    that check is the security-relevant line: without it the value would reach
    `get_llm(name)`, and a JSON blob from a request body would be choosing which
    vendor to call. An unknown name is logged and ignored rather than raising —
    a typo in one bot's settings must not take that bot offline when the default
    provider works perfectly.

    Note the asymmetry: an overridden bot gets NO automatic failover. Falling
    back from a deliberately chosen premium model to the free default would
    silently answer with a model the customer did not ask for and is not paying
    for, which is worse than a clean error.
    """
    override = (model_overrides or {}).get("llm_provider")
    if not override:
        return get_llm_with_fallback()

    if override not in available_llms():
        logger.warning(
            "ignoring unknown llm_provider override",
            extra={"override": override, "valid": available_llms()},
        )
        return get_llm_with_fallback()

    return get_llm(override)


async def run_turn(
    question: str,
    *,
    org_id: str,
    chatbot_id: str,
    store: VectorStore,
    history: Sequence[Message] = (),
    persona: str | None = None,
    model_overrides: dict | None = None,
    llm: LLMProvider | None = None,
    embedder: EmbeddingProvider | None = None,
    reranker: RerankProvider | None = None,
    candidates: int = DEFAULT_CANDIDATES,
    top_k: int = DEFAULT_TOP_K,
    max_answer_tokens: int = DEFAULT_MAX_ANSWER_TOKENS,
) -> AsyncIterator[ChatEvent]:
    """Run one turn, yielding `TokenChunk`s and finally one `TurnCompleted`.

    `org_id` and `chatbot_id` are required keyword arguments passed straight to
    `retrieve`, which passes them to the mandatory Qdrant filter. There is no
    path through this function that searches without them (Hard Rule #3).

    Raises `ValueError` for an empty or oversized question and `ProviderError`
    if generation fails — see the module docstring on why errors are raised
    rather than yielded.
    """
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(
            f"question is {len(question)} characters; the limit is {MAX_QUESTION_CHARS}"
        )

    started = time.perf_counter()
    log = {"org_id": org_id, "chatbot_id": chatbot_id}

    # Trim history before it is used anywhere. It is replayed to the model on
    # every turn, so an unbounded transcript grows the prompt — and therefore
    # the latency and the cost — without limit. Keep the MOST RECENT messages:
    # the near turns are what a follow-up refers to.
    trimmed = list(history)[-settings.chat_history_messages :]

    retrieval_query = build_retrieval_query(question, trimmed)
    chunks = await retrieve(
        retrieval_query,
        org_id=org_id,
        chatbot_id=chatbot_id,
        store=store,
        embedder=embedder,
        reranker=reranker,
        candidates=candidates,
        top_k=top_k,
    )

    provider = llm or resolve_llm(model_overrides)
    result = TurnCompleted(
        answer="",
        citations=[Citation.from_chunk(chunk) for chunk in chunks],
        provider=provider.name,
        chunks_retrieved=len(chunks),
        tokens_in=_estimate_prompt_tokens(question, chunks, trimmed),
    )

    parts: list[str] = []
    first_token_at: float | None = None

    async for fragment in stream_answer(
        question,
        chunks,
        llm=provider,
        history=list(trimmed),
        persona=persona,
        max_tokens=max_answer_tokens,
    ):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        parts.append(fragment)
        yield TokenChunk(text=fragment)

    result.answer = "".join(parts)
    result.tokens_out = estimate_tokens(result.answer)
    result.latency_ms = _elapsed_ms(started)
    result.ttft_ms = (
        int((first_token_at - started) * 1000) if first_token_at is not None else None
    )
    result.refused = is_refusal(result.answer)

    if result.refused:
        # Citations on a refusal would be a lie: nothing was used. Clearing them
        # also stops the widget rendering "sources" beneath "I don't have that
        # information", which reads as a bug to a user.
        result.citations = []

    logger.info(
        "chat turn complete",
        extra={
            **log,
            "provider": result.provider,
            "chunks": result.chunks_retrieved,
            "refused": result.refused,
            "ttft_ms": result.ttft_ms,
            "latency_ms": result.latency_ms,
            "tokens_out": result.tokens_out,
            "rewritten_query": retrieval_query != question,
        },
    )
    yield result


def _estimate_prompt_tokens(
    question: str, chunks: Sequence[RetrievedChunk], history: Sequence[Message]
) -> int:
    """Approximate the size of what we sent the model.

    Approximate and knowingly so, for the reason `generation.build_context`
    already documents: `LLMProvider` yields text and nothing else, so there is no
    vendor usage number to read without changing a Phase 1 contract that five
    providers implement. Adding `count_tokens` to the LLM contract would mean a
    network call per vendor (Gemini's tokenizer is remote) on the hot path.

    The estimate excludes the fixed prompt template (~350 tokens), so it
    understates by a constant. That is fine for what these numbers are for —
    spotting the day a prompt doubles in size — and it would not be fine if
    anything were billed on it, which is why nothing is.
    """
    return (
        estimate_tokens(question)
        + sum(estimate_tokens(chunk.text) for chunk in chunks)
        + sum(estimate_tokens(message.content) for message in history)
    )


def _elapsed_ms(since: float) -> int:
    """Milliseconds since a `perf_counter` reading — monotonic, so an NTP
    correction mid-request cannot produce a negative latency."""
    return int((time.perf_counter() - since) * 1000)

"""The chat endpoints: SSE streaming, metered, rate limited, logged.

Two routes, one pipeline:

    POST /widget/chat              X-Bot-Key      channel=widget      public
    POST /api/bots/{id}/chat       Clerk JWT      channel=playground  dashboard

**Why two routes rather than one that accepts either credential.**

A single endpoint would need to look at the request and decide which credential
it is holding — and "guess the credential type" is how authentication-confusion
bugs are written. Here each route depends on exactly one authenticator, and the
route is what decides which is acceptable. A request carrying both a bot key and
a session token is not ambiguous: whichever route it reached honours its own and
ignores the other.

**Request lifecycle, and why this order.**

    1. authenticate            -> org + chatbot, server-side, never from the body
    2. rate limit              -> cheapest possible rejection of a flood
    3. idempotency replay      -> a retry costs nothing and bills nothing
    4. reserve one message     -> the plan cap, claimed BEFORE the work
    5. resolve conversation    -> forged ids start a new thread, never join one
    6. stream the answer       -> tokens leave as they arrive
    7. persist + finish        -> transcript, timings, citations
    8. on failure, refund      -> the customer is not charged for silence

Steps 2 and 3 come before 4 deliberately. Reserving quota for a request we are
about to reject on rate limit, or answer from storage, would consume a message
the customer never sent — and `test_limits.py` exists to make sure the counter
means what it says.

**Blocking database calls inside an async endpoint.**

These handlers are `async def` because they stream, but SQLAlchemy's `Session` is
blocking, and a blocking call on the event loop stalls every other in-flight
request. Every DB call here therefore goes through `run_in_threadpool`.

The streaming half goes further and uses its own short-lived session rather than
the request-scoped one: generation takes seconds, and holding a pooled Postgres
connection open for the duration of an LLM call is how a connection pool is
exhausted by twenty concurrent users. Pre-flight commits and lets its connection
go; the transcript write opens a fresh one at the end.
"""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.api.chatbots import CurrentOrg, DbSession, owned_chatbot
from app.api.schemas import ChatRequest
from app.config import settings
from app.core.chat import TokenChunk, TurnCompleted, run_turn
from app.core.metering import (
    PlanLimitExceededError,
    UsageSnapshot,
    release_message,
    reserve_message,
)
from app.core.vectorstore import VectorStore, VectorStoreError
from app.db.base import SessionLocal
from app.db.tables import (
    Channel,
    Chatbot,
    Conversation,
    Message,
    MessageRole,
    Organization,
)
from app.providers.base import Message as ProviderMessage
from app.providers.base import ProviderError
from app.security.bot_key import BotContext, require_bot
from app.security.clerk import require_org
from app.security.rate_limit import RateLimitExceededError, get_rate_limiter

logger = logging.getLogger(__name__)

widget_router = APIRouter(
    prefix="/widget",
    tags=["chat"],
    # Router-level, like every other authenticated router in this codebase: an
    # endpoint added below is authenticated whether or not its author asked.
    dependencies=[Depends(require_bot)],
)

playground_router = APIRouter(
    prefix="/api/bots/{chatbot_id}",
    tags=["chat"],
    dependencies=[Depends(require_org)],
)


@dataclass(frozen=True)
class ChatScope:
    """The resolved tenant scope of one chat request.

    Built by each route from its own credential and passed to shared code, so
    the pipeline below cannot tell — and must not care — how the caller
    authenticated. Every value here came from the server.
    """

    organization: Organization
    chatbot: Chatbot
    channel: Channel
    rate_limit_identity: str

    @property
    def org_id(self) -> uuid.UUID:
        return self.organization.id

    @property
    def chatbot_id(self) -> uuid.UUID:
        return self.chatbot.id

    @property
    def log(self) -> dict:
        return {"org_id": str(self.org_id), "chatbot_id": str(self.chatbot_id)}


# --------------------------------------------------------------------------
# Server-Sent Events
# --------------------------------------------------------------------------
#
# The wire format is four lines of code and worth understanding rather than
# importing: an SSE frame is `event: <name>`, one or more `data: <line>`, and a
# blank line that terminates it. The blank line is the frame delimiter, which is
# why every payload here is JSON — a raw answer fragment containing "\n\n" would
# otherwise end the frame early and split one token into two events.


def _sse(event: str, data: dict) -> str:
    """Encode one SSE frame."""
    # `separators` trims the whitespace json.dumps adds by default. On a stream
    # of thousands of single-token frames that is a measurable fraction of the
    # bytes on the wire, for zero readability cost on a machine-read format.
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


SSE_HEADERS = {
    # Without this a shared proxy may serve one user's answer to another.
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # nginx buffers proxied responses by default, which holds the whole answer
    # until generation finishes and turns a streaming endpoint into a slow
    # non-streaming one. This header is the documented opt-out, and it is
    # ignored harmlessly by every proxy that does not implement it.
    "X-Accel-Buffering": "no",
}


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@widget_router.post("/chat")
async def widget_chat(
    request: Request,
    payload: ChatRequest,
    context: Annotated[BotContext, Depends(require_bot)],
) -> StreamingResponse:
    """Chat from an embedded widget, authenticated by the public bot key.

    The scope comes entirely from the key. Note what is NOT read from the body:
    the chatbot, the organization, the model, the prompt. A visitor on a
    customer's website can choose their question and nothing else.
    """
    scope = ChatScope(
        organization=context.organization,
        chatbot=context.chatbot,
        channel=Channel.WIDGET,
        # Per KEY, so one busy site cannot throttle another's widget.
        rate_limit_identity=context.rate_limit_key,
    )
    return await _chat(request, payload, scope)


@playground_router.post("/chat")
async def playground_chat(
    request: Request,
    chatbot_id: uuid.UUID,
    payload: ChatRequest,
    db: DbSession,
    principal: CurrentOrg,
) -> StreamingResponse:
    """Chat from the dashboard's playground, authenticated by a session token.

    Same pipeline, same metering — a message tested by the owner costs the same
    quota as one from a visitor, because it costs us the same. `channel`
    distinguishes them so Phase 6 can answer "how much of this plan was spent on
    testing?".
    """
    chatbot = await run_in_threadpool(owned_chatbot, db, principal, chatbot_id)
    organization = await run_in_threadpool(db.get, Organization, principal.org_id)
    if organization is None:
        raise HTTPException(status_code=403, detail="organization not found")

    scope = ChatScope(
        organization=organization,
        chatbot=chatbot,
        channel=Channel.PLAYGROUND,
        # Per ORG here, not per key: the dashboard has no bot key, and the
        # abusable identity is the authenticated tenant.
        rate_limit_identity=f"org:{principal.org_id}",
    )
    return await _chat(request, payload, scope)


# --------------------------------------------------------------------------
# The shared pipeline
# --------------------------------------------------------------------------


async def _chat(
    request: Request, payload: ChatRequest, scope: ChatScope
) -> StreamingResponse:
    """Pre-flight everything that can fail with a real status code, then stream.

    The split matters: once the first byte of a 200 has been sent, the status
    code is spent and every later failure can only be an `error` frame inside
    the stream. So anything that deserves a 429 or a 503 has to be decided
    here, before the response begins.
    """
    limiter = get_rate_limiter()
    try:
        await limiter.enforce(scope.rate_limit_identity)
    except RateLimitExceededError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "rate_limited",
                "message": "too many messages; please slow down",
                "retry_after": exc.decision.retry_after,
            },
            headers=exc.decision.headers(),
        ) from exc

    db: Session = SessionLocal()
    try:
        conversation = await run_in_threadpool(
            _resolve_conversation, db, scope, payload
        )

        # Idempotent retry: the same client_message_id in the same conversation
        # already has an answer. Replay it — no LLM call, no metering, no second
        # row. This is what makes a dropped connection safe to retry.
        replayed = await run_in_threadpool(
            _find_replayable_answer, db, conversation.id, payload.client_message_id
        )
        if replayed is not None:
            logger.info(
                "replaying idempotent chat request",
                extra={**scope.log, "conversation_id": str(conversation.id)},
            )
            return StreamingResponse(
                _replay_stream(conversation.id, replayed),
                media_type="text/event-stream",
                headers=SSE_HEADERS,
            )

        try:
            usage = await run_in_threadpool(reserve_message, db, scope.organization)
        except PlanLimitExceededError as exc:
            # PLAN §6: "the 101st free-tier message is politely refused". Politely
            # means a structured payload the widget can render as a sentence,
            # not a bare status code.
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=exc.payload(),
            ) from exc

        history = await run_in_threadpool(_load_history, db, conversation.id)
    finally:
        # Released before generation starts. Holding it across an LLM call would
        # tie up one Postgres connection per concurrent chat, for seconds.
        db.close()

    store = _vector_store(request)

    return StreamingResponse(
        _chat_stream(
            scope=scope,
            payload=payload,
            conversation_id=conversation.id,
            history=history,
            usage=usage,
            store=store,
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


async def _chat_stream(
    *,
    scope: ChatScope,
    payload: ChatRequest,
    conversation_id: uuid.UUID,
    history: list[ProviderMessage],
    usage: UsageSnapshot,
    store: VectorStore,
) -> AsyncIterator[str]:
    """Yield the SSE frames of one answer, then persist it.

    Frame order is `meta` (immediately), `token`* (as generated), then exactly
    one of `done` or `error`.

    `meta` first, before any model work, is deliberate: it carries the
    conversation id, so a client whose connection drops mid-answer already knows
    which thread to resume and which id to retry with.
    """
    yield _sse(
        "meta",
        {
            "conversation_id": str(conversation_id),
            "chatbot_id": str(scope.chatbot_id),
            "usage": usage.as_dict(),
        },
    )

    result: TurnCompleted | None = None
    delivered = False

    try:
        async for event in run_turn(
            payload.question,
            org_id=str(scope.org_id),
            chatbot_id=str(scope.chatbot_id),
            store=store,
            history=history,
            persona=scope.chatbot.system_prompt,
            model_overrides=scope.chatbot.model_overrides,
        ):
            if isinstance(event, TokenChunk):
                delivered = True
                yield _sse("token", {"text": event.text})
            else:
                result = event

    except (ProviderError, VectorStoreError, ValueError) as exc:
        # The answer did not happen, so the reservation is refunded. Without
        # this, a provider outage would quietly spend a customer's monthly quota
        # on answers they never received.
        await run_in_threadpool(_refund, scope.org_id)
        logger.warning(
            "chat turn failed",
            extra={**scope.log, "error": str(exc), "delivered_tokens": delivered},
        )
        yield _sse(
            "error",
            {
                "error": _error_code(exc),
                # The typed provider errors are already written for humans and
                # carry no secrets — they name the provider and the condition,
                # never a key or a prompt.
                "message": str(exc),
                "conversation_id": str(conversation_id),
            },
        )
        return

    except asyncio.CancelledError:
        # The client went away mid-answer. The work was done and the tokens were
        # generated, so the message stays counted — refunding here would let a
        # client get free answers by disconnecting just before the end.
        logger.info(
            "client disconnected during chat",
            extra={**scope.log, "conversation_id": str(conversation_id)},
        )
        raise

    if result is None:  # pragma: no cover - run_turn always ends with one
        await run_in_threadpool(_refund, scope.org_id)
        yield _sse(
            "error", {"error": "internal_error", "message": "no answer produced"}
        )
        return

    message_id = await run_in_threadpool(
        _persist_turn,
        scope,
        conversation_id,
        payload,
        result,
    )

    yield _sse(
        "done",
        {
            "conversation_id": str(conversation_id),
            "message_id": str(message_id) if message_id else None,
            "citations": result.citation_dicts(),
            "refused": result.refused,
            "provider": result.provider,
            "latency_ms": result.latency_ms,
            "ttft_ms": result.ttft_ms,
            "usage": usage.as_dict(),
        },
    )


async def _replay_stream(
    conversation_id: uuid.UUID, stored: Message
) -> AsyncIterator[str]:
    """Re-send a stored answer for an idempotent retry.

    Delivered as a single `token` frame rather than re-chunked: the client's
    renderer appends fragments either way, and pretending to stream something
    that was read from a database in one query would be theatre.
    """
    yield _sse(
        "meta",
        {"conversation_id": str(conversation_id), "replayed": True},
    )
    yield _sse("token", {"text": stored.content})
    yield _sse(
        "done",
        {
            "conversation_id": str(conversation_id),
            "message_id": str(stored.id),
            "citations": stored.citations or [],
            "replayed": True,
            "latency_ms": stored.latency_ms,
        },
    )


# --------------------------------------------------------------------------
# Database work (all of it runs in a threadpool)
# --------------------------------------------------------------------------


def _resolve_conversation(
    db: Session, scope: ChatScope, payload: ChatRequest
) -> Conversation:
    """Continue the named conversation, or start a new one.

    `conversation_id` is the only forgeable value in the request, so it is
    looked up with the full scope — id AND org AND chatbot. A mismatch does not
    404: it silently starts a new thread. That is the friendlier behaviour for a
    widget holding a stale id, and it leaks nothing, because "not yours" and
    "does not exist" produce identical results.
    """
    if payload.conversation_id is not None:
        existing = db.scalar(
            select(Conversation).where(
                Conversation.id == payload.conversation_id,
                Conversation.org_id == scope.org_id,
                Conversation.chatbot_id == scope.chatbot_id,
            )
        )
        if existing is not None:
            return existing
        logger.info(
            "unknown conversation id; starting a new conversation",
            extra={**scope.log, "requested": str(payload.conversation_id)},
        )

    conversation = Conversation(
        org_id=scope.org_id,
        chatbot_id=scope.chatbot_id,
        channel=scope.channel,
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


def _find_replayable_answer(
    db: Session, conversation_id: uuid.UUID, client_message_id: str | None
) -> Message | None:
    """The stored answer for this idempotency key, if there is one.

    The key is recorded on the ASSISTANT row — the artifact being deduplicated
    is the answer, and the table's unique constraint on
    (conversation_id, client_message_id) is what makes the lookup exact.
    """
    if not client_message_id:
        return None

    return db.scalar(
        select(Message).where(
            Message.conversation_id == conversation_id,
            Message.client_message_id == client_message_id,
            Message.role == MessageRole.ASSISTANT,
        )
    )


def _load_history(db: Session, conversation_id: uuid.UUID) -> list[ProviderMessage]:
    """The recent turns of this conversation, oldest first.

    Fetched newest-first with a LIMIT and then reversed, rather than ordered
    ascending: a long-running conversation would otherwise read every message
    ever sent in it to keep the last six.
    """
    rows = list(
        db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(settings.chat_history_messages)
        )
    )
    return [
        ProviderMessage(role=str(row.role), content=row.content)
        for row in reversed(rows)
    ]


def _persist_turn(
    scope: ChatScope,
    conversation_id: uuid.UUID,
    payload: ChatRequest,
    result: TurnCompleted,
) -> uuid.UUID | None:
    """Write both messages and bump the conversation, in one transaction.

    Opens its own session: by the time this runs the request-scoped one is long
    closed (see `_chat`), and the write should hold a connection for
    milliseconds rather than for the length of a generation.

    Failure here is logged and swallowed. The user has already received their
    answer; turning a transcript-write failure into an error frame would tell
    them something went wrong with a message they can see on their screen. The
    metering counter is separate and already committed, so usage stays correct
    even when logging does not.
    """
    db = SessionLocal()
    try:
        now = datetime.now(UTC)
        question = Message(
            org_id=scope.org_id,
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=payload.question,
            tokens_in=result.tokens_in,
        )
        answer = Message(
            org_id=scope.org_id,
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=result.answer,
            citations=result.citation_dicts(),
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            latency_ms=result.latency_ms,
            client_message_id=payload.client_message_id,
        )
        db.add_all([question, answer])

        conversation = db.get(Conversation, conversation_id)
        if conversation is not None:
            conversation.last_message_at = now

        db.commit()
        return answer.id
    except IntegrityError:
        # Two retries of the same client_message_id raced and both generated.
        # The unique constraint caught the second one; the first answer stands.
        db.rollback()
        logger.info(
            "duplicate client_message_id; keeping the first answer",
            extra={**scope.log, "conversation_id": str(conversation_id)},
        )
        existing = _find_replayable_answer(
            db, conversation_id, payload.client_message_id
        )
        return existing.id if existing else None
    except Exception:  # noqa: BLE001 - never fail a delivered answer
        db.rollback()
        logger.exception("could not persist chat turn", extra=scope.log)
        return None
    finally:
        db.close()


def _refund(org_id: uuid.UUID) -> None:
    """Return the reserved message. Its own session, for the same reason as
    `_persist_turn`, and never allowed to raise — a refund failing must not
    replace the error the user actually needs to see."""
    db = SessionLocal()
    try:
        release_message(db, org_id)
    except Exception:  # noqa: BLE001
        logger.exception(
            "could not refund reserved message", extra={"org_id": str(org_id)}
        )
    finally:
        db.close()


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


def _vector_store(request: Request) -> VectorStore:
    """The process-wide Qdrant client, created once in the app lifespan.

    Per-request construction would build a fresh HTTP connection pool for every
    message and discard it, which is both wasteful and a reliable way to run out
    of sockets under load.
    """
    store = getattr(request.app.state, "vector_store", None)
    if store is None:  # pragma: no cover - lifespan always sets it
        raise HTTPException(status_code=503, detail="vector store is not available")
    return store


def _error_code(exc: Exception) -> str:
    """Map an exception to a stable machine-readable code for the client.

    Codes rather than class names: the client branches on these, and renaming a
    Python class should not break someone's widget.
    """
    if isinstance(exc, ProviderError):
        return "provider_unavailable" if exc.retryable else "provider_error"
    if isinstance(exc, VectorStoreError):
        return "search_unavailable"
    return "bad_request"

"""End-to-end tests for the Phase 3 chat endpoints.

These drive real HTTP through the real routers, the real bot-key authenticator
and the real metering code. Exactly one thing is faked — `run_turn` — and the
reason is worth stating, because faking too much is how an integration test
becomes a mirror:

`run_turn` is where the LLM, the embedder and the reranker live. Calling it for
real would download a 130MB model, need a Qdrant, and hit a rate-limited free
API from a unit test. Everything this file is actually about — SSE framing, who
is allowed to chat, what the counter says afterwards, what happens when
generation fails — sits either side of that call. `core/chat.py` is what tests
the pipeline itself.

The database is SQLite in memory (see `conftest.db_engine`), so all of this runs
without Docker, which matters because two of these tests guard security
boundaries and a skipped test guards nothing.

**PLAN §6 "done when" lives here:**

    curl streams an answer token-by-token   -> test_widget_chat_streams_tokens_*
    the 101st free message is refused       -> test_the_101st_free_message_*
    usage numbers match messages sent       -> test_usage_matches_messages_sent
"""

import json
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.api import chat as chat_api
from app.core.chat import Citation, TokenChunk, TurnCompleted
from app.core.metering import current_period
from app.db.base import get_db
from app.db.tables import (
    Channel,
    Chatbot,
    ChatbotStatus,
    Conversation,
    Message,
    MessageRole,
    Organization,
    PlanTier,
    UsageRecord,
    User,
    UserRole,
    generate_public_key,
)
from app.main import app
from app.providers.base import ProviderRateLimitError
from app.security.bot_key import BOT_KEY_HEADER
from app.security.clerk import Principal, require_org
from app.security.rate_limit import TokenBucketLimiter

# The rate limiter must not reach a real Redis: a developer running one locally
# would otherwise carry token-bucket state between test runs, and the suite
# would pass or fail depending on how recently it last ran. Port 1 is reserved
# and nothing listens there, so the limiter falls back to its in-process bucket.
UNREACHABLE_REDIS = "redis://127.0.0.1:1"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@dataclass
class Tenant:
    """One seeded org + bot, and the credentials to talk to it."""

    organization: Organization
    chatbot: Chatbot
    key: str

    @property
    def headers(self) -> dict[str, str]:
        return {BOT_KEY_HEADER: self.key}

    @property
    def principal(self) -> Principal:
        """What `require_org` would return for this org's owner."""
        return Principal(
            org_id=self.organization.id,
            clerk_org_id=f"org_{self.organization.name}",
            clerk_user_id=f"user_{self.organization.name}",
            role=UserRole.OWNER,
        )


def _seed_tenant(db: Session, name: str) -> Tenant:
    organization = Organization(
        name=name, clerk_org_id=f"org_{name}", plan=PlanTier.FREE
    )
    db.add(organization)
    db.flush()

    db.add(
        User(
            clerk_user_id=f"user_{name}",
            org_id=organization.id,
            role=UserRole.OWNER,
        )
    )
    chatbot = Chatbot(
        org_id=organization.id,
        name=f"{name}-bot",
        system_prompt="Be brief.",
        embedding_model="BAAI/bge-small-en-v1.5",
        dimension=384,
        status=ChatbotStatus.READY,
        public_key=generate_public_key(),
    )
    db.add(chatbot)
    db.commit()
    return Tenant(organization=organization, chatbot=chatbot, key=chatbot.public_key)


@pytest.fixture
def acme(db: Session) -> Tenant:
    return _seed_tenant(db, "acme")


@pytest.fixture
def globex(db: Session) -> Tenant:
    """A second tenant. Every cross-tenant test needs someone to be the victim."""
    return _seed_tenant(db, "globex")


@dataclass
class FakeTurn:
    """A stand-in for `run_turn`, recording how it was called.

    Records `calls` because several tests assert on what the endpoint passed
    IN — the history it loaded, the persona it read off the bot row — which is
    invisible from the response body.
    """

    tokens: tuple[str, ...] = ("Hello", ", ", "world")
    error: Exception | None = None
    refused: bool = False

    def __post_init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, question: str, **kwargs) -> AsyncIterator:
        self.calls.append({"question": question, **kwargs})
        return self._run()

    async def _run(self) -> AsyncIterator:
        if self.error is not None:
            raise self.error
        for token in self.tokens:
            yield TokenChunk(text=token)
        yield TurnCompleted(
            answer="".join(self.tokens),
            citations=(
                []
                if self.refused
                else [
                    Citation(
                        citation="[1] handbook.pdf p.3",
                        document_id=str(uuid.uuid4()),
                        source="handbook.pdf",
                        page_start=3,
                        page_end=3,
                        score=0.91,
                    )
                ]
            ),
            provider="gemini",
            chunks_retrieved=0 if self.refused else 1,
            tokens_in=42,
            tokens_out=7,
            latency_ms=123,
            ttft_ms=45,
            refused=self.refused,
        )


@pytest.fixture
def turn(monkeypatch: pytest.MonkeyPatch) -> FakeTurn:
    """Replace `run_turn` for the duration of one test.

    Patched on `app.api.chat`, not on `app.core.chat`: rebinding the name where
    it is USED is what a `from x import y` import requires. Patching the source
    module would leave the endpoint holding its original reference and the test
    would silently exercise the real pipeline.
    """
    fake = FakeTurn()
    monkeypatch.setattr(chat_api, "run_turn", fake)
    return fake


@pytest.fixture
def limiter(monkeypatch: pytest.MonkeyPatch) -> TokenBucketLimiter:
    """A permissive, process-local rate limiter, fresh per test.

    Fresh matters: `get_rate_limiter()` is `lru_cache`d for production, and a
    shared bucket would make a test fail because of how many requests the
    previous test sent.
    """
    instance = TokenBucketLimiter(
        per_minute=600, burst=600, redis_url=UNREACHABLE_REDIS
    )
    monkeypatch.setattr(chat_api, "get_rate_limiter", lambda: instance)
    return instance


@pytest.fixture
def chat_client(
    db_engine: Engine, monkeypatch: pytest.MonkeyPatch, limiter: TokenBucketLimiter
) -> Iterator[TestClient]:
    """A client wired to the in-memory database.

    Three redirections, each for a different reason:

    - `get_db` is a FastAPI dependency, so it is replaced through
      `dependency_overrides` — the supported hook, and the one that unwinds
      cleanly if a test fails mid-request.
    - `SessionLocal` is called directly by the streaming path (it deliberately
      does not hold the request session across an LLM call), so it is
      monkeypatched on the module.
    - `app.state.vector_store` is a sentinel. `run_turn` is faked, so the store
      is passed around and never touched; the endpoint only checks it exists.

    Constructed WITHOUT `with`, unlike the `client` fixture: that skips the
    lifespan, which would otherwise build a real Qdrant client and overwrite the
    sentinel.
    """
    factory = sessionmaker(bind=db_engine, autoflush=False, expire_on_commit=False)

    def override_get_db() -> Iterator[Session]:
        session = factory()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr(chat_api, "SessionLocal", factory)
    app.dependency_overrides[get_db] = override_get_db
    app.state.vector_store = object()
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def sse_events(response: Response) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event name, decoded payload) pairs.

    Written out rather than imported so the test asserts on the WIRE FORMAT.
    If `_sse` ever stopped emitting `event:`/`data:`/blank-line, sharing the
    encoder would hide the break; this parser would fail, which is the point.
    """
    events: list[tuple[str, dict]] = []
    for block in response.text.split("\n\n"):
        if not block.strip():
            continue
        name = ""
        data = ""
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        events.append((name, json.loads(data)))
    return events


def names(events: list[tuple[str, dict]]) -> list[str]:
    return [name for name, _ in events]


def payload_of(events: list[tuple[str, dict]], event: str) -> dict:
    return next(data for name, data in events if name == event)


def tokens_of(events: list[tuple[str, dict]]) -> str:
    return "".join(data["text"] for name, data in events if name == "token")


def messages_used(db: Session, org_id: uuid.UUID) -> int:
    record = db.scalar(
        select(UsageRecord).where(
            UsageRecord.org_id == org_id, UsageRecord.period == current_period()
        )
    )
    return record.messages_used if record else 0


def message_count(db: Session, org_id: uuid.UUID, role: MessageRole) -> int:
    return (
        db.scalar(
            select(func.count(Message.id)).where(
                Message.org_id == org_id, Message.role == role
            )
        )
        or 0
    )


def ask(
    client: TestClient,
    tenant: Tenant,
    question: str = "what is the refund policy?",
    **body,
) -> Response:
    return client.post(
        "/widget/chat", json={"question": question, **body}, headers=tenant.headers
    )


# --------------------------------------------------------------------------
# Streaming — PLAN §6: "curl streams an answer token-by-token"
# --------------------------------------------------------------------------


def test_widget_chat_streams_tokens_then_done(
    chat_client: TestClient, acme: Tenant, turn: FakeTurn
) -> None:
    response = ask(chat_client, acme)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = sse_events(response)
    assert names(events) == ["meta", "token", "token", "token", "done"]
    assert tokens_of(events) == "Hello, world"


def test_answer_arrives_as_separate_frames_not_one_blob(
    chat_client: TestClient, acme: Tenant, turn: FakeTurn
) -> None:
    """The token-BY-token part of the criterion.

    A handler that collected the answer and sent it in one frame would pass the
    test above. Counting frames is what distinguishes streaming from a slow
    request that happens to use the SSE content type.
    """
    turn.tokens = ("a", "b", "c", "d", "e")

    events = sse_events(ask(chat_client, acme))

    assert names(events).count("token") == 5


def test_meta_frame_leads_and_carries_the_conversation_id(
    chat_client: TestClient, acme: Tenant, turn: FakeTurn
) -> None:
    """`meta` is sent before generation starts, so a client whose connection
    drops mid-answer already knows which thread to resume."""
    events = sse_events(ask(chat_client, acme))

    name, meta = events[0]
    assert name == "meta"
    assert uuid.UUID(meta["conversation_id"])
    assert meta["chatbot_id"] == str(acme.chatbot.id)
    assert meta["usage"]["messages_used"] == 1


def test_stream_headers_defeat_caches_and_proxy_buffering(
    chat_client: TestClient, acme: Tenant, turn: FakeTurn
) -> None:
    response = ask(chat_client, acme)

    assert response.headers["cache-control"] == "no-cache"
    # Without this, nginx holds the whole answer until generation finishes.
    assert response.headers["x-accel-buffering"] == "no"


def test_done_frame_reports_citations_and_timings(
    chat_client: TestClient, acme: Tenant, turn: FakeTurn
) -> None:
    done = payload_of(sse_events(ask(chat_client, acme)), "done")

    assert done["provider"] == "gemini"
    assert done["latency_ms"] == 123
    assert done["ttft_ms"] == 45
    assert done["refused"] is False
    assert done["citations"][0]["source"] == "handbook.pdf"
    assert uuid.UUID(done["message_id"])


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def test_both_turns_are_logged_with_observability_columns(
    chat_client: TestClient, acme: Tenant, turn: FakeTurn, db: Session
) -> None:
    ask(chat_client, acme, "how do refunds work?")

    rows = list(db.scalars(select(Message).order_by(Message.role)))
    assert len(rows) == 2

    assistant = next(row for row in rows if row.role == MessageRole.ASSISTANT)
    user = next(row for row in rows if row.role == MessageRole.USER)

    assert user.content == "how do refunds work?"
    assert user.citations is None
    assert assistant.content == "Hello, world"
    assert assistant.citations[0]["citation"] == "[1] handbook.pdf p.3"
    assert (assistant.tokens_in, assistant.tokens_out) == (42, 7)
    assert assistant.latency_ms == 123


def test_conversation_records_the_channel_and_stays_scoped(
    chat_client: TestClient, acme: Tenant, turn: FakeTurn, db: Session
) -> None:
    ask(chat_client, acme)

    conversation = db.scalar(select(Conversation))
    assert conversation is not None
    assert conversation.org_id == acme.organization.id
    assert conversation.chatbot_id == acme.chatbot.id
    assert conversation.channel == Channel.WIDGET


def test_follow_up_continues_the_same_conversation(
    chat_client: TestClient, acme: Tenant, turn: FakeTurn, db: Session
) -> None:
    first = payload_of(
        sse_events(ask(chat_client, acme, "what is your refund policy?")), "meta"
    )

    second = payload_of(
        sse_events(
            ask(
                chat_client,
                acme,
                "and for gifts?",
                conversation_id=first["conversation_id"],
            )
        ),
        "meta",
    )

    assert second["conversation_id"] == first["conversation_id"]
    assert db.scalar(select(func.count(Conversation.id))) == 1
    assert db.scalar(select(func.count(Message.id))) == 4


def test_history_of_the_thread_is_replayed_to_the_model(
    chat_client: TestClient, acme: Tenant, turn: FakeTurn
) -> None:
    """The multi-turn guarantee, checked at the boundary it actually crosses."""
    meta = payload_of(
        sse_events(ask(chat_client, acme, "what is your refund policy?")), "meta"
    )
    ask(chat_client, acme, "and for gifts?", conversation_id=meta["conversation_id"])

    history = turn.calls[1]["history"]
    assert [(m.role, m.content) for m in history] == [
        ("user", "what is your refund policy?"),
        ("assistant", "Hello, world"),
    ]


def test_the_bots_system_prompt_is_passed_as_the_persona(
    chat_client: TestClient, acme: Tenant, turn: FakeTurn
) -> None:
    ask(chat_client, acme)

    assert turn.calls[0]["persona"] == "Be brief."


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------


def test_a_request_without_a_bot_key_is_rejected(
    chat_client: TestClient, acme: Tenant, turn: FakeTurn, db: Session
) -> None:
    response = chat_client.post("/widget/chat", json={"question": "hello?"})

    assert response.status_code == 401
    assert turn.calls == []
    assert db.scalar(select(func.count(Conversation.id))) == 0


def test_an_unknown_bot_key_is_rejected(
    chat_client: TestClient, acme: Tenant, turn: FakeTurn
) -> None:
    response = chat_client.post(
        "/widget/chat",
        json={"question": "hello?"},
        headers={BOT_KEY_HEADER: generate_public_key()},
    )

    assert response.status_code == 401
    assert turn.calls == []


def test_a_disabled_bot_refuses_to_chat(
    chat_client: TestClient, acme: Tenant, turn: FakeTurn, db: Session
) -> None:
    """Disabling is how a customer turns their widget off without destroying
    the vectors — so it has to actually stop traffic."""
    acme.chatbot.status = ChatbotStatus.DISABLED
    db.commit()

    response = ask(chat_client, acme)

    assert response.status_code == 403
    assert turn.calls == []


def test_the_body_cannot_choose_a_chatbot(
    chat_client: TestClient, acme: Tenant, globex: Tenant, turn: FakeTurn
) -> None:
    """`extra="forbid"` in action. If this ever became a 200, one public key
    would be able to chat with every bot in the system."""
    response = chat_client.post(
        "/widget/chat",
        json={"question": "hello?", "chatbot_id": str(globex.chatbot.id)},
        headers=acme.headers,
    )

    assert response.status_code == 422
    assert turn.calls == []


def test_a_forged_conversation_id_starts_a_new_thread(
    chat_client: TestClient, acme: Tenant, globex: Tenant, turn: FakeTurn, db: Session
) -> None:
    """The one forgeable field in the request.

    Another tenant's conversation id must not join that conversation — and must
    not 404 either, which would confirm the id exists.
    """
    victim = Conversation(
        org_id=globex.organization.id,
        chatbot_id=globex.chatbot.id,
        channel=Channel.WIDGET,
    )
    db.add(victim)
    db.commit()

    meta = payload_of(
        sse_events(ask(chat_client, acme, "hello?", conversation_id=str(victim.id))),
        "meta",
    )

    assert meta["conversation_id"] != str(victim.id)
    assert (
        db.scalar(
            select(func.count(Message.id)).where(Message.conversation_id == victim.id)
        )
        == 0
    )
    assert message_count(db, globex.organization.id, MessageRole.USER) == 0


# --------------------------------------------------------------------------
# Plan limits — PLAN §6: "the 101st free-tier message is politely refused"
# --------------------------------------------------------------------------


def test_the_101st_free_message_is_politely_refused(
    chat_client: TestClient, acme: Tenant, turn: FakeTurn, db: Session
) -> None:
    """Seeded to 100 rather than sending 100 requests: the counter is what the
    cap is enforced against, and `test_limits.py` already proves that one
    message moves it by exactly one."""
    db.add(
        UsageRecord(
            org_id=acme.organization.id, period=current_period(), messages_used=100
        )
    )
    db.commit()

    response = ask(chat_client, acme)

    assert response.status_code == 429
    detail = response.json()["detail"]
    assert detail["error"] == "plan_limit_reached"
    # "Politely" means a payload the widget can render as a sentence.
    assert detail["plan"] == "free"
    assert detail["used"] == 100
    assert detail["limit"] == 100
    assert "resets_at" in detail
    assert turn.calls == []


def test_a_refused_message_does_not_consume_quota(
    chat_client: TestClient, acme: Tenant, turn: FakeTurn, db: Session
) -> None:
    db.add(
        UsageRecord(
            org_id=acme.organization.id, period=current_period(), messages_used=100
        )
    )
    db.commit()

    ask(chat_client, acme)

    db.expire_all()
    assert messages_used(db, acme.organization.id) == 100
    assert message_count(db, acme.organization.id, MessageRole.USER) == 0


def test_the_100th_message_still_goes_through(
    chat_client: TestClient, acme: Tenant, turn: FakeTurn, db: Session
) -> None:
    """The off-by-one that matters commercially: a cap of 100 must allow 100."""
    db.add(
        UsageRecord(
            org_id=acme.organization.id, period=current_period(), messages_used=99
        )
    )
    db.commit()

    assert ask(chat_client, acme).status_code == 200

    db.expire_all()
    assert messages_used(db, acme.organization.id) == 100


def test_a_paid_plan_is_not_capped_at_the_free_limit(
    chat_client: TestClient, acme: Tenant, turn: FakeTurn, db: Session
) -> None:
    acme.organization.plan = PlanTier.PRO
    db.add(
        UsageRecord(
            org_id=acme.organization.id, period=current_period(), messages_used=100
        )
    )
    db.commit()

    assert ask(chat_client, acme).status_code == 200


# --------------------------------------------------------------------------
# Metering — PLAN §6: "usage numbers in DB match messages sent"
# --------------------------------------------------------------------------


def test_usage_matches_messages_sent(
    chat_client: TestClient, acme: Tenant, turn: FakeTurn, db: Session
) -> None:
    for index in range(5):
        assert ask(chat_client, acme, f"question {index}").status_code == 200

    db.expire_all()
    assert messages_used(db, acme.organization.id) == 5
    # Reconciled against the transcript, not just against the request count:
    # the counter and the log are written by different code paths, and this is
    # the only test that would notice them drifting apart.
    assert message_count(db, acme.organization.id, MessageRole.USER) == 5
    assert message_count(db, acme.organization.id, MessageRole.ASSISTANT) == 5


def test_usage_is_counted_per_organization(
    chat_client: TestClient, acme: Tenant, globex: Tenant, turn: FakeTurn, db: Session
) -> None:
    ask(chat_client, acme)
    ask(chat_client, globex)
    ask(chat_client, globex)

    db.expire_all()
    assert messages_used(db, acme.organization.id) == 1
    assert messages_used(db, globex.organization.id) == 2


# --------------------------------------------------------------------------
# Idempotency
# --------------------------------------------------------------------------


def test_a_retry_with_the_same_client_id_replays_the_stored_answer(
    chat_client: TestClient, acme: Tenant, turn: FakeTurn, db: Session
) -> None:
    """A widget whose connection dropped mid-answer retries. It must get the
    same answer back, not a second generation on the customer's quota."""
    first = sse_events(ask(chat_client, acme, "hi?", client_message_id="msg-1"))
    conversation_id = payload_of(first, "meta")["conversation_id"]

    second = sse_events(
        ask(
            chat_client,
            acme,
            "hi?",
            client_message_id="msg-1",
            conversation_id=conversation_id,
        )
    )

    assert payload_of(second, "meta")["replayed"] is True
    assert tokens_of(second) == tokens_of(first) == "Hello, world"
    assert len(turn.calls) == 1  # the model was called once, not twice


def test_a_replayed_retry_is_not_billed_or_logged_twice(
    chat_client: TestClient, acme: Tenant, turn: FakeTurn, db: Session
) -> None:
    meta = payload_of(
        sse_events(ask(chat_client, acme, "hi?", client_message_id="msg-1")), "meta"
    )
    ask(
        chat_client,
        acme,
        "hi?",
        client_message_id="msg-1",
        conversation_id=meta["conversation_id"],
    )

    db.expire_all()
    assert messages_used(db, acme.organization.id) == 1
    assert db.scalar(select(func.count(Message.id))) == 2


def test_a_different_client_id_is_a_different_message(
    chat_client: TestClient, acme: Tenant, turn: FakeTurn, db: Session
) -> None:
    """The mirror image: idempotency must not deduplicate genuinely new
    questions that happen to share a conversation."""
    meta = payload_of(
        sse_events(ask(chat_client, acme, "hi?", client_message_id="msg-1")), "meta"
    )
    ask(
        chat_client,
        acme,
        "and again?",
        client_message_id="msg-2",
        conversation_id=meta["conversation_id"],
    )

    db.expire_all()
    assert len(turn.calls) == 2
    assert messages_used(db, acme.organization.id) == 2


def test_the_same_client_id_in_another_conversation_is_not_a_replay(
    chat_client: TestClient, acme: Tenant, turn: FakeTurn
) -> None:
    """Idempotency keys come from browsers and are only unique where they are
    compared — which is inside one conversation."""
    ask(chat_client, acme, "hi?", client_message_id="msg-1")
    second = sse_events(ask(chat_client, acme, "hi?", client_message_id="msg-1"))

    assert "replayed" not in payload_of(second, "meta")
    assert len(turn.calls) == 2


# --------------------------------------------------------------------------
# Failure handling
# --------------------------------------------------------------------------


def test_a_provider_failure_becomes_an_error_frame(
    chat_client: TestClient, acme: Tenant, turn: FakeTurn
) -> None:
    """Once the 200 has been sent the status code is spent, so a mid-stream
    failure can only be reported inside the stream."""
    turn.error = ProviderRateLimitError("quota exhausted", provider="gemini")

    response = ask(chat_client, acme)
    events = sse_events(response)

    assert response.status_code == 200
    assert names(events) == ["meta", "error"]
    assert payload_of(events, "error")["error"] == "provider_unavailable"


def test_a_failed_answer_is_refunded(
    chat_client: TestClient, acme: Tenant, turn: FakeTurn, db: Session
) -> None:
    """Without the refund, a provider outage silently spends a customer's
    monthly quota on answers they never received."""
    turn.error = ProviderRateLimitError("quota exhausted", provider="gemini")

    ask(chat_client, acme)

    db.expire_all()
    assert messages_used(db, acme.organization.id) == 0
    assert db.scalar(select(func.count(Message.id))) == 0


def test_a_failure_after_a_successful_turn_leaves_the_earlier_usage_alone(
    chat_client: TestClient, acme: Tenant, turn: FakeTurn, db: Session
) -> None:
    """The refund must return exactly one message, not reset the counter."""
    ask(chat_client, acme)
    turn.error = ProviderRateLimitError("quota exhausted", provider="gemini")
    ask(chat_client, acme)

    db.expire_all()
    assert messages_used(db, acme.organization.id) == 1


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------


def test_a_flood_from_one_widget_is_throttled(
    chat_client: TestClient,
    acme: Tenant,
    turn: FakeTurn,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strict = TokenBucketLimiter(per_minute=60, burst=1, redis_url=UNREACHABLE_REDIS)
    monkeypatch.setattr(chat_api, "get_rate_limiter", lambda: strict)

    assert ask(chat_client, acme).status_code == 200
    response = ask(chat_client, acme)

    assert response.status_code == 429
    assert response.json()["detail"]["error"] == "rate_limited"
    assert int(response.headers["retry-after"]) >= 0


def test_throttling_happens_before_metering(
    chat_client: TestClient,
    acme: Tenant,
    turn: FakeTurn,
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Order-of-operations, and it is a billing bug if it changes: a request
    refused for flooding was never answered, so it must not be charged."""
    strict = TokenBucketLimiter(per_minute=60, burst=1, redis_url=UNREACHABLE_REDIS)
    monkeypatch.setattr(chat_api, "get_rate_limiter", lambda: strict)

    ask(chat_client, acme)
    ask(chat_client, acme)

    db.expire_all()
    assert messages_used(db, acme.organization.id) == 1
    assert len(turn.calls) == 1


def test_one_tenants_flood_does_not_throttle_another(
    chat_client: TestClient,
    acme: Tenant,
    globex: Tenant,
    turn: FakeTurn,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strict = TokenBucketLimiter(per_minute=60, burst=1, redis_url=UNREACHABLE_REDIS)
    monkeypatch.setattr(chat_api, "get_rate_limiter", lambda: strict)

    assert ask(chat_client, acme).status_code == 200
    assert ask(chat_client, acme).status_code == 429
    # Different key, different bucket.
    assert ask(chat_client, globex).status_code == 200


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_an_empty_question_is_rejected(
    chat_client: TestClient, acme: Tenant, turn: FakeTurn
) -> None:
    assert ask(chat_client, acme, "").status_code == 422
    assert turn.calls == []


def test_an_oversized_question_is_rejected(
    chat_client: TestClient, acme: Tenant, turn: FakeTurn
) -> None:
    """Rejected at the schema, not after embedding it: bge truncates silently
    past its window, so an unbounded question is a cost and a correctness bug."""
    assert ask(chat_client, acme, "x" * 4_001).status_code == 422
    assert turn.calls == []


# --------------------------------------------------------------------------
# The playground route
# --------------------------------------------------------------------------


@pytest.fixture
def playground_client(chat_client: TestClient, acme: Tenant) -> Iterator[TestClient]:
    """`chat_client`, with the Clerk dependency answered as acme's owner.

    Overriding `require_org` rather than minting a JWT: token verification has
    its own tests, and reproducing a signed Clerk token here would test the
    fixture more than the endpoint.
    """
    app.dependency_overrides[require_org] = lambda: acme.principal
    try:
        yield chat_client
    finally:
        app.dependency_overrides.pop(require_org, None)


def test_the_playground_streams_and_meters_like_the_widget(
    playground_client: TestClient, acme: Tenant, turn: FakeTurn, db: Session
) -> None:
    """A message the owner sends while testing costs the same quota as a
    visitor's, because it costs us the same."""
    response = playground_client.post(
        f"/api/bots/{acme.chatbot.id}/chat", json={"question": "does this work?"}
    )

    assert response.status_code == 200
    assert tokens_of(sse_events(response)) == "Hello, world"

    db.expire_all()
    assert messages_used(db, acme.organization.id) == 1


def test_the_playground_marks_its_conversations(
    playground_client: TestClient, acme: Tenant, turn: FakeTurn, db: Session
) -> None:
    """So Phase 6 can answer "how much of this plan was spent on testing?"."""
    playground_client.post(
        f"/api/bots/{acme.chatbot.id}/chat", json={"question": "does this work?"}
    )

    conversation = db.scalar(select(Conversation))
    assert conversation is not None
    assert conversation.channel == Channel.PLAYGROUND


def test_the_playground_cannot_chat_with_another_orgs_bot(
    playground_client: TestClient, globex: Tenant, turn: FakeTurn
) -> None:
    """404 rather than 403 — see `api/chatbots.py` on why confirming existence
    is itself a leak."""
    response = playground_client.post(
        f"/api/bots/{globex.chatbot.id}/chat", json={"question": "hello?"}
    )

    assert response.status_code == 404
    assert turn.calls == []

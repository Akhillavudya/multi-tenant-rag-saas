"""ORM tables: organizations, users, chatbots, documents (Phase 2) and
conversations, messages, usage_records (Phase 3).

PLAN §4's data model, restricted to what has a writer. `subscriptions` is still
deliberately absent — it belongs to Phase 6 (billing), and a table nothing
writes to is a schema guess that hardens into a migration before anyone learns
whether it was right.

**The organizing principle: `org_id` is on every tenant-scoped row.**

`documents.org_id` is formally redundant — a document belongs to a chatbot, and
the chatbot knows its org — but it is stored anyway, and that redundancy is the
point. It means every tenant-scoped query filters on a column that is physically
present on the row, with no join to forget. The alternative makes isolation
depend on remembering to join through `chatbots`, and a security boundary that
depends on remembering is the exact failure mode Hard Rule #3 exists to prevent.
It also mirrors the Qdrant payload (`org_id` + `chatbot_id` on every point), so
the SQL and vector sides of a tenant scope look identical rather than merely
equivalent.

The cost is the usual denormalization cost: two places can disagree. That is
bounded here because a document's owning chatbot never changes — there is no
"move document to another bot" operation, and if one is ever added it must move
both columns. A CHECK constraint cannot express this (it spans rows), so it is
enforced by there being exactly one INSERT path, in `api/documents.py`.

**Deviation from PLAN §4, deliberate.** PLAN names the chatbot's override column
`model_config`. Pydantic v2 reserves that attribute name for its own class
configuration, and the API schemas mirror these field names — a Pydantic model
with a `model_config` *field* raises at class-definition time. Renamed to
`model_overrides`; identical purpose and JSON shape.

**Why UUID primary keys rather than autoincrementing integers.** Three reasons,
in order of weight: (1) sequential ids are guessable, and a tenant id an
attacker can enumerate turns every missing authorization check into a walk
through the customer list; (2) ids appear in Qdrant payloads and in URLs, so
they must be generatable without a database round trip; (3) they are unique
across environments, so a row exported from staging cannot collide in
production. The cost is 16 bytes instead of 4 and a slightly less cache-friendly
index — irrelevant at this scale.
"""

import enum
import secrets
import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)

# Aliased: the stdlib `enum` module is imported above, and SQLAlchemy's Enum is
# a column type, not a Python enum. Same word, unrelated things.
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

# Prefix on the widget key. Not decoration: a leaked credential is far easier to
# recognise and revoke when its first three characters say what it is, and
# secret-scanners key off exactly this kind of prefix.
PUBLIC_KEY_PREFIX = "pk_"

# 32 bytes -> 43 url-safe characters. Well past the point where brute-forcing a
# key is more expensive than attacking anything else in the system.
PUBLIC_KEY_BYTES = 32

# JSONB is a PostgreSQL type and does not compile on any other dialect.
# `with_variant` hands SQLite the generic JSON type instead, which is what lets
# `tests/test_limits.py` build this entire schema in memory with no Docker —
# the same trade Phase 2 made when `VectorStore` gained `location=":memory:"`
# for the isolation test. On Postgres this is byte-for-byte JSONB, so the
# emitted production DDL (and migration 0001) is unchanged.
JSON_COLUMN = JSONB().with_variant(JSON(), "sqlite")


def generate_public_key() -> str:
    """Create a widget key.

    `secrets`, never `random`: the latter is a Mersenne Twister seeded from the
    clock, and observing a few outputs is enough to predict the rest. This value
    is a bearer credential — anyone holding it can talk to the bot — so it needs
    a CSPRNG. This is the single most security-sensitive line in the file.
    """
    return f"{PUBLIC_KEY_PREFIX}{secrets.token_urlsafe(PUBLIC_KEY_BYTES)}"


# --------------------------------------------------------------------------
# Enumerated values
# --------------------------------------------------------------------------
#
# StrEnum members compare equal to their string value, so these read naturally
# in queries and serialize to plain JSON without a custom encoder. They map to
# real PostgreSQL enum types, which means the database rejects a typo'd status
# rather than storing it — a `documents` row with status "reddy" would never be
# picked up by any query and the document would appear to hang forever.


class PlanTier(enum.StrEnum):
    """Subscription tier. Limits live in `PLAN_LIMITS`, not in the enum."""

    FREE = "free"
    PRO = "pro"
    BUSINESS = "business"


class UserRole(enum.StrEnum):
    """Role within one organization.

    Two roles, not a permission system. Phase 2 needs only "can this person
    change billing and delete bots" (owner) versus "can they use them" (member).
    A richer model invented now would be a guess.
    """

    OWNER = "owner"
    MEMBER = "member"


class ChatbotStatus(enum.StrEnum):
    """Whether a bot is usable.

    DRAFT means created but with no ingested document yet — it would answer
    every question with the refusal string, so the dashboard shows it as
    unfinished rather than broken.
    """

    DRAFT = "draft"
    READY = "ready"
    DISABLED = "disabled"


class DocumentStatus(enum.StrEnum):
    """Ingestion lifecycle: PENDING -> PROCESSING -> READY | FAILED.

    PENDING and PROCESSING are distinct on purpose. PENDING means the row exists
    and Celery has been told about it; PROCESSING means a worker actually picked
    it up. Collapsing them hides the most common production symptom — a queue
    with no worker draining it — behind a status that looks like normal work in
    progress.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class SourceType(enum.StrEnum):
    """Where a document came from. URL ingestion is a PLAN stretch goal; the
    enum admits it now so adding it later is not a migration on a hot table."""

    FILE = "file"
    URL = "url"


class Channel(enum.StrEnum):
    """Where a conversation came from.

    Worth storing because the two have different economics and different
    trust: PLAYGROUND is the owner testing their own bot from the dashboard,
    WIDGET is an anonymous member of the public on someone else's website.
    Phase 6 will want to know whether a plan's message quota was spent on
    customers or on testing, and that question is unanswerable retroactively.
    """

    WIDGET = "widget"
    PLAYGROUND = "playground"


class MessageRole(enum.StrEnum):
    """Who produced a message.

    Only two values, matching what is actually persisted. The system prompt is
    not stored per message: it is rebuilt from the chatbot row and the retrieved
    context on every request, so a copy here would be a stale duplicate of two
    things that already exist.
    """

    USER = "user"
    ASSISTANT = "assistant"


# Plan limits, seeded from PLAN §4. In code rather than in a `plans` table
# because these are product decisions that change with a deploy, not data an
# operator edits at runtime — and a table would need a migration to change a
# number that lives in version control perfectly well here. Phase 6 reads this
# when Stripe flips a plan; Phase 3 reads it to enforce caps.
PLAN_LIMITS: dict[PlanTier, dict[str, int]] = {
    PlanTier.FREE: {"chatbots": 1, "messages_per_month": 100, "documents": 5},
    PlanTier.PRO: {"chatbots": 5, "messages_per_month": 2_000, "documents": 100},
    PlanTier.BUSINESS: {
        "chatbots": 20,
        "messages_per_month": 10_000,
        "documents": 1_000,
    },
}


# --------------------------------------------------------------------------
# Shared column helpers
# --------------------------------------------------------------------------


def _pk() -> Mapped[uuid.UUID]:
    """Primary key column. `default=uuid4` is a PYTHON-side default, so the id
    exists on the object before flush — which matters because ingestion needs
    the document id to build Qdrant point ids, and waiting for a round trip to
    learn it would serialize work that has no reason to be serial."""
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _created_at() -> Mapped[datetime]:
    """`server_default=now()` rather than a Python default: the database clock
    is the one authority every writer shares. A Python default records the
    application server's clock, and two servers with drifting clocks produce a
    created_at ordering that disagrees with reality."""
    return mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


class Organization(Base):
    """A tenant. The root of every ownership chain in the system.

    `clerk_org_id` is the join to the identity provider and is what the JWT
    middleware looks up on every authenticated request. It is unique and
    indexed for that reason — it is read far more often than the primary key.

    Clerk owns identity; we own everything else. Deliberately no email, no
    member list, no name-of-owner here: duplicating profile data means keeping
    it in sync with Clerk forever, and the stale copy always wins an argument it
    should lose.
    """

    __tablename__ = "organizations"

    id: Mapped[uuid.UUID] = _pk()
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    clerk_org_id: Mapped[str] = mapped_column(
        String(120), nullable=False, unique=True, index=True
    )
    plan: Mapped[PlanTier] = mapped_column(
        # Explicit type name so Alembic emits a stable CREATE TYPE; an
        # auto-named type is painful to alter later.
        SAEnum(PlanTier, name="plan_tier"),
        nullable=False,
        default=PlanTier.FREE,
        server_default=PlanTier.FREE.value,
    )
    created_at: Mapped[datetime] = _created_at()

    # cascade="all, delete-orphan" is ORM-level; the ondelete="CASCADE" on the
    # child FKs is database-level. Both are declared because they fire in
    # different situations: the ORM one when you delete a loaded object, the DB
    # one when a bulk DELETE or a manual statement bypasses the session.
    users: Mapped[list["User"]] = relationship(
        back_populates="organization", cascade="all, delete-orphan"
    )
    chatbots: Mapped[list["Chatbot"]] = relationship(
        back_populates="organization", cascade="all, delete-orphan"
    )

    @property
    def limits(self) -> dict[str, int]:
        """This org's plan caps. Phase 3 enforces them; here so there is one
        place that maps a plan to its numbers."""
        return PLAN_LIMITS[self.plan]


class User(Base):
    """A person, scoped to exactly one organization.

    One org per user is a Phase 2 simplification, and an honest one: PLAN lists
    "team members per org" as a stretch goal, and modelling a many-to-many
    membership now would add a join table that every query pays for and no
    feature uses. Widening it later is a migration; narrowing a premature
    many-to-many is a rewrite.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _pk()
    clerk_user_id: Mapped[str] = mapped_column(
        String(120), nullable=False, unique=True, index=True
    )
    org_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[UserRole] = mapped_column(
        SAEnum(UserRole, name="user_role"),
        nullable=False,
        default=UserRole.MEMBER,
        server_default=UserRole.MEMBER.value,
    )
    created_at: Mapped[datetime] = _created_at()

    organization: Mapped[Organization] = relationship(back_populates="users")


class Chatbot(Base):
    """One assistant, owned by one org. The unit a widget key points at.

    `embedding_model` and `dimension` are stored PER BOT rather than read from
    config at query time, and that is PLAN §3 rule 1 made concrete. The vectors
    in Qdrant were produced by whatever embedder ran at ingestion; if
    EMBEDDING_PROVIDER later changes, this bot's existing vectors do not.
    Recording the model that actually produced them is what lets `reembed.py`
    detect the mismatch and fix it deliberately, instead of the bot silently
    returning nonsense because 384-dim queries are being compared against
    1536-dim history.
    """

    __tablename__ = "chatbots"
    __table_args__ = (
        # Two bots in the same org may not share a name — the dashboard lists
        # bots by name, and two identically named bots is a support ticket.
        # Scoped to the org: different customers naturally both have a "Support"
        # bot, and a global unique constraint would be a cross-tenant collision
        # where one customer's naming choice blocks another's.
        UniqueConstraint("org_id", "name", name="uq_chatbot_org_name"),
        CheckConstraint("dimension > 0", name="ck_chatbot_dimension_positive"),
    )

    id: Mapped[uuid.UUID] = _pk()
    org_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)

    # Text, not String(n): a system prompt is prose with no natural bound, and
    # picking one guarantees the first customer who exceeds it hits a 500.
    # Postgres stores both identically anyway — a length limit on TEXT buys
    # nothing but a failure mode.
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)

    # JSONB rather than JSON: it is stored parsed, so it can be indexed and
    # queried. Per-bot provider overrides (PLAN §3 rule 4) live here — a
    # premium client running Claude while everyone else runs Gemini, without a
    # migration per knob. Renamed from PLAN's `model_config`; see module docs.
    model_overrides: Mapped[dict] = mapped_column(
        JSON_COLUMN, nullable=False, default=dict, server_default="{}"
    )

    embedding_model: Mapped[str] = mapped_column(String(120), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)

    status: Mapped[ChatbotStatus] = mapped_column(
        SAEnum(ChatbotStatus, name="chatbot_status"),
        nullable=False,
        default=ChatbotStatus.DRAFT,
        server_default=ChatbotStatus.DRAFT.value,
    )

    # The widget credential. Unique and indexed because Phase 5's traffic
    # arrives with nothing else: the key is looked up on every widget message
    # to resolve org and bot, making this the hottest read in the system.
    public_key: Mapped[str] = mapped_column(
        String(80), nullable=False, unique=True, index=True, default=generate_public_key
    )

    created_at: Mapped[datetime] = _created_at()

    organization: Mapped[Organization] = relationship(back_populates="chatbots")
    documents: Mapped[list["Document"]] = relationship(
        back_populates="chatbot", cascade="all, delete-orphan"
    )
    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="chatbot", cascade="all, delete-orphan"
    )


class Document(Base):
    """One uploaded file and the state of its ingestion.

    Carries `org_id` as well as `chatbot_id` — see the module docstring for why
    that redundancy is deliberate.

    This row is the ONLY thing the dashboard can show while a Celery job runs,
    so its status fields are a user-facing feature rather than bookkeeping. In
    particular `error` holds the message a human reads when ingestion fails;
    "failed" with no reason is indistinguishable from a bug in our product, and
    the most common cause is a scanned PDF the user can fix themselves by
    uploading a better copy.
    """

    __tablename__ = "documents"
    __table_args__ = (
        # The dashboard's sources tab query: this bot's documents, newest first.
        Index("ix_documents_scope", "org_id", "chatbot_id", "created_at"),
        # Re-uploading the same file to the same bot should update that
        # document, not create a second one — chunk point ids are derived from
        # document_id, so a duplicate row would produce a second full copy of
        # every chunk and let one passage occupy several top-40 slots.
        UniqueConstraint("chatbot_id", "content_hash", name="uq_document_bot_content"),
        CheckConstraint(
            # A document is only allowed to carry an error message when it has
            # actually failed. Without this, a retry that succeeds but forgets
            # to clear `error` leaves a ready document displaying a stale
            # failure in the dashboard forever.
            "(status = 'failed') OR (error IS NULL)",
            name="ck_document_error_only_when_failed",
        ),
    )

    id: Mapped[uuid.UUID] = _pk()
    org_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chatbot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chatbots.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    source_type: Mapped[SourceType] = mapped_column(
        SAEnum(SourceType, name="source_type"),
        nullable=False,
        default=SourceType.FILE,
        server_default=SourceType.FILE.value,
    )
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    # SHA-256 of the file bytes, from `ingestion.content_document_id`. Makes
    # re-upload idempotent (see the unique constraint above) and lets the UI say
    # "you already uploaded this" instead of silently doing the work twice.
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    status: Mapped[DocumentStatus] = mapped_column(
        SAEnum(DocumentStatus, name="document_status"),
        nullable=False,
        default=DocumentStatus.PENDING,
        server_default=DocumentStatus.PENDING.value,
        index=True,
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    pages: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )

    created_at: Mapped[datetime] = _created_at()
    # Bumped on every status transition, so the dashboard can show "processing
    # for 4 minutes" and an operator can spot a job that died mid-flight.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    chatbot: Mapped[Chatbot] = relationship(back_populates="documents")


# --------------------------------------------------------------------------
# Phase 3: chat transcripts and metering
# --------------------------------------------------------------------------


class Conversation(Base):
    """One continuous chat session with one bot.

    Created on the first message and reused for every follow-up, which is what
    makes multi-turn possible: the id is what the client sends back to say "same
    conversation". Anonymous by design — there is no user column, because widget
    traffic is the public on a customer's website and we neither have nor want
    their identity.

    Carries `org_id` for the same reason `documents` does: every tenant-scoped
    query then filters a column that is physically on the row, with no join to
    forget.
    """

    __tablename__ = "conversations"
    __table_args__ = (
        # The dashboard's conversations tab: this bot's threads, most recently
        # active first. `last_message_at` leads over `started_at` because a
        # thread that is still alive matters more than one that started earlier.
        Index("ix_conversations_scope", "org_id", "chatbot_id", "last_message_at"),
    )

    id: Mapped[uuid.UUID] = _pk()
    org_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chatbot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chatbots.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    channel: Mapped[Channel] = mapped_column(
        SAEnum(Channel, name="conversation_channel"),
        nullable=False,
        default=Channel.WIDGET,
        server_default=Channel.WIDGET.value,
    )

    started_at: Mapped[datetime] = _created_at()

    # Denormalized, and deliberately so. The conversations list sorts by "last
    # active", and deriving that from `max(messages.created_at)` costs either a
    # correlated subquery per row or a join+group-by over the largest table in
    # the schema — on every page load. One column written inside the same
    # transaction that writes the message keeps it exact.
    last_message_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    chatbot: Mapped[Chatbot] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        # Transcripts are read in the order they happened; without this the ORM
        # returns whatever order the database felt like.
        order_by="Message.created_at",
    )


class Message(Base):
    """One turn: a question from a user or a grounded answer from the bot.

    The assistant rows are where Phase 3's observability lives — `citations`,
    `tokens_*` and `latency_ms` are the raw material for PLAN §7's metrics
    section, and none of them can be reconstructed after the fact.

    **Token counts are estimates, and the column names do not lie about it: see
    `core/chat.py`.** `LLMProvider.stream()` yields text and nothing else, so
    there is no vendor usage number to record without changing a Phase 1
    contract that four providers already implement. Plan caps are counted in
    messages, not tokens, so nothing billable depends on these — they are for
    spotting a prompt that has quietly doubled in size.
    """

    __tablename__ = "messages"
    __table_args__ = (
        # Transcript read: one conversation, in order.
        Index("ix_messages_conversation", "conversation_id", "created_at"),
        # Usage reconciliation: "how many messages did this org send this
        # month" must not scan every tenant's traffic.
        Index("ix_messages_org_created", "org_id", "created_at"),
        # The idempotency guarantee. A widget whose connection dropped mid-answer
        # retries with the SAME client_message_id; this constraint is what makes
        # the retry return the stored answer instead of generating (and
        # metering) a second one. Scoped to the conversation rather than global
        # because the id comes from a browser and only has to be unique where it
        # is actually compared.
        UniqueConstraint(
            "conversation_id", "client_message_id", name="uq_message_client_id"
        ),
        CheckConstraint(
            "tokens_in >= 0 AND tokens_out >= 0 AND latency_ms >= 0",
            name="ck_message_counters_non_negative",
        ),
    )

    id: Mapped[uuid.UUID] = _pk()
    org_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[MessageRole] = mapped_column(
        SAEnum(MessageRole, name="message_role"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # NULL on user messages, a list of citation objects on assistant messages.
    # Stored as JSON rather than a join table because it is written once, read
    # whole, and never queried by its contents — a `citations` table would be
    # three joins to render one chat bubble.
    citations: Mapped[list | None] = mapped_column(JSON_COLUMN, nullable=True)

    tokens_in: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_out: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Wall-clock time from request received to stream finished. The number that
    # tells you whether the product feels fast; PLAN §7 reports p50/p95 of it.
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Client-supplied idempotency key. Nullable because the dashboard playground
    # has no reconnect logic and does not need one; NULLs do not collide in a
    # unique index, so many messages may omit it.
    client_message_id: Mapped[str | None] = mapped_column(String(80), nullable=True)

    created_at: Mapped[datetime] = _created_at()

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class UsageRecord(Base):
    """One organization's consumption in one calendar month.

    A counter, not a log. The messages table already holds the log; this row
    exists so that "are you over your plan?" is a single indexed lookup on the
    hot path of every chat request rather than a COUNT over a table that grows
    forever.

    The pair (org_id, period) is unique, and that constraint is load-bearing
    rather than tidy: `core/metering.py` relies on it to turn increment-or-create
    into one atomic statement. Without it, two simultaneous first-messages of the
    month create two rows and every subsequent count is wrong.
    """

    __tablename__ = "usage_records"
    __table_args__ = (
        UniqueConstraint("org_id", "period", name="uq_usage_org_period"),
        CheckConstraint(
            "messages_used >= 0 AND docs_ingested >= 0",
            name="ck_usage_counters_non_negative",
        ),
    )

    id: Mapped[uuid.UUID] = _pk()
    org_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # "YYYY-MM", UTC. A string rather than a date because it is an identifier
    # for a billing window, not a point in time — and because a DATE column
    # invites someone to store the 3rd of the month and break the uniqueness
    # that the whole design rests on. UTC rather than the org's local timezone:
    # a tenant that moves country would otherwise get two partial months.
    period: Mapped[str] = mapped_column(String(7), nullable=False, index=True)

    messages_used: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    docs_ingested: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )

    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

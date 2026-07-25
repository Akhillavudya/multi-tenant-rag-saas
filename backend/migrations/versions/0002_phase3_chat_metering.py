"""Phase 3: conversations, messages, usage records.

Revision ID: 0002_phase3_chat
Revises: 0001_phase2_tenancy
Create Date: 2026-07-25

Hand-written, for the same reason 0001 was: `--autogenerate` diffs against a
live database, and this was authored without one. It is a normal revision —
`alembic upgrade head` applies it identically. Confirm it agrees with the models
once Postgres is up:

    docker compose up -d postgres
    alembic upgrade head
    alembic check          # clean exit == this file and db/tables.py agree

**Additive only.** Three new tables, two new enum types, nothing altered or
dropped. That is what makes this deployable without a maintenance window: the
currently-running Phase 2 code neither reads nor writes any of it, so the
migration can be applied before the new code ships, which is the order you want
— a new pod that boots against an old schema crashes, an old pod that runs
against a new schema simply ignores the extra tables.

Enum note, repeated from 0001 because it is the trap that bites twice:
PostgreSQL enums are schema objects, not column constraints, so they outlive the
table that used them. `downgrade()` drops each type explicitly, or a
downgrade-then-upgrade fails with "type message_role already exists".
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002_phase3_chat"
down_revision = "0001_phase2_tenancy"
branch_labels = None
depends_on = None

_ENUM_TYPES = ("conversation_channel", "message_role")


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chatbot_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "channel",
            sa.Enum("widget", "playground", name="conversation_channel"),
            server_default="widget",
            nullable=False,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "last_message_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["chatbot_id"], ["chatbots.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["org_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_conversations_org_id", "conversations", ["org_id"])
    op.create_index("ix_conversations_chatbot_id", "conversations", ["chatbot_id"])
    # The dashboard's list query, in the order it reads: narrow by tenant, then
    # by bot, then sort. A composite index only helps when the leading columns
    # are the ones being equality-matched, which is why org_id comes first.
    op.create_index(
        "ix_conversations_scope",
        "conversations",
        ["org_id", "chatbot_id", "last_message_at"],
    )

    op.create_table(
        "messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "role",
            sa.Enum("user", "assistant", name="message_role"),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("citations", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("tokens_in", sa.Integer(), nullable=True),
        sa.Column("tokens_out", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("client_message_id", sa.String(length=80), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "tokens_in >= 0 AND tokens_out >= 0 AND latency_ms >= 0",
            name="ck_message_counters_non_negative",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"], ["conversations.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["org_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        # Idempotency. NULLs never collide in a Postgres unique index, so the
        # many messages that omit a client id coexist happily; two messages that
        # DO carry the same id in the same conversation cannot.
        sa.UniqueConstraint(
            "conversation_id", "client_message_id", name="uq_message_client_id"
        ),
    )
    op.create_index("ix_messages_org_id", "messages", ["org_id"])
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"])
    op.create_index(
        "ix_messages_conversation", "messages", ["conversation_id", "created_at"]
    )
    op.create_index("ix_messages_org_created", "messages", ["org_id", "created_at"])

    op.create_table(
        "usage_records",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("period", sa.String(length=7), nullable=False),
        sa.Column("messages_used", sa.Integer(), server_default="0", nullable=False),
        sa.Column("docs_ingested", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "messages_used >= 0 AND docs_ingested >= 0",
            name="ck_usage_counters_non_negative",
        ),
        sa.ForeignKeyConstraint(["org_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        # Load-bearing, not tidiness: `core/metering.py` turns
        # increment-or-create into one atomic statement by relying on this
        # constraint to reject the second concurrent INSERT. Drop it and two
        # simultaneous first-messages of the month create two rows, after which
        # every count is quietly wrong.
        sa.UniqueConstraint("org_id", "period", name="uq_usage_org_period"),
    )
    op.create_index("ix_usage_records_org_id", "usage_records", ["org_id"])
    op.create_index("ix_usage_records_period", "usage_records", ["period"])


def downgrade() -> None:
    # Reverse creation order so foreign keys never block a drop.
    op.drop_table("usage_records")
    op.drop_table("messages")
    op.drop_table("conversations")

    # Enum types outlive their tables; drop them explicitly. See module docs.
    for type_name in _ENUM_TYPES:
        postgresql.ENUM(name=type_name).drop(op.get_bind(), checkfirst=True)

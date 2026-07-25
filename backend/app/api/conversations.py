"""Conversation history, scoped to the caller's organization.

The read side of what `api/chat.py` writes. Phase 4's dashboard renders these;
Phase 3 needs them for a plainer reason — logging that cannot be read is logging
nobody can verify, and "usage numbers match messages sent" is a claim you check
by looking at both.

Same rules as every other tenant-scoped router here, and they are worth
restating because repetition is the mechanism: `org_id` is in the WHERE clause
rather than in an `if` after the fetch, and a row belonging to another tenant
produces 404 rather than 403, so ids cannot be enumerated for a census of the
customer base.

Handlers are sync (`def`), unlike `api/chat.py`. There is nothing to stream and
nothing to await, so FastAPI's threadpool is the right home for a blocking
`Session` — see `api/chatbots.py` for the full reasoning.

**No message-editing or deletion endpoints, deliberately.** A transcript is a
record of what actually happened, and it is the evidence for a billing dispute
("we were charged for 400 messages") as much as it is a product feature. Whole
conversations can be deleted, because a customer has a legitimate right to
remove a visitor's data; individual messages cannot, because editing one turn of
a two-turn exchange leaves a record that reads as a fact and is not one.
"""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select

from app.api.chatbots import CurrentOrg, CurrentOwner, DbSession, owned_chatbot
from app.api.schemas import ConversationDetailOut, ConversationOut
from app.db.tables import Conversation, Message
from app.security.clerk import require_org

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/bots/{chatbot_id}/conversations",
    tags=["conversations"],
    dependencies=[Depends(require_org)],
)

# Page size ceiling. A busy bot accumulates thousands of threads, and an
# unbounded list endpoint is a denial of service one `curl` away — the query is
# cheap, serializing 50,000 rows to JSON is not.
MAX_PAGE_SIZE = 100
DEFAULT_PAGE_SIZE = 50


@router.get("", response_model=list[ConversationOut])
def list_conversations(
    chatbot_id: uuid.UUID,
    db: DbSession,
    principal: CurrentOrg,
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(0, ge=0),
) -> list[Conversation]:
    """This bot's conversations, most recently active first.

    Ordered by `last_message_at` rather than `started_at`: a thread that is
    still alive matters more than one that began earlier, and this is the column
    `ix_conversations_scope` was built to serve.

    Offset pagination rather than a cursor. Honest about its weakness — a new
    message arriving between page loads shifts rows and can duplicate or skip
    one — and correct for the shape of this data, which is read by a human
    scrolling a support inbox, not by a job that must see every row exactly once.
    """
    chatbot = owned_chatbot(db, principal, chatbot_id)

    return list(
        db.scalars(
            select(Conversation)
            .where(
                Conversation.org_id == principal.org_id,
                Conversation.chatbot_id == chatbot.id,
            )
            .order_by(Conversation.last_message_at.desc())
            .limit(limit)
            .offset(offset)
        )
    )


@router.get("/{conversation_id}", response_model=ConversationDetailOut)
def get_conversation(
    chatbot_id: uuid.UUID,
    conversation_id: uuid.UUID,
    db: DbSession,
    principal: CurrentOrg,
) -> Conversation:
    """One conversation with its full transcript, oldest message first.

    Scoped on org, bot AND conversation id together. Filtering on the
    conversation id alone would be correct while ids are unique and become a
    cross-tenant read the moment one is guessed — the same reasoning as the
    Qdrant filter, applied to SQL.

    The messages come from the relationship's `order_by`, so ordering is a
    property of the model rather than something each caller remembers.
    """
    conversation = _owned_conversation(db, principal, chatbot_id, conversation_id)
    return conversation


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    chatbot_id: uuid.UUID,
    conversation_id: uuid.UUID,
    db: DbSession,
    principal: CurrentOwner,
) -> None:
    """Delete a conversation and its messages.

    Owner-only, and it does NOT decrement usage. The messages were served: the
    quota was spent when the answers were generated, and letting a delete refund
    it would make "clear your history" a way to reset your monthly bill.
    """
    conversation = _owned_conversation(db, principal, chatbot_id, conversation_id)

    db.delete(conversation)  # messages follow via cascade
    db.commit()

    logger.info(
        "deleted conversation",
        extra={
            "org_id": str(principal.org_id),
            "chatbot_id": str(chatbot_id),
            "conversation_id": str(conversation_id),
        },
    )


@router.get("/{conversation_id}/count", response_model=dict)
def count_messages(
    chatbot_id: uuid.UUID,
    conversation_id: uuid.UUID,
    db: DbSession,
    principal: CurrentOrg,
) -> dict:
    """How many messages a conversation holds, without transferring them.

    Exists so the dashboard can show "42 messages" in a list row without
    fetching 42 bodies — the same reason `ConversationOut` does not nest them.
    """
    conversation = _owned_conversation(db, principal, chatbot_id, conversation_id)
    total = (
        db.scalar(
            select(func.count(Message.id)).where(
                Message.conversation_id == conversation.id
            )
        )
        or 0
    )
    return {"conversation_id": str(conversation.id), "messages": total}


def _owned_conversation(
    db: DbSession,
    principal: CurrentOrg,
    chatbot_id: uuid.UUID,
    conversation_id: uuid.UUID,
) -> Conversation:
    """Load one conversation belonging to the caller, or 404.

    The single place this module resolves a client-supplied conversation id,
    mirroring `owned_chatbot` in `api/chatbots.py`. Two implementations of an
    ownership check are two chances to get it wrong, and only one of them would
    be tested.
    """
    owned_chatbot(db, principal, chatbot_id)  # 404s first if the bot is not ours

    conversation = db.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.org_id == principal.org_id,
            Conversation.chatbot_id == chatbot_id,
        )
    )
    if conversation is None:
        logger.info(
            "conversation not found in scope",
            extra={
                "org_id": str(principal.org_id),
                "chatbot_id": str(chatbot_id),
                "conversation_id": str(conversation_id),
            },
        )
        raise HTTPException(status_code=404, detail="conversation not found")
    return conversation

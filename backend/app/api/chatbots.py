"""Chatbot CRUD, scoped to the caller's organization.

Every query in this file filters on `principal.org_id`, and that filter is in
the WHERE clause rather than in an `if` after the fetch. The difference is not
style: fetch-then-check leaves a window in which another tenant's row is loaded
into memory, ready to be logged, returned in an error message, or acted on by
whatever gets added below the check later.

**404, not 403, for rows belonging to another tenant.**

403 is the honest status code, and it is the wrong one, because it confirms the
resource exists. An attacker walking through ids learns which are real and gets
a census of the customer base. A 404 makes "not yours" and "never existed"
indistinguishable. `_owned_chatbot` is the only way this module loads a bot, so
there is one place that decision is made.

**Handlers are sync (`def`, not `async def`).**

SQLAlchemy's `Session` is blocking. In an `async def` handler a blocking call
runs ON the event loop and stalls every other in-flight request; in a plain
`def` handler FastAPI runs the whole thing in a threadpool, where blocking is
exactly what threads are for. The cost is that we cannot `await` the Qdrant
client here — which is why vector cleanup is enqueued to Celery rather than
performed inline. That is the right shape anyway: deleting thousands of points
should not hold an HTTP connection open, and it must survive the request dying
halfway through.
"""

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.schemas import ChatbotCreate, ChatbotOut, ChatbotUpdate, UsageOut
from app.db.base import get_db
from app.db.tables import Chatbot, ChatbotStatus, Document, Organization
from app.providers.factory import get_embedder
from app.security.clerk import Principal, require_org, require_owner

logger = logging.getLogger(__name__)

# `dependencies=` at the ROUTER level, not per-route. An endpoint added to this
# router below is authenticated whether or not its author remembered to ask —
# see the deviation note in security/clerk.py for why that granularity was
# chosen over middleware.
router = APIRouter(
    prefix="/api/bots",
    tags=["chatbots"],
    dependencies=[Depends(require_org)],
)

DbSession = Annotated[Session, Depends(get_db)]
CurrentOrg = Annotated[Principal, Depends(require_org)]
CurrentOwner = Annotated[Principal, Depends(require_owner)]


def owned_chatbot(db: Session, principal: Principal, chatbot_id: uuid.UUID) -> Chatbot:
    """Load one chatbot belonging to the caller, or raise 404.

    The ONLY way this codebase loads a `Chatbot` from a client-supplied id.
    Shared with `api/documents.py` so that both routers make the ownership
    decision identically — two implementations of this function would be two
    chances to get it wrong, and only one of them would be tested.
    """
    chatbot = db.scalar(
        select(Chatbot).where(
            Chatbot.id == chatbot_id,
            # The load-bearing line. Not a post-fetch check.
            Chatbot.org_id == principal.org_id,
        )
    )
    if chatbot is None:
        logger.info(
            "chatbot not found in scope",
            extra={"org_id": str(principal.org_id), "chatbot_id": str(chatbot_id)},
        )
        raise HTTPException(status_code=404, detail="chatbot not found")
    return chatbot


def _usage(db: Session, organization: Organization) -> UsageOut:
    """Current counts against this org's plan caps."""
    limits = organization.limits
    chatbots_used = (
        db.scalar(
            select(func.count(Chatbot.id)).where(Chatbot.org_id == organization.id)
        )
        or 0
    )
    documents_used = (
        db.scalar(
            select(func.count(Document.id)).where(Document.org_id == organization.id)
        )
        or 0
    )
    return UsageOut(
        plan=organization.plan,
        chatbots_used=chatbots_used,
        chatbots_limit=limits["chatbots"],
        documents_used=documents_used,
        documents_limit=limits["documents"],
    )


def _organization(db: Session, principal: Principal) -> Organization:
    organization = db.get(Organization, principal.org_id)
    if organization is None:
        # The token verified against a row that has since vanished — a delete
        # racing an in-flight request. 403 rather than 404: the caller is
        # authenticated, they just no longer belong anywhere.
        raise HTTPException(status_code=403, detail="organization not found")
    return organization


@router.get("", response_model=list[ChatbotOut])
def list_chatbots(db: DbSession, principal: CurrentOrg) -> list[Chatbot]:
    """Every bot in the caller's org, newest first.

    Takes no filter arguments at all. An `org_id` query parameter is the classic
    version of this bug — it looks like a convenience and it is a tenancy hole.
    """
    return list(
        db.scalars(
            select(Chatbot)
            .where(Chatbot.org_id == principal.org_id)
            .order_by(Chatbot.created_at.desc())
        )
    )


@router.get("/usage", response_model=UsageOut)
def get_usage(db: DbSession, principal: CurrentOrg) -> UsageOut:
    """Consumption against plan caps, for the dashboard's usage bar.

    Declared BEFORE `/{chatbot_id}` on purpose: FastAPI matches routes in
    declaration order, so a path parameter registered first would swallow
    "/usage" and try to parse it as a UUID.
    """
    return _usage(db, _organization(db, principal))


@router.post("", response_model=ChatbotOut, status_code=status.HTTP_201_CREATED)
def create_chatbot(
    payload: ChatbotCreate, db: DbSession, principal: CurrentOrg
) -> Chatbot:
    """Create a bot, subject to the org's plan cap.

    `embedding_model` and `dimension` are read from the LIVE embedder rather
    than from the request. This is PLAN §3 rule 1 in practice: whatever provider
    is configured right now is what will embed this bot's documents, so that is
    what gets recorded. A client-supplied dimension would let someone create a
    bot whose vectors the collection can never accept.
    """
    organization = _organization(db, principal)
    usage = _usage(db, organization)
    if usage.chatbots_used >= usage.chatbots_limit:
        # 429 with the numbers attached, so the dashboard can render "1 of 1
        # bots used — upgrade to add more" without a second request.
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "plan_limit_reached",
                "message": (
                    f"the {organization.plan} plan allows "
                    f"{usage.chatbots_limit} chatbot(s)"
                ),
                "usage": usage.model_dump(mode="json"),
            },
        )

    embedder = get_embedder()
    chatbot = Chatbot(
        org_id=principal.org_id,
        name=payload.name,
        system_prompt=payload.system_prompt,
        model_overrides=payload.model_overrides,
        embedding_model=embedder.model,
        dimension=embedder.dimension,
        status=ChatbotStatus.DRAFT,
    )
    db.add(chatbot)

    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        # The only unique constraint reachable here is (org_id, name). Mapping
        # it to 409 rather than letting a 500 escape means the dashboard can say
        # "you already have a bot called that" instead of "something went wrong".
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"a chatbot named {payload.name!r} already exists",
        ) from exc

    db.refresh(chatbot)
    logger.info(
        "created chatbot",
        extra={"org_id": str(principal.org_id), "chatbot_id": str(chatbot.id)},
    )
    return chatbot


@router.get("/{chatbot_id}", response_model=ChatbotOut)
def get_chatbot(chatbot_id: uuid.UUID, db: DbSession, principal: CurrentOrg) -> Chatbot:
    return owned_chatbot(db, principal, chatbot_id)


@router.patch("/{chatbot_id}", response_model=ChatbotOut)
def update_chatbot(
    chatbot_id: uuid.UUID,
    payload: ChatbotUpdate,
    db: DbSession,
    principal: CurrentOrg,
) -> Chatbot:
    """Partial update.

    `exclude_unset=True` is what makes PATCH semantics correct: it distinguishes
    "the client did not mention system_prompt" from "the client sent
    system_prompt: null to clear it". Using the model's plain dict would treat
    every omitted field as an explicit null and wipe settings the client never
    touched.
    """
    chatbot = owned_chatbot(db, principal, chatbot_id)

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(chatbot, field, value)

    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="a chatbot with that name already exists",
        ) from exc

    db.refresh(chatbot)
    return chatbot


@router.delete("/{chatbot_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chatbot(
    chatbot_id: uuid.UUID, db: DbSession, principal: CurrentOwner
) -> None:
    """Delete a bot, its documents, and its vectors.

    Owner-only: this is irreversible and destroys ingested work, which is not
    something a member should be able to do by clicking the wrong row.

    The SQL rows go now (ON DELETE CASCADE reaches documents); the Qdrant points
    are removed by a Celery task. Ordering matters and is deliberate — the rows
    are the source of truth, so once they are gone the bot is unreachable even
    if vector cleanup is delayed or fails. Orphaned points are wasted storage,
    not a leak: nothing can search them, because every search filters on an
    org/bot pair that no longer resolves to anything.
    """
    chatbot = owned_chatbot(db, principal, chatbot_id)
    org_id, bot_id = str(chatbot.org_id), str(chatbot.id)

    db.delete(chatbot)
    db.commit()

    # Imported here, not at module scope: importing the Celery app pulls the
    # broker configuration into the web process, and a router module should not
    # fail to import because Redis is unreachable.
    from app.workers.ingest_tasks import purge_chatbot_vectors

    purge_chatbot_vectors.delay(org_id=org_id, chatbot_id=bot_id)

    logger.info("deleted chatbot", extra={"org_id": org_id, "chatbot_id": bot_id})

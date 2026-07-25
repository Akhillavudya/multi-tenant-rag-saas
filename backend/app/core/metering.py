"""Message metering and plan-cap enforcement.

One question, asked on the hot path of every chat request: *may this
organization send another message this month, and what is the new count?*
Answering it correctly is the difference between a free tier and a free
product.

**Reserve before the work, not after.**

The counter is incremented BEFORE the LLM is called, and refunded if the answer
never happens. The tempting alternative — meter the successful answer — has a
hole that is obvious once seen: a client sitting at its cap fires a hundred
concurrent requests, every one of them checks the counter before any of them
finishes, and all hundred are served. Reserving first means the hundredth
request is refused while the ninety-nine are still generating.

The cost is that a failed generation must give the reservation back, or the
counter drifts above reality. `release_message` is that refund, and
`tests/test_limits.py` holds the line on it.

**The increment is one statement, and that is the whole design.**

    UPDATE usage_records
       SET messages_used = messages_used + 1
     WHERE org_id = :org AND period = :period
    RETURNING messages_used

The new value is computed by the database, inside a statement it serializes on
the row lock, and returned. Nothing is read into Python and written back. The
shape being avoided is:

    used = SELECT messages_used   -- both requests read 99
    if used < limit: UPDATE SET messages_used = 100   -- both write 100

which is the classic *lost update*: two requests, one increment, one free
message. It cannot be caught by a single-threaded test, which is why
`test_increment_is_computed_by_the_database` asserts on the source instead.

**Why the row is created lazily.** An organization that never chats has no
usage row. Creating one per org per month at signup would mean a scheduled job
writing rows nobody reads. The cost of laziness is that "increment" is really
"increment or create", and two first-messages of the month can race to create
the same row — which the unique constraint on (org_id, period) turns into an
IntegrityError we catch and retry as an update. The constraint is doing real
work here, not documentation.

**Why the limit is read from the plan every time.** `UsageSnapshot.messages_limit`
comes from `organization.plan` at call time, never from a column on the usage
row. Phase 6's Stripe webhook changes the plan and nothing else; a limit copied
onto the row at creation would leave an upgraded customer capped at the tier
they just left, for the rest of the month.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import case, insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.tables import Organization, PlanTier, UsageRecord

logger = logging.getLogger(__name__)

# The billing-period key. Two digits for the month, always — "2026-7" and
# "2026-07" would be two counters for one month, and the bug would surface as a
# customer whose quota mysteriously reset mid-month.
PERIOD_FORMAT = "%Y-%m"

# Which plan number this module enforces. Named rather than inlined because the
# 429 payload reports it and Phase 6 adds siblings ("chatbots", "documents").
MESSAGES = "messages_per_month"


def current_period(now: datetime | None = None) -> str:
    """The current billing period as "YYYY-MM", in UTC.

    UTC rather than server-local time: the month must roll over at the same
    instant everywhere, or a laptop in IST and a server in UTC disagree about
    which month a message belongs to for five and a half hours every month.

    A naive datetime is treated as UTC rather than rejected — callers pass one
    only from tests, and guessing "local" would reintroduce exactly the
    ambiguity this function exists to remove.
    """
    moment = now or datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).strftime(PERIOD_FORMAT)


@dataclass(frozen=True)
class UsageSnapshot:
    """What an organization has spent this period, and what it is allowed.

    Frozen and self-contained so it can be handed straight to a response model
    or attached to an error — the dashboard's usage bar and the 429 body are
    the same four numbers.
    """

    plan: PlanTier
    period: str
    messages_used: int
    messages_limit: int

    @property
    def remaining(self) -> int:
        """Messages left, clamped at zero.

        Never negative, because usage legitimately exceeds the limit after a
        mid-month downgrade: the customer keeps the messages they already sent
        and simply cannot send more. A negative "remaining" would render as a
        progress bar past 100% and read as a bug in our accounting.
        """
        return max(0, self.messages_limit - self.messages_used)

    @property
    def exceeded(self) -> bool:
        return self.messages_used >= self.messages_limit

    def as_dict(self) -> dict:
        return {
            "plan": str(self.plan),
            "period": self.period,
            "messages_used": self.messages_used,
            "messages_limit": self.messages_limit,
            "messages_remaining": self.remaining,
        }


class PlanLimitExceededError(Exception):
    """The organization has spent its allowance for this period.

    An exception rather than a boolean return, deliberately: a returned flag can
    be ignored by a caller who forgets to check it, and the failure mode of
    forgetting is "the cap silently does not exist". This cannot be dropped
    accidentally.

    Carries the numbers, not just a message, so `api/chat.py` can build the 429
    body without a second query and the widget can say something specific to a
    stranger on a customer's website.
    """

    def __init__(
        self,
        *,
        plan: PlanTier,
        period: str,
        used: int,
        limit: int,
        resource: str = MESSAGES,
    ) -> None:
        super().__init__(
            f"the {plan} plan allows {limit} messages per month; "
            f"{used} used in {period}"
        )
        self.plan = plan
        self.period = period
        self.used = used
        self.limit = limit
        self.resource = resource

    def payload(self) -> dict:
        """The JSON body of the 429. Shaped like the plan-limit payloads
        `api/chatbots.py` already returns, so a client has one error format to
        handle rather than three."""
        return {
            "error": "plan_limit_reached",
            "message": str(self),
            "resource": self.resource,
            "plan": str(self.plan),
            "period": self.period,
            "used": self.used,
            "limit": self.limit,
            # When the allowance refills. The widget shows a stranger "try again
            # on 1 August" instead of an unexplained refusal.
            "resets_at": _next_period_start(self.period).isoformat(),
        }


def _next_period_start(period: str) -> datetime:
    """First instant of the month after `period`, in UTC."""
    year, month = (int(part) for part in period.split("-"))
    return (
        datetime(year + 1, 1, 1, tzinfo=UTC)
        if month == 12
        else datetime(year, month + 1, 1, tzinfo=UTC)
    )


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def message_usage(
    db: Session, organization: Organization, *, period: str | None = None
) -> UsageSnapshot:
    """Current consumption for `organization`, without consuming anything.

    Reads the counter column directly rather than loading the `UsageRecord`
    entity: the writes below go through Core statements, so an ORM object left
    in the session's identity map would report the value it had when it was
    loaded, not the value in the database.
    """
    period = period or current_period()
    used = db.scalar(
        select(UsageRecord.messages_used).where(
            UsageRecord.org_id == organization.id,
            UsageRecord.period == period,
        )
    )
    return UsageSnapshot(
        plan=organization.plan,
        period=period,
        # No row is not an error; it is zero. Anything else would break the
        # dashboard for exactly the customers who just signed up.
        messages_used=used or 0,
        messages_limit=organization.limits[MESSAGES],
    )


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def reserve_message(
    db: Session,
    organization: Organization,
    *,
    period: str | None = None,
) -> UsageSnapshot:
    """Claim one message of quota, or raise `PlanLimitExceededError`.

    Takes the `Organization` row rather than an org_id, on purpose: the cap
    comes from `organization.plan`, and a signature that accepted a bare id
    would need either a hidden extra query or a caller-supplied limit — and a
    caller-supplied limit is a plan cap the caller gets to choose.

    Commits before returning. The reservation must be durable and its row lock
    released before the request goes off to spend seconds inside an LLM call;
    holding a transaction open for the length of a generation would serialize
    every message that organization sends.

    On refusal the increment is undone in the SAME transaction, so no other
    request ever observes the inflated count and the committed number is
    exactly "messages actually served".
    """
    period = period or current_period()
    limit = organization.limits[MESSAGES]

    used = _bump_messages(db, organization.id, period, delta=1)

    if used > limit:
        # One past the cap: this request is the refusal. Give the reservation
        # back before committing, so the stored counter matches reality.
        _bump_messages(db, organization.id, period, delta=-1)
        db.commit()
        logger.info(
            "plan message limit reached",
            extra={
                "org_id": str(organization.id),
                "plan": str(organization.plan),
                "period": period,
                "limit": limit,
            },
        )
        raise PlanLimitExceededError(
            plan=organization.plan, period=period, used=used - 1, limit=limit
        )

    db.commit()
    return UsageSnapshot(
        plan=organization.plan,
        period=period,
        messages_used=used,
        messages_limit=limit,
    )


def release_message(
    db: Session, org_id: uuid.UUID, *, period: str | None = None
) -> None:
    """Refund one reserved message.

    Called when the answer did not happen — every provider failed, the client
    disconnected before the first token, the request was rejected downstream.
    Without it, a provider outage would silently spend a customer's monthly
    quota on answers they never received.

    Never creates a row and never goes below zero: a duplicate refund must not
    manufacture free quota, and a negative counter would eventually trip the
    table's CHECK constraint somewhere far away from the cause.
    """
    period = period or current_period()
    result = db.execute(
        update(UsageRecord)
        .where(UsageRecord.org_id == org_id, UsageRecord.period == period)
        .values(
            {
                UsageRecord.messages_used: case(
                    (UsageRecord.messages_used > 0, UsageRecord.messages_used - 1),
                    else_=0,
                )
            }
        )
        .execution_options(synchronize_session=False)
    )
    db.commit()

    if result.rowcount == 0:
        # Nothing was reserved for this period. Not an error: a request can be
        # refused before it ever reserves.
        logger.debug(
            "release with no usage row",
            extra={"org_id": str(org_id), "period": period},
        )


def record_document_ingested(
    db: Session, org_id: uuid.UUID, *, period: str | None = None
) -> None:
    """Count one successfully ingested document against this period.

    Separate from the document CAP, which `api/documents.py` enforces by
    counting rows in `documents` — that is a stock (how many you store, freed
    when you delete one), while this is a flow (how much ingestion work you
    caused this month). Phase 6 prices the flow; the stock is what the plan
    limits.

    Best-effort by design: this is called at the end of a successful Celery
    ingestion, and a metering failure must never turn a document the user can
    see and query into a FAILED row.
    """
    period = period or current_period()
    try:
        _bump(
            db,
            org_id,
            period,
            updates={UsageRecord.docs_ingested: UsageRecord.docs_ingested + 1},
            returning=UsageRecord.docs_ingested,
            insert_values={"docs_ingested": 1},
        )
        db.commit()
    except Exception:  # noqa: BLE001 - metering must not fail ingestion
        db.rollback()
        logger.warning(
            "could not record document ingestion",
            extra={"org_id": str(org_id), "period": period},
            exc_info=True,
        )


# --------------------------------------------------------------------------
# The one statement everything above is built from
# --------------------------------------------------------------------------


def _bump_messages(db: Session, org_id: uuid.UUID, period: str, *, delta: int) -> int:
    """Add `delta` to `messages_used` and return the new value.

    `delta` is only ever +1 (reserve) or -1 (the compensating undo inside
    `reserve_message`). The refund path used by callers is `release_message`,
    which clamps at zero; this one deliberately does not, because it is only
    ever undoing an increment it just made in the same transaction.
    """
    return _bump(
        db,
        org_id,
        period,
        updates={UsageRecord.messages_used: UsageRecord.messages_used + delta},
        returning=UsageRecord.messages_used,
        insert_values={"messages_used": max(delta, 0)},
    )


def _bump(
    db: Session,
    org_id: uuid.UUID,
    period: str,
    *,
    updates: dict,
    returning,
    insert_values: dict,
) -> int:
    """Increment-or-create, atomically, returning the new counter value.

    Three steps, in this order for a reason:

    1. UPDATE ... RETURNING. Succeeds for every request after the first of the
       month, which is the overwhelmingly common case, so the cheap path runs
       first.
    2. If no row was updated, INSERT ... RETURNING. Only the first message of a
       period reaches this.
    3. If that INSERT hits the unique constraint, another request created the
       row a microsecond earlier. Roll back and retry the UPDATE — the row now
       exists, so the retry cannot fail the same way twice.

    Step 3 is the part that would be tempting to skip. Skipping it means the
    first two messages of a month can produce an unhandled IntegrityError, which
    is a 500 on a request that did nothing wrong, once a month, only under
    concurrency. `synchronize_session=False` throughout: these are Core
    statements against a table no ORM object in this session holds.
    """
    statement = (
        update(UsageRecord)
        .where(UsageRecord.org_id == org_id, UsageRecord.period == period)
        .values(updates)
        .returning(returning)
        .execution_options(synchronize_session=False)
    )

    value = db.execute(statement).scalar_one_or_none()
    if value is not None:
        return value

    try:
        # SAVEPOINT: a failed INSERT would otherwise poison the whole
        # transaction, taking the caller's uncommitted work down with it.
        with db.begin_nested():
            value = db.execute(
                insert(UsageRecord)
                .values(org_id=org_id, period=period, **insert_values)
                .returning(returning)
            ).scalar_one()
        return value
    except IntegrityError:
        # Lost the race to create the row. It exists now, so the update works.
        value = db.execute(statement).scalar_one_or_none()
        if value is None:
            # The row vanished between the INSERT conflict and this retry —
            # only possible if the organization was deleted mid-request.
            raise
        return value

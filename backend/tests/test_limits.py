"""Plan-cap enforcement and message metering.

The second test file PLAN §7 runs in CI on every push, and the second one
written BEFORE its implementation (Hard Rule #4). `test_isolation.py` protects
the promise that tenants cannot see each other; this one protects the promise
that a free plan is actually free-tier-shaped — that the product cannot be used
without limit by someone who is not paying, and that the number we show a
customer matches the number we charged them for.

**What "correct" means here, precisely.** PLAN §6's acceptance criterion is
"the 101st free-tier message is politely refused; usage numbers in DB match
messages sent". Those are two different guarantees and both are tested:

1. *The cap holds.* Message 100 succeeds, message 101 does not.
2. *The counter is honest.* A refused message must not increment anything. A
   counter that climbs past the cap is not merely untidy — it means the usage
   bar the customer sees, and the invoice Phase 6 builds from it, describe work
   that was never done.

**Why the increment is the interesting part.** The naive implementation is

    used = SELECT messages_used ...      # read
    if used >= limit: refuse
    UPDATE ... SET messages_used = used + 1   # write what we read

which is a read-modify-write race: two requests arriving together both read 99,
both decide they are under the cap, and both write 100. One message is free, and
under real concurrency the leak is proportional to traffic. The fix is to make
the increment a single SQL statement that the database serializes on the row
lock — `UPDATE ... SET messages_used = messages_used + 1 RETURNING messages_used`
— and to decide from the value the database returns rather than from a value we
read earlier. `test_increment_is_computed_by_the_database` asserts the
implementation stayed that shape, because the race it prevents is invisible in a
single-threaded test and expensive in production.

**SQLite, not Postgres, and not Docker.** The schema builds on SQLite because
`db/tables.py` declares its one JSONB column with a SQLite variant (see
`JSON_COLUMN` there). Same reasoning as the isolation test's in-memory Qdrant:
a security test that needs a container is a security test CI eventually skips.
What this trades away is honest and worth stating — SQLite does not exercise
Postgres row locking, so the concurrency guarantee above is covered
structurally rather than behaviourally, and enum types here are CHECK
constraints rather than real PG enums.
"""

import inspect
import re
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core import metering
from app.core.metering import (
    PlanLimitExceededError,
    current_period,
    message_usage,
    record_document_ingested,
    release_message,
    reserve_message,
)
from app.db.tables import (
    PLAN_LIMITS,
    Channel,
    Chatbot,
    Conversation,
    Message,
    MessageRole,
    Organization,
    PlanTier,
    UsageRecord,
)

FREE_MESSAGE_CAP = PLAN_LIMITS[PlanTier.FREE]["messages_per_month"]


def _org(
    db: Session, plan: PlanTier = PlanTier.FREE, name: str = "Acme"
) -> Organization:
    """Create and commit one organization.

    `clerk_org_id` is randomised because it is globally unique; a test that
    creates two orgs must not collide on a constant.
    """
    organization = Organization(
        name=name,
        clerk_org_id=f"org_{uuid.uuid4().hex[:12]}",
        plan=plan,
    )
    db.add(organization)
    db.commit()
    return organization


def _spend(db: Session, organization: Organization, count: int) -> None:
    """Consume `count` messages of quota."""
    for _ in range(count):
        reserve_message(db, organization)


# --------------------------------------------------------------------------
# 1. The billing period
# --------------------------------------------------------------------------


def test_period_is_utc_year_month() -> None:
    """The period key is "YYYY-MM" in UTC.

    Format matters because it is a primary-key component: "2026-7" and "2026-07"
    are different strings and would silently become two counters for one month.
    """
    assert current_period(datetime(2026, 7, 25, 12, 0, tzinfo=UTC)) == "2026-07"
    assert current_period(datetime(2026, 1, 1, 0, 0, tzinfo=UTC)) == "2026-01"


def test_period_of_a_naive_datetime_is_still_utc() -> None:
    """A naive timestamp is interpreted as UTC rather than as local time.

    The alternative — trusting the server's timezone — means the month rolls
    over at a different instant on a developer's laptop than on the Oracle box,
    and the bug appears once a month, at midnight, in production only.
    """
    assert current_period(datetime(2026, 12, 31, 23, 59)) == "2026-12"


def test_default_period_tracks_the_clock() -> None:
    """Called with no argument, the period is *now*."""
    assert current_period() == datetime.now(UTC).strftime("%Y-%m")


# --------------------------------------------------------------------------
# 2. Reserving a message
# --------------------------------------------------------------------------


def test_first_message_creates_the_usage_row(db: Session) -> None:
    """A brand-new org has no usage row until it sends something.

    Rows are created on demand rather than seeded at signup: an org that never
    chats should not accumulate a row per month forever.
    """
    organization = _org(db)
    assert db.scalar(select(func.count(UsageRecord.id))) == 0

    usage = reserve_message(db, organization)

    assert usage.messages_used == 1
    assert usage.messages_limit == FREE_MESSAGE_CAP
    assert db.scalar(select(func.count(UsageRecord.id))) == 1


def test_each_message_increments_by_exactly_one(db: Session) -> None:
    organization = _org(db)

    for expected in range(1, 6):
        assert reserve_message(db, organization).messages_used == expected

    record = db.scalar(select(UsageRecord).where(UsageRecord.org_id == organization.id))
    assert record is not None and record.messages_used == 5


def test_usage_snapshot_reports_without_consuming(db: Session) -> None:
    """Reading usage must not spend it.

    The dashboard polls this for the usage bar. If reading were a write, simply
    watching your quota would exhaust it.
    """
    organization = _org(db)
    _spend(db, organization, 3)

    first = message_usage(db, organization)
    second = message_usage(db, organization)

    assert first.messages_used == second.messages_used == 3
    assert first.remaining == FREE_MESSAGE_CAP - 3


def test_usage_snapshot_of_an_untouched_org_is_zero(db: Session) -> None:
    """No usage row is not an error — it is zero.

    A KeyError or None here would make the dashboard fail for exactly the
    customers who just signed up.
    """
    usage = message_usage(db, _org(db))

    assert usage.messages_used == 0
    assert usage.messages_limit == FREE_MESSAGE_CAP
    assert not usage.exceeded


# --------------------------------------------------------------------------
# 3. The cap — PLAN §6's "the 101st free-tier message is politely refused"
# --------------------------------------------------------------------------


def test_free_plan_allows_exactly_its_quota(db: Session) -> None:
    """Messages 1..100 succeed. Off-by-one in either direction is a real bug:
    refusing at 100 cheats the customer, allowing 101 cheats us."""
    organization = _org(db)

    _spend(db, organization, FREE_MESSAGE_CAP)

    assert message_usage(db, organization).messages_used == FREE_MESSAGE_CAP


def test_the_next_message_over_the_cap_is_refused(db: Session) -> None:
    organization = _org(db)
    _spend(db, organization, FREE_MESSAGE_CAP)

    with pytest.raises(PlanLimitExceededError):
        reserve_message(db, organization)


def test_a_refused_message_does_not_consume_quota(db: Session) -> None:
    """THE metering guarantee: usage in the database equals messages served.

    The implementation increments first and compensates on refusal, because the
    increment is what makes the check atomic. This test is what stops that
    compensation from being quietly dropped — without it the counter would drift
    upward on every rejected request, and a customer who hit their cap once
    would appear to be using 500% of a plan they cannot exceed.
    """
    organization = _org(db)
    _spend(db, organization, FREE_MESSAGE_CAP)

    for _ in range(5):
        with pytest.raises(PlanLimitExceededError):
            reserve_message(db, organization)

    assert message_usage(db, organization).messages_used == FREE_MESSAGE_CAP


def test_refusal_carries_the_numbers_a_client_needs(db: Session) -> None:
    """The error is a payload, not just a status.

    The dashboard renders "100 of 100 messages used on the free plan — upgrade
    to continue" from this without a second request, and the widget shows the
    end user a sentence rather than a stack trace.
    """
    organization = _org(db)
    _spend(db, organization, FREE_MESSAGE_CAP)

    with pytest.raises(PlanLimitExceededError) as raised:
        reserve_message(db, organization)

    error = raised.value
    assert error.plan is PlanTier.FREE
    assert error.used == FREE_MESSAGE_CAP
    assert error.limit == FREE_MESSAGE_CAP
    assert error.period == current_period()
    assert "free" in str(error).lower()


def test_release_returns_quota_to_the_pool(db: Session) -> None:
    """A reservation that could not be fulfilled is refunded.

    A chat request reserves before calling the LLM, because the alternative —
    metering after a successful answer — lets a client at the cap start a
    hundred concurrent generations for free. When the provider then fails, the
    customer must not be charged a message they never received.
    """
    organization = _org(db)
    _spend(db, organization, 10)

    release_message(db, organization.id)

    assert message_usage(db, organization).messages_used == 9


def test_release_never_drives_the_counter_negative(db: Session) -> None:
    """A double refund (a retry, a buggy caller) must not create free quota.

    The CHECK constraint would reject a negative value with an IntegrityError
    somewhere far from the cause, so the floor is enforced in the statement.
    """
    organization = _org(db)
    reserve_message(db, organization)

    release_message(db, organization.id)
    release_message(db, organization.id)

    assert message_usage(db, organization).messages_used == 0


def test_release_for_an_org_with_no_usage_row_is_a_no_op(db: Session) -> None:
    """Refunding something never reserved does nothing and raises nothing."""
    organization = _org(db)

    release_message(db, organization.id)

    assert db.scalar(select(func.count(UsageRecord.id))) == 0


# --------------------------------------------------------------------------
# 4. Counters do not bleed across tenants, plans, or months
# --------------------------------------------------------------------------


def test_usage_is_scoped_per_organization(db: Session) -> None:
    """One tenant's traffic must never spend another's quota.

    The same class of bug as a missing Qdrant filter, on the SQL side: a
    counter keyed on the wrong column is a cross-tenant leak that presents as a
    billing dispute rather than as a data breach.
    """
    first = _org(db, name="First")
    second = _org(db, name="Second")

    _spend(db, first, 7)

    assert message_usage(db, first).messages_used == 7
    assert message_usage(db, second).messages_used == 0


def test_one_org_at_its_cap_does_not_block_another(db: Session) -> None:
    exhausted = _org(db, name="Exhausted")
    fresh = _org(db, name="Fresh")
    _spend(db, exhausted, FREE_MESSAGE_CAP)

    assert reserve_message(db, fresh).messages_used == 1


def test_usage_is_scoped_per_period(db: Session) -> None:
    """Last month's traffic does not count against this month's cap.

    This is what makes the quota a monthly allowance rather than a lifetime
    one. Written by pinning the period explicitly, since a test cannot wait for
    the calendar.
    """
    organization = _org(db)
    last_month = "2026-06"

    for _ in range(FREE_MESSAGE_CAP):
        reserve_message(db, organization, period=last_month)

    assert message_usage(db, organization, period=last_month).exceeded
    assert message_usage(db, organization).messages_used == 0
    assert reserve_message(db, organization).messages_used == 1


def test_upgrading_the_plan_raises_the_cap_immediately(db: Session) -> None:
    """A plan change takes effect on the next message, with no counter reset.

    Phase 6's Stripe webhook flips `organizations.plan` and nothing else. That
    only works because the limit is read from the plan at check time rather
    than copied onto the usage row when it was created — a denormalized limit
    would leave an upgraded customer capped at the tier they left.
    """
    organization = _org(db)
    _spend(db, organization, FREE_MESSAGE_CAP)

    organization.plan = PlanTier.PRO
    db.commit()

    usage = reserve_message(db, organization)

    assert usage.messages_limit == PLAN_LIMITS[PlanTier.PRO]["messages_per_month"]
    assert usage.messages_used == FREE_MESSAGE_CAP + 1
    assert not usage.exceeded


def test_downgrading_below_current_usage_refuses_rather_than_crashes(
    db: Session,
) -> None:
    """A cancelled subscription mid-month leaves usage above the new cap.

    That state is legitimate and must be handled: the customer keeps what they
    already sent and simply cannot send more. Any assertion that usage <= limit
    would turn a normal downgrade into a 500 on every subsequent request.
    """
    organization = _org(db, plan=PlanTier.PRO)
    _spend(db, organization, FREE_MESSAGE_CAP + 10)

    organization.plan = PlanTier.FREE
    db.commit()

    usage = message_usage(db, organization)
    assert usage.exceeded
    assert usage.remaining == 0  # clamped, never negative
    with pytest.raises(PlanLimitExceededError):
        reserve_message(db, organization)


# --------------------------------------------------------------------------
# 5. Documents ingested — the other half of usage_records
# --------------------------------------------------------------------------


def test_document_ingestion_is_counted_separately(db: Session) -> None:
    """`docs_ingested` is a monthly flow, not the stock of stored documents.

    The document CAP is enforced against `count(documents)` in
    `api/documents.py`, because deleting a file should free a slot. This
    counter answers a different question — how much ingestion work an org
    caused this month — and Phase 6 prices that separately.
    """
    organization = _org(db)

    record_document_ingested(db, organization.id)
    record_document_ingested(db, organization.id)
    reserve_message(db, organization)

    record = db.scalar(select(UsageRecord).where(UsageRecord.org_id == organization.id))
    assert record is not None
    assert record.docs_ingested == 2
    assert record.messages_used == 1  # the two counters do not interfere


# --------------------------------------------------------------------------
# 6. Reconciliation — the number matches the transcript
# --------------------------------------------------------------------------


def test_usage_matches_the_messages_actually_stored(db: Session) -> None:
    """PLAN §6: "usage numbers in DB match messages sent".

    Metering and transcript logging are separate writes to separate tables, so
    they can drift. This test reconciles them the way an operator would: count
    the user messages in the transcript, compare to the counter.
    """
    organization = _org(db)
    chatbot = Chatbot(
        org_id=organization.id,
        name="Support",
        embedding_model="stub",
        dimension=16,
    )
    db.add(chatbot)
    db.commit()

    conversation = Conversation(
        org_id=organization.id, chatbot_id=chatbot.id, channel=Channel.WIDGET
    )
    db.add(conversation)
    db.commit()

    sent = 12
    for index in range(sent):
        reserve_message(db, organization)
        db.add_all(
            [
                Message(
                    org_id=organization.id,
                    conversation_id=conversation.id,
                    role=MessageRole.USER,
                    content=f"question {index}",
                ),
                Message(
                    org_id=organization.id,
                    conversation_id=conversation.id,
                    role=MessageRole.ASSISTANT,
                    content=f"answer {index}",
                    citations=[{"citation": "handbook.pdf p.1"}],
                    tokens_in=100,
                    tokens_out=40,
                    latency_ms=850,
                ),
            ]
        )
    db.commit()

    questions_logged = db.scalar(
        select(func.count(Message.id)).where(
            Message.org_id == organization.id,
            Message.role == MessageRole.USER,
        )
    )

    assert questions_logged == sent
    assert message_usage(db, organization).messages_used == sent


# --------------------------------------------------------------------------
# 7. Structural — the race stays unwritable
# --------------------------------------------------------------------------


def test_increment_is_computed_by_the_database() -> None:
    """The counter must be incremented by SQL, never by Python.

    A structural test, for the same reason `test_isolation.py` has two: the bug
    it guards against cannot be reproduced here. SQLite serializes everything,
    so a read-modify-write implementation passes every behavioural test above
    and still loses messages under real concurrency on Postgres.

    What it checks: the counter appears on BOTH sides of its own assignment
    (`UsageRecord.messages_used: UsageRecord.messages_used + <something>`), which
    is only expressible as a SQL expression — the database computes the new value
    inside a statement it serializes on the row lock. A Python-side `+= 1` on a
    loaded ORM attribute, or a value read by an earlier SELECT, cannot match.

    Matched by pattern rather than by an exact string: the delta is
    parameterised so `reserve_message` can undo its own increment, and pinning
    the literal `+ 1` would fail on a refactor that changes nothing about the
    property being protected.
    """
    source = inspect.getsource(metering)

    assert re.search(
        r"UsageRecord\.messages_used\s*\+\s*\w+", source
    ), "the increment must be a SQL expression over the column, not a Python value"
    assert (
        "messages_used += " not in source
    ), "a Python-side increment is a read-modify-write race under concurrency"

    # Deliberately only two assertions. A third — "no previously-read value is
    # ever assigned to the column" — cannot be expressed by scanning text
    # without also matching innocent things like `UsageSnapshot(messages_used=used)`,
    # and a structural test that cries wolf is one people delete.


def test_reserve_requires_an_organization_not_an_id() -> None:
    """`reserve_message` takes the Organization row, not an org_id.

    Deliberate: the limit comes from `organization.plan`, so accepting a bare id
    would mean either a second query hidden inside metering or a caller-supplied
    limit — and a caller-supplied limit is a plan cap the caller can choose.
    """
    parameters = inspect.signature(reserve_message).parameters

    assert "organization" in parameters
    assert (
        "limit" not in parameters
    ), "the cap must be derived from the plan, never passed in"

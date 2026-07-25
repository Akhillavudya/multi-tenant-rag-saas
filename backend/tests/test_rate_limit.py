"""Token-bucket rate limiting.

Written AFTER `security/rate_limit.py`, unlike `test_limits.py` and
`test_bot_key.py`. That ordering is deliberate rather than lazy: Hard Rule #4
puts tests first for isolation, plan limits, auth and webhooks — the things
whose failure is a breach or a billing error. A rate limiter protects budget and
availability, and its failure mode is "some abuse got through", which is a
different severity.

Every test here drives the **in-process fallback**, by pointing the limiter at a
Redis that cannot exist. That is not a shortcut around Redis — it is the path
that runs during a Redis outage, so it is the path most worth having covered.
The Lua script's arithmetic is identical; what Redis adds is sharing the bucket
between processes, which a single-process test could not observe anyway.

Time is injected rather than waited for. A test that sleeps to watch a bucket
refill is a slow test that is also flaky on a loaded CI machine.
"""

import pytest

from app.security import rate_limit
from app.security.rate_limit import RateLimitExceededError, TokenBucketLimiter

# A port nothing listens on: connection is refused immediately, so every Redis
# call fails fast and the limiter degrades to its in-process buckets.
UNREACHABLE_REDIS = "redis://127.0.0.1:1"


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch):
    """A controllable clock for the limiter.

    Returns a callable that advances time. The limiter reads `time.time()`
    exactly once per check, so a fake is enough to simulate any elapsed
    interval instantly.
    """
    current = {"now": 1_000_000.0}

    monkeypatch.setattr(rate_limit.time, "time", lambda: current["now"])

    def advance(seconds: float) -> None:
        current["now"] += seconds

    return advance


def _limiter(per_minute: int = 60, burst: int = 3) -> TokenBucketLimiter:
    return TokenBucketLimiter(
        per_minute=per_minute, burst=burst, redis_url=UNREACHABLE_REDIS
    )


async def test_a_full_bucket_allows_a_burst(clock) -> None:
    """`burst` requests may arrive back to back.

    This is the behaviour a fixed-window counter gets wrong in the other
    direction, and it matters: a person asks a question, reads, asks a
    follow-up. Throttling that pattern would make a working product feel broken.
    """
    limiter = _limiter(burst=3)

    decisions = [await limiter.check("bot:abc") for _ in range(3)]

    assert all(decision.allowed for decision in decisions)


async def test_the_request_after_the_burst_is_refused(clock) -> None:
    limiter = _limiter(burst=3)
    for _ in range(3):
        await limiter.check("bot:abc")

    decision = await limiter.check("bot:abc")

    assert not decision.allowed
    assert decision.retry_after >= 1


async def test_tokens_refill_with_elapsed_time(clock) -> None:
    """Refill is lazy arithmetic on the timestamp, not a background timer.

    60/min is one token per second, so one second of silence buys exactly one
    request back.
    """
    limiter = _limiter(per_minute=60, burst=3)
    for _ in range(3):
        await limiter.check("bot:abc")
    assert not (await limiter.check("bot:abc")).allowed

    clock(1.01)

    assert (await limiter.check("bot:abc")).allowed


def test_a_bucket_never_refills_past_its_capacity() -> None:
    """An hour of silence does not buy an hour's worth of burst.

    Without the `min(capacity, ...)` clamp, an idle key would accumulate
    thousands of tokens and could then spend them all at once — which is
    precisely the flood the limiter exists to prevent.
    """
    limiter = _limiter(per_minute=60, burst=3)

    # Called directly: capacity is a property of the arithmetic, not of the
    # transport, and this keeps the assertion on the one line that matters.
    allowed, tokens = limiter._check_local("bot:abc", 1_000_000.0)
    assert allowed and tokens == 2

    _, tokens_after_an_hour = limiter._check_local("bot:abc", 1_003_600.0)
    assert tokens_after_an_hour == 2, "tokens must be capped at burst capacity"


async def test_identities_have_independent_buckets(clock) -> None:
    """One customer's traffic must not throttle another's.

    A shared bucket would be a cross-tenant denial of service: the busiest
    website embedding a widget would silently rate-limit every other one.
    """
    limiter = _limiter(burst=2)
    for _ in range(2):
        await limiter.check("bot:first")

    assert not (await limiter.check("bot:first")).allowed
    assert (await limiter.check("bot:second")).allowed


async def test_enforce_raises_with_the_decision_attached(clock) -> None:
    """The exception carries `Retry-After`, so the handler does not recompute
    it and cannot report a different number than the one that was decided."""
    limiter = _limiter(burst=1)
    await limiter.enforce("bot:abc")

    with pytest.raises(RateLimitExceededError) as raised:
        await limiter.enforce("bot:abc")

    assert raised.value.decision.retry_after >= 1
    assert raised.value.decision.limit == limiter.per_minute


async def test_headers_are_present_on_allowed_and_refused_requests(clock) -> None:
    """A client that can see its remaining budget can slow itself down.

    `Retry-After` appears only on the refusal, per RFC 9110 — sending it on a
    successful response tells a client to wait when it need not.
    """
    limiter = _limiter(burst=1)

    allowed = (await limiter.check("bot:abc")).headers()
    refused = (await limiter.check("bot:abc")).headers()

    assert allowed["X-RateLimit-Limit"] == str(limiter.per_minute)
    assert "Retry-After" not in allowed
    assert "Retry-After" in refused


async def test_a_redis_outage_degrades_instead_of_refusing(clock, caplog) -> None:
    """Fail open, loudly.

    Every test in this file already exercises the degraded path, so this one
    asserts the two things that make it defensible: requests still succeed, and
    the degradation is visible in the logs rather than silent. Failing closed
    would mean a Redis restart takes every customer's chatbot offline to enforce
    a limit that protects a budget Postgres is already protecting.
    """
    limiter = _limiter(burst=2)

    with caplog.at_level("WARNING"):
        assert (await limiter.check("bot:abc")).allowed

    assert any("in-process buckets" in record.message for record in caplog.records)


async def test_a_burst_of_zero_is_treated_as_one(clock) -> None:
    """Misconfiguration must not silently refuse all traffic.

    A capacity of zero can never hold the token a request spends, so every
    request would 429 — a config typo that reads as a total outage. The floor
    turns it into the strictest working limit instead.
    """
    limiter = TokenBucketLimiter(per_minute=60, burst=0, redis_url=UNREACHABLE_REDIS)

    assert limiter.burst == 1
    assert (await limiter.check("bot:abc")).allowed

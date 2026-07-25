"""Per-key rate limiting: a token bucket, shared across API processes via Redis.

A widget key is public by design (see `security/bot_key.py`), so the thing that
stops a copied key from being used as a free LLM proxy is this file. The plan
cap in `core/metering.py` bounds the month; this bounds the minute, which is
what keeps one abusive visitor from burning a customer's entire monthly quota
before anyone notices.

**Token bucket, and why not the obvious alternative.**

    capacity C tokens, refilling at R tokens/second
    each request removes one token; no token, no request

The state is two numbers per key — how many tokens are left and when they were
last counted — and refill is computed lazily on read rather than by a timer:

    tokens = min(C, tokens + (now - last_seen) * R)

*Fixed window* ("20 requests per calendar minute") is simpler and wrong at the
edges: 20 requests at 11:59:59 and 20 more at 12:00:00 is 40 requests in one
second, all allowed. *Sliding window log* (store every timestamp) is exact but
stores one entry per request per key. The bucket is O(1) state, allows a
deliberate burst — which is what real usage looks like: a visitor asks, reads
for a minute, asks again — and still bounds the sustained rate at R.

**Why the arithmetic happens inside Redis.**

Read-modify-write from Python has the same lost-update race as the usage counter
(see `core/metering.py`): two API workers both read 1 token left, both decide
they may proceed, both write 0. A Lua script runs on the Redis server as a
single atomic unit, so the read, the refill, the spend and the write cannot
interleave. This is the standard reason rate limiters ship a Lua script rather
than three commands.

**Failure policy: open, loudly.**

If Redis is unreachable this degrades to a per-process bucket rather than
refusing traffic. That is a deliberate availability choice and it is only
defensible because the *money* is protected elsewhere: the monthly plan cap
lives in Postgres and is unaffected by a Redis outage. Failing closed here would
mean a Redis restart takes every customer's chatbot offline in order to enforce
a limit whose whole purpose is to protect a budget that is already protected.
The degraded state is logged at WARNING, once per interval, so it cannot pass
unnoticed.
"""

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache

from app.config import settings

logger = logging.getLogger(__name__)

# Redis key prefix. Namespaced so a shared Redis (Celery's broker lives there
# too) cannot collide with a queue name.
_KEY_PREFIX = "rl:"

# How long an idle bucket lives in Redis. Slightly longer than the time it takes
# a fully drained bucket to refill; after that the key is worthless (an absent
# key means "full bucket") and keeping it wastes memory.
_IDLE_TTL_SECONDS = 300

# Cap on the in-process fallback's memory. Ten thousand buckets is a few hundred
# KB; without a cap, an attacker rotating identities would grow the dict without
# bound, which is a memory exhaustion bug hiding inside a defence mechanism.
_LOCAL_MAX_BUCKETS = 10_000

# Budget for one Redis round trip. Deliberately tight: this call sits in front
# of every chat message, and a limiter that adds a second of latency has become
# a worse problem than the abuse it prevents. Redis on the same box answers in
# well under a millisecond.
_REDIS_TIMEOUT_SECONDS = 0.25

# How long to stop attempting Redis after a failure. Long enough that an outage
# does not charge every request a connect timeout; short enough that recovery is
# noticed within seconds rather than requiring a deploy.
_CIRCUIT_COOLDOWN_SECONDS = 30.0

# The whole limiter, atomically. Arguments are passed in rather than read from
# the server's clock so the script stays deterministic and so tests can inject
# time; the cost is that badly skewed clocks across API hosts would blur the
# window, which NTP already prevents.
_TOKEN_BUCKET_LUA = """
local key       = KEYS[1]
local rate      = tonumber(ARGV[1])   -- tokens per second
local capacity  = tonumber(ARGV[2])
local now       = tonumber(ARGV[3])   -- unix seconds, float
local ttl       = tonumber(ARGV[4])

local state  = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts     = tonumber(state[2])

if tokens == nil then
  tokens = capacity
  ts = now
end

-- Lazy refill: no timer, no background job. Elapsed time IS the refill.
local elapsed = math.max(0, now - ts)
tokens = math.min(capacity, tokens + elapsed * rate)

local allowed = 0
if tokens >= 1 then
  tokens = tokens - 1
  allowed = 1
end

redis.call('HSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, ttl)

-- Redis converts Lua numbers to integers, so tokens are returned in
-- milli-units to survive the round trip with their fractional part intact.
return {allowed, math.floor(tokens * 1000)}
"""


@dataclass(frozen=True)
class RateLimitDecision:
    """The outcome of one rate-limit check."""

    allowed: bool
    remaining: int
    retry_after: int
    limit: int

    def headers(self) -> dict[str, str]:
        """Standard rate-limit response headers.

        Returned on allowed requests too, not just on the 429: a client that
        can see it has two of twenty tokens left can slow down on its own,
        which is the whole point of telling it.
        """
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
        }
        if not self.allowed:
            headers["Retry-After"] = str(self.retry_after)
        return headers


class RateLimitExceededError(Exception):
    """Too many requests from one key -> 429.

    Carries the decision so the handler can attach `Retry-After` without
    recomputing it.
    """

    def __init__(self, decision: RateLimitDecision) -> None:
        super().__init__(
            f"rate limit of {decision.limit}/min exceeded; "
            f"retry in {decision.retry_after}s"
        )
        self.decision = decision


class TokenBucketLimiter:
    """Token bucket over Redis, with a per-process fallback.

    One instance per process, built by `get_rate_limiter()`. The Redis client is
    created lazily on first use because it binds to the running event loop —
    constructing it at import time would bind it to whichever loop happened to
    exist then, and every later call would fail with "attached to a different
    loop".
    """

    def __init__(
        self,
        *,
        per_minute: int | None = None,
        burst: int | None = None,
        redis_url: str | None = None,
    ) -> None:
        # `is not None` rather than `or`: an explicitly configured 0 must reach
        # the clamp below and become 1, not be silently swapped for the default.
        # `0 or default` is the same expression that has quietly disabled many a
        # feature flag.
        self.per_minute = (
            per_minute
            if per_minute is not None
            else settings.chat_rate_limit_per_minute
        )
        # Both floors are 1. A bucket with capacity 0 can never hold the token a
        # request spends, so every request would 429 — a config typo that
        # presents as a total outage rather than as a strict limit.
        self.per_minute = max(1, self.per_minute)
        self.burst = max(
            1, burst if burst is not None else settings.chat_rate_limit_burst
        )
        self.rate = self.per_minute / 60.0
        self._redis_url = redis_url or settings.redis_url
        self._redis = None
        self._script = None
        self._degraded = False
        # Circuit breaker: while set in the future, Redis is not even attempted.
        self._skip_redis_until = 0.0
        self._local: OrderedDict[str, tuple[float, float]] = OrderedDict()

    async def check(self, identity: str) -> RateLimitDecision:
        """Spend one token for `identity`, or report that there is none.

        Consuming rather than peeking: a check that does not spend would let a
        caller that ignores the answer proceed anyway, and this must be safe
        against a caller who does exactly that.
        """
        now = time.time()
        if now < self._skip_redis_until:
            # Circuit open: Redis failed recently, so do not spend the timeout
            # again on this request. Without this, an outage adds the connect
            # timeout to the latency of EVERY chat message for as long as it
            # lasts — a defence mechanism becoming the performance problem.
            allowed, tokens = self._check_local(identity, now)
        else:
            try:
                allowed, tokens = await self._check_redis(identity, now)
            except Exception as exc:  # noqa: BLE001 - any Redis failure degrades
                self._log_degraded(exc, now)
                allowed, tokens = self._check_local(identity, now)

        return RateLimitDecision(
            allowed=allowed,
            remaining=int(tokens),
            # Time until one whole token exists again. Rounded UP, because a
            # Retry-After the client trusts and retries at must not land a
            # moment before the bucket is actually ready.
            retry_after=max(1, int((1 - tokens) / self.rate) + 1) if not allowed else 0,
            limit=self.per_minute,
        )

    async def enforce(self, identity: str) -> RateLimitDecision:
        """`check`, raising `RateLimitExceededError` when refused.

        Two entry points because the widget endpoint wants an exception it
        cannot forget to handle, while a future admin view may want to read a
        decision without acting on it.
        """
        decision = await self.check(identity)
        if not decision.allowed:
            logger.info(
                "rate limit exceeded",
                extra={"identity": identity, "limit": self.limit_description},
            )
            raise RateLimitExceededError(decision)
        return decision

    @property
    def limit_description(self) -> str:
        return f"{self.per_minute}/min (burst {self.burst})"

    # -- redis --------------------------------------------------------------

    async def _check_redis(self, identity: str, now: float) -> tuple[bool, float]:
        script = await self._get_script()
        allowed, milli_tokens = await script(
            keys=[f"{_KEY_PREFIX}{identity}"],
            args=[self.rate, self.burst, now, _IDLE_TTL_SECONDS],
        )
        if self._degraded:
            logger.info("rate limiter reconnected to redis")
            self._degraded = False
        return bool(allowed), milli_tokens / 1000.0

    async def _get_script(self):
        """Register the Lua script once and reuse its SHA.

        `register_script` sends the body on first call and uses EVALSHA
        afterwards, so the script travels the network once per process rather
        than once per request.
        """
        if self._script is None:
            # Imported here, not at module scope: the API process should not
            # fail to import this module because redis is not installed in some
            # future slimmed-down deployment.
            from redis.asyncio import Redis
            from redis.backoff import NoBackoff
            from redis.retry import Retry

            self._redis = Redis.from_url(
                self._redis_url,
                decode_responses=True,
                # Every one of these fights the same failure: a rate limiter
                # that makes requests SLOWER when its backing store is sick.
                # redis-py retries connection errors with exponential backoff by
                # default, which turns a refused connection into seconds of
                # waiting on the hot path of a chat request.
                socket_connect_timeout=_REDIS_TIMEOUT_SECONDS,
                socket_timeout=_REDIS_TIMEOUT_SECONDS,
                retry=Retry(NoBackoff(), 0),
                retry_on_timeout=False,
            )
            self._script = self._redis.register_script(_TOKEN_BUCKET_LUA)
        return self._script

    def _log_degraded(self, exc: Exception, now: float) -> None:
        """Log the first failure loudly, then open the circuit and stay quiet.

        A Redis outage generates one of these per request; logging every one
        turns an incident into a log flood that hides the incident. Opening the
        circuit for a cooldown means the next attempt is a probe rather than
        another timeout charged to a user's message.
        """
        if not self._degraded:
            self._degraded = True
            logger.warning(
                "rate limiter falling back to in-process buckets",
                extra={"error": str(exc), "cooldown_s": _CIRCUIT_COOLDOWN_SECONDS},
            )
        self._skip_redis_until = now + _CIRCUIT_COOLDOWN_SECONDS
        # Drop the client so the next probe reconnects rather than reusing a
        # socket that is already known to be broken.
        self._script = None
        self._redis = None

    # -- in-process fallback -------------------------------------------------

    def _check_local(self, identity: str, now: float) -> tuple[bool, float]:
        """The same arithmetic, in this process only.

        Weaker on purpose and in a known direction: with N API workers the
        effective limit becomes N x the configured rate, because each holds its
        own bucket. That is the honest cost of staying available during a Redis
        outage, and it is still bounded — the alternative during an outage is no
        limit at all.
        """
        tokens, last_seen = self._local.get(identity, (float(self.burst), now))
        tokens = min(self.burst, tokens + max(0.0, now - last_seen) * self.rate)

        allowed = tokens >= 1
        if allowed:
            tokens -= 1

        self._local[identity] = (tokens, now)
        self._local.move_to_end(identity)
        while len(self._local) > _LOCAL_MAX_BUCKETS:
            # Evict the least recently used bucket. Evicting means that identity
            # gets a full bucket next time — acceptable, because the eviction
            # only happens under a flood of distinct identities, which is
            # exactly when this fallback is already the wrong tool.
            self._local.popitem(last=False)

        return allowed, tokens

    async def close(self) -> None:
        """Release the Redis connection pool. Called from FastAPI's lifespan."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            self._script = None


@lru_cache
def get_rate_limiter() -> TokenBucketLimiter:
    """The process-wide limiter.

    Cached for the same reason the provider factory caches: one instance holds
    the connection pool and the fallback state, and a second one would defeat
    the fallback by giving each request a fresh, full bucket.
    """
    return TokenBucketLimiter()


def reset_rate_limiter() -> None:
    """Drop the cached limiter. For tests that change the configured rate."""
    get_rate_limiter.cache_clear()

"""Widget bot-key authentication: `pk_...` -> one organization, one chatbot.

The second trust boundary in the system. `security/clerk.py` authenticates a
*person* using the dashboard; this authenticates *a website* embedding a bot.
They share one rule, which is the rule that makes the product safe: the tenant
scope is resolved server-side from the credential, never read from the request.

**The credential is public, and the design accounts for that.**

The key ships inside a `<script>` tag on the customer's own site (PLAN §5), so
anyone who views source can read it. Treating it as a secret would be a lie we
tell ourselves; instead:

- It authorizes exactly one thing — chatting with one bot. It cannot list
  documents, read conversations, change settings, or reach another tenant. The
  dashboard's API is a different credential entirely (a Clerk session token).
- Abuse is bounded by the per-key rate limit (`security/rate_limit.py`) and by
  the org's monthly plan cap (`core/metering.py`), not by the key's secrecy.
- Revocation is rotation: `chatbots.public_key` is read from the database on
  every request, with no cache, so a rotated key stops working on the very next
  message rather than at the end of some TTL.

**Why the header is `X-Bot-Key` rather than `Authorization: Bearer`.**

Two credentials sharing one header would mean a parser that inspects the value
and decides which kind it is holding. "Guess the credential type" is a
well-worn source of authentication-confusion bugs, and the fix costs one header
name. A request carrying both headers is not ambiguous either: each route
depends on exactly one of them and ignores the other, because the route is what
decides which credential is acceptable — not the request.

**Why one query returning both rows.**

Chat needs the chatbot (system prompt, dimension) and the organization (plan,
for the cap) on every single message. Resolving the key and then loading the org
would be two round trips per message on the hottest path in the product, so the
lookup joins them. The join is on the foreign key and both sides are indexed.
"""

import hashlib
import logging
import uuid
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.db.tables import Chatbot, ChatbotStatus, Organization

logger = logging.getLogger(__name__)

# The dedicated header carrying a widget key. Lowercase comparison happens in
# HTTP-land; FastAPI's `Header` parameter handles the casing for us.
BOT_KEY_HEADER = "X-Bot-Key"

# How many hex characters of the key's SHA-256 identify it for rate limiting and
# logging. 16 hex chars is 64 bits — far past any realistic collision, and short
# enough to read in a log line.
_FINGERPRINT_CHARS = 16


class BotKeyError(Exception):
    """Base for widget-credential failures.

    Typed rather than raising HTTPException, for the same reason
    `security/clerk.py` does it: this module must be testable without a request
    object, and the mapping to status codes belongs to the web layer.
    """


class MissingBotKeyError(BotKeyError):
    """No key was presented -> 401.

    Distinguished from `UnknownBotKeyError` in our logs but not in the response:
    a wave of missing keys is a customer's embed snippet installed wrongly,
    while a wave of unknown ones is somebody probing. The caller is told the
    same thing either way.
    """


class UnknownBotKeyError(BotKeyError):
    """The key does not match any chatbot -> 401."""


class BotDisabledError(BotKeyError):
    """The key is valid, the bot is switched off -> 403.

    A separate status because it is a different situation: the credential is
    genuine and the site owner disabled the bot on purpose. Retrying with a new
    key would not help, and a 401 would send a widget into a reload loop.
    """


@dataclass(frozen=True)
class BotContext:
    """Everything a widget chat request is allowed to know about its scope.

    Holds the ORM rows because the chat handler needs them immediately —
    `chatbot.system_prompt` and `organization.plan` are both required before the
    first token can be generated. Frozen so that nothing downstream can rebind
    the scope after it has been resolved.
    """

    chatbot: Chatbot
    organization: Organization
    fingerprint: str

    @property
    def org_id(self) -> uuid.UUID:
        return self.organization.id

    @property
    def chatbot_id(self) -> uuid.UUID:
        return self.chatbot.id

    @property
    def rate_limit_key(self) -> str:
        """Stable per-key identity for the rate limiter.

        A fingerprint rather than the key itself: Redis keys show up in
        `MONITOR`, in slow-log output, and in any keyspace dump, and none of
        those are places a working credential should be sitting. The hash is
        just as stable and carries nothing replayable.
        """
        return f"bot:{self.fingerprint}"


def fingerprint(key: str) -> str:
    """Short, non-reversible identifier for a key.

    Plain SHA-256, not a password hash. The threat here is a credential
    appearing in logs, not an offline cracking attempt against a stolen
    database — and the key is 256 bits of `secrets` output, so brute-forcing the
    preimage is not a thing that happens. A slow KDF would add latency to every
    single widget message to defend against an attack the key's entropy already
    prevents.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:_FINGERPRINT_CHARS]


def resolve_bot_key(db: Session, key: str | None) -> BotContext:
    """Resolve a widget key to its organization and chatbot.

    Takes ONLY the key. There is deliberately no org_id or chatbot_id parameter:
    the entire tenant scope of a widget request is derived here, from a value the
    caller cannot forge into pointing somewhere else, and a scope argument would
    be a scope a caller could supply.

    Exact equality, never a prefix or LIKE match — a pattern match written for
    some future admin search would quietly let one key resolve to whichever bot
    happened to sort first.
    """
    if not key or not key.strip():
        raise MissingBotKeyError("missing bot key")

    row = db.execute(
        select(Chatbot, Organization)
        .join(Organization, Chatbot.org_id == Organization.id)
        .where(Chatbot.public_key == key.strip())
    ).one_or_none()

    if row is None:
        # INFO, not WARNING: the common cause is a stale embed snippet after a
        # key rotation, not an attack. The fingerprint is logged so a flood can
        # be correlated without the credential itself ever being written down.
        logger.info("unknown bot key", extra={"bot_key_fp": fingerprint(key.strip())})
        raise UnknownBotKeyError("invalid bot key")

    chatbot, organization = row

    if chatbot.status is ChatbotStatus.DISABLED:
        logger.info(
            "bot key for a disabled chatbot",
            extra={"org_id": str(organization.id), "chatbot_id": str(chatbot.id)},
        )
        raise BotDisabledError("this chatbot is disabled")

    return BotContext(
        chatbot=chatbot,
        organization=organization,
        fingerprint=fingerprint(key.strip()),
    )


# --------------------------------------------------------------------------
# FastAPI wiring
# --------------------------------------------------------------------------


def require_bot(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    x_bot_key: Annotated[str | None, Header()] = None,
) -> BotContext:
    """Dependency for widget routes: authenticate the key, yield the scope.

    Mirrors `security/clerk.require_org` on purpose — same shape, same
    router-level attachment, same population of `request.state` so the logging
    middleware can tag the access log with the tenant. Two auth mechanisms that
    look alike are two mechanisms a reviewer can check at the same speed.
    """
    try:
        context = resolve_bot_key(db, x_bot_key)
    except MissingBotKeyError as exc:
        raise HTTPException(
            status_code=401,
            detail=f"missing {BOT_KEY_HEADER} header",
            headers={"WWW-Authenticate": BOT_KEY_HEADER},
        ) from exc
    except UnknownBotKeyError as exc:
        raise HTTPException(status_code=401, detail="invalid bot key") from exc
    except BotDisabledError as exc:
        # 403: authenticated, deliberately not authorized. See the class
        # docstring on why this is not a 401.
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    request.state.org_id = str(context.org_id)
    request.state.chatbot_id = str(context.chatbot_id)
    return context

"""Widget bot-key authentication.

The third security-critical file written before its implementation (Hard Rule
#4). `test_isolation.py` protects the filter, `test_limits.py` protects the
counter, and this one protects the *entry point those two defend* — the only
credential in the system that arrives from a browser we do not control, on a
website we have never seen.

**What a bot key is, and what it deliberately is not.**

It ships inside a `<script>` tag on the customer's own site, so it is readable
by anyone who views source. That is by design (PLAN §5), and it means secrecy is
not the security property. The property is *resolution*: a key names exactly one
(organization, chatbot) pair, server-side, and nothing a caller sends can point
it at a different one. Everything downstream — the Qdrant filter, the plan cap,
the rate limit — is derived from what this module returns, so a bug here
bypasses all three at once.

Because the key is public, abuse is bounded by the other two mechanisms rather
than by the credential: a stranger copying a key can chat with a bot they could
already chat with on the customer's website, at a limited rate, against the
customer's monthly quota. They can never read a document, list conversations,
or reach another tenant.

**The attack this file is really about.** The chat request body contains a
question and, optionally, a conversation id. It must never contain a chatbot id
that is honoured — that would let anyone holding one public key chat with every
bot in the system. `test_resolution_takes_only_the_key` asserts on the signature
so that the argument does not exist to be trusted.
"""

import inspect
import uuid

import pytest
from sqlalchemy.orm import Session

from app.db.tables import (
    PUBLIC_KEY_PREFIX,
    Chatbot,
    ChatbotStatus,
    Organization,
    generate_public_key,
)
from app.security.bot_key import (
    BOT_KEY_HEADER,
    BotDisabledError,
    BotKeyError,
    MissingBotKeyError,
    UnknownBotKeyError,
    resolve_bot_key,
)


def _org(db: Session, name: str = "Acme") -> Organization:
    organization = Organization(name=name, clerk_org_id=f"org_{uuid.uuid4().hex[:12]}")
    db.add(organization)
    db.commit()
    return organization


def _bot(
    db: Session,
    organization: Organization,
    *,
    name: str = "Support",
    status: ChatbotStatus = ChatbotStatus.READY,
) -> Chatbot:
    chatbot = Chatbot(
        org_id=organization.id,
        name=name,
        embedding_model="stub",
        dimension=16,
        status=status,
    )
    db.add(chatbot)
    db.commit()
    return chatbot


# --------------------------------------------------------------------------
# 1. The key generator
# --------------------------------------------------------------------------


def test_generated_keys_are_prefixed_and_unique() -> None:
    """Prefix for recognisability, `secrets` for unguessability.

    The prefix is not decoration: a leaked credential is far easier to spot in a
    log or a public repo when its first characters say what it is, and secret
    scanners key off exactly this pattern.
    """
    keys = {generate_public_key() for _ in range(500)}

    assert len(keys) == 500, "a collision in 500 keys means the generator is not random"
    assert all(key.startswith(PUBLIC_KEY_PREFIX) for key in keys)
    # 32 random bytes -> 43 url-safe characters, plus the prefix.
    assert all(len(key) >= 40 for key in keys)


# --------------------------------------------------------------------------
# 2. Resolution
# --------------------------------------------------------------------------


def test_a_valid_key_resolves_to_its_own_bot_and_org(db: Session) -> None:
    """The happy path, and the shape everything downstream depends on."""
    organization = _org(db)
    chatbot = _bot(db, organization)

    context = resolve_bot_key(db, chatbot.public_key)

    assert context.chatbot.id == chatbot.id
    assert context.organization.id == organization.id
    assert context.org_id == organization.id
    assert context.chatbot_id == chatbot.id


def test_resolution_loads_org_and_bot_in_one_query(db: Session) -> None:
    """Both rows come back together, not in two round trips.

    This runs on every widget message, so an N+1 here is a per-message cost
    paid by every customer's every visitor. Asserted by emitting SQL through a
    counting listener rather than by trusting the implementation to have joined.
    """
    from sqlalchemy import event

    organization = _org(db)
    chatbot = _bot(db, organization)
    db.expunge_all()  # force real queries rather than identity-map hits

    statements: list[str] = []

    def record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(db.get_bind(), "before_cursor_execute", record)
    try:
        context = resolve_bot_key(db, chatbot.public_key)
        assert context.organization.name == organization.name  # touch both rows
        assert context.chatbot.name == chatbot.name
    finally:
        event.remove(db.get_bind(), "before_cursor_execute", record)

    selects = [s for s in statements if s.lstrip().upper().startswith("SELECT")]
    assert len(selects) == 1, f"expected one SELECT, got {len(selects)}: {selects}"


def test_two_orgs_keys_resolve_to_their_own_tenant(db: Session) -> None:
    """The cross-tenant case, stated explicitly.

    Same reasoning as `test_isolation.py`'s symmetric check: an implementation
    that captured the wrong row could pass a one-tenant test.
    """
    first_org = _org(db, name="First")
    second_org = _org(db, name="Second")
    first_bot = _bot(db, first_org)
    second_bot = _bot(db, second_org)

    assert resolve_bot_key(db, first_bot.public_key).org_id == first_org.id
    assert resolve_bot_key(db, second_bot.public_key).org_id == second_org.id


# --------------------------------------------------------------------------
# 3. Rejection
# --------------------------------------------------------------------------


def test_an_unknown_key_is_rejected(db: Session) -> None:
    _bot(db, _org(db))

    with pytest.raises(UnknownBotKeyError):
        resolve_bot_key(db, generate_public_key())


@pytest.mark.parametrize("key", ["", "   ", None])
def test_a_missing_key_is_a_distinct_error(db: Session, key: str | None) -> None:
    """ "No credential" and "wrong credential" are different failures.

    They map to the same 401 for the caller — telling a stranger which half of
    their forgery to fix is free help — but the distinction matters in our logs,
    where a flood of unknown keys is an attack and a flood of missing ones is a
    broken embed snippet on a customer's site.
    """
    with pytest.raises(MissingBotKeyError):
        resolve_bot_key(db, key)


def test_a_key_prefix_alone_does_not_resolve(db: Session) -> None:
    """Lookup is exact equality, never a prefix or LIKE match.

    Worth an explicit test: a `LIKE 'pk_%'` written for a "search keys" admin
    feature would make the first bot in the table answer for everyone.
    """
    _bot(db, _org(db))

    with pytest.raises(BotKeyError):
        resolve_bot_key(db, PUBLIC_KEY_PREFIX)


def test_a_session_token_is_not_a_bot_key(db: Session) -> None:
    """The two credentials do not interchange.

    A JWT presented as a bot key must fail, and a bot key presented as a bearer
    token must fail (the Clerk path rejects it as an unverifiable signature).
    Credential confusion between two schemes that each authorize different
    things is a real class of bug; this is the cheap half of preventing it.
    """
    _bot(db, _org(db))

    with pytest.raises(BotKeyError):
        resolve_bot_key(db, "eyJhbGciOiJSUzI1NiIsImtpZCI6ImFiYyJ9.e30.sig")


def test_a_disabled_bot_refuses_its_own_key(db: Session) -> None:
    """DISABLED is how a customer switches a bot off without destroying it.

    If the key kept working, "disable" would mean nothing to the only traffic
    that actually uses it — every visitor on the customer's website.
    """
    organization = _org(db)
    chatbot = _bot(db, organization, status=ChatbotStatus.DISABLED)

    with pytest.raises(BotDisabledError):
        resolve_bot_key(db, chatbot.public_key)


def test_a_draft_bot_still_answers(db: Session) -> None:
    """DRAFT means "no documents ingested yet", not "off".

    Such a bot answers every question with the refusal sentence, which is the
    correct behaviour and a much better first experience than an auth error the
    site owner cannot interpret.
    """
    organization = _org(db)
    chatbot = _bot(db, organization, status=ChatbotStatus.DRAFT)

    assert resolve_bot_key(db, chatbot.public_key).chatbot_id == chatbot.id


def test_a_revoked_key_stops_working_immediately(db: Session) -> None:
    """Rotating `public_key` is the revocation mechanism.

    There is no key cache anywhere: resolution reads the row on every request,
    so a rotated key is dead on the next message rather than at the end of some
    TTL. That is worth the per-request lookup — the alternative is explaining to
    a customer why their leaked key still works for another five minutes.
    """
    organization = _org(db)
    chatbot = _bot(db, organization)
    old_key = chatbot.public_key

    chatbot.public_key = generate_public_key()
    db.commit()

    with pytest.raises(UnknownBotKeyError):
        resolve_bot_key(db, old_key)
    assert resolve_bot_key(db, chatbot.public_key).chatbot_id == chatbot.id


# --------------------------------------------------------------------------
# 4. Identity for rate limiting, without leaking the credential
# --------------------------------------------------------------------------


def test_rate_limit_identity_is_derived_and_not_the_raw_key(db: Session) -> None:
    """The rate limiter needs a stable per-key identity; Redis and our logs
    should never hold the credential itself.

    A raw key used as a Redis key ends up in `MONITOR` output, in slow-query
    logs, and in any dump of the keyspace. A hash is just as stable and carries
    nothing an attacker could replay.
    """
    organization = _org(db)
    chatbot = _bot(db, organization)

    context = resolve_bot_key(db, chatbot.public_key)
    again = resolve_bot_key(db, chatbot.public_key)

    assert context.rate_limit_key == again.rate_limit_key
    assert chatbot.public_key not in context.rate_limit_key
    assert context.rate_limit_key  # non-empty


def test_different_keys_get_different_rate_limit_identities(db: Session) -> None:
    """Two bots must not share a bucket — one busy customer would throttle
    another, which is a cross-tenant denial of service."""
    organization = _org(db)
    first = _bot(db, organization, name="First")
    second = _bot(db, organization, name="Second")

    assert (
        resolve_bot_key(db, first.public_key).rate_limit_key
        != resolve_bot_key(db, second.public_key).rate_limit_key
    )


# --------------------------------------------------------------------------
# 5. Structural — the forgeable argument must not exist
# --------------------------------------------------------------------------


def test_resolution_takes_only_the_key(db: Session) -> None:
    """`resolve_bot_key(db, key)` — no org_id, no chatbot_id, no scope.

    The same structural argument as `test_isolation.py`'s signature test: a
    behavioural test proves the current code does not trust a client-supplied
    bot id, while this one fails the moment such a parameter is added, which is
    the change that would make trusting it possible.
    """
    parameters = list(inspect.signature(resolve_bot_key).parameters)

    assert parameters == ["db", "key"], (
        "resolve_bot_key must derive the entire tenant scope from the key; "
        f"got parameters {parameters}"
    )


def test_the_header_name_is_dedicated(db: Session) -> None:
    """Bot keys travel in their own header, not in `Authorization`.

    Sharing `Authorization` with Clerk session tokens would mean one parser
    deciding which credential it is holding — and "guess the credential type"
    is how authentication-confusion bugs happen.
    """
    assert BOT_KEY_HEADER.lower() == "x-bot-key"

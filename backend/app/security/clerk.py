"""Clerk session-token verification and tenant resolution.

The trust boundary. Everything upstream of this module is attacker-controlled
bytes; everything downstream may assume `org_id` is a fact. PLAN's rule — "the
backend resolves org_id SERVER-SIDE (always)" — is implemented here and nowhere
else, which is the point: one place to audit, one place to get right.

**The flow.**

    Authorization: Bearer <jwt>
        -> decode header, find `kid`
        -> fetch/lookup that key in Clerk's JWKS
        -> verify RS256 signature, exp, nbf, iss
        -> read `org_id` claim (Clerk's, a string like "org_2abc...")
        -> look up OUR Organization row by clerk_org_id
        -> hand back a Principal carrying our UUID org_id

Note the last step. The token's org id is Clerk's identifier, not ours, and the
two are deliberately not interchangeable: a token proves *who Clerk says you
are*, and the database decides *whether that maps to a tenant here*. An org
deleted on our side stops working immediately even though Clerk still issues
perfectly valid tokens for it.

**What is verified, and why each matters.**

- *Signature*, with `algorithms=["RS256"]` passed explicitly. This single
  argument defeats the classic JWT attack: an attacker rewrites the header to
  `alg: none` (no signature required) or to `alg: HS256` and signs with the RSA
  *public* key, which is public by definition — a library that reads the
  algorithm from the token itself will happily verify both. Never let the token
  choose its own verification algorithm.
- *`exp`*, so a stolen token expires. Clerk's are short-lived (~60s) precisely
  because theft is assumed.
- *`nbf`/`iat`*, with a small leeway for clock skew between Clerk and us.
- *`iss`*, when configured. This one is easy to skip and important: anyone can
  create their own free Clerk instance, and tokens from it are validly signed
  against *its* JWKS. Without an issuer check, the only thing stopping a
  stranger's token from authenticating here is that we happen to fetch keys
  from a different URL — true today, and not a guarantee.

**What is NOT verified: `aud`.** Clerk does not populate an audience claim on
standard session tokens, so requiring one would reject every real token. The
issuer check covers the same ground here.

**Deliberate deviation from PLAN §6.** PLAN says "JWT verified in FastAPI
middleware -> request.state.org_id". This is a dependency (`require_org`)
applied at ROUTER level instead, for two reasons. First, middleware runs on
every path — `/health`, `/docs`, future webhook endpoints that authenticate by
Stripe signature rather than by JWT — so it needs a growing list of exempt paths,
and a path-prefix exemption list is a well-known source of auth bypasses.
Second, a dependency composes with FastAPI's DI, so the DB session used to
resolve the org is the same one the handler uses.

The honest cost: with middleware, forgetting to configure a route leaves it
protected; with dependencies, forgetting leaves it PUBLIC. That is why the
dependency is attached to `APIRouter(dependencies=[...])` rather than to
individual routes — the unit you can forget is a whole router, not an endpoint,
and `request.state` is still populated so logging middleware sees the org.
"""

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Annotated

import httpx
import jwt
from fastapi import Depends, Header, HTTPException, Request
from jwt import PyJWKClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.base import get_db
from app.db.tables import Organization, User, UserRole

logger = logging.getLogger(__name__)

# The only algorithm we accept. A list of exactly one, on purpose — see module
# docstring on algorithm confusion.
_ALGORITHMS = ["RS256"]

# Tolerance for clock drift between Clerk's servers and ours, in seconds.
# Without it, a machine whose clock is two seconds fast rejects freshly issued
# tokens with "token not yet valid", which is a maddening intermittent failure.
# Kept small: leeway is time during which an expired token still works.
_CLOCK_SKEW_LEEWAY = 30


class AuthError(Exception):
    """Base for authentication failures.

    Typed rather than raising HTTPException directly so this module stays
    independent of the web layer and can be unit-tested without a request. The
    router-level translation to 401/403 lives in `_to_http` below.
    """


class InvalidTokenError(AuthError):
    """Missing, malformed, expired, or badly signed token -> 401."""


class UnknownOrganizationError(AuthError):
    """Token is valid but names an org we have no row for -> 403.

    Distinct from InvalidTokenError because the remedy is different: the client
    should not retry with a fresh token, because the token was never the
    problem.
    """


class MissingOrganizationError(AuthError):
    """Valid personal token with no active organization -> 403.

    Clerk issues session tokens for a user with no org selected. Everything in
    this product is org-scoped, so such a token authenticates a person but
    authorizes nothing, and the dashboard must send them to org creation.
    """


class AuthNotConfiguredError(AuthError):
    """CLERK_JWKS_URL is unset -> 500, not 401.

    A misconfigured server must never look like a rejected credential. Returning
    401 here would send developers hunting for a bad token when the actual fault
    is an empty .env, and — worse — would make an auth outage indistinguishable
    from normal traffic in the logs.
    """


@dataclass(frozen=True)
class Principal:
    """The verified identity of a request. The output of this module.

    Frozen, and carrying OUR `org_id` (a UUID from the organizations table)
    rather than Clerk's string id. Handlers receive this and never see the raw
    token, so there is no route by which a handler can accidentally trust an
    unverified claim.
    """

    org_id: uuid.UUID
    clerk_org_id: str
    clerk_user_id: str
    role: UserRole
    user_id: uuid.UUID | None = None

    @property
    def is_owner(self) -> bool:
        """Owners may delete bots and change billing; members may not."""
        return self.role is UserRole.OWNER


# --------------------------------------------------------------------------
# JWKS
# --------------------------------------------------------------------------


class _JWKSCache:
    """Holds one `PyJWKClient`, rebuilt when its cache window expires.

    PyJWKClient already caches keys in memory and refetches on an unknown `kid`,
    which is exactly the behaviour key rotation needs: a new signing key appears
    in a token before we have ever seen it, and the client goes and gets it
    rather than rejecting the token.

    The outer TTL exists for the opposite case — a key that was REVOKED. An
    unknown kid triggers a refetch, but a known-and-since-revoked kid does not,
    so without a ceiling a compromised key would stay accepted for the life of
    the process. `clerk_jwks_cache_seconds` bounds that window.

    Not thread-safe in the strict sense: two threads can rebuild concurrently
    and one result is discarded. That is a wasted HTTP call, not a correctness
    problem, and a lock on the hot path of every request would cost more than
    the occasional duplicate fetch.
    """

    def __init__(self) -> None:
        self._client: PyJWKClient | None = None
        self._built_at: float = 0.0
        self._url: str | None = None

    def get(self) -> PyJWKClient:
        url = settings.clerk_jwks_url
        if not url:
            raise AuthNotConfiguredError(
                "CLERK_JWKS_URL is not set; the API cannot verify session tokens"
            )

        expired = time.monotonic() - self._built_at > settings.clerk_jwks_cache_seconds
        # Rebuilding when the URL changes matters for tests, which point this at
        # a fixture server between cases.
        if self._client is None or expired or self._url != url:
            logger.info("building JWKS client", extra={"jwks_url": url})
            self._client = PyJWKClient(url, cache_keys=True)
            self._built_at = time.monotonic()
            self._url = url
        return self._client

    def reset(self) -> None:
        """Drop the cached client. For tests, and for a future admin endpoint
        that forces a key refresh after a rotation incident."""
        self._client = None
        self._built_at = 0.0
        self._url = None


_jwks = _JWKSCache()


def reset_jwks_cache() -> None:
    _jwks.reset()


# --------------------------------------------------------------------------
# Token verification
# --------------------------------------------------------------------------


def verify_token(token: str) -> dict:
    """Verify a Clerk session token and return its claims.

    Pure with respect to our database — it proves the token is authentic and
    says nothing about whether the org exists here. Split that way so the
    cryptography can be tested without a database, and so the "valid token,
    unknown tenant" case produces a distinguishable error.

    Every failure below becomes the SAME `InvalidTokenError` message on purpose.
    Distinguishing "expired" from "bad signature" from "wrong issuer" in the
    response body tells an attacker which part of their forgery to fix; the
    detail goes to the log, where we can see it and they cannot.
    """
    if not token:
        raise InvalidTokenError("missing bearer token")

    try:
        signing_key = _jwks.get().get_signing_key_from_jwt(token)
    except AuthNotConfiguredError:
        raise
    except (httpx.HTTPError, jwt.PyJWKClientError) as exc:
        # Reaching Clerk's JWKS endpoint failed, or the token names a key that
        # does not exist. The first is our outage and the second is a forgery,
        # and PyJWT does not cleanly separate them — logged either way.
        logger.warning("could not resolve JWKS signing key", extra={"error": str(exc)})
        raise InvalidTokenError("token signature could not be verified") from exc

    try:
        return jwt.decode(
            token,
            signing_key.key,
            # The single most important argument in this module. Never derive
            # this from the token's own header.
            algorithms=_ALGORITHMS,
            issuer=settings.clerk_issuer,
            leeway=_CLOCK_SKEW_LEEWAY,
            options={
                "require": ["exp", "iat", "sub"],
                "verify_exp": True,
                "verify_iat": True,
                "verify_nbf": True,
                "verify_signature": True,
                # Clerk does not set `aud` on session tokens; requiring one
                # would reject every genuine token. See module docstring.
                "verify_aud": False,
                # Only meaningful when clerk_issuer is configured. Made explicit
                # so an unset issuer is a visible gap rather than a silent one.
                "verify_iss": settings.clerk_issuer is not None,
            },
        )
    except jwt.PyJWTError as exc:
        logger.info(
            "rejected session token",
            extra={"reason": type(exc).__name__, "detail": str(exc)},
        )
        raise InvalidTokenError("invalid or expired session token") from exc


def extract_clerk_org_id(claims: dict) -> str:
    """Pull the active organization id out of verified claims.

    Clerk has shipped two shapes over time: a flat `org_id`, and a nested
    `o: {id, rol}` in newer versions of the default session-token template.
    Both are checked because which one you get depends on the instance's
    template, and a deployment silently authenticating with the wrong shape
    would fail closed in a way that looks like "my org disappeared".
    """
    organization = claims.get("o")
    if isinstance(organization, dict) and organization.get("id"):
        return str(organization["id"])

    if claims.get("org_id"):
        return str(claims["org_id"])

    raise MissingOrganizationError(
        "session token has no active organization; the user must create or "
        "select an organization before using the API"
    )


def extract_role(claims: dict) -> UserRole:
    """Best-effort role from the token, defaulting to the LEAST privilege.

    Clerk's role strings ("org:admin", "admin", "org:member") are its own
    vocabulary, not ours. Anything unrecognised becomes MEMBER: defaulting to
    OWNER on an unfamiliar string would mean a Clerk-side rename silently
    promoting every user in the system.

    The token's role is a hint. The database row is authoritative — see
    `_resolve_principal`, which prefers the stored role whenever a User row
    exists.
    """
    organization = claims.get("o")
    raw = ""
    if isinstance(organization, dict):
        raw = str(organization.get("rol") or "")
    raw = raw or str(claims.get("org_role") or "")

    return (
        UserRole.OWNER if raw.split(":")[-1] in {"admin", "owner"} else UserRole.MEMBER
    )


# --------------------------------------------------------------------------
# Tenant resolution
# --------------------------------------------------------------------------


def resolve_principal(claims: dict, db: Session) -> Principal:
    """Map verified claims onto a row in OUR database.

    A valid token is necessary but not sufficient: the org must exist here. That
    separation is what makes off-boarding work — delete the organization row and
    every outstanding token for it stops authorizing, without waiting for Clerk.
    """
    clerk_org_id = extract_clerk_org_id(claims)
    clerk_user_id = str(claims["sub"])

    organization = db.scalar(
        select(Organization).where(Organization.clerk_org_id == clerk_org_id)
    )
    if organization is None:
        # INFO, not WARNING: the common cause is a brand-new org whose webhook
        # has not landed yet, not an attack.
        logger.info(
            "token names an unknown organization",
            extra={"clerk_org_id": clerk_org_id, "clerk_user_id": clerk_user_id},
        )
        raise UnknownOrganizationError("organization not found")

    user = db.scalar(select(User).where(User.clerk_user_id == clerk_user_id))

    # The stored role wins over the token's when we have a row. The token is
    # authentic but reflects Clerk's view; our row reflects decisions made here.
    role = user.role if user is not None else extract_role(claims)

    # A user row belonging to a DIFFERENT org than the token names is not a
    # normal state — it means our copy is stale or the ids were crossed. Refuse
    # rather than guess which one is right.
    if user is not None and user.org_id != organization.id:
        logger.warning(
            "user/org mismatch between token and database",
            extra={
                "clerk_user_id": clerk_user_id,
                "token_org": str(organization.id),
                "row_org": str(user.org_id),
            },
        )
        raise UnknownOrganizationError("organization not found")

    return Principal(
        org_id=organization.id,
        clerk_org_id=clerk_org_id,
        clerk_user_id=clerk_user_id,
        role=role,
        user_id=user.id if user is not None else None,
    )


# --------------------------------------------------------------------------
# FastAPI wiring
# --------------------------------------------------------------------------


def _bearer_token(authorization: str | None) -> str:
    """Extract the token from an `Authorization: Bearer <token>` header.

    The scheme comparison is case-insensitive because the RFC says it is, and
    some HTTP clients send "bearer".
    """
    if not authorization:
        raise InvalidTokenError("missing Authorization header")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise InvalidTokenError("Authorization header must be 'Bearer <token>'")
    return token.strip()


def require_org(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    authorization: Annotated[str | None, Header()] = None,
) -> Principal:
    """FastAPI dependency: authenticate, and yield the caller's Principal.

    Attach at ROUTER level:

        router = APIRouter(dependencies=[Depends(require_org)])

    so that adding an endpoint to an already-protected router cannot forget it.

    `request.state` is populated as a side effect so the logging middleware can
    tag the response log line with org_id (PLAN §7). Middleware wraps the whole
    call, so it sees this value on the way out, after dependencies have run —
    which is also why an unauthenticated request logs no org_id at all.

    Handlers should take `Principal` as a parameter rather than reading
    `request.state`: the parameter is typed and cannot be absent, the state bag
    is neither.
    """
    principal = _resolve(request, authorization, db)
    request.state.org_id = str(principal.org_id)
    request.state.clerk_user_id = principal.clerk_user_id
    return principal


def _resolve(request: Request, authorization: str | None, db: Session) -> Principal:
    try:
        claims = verify_token(_bearer_token(authorization))
        return resolve_principal(claims, db)
    except AuthNotConfiguredError as exc:
        # 500: our fault, not the caller's. See the class docstring.
        logger.error("auth is not configured", extra={"error": str(exc)})
        raise HTTPException(
            status_code=500, detail="authentication unavailable"
        ) from exc
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=401,
            detail=str(exc),
            # Required by RFC 6750 for a 401 on a bearer-token resource, and it
            # is what tells a browser client to refresh its session rather than
            # showing the user an error.
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except (MissingOrganizationError, UnknownOrganizationError) as exc:
        # 403, not 401: the caller IS authenticated. Retrying with a new token
        # will not help, and a 401 would send the client into a refresh loop.
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def require_owner(
    principal: Annotated[Principal, Depends(require_org)],
) -> Principal:
    """Dependency for destructive or billing-related endpoints.

    Layered on top of `require_org` rather than replacing it, so an endpoint
    cannot accidentally get the owner check without the authentication that must
    precede it.
    """
    if not principal.is_owner:
        logger.info(
            "denied owner-only action",
            extra={
                "org_id": str(principal.org_id),
                "clerk_user_id": principal.clerk_user_id,
            },
        )
        raise HTTPException(
            status_code=403, detail="this action requires the organization owner role"
        )
    return principal

"""FastAPI application entrypoint.

Run locally:  uvicorn app.main:app --reload   (from backend/)
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import playground_router, widget_router
from app.api.chatbots import router as chatbots_router
from app.api.conversations import router as conversations_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.config import settings
from app.core.logging import configure_logging
from app.core.middleware import RequestContextMiddleware
from app.core.vectorstore import VectorStore
from app.providers.factory import get_embedder
from app.security.rate_limit import get_rate_limiter

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown: build the long-lived clients once, close them once.

    **The Qdrant client is created here, not per request.** It owns an HTTP
    connection pool; building one per chat message would open and discard a pool
    per message, which under load runs a machine out of sockets. It is created
    eagerly but does not *connect* eagerly — `AsyncQdrantClient` performs no I/O
    until the first query — so a Qdrant that is down delays a 503 on the first
    search instead of preventing the API from starting. That distinction
    matters: `/health` has to answer even when a dependency is sick, or a load
    balancer removes the last healthy instance during an outage in something
    else entirely.

    `ensure_collection()` is deliberately NOT called here. It writes, and the
    web process should not create infrastructure — the ingestion worker already
    does that before its first upsert, which is the only moment the collection
    genuinely has to exist.
    """
    configure_logging(
        level=settings.log_level,
        # Human-readable in dev, JSON in production where logs get parsed.
        json_output=settings.is_production,
    )
    logger.info("app starting", extra={"env": settings.app_env})

    # `get_embedder()` is cached by the factory, so this is also where the model
    # warms up — paying the load cost at boot rather than inside the first
    # customer's chat request.
    embedder = get_embedder()
    app.state.vector_store = VectorStore(dimension=embedder.dimension)

    try:
        yield
    finally:
        # Both closes are best-effort. A shutdown that raises leaves sockets in
        # TIME_WAIT and, worse, buries whatever actually caused the shutdown.
        try:
            await app.state.vector_store.close()
        except Exception:  # noqa: BLE001
            logger.warning("vector store did not close cleanly", exc_info=True)
        try:
            await get_rate_limiter().close()
        except Exception:  # noqa: BLE001
            logger.warning("rate limiter did not close cleanly", exc_info=True)
        logger.info("app stopped")


app = FastAPI(
    title="Multi-Tenant RAG SaaS",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(RequestContextMiddleware)

# CORS. The widget runs on customers' own domains, which we do not know in
# advance, so the default origin list is "*". That is safe here for one specific
# reason, and it stops being safe the day that reason changes: every credential
# in this API travels in a HEADER (`X-Bot-Key`, `Authorization`), never in a
# cookie. A hostile page can send a cross-origin request to us, but the browser
# will not attach a credential the page does not already hold, and
# `allow_credentials=False` keeps it that way. Introduce cookie auth and this
# list must become explicit on the same day.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Bot-Key", "X-Request-ID"],
    # Response headers are invisible to browser JS unless exposed. These three
    # let a client show its own rate-limit budget and quote a request id when
    # reporting a problem.
    expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
)

# /health stays unauthenticated — load balancers and uptime checks have no
# session token. Everything below it carries a router-level auth dependency;
# see app/security/clerk.py on why that is a dependency rather than middleware.
app.include_router(health_router)
app.include_router(chatbots_router)
app.include_router(documents_router)
app.include_router(conversations_router)
app.include_router(playground_router)
# The only router authenticated by bot key rather than by a session token.
app.include_router(widget_router)

"""FastAPI application entrypoint.

Run locally:  uvicorn app.main:app --reload   (from backend/)
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.health import router as health_router
from app.config import settings
from app.core.logging import configure_logging
from app.core.middleware import RequestContextMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown. Later phases warm the embedding model and open the
    Qdrant client here so the first request doesn't pay that cost."""
    configure_logging(
        level=settings.log_level,
        # Human-readable in dev, JSON in production where logs get parsed.
        json_output=settings.is_production,
    )
    logger.info("app starting", extra={"env": settings.app_env})
    yield
    logger.info("app stopped")


app = FastAPI(
    title="Multi-Tenant RAG SaaS",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(RequestContextMiddleware)

app.include_router(health_router)

"""Health endpoints.

/health is a liveness probe: is this process alive? It deliberately touches no
dependencies — if it queried Postgres, a slow DB would make an orchestrator
kill an otherwise healthy app. A /ready readiness probe that does check
dependencies arrives in Phase 2, once there are dependencies worth checking.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    env: str
    version: str


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe. Deliberately dependency-free."""
    return HealthResponse(status="ok", env=settings.app_env, version="0.1.0")

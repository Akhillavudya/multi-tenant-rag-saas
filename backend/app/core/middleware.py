"""HTTP middleware: request correlation and access logging."""

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import request_id_var

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns every request an ID, binds it to the logging context, echoes it
    back in the response header, and logs one line per completed request.

    Accepting an inbound X-Request-ID lets a trace span the widget → API →
    worker hops later; we generate one when the caller didn't supply it.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:16]
        token = request_id_var.set(request_id)
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            # Log with context bound, then re-raise so FastAPI's exception
            # handlers still produce the response.
            logger.exception(
                "request failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": _elapsed_ms(start),
                },
            )
            raise
        else:
            response.headers[REQUEST_ID_HEADER] = request_id
            logger.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": _elapsed_ms(start),
                },
            )
            return response
        finally:
            # Reset on every path, or the ID leaks into whatever task reuses
            # this context next.
            request_id_var.reset(token)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)

"""Structured JSON logging with per-request context.

PLAN §7 requires org_id / bot_id / request_id on every log line. Threading those
through every function call would poison every signature, so they live in
contextvars: set once at the edge (middleware), read automatically by the
formatter. contextvars — unlike threading.local — are correct under asyncio,
where many requests interleave on one thread.
"""

import json
import logging
import sys
from contextvars import ContextVar
from typing import Any

# Set by middleware at request start; None outside a request (startup, workers).
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
org_id_var: ContextVar[str | None] = ContextVar("org_id", default=None)
bot_id_var: ContextVar[str | None] = ContextVar("bot_id", default=None)

# Attributes LogRecord always carries; anything else was passed via `extra=`
# and belongs in the JSON output.
_STANDARD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """One JSON object per line — greppable locally, queryable once shipped."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Request context, omitted when absent so lines stay readable.
        for key, var in (
            ("request_id", request_id_var),
            ("org_id", org_id_var),
            ("bot_id", bot_id_var),
        ):
            value = var.get()
            if value is not None:
                payload[key] = value

        # Anything the caller passed as extra={...}
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Install the formatter on the root logger. Called once, at app startup."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter()
        if json_output
        else logging.Formatter("%(levelname)-8s %(name)s: %(message)s")
    )

    root = logging.getLogger()
    root.handlers.clear()  # replace uvicorn's default handler
    root.addHandler(handler)
    root.setLevel(level)

    # uvicorn installs its own handlers; clearing them makes its logs flow
    # through ours instead of printing twice in two different formats.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).handlers.clear()
        logging.getLogger(name).propagate = True

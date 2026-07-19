"""Celery application: the worker process's entrypoint and configuration.

Run it with:

    celery -A app.workers.celery_app worker --loglevel=info --pool=solo

`--pool=solo` on Windows. Celery's default prefork pool relies on `fork()`,
which Windows does not have; without it the worker starts, accepts a task, and
dies with an obscure pickling error. On the Linux box in Phase 7 the default
prefork pool is correct and this flag comes off.

**Why a separate broker process at all.** The API must answer in milliseconds
and ingestion takes minutes, so the two cannot share a request. Redis holds the
queue: the API pushes a task name plus arguments, the worker pops it. The
queue is also the durability boundary — a task that has been published survives
the API process dying, which is the entire reason the document row is committed
BEFORE the task is published (see api/documents.py).

**What goes in the message: an id, and nothing else.**

`ingest_document.delay(document_id=...)` sends a UUID string, not the document
row and not the file. Three reasons, all of which have bitten people:

1. A message is a snapshot. Sending the row means the worker acts on state that
   may be minutes stale by the time it runs — including a status another job
   already changed.
2. Broker messages are not a private channel. Sending file contents or tenant
   details puts them in Redis, in logs, and in any monitoring tool attached to
   the queue.
3. Messages must stay small. Redis holds them in memory, and a queue of
   megabyte payloads is a memory-exhaustion incident waiting for a busy day.

The worker re-reads everything it needs from Postgres, where the current truth
lives.
"""

import logging

from celery import Celery

from app.config import settings
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)

celery_app = Celery(
    "ragsaas",
    broker=settings.redis_url,
    backend=settings.redis_url,
    # Import path of the module holding @task functions. Celery only knows
    # about tasks in modules it imports; a task defined somewhere unlisted
    # raises "Received unregistered task" at runtime, which reads like a broker
    # problem and is not.
    include=["app.workers.ingest_tasks"],
)

celery_app.conf.update(
    # JSON, not pickle. Pickle deserialization executes arbitrary code, so a
    # broker an attacker can write to becomes remote code execution on the
    # worker. JSON cannot express arbitrary objects, which is exactly the
    # constraint that keeps messages to plain ids.
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Acknowledge a task AFTER it finishes, not when it is received. If the
    # worker is killed mid-ingestion the message returns to the queue and is
    # retried, instead of vanishing and leaving a document stuck in PROCESSING
    # forever. Safe here because ingestion is idempotent — deterministic point
    # ids mean a re-run overwrites its own partial work (see vectorstore).
    task_acks_late=True,
    # One task at a time per worker process. The local embedder holds the model
    # weights and is CPU-bound; running several in parallel in one process
    # multiplies memory without multiplying throughput.
    worker_prefetch_multiplier=1,
    # A hard ceiling so one pathological document cannot occupy a worker
    # indefinitely. The soft limit fires first and raises inside the task, which
    # gives the handler a chance to mark the document FAILED with a real reason;
    # the hard limit kills the process if that does not work.
    task_soft_time_limit=25 * 60,
    task_time_limit=30 * 60,
    # Results are only used for debugging; expire them so Redis does not grow
    # without bound.
    result_expires=3600,
)


@celery_app.on_after_configure.connect
def _setup_worker_logging(sender: Celery, **_: object) -> None:
    """Give the worker the same structured logging as the API.

    Without this the worker uses Celery's own format and its lines lose the
    org_id/document_id fields that make an ingestion failure traceable — which
    is precisely when you need them most.
    """
    configure_logging(level=settings.log_level, json_output=settings.is_production)
    logger.info("celery worker configured", extra={"broker": "redis"})

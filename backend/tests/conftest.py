"""Shared pytest fixtures.

pytest auto-discovers this file — tests never import it. Any fixture defined
here is available to every test in this directory tree by parameter name.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.main import app


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """An in-process HTTP client for the API.

    Used as a context manager on purpose: that is what triggers FastAPI's
    lifespan (startup/shutdown). Constructing TestClient without `with` skips
    lifespan entirely, so anything initialized at startup would be missing —
    a subtle source of "works in prod, fails in tests" bugs later.

    session scope: the app is built once for the whole run rather than per
    test, since these tests don't mutate app state.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def db_engine() -> Iterator[Engine]:
    """A fresh, empty database for one test — SQLite, in memory, no Docker.

    Two arguments are load-bearing and both are easy to get wrong:

    `StaticPool` plus `check_same_thread=False` make every connection the SAME
    connection. Without them, `sqlite://` gives each new connection its own
    blank database — so the tables created here are invisible to the session
    under test, and every query fails with "no such table" for reasons that look
    nothing like the actual cause.

    The schema is built with `create_all`, NOT with Alembic. That is a real
    trade: these tests therefore cannot catch a migration that has drifted from
    the models. `alembic check` catches that, and it is the right tool for it —
    running migrations here would tie every unit test to Postgres-only DDL and
    put a container back in the loop we just removed.
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def db(db_engine: Engine) -> Iterator[Session]:
    """One session on the test database.

    `expire_on_commit=False` mirrors `db/base.py`: without it, reading any
    attribute after a commit fires a fresh SELECT, and tests that hold an ORM
    object across a commit start failing in ways production would not.

    Function-scoped, like `db_engine`. Metering tests mutate counters and chat
    tests write transcripts; a shared database would make every result depend
    on which test ran first.
    """
    factory = sessionmaker(bind=db_engine, autoflush=False, expire_on_commit=False)
    session = factory()
    try:
        yield session
    finally:
        session.close()

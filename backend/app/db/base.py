"""SQLAlchemy declarative base, engine, and session factory.

Models arrive in Phase 2. Alembic imports `Base` to autogenerate migrations by
diffing `Base.metadata` against the live database — so a model Alembic never
imports is a model Alembic will happily generate a DROP for.
"""

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    """Parent of every ORM model. Carries the metadata Alembic diffs against."""


# pool_pre_ping issues a cheap SELECT 1 before handing out a pooled connection.
# Serverless Postgres (Neon) drops idle connections, and without this the first
# query after an idle period fails with a stale-connection error.
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    echo=False,  # set True to see every emitted SQL statement while learning
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed.

    expire_on_commit=False above means ORM objects stay readable after commit —
    without it, touching any attribute post-commit triggers a fresh SELECT, and
    in a request handler that often means a DetachedInstanceError.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

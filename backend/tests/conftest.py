"""Shared pytest fixtures.

pytest auto-discovers this file — tests never import it. Any fixture defined
here is available to every test in this directory tree by parameter name.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

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

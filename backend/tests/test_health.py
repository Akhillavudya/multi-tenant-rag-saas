"""Smoke tests for the health endpoint.

Small, but they prove a lot: the app imports, config validates, lifespan runs,
middleware doesn't crash, routing resolves, and the response matches its
declared schema.
"""

from fastapi.testclient import TestClient

from app.core.middleware import REQUEST_ID_HEADER


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_reports_env_and_version(client: TestClient) -> None:
    """The response_model contract, not just the status code."""
    body = client.get("/health").json()

    assert set(body) == {"status", "env", "version"}
    assert body["env"] in {"development", "production"}
    assert body["version"] == "0.1.0"


def test_response_carries_request_id(client: TestClient) -> None:
    """Middleware must stamp every response, even when the caller sends none."""
    response = client.get("/health")

    assert response.headers.get(REQUEST_ID_HEADER)


def test_inbound_request_id_is_preserved(client: TestClient) -> None:
    """A caller-supplied ID must survive, so traces span services."""
    sent = "trace-me-12345"

    response = client.get("/health", headers={REQUEST_ID_HEADER: sent})

    assert response.headers[REQUEST_ID_HEADER] == sent


def test_unknown_route_returns_404(client: TestClient) -> None:
    assert client.get("/no-such-route").status_code == 404

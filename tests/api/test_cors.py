"""CORS is load-bearing for the React dashboard: it runs on its own origin, so
a missing or wrong header means every dashboard call fails in the browser while
curl keeps working - the kind of bug that eats an afternoon.

These use TestClient without entering its lifespan context, so no trained
artifacts or database are needed; only the middleware is under test.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app
from src.common.config import get_settings

client = TestClient(app)
ALLOWED_ORIGIN = get_settings().cors_origins[0]


def test_preflight_from_the_dashboard_origin_is_allowed():
    response = client.options(
        "/flags",
        headers={
            "Origin": ALLOWED_ORIGIN,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN


def test_authorization_header_is_allowed_through_preflight():
    """Every dashboard call carries a bearer token; if Authorization isn't in
    the allowed headers the browser blocks the request before it is sent."""
    response = client.options(
        "/flags",
        headers={
            "Origin": ALLOWED_ORIGIN,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    allowed = response.headers.get("access-control-allow-headers", "").lower()
    assert "authorization" in allowed


def test_an_unlisted_origin_is_not_granted_access():
    """The allowlist is the point: with allow_credentials on, echoing any
    origin would let a page the analyst has open call the API as them."""
    response = client.options(
        "/flags",
        headers={
            "Origin": "https://not-our-dashboard.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.headers.get("access-control-allow-origin") != (
        "https://not-our-dashboard.example"
    )


def test_wildcard_origin_is_never_returned():
    response = client.get("/health", headers={"Origin": ALLOWED_ORIGIN})
    assert response.headers.get("access-control-allow-origin") != "*"

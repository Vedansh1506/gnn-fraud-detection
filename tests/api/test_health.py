"""/health must always *answer*, including when its dependencies are down.

This is the endpoint the dashboard polls to decide whether to show the
degraded-mode banner (Design Doc 7). A health check that hangs is worse than
one that reports "unhealthy": the UI cannot tell a hung check from a hung app,
so it shows a spinner forever instead of an honest banner.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from src.api.main import app
from src.db.session import CONNECT_TIMEOUT_SECONDS, build_engine

client = TestClient(app)

# Generous: the point is "bounded", not "fast". The unbounded version of this
# never returned at all.
MAX_HEALTH_SECONDS = CONNECT_TIMEOUT_SECONDS + 12


def test_database_connections_are_bounded():
    """Regression guard for the real defect: without connect_timeout, a
    connection attempt to an unreachable Postgres never returned, so /health
    hung instead of degrading.

    Built through the same helper the app uses, pointed at an address that
    cannot answer, so it tests the timeout rather than the local DB's state.
    """
    engine = build_engine("postgresql+psycopg://u:p@10.255.255.1:5432/nope")

    started = time.perf_counter()
    with pytest.raises(OperationalError):
        with engine.connect():
            pass
    elapsed = time.perf_counter() - started

    assert elapsed < CONNECT_TIMEOUT_SECONDS + 5, (
        f"connection attempt took {elapsed:.1f}s - it must be bounded"
    )


def test_health_answers_even_when_dependencies_are_down():
    started = time.perf_counter()
    response = client.get("/health")
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    assert elapsed < MAX_HEALTH_SECONDS, (
        f"/health took {elapsed:.1f}s - it must degrade, not hang"
    )


def test_health_reports_component_status_honestly():
    """Whatever is actually up, the response says so per component rather than
    collapsing to a single opaque boolean."""
    body = client.get("/health").json()

    assert body["status"] in {"ok", "degraded", "unhealthy"}
    assert set(body["components"]) == {
        "model_loaded",
        "database_reachable",
        "embeddings_loaded",
    }
    # The serving versions are pinned, never "latest" - the dashboard shows
    # this string, so a blank one would be a silent misreport.
    assert body["model_version"]
    assert body["embedding_version"]

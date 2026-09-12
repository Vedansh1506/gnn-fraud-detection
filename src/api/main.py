"""FastAPI scoring service (SAD C8) - the modular monolith's online path.

    uv run uvicorn src.api.main:app --reload

Currently serves the core scoring path: POST /score and GET /health.
/flags, /feedback, /graph and auth arrive with the dashboard that consumes them.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from sqlalchemy import text

from src.api.schemas import (
    ComponentHealth,
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
)
from src.api.scoring import load_artifacts, score_transaction
from src.common.config import get_settings
from src.db.models import AuditLogEntry
from src.db.session import create_tables, session_scope

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("scoring")

_state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the pinned artifacts once, at startup - not per request. Scoring has
    a <1s p95 budget and re-reading 700k embeddings per call would blow it."""
    settings = get_settings()
    _state["artifacts"] = load_artifacts(settings.model_version, settings.embedding_version)
    logger.info(
        "loaded model_version=%s embedding_version=%s",
        settings.model_version,
        settings.embedding_version,
    )

    # The audit log is a product feature; if the table can't be created the
    # operator should see it now, not on the first scored transaction. Startup
    # still proceeds - scoring without an audit trail beats no service at all.
    try:
        create_tables()
        _state["database_reachable"] = True
    except Exception:
        logger.exception("database unavailable at startup - scoring will run unaudited")
        _state["database_reachable"] = False

    yield
    _state.clear()


app = FastAPI(title="GNN Fraud Detection - Scoring API", lifespan=lifespan)


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    artifacts = _state.get("artifacts")
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    response = score_transaction(request, artifacts)
    _write_audit_log(request, response)

    logger.info(
        "scored tx_id=%s score=%.4f flagged=%s degraded=%s latency_ms=%d",
        response.tx_id,
        response.score,
        response.is_flagged,
        response.degraded,
        response.latency_ms,
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    artifacts = _state.get("artifacts")
    database_reachable = _check_database()

    components = ComponentHealth(
        model_loaded=artifacts is not None,
        database_reachable=database_reachable,
        embeddings_loaded=artifacts is not None and not artifacts.embeddings.empty,
    )
    # Degraded rather than unhealthy when only the database is down: the
    # service can still score, it just can't record an audit trail.
    status = "ok" if all(vars(components).values()) else (
        "degraded" if components.model_loaded else "unhealthy"
    )
    return HealthResponse(
        status=status,
        model_version=settings.model_version,
        embedding_version=settings.embedding_version,
        components=components,
    )


def _check_database() -> bool:
    try:
        with session_scope() as session:
            session.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.warning("database health check failed", exc_info=True)
        return False


def _write_audit_log(request: ScoreRequest, response: ScoreResponse) -> None:
    """Never let an audit-log failure lose an already-computed score: log it
    loudly and still return the result (graceful degradation, locked rule)."""
    try:
        with session_scope() as session:
            session.add(
                AuditLogEntry(
                    tx_id=response.tx_id,
                    account_key=request.sender_account_key,
                    score=response.score,
                    is_flagged=response.is_flagged,
                    explanation_json={
                        "factors": [factor.model_dump() for factor in response.explanation]
                    },
                    model_version=response.model_version,
                    embedding_version=response.embedding_version,
                )
            )
    except Exception:
        logger.exception("audit log write failed for tx_id=%s", response.tx_id)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Full detail to the logs, a plain message to the caller (rules.md 4)."""
    logger.exception("unhandled error on %s", request.url.path)
    raise HTTPException(status_code=500, detail="Internal error - see service logs")

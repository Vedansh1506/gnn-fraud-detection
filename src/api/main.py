"""FastAPI scoring service (SAD C8) - the modular monolith's online path.

    uv run uvicorn src.api.main:app --reload

Endpoints (TRD 5.1): POST /score (X-API-Key, service-to-service), GET /health
(open, for liveness probes), and the dashboard-facing set behind JWT -
POST /auth/login, GET /flags, POST /feedback, GET /graph/{account_key}, and
POST /retrain (operator role only).
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from sqlalchemy import select, text

from src.api.auth import (
    authenticate_user,
    create_access_token,
    require_api_key,
    require_operator,
    require_user,
)
from src.api.graph_context import fetch_neighbourhood
from src.api.schemas import (
    ComponentHealth,
    FeedbackRequest,
    FeedbackResponse,
    FlaggedTransaction,
    FlagsResponse,
    GraphResponse,
    HealthResponse,
    LoginRequest,
    LoginResponse,
    RetrainResponse,
    ScoreRequest,
    ScoreResponse,
)
from src.api.scoring import load_artifacts, score_transaction
from src.common.config import get_settings
from src.db.models import AuditLogEntry, Feedback
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


@app.post("/score", response_model=ScoreResponse, dependencies=[Depends(require_api_key)])
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


@app.post("/auth/login", response_model=LoginResponse)
def login(request: LoginRequest) -> LoginResponse:
    """Not in TRD 5.1, but the Design Doc's login screen and SAD 8's JWT
    requirement need somewhere to exchange credentials for a token. Added by
    decision on 2026-09-12, with the users table added to TRD 4.3 to match."""
    user = authenticate_user(request.username, request.password)
    if user is None:
        # Deliberately identical for unknown user and wrong password.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password"
        )

    settings = get_settings()
    logger.info("login username=%s role=%s", user.username, user.role)
    return LoginResponse(
        access_token=create_access_token(user.username, user.role),
        role=user.role,
        expires_in_minutes=settings.jwt_expire_minutes,
    )


@app.get("/flags", response_model=FlagsResponse)
def flags(
    _claims: Annotated[dict, Depends(require_user)],
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    since: datetime | None = None,
) -> FlagsResponse:
    """The analyst queue: most recent flags first (Design Doc 4.2). Capped at
    500 to keep the dashboard's <2s load target honest."""
    query = select(AuditLogEntry).where(AuditLogEntry.is_flagged.is_(True))
    if since is not None:
        query = query.where(AuditLogEntry.scored_at >= since)
    query = query.order_by(AuditLogEntry.scored_at.desc()).limit(limit)

    with session_scope() as session:
        entries = list(session.scalars(query))

    return FlagsResponse(
        flags=[
            FlaggedTransaction(
                tx_id=entry.tx_id,
                account_key=entry.account_key,
                score=entry.score,
                is_flagged=entry.is_flagged,
                model_version=entry.model_version,
                embedding_version=entry.embedding_version,
                scored_at=entry.scored_at,
                explanation=(entry.explanation_json or {}).get("factors", []),
            )
            for entry in entries
        ],
        count=len(entries),
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(
    request: FeedbackRequest, claims: Annotated[dict, Depends(require_user)]
) -> FeedbackResponse:
    """Records an analyst decision. used_in_training_run stays NULL until a
    retrain consumes it, which is how the next retrain finds new labels."""
    with session_scope() as session:
        session.add(
            Feedback(
                tx_id=request.tx_id,
                analyst_decision=request.analyst_decision,
                decided_by=str(claims.get("sub", "unknown")),
            )
        )

    logger.info(
        "feedback tx_id=%s decision=%s by=%s",
        request.tx_id,
        request.analyst_decision,
        claims.get("sub"),
    )
    return FeedbackResponse(status="recorded", tx_id=request.tx_id)


@app.post("/retrain", response_model=RetrainResponse)
def retrain(claims: Annotated[dict, Depends(require_operator)]) -> RetrainResponse:
    """Operator-only. GNN training runs on Kaggle/Colab (locked compute
    decision), so for the MVP this returns a reference id and the steps rather
    than kicking off a job the API has no GPU to run - TRD 5.1 allows exactly
    that. Nothing is persisted yet; a retrain-history table can come with the
    dashboard's Model & Ops screen if it's wanted there.
    """
    job_id = f"retrain-{uuid.uuid4().hex[:12]}"
    logger.info("retrain requested job_id=%s by=%s", job_id, claims.get("sub"))
    return RetrainResponse(
        job_id=job_id,
        status="accepted",
        instructions=(
            "1) uv run --group gnn python -m src.models.gnn.train_graphsage "
            "--version <new_embedding_version>  "
            "2) uv run python -m src.models.classifier.train_baseline "
            "--embeddings artifacts/embeddings/<new_embedding_version>/embeddings.parquet "
            "--version <new_model_version>  "
            "3) uv run python -m src.models.classifier.evaluate --version <new_model_version>  "
            "4) promote by pointing MODEL_VERSION/EMBEDDING_VERSION at the new "
            "versions only after they clear the eval gate."
        ),
    )


@app.get("/graph/{account_key}", response_model=GraphResponse)
def graph(
    account_key: str,
    _claims: Annotated[dict, Depends(require_user)],
    hops: Annotated[int, Query(ge=1, le=3)] = 2,
) -> GraphResponse:
    """Money-flow context for the flag detail view. Neo4j being down fails just
    this panel with a clear message - it never blocks scoring, which reads no
    graph at all."""
    try:
        neighbourhood = fetch_neighbourhood(account_key, hops)
    except Exception as exc:
        logger.exception("graph lookup failed for %s", account_key)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph context unavailable - the graph database is unreachable.",
        ) from exc

    return GraphResponse(
        account_key=neighbourhood.account_key,
        hops=hops,
        nodes=neighbourhood.nodes,
        edges=[vars(edge) for edge in neighbourhood.edges],
        truncated=neighbourhood.truncated,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Full detail to the logs, a plain message to the caller (rules.md 4)."""
    logger.exception("unhandled error on %s", request.url.path)
    raise HTTPException(status_code=500, detail="Internal error - see service logs")

"""FastAPI scoring service (SAD C8) - the modular monolith's online path.

    uv run uvicorn src.api.main:app --reload

Endpoints (TRD 5.1): POST /score (X-API-Key, service-to-service), GET /health
(open, for liveness probes), and the dashboard-facing set behind JWT -
POST /auth/login, GET /flags, POST /feedback, GET /graph/{account_key},
GET /models, and POST /retrain (operator role only).
"""

from __future__ import annotations

import datetime as dt
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func as sql_func
from sqlalchemy import select, text

from src.api.auth import (
    authenticate_user,
    create_access_token,
    require_api_key,
    require_operator,
    require_user,
)
from src.api.graph_context import fetch_neighbourhood
from src.api.model_registry import baseline_and_gnn, lift_pct, list_versions
from src.api.schemas import (
    ComponentHealth,
    DriftStatus,
    FeedbackRequest,
    FeedbackResponse,
    FlaggedTransaction,
    FlagsResponse,
    GraphResponse,
    HealthResponse,
    LoginRequest,
    LoginResponse,
    ModelsResponse,
    ModelVersionOut,
    QueueSummary,
    RetrainResponse,
    RetrainRunOut,
    ScoreRequest,
    ScoreResponse,
)
from src.api.scoring import load_artifacts, score_transaction
from src.common.config import get_settings
from src.db.models import (
    CONFIRMED_FRAUD,
    FALSE_POSITIVE,
    REQUESTED,
    AuditLogEntry,
    Feedback,
    RetrainRun,
)
from src.db.session import create_tables, session_scope
from src.mlops.drift import check_drift

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("scoring")

_state: dict[str, object] = {}

# How long a computed drift report stays good for. Drift moves over hours, not
# seconds; recomputing per request would make the Ops screen slow for no gain.
DRIFT_CACHE_SECONDS = 300


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

# The dashboard is a separate origin (Vite dev server / the nginx container),
# so every dashboard call is preflighted. Origins come from config and are
# explicit: allow_credentials with a wildcard is both forbidden by the spec and
# exactly the hole that would let any page call this API with a live token.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)


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
                    # Transaction detail the analyst queue shows on each row.
                    # Copied from the request rather than re-fetched later:
                    # the audit row is meant to be a complete record of what
                    # was scored, readable without the source system.
                    receiver_account_key=request.receiver_account_key,
                    amount_paid=request.amount_paid,
                    payment_currency=request.payment_currency,
                    payment_format=request.payment_format,
                    tx_timestamp=request.timestamp,
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


# The feedback row that decides a flag: latest decision per tx_id, so an
# analyst changing their mind doesn't leave the queue showing the stale verdict.
_LATEST_FEEDBACK = (
    select(
        Feedback.tx_id.label("tx_id"),
        sql_func.max(Feedback.decided_at).label("decided_at"),
    )
    .group_by(Feedback.tx_id)
    .subquery()
)

# The newest audit row per transaction.
#
# The audit log deliberately keeps *every* scoring event - re-streaming a
# transaction writes a second row, and deleting that history would defeat the
# point of an audit trail. But the queue is a worklist of transactions, not of
# scoring events: without this, a transaction scored twice appeared in the
# analyst's list twice (observed live: 179 rows for 177 transactions). Highest
# id wins rather than latest `scored_at`, because two rows written in the same
# instant would otherwise both survive.
_LATEST_AUDIT = (
    select(sql_func.max(AuditLogEntry.id).label("id"))
    .where(AuditLogEntry.is_flagged.is_(True))
    .group_by(AuditLogEntry.tx_id)
    .subquery()
)


@app.get("/flags", response_model=FlagsResponse)
def flags(
    _claims: Annotated[dict, Depends(require_user)],
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    since: datetime | None = None,
    status_filter: Annotated[
        Literal["open", "reviewed", "all"], Query(alias="status")
    ] = "open",
    min_score: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
) -> FlagsResponse:
    """The analyst queue: highest risk first (Design Doc 4.2).

    Left-joined to feedback so a reviewed flag carries its verdict and can
    leave the default view - without it the queue grows forever and the
    analyst loop never visibly closes. Capped at 500 to keep the dashboard's
    load target honest.
    """
    query = (
        select(AuditLogEntry, Feedback)
        .join(_LATEST_AUDIT, AuditLogEntry.id == _LATEST_AUDIT.c.id)
        .outerjoin(_LATEST_FEEDBACK, AuditLogEntry.tx_id == _LATEST_FEEDBACK.c.tx_id)
        .outerjoin(
            Feedback,
            (Feedback.tx_id == _LATEST_FEEDBACK.c.tx_id)
            & (Feedback.decided_at == _LATEST_FEEDBACK.c.decided_at),
        )
    )
    if since is not None:
        query = query.where(AuditLogEntry.scored_at >= since)
    if min_score > 0.0:
        query = query.where(AuditLogEntry.score >= min_score)
    if status_filter == "open":
        query = query.where(_LATEST_FEEDBACK.c.tx_id.is_(None))
    elif status_filter == "reviewed":
        query = query.where(_LATEST_FEEDBACK.c.tx_id.is_not(None))

    # Score-descending is the Design Doc default: the queue is a risk-ordered
    # worklist, so the riskiest item must be reachable without paging.
    query = query.order_by(AuditLogEntry.score.desc(), AuditLogEntry.scored_at.desc()).limit(limit)

    with session_scope() as session:
        rows = list(session.execute(query))
        summary = _queue_summary(session)

    return FlagsResponse(
        flags=[_to_flag(entry, decision) for entry, decision in rows],
        count=len(rows),
        summary=summary,
    )


def _to_flag(entry: AuditLogEntry, decision: Feedback | None) -> FlaggedTransaction:
    return FlaggedTransaction(
        tx_id=entry.tx_id,
        account_key=entry.account_key,
        score=entry.score,
        is_flagged=entry.is_flagged,
        model_version=entry.model_version,
        embedding_version=entry.embedding_version,
        scored_at=entry.scored_at,
        explanation=(entry.explanation_json or {}).get("factors", []),
        receiver_account_key=entry.receiver_account_key,
        amount_paid=entry.amount_paid,
        payment_currency=entry.payment_currency,
        payment_format=entry.payment_format,
        tx_timestamp=entry.tx_timestamp,
        analyst_decision=decision.analyst_decision if decision else None,
        decided_at=decision.decided_at if decision else None,
        decided_by=decision.decided_by if decision else None,
    )


def _queue_summary(session) -> QueueSummary:
    """Counts for the summary strip, over the whole audit log rather than the
    returned page - a count that shrank when you changed page size would be
    actively misleading."""
    reviewed = select(Feedback.tx_id).distinct().scalar_subquery()
    # Counts distinct transactions, matching what the list returns - a summary
    # that said 179 above a list of 177 would undermine both numbers.
    open_flags = session.scalar(
        select(sql_func.count(sql_func.distinct(AuditLogEntry.tx_id)))
        .select_from(AuditLogEntry)
        .where(AuditLogEntry.is_flagged.is_(True), AuditLogEntry.tx_id.not_in(reviewed))
    )

    # "Today" is deliberately the last 24 hours rather than a calendar day:
    # there is no per-user timezone in the MVP, and a rolling window is the
    # same number for everyone looking at the demo.
    since_cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(hours=24)
    decided = session.execute(
        select(Feedback.analyst_decision, sql_func.count())
        .where(Feedback.decided_at >= since_cutoff)
        .group_by(Feedback.analyst_decision)
    ).all()
    counts = dict(decided)

    return QueueSummary(
        open_flags=open_flags or 0,
        confirmed_today=counts.get(CONFIRMED_FRAUD, 0),
        dismissed_today=counts.get(FALSE_POSITIVE, 0),
        model_version=get_settings().model_version,
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
    that. The request is persisted to `retrain_runs` so the Model & Ops screen
    shows real history - the row is a work order recording that an operator
    asked for a retrain, not evidence that one ran.
    """
    job_id = f"retrain-{uuid.uuid4().hex[:12]}"
    requested_by = str(claims.get("sub", "unknown"))

    # Record the request before returning it. Previously this was
    # fire-and-forget, which left the Ops screen with nothing real to show;
    # the row also captures *why* it was requested (how much unused feedback
    # had accumulated) at the moment the operator decided.
    with session_scope() as session:
        pending = session.scalar(
            select(sql_func.count())
            .select_from(Feedback)
            .where(Feedback.used_in_training_run.is_(None))
        ) or 0
        run = RetrainRun(
            job_id=job_id,
            status=REQUESTED,
            requested_by=requested_by,
            feedback_rows_pending=pending,
        )
        session.add(run)
        session.flush()
        requested_at = run.requested_at or dt.datetime.now(dt.UTC)

    logger.info(
        "retrain requested job_id=%s by=%s feedback_pending=%d", job_id, requested_by, pending
    )
    return RetrainResponse(
        job_id=job_id,
        status=REQUESTED,
        requested_at=requested_at,
        feedback_rows_pending=pending,
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


@app.get("/models", response_model=ModelsResponse)
def models(_claims: Annotated[dict, Depends(require_user)]) -> ModelsResponse:
    """Model & Ops data (Design Doc 4.4).

    Every metric here was measured by an evaluation run and is read from that
    run's `metrics.json` artifact - this endpoint computes no model quality
    numbers of its own. Drift is reported as not-instrumented rather than
    green, because Evidently is a later build step and a health light nobody
    computed would be the exact dishonesty this project argues against.
    """
    settings = get_settings()
    versions = list_versions()
    baseline, gnn = baseline_and_gnn(versions)

    reviewed, overrides = _feedback_totals()
    recent = _recent_retrains()

    return ModelsResponse(
        serving_model_version=settings.model_version,
        serving_embedding_version=settings.embedding_version,
        flag_threshold=settings.flag_threshold,
        versions=[
            ModelVersionOut(
                version=metrics.version,
                auprc=metrics.auprc,
                roc_auc=metrics.roc_auc,
                test_rows=metrics.test_rows,
                test_positives=metrics.test_positives,
                best_f1_threshold=metrics.best_f1_threshold,
                best_f1_precision=metrics.best_f1_precision,
                best_f1_recall=metrics.best_f1_recall,
                best_f1=metrics.best_f1,
                embedding_version=metrics.embedding_version,
                is_serving=metrics.version == settings.model_version,
            )
            for metrics in versions
        ],
        baseline_auprc=baseline.auprc if baseline else None,
        gnn_auprc=gnn.auprc if gnn else None,
        auprc_lift_pct=lift_pct(baseline, gnn),
        # None, not 0.0, when nothing has been reviewed: "no analyst has ever
        # disagreed" and "no analyst has ever looked" are different facts.
        override_rate=(overrides / reviewed) if reviewed else None,
        reviewed_count=reviewed,
        drift=_drift_status(settings.model_version),
        recent_retrains=recent,
    )


def _drift_status(model_version: str) -> DriftStatus:
    """Real drift, computed against the pinned reference (Design Doc 4.4).

    Cached: an Evidently run over tens of thousands of rows is not something to
    repeat on every page load, and drift is a slow-moving signal - a value
    minutes old is still true. A failure here degrades this one panel and never
    takes the Ops screen down.
    """
    try:
        report = _cached_drift(model_version, _drift_cache_key())
    except Exception:
        logger.exception("drift check failed")
        return DriftStatus(
            state="not_instrumented",
            message="The drift check could not be computed - see the service logs.",
        )

    return DriftStatus(
        state=report.state, message=report.message, checked_at=report.checked_at
    )


def _drift_cache_key() -> int:
    """Buckets time so the cached report refreshes on its own every few
    minutes without needing a background scheduler."""
    return int(time.time() // DRIFT_CACHE_SECONDS)


@lru_cache(maxsize=4)
def _cached_drift(model_version: str, _bucket: int):
    return check_drift(model_version)


def _feedback_totals() -> tuple[int, int]:
    """(reviewed, overrides). An override is a flag the model raised and the
    analyst rejected - the only live quality signal the MVP has."""
    try:
        with session_scope() as session:
            reviewed = session.scalar(select(sql_func.count()).select_from(Feedback)) or 0
            overrides = (
                session.scalar(
                    select(sql_func.count())
                    .select_from(Feedback)
                    .where(Feedback.analyst_decision == FALSE_POSITIVE)
                )
                or 0
            )
        return reviewed, overrides
    except Exception:
        # One panel's data being unavailable must not blank the Ops screen.
        logger.exception("feedback totals unavailable")
        return 0, 0


def _recent_retrains(limit: int = 5) -> list[RetrainRunOut]:
    try:
        with session_scope() as session:
            runs = list(
                session.scalars(
                    select(RetrainRun).order_by(RetrainRun.requested_at.desc()).limit(limit)
                )
            )
        return [
            RetrainRunOut(
                job_id=run.job_id,
                status=run.status,
                requested_by=run.requested_by,
                requested_at=run.requested_at,
                feedback_rows_pending=run.feedback_rows_pending,
                resulting_model_version=run.resulting_model_version,
            )
            for run in runs
        ]
    except Exception:
        logger.exception("retrain history unavailable")
        return []


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

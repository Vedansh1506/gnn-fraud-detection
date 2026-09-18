"""Request/response contracts for the scoring API (TRD 5.1, 4.4).

Strict by design: unknown fields and malformed values are rejected with 422
rather than coerced or sanitised, per the locked validation rule.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ScoreRequest(BaseModel):
    """Mirrors the stream event schema (TRD 4.4) so the consumer can forward an
    event straight through without reshaping it."""

    model_config = ConfigDict(extra="forbid")

    tx_id: str = Field(min_length=1)
    sender_account_key: str = Field(min_length=1)
    receiver_account_key: str = Field(min_length=1)
    amount_paid: float = Field(ge=0)
    payment_currency: str = Field(min_length=1)
    amount_received: float = Field(ge=0)
    receiving_currency: str = Field(min_length=1)
    payment_format: str = Field(min_length=1)
    timestamp: datetime


class ExplanationFactor(BaseModel):
    feature: str
    contribution: float
    plain: str


class ScoreResponse(BaseModel):
    tx_id: str
    score: float
    is_flagged: bool
    model_version: str
    embedding_version: str
    explanation: list[ExplanationFactor]
    latency_ms: int
    degraded: bool = False


class ComponentHealth(BaseModel):
    model_loaded: bool
    database_reachable: bool
    embeddings_loaded: bool


class HealthResponse(BaseModel):
    status: str
    model_version: str
    embedding_version: str
    components: ComponentHealth


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    expires_in_minutes: int


class FlaggedTransaction(BaseModel):
    """One row of the analyst queue (Design Doc 4.2).

    The transaction detail fields are optional because audit rows written
    before those columns existed genuinely do not have them - the UI shows an
    honest em-dash rather than a fabricated amount.
    """

    tx_id: str
    account_key: str
    score: float
    is_flagged: bool
    model_version: str
    embedding_version: str
    scored_at: datetime
    explanation: list[ExplanationFactor]

    receiver_account_key: str | None = None
    amount_paid: float | None = None
    payment_currency: str | None = None
    payment_format: str | None = None
    tx_timestamp: datetime | None = None

    # None = nobody has reviewed this flag yet, which is what makes it "open".
    analyst_decision: Literal["confirmed_fraud", "false_positive"] | None = None
    decided_at: datetime | None = None
    decided_by: str | None = None


class QueueSummary(BaseModel):
    """The Flag Queue's summary strip (Design Doc 4.2). Counted over the whole
    audit log, not just the returned page - a count that changes with page size
    would be worse than no count."""

    open_flags: int
    confirmed_today: int
    dismissed_today: int
    model_version: str


class FlagsResponse(BaseModel):
    flags: list[FlaggedTransaction]
    count: int
    summary: QueueSummary


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tx_id: str = Field(min_length=1)
    # Constrained to the two decisions the feedback table's CHECK allows, so a
    # bad value is a 422 at the edge rather than a database error deeper in.
    analyst_decision: Literal["confirmed_fraud", "false_positive"]


class FeedbackResponse(BaseModel):
    status: str
    tx_id: str


class RetrainResponse(BaseModel):
    job_id: str
    status: str
    instructions: str
    requested_at: datetime
    feedback_rows_pending: int


class RetrainRunOut(BaseModel):
    job_id: str
    status: str
    requested_by: str
    requested_at: datetime
    feedback_rows_pending: int
    resulting_model_version: str | None = None


class ModelVersionOut(BaseModel):
    """A trained version as recorded in its own `metrics.json` artifact. Every
    number here was measured by an evaluation run; nothing is computed or
    guessed at read time."""

    model_config = ConfigDict(protected_namespaces=())

    version: str
    auprc: float
    roc_auc: float
    test_rows: int
    test_positives: int
    best_f1_threshold: float
    best_f1_precision: float
    best_f1_recall: float
    best_f1: float
    is_serving: bool
    embedding_version: str | None = None
    # Features this version was trained WITHOUT (ablation runs). The UI must
    # label these, or an ablation row reads as a normal model that simply
    # scored badly.
    dropped_features: list[str] = []
    # Operational view. None where a version predates this being recorded -
    # the UI shows an em-dash rather than implying zero alerts a day.
    alerts_per_day: float | None = None
    precision_at_threshold: float | None = None
    recall_at_threshold: float | None = None
    transactions_per_day: float | None = None


class DriftStatus(BaseModel):
    """Deliberately shaped to be able to say "not measured yet". Evidently
    lands in a later build step, and a green light nobody computed would be a
    lie in exactly the place this project claims honesty."""

    state: Literal["ok", "warning", "alert", "not_instrumented"]
    message: str
    checked_at: datetime | None = None


class ModelsResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    serving_model_version: str
    serving_embedding_version: str
    flag_threshold: float
    versions: list[ModelVersionOut]
    baseline_auprc: float | None = None
    gnn_auprc: float | None = None
    auprc_lift_pct: float | None = None
    # Share of reviewed flags the analyst rejected - the closest thing to a
    # live quality signal the MVP actually has.
    override_rate: float | None = None
    reviewed_count: int
    drift: DriftStatus
    recent_retrains: list[RetrainRunOut]


class GraphEdgeOut(BaseModel):
    source: str
    target: str
    tx_id: str
    amount_paid: float
    is_laundering: bool


class GraphResponse(BaseModel):
    account_key: str
    hops: int
    nodes: list[str]
    edges: list[GraphEdgeOut]
    truncated: bool

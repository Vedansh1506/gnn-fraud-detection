"""Relational schema (TRD 4.3). Postgres in Docker for dev, RDS in cloud - the
same SQLAlchemy code runs against both, which is why dev doesn't use SQLite.

The Model & Ops screen reads trained-model metadata straight from the pinned
artifacts on disk (each version's `metrics.json`), so there is still no
`model_registry_meta` table - the artifacts are already the source of truth and
a table would just be a copy that can drift from them.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    Integer,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

ANALYST_ROLE = "analyst"
OPERATOR_ROLE = "operator"
ROLES = (ANALYST_ROLE, OPERATOR_ROLE)

CONFIRMED_FRAUD = "confirmed_fraud"
FALSE_POSITIVE = "false_positive"
ANALYST_DECISIONS = (CONFIRMED_FRAUD, FALSE_POSITIVE)

REQUESTED = "requested"
RETRAIN_STATUSES = ("requested", "running", "completed", "failed")


class Base(DeclarativeBase):
    pass


class AuditLogEntry(Base):
    """One row per scored transaction - score, explanation, model version and
    time. This is a product feature, not just logging: "every decision
    traceable" is an explicit acceptance criterion, and it's what /flags and
    the dashboard read from.
    """

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tx_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    # The sending side. Kept as `account_key` rather than renamed to
    # `sender_account_key`: it is the audit row's subject account and the key
    # /graph is opened on, and the TRD vocabulary rule forbids synonyms.
    account_key: Mapped[str] = mapped_column(String, nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    is_flagged: Mapped[bool] = mapped_column(Boolean, nullable=False)
    # JSONB rather than text so explanations stay queryable (TRD 4.3).
    explanation_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    embedding_version: Mapped[str] = mapped_column(String, nullable=False)
    scored_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Transaction detail carried on the audit row so the analyst queue can show
    # a triageable line (amount, counterparty, when it happened) without a
    # second lookup per row. Added 2026-09-13 with TRD 4.3 updated to match;
    # nullable because rows written before that change have no values and
    # backfilling a synthetic amount would be inventing data.
    receiver_account_key: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    amount_paid: Mapped[float | None] = mapped_column(Float, nullable=True)
    payment_currency: Mapped[str | None] = mapped_column(String, nullable=True)
    payment_format: Mapped[str | None] = mapped_column(String, nullable=True)
    # When the transaction happened, as opposed to scored_at (when we saw it).
    # A stream replay makes these differ by years, and the analyst needs the
    # former to judge the transaction.
    tx_timestamp: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class Feedback(Base):
    """Analyst decisions, which are also the retraining label source (TRD 4.3).

    used_in_training_run stays NULL until a retrain consumes the row - that's
    how a retrain knows which feedback is new.
    """

    __tablename__ = "feedback"
    __table_args__ = (
        CheckConstraint(
            "analyst_decision IN ('confirmed_fraud', 'false_positive')",
            name="feedback_decision_check",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tx_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    analyst_decision: Mapped[str] = mapped_column(String, nullable=False)
    decided_by: Mapped[str] = mapped_column(String, nullable=False)
    decided_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    used_in_training_run: Mapped[str | None] = mapped_column(String, nullable=True)


class RetrainRun(Base):
    """Retrain requests, so the Ops screen can show real history rather than a
    fire-and-forget job id.

    Added 2026-09-13 (TRD 4.3) when the Model & Ops screen needed it. Honest
    about what it is: the locked compute decision puts GNN training on
    Kaggle/Colab, so the API cannot execute a retrain. A row records that an
    operator *requested* one and stays `requested` until someone runs the steps
    and closes it out - it is a work order, not a job runner.
    """

    __tablename__ = "retrain_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('requested', 'running', 'completed', 'failed')",
            name="retrain_runs_status_check",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default=REQUESTED)
    requested_by: Mapped[str] = mapped_column(String, nullable=False)
    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # How many feedback rows were unused at request time - the reason to
    # retrain, captured when the decision was made rather than recomputed later.
    feedback_rows_pending: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Set when the operator closes the run out with the versions it produced.
    resulting_model_version: Mapped[str | None] = mapped_column(String, nullable=True)
    notes: Mapped[str | None] = mapped_column(String, nullable=True)


class User(Base):
    """Dashboard/API accounts for the MVP's two personas (SAD 8).

    Not in the original TRD 4.3 schema - added by decision on 2026-09-12 so
    JWTs have something to authenticate against, with TRD updated to match.
    Deliberately minimal: enterprise SSO/RBAC is explicitly out of scope, so
    there is one role column rather than a permissions model.
    """

    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("role IN ('analyst', 'operator')", name="users_role_check"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    # bcrypt hash, never the password itself.
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

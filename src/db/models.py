"""Relational schema (TRD 4.3). Postgres in Docker for dev, RDS in cloud - the
same SQLAlchemy code runs against both, which is why dev doesn't use SQLite.

model_registry_meta is not here yet: nothing reads it until the dashboard's
Model & Ops screen exists.
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

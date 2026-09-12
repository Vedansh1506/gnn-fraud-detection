"""Relational schema (TRD 4.3). Postgres in Docker for dev, RDS in cloud - the
same SQLAlchemy code runs against both, which is why dev doesn't use SQLite.

Only audit_log exists so far: it's what the scoring path writes. The feedback
and model_registry_meta tables arrive with the endpoints that use them.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


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

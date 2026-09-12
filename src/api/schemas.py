"""Request/response contracts for the scoring API (TRD 5.1, 4.4).

Strict by design: unknown fields and malformed values are rejected with 422
rather than coerced or sanitised, per the locked validation rule.
"""

from __future__ import annotations

from datetime import datetime

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

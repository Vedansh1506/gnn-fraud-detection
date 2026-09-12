"""Serving-time scoring: rebuild one transaction's features, score, explain.

The train/serve contract is that this module *reads* what training wrote - the
pinned `feature_spec.json`, the fitted `account_features.parquet`, and the
embeddings parquet - and never recomputes statistics of its own. Recomputing
from live data is exactly how train/serve skew (TRD 7.1's "#1 silent ML bug")
gets in.

Note this path does not touch Neo4j at all: account context comes from the
pinned artifacts, so the graph store being down cannot stop a transaction being
scored. Neo4j is only needed for the dashboard's graph-context view.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.api.schemas import ExplanationFactor, ScoreRequest, ScoreResponse
from src.common.config import get_settings
from src.features.tx_features import safe_zscore

ARTIFACTS_ROOT = Path("artifacts/models")
EMBEDDINGS_ROOT = Path("artifacts/embeddings")
MISSING_EMBEDDING_VERSION = "missing"
TOP_K_FACTORS = 3

# Plain-language wording for the explanation (TRD 5.1 shows explanations a
# non-technical reviewer could follow). Unmapped features fall back to their
# own name rather than inventing a description.
FEATURE_DESCRIPTIONS = {
    "amount_paid": "Transaction amount",
    "amount_received": "Amount received",
    "amount_zscore": "Amount is unusual for this sending account",
    "hour_of_day": "Time of day the transaction was made",
    "payment_format": "Payment method used",
    "cross_currency_flag": "Payment crosses currencies",
    "cross_bank_flag": "Payment crosses banks",
    "sender_in_degree": "How many payments the sender receives",
    "sender_out_degree": "How many payments the sender makes",
    "sender_distinct_counterparties": "How many different accounts the sender deals with",
    "sender_avg_amount": "Sender's typical payment size",
    "sender_tx_count_24h": "Sender's recent activity burst",
    "receiver_in_degree": "How many payments the receiver takes in",
    "receiver_out_degree": "How many payments the receiver sends on",
    "receiver_distinct_counterparties": "How many different accounts the receiver deals with",
    "receiver_avg_amount": "Receiver's typical payment size",
    "receiver_tx_count_24h": "Receiver's recent activity burst",
}


@dataclass
class ScoringArtifacts:
    """Everything loaded once at startup and held in memory - the lookups have
    to be fast to keep scoring inside the <1s p95 target."""

    model: xgb.XGBClassifier
    explainer: shap.TreeExplainer
    feature_columns: list[str]
    categorical_values: dict[str, list[str]]
    account_features: pd.DataFrame
    embeddings: pd.DataFrame
    model_version: str
    embedding_version: str


def load_artifacts(model_version: str, embedding_version: str) -> ScoringArtifacts:
    spec = json.loads((ARTIFACTS_ROOT / model_version / "feature_spec.json").read_text())

    model = xgb.XGBClassifier(enable_categorical=True)
    model.load_model(ARTIFACTS_ROOT / model_version / "model.json")

    account_features = pd.read_parquet(
        ARTIFACTS_ROOT / model_version / "account_features.parquet"
    ).set_index("account_key")
    embeddings = pd.read_parquet(
        EMBEDDINGS_ROOT / embedding_version / "embeddings.parquet"
    ).set_index("account_key")

    return ScoringArtifacts(
        model=model,
        explainer=shap.TreeExplainer(model),
        feature_columns=spec["feature_columns"],
        categorical_values=spec.get("categorical_values", {}),
        account_features=account_features,
        embeddings=embeddings,
        model_version=model_version,
        embedding_version=embedding_version,
    )


def _bank_of(account_key: str) -> str:
    """account_key is "<bank>_<account>" (TRD 4.1), so the bank is recoverable
    from the key alone - the event schema doesn't carry it separately."""
    return account_key.split("_", 1)[0]


def build_feature_row(
    request: ScoreRequest, artifacts: ScoringArtifacts
) -> tuple[pd.DataFrame, bool]:
    """Reconstruct the model's feature vector for a single transaction.

    Returns the row plus whether any embedding was missing, which the caller
    surfaces as degraded mode rather than silently scoring on zeros.
    """
    sender = artifacts.account_features.reindex([request.sender_account_key]).iloc[0]
    receiver = artifacts.account_features.reindex([request.receiver_account_key]).iloc[0]

    row: dict[str, object] = {
        "amount_paid": request.amount_paid,
        "amount_received": request.amount_received,
        "hour_of_day": request.timestamp.hour,
        "amount_zscore": float(
            safe_zscore(
                pd.Series([request.amount_paid]),
                pd.Series([sender.get("avg_amount", np.nan)]),
                pd.Series([sender.get("amount_std", np.nan)]),
            ).iloc[0]
        ),
        "payment_format": request.payment_format,
        "cross_currency_flag": request.payment_currency != request.receiving_currency,
        "cross_bank_flag": _bank_of(request.sender_account_key)
        != _bank_of(request.receiver_account_key),
    }

    for side, stats in (("sender", sender), ("receiver", receiver)):
        for column, value in stats.items():
            row[f"{side}_{column}"] = 0.0 if pd.isna(value) else float(value)

    embeddings_missing = False
    for side, key in (
        ("sender", request.sender_account_key),
        ("receiver", request.receiver_account_key),
    ):
        embedding = artifacts.embeddings.reindex([key]).iloc[0]
        if embedding.isna().all():
            embeddings_missing = True
        for column, value in embedding.items():
            row[f"{side}_{column}"] = 0.0 if pd.isna(value) else float(value)

    frame = pd.DataFrame([row])
    for column, categories in artifacts.categorical_values.items():
        frame[column] = pd.Categorical(frame[column], categories=categories)

    # Reindex to the trained column order - a mismatch here is the classic
    # silent serving bug, so it is enforced rather than assumed.
    return frame.reindex(columns=artifacts.feature_columns), embeddings_missing


def explain_row(features: pd.DataFrame, artifacts: ScoringArtifacts) -> list[ExplanationFactor]:
    contributions = artifacts.explainer.shap_values(features)[0]
    ranked = sorted(
        range(len(artifacts.feature_columns)),
        key=lambda i: abs(contributions[i]),
        reverse=True,
    )[:TOP_K_FACTORS]
    return [
        ExplanationFactor(
            feature=artifacts.feature_columns[i],
            contribution=float(contributions[i]),
            plain=FEATURE_DESCRIPTIONS.get(
                artifacts.feature_columns[i],
                _describe_embedding(artifacts.feature_columns[i]),
            ),
        )
        for i in ranked
    ]


def _describe_embedding(feature: str) -> str:
    """Embedding dimensions have no individual human meaning - saying so is
    more honest than inventing one."""
    if "_emb_" in feature:
        side = "sender" if feature.startswith("sender") else "receiver"
        return f"Learned graph position of the {side} (money-flow network structure)"
    return feature


def score_transaction(request: ScoreRequest, artifacts: ScoringArtifacts) -> ScoreResponse:
    started = time.perf_counter()

    features, embeddings_missing = build_feature_row(request, artifacts)
    score = float(artifacts.model.predict_proba(features)[0, 1])
    explanation = explain_row(features, artifacts)

    return ScoreResponse(
        tx_id=request.tx_id,
        score=score,
        is_flagged=score >= get_settings().flag_threshold,
        model_version=artifacts.model_version,
        embedding_version=(
            MISSING_EMBEDDING_VERSION if embeddings_missing else artifacts.embedding_version
        ),
        explanation=explanation,
        latency_ms=int((time.perf_counter() - started) * 1000),
        degraded=embeddings_missing,
    )

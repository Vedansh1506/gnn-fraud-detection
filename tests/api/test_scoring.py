"""Serving-path tests, including the train/serve consistency check TRD 7.1
calls the guard against "the #1 silent ML bug".

These build their own tiny artifacts rather than depending on the real trained
model, so they stay fast and run without a prior training run.
"""

from datetime import datetime

import pandas as pd
import pytest
import shap
import xgboost as xgb

from src.api.schemas import ScoreRequest
from src.api.scoring import (
    MISSING_EMBEDDING_VERSION,
    ScoringArtifacts,
    build_feature_row,
    score_transaction,
)
from src.features.account_features import fit_account_features
from src.models.classifier.dataset import build_features, feature_columns_with_embeddings

EMBEDDING_DIM = 2


def _transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tx_id": ["T1", "T2", "T3"],
            "sender_account_key": ["011_A", "011_A", "020_B"],
            "receiver_account_key": ["020_B", "030_C", "011_A"],
            "Timestamp": pd.to_datetime(
                ["2022-09-01 10:00", "2022-09-01 14:00", "2022-09-02 09:00"]
            ),
            "Amount Paid": [100.0, 300.0, 50.0],
            "Amount Received": [100.0, 300.0, 50.0],
            "Payment Currency": ["US Dollar", "US Dollar", "Euro"],
            "Receiving Currency": ["US Dollar", "Euro", "Euro"],
            "Payment Format": ["ACH", "Wire", "ACH"],
            "From Bank": ["011", "011", "020"],
            "To Bank": ["020", "030", "011"],
            "Is Laundering": [0, 0, 1],
        }
    )


def _embeddings(account_keys: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "account_key": account_keys,
            **{
                f"emb_{i}": [0.1 * (i + 1) * (j + 1) for j in range(len(account_keys))]
                for i in range(EMBEDDING_DIM)
            },
        }
    )


@pytest.fixture
def artifacts() -> ScoringArtifacts:
    df = _transactions()
    account_features = fit_account_features(df)
    embeddings = _embeddings(account_features["account_key"].tolist())
    columns = feature_columns_with_embeddings(embeddings)

    x = build_features(df, account_features, embeddings)[columns]
    model = xgb.XGBClassifier(n_estimators=5, max_depth=2, enable_categorical=True)
    model.fit(x, df["Is Laundering"])

    return ScoringArtifacts(
        model=model,
        explainer=shap.TreeExplainer(model),
        feature_columns=columns,
        categorical_values={"payment_format": list(map(str, x["payment_format"].cat.categories))},
        account_features=account_features.set_index("account_key"),
        embeddings=embeddings.set_index("account_key"),
        model_version="test_v1",
        embedding_version="test_emb_v1",
    )


def _request(**overrides) -> ScoreRequest:
    payload = {
        "tx_id": "T1",
        "sender_account_key": "011_A",
        "receiver_account_key": "020_B",
        "amount_paid": 100.0,
        "payment_currency": "US Dollar",
        "amount_received": 100.0,
        "receiving_currency": "US Dollar",
        "payment_format": "ACH",
        "timestamp": datetime(2022, 9, 1, 10, 0),
    }
    return ScoreRequest(**{**payload, **overrides})


def test_serving_features_match_training_features(artifacts):
    """The load-bearing test: the same transaction must produce the same feature
    vector through the serving path as through the training pipeline."""
    df = _transactions()
    account_features = fit_account_features(df)
    embeddings = artifacts.embeddings.reset_index()
    training_row = build_features(df, account_features, embeddings)[artifacts.feature_columns].iloc[
        [0]
    ]

    serving_row, _ = build_feature_row(_request(), artifacts)

    assert list(serving_row.columns) == list(training_row.columns)
    for column in artifacts.feature_columns:
        training_value = training_row[column].iloc[0]
        serving_value = serving_row[column].iloc[0]
        if column == "payment_format":
            assert str(training_value) == str(serving_value)
        else:
            assert float(training_value) == pytest.approx(float(serving_value)), column


def test_column_order_matches_the_trained_spec(artifacts):
    row, _ = build_feature_row(_request(), artifacts)
    assert list(row.columns) == artifacts.feature_columns


def test_unknown_account_degrades_instead_of_failing(artifacts):
    """An account with no embedding scores on neutral defaults and says so,
    rather than erroring - the locked graceful-degradation rule."""
    response = score_transaction(_request(sender_account_key="999_UNSEEN"), artifacts)

    assert response.degraded is True
    assert response.embedding_version == MISSING_EMBEDDING_VERSION
    assert 0.0 <= response.score <= 1.0


def test_known_accounts_are_not_marked_degraded(artifacts):
    response = score_transaction(_request(), artifacts)
    assert response.degraded is False
    assert response.embedding_version == "test_emb_v1"


def test_response_carries_pinned_versions_and_explanation(artifacts):
    response = score_transaction(_request(), artifacts)

    assert response.model_version == "test_v1"
    assert response.tx_id == "T1"
    assert len(response.explanation) > 0
    # Every factor needs human-readable wording - the dashboard shows these.
    assert all(factor.plain for factor in response.explanation)
    assert response.latency_ms >= 0


def test_cross_bank_flag_derived_from_account_key_prefix(artifacts):
    """The event schema carries no bank field, so it comes out of the
    "<bank>_<account>" key format."""
    same_bank, _ = build_feature_row(
        _request(sender_account_key="011_A", receiver_account_key="011_A"), artifacts
    )
    cross_bank, _ = build_feature_row(
        _request(sender_account_key="011_A", receiver_account_key="020_B"), artifacts
    )

    assert bool(same_bank["cross_bank_flag"].iloc[0]) is False
    assert bool(cross_bank["cross_bank_flag"].iloc[0]) is True

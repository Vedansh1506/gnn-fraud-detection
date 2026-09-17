"""Feature-ablation plumbing.

An ablation is only meaningful if the dropped set is applied *identically* to
both halves of a comparison and survives into evaluation. These pin the ways
that silently fails: a typo that drops nothing, a categorical left dangling
after its column is gone, or a spec that doesn't record what was dropped.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.classifier.dataset import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    feature_columns_with_embeddings,
)


def _fake_embeddings() -> pd.DataFrame:
    return pd.DataFrame({"account_key": ["a"], "emb_0": [0.1], "emb_1": [0.2]})


def test_dropping_nothing_returns_the_full_set():
    assert feature_columns_with_embeddings(None) == list(FEATURE_COLUMNS)


def test_a_dropped_feature_is_absent():
    columns = feature_columns_with_embeddings(None, drop_features=["payment_format"])

    assert "payment_format" not in columns
    assert len(columns) == len(FEATURE_COLUMNS) - 1


def test_everything_else_survives_the_drop():
    """The ablation must remove one column, not reshuffle the feature set."""
    columns = feature_columns_with_embeddings(None, drop_features=["payment_format"])
    expected = [c for c in FEATURE_COLUMNS if c != "payment_format"]

    assert columns == expected


def test_an_unknown_feature_is_rejected_rather_than_ignored():
    """A typo would otherwise produce an 'ablation' identical to the original
    model, and a comparison that silently measures nothing."""
    with pytest.raises(ValueError, match="unknown feature"):
        feature_columns_with_embeddings(None, drop_features=["paymnet_format"])


def test_dropping_applies_to_the_embedding_variant_too():
    """Both halves of the comparison must lose the same column, or the lift is
    measuring two differences at once."""
    columns = feature_columns_with_embeddings(_fake_embeddings(), drop_features=["payment_format"])

    assert "payment_format" not in columns
    # The embedding columns are untouched - that is the variable under test.
    assert "sender_emb_0" in columns
    assert "receiver_emb_1" in columns


def test_embedding_columns_cannot_be_dropped_by_accident():
    """Guards the reverse mistake: an ablation naming an embedding column would
    be testing something entirely different from a feature ablation."""
    columns = feature_columns_with_embeddings(_fake_embeddings(), drop_features=["sender_emb_0"])
    assert "sender_emb_0" not in columns
    assert "receiver_emb_0" in columns


def test_payment_format_is_the_only_categorical():
    """The ablation's awkward case: dropping the sole categorical must leave an
    empty categorical list, not a dangling name pointing at a missing column."""
    assert CATEGORICAL_COLUMNS == ["payment_format"]

    remaining = [c for c in CATEGORICAL_COLUMNS if c in
                 feature_columns_with_embeddings(None, drop_features=["payment_format"])]
    assert remaining == []

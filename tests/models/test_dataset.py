import pandas as pd
import pytest

from src.features.account_features import fit_account_features
from src.models.classifier.dataset import (
    FEATURE_COLUMNS,
    Dataset,
    build_features,
    split_by_time,
)


def _transactions() -> pd.DataFrame:
    """Spans the train (09-01..09-07), val (09-08) and test (09-09+) windows."""
    rows = [
        ("T0", "A", "B", "2022-09-01 00:00", 100.0, 0),
        ("T1", "A", "C", "2022-09-02 00:00", 300.0, 0),
        ("T2", "B", "A", "2022-09-03 00:00", 50.0, 1),
        ("T3", "A", "B", "2022-09-08 12:00", 200.0, 0),  # val
        ("T4", "C", "D", "2022-09-10 00:00", 400.0, 1),  # test
        ("T5", "Z", "Y", "2022-09-11 00:00", 500.0, 0),  # test, unseen accounts
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "tx_id",
            "sender_account_key",
            "receiver_account_key",
            "Timestamp",
            "Amount Paid",
            "Is Laundering",
        ],
    )
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Amount Received"] = df["Amount Paid"]
    df["Payment Currency"] = "US Dollar"
    df["Receiving Currency"] = "US Dollar"
    df["Payment Format"] = "ACH"
    df["From Bank"] = "011"
    df["To Bank"] = "020"
    return df


def test_split_is_chronological_disjoint_and_complete():
    train, val, test = split_by_time(_transactions())

    assert list(train["tx_id"]) == ["T0", "T1", "T2"]
    assert list(val["tx_id"]) == ["T3"]
    assert list(test["tx_id"]) == ["T4", "T5"]

    assert train["Timestamp"].max() < val["Timestamp"].min()
    assert val["Timestamp"].max() < test["Timestamp"].min()
    assert len(train) + len(val) + len(test) == len(_transactions())


def test_account_stats_fitted_on_train_ignore_later_rows():
    """The leakage guard: fitting on train must give identical statistics
    whether or not val/test rows exist in the source frame."""
    df = _transactions()
    train, _, _ = split_by_time(df)

    from_train_only = fit_account_features(train)
    from_full_frame_train_slice = fit_account_features(split_by_time(df)[0])

    pd.testing.assert_frame_equal(from_train_only, from_full_frame_train_slice)

    # Account A sends 100 and 300 in train; its 200.0 val send must not count.
    account_a = from_train_only.set_index("account_key").loc["A"]
    assert account_a["avg_amount"] == pytest.approx(200.0)
    assert account_a["out_degree"] == 2


def test_unseen_accounts_get_neutral_defaults():
    df = _transactions()
    train, _, test = split_by_time(df)
    account_features = fit_account_features(train)

    features = build_features(test, account_features)
    unseen = features[features["tx_id"] == "T5"].iloc[0]

    # Z and Y never appear in training - every account feature defaults to 0,
    # and the z-score has no distribution to measure against.
    assert unseen["sender_in_degree"] == 0
    assert unseen["sender_out_degree"] == 0
    assert unseen["receiver_distinct_counterparties"] == 0
    assert unseen["amount_zscore"] == 0.0


def test_build_features_produces_the_declared_feature_columns():
    df = _transactions()
    train, _, _ = split_by_time(df)
    features = build_features(train, fit_account_features(train))
    assert set(FEATURE_COLUMNS).issubset(features.columns)


def test_scale_pos_weight_uses_train_split_only():
    y_train = pd.Series([0] * 98 + [1, 1])
    dataset = Dataset(
        x_train=pd.DataFrame(),
        y_train=y_train,
        x_val=pd.DataFrame(),
        y_val=pd.Series(dtype=int),
        x_test=pd.DataFrame(),
        y_test=pd.Series([1] * 50),  # a wildly different test balance...
        account_features=pd.DataFrame(),
    )
    assert dataset.scale_pos_weight == pytest.approx(49.0)  # ...must not affect it

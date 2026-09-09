import pandas as pd
import pytest

from src.features.account_features import fit_account_features, join_account_features


def _sample_df() -> pd.DataFrame:
    # A -> B (100), A -> C (300) on day 1; B -> A (50) on day 1; A -> B (200) on day 3.
    return pd.DataFrame(
        {
            "sender_account_key": ["A", "A", "B", "A"],
            "receiver_account_key": ["B", "C", "A", "B"],
            "Timestamp": pd.to_datetime(
                ["2022-09-01 00:00", "2022-09-01 12:00", "2022-09-01 23:00", "2022-09-03 00:00"]
            ),
            "Amount Paid": [100.0, 300.0, 50.0, 200.0],
        }
    )


def _row(result: pd.DataFrame, account_key: str) -> pd.Series:
    return result.set_index("account_key").loc[account_key]


def test_degrees_and_counterparties():
    result = fit_account_features(_sample_df())
    a = _row(result, "A")
    assert a["out_degree"] == 3
    assert a["in_degree"] == 1
    assert a["distinct_counterparties"] == 2  # B and C

    c = _row(result, "C")
    assert c["out_degree"] == 0
    assert c["in_degree"] == 1


def test_avg_amount_is_average_sent():
    result = fit_account_features(_sample_df())
    assert _row(result, "A")["avg_amount"] == pytest.approx(200.0)  # (100+300+200)/3
    assert _row(result, "B")["avg_amount"] == pytest.approx(50.0)


def test_avg_amount_zero_for_pure_receiver():
    # C only ever receives - never sends, so avg_amount defaults to 0.0, not NaN.
    assert _row(fit_account_features(_sample_df()), "C")["avg_amount"] == 0.0


def test_amount_std_undefined_stays_nan():
    """Deliberately not filled with 0: a 0 std would divide to infinity in the
    z-score, whereas NaN signals "no usable spread" and collapses to 0.0."""
    result = fit_account_features(_sample_df())
    assert pd.isna(_row(result, "B")["amount_std"])  # single send
    assert pd.isna(_row(result, "C")["amount_std"])  # never sends
    assert _row(result, "A")["amount_std"] > 0


def test_tx_count_24h_is_trailing_window_as_of_last_activity():
    # A's last event is 2022-09-03 00:00; only that event itself falls within
    # the trailing 24h window ending there (the day-1 cluster is >24h earlier).
    result = fit_account_features(_sample_df())
    assert _row(result, "A")["tx_count_24h"] == 1
    assert _row(result, "B")["tx_count_24h"] == 1
    assert _row(result, "C")["tx_count_24h"] == 1


def test_join_prefixes_columns_and_defaults_unknown_accounts():
    fitted = fit_account_features(_sample_df())
    transactions = pd.DataFrame(
        {"sender_account_key": ["A", "UNKNOWN"], "receiver_account_key": ["B", "B"]}
    )

    joined = join_account_features(transactions, fitted, "sender")

    assert joined.loc[0, "sender_out_degree"] == 3
    assert joined.loc[1, "sender_out_degree"] == 0  # unseen -> neutral default
    assert "sender_avg_amount" in joined.columns

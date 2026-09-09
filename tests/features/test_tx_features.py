import pandas as pd
import pytest

from src.features.account_features import fit_account_features
from src.features.tx_features import compute_tx_features


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tx_id": ["T1", "T2", "T3"],
            "sender_account_key": ["A", "A", "B"],
            "receiver_account_key": ["B", "C", "A"],
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


def _features() -> pd.DataFrame:
    df = _sample_df()
    return compute_tx_features(df, fit_account_features(df))


def test_hour_of_day():
    assert list(_features()["hour_of_day"]) == [10, 14, 9]


def test_cross_currency_flag():
    assert list(_features()["cross_currency_flag"]) == [False, True, False]


def test_cross_bank_flag():
    assert list(_features()["cross_bank_flag"]) == [True, True, True]


def test_amount_zscore_is_per_sending_account():
    # Account A sent two tx (100, 300): mean=200, sample std (ddof=1)=141.42...
    result = _features().set_index("tx_id")
    assert result.loc["T1", "amount_zscore"] == pytest.approx(-0.7071067811865475)
    assert result.loc["T2", "amount_zscore"] == pytest.approx(0.7071067811865475)


def test_amount_zscore_single_tx_account_defaults_to_zero():
    # Account B has only one outgoing transaction -> std is undefined, not zero.
    assert _features().set_index("tx_id").loc["T3", "amount_zscore"] == 0.0


def test_amount_zscore_uses_supplied_stats_not_the_frames_own():
    """The leakage fix: statistics come from the fitted table passed in, so a
    row can be scored against a window it wasn't part of."""
    df = _sample_df()
    fitted = fit_account_features(df)
    # Pretend training saw a much larger average for account A.
    fitted.loc[fitted["account_key"] == "A", ["avg_amount", "amount_std"]] = [1000.0, 100.0]

    result = compute_tx_features(df, fitted).set_index("tx_id")
    assert result.loc["T1", "amount_zscore"] == pytest.approx((100.0 - 1000.0) / 100.0)


def test_amount_zscore_zero_when_account_unseen_in_fitted_stats():
    df = _sample_df()
    fitted = fit_account_features(df)
    fitted = fitted[fitted["account_key"] != "A"]  # A now unknown

    result = compute_tx_features(df, fitted).set_index("tx_id")
    assert result.loc["T1", "amount_zscore"] == 0.0

"""Transaction-level features (TRD 4.2 tx_features view), keyed by tx_id.

Pure functions over a DataFrame from src.data.load_amlworld.load_transactions -
no I/O, no Neo4j/Feast dependency, so the identical logic can later be called
by both the offline batch pipeline and the real-time streaming consumer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TX_FEATURE_COLUMNS = [
    "amount_paid",
    "amount_received",
    "hour_of_day",
    "amount_zscore",
    "payment_format",
    "cross_currency_flag",
    "cross_bank_flag",
]


def compute_tx_features(df: pd.DataFrame, account_features: pd.DataFrame) -> pd.DataFrame:
    """Per-transaction features, keyed by tx_id.

    amount_zscore is measured against the *sending account's* own history, not
    a global distribution - TRD 5.1's own /score example describes the feature
    as "Amount is unusually high for THIS account". The mean/std come from the
    fitted account table (training window only) rather than from `df`, so a
    test-period transaction is scored against training-period behaviour and
    no future information leaks backwards.
    """
    stats = account_features.set_index("account_key")
    mean = df["sender_account_key"].map(stats["avg_amount"])
    std = df["sender_account_key"].map(stats["amount_std"])

    return pd.DataFrame(
        {
            "tx_id": df["tx_id"],
            "amount_paid": df["Amount Paid"],
            "amount_received": df["Amount Received"],
            "hour_of_day": df["Timestamp"].dt.hour,
            "amount_zscore": _safe_zscore(df["Amount Paid"], mean, std),
            "payment_format": df["Payment Format"],
            # Payment/Receiving Currency are category dtype (memory-efficient
            # on the full file) with independently-built category sets, so
            # pandas refuses a direct Categorical != Categorical comparison
            # even though the values compare fine as plain strings.
            "cross_currency_flag": (
                df["Payment Currency"].astype(str) != df["Receiving Currency"].astype(str)
            ),
            "cross_bank_flag": df["From Bank"] != df["To Bank"],
        }
    )


def _safe_zscore(values: pd.Series, mean: pd.Series, std: pd.Series) -> pd.Series:
    """0.0 wherever the z-score isn't defined.

    Three cases collapse to "no usable signal": an account unseen in training
    (mean/std NaN), one with a single sent transaction (std NaN), and one whose
    amounts never varied (std 0, which would otherwise divide to +/-inf).
    """
    zscore = (values - mean) / std
    return zscore.replace([np.inf, -np.inf], 0.0).fillna(0.0)

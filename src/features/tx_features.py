"""Transaction-level features (TRD 4.2 tx_features view), keyed by tx_id.

Pure function over a DataFrame from src.data.load_amlworld.load_transactions -
no I/O, no Neo4j/Feast dependency, so the identical logic can later be called
by both the offline batch loader and the real-time streaming consumer.
"""

from __future__ import annotations

import pandas as pd


def compute_tx_features(df: pd.DataFrame) -> pd.DataFrame:
    """amount_zscore is per sending-account, not global.

    TRD's own /score example describes it as "Amount is unusually high for
    THIS account" (TRD 5.1) - a global z-score wouldn't match that framing.
    Accounts with too little history for a defined std (single transaction,
    or all-identical amounts) get 0.0 rather than NaN/inf.
    """
    grouped = df.groupby("sender_account_key")["Amount Paid"]
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    amount_zscore = ((df["Amount Paid"] - mean) / std).fillna(0.0)

    return pd.DataFrame(
        {
            "tx_id": df["tx_id"],
            "amount_paid": df["Amount Paid"],
            "amount_received": df["Amount Received"],
            "hour_of_day": df["Timestamp"].dt.hour,
            "amount_zscore": amount_zscore,
            "payment_format": df["Payment Format"],
            # Payment/Receiving Currency are category dtype (memory-efficient
            # for the full file) with independently-built category sets, so
            # pandas refuses a direct Categorical != Categorical comparison
            # ("Categoricals can only be compared if categories are the
            # same") even though the values compare fine as plain strings.
            "cross_currency_flag": (
                df["Payment Currency"].astype(str) != df["Receiving Currency"].astype(str)
            ),
            "cross_bank_flag": df["From Bank"] != df["To Bank"],
        }
    )

"""Account-level features (TRD 4.2 account_features view), keyed by account_key.

These are *fitted* on a training window and then applied forward to later
rows - never recomputed over data the model shouldn't have seen yet. That
mirrors the locked architecture (account features/embeddings are batch-built
offline per work-session and only read at serving time) and keeps the
train/serve contract honest: at scoring time you can only know the past.
"""

from __future__ import annotations

import pandas as pd

# Columns a caller can rely on being present after a fit, in this order.
ACCOUNT_FEATURE_COLUMNS = [
    "in_degree",
    "out_degree",
    "distinct_counterparties",
    "avg_amount",
    "amount_std",
    "tx_count_24h",
]


def fit_account_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-account feature table from the training window only.

    avg_amount is the account's average amount SENT, and amount_std its
    standard deviation (used by tx_features for amount_zscore). Pure-receiver
    accounts that never send get avg_amount 0.0; amount_std is deliberately
    left NaN when undefined (fewer than 2 sent transactions) so the z-score
    calculation can recognise "no usable spread" rather than dividing by zero.
    """
    events = _build_events(train_df)
    sent = events.loc[events["role"] == "sent"]

    in_degree = events.loc[events["role"] == "received"].groupby("account_key").size()
    out_degree = sent.groupby("account_key").size()
    distinct_counterparties = events.groupby("account_key")["counterparty_key"].nunique()
    avg_amount = sent.groupby("account_key")["amount"].mean()
    amount_std = sent.groupby("account_key")["amount"].std()
    tx_count_24h = _trailing_24h_count_at_last_activity(events)

    result = pd.DataFrame(
        {
            "in_degree": in_degree,
            "out_degree": out_degree,
            "distinct_counterparties": distinct_counterparties,
            "avg_amount": avg_amount,
            "amount_std": amount_std,
            "tx_count_24h": tx_count_24h,
        }
    )
    # amount_std stays NaN where undefined; everything else defaults to 0.
    result[[c for c in ACCOUNT_FEATURE_COLUMNS if c != "amount_std"]] = result[
        [c for c in ACCOUNT_FEATURE_COLUMNS if c != "amount_std"]
    ].fillna(0)
    result.index.name = "account_key"
    return result.reset_index()


def _build_events(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (account, transaction leg) - each transaction contributes a
    'sent' event for its sender and a 'received' event for its receiver."""
    sent = df[["sender_account_key", "Timestamp", "receiver_account_key", "Amount Paid"]].rename(
        columns={
            "sender_account_key": "account_key",
            "receiver_account_key": "counterparty_key",
            "Amount Paid": "amount",
        }
    )
    sent["role"] = "sent"
    received_cols = ["receiver_account_key", "Timestamp", "sender_account_key", "Amount Paid"]
    received = df[received_cols].rename(
        columns={
            "receiver_account_key": "account_key",
            "sender_account_key": "counterparty_key",
            "Amount Paid": "amount",
        }
    )
    received["role"] = "received"
    return pd.concat([sent, received], ignore_index=True)


def _trailing_24h_count_at_last_activity(events: pd.DataFrame) -> pd.Series:
    """Count of each account's events within 24h of its own most recent one.

    Equivalent to a per-event groupby().rolling("24h").count() keeping only the
    last value per account - computed directly instead, since pandas'
    groupby+rolling scales badly here: ~700k mostly single-digit-size account
    groups is the "many small groups" case it's known to be slow on (~34s vs
    ~6s measured on the full dataset).
    """
    last_time = events.groupby("account_key")["Timestamp"].transform("max")
    in_window = events["Timestamp"] > (last_time - pd.Timedelta("24h"))
    return events.loc[in_window].groupby("account_key").size()


def join_account_features(
    df: pd.DataFrame, account_features: pd.DataFrame, side: str
) -> pd.DataFrame:
    """Left-join the fitted account table onto transactions for one side.

    `side` is "sender" or "receiver"; columns come back prefixed accordingly.
    Accounts absent from the fitted table (never seen during training) get
    neutral zero defaults - the same thing that happens at serving time for an
    account with no history, per the locked graceful-degradation rule.
    """
    key_column = f"{side}_account_key"
    renamed = account_features.rename(
        columns={c: f"{side}_{c}" for c in ACCOUNT_FEATURE_COLUMNS}
    )
    joined = df.merge(
        renamed, how="left", left_on=key_column, right_on="account_key", suffixes=("", "_acct")
    )
    prefixed = [f"{side}_{c}" for c in ACCOUNT_FEATURE_COLUMNS]
    joined[prefixed] = joined[prefixed].fillna(0)
    return joined.drop(columns=["account_key"])

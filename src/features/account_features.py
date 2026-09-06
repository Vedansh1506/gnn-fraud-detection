"""Account-level features (TRD 4.2 account_features view), keyed by account_key.

Pure function over a DataFrame from src.data.load_amlworld.load_transactions -
no I/O, no Neo4j/Feast dependency (see tx_features.py for why that matters).
"""

from __future__ import annotations

import pandas as pd


def compute_account_features(df: pd.DataFrame) -> pd.DataFrame:
    """avg_amount is the account's average amount SENT (0.0 for pure-receiver
    accounts that never send, rather than NaN - a receiver with no outgoing
    activity legitimately has nothing to average).

    tx_count_24h is the odd one out: TRD lists it as a per-account_key field,
    but it's inherently a point-in-time rolling quantity (how active was this
    account in the trailing 24h as of a given moment) - the same definition a
    live streaming consumer would evaluate "as of now" per event. For this
    static training snapshot, each account's *most recent* such value (as of
    its own last transaction) is used - see _trailing_24h_count_at_last_activity,
    whose "count of events in (last_time - 24h, last_time]" definition is what
    stays consistent between batch and a future streaming consumer, not any
    particular pandas implementation of it.
    """
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
    events = pd.concat([sent, received], ignore_index=True)

    in_degree = events.loc[events["role"] == "received"].groupby("account_key").size()
    out_degree = events.loc[events["role"] == "sent"].groupby("account_key").size()
    distinct_counterparties = events.groupby("account_key")["counterparty_key"].nunique()
    avg_amount = sent.groupby("account_key")["amount"].mean()
    tx_count_24h = _trailing_24h_count_at_last_activity(events)

    result = pd.DataFrame(
        {
            "in_degree": in_degree,
            "out_degree": out_degree,
            "distinct_counterparties": distinct_counterparties,
            "avg_amount": avg_amount,
            "tx_count_24h": tx_count_24h,
        }
    ).fillna(0)
    result.index.name = "account_key"
    return result.reset_index()


def _trailing_24h_count_at_last_activity(events: pd.DataFrame) -> pd.Series:
    """Count of each account's events within 24h of its own most recent one.

    Equivalent to a per-event groupby().rolling("24h").count() with only the
    last (most recent) value per account kept - but computed directly instead
    of materializing the full rolling series, since pandas' groupby+rolling
    scales badly here: ~700k mostly single-digit-size account groups is the
    "many small groups" case it's known to be slow on (~34s vs ~6s measured
    on the full ~6.9M-row/13.8M-event dataset).
    """
    last_time = events.groupby("account_key")["Timestamp"].transform("max")
    in_window = events["Timestamp"] > (last_time - pd.Timedelta("24h"))
    return events.loc[in_window].groupby("account_key").size()

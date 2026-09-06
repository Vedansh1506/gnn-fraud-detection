"""Full-column loader for the IBM AML LI-Small transaction file.

Extends the dtype-safety fix already used in check_graph_richness.py (bank
codes read as str to preserve leading zeros) to every column the graph
loader and feature engineering modules need.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_TRANSACTIONS_PATH = Path("data/raw/LI-Small_Trans.csv")

_DTYPES = {
    "From Bank": str,
    "Account": str,
    "To Bank": str,
    "Account.1": str,
    "Receiving Currency": "category",
    "Payment Currency": "category",
    "Payment Format": "category",
    "Is Laundering": "int8",
}


def load_transactions(
    path: Path = DEFAULT_TRANSACTIONS_PATH, limit: int | None = None
) -> pd.DataFrame:
    """Load transactions with derived account_key and tx_id columns.

    tx_id has no source column in the raw dataset, so it is synthesized from
    each row's position in the (static, always-read-in-order) source file -
    "T00000001" etc. This is stable across reruns, which is what keeps the
    Neo4j MERGE upserts in build_graph.py idempotent (TRD 6: idempotent
    replay on tx_id/account_key).
    """
    df = pd.read_csv(
        path,
        dtype=_DTYPES,
        parse_dates=["Timestamp"],
        date_format="%Y/%m/%d %H:%M",
        nrows=limit,
    )
    df["tx_id"] = [f"T{i:08d}" for i in range(len(df))]
    df["sender_account_key"] = df["From Bank"] + "_" + df["Account"]
    df["receiver_account_key"] = df["To Bank"] + "_" + df["Account.1"]
    return df

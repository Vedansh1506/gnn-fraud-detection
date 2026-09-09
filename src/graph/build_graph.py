"""Batch-load the historical LI-Small file into Neo4j.

    uv run python -m src.graph.build_graph [--limit N] [--batch-size N]

Offline/historical bootstrap only (see docs/EXECUTION-GUIDE.md Build Loop step 1
and the approved plan) - the live streaming consumer (SAD C3) is a later step.

Feature generation deliberately does NOT live here: features must be fitted on
a training window to stay leakage-free, which only the training pipeline knows
about, so src/models/classifier/dataset.py owns it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_amlworld import DEFAULT_TRANSACTIONS_PATH, load_transactions
from src.graph.client import session_scope
from src.graph.schema import ensure_schema

DEFAULT_BATCH_SIZE = 5000

_ACCOUNT_UPSERT = (
    "UNWIND $batch AS row "
    "MERGE (a:Account {account_key: row.account_key}) "
    "ON CREATE SET a.first_seen = row.first_seen"
)

# ON CREATE (not plain SET): tx_id is immutable historical fact, and a later
# scoring-service write to t.score/t.is_flagged must never be clobbered by a
# rerun of this loader (TRD 6: idempotent replay).
_TRANSACTED_UPSERT = (
    "UNWIND $batch AS row "
    "MATCH (s:Account {account_key: row.sender_account_key}) "
    "MATCH (r:Account {account_key: row.receiver_account_key}) "
    "MERGE (s)-[t:TRANSACTED {tx_id: row.tx_id}]->(r) "
    "ON CREATE SET "
    "  t.timestamp = row.timestamp, "
    "  t.amount_paid = row.amount_paid, "
    "  t.paid_currency = row.paid_currency, "
    "  t.amount_received = row.amount_received, "
    "  t.received_currency = row.received_currency, "
    "  t.payment_format = row.payment_format, "
    "  t.is_laundering = row.is_laundering"
)


def _batched(records: list[dict], batch_size: int):
    for i in range(0, len(records), batch_size):
        yield records[i : i + batch_size]


def _prepare_account_records(df: pd.DataFrame) -> list[dict]:
    sent = df[["sender_account_key", "Timestamp"]].rename(
        columns={"sender_account_key": "account_key"}
    )
    received = df[["receiver_account_key", "Timestamp"]].rename(
        columns={"receiver_account_key": "account_key"}
    )
    first_seen = pd.concat([sent, received]).groupby("account_key")["Timestamp"].min().reset_index()
    # np.array(...) pins today's return type (an ndarray of datetime.datetime)
    # against a pending pandas change - see the FutureWarning on .dt.to_pydatetime.
    first_seen["Timestamp"] = np.array(first_seen["Timestamp"].dt.to_pydatetime())
    return first_seen.rename(columns={"Timestamp": "first_seen"}).to_dict("records")


def _prepare_transaction_records(df: pd.DataFrame) -> list[dict]:
    # Cast explicitly to native Python types the Bolt protocol accepts -
    # category/int8 dtypes from load_amlworld's dtype-safety pass don't
    # serialize directly as Cypher parameters.
    prepared = pd.DataFrame(
        {
            "tx_id": df["tx_id"],
            "sender_account_key": df["sender_account_key"],
            "receiver_account_key": df["receiver_account_key"],
            "timestamp": np.array(df["Timestamp"].dt.to_pydatetime()),
            "amount_paid": df["Amount Paid"].astype(float),
            "paid_currency": df["Payment Currency"].astype(str),
            "amount_received": df["Amount Received"].astype(float),
            "received_currency": df["Receiving Currency"].astype(str),
            "payment_format": df["Payment Format"].astype(str),
            "is_laundering": df["Is Laundering"].astype(bool),
        }
    )
    return prepared.to_dict("records")


def _upsert_accounts(records: list[dict], batch_size: int) -> None:
    print(f"Upserting {len(records):,} accounts...")
    with session_scope() as session:
        for batch in _batched(records, batch_size):
            session.execute_write(lambda tx, b=batch: tx.run(_ACCOUNT_UPSERT, batch=b))


def _upsert_transactions(records: list[dict], batch_size: int) -> None:
    print(f"Upserting {len(records):,} transactions...")
    total = len(records)
    with session_scope() as session:
        for i, batch in enumerate(_batched(records, batch_size)):
            session.execute_write(lambda tx, b=batch: tx.run(_TRANSACTED_UPSERT, batch=b))
            done = min((i + 1) * batch_size, total)
            if (i + 1) % 20 == 0 or done == total:
                print(f"  ... {done:,} / {total:,}")


def build_graph(path: Path, limit: int | None, batch_size: int) -> None:
    ensure_schema()
    print(f"Loading {path} ...")
    df = load_transactions(path, limit=limit)
    print(f"Loaded {len(df):,} transactions.")

    _upsert_accounts(_prepare_account_records(df), batch_size)
    _upsert_transactions(_prepare_transaction_records(df), batch_size)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_TRANSACTIONS_PATH)
    parser.add_argument(
        "--limit", type=int, default=None, help="Load only the first N rows (fast iteration)."
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    build_graph(args.path, args.limit, args.batch_size)

"""Replays historical transactions onto the stream as events (SAD C1).

    uv run python -m src.streaming.producer --limit 500 --rate 20

Replays the **test window** by default (2022-09-09 onward). That matters: the
model never saw those transactions during training, so a demo replaying them
shows real scoring rather than recall of memorised rows. Replaying the train
window would look better and mean nothing.

Events match the TRD 4.4 stream schema exactly, so the consumer can forward
one to POST /score without reshaping it.

Replay is **chronological**. The source file is only roughly time-ordered, and
taking rows in file order lands in a laundering-dense region - the first 300
rows that way are 51% laundering against a true rate of ~0.07%, which would
make a demo look absurd. Sorted by timestamp, the first 300 contain 1
laundering transaction, which is the honest base rate: laundering is a needle
in a haystack here, and that is precisely why AUPRC is the headline metric. A
short replay may legitimately flag nothing.
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from src.common.config import get_settings
from src.data.load_amlworld import DEFAULT_TRANSACTIONS_PATH, load_transactions
from src.models.classifier.dataset import split_by_time
from src.streaming.client import build_producer

DEFAULT_RATE = 20.0


def to_event(row: pd.Series) -> dict:
    """One transaction as a TRD 4.4 stream event."""
    return {
        "tx_id": row["tx_id"],
        "sender_account_key": row["sender_account_key"],
        "receiver_account_key": row["receiver_account_key"],
        "amount_paid": float(row["Amount Paid"]),
        "payment_currency": str(row["Payment Currency"]),
        "amount_received": float(row["Amount Received"]),
        "receiving_currency": str(row["Receiving Currency"]),
        "payment_format": str(row["Payment Format"]),
        "timestamp": row["Timestamp"].isoformat(),
    }


def replay(frame: pd.DataFrame, rate: float) -> int:
    settings = get_settings()
    producer = build_producer(**{"message.timeout.ms": 10_000})
    topic = settings.transactions_topic
    interval = 1.0 / rate if rate > 0 else 0.0

    failures: list[str] = []

    def on_delivery(err, msg) -> None:
        # flush() returning is not proof of delivery - only the callback is.
        if err is not None:
            failures.append(str(err))

    sent = 0
    for _, row in frame.iterrows():
        event = to_event(row)
        producer.produce(
            topic,
            key=event["tx_id"].encode(),
            value=json.dumps(event).encode(),
            callback=on_delivery,
        )
        producer.poll(0)
        sent += 1
        if sent % 100 == 0:
            print(f"  ... produced {sent:,}/{len(frame):,}")
        if interval:
            time.sleep(interval)

    producer.flush(30)
    if failures:
        print(f"WARNING: {len(failures)} message(s) failed to deliver, e.g. {failures[0]}")
    return sent - len(failures)


def main(limit: int | None, rate: float, use_train_window: bool) -> None:
    _, _, test_df = split_by_time(load_transactions(DEFAULT_TRANSACTIONS_PATH))
    frame = test_df if not use_train_window else load_transactions(DEFAULT_TRANSACTIONS_PATH)
    # Chronological, not file order - see the module docstring.
    frame = frame.sort_values("Timestamp")
    if limit:
        frame = frame.head(limit)

    laundering = int(frame["Is Laundering"].sum())
    print(
        f"Replaying {len(frame):,} transactions ({laundering} laundering-labelled) "
        f"at {rate}/s onto '{get_settings().transactions_topic}'"
    )
    delivered = replay(frame, rate)
    print(f"Delivered {delivered:,} events.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=500, help="Events to replay (default 500).")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE, help="Events per second.")
    parser.add_argument(
        "--train-window",
        action="store_true",
        help="Replay from the start of the dataset instead of the held-out test window. "
        "Only for load testing - scores on data the model trained on are not meaningful.",
    )
    args = parser.parse_args()
    main(args.limit, args.rate, args.train_window)

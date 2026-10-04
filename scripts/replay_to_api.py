"""Replay held-out transactions straight into a running API's /score endpoint.

    uv run python -m scripts.replay_to_api --api-url https://demo.example.com --limit 40000

This is how the *cloud* demo gets a populated flag queue. Locally the stream is
Redpanda (producer -> broker -> consumer -> /score), and that path is unchanged.
In the cloud there is no broker: the free-plan account cannot use Kinesis, and
the consumer speaks the Kafka API rather than Kinesis's, so adopting it would
mean new code and a new failure point to demo. Posting events directly to
/score exercises the same scoring path and produces the same audit rows and
flags - what it skips is the broker hop and the consumer's Neo4j upsert.

The skipped upsert is harmless here: the graph is loaded ahead of time and
scoring reads precomputed artifacts, never the graph (rules.md: the offline /
online boundary). The cost is that the "streaming ingestion" claim is
demonstrated locally, not on the deployed box - say so if asked.

Replays the **test window** (2022-09-09 onward) chronologically, exactly like
the producer, for the same reason: the model never trained on it, and file
order lands in a laundering-dense region that would look absurd.

**How many events you need.** At the pinned threshold roughly 4.7 per 1,000
transactions flag (README), so a short replay legitimately flags almost nothing.
Plan on ~40,000 events for a couple of dozen open flags. There is deliberately
no option to bias the slice toward laundering: an earlier draft had one, and
running it showed it did nothing (the first laundering transaction in the
chronological test window is at row 108, so the slices were identical) while
its comment claimed a benefit nobody had measured.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import pandas as pd

from src.common.config import get_settings
from src.data.load_amlworld import DEFAULT_TRANSACTIONS_PATH, load_transactions
from src.models.classifier.dataset import split_by_time
from src.streaming.producer import to_event

TIMEOUT = 30.0
MAX_ATTEMPTS = 3


def select_window(test_df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """The slice to replay: the earliest `limit` transactions, chronologically.

    Pure so it can be tested without the 650 MB dataset.
    """
    frame = test_df.sort_values("Timestamp").reset_index(drop=True)
    return frame.head(limit) if limit else frame


def post_event(client: httpx.Client, url: str, api_key: str, event: dict) -> str:
    """Send one event; returns 'ok', 'client_error' or 'server_error'.

    Retries only transient failures. A 4xx will never succeed on retry (the
    event is malformed or auth is wrong), so retrying it would just hide the
    problem behind delay - the same rule the stream consumer follows.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.post(
                f"{url}/score", json=event, headers={"X-API-Key": api_key}, timeout=TIMEOUT
            )
        except httpx.HTTPError:
            if attempt == MAX_ATTEMPTS:
                return "server_error"
            time.sleep(0.5 * attempt)
            continue

        if response.status_code == 200:
            return "ok"
        if 400 <= response.status_code < 500:
            return "client_error"
        if attempt == MAX_ATTEMPTS:
            return "server_error"
        time.sleep(0.5 * attempt)

    return "server_error"


def replay(frame: pd.DataFrame, url: str, api_key: str, workers: int) -> dict[str, int]:
    events = [to_event(row) for _, row in frame.iterrows()]
    counts = {"ok": 0, "client_error": 0, "server_error": 0}
    started = time.perf_counter()

    with httpx.Client() as client, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(post_event, client, url, api_key, e) for e in events]
        for done, future in enumerate(as_completed(futures), start=1):
            counts[future.result()] += 1
            if done % 250 == 0:
                rate = done / (time.perf_counter() - started)
                print(f"  ... {done:,}/{len(events):,} scored ({rate:.0f}/s)")

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=None, help="API base URL (default: from config).")
    parser.add_argument("--limit", type=int, default=2000, help="Events to replay.")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent requests. Keep low against a small instance - scoring is CPU-bound.",
    )
    args = parser.parse_args()

    settings = get_settings()
    url = (args.api_url or settings.scoring_api_base_url).rstrip("/")

    _, _, test_df = split_by_time(load_transactions(DEFAULT_TRANSACTIONS_PATH))
    frame = select_window(test_df, args.limit)

    laundering = int(frame["Is Laundering"].sum())
    print(f"Replaying {len(frame):,} test-window transactions to {url}")
    print(f"  {laundering} laundering-labelled in this slice ({laundering / len(frame):.2%})")

    counts = replay(frame, url, settings.service_api_key, args.workers)

    print(
        f"\nscored {counts['ok']:,} | rejected {counts['client_error']:,} | "
        f"failed {counts['server_error']:,}"
    )
    if counts["client_error"]:
        print("Rejections are 4xx: check SERVICE_API_KEY and the event schema.")
    return 1 if counts["server_error"] or counts["client_error"] else 0


if __name__ == "__main__":
    sys.exit(main())

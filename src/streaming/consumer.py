"""Consumes transaction events, updates the graph, and gets each one scored
(SAD C3).

    uv run python -m src.streaming.consumer --max-events 500

Two responsibilities, both from TRD 5.2:
- **The consumer owns graph writes.** The scoring API is read-only on Neo4j, so
  this is the only thing that upserts accounts and :TRANSACTED edges live.
- **It calls POST /score with the static X-API-Key**, the service-to-service
  credential, rather than a user's JWT.

Offsets are committed only after an event is fully handled, so a crash
mid-event redelivers it instead of dropping it. Redelivery is safe: the graph
upserts are MERGE-based and already-scored tx_ids are skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import httpx
from confluent_kafka import KafkaError

from src.common.config import get_settings
from src.graph.client import session_scope
from src.streaming.client import build_consumer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("consumer")

CONSUMER_GROUP = "transaction-scorer"
SCORE_TIMEOUT_SECONDS = 10.0
SCORE_RETRIES = 3

# Same MERGE shape as the batch loader, so a replayed event and a bulk-loaded
# row converge on identical graph state (TRD 6: idempotent replay).
_UPSERT_TRANSACTION = (
    "MERGE (s:Account {account_key: $sender_account_key}) "
    "  ON CREATE SET s.first_seen = datetime($timestamp) "
    "MERGE (r:Account {account_key: $receiver_account_key}) "
    "  ON CREATE SET r.first_seen = datetime($timestamp) "
    "MERGE (s)-[t:TRANSACTED {tx_id: $tx_id}]->(r) "
    "  ON CREATE SET t.timestamp = datetime($timestamp), "
    "                t.amount_paid = $amount_paid, "
    "                t.paid_currency = $payment_currency, "
    "                t.amount_received = $amount_received, "
    "                t.received_currency = $receiving_currency, "
    "                t.payment_format = $payment_format"
)


def upsert_transaction(event: dict) -> None:
    with session_scope() as session:
        session.run(_UPSERT_TRANSACTION, **event)


def score_event(client: httpx.Client, event: dict) -> dict | None:
    """POST to /score, retrying transient failures with backoff.

    Returns None when the API stays unreachable: one unscored transaction is
    worth logging and moving past, not stopping the whole stream for.
    """
    settings = get_settings()
    for attempt in range(1, SCORE_RETRIES + 1):
        try:
            response = client.post(
                "/score",
                json=event,
                headers={"X-API-Key": settings.service_api_key},
                timeout=SCORE_TIMEOUT_SECONDS,
            )
            if response.status_code == 200:
                return response.json()
            # 4xx means this event will never succeed - retrying is pointless.
            if 400 <= response.status_code < 500:
                logger.error(
                    "scoring rejected tx_id=%s status=%s body=%s",
                    event.get("tx_id"),
                    response.status_code,
                    response.text[:200],
                )
                return None
            logger.warning(
                "scoring returned %s for tx_id=%s (attempt %d)",
                response.status_code,
                event.get("tx_id"),
                attempt,
            )
        except httpx.HTTPError as exc:
            logger.warning(
                "scoring unreachable for tx_id=%s (attempt %d): %s",
                event.get("tx_id"),
                attempt,
                exc,
            )
        time.sleep(2**attempt * 0.1)
    return None


def run(max_events: int | None, poll_timeout: float = 1.0, idle_limit: int = 15) -> dict:
    settings = get_settings()
    consumer = build_consumer(CONSUMER_GROUP)
    consumer.subscribe([settings.transactions_topic])

    processed: set[str] = set()
    stats = {"consumed": 0, "scored": 0, "flagged": 0, "skipped_duplicate": 0, "score_failed": 0}
    idle_polls = 0

    logger.info(
        "consuming '%s' -> graph upsert + %s/score",
        settings.transactions_topic,
        settings.scoring_api_base_url,
    )
    with httpx.Client(base_url=settings.scoring_api_base_url) as client:
        try:
            while max_events is None or stats["consumed"] < max_events:
                message = consumer.poll(poll_timeout)
                if message is None:
                    idle_polls += 1
                    if idle_polls >= idle_limit:
                        logger.info("no new events for %ds - stopping", idle_limit)
                        break
                    continue
                if message.error():
                    if message.error().code() != KafkaError._PARTITION_EOF:
                        logger.error("consume error: %s", message.error())
                    continue

                idle_polls = 0
                event = json.loads(message.value())
                stats["consumed"] += 1

                # At-least-once delivery means redelivery is normal, not an
                # error - skip work already done rather than double-scoring.
                if event["tx_id"] in processed:
                    stats["skipped_duplicate"] += 1
                    consumer.commit(message)
                    continue

                upsert_transaction(event)
                result = score_event(client, event)
                if result is None:
                    stats["score_failed"] += 1
                else:
                    stats["scored"] += 1
                    stats["flagged"] += int(result["is_flagged"])
                    if result["is_flagged"]:
                        logger.info(
                            "FLAGGED tx_id=%s score=%.4f degraded=%s",
                            result["tx_id"],
                            result["score"],
                            result["degraded"],
                        )

                processed.add(event["tx_id"])
                # Commit last: anything above failing means redelivery.
                consumer.commit(message)
        finally:
            consumer.close()

    return stats


def main(max_events: int | None) -> None:
    stats = run(max_events)
    print(
        f"consumed={stats['consumed']:,} scored={stats['scored']:,} "
        f"flagged={stats['flagged']:,} duplicates_skipped={stats['skipped_duplicate']:,} "
        f"score_failures={stats['score_failed']:,}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-events", type=int, default=None, help="Stop after N events (default: until idle)."
    )
    args = parser.parse_args()
    main(args.max_events)

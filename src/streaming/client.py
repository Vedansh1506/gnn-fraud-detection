"""Shared Kafka-API client configuration (SAD C2).

Redpanda locally, Kinesis in the cloud - the client code stays Kafka-API
compatible either way, which is the whole reason Redpanda was chosen over a
Redpanda-specific API.
"""

from __future__ import annotations

from confluent_kafka import Consumer, Producer

from src.common.config import get_settings


def _base_config() -> dict:
    return {
        "bootstrap.servers": get_settings().redpanda_bootstrap_servers,
        # Without this, librdkafka resolves "localhost" to ::1 first and stalls
        # ~21s per connection attempt before falling back to IPv4 - long enough
        # that consumers appear to receive nothing at all. Measured: 0.2s
        # round trip with v4 forced, timeout without it.
        "broker.address.family": "v4",
    }


def build_producer(**overrides) -> Producer:
    return Producer({**_base_config(), **overrides})


def build_consumer(group_id: str, **overrides) -> Consumer:
    return Consumer(
        {
            **_base_config(),
            "group.id": group_id,
            # earliest: a demo replay should process the backlog it was just
            # given, not silently skip everything produced before it started.
            "auto.offset.reset": "earliest",
            # Commit only after the work is done, so a crash mid-transaction
            # redelivers rather than losing the event.
            "enable.auto.commit": False,
            **overrides,
        }
    )

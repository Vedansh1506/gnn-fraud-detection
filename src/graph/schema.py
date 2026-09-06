"""Idempotent Neo4j schema setup - safe to run on every startup/load (TRD 4.1 indexing spec)."""

from __future__ import annotations

from neo4j import Driver

from src.graph.client import get_driver

_CONSTRAINTS_AND_INDEXES = (
    "CREATE CONSTRAINT account_key_unique IF NOT EXISTS "
    "FOR (a:Account) REQUIRE a.account_key IS UNIQUE",
    "CREATE INDEX transacted_timestamp IF NOT EXISTS "
    "FOR ()-[r:TRANSACTED]-() ON (r.timestamp)",
    # Not a uniqueness constraint (Neo4j Community doesn't support those on
    # relationship properties) - a plain index so MERGE ... {tx_id: ...} can
    # look tx_id up directly instead of scanning every relationship between
    # the two matched accounts, which gets slow for high-multiplicity pairs
    # (e.g. accounts with many same-account "Reinvestment" transactions).
    "CREATE INDEX transacted_tx_id IF NOT EXISTS " "FOR ()-[r:TRANSACTED]-() ON (r.tx_id)",
)


def ensure_schema(driver: Driver | None = None) -> None:
    driver = driver or get_driver()
    with driver.session() as session:
        for statement in _CONSTRAINTS_AND_INDEXES:
            session.run(statement)


if __name__ == "__main__":
    ensure_schema()
    print("Schema ensured: account_key uniqueness constraint + TRANSACTED.timestamp index.")

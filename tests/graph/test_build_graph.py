"""Integration test - requires Neo4j reachable (docker compose up -d neo4j).

Skips rather than fails when Neo4j isn't running, per the plan's testing
strategy (TRD 7.1 "Integration"/"Idempotency" rows use a small fixture
dataset against a real Neo4j, not a mock).
"""

from pathlib import Path

import pytest

from src.graph.build_graph import build_graph
from src.graph.client import get_driver, session_scope

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "sample_transactions.csv"
# All fixture account_keys start with this - guaranteed disjoint from real
# IBM AML bank codes, so cleanup here can never touch real graph data.
TEST_ACCOUNT_PREFIX = "999"


@pytest.fixture
def neo4j_available():
    try:
        get_driver().verify_connectivity()
    except Exception:
        pytest.skip("Neo4j not reachable - run `docker compose up -d neo4j` to enable this test.")


def _counts() -> tuple[int, int]:
    with session_scope() as session:
        node_count = session.run(
            "MATCH (a:Account) WHERE a.account_key STARTS WITH $prefix RETURN count(a) AS c",
            prefix=TEST_ACCOUNT_PREFIX,
        ).single()["c"]
        edge_count = session.run(
            "MATCH (a:Account)-[t:TRANSACTED]->(b:Account) "
            "WHERE a.account_key STARTS WITH $prefix OR b.account_key STARTS WITH $prefix "
            "RETURN count(t) AS c",
            prefix=TEST_ACCOUNT_PREFIX,
        ).single()["c"]
    return node_count, edge_count


def _cleanup() -> None:
    with session_scope() as session:
        session.run(
            "MATCH (a:Account) WHERE a.account_key STARTS WITH $prefix DETACH DELETE a",
            prefix=TEST_ACCOUNT_PREFIX,
        )


def test_build_graph_is_idempotent(neo4j_available, tmp_path):
    # tmp_path for the feature output: without it this test overwrites the
    # real data/processed/*.parquet built from the full 6.9M-row dataset
    # with its own 5-row fixture (which is exactly what happened once).
    _cleanup()
    try:
        build_graph(FIXTURE_PATH, limit=None, batch_size=100, processed_dir=tmp_path)
        first_nodes, first_edges = _counts()
        assert first_nodes > 0
        assert first_edges > 0

        build_graph(FIXTURE_PATH, limit=None, batch_size=100, processed_dir=tmp_path)
        second_nodes, second_edges = _counts()

        assert second_nodes == first_nodes
        assert second_edges == first_edges
    finally:
        _cleanup()


def test_build_graph_writes_features_only_to_given_dir(neo4j_available, tmp_path):
    """Guards the clobbering bug above: features must land in the caller's
    directory, never the default one, when processed_dir is passed."""
    _cleanup()
    try:
        build_graph(FIXTURE_PATH, limit=None, batch_size=100, processed_dir=tmp_path)
        assert (tmp_path / "tx_features.parquet").exists()
        assert (tmp_path / "account_features.parquet").exists()
    finally:
        _cleanup()

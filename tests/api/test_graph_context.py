"""Graph-context tests, with a stubbed Neo4j session so they need no database.

The truncation assertions exist because the first implementation returned
exactly the query LIMIT while reporting truncated=False - a partial
neighbourhood presented as complete, which would quietly mislead an analyst
about who a flagged account deals with.
"""

from contextlib import contextmanager

import pytest

from src.api import graph_context
from src.api.graph_context import MAX_EDGES, MAX_NODES, fetch_neighbourhood


def _record(source: str, target: str, tx_id: str, laundering: bool = False) -> dict:
    return {
        "source": source,
        "target": target,
        "tx_id": tx_id,
        "amount_paid": 100.0,
        "is_laundering": laundering,
    }


@pytest.fixture
def stub_neo4j(monkeypatch):
    """Patch the session used by graph_context to return canned records."""

    def _install(records: list[dict]):
        @contextmanager
        def fake_session_scope():
            class FakeSession:
                def run(self, _query, **_params):
                    return records

            yield FakeSession()

        monkeypatch.setattr(graph_context, "session_scope", fake_session_scope)

    return _install


def test_small_neighbourhood_is_returned_whole(stub_neo4j):
    stub_neo4j([_record("A", "B", "T1"), _record("B", "C", "T2", laundering=True)])

    result = fetch_neighbourhood("A", hops=2)

    assert result.account_key == "A"
    assert set(result.nodes) == {"A", "B", "C"}
    assert len(result.edges) == 2
    assert result.truncated is False
    assert result.edges[1].is_laundering is True


def test_edge_cap_is_enforced_and_reported(stub_neo4j):
    # One more record than the cap allows - the extra must be dropped *and*
    # the caller told about it.
    stub_neo4j([_record(f"A{i}", f"B{i}", f"T{i}") for i in range(MAX_EDGES + 1)])

    result = fetch_neighbourhood("A0", hops=2)

    assert len(result.edges) <= MAX_EDGES
    assert result.truncated is True


def test_node_cap_is_enforced_and_reported(stub_neo4j):
    # Every edge introduces two fresh accounts, so the node cap bites first.
    stub_neo4j([_record(f"S{i}", f"R{i}", f"T{i}") for i in range(MAX_NODES)])

    result = fetch_neighbourhood("S0", hops=2)

    assert len(result.nodes) <= MAX_NODES
    assert result.truncated is True


def test_empty_neighbourhood_is_not_an_error(stub_neo4j):
    """An account with no counterparties is a valid, if dull, answer - the
    dashboard shows an empty state rather than an error."""
    stub_neo4j([])

    result = fetch_neighbourhood("LONELY", hops=2)

    assert result.nodes == []
    assert result.edges == []
    assert result.truncated is False


def test_missing_amount_does_not_break_the_view(stub_neo4j):
    record = _record("A", "B", "T1")
    record["amount_paid"] = None
    stub_neo4j([record])

    result = fetch_neighbourhood("A", hops=2)

    assert result.edges[0].amount_paid == 0.0

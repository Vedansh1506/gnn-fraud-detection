"""N-hop money-flow neighbourhood for the dashboard's graph panel (TRD 5.1).

Read-only: the consumer owns graph writes (TRD 5.2). This is the only place the
API touches Neo4j at all - scoring deliberately doesn't, so the graph store
being down degrades one panel instead of stopping the service.
"""

from __future__ import annotations

from dataclasses import dataclass

from neo4j import Driver

from src.graph.client import session_scope

# rules.md caps the rendered graph at ~30 nodes, for legibility as much as
# render speed - an analyst cannot read a hairball. Edges need their own cap
# too: a handful of busy accounts can produce dozens of transactions between
# them, which is just as unreadable as too many nodes. Laundering-labelled and
# larger-value edges are kept first, so what survives the cap is what matters.
MAX_NODES = 30
MAX_EDGES = 40



@dataclass
class GraphEdge:
    source: str
    target: str
    tx_id: str
    amount_paid: float
    is_laundering: bool


@dataclass
class GraphNeighbourhood:
    account_key: str
    nodes: list[str]
    edges: list[GraphEdge]
    truncated: bool


def fetch_neighbourhood(
    account_key: str, hops: int, driver: Driver | None = None
) -> GraphNeighbourhood:
    """Collect the distinct accounts and transactions within `hops` of the
    given account, capped at MAX_NODES.

    `driver` targets a different graph than the configured one. It exists so
    the cloud subgraph export and its verification run *this* query rather
    than a copy of it - a copy would drift, and the export is only correct if
    it reproduces exactly what the panel shows.
    """
    # Fetch one more than the cap so "did we cut anything off?" is answerable:
    # reporting a partial neighbourhood as complete would quietly mislead the
    # analyst about who a flagged account deals with.
    fetch_limit = MAX_EDGES + 1
    query = (
        "MATCH (a:Account {account_key: $account_key})"
        f"-[r:TRANSACTED*1..{hops}]-(connected) "
        "UNWIND r AS rel "
        "WITH DISTINCT rel, startNode(rel) AS s, endNode(rel) AS e "
        "RETURN s.account_key AS source, e.account_key AS target, rel.tx_id AS tx_id, "
        "rel.amount_paid AS amount_paid, rel.is_laundering AS is_laundering "
        "ORDER BY rel.is_laundering DESC, rel.amount_paid DESC "
        f"LIMIT {fetch_limit}"
    )

    if driver is not None:
        with driver.session() as session:
            records = list(session.run(query, account_key=account_key))
    else:
        with session_scope() as session:
            records = list(session.run(query, account_key=account_key))

    truncated = len(records) > MAX_EDGES
    edges: list[GraphEdge] = []
    nodes: list[str] = []
    for record in records[:MAX_EDGES]:
        if len(set(nodes) | {record["source"], record["target"]}) > MAX_NODES:
            truncated = True
            break
        for key in (record["source"], record["target"]):
            if key not in nodes:
                nodes.append(key)
        edges.append(
            GraphEdge(
                source=record["source"],
                target=record["target"],
                tx_id=record["tx_id"],
                amount_paid=float(record["amount_paid"] or 0.0),
                is_laundering=bool(record["is_laundering"]),
            )
        )

    return GraphNeighbourhood(
        account_key=account_key,
        nodes=nodes,
        edges=edges,
        truncated=truncated,
    )

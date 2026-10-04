"""Export the money-flow subgraph the dashboard needs, and load it into AuraDB.

    uv run python -m scripts.subgraph export --out subgraph.json
    uv run python -m scripts.subgraph load   --file subgraph.json --uri neo4j+s://xxxx.databases.neo4j.io
    uv run python -m scripts.subgraph verify --file subgraph.json --uri neo4j+s://xxxx.databases.neo4j.io

The target password is read from TARGET_NEO4J_PASSWORD, never a CLI argument,
so it does not land in shell history (same rule as src.db.seed_users).

**Why a subgraph.** The full graph is 705,907 accounts and 6.9M payments.
AuraDB Free holds at most 200,000 nodes and 400,000 relationships, so it cannot
take the full graph. It does not need to: the dashboard only ever shows a flagged account's
capped neighbourhood (`src/api/graph_context.py`: 30 nodes / 40 edges). Every
flag in the queue fits in about 0.1% of the free limit.

**Why not just copy what the panel shows.** The panel keeps the top 40 edges by
laundering label then amount. A 2-hop edge can survive that cut while the 1-hop
edge linking it back to the flagged account does not. Loaded on its own, that
2-hop edge would be unreachable in the target graph and the cloud panel would
quietly show less than the local one. So for every account the panel shows, the
export also includes one shortest connecting path back to the flagged account.
`verify` then checks the target really reproduces the local neighbourhoods.

Which accounts: every account with a flagged audit row. Replaying the same
held-out events through the same pinned model produces the same flags in the
cloud, so these are the neighbourhoods the deployed queue will ask for. A flag
whose neighbourhood is missing degrades to the panel's empty state, not an error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from neo4j import Driver, GraphDatabase
from sqlalchemy import select

from src.api.graph_context import fetch_neighbourhood
from src.db.models import AuditLogEntry
from src.db.session import session_scope
from src.graph.client import get_driver
from src.graph.schema import ensure_schema

HOPS = 2
BATCH = 500

# The same edge properties the panel's query reads. Anything else would be
# load we never use.
CONNECTOR_QUERY = (
    "MATCH p = shortestPath((a:Account {account_key: $source})"
    f"-[:TRANSACTED*..{HOPS}]-(b:Account {{account_key: $target}})) "
    "UNWIND relationships(p) AS rel "
    "RETURN startNode(rel).account_key AS source, endNode(rel).account_key AS target, "
    "rel.tx_id AS tx_id, rel.amount_paid AS amount_paid, rel.is_laundering AS is_laundering"
)


def flagged_accounts() -> list[str]:
    with session_scope() as session:
        rows = session.scalars(
            select(AuditLogEntry.account_key).where(AuditLogEntry.is_flagged.is_(True)).distinct()
        )
        return sorted(rows)


def _edge(source, target, tx_id, amount_paid, is_laundering) -> dict:
    return {
        "source": source,
        "target": target,
        "tx_id": tx_id,
        "amount_paid": float(amount_paid or 0.0),
        "is_laundering": bool(is_laundering),
    }


def export(out: Path) -> dict:
    accounts = flagged_accounts()
    driver = get_driver()
    edges: dict[str, dict] = {}
    connectors_added = 0

    for i, account in enumerate(accounts, start=1):
        shown = fetch_neighbourhood(account, HOPS)
        for e in shown.edges:
            edges[e.tx_id] = _edge(e.source, e.target, e.tx_id, e.amount_paid, e.is_laundering)

        # Make every shown account reachable from the flagged one in the
        # target graph, even if the panel's cap dropped its connecting edge.
        with driver.session() as session:
            for node in shown.nodes:
                if node == account:
                    continue
                for r in session.run(CONNECTOR_QUERY, source=account, target=node):
                    if r["tx_id"] not in edges:
                        connectors_added += 1
                    edges[r["tx_id"]] = _edge(
                        r["source"], r["target"], r["tx_id"], r["amount_paid"], r["is_laundering"]
                    )
        print(f"  [{i}/{len(accounts)}] {account}: {len(shown.nodes)} accounts shown")

    nodes = sorted({e["source"] for e in edges.values()} | {e["target"] for e in edges.values()})
    payload = {"flagged_accounts": accounts, "nodes": nodes, "edges": list(edges.values())}
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print(f"\nexported {len(nodes):,} accounts / {len(edges):,} payments for {len(accounts)} flags")
    print(f"  of which {connectors_added} connecting edges the panel's cap would have dropped")
    print(f"  AuraDB Free headroom: {len(nodes) / 200_000:.2%} of nodes, "
          f"{len(edges) / 400_000:.2%} of relationships")
    return payload


def _target_driver(uri: str, user: str) -> Driver:
    password = os.environ.get("TARGET_NEO4J_PASSWORD")
    if not password:
        raise SystemExit("Set TARGET_NEO4J_PASSWORD (kept out of argv so it skips shell history).")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    return driver


def load(file: Path, uri: str, user: str) -> None:
    payload = json.loads(file.read_text(encoding="utf-8"))
    driver = _target_driver(uri, user)
    ensure_schema(driver)

    with driver.session() as session:
        for start in range(0, len(payload["nodes"]), BATCH):
            session.run(
                "UNWIND $keys AS key MERGE (:Account {account_key: key})",
                keys=payload["nodes"][start : start + BATCH],
            )
        # MERGE on tx_id: re-running the load is idempotent, same rule as the
        # consumer's writes, so a half-finished load can simply be repeated.
        for start in range(0, len(payload["edges"]), BATCH):
            session.run(
                "UNWIND $edges AS e "
                "MATCH (s:Account {account_key: e.source}), (t:Account {account_key: e.target}) "
                "MERGE (s)-[r:TRANSACTED {tx_id: e.tx_id}]->(t) "
                "SET r.amount_paid = e.amount_paid, r.is_laundering = e.is_laundering",
                edges=payload["edges"][start : start + BATCH],
            )
        counts = session.run(
            "MATCH (a:Account) WITH count(a) AS n "
            "MATCH ()-[r:TRANSACTED]->() RETURN n, count(r) AS m"
        ).single()

    driver.close()
    print(
        f"loaded into {uri}: target now holds "
        f"{counts['n']:,} accounts / {counts['m']:,} payments"
    )


def verify(file: Path, uri: str, user: str) -> int:
    """Run the panel's own query against both graphs and compare."""
    payload = json.loads(file.read_text(encoding="utf-8"))
    target = _target_driver(uri, user)
    mismatches = 0

    for account in payload["flagged_accounts"]:
        local = fetch_neighbourhood(account, HOPS)
        remote = fetch_neighbourhood(account, HOPS, driver=target)
        local_nodes, remote_nodes = set(local.nodes), set(remote.nodes)
        missing = local_nodes - remote_nodes
        status = "ok" if not missing else f"MISSING {len(missing)} account(s)"
        mismatches += bool(missing)
        print(
            f"  {account}: local {len(local_nodes)} / "
            f"target {len(remote_nodes)} accounts - {status}"
        )

    target.close()
    print(f"\n{len(payload['flagged_accounts']) - mismatches}/{len(payload['flagged_accounts'])} "
          "neighbourhoods fully reproduced")
    return 1 if mismatches else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_export = sub.add_parser("export", help="Write the needed subgraph from the local graph.")
    p_export.add_argument("--out", type=Path, default=Path("subgraph.json"))

    for name in ("load", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--file", type=Path, default=Path("subgraph.json"))
        p.add_argument("--uri", required=True, help="Target, e.g. neo4j+s://xxxx.databases.neo4j.io")
        p.add_argument("--user", default="neo4j")

    args = parser.parse_args()
    if args.command == "export":
        export(args.out)
        return 0
    if args.command == "load":
        load(args.file, args.uri, args.user)
        return 0
    return verify(args.file, args.uri, args.user)


if __name__ == "__main__":
    sys.exit(main())

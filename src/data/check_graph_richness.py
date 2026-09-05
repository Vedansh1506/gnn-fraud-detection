"""Graph-richness diagnostic for the IBM AML LI-Small transaction file.

Validates the assumption the whole project depends on: that account-level
money-flow chains exist in this data (unlike PaySim, which had none - see
docs/memory.md). Run before trusting any GNN work on a given slice:

    uv run python src/data/check_graph_richness.py [path/to/LI-Small_Trans.csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DEFAULT_TRANSACTIONS_PATH = Path("data/raw/LI-Small_Trans.csv")

REQUIRED_COLUMNS = ["From Bank", "Account", "To Bank", "Account.1", "Timestamp", "Is Laundering"]


def load_transactions(path: Path) -> pd.DataFrame:
    """Load the transaction CSV and derive sender/receiver account_key columns.

    Bank codes carry meaningful leading zeros (e.g. "011"), so they must be
    read as strings - reading them as integers would silently corrupt the
    account_key by dropping those zeros (TRD: account_key = bank + account,
    joined as "<bank>_<account>", because banks reuse account numbers).
    """
    df = pd.read_csv(
        path,
        usecols=REQUIRED_COLUMNS,
        dtype={
            "From Bank": str,
            "Account": str,
            "To Bank": str,
            "Account.1": str,
            "Is Laundering": "int8",
        },
        parse_dates=["Timestamp"],
        date_format="%Y/%m/%d %H:%M",
    )
    df["sender_account_key"] = df["From Bank"] + "_" + df["Account"]
    df["receiver_account_key"] = df["To Bank"] + "_" + df["Account.1"]
    return df


def report_pass_through_accounts(df: pd.DataFrame) -> set[str]:
    """(1) Accounts seen as both a sender and a receiver, across all transactions."""
    senders = set(df["sender_account_key"])
    receivers = set(df["receiver_account_key"])
    pass_through = senders & receivers
    all_accounts = senders | receivers

    print(f"Unique sender accounts:   {len(senders):,}")
    print(f"Unique receiver accounts: {len(receivers):,}")
    print(f"Unique accounts overall:  {len(all_accounts):,}")
    print(f"Pass-through accounts (appear as BOTH sender and receiver): {len(pass_through):,}")
    if all_accounts:
        pct = 100 * len(pass_through) / len(all_accounts)
        print(f"  -> {pct:.2f}% of all accounts relay funds at least once (a chain link).")
    return pass_through


def report_laundering_multihop(df: pd.DataFrame) -> set[str]:
    """(2) Whether laundering-labelled transactions themselves form multi-hop chains.

    An account that both receives and sends laundering-labelled money is
    evidence of a real path (A)-[:TRANSACTED]->(B)-[:TRANSACTED]->(C), not
    just isolated flagged edges - see TRD 4.2 "money-flow paths".
    """
    laundering = df[df["Is Laundering"] == 1]
    l_senders = set(laundering["sender_account_key"])
    l_receivers = set(laundering["receiver_account_key"])
    multihop = l_senders & l_receivers

    print(f"Laundering-labelled transactions: {len(laundering):,}")
    print(f"Laundering-labelled sender accounts:   {len(l_senders):,}")
    print(f"Laundering-labelled receiver accounts: {len(l_receivers):,}")
    print(f"Accounts that both RECEIVE and SEND laundering-labelled money: {len(multihop):,}")

    if multihop:
        example_key = sorted(multihop)[0]
        is_incoming = laundering["receiver_account_key"] == example_key
        is_outgoing = laundering["sender_account_key"] == example_key
        incoming = laundering[is_incoming].sort_values("Timestamp")
        outgoing = laundering[is_outgoing].sort_values("Timestamp")
        print(f"  Example multi-hop account: {example_key}")
        print(
            f"    receives {len(incoming):,} laundering-labelled payment(s), "
            f"then relays {len(outgoing):,} onward -> a real chain, not an isolated flag."
        )

    return multihop


def report_class_ratio(df: pd.DataFrame) -> None:
    """(3) Laundering vs. clean ratio - the imbalance behind picking AUPRC over accuracy."""
    total = len(df)
    laundering_n = int((df["Is Laundering"] == 1).sum())
    clean_n = total - laundering_n
    ratio = clean_n / laundering_n if laundering_n else float("inf")

    print(f"Total transactions: {total:,}")
    print(f"  Laundering (Is Laundering=1): {laundering_n:,} ({100 * laundering_n / total:.4f}%)")
    print(f"  Clean      (Is Laundering=0): {clean_n:,} ({100 * clean_n / total:.4f}%)")
    print(f"  Class imbalance: ~1 laundering transaction per {ratio:,.0f} clean transactions.")


def print_verdict(pass_through: set[str], multihop: set[str]) -> None:
    print()
    print("=" * 78)
    if pass_through and multihop:
        print("VERDICT: Real money-flow chains exist in this data.")
        print(f"  {len(pass_through):,} accounts relay funds in general, and {len(multihop):,} of")
        print("  those relays happen specifically within laundering-labelled transactions - i.e.")
        print("  multi-hop laundering paths, not just isolated flagged edges.")
        print("  This is the structure a GraphSAGE model needs. Safe to proceed.")
    elif pass_through:
        print("VERDICT: Money-flow chains exist overall, but none were found within")
        print("  laundering-labelled transactions specifically (no multihop accounts).")
        print("  Graph structure exists, but the labelled laundering patterns may look")
        print("  point-like on this slice - investigate before trusting graph-based lift.")
    else:
        print("VERDICT: NO pass-through accounts found anywhere in this slice - this")
        print("  data is effectively a bipartite/star graph, the same failure mode that")
        print("  disproved PaySim. A GNN would have nothing to learn. STOP and re-check")
        print("  the dataset file / time-slice before building further.")
    print("=" * 78)


def main(path: Path) -> None:
    print(f"Loading {path} ...")
    df = load_transactions(path)
    print(f"Loaded {len(df):,} transactions.\n")

    print("--- (1) Pass-through accounts (chains, all transactions) ---")
    pass_through = report_pass_through_accounts(df)

    print("\n--- (2) Multi-hop laundering chains ---")
    multihop = report_laundering_multihop(df)

    print("\n--- (3) Laundering class ratio ---")
    report_class_ratio(df)

    print_verdict(pass_through, multihop)


if __name__ == "__main__":
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TRANSACTIONS_PATH
    main(csv_path)

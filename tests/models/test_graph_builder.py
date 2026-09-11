import pandas as pd

from src.models.classifier.dataset import split_by_time
from src.models.gnn.graph_builder import build_account_graph


def _transactions() -> pd.DataFrame:
    rows = [
        ("T0", "A", "B", "2022-09-01 00:00", 100.0, 0),
        ("T1", "B", "C", "2022-09-02 00:00", 90.0, 1),
        ("T2", "C", "A", "2022-09-03 00:00", 80.0, 0),
        ("T3", "A", "C", "2022-09-04 00:00", 70.0, 0),
        # After the train cutoff - must not reach the graph at all.
        ("T4", "X", "Y", "2022-09-12 00:00", 500.0, 1),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "tx_id",
            "sender_account_key",
            "receiver_account_key",
            "Timestamp",
            "Amount Paid",
            "Is Laundering",
        ],
    )
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


def _train_graph():
    train_df, _, _ = split_by_time(_transactions())
    return build_account_graph(train_df)


def test_nodes_are_train_window_accounts_only():
    graph = _train_graph()
    assert set(graph.account_keys) == {"A", "B", "C"}
    assert "X" not in graph.account_keys  # post-cutoff account
    assert "Y" not in graph.account_keys


def test_edges_are_bidirectional_over_train_transactions():
    graph = _train_graph()
    # 4 train transactions, each contributing a forward and a reverse edge.
    assert graph.data.edge_index.shape == (2, 8)
    assert int(graph.data.edge_index.max()) < graph.num_nodes


def test_node_labels_come_from_train_window_laundering_only():
    graph = _train_graph()
    labels = dict(zip(graph.account_keys, graph.data.y.tolist(), strict=True))

    # T1 (B -> C) is the only laundering transaction inside the train window.
    assert labels["B"] == 1.0
    assert labels["C"] == 1.0
    assert labels["A"] == 0.0


def test_node_features_are_finite_and_shaped_per_account():
    graph = _train_graph()
    assert graph.data.x.shape[0] == graph.num_nodes
    assert graph.data.x.shape[1] == 6  # ACCOUNT_FEATURE_COLUMNS
    assert bool(graph.data.x.isfinite().all())

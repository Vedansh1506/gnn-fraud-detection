"""Builds the account money-flow graph the GNN trains on (SAD C6).

Nodes are accounts and edges are money flows, which is what the locked
decisions require: `account_embeddings` is keyed by `account_key` and serving
reads one embedding per account. (TRD 4.1 mentions a transaction-to-transaction
formulation as an option - "the offline training graph *may* use that tx->tx
formulation" - which this deliberately does not use.)

Everything here is built from the **training window only**, for the same reason
the tabular features are: an embedding that encoded test-period structure would
inflate the headline comparison and could never be reproduced at serving time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.features.account_features import ACCOUNT_FEATURE_COLUMNS, fit_account_features


@dataclass
class AccountGraph:
    """A PyG graph plus the account_key <-> node index mapping needed to get
    embeddings back out keyed by account."""

    data: Data
    account_keys: list[str]

    @property
    def num_nodes(self) -> int:
        return len(self.account_keys)


def build_account_graph(train_df: pd.DataFrame) -> AccountGraph:
    """Assemble the train-window account graph.

    Edges are added in both directions. Money flows one way, but for message
    passing a receiver's risk is just as informative about its sender as the
    reverse - and GraphSAGE aggregates over incoming edges only, so a
    directed-only graph would leave every pure-sender node with no neighbourhood
    to learn from.
    """
    account_features = fit_account_features(train_df)
    account_keys = account_features["account_key"].tolist()
    index_of = {key: i for i, key in enumerate(account_keys)}

    x = _node_features(account_features)
    edge_index = _edge_index(train_df, index_of)
    y = _node_labels(train_df, index_of, len(account_keys))

    return AccountGraph(data=Data(x=x, edge_index=edge_index, y=y), account_keys=account_keys)


def _node_features(account_features: pd.DataFrame) -> torch.Tensor:
    """Degree/amount features on wildly different scales (counts in the tens,
    amounts in the millions), so log1p-compress then standardise - otherwise the
    amount columns dominate the first layer purely by magnitude."""
    raw = account_features[ACCOUNT_FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=np.float64)
    compressed = np.log1p(np.clip(raw, 0.0, None))
    mean = compressed.mean(axis=0)
    std = compressed.std(axis=0)
    std[std == 0] = 1.0  # a constant column carries no signal; don't divide by 0
    return torch.tensor((compressed - mean) / std, dtype=torch.float32)


def _edge_index(train_df: pd.DataFrame, index_of: dict[str, int]) -> torch.Tensor:
    senders = train_df["sender_account_key"].map(index_of).to_numpy(dtype=np.int64)
    receivers = train_df["receiver_account_key"].map(index_of).to_numpy(dtype=np.int64)
    forward = np.stack([senders, receivers])
    backward = np.stack([receivers, senders])
    return torch.tensor(np.concatenate([forward, backward], axis=1), dtype=torch.long)


def _node_labels(
    train_df: pd.DataFrame, index_of: dict[str, int], num_nodes: int
) -> torch.Tensor:
    """An account is positive if it took either side of a laundering-labelled
    transaction during the training window. The dataset labels transactions, not
    accounts, so this projection is what makes node-level training possible.
    """
    laundering = train_df[train_df["Is Laundering"] == 1]
    involved = pd.concat(
        [laundering["sender_account_key"], laundering["receiver_account_key"]]
    ).unique()

    labels = torch.zeros(num_nodes, dtype=torch.float32)
    indices = [index_of[key] for key in involved if key in index_of]
    labels[indices] = 1.0
    return labels

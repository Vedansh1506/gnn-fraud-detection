"""Train GraphSAGE on the account graph and export per-account embeddings.

    uv run --group gnn python -m src.models.gnn.train_graphsage

Device-agnostic: uses CUDA when present, CPU otherwise. The same script runs on
a Kaggle/Colab GPU (the locked compute decision) after installing the matching
sampler wheel there, e.g.

    pip install pyg-lib -f https://data.pyg.org/whl/torch-2.12.0+cu121.html

Note `uv sync` without `--group gnn` uninstalls pyg-lib, so GNN commands need
the group flag.

This is the offline half of the offline/online split: the live scoring path
never runs this, it only reads the embeddings this produces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

from src.data.load_amlworld import DEFAULT_TRANSACTIONS_PATH, load_transactions
from src.models.classifier.dataset import split_by_time
from src.models.gnn.graph_builder import AccountGraph, build_account_graph

EMBEDDING_VERSION = "gnn_emb_v1"
ARTIFACTS_ROOT = Path("artifacts/embeddings")

HIDDEN_DIM = 64
EMBEDDING_DIM = 32
NUM_NEIGHBORS = [15, 10]
BATCH_SIZE = 1024
EPOCHS = 3
LEARNING_RATE = 0.01


class GraphSAGE(torch.nn.Module):
    """Two message-passing layers, then a linear head.

    The exported embedding is the second layer's output, not the head's logit:
    a single risk score would throw away the structural detail that makes these
    useful as XGBoost features.
    """

    def __init__(self, in_dim: int, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, embedding_dim)
        self.head = torch.nn.Linear(embedding_dim, 1)

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x, edge_index)).squeeze(-1)


def train(graph: AccountGraph, device: torch.device, epochs: int = EPOCHS) -> GraphSAGE:
    model = GraphSAGE(graph.data.num_features, HIDDEN_DIM, EMBEDDING_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Same imbalance stance as the classifier: weight the loss, never resample
    # the graph (dropping nodes would sever the money-flow chains outright).
    positives = float(graph.data.y.sum())
    pos_weight = torch.tensor([(graph.num_nodes - positives) / max(positives, 1.0)], device=device)
    print(
        f"nodes {graph.num_nodes:,} | positive nodes {int(positives):,} | "
        f"pos_weight {pos_weight.item():,.1f}"
    )

    loader = NeighborLoader(
        graph.data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        input_nodes=None,
    )

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        seen = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index)[: batch.batch_size]
            loss = F.binary_cross_entropy_with_logits(
                logits, batch.y[: batch.batch_size], pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item() * batch.batch_size
            seen += batch.batch_size
        print(f"epoch {epoch}/{epochs}  loss {total_loss / max(seen, 1):.4f}")

    return model


@torch.no_grad()
def export_embeddings(model: GraphSAGE, graph: AccountGraph, device: torch.device) -> pd.DataFrame:
    """Full-graph forward pass - inference has no sampling variance to worry
    about, and every account needs an embedding, not just sampled ones."""
    model.eval()
    embeddings = model.embed(graph.data.x.to(device), graph.data.edge_index.to(device))
    frame = pd.DataFrame(
        embeddings.cpu().numpy(), columns=[f"emb_{i}" for i in range(EMBEDDING_DIM)]
    )
    frame.insert(0, "account_key", graph.account_keys)
    return frame


def main(version: str, epochs: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_df, _, _ = split_by_time(load_transactions(DEFAULT_TRANSACTIONS_PATH))
    print(f"train-window transactions: {len(train_df):,}")

    graph = build_account_graph(train_df)
    print(f"graph: {graph.num_nodes:,} nodes / {graph.data.edge_index.shape[1]:,} directed edges")

    model = train(graph, device, epochs)
    embeddings = export_embeddings(model, graph, device)

    out_dir = ARTIFACTS_ROOT / version
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings.to_parquet(out_dir / "embeddings.parquet", index=False)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "embedding_version": version,
                "embedding_dim": EMBEDDING_DIM,
                "accounts": int(len(embeddings)),
                "positive_nodes": int(graph.data.y.sum()),
                "epochs": epochs,
                "trained_on": "train window only (< dataset.TRAIN_END)",
            },
            indent=2,
        )
    )
    print(f"Wrote {len(embeddings):,} embeddings to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=str, default=EMBEDDING_VERSION)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    main(args.version, args.epochs)

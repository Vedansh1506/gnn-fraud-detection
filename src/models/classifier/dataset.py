"""Assembles the train/val/test matrices for the tabular baseline classifier.

The split is by time, never random - random sampling severs the money-flow
chains this project depends on (locked decision), and a random split would
also let future transactions inform past ones.

Split boundaries are fixed constants rather than percentages because the
dataset's volume is heavily front-loaded (days 11-17 hold ~215 transactions
between them); they were chosen so each side holds a usable number of
laundering-labelled positives, not equal row counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data.load_amlworld import DEFAULT_TRANSACTIONS_PATH, load_transactions
from src.features.account_features import (
    ACCOUNT_FEATURE_COLUMNS,
    fit_account_features,
    join_account_features,
)
from src.features.tx_features import TX_FEATURE_COLUMNS, compute_tx_features

# Inclusive day boundaries (the dataset spans 2022-09-01 .. 2022-09-17).
TRAIN_END = pd.Timestamp("2022-09-08")  # train is everything before this
VAL_END = pd.Timestamp("2022-09-09")  # val is [TRAIN_END, VAL_END)
LABEL_COLUMN = "is_laundering"

FEATURE_COLUMNS = (
    TX_FEATURE_COLUMNS
    + [f"sender_{c}" for c in ACCOUNT_FEATURE_COLUMNS]
    + [f"receiver_{c}" for c in ACCOUNT_FEATURE_COLUMNS]
)
CATEGORICAL_COLUMNS = ["payment_format"]


def embedding_columns(embeddings: pd.DataFrame) -> list[str]:
    return [c for c in embeddings.columns if c.startswith("emb_")]


def feature_columns_with_embeddings(embeddings: pd.DataFrame | None) -> list[str]:
    """The baseline and the GNN-augmented model differ only by these columns -
    everything else about the pipeline is deliberately identical, so the AUPRC
    comparison isolates the embeddings and nothing else."""
    if embeddings is None:
        return list(FEATURE_COLUMNS)
    emb = embedding_columns(embeddings)
    return (
        list(FEATURE_COLUMNS)
        + [f"sender_{c}" for c in emb]
        + [f"receiver_{c}" for c in emb]
    )


def join_embeddings(
    features: pd.DataFrame, embeddings: pd.DataFrame, side: str
) -> pd.DataFrame:
    """Accounts with no embedding (never seen in the training-window graph) get
    zeros - the locked "missing embedding -> neutral default" rule, and the same
    thing the serving path does on a Feast cache miss."""
    emb = embedding_columns(embeddings)
    renamed = embeddings.rename(columns={c: f"{side}_{c}" for c in emb})
    joined = features.merge(
        renamed, how="left", left_on=f"{side}_account_key", right_on="account_key"
    )
    prefixed = [f"{side}_{c}" for c in emb]
    joined[prefixed] = joined[prefixed].fillna(0.0)
    return joined.drop(columns=["account_key"])


@dataclass
class Dataset:
    """Feature matrices and labels for one modelling run, plus the fitted
    account table (which must ship with the model - serving has to apply the
    exact same fitted statistics, not recompute its own)."""

    x_train: pd.DataFrame
    y_train: pd.Series
    x_val: pd.DataFrame
    y_val: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    account_features: pd.DataFrame
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    @property
    def scale_pos_weight(self) -> float:
        """Negative/positive ratio from the TRAIN split only - deriving it from
        the full dataset would leak the test set's class balance."""
        positives = int(self.y_train.sum())
        return (len(self.y_train) - positives) / positives


def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["Timestamp"] < TRAIN_END]
    val = df[(df["Timestamp"] >= TRAIN_END) & (df["Timestamp"] < VAL_END)]
    test = df[df["Timestamp"] >= VAL_END]
    return train, val, test


def _require_non_empty_splits(*splits: pd.DataFrame) -> None:
    """The source file is time-ordered, so `--limit N` truncates to the earliest
    days and can silently leave val/test empty. Say so plainly instead of
    failing later inside XGBoost's early stopping.
    """
    names = ("train", "val", "test")
    empty = [name for name, split in zip(names, splits, strict=True) if split.empty]
    if empty:
        raise ValueError(
            f"Empty split(s): {', '.join(empty)}. The dataset is time-ordered, so a row "
            f"limit below ~5.1M never reaches the {TRAIN_END.date()} val/test windows - "
            f"run without --limit to train for real."
        )


def build_features(
    df: pd.DataFrame,
    account_features: pd.DataFrame,
    embeddings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply already-fitted account statistics to a set of transactions."""
    features = compute_tx_features(df, account_features)
    keys = df[["sender_account_key", "receiver_account_key"]].reset_index(drop=True)
    features = pd.concat([features.reset_index(drop=True), keys], axis=1)
    features = join_account_features(features, account_features, "sender")
    features = join_account_features(features, account_features, "receiver")
    if embeddings is not None:
        features = join_embeddings(features, embeddings, "sender")
        features = join_embeddings(features, embeddings, "receiver")
    for column in CATEGORICAL_COLUMNS:
        features[column] = features[column].astype("category")
    return features


def build_dataset(
    path: Path = DEFAULT_TRANSACTIONS_PATH,
    limit: int | None = None,
    embeddings_path: Path | None = None,
) -> Dataset:
    df = load_transactions(path, limit=limit)
    train_df, val_df, test_df = split_by_time(df)
    _require_non_empty_splits(train_df, val_df, test_df)

    embeddings = pd.read_parquet(embeddings_path) if embeddings_path else None
    columns = feature_columns_with_embeddings(embeddings)

    # Fitted on train only - this is the whole point (see module docstring).
    account_features = fit_account_features(train_df)

    return Dataset(
        x_train=build_features(train_df, account_features, embeddings)[columns],
        y_train=train_df["Is Laundering"].astype(int).reset_index(drop=True),
        x_val=build_features(val_df, account_features, embeddings)[columns],
        y_val=val_df["Is Laundering"].astype(int).reset_index(drop=True),
        x_test=build_features(test_df, account_features, embeddings)[columns],
        y_test=test_df["Is Laundering"].astype(int).reset_index(drop=True),
        account_features=account_features,
        feature_columns=columns,
    )

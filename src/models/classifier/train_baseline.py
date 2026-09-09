"""Train the tabular-only XGBoost baseline (Build Plan Phase 3).

    uv run python -m src.models.classifier.train_baseline [--limit N]

This is the project's fallback deliverable and the baseline half of the
headline baseline-vs-GNN AUPRC comparison. No graph embeddings here - those
arrive in Phase 4 and are what the comparison is meant to isolate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import xgboost as xgb

from src.models.classifier.dataset import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    Dataset,
    build_dataset,
)

MODEL_VERSION = "baseline_v1"
ARTIFACTS_ROOT = Path("artifacts/models")

# Modest depth/learning rate: with ~1 positive per 2,000 rows, an
# unconstrained model memorises the majority class long before it learns
# anything transferable.
PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}
NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 30


def train(dataset: Dataset) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        **PARAMS,
        n_estimators=NUM_BOOST_ROUND,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        scale_pos_weight=dataset.scale_pos_weight,
        enable_categorical=True,
    )
    model.fit(
        dataset.x_train,
        dataset.y_train,
        eval_set=[(dataset.x_val, dataset.y_val)],
        verbose=25,
    )
    return model


def save_artifacts(model: xgb.XGBClassifier, dataset: Dataset, version: str) -> Path:
    """Everything serving needs, pinned together under one version directory.

    feature_spec.json is the train/serve contract: the exact ordered feature
    names the model expects. account_features.parquet is the fitted statistics
    those features were built from - serving joins these, it does not
    recompute them from live data.
    """
    out_dir = ARTIFACTS_ROOT / version
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(out_dir / "model.json")
    dataset.account_features.to_parquet(out_dir / "account_features.parquet", index=False)
    (out_dir / "feature_spec.json").write_text(
        json.dumps(
            {
                "model_version": version,
                "feature_columns": FEATURE_COLUMNS,
                "categorical_columns": CATEGORICAL_COLUMNS,
                "scale_pos_weight": dataset.scale_pos_weight,
                "best_iteration": int(model.best_iteration),
                "train_rows": int(len(dataset.y_train)),
                "train_positives": int(dataset.y_train.sum()),
            },
            indent=2,
        )
    )
    return out_dir


def main(limit: int | None, version: str) -> None:
    print("Building dataset (this loads and re-derives features - a few minutes)...")
    dataset = build_dataset(limit=limit)
    print(
        f"train {len(dataset.y_train):,} rows / {int(dataset.y_train.sum()):,} positives | "
        f"val {len(dataset.y_val):,} / {int(dataset.y_val.sum()):,} | "
        f"test {len(dataset.y_test):,} / {int(dataset.y_test.sum()):,}"
    )
    print(f"scale_pos_weight (train only): {dataset.scale_pos_weight:,.1f}")

    model = train(dataset)
    out_dir = save_artifacts(model, dataset, version)
    print(f"Best iteration: {model.best_iteration}")
    print(f"Wrote artifacts to {out_dir}/")
    print("Run evaluate.py for held-out test metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Load only the first N rows.")
    parser.add_argument("--version", type=str, default=MODEL_VERSION)
    args = parser.parse_args()
    main(args.limit, args.version)

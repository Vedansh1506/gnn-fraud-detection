"""Reproducible held-out evaluation for a pinned model version.

    uv run python -m src.models.classifier.evaluate [--version baseline_v1]

AUPRC is the headline metric, not accuracy: with ~1 laundering transaction per
2,000, a model predicting "clean" every time scores 99.95% accuracy and
catches nothing (PRD). Phase 4 reruns this same script against the
GNN-augmented model so the lift comparison is apples-to-apples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from src.models.classifier.dataset import build_dataset
from src.models.classifier.train_baseline import ARTIFACTS_ROOT, MODEL_VERSION


def load_model(version: str) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(enable_categorical=True)
    model.load_model(ARTIFACTS_ROOT / version / "model.json")
    return model


def evaluate(model: xgb.XGBClassifier, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    scores = model.predict_proba(x_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, scores)

    # Report the threshold maximising F1 as a concrete operating point - the
    # production threshold is a business decision (analyst review capacity),
    # not something to hardcode here.
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    best = int(f1[:-1].argmax())

    return {
        "auprc": float(average_precision_score(y_test, scores)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "positive_rate": float(y_test.mean()),
        "test_rows": int(len(y_test)),
        "test_positives": int(y_test.sum()),
        "best_f1_operating_point": {
            "threshold": float(thresholds[best]),
            "precision": float(precision[best]),
            "recall": float(recall[best]),
            "f1": float(f1[best]),
        },
    }


def main(version: str) -> None:
    # Rebuild features exactly as this model version was trained - the spec
    # records whether embeddings were fused, so baseline and GNN models are
    # evaluated through one code path on one split.
    spec = json.loads((ARTIFACTS_ROOT / version / "feature_spec.json").read_text())
    embeddings_path = spec.get("embeddings_path")
    dataset = build_dataset(embeddings_path=Path(embeddings_path) if embeddings_path else None)
    model = load_model(version)
    metrics = evaluate(model, dataset.x_test, dataset.y_test)
    metrics["model_version"] = version

    out_path: Path = ARTIFACTS_ROOT / version / "metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"\nAUPRC (headline): {metrics['auprc']:.4f}")
    print(f"Baseline for reference - always-predict-positive AUPRC would be "
          f"{metrics['positive_rate']:.6f}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=str, default=MODEL_VERSION)
    args = parser.parse_args()
    main(args.version)

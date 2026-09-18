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

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from src.common.config import get_settings
from src.models.classifier.dataset import build_dataset
from src.models.classifier.train_baseline import ARTIFACTS_ROOT, MODEL_VERSION

# Daily review capacities to report precision/recall at. Chosen to bracket what
# a small team plausibly gets through in a day - the point is to answer "if we
# review N alerts a day, what do we catch?", which is the question an operations
# manager actually asks and AUPRC does not answer on its own.
DAILY_CAPACITIES = (50, 100, 250, 500, 1000)


def load_model(version: str) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(enable_categorical=True)
    model.load_model(ARTIFACTS_ROOT / version / "model.json")
    return model


def alert_capacity(
    scores: np.ndarray,
    y_test: pd.Series,
    timestamps: pd.Series | None,
    flag_threshold: float,
    capacities: tuple[int, ...] = DAILY_CAPACITIES,
) -> dict:
    """Translate ranking quality into review-queue terms.

    AUPRC answers "how good is the ranking"; this answers "if analysts review N
    alerts a day, how many are real and how much laundering do we catch". They
    are the same model, but only the second is actionable for staffing.

    Days come from the test window's own timestamps rather than a constant, so
    the rate stays correct if the split boundaries ever move.
    """
    positives = int(y_test.sum())
    if timestamps is None or timestamps.empty:
        days = None
    else:
        # Distinct calendar days actually present, not a span - a gap in the
        # data would otherwise inflate the denominator and understate volume.
        days = int(pd.to_datetime(timestamps).dt.normalize().nunique())

    order = np.argsort(-scores)
    labels_by_rank = y_test.to_numpy()[order]
    cumulative_hits = np.cumsum(labels_by_rank)

    def at_k(k: int) -> dict:
        k = max(1, min(k, len(labels_by_rank)))
        hits = int(cumulative_hits[k - 1])
        return {
            "reviewed": k,
            "true_positives": hits,
            "precision": hits / k,
            "recall": hits / positives if positives else 0.0,
        }

    by_capacity = []
    if days:
        for per_day in capacities:
            point = at_k(per_day * days)
            point["alerts_per_day"] = per_day
            by_capacity.append(point)

    flagged = int((scores >= flag_threshold).sum())
    flagged_hits = int(y_test.to_numpy()[scores >= flag_threshold].sum())

    return {
        "test_days": days,
        "transactions_per_day": (len(y_test) / days) if days else None,
        "at_flag_threshold": {
            "threshold": flag_threshold,
            "alerts_total": flagged,
            "alerts_per_day": (flagged / days) if days else None,
            "precision": (flagged_hits / flagged) if flagged else None,
            "recall": (flagged_hits / positives) if positives else None,
        },
        "by_daily_capacity": by_capacity,
    }


def evaluate(
    model: xgb.XGBClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    timestamps: pd.Series | None = None,
    flag_threshold: float | None = None,
) -> dict:
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
        "alert_capacity": alert_capacity(
            scores,
            y_test,
            timestamps,
            flag_threshold if flag_threshold is not None else get_settings().flag_threshold,
        ),
    }


def main(version: str) -> None:
    # Rebuild features exactly as this model version was trained - the spec
    # records whether embeddings were fused, so baseline and GNN models are
    # evaluated through one code path on one split.
    spec = json.loads((ARTIFACTS_ROOT / version / "feature_spec.json").read_text())
    embeddings_path = spec.get("embeddings_path")
    # Ablation runs record which features they were trained without; evaluating
    # them against the full feature set would be measuring a different model.
    dropped = tuple(spec.get("dropped_features") or ())
    dataset = build_dataset(
        embeddings_path=Path(embeddings_path) if embeddings_path else None,
        drop_features=dropped,
    )
    model = load_model(version)
    metrics = evaluate(model, dataset.x_test, dataset.y_test, dataset.test_timestamps)
    metrics["model_version"] = version
    if dropped:
        metrics["dropped_features"] = list(dropped)

    out_path: Path = ARTIFACTS_ROOT / version / "metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"\nAUPRC (headline): {metrics['auprc']:.4f}")
    print(f"Baseline for reference - always-predict-positive AUPRC would be "
          f"{metrics['positive_rate']:.6f}")

    capacity = metrics["alert_capacity"]
    if capacity["test_days"]:
        at_threshold = capacity["at_flag_threshold"]
        print()
        if at_threshold["precision"] is None:
            # A model that never reaches the threshold raises nothing at all.
            # That is a real and important result, not a formatting edge case:
            # at the pinned operating point this model would catch zero
            # laundering, however good its ranking looks.
            print(
                f"At the pinned threshold {at_threshold['threshold']}: "
                f"NO alerts at all - this model never reaches the threshold, "
                f"so it would flag nothing in production."
            )
        else:
            print(
                f"At the pinned threshold {at_threshold['threshold']}: "
                f"{at_threshold['alerts_per_day']:,.0f} alerts/day, "
                f"precision {at_threshold['precision']:.1%}, recall {at_threshold['recall']:.1%}"
            )
        print("If analysts review N alerts a day:")
        for point in capacity["by_daily_capacity"]:
            print(
                f"  {point['alerts_per_day']:>5}/day -> precision {point['precision']:6.1%}, "
                f"recall {point['recall']:5.1%} ({point['true_positives']} caught)"
            )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=str, default=MODEL_VERSION)
    args = parser.parse_args()
    main(args.version)

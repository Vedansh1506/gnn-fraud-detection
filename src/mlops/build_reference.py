"""Build the drift reference for a model version.

    uv run python -m src.mlops.build_reference [--version gnn_v2]

The reference is the pre-serving distribution that live traffic is later
compared against. It is built **once, from the validation window**, and saved
as an artifact next to the model it belongs to.

Three decisions worth knowing:

- **The validation window, not the training window.** This was changed after
  measuring: built from training rows, the reference score distribution is the
  one the model produces on data it was *fitted to*, and live traffic then
  registers enormous "prediction drift" that is really just the gap between
  fitted and unseen data (measured: a Wasserstein distance of 28 on `score`,
  almost all of it overfitting rather than drift). The validation split sits
  before the serving period and the model never fit it, so it is the honest
  baseline for what a normal score distribution looks like. The held-out test
  window is deliberately still excluded - folding it in would let a genuine
  shift partly compare against itself.
- **Scores are real.** The sample is scored through the model's *full* feature
  matrix - all 83 columns, exactly as training built them - so the reference
  score distribution is one the model actually produced. Scoring a stub row
  with the other features zeroed would be fast and completely fictitious.
- **`payment_currency` is absent on purpose.** The training matrix carries
  `cross_currency_flag`, not the currency name the audit log stores, so it
  cannot be reconstructed here without guessing. The drift check compares the
  columns that genuinely exist on both sides and says how many that was.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.common.config import get_settings
from src.mlops import drift

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("mlops.build_reference")

# Attributes the audit log also stores, so both sides of the comparison have them.
SHARED_ATTRIBUTES = ("amount_paid", "hour_of_day", "payment_format")


def build(version: str, sample: int, with_scores: bool = True) -> pd.DataFrame:
    """Sample the validation window and, optionally, score it with the pinned model.

    Reuses the classifier's own dataset builder so the train/val/test boundary
    is defined in exactly one place - a reference built against a different
    cutoff than the model trained on would be quietly wrong.
    """
    from src.models.classifier.dataset import build_dataset

    logger.info("loading the dataset (a few minutes)")
    dataset = build_dataset(limit=None)

    features = dataset.x_val
    if len(features) > sample:
        # Fixed seed: a reference that shifts between builds would show drift
        # that is entirely our own sampling noise.
        features = features.sample(sample, random_state=42)

    reference = pd.DataFrame(
        {
            "amount_paid": features["amount_paid"].astype(float).to_numpy(),
            "hour_of_day": features["hour_of_day"].astype(int).to_numpy(),
            "payment_format": features["payment_format"].astype(str).to_numpy(),
        }
    )

    if with_scores:
        reference["score"] = _score(features, version)

    return reference


def _score(features: pd.DataFrame, version: str):
    """Score the sampled validation rows with the pinned serving model.

    Uses the full feature matrix the dataset builder produced, reindexed to the
    model's trained column order - the same contract serving enforces.
    """
    from src.api.scoring import load_artifacts

    artifacts = load_artifacts(version, get_settings().embedding_version)
    aligned = features.reindex(columns=artifacts.feature_columns)
    logger.info("scoring %d validation rows with %s", len(aligned), version)
    return artifacts.model.predict_proba(aligned)[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        default=None,
        help="Model version to build the reference for (default: the serving version).",
    )
    parser.add_argument("--sample", type=int, default=50_000)
    parser.add_argument(
        "--no-scores",
        action="store_true",
        help="Skip scoring - input drift only, no prediction drift.",
    )
    args = parser.parse_args()

    version = args.version or get_settings().model_version
    reference = build(version, args.sample, with_scores=not args.no_scores)
    path = drift.build_reference(reference, version, sample=args.sample)

    print(f"Wrote {path}")
    print(f"  {len(reference):,} rows | columns: {', '.join(reference.columns)}")
    if "score" in reference:
        print(
            f"  reference score: mean {reference['score'].mean():.4f}, "
            f"p95 {reference['score'].quantile(0.95):.4f}"
        )


if __name__ == "__main__":
    main()

"""Backfill MLflow with the models that were trained before tracking existed.

    uv run python -m src.mlops.backfill [--dry-run]

Every value logged here is read from an artifact that a real training or
evaluation run wrote (`feature_spec.json`, `metrics.json`). Nothing is
estimated, interpolated, or invented - a version with no readable metrics is
skipped rather than logged with placeholder numbers.

Each backfilled run is tagged `backfilled=true` with the source path, so it is
always distinguishable from a run MLflow actually observed. That distinction
matters: these runs have no start/end times, no live system metrics and no
parameter history, and presenting them as if MLflow had watched them happen
would be a small lie told by a tool whose whole purpose is provenance.

Re-running is safe: a version already backfilled is skipped.
"""

from __future__ import annotations

import argparse
import logging

import mlflow

from src.api.model_registry import ARTIFACTS_ROOT
from src.mlops import tracking

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("mlops.backfill")

BACKFILL_TAG = "backfilled"


def already_backfilled(experiment_name: str) -> set[str]:
    """Model versions that already have a backfilled run, so re-running this
    command doesn't duplicate them."""
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return set()

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.{BACKFILL_TAG} = 'true'",
        max_results=500,
    )
    return {
        run.data.tags.get("model_version")
        for run in runs
        if run.data.tags.get("model_version")
    }


def discover_versions() -> list[str]:
    if not ARTIFACTS_ROOT.exists():
        return []
    return sorted(child.name for child in ARTIFACTS_ROOT.iterdir() if child.is_dir())


def backfill(dry_run: bool = False) -> int:
    if not tracking.is_reachable():
        logger.error(
            "MLflow is not reachable at %s - start it with "
            "`docker compose up -d mlflow`",
            tracking.get_settings().mlflow_tracking_uri,
        )
        return 0

    tracking.configure()
    existing = already_backfilled(tracking.get_settings().mlflow_experiment)
    logged = 0

    for version in discover_versions():
        metrics = tracking.read_metrics(version)
        if not metrics:
            logger.warning("%s: no metrics.json - skipped (nothing to log honestly)", version)
            continue
        if version in existing:
            logger.info("%s: already backfilled - skipped", version)
            continue

        spec = tracking.read_spec(version)
        params = tracking.params_from_spec(spec)
        flat_metrics = tracking.metrics_from_eval(metrics)

        if dry_run:
            logger.info(
                "%s: would log %d params / %d metrics", version, len(params), len(flat_metrics)
            )
            continue

        with mlflow.start_run(run_name=f"{version} (backfilled)"):
            mlflow.set_tags(
                {
                    "model_version": version,
                    BACKFILL_TAG: "true",
                    "source": str(ARTIFACTS_ROOT / version),
                    "note": (
                        "Logged from artifacts after the fact; "
                        "MLflow did not observe this run."
                    ),
                }
            )
            mlflow.log_params(params)
            mlflow.log_metrics(flat_metrics)
            for filename in ("metrics.json", "feature_spec.json"):
                tracking.log_artifact(ARTIFACTS_ROOT / version / filename)

        logger.info(
            "%s: logged auprc=%.6f roc_auc=%.4f (%s)",
            version,
            flat_metrics.get("auprc", float("nan")),
            flat_metrics.get("roc_auc", float("nan")),
            params.get("embedding_version"),
        )
        logged += 1

    return logged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be logged without writing to MLflow.",
    )
    args = parser.parse_args()

    count = backfill(dry_run=args.dry_run)
    if args.dry_run:
        print("Dry run - nothing written.")
    else:
        print(f"Backfilled {count} run(s) into MLflow.")


if __name__ == "__main__":
    main()

"""MLflow experiment tracking (SAD C11).

The rule this module exists to enforce: **tracking never breaks training.**
A run that trained a good model and failed to log it is a nuisance; a run that
threw away three hours of GPU time because a tracking server was down is a
disaster. Every entry point here degrades to a warning and lets the caller
continue, which is the same graceful-degradation stance the scoring path takes
when Neo4j or the audit database is unavailable (rules.md 4).

What gets logged is deliberately what the artifacts already record - params
from `feature_spec.json`, metrics from `metrics.json`. MLflow is a second,
queryable view of the artifact tree, never a second source of truth: if the two
disagree, the artifact is right, because that is what serving actually loads.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow

from src.common.config import get_settings

logger = logging.getLogger("mlops.tracking")

ARTIFACTS_ROOT = Path("artifacts/models")

# Params worth comparing runs by. Pulled from feature_spec.json rather than
# re-derived, so a run's recorded params are the ones serving will honour.
_SPEC_PARAMS = (
    "scale_pos_weight",
    "best_iteration",
    "train_rows",
    "train_positives",
)


def make_stdout_unicode_safe() -> None:
    """Stop MLflow's console output from killing a Windows CLI.

    MLflow prints a "View run" line containing an emoji when a run ends. On a
    Windows console defaulting to cp1252 that raises UnicodeEncodeError from
    inside `set_terminated`, which is about the worst place for it: the run has
    already been logged, so the crash leaves it stranded in RUNNING forever.
    Observed exactly that - two backfilled runs logged, both stuck RUNNING, the
    third never written.

    Re-encoding with `errors="replace"` costs a mangled emoji and nothing else.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                # A redirected or already-detached stream; not worth failing for.
                pass


def configure() -> str:
    """Point the MLflow client at the configured server and experiment."""
    make_stdout_unicode_safe()
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    return settings.mlflow_tracking_uri


def is_reachable(timeout_seconds: float = 3.0) -> bool:
    """Cheap pre-flight so callers can say "tracking is off" once, up front,
    instead of discovering it through a stack trace at the end of training."""
    settings = get_settings()
    try:
        import urllib.request

        urllib.request.urlopen(  # noqa: S310 - fixed, operator-supplied URL
            f"{settings.mlflow_tracking_uri.rstrip('/')}/health",
            timeout=timeout_seconds,
        )
        return True
    except Exception:
        return False


@contextmanager
def training_run(
    version: str,
    *,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> Generator[Any, None, None]:
    """Wrap a training run, logging to MLflow when it is available.

    Yields the active run, or `None` when tracking is unavailable - callers
    must tolerate `None` rather than assume a run exists.
    """
    if not is_reachable():
        logger.warning(
            "MLflow unreachable at %s - training will proceed untracked",
            get_settings().mlflow_tracking_uri,
        )
        yield None
        return

    configure()
    try:
        with mlflow.start_run(run_name=run_name or version) as run:
            mlflow.set_tags({"model_version": version, **(tags or {})})
            yield run
    except Exception:
        # Losing the tracking connection mid-run must not lose the model.
        logger.exception("MLflow run failed - training output is unaffected")
        yield None


def log_params(params: dict[str, Any]) -> None:
    _safely(lambda: mlflow.log_params(params), "params")


def log_metrics(metrics: dict[str, float]) -> None:
    _safely(lambda: mlflow.log_metrics(metrics), "metrics")


def log_artifact(path: Path) -> None:
    if not path.exists():
        return
    _safely(lambda: mlflow.log_artifact(str(path)), f"artifact {path.name}")


def _safely(action, what: str) -> None:
    try:
        action()
    except Exception:
        logger.warning("could not log %s to MLflow", what, exc_info=True)


# --------------------------------------------------------------------------
# Reading the artifact tree
# --------------------------------------------------------------------------


def read_spec(version: str, root: Path = ARTIFACTS_ROOT) -> dict[str, Any]:
    path = root / version / "feature_spec.json"
    return json.loads(path.read_text()) if path.exists() else {}


def read_metrics(version: str, root: Path = ARTIFACTS_ROOT) -> dict[str, Any]:
    path = root / version / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


def params_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Flatten a feature spec into MLflow params.

    `embeddings_path` becomes a first-class param because it is the single
    thing that distinguishes the baseline from the graph-augmented model - the
    whole comparison this project reports hinges on it.
    """
    params: dict[str, Any] = {
        key: spec[key] for key in _SPEC_PARAMS if spec.get(key) is not None
    }
    embeddings_path = spec.get("embeddings_path")
    params["uses_embeddings"] = embeddings_path is not None
    params["embedding_version"] = (
        Path(embeddings_path).parent.name if embeddings_path else "none"
    )
    if spec.get("feature_columns"):
        params["n_features"] = len(spec["feature_columns"])

    # Ablation runs must be self-describing in MLflow. Without this, an
    # ablation sits in the run list looking like a model that simply scored
    # badly, and someone comparing AUPRC across runs would be comparing models
    # trained on different feature sets without knowing it.
    dropped = spec.get("dropped_features") or []
    params["is_ablation"] = bool(dropped)
    params["dropped_features"] = ",".join(sorted(dropped)) if dropped else "none"
    return params


def metrics_from_eval(metrics: dict[str, Any]) -> dict[str, float]:
    """Flatten `metrics.json` into MLflow metrics.

    AUPRC is named `auprc` and listed first deliberately: it is the headline
    metric, and ROC-AUC is logged alongside it precisely so a run that improved
    one and regressed the other cannot be reported as an unqualified win.
    """
    flat: dict[str, float] = {}
    for key in ("auprc", "roc_auc", "positive_rate"):
        if metrics.get(key) is not None:
            flat[key] = float(metrics[key])

    operating_point = metrics.get("best_f1_operating_point") or {}
    for key in ("threshold", "precision", "recall", "f1"):
        if operating_point.get(key) is not None:
            flat[f"best_f1_{key}"] = float(operating_point[key])

    for key in ("test_rows", "test_positives"):
        if metrics.get(key) is not None:
            flat[key] = float(metrics[key])
    return flat

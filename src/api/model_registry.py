"""Reads trained-model metadata for the Model & Ops screen (Design Doc 4.4).

The source of truth is the artifact tree itself: every training run writes a
`metrics.json` next to its model, and every embedding run writes a
`metadata.json`. This module reads those rather than a database table, because
a table would be a *copy* of the artifacts and copies drift - the failure mode
where the dashboard proudly reports an AUPRC the serving model never had.

Nothing here computes a metric. If a number isn't in an artifact, this module
reports it as absent instead of deriving a plausible-looking substitute.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

ARTIFACTS_ROOT = Path("artifacts/models")
EMBEDDINGS_ROOT = Path("artifacts/embeddings")

logger = logging.getLogger("scoring.registry")


@dataclass(frozen=True)
class VersionMetrics:
    """One trained version's measured evaluation result."""

    version: str
    auprc: float
    roc_auc: float
    test_rows: int
    test_positives: int
    best_f1_threshold: float
    best_f1_precision: float
    best_f1_recall: float
    best_f1: float
    embedding_version: str | None
    # Features this version was trained WITHOUT. Empty for the main models;
    # populated for ablation runs. Comparisons must only be made within a
    # matching set - see `baseline_and_gnn`.
    dropped_features: tuple[str, ...] = ()


def _read_spec(version_dir: Path) -> dict:
    spec_path = version_dir / "feature_spec.json"
    if not spec_path.exists():
        return {}
    try:
        return json.loads(spec_path.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning("unreadable feature_spec.json in %s", version_dir, exc_info=True)
        return {}


def _read_embedding_version(version_dir: Path) -> str | None:
    """Which embeddings a version was trained on, or None if it is tabular-only.

    The spec records `embeddings_path`
    (`artifacts/embeddings/<version>/embeddings.parquet`), so the version is the
    parent directory name. Absent is a real answer - the tabular baseline has
    no embeddings - and is what separates the two sides of the headline
    comparison.
    """
    embeddings_path = _read_spec(version_dir).get("embeddings_path")
    if not embeddings_path:
        return None
    return Path(embeddings_path).parent.name


def _load_one(version_dir: Path) -> VersionMetrics | None:
    metrics_path = version_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        raw = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning("unreadable metrics.json in %s", version_dir, exc_info=True)
        return None

    operating_point = raw.get("best_f1_operating_point", {})
    try:
        return VersionMetrics(
            version=raw.get("model_version", version_dir.name),
            auprc=float(raw["auprc"]),
            roc_auc=float(raw["roc_auc"]),
            test_rows=int(raw["test_rows"]),
            test_positives=int(raw["test_positives"]),
            best_f1_threshold=float(operating_point["threshold"]),
            best_f1_precision=float(operating_point["precision"]),
            best_f1_recall=float(operating_point["recall"]),
            best_f1=float(operating_point["f1"]),
            embedding_version=_read_embedding_version(version_dir),
            dropped_features=tuple(_read_spec(version_dir).get("dropped_features") or ()),
        )
    except (KeyError, TypeError, ValueError):
        # A half-written metrics file is a real possibility mid-training run.
        # Skipping it shows one fewer row; guessing would show a wrong one.
        logger.warning("incomplete metrics.json in %s", version_dir, exc_info=True)
        return None


def list_versions(root: Path = ARTIFACTS_ROOT) -> list[VersionMetrics]:
    """Every trained version that has a readable metrics.json, best AUPRC first."""
    if not root.exists():
        return []
    found = [_load_one(child) for child in sorted(root.iterdir()) if child.is_dir()]
    return sorted((m for m in found if m is not None), key=lambda m: m.auprc, reverse=True)


def baseline_and_gnn(
    versions: list[VersionMetrics],
    dropped_features: tuple[str, ...] = (),
) -> tuple[VersionMetrics | None, VersionMetrics | None]:
    """Pick the pair behind a headline lift number.

    "Baseline" is the best version trained without embeddings and "GNN" the
    best with them, which is exactly the comparison the project claims: same
    data, same split, embeddings as the only difference.

    **Only versions with a matching `dropped_features` set are eligible.**
    Without that filter an ablation run could be paired against a full-feature
    model, and the resulting "lift" would be measuring two changes at once
    while presenting itself as one. Defaults to the full-feature comparison,
    which is the project's headline.
    """
    eligible = [m for m in versions if m.dropped_features == dropped_features]
    tabular = [m for m in eligible if m.embedding_version is None]
    fused = [m for m in eligible if m.embedding_version is not None]
    best = lambda group: max(group, key=lambda m: m.auprc) if group else None  # noqa: E731
    return best(tabular), best(fused)


def ablation_sets(versions: list[VersionMetrics]) -> list[tuple[str, ...]]:
    """Every distinct non-empty `dropped_features` set present, so callers can
    enumerate the ablations without hardcoding which were run."""
    return sorted({m.dropped_features for m in versions if m.dropped_features})


def lift_pct(baseline: VersionMetrics | None, gnn: VersionMetrics | None) -> float | None:
    if baseline is None or gnn is None or baseline.auprc <= 0:
        return None
    return (gnn.auprc - baseline.auprc) / baseline.auprc * 100.0

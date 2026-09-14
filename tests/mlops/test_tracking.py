"""MLflow tracking helpers.

The load-bearing property here is the one in the module docstring: **tracking
never breaks training.** Most of these tests are about what happens when MLflow
is absent or broken, because that is the path that must not take a training run
down with it.
"""

from __future__ import annotations

import json

import pytest

from src.mlops import tracking


@pytest.fixture
def artifact_tree(tmp_path):
    version = tmp_path / "gnn_test"
    version.mkdir()
    (version / "feature_spec.json").write_text(
        json.dumps(
            {
                "model_version": "gnn_test",
                "embeddings_path": "artifacts/embeddings/gnn_emb_test/embeddings.parquet",
                "feature_columns": ["amount_paid", "hour_of_day", "sender_emb_0"],
                "scale_pos_weight": 2155.47,
                "best_iteration": 22,
                "train_rows": 5091438,
                "train_positives": 2361,
            }
        )
    )
    (version / "metrics.json").write_text(
        json.dumps(
            {
                "auprc": 0.0218,
                "roc_auc": 0.9079,
                "positive_rate": 0.00068,
                "test_rows": 1174673,
                "test_positives": 802,
                "best_f1_operating_point": {
                    "threshold": 0.96,
                    "precision": 0.0568,
                    "recall": 0.1584,
                    "f1": 0.0837,
                },
            }
        )
    )
    return tmp_path


def test_params_capture_the_embedding_distinction(artifact_tree):
    """`uses_embeddings` is the one param the whole baseline-vs-GNN comparison
    turns on, so it must always be present and correct."""
    params = tracking.params_from_spec(tracking.read_spec("gnn_test", root=artifact_tree))

    assert params["uses_embeddings"] is True
    assert params["embedding_version"] == "gnn_emb_test"
    assert params["n_features"] == 3
    assert params["train_rows"] == 5091438


def test_a_tabular_model_reports_no_embeddings(tmp_path):
    version = tmp_path / "baseline"
    version.mkdir()
    (version / "feature_spec.json").write_text(
        json.dumps({"model_version": "baseline", "feature_columns": ["amount_paid"]})
    )

    params = tracking.params_from_spec(tracking.read_spec("baseline", root=tmp_path))
    assert params["uses_embeddings"] is False
    assert params["embedding_version"] == "none"


def test_metrics_keep_auprc_and_roc_auc_together(artifact_tree):
    """Logging AUPRC without ROC-AUC would let a run that improved one and
    regressed the other be reported as an unqualified win."""
    flat = tracking.metrics_from_eval(tracking.read_metrics("gnn_test", root=artifact_tree))

    assert flat["auprc"] == pytest.approx(0.0218)
    assert flat["roc_auc"] == pytest.approx(0.9079)
    assert flat["best_f1_recall"] == pytest.approx(0.1584)
    assert flat["test_positives"] == 802


def test_missing_artifacts_read_as_empty(tmp_path):
    assert tracking.read_spec("nope", root=tmp_path) == {}
    assert tracking.read_metrics("nope", root=tmp_path) == {}


def test_metrics_from_an_empty_file_are_empty_not_zero():
    """A version with no evaluation must log no metrics, rather than a
    confident set of zeros."""
    assert tracking.metrics_from_eval({}) == {}


def test_training_run_yields_none_when_mlflow_is_down(monkeypatch):
    """The whole point: a tracking outage must not stop training."""
    monkeypatch.setattr(tracking, "is_reachable", lambda *a, **k: False)

    with tracking.training_run("v1") as run:
        assert run is None  # caller continues, untracked


def test_logging_helpers_swallow_failures(monkeypatch):
    """log_* are called from inside training; they must never raise."""

    def explode(*args, **kwargs):
        raise RuntimeError("tracking server went away")

    monkeypatch.setattr(tracking.mlflow, "log_params", explode)
    monkeypatch.setattr(tracking.mlflow, "log_metrics", explode)

    tracking.log_params({"a": 1})
    tracking.log_metrics({"b": 2.0})


def test_unreachable_server_reports_false(monkeypatch):
    monkeypatch.setattr(
        tracking.get_settings(), "mlflow_tracking_uri", "http://127.0.0.1:1", raising=False
    )
    assert tracking.is_reachable(timeout_seconds=1.0) is False

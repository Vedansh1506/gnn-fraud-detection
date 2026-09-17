"""Model & Ops data comes from the artifact tree, so these tests pin the two
things that would quietly corrupt the headline number: which versions count as
"baseline" vs "GNN", and what happens to an artifact that can't be read.

No database and no trained model needed - the registry reads JSON files.
"""

from __future__ import annotations

import json

import pytest

from src.api.model_registry import ablation_sets, baseline_and_gnn, lift_pct, list_versions


def _write_version(root, name, auprc, roc_auc=0.9, embeddings_version=None, dropped=None):
    version_dir = root / name
    version_dir.mkdir(parents=True)
    (version_dir / "metrics.json").write_text(
        json.dumps(
            {
                "auprc": auprc,
                "roc_auc": roc_auc,
                "test_rows": 1000,
                "test_positives": 10,
                "best_f1_operating_point": {
                    "threshold": 0.9,
                    "precision": 0.05,
                    "recall": 0.15,
                    "f1": 0.08,
                },
                "model_version": name,
            }
        )
    )
    spec = {"model_version": name, "feature_columns": ["amount_paid"]}
    if embeddings_version is not None:
        spec["embeddings_path"] = f"artifacts/embeddings/{embeddings_version}/embeddings.parquet"
    if dropped is not None:
        spec["dropped_features"] = list(dropped)
    (version_dir / "feature_spec.json").write_text(json.dumps(spec))
    return version_dir


@pytest.fixture
def registry_root(tmp_path):
    _write_version(tmp_path, "baseline_v1", auprc=0.02)
    _write_version(tmp_path, "gnn_v1", auprc=0.021, embeddings_version="gnn_emb_v1")
    _write_version(tmp_path, "gnn_v2", auprc=0.022, embeddings_version="gnn_emb_v2")
    return tmp_path


def test_versions_are_returned_best_auprc_first(registry_root):
    versions = list_versions(registry_root)
    assert [v.version for v in versions] == ["gnn_v2", "gnn_v1", "baseline_v1"]


def test_embedding_version_is_derived_from_the_embeddings_path(registry_root):
    versions = {v.version: v for v in list_versions(registry_root)}
    assert versions["gnn_v2"].embedding_version == "gnn_emb_v2"
    # The tabular baseline genuinely has none - this is what splits the pair.
    assert versions["baseline_v1"].embedding_version is None


def test_headline_pair_is_best_tabular_versus_best_fused(registry_root):
    baseline, gnn = baseline_and_gnn(list_versions(registry_root))
    assert baseline.version == "baseline_v1"
    assert gnn.version == "gnn_v2"
    assert lift_pct(baseline, gnn) == pytest.approx(10.0)


def test_a_version_with_no_metrics_is_skipped_not_guessed(registry_root):
    """A training run that hasn't been evaluated yet must not appear with a
    fabricated score - it must not appear at all."""
    (registry_root / "gnn_v3").mkdir()
    assert "gnn_v3" not in {v.version for v in list_versions(registry_root)}


def test_an_unreadable_metrics_file_is_skipped_not_fatal(registry_root):
    """One corrupt artifact must not blank the whole Ops screen."""
    broken = registry_root / "gnn_v4"
    broken.mkdir()
    (broken / "metrics.json").write_text("{not json")

    versions = list_versions(registry_root)
    assert "gnn_v4" not in {v.version for v in versions}
    assert len(versions) == 3


def test_incomplete_metrics_file_is_skipped(registry_root):
    """Half-written metrics (a crashed eval run) must not surface as a row with
    zeros in the missing fields."""
    partial = registry_root / "gnn_v5"
    partial.mkdir()
    (partial / "metrics.json").write_text(json.dumps({"auprc": 0.05}))
    assert "gnn_v5" not in {v.version for v in list_versions(registry_root)}


def test_missing_artifact_root_returns_empty_not_error(tmp_path):
    assert list_versions(tmp_path / "does-not-exist") == []


def test_lift_is_none_when_either_side_is_missing(registry_root):
    versions = list_versions(registry_root)
    _, gnn = baseline_and_gnn(versions)
    assert lift_pct(None, gnn) is None
    assert lift_pct(gnn, None) is None


def test_real_artifacts_reproduce_the_documented_lift():
    """Guards the project's headline claim against a silent artifact change.

    The documented result is baseline 0.0198 -> GNN 0.0218, about +10%. If this
    fails, either the artifacts changed or the pairing logic did - both are
    things that must never happen quietly.
    """
    versions = list_versions()
    if not versions:
        pytest.skip("no trained artifacts present - run the training pipeline first")

    baseline, gnn = baseline_and_gnn(versions)
    if baseline is None or gnn is None:
        pytest.skip("need both a tabular and an embedding-fused version")

    assert baseline.auprc == pytest.approx(0.0198, abs=5e-4)
    assert gnn.auprc == pytest.approx(0.0218, abs=5e-4)
    assert lift_pct(baseline, gnn) == pytest.approx(10.3, abs=0.5)


def test_an_ablation_is_never_paired_against_a_full_feature_model(registry_root):
    """The failure this guards: an ablation run scoring higher than the real
    baseline would be picked as "baseline", and the reported lift would be
    measuring the dropped feature AND the embeddings at once while presenting
    itself as measuring only the embeddings."""
    _write_version(registry_root, "baseline_nopf_v1", auprc=0.05, dropped=["payment_format"])
    _write_version(
        registry_root, "gnn_nopf_v1", auprc=0.09,
        embeddings_version="gnn_emb_v2", dropped=["payment_format"],
    )
    versions = list_versions(registry_root)

    baseline, gnn = baseline_and_gnn(versions)
    assert baseline.version == "baseline_v1"
    assert gnn.version == "gnn_v2"
    assert baseline.dropped_features == ()
    assert gnn.dropped_features == ()


def test_the_ablation_pair_can_be_requested_explicitly(registry_root):
    _write_version(registry_root, "baseline_nopf_v1", auprc=0.05, dropped=["payment_format"])
    _write_version(
        registry_root, "gnn_nopf_v1", auprc=0.09,
        embeddings_version="gnn_emb_v2", dropped=["payment_format"],
    )
    versions = list_versions(registry_root)

    baseline, gnn = baseline_and_gnn(versions, dropped_features=("payment_format",))
    assert baseline.version == "baseline_nopf_v1"
    assert gnn.version == "gnn_nopf_v1"
    assert lift_pct(baseline, gnn) == pytest.approx(80.0)


def test_ablation_sets_enumerates_what_was_run(registry_root):
    _write_version(registry_root, "baseline_nopf_v1", auprc=0.05, dropped=["payment_format"])
    assert ablation_sets(list_versions(registry_root)) == [("payment_format",)]


def test_no_ablations_means_an_empty_list(registry_root):
    assert ablation_sets(list_versions(registry_root)) == []

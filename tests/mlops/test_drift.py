"""Drift detection.

Weighted toward the two things that would quietly produce a *wrong verdict*
rather than a visible error: parsing Evidently's result payload, and the rule
that turns per-column results into a single state. A drift panel that says
"ok" when it could not actually tell is worse than one that says nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mlops import drift

# --------------------------------------------------------------- parsing

def _value_drift_metric(
    column: str,
    value: float,
    threshold: float = 0.1,
    method: str = "Wasserstein distance (normed)",
):
    return {
        "metric_name": f"ValueDrift(column={column},method={method},threshold={threshold})",
        "value": value,
        "config": {"type": "evidently:metric_v2:ValueDrift", "threshold": threshold},
    }


def test_parses_column_name_and_threshold():
    parsed = drift._parse_result({"metrics": [_value_drift_metric("amount_paid", 14.9)]})

    assert len(parsed) == 1
    assert parsed[0].column == "amount_paid"
    assert parsed[0].drifted is True
    assert parsed[0].score == pytest.approx(14.9)


def test_value_below_threshold_is_not_drift():
    parsed = drift._parse_result({"metrics": [_value_drift_metric("hour_of_day", 0.036)]})
    assert parsed[0].drifted is False


def test_value_exactly_at_threshold_counts_as_drift():
    """Evidently's own convention, applied rather than reinvented."""
    parsed = drift._parse_result({"metrics": [_value_drift_metric("score", 0.1)]})
    assert parsed[0].drifted is True


def test_dataset_level_metric_is_ignored():
    """DriftedColumnsCount is a summary, not a column - counting it as one
    would inflate the denominator and dilute the drifted share."""
    payload = {
        "metrics": [
            {
                "metric_name": "DriftedColumnsCount(drift_share=0.5)",
                "value": {"count": 2.0, "share": 0.4},
            },
            _value_drift_metric("amount_paid", 14.9),
        ]
    }
    parsed = drift._parse_result(payload)
    assert [c.column for c in parsed] == ["amount_paid"]


def test_unrecognised_payload_yields_nothing_not_a_guess():
    """If Evidently's shape changes, the caller must report 'uncomputable' -
    never a confident 'no drift'."""
    assert drift._parse_result({"metrics": [{"metric_name": None, "value": None}]}) == []
    assert drift._parse_result({}) == []


def test_metric_without_threshold_is_skipped():
    payload = {
        "metrics": [
            {
                "metric_name": "ValueDrift(column=x,method=m,threshold=0.1)",
                "value": 5.0,
                "config": {},
            }
        ]
    }
    assert drift._parse_result(payload) == []


# --------------------------------------------------- reference artifacts

def _frame(rows: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "amount_paid": rng.normal(1000, 200, rows),
            "hour_of_day": rng.integers(0, 24, rows),
            "payment_format": rng.choice(["ACH", "Wire"], rows),
            "score": rng.beta(2, 30, rows),
        }
    )


def test_reference_round_trips(tmp_path):
    drift.build_reference(_frame(), "test_v1", root=tmp_path)
    loaded = drift.load_reference("test_v1", root=tmp_path)

    assert loaded is not None
    assert len(loaded) == 500
    assert set(loaded.columns) >= {"amount_paid", "hour_of_day", "payment_format", "score"}


def test_reference_sampling_is_deterministic(tmp_path):
    """A reference that shifted between builds would show drift that is purely
    our own sampling noise."""
    frame = _frame(rows=2000)
    drift.build_reference(frame, "a", root=tmp_path, sample=100)
    drift.build_reference(frame, "b", root=tmp_path, sample=100)

    pd.testing.assert_frame_equal(
        drift.load_reference("a", root=tmp_path), drift.load_reference("b", root=tmp_path)
    )


def test_missing_reference_loads_as_none(tmp_path):
    assert drift.load_reference("never_trained", root=tmp_path) is None


# ------------------------------------------------------- the real report

def test_evidently_detects_a_real_shift():
    """End-to-end through Evidently: a moved amount distribution must register."""
    reference = _frame(rows=1500, seed=1)
    current = _frame(rows=800, seed=2)
    current["amount_paid"] = current["amount_paid"] + 5000  # unmistakable shift

    parsed = drift._evaluate(reference, current)
    by_column = {c.column: c for c in parsed}

    assert by_column["amount_paid"].drifted is True
    # An untouched column must not be swept along with it.
    assert by_column["hour_of_day"].drifted is False


def test_identical_data_shows_no_drift():
    frame = _frame(rows=1200, seed=3)
    parsed = drift._evaluate(frame, frame.copy())
    assert all(not c.drifted for c in parsed)


# ------------------------------------------------- representativeness guards

def _timed_frame(rows: int, span_hours: float, seed: int = 7) -> pd.DataFrame:
    """A current-side frame spanning a given amount of transaction time."""
    start = pd.Timestamp("2022-09-09T00:00:00Z")
    frame = _frame(rows=rows, seed=seed)
    offsets = pd.to_timedelta(np.linspace(0, span_hours, rows), unit="h")
    frame["tx_timestamp"] = start + offsets
    frame["hour_of_day"] = frame["tx_timestamp"].dt.hour
    return frame


def test_span_hours_measures_transaction_time():
    frame = _timed_frame(rows=100, span_hours=12)
    assert drift._span_hours(frame) == pytest.approx(12.0, abs=0.01)


def test_span_is_none_without_timestamps():
    assert drift._span_hours(_frame()) is None


def test_a_constant_column_is_excluded_from_comparison():
    """Drift on a constant is not a verdict. Observed live: a short replay put
    every row in one clock hour, and `hour_of_day` then registered as severe
    drift that was purely the replay window."""
    reference = _frame(rows=800, seed=11)
    current = _frame(rows=400, seed=12)
    current["hour_of_day"] = 0  # what a sub-hour replay actually looks like

    assert "hour_of_day" not in drift._comparable_columns(reference, current)
    # The columns that still vary are unaffected.
    assert "amount_paid" in drift._comparable_columns(reference, current)


def test_narrow_replay_reports_uncomputable_not_drift(tmp_path, monkeypatch):
    """The guard that matters for demos: 20,000 transactions covering 17
    minutes must not be reported as an alert."""
    drift.build_reference(_frame(rows=2000, seed=1), "guard_v1", root=tmp_path)
    monkeypatch.setattr(drift, "DRIFT_ROOT", tmp_path)
    monkeypatch.setattr(drift, "load_current", lambda *a, **k: _timed_frame(1000, span_hours=0.3))

    report = drift.check_drift("guard_v1")
    assert report.state == "not_instrumented"
    assert "too narrow" in report.message
    assert report.checked_at is None


def test_a_wide_window_does_produce_a_verdict(tmp_path, monkeypatch):
    """The mirror of the guard above - given a representative span, the check
    must actually commit to an answer rather than always abstaining."""
    reference = _frame(rows=3000, seed=1)
    drift.build_reference(reference, "wide_v1", root=tmp_path)
    monkeypatch.setattr(drift, "DRIFT_ROOT", tmp_path)

    current = _timed_frame(1500, span_hours=48, seed=1)
    monkeypatch.setattr(drift, "load_current", lambda *a, **k: current)

    report = drift.check_drift("wide_v1")
    assert report.state in {"ok", "warning", "alert"}
    assert report.checked_at is not None
    assert report.total_columns > 0


def test_a_wide_window_with_a_real_shift_escalates(tmp_path, monkeypatch):
    drift.build_reference(_frame(rows=3000, seed=1), "shift_v1", root=tmp_path)
    monkeypatch.setattr(drift, "DRIFT_ROOT", tmp_path)

    shifted = _timed_frame(1500, span_hours=48, seed=2)
    shifted["amount_paid"] = shifted["amount_paid"] + 8000
    shifted["score"] = shifted["score"] + 0.4
    monkeypatch.setattr(drift, "load_current", lambda *a, **k: shifted)

    report = drift.check_drift("shift_v1")
    assert report.state in {"warning", "alert"}
    assert report.drifted_columns > 0


def test_missing_reference_is_not_instrumented(tmp_path, monkeypatch):
    monkeypatch.setattr(drift, "DRIFT_ROOT", tmp_path)
    report = drift.check_drift("never_built")

    assert report.state == "not_instrumented"
    assert "build_reference" in report.message
    assert report.checked_at is None

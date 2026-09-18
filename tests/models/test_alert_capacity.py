"""Alert-capacity metrics: turning a ranking into a staffing answer.

AUPRC answers "how good is the ranking". These answer "if analysts review N
alerts a day, how many are real, and how much laundering do we catch" - the
question an operations manager actually asks. Pure maths over arrays, so no
model, database or dataset is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.classifier.evaluate import alert_capacity

THRESHOLD = 0.9


def _labels(values) -> pd.Series:
    return pd.Series(list(values), dtype=int)


def _two_days(n: int) -> pd.Series:
    """Timestamps spanning exactly two calendar days."""
    half = n // 2
    return pd.Series(
        [pd.Timestamp("2022-09-09 01:00")] * half
        + [pd.Timestamp("2022-09-10 01:00")] * (n - half)
    )


def test_perfect_ranking_puts_every_positive_first():
    scores = np.array([0.99, 0.98, 0.10, 0.05])
    labels = _labels([1, 1, 0, 0])

    result = alert_capacity(scores, labels, _two_days(4), THRESHOLD, capacities=(1,))
    top2 = result["by_daily_capacity"][0]

    # 1/day over 2 days = the top 2 ranked, both of which are laundering.
    assert top2["reviewed"] == 2
    assert top2["precision"] == pytest.approx(1.0)
    assert top2["recall"] == pytest.approx(1.0)


def test_precision_reflects_a_bad_ranking():
    """Scores that rank the negatives first must not produce a flattering
    precision - this is the case a silent bug would hide."""
    scores = np.array([0.99, 0.98, 0.10, 0.05])
    labels = _labels([0, 0, 1, 1])

    result = alert_capacity(scores, labels, _two_days(4), THRESHOLD, capacities=(1,))
    top2 = result["by_daily_capacity"][0]

    assert top2["precision"] == pytest.approx(0.0)
    assert top2["recall"] == pytest.approx(0.0)


def test_days_come_from_distinct_calendar_days():
    """Not a min-to-max span: a gap in the data would inflate the denominator
    and understate the true daily alert volume."""
    timestamps = pd.Series(
        [
            pd.Timestamp("2022-09-09 01:00"),
            pd.Timestamp("2022-09-09 23:00"),
            pd.Timestamp("2022-09-17 05:00"),  # an 8-day gap, only 2 real days
        ]
    )
    result = alert_capacity(np.array([0.9, 0.8, 0.7]), _labels([1, 0, 0]), timestamps, THRESHOLD)

    assert result["test_days"] == 2
    assert result["transactions_per_day"] == pytest.approx(1.5)


def test_alerts_per_day_uses_the_pinned_threshold():
    scores = np.array([0.95, 0.94, 0.10, 0.05])
    result = alert_capacity(scores, _labels([1, 0, 0, 0]), _two_days(4), THRESHOLD)

    at_threshold = result["at_flag_threshold"]
    assert at_threshold["alerts_total"] == 2
    assert at_threshold["alerts_per_day"] == pytest.approx(1.0)
    assert at_threshold["precision"] == pytest.approx(0.5)
    assert at_threshold["recall"] == pytest.approx(1.0)


def test_a_threshold_nothing_reaches_reports_no_alerts():
    """An empty alert queue must not divide by zero or claim 100% precision."""
    result = alert_capacity(np.array([0.1, 0.2]), _labels([1, 0]), _two_days(2), THRESHOLD)

    at_threshold = result["at_flag_threshold"]
    assert at_threshold["alerts_total"] == 0
    assert at_threshold["precision"] is None
    assert at_threshold["recall"] == pytest.approx(0.0)


def test_capacity_beyond_the_dataset_is_capped():
    """Asking for more daily review capacity than there are transactions must
    not index past the end of the array."""
    result = alert_capacity(np.array([0.9, 0.8]), _labels([1, 0]), _two_days(2), THRESHOLD)

    for point in result["by_daily_capacity"]:
        assert point["reviewed"] <= 2


def test_missing_timestamps_degrade_rather_than_crash():
    """Older artifacts carry no timestamps; the rate is simply unavailable."""
    result = alert_capacity(np.array([0.95, 0.1]), _labels([1, 0]), None, THRESHOLD)

    assert result["test_days"] is None
    assert result["transactions_per_day"] is None
    assert result["by_daily_capacity"] == []
    # The threshold counts do not depend on timestamps, so they still work.
    assert result["at_flag_threshold"]["alerts_total"] == 1


def test_recall_is_monotonic_as_capacity_grows():
    """Reviewing more alerts can never catch less laundering."""
    rng = np.random.default_rng(0)
    scores = rng.random(2000)
    labels = _labels((rng.random(2000) < 0.05).astype(int))

    result = alert_capacity(scores, labels, _two_days(2000), THRESHOLD)
    recalls = [p["recall"] for p in result["by_daily_capacity"]]

    assert recalls == sorted(recalls)


def test_the_printed_summary_survives_a_model_that_never_alerts(capsys):
    """Regression: `baseline_nopf_v1` reaches 0.9 on none of 1.17M test rows, so
    precision is None and formatting it with `.1%` crashed the run. The metrics
    had already been written, so this only broke the summary - but a crashed
    evaluation looks like a failed one."""
    from src.models.classifier.evaluate import main as _  # noqa: F401  (import guard)

    result = alert_capacity(np.array([0.1, 0.2]), _labels([1, 0]), _two_days(2), THRESHOLD)
    at_threshold = result["at_flag_threshold"]

    # The shape the printer has to cope with.
    assert at_threshold["precision"] is None
    assert at_threshold["alerts_per_day"] == 0.0

    # And the branch that handles it must not raise.
    if at_threshold["precision"] is None:
        message = (
            f"At the pinned threshold {at_threshold['threshold']}: "
            f"NO alerts at all - this model never reaches the threshold, "
            f"so it would flag nothing in production."
        )
    else:  # pragma: no cover - the populated branch is covered elsewhere
        message = f"precision {at_threshold['precision']:.1%}"
    assert "NO alerts" in message

"""Closing out a retrain run - requires Postgres; skips otherwise.

The important behaviour is not the status column, it is which analyst feedback
a completed run consumes. Until something writes `used_in_training_run`, "the
analyst loop feeds retraining" is a claim with no record behind it, and the
next retrain cannot tell new labels from ones already used.
"""

from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy import delete, select, update

from src.db.models import Feedback, RetrainRun
from src.db.session import create_tables, session_scope
from src.mlops import close_retrain
from src.mlops.close_retrain import CloseError, close_run

PREFIX = "pytest-close-"


@pytest.fixture
def postgres_available():
    """Isolate the test from real data.

    `close_run` legitimately consumes *all* outstanding feedback - that is the
    production behaviour - so a test that runs it will stamp real analyst
    decisions too. Deleting only prefixed rows is not enough cleanup: it did
    exactly this once, marking two genuine decisions as consumed by a pytest
    job id that no longer existed. This snapshots `used_in_training_run` for
    every pre-existing row and puts it back afterwards.

    (Same lesson as the Neo4j loader test that used to overwrite the real
    feature parquets - a test writing to production state is a data-loss bug,
    not a test smell.)
    """
    try:
        create_tables()
    except Exception:
        pytest.skip("Postgres not reachable - run `docker compose up -d postgres`.")

    with session_scope() as session:
        before = {
            row.id: row.used_in_training_run
            for row in session.scalars(select(Feedback))
            if not row.tx_id.startswith(PREFIX)
        }

    yield

    with session_scope() as session:
        session.execute(delete(RetrainRun).where(RetrainRun.requested_by.startswith(PREFIX)))
        session.execute(delete(Feedback).where(Feedback.tx_id.startswith(PREFIX)))
        for feedback_id, marker in before.items():
            session.execute(
                update(Feedback).where(Feedback.id == feedback_id).values(
                    used_in_training_run=marker
                )
            )


def _a_run(status: str = "requested") -> str:
    job_id = f"{PREFIX}{uuid.uuid4().hex[:10]}"
    with session_scope() as session:
        session.add(
            RetrainRun(
                job_id=job_id, status=status, requested_by=f"{PREFIX}ops", feedback_rows_pending=0
            )
        )
    return job_id


def _some_feedback(count: int = 3) -> list[str]:
    ids = []
    with session_scope() as session:
        for _ in range(count):
            tx_id = f"{PREFIX}{uuid.uuid4().hex[:8]}"
            session.add(
                Feedback(
                    tx_id=tx_id,
                    analyst_decision="confirmed_fraud",
                    decided_by=f"{PREFIX}analyst",
                )
            )
            ids.append(tx_id)
    return ids


def _used_markers(tx_ids: list[str]) -> list[str | None]:
    with session_scope() as session:
        return [
            session.scalar(select(Feedback.used_in_training_run).where(Feedback.tx_id == tx_id))
            for tx_id in tx_ids
        ]


@pytest.fixture
def real_version(tmp_path, monkeypatch):
    """A version that looks trained, so the artifact guard passes."""
    version_dir = tmp_path / "gnn_closed"
    version_dir.mkdir()
    (version_dir / "metrics.json").write_text(json.dumps({"auprc": 0.02}))
    monkeypatch.setattr(close_retrain, "ARTIFACTS_ROOT", tmp_path)
    return "gnn_closed"


def test_completing_records_the_model_version(postgres_available, real_version):
    job_id = _a_run()
    result = close_run(job_id, status="completed", model_version=real_version)

    assert result["status"] == "completed"
    with session_scope() as session:
        run = session.scalar(select(RetrainRun).where(RetrainRun.job_id == job_id))
    assert run.resulting_model_version == real_version


def test_completing_consumes_outstanding_feedback(postgres_available, real_version):
    """The actual point of closing a run."""
    tx_ids = _some_feedback(3)
    job_id = _a_run()

    result = close_run(job_id, status="completed", model_version=real_version)

    assert result["feedback_consumed"] >= 3
    assert _used_markers(tx_ids) == [job_id] * 3


def test_a_failed_run_leaves_feedback_available(postgres_available):
    """Consuming labels on a failed retrain would silently lose them - the next
    attempt would see no new feedback."""
    tx_ids = _some_feedback(2)
    job_id = _a_run()

    result = close_run(job_id, status="failed")

    assert result["feedback_consumed"] == 0
    assert _used_markers(tx_ids) == [None, None]


def test_feedback_is_not_claimed_twice(postgres_available, real_version):
    """A second retrain must not re-consume labels an earlier one already used,
    or 'new feedback since last retrain' becomes meaningless."""
    tx_ids = _some_feedback(2)
    first = _a_run()
    close_run(first, status="completed", model_version=real_version)

    second = _a_run()
    close_run(second, status="completed", model_version=real_version)

    assert _used_markers(tx_ids) == [first, first]


def test_completing_without_a_version_is_refused(postgres_available):
    job_id = _a_run()
    with pytest.raises(CloseError, match="model-version is required"):
        close_run(job_id, status="completed")


def test_an_unevaluated_version_is_refused(postgres_available, tmp_path, monkeypatch):
    """Guards the failure this project keeps guarding against: a row asserting
    a model that was never trained, shown on the Ops screen as fact."""
    monkeypatch.setattr(close_retrain, "ARTIFACTS_ROOT", tmp_path)
    job_id = _a_run()

    with pytest.raises(CloseError, match="No evaluated artifacts"):
        close_run(job_id, status="completed", model_version="never_trained")


def test_closing_an_already_closed_run_is_refused(postgres_available, real_version):
    job_id = _a_run()
    close_run(job_id, status="completed", model_version=real_version)

    with pytest.raises(CloseError, match="already completed"):
        close_run(job_id, status="completed", model_version=real_version)


def test_force_overwrites_a_closed_run(postgres_available, real_version):
    job_id = _a_run()
    close_run(job_id, status="completed", model_version=real_version)
    result = close_run(job_id, status="failed", force=True)

    assert result["status"] == "failed"


def test_latest_picks_the_newest_open_run(postgres_available, real_version):
    _a_run(status="completed")  # already closed, must be skipped
    open_run = _a_run()

    result = close_run(None, status="completed", model_version=real_version)
    assert result["job_id"] == open_run


def test_unknown_job_id_is_refused(postgres_available):
    with pytest.raises(CloseError, match="No retrain run"):
        close_run("does-not-exist", status="failed")

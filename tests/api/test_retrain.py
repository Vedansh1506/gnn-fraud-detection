"""/retrain records a work order (Design Doc 4.4 needs real history to show).

The honest framing matters here and is asserted below: the locked compute
decision puts GNN training on Kaggle/Colab, so the API cannot run a retrain. A
row means an operator *requested* one - it must never read as "a retrain ran".

Requires Postgres; skips otherwise.
"""

from __future__ import annotations

import pytest
from sqlalchemy import delete, select

from src.db.models import REQUESTED, Feedback, RetrainRun
from src.db.session import create_tables, session_scope

TEST_PREFIX = "pytest-retrain-"


@pytest.fixture
def postgres_available():
    try:
        create_tables()
    except Exception:
        pytest.skip("Postgres not reachable - run `docker compose up -d postgres`.")
    yield
    with session_scope() as session:
        session.execute(delete(RetrainRun).where(RetrainRun.requested_by.startswith(TEST_PREFIX)))
        session.execute(delete(Feedback).where(Feedback.tx_id.startswith(TEST_PREFIX)))


def _request_retrain(username: str):
    from src.api.main import retrain

    return retrain(claims={"sub": username, "role": "operator"})


def test_a_request_is_persisted(postgres_available):
    user = f"{TEST_PREFIX}ops"
    response = _request_retrain(user)

    with session_scope() as session:
        run = session.scalar(select(RetrainRun).where(RetrainRun.job_id == response.job_id))

    assert run is not None
    assert run.requested_by == user
    assert run.job_id == response.job_id


def test_status_is_requested_not_completed(postgres_available):
    """The API has no GPU and runs nothing. A row claiming otherwise would be
    the dashboard reporting work that never happened."""
    response = _request_retrain(f"{TEST_PREFIX}ops")

    assert response.status == REQUESTED
    with session_scope() as session:
        run = session.scalar(select(RetrainRun).where(RetrainRun.job_id == response.job_id))
    assert run.status == REQUESTED
    assert run.resulting_model_version is None


def test_pending_feedback_is_captured_at_request_time(postgres_available):
    """Why the operator retrained, recorded when they decided - not recomputed
    later, by which time the number has moved."""
    before = _request_retrain(f"{TEST_PREFIX}ops").feedback_rows_pending

    with session_scope() as session:
        session.add(
            Feedback(
                tx_id=f"{TEST_PREFIX}tx1",
                analyst_decision="confirmed_fraud",
                decided_by=f"{TEST_PREFIX}analyst",
            )
        )

    after = _request_retrain(f"{TEST_PREFIX}ops").feedback_rows_pending
    assert after == before + 1


def test_job_ids_are_unique_across_requests(postgres_available):
    first = _request_retrain(f"{TEST_PREFIX}ops").job_id
    second = _request_retrain(f"{TEST_PREFIX}ops").job_id
    assert first != second


def test_instructions_describe_the_manual_steps(postgres_available):
    """The response has to tell the operator what to actually run, because the
    service isn't going to do it."""
    response = _request_retrain(f"{TEST_PREFIX}ops")
    assert "train_graphsage" in response.instructions
    assert "evaluate" in response.instructions

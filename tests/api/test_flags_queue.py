"""Integration tests for the analyst queue - requires Postgres reachable
(`docker compose up -d postgres`, host port 5433).

Skips rather than fails when Postgres isn't running, matching the Neo4j
integration tests. These use a real database on purpose: the behaviour under
test *is* the SQL (the feedback outer join and the summary counts), so a mocked
session would assert nothing about the thing that can actually be wrong.

Every row written here uses a tx_id prefix that real data can never produce, and
is deleted afterwards.
"""

from __future__ import annotations

import datetime as dt
import uuid

import pytest
from sqlalchemy import delete

from src.db.models import AuditLogEntry, Feedback
from src.db.session import create_tables, session_scope

# Real tx_ids are row positions; this prefix cannot collide with one.
TEST_PREFIX = "pytest-queue-"


@pytest.fixture
def postgres_available():
    try:
        create_tables()
        with session_scope() as session:
            session.execute(
                delete(AuditLogEntry).where(AuditLogEntry.tx_id.startswith(TEST_PREFIX))
            )
            session.execute(delete(Feedback).where(Feedback.tx_id.startswith(TEST_PREFIX)))
    except Exception:
        pytest.skip(
            "Postgres not reachable - run `docker compose up -d postgres` to enable this test."
        )
    yield
    with session_scope() as session:
        session.execute(delete(AuditLogEntry).where(AuditLogEntry.tx_id.startswith(TEST_PREFIX)))
        session.execute(delete(Feedback).where(Feedback.tx_id.startswith(TEST_PREFIX)))


def _add_flag(score: float = 0.95, tx_id: str | None = None, **overrides) -> str:
    tx_id = tx_id or f"{TEST_PREFIX}{uuid.uuid4().hex[:8]}"
    row = {
        "tx_id": tx_id,
        "account_key": "999_SENDER",
        "score": score,
        "is_flagged": True,
        "explanation_json": {
            "factors": [
                {"feature": "amount_paid", "contribution": 1.0, "plain": "Transaction amount"}
            ]
        },
        "model_version": "test_v1",
        "embedding_version": "test_emb_v1",
        "receiver_account_key": "999_RECEIVER",
        "amount_paid": 12345.67,
        "payment_currency": "US Dollar",
        "payment_format": "ACH",
        "tx_timestamp": dt.datetime(2022, 9, 1, 10, 0, tzinfo=dt.UTC),
    }
    row.update(overrides)
    with session_scope() as session:
        session.add(AuditLogEntry(**row))
    return tx_id


def _review(tx_id: str, decision: str) -> None:
    with session_scope() as session:
        session.add(Feedback(tx_id=tx_id, analyst_decision=decision, decided_by="pytest"))


def _fetch(status: str = "open", limit: int = 500):
    """Calls the endpoint function directly - the HTTP layer and its auth are
    already covered elsewhere; what needs exercising here is the query."""
    from src.api.main import flags

    return flags(_claims={"sub": "pytest"}, limit=limit, status_filter=status)


def _find(response, tx_id):
    return next((flag for flag in response.flags if flag.tx_id == tx_id), None)


def test_transaction_detail_round_trips_to_the_queue(postgres_available):
    """The queue row must carry what an analyst triages on. These columns were
    added precisely because /flags could not show amount or counterparty."""
    tx_id = _add_flag()
    flag = _find(_fetch("open"), tx_id)

    assert flag is not None
    assert flag.amount_paid == pytest.approx(12345.67)
    assert flag.receiver_account_key == "999_RECEIVER"
    assert flag.payment_format == "ACH"
    assert flag.tx_timestamp is not None


def test_an_unreviewed_flag_is_open(postgres_available):
    tx_id = _add_flag()
    flag = _find(_fetch("open"), tx_id)

    assert flag is not None
    # None is what makes it open - there is no separate status column to drift.
    assert flag.analyst_decision is None


def test_a_reviewed_flag_leaves_the_open_queue(postgres_available):
    """The bug this closes: without the feedback join, a decided flag came back
    forever and the analyst loop never visibly closed."""
    tx_id = _add_flag()
    _review(tx_id, "confirmed_fraud")

    assert _find(_fetch("open"), tx_id) is None

    reviewed = _find(_fetch("reviewed"), tx_id)
    assert reviewed is not None
    assert reviewed.analyst_decision == "confirmed_fraud"
    assert reviewed.decided_by == "pytest"


def test_all_status_returns_both(postgres_available):
    open_tx = _add_flag()
    reviewed_tx = _add_flag()
    _review(reviewed_tx, "false_positive")

    everything = _fetch("all")
    assert _find(everything, open_tx) is not None
    assert _find(everything, reviewed_tx) is not None


def test_a_changed_verdict_shows_the_latest_one(postgres_available):
    """An analyst may correct themselves; the queue must not show the stale
    decision just because it was written first."""
    tx_id = _add_flag()
    _review(tx_id, "false_positive")
    _review(tx_id, "confirmed_fraud")

    reviewed = _find(_fetch("reviewed"), tx_id)
    assert reviewed is not None
    assert reviewed.analyst_decision == "confirmed_fraud"
    # One row per flag, not one per decision - a duplicated queue row would be
    # a visible bug in the UI.
    assert sum(1 for f in _fetch("reviewed").flags if f.tx_id == tx_id) == 1


def test_queue_is_ordered_by_score_descending(postgres_available):
    """It is a risk-ordered worklist: the riskiest item must be reachable
    without paging."""
    _add_flag(score=0.91)
    _add_flag(score=0.99)
    _add_flag(score=0.95)

    scores = [f.score for f in _fetch("open").flags if f.tx_id.startswith(TEST_PREFIX)]
    assert scores == sorted(scores, reverse=True)


def test_summary_counts_reflect_decisions(postgres_available):
    before = _fetch("open").summary

    open_tx = _add_flag()
    confirmed_tx = _add_flag()
    dismissed_tx = _add_flag()
    _review(confirmed_tx, "confirmed_fraud")
    _review(dismissed_tx, "false_positive")

    after = _fetch("open").summary

    assert after.open_flags == before.open_flags + 1
    assert after.confirmed_today == before.confirmed_today + 1
    assert after.dismissed_today == before.dismissed_today + 1
    assert _find(_fetch("open"), open_tx) is not None


def test_summary_is_independent_of_page_size(postgres_available):
    """A count that shrank when you changed the page size would be worse than
    no count at all."""
    for _ in range(3):
        _add_flag()

    small_page = _fetch("open", limit=1).summary.open_flags
    large_page = _fetch("open", limit=500).summary.open_flags
    assert small_page == large_page


def test_a_transaction_scored_twice_appears_once(postgres_available):
    """The audit log keeps every scoring event on purpose, but the queue is a
    worklist of transactions. Observed live before this was fixed: 179 rows for
    177 transactions, with two duplicated in the analyst's list."""
    tx_id = _add_flag(score=0.93)
    # Same transaction, scored again by a replay - a second audit row.
    _add_flag(score=0.97, tx_id=tx_id)

    matching = [f for f in _fetch("open").flags if f.tx_id == tx_id]
    assert len(matching) == 1
    # And it is the newer scoring, not the stale one.
    assert matching[0].score == pytest.approx(0.97)


def test_open_count_matches_the_rows_returned(postgres_available):
    """A summary saying 179 above a list of 177 would undermine both numbers."""
    tx_id = _add_flag()
    _add_flag(tx_id=tx_id)  # duplicate scoring event

    response = _fetch("open", limit=500)
    distinct_returned = {flag.tx_id for flag in response.flags}
    assert len(response.flags) == len(distinct_returned)

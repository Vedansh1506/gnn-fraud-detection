"""The additive-migration helper in `src.db.session`.

It exists because `create_all` leaves an already-provisioned database behind
when a model gains a column - the app then starts cleanly and fails on the
first query that mentions it. These tests pin both what it does and, more
importantly, what it refuses to do.

Requires Postgres (`docker compose up -d postgres`); skips otherwise, matching
the other integration tests.
"""

from __future__ import annotations

import pytest
from sqlalchemy import Column, Float, Integer, MetaData, String, Table, inspect, text

from src.db.session import add_missing_columns, get_engine


@pytest.fixture
def engine():
    engine = get_engine()
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("Postgres not reachable - run `docker compose up -d postgres`.")
    return engine


@pytest.fixture
def scratch_table(engine):
    """A throwaway table, created narrow so a column can be 'added' to it."""
    name = "pytest_migration_scratch"
    with engine.begin() as connection:
        connection.execute(text(f"DROP TABLE IF EXISTS {name}"))
        connection.execute(text(f"CREATE TABLE {name} (id INTEGER PRIMARY KEY)"))
    yield name
    with engine.begin() as connection:
        connection.execute(text(f"DROP TABLE IF EXISTS {name}"))


def _metadata_with(name: str, *columns: Column) -> MetaData:
    metadata = MetaData()
    Table(name, metadata, Column("id", Integer, primary_key=True), *columns)
    return metadata


def _columns(engine, name: str) -> set[str]:
    return {column["name"] for column in inspect(engine).get_columns(name)}


def test_a_new_nullable_column_is_added(engine, scratch_table, monkeypatch):
    monkeypatch.setattr(
        "src.db.session.Base",
        type("FakeBase", (), {"metadata": _metadata_with(scratch_table, Column("amount", Float))}),
    )
    added = add_missing_columns(engine)

    assert f"{scratch_table}.amount" in added
    assert "amount" in _columns(engine, scratch_table)


def test_running_it_twice_adds_nothing_the_second_time(engine, scratch_table, monkeypatch):
    monkeypatch.setattr(
        "src.db.session.Base",
        type("FakeBase", (), {"metadata": _metadata_with(scratch_table, Column("amount", Float))}),
    )
    add_missing_columns(engine)
    assert add_missing_columns(engine) == []


def test_a_non_nullable_column_is_refused_not_guessed(engine, scratch_table, monkeypatch):
    """Adding NOT NULL to a populated table needs a default or a backfill -
    that is a decision, not something to infer at startup."""
    monkeypatch.setattr(
        "src.db.session.Base",
        type(
            "FakeBase",
            (),
            {"metadata": _metadata_with(scratch_table, Column("owner", String, nullable=False))},
        ),
    )
    added = add_missing_columns(engine)

    assert added == []
    assert "owner" not in _columns(engine, scratch_table)


def test_existing_columns_are_left_alone(engine, scratch_table, monkeypatch):
    """It must never retype or rewrite a column that is already there."""
    with engine.begin() as connection:
        connection.execute(text(f"ALTER TABLE {scratch_table} ADD COLUMN label VARCHAR"))
        connection.execute(text(f"INSERT INTO {scratch_table} (id, label) VALUES (1, 'keep me')"))

    monkeypatch.setattr(
        "src.db.session.Base",
        type("FakeBase", (), {"metadata": _metadata_with(scratch_table, Column("label", String))}),
    )
    assert add_missing_columns(engine) == []

    with engine.connect() as connection:
        value = connection.execute(text(f"SELECT label FROM {scratch_table} WHERE id = 1")).scalar()
    assert value == "keep me"


def test_the_real_audit_log_has_the_queue_columns(engine):
    """The columns the analyst queue reads must actually exist after startup -
    this is the failure the helper was written for."""
    from src.db.session import create_tables

    create_tables()
    columns = _columns(engine, "audit_log")
    assert {
        "receiver_account_key",
        "amount_paid",
        "payment_currency",
        "payment_format",
        "tx_timestamp",
    } <= columns

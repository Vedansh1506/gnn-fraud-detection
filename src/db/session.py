"""Postgres engine/session wiring."""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from src.common.config import get_settings
from src.db.models import Base

logger = logging.getLogger("db")


# Seconds to wait for a Postgres connection before giving up. Without a bound,
# /health hangs instead of answering when the database is down - measured: the
# request never returned. That is the worst possible failure for the one
# endpoint the dashboard polls to decide whether to show its degraded banner,
# because a hung health check is indistinguishable from a hung app. Short
# enough that a degraded answer arrives promptly, long enough for a loaded box.
CONNECT_TIMEOUT_SECONDS = 3


def build_engine(dsn: str) -> Engine:
    """One construction path for every engine, so the connection settings that
    matter can't be right in the app and absent wherever else one is made."""
    # pool_pre_ping: the demo box sits idle between runs and Postgres drops
    # stale connections; without it the first score after a pause fails.
    return create_engine(
        dsn,
        pool_pre_ping=True,
        connect_args={"connect_timeout": CONNECT_TIMEOUT_SECONDS},
    )


@lru_cache
def get_engine() -> Engine:
    return build_engine(get_settings().postgres_dsn)


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def create_tables() -> None:
    """Idempotent - safe on every startup.

    `create_all` creates missing *tables* but never alters an existing one, so
    adding a column to a model leaves an already-provisioned database silently
    behind: the app starts fine and then fails on the first query mentioning
    the new column. `add_missing_columns` closes that gap for the additive
    changes this project actually makes.
    """
    engine = get_engine()
    Base.metadata.create_all(engine)
    add_missing_columns(engine)


def add_missing_columns(engine: Engine) -> list[str]:
    """Add nullable columns that exist on the models but not yet in the database.

    A deliberately small stand-in for a migration tool, and bounded so it can
    only ever do the safe thing: it adds nullable columns and nothing else - it
    never drops, renames, retypes, or backfills. Anything beyond that (a NOT
    NULL column, a rename, a data migration) is not expressible here and is the
    point at which this project should adopt Alembic rather than grow this
    function.

    Returns what it added, so startup can say so out loud.
    """
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    added: list[str] = []

    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue  # create_all just made it, with every column.
        present = {column["name"] for column in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in present:
                continue
            if not column.nullable:
                # Refuse rather than guess: adding NOT NULL to a populated
                # table needs a default or a backfill, which is a decision, not
                # a detail to infer at startup.
                logger.error(
                    "cannot add non-nullable column %s.%s automatically - needs a migration",
                    table.name,
                    column.name,
                )
                continue
            # Identifiers come from our own model metadata, never from user
            # input, so interpolating them here is not an injection path.
            statement = (
                f'ALTER TABLE "{table.name}" '
                f'ADD COLUMN IF NOT EXISTS "{column.name}" '
                f"{column.type.compile(engine.dialect)}"
            )
            with engine.begin() as connection:
                connection.execute(text(statement))
            added.append(f"{table.name}.{column.name}")

    if added:
        logger.info("added missing columns: %s", ", ".join(added))
    return added


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

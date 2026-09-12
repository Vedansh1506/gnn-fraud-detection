"""Postgres engine/session wiring."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.common.config import get_settings
from src.db.models import Base


@lru_cache
def get_engine() -> Engine:
    # pool_pre_ping: the demo box sits idle between runs and Postgres drops
    # stale connections; without it the first score after a pause fails.
    return create_engine(get_settings().postgres_dsn, pool_pre_ping=True)


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def create_tables() -> None:
    """Idempotent - safe on every startup. Alembic would be the answer once
    the schema starts changing; for a fixed demo schema it is overkill."""
    Base.metadata.create_all(get_engine())


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

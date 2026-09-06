"""Neo4j driver access - one driver per process, short-lived sessions per unit of work."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from functools import lru_cache

from neo4j import Driver, GraphDatabase, Session

from src.common.config import get_settings


@lru_cache
def get_driver() -> Driver:
    settings = get_settings()
    auth = (settings.neo4j_user, settings.neo4j_password)
    return GraphDatabase.driver(settings.neo4j_uri, auth=auth)


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    driver = get_driver()
    session = driver.session()
    try:
        yield session
    finally:
        session.close()

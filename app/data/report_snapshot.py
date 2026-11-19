"""One MVCC snapshot shared by all database reads for a generated report."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, Iterator

from sqlalchemy import text

from app.data.database import engine

_REPORT_SNAPSHOT_CONNECTION: ContextVar[Any | None] = ContextVar(
    "report_snapshot_connection", default=None,
)


@contextmanager
def report_read_snapshot() -> Iterator[Any]:
    """Share one PostgreSQL REPEATABLE READ transaction across report queries."""
    active = _REPORT_SNAPSHOT_CONNECTION.get()
    if active is not None:
        yield active
        return

    with engine.connect() as connection:
        with connection.begin():
            connection.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ"))
            token = _REPORT_SNAPSHOT_CONNECTION.set(connection)
            try:
                yield connection
            finally:
                _REPORT_SNAPSHOT_CONNECTION.reset(token)


def current_report_snapshot_connection() -> Any | None:
    """Return the active report transaction connection, if any."""
    return _REPORT_SNAPSHOT_CONNECTION.get()


@contextmanager
def repository_connection(connection_engine=None) -> Iterator[Any]:
    """Reuse a report's consistent-read connection, otherwise lease a new one."""
    active = current_report_snapshot_connection()
    if active is not None:
        yield active
        return
    with (connection_engine or engine).connect() as connection:
        yield connection


def consistent_report_reads(function):
    """Run a production report build against one database MVCC snapshot."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        # Injected fetchers are deterministic test/preview adapters and do not
        # read the repository database.
        if kwargs.get("fetcher") is not None or current_report_snapshot_connection() is not None:
            return function(*args, **kwargs)
        with report_read_snapshot():
            return function(*args, **kwargs)

    return wrapped

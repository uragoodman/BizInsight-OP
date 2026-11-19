"""Stable test-process defaults.

The application loads ``.env`` for local development.  A developer can enable
real account authentication there, but the legacy unit/integration suite is an
anonymous compatibility suite unless a test explicitly opts into authentication.
Set the process value before any test imports :mod:`app` so a local ``.env``
cannot make unrelated tests fail with 401 responses.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import dotenv_values
from sqlalchemy.engine import make_url


# ``Settings`` gives real environment variables precedence over .env values.
# Individual authentication tests can still override this with monkeypatch and
# clear ``get_settings``'s cache before importing/calling application code.
os.environ["TENANT_AUTH_REQUIRED"] = "false"
# Unit tests use the checked-in bootstrap policy unless a test explicitly
# mocks/publishes database ABAC rules. Do not make unrelated graph and
# authorization-contract tests wait on a developer's local PostgreSQL service.
os.environ["ABAC_DATABASE_ENABLED"] = "false"


def pytest_configure(config: pytest.Config) -> None:
    """Fail closed unless the shard runner supplied a disposable database URL.

    Settings loads ``.env`` at import time, so this check must run before test
collection imports application modules. A raw ``pytest`` invocation is
deliberately rejected: use ``scripts/run_test_shard.py`` to set a unique safe
URL first.
    """
    if os.environ.get("BIZINSIGHT_TEST_RUNNER") != "1":
        raise pytest.UsageError(
            "Run tests through scripts/run_test_shard.py; raw pytest may resolve .env"
        )

    database_url = os.environ.get("POSTGRES_URL", "").strip()
    if not database_url:
        raise pytest.UsageError("test shard runner must set POSTGRES_URL explicitly")
    try:
        requested_database = make_url(database_url).database
    except Exception as exc:  # noqa: BLE001 - present a safe, actionable error
        raise pytest.UsageError("test shard runner supplied an invalid POSTGRES_URL") from exc

    env_file_url = dotenv_values(Path(__file__).resolve().parents[1] / ".env").get("POSTGRES_URL")
    env_file_database = make_url(str(env_file_url)).database if env_file_url else None
    if not requested_database or requested_database.lower() == "bizinsight":
        raise pytest.UsageError("tests may not connect to the developer bizinsight database")
    if env_file_database and requested_database.lower() == env_file_database.lower():
        raise pytest.UsageError("test POSTGRES_URL must differ from the database named in .env")

    shard = os.environ.get("BIZINSIGHT_TEST_SHARD", "")
    if shard == "no-db":
        if not requested_database.startswith("codex_test_no_db_"):
            raise pytest.UsageError("no-db shard must use the non-routable codex_test_no_db_* URL")
    elif shard == "postgres":
        if not requested_database.startswith("codex_test_"):
            raise pytest.UsageError("PostgreSQL shard must use a unique codex_test_* database")
    else:
        raise pytest.UsageError("BIZINSIGHT_TEST_SHARD must be 'no-db' or 'postgres'")

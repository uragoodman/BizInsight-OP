"""Opt-in WP6 rollback smoke on an isolated PostgreSQL clone.

Runs the durable chat chain with ``MEM0_ENABLED=false`` and a deterministic
graph edge, verifying that Turn Commit, SSE completion and task continuity do
not depend on Mem0 personalization.  Skipped unless an isolated PostgreSQL
admin URL and template are configured.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not os.getenv("ISOLATED_TEST_POSTGRES_ADMIN_URL"),
    reason="Set ISOLATED_TEST_POSTGRES_ADMIN_URL to run the WP6 rollback smoke",
)


def test_durable_chain_survives_mem0_disabled_on_isolated_clone() -> None:
    admin_url = make_url(os.environ["ISOLATED_TEST_POSTGRES_ADMIN_URL"])
    admin = create_engine(admin_url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    template = os.getenv("WP2_TEST_POSTGRES_TEMPLATE", "bizinsight")
    database = f"wp6_rollback_{uuid.uuid4().hex[:10]}"
    clone_url = admin_url.set(database=database).render_as_string(hide_password=False)
    try:
        with admin.connect() as connection:
            if not connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname=:name"), {"name": template},
            ).scalar():
                pytest.skip(f"template database {template!r} does not exist")
            connection.execute(text(f'CREATE DATABASE "{database}" TEMPLATE "{template}"'))

        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/verify_wp6_mem0_disabled_chain.py")],
            cwd=ROOT,
            env={
                **os.environ,
                "POSTGRES_URL": clone_url,
                "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                "MEM0_ENABLED": "false",
                "TENANT_AUTH_REQUIRED": "false",
                "RELATION_RESOLVER_LLM_ENABLED": "false",
            },
            capture_output=True,
            text=True,
            timeout=600,
        )
        if completed.returncode == 2 and "WP6_MEM0_DISABLED_SKIPPED" in completed.stdout:
            pytest.skip(completed.stdout.strip())
        assert completed.returncode == 0, (
            "WP6 rollback smoke failed\n"
            f"stdout:\n{completed.stdout[-6000:]}\n"
            f"stderr:\n{completed.stderr[-6000:]}"
        )
        assert "WP6_MEM0_DISABLED_OK" in completed.stdout
        assert '"commits": [1, 2, 3]' in completed.stdout.replace("\n", " ")
        assert '"tasks": 1' in completed.stdout.replace("\n", " ")
    finally:
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)'))
        admin.dispose()

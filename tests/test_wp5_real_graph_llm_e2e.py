"""Opt-in real-graph + real-LLM smoke for the WP5 durable chat chain.

Skipped unless ``WP5_REAL_LLM_E2E=1`` (or ``true``), an isolated PostgreSQL
admin URL and template are provided, and an LLM API key is configured.  The
smoke clones the template, runs ``scripts/verify_wp5_real_graph_llm.py`` in a
subprocess and drops the clone.  It never touches the configured application
database and is intentionally not part of the default PR gate.
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


def _enabled() -> bool:
    return os.getenv("WP5_REAL_LLM_E2E", "").strip().lower() in {"1", "true", "yes"}


def _has_llm_key() -> bool:
    if (os.getenv("DEEPSEEK_API_KEY") or "").strip():
        return True
    try:
        from dotenv import dotenv_values
    except Exception:  # noqa: BLE001 - optional local convenience
        return False
    return bool((dotenv_values(ROOT / ".env").get("DEEPSEEK_API_KEY") or "").strip())


pytestmark = pytest.mark.skipif(
    not _enabled()
    or not os.getenv("ISOLATED_TEST_POSTGRES_ADMIN_URL")
    or not _has_llm_key(),
    reason=(
        "Set WP5_REAL_LLM_E2E=1, ISOLATED_TEST_POSTGRES_ADMIN_URL and DEEPSEEK_API_KEY "
        "to run the real-graph/real-LLM WP5 smoke"
    ),
)


def test_real_graph_llm_smoke_on_isolated_clone() -> None:
    admin_url = make_url(os.environ["ISOLATED_TEST_POSTGRES_ADMIN_URL"])
    admin = create_engine(admin_url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    template = os.getenv("WP2_TEST_POSTGRES_TEMPLATE", "bizinsight")
    database = f"wp5_real_llm_{uuid.uuid4().hex[:10]}"
    clone_url = admin_url.set(database=database).render_as_string(hide_password=False)
    try:
        with admin.connect() as connection:
            if not connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname=:name"), {"name": template},
            ).scalar():
                pytest.skip(f"template database {template!r} does not exist")
            connection.execute(text(f'CREATE DATABASE "{database}" TEMPLATE "{template}"'))

        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/verify_wp5_real_graph_llm.py")],
            cwd=ROOT,
            env={
                **os.environ,
                "POSTGRES_URL": clone_url,
                "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                "TENANT_AUTH_REQUIRED": "false",
                "RELATION_RESOLVER_LLM_ENABLED": "false",
            },
            capture_output=True,
            text=True,
            timeout=600,
        )
        if completed.returncode == 2 and "WP5_REAL_GRAPH_LLM_SKIPPED" in completed.stdout:
            pytest.skip(completed.stdout.strip())
        assert completed.returncode == 0, (
            "real-graph/real-LLM WP5 smoke failed\n"
            f"stdout:\n{completed.stdout[-6000:]}\n"
            f"stderr:\n{completed.stderr[-6000:]}"
        )
        assert "WP5_REAL_GRAPH_LLM_OK" in completed.stdout
        assert '"commits": [1, 2, 3]' in completed.stdout.replace("\n", " ")
    finally:
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)'))
        admin.dispose()

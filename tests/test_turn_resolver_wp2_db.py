"""Explicit isolated-PostgreSQL verification for the WP2/WP1 connection.

This test is skipped unless both the maintenance URL and a template database
are explicitly configured.  It never falls back to unit mocks, and it does not
touch the configured application database: the script clones the template into
a temporary database and drops it after the assertions.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not os.getenv("ISOLATED_TEST_POSTGRES_ADMIN_URL")
    or not os.getenv("WP2_TEST_POSTGRES_TEMPLATE"),
    reason=(
        "Set ISOLATED_TEST_POSTGRES_ADMIN_URL and WP2_TEST_POSTGRES_TEMPLATE "
        "to run the real PostgreSQL WP2 chain"
    ),
)
def test_wp2_real_chain_against_isolated_postgres() -> None:
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify_wp2_real_chain.py")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode == 2 and "WP2_REAL_CHAIN_SKIPPED" in completed.stdout:
        pytest.skip(completed.stdout.strip())
    assert completed.returncode == 0, (
        f"WP2 real-chain verification failed\n"
        f"stdout:\n{completed.stdout[-6000:]}\n"
        f"stderr:\n{completed.stderr[-6000:]}"
    )
    assert "WP2_REAL_CHAIN_OK" in completed.stdout
    assert '"relation_multiple": "clarify"' in completed.stdout
    assert '"relation_selected": "continue"' in completed.stdout
    assert '"source": "task_state"' in completed.stdout
    assert '"bundle_turn_id": "wp2-real-turn-' in completed.stdout
    assert '"bundle_client_turn_id": "client-' in completed.stdout

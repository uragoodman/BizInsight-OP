"""Opt-in real PostgreSQL migration and lifecycle test for WP5."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not os.getenv("ISOLATED_TEST_POSTGRES_ADMIN_URL"),
    reason="Set ISOLATED_TEST_POSTGRES_ADMIN_URL to run WP5 on an isolated PostgreSQL clone",
)
def test_wp5_turn_commit_on_isolated_postgres() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify_wp5_turn_commit.py")],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")},
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode == 2 and "WP5_TURN_COMMIT_SKIPPED" in completed.stdout:
        pytest.skip(completed.stdout.strip())
    assert completed.returncode == 0, (
        f"WP5 isolated PostgreSQL verification failed\n"
        f"stdout:\n{completed.stdout[-6000:]}\n"
        f"stderr:\n{completed.stderr[-6000:]}"
    )
    assert "WP5_TURN_COMMIT_OK" in completed.stdout

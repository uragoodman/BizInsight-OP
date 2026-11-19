"""Opt-in migration test for the WP5 summary backfill budget.

The test clones an isolated template database, downgrades the clone to the
revision immediately before the WP5 migration, inserts terminal turns that
predate it, upgrades back to head and verifies that:

* every terminal turn received a ``commit_version``;
* the backfilled summary covers a newest *contiguous* suffix of the turns
  older than the recent window;
* the stored coverage ids/versions match the rendered text exactly;
* the stored summary stays inside the frozen 4k-token summary budget.

It never touches the configured application database.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


ROOT = Path(__file__).resolve().parents[1]
PRE_WP5_REVISION = "20261101_60"
TOTAL_TURNS = 30
RECENT_WINDOW = 20


pytestmark = pytest.mark.skipif(
    not os.getenv("ISOLATED_TEST_POSTGRES_ADMIN_URL"),
    reason="Set ISOLATED_TEST_POSTGRES_ADMIN_URL to run the WP5 migration backfill test",
)


def _alembic(url: str, action: str, revision: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "alembic", action, revision],
        cwd=ROOT,
        env={**os.environ, "POSTGRES_URL": url, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, f"alembic {action} {revision} failed:\n{completed.stderr[-4000:]}"


def test_migration_backfill_keeps_summary_within_budget_and_coverage() -> None:
    from app.context.packer import estimate_tokens

    admin_url = make_url(os.environ["ISOLATED_TEST_POSTGRES_ADMIN_URL"])
    admin = create_engine(admin_url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    template = os.getenv("WP2_TEST_POSTGRES_TEMPLATE", "bizinsight")
    database = f"wp5_backfill_{uuid.uuid4().hex[:10]}"
    clone_url = admin_url.set(database=database).render_as_string(hide_password=False)
    engine = None
    try:
        with admin.connect() as connection:
            if not connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname=:name"), {"name": template},
            ).scalar():
                pytest.skip(f"template database {template!r} does not exist")
            connection.execute(text(f'CREATE DATABASE "{database}" TEMPLATE "{template}"'))

        _alembic(clone_url, "downgrade", PRE_WP5_REVISION)

        suffix = uuid.uuid4().hex[:10]
        tenant, user = f"wp5-backfill-{suffix}", f"wp5-backfill-user-{suffix}"
        session = f"wp5-backfill-session-{suffix}"
        engine = create_engine(clone_url)
        with engine.begin() as connection:
            connection.execute(text("""
                INSERT INTO iam.tenants(tenant_id, tenant_name, scope_type, status)
                VALUES (:tenant, 'wp5 backfill', 'platform', 'active')
            """), {"tenant": tenant})
            connection.execute(text("""
                INSERT INTO iam.users(user_id, display_name, email, status)
                VALUES (:user, 'wp5 backfill', :email, 'active')
            """), {"user": user, "email": f"{user}@example.invalid"})
            connection.execute(text("""
                INSERT INTO governance.chat_sessions(tenant_id, user_id, chat_session_id, title)
                VALUES (:tenant, :user, :session, 'wp5 backfill')
            """), {"tenant": tenant, "user": user, "session": session})
            base = datetime(2026, 1, 1)
            for index in range(1, TOTAL_TURNS + 1):
                connection.execute(text("""
                    INSERT INTO governance.chat_turns
                        (turn_id, tenant_id, user_id, chat_session_id, question,
                         answer, task_type, trace_id, status, created_at)
                    VALUES (:turn, :tenant, :user, :session, :question, :answer,
                            'metric_query', :trace, 'completed', :created_at)
                """), {
                    "turn": f"turn-{index:02d}", "tenant": tenant, "user": user,
                    "session": session,
                    "question": f"问题{index} " + "很长的中文内容" * 45,
                    "answer": f"回答{index} " + "详细解释" * 160,
                    "trace": f"trace-{index}",
                    "created_at": base + timedelta(minutes=index),
                })
        engine.dispose()
        engine = None

        _alembic(clone_url, "upgrade", "head")

        engine = create_engine(clone_url)
        with engine.connect() as connection:
            versioned = connection.execute(text("""
                SELECT count(*) FROM governance.chat_turns
                 WHERE tenant_id=:tenant AND user_id=:user AND commit_version IS NOT NULL
            """), {"tenant": tenant, "user": user}).scalar_one()
            assert versioned == TOTAL_TURNS

            row = connection.execute(text("""
                SELECT conversation_summary, summary_covered_from_version,
                       summary_covered_to_version, summary_covered_from_turn_id,
                       summary_covered_to_turn_id
                  FROM governance.chat_sessions
                 WHERE tenant_id=:tenant AND user_id=:user AND chat_session_id=:session
            """), {"tenant": tenant, "user": user, "session": session}).mappings().one()

        summary = str(row["conversation_summary"] or "")
        covered_from = int(row["summary_covered_from_version"])
        covered_to = int(row["summary_covered_to_version"])
        # The recent raw window owns the newest 20 committed turns.
        assert covered_to <= TOTAL_TURNS - RECENT_WINDOW
        assert 1 <= covered_from <= covered_to
        # Every eligible line is ~960 characters, so the summary budget must
        # have forced a suffix; a backfill that concatenated all eligible
        # turns would fail here.
        assert covered_from > 1
        assert row["summary_covered_from_turn_id"] == f"turn-{covered_from:02d}"
        assert row["summary_covered_to_turn_id"] == f"turn-{covered_to:02d}"

        lines = summary.split("\n") if summary else []
        assert len(lines) == covered_to - covered_from + 1
        for version in range(covered_from, covered_to + 1):
            assert f"问题{version} " in summary
        for version in range(1, covered_from):
            assert f"问题{version} " not in summary
        for version in range(covered_to + 1, TOTAL_TURNS + 1):
            assert f"问题{version} " not in summary

        # The stored text must stay inside the frozen summary budget even
        # before the next Turn Commit rebuilds it with the real tokenizer.
        assert estimate_tokens(summary) <= 4000
    finally:
        if engine is not None:
            engine.dispose()
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)'))
        admin.dispose()

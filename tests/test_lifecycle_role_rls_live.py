"""Opt-in live check of the lifecycle worker's least-privilege PostgreSQL role.

Run with RUN_LIFECYCLE_RLS_INTEGRATION=1 after applying migrations and
scripts.setup_rls_role. Every synthetic row is removed in a finally block;
the worker-side update/audit probe is rolled back.
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import make_url

from app.core.config import get_settings
from app.data.database import _rls_role_statement


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIFECYCLE_RLS_INTEGRATION") != "1",
    reason="set RUN_LIFECYCLE_RLS_INTEGRATION=1 for the live lifecycle-role check",
)


def test_lifecycle_role_is_narrow_and_enforces_tenant_user_rls() -> None:
    settings = get_settings()
    admin_engine = create_engine(
        settings.postgres_url,
        future=True,
        connect_args={"connect_timeout": int(settings.database_timeout_seconds)},
    )
    worker_url = make_url(settings.postgres_url).set(
        username="bizinsight_lifecycle_login",
        password=settings.lifecycle_reconcile_db_password,
    )
    worker_engine = create_engine(
        worker_url,
        future=True,
        connect_args={"connect_timeout": int(settings.database_timeout_seconds)},
    )

    @event.listens_for(worker_engine, "checkout")
    def _assume_lifecycle_role(dbapi_connection, _record, _proxy) -> None:  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute(_rls_role_statement("bizinsight_lifecycle_app"))
        finally:
            cursor.close()

    tag = uuid4().hex
    data = {
        "tenant_a": f"lifecycle_rls_a_{tag}",
        "tenant_b": f"lifecycle_rls_b_{tag}",
        "user_a": f"lifecycle_user_a_{tag}",
        "user_b": f"lifecycle_user_b_{tag}",
        "session_a": f"lifecycle_session_a_{tag}",
        "session_b": f"lifecycle_session_b_{tag}",
        "task_a": f"lifecycle_task_a_{tag}",
        "task_b": f"lifecycle_task_b_{tag}",
    }
    try:
        with admin_engine.begin() as conn:
            for suffix in ("a", "b"):
                conn.execute(text("""
                    INSERT INTO governance.chat_sessions
                      (tenant_id,user_id,chat_session_id,title)
                    VALUES (:tenant,:user,:session,'temporary RLS verification')
                """), {
                    "tenant": data[f"tenant_{suffix}"],
                    "user": data[f"user_{suffix}"],
                    "session": data[f"session_{suffix}"],
                })
                conn.execute(text("""
                    INSERT INTO governance.conversation_tasks
                      (task_id,tenant_id,user_id,chat_session_id,title,objective,status)
                    VALUES (:task,:tenant,:user,:session,
                            'temporary RLS verification','rollback-only fixture','interrupted')
                """), {
                    "task": data[f"task_{suffix}"],
                    "tenant": data[f"tenant_{suffix}"],
                    "user": data[f"user_{suffix}"],
                    "session": data[f"session_{suffix}"],
                })

        with worker_engine.begin() as conn:
            role = conn.execute(text("""
                SELECT session_user,current_user,
                       (SELECT rolsuper FROM pg_roles WHERE rolname=current_user),
                       (SELECT rolbypassrls FROM pg_roles WHERE rolname=current_user)
            """)).one()
            assert role == (
                "bizinsight_lifecycle_login", "bizinsight_lifecycle_app", False, False,
            )
            assert not conn.execute(text(
                "SELECT has_schema_privilege(current_user,'analytics','USAGE')"
            )).scalar_one()
            assert conn.execute(text("""
                SELECT has_column_privilege(current_user,'iam.users','user_id','SELECT'),
                       has_column_privilege(current_user,'iam.users','password_hash','SELECT'),
                       has_column_privilege(current_user,'governance.conversation_tasks','summary_data','SELECT'),
                       has_column_privilege(current_user,'governance.conversation_tasks','objective','SELECT'),
                       has_column_privilege(current_user,'governance.conversation_tasks','summary_data','UPDATE'),
                       has_column_privilege(current_user,'governance.lifecycle_audit_events','event_name','INSERT')
            """)).one() == (True, False, True, False, True, True)

            def set_context(tenant_id: str, user_id: str) -> None:
                conn.execute(text("""
                    SELECT set_config('app.tenant_id',:tenant,true),
                           set_config('app.user_id',:user,true),
                           set_config('app.scope_type','seller',true)
                """), {"tenant": tenant_id, "user": user_id})

            query = text("""
                SELECT task_id FROM governance.conversation_tasks
                WHERE task_id IN (:task_a,:task_b) ORDER BY task_id
            """)
            set_context(data["tenant_a"], data["user_a"])
            assert conn.execute(query, {
                "task_a": data["task_a"], "task_b": data["task_b"],
            }).scalars().all() == [data["task_a"]]
            conn.execute(text("""
                UPDATE governance.conversation_tasks
                SET summary_data=summary_data WHERE task_id=:task
            """), {"task": data["task_a"]})
            conn.execute(text("""
                INSERT INTO governance.lifecycle_audit_events
                  (event_name,tenant_id,user_id,payload)
                VALUES ('chat.lifecycle.rls_probe',:tenant,:user,'{}'::jsonb)
            """), {"tenant": data["tenant_a"], "user": data["user_a"]})

            set_context(data["tenant_b"], data["user_b"])
            assert conn.execute(query, {
                "task_a": data["task_a"], "task_b": data["task_b"],
            }).scalars().all() == [data["task_b"]]
            set_context("", "")
            assert conn.execute(query, {
                "task_a": data["task_a"], "task_b": data["task_b"],
            }).scalars().all() == []
        # worker_engine.begin() rolled back its update and audit probe.
    finally:
        with admin_engine.begin() as conn:
            conn.execute(text("""
                DELETE FROM governance.conversation_tasks
                WHERE task_id IN (:task_a,:task_b)
            """), {"task_a": data["task_a"], "task_b": data["task_b"]})
            conn.execute(text("""
                DELETE FROM governance.chat_sessions
                WHERE chat_session_id IN (:session_a,:session_b)
            """), {"session_a": data["session_a"], "session_b": data["session_b"]})
            remaining = conn.execute(text("""
                SELECT count(*) FROM governance.conversation_tasks
                WHERE task_id IN (:task_a,:task_b)
            """), {"task_a": data["task_a"], "task_b": data["task_b"]}).scalar_one()
            assert remaining == 0
        worker_engine.dispose()
        admin_engine.dispose()

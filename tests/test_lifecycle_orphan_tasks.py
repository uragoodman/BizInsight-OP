from __future__ import annotations

from uuid import uuid4

from sqlalchemy import text

from app.auth.context import build_authorization_context
from app.chat_lifecycle.integration import begin_chat_lifecycle
from app.chat_lifecycle.reconciliation import run_lifecycle_reconciliation
from app.data.database import engine
from app.data.rls import set_local_context
from app.tasks.service import create_task, get_task
from app.tenancy import TenantContext


def test_reconcile_stale_legacy_task_without_turn_or_trace() -> None:
    context = TenantContext(
        tenant_id="platform", user_id="system", role="admin", scope_type="platform",
    )
    authz = build_authorization_context(context, metric_scope_type="platform")
    session_id = f"orphan-reconcile-{uuid4().hex}"
    protected_session_id = f"orphan-reconcile-active-{uuid4().hex}"
    task = create_task(
        context, session_id, "stale legacy task reconciliation test", status="running",
    )
    task_id = task["task_id"]
    protected_task = create_task(
        context, protected_session_id, "task linked to live turn", status="running",
    )
    protected_binding = begin_chat_lifecycle(
        context,
        session_id=protected_session_id,
        client_turn_id=f"client-{uuid4().hex}",
        question="active task reconciliation exclusion test",
        explorer_payload={
            "seller_id": "1f50f920176a81dab994f9023523100",
            "metric_name": "gmv",
            "start_date": "2018-08-01",
            "end_date": "2018-08-03",
            "granularity": "day",
        },
        thread_id=f"thread-{uuid4().hex}",
        trace_id=str(uuid4()),
    )
    assert protected_binding is not None and protected_binding.execution is not None

    try:
        with engine.begin() as conn:
            set_local_context(conn, context)
            conn.execute(text("""
                UPDATE governance.conversation_tasks
                   SET updated_at=now() - interval '2 days'
                 WHERE task_id=:task AND tenant_id=:tenant AND user_id=:user
            """), {
                "task": task_id,
                "tenant": context.tenant_id,
                "user": context.user_id,
            })
            conn.execute(text("""
                UPDATE governance.chat_turns SET task_id=:task
                 WHERE turn_id=:turn AND tenant_id=:tenant AND user_id=:user
            """), {
                "task": protected_task["task_id"],
                "turn": protected_binding.turn.turn_id,
                "tenant": context.tenant_id,
                "user": context.user_id,
            })
            conn.execute(text("""
                UPDATE governance.conversation_tasks
                   SET updated_at=now() - interval '2 days'
                 WHERE task_id=:task AND tenant_id=:tenant AND user_id=:user
            """), {
                "task": protected_task["task_id"],
                "tenant": context.tenant_id,
                "user": context.user_id,
            })

        result = run_lifecycle_reconciliation(context, authz, stale_seconds=3600)
        assert result["orphan_tasks_interrupted"] == 1

        repaired = get_task(context, task_id)
        assert repaired is not None
        assert repaired["status"] == "interrupted"
        assert repaired["completed_at"] is not None
        audit = repaired["summary_data"]["lifecycle_reconciliation"]
        assert audit["reason"] == "stale_legacy_task_without_turn_or_trace"
        assert audit["stale_seconds"] == 86_400
        assert audit["reconciled_at"] is not None

        still_running = get_task(context, protected_task["task_id"])
        assert still_running is not None
        assert still_running["status"] == "running"
    finally:
        with engine.begin() as conn:
            set_local_context(conn, context)
            conn.execute(text("""
                DELETE FROM governance.chat_sessions
                 WHERE tenant_id=:tenant AND user_id=:user AND chat_session_id=:session
            """), {
                "tenant": context.tenant_id,
                "user": context.user_id,
                "session": session_id,
            })
            conn.execute(text("""
                DELETE FROM governance.chat_sessions
                 WHERE tenant_id=:tenant AND user_id=:user AND chat_session_id=:session
            """), {
                "tenant": context.tenant_id,
                "user": context.user_id,
                "session": protected_session_id,
            })

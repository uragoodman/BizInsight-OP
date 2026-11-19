# -*- coding: utf-8 -*-
"""WP6 rollback smoke: durable chat with MEM0_ENABLED=false.

Runs the real FastAPI routes, real PostgreSQL Turn Commit and the real context
compiler, with the graph edges replaced by deterministic results (no model
calls).  It proves that disabling Mem0 personalization does not break the
durable conversation chain and that no Mem0 preference adapter is touched.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import text


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPLORER = {
    "seller_id": "seller-1", "metric_name": "gmv",
    "start_date": "2018-08-01", "end_date": "2018-08-03", "granularity": "day",
}


def main() -> int:
    from app.core.config import get_settings
    from app.data.database import engine
    from app.main import app
    from app.api.auth import require_tenant_context
    from app.api.routes import chat as chat_routes
    from app.graph import workflow
    from app.memory import mem0_adapter
    from app.tenancy import TenantContext

    settings = get_settings()
    if bool(getattr(settings, "mem0_enabled", True)):
        print("WP6_MEM0_DISABLED_SKIPPED: MEM0_ENABLED is not false")
        return 2

    context = TenantContext(
        tenant_id="platform", user_id=f"wp6-rollback-{uuid.uuid4().hex[:8]}",
        role="admin", scope_type="platform",
    )
    with engine.begin() as connection:
        connection.execute(text("""
            INSERT INTO iam.tenants(tenant_id, tenant_name, scope_type, status)
            VALUES ('platform', 'wp6 rollback smoke', 'platform', 'active')
            ON CONFLICT (tenant_id) DO NOTHING
        """))
        connection.execute(text("""
            INSERT INTO iam.users(user_id, display_name, email, status)
            VALUES (:user, 'wp6 rollback smoke', :email, 'active')
            ON CONFLICT (user_id) DO NOTHING
        """), {"user": context.user_id, "email": f"{context.user_id}@example.invalid"})
        connection.execute(text("""
            INSERT INTO iam.tenant_memberships(tenant_id, user_id, role)
            VALUES ('platform', :user, 'admin') ON CONFLICT DO NOTHING
        """), {"user": context.user_id})

    compiled_preferences: list[dict] = []

    def fake_workflow(message, _thread_id=None, _tenant_context=None, _explorer=None,
                      _use_scope=False, _memory_session=None, _memory_enabled=True,
                      _metric_confirmation=None, _session=None, _turn=None,
                      task_id=None, _semantic_ids=None, _skipped_ids=None, **kwargs):
        boundary = kwargs.get("authorization_boundary") or {}
        compiled = workflow.compile_turn_context({
            "user_message": message,
            "current_query_scope_fingerprint": boundary.get("query_scope_fingerprint"),
            "data_version_manifest_hash": _expected_manifest,
            "task_progress_context": {},
            "node_history": [],
        })
        compiled_preferences.append(dict(compiled.get("preference_context") or {}))
        return {
            "task_type": "metric_query",
            "answer": f"deterministic answer for {message}",
            "trace_id": kwargs.get("trace_id") or str(uuid.uuid4()),
            "error": False, "errors": [], "clarification": None, "presentation": None,
        }

    def fake_stream(*args, **kwargs):
        trace = args[11] if len(args) > 11 else kwargs.get("trace_id") or "trace-stream"
        yield {"event": "started", "trace_id": trace, "task_type": "metric_query"}
        yield {
            "event": "completed", "task_type": "metric_query",
            "answer": "deterministic streamed answer", "trace_id": trace,
        }

    _expected_manifest = "wp6-manifest"
    original_manifest = chat_routes._expected_data_manifest_hash
    original_workflow = chat_routes.run_chat_workflow
    original_stream = chat_routes.stream_chat_workflow
    original_adapter_init = mem0_adapter.Mem0PreferenceAdapter.__init__

    def forbidden_adapter_init(self, *args, **kwargs):
        raise AssertionError("disabled Mem0 must not construct a preference adapter")

    chat_routes._expected_data_manifest_hash = lambda *_args, **_kwargs: _expected_manifest
    chat_routes.run_chat_workflow = fake_workflow
    chat_routes.stream_chat_workflow = fake_stream
    mem0_adapter.Mem0PreferenceAdapter.__init__ = forbidden_adapter_init
    app.dependency_overrides[require_tenant_context] = lambda: context
    client = TestClient(app)
    session_id = f"wp6-rollback-{uuid.uuid4().hex}"
    try:
        def durable(message: str) -> dict:
            return client.post("/chat", json={
                "message": message,
                "use_current_explorer_context": True,
                "explorer_context": EXPLORER,
                "chat_session_id": session_id,
                "chat_turn_id": f"turn-{uuid.uuid4().hex}",
            })

        first = durable("查询 2018-08-01 到 2018-08-03 的 GMV")
        assert first.status_code == 200, first.text
        body1 = first.json()
        assert body1.get("durable_turn_id") and body1.get("commit_version") == 1, body1
        assert compiled_preferences and all(
            not (entry.get("items") or []) for entry in compiled_preferences
        ), compiled_preferences

        second = durable("继续看一下 2018-08-02 的 GMV")
        body2 = second.json()
        assert second.status_code == 200, second.text
        assert body2.get("commit_version") == 2, body2

        turn3 = f"turn-{uuid.uuid4().hex}"
        completed = None
        row_at_completed = None
        with client.stream("POST", "/chat/stream", json={
            "message": "继续看一下 2018-08-03 的 GMV",
            "use_current_explorer_context": True,
            "explorer_context": EXPLORER,
            "chat_session_id": session_id,
            "chat_turn_id": turn3,
        }) as response:
            assert response.status_code == 200, response.read()
            for line in response.iter_lines():
                if not line or not str(line).startswith("data: "):
                    continue
                event = json.loads(str(line)[6:])
                if event.get("event") != "completed":
                    continue
                completed = event
                durable_turn = event.get("durable_turn_id")
                assert durable_turn, event
                with engine.connect() as connection:
                    row_at_completed = connection.execute(text("""
                        SELECT status, commit_version FROM governance.chat_turns
                         WHERE tenant_id=:tenant AND user_id=:user AND turn_id=:turn
                    """), {
                        "tenant": context.tenant_id, "user": context.user_id,
                        "turn": durable_turn,
                    }).mappings().one_or_none()
                break

        assert completed is not None, "stream never emitted completed"
        assert row_at_completed is not None, "completed was emitted before the durable commit"
        assert row_at_completed["status"] == "completed", row_at_completed
        assert int(row_at_completed["commit_version"]) == 3, row_at_completed
        assert all(not (entry.get("items") or []) for entry in compiled_preferences)

        with engine.connect() as connection:
            tasks = connection.execute(text("""
                SELECT count(DISTINCT task_id) FROM governance.chat_turns
                 WHERE tenant_id=:tenant AND user_id=:user AND chat_session_id=:session
                   AND task_id IS NOT NULL
            """), {
                "tenant": context.tenant_id, "user": context.user_id, "session": session_id,
            }).scalar_one()
        assert tasks == 1, f"durable rollback chain created {tasks} tasks"
        print("WP6_MEM0_DISABLED_OK " + json.dumps({
            "commits": [1, 2, int(row_at_completed["commit_version"])],
            "tasks": tasks,
            "preference_entries": len(compiled_preferences),
            "completed_event": sorted(completed.keys()),
        }, ensure_ascii=False))
        return 0
    finally:
        chat_routes._expected_data_manifest_hash = original_manifest
        chat_routes.run_chat_workflow = original_workflow
        chat_routes.stream_chat_workflow = original_stream
        mem0_adapter.Mem0PreferenceAdapter.__init__ = original_adapter_init
        app.dependency_overrides.pop(require_tenant_context, None)


if __name__ == "__main__":
    raise SystemExit(main())

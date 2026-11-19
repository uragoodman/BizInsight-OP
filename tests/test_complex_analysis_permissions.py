from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from app.api.auth import require_tenant_context
from app.api.routes import complex_analysis as routes
from app.main import app
from app.tenancy import SellerGrant, TenantContext


def test_complex_analysis_write_routes_require_analyze_permission(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(routes, "revise_plan", lambda *_args, **_kwargs: calls.append("revise") or {"plan_id": "p"})
    monkeypatch.setattr(routes, "cancel_plan", lambda *_args, **_kwargs: calls.append("cancel") or True)
    monkeypatch.setattr(routes, "execute_plan", lambda *_args, **_kwargs: calls.append("execute") or {
        "answer": "done", "trace_id": "trace", "task_type": "complex_analysis",
    })
    # This test checks role gates, not chat lifecycle persistence.
    monkeypatch.setattr(routes, "get_plan_v2", lambda *_args, **_kwargs: {
        "plan_id": "p", "status": "pending_confirmation",
    })
    monkeypatch.setattr(routes, "_resume_plan_lifecycle", lambda *_args, **_kwargs: (None, None))

    previous = app.dependency_overrides.get(require_tenant_context)
    app.dependency_overrides[require_tenant_context] = lambda: TenantContext(
        tenant_id="phase21-permission-tenant", user_id="phase21-viewer",
        role="viewer", scope_type="seller", seller_id="phase21-seller",
    )
    try:
        with TestClient(app) as client:
            create = client.post("/complex-analysis/plans", json={
                "question": "客户留存分析",
                "explorer_context": {
                    "seller_id": "phase21-seller",
                    "start_date": "2025-01-01",
                    "end_date": "2025-01-31",
                },
            })
            revise = client.post("/complex-analysis/plans/p/revise", json={"focus_key": "retention_curve"})
            cancel = client.post("/complex-analysis/plans/p/cancel")
            confirm = client.post("/complex-analysis/plans/p/confirm")
            assert [create.status_code, revise.status_code, cancel.status_code, confirm.status_code] == [403] * 4
            assert calls == []

            analyst = TenantContext(
                tenant_id="phase21-permission-tenant", user_id="phase21-analyst",
                role="analyst", scope_type="seller", seller_id="phase21-seller",
            )
            app.dependency_overrides[require_tenant_context] = lambda: analyst
            allowed = client.post("/complex-analysis/plans", json={
                "question": "客户留存分析",
                "explorer_context": {
                    "seller_id": "phase21-seller",
                    "start_date": "2025-01-01",
                    "end_date": "2025-01-31",
                },
            })
            revised = client.post("/complex-analysis/plans/p/revise", json={"focus_key": "retention_curve"})
            cancelled = client.post("/complex-analysis/plans/p/cancel")
            confirmed = client.post("/complex-analysis/plans/p/confirm")
            assert allowed.status_code == 409
            assert allowed.json()["detail"] == "sealed_route_decision_required"
            assert [revised.status_code, cancelled.status_code, confirmed.status_code] == [200] * 3
            assert '"event": "completed"' in confirmed.text
            assert calls == ["revise", "cancel", "execute"]
    finally:
        if previous is None:
            app.dependency_overrides.pop(require_tenant_context, None)
        else:
            app.dependency_overrides[require_tenant_context] = previous


def test_action_plan_mutations_require_analyze_permission(monkeypatch) -> None:
    previous = app.dependency_overrides.get(require_tenant_context)
    app.dependency_overrides[require_tenant_context] = lambda: TenantContext(
        tenant_id="action-plan-permission-tenant", user_id="action-plan-viewer",
        role="viewer", scope_type="seller", seller_id="action-plan-seller",
    )
    # The role check must reject the request before touching persistence.
    monkeypatch.setattr(routes, "engine", object())
    try:
        with TestClient(app) as client:
            updated = client.put(
                "/complex-analysis/action-plans/action-plan-1",
                json={"title": "unauthorized edit"},
            )
            deleted = client.delete(
                "/complex-analysis/action-plans/action-plan-1",
            )
        assert updated.status_code == 403
        assert deleted.status_code == 403
    finally:
        if previous is None:
            app.dependency_overrides.pop(require_tenant_context, None)
        else:
            app.dependency_overrides[require_tenant_context] = previous


def test_action_plan_and_execution_queries_bind_tenant_and_user_scope(monkeypatch) -> None:
    statements: list[tuple[str, dict]] = []
    context = TenantContext(
        tenant_id="scoped-tenant", user_id="scoped-user", role="owner",
        scope_type="seller", seller_id="scoped-seller",
    )
    action_plan = {
        "action_plan_id": "plan-1", "title": "Plan", "content": {},
        "status": "draft", "version": 1,
    }

    class Result:
        def __init__(self, sql: str):
            self.sql = sql

        def mappings(self):
            return self

        def all(self):
            return []

        def one_or_none(self):
            if "DELETE FROM governance.action_plans" in self.sql:
                return {"action_plan_id": "plan-1"}
            if "SELECT title,content,status FROM governance.action_plans" in self.sql:
                return {"title": "Plan", "content": {}, "status": "draft"}
            return None

        def one(self):
            return {**action_plan, "title": "Updated", "version": 2}

    class Connection:
        def execute(self, statement, params=None):
            sql = str(statement)
            statements.append((sql, dict(params or {})))
            return Result(sql)

    class Begin:
        def __enter__(self):
            return Connection()

        def __exit__(self, *_args):
            return False

    class Engine:
        def begin(self):
            return Begin()

    monkeypatch.setattr(routes, "engine", Engine())

    routes.list_executions(None, 0, 20, context)
    routes.list_action_plans(0, 20, context)
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        routes.get_action_plan("plan-1", context=context)
    routes.update_action_plan("plan-1", routes.ActionPlanUpdate(title="Updated"), context=context)
    routes.delete_action_plan("plan-1", context=context)

    scoped_queries = [
        (sql, params) for sql, params in statements
        if "governance.action_plans" in sql or "governance.analysis_execution_audits" in sql
    ]
    assert len(scoped_queries) == 6
    for sql, params in scoped_queries:
        assert "tenant_id=:tenant" in sql
        assert "user_id=:user" in sql
        assert params["tenant"] == context.tenant_id
        assert params["user"] == context.user_id


def test_expired_plan_revision_finalizes_its_suspended_chat_lifecycle(monkeypatch) -> None:
    context = TenantContext(
        tenant_id="expired-revision-tenant", user_id="expired-revision-analyst",
        role="analyst", scope_type="seller", seller_id="expired-revision-seller",
    )
    finalized: list[dict] = []
    monkeypatch.setattr(routes, "revise_plan", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("Plan has expired; create a new analysis request")))
    monkeypatch.setattr(routes, "get_plan_v2", lambda *_args, **_kwargs: {
        "plan_id": "expired-plan", "status": "expired",
        "chat_session_id": "session-expired", "chat_turn_id": "turn-expired",
    })
    monkeypatch.setattr(routes, "_resume_plan_lifecycle", lambda *_args, **_kwargs: (object(), "task-expired"))
    monkeypatch.setattr(
        routes, "_finalize_plan_lifecycle",
        lambda binding, **kwargs: finalized.append({"binding": binding, **kwargs}),
    )

    previous = app.dependency_overrides.get(require_tenant_context)
    app.dependency_overrides[require_tenant_context] = lambda: context
    try:
        response = TestClient(app).post(
            "/complex-analysis/plans/expired-plan/revise",
            json={"focus_key": "retention_curve"},
        )
    finally:
        if previous is None:
            app.dependency_overrides.pop(require_tenant_context, None)
        else:
            app.dependency_overrides[require_tenant_context] = previous

    assert response.status_code == 409
    assert len(finalized) == 1
    assert finalized[0]["status"] == "failed"
    assert finalized[0]["task_id"] == "task-expired"
    assert finalized[0]["result"]["error_code"] == "COMPLEX_PLAN_EXPIRED"


@pytest.mark.parametrize(
    ("terminal_status", "error"),
    [("expired", "Plan has expired"), ("failed", "Plan integrity validation failed")],
)
def test_confirm_finalizes_lifecycle_after_persisted_terminal_plan_failure(
    monkeypatch, terminal_status: str, error: str,
) -> None:
    context = TenantContext(
        tenant_id="terminal-plan-tenant", user_id="terminal-plan-analyst",
        role="analyst", scope_type="seller", seller_id="terminal-plan-seller",
    )
    plan = {
        "plan_id": "terminal-plan", "status": "pending_confirmation",
        "chat_session_id": "session-terminal", "chat_turn_id": "turn-terminal",
    }
    finalized: list[dict] = []
    get_calls = 0

    def get_plan(*_args, **_kwargs):
        nonlocal get_calls
        get_calls += 1
        return plan if get_calls == 1 else {**plan, "status": terminal_status}

    monkeypatch.setattr(routes, "get_plan_v2", get_plan)
    monkeypatch.setattr(routes, "_resume_plan_lifecycle", lambda *_args, **_kwargs: (object(), "task-terminal"))
    monkeypatch.setattr(routes, "execute_plan", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError(error)))
    monkeypatch.setattr(
        routes, "_finalize_plan_lifecycle",
        lambda binding, **kwargs: finalized.append({"binding": binding, **kwargs}),
    )

    previous = app.dependency_overrides.get(require_tenant_context)
    app.dependency_overrides[require_tenant_context] = lambda: context
    try:
        response = TestClient(app).post("/complex-analysis/plans/terminal-plan/confirm")
    finally:
        if previous is None:
            app.dependency_overrides.pop(require_tenant_context, None)
        else:
            app.dependency_overrides[require_tenant_context] = previous

    assert response.status_code == 200
    assert '"event": "failed"' in response.text
    assert len(finalized) == 1
    assert finalized[0]["status"] == "failed"
    assert finalized[0]["task_id"] == "task-terminal"

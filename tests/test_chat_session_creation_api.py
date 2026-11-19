from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.auth import require_tenant_context
from app.api.routes import chat_history
from app.main import app
from app.tenancy import TenantContext


def test_create_chat_session_uses_authenticated_context(monkeypatch):
    context = TenantContext(
        tenant_id="tenant-session", user_id="user-session", role="admin", scope_type="platform",
    )
    previous = app.dependency_overrides.get(require_tenant_context)
    app.dependency_overrides[require_tenant_context] = lambda: context
    seen: dict[str, object] = {}

    class FakeRepository:
        def __init__(self, actual_context):
            seen["context"] = actual_context

        def create_session(self, title):
            seen["title"] = title
            return {"id": "session-created", "title": title}

    monkeypatch.setattr(chat_history, "ChatHistoryRepository", FakeRepository)
    try:
        with TestClient(app) as client:
            response = client.post("/chat/sessions", json={"title": "偏好设置"})
    finally:
        app.dependency_overrides.pop(require_tenant_context, None)
        if previous is not None:
            app.dependency_overrides[require_tenant_context] = previous

    assert response.status_code == 201
    assert response.json() == {"id": "session-created", "title": "偏好设置"}
    assert seen["context"] is context
    assert seen["title"] == "偏好设置"

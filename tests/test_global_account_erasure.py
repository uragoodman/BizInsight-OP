from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.auth import require_tenant_context
from app.api.routes import auth as auth_routes
from app.main import app
from app.tenancy import TenantContext


def test_global_account_erasure_requires_explicit_confirmation(monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("account erasure must not be queued without confirmation")

    monkeypatch.setattr(auth_routes, "request_account_erasure", unexpected)
    context = TenantContext(tenant_id="tenant-a", user_id="user-a", role="admin", scope_type="platform")
    previous = app.dependency_overrides.get(require_tenant_context)
    app.dependency_overrides[require_tenant_context] = lambda: context
    try:
        response = TestClient(app).post("/auth/account/erasure", json={
            "confirmation_phrase": "DELETE_MY_ACCOUNT",
            "idempotency_key": "erase-1",
        })
    finally:
        if previous is None:
            app.dependency_overrides.pop(require_tenant_context, None)
        else:
            app.dependency_overrides[require_tenant_context] = previous
    assert response.status_code == 422
    assert response.json()["detail"] == "second_confirmation_required"


def test_global_account_erasure_returns_only_job_and_revocation_status(monkeypatch):
    seen = {}

    def request(context, *, idempotency_key):
        seen["user_id"] = context.user_id
        seen["idempotency_key"] = idempotency_key
        return {"job_id": "aerase-1", "status": "pending"}

    monkeypatch.setattr(auth_routes, "request_account_erasure", request)
    context = TenantContext(tenant_id="tenant-a", user_id="user-a", role="admin", scope_type="platform")
    previous = app.dependency_overrides.get(require_tenant_context)
    app.dependency_overrides[require_tenant_context] = lambda: context
    try:
        response = TestClient(app).post("/auth/account/erasure", json={
            "confirmation_phrase": "DELETE_MY_ACCOUNT_AND_ERASE_DATA",
            "idempotency_key": "erase-1",
        })
    finally:
        if previous is None:
            app.dependency_overrides.pop(require_tenant_context, None)
        else:
            app.dependency_overrides[require_tenant_context] = previous
    assert response.status_code == 202
    assert response.json() == {"job_id": "aerase-1", "status": "pending", "access_revoked": True}
    assert seen == {"user_id": "user-a", "idempotency_key": "erase-1"}

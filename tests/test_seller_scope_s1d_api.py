from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.auth import require_tenant_context
from app.main import app
from app.tenancy import SellerGrant, TenantContext


@pytest.fixture
def seller_scope_client(monkeypatch) -> Iterator[tuple[TestClient, TenantContext]]:
    from app.core.config import Settings

    context = TenantContext(
        tenant_id="seller-scope-tenant",
        user_id="seller-scope-admin",
        role="admin",
        scope_type="platform",
        seller_grants=(
            SellerGrant("A", "2024-01-01", "2024-12-31"),
            SellerGrant("B", "2024-01-01", "2024-12-31"),
        ),
    )
    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: Settings(
            tenant_auth_secret="test-seller-scope-secret",
            seller_dimension_enabled=True,
            seller_multi_scope_enabled=True,
        ),
    )
    previous = app.dependency_overrides.get(require_tenant_context)
    app.dependency_overrides[require_tenant_context] = lambda: context
    try:
        with TestClient(app) as client:
            yield client, context
    finally:
        if previous is None:
            app.dependency_overrides.pop(require_tenant_context, None)
        else:
            app.dependency_overrides[require_tenant_context] = previous


def test_resolve_api_reauthorizes_and_returns_signed_selected_payload(seller_scope_client, monkeypatch):
    from app.api.routes import explorer as routes
    from app.scope.seller_scope import verify_seller_scope_payload

    client, context = seller_scope_client
    observed = {}

    def resolve(intent, *, tenant, authz):
        observed.update(intent=intent, tenant=tenant, authz=authz)
        return object()

    monkeypatch.setattr(routes, "resolve_seller_scope", resolve)
    response = client.post("/data-explorer/seller-scope/resolve", json={
        "seller_scope_type": "selected",
        "seller_ids": ["B", "A", "A"],
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "metric_ids": ["gmv"],
        "granularity": "day",
    })

    assert response.status_code == 200, response.text
    payload = response.json()["seller_scope_payload"]
    intent = verify_seller_scope_payload(payload, tenant=context)
    assert intent.seller_scope_type == "selected"
    assert intent.seller_ids == ("A", "B")
    assert intent.dimensions == ("seller",)
    assert intent.include_unattributed is False
    assert observed["tenant"] is context
    assert payload["provenance"]["source"] == "explorer_selector"
    assert payload["provenance"]["token"]


def test_resolve_api_rejects_scope_expansion_and_invalid_shapes(seller_scope_client):
    client, _context = seller_scope_client
    all_with_ids = client.post("/data-explorer/seller-scope/resolve", json={
        "seller_scope_type": "all", "seller_ids": ["A"],
        "start_date": "2024-01-01", "end_date": "2024-01-31",
    })
    assert all_with_ids.status_code == 422

    selected_unknown = client.post("/data-explorer/seller-scope/resolve", json={
        "seller_scope_type": "selected", "seller_ids": ["missing"],
        "start_date": "2024-01-01", "end_date": "2024-01-31",
    })
    assert selected_unknown.status_code == 403

    selected_include_unknown = client.post("/data-explorer/seller-scope/resolve", json={
        "seller_scope_type": "selected", "seller_ids": ["A"],
        "start_date": "2024-01-01", "end_date": "2024-01-31",
        "include_unattributed": True,
    })
    assert selected_include_unknown.status_code == 422


def test_resolve_api_fails_closed_without_signing_secret(seller_scope_client, monkeypatch):
    from types import SimpleNamespace
    from app.api.routes import explorer as routes

    client, _context = seller_scope_client
    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: SimpleNamespace(
            tenant_auth_secret="",
            seller_dimension_enabled=True,
            seller_multi_scope_enabled=True,
        ),
    )
    monkeypatch.setattr(routes, "resolve_seller_scope", lambda *_args, **_kwargs: object())
    response = client.post("/data-explorer/seller-scope/resolve", json={
        "seller_scope_type": "selected", "seller_ids": ["A"],
        "start_date": "2024-01-01", "end_date": "2024-01-31",
    })
    assert response.status_code == 403
    assert "provenance signing is not configured" in response.json()["detail"]


def test_chat_seller_scope_payload_is_exclusive_with_legacy_explorer_context():
    from app.schemas.chat import ChatRequest

    with pytest.raises(ValidationError, match="cannot be combined with Explorer scope"):
        ChatRequest(
            message="按店铺拆解 GMV",
            use_current_explorer_context=True,
            seller_scope_payload={
                "seller_scope_intent": {},
                "provenance": {
                    "source": "explorer_selector", "token": "token", "expires_at": "2024-01-01T00:00:00+00:00",
                },
            },
        )

"""HTTP-boundary acceptance checks for business APIs with isolated service edges.

These tests exercise FastAPI routing, request validation, auth dependencies,
status mapping, and response serialization. Database/provider behavior is
covered by the dedicated live integration suites.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.auth import require_tenant_context
from app.main import app
from app.tenancy import SellerGrant, TenantContext


@pytest.fixture
def api_client() -> Iterator[TestClient]:
    previous = app.dependency_overrides.get(require_tenant_context)
    context = TenantContext(
        tenant_id="api-acceptance-tenant",
        user_id="api-acceptance-user",
        role="owner",
        scope_type="seller",
        seller_id="api-acceptance-seller",
        seller_grants=(SellerGrant("api-acceptance-seller", "2024-01-01", "2026-12-31"),),
    )
    app.dependency_overrides[require_tenant_context] = lambda: context
    try:
        with TestClient(app) as client:
            yield client
    finally:
        if previous is None:
            app.dependency_overrides.pop(require_tenant_context, None)
        else:
            app.dependency_overrides[require_tenant_context] = previous


def test_auth_profile_http_returns_trusted_context(api_client: TestClient) -> None:
    from app.api.routes import auth as routes

    routes_profile = {
        "tenant_id": "api-acceptance-tenant",
        "user_id": "api-acceptance-user",
        "role": "owner",
    }
    # Profile projection reads the IAM persistence layer; keep this HTTP
    # contract case isolated from the live account fixture.
    from unittest.mock import patch

    with patch.object(routes, "profile", return_value=routes_profile):
        response = api_client.get("/auth/me")
    assert response.status_code == 200
    body = response.json()
    assert body["tenant_id"] == "api-acceptance-tenant"
    assert body["user_id"] == "api-acceptance-user"
    assert body["role"] == "owner"


def test_auth_member_list_http_checks_permission(api_client: TestClient, monkeypatch) -> None:
    from app.api.routes import auth as routes

    monkeypatch.setattr(routes, "list_members", lambda _context: [{"user_id": "member-1"}])
    response = api_client.get("/auth/members")
    assert response.status_code == 200
    assert response.json() == {"items": [{"user_id": "member-1"}]}

    app.dependency_overrides[require_tenant_context] = lambda: TenantContext(
        tenant_id="api-acceptance-tenant", user_id="api-acceptance-viewer",
        role="viewer", scope_type="seller",
        seller_grants=(SellerGrant("api-acceptance-seller"),),
    )
    denied = api_client.get("/auth/members")
    assert denied.status_code == 403


def test_auth_bootstrap_http_rejects_missing_token(api_client: TestClient, monkeypatch) -> None:
    from app.api.routes import auth as routes

    settings = routes.get_settings()
    monkeypatch.setattr(settings, "auth_bootstrap_token", "bootstrap-test-secret")
    response = api_client.post("/auth/bootstrap", json={
        "email": "owner@example.invalid", "password": "not-a-real-password",
        "display_name": "Test Owner",
    })
    assert response.status_code == 403


def test_explorer_seller_lookup_http_returns_items(api_client: TestClient, monkeypatch) -> None:
    from app.api.routes import explorer as routes

    monkeypatch.setattr(routes, "search_sellers", lambda *_args: [{"seller_id": "api-acceptance-seller", "label": "Test shop"}])
    response = api_client.get("/data-explorer/sellers?query=test&limit=5")
    assert response.status_code == 200
    assert response.json()["items"][0]["seller_id"] == "api-acceptance-seller"


def test_explorer_preview_http_serializes_scoped_result(api_client: TestClient, monkeypatch) -> None:
    from app.api.routes import explorer as routes

    monkeypatch.setattr(routes, "preview_explorer", lambda _context, payload: {
        "context": payload.model_dump(mode="json"),
        "points": [{"metric_date": "2026-09-01", "value": 12.5}],
    })
    response = api_client.post("/data-explorer/preview", json={
        "seller_id": "api-acceptance-seller", "metric_name": "seller_avg_freight_per_order",
        "start_date": "2026-09-01", "end_date": "2026-09-02", "granularity": "day",
    })
    assert response.status_code == 200
    assert response.json()["context"]["granularity"] == "day"
    assert response.json()["points"][0]["value"] == 12.5


def test_explorer_rejects_invalid_time_preset_over_http(api_client: TestClient) -> None:
    response = api_client.post("/data-explorer/time-range/resolve", json={"kind": "not-a-preset"})
    assert response.status_code == 422


def test_inventory_query_http_passes_request_to_repository(api_client: TestClient, monkeypatch) -> None:
    from app.data.repositories import inventory

    monkeypatch.setattr(inventory, "query_inventory_snapshot", lambda scope, query: {
        "status": "ok", "seller_id": scope.seller_id,
        "as_of": query.as_of.isoformat(), "items": [], "coverage_status": "unknown",
    })
    response = api_client.post("/data-explorer/inventory-snapshots/query", json={
        "seller_id": "api-acceptance-seller", "start_date": "2026-09-01",
        "end_date": "2026-09-30", "as_of": "2026-09-15T23:59:59Z",
        "quantity_basis": "available",
    })
    assert response.status_code == 200, response.text
    assert response.json()["seller_id"] == "api-acceptance-seller"
    assert response.json()["coverage_status"] == "unknown"


def test_inventory_query_http_rejects_timezone_less_as_of(api_client: TestClient) -> None:
    response = api_client.post("/data-explorer/inventory-snapshots/query", json={
        "seller_id": "api-acceptance-seller", "start_date": "2026-09-01",
        "end_date": "2026-09-30", "as_of": "2026-09-15T12:00:00",
        "quantity_basis": "available",
    })
    assert response.status_code == 422


def test_inventory_ingest_http_serializes_batch_result(api_client: TestClient, monkeypatch) -> None:
    from app.data.repositories import inventory

    monkeypatch.setattr(inventory, "ingest_inventory_snapshot_batch", lambda batch: {
        "status": "ok", "source_batch_id": batch.source_batch_id,
        "received_rows": len(batch.rows), "inserted_rows": len(batch.rows),
        "duplicate_rows_ignored": 0,
    })
    response = api_client.post("/data-explorer/inventory-snapshots/ingest", json={
        "source_system": "acceptance-test", "source_batch_id": "batch-http-001",
        "rows": [{
            "seller_id": "api-acceptance-seller", "warehouse_id": "wh-1", "product_id": "sku-1",
            "snapshot_at": "2026-09-15T23:59:59Z", "snapshot_semantics": "daily_close",
            "available_quantity": 7, "quantity_unit": "unit",
        }],
    })
    assert response.status_code == 200, response.text
    assert response.json()["inserted_rows"] == 1
    assert response.json()["source_batch_id"] == "batch-http-001"


def test_code_analysis_file_upload_http_returns_contract(api_client: TestClient, monkeypatch) -> None:
    from app.api.routes import code_analysis as routes

    monkeypatch.setattr(routes, "upload_file", lambda _context, name, _content_type, raw: {
        "file_id": "file_acceptance_01", "filename": name,
        "size_bytes": len(raw), "file_type": "csv", "sha256": "a" * 64,
        "data_contract": {"file_type": "csv", "row_count": 1},
    })
    response = api_client.post(
        "/code-analysis/files", files={"file": ("sales.csv", b"amount\n10\n", "text/csv")},
    )
    assert response.status_code == 200, response.text
    assert response.json()["file_type"] == "csv"
    assert response.json()["data_contract"]["row_count"] == 1


def test_code_analysis_job_create_http_returns_accepted_job(api_client: TestClient, monkeypatch) -> None:
    from app.api.routes import code_analysis as routes

    monkeypatch.setattr(routes, "create_job", lambda _context, file_id, question, *_args: {
        "job_id": "caj_acceptance_01", "file_id": file_id,
        "question": question, "status": "queued",
    })
    response = api_client.post("/code-analysis/jobs", json={
        "file_id": "file_acceptance_01", "question": "按月汇总销售额并说明限制",
    })
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "queued"
    assert response.json()["file_id"] == "file_acceptance_01"


def test_code_analysis_job_events_http_returns_sse_terminal_status(api_client: TestClient, monkeypatch) -> None:
    from app.api.routes import code_analysis as routes

    monkeypatch.setattr(routes, "get_job", lambda *_args: {"job_id": "caj_acceptance_01", "status": "completed"})
    response = api_client.get("/code-analysis/jobs/caj_acceptance_01/events")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"event": "completed"' in response.text


def test_complex_analysis_plan_create_http_fails_closed_without_sealed_decision(api_client: TestClient) -> None:
    response = api_client.post("/complex-analysis/plans", json={
        "question": "比较最近两期销售变化并分析商品类别贡献",
        "explorer_context": {"seller_id": "api-acceptance-seller", "granularity": "month"},
    })
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == "sealed_route_decision_required"


def test_complex_analysis_plan_read_http_returns_not_found(api_client: TestClient, monkeypatch) -> None:
    from app.api.routes import complex_analysis as routes

    monkeypatch.setattr(routes, "get_plan_v2", lambda *_args: None)
    response = api_client.get("/complex-analysis/plans/missing-plan")
    assert response.status_code == 404


def test_complex_analysis_cancel_http_serializes_state_change(api_client: TestClient, monkeypatch) -> None:
    from app.api.routes import complex_analysis as routes

    monkeypatch.setattr(routes, "cancel_plan", lambda *_args: True)
    response = api_client.post("/complex-analysis/plans/cap_acceptance_01/cancel")
    assert response.status_code == 200
    assert response.json() == {"cancelled": True}

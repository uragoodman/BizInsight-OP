from datetime import date

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.api.routes import explorer
from app.data.inventory_contracts import InventorySnapshotBatch, InventorySnapshotRequest
from app.tenancy import SellerGrant, TenantContext


def _tenant(role="analyst"):
    return TenantContext(
        tenant_id="tenant-1", user_id="user-1", role=role,
        scope_type="seller", seller_id="seller-1",
        seller_grants=(SellerGrant("seller-1", "2024-01-01", "2024-12-31"),),
    )


def test_inventory_request_requires_timezone_and_as_of_inside_scope():
    values = {
        "seller_id": "seller-1", "start_date": "2024-01-01",
        "end_date": "2024-01-31", "as_of": "2024-01-20T12:00:00Z",
        "quantity_basis": "available",
    }
    assert InventorySnapshotRequest.model_validate(values).as_of.tzinfo is not None
    with pytest.raises(ValidationError, match="timezone"):
        InventorySnapshotRequest.model_validate({**values, "as_of": "2024-01-20T12:00:00"})
    with pytest.raises(ValidationError, match="inside the requested date scope"):
        InventorySnapshotRequest.model_validate({**values, "as_of": "2024-02-01T00:00:00Z"})


def test_inventory_query_route_requires_analysis_permission_and_passes_locked_bounds(monkeypatch):
    request = InventorySnapshotRequest.model_validate({
        "seller_id": "seller-1", "start_date": "2024-01-01",
        "end_date": "2024-01-31", "as_of": "2024-01-20T12:00:00Z",
        "quantity_basis": "available", "limit": 120,
    })
    seen = {}
    monkeypatch.setattr(explorer, "require_permission", lambda _tenant, permission: seen.setdefault("permission", permission))

    def query(scope, query):
        seen["scope"] = scope
        seen["query"] = query
        return {"status": "ok", "coverage_status": "unknown"}

    monkeypatch.setattr("app.data.repositories.inventory.query_inventory_snapshot", query)
    response = explorer.inventory_snapshot_query(request, _tenant())

    assert response["status"] == "ok"
    assert seen["permission"] == "analyze"
    assert seen["scope"].seller_id == "seller-1"
    assert seen["scope"].default_start_date == date(2024, 1, 1)
    assert seen["scope"].default_end_date == date(2024, 1, 31)
    assert seen["scope"].max_rows == 120
    assert seen["query"].quantity_basis == "available"


def test_inventory_query_route_maps_permission_failure_to_403(monkeypatch):
    request = InventorySnapshotRequest.model_validate({
        "seller_id": "seller-1", "start_date": "2024-01-01",
        "end_date": "2024-01-31", "as_of": "2024-01-20T12:00:00Z",
        "quantity_basis": "on_hand",
    })
    monkeypatch.setattr(explorer, "require_permission", lambda *_: (_ for _ in ()).throw(PermissionError("denied")))
    with pytest.raises(HTTPException) as caught:
        explorer.inventory_snapshot_query(request, _tenant())
    assert caught.value.status_code == 403


def test_inventory_ingest_route_requires_management_permission_and_passes_batch(monkeypatch):
    batch = InventorySnapshotBatch.model_validate({
        "source_system": "erp",
        "source_batch_id": "batch-1",
        "rows": [{
            "seller_id": "seller-1",
            "warehouse_id": "warehouse-1",
            "product_id": "product-1",
            "snapshot_at": "2024-01-20T23:59:59Z",
            "snapshot_semantics": "daily_close",
            "on_hand_quantity": 12,
            "available_quantity": 10,
            "quantity_unit": "unit",
        }],
    })
    seen = {}
    monkeypatch.setattr(explorer, "require_permission", lambda _tenant, permission: seen.setdefault("permission", permission))

    def ingest(received_batch):
        seen["batch"] = received_batch
        return {"status": "ok", "received_rows": 1, "inserted_rows": 1}

    monkeypatch.setattr("app.data.repositories.inventory.ingest_inventory_snapshot_batch", ingest)
    result = explorer.inventory_snapshot_ingest(batch, _tenant(role="owner"))

    assert result["status"] == "ok"
    assert seen["permission"] == "manage_sellers"
    assert seen["batch"] is batch


def test_inventory_ingest_route_maps_missing_management_permission_to_403(monkeypatch):
    batch = InventorySnapshotBatch.model_validate({
        "source_system": "erp",
        "source_batch_id": "batch-1",
        "rows": [{
            "seller_id": "seller-1",
            "warehouse_id": "warehouse-1",
            "product_id": "product-1",
            "snapshot_at": "2024-01-20T23:59:59Z",
            "snapshot_semantics": "daily_close",
            "available_quantity": 10,
            "quantity_unit": "unit",
        }],
    })
    monkeypatch.setattr(explorer, "require_permission", lambda *_: (_ for _ in ()).throw(PermissionError("denied")))
    with pytest.raises(HTTPException) as caught:
        explorer.inventory_snapshot_ingest(batch, _tenant(role="analyst"))
    assert caught.value.status_code == 403

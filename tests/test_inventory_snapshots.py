from contextlib import contextmanager
from datetime import date, datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.data.inventory_contracts import (
    InventorySnapshotBatch,
    InventorySnapshotInput,
    InventorySnapshotQuery,
)
from app.data.repositories import inventory
from app.schemas.explorer import ControlledAnalysisScope
from app.tenancy import SellerGrant, TenantContext


def _row(**updates):
    values = {
        "seller_id": "seller-1",
        "warehouse_id": "warehouse-1",
        "product_id": "product-1",
        "snapshot_at": "2024-01-15T23:00:00-03:00",
        "snapshot_semantics": "daily_close",
        "on_hand_quantity": "12",
        "reserved_quantity": "3",
        "available_quantity": "8",
        "quantity_unit": "unit",
    }
    values.update(updates)
    return values


def _scope():
    return ControlledAnalysisScope.model_validate({
        "seller_id": "seller-1",
        "allowed_metrics": ("gmv",),
        "default_start_date": date(2024, 1, 1),
        "default_end_date": date(2024, 1, 31),
        "metric_name": "gmv",
        "granularity": "day",
    })


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _Connection:
    def __init__(self, rows):
        self.rows = rows
        self.statement = ""
        self.params = None

    def execute(self, statement, params=None):
        self.statement = str(statement)
        self.params = params
        if isinstance(params, list):
            return _Result([{"snapshot_row_id": index} for index, _ in enumerate(params[:1], 1)])
        return _Result(self.rows)


def _tenant(role="owner"):
    return TenantContext(
        tenant_id="tenant-1", user_id="user-1", role=role,
        scope_type="seller", seller_id="seller-1",
        seller_grants=(SellerGrant(seller_id="seller-1"),),
    )


def test_inventory_contract_requires_tz_explicit_basis_and_finite_quantities():
    valid = InventorySnapshotInput.model_validate(_row())
    assert valid.snapshot_at.utcoffset() is not None
    with pytest.raises(ValidationError):
        InventorySnapshotInput.model_validate(_row(snapshot_at="2024-01-15T23:00:00"))
    with pytest.raises(ValidationError):
        InventorySnapshotInput.model_validate(_row(on_hand_quantity=None, available_quantity=None, reserved_quantity=None))
    with pytest.raises(ValidationError, match="reserved quantity alone"):
        InventorySnapshotInput.model_validate(_row(on_hand_quantity=None, available_quantity=None, reserved_quantity="3"))
    with pytest.raises(ValidationError):
        InventorySnapshotInput.model_validate(_row(on_hand_quantity="NaN"))


def test_inventory_batch_rejects_multiple_sellers():
    with pytest.raises(ValidationError):
        InventorySnapshotBatch.model_validate({
            "source_system": "erp",
            "source_batch_id": "batch-1",
            "rows": [_row(), _row(seller_id="seller-2")],
        })


def test_inventory_ingest_is_owner_only_transaction_scoped_and_idempotent(monkeypatch):
    connection = _Connection([])
    calls = []

    @contextmanager
    def begin():
        yield connection

    class _Authz:
        def require_sellers(self, seller_ids):
            calls.append(("sellers", seller_ids))

        def require_date_range(self, seller_id, start, end):
            calls.append(("dates", seller_id, start, end))

    monkeypatch.setattr(inventory.engine, "begin", begin)
    monkeypatch.setattr("app.data.repositories.inventory.current_tenant_context", lambda: _tenant())
    monkeypatch.setattr("app.data.repositories.inventory.set_local_context", lambda *_a, **_kw: None)
    monkeypatch.setattr("app.auth.context.build_authorization_context", lambda *_a, **_kw: _Authz())
    batch = InventorySnapshotBatch.model_validate({
        "source_system": "erp",
        "source_batch_id": "batch-1",
        "rows": [_row(), _row(product_id="product-2")],
    })

    result = inventory.ingest_inventory_snapshot_batch(batch)

    assert result["received_rows"] == 2
    assert result["inserted_rows"] == 1
    assert result["duplicate_rows_ignored"] == 1
    assert "ON CONFLICT" in connection.statement
    assert connection.params[0]["tenant_id"] == "tenant-1"
    assert len(calls) == 2  # one seller check and one bounded date check


def test_inventory_ingest_denies_non_owner_and_out_of_scope_seller(monkeypatch):
    batch = InventorySnapshotBatch.model_validate({
        "source_system": "erp", "source_batch_id": "b1", "rows": [_row()],
    })
    monkeypatch.setattr("app.data.repositories.inventory.current_tenant_context", lambda: _tenant("analyst"))
    with pytest.raises(PermissionError, match="owner"):
        inventory.ingest_inventory_snapshot_batch(batch)


def test_inventory_as_of_query_keeps_missing_distinct_from_zero(monkeypatch):
    connection = _Connection([{
        "warehouse_id": "warehouse-1",
        "product_id": "product-1",
        "snapshot_at": datetime(2024, 1, 15, 23, tzinfo=timezone.utc),
        "snapshot_semantics": "daily_close",
        "quantity": None,
        "quantity_unit": "unit",
        "source_system": "erp",
    }])

    @contextmanager
    def begin():
        yield connection

    class _Authz:
        def require_sellers(self, _seller_ids):
            return None

        def require_date_range(self, *_args):
            return None

    monkeypatch.setattr(inventory.engine, "begin", begin)
    monkeypatch.setattr("app.data.repositories.inventory.current_tenant_context", lambda: _tenant("analyst"))
    monkeypatch.setattr("app.data.repositories.inventory.set_local_context", lambda *_a, **_kw: None)
    monkeypatch.setattr("app.auth.context.build_authorization_context", lambda *_a, **_kw: _Authz())
    query = InventorySnapshotQuery.model_validate({
        "seller_id": "seller-1",
        "as_of": "2024-01-20T00:00:00Z",
        "quantity_basis": "available",
    })

    result = inventory.query_inventory_snapshot(_scope(), query)

    assert result["status"] == "ok"
    assert result["items"][0]["quantity"] is None
    assert result["items"][0]["measurement_status"] == "missing"
    assert result["coverage_status"] == "unknown_without_authoritative_product_warehouse_catalog"
    assert "DISTINCT ON (warehouse_id, product_id)" in connection.statement
    assert "ORDER BY warehouse_id, product_id, snapshot_at DESC" in connection.statement

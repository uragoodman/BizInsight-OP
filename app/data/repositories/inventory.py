"""Tenant- and seller-scoped inventory snapshot ingestion and as-of lookup."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Any

from sqlalchemy import text

from app.data.database import engine
from app.data.rls import set_local_context
from app.data.inventory_contracts import InventorySnapshotBatch, InventorySnapshotQuery
from app.schemas.explorer import ControlledAnalysisScope
from app.tenancy import current_tenant_context


def _require_seller_access(
    tenant: Any, seller_id: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> None:
    from app.auth.context import AuthorizationDenied, build_authorization_context

    authz = build_authorization_context(tenant, metric_scope_type="seller")
    try:
        authz.require_sellers([seller_id])
        if start_date is not None:
            authz.require_date_range(seller_id, start_date, end_date or start_date)
    except AuthorizationDenied as exc:
        raise PermissionError(str(exc)) from exc


def ingest_inventory_snapshot_batch(
    batch: InventorySnapshotBatch,
) -> dict[str, object]:
    """Insert immutable source snapshots; duplicate natural keys are ignored.

    Ingestion is currently an owner-only backend operation. The tenant and
    actor are always derived from trusted request context, never the batch.
    Corrections should arrive as an explicitly governed new source/snapshot,
    not silently overwrite a previously recorded stock observation.
    """
    tenant = current_tenant_context()
    if tenant is None:
        raise PermissionError("Trusted tenant context is required for inventory ingestion")
    if tenant.role != "owner" or tenant.scope_type != "seller":
        raise PermissionError("Only a seller owner may ingest inventory snapshots")
    seller_ids = {row.seller_id for row in batch.rows}
    if len(seller_ids) != 1:
        raise ValueError("one inventory batch may contain rows for only one seller")
    seller_id = next(iter(seller_ids))
    snapshot_dates = [row.snapshot_at.date() for row in batch.rows]
    _require_seller_access(tenant, seller_id, min(snapshot_dates), max(snapshot_dates))

    statement = text("""
        INSERT INTO analytics.inventory_snapshots (
          tenant_id, seller_id, warehouse_id, product_id, sku, snapshot_at,
          snapshot_semantics, on_hand_quantity, reserved_quantity,
          available_quantity, quantity_unit, source_system, source_batch_id,
          source_row_number, ingested_by
        ) VALUES (
          :tenant_id, :seller_id, :warehouse_id, :product_id, :sku, :snapshot_at,
          :snapshot_semantics, :on_hand_quantity, :reserved_quantity,
          :available_quantity, :quantity_unit, :source_system, :source_batch_id,
          :source_row_number, :ingested_by
        )
        ON CONFLICT (tenant_id, seller_id, warehouse_id, product_id, snapshot_at, source_system)
        DO NOTHING
        RETURNING snapshot_row_id
    """)
    params = [
        {
            "tenant_id": tenant.tenant_id,
            "seller_id": row.seller_id,
            "warehouse_id": row.warehouse_id,
            "product_id": row.product_id,
            "sku": row.sku,
            "snapshot_at": row.snapshot_at,
            "snapshot_semantics": row.snapshot_semantics,
            "on_hand_quantity": row.on_hand_quantity,
            "reserved_quantity": row.reserved_quantity,
            "available_quantity": row.available_quantity,
            "quantity_unit": row.quantity_unit,
            "source_system": batch.source_system,
            "source_batch_id": batch.source_batch_id,
            "source_row_number": index,
            "ingested_by": tenant.user_id,
        }
        for index, row in enumerate(batch.rows, start=1)
    ]
    with engine.begin() as connection:
        set_local_context(
            connection, tenant, scope_type="seller",
            seller_access={seller_id: (min(snapshot_dates), max(snapshot_dates))},
        )
        inserted = connection.execute(statement, params).mappings().all()
    inserted_count = len(inserted)
    return {
        "status": "ok",
        "seller_id": seller_id,
        "source_system": batch.source_system,
        "source_batch_id": batch.source_batch_id,
        "received_rows": len(batch.rows),
        "inserted_rows": inserted_count,
        "duplicate_rows_ignored": len(batch.rows) - inserted_count,
        "snapshot_semantics": sorted({row.snapshot_semantics for row in batch.rows}),
        "quantity_units": sorted({row.quantity_unit for row in batch.rows}),
    }


def query_inventory_snapshot(
    scope: ControlledAnalysisScope,
    query: InventorySnapshotQuery,
) -> dict[str, object]:
    """Return each product/warehouse's latest observed row at ``as_of``.

    A latest row with NULL for the requested quantity basis stays NULL. Missing
    product/warehouse observations are not turned into zero and are not treated
    as a complete catalog because the source does not provide that catalog.
    """
    tenant = current_tenant_context()
    if tenant is None:
        raise PermissionError("Trusted tenant context is required for inventory queries")
    if query.seller_id != scope.seller_id:
        raise PermissionError("Inventory seller differs from the locked analysis scope")
    _require_seller_access(tenant, query.seller_id, query.as_of.date(), query.as_of.date())
    if query.as_of.date() < scope.default_start_date or query.as_of.date() > scope.default_end_date:
        raise ValueError("Inventory as_of is outside the locked date scope")
    if query.limit > scope.max_rows:
        raise ValueError("Inventory result limit exceeds the locked row policy")

    quantity_column = {
        "on_hand": "on_hand_quantity",
        "available": "available_quantity",
    }[query.quantity_basis]
    filters = [
        "tenant_id = :tenant_id",
        "seller_id = :seller_id",
        "snapshot_at >= :scope_start",
        "snapshot_at <= :as_of",
    ]
    params: dict[str, object] = {
        "tenant_id": tenant.tenant_id,
        "seller_id": query.seller_id,
        "scope_start": datetime.combine(scope.default_start_date, time.min, tzinfo=timezone.utc),
        "as_of": query.as_of,
        "limit": query.limit + 1,
    }
    if query.warehouse_id:
        filters.append("warehouse_id = :warehouse_id")
        params["warehouse_id"] = query.warehouse_id
    if query.product_id:
        filters.append("product_id = :product_id")
        params["product_id"] = query.product_id
    statement = text(f"""
        WITH latest AS (
          SELECT DISTINCT ON (warehouse_id, product_id)
                 warehouse_id, product_id, snapshot_at, snapshot_semantics,
                 {quantity_column} AS quantity, quantity_unit, source_system
          FROM analytics.inventory_snapshots
          WHERE {' AND '.join(filters)}
          ORDER BY warehouse_id, product_id, snapshot_at DESC, created_at DESC
        )
        SELECT * FROM latest
        ORDER BY warehouse_id, product_id
        LIMIT :limit
    """)
    with engine.begin() as connection:
        set_local_context(
            connection, tenant, scope_type="seller",
            seller_access={query.seller_id: (scope.default_start_date, query.as_of.date())},
        )
        rows = connection.execute(statement, params).mappings().all()
    truncated = len(rows) > query.limit
    items = [
        {
            "warehouse_id": str(row["warehouse_id"]),
            "product_id": str(row["product_id"]),
            "snapshot_at": row["snapshot_at"].isoformat(),
            "snapshot_semantics": str(row["snapshot_semantics"]),
            "quantity": float(row["quantity"]) if row["quantity"] is not None else None,
            "quantity_unit": str(row["quantity_unit"]),
            "source_system": str(row["source_system"]),
            "measurement_status": "observed" if row["quantity"] is not None else "missing",
        }
        for row in rows[:query.limit]
    ]
    return {
        "status": "ok" if items else "not_found",
        "seller_id": query.seller_id,
        "as_of": query.as_of.isoformat(),
        "quantity_basis": query.quantity_basis,
        "observed_product_warehouse_count": len(items),
        "truncated": truncated,
        "items": items,
        "coverage_status": "unknown_without_authoritative_product_warehouse_catalog",
        "missing_policy": "No observed row is not interpreted as zero stock; a NULL latest measurement remains missing.",
        "source_table": "analytics.inventory_snapshots",
    }

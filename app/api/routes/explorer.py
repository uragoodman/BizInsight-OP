"""数据浏览器 API：前端获得展示快照，Agent 仍在后端重新取数。"""

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.auth import require_tenant_context
from app.core.time_expression import data_time_bounds, resolve_preset
from app.data.repositories.explorer import preview_explorer, search_sellers
from app.schemas.explorer import ExplorerQuery, SellerScopeResolutionRequest, TimeRangePresetRequest
from app.scope.canonical import ScopeError
from app.scope.seller_scope import SellerScopeIntent, issue_seller_scope_payload, resolve_seller_scope
from app.data.inventory_contracts import (
    InventorySnapshotBatch,
    InventorySnapshotQuery,
    InventorySnapshotRequest,
)
from app.tenancy import TenantContext, activate_tenant_context
from app.iam.service import require_permission

router = APIRouter(prefix="/data-explorer", tags=["data-explorer"])


@router.get("/time-bounds")
def time_bounds(tenant_context: TenantContext = Depends(require_tenant_context)) -> dict[str, str]:
    first, latest = data_time_bounds()
    if tenant_context.scope_type == "seller" and len(tenant_context.seller_grants) == 1:
        grant = tenant_context.seller_grants[0]
        if grant.start_date:
            first = max(first, type(first).fromisoformat(grant.start_date))
        if grant.end_date:
            latest = min(latest, type(latest).fromisoformat(grant.end_date))
    return {"min_date": first.isoformat(), "max_date": latest.isoformat(), "relative_anchor": "dataset_latest_date"}


@router.post("/time-range/resolve")
def resolve_time_range(payload: TimeRangePresetRequest, tenant_context: TenantContext = Depends(require_tenant_context)) -> dict[str, object]:
    try:
        return resolve_preset(payload.kind, year=payload.year, value=payload.value).as_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.get("/sellers")
def sellers(
    query: str = Query(default="", max_length=64), limit: int = Query(default=20, ge=1, le=50),
    tenant_context: TenantContext = Depends(require_tenant_context),
) -> dict[str, object]:
    if tenant_context.is_platform_admin:
        # The shared seller search is global for platform administrators. The
        # selector must expose only explicit server grants so the UI cannot
        # offer sellers that the later resolver will reject.
        items: list[dict[str, object]] = []
        needle = query.strip().casefold()
        for grant in tenant_context.seller_grants:
            seller_id = grant.seller_id
            if needle and needle not in seller_id.casefold():
                continue
            rows = search_sellers(tenant_context, seller_id, 50)
            match = next((item for item in rows if item.get("seller_id") == seller_id), None)
            if match is not None:
                items.append(match)
            if len(items) >= limit:
                break
        return {"items": items}
    return {"items": search_sellers(tenant_context, query, limit)}


@router.post("/seller-scope/resolve")
def resolve_seller_scope_selection(
    payload: SellerScopeResolutionRequest,
    tenant_context: TenantContext = Depends(require_tenant_context),
) -> dict[str, object]:
    """Reauthorize a UI selection and sign a short-lived execution payload."""
    try:
        require_permission(tenant_context, "analyze")
        from app.auth.context import build_authorization_context

        auth_scope = "seller" if tenant_context.scope_type == "seller" else "platform"
        authz = build_authorization_context(tenant_context, metric_scope_type=auth_scope)
        intent = SellerScopeIntent(
            seller_scope_type=payload.seller_scope_type,
            seller_ids=payload.seller_ids,
            start_date=payload.start_date,
            end_date=payload.end_date,
            metric_ids=payload.metric_ids,
            dimensions=("seller",),
            granularity=payload.granularity,
            include_unattributed=payload.include_unattributed,
        )
        # Resolve before minting provenance: flags, actor, grants, dates and
        # metric/dimension capability are all checked against current server
        # authorization facts.
        resolve_seller_scope(intent, tenant=tenant_context, authz=authz)
        return {"seller_scope_payload": issue_seller_scope_payload(intent, tenant=tenant_context)}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ScopeError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/preview")
def preview(payload: ExplorerQuery, tenant_context: TenantContext = Depends(require_tenant_context)) -> dict[str, object]:
    try:
        require_permission(tenant_context, "analyze")
        return preview_explorer(tenant_context, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/inventory-snapshots/query")
def inventory_snapshot_query(
    payload: InventorySnapshotRequest,
    tenant_context: TenantContext = Depends(require_tenant_context),
) -> dict[str, object]:
    """Query an observed inventory snapshot without imputing missing stock as zero."""
    from app.data.repositories.inventory import query_inventory_snapshot
    from app.schemas.explorer import ControlledAnalysisScope

    try:
        require_permission(tenant_context, "analyze")
        scope = ControlledAnalysisScope.model_validate({
            "seller_id": payload.seller_id,
            "allowed_metrics": (),
            "default_start_date": payload.start_date,
            "default_end_date": payload.end_date,
            "metric_name": "inventory_snapshot_observed",
            "granularity": "day",
            "max_rows": min(10_000, payload.limit),
        })
        query = InventorySnapshotQuery.model_validate({
            "seller_id": payload.seller_id,
            "as_of": payload.as_of,
            "quantity_basis": payload.quantity_basis,
            "warehouse_id": payload.warehouse_id,
            "product_id": payload.product_id,
            "limit": payload.limit,
        })
        with activate_tenant_context(tenant_context):
            return query_inventory_snapshot(scope, query)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/inventory-snapshots/ingest")
def inventory_snapshot_ingest(
    payload: InventorySnapshotBatch,
    tenant_context: TenantContext = Depends(require_tenant_context),
) -> dict[str, object]:
    """Owner-only bounded ingestion; trusted tenant/actor are never request fields."""
    from app.data.repositories.inventory import ingest_inventory_snapshot_batch

    try:
        require_permission(tenant_context, "manage_sellers")
        with activate_tenant_context(tenant_context):
            return ingest_inventory_snapshot_batch(payload)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

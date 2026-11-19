"""GMV data-version manifests backed by immutable PostgreSQL rows."""
from __future__ import annotations

import hashlib
import json
from datetime import date

from sqlalchemy import text

from app.data.database import engine
from app.metrics.registry import get_registry
from app.tenancy import TenantContext

GMV_DATASET = "tenant_gmv_items"


def validate_gmv_dependency_contract() -> None:
    """Fail fast if registry declaration diverges from the audited GMV source.

    The controlled seller implementation ultimately reads tenant_fact_order_items;
    the versioned adapter snapshots that table under the explicit dataset id.
    Keeping this assertion executable prevents a cosmetic manifest contract.
    """
    declared = get_registry().get("gmv").data_dependencies
    if declared != (GMV_DATASET,):
        raise RuntimeError(f"GMV data_dependencies must be {(GMV_DATASET,)!r}, got {declared!r}")


def _scope(conn, context: TenantContext) -> None:
    conn.execute(text("SELECT set_config('app.tenant_id', :v, true)"), {"v": context.tenant_id})
    conn.execute(text("SELECT set_config('app.user_id', :v, true)"), {"v": context.user_id})
    conn.execute(text("SELECT set_config('app.scope_type', :v, true)"), {"v": context.scope_type})


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def seal_gmv_version(context: TenantContext, parent_version_id: str | None = None) -> dict:
    """Copy tenant GMV source rows, hash the final stored representation, then seal it."""
    with engine.begin() as conn:
        _scope(conn, context)
        rows = conn.execute(text("""SELECT order_id,order_item_id,seller_id,purchase_date,price,freight_value,order_status
            FROM analytics.tenant_fact_order_items WHERE tenant_id=:tenant ORDER BY order_id,order_item_id"""), {"tenant": context.tenant_id}).mappings().all()
        canonical = [dict(row) for row in rows]
        content_hash = _hash(canonical)
        existing = conn.execute(text("""SELECT * FROM governance.data_versions
            WHERE tenant_id=:tenant AND dataset_id=:dataset AND content_hash=:hash"""), {"tenant": context.tenant_id, "dataset": GMV_DATASET, "hash": content_hash}).mappings().first()
        if existing:
            return dict(existing)
        version_id = f"gmv_{content_hash[:48]}"
        schema_hash = _hash(["order_id","order_item_id","seller_id","purchase_date","price","freight_value","order_status"])
        conn.execute(text("""INSERT INTO governance.data_versions
            (version_id,tenant_id,dataset_id,status,content_hash,schema_hash,row_count,min_business_date,max_business_date,affected_sellers,storage_ref,parent_version_id,sealed_at)
            VALUES (:id,:tenant,:dataset,'writing',:content,:schema,:count,:min,:max,CAST(:sellers AS jsonb),:storage,:parent,now())"""), {
                "id": version_id, "tenant": context.tenant_id, "dataset": GMV_DATASET, "content": content_hash, "schema": schema_hash,
                "count": len(canonical), "min": min((r["purchase_date"] for r in canonical), default=None), "max": max((r["purchase_date"] for r in canonical), default=None),
                "sellers": json.dumps(sorted({r["seller_id"] for r in canonical})), "storage": "analytics.tenant_gmv_version_rows", "parent": parent_version_id,
            })
        for row in canonical:
            conn.execute(text("""INSERT INTO analytics.tenant_gmv_version_rows
                (tenant_id,ingestion_version,order_id,order_item_id,seller_id,purchase_date,price,freight_value,order_status)
                VALUES (:tenant,:version,:order,:item,:seller,:date,:price,:freight,:status)"""), {
                    "tenant": context.tenant_id, "version": version_id, "order": row["order_id"], "item": row["order_item_id"], "seller": row["seller_id"], "date": row["purchase_date"], "price": row["price"], "freight": row["freight_value"], "status": row["order_status"],
                })
        conn.execute(text("UPDATE governance.data_versions SET status='sealed' WHERE version_id=:id AND status='writing'"), {"id": version_id})
        return dict(conn.execute(text("SELECT * FROM governance.data_versions WHERE version_id=:id"), {"id": version_id}).mappings().one())


def resolve_gmv_manifest(context: TenantContext, scope: dict, scope_fingerprint: str) -> dict:
    definition = get_registry().get("gmv")
    metric_version = _hash({"metric_id": definition.metric_id, "version": definition.version, "unit": definition.unit, "aggregation": definition.aggregation})
    with engine.begin() as conn:
        _scope(conn, context)
        version = conn.execute(text("""SELECT version_id FROM governance.data_versions
            WHERE tenant_id=:tenant AND dataset_id=:dataset AND status='sealed' ORDER BY sealed_at DESC LIMIT 1"""), {"tenant": context.tenant_id, "dataset": GMV_DATASET}).scalar_one_or_none()
        if not version:
            raise ValueError("No sealed GMV data version is available")
        versions = {GMV_DATASET: version}
        manifest_hash = _hash({"versions": versions, "metric_definition_version": metric_version, "scope_fingerprint": scope_fingerprint})
        existing = conn.execute(text("SELECT * FROM governance.data_version_manifests WHERE tenant_id=:tenant AND manifest_hash=:hash"), {"tenant": context.tenant_id, "hash": manifest_hash}).mappings().first()
        if existing:
            return dict(existing)
        manifest_id = f"mfest_{manifest_hash[:48]}"
        conn.execute(text("""INSERT INTO governance.data_version_manifests
            (manifest_id,tenant_id,versions,metric_definition_version,scope_fingerprint,manifest_hash)
            VALUES (:id,:tenant,CAST(:versions AS jsonb),:metric,:scope,:hash)"""), {
                "id": manifest_id, "tenant": context.tenant_id, "versions": json.dumps(versions), "metric": metric_version, "scope": scope_fingerprint, "hash": manifest_hash,
            })
        return dict(conn.execute(text("SELECT * FROM governance.data_version_manifests WHERE manifest_id=:id"), {"id": manifest_id}).mappings().one())


def current_gmv_manifest_hash(context: TenantContext, scope: dict, scope_fingerprint: str) -> str | None:
    """Return the manifest hash a new GMV query would use, without creating it.

    Relation routing runs before the graph resolves its data snapshot. It may
    offer prior evidence IDs only when the exact scope and latest sealed data
    version still match, so this lookup must remain read-only.
    """
    definition = get_registry().get("gmv")
    metric_version = _hash({
        "metric_id": definition.metric_id,
        "version": definition.version,
        "unit": definition.unit,
        "aggregation": definition.aggregation,
    })
    with engine.begin() as conn:
        _scope(conn, context)
        version = conn.execute(text("""SELECT version_id FROM governance.data_versions
            WHERE tenant_id=:tenant AND dataset_id=:dataset AND status='sealed'
            ORDER BY sealed_at DESC LIMIT 1"""), {
            "tenant": context.tenant_id, "dataset": GMV_DATASET,
        }).scalar_one_or_none()
    if not version:
        return None
    return _hash({
        "versions": {GMV_DATASET: version},
        "metric_definition_version": metric_version,
        "scope_fingerprint": scope_fingerprint,
    })


def query_gmv_manifest(context: TenantContext, manifest_id: str, seller_id: str, start_date: date, end_date: date) -> dict:
    with engine.begin() as conn:
        _scope(conn, context)
        manifest = _load_manifest(conn, context, manifest_id)
        version = manifest["versions"][GMV_DATASET]
        row = conn.execute(text("""SELECT COALESCE(SUM(price + freight_value),0) AS value
            FROM analytics.v_tenant_gmv_sealed_rows WHERE tenant_id=:tenant AND ingestion_version=:version AND seller_id=:seller
              AND purchase_date BETWEEN :start AND :end AND order_status='delivered'"""), {"tenant": context.tenant_id, "version": version, "seller": seller_id, "start": start_date, "end": end_date}).mappings().one()
        return {"metric_name": "gmv", "value": float(row["value"]), "manifest_id": manifest_id, "manifest_hash": manifest["manifest_hash"], "source_table": "analytics.tenant_gmv_version_rows", "status": "ok"}


def query_gmv_manifest_trend(context: TenantContext, manifest_id: str, seller_id: str, start_date: date, end_date: date) -> dict:
    with engine.begin() as conn:
        _scope(conn, context)
        manifest = _load_manifest(conn, context, manifest_id)
        rows = conn.execute(text("""SELECT purchase_date AS metric_date, COALESCE(SUM(price + freight_value),0) AS value
          FROM analytics.v_tenant_gmv_sealed_rows WHERE tenant_id=:tenant AND ingestion_version=:version AND seller_id=:seller
          AND purchase_date BETWEEN :start AND :end AND order_status='delivered'
          GROUP BY purchase_date ORDER BY purchase_date"""), {"tenant": context.tenant_id, "version": manifest["versions"][GMV_DATASET], "seller": seller_id, "start": start_date, "end": end_date}).mappings().all()
        return {"metric_name": "gmv", "start_date": start_date.isoformat(), "end_date": end_date.isoformat(), "granularity": "day", "points": [{"metric_date": str(r["metric_date"]), "value": float(r["value"])} for r in rows], "manifest_id": manifest_id, "manifest_hash": manifest["manifest_hash"], "source_table": "analytics.tenant_gmv_version_rows", "status": "ok"}


def versioned_growth_diagnostics(context: TenantContext, manifest_id: str, seller_id: str, start_date: date, end_date: date) -> dict:
    """The one complex-analysis template in the GMV vertical slice.

    Categories/AOV depend on datasets outside the declared GMV contract, so
    they are intentionally omitted rather than silently read from latest data.
    """
    trend = query_gmv_manifest_trend(context, manifest_id, seller_id, start_date, end_date)
    monthly: dict[str, float] = {}
    for point in trend["points"]:
        month = point["metric_date"][:7] + "-01"
        monthly[month] = monthly.get(month, 0.0) + point["value"]
    items = [{"period": month, "gmv": round(value, 2)} for month, value in sorted(monthly.items())]
    summary = "No delivered GMV exists in the locked period."
    if items:
        summary = f"Versioned GMV covers {len(items)} month(s), from {items[0]['gmv']} to {items[-1]['gmv']}."
    return {"status": "ok" if items else "insufficient_data", "summary": summary, "monthly": items,
            "top_categories": [], "manifest_id": manifest_id, "manifest_hash": trend["manifest_hash"],
            "limitations": ["Category and AOV analysis is outside the GMV-only version contract."]}


def get_manifest(context: TenantContext, manifest_hash: str) -> dict | None:
    with engine.begin() as conn:
        _scope(conn, context)
        row = conn.execute(text("SELECT * FROM governance.data_version_manifests WHERE tenant_id=:tenant AND manifest_hash=:hash"), {"tenant": context.tenant_id, "hash": manifest_hash}).mappings().first()
        return dict(row) if row else None


def _load_manifest(conn, context: TenantContext, manifest_id: str) -> dict:
    manifest = conn.execute(text("SELECT * FROM governance.data_version_manifests WHERE manifest_id=:id AND tenant_id=:tenant AND status='sealed'"), {"id": manifest_id, "tenant": context.tenant_id}).mappings().first()
    if not manifest or GMV_DATASET not in manifest["versions"]:
        raise ValueError("Data version manifest is unavailable for this tenant or GMV dependency")
    valid = conn.execute(text("SELECT 1 FROM governance.data_versions WHERE version_id=:id AND tenant_id=:tenant AND dataset_id=:dataset AND status='sealed'"), {"id": manifest["versions"][GMV_DATASET], "tenant": context.tenant_id, "dataset": GMV_DATASET}).scalar_one_or_none()
    if not valid:
        raise ValueError("Manifest references a non-sealed or expired GMV version")
    return dict(manifest)


def impact_memories(context: TenantContext, new_version_id: str, *, apply: bool = False) -> list[dict]:
    """Find (and optionally stale) only memories whose tenant/seller/date overlap a revision."""
    with engine.begin() as conn:
        _scope(conn, context)
        revision = conn.execute(text("SELECT * FROM governance.data_versions WHERE version_id=:id AND tenant_id=:tenant"), {"id": new_version_id, "tenant": context.tenant_id}).mappings().first()
        if not revision:
            raise ValueError("Data version not found")
        parent = revision["parent_version_id"]
        if not parent:
            return []
        manifests = conn.execute(text("SELECT manifest_id,manifest_hash FROM governance.data_version_manifests WHERE tenant_id=:tenant AND versions->>:dataset=:parent"), {"tenant": context.tenant_id, "dataset": GMV_DATASET, "parent": parent}).mappings().all()
        hashes = [m["manifest_hash"] for m in manifests]
        if not hashes:
            return []
        rows = conn.execute(text("""SELECT memory_id,seller_id,start_date,end_date,status,source_data_version
            FROM governance.analysis_memories WHERE source_data_version = ANY(:hashes)
              AND status IN ('active','pending_review') AND start_date <= :max_date AND end_date >= :min_date"""), {"hashes": hashes, "min_date": revision["min_business_date"], "max_date": revision["max_business_date"]}).mappings().all()
        # The JSON predicate above cannot compare a per-row seller through a bound
        # scalar on every PostgreSQL version; enforce the final partition check in
        # Python as a second deterministic guard.
        sellers = set(revision["affected_sellers"] or [])
        impacted = [dict(r) for r in rows if not sellers or r["seller_id"] in sellers]
        if apply:
            for row in impacted:
                conn.execute(text("UPDATE governance.analysis_memories SET status='stale',reviewed_by='system',reviewed_at=now(),review_reason='sealed GMV revision overlaps this artifact' WHERE memory_id=:id AND status IN ('active','pending_review')"), {"id": row["memory_id"]})
                conn.execute(text("""INSERT INTO governance.memory_decisions(decision_id,tenant_id,user_id,memory_id,action,actor,reason,rule_scores,source_memory_ids)
                    VALUES (:id,:tenant,:user,:memory,'staled','system','sealed GMV revision overlaps this artifact',CAST(:scores AS jsonb),'[]'::jsonb)"""), {"id": f"mdec_{hashlib.sha256((str(row['memory_id'])+new_version_id).encode()).hexdigest()[:32]}", "tenant": context.tenant_id, "user": context.user_id, "memory": row["memory_id"], "scores": json.dumps({"new_version_id": new_version_id})})
        return impacted

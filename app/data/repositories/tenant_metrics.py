"""Seller-scoped metrics backed by RLS-protected tenant marts.

This module intentionally exposes only metrics with valid item-level seller
semantics. Platform-only dimensions stay in the legacy repository and are never
silently reinterpreted for a seller.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from sqlalchemy import text

from app.data.database import engine
from app.tenancy import SELLER_SCOPE, TenantContext


SELLER_METRICS = {
    "sales_gmv", "order_count", "customer_count", "avg_order_value", "repeat_purchase_rate",
}


def _require_seller(context: TenantContext) -> None:
    if context.scope_type != SELLER_SCOPE:
        raise PermissionError("Seller metrics require a seller tenant context")


def _set_rls_context(connection, context: TenantContext) -> None:
    connection.execute(text("SELECT set_config('app.tenant_id', :tenant_id, true)"), {"tenant_id": context.tenant_id})
    connection.execute(text("SELECT set_config('app.scope_type', :scope_type, true)"), {"scope_type": context.scope_type})


def query_seller_metric(context: TenantContext, metric_name: str, start_date: str, end_date: str | None = None) -> dict:
    _require_seller(context)
    if metric_name not in SELLER_METRICS:
        raise ValueError(f"Metric {metric_name!r} is not available for seller scope")
    end_date = end_date or start_date
    with engine.begin() as connection:
        _set_rls_context(connection, context)
        row = connection.execute(text(f"""
            SELECT COALESCE(SUM({metric_name}), 0) AS value
            FROM analytics.tenant_daily_metrics
            WHERE tenant_id=:tenant_id AND metric_date BETWEEN :start_date AND :end_date
        """), {"tenant_id": context.tenant_id, "start_date": start_date, "end_date": end_date}).mappings().one()
    # Ratio metrics must be recomputed from their daily numerator/denominator;
    # never sum percentages across dates.
    if metric_name in {"avg_order_value", "repeat_purchase_rate"}:
        with engine.begin() as connection:
            _set_rls_context(connection, context)
            if metric_name == "avg_order_value":
                row = connection.execute(text("""SELECT CASE WHEN SUM(order_count)=0 THEN 0
                    ELSE ROUND(SUM(sales_gmv)/SUM(order_count), 2) END AS value
                    FROM analytics.tenant_daily_metrics WHERE tenant_id=:tenant_id AND metric_date BETWEEN :start_date AND :end_date"""), {"tenant_id": context.tenant_id, "start_date": start_date, "end_date": end_date}).mappings().one()
            else:
                row = connection.execute(text("SELECT ROUND(AVG(repeat_purchase_rate), 2) AS value FROM analytics.tenant_daily_metrics WHERE tenant_id=:tenant_id AND metric_date BETWEEN :start_date AND :end_date"), {"tenant_id": context.tenant_id, "start_date": start_date, "end_date": end_date}).mappings().one()
    return {"metric_name": metric_name, "value": float(row["value"] or 0), "scope": "seller", "tenant_id": context.tenant_id, "seller_id": context.seller_id, "start_date": start_date, "end_date": end_date, "source_table": "analytics.tenant_daily_metrics"}


def query_seller_trend(context: TenantContext, metric_name: str, start_date: str, end_date: str) -> dict:
    _require_seller(context)
    if metric_name not in SELLER_METRICS:
        raise ValueError(f"Metric {metric_name!r} is not available for seller scope")
    with engine.begin() as connection:
        _set_rls_context(connection, context)
        rows = connection.execute(text(f"""SELECT metric_date, {metric_name} AS value
            FROM analytics.tenant_daily_metrics WHERE tenant_id=:tenant_id AND metric_date BETWEEN :start_date AND :end_date ORDER BY metric_date"""), {"tenant_id": context.tenant_id, "start_date": start_date, "end_date": end_date}).mappings().all()
    return {"metric_name": metric_name, "scope": "seller", "tenant_id": context.tenant_id, "seller_id": context.seller_id, "points": [{"metric_date": str(row["metric_date"]), "value": float(row["value"])} for row in rows]}

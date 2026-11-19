"""Exact, item-grain multi-dimension metric breakdown queries."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import text

from app.data.database import engine
from app.data.report_snapshot import repository_connection


# These expressions are server-owned SQL fragments. Callers can only choose a
# key from this map; dimension names and SQL identifiers are never interpolated
# directly from user input.
_ITEM_DIMENSION_EXPRESSIONS: dict[str, str] = {
    "customer_state": "COALESCE(NULLIF(fo.customer_state, ''), 'UNKNOWN')",
    "customer_city": "COALESCE(NULLIF(fo.customer_city, ''), 'UNKNOWN')",
    "seller_state": "COALESCE(NULLIF(s.seller_state, ''), 'UNKNOWN')",
    "product_category": "COALESCE(NULLIF(p.product_category_name, ''), 'UNKNOWN')",
    "month_position": (
        "CASE WHEN EXTRACT(DAY FROM fo.purchase_date) BETWEEN 1 AND 10 "
        "THEN '月初(1-10日)' WHEN EXTRACT(DAY FROM fo.purchase_date) BETWEEN 11 AND 20 "
        "THEN '月中(11-20日)' ELSE '月末(21-31日)' END"
    ),
    "day_of_week": (
        "CASE EXTRACT(ISODOW FROM fo.purchase_date) "
        "WHEN 1 THEN '周一' WHEN 2 THEN '周二' WHEN 3 THEN '周三' "
        "WHEN 4 THEN '周四' WHEN 5 THEN '周五' WHEN 6 THEN '周六' "
        "ELSE '周日' END"
    ),
    "hour_of_day": "LPAD(EXTRACT(HOUR FROM fo.order_purchase_timestamp)::int::text, 2, '0') || ':00'",
    "quarter_of_year": "'Q' || EXTRACT(QUARTER FROM fo.purchase_date)::int::text",
    "half_of_year": "CASE WHEN EXTRACT(MONTH FROM fo.purchase_date) <= 6 THEN '上半年' ELSE '下半年' END",
}

# Server-owned SQL axes available to the exact aggregate query. Identity
# authorization applies a narrower allow-list before a plan reaches this set.
MULTI_DIMENSION_QUERY_DIMENSIONS = frozenset(_ITEM_DIMENSION_EXPRESSIONS)

_METRIC_VALUE_EXPRESSIONS: dict[str, str] = {
    "gmv": "SUM(item.price + item.freight_value)",
    "freight_value": "SUM(item.freight_value)",
    "order_count": "COUNT(DISTINCT item.order_id)",
    "customer_count": "COUNT(DISTINCT fo.customer_unique_id)",
}

_METRIC_SEMANTICS: dict[str, str] = {
    "gmv": "已交付商品行的商品金额与运费之和；按商品行精确归入每个维度组合。",
    "freight_value": "已交付商品行的运费之和；按商品行精确归入每个维度组合。",
    "order_count": "每个维度组合内去重订单数；同一订单可能出现在多个商品维度组合中。",
    "customer_count": "每个维度组合内去重客户数；同一客户可能出现在多个维度组合中。",
}


def _unsupported(
    metric_name: str,
    dimensions: list[str],
    reason_code: str,
    message: str,
) -> dict[str, Any]:
    return {
        "metric_name": metric_name,
        "dimensions": dimensions,
        "items": [],
        "status": "unsupported",
        "reason_code": reason_code,
        "message": message,
    }


def query_metric_by_dimensions(
    metric_name: str,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    dimensions: list[str] | tuple[str, ...],
    limit: int = 100,
) -> dict[str, Any]:
    """Group an exact item-grain metric by two to four registered dimensions.

    This tool deliberately does not infer a relationship between order-item
    facts and payment-type facts. In particular, it never allocates GMV across
    payment methods. Additive item measures partition item rows; distinct
    order/customer counts are only unique within each returned cell.
    """
    from app.metrics.registry import get_registry

    selected_dimensions = list(dimensions)
    if len(selected_dimensions) < 2 or len(selected_dimensions) > 4:
        raise ValueError("dimensions must contain between 2 and 4 entries")
    if len(selected_dimensions) != len(set(selected_dimensions)):
        raise ValueError("dimensions must be unique")
    if not 1 <= int(limit) <= 100:
        raise ValueError("limit must be between 1 and 100")

    start = _normalize_date(start_date)
    end = _normalize_date(end_date)
    if end < start:
        raise ValueError("end_date must be greater than or equal to start_date")

    registry = get_registry()
    canonical_metric = registry.resolve(metric_name)
    definition = registry.get(canonical_metric)
    if not registry.supports(canonical_metric, "breakdown", "platform"):
        return _unsupported(
            canonical_metric, selected_dimensions, "metric_breakdown_not_registered",
            f"指标 {canonical_metric} 未注册平台维度拆解能力。",
        )

    denied_dimensions = [
        dimension for dimension in selected_dimensions
        if not registry.is_dimension_allowed(canonical_metric, dimension, "platform")
    ]
    if denied_dimensions:
        return _unsupported(
            canonical_metric, selected_dimensions, "metric_dimension_not_registered",
            f"指标 {canonical_metric} 未注册这些维度：{', '.join(denied_dimensions)}。",
        )

    if canonical_metric not in _METRIC_VALUE_EXPRESSIONS:
        return _unsupported(
            canonical_metric, selected_dimensions, "multidimension_metric_not_supported",
            f"多维工具目前没有为指标 {canonical_metric} 定义精确的商品行级聚合口径。",
        )

    # Seller identity requires a controlled seller scope. Payment is at a
    # different fact grain from product rows, so joining it here would create
    # false category/payment associations or duplicate the item amount.
    if any(dimension in {"seller", "seller_id"} for dimension in selected_dimensions):
        return _unsupported(
            canonical_metric, selected_dimensions, "seller_scope_required",
            "按店铺身份分组需要服务端授权的店铺范围；本平台多维工具不接收任意店铺标识。",
        )
    if "payment_type" in selected_dimensions:
        return _unsupported(
            canonical_metric, selected_dimensions, "dimension_grain_mismatch",
            "当前数据只能观察订单与支付方式的关系，不能把支付方式对应到订单内的具体商品行。"
            "为保持精确口径，本工具拒绝将商品金额或运费按支付方式分摊。",
        )

    unsupported_dimensions = [
        dimension for dimension in selected_dimensions
        if dimension not in _ITEM_DIMENSION_EXPRESSIONS
    ]
    if unsupported_dimensions:
        return _unsupported(
            canonical_metric, selected_dimensions, "dimension_grain_mismatch",
            f"这些维度不属于当前精确商品行分析的支持范围：{', '.join(unsupported_dimensions)}。",
        )

    select_dimensions: list[str] = []
    group_dimensions: list[str] = []
    for index, dimension in enumerate(selected_dimensions):
        alias = f"dim_{index}"
        expression = _ITEM_DIMENSION_EXPRESSIONS[dimension]
        select_dimensions.append(f"{expression} AS {alias}")
        group_dimensions.append(expression)

    joins: list[str] = []
    if "product_category" in selected_dimensions:
        joins.append("LEFT JOIN raw.products p ON p.product_id = item.product_id")
    if "seller_state" in selected_dimensions:
        joins.append("LEFT JOIN raw.sellers s ON s.seller_id = item.seller_id")

    value_expression = _METRIC_VALUE_EXPRESSIONS[canonical_metric]
    group_by_sql = ", ".join(group_dimensions)
    select_dimensions_sql = ",\n                ".join(select_dimensions)
    order_by_sql = ", ".join(f"dim_{index}" for index in range(len(selected_dimensions)))
    query = text(
        f"""
        SELECT
            {select_dimensions_sql},
            {value_expression} AS value,
            COUNT(*) AS item_rows,
            COUNT(DISTINCT item.order_id) AS order_count,
            COUNT(DISTINCT fo.customer_unique_id) AS customer_count
        FROM analytics.v_delivered_order_items item
        JOIN analytics.v_delivered_orders fo USING (order_id)
        {' '.join(joins)}
        WHERE fo.purchase_date BETWEEN :start_date AND :end_date
        GROUP BY {group_by_sql}
        ORDER BY value DESC NULLS LAST, {order_by_sql}
        LIMIT :fetch_limit
        """
    )
    with repository_connection(engine) as conn:
        rows = conn.execute(
            query,
            {"start_date": start, "end_date": end, "fetch_limit": int(limit) + 1},
        ).mappings().all()

    truncated = len(rows) > int(limit)
    rows = rows[:int(limit)]
    items: list[dict[str, Any]] = []
    for row in rows:
        values = {
            dimension: str(row[f"dim_{index}"])
            for index, dimension in enumerate(selected_dimensions)
        }
        raw_value = row["value"]
        value: int | float | None
        if raw_value is None:
            value = None
        elif canonical_metric in {"order_count", "customer_count"}:
            value = int(raw_value)
        else:
            value = round(float(raw_value), 2)
        items.append({
            "dimensions": values,
            "value": value,
            "item_rows": int(row["item_rows"]),
            "order_count": int(row["order_count"]),
            "customer_count": int(row["customer_count"]),
        })

    counts_may_overlap = canonical_metric in {"order_count", "customer_count"}
    source_table = "analytics.v_delivered_order_items + analytics.v_delivered_orders"
    if "product_category" in selected_dimensions:
        source_table += " + raw.products"
    if "seller_state" in selected_dimensions:
        source_table += " + raw.sellers"

    return {
        "metric_name": canonical_metric,
        "metric_display_name": definition.display_name,
        "unit": definition.unit,
        "dimensions": selected_dimensions,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "source_grain": "delivered_order_item",
        "metric_semantics": _METRIC_SEMANTICS[canonical_metric],
        "cell_values_additive": canonical_metric in {"gmv", "freight_value"},
        "cell_counts_may_overlap": counts_may_overlap,
        "share": None,
        "items": items,
        "limit": int(limit),
        "truncated": truncated,
        "source_table": source_table,
        "status": "ok" if items else "not_found",
    }


def _normalize_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)

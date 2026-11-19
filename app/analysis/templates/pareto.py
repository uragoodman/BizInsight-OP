"""Pareto 集中度分析 — 累计销售额占比曲线，找到覆盖 80% 的头部实体数。"""

from __future__ import annotations

from datetime import date

from sqlalchemy import text

from app.data.database import engine
from app.analysis.templates.registry import register
from app.analysis.templates.runtime import apply_statement_timeout


@register(
    "pareto_analysis", "Pareto 集中度分析",
    ["seller_id", "start_date", "end_date", "dimension"],
    ["pareto", "集中度", "8020", "80/20", "占比", "累积", "集中"],
    "按商品/品类计算累计销售额占比曲线，确定覆盖80%的头部实体数。",
)
def compute_pareto(
    seller_id: str, start_date: date, end_date: date,
    dimension: str = "product_category", **kwargs,
) -> dict:
    """计算 seller 范围内指定维度的 Pareto 集中度。"""
    if dimension == "product_category":
        name_expr = "COALESCE(v.primary_category, 'UNKNOWN')"
        source = "analytics.v_order_unified v"
        join_clause = ""
        where_clause = "v.seller_id = :seller_id AND v.purchase_date BETWEEN :start_date AND :end_date AND v.is_single_seller_order"
        group_key = name_expr
    else:
        return {"status": "unsupported", "dimension": dimension, "items": [], "error": f"unsupported dimension: {dimension}"}

    with engine.connect() as conn:
        apply_statement_timeout(conn)
        rows = conn.execute(text(f"""
            SELECT {name_expr} AS entity_name,
                   ROUND(SUM(v.price_sum + v.freight_sum)::numeric, 2) AS value
            FROM {source}
            WHERE {where_clause}
            GROUP BY {group_key}
            ORDER BY value DESC
        """), {
            "seller_id": seller_id, "start_date": start_date, "end_date": end_date,
        }).mappings().all()

    items = []
    total = sum(float(r["value"] or 0) for r in rows) if rows else 0
    if total == 0:
        return {"status": "not_found", "dimension": dimension, "items": [], "p80_cutoff": None}

    cumulative = 0.0
    p80_entity_count = 0
    p80_entity_pct = 0.0
    found_p80 = False

    for idx, r in enumerate(rows):
        val = float(r["value"] or 0)
        cumulative += val
        cum_pct = round(cumulative * 100.0 / total, 2)
        items.append({
            "rank": idx + 1,
            "name": str(r["entity_name"]),
            "value": val,
            "cumulative_pct": cum_pct,
        })
        if not found_p80 and cum_pct >= 80.0:
            p80_entity_count = idx + 1
            p80_entity_pct = round((idx + 1) * 100.0 / len(rows), 1) if rows else 0
            found_p80 = True

    if len(items) < 2:
        return {"status": "insufficient_data", "dimension": dimension, "items": items, "p80_cutoff": None}
    return {
        "status": "ok",
        "dimension": dimension,
        "total_value": total,
        "items": items,
        "p80_cutoff": {
            "entity_count": p80_entity_count,
            "entity_pct": p80_entity_pct,
            "value_pct": 80.0,
        } if found_p80 else None,
    }

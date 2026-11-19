"""Cohort 留存分析 — 按首次购买月份分组，追踪后续每月复购率。

Olist 复购率较低（约 3%），因此留存率在 M+1 后会急剧下降。
这本身是有意义的业务发现——大多数买家是单次交易客户。
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import text

from app.data.database import engine
from app.analysis.templates.registry import register
from app.analysis.templates.runtime import apply_statement_timeout

_MIN_CUSTOMERS = 30


@register(
    "cohort_retention", "Cohort 留存分析",
    ["seller_id", "start_date", "end_date", "granularity"],
    ["留存", "retention", "回头", "复购", "repeat", "保留"],
    "按首次购买月份分组，追踪每组后续每月的留存率。",
)
def compute_cohort_retention(
    seller_id: str, start_date: date, end_date: date,
    granularity: str = "month", max_periods: int = 6,
) -> dict:
    """计算 seller 的 Cohort 留存矩阵。"""
    with engine.connect() as conn:
        apply_statement_timeout(conn)
        rows = conn.execute(text(f"""
            WITH first_purchase AS (
                SELECT customer_unique_id,
                       DATE_TRUNC('{granularity}', MIN(purchase_date))::date AS first_period
                FROM analytics.v_order_unified
                WHERE seller_id = :seller_id
                  AND purchase_date BETWEEN :start_date AND :end_date
                GROUP BY customer_unique_id
            ),
            all_periods AS (
                SELECT fp.customer_unique_id, fp.first_period,
                       DATE_TRUNC('{granularity}', v.purchase_date)::date AS activity_period,
                       (EXTRACT(YEAR FROM v.purchase_date) * 12
                        + EXTRACT(MONTH FROM v.purchase_date))
                       - (EXTRACT(YEAR FROM fp.first_period) * 12
                          + EXTRACT(MONTH FROM fp.first_period))
                           AS period_offset
                FROM first_purchase fp
                JOIN analytics.v_order_unified v USING(customer_unique_id)
                WHERE v.seller_id = :seller_id
                  AND v.purchase_date BETWEEN :start_date AND :end_date
            )
            SELECT first_period AS cohort,
                   COUNT(DISTINCT customer_unique_id) AS cohort_size,
                   period_offset,
                   COUNT(DISTINCT customer_unique_id) FILTER (
                       WHERE period_offset >= 0
                   ) AS active_customers
            FROM all_periods
            WHERE period_offset >= 0 AND period_offset < :max_periods
            GROUP BY first_period, period_offset
            ORDER BY first_period, period_offset
        """), {
            "seller_id": seller_id, "start_date": start_date, "end_date": end_date,
            "max_periods": max_periods,
        }).mappings().all()

    if not rows:
        return {"status": "insufficient_data", "cohorts": []}

    # 重组为 cohort → periods 结构
    cohort_map: dict[str, dict] = {}
    for r in rows:
        label = str(r["cohort"])
        size = int(r["cohort_size"])
        if label not in cohort_map:
            cohort_map[label] = {
                "cohort_label": label,
                "customer_count": size,
                "retention": [],
            }
        cohort_map[label]["retention"].append({
            "period": f"M+{r['period_offset']}",
            "rate": round(int(r["active_customers"]) * 100.0 / size, 1),
        })

    cohorts = list(cohort_map.values())
    total = sum(c["customer_count"] for c in cohorts)
    if total < _MIN_CUSTOMERS:
        return {"status": "insufficient_data", "cohorts": cohorts}

    return {
        "status": "ok",
        "cohorts": cohorts,
        "granularity": granularity,
        "total_customers": total,
    }

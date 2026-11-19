"""履约—评价—销量联合分析查询仓库。

该仓库提供跨域指标的一次查询，将履约（准时率/运费率）、评价（评分/差评率）
和销量（GMV/订单量）在同一受控范围内返回。  ``v_order_unified`` 视图通过
分层 CTE 预聚合避免了多表 JOIN 的乘法膨胀，每行对应一次 ``order_id + seller_id``
组合（``is_single_seller_order`` 过滤保证评论归因可追溯）。
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import text

from app.data.database import engine
from app.schemas.explorer import ControlledAnalysisScope


def query_joint_cluster(scope: ControlledAnalysisScope) -> dict[str, object]:
    """在受控范围内查询履约×评价×销量联合指标。

    限定 ``is_single_seller_order = true`` 以避免多卖家订单的评论归因歧义。
    """
    from app.data.repositories.explorer import _scope_params

    params = _scope_params(scope)
    with engine.connect() as connection:
        row = connection.execute(text("""
            SELECT
                COUNT(DISTINCT order_id) AS order_count,
                ROUND(AVG(review_score)::numeric, 2) AS avg_score,
                ROUND(COUNT(*) FILTER (WHERE review_score <= 2) * 100.0
                    / NULLIF(COUNT(*) FILTER (WHERE review_score IS NOT NULL), 0), 2)
                    AS bad_review_rate,
                ROUND(COUNT(*) FILTER (WHERE is_on_time) * 100.0
                    / NULLIF(COUNT(*) FILTER (WHERE is_on_time IS NOT NULL), 0), 2)
                    AS on_time_rate,
                ROUND(SUM(price_sum + freight_sum)::numeric, 2) AS gross_value,
                ROUND((SUM(freight_sum) * 100.0
                    / NULLIF(SUM(price_sum + freight_sum), 0))::numeric, 2)
                    AS freight_rate
            FROM analytics.v_order_unified
            WHERE seller_id = :seller_id
              AND purchase_date BETWEEN :start_date AND :end_date
              AND is_single_seller_order
        """), params).mappings().one()
    return {
        "metric_name": "joint_cluster",
        "order_count": int(row["order_count"] or 0),
        "avg_score": float(row["avg_score"] or 0),
        "bad_review_rate": float(row["bad_review_rate"] or 0),
        "on_time_rate": float(row["on_time_rate"] or 0),
        "gross_value": float(row["gross_value"] or 0),
        "freight_rate": float(row["freight_rate"] or 0),
        "sample_size": int(row["order_count"] or 0),
        "source_table": "analytics.v_order_unified",
        "status": "ok" if row["order_count"] else "not_found",
    }


def compare_joint_cluster(
    scope: ControlledAnalysisScope,
    current_start: date, current_end: date,
    baseline_start: date, baseline_end: date,
) -> dict[str, object]:
    """两期联合指标对比，限定 ``is_single_seller_order=true``。"""
    current_scope = scope.model_copy(update={
        "default_start_date": current_start, "default_end_date": current_end,
    })
    baseline_scope = scope.model_copy(update={
        "default_start_date": baseline_start, "default_end_date": baseline_end,
    })
    current = query_joint_cluster(current_scope)
    baseline = query_joint_cluster(baseline_scope)

    metrics = ["avg_score", "bad_review_rate", "on_time_rate", "gross_value", "freight_rate"]
    comparisons = {}
    for m in metrics:
        cv = current.get(m)
        bv = baseline.get(m)
        dv = round(float(cv) - float(bv), 2) if cv is not None and bv is not None else None
        dr = round(dv * 100.0 / float(bv), 2) if dv is not None and bv and float(bv) != 0 else None
        comparisons[m] = {
            "current_value": cv, "baseline_value": bv,
            "delta_value": dv, "delta_ratio": dr,
        }
    return {
        "metric_name": "joint_cluster",
        "current": current,
        "baseline": baseline,
        "comparisons": comparisons,
        "current_period": f"{current_start.isoformat()} 至 {current_end.isoformat()}",
        "baseline_period": f"{baseline_start.isoformat()} 至 {baseline_end.isoformat()}",
        "sample_size": current.get("order_count", 0),
        "status": "ok" if current.get("order_count") else "not_found",
    }

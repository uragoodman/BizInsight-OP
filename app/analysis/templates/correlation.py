"""延迟-低评分关联分析 — 延迟订单的差评率 vs 准时订单的差评率。

计算 Lift = 延迟订单差评率 / 准时订单差评率，以及卡方独立性检验。
"""

from __future__ import annotations

import math
from datetime import date

from sqlalchemy import text

from app.data.database import engine
from app.analysis.templates.registry import register
from app.analysis.templates.runtime import apply_statement_timeout
from app.analysis.association_stats import summarize_binary_association

_MIN_SAMPLE = 50


def _chi_square(a: int, b: int, c: int, d: int) -> dict:
    """2x2 列联表卡方检验。返回 {statistic, p_value}。"""
    total = a + b + c + d
    if total == 0:
        return {"statistic": None, "p_value": None}
    expected_a = (a + b) * (a + c) / total
    expected_b = (a + b) * (b + d) / total
    expected_c = (c + d) * (a + c) / total
    expected_d = (c + d) * (b + d) / total

    def _cell(obs: int, exp: float) -> float:
        return 0.0 if exp == 0 else (obs - exp) ** 2 / exp

    chi2 = _cell(a, expected_a) + _cell(b, expected_b) + _cell(c, expected_c) + _cell(d, expected_d)
    # 自由度 = 1 的卡方分布近似 p 值
    p_value = _chi2_pval(chi2) if chi2 > 0 else 1.0
    return {"statistic": round(chi2, 4), "p_value": round(p_value, 6)}


def _chi2_pval(chi2: float) -> float:
    """自由度=1 的卡方分布右尾概率近似。"""
    # For one degree of freedom, chi-square is the square of a standard
    # normal variate, so its survival function is erfc(sqrt(x / 2)).
    return math.erfc(math.sqrt(chi2 / 2)) if chi2 > 0 else 1.0


@register(
    "delay_review_correlation", "延迟-评分关联分析",
    ["seller_id", "start_date", "end_date"],
    ["延迟", "差评", "关联", "相关", "correlation", "准时", "送达"],
    "计算延迟送达与低评分之间的统计关联强度。",
)
def compute_delay_review_correlation(
    seller_id: str, start_date: date, end_date: date, **kwargs,
) -> dict:
    """计算延迟送达与差评率的关联。"""
    with engine.connect() as conn:
        apply_statement_timeout(conn)
        row = conn.execute(text("""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE is_on_time AND review_score > 2) AS on_time_good,
                COUNT(*) FILTER (WHERE is_on_time AND review_score <= 2) AS on_time_bad,
                COUNT(*) FILTER (WHERE NOT is_on_time AND review_score > 2) AS late_good,
                COUNT(*) FILTER (WHERE NOT is_on_time AND review_score <= 2) AS late_bad,
                AVG(review_score) FILTER (WHERE is_on_time) AS on_time_avg_score,
                AVG(review_score) FILTER (WHERE NOT is_on_time) AS late_avg_score
            FROM analytics.v_order_unified
            WHERE seller_id = :seller_id
              AND purchase_date BETWEEN :start_date AND :end_date
              AND is_single_seller_order
              AND review_score IS NOT NULL
              AND is_on_time IS NOT NULL
        """), {"seller_id": seller_id, "start_date": start_date, "end_date": end_date}).mappings().one()

    total = int(row["total"] or 0)
    a = int(row["on_time_good"] or 0)
    b = int(row["on_time_bad"] or 0)
    c = int(row["late_good"] or 0)
    d = int(row["late_bad"] or 0)
    on_time_total = a + b
    late_total = c + d

    on_time_bad_rate = round(b * 100.0 / on_time_total, 1) if on_time_total else 0.0
    on_time_avg_score = round(float(row["on_time_avg_score"] or 0), 2)
    late_bad_rate = round(d * 100.0 / late_total, 1) if late_total else 0.0
    late_avg_score = round(float(row["late_avg_score"] or 0), 2)

    lift = round(late_bad_rate / on_time_bad_rate, 2) if on_time_bad_rate > 0 else None
    chi2 = _chi_square(a, b, c, d)
    association = summarize_binary_association(
        exposed_events=d, exposed_total=late_total,
        comparison_events=b, comparison_total=on_time_total,
        min_total=_MIN_SAMPLE,
    )
    delayed_ci = association["exposed"]["rate_ci_95"]
    on_time_ci = association["comparison"]["rate_ci_95"]
    risk_difference_ci = association["risk_difference_ci_95"]

    return {
        "status": association["status"],
        "sample_size": total,
        "overall": {
            "order_count": total,
            "bad_review_order_count": d + b,
            "bad_review_rate": round((d + b) * 100.0 / total, 1) if total else None,
        },
        "delayed": {
            "order_count": late_total,
            "bad_review_rate": late_bad_rate,
            "bad_review_rate_ci_95_pct": [round(float(value) * 100.0, 1) for value in delayed_ci] if delayed_ci else None,
            "avg_score": late_avg_score,
        },
        "on_time": {
            "order_count": on_time_total,
            "bad_review_rate": on_time_bad_rate,
            "bad_review_rate_ci_95_pct": [round(float(value) * 100.0, 1) for value in on_time_ci] if on_time_ci else None,
            "avg_score": on_time_avg_score,
        },
        "lift": lift,
        "risk_difference_percentage_points": round(float(association["risk_difference"]) * 100.0, 2) if association["risk_difference"] is not None else None,
        "risk_difference_ci_95_percentage_points": [round(float(value) * 100.0, 2) for value in risk_difference_ci] if risk_difference_ci else None,
        "association_method": association["method_id"],
        "multiple_comparison_adjustment": "not_applicable_single_pre_registered_pair",
        "chi_square": chi2,
        "interpretation_limit": "关联不代表因果；区间描述抽样不确定性，不代表所有偏差均已消除。",
    }

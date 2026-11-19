"""跨域风险评分：差评率×延迟率×运费率合成 0-100 风险分。

数据源：analytics.v_order_unified（M2 已建成）。
过滤：is_single_seller_order = true 避免多卖家评论归因歧义。
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import text

from app.data.database import engine

# 维度权重
_BAD_REVIEW_WEIGHT = 0.4
_DELAY_WEIGHT = 0.3
_FREIGHT_WEIGHT = 0.3

# 风险等级阈值（>= threshold → level）
_LEVEL_THRESHOLDS: list[tuple[float, str]] = [
    (60, "high"),
    (30, "medium"),
    (0, "low"),
]

_MIN_SAMPLE = 5


def _risk_level(score: float) -> str:
    for threshold, level in _LEVEL_THRESHOLDS:
        if score >= threshold:
            return level
    return "low"


def query_risk_score(seller_id: str, start_date: date, end_date: date) -> dict[str, object]:
    """查询指定 seller+ 日期范围的跨域风险评分。

    返回:
      - score: 0-100 复合风险分
      - level: "low" | "medium" | "high"
      - dimensions: {bad_review_rate, delay_rate, freight_rate}
      - sample_size: 参与计算的订单数
      - status: "ok" | "insufficient_data"
    """
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT
                COUNT(DISTINCT order_id) AS sample_size,
                ROUND(COUNT(*) FILTER (WHERE review_score <= 2) * 100.0
                    / NULLIF(COUNT(*) FILTER (WHERE review_score IS NOT NULL), 0), 1)
                    AS bad_review_rate,
                ROUND(COUNT(*) FILTER (WHERE NOT is_on_time) * 100.0
                    / NULLIF(COUNT(*) FILTER (WHERE is_on_time IS NOT NULL), 0), 1)
                    AS delay_rate,
                ROUND((SUM(freight_sum) * 100.0
                    / NULLIF(SUM(price_sum + freight_sum), 0))::numeric, 1)
                    AS freight_rate
            FROM analytics.v_order_unified
            WHERE seller_id = :seller_id
              AND purchase_date BETWEEN :start_date AND :end_date
              AND is_single_seller_order
        """), {"seller_id": seller_id, "start_date": start_date, "end_date": end_date}).mappings().one()

    sample = int(row["sample_size"] or 0)
    if sample < _MIN_SAMPLE:
        return {
            "status": "insufficient_data",
            "score": 0, "level": "low",
            "sample_size": sample,
            "dimensions": {},
        }

    br = float(row["bad_review_rate"] or 0)
    dr = float(row["delay_rate"] or 0)
    fr = float(row["freight_rate"] or 0)

    score = round(br * _BAD_REVIEW_WEIGHT + dr * _DELAY_WEIGHT + fr * _FREIGHT_WEIGHT, 1)
    score = max(0.0, min(100.0, score))

    return {
        "status": "ok",
        "score": score,
        "level": _risk_level(score),
        "sample_size": sample,
        "dimensions": {
            "bad_review_rate": br,
            "delay_rate": dr,
            "freight_rate": fr,
        },
    }

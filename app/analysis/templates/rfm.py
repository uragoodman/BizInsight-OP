"""RFM 分群 — 按 Recency / Frequency / Monetary 三维归入客户群组。

Olist 复购率低（约 3%），因此大多数客户将落入"低频"分箱。
分箱在 seller 内部进行（NTILE），不跨 seller 比较。
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import text

from app.data.database import engine
from app.analysis.templates.registry import register
from app.analysis.templates.runtime import apply_statement_timeout

_MIN_CUSTOMERS = 30

_SEGMENT_MAP: dict[tuple[int, int, int], str] = {
    (5, 5, 5): "高价值核心",
    (5, 4, 5): "高价值核心",
    (5, 5, 4): "高价值核心",
    (4, 5, 5): "高价值核心",
    (5, 4, 4): "重要价值",
    (4, 5, 4): "重要发展",
    (4, 4, 5): "重要价值",
    (4, 4, 4): "重要发展",
    (3, 4, 4): "一般价值",
    (3, 3, 4): "一般发展",
    (2, 3, 3): "一般挽留",
    (1, 2, 2): "一般挽留",
    (1, 1, 2): "流失预警",
    (1, 1, 1): "已流失",
}

_RFM_LABELS = [(5, "高"), (4, "中高"), (3, "中"), (2, "低"), (1, "低")]


def _segment(r: int, f: int, m: int) -> str:
    return _SEGMENT_MAP.get((r, f, m), "其他")


@register(
    "rfm_segmentation", "RFM 客户分群",
    ["seller_id", "end_date"],
    ["rfm", "客户分群", "客户价值", "客户分层", "segment", "分层"],
    "按最近购买时间/频率/金额三维评分，归入高价值/需挽留/流失等群组。",
)
def compute_rfm(seller_id: str, end_date: date, **kwargs) -> dict:
    """计算 seller 的 RFM 客户分群。"""
    with engine.connect() as conn:
        apply_statement_timeout(conn)
        raw = conn.execute(text("""
            WITH customer_stats AS (
                SELECT customer_unique_id,
                       CAST(:end_date AS date) - MAX(purchase_date) AS recency,
                       COUNT(DISTINCT order_id) AS frequency,
                       SUM(price_sum + freight_sum) AS monetary
                FROM analytics.v_order_unified
                WHERE seller_id = :seller_id
                  AND purchase_date <= :end_date
                GROUP BY customer_unique_id
            ),
            scored AS (
                SELECT customer_unique_id, recency, frequency, monetary,
                       NTILE(5) OVER (ORDER BY recency DESC) AS r_score,
                       NTILE(5) OVER (ORDER BY frequency ASC) AS f_score,
                       NTILE(5) OVER (ORDER BY monetary ASC) AS m_score
                FROM customer_stats
            )
            SELECT r_score, f_score, m_score,
                   COUNT(*) AS customer_count,
                   ROUND(AVG(monetary)::numeric, 2) AS avg_value
            FROM scored
            GROUP BY r_score, f_score, m_score
            ORDER BY r_score DESC, f_score DESC, m_score DESC
        """), {"seller_id": seller_id, "end_date": end_date}).mappings().all()

    if not raw:
        return {"status": "insufficient_data", "segments": [], "total_customers": 0}

    # 按 segment 聚合
    seg_map: dict[str, dict] = {}
    for r in raw:
        seg = _segment(int(r["r_score"]), int(r["f_score"]), int(r["m_score"]))
        cnt = int(r["customer_count"])
        val = float(r["avg_value"])
        if seg not in seg_map:
            seg_map[seg] = {"segment_name": seg, "customer_count": 0, "pct": 0.0, "avg_value": 0.0}
        previous_count = seg_map[seg]["customer_count"]
        previous_avg = seg_map[seg]["avg_value"]
        seg_map[seg]["avg_value"] = round(
            (previous_avg * previous_count + val * cnt) / (previous_count + cnt), 2,
        ) if previous_count + cnt > 0 else 0.0
        seg_map[seg]["customer_count"] = previous_count + cnt

    total = sum(s["customer_count"] for s in seg_map.values())
    if total < _MIN_CUSTOMERS:
        return {"status": "insufficient_data", "segments": []}

    segments = sorted(seg_map.values(), key=lambda x: -x["customer_count"])
    for s in segments:
        s["pct"] = round(s["customer_count"] * 100.0 / total, 1)

    return {
        "status": "ok",
        "segments": segments,
        "total_customers": total,
    }

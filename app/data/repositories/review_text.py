"""评论文本受控查询：聚合统计 + 确定性抽样 + PII 脱敏。

所有输出默认脱敏，不包含客户标识符。text 字段截断至 120 字符，
且经过 PII 正则遮蔽（邮箱、电话、URL、疑似订单引用）。
order_id 替换为 trace 内稳定的短 hash（SHA256 前 12 位）。
"""

from __future__ import annotations

import hashlib
import re
from datetime import date

from sqlalchemy import text

from app.data.database import engine
from app.schemas.explorer import ControlledAnalysisScope

# 脱敏正则：替换为占位符，保留阅读连续性
_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), " [EMAIL] "),
    (re.compile(r"https?://\S+"), " [URL] "),
    (re.compile(r"(?<!\w)(?:\+?55)?\s*\(?\d{2,3}\)?\s*\d{4,5}-?\d{4}(?!\w)"), " [PHONE] "),
    # 疑似订单引用 — 32 位 hex（Olist order_id 格式）
    (re.compile(r"\b[a-f0-9]{32}\b", re.I), " [ORDER_REF] "),
]


def _sanitize_comment(text: str) -> str:
    """对评论正文做 PII 脱敏，返回替换后的文本。"""
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _short_hash(order_id: str) -> str:
    """返回 SHA256 前 12 位作为 trace 内稳定的短标识。"""
    return hashlib.sha256(order_id.encode()).hexdigest()[:12]


def _scope_params(scope: ControlledAnalysisScope) -> dict[str, object]:
    return {
        "seller_id": scope.seller_id,
        "start_date": scope.default_start_date,
        "end_date": scope.default_end_date,
    }


def query_review_comment_stats(seller_id: str, start_date: date, end_date: date) -> dict[str, object]:
    """聚合统计：含文本率、平均长度、差评含文本率。

    从 ``v_order_unified`` 取数，该视图的 review 已做 DISTINCT ON (order_id) 去重。
    """
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT
                COUNT(*) AS total_orders,
                COUNT(*) FILTER (WHERE review_comment_message IS NOT NULL
                    AND TRIM(review_comment_message) != '') AS with_comment,
                ROUND((AVG(LENGTH(review_comment_message))
                    FILTER (WHERE review_comment_message IS NOT NULL))::numeric, 1) AS avg_length,
                COUNT(*) FILTER (WHERE review_score <= 2
                    AND review_comment_message IS NOT NULL
                    AND TRIM(review_comment_message) != '') AS bad_with_comment
            FROM analytics.v_order_unified
            WHERE seller_id = :seller_id
              AND purchase_date BETWEEN :start_date AND :end_date
        """), {"seller_id": seller_id, "start_date": start_date, "end_date": end_date}).mappings().one()
    total = int(row["total_orders"] or 0)
    with_comm = int(row["with_comment"] or 0)
    return {
        "total_orders": total,
        "with_comment_count": with_comm,
        "comment_rate": round(with_comm * 100.0 / total, 1) if total else 0.0,
        "avg_comment_length": float(row["avg_length"] or 0),
        "bad_with_comment_count": int(row["bad_with_comment"] or 0),
        "source_table": "analytics.v_order_unified",
        "status": "ok" if total else "not_found",
    }


def query_review_samples(
    seller_id: str, start_date: date, end_date: date,
    max_score: int = 5, min_score: int = 1, limit: int = 5,
) -> list[dict[str, object]]:
    """确定性抽样，脱敏输出。

    排序：按 review_score 升序（差评优先）、comment 长度降序 → 同请求结果可重现。
    输出不包含 customer 字段。comment 截断至 120 字符。
    """
    params: dict[str, object] = {
        "seller_id": seller_id, "start_date": start_date, "end_date": end_date,
        "min_score": max(1, min_score), "max_score": min(5, max_score),
        "limit": max(1, min(limit, 20)),
    }
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT review_comment_message,
                   review_comment_title,
                   review_score,
                   order_id, purchase_date
            FROM analytics.v_order_unified
            WHERE seller_id = :seller_id
              AND purchase_date BETWEEN :start_date AND :end_date
              AND review_score BETWEEN :min_score AND :max_score
              AND review_comment_message IS NOT NULL
              AND TRIM(review_comment_message) != ''
            ORDER BY review_score ASC, LENGTH(review_comment_message) DESC
            LIMIT :limit
        """), params).mappings().all()
    samples = []
    for r in rows:
        msg = (r["review_comment_message"] or "")[:120]
        sanitized = _sanitize_comment(msg)
        samples.append({
            "order_id_abbr": _short_hash(str(r["order_id"])),
            "score": int(r["review_score"]),
            "comment_excerpt": sanitized,
            "has_title": bool(r["review_comment_title"] and r["review_comment_title"].strip()),
            "purchase_date": str(r["purchase_date"]),
        })
    return samples

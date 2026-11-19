"""Collect real, reproducible quality samples through the production workflow.

This is intentionally a bounded local-operations script. It does not fabricate
quality metrics: every sample is produced by the same chat workflow and the
configured model, then persisted by the configured trace backend.
"""

from __future__ import annotations

import os
from datetime import date

from sqlalchemy import text

from app.data.database import engine
from app.graph.workflow import run_chat_workflow
from app.tenancy import TenantContext


PROMPTS = [
    "分析当前平台 GMV 趋势，并说明最明显的变化。",
    "比较订单数和客单价的近期变化，给出有证据的结论。",
    "分析销售额下降可能涉及哪些维度，并说明数据限制。",
    "查看最近经营指标是否存在异常，输出简洁结论。",
    "生成一份经营数据概览，包含 GMV、订单数和客单价。",
    "分析订单履约表现的变化趋势，并指出需要复核的地方。",
    "比较当前周期与上一周期的核心指标差异。",
    "针对当前经营数据提出两条可验证的分析建议。",
    "分析客户数量与复购表现的变化。",
    "检查核心指标是否存在明显的异常波动。",
]


def _sample_sellers(limit: int = 8, offset: int = 0) -> list[str]:
    """Choose active sellers from real delivered-order data, not fixtures."""
    with engine.connect() as connection:
        rows = connection.execute(text("""
            SELECT oi.seller_id
            FROM analytics.v_delivered_order_items oi
            JOIN analytics.v_delivered_orders fo ON fo.order_id = oi.order_id
            GROUP BY oi.seller_id
            HAVING COUNT(DISTINCT oi.order_id) >= 500
            ORDER BY COUNT(DISTINCT oi.order_id) DESC, oi.seller_id
            LIMIT :limit OFFSET :offset
        """), {"limit": limit, "offset": offset}).scalars().all()
    return [str(value) for value in rows]


def main() -> int:
    os.environ.setdefault("TRACE_BACKEND", "postgres")
    seller_limit = int(os.environ.get("QUALITY_SELLER_LIMIT", "8"))
    seller_offset = int(os.environ.get("QUALITY_SELLER_OFFSET", "0"))
    sellers = _sample_sellers(seller_limit, seller_offset)
    if not sellers:
        print("No active sellers found")
        return 1
    jobs = [(seller_id, prompt) for seller_id in sellers for prompt in PROMPTS[:2]]
    passed = 0
    for index, (seller_id, prompt) in enumerate(jobs, start=1):
        context = TenantContext(
            tenant_id="platform",
            user_id="usr_0de942d7beaf491288aa5798060fe895",
            role="admin",
            scope_type="seller",
            seller_id=seller_id,
        )
        explorer_context = {
            "seller_id": seller_id,
            "metric_name": "gmv" if index % 2 else "order_count",
            "start_date": date(2017, 1, 1),
            "end_date": date(2018, 8, 31),
            "granularity": "month",
        }
        try:
            result = run_chat_workflow(
                prompt,
                thread_id=f"quality-sample-{seller_id[:8]}-{index}",
                tenant_context=context,
                explorer_context=explorer_context,
                use_current_explorer_context=True,
                memory_enabled=False,
            )
            verdict = result.get("critic_verdict") or "missing"
            timings = len(result.get("node_timings") or [])
            print(f"[{index}/{len(jobs)}] seller={seller_id} critic={verdict} node_timings={timings}")
            passed += 1
        except Exception as exc:  # noqa: BLE001 - keep collecting bounded samples
            print(f"[{index}/{len(jobs)}] seller={seller_id} failed={type(exc).__name__}: {exc}")
    print(f"sellers={len(sellers)} collected={passed}/{len(jobs)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

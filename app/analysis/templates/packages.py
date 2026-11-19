"""Six fixed, evidence-oriented ecommerce analysis packages.

Each package uses only static SQL against the governed unified-order view.
They are registered in the same whitelist as the Phase 9 templates.
"""
from __future__ import annotations

from datetime import date
from sqlalchemy import text

from app.analysis.templates.registry import register
from app.analysis.templates.runtime import apply_statement_timeout
from app.analysis.templates.cohort import compute_cohort_retention
from app.analysis.templates.rfm import compute_rfm
from app.analysis.templates.pareto import compute_pareto
from app.analysis.templates.correlation import compute_delay_review_correlation
from app.data.database import engine


def _query(sql: str, params: dict) -> list[dict]:
    with engine.connect() as conn:
        apply_statement_timeout(conn)
        return [dict(row) for row in conn.execute(text(sql), params).mappings().all()]


@register("growth_diagnostics", "GMV 异动诊断包", ["seller_id", "start_date", "end_date"],
          ["经营诊断", "gmv异动", "gmv 诊断", "收入诊断", "增长诊断", "销售诊断", "growth diagnostics", "gmv diagnostics"],
          "按月追踪 GMV、订单、客单价，并给出品类贡献。",
          )
def compute_growth_diagnostics(seller_id: str, start_date: date, end_date: date, **_: object) -> dict:
    monthly = _query("""SELECT DATE_TRUNC('month',purchase_date)::date AS period,
          ROUND(SUM(price_sum+freight_sum)::numeric,2) AS gmv, COUNT(DISTINCT order_id) AS orders,
          ROUND(AVG(price_sum+freight_sum)::numeric,2) AS aov
        FROM analytics.v_order_unified WHERE seller_id=:seller_id AND purchase_date BETWEEN :start AND :end
          AND is_single_seller_order GROUP BY 1 ORDER BY 1""", {"seller_id":seller_id,"start":start_date,"end":end_date})
    categories = _query("""SELECT COALESCE(primary_category,'UNKNOWN') AS category,
          ROUND(SUM(price_sum+freight_sum)::numeric,2) AS gmv, COUNT(DISTINCT order_id) AS orders
        FROM analytics.v_order_unified WHERE seller_id=:seller_id AND purchase_date BETWEEN :start AND :end
          AND is_single_seller_order GROUP BY 1 ORDER BY gmv DESC LIMIT 10""", {"seller_id":seller_id,"start":start_date,"end":end_date})
    summary = "未找到锁定范围内的成交订单。"
    if monthly:
        first, last = monthly[0], monthly[-1]
        summary = f"覆盖 {len(monthly)} 个自然月：GMV 从 {first['gmv']} 变化至 {last['gmv']}。"
        if categories:
            summary += f"GMV 最高的品类为 {categories[0]['category']}（{categories[0]['gmv']}）。"
    return {"status":"ok" if monthly else "insufficient_data", "summary":summary, "monthly":monthly, "top_categories":categories}


@register("fulfillment_funnel", "订单履约漏斗包", ["seller_id", "start_date", "end_date"],
          ["履约漏斗", "订单漏斗", "订单状态", "发货漏斗", "签收漏斗", "funnel", "fulfillment funnel", "order funnel"],
          "统计订单状态、准时率和关键履约时长，定位订单损失环节。")
def compute_fulfillment_funnel(seller_id: str, start_date: date, end_date: date, **_: object) -> dict:
    stages = _query("""SELECT order_status AS stage, COUNT(DISTINCT order_id) AS orders
        FROM analytics.v_order_unified WHERE seller_id=:seller_id AND purchase_date BETWEEN :start AND :end
        GROUP BY 1 ORDER BY orders DESC""", {"seller_id":seller_id,"start":start_date,"end":end_date})
    quality = _query("""SELECT COUNT(DISTINCT order_id) AS orders,
          ROUND(AVG(approval_lead_hours)::numeric,2) AS approval_hours,
          ROUND(AVG(dispatch_lead_hours)::numeric,2) AS dispatch_hours,
          ROUND(AVG(delivery_transit_hours)::numeric,2) AS transit_hours,
          ROUND(100.0*COUNT(DISTINCT order_id) FILTER(WHERE is_on_time=false)/NULLIF(COUNT(DISTINCT order_id) FILTER(WHERE is_on_time IS NOT NULL),0),1) AS delay_rate
        FROM analytics.v_order_unified WHERE seller_id=:seller_id AND purchase_date BETWEEN :start AND :end""", {"seller_id":seller_id,"start":start_date,"end":end_date})
    summary = "未找到锁定范围内的订单状态数据。"
    if stages:
        top = stages[0]
        summary = f"共识别 {sum(item['orders'] for item in stages)} 笔订单状态记录，数量最多的状态为 {top['stage']}（{top['orders']} 笔）。"
    return {"status":"ok" if stages else "insufficient_data", "summary":summary, "stages":stages, "quality":quality[0] if quality else {}}


@register("customer_lifecycle", "客户价值与留存包", ["seller_id", "start_date", "end_date"],
          ["客户生命周期", "客户留存", "客户价值", "复购分析", "客户健康", "customer lifecycle", "customer retention"],
          "组合 Cohort 留存、RFM 客户价值分群与复购间隔。",
          )
def compute_customer_lifecycle(seller_id: str, start_date: date, end_date: date, **_: object) -> dict:
    intervals = _query("""WITH purchases AS (SELECT customer_unique_id,purchase_date,
          LAG(purchase_date) OVER(PARTITION BY customer_unique_id ORDER BY purchase_date) AS previous_date
          FROM analytics.v_order_unified WHERE seller_id=:seller_id AND purchase_date BETWEEN :start AND :end)
        SELECT COUNT(*) FILTER(WHERE previous_date IS NOT NULL) AS repeat_orders,
          ROUND(AVG(purchase_date-previous_date) FILTER(WHERE previous_date IS NOT NULL),1) AS avg_repeat_days
        FROM purchases""", {"seller_id":seller_id,"start":start_date,"end":end_date})
    cohort = compute_cohort_retention(seller_id, start_date, end_date)
    rfm = compute_rfm(seller_id, end_date)
    status = "ok" if cohort.get("status")=="ok" or rfm.get("status")=="ok" else "insufficient_data"
    repeat = intervals[0] if intervals else {}
    summary = f"复购订单 {repeat.get('repeat_orders', 0)} 笔，平均复购间隔 {repeat.get('avg_repeat_days', '无')} 天。"
    return {"status":status, "summary":summary, "cohort":cohort,"rfm":rfm,"repeat":repeat}


@register("category_portfolio", "品类经营组合包", ["seller_id", "start_date", "end_date"],
          ["品类经营", "品类组合", "品类诊断", "商品组合", "品类表现", "category portfolio", "category diagnostics"],
          "结合品类集中度、客单价、运费占比与评分评估品类经营结构。")
def compute_category_portfolio(seller_id: str, start_date: date, end_date: date, **_: object) -> dict:
    metrics = _query("""SELECT COALESCE(primary_category,'UNKNOWN') AS category,
          ROUND(SUM(price_sum+freight_sum)::numeric,2) AS gmv, COUNT(DISTINCT order_id) AS orders,
          ROUND(AVG(price_sum+freight_sum)::numeric,2) AS aov,
          ROUND(100::numeric*SUM(freight_sum)::numeric/NULLIF(SUM(price_sum+freight_sum)::numeric,0),1) AS freight_rate,
          ROUND(AVG(review_score)::numeric,2) AS review_score
        FROM analytics.v_order_unified WHERE seller_id=:seller_id AND purchase_date BETWEEN :start AND :end
          AND is_single_seller_order GROUP BY 1 ORDER BY gmv DESC LIMIT 15""", {"seller_id":seller_id,"start":start_date,"end":end_date})
    summary = "至少需要两个非零品类，才能形成品类组合结论。"
    if metrics:
        summary = f"共 {len(metrics)} 个品类有成交，GMV 最高的是 {metrics[0]['category']}（{metrics[0]['gmv']}）。"
    return {"status":"ok" if len(metrics)>=2 else "insufficient_data", "summary":summary, "pareto":compute_pareto(seller_id,start_date,end_date),"categories":metrics}


@register("logistics_experience", "物流体验诊断包", ["seller_id", "start_date", "end_date"],
          ["物流体验", "配送体验", "履约体验", "物流诊断", "延迟诊断", "logistics experience", "logistics diagnostics"],
          "分析延迟、配送时长、评分和客户州的履约体验差异。")
def compute_logistics_experience(seller_id: str, start_date: date, end_date: date, **_: object) -> dict:
    states = _query("""SELECT customer_state AS state, COUNT(DISTINCT order_id) AS orders,
          ROUND(100.0*COUNT(DISTINCT order_id) FILTER(WHERE is_on_time=false)/NULLIF(COUNT(DISTINCT order_id) FILTER(WHERE is_on_time IS NOT NULL),0),1) AS delay_rate,
          ROUND(AVG(review_score)::numeric,2) AS review_score
        FROM analytics.v_order_unified WHERE seller_id=:seller_id AND purchase_date BETWEEN :start AND :end
          AND is_single_seller_order GROUP BY 1 HAVING COUNT(DISTINCT order_id)>=10 ORDER BY delay_rate DESC NULLS LAST LIMIT 10""", {"seller_id":seller_id,"start":start_date,"end":end_date})
    corr = compute_delay_review_correlation(seller_id,start_date,end_date)
    status = "ok" if states or corr.get("status")=="ok" else "insufficient_data"
    summary = "未形成足够的地区履约体验样本。"
    if states:
        worst = states[0]
        summary = f"至少 10 单的地区中，延迟率最高的是 {worst['state']}（{worst.get('delay_rate')}%）。"
    return {"status":status, "summary":summary, "correlation":corr,"states":states}


@register("payment_cancellation", "支付与取消诊断包", ["seller_id", "start_date", "end_date"],
          ["支付取消", "支付分析", "取消分析", "分期分析", "支付方式", "payment cancellation", "payment analysis"],
          "按支付方式和是否分期分析订单量、取消率及客单价。")
def compute_payment_cancellation(seller_id: str, start_date: date, end_date: date, **_: object) -> dict:
    rows = _query("""SELECT COALESCE(payment_types,'unknown') AS payment_type, COALESCE(has_installments,false) AS installments,
          COUNT(DISTINCT order_id) AS orders, ROUND(100.0*COUNT(DISTINCT order_id) FILTER(WHERE order_status IN ('canceled','unavailable'))/NULLIF(COUNT(DISTINCT order_id),0),1) AS cancellation_rate,
          ROUND(AVG(price_sum+freight_sum)::numeric,2) AS aov
        FROM analytics.v_order_unified WHERE seller_id=:seller_id AND purchase_date BETWEEN :start AND :end
        GROUP BY 1,2 HAVING COUNT(DISTINCT order_id)>=5 ORDER BY orders DESC""", {"seller_id":seller_id,"start":start_date,"end":end_date})
    summary = "没有满足最少 5 笔订单的支付分组。"
    if rows:
        highest = max(rows, key=lambda item: item.get("cancellation_rate") or 0)
        summary = f"共 {len(rows)} 个支付分组满足样本要求，最高取消率为 {highest.get('cancellation_rate')}%。"
    return {"status":"ok" if rows else "insufficient_data", "summary":summary, "payment_segments":rows}

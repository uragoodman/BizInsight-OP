"""受控的 Olist 店铺扩展指标。

所有 SQL 模板均在服务端固定；调用方只能传入 ``ControlledAnalysisScope``，
不能自行指定店铺、表名或表达式。

分类说明：
  - 基础聚合指标（_DEFINITIONS 字典）：单表 JOIN + 聚合表达式，query/trend 通用
  - 集中度/新品指标（_concentration_*）：需子查询计算商品/品类占比
  - 客户生命周期指标（_customer_lifecycle_*）：需识别首购/复购客户
  - 评论评价指标（_review_*）：关联 order_reviews 计算评分和差评率
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import text

from app.data.database import engine
from app.data.rls import seller_query_transaction
from app.schemas.explorer import ControlledAnalysisScope


_DEFINITIONS: dict[str, tuple[str, str, str, str, str]] = {
    # metric_id: (FROM source, seller column, purchase-date column, expression, lineage)
    # 以下为通用聚合指标，所有店铺维度的基础 KPI 定义在此
    "seller_item_gross_value": (
        "analytics.v_delivered_order_items oi JOIN analytics.v_delivered_orders fo USING(order_id)",
        "oi.seller_id", "fo.purchase_date", "COALESCE(SUM(oi.price), 0)",
        "仅已交付商品行的商品金额，不含运费",
    ),
    "seller_avg_item_price": (
        "analytics.v_delivered_order_items oi JOIN analytics.v_delivered_orders fo USING(order_id)",
        "oi.seller_id", "fo.purchase_date", "AVG(oi.price)",
        "仅已交付商品行的平均商品售价，不含运费",
    ),
    "seller_avg_freight_per_order": (
        "analytics.v_delivered_order_items oi JOIN analytics.v_delivered_orders fo USING(order_id)",
        "oi.seller_id", "fo.purchase_date", "SUM(oi.freight_value) / NULLIF(COUNT(DISTINCT oi.order_id), 0)",
        "仅已交付订单的店铺运费总额 / 店铺订单数",
    ),
    "seller_free_shipping_order_rate": (
        """(SELECT oi.seller_id,oi.order_id,fo.purchase_date,SUM(oi.freight_value) seller_freight FROM analytics.v_delivered_order_items oi JOIN analytics.v_delivered_orders fo USING(order_id) GROUP BY oi.seller_id,oi.order_id,fo.purchase_date) seller_orders""",
        "seller_orders.seller_id", "seller_orders.purchase_date",
        "COUNT(*) FILTER (WHERE seller_freight=0) * 100.0 / NULLIF(COUNT(*),0)",
        "店铺订单中商品行运费合计为零的订单占比",
    ),
    "seller_avg_payment_installments": (
        "analytics.fact_order_payments p JOIN analytics.fact_orders o USING(order_id)",
        "p.seller_id", "o.purchase_date",
        "SUM(COALESCE(p.payment_installments, 0) * p.allocated_payment_value) / NULLIF(SUM(p.allocated_payment_value), 0)",
        "以分摊支付金额加权的平均分期数",
    ),
    "seller_credit_card_payment_share": (
        "analytics.fact_order_payments p JOIN analytics.fact_orders o USING(order_id)",
        "p.seller_id", "o.purchase_date",
        "SUM(p.allocated_payment_value) FILTER (WHERE p.payment_type='credit_card') * 100.0 / NULLIF(SUM(p.allocated_payment_value), 0)",
        "信用卡分摊支付金额 / 总分摊支付金额",
    ),
    "seller_boleto_payment_share": (
        "analytics.fact_order_payments p JOIN analytics.fact_orders o USING(order_id)",
        "p.seller_id", "o.purchase_date",
        "SUM(p.allocated_payment_value) FILTER (WHERE p.payment_type='boleto') * 100.0 / NULLIF(SUM(p.allocated_payment_value), 0)",
        "Boleto 分摊支付金额 / 总分摊支付金额",
    ),
    "seller_active_product_count": (
        "analytics.v_delivered_order_items oi JOIN analytics.v_delivered_orders fo USING(order_id)",
        "oi.seller_id", "fo.purchase_date", "COUNT(DISTINCT oi.product_id)",
        "当前范围内至少成交一次的商品数",
    ),
    "seller_active_category_count": (
        "analytics.v_delivered_order_items oi JOIN analytics.v_delivered_orders fo USING(order_id) LEFT JOIN analytics.v_product_category p USING(product_id)",
        "oi.seller_id", "fo.purchase_date", "COUNT(DISTINCT p.product_category_name_english)",
        "当前范围内至少成交一次的非空商品品类数（翻译后英文名）",
    ),
    "seller_avg_early_delivery_days": (
        "analytics.v_seller_fulfillment", "seller_id", "purchase_date",
        "AVG(EXTRACT(EPOCH FROM (estimated_delivery_at-delivered_customer_at))/86400.0) FILTER (WHERE delivered_customer_at < estimated_delivery_at)",
        "仅提前签收订单的预计送达日期减签收日期平均值",
    ),
    "seller_avg_late_delivery_days": (
        "analytics.v_seller_fulfillment", "seller_id", "purchase_date",
        "AVG(EXTRACT(EPOCH FROM (delivered_customer_at-estimated_delivery_at))/86400.0) FILTER (WHERE delivered_customer_at > estimated_delivery_at)",
        "仅延迟签收订单的签收日期减预计送达日期平均值",
    ),
    "seller_avg_delivery_estimate_abs_error_days": (
        "analytics.v_seller_fulfillment", "seller_id", "purchase_date",
        "AVG(ABS(EXTRACT(EPOCH FROM (delivered_customer_at-estimated_delivery_at))/86400.0)) FILTER (WHERE delivered_customer_at IS NOT NULL AND estimated_delivery_at IS NOT NULL)",
        "已签收且有预计送达日期订单的预计送达绝对误差平均值",
    ),
    "seller_p90_order_to_delivery_days": (
        "analytics.v_seller_fulfillment", "seller_id", "purchase_date",
        "percentile_cont(0.9) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (delivered_customer_at-purchase_at))/86400.0) FILTER (WHERE delivered_customer_at IS NOT NULL AND purchase_at IS NOT NULL)",
        "下单至签收全链路时长的 P90，仅已签收订单",
    ),
    "seller_missing_delivery_timestamp_rate": (
        "analytics.v_seller_fulfillment", "seller_id", "purchase_date",
        "COUNT(*) FILTER (WHERE order_status='delivered' AND delivered_customer_at IS NULL)*100.0 / NULLIF(COUNT(*) FILTER (WHERE order_status='delivered'),0)",
        "已交付订单中缺少签收时间字段的占比，反映数据质量异常",
    ),
}


def supports(metric_name: str) -> bool:
    """判断指标是否属于店铺扩展指标（含基础聚合和四类特殊指标）。"""
    return metric_name in _DEFINITIONS or metric_name in {
        "seller_top1_product_gross_share", "seller_top3_category_gross_share",
        "seller_new_product_gross_share",
        "seller_historical_returning_customer_count", "seller_historical_returning_customer_rate",
        "seller_new_customer_count", "seller_new_customer_rate",
        "seller_single_item_order_product_review_score_avg", "seller_single_item_order_product_bad_review_rate",
        "seller_product_attributed_review_score_avg", "seller_product_attributed_bad_review_rate",
    }


def query(scope: ControlledAnalysisScope, metric_name: str, metric_date: date | None = None) -> dict[str, object]:
    """按指标名称路由到对应的查询函数，返回单点值。"""
    if metric_name in {"seller_top1_product_gross_share", "seller_top3_category_gross_share", "seller_new_product_gross_share"}:
        return _concentration_query(scope, metric_name, metric_date)
    if metric_name.startswith("seller_historical_") or metric_name.startswith("seller_new_customer_"):
        return _customer_lifecycle_query(scope, metric_name, metric_date)
    if "product_review" in metric_name or "product_attributed" in metric_name or "single_item_order_product_bad_review_rate" in metric_name:
        return _review_query(scope, metric_name, metric_date)
    source, seller_column, date_column, expression, note = _DEFINITIONS[metric_name]
    params: dict[str, object] = {"seller_id": scope.seller_id, "start_date": scope.default_start_date, "end_date": scope.default_end_date}
    predicate = f"{date_column} BETWEEN :start_date AND :end_date"
    if metric_date is not None:
        params["metric_date"] = metric_date
        predicate = f"{date_column} = :metric_date"
    with seller_query_transaction(
        engine, scope,
        start_date=metric_date or scope.default_start_date,
        end_date=metric_date or scope.default_end_date,
    ) as conn:
        row = conn.execute(text(f"SELECT {expression} AS value FROM {source} WHERE {seller_column}=:seller_id AND {predicate}"), params).mappings().one()
    return {"metric_name": metric_name, "value": float(row["value"] or 0), "source_table": source, "lineage_note": note, "status": "ok"}


def trend(scope: ControlledAnalysisScope, metric_name: str, start_date: date | None = None, end_date: date | None = None) -> dict[str, object]:
    """按指标名称路由到对应的趋势查询函数，返回时间序列。"""
    if metric_name in {"seller_top1_product_gross_share", "seller_top3_category_gross_share", "seller_new_product_gross_share"}:
        return _concentration_trend(scope, metric_name, start_date, end_date)
    if metric_name.startswith("seller_historical_") or metric_name.startswith("seller_new_customer_"):
        return _customer_lifecycle_trend(scope, metric_name, start_date, end_date)
    if "product_review" in metric_name or "product_attributed" in metric_name or "single_item_order_product_bad_review_rate" in metric_name:
        return _review_trend(scope, metric_name, start_date, end_date)
    source, seller_column, date_column, expression, note = _DEFINITIONS[metric_name]
    params: dict[str, object] = {"seller_id": scope.seller_id, "start_date": start_date or scope.default_start_date, "end_date": end_date or scope.default_end_date}
    bucket = f"date_trunc('{scope.granularity}', {date_column})::date"
    with seller_query_transaction(
        engine, scope,
        start_date=start_date or scope.default_start_date,
        end_date=end_date or scope.default_end_date,
    ) as conn:
        rows = conn.execute(text(f"SELECT {bucket} AS bucket, {expression} AS value FROM {source} WHERE {seller_column}=:seller_id AND {date_column} BETWEEN :start_date AND :end_date GROUP BY {bucket} ORDER BY {bucket}"), params).mappings().all()
    return {"metric_name": metric_name, "points": [{"bucket": str(r["bucket"]), "value": float(r["value"] or 0)} for r in rows], "source_table": source, "lineage_note": note, "status": "ok" if rows else "not_found"}


def _concentration_query(scope: ControlledAnalysisScope, metric_name: str, metric_date: date | None) -> dict[str, object]:
    """集中度/新品占比的查询点。

    三个指标共用一个模板，区别在于外层子查询：
      - top1_product：最高单品销售额占比
      - top3_category：最高三个品类销售额占比
      - new_product：新品（在该店首次成交的商品）销售额占比
    """
    params: dict[str, object] = {"seller_id": scope.seller_id, "start_date": scope.default_start_date, "end_date": scope.default_end_date}
    predicate = "fo.purchase_date BETWEEN :start_date AND :end_date"
    if metric_date:
        params["metric_date"] = metric_date
        predicate = "fo.purchase_date=:metric_date"
    if metric_name == "seller_top1_product_gross_share":
        sql = f"""WITH x AS (SELECT oi.product_id, SUM(oi.price) v FROM analytics.v_delivered_order_items oi JOIN analytics.v_delivered_orders fo USING(order_id) WHERE oi.seller_id=:seller_id AND {predicate} GROUP BY oi.product_id) SELECT MAX(v)*100.0/NULLIF(SUM(v),0) value FROM x"""
        note = "销售额最高商品 / 店铺商品金额总额"
    elif metric_name == "seller_top3_category_gross_share":
        sql = f"""WITH x AS (SELECT p.product_category_name_english k, SUM(oi.price) v FROM analytics.v_delivered_order_items oi JOIN analytics.v_delivered_orders fo USING(order_id) LEFT JOIN analytics.v_product_category p USING(product_id) WHERE oi.seller_id=:seller_id AND {predicate} GROUP BY k), t AS (SELECT SUM(v) v FROM x), r AS (SELECT v FROM x ORDER BY v DESC LIMIT 3) SELECT (SELECT SUM(v) FROM r)*100.0/NULLIF((SELECT v FROM t),0) value"""
        note = "销售额最高三个品类 / 店铺商品金额总额"
    else:
        sql = f"""WITH first_sale AS (SELECT seller_id, product_id, MIN(purchase_date) first_date FROM analytics.v_delivered_order_items GROUP BY seller_id,product_id), x AS (SELECT oi.price, fo.purchase_date, fs.first_date FROM analytics.v_delivered_order_items oi JOIN analytics.v_delivered_orders fo USING(order_id) JOIN first_sale fs ON fs.seller_id=oi.seller_id AND fs.product_id=oi.product_id WHERE oi.seller_id=:seller_id AND {predicate}) SELECT SUM(price) FILTER (WHERE first_date BETWEEN :start_date AND :end_date)*100.0/NULLIF(SUM(price),0) value FROM x"""
        note = "首次在该店铺成交日期落入当前范围的商品金额 / 商品金额总额"
    with seller_query_transaction(
        engine, scope,
        start_date=metric_date or scope.default_start_date,
        end_date=metric_date or scope.default_end_date,
    ) as conn:
        row = conn.execute(text(sql), params).mappings().one()
    return {"metric_name": metric_name, "value": float(row["value"] or 0), "source_table": "analytics.v_delivered_order_items", "lineage_note": note, "status": "ok"}


def _concentration_trend(scope: ControlledAnalysisScope, metric_name: str, start_date: date | None, end_date: date | None) -> dict[str, object]:
    """集中度/新品占比的趋势查询。

    对每个时间桶独立计算集中度/新品占比，首次成交日仍基于完整历史。"""
    points = []
    current = start_date or scope.default_start_date
    end = end_date or scope.default_end_date
    # 复用数据库 date_trunc，避免在 Python 侧重新定义周/月/季边界。
    bucket = f"date_trunc('{scope.granularity}', fo.purchase_date)::date"
    params = {"seller_id": scope.seller_id, "start_date": current, "end_date": end}
    if metric_name == "seller_top1_product_gross_share":
        sql = f"""WITH x AS (SELECT {bucket} b, oi.product_id, SUM(oi.price) v FROM analytics.v_delivered_order_items oi JOIN analytics.v_delivered_orders fo USING(order_id) WHERE oi.seller_id=:seller_id AND fo.purchase_date BETWEEN :start_date AND :end_date GROUP BY b,oi.product_id), y AS (SELECT b, MAX(v) top_v, SUM(v) total_v FROM x GROUP BY b) SELECT b bucket, top_v*100.0/NULLIF(total_v,0) value FROM y ORDER BY b"""
    elif metric_name == "seller_top3_category_gross_share":
        sql = f"""WITH x AS (SELECT {bucket} b, p.product_category_name_english k, SUM(oi.price) v FROM analytics.v_delivered_order_items oi JOIN analytics.v_delivered_orders fo USING(order_id) LEFT JOIN analytics.v_product_category p USING(product_id) WHERE oi.seller_id=:seller_id AND fo.purchase_date BETWEEN :start_date AND :end_date GROUP BY b,k), r AS (SELECT b,v,ROW_NUMBER() OVER(PARTITION BY b ORDER BY v DESC) rn,SUM(v) OVER(PARTITION BY b) total_v FROM x) SELECT b bucket,SUM(v) FILTER(WHERE rn<=3)*100.0/NULLIF(MAX(total_v),0) value FROM r GROUP BY b ORDER BY b"""
    else:
        first_sale_bucket = f"date_trunc('{scope.granularity}', first_date)::date"
        sql = f"""WITH first_sale AS (SELECT seller_id,product_id,MIN(purchase_date) first_date FROM analytics.v_delivered_order_items GROUP BY seller_id,product_id), x AS (SELECT {bucket} b,oi.price,fo.purchase_date,fs.first_date FROM analytics.v_delivered_order_items oi JOIN analytics.v_delivered_orders fo USING(order_id) JOIN first_sale fs ON fs.seller_id=oi.seller_id AND fs.product_id=oi.product_id WHERE oi.seller_id=:seller_id AND fo.purchase_date BETWEEN :start_date AND :end_date) SELECT b bucket,SUM(price) FILTER(WHERE {first_sale_bucket}=b)*100.0/NULLIF(SUM(price),0) value FROM x GROUP BY b ORDER BY b"""
    with seller_query_transaction(
        engine, scope,
        start_date=current,
        end_date=end,
    ) as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return {"metric_name": metric_name, "points": [{"bucket": str(r["bucket"]), "value": float(r["value"] or 0)} for r in rows], "source_table": "analytics.v_delivered_order_items", "lineage_note": "按时间桶计算的店铺商品结构指标", "status": "ok" if rows else "not_found"}


def _customer_lifecycle_sql(metric_name: str, date_filter: str, bucket: str | None = None) -> tuple[str, str]:
    """生成客户生命周期指标的 SQL 模板。

    通过窗口函数识别每个客户在该店的首购日期和购买序号，
    以此区分新客户和回头客。
    """
    b = f"{bucket} bucket," if bucket else ""
    group = "GROUP BY bucket" if bucket else ""
    base = """WITH seller_orders AS (SELECT DISTINCT oi.seller_id,oi.order_id,fo.customer_unique_id,fo.purchase_date FROM analytics.v_delivered_order_items oi JOIN analytics.v_delivered_orders fo USING(order_id)), ranked AS (SELECT *,MIN(purchase_date) OVER(PARTITION BY seller_id,customer_unique_id) first_date,ROW_NUMBER() OVER(PARTITION BY seller_id,customer_unique_id ORDER BY purchase_date,order_id) purchase_number FROM seller_orders)"""
    if metric_name.startswith("seller_historical_returning"):
        numerator = "COUNT(DISTINCT customer_unique_id) FILTER (WHERE purchase_number > 1)"
        note = "当前范围内曾在此前向同一店铺购买过的去重客户数"
    else:
        numerator = "COUNT(DISTINCT customer_unique_id) FILTER (WHERE first_date BETWEEN :start_date AND :end_date)"
        note = "首次向该店铺购买日期落在当前范围内的去重客户数"
    value = numerator if metric_name.endswith("_count") else f"{numerator} * 100.0 / NULLIF(COUNT(DISTINCT customer_unique_id),0)"
    return f"{base} SELECT {b}{value} value FROM ranked WHERE seller_id=:seller_id AND {date_filter} {group}", note


def _customer_lifecycle_query(scope: ControlledAnalysisScope, metric_name: str, metric_date: date | None) -> dict[str, object]:
    """客户生命周期指标的单点查询。"""
    predicate = "purchase_date BETWEEN :start_date AND :end_date" if metric_date is None else "purchase_date=:metric_date"
    params = {"seller_id": scope.seller_id, "start_date": scope.default_start_date, "end_date": scope.default_end_date}
    if metric_date: params["metric_date"] = metric_date
    sql, note = _customer_lifecycle_sql(metric_name, predicate)
    with seller_query_transaction(
        engine, scope,
        start_date=metric_date or scope.default_start_date,
        end_date=metric_date or scope.default_end_date,
    ) as conn:
        row = conn.execute(text(sql), params).mappings().one()
    return {"metric_name": metric_name, "value": float(row["value"] or 0), "source_table": "analytics.v_delivered_order_items", "lineage_note": note, "status": "ok"}


def _customer_lifecycle_trend(scope: ControlledAnalysisScope, metric_name: str, start_date: date | None, end_date: date | None) -> dict[str, object]:
    """客户生命周期指标的趋势查询。"""
    params = {"seller_id": scope.seller_id, "start_date": start_date or scope.default_start_date, "end_date": end_date or scope.default_end_date}
    bucket = f"date_trunc('{scope.granularity}', purchase_date)::date"
    sql, note = _customer_lifecycle_sql(metric_name, "purchase_date BETWEEN :start_date AND :end_date", bucket)
    with seller_query_transaction(
        engine, scope,
        start_date=start_date or scope.default_start_date,
        end_date=end_date or scope.default_end_date,
    ) as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return {"metric_name": metric_name, "points": [{"bucket": str(r["bucket"]), "value": float(r["value"] or 0)} for r in rows], "source_table": "analytics.v_delivered_order_items", "lineage_note": note, "status": "ok" if rows else "not_found"}


def _review_source(metric_name: str) -> tuple[str, str, str]:
    """返回评论指标的 FROM 源和过滤列。

    单商品订单（seller_single_item_*）用子查询过滤出仅含 1 件商品的订单，
    然后关联评论表；通用评论指标直接在订单明细上关联。"""
    if metric_name.startswith("seller_single_item"):
        return ("""(SELECT oi.seller_id,oi.order_id,fo.purchase_date FROM analytics.v_delivered_order_items oi JOIN analytics.v_delivered_orders fo USING(order_id) GROUP BY oi.seller_id,oi.order_id,fo.purchase_date HAVING COUNT(*)=1) seller_orders JOIN raw.order_reviews r USING(order_id)""", "seller_orders.seller_id", "seller_orders.purchase_date")
    return ("analytics.v_delivered_order_items oi JOIN analytics.v_delivered_orders fo USING(order_id) JOIN raw.order_reviews r USING(order_id)", "oi.seller_id", "fo.purchase_date")


def _review_expression(metric_name: str) -> tuple[str, str]:
    """返回评论指标的聚合表达式和中文注释。

    以 _score_avg 结尾→平均评分，否则→差评率（≤2 分）。"""
    if metric_name.endswith("score_avg"): return "AVG(r.review_score)", "平均订单评论分；多商品订单评论会按商品关联次数加权"
    return "COUNT(*) FILTER (WHERE r.review_score<=2)*100.0/NULLIF(COUNT(r.review_score),0)", "评分小于等于 2 的评论占比；多商品订单评论按商品关联次数加权"


def _review_query(scope: ControlledAnalysisScope, metric_name: str, metric_date: date | None) -> dict[str, object]:
    """评论评价指标的单点查询。"""
    source, seller_col, date_col = _review_source(metric_name); expr, note = _review_expression(metric_name)
    params = {"seller_id": scope.seller_id, "start_date": scope.default_start_date, "end_date": scope.default_end_date}; pred = f"{date_col} BETWEEN :start_date AND :end_date"
    if metric_date: params["metric_date"] = metric_date; pred = f"{date_col}=:metric_date"
    with seller_query_transaction(
        engine, scope,
        start_date=metric_date or scope.default_start_date,
        end_date=metric_date or scope.default_end_date,
    ) as conn:
        row=conn.execute(text(f"SELECT {expr} value FROM {source} WHERE {seller_col}=:seller_id AND {pred}"),params).mappings().one()
    return {"metric_name":metric_name,"value":float(row["value"] or 0),"source_table":source,"lineage_note":note,"status":"ok"}


def _review_trend(scope: ControlledAnalysisScope, metric_name: str, start_date: date | None, end_date: date | None) -> dict[str, object]:
    """评论评价指标的趋势查询。"""
    source,seller_col,date_col=_review_source(metric_name); expr,note=_review_expression(metric_name); params={"seller_id":scope.seller_id,"start_date":start_date or scope.default_start_date,"end_date":end_date or scope.default_end_date}; bucket=f"date_trunc('{scope.granularity}', {date_col})::date"
    with seller_query_transaction(
        engine, scope,
        start_date=start_date or scope.default_start_date,
        end_date=end_date or scope.default_end_date,
    ) as conn:
        rows=conn.execute(text(f"SELECT {bucket} bucket,{expr} value FROM {source} WHERE {seller_col}=:seller_id AND {date_col} BETWEEN :start_date AND :end_date GROUP BY {bucket} ORDER BY {bucket}"),params).mappings().all()
    return {"metric_name":metric_name,"points":[{"bucket":str(r["bucket"]),"value":float(r["value"] or 0)} for r in rows],"source_table":source,"lineage_note":note,"status":"ok" if rows else "not_found"}

"""每日经营指标查询仓库。

这个文件是当前项目里"经营分析指标"的核心数据访问层，负责三类查询：

1. 单点查询
   例子：`2018-08-10 的 GMV 是多少`
   含义：查询某个指标在某一天的值
2. 趋势查询
   例子：`2018-08-01 到 2018-08-03 的 GMV 趋势`
   含义：查询某个指标在一段时间内每天的值
3. 对比 / 拆解查询
   例子：
   - `对比 2018-08-09 和 2018-08-10 的 GMV`
   - `查询 2018-08-10 的 GMV 按州拆解`

当前已实现的指标及业务含义：

- `gmv`
  成交总额。口径是当天已交付订单的商品金额与运费之和。
- `order_count`
  成交订单量。口径是当天已交付订单数。
- `avg_order_value`
  客单价。口径是 `GMV / order_count`。
- `cancel_rate`
  取消率。口径是当天创建订单中，最终状态为 `canceled` 的订单占比。
- `freight_value`
  运费总额。口径是当天已交付订单对应的运费总和。
- `review_score_avg`
  平均评论分。口径是当天创建订单对应评论记录的平均 `review_score`。
- `delivery_delay_rate`
  延迟送达率。口径是当天创建且最终已送达订单中，实际送达时间晚于预计送达时间的订单占比。
- `customer_count`
  成交客户数。口径是当天已交付订单对应的去重客户数，按 `customer_unique_id` 去重。
- `repeat_customer_count`
  复购客户数。口径是当天成交客户中，在当天之前已经有过历史成交订单的客户数。
- `repeat_purchase_rate`
  复购率。口径是 `repeat_customer_count / customer_count * 100`。
- `avg_items_per_order`
  平均每单件数。口径是当天已交付订单的平均商品件数。
- `avg_delivery_lead_time`
  平均送达时长。口径是当天创建且已送达订单，从下单到客户签收的平均天数。
- `on_time_delivery_rate`
  准时送达率。口径是当天创建且已送达订单中，实际送达时间不晚于预计送达时间的订单占比。
- `bad_review_rate`
  差评率。口径是当天有评论记录的订单中，评论分小于等于 2 分的订单占比。
- `freight_rate`
  运费率。口径是 `freight_value / gmv * 100`。

设计原则：
- 能直接从 `analytics.daily_metrics` 读取的，优先走聚合表，性能更稳定。
- 聚合表里还没有的指标，就临时从 `raw` 或 `analytics.fact_orders` 现算。
- 上层 MCP、workflow 不需要关心这些差异，只调用统一查询函数即可。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from functools import partial

from sqlalchemy import text

from app.data.database import engine
from app.data.report_snapshot import repository_connection
from app.data.repositories.order_allocation import (
    ALLOCATION_METHOD,
    ALLOCATION_NOTE,
    allocated_items,
    paired_allocated_items,
    require_reconciled,
)


# 文件作用：
# - 提供每日经营指标的统一查询入口。
# - 支持单点查询、趋势查询、日期对比、维度拆解四类能力。
# - 根据指标类型自动选择聚合表、事实表或原始表作为数据来源。
# SUPPORTED_METRICS 现在由注册中心管理。
# 保留此集合供旧代码路径做快速校验；新代码应使用 registry.list()。
_SUPPORTED_METRICS_CACHE: set[str] | None = None


def _get_supported_metrics() -> set[str]:
    """Lazily populate from the registry."""
    global _SUPPORTED_METRICS_CACHE
    if _SUPPORTED_METRICS_CACHE is None:
        from app.metrics.registry import get_registry

        reg = get_registry()
        _SUPPORTED_METRICS_CACHE = {m.metric_id for m in reg.list()}
    return _SUPPORTED_METRICS_CACHE


SUPPORTED_METRICS = _get_supported_metrics()
SUPPORTED_BREAKDOWN_DIMENSIONS = {
    "customer_state",
    "customer_city",
    "seller_state",
    "product_category",
    "payment_type",
    "month_position",
    "day_of_week",
    "hour_of_day",
    "quarter_of_year",
    "half_of_year",
}

# Ratio attribution only permits dimensions that partition the one-row-per-order
# fact table without a fan-out join or widening customer-data access.
SUPPORTED_RATIO_FACT_DIMENSIONS = {
    "month_position": "CASE WHEN EXTRACT(DAY FROM purchase_date) BETWEEN 1 AND 10 "
                      "THEN '月初(1-10日)' WHEN EXTRACT(DAY FROM purchase_date) BETWEEN 11 AND 20 "
                      "THEN '月中(11-20日)' ELSE '月末(21-31日)' END",
}

# Exact sufficient-statistic contracts for the one-row-per-order fact table.
# SQL expressions are server constants; callers cannot supply identifiers.
SUPPORTED_RATIO_FACT_METRICS = {
    ("freight_value", "gmv"): {
        "metric_name": "freight_rate",
        "numerator_sql": "SUM(freight_value)",
        "denominator_sql": "SUM(item_gross_value + freight_value)",
    },
    ("gmv", "order_count"): {
        "metric_name": "avg_order_value",
        "numerator_sql": "SUM(item_gross_value + freight_value)",
        "denominator_sql": "COUNT(*)",
    },
}

AGGREGATED_METRICS = {
    "gmv",
    "order_count",
    "avg_order_value",
}

# 支付与履约数据域使用独立事实表；日期始终按订单下单日归属。
_DOMAIN_METRIC_SQL = {
    # fact_order_payments is at payment × seller grain. payment_value repeats
    # on every seller allocation row; allocated_payment_value is the additive
    # platform-safe amount (the migration assigns any rounding remainder).
    "payment_value": ("analytics.fact_order_payments p JOIN analytics.fact_orders o USING(order_id)", "SUM(p.allocated_payment_value)", "o.purchase_date"),
    "installment_order_share": ("analytics.fact_order_payments p JOIN analytics.fact_orders o USING(order_id)", "SUM(CASE WHEN p.payment_installments > 1 THEN p.allocated_payment_value ELSE 0 END) * 100.0 / NULLIF(SUM(p.allocated_payment_value), 0)", "o.purchase_date"),
    "avg_approval_lead_hours": ("analytics.fact_fulfillment", "AVG(approval_lead_hours)", "purchase_date"),
    "avg_dispatch_lead_hours": ("analytics.fact_fulfillment", "AVG(dispatch_lead_hours)", "purchase_date"),
    "avg_delivery_transit_hours": ("analytics.fact_fulfillment", "AVG(delivery_transit_hours)", "purchase_date"),
    # Order funnel metrics (Phase 6 M1b)
    "order_approval_rate": ("analytics.v_order_cohort_funnel",
        "SUM(stage_approved) * 100.0 / NULLIF(SUM(total_orders), 0)", "purchase_date"),
    "order_shipment_rate": ("analytics.v_order_cohort_funnel",
        "SUM(stage_shipped) * 100.0 / NULLIF(SUM(total_orders), 0)", "purchase_date"),
    "order_final_delivery_rate": ("analytics.v_order_cohort_funnel",
        "SUM(final_delivered) * 100.0 / NULLIF(SUM(total_orders), 0)", "purchase_date"),
    # Review text (Phase 6 M3)
    "review_comment_rate": (
        "raw.orders o LEFT JOIN raw.order_reviews r ON r.order_id = o.order_id",
        "COUNT(*) FILTER (WHERE r.review_comment_message IS NOT NULL"
        " AND TRIM(r.review_comment_message) != '')"
        " * 100.0 / NULLIF(COUNT(r.order_id), 0)",
        "DATE(o.order_purchase_timestamp)"),
}

_PERIOD_DOMAIN_METRICS = {
    "payment_value",
    "installment_order_share",
    "avg_approval_lead_hours",
    "avg_dispatch_lead_hours",
    "avg_delivery_transit_hours",
}


def _query_domain_metric(metric_name: str, metric_date: date) -> dict:
    source, expression, date_column = _DOMAIN_METRIC_SQL[metric_name]
    with repository_connection(engine) as conn:
        row = conn.execute(text(f"SELECT {expression} AS value FROM {source} WHERE {date_column}=:metric_date"), {"metric_date": metric_date}).mappings().one()
    value = row["value"]
    return _build_result(metric_name, metric_date, round(float(value), 2) if value is not None else None, source)


#
# 作用：统一规范日期入参，避免各查询函数重复解析日期。
# 参数：
# - value: `YYYY-MM-DD` 字符串、`date` 或 `datetime`。
# 返回：
# - `date` 对象，作为 SQL 查询参数使用。
def _normalize_date(value: str | date | datetime) -> date:
    """把字符串 / `date` / `datetime` 统一转换成 `date` 对象。"""

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


#
# 作用：构造统一的指标查询结果字典，保证上层调用拿到稳定结构。
# 参数：
# - metric_name: 指标名称。
# - metric_date: 指标日期。
# - value: 指标值，无数据时为 `None`。
# - source_table: 指标来源表名。
# 返回：
# - 标准化结果字典，包含指标名、日期、数值、来源表和状态。
def _build_result(
    metric_name: str,
    metric_date: date,
    value: float | None,
    source_table: str,
    *,
    source_lineage: str | None = None,
) -> dict:
    """构造统一的单点查询返回结构。"""

    result = {
        "metric_name": metric_name,
        "metric_date": metric_date.isoformat(),
        "value": value,
        "source_table": source_table,
        "status": "ok" if value is not None else "not_found",
    }
    if source_lineage is not None:
        result["source_lineage"] = source_lineage
    return result


#
# 作用：查询当前数据库中已构建指标表的最新日期，便于上层默认取最近一天。
# 参数：无。
# 返回：
# - 最新指标日期的 ISO 字符串。
# - 若聚合表为空，则返回 `None`。
def get_latest_metric_date() -> str | None:
    """返回当前指标表里最新的可用日期。"""

    query = text("SELECT MAX(metric_date) AS max_metric_date FROM analytics.daily_metrics")
    with repository_connection(engine) as conn:
        row = conn.execute(query).mappings().first()
    if row is None or row["max_metric_date"] is None:
        return None
    return row["max_metric_date"].isoformat()


#
# 作用：从 `analytics.daily_metrics` 读取某天的核心聚合指标。
# 参数：
# - metric_date: 待查询日期，支持字符串、`date`、`datetime`。
# 返回：
# - 命中时返回聚合行字典。
# - 未命中时返回 `None`。
def get_daily_metrics(metric_date: str | date | datetime) -> dict | None:
    """从 `analytics.daily_metrics` 读取某一天的核心聚合指标。

    这里保持最小字段集，只依赖旧版数据库也稳定存在的三列：
    `gmv / order_count / avg_order_value`。

    其他指标即使未来也会聚合进 `analytics.daily_metrics`，运行时查询仍优先走
    现算逻辑，这样可以兼容当前还没有重建过数据库的环境。
    """

    normalized = _normalize_date(metric_date)
    query = text(
        """
        SELECT
            metric_date,
            gmv,
            order_count,
            avg_order_value
        FROM analytics.daily_metrics
        WHERE metric_date = :metric_date
        """
    )
    with repository_connection(engine) as conn:
        row = conn.execute(query, {"metric_date": normalized}).mappings().first()
    return dict(row) if row else None


#
# 作用：优先从日聚合表读取已预计算完成的指标，减少实时计算成本。
# 参数：
# - metric_name: 指标名称。
# - metric_date: 待查询日期。
# 返回：
# - 标准化指标结果字典。
def _query_from_aggregated_table(metric_name: str, metric_date: date) -> dict:
    """从日聚合表读取已经算好的单日指标。"""

    metrics = get_daily_metrics(metric_date)
    if metrics is None:
        return _build_result(metric_name, metric_date, None, "analytics.daily_metrics")
    value = metrics.get(metric_name)
    return _build_result(metric_name, metric_date, float(value) if value is not None else None, "analytics.daily_metrics")


#
# 作用：计算某天创建订单中的取消率。
# 参数：
# - metric_date: 待查询日期。
# 返回：
# - 取消率结果字典；若当天没有订单则 `value` 为 `None`。
def _query_cancel_rate(metric_date: date) -> dict:
    """查询取消率。"""

    query = text(
        """
        SELECT
            COUNT(*) AS total_orders,
            SUM(CASE WHEN order_status = 'canceled' THEN 1 ELSE 0 END) AS canceled_orders
        FROM raw.orders
        WHERE DATE(order_purchase_timestamp) = :metric_date
        """
    )
    with repository_connection(engine) as conn:
        row = conn.execute(query, {"metric_date": metric_date}).mappings().first()

    total_orders = int(row["total_orders"] or 0) if row else 0
    canceled_orders = int(row["canceled_orders"] or 0) if row else 0
    if total_orders == 0:
        return _build_result("cancel_rate", metric_date, None, "raw.orders")

    value = round(canceled_orders * 100.0 / total_orders, 2)
    return _build_result("cancel_rate", metric_date, value, "raw.orders")


#
# 作用：计算某天成交订单对应的运费总额。
# 参数：
# - metric_date: 待查询日期。
# 返回：
# - 运费总额结果字典；若无成交订单则 `value` 为 `None`。
def _query_freight_value(metric_date: date) -> dict:
    """查询运费总额。"""

    query = text(
        """
        SELECT
            SUM(freight_value) AS freight_value
        FROM analytics.v_delivered_orders
        WHERE purchase_date = :metric_date
        """
    )
    with repository_connection(engine) as conn:
        row = conn.execute(query, {"metric_date": metric_date}).mappings().first()

    if row is None or row["freight_value"] is None:
        return _build_result("freight_value", metric_date, None, "analytics.v_delivered_orders")
    return _build_result("freight_value", metric_date, float(row["freight_value"]), "analytics.v_delivered_orders")


#
# 作用：计算某天订单对应评论记录的平均评分。
# 参数：
# - metric_date: 待查询日期。
# 返回：
# - 平均评论分结果字典；若没有评论数据则 `value` 为 `None`。
def _query_review_score_avg(metric_date: date) -> dict:
    """查询平均评论分。"""

    query = text(
        """
        SELECT
            AVG(r.review_score) AS review_score_avg
        FROM raw.orders o
        LEFT JOIN raw.order_reviews r
            ON o.order_id = r.order_id
        WHERE DATE(o.order_purchase_timestamp) = :metric_date
        """
    )
    with repository_connection(engine) as conn:
        row = conn.execute(query, {"metric_date": metric_date}).mappings().first()

    if row is None or row["review_score_avg"] is None:
        return _build_result("review_score_avg", metric_date, None, "raw.order_reviews")
    return _build_result("review_score_avg", metric_date, round(float(row["review_score_avg"]), 2), "raw.order_reviews")


#
# 作用：计算某天已送达订单中的延迟送达比例。
# 参数：
# - metric_date: 待查询日期。
# 返回：
# - 延迟送达率结果字典；若无可比较送达订单则 `value` 为 `None`。
def _query_delivery_delay_rate(metric_date: date) -> dict:
    """查询延迟送达率。"""

    query = text(
        """
        SELECT
            SUM(
                CASE
                    WHEN order_status = 'delivered'
                         AND order_delivered_customer_date IS NOT NULL
                         AND order_estimated_delivery_date IS NOT NULL
                    THEN 1
                    ELSE 0
                END
            ) AS delivered_orders,
            SUM(
                CASE
                    WHEN order_status = 'delivered'
                         AND order_delivered_customer_date IS NOT NULL
                         AND order_estimated_delivery_date IS NOT NULL
                         AND order_delivered_customer_date > order_estimated_delivery_date
                    THEN 1
                    ELSE 0
                END
            ) AS delayed_orders
        FROM raw.orders
        WHERE DATE(order_purchase_timestamp) = :metric_date
        """
    )
    with repository_connection(engine) as conn:
        row = conn.execute(query, {"metric_date": metric_date}).mappings().first()

    delivered_orders = int(row["delivered_orders"] or 0) if row else 0
    delayed_orders = int(row["delayed_orders"] or 0) if row else 0
    if delivered_orders == 0:
        return _build_result("delivery_delay_rate", metric_date, None, "raw.orders")

    value = round(delayed_orders * 100.0 / delivered_orders, 2)
    return _build_result("delivery_delay_rate", metric_date, value, "raw.orders")


#
# 作用：统计某天成交客户数，按 `customer_unique_id` 去重。
# 参数：
# - metric_date: 待查询日期。
# 返回：
# - 成交客户数结果字典；没有成交客户时 `value` 为 `None`。
def _query_customer_count(metric_date: date) -> dict:
    """查询成交客户数。"""

    query = text(
        """
        SELECT COUNT(DISTINCT customer_unique_id) AS customer_count
        FROM analytics.v_delivered_orders
        WHERE purchase_date = :metric_date
          AND customer_unique_id IS NOT NULL
        """
    )
    with repository_connection(engine) as conn:
        row = conn.execute(query, {"metric_date": metric_date}).mappings().first()

    if row is None or row["customer_count"] is None:
        return _build_result("customer_count", metric_date, None, "analytics.v_delivered_orders")
    return _build_result("customer_count", metric_date, float(row["customer_count"]), "analytics.v_delivered_orders")


#
# 作用：统计某天成交客户中历史上曾下过单的复购客户数。
# 参数：
# - metric_date: 待查询日期。
# 返回：
# - 复购客户数结果字典；无数据时 `value` 为 `None`。
def _query_repeat_customer_count(metric_date: date) -> dict:
    """查询复购客户数。"""

    query = text(
        """
        SELECT COUNT(DISTINCT fo.customer_unique_id) AS repeat_customer_count
        FROM analytics.v_delivered_orders fo
        WHERE fo.purchase_date = :metric_date
          AND fo.customer_unique_id IS NOT NULL
          AND EXISTS (
              SELECT 1
              FROM analytics.v_delivered_orders prev
              WHERE prev.customer_unique_id = fo.customer_unique_id
                AND prev.purchase_date < :metric_date
          )
        """
    )
    with repository_connection(engine) as conn:
        row = conn.execute(query, {"metric_date": metric_date}).mappings().first()

    if row is None or row["repeat_customer_count"] is None:
        value = None
    else:
        value = float(row["repeat_customer_count"])
    return _build_result(
        "repeat_customer_count", metric_date, value, "analytics.v_delivered_orders",
        source_lineage="analytics.fact_orders",
    )


#
# 作用：基于成交客户数和复购客户数计算复购率。
# 参数：
# - metric_date: 待查询日期。
# 返回：
# - 复购率结果字典；客户数为 0 或缺失时 `value` 为 `None`。
def _query_repeat_purchase_rate(metric_date: date) -> dict:
    """查询复购率。"""

    customer_count = _query_customer_count(metric_date)["value"]
    repeat_customer_count = _query_repeat_customer_count(metric_date)["value"]
    if customer_count in (None, 0):
        return _build_result(
            "repeat_purchase_rate", metric_date, None, "analytics.v_delivered_orders",
            source_lineage="analytics.fact_orders",
        )

    value = round(float(repeat_customer_count or 0) * 100.0 / float(customer_count), 2)
    return _build_result(
        "repeat_purchase_rate", metric_date, value, "analytics.v_delivered_orders",
        source_lineage="analytics.fact_orders",
    )


def query_seller_repeat_purchase_rate(scope, metric_date: date | None = None) -> dict:
    """Query the registered seller-scoped repeat-purchase-rate implementation.

    The platform daily-metrics query above is customer-grain and does not
    apply seller attribution. Seller scope therefore evaluates returning
    customers using delivered item rows for the same seller on both the
    measured date and an earlier purchase date.
    """
    if "repeat_purchase_rate" not in scope.allowed_metrics:
        raise ValueError("repeat_purchase_rate is outside the locked analysis scope")
    target_date = metric_date or scope.default_start_date
    if target_date < scope.default_start_date or target_date > scope.default_end_date:
        raise ValueError("metric_date is outside the locked seller analysis range")

    query = text("""
        SELECT
            COUNT(DISTINCT fo.customer_unique_id) FILTER (
                WHERE EXISTS (
                    SELECT 1
                    FROM analytics.v_delivered_order_items previous_items
                    JOIN analytics.v_delivered_orders previous_orders
                      ON previous_items.order_id = previous_orders.order_id
                    WHERE previous_items.seller_id = :seller_id
                      AND previous_orders.customer_unique_id = fo.customer_unique_id
                      AND previous_orders.purchase_date < :metric_date
                )
            ) * 100.0 / NULLIF(COUNT(DISTINCT fo.customer_unique_id), 0) AS value
        FROM analytics.v_delivered_order_items oi
        JOIN analytics.v_delivered_orders fo ON oi.order_id = fo.order_id
        WHERE oi.seller_id = :seller_id
          AND fo.purchase_date = :metric_date
          AND fo.customer_unique_id IS NOT NULL
    """)
    from app.data.rls import seller_query_transaction

    with seller_query_transaction(
        engine, scope, start_date=target_date, end_date=target_date,
    ) as connection:
        row = connection.execute(query, {
            "seller_id": scope.seller_id,
            "metric_date": target_date,
        }).mappings().one()
    value = row["value"]
    return _build_result(
        "repeat_purchase_rate",
        target_date,
        round(float(value), 2) if value is not None else None,
        "analytics.v_delivered_order_items + analytics.v_delivered_orders",
        source_lineage="analytics.fact_order_items + analytics.fact_orders",
    )


#
# 作用：计算某天成交订单的平均每单件数。
# 参数：
# - metric_date: 待查询日期。
# 返回：
# - 平均每单件数结果字典；无订单时 `value` 为 `None`。
def _query_avg_items_per_order(metric_date: date) -> dict:
    """查询平均每单件数。"""

    query = text(
        """
        SELECT AVG(item_count) AS avg_items_per_order
        FROM analytics.v_delivered_orders
        WHERE purchase_date = :metric_date
        """
    )
    with repository_connection(engine) as conn:
        row = conn.execute(query, {"metric_date": metric_date}).mappings().first()

    if row is None or row["avg_items_per_order"] is None:
        value = None
    else:
        value = round(float(row["avg_items_per_order"]), 2)
    return _build_result(
        "avg_items_per_order", metric_date, value, "analytics.v_delivered_orders",
        source_lineage="analytics.fact_orders",
    )


#
# 作用：计算某天已送达订单从下单到签收的平均天数。
# 参数：
# - metric_date: 待查询日期。
# 返回：
# - 平均送达时长结果字典；无符合条件订单时 `value` 为 `None`。
def _query_avg_delivery_lead_time(metric_date: date) -> dict:
    """查询平均送达时长，单位为天。"""

    query = text(
        """
        SELECT
            AVG(EXTRACT(EPOCH FROM (order_delivered_customer_date - order_purchase_timestamp)) / 86400.0) AS avg_delivery_lead_time
        FROM raw.orders
        WHERE DATE(order_purchase_timestamp) = :metric_date
          AND order_status = 'delivered'
          AND order_delivered_customer_date IS NOT NULL
        """
    )
    with repository_connection(engine) as conn:
        row = conn.execute(query, {"metric_date": metric_date}).mappings().first()

    if row is None or row["avg_delivery_lead_time"] is None:
        return _build_result("avg_delivery_lead_time", metric_date, None, "raw.orders")
    return _build_result("avg_delivery_lead_time", metric_date, round(float(row["avg_delivery_lead_time"]), 2), "raw.orders")


#
# 作用：计算某天已送达订单中的准时送达比例。
# 参数：
# - metric_date: 待查询日期。
# 返回：
# - 准时送达率结果字典；无可比较送达订单时 `value` 为 `None`。
def _query_on_time_delivery_rate(metric_date: date) -> dict:
    """查询准时送达率。"""

    query = text(
        """
        SELECT
            SUM(
                CASE
                    WHEN order_status = 'delivered'
                         AND order_delivered_customer_date IS NOT NULL
                         AND order_estimated_delivery_date IS NOT NULL
                    THEN 1
                    ELSE 0
                END
            ) AS delivered_orders,
            SUM(
                CASE
                    WHEN order_status = 'delivered'
                         AND order_delivered_customer_date IS NOT NULL
                         AND order_estimated_delivery_date IS NOT NULL
                         AND order_delivered_customer_date <= order_estimated_delivery_date
                    THEN 1
                    ELSE 0
                END
            ) AS on_time_orders
        FROM raw.orders
        WHERE DATE(order_purchase_timestamp) = :metric_date
        """
    )
    with repository_connection(engine) as conn:
        row = conn.execute(query, {"metric_date": metric_date}).mappings().first()

    delivered_orders = int(row["delivered_orders"] or 0) if row else 0
    on_time_orders = int(row["on_time_orders"] or 0) if row else 0
    if delivered_orders == 0:
        return _build_result("on_time_delivery_rate", metric_date, None, "raw.orders")

    value = round(on_time_orders * 100.0 / delivered_orders, 2)
    return _build_result("on_time_delivery_rate", metric_date, value, "raw.orders")


#
# 作用：计算某天有评论订单中的差评率，差评定义为评分小于等于 2。
# 参数：
# - metric_date: 待查询日期。
# 返回：
# - 差评率结果字典；无评论数据时 `value` 为 `None`。
def _query_bad_review_rate(metric_date: date) -> dict:
    """查询差评率。"""

    query = text(
        """
        SELECT
            COUNT(r.review_score) AS reviewed_orders,
            SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END) AS bad_review_orders
        FROM raw.orders o
        LEFT JOIN raw.order_reviews r
            ON o.order_id = r.order_id
        WHERE DATE(o.order_purchase_timestamp) = :metric_date
        """
    )
    with repository_connection(engine) as conn:
        row = conn.execute(query, {"metric_date": metric_date}).mappings().first()

    reviewed_orders = int(row["reviewed_orders"] or 0) if row else 0
    bad_review_orders = int(row["bad_review_orders"] or 0) if row else 0
    if reviewed_orders == 0:
        return _build_result("bad_review_rate", metric_date, None, "raw.order_reviews")

    value = round(bad_review_orders * 100.0 / reviewed_orders, 2)
    return _build_result("bad_review_rate", metric_date, value, "raw.order_reviews")


#
# 作用：计算某天运费占 GMV 的比例。
# 参数：
# - metric_date: 待查询日期。
# 返回：
# - 运费率结果字典；GMV 为 0 时 `value` 为 `None`。
def _query_freight_rate(metric_date: date) -> dict:
    """查询运费率。"""

    query = text(
        """
        SELECT
            SUM(freight_value) AS total_freight_value,
            SUM(item_gross_value + freight_value) AS total_gmv
        FROM analytics.fact_orders
        WHERE purchase_date = :metric_date AND order_status = 'delivered'
        """
    )
    with repository_connection(engine) as conn:
        row = conn.execute(query, {"metric_date": metric_date}).mappings().first()

    total_gmv = float(row["total_gmv"] or 0) if row else 0
    total_freight_value = float(row["total_freight_value"] or 0) if row else 0
    if total_gmv == 0:
        return _build_result("freight_rate", metric_date, None, "analytics.fact_orders")

    value = round(total_freight_value * 100.0 / total_gmv, 2)
    return _build_result("freight_rate", metric_date, value, "analytics.fact_orders")


DIRECT_QUERY_HANDLERS = {
    "cancel_rate": _query_cancel_rate,
    "freight_value": _query_freight_value,
    "review_score_avg": _query_review_score_avg,
    "delivery_delay_rate": _query_delivery_delay_rate,
    "customer_count": _query_customer_count,
    "repeat_customer_count": _query_repeat_customer_count,
    "repeat_purchase_rate": _query_repeat_purchase_rate,
    "avg_items_per_order": _query_avg_items_per_order,
    "avg_delivery_lead_time": _query_avg_delivery_lead_time,
    "on_time_delivery_rate": _query_on_time_delivery_rate,
    "bad_review_rate": _query_bad_review_rate,
    "freight_rate": _query_freight_rate,
}

# Domain metrics share one trusted SQL catalog and one executor. Keep any
# dedicated direct handler above authoritative if a metric appears in both.
for _metric_name in _DOMAIN_METRIC_SQL:
    if _metric_name not in AGGREGATED_METRICS:
        DIRECT_QUERY_HANDLERS.setdefault(
            _metric_name, partial(_query_domain_metric, _metric_name)
        )


#
# 作用：查询某个指标在指定日期上的值，是上层最常用的统一单点查询入口。
# 参数：
# - metric_name: 指标名称，必须属于 `SUPPORTED_METRICS`。
# - metric_date: 待查询日期，支持字符串、`date`、`datetime`。
# 返回：
# - 标准化指标结果字典，包含指标值、来源表和状态。
def query_metric(metric_name: str, metric_date: str | date | datetime) -> dict:
    """查询某个指标在某一天的值。"""

    if metric_name not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric: {metric_name}")

    normalized = _normalize_date(metric_date)
    if metric_name in AGGREGATED_METRICS:
        return _query_from_aggregated_table(metric_name, normalized)
    return DIRECT_QUERY_HANDLERS[metric_name](normalized)


#
# 作用：查询某个指标在一段时间内的按天趋势。
# 参数：
# - metric_name: 指标名称。
# - start_date: 起始日期。
# - end_date: 结束日期，必须大于等于起始日期。
# - granularity: 时间粒度，当前仅支持 `day`。
# 返回：
# - 趋势结果字典，包含 `points` 数组、来源表和状态。
def query_metric_trend(
    metric_name: str,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    granularity: str = "day",
) -> dict:
    """查询日、ISO 周、月、季度或年度粒度的指标趋势。"""

    if metric_name not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric: {metric_name}")
    if granularity not in {"day", "week", "month", "quarter", "year"}:
        raise ValueError(f"Unsupported granularity: {granularity}")

    start = _normalize_date(start_date)
    end = _normalize_date(end_date)
    if end < start:
        raise ValueError("end_date must be greater than or equal to start_date")

    if granularity == "week" and (
        start.weekday() != 0 or end.weekday() != 6
        or ((end - start).days + 1) % 7 != 0
    ):
        raise ValueError("week trend requires complete Monday-Sunday buckets")
    if granularity == "month":
        from calendar import monthrange

        if start.day != 1 or end.day != monthrange(end.year, end.month)[1]:
            raise ValueError("month trend requires complete calendar-month buckets")

    if granularity == "day" and metric_name in AGGREGATED_METRICS:
        query = text(f"""
            SELECT metric_date AS bucket, {metric_name} AS value
            FROM analytics.daily_metrics
            WHERE metric_date BETWEEN :start_date AND :end_date
            ORDER BY metric_date
        """)
        with repository_connection(engine) as conn:
            rows = conn.execute(query, {"start_date": start, "end_date": end}).mappings().all()
        by_date = {row["bucket"]: row["value"] for row in rows}
        aggregated_points: list[dict] = []
        current = start
        while current <= end:
            raw_value = by_date.get(current)
            aggregated_points.append({
                "bucket": current.isoformat(),
                "value": round(float(raw_value), 2) if raw_value is not None else None,
            })
            current += timedelta(days=1)
        return {
            "metric_name": metric_name, "start_date": start.isoformat(), "end_date": end.isoformat(),
            "granularity": granularity, "points": aggregated_points, "source_table": "analytics.daily_metrics",
            "status": "ok" if any(point["value"] is not None for point in aggregated_points) else "not_found",
        }

    points: list[dict] = []
    current = start
    source_table = "unknown"
    while current <= end:
        if granularity == "day":
            bucket_end = current
            result = query_metric(metric_name, current)
        elif granularity == "week":
            bucket_end = current + timedelta(days=6)
            result = query_metric_period(metric_name, current, bucket_end)
        elif granularity == "month":
            from calendar import monthrange

            bucket_end = current.replace(day=monthrange(current.year, current.month)[1])
            result = query_metric_period(metric_name, current, bucket_end)
        elif granularity == "quarter":
            from calendar import monthrange
            quarter_end_month = ((current.month - 1) // 3 + 1) * 3
            bucket_end = current.replace(month=quarter_end_month, day=monthrange(current.year, quarter_end_month)[1])
            result = query_metric_period(metric_name, current, min(bucket_end, end))
        else:
            bucket_end = current.replace(month=12, day=31)
            result = query_metric_period(metric_name, current, min(bucket_end, end))
        source_table = result.get("source_table", source_table)
        points.append({"bucket": current.isoformat(), "value": result["value"]})
        current = min(bucket_end, end) + timedelta(days=1)

    has_value = any(point["value"] is not None for point in points)
    return {
        "metric_name": metric_name,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "granularity": granularity,
        "points": points,
        "source_table": source_table,
        "status": "ok" if has_value else "not_found",
    }


def query_metric_statistics(
    metric_name: str,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    granularity: str = "day",
    *,
    requested: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Return descriptive statistics over the metric's time-bucket values."""

    from app.metrics.statistics import summarize_points

    trend = query_metric_trend(metric_name, start_date, end_date, granularity)
    summary = summarize_points(trend.get("points", []), requested=requested)
    return {
        "metric_name": metric_name,
        "start_date": trend.get("start_date"),
        "end_date": trend.get("end_date"),
        "granularity": trend.get("granularity", granularity),
        "statistics": summary["statistics"],
        "count": summary["count"],
        "min_bucket": summary["min_bucket"],
        "max_bucket": summary["max_bucket"],
        "source_table": trend.get("source_table", ""),
        "status": summary["status"],
    }


def query_metric_period(
    metric_name: str,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
) -> dict:
    """Aggregate one metric over an inclusive period using entity-level facts.

    Unlike combining daily values, this query preserves exact denominators for
    ratios/averages and de-duplicates customers across the whole period.
    """

    if metric_name not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric: {metric_name}")
    start = _normalize_date(start_date)
    end = _normalize_date(end_date)
    if end < start:
        raise ValueError("end_date must be greater than or equal to start_date")

    source_table: str
    if metric_name in {
        "gmv", "order_count", "avg_order_value", "freight_value",
        "customer_count", "avg_items_per_order", "freight_rate",
    }:
        query = text("""
            SELECT SUM(item_gross_value + freight_value) AS gmv,
                   COUNT(DISTINCT order_id) AS order_count,
                   SUM(freight_value) AS freight_value,
                   COUNT(DISTINCT customer_unique_id) AS customer_count,
                   AVG(item_count) AS avg_items_per_order
            FROM analytics.fact_orders
            WHERE purchase_date BETWEEN :start_date AND :end_date
              AND order_status = 'delivered'
        """)
        source_table = "analytics.fact_orders"
    elif metric_name in {"repeat_customer_count", "repeat_purchase_rate"}:
        query = text("""
            WITH fact AS (
                SELECT customer_unique_id FROM analytics.fact_orders
                WHERE purchase_date BETWEEN :start_date AND :end_date
            ),
            prior_customers AS (
                SELECT DISTINCT customer_unique_id
                FROM analytics.fact_orders
                WHERE purchase_date < :start_date
            )
            SELECT COUNT(DISTINCT fact.customer_unique_id) AS customer_count,
                   COUNT(DISTINCT prior_customers.customer_unique_id) AS repeat_customer_count
            FROM fact
            LEFT JOIN prior_customers USING (customer_unique_id)
        """)
        source_table = "analytics.fact_orders"
    elif metric_name in {
        "cancel_rate", "delivery_delay_rate", "on_time_delivery_rate", "avg_delivery_lead_time",
    }:
        query = text("""
            SELECT COUNT(*) AS total_orders,
                   SUM(CASE WHEN order_status = 'canceled' THEN 1 ELSE 0 END) AS canceled_orders,
                   SUM(CASE WHEN order_status = 'delivered'
                             AND order_delivered_customer_date IS NOT NULL
                             AND order_estimated_delivery_date IS NOT NULL THEN 1 ELSE 0 END) AS delivered_orders,
                   SUM(CASE WHEN order_status = 'delivered'
                             AND order_delivered_customer_date > order_estimated_delivery_date THEN 1 ELSE 0 END) AS delayed_orders,
                   SUM(CASE WHEN order_status = 'delivered'
                             AND order_delivered_customer_date <= order_estimated_delivery_date THEN 1 ELSE 0 END) AS on_time_orders,
                   AVG(CASE WHEN order_status = 'delivered' AND order_delivered_customer_date IS NOT NULL
                            THEN EXTRACT(EPOCH FROM (order_delivered_customer_date - order_purchase_timestamp)) / 86400.0 END)
                       AS avg_delivery_lead_time
            FROM raw.orders
            WHERE DATE(order_purchase_timestamp) BETWEEN :start_date AND :end_date
        """)
        source_table = "raw.orders"
    elif metric_name in _PERIOD_DOMAIN_METRICS:
        # Keep period aggregation on the registered fact source and formula.
        # SUM ratios therefore aggregate numerator/denominator over the whole
        # period, while AVG metrics average period facts rather than daily AVGs.
        source, expression, date_column = _DOMAIN_METRIC_SQL[metric_name]
        query = text(f"""
            SELECT {expression} AS value
            FROM {source}
            WHERE {date_column} BETWEEN :start_date AND :end_date
        """)
        source_table = source
    elif metric_name in {"review_score_avg", "bad_review_rate"}:
        # These review metrics use an explicit review source and projection;
        # they must not act as a catch-all for other registered metrics.
        query = text("""
            SELECT AVG(r.review_score) AS review_score_avg,
                   COUNT(r.review_score) AS reviewed_orders,
                   SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END) AS bad_review_orders
            FROM raw.orders o
            JOIN raw.order_reviews r ON r.order_id = o.order_id
            WHERE DATE(o.order_purchase_timestamp) BETWEEN :start_date AND :end_date
        """)
        source_table = "raw.orders+raw.order_reviews"
    else:
        raise NotImplementedError(f"Period query is not implemented for metric: {metric_name}")
    with repository_connection(engine) as conn:
        row = conn.execute(query, {"start_date": start, "end_date": end}).mappings().first()
    data = dict(row or {})
    value: float | None
    if metric_name in _PERIOD_DOMAIN_METRICS:
        metric_value = data.get("value")
        value = float(metric_value) if metric_value is not None else None
    elif metric_name == "avg_order_value":
        denominator = float(data.get("order_count") or 0)
        value = float(data.get("gmv") or 0) / denominator if denominator else None
    elif metric_name == "cancel_rate":
        denominator = float(data.get("total_orders") or 0)
        value = float(data.get("canceled_orders") or 0) * 100 / denominator if denominator else None
    elif metric_name == "delivery_delay_rate":
        denominator = float(data.get("delivered_orders") or 0)
        value = float(data.get("delayed_orders") or 0) * 100 / denominator if denominator else None
    elif metric_name == "on_time_delivery_rate":
        denominator = float(data.get("delivered_orders") or 0)
        value = float(data.get("on_time_orders") or 0) * 100 / denominator if denominator else None
    elif metric_name == "repeat_purchase_rate":
        denominator = float(data.get("customer_count") or 0)
        value = float(data.get("repeat_customer_count") or 0) * 100 / denominator if denominator else None
    elif metric_name == "bad_review_rate":
        denominator = float(data.get("reviewed_orders") or 0)
        value = float(data.get("bad_review_orders") or 0) * 100 / denominator if denominator else None
    elif metric_name == "freight_rate":
        denominator = float(data.get("gmv") or 0)
        value = float(data.get("freight_value") or 0) * 100 / denominator if denominator else None
    else:
        metric_value = data.get(metric_name)
        value = float(metric_value) if metric_value is not None else None
    return {
        "metric_name": metric_name,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "value": round(value, 2) if value is not None else None,
        "source_table": source_table,
        "status": "ok" if value is not None else "not_found",
    }


#
# 作用：比较同一指标在两个日期之间的差值和变化比例。
# 参数：
# - metric_name: 指标名称。
# - left_date: 基准日期。
# - right_date: 对比日期。
# 返回：
# - 对比结果字典，包含左右值、差值、变化比例和状态。
def compare_metric(
    metric_name: str,
    left_date: str | date | datetime,
    right_date: str | date | datetime,
) -> dict:
    """比较同一指标在两个日期上的数值差异。"""

    if metric_name not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric: {metric_name}")

    left = query_metric(metric_name, left_date)
    right = query_metric(metric_name, right_date)
    left_value = left["value"]
    right_value = right["value"]

    delta_value = None
    delta_ratio = None
    if left_value is not None and right_value is not None:
        delta_value = round(right_value - left_value, 2)
        if left_value != 0:
            delta_ratio = round((right_value - left_value) * 100.0 / left_value, 2)

    result = {
        "metric_name": metric_name,
        "left_date": left["metric_date"],
        "right_date": right["metric_date"],
        "left_value": left_value,
        "right_value": right_value,
        "delta_value": delta_value,
        "delta_ratio": delta_ratio,
        "source_table": left["source_table"],
        # A comparison is complete only when both operands exist. Keep any
        # available side in the payload so callers can explain the limitation.
        "status": "ok" if left_value is not None and right_value is not None else "not_found",
    }
    if left.get("source_lineage") is not None:
        result["source_lineage"] = left["source_lineage"]
    return result


#
# 作用：把维度拆解 SQL 返回的中间聚合字段转换为最终指标值。
# 参数：
# - metric_name: 指标名称。
# - row: 单行 SQL 查询结果字典。
# 返回：
# - 计算后的浮点数指标值；若无法计算则返回 `None`。
def _metric_value_from_row(metric_name: str, row: dict) -> float | None:
    """把拆解 SQL 返回的中间字段转换成最终指标值。"""

    if metric_name == "cancel_rate":
        total_orders = int(row.get("total_orders") or 0)
        canceled_orders = int(row.get("canceled_orders") or 0)
        return round(canceled_orders * 100.0 / total_orders, 2) if total_orders > 0 else None

    if metric_name == "delivery_delay_rate":
        delivered_orders = int(row.get("delivered_orders") or 0)
        delayed_orders = int(row.get("delayed_orders") or 0)
        return round(delayed_orders * 100.0 / delivered_orders, 2) if delivered_orders > 0 else None

    if metric_name == "on_time_delivery_rate":
        delivered_orders = int(row.get("delivered_orders") or 0)
        on_time_orders = int(row.get("on_time_orders") or 0)
        return round(on_time_orders * 100.0 / delivered_orders, 2) if delivered_orders > 0 else None

    if metric_name == "repeat_purchase_rate":
        customer_count = int(row.get("customer_count") or 0)
        repeat_customer_count = int(row.get("repeat_customer_count") or 0)
        return round(repeat_customer_count * 100.0 / customer_count, 2) if customer_count > 0 else None

    if metric_name == "repeat_customer_count":
        value = row.get("repeat_customer_count")
        return float(value) if value is not None else None

    if metric_name == "customer_count":
        value = row.get("customer_count")
        return float(value) if value is not None else None

    if metric_name == "avg_delivery_lead_time":
        value = row.get("avg_delivery_lead_time")
        return round(float(value), 2) if value is not None else None

    if metric_name == "bad_review_rate":
        reviewed_orders = int(row.get("reviewed_orders") or 0)
        bad_review_orders = int(row.get("bad_review_orders") or 0)
        return round(bad_review_orders * 100.0 / reviewed_orders, 2) if reviewed_orders > 0 else None

    if metric_name == "freight_rate":
        gmv = float(row.get("gmv") or 0)
        freight_value = float(row.get("freight_value") or 0)
        return round(freight_value * 100.0 / gmv, 2) if gmv > 0 else None

    raw_value = row.get(metric_name)
    return round(float(raw_value), 2) if raw_value is not None else None


#
# 作用：判断某个指标在做维度拆解后是否适合继续计算占比 share。
# 参数：
# - metric_name: 指标名称。
# 返回：
# - 可计算 share 返回 `True`，否则返回 `False`。
def _is_share_supported(metric_name: str) -> bool:
    """判断某个指标拆解后是否适合计算 share。"""

    return metric_name not in {
        "cancel_rate",
        "delivery_delay_rate",
        "review_score_avg",
        "on_time_delivery_rate",
        "repeat_purchase_rate",
        "avg_delivery_lead_time",
        "bad_review_rate",
        "freight_rate",
    }


#
# 作用：为依赖订单商品行的维度生成拆解 SQL，例如卖家州和商品品类。
# 参数：
# - metric_name: 指标名称。
# - dimension_expr: SQL 中的维度字段或表达式。
# 返回：
# - 二元组 `(sql, source_table)`，分别表示查询模板和来源表名。
def _item_dimension_metric_sql(metric_name: str, dimension_expr: str) -> tuple[str, str]:
    """返回卖家/品类维度拆解所需的 SQL。"""

    if metric_name in {
        "gmv",
        "order_count",
        "avg_order_value",
        "freight_value",
        "customer_count",
        "repeat_customer_count",
        "repeat_purchase_rate",
        "avg_items_per_order",
        "avg_delivery_lead_time",
        "on_time_delivery_rate",
        "delivery_delay_rate",
        "review_score_avg",
        "bad_review_rate",
        "freight_rate",
    }:
        sql = f"""
        WITH item_dimension_orders AS (
            SELECT DISTINCT
                COALESCE({dimension_expr}, 'UNKNOWN') AS dimension_value,
                o.order_id,
                c.customer_unique_id,
                o.order_purchase_timestamp,
                o.order_delivered_customer_date,
                o.order_estimated_delivery_date,
                o.order_status,
                r.review_score
            FROM raw.orders o
            JOIN raw.order_items oi
                ON o.order_id = oi.order_id
            LEFT JOIN raw.customers c
                ON o.customer_id = c.customer_id
            LEFT JOIN raw.sellers s
                ON oi.seller_id = s.seller_id
            LEFT JOIN raw.products p
                ON oi.product_id = p.product_id
            LEFT JOIN raw.order_reviews r
                ON o.order_id = r.order_id
            WHERE DATE(o.order_purchase_timestamp) = :metric_date
        ),
        item_dimension_sales AS (
            SELECT
                COALESCE({dimension_expr}, 'UNKNOWN') AS dimension_value,
                SUM(oi.price + oi.freight_value) AS gmv,
                SUM(oi.freight_value) AS freight_value,
                COUNT(*) AS item_rows
            FROM raw.orders o
            JOIN raw.order_items oi
                ON o.order_id = oi.order_id
            LEFT JOIN raw.sellers s
                ON oi.seller_id = s.seller_id
            LEFT JOIN raw.products p
                ON oi.product_id = p.product_id
            WHERE DATE(o.order_purchase_timestamp) = :metric_date
              AND o.order_status = 'delivered'
            GROUP BY COALESCE({dimension_expr}, 'UNKNOWN')
        )
        SELECT
            odo.dimension_value,
            COUNT(DISTINCT CASE WHEN odo.order_status = 'delivered' THEN odo.order_id END) AS order_count,
            COUNT(DISTINCT CASE WHEN odo.order_status = 'delivered' THEN odo.customer_unique_id END) AS customer_count,
            COUNT(
                DISTINCT CASE
                    WHEN odo.order_status = 'delivered'
                         AND EXISTS (
                             SELECT 1
                             FROM analytics.fact_orders prev
                             WHERE prev.customer_unique_id = odo.customer_unique_id
                               AND prev.purchase_date < :metric_date
                         )
                    THEN odo.customer_unique_id
                    ELSE NULL
                END
            ) AS repeat_customer_count,
            AVG(
                CASE
                    WHEN odo.order_status = 'delivered'
                         AND odo.order_delivered_customer_date IS NOT NULL
                    THEN EXTRACT(EPOCH FROM (odo.order_delivered_customer_date - odo.order_purchase_timestamp)) / 86400.0
                    ELSE NULL
                END
            ) AS avg_delivery_lead_time,
            SUM(
                CASE
                    WHEN odo.order_status = 'delivered'
                         AND odo.order_delivered_customer_date IS NOT NULL
                         AND odo.order_estimated_delivery_date IS NOT NULL
                    THEN 1
                    ELSE 0
                END
            ) AS delivered_orders,
            SUM(
                CASE
                    WHEN odo.order_status = 'delivered'
                         AND odo.order_delivered_customer_date IS NOT NULL
                         AND odo.order_estimated_delivery_date IS NOT NULL
                         AND odo.order_delivered_customer_date > odo.order_estimated_delivery_date
                    THEN 1
                    ELSE 0
                END
            ) AS delayed_orders,
            SUM(
                CASE
                    WHEN odo.order_status = 'delivered'
                         AND odo.order_delivered_customer_date IS NOT NULL
                         AND odo.order_estimated_delivery_date IS NOT NULL
                         AND odo.order_delivered_customer_date <= odo.order_estimated_delivery_date
                    THEN 1
                    ELSE 0
                END
            ) AS on_time_orders,
            AVG(odo.review_score) AS review_score_avg,
            COUNT(odo.review_score) AS reviewed_orders,
            SUM(CASE WHEN odo.review_score <= 2 THEN 1 ELSE 0 END) AS bad_review_orders,
            COALESCE(ids.gmv, 0) AS gmv,
            COALESCE(ids.freight_value, 0) AS freight_value,
            COALESCE(ids.item_rows, 0) AS item_rows
        FROM item_dimension_orders odo
        LEFT JOIN item_dimension_sales ids
            ON odo.dimension_value = ids.dimension_value
        GROUP BY
            odo.dimension_value,
            ids.gmv,
            ids.freight_value,
            ids.item_rows
        """
        return sql, "raw.order_items"

    raise ValueError(f"Unsupported metric for item breakdown: {metric_name}")


#
# 作用：为支付方式维度生成对应的指标拆解 SQL。
# 参数：
# - metric_name: 指标名称。
# 返回：
# - 二元组 `(sql, source_table)`，分别表示查询模板和来源表名。
def _payment_dimension_metric_sql(metric_name: str) -> tuple[str, str]:
    """返回支付方式维度拆解所需的 SQL。"""

    if metric_name == "gmv":
        sql = """
        WITH order_gmv AS (
            SELECT
                fo.order_id,
                fo.purchase_date,
                SUM(oi.price + oi.freight_value)::numeric(18, 2) AS order_gmv
            FROM analytics.v_delivered_orders fo
            JOIN analytics.v_delivered_order_items oi USING (order_id)
            WHERE fo.purchase_date = :metric_date
            GROUP BY fo.order_id, fo.purchase_date
        ), payment_type_values AS (
            SELECT
                order_id,
                COALESCE(NULLIF(payment_type, ''), 'UNKNOWN') AS payment_type,
                SUM(payment_value)::numeric(18, 2) AS type_payment_value
            FROM raw.order_payments
            GROUP BY order_id, COALESCE(NULLIF(payment_type, ''), 'UNKNOWN')
        ), order_payment_values AS (
            SELECT order_id, SUM(type_payment_value)::numeric(18, 2) AS total_payment_value
            FROM payment_type_values
            GROUP BY order_id
        ), rounded_allocations AS (
            SELECT
                g.order_id,
                g.order_gmv,
                COALESCE(p.payment_type, 'UNKNOWN') AS dimension_value,
                t.total_payment_value,
                CASE
                    WHEN t.total_payment_value > 0
                    THEN ROUND(g.order_gmv * p.type_payment_value / t.total_payment_value, 2)
                    ELSE g.order_gmv
                END AS rounded_gmv,
                ROW_NUMBER() OVER (
                    PARTITION BY g.order_id
                    ORDER BY p.type_payment_value DESC NULLS LAST,
                             COALESCE(p.payment_type, 'UNKNOWN')
                ) AS allocation_rank,
                SUM(
                    CASE
                        WHEN t.total_payment_value > 0
                        THEN ROUND(g.order_gmv * p.type_payment_value / t.total_payment_value, 2)
                        ELSE g.order_gmv
                    END
                ) OVER (PARTITION BY g.order_id) AS rounded_order_gmv
            FROM order_gmv g
            LEFT JOIN order_payment_values t USING (order_id)
            LEFT JOIN payment_type_values p
                ON p.order_id = g.order_id
               AND t.total_payment_value > 0
               AND p.type_payment_value <> 0
        )
        SELECT
            dimension_value,
            SUM(
                rounded_gmv + CASE
                    WHEN allocation_rank = 1 AND total_payment_value > 0
                    THEN order_gmv - rounded_order_gmv
                    ELSE 0
                END
            ) AS gmv
        FROM rounded_allocations
        GROUP BY dimension_value
        """
        return sql, (
            "analytics.v_delivered_orders + analytics.v_delivered_order_items "
            "+ raw.order_payments"
        )

    if metric_name in {
        "order_count",
        "customer_count",
        "repeat_customer_count",
        "repeat_purchase_rate",
        "payment_value",
        "avg_delivery_lead_time",
        "on_time_delivery_rate",
        "delivery_delay_rate",
        "review_score_avg",
        "bad_review_rate",
    }:
        sql = """
        SELECT
            COALESCE(op.payment_type, 'UNKNOWN') AS dimension_value,
            SUM(op.payment_value) AS payment_value,
            COUNT(DISTINCT o.order_id) AS order_count,
            COUNT(DISTINCT c.customer_unique_id) AS customer_count,
            COUNT(
                DISTINCT CASE
                    WHEN EXISTS (
                        SELECT 1
                        FROM analytics.fact_orders prev
                        WHERE prev.customer_unique_id = c.customer_unique_id
                          AND prev.purchase_date < :metric_date
                    )
                    THEN c.customer_unique_id
                    ELSE NULL
                END
            ) AS repeat_customer_count,
            AVG(
                CASE
                    WHEN o.order_status = 'delivered'
                         AND o.order_delivered_customer_date IS NOT NULL
                    THEN EXTRACT(EPOCH FROM (o.order_delivered_customer_date - o.order_purchase_timestamp)) / 86400.0
                    ELSE NULL
                END
            ) AS avg_delivery_lead_time,
            SUM(
                CASE
                    WHEN o.order_status = 'delivered'
                         AND o.order_delivered_customer_date IS NOT NULL
                         AND o.order_estimated_delivery_date IS NOT NULL
                    THEN 1
                    ELSE 0
                END
            ) AS delivered_orders,
            SUM(
                CASE
                    WHEN o.order_status = 'delivered'
                         AND o.order_delivered_customer_date IS NOT NULL
                         AND o.order_estimated_delivery_date IS NOT NULL
                         AND o.order_delivered_customer_date > o.order_estimated_delivery_date
                    THEN 1
                    ELSE 0
                END
            ) AS delayed_orders,
            SUM(
                CASE
                    WHEN o.order_status = 'delivered'
                         AND o.order_delivered_customer_date IS NOT NULL
                         AND o.order_estimated_delivery_date IS NOT NULL
                         AND o.order_delivered_customer_date <= o.order_estimated_delivery_date
                    THEN 1
                    ELSE 0
                END
            ) AS on_time_orders,
            AVG(r.review_score) AS review_score_avg,
            COUNT(r.review_score) AS reviewed_orders,
            SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END) AS bad_review_orders
        FROM raw.orders o
        JOIN raw.order_payments op
            ON o.order_id = op.order_id
        LEFT JOIN raw.customers c
            ON o.customer_id = c.customer_id
        LEFT JOIN raw.order_reviews r
            ON o.order_id = r.order_id
        WHERE DATE(o.order_purchase_timestamp) = :metric_date
        GROUP BY COALESCE(op.payment_type, 'UNKNOWN')
        """
        return sql, "raw.order_payments"

    raise ValueError(f"Unsupported metric for payment type breakdown: {metric_name}")


#
# 派生时间维度的 SQL 表达式（{col} 由各指标簇的表名/别名决定）。
# 四个维度的指标覆盖范围与 _month_position_metric_sql 对齐：
# - fact_orders 簇：gmv, order_count, avg_order_value, freight_value, customer_count, avg_items_per_order, freight_rate
# - repeat 簇：repeat_customer_count, repeat_purchase_rate
# - raw.orders 簇：cancel_rate, delivery_delay_rate, on_time_delivery_rate, avg_delivery_lead_time
# - review 簇：review_score_avg, bad_review_rate
_DERIVED_TIME_EXPRESSIONS: dict[str, str] = {
    "day_of_week": (
        "CASE EXTRACT(ISODOW FROM {col}) "
        "WHEN 1 THEN '周一' WHEN 2 THEN '周二' WHEN 3 THEN '周三' "
        "WHEN 4 THEN '周四' WHEN 5 THEN '周五' WHEN 6 THEN '周六' "
        "ELSE '周日' END"
    ),
    "hour_of_day": "LPAD(EXTRACT(HOUR FROM {col})::text, 2, '0') || CHR(58) || '00'",
    "quarter_of_year": "'Q' || EXTRACT(QUARTER FROM {col})::text",
    "half_of_year": "CASE WHEN EXTRACT(MONTH FROM {col}) <= 6 THEN '上半年' ELSE '下半年' END",
}

# 派生维度 → fact_orders 和 raw.orders 各自使用的日期列；
# review 簇沿用 raw.orders 的列（通过别名 o. 引用）。
_DERIVED_TIME_COLUMNS: dict[str, dict[str, str]] = {
    "day_of_week": {"fact": "purchase_date", "raw": "order_purchase_timestamp"},
    "hour_of_day": {"fact": "order_purchase_timestamp", "raw": "order_purchase_timestamp"},
    "quarter_of_year": {"fact": "purchase_date", "raw": "order_purchase_timestamp"},
    "half_of_year": {"fact": "purchase_date", "raw": "order_purchase_timestamp"},
}


def _derived_time_dimension_metric_sql(metric_name: str, dimension: str) -> tuple[str, str]:
    """为派生时间维度生成按指标簇分类的完整拆解 SQL。

    day_of_week / hour_of_day / quarter_of_year / half_of_year 四个维度共用；
    month_position 因其语义特殊（固定三桶、依赖 DAY 而非 ISODOW/QUARTER 等）
    仍由独立的 _month_position_metric_sql 处理。
    """

    expr_template = _DERIVED_TIME_EXPRESSIONS[dimension]
    cols = _DERIVED_TIME_COLUMNS[dimension]
    fact_expr = expr_template.format(col=cols["fact"])
    raw_expr = expr_template.format(col=cols["raw"])
    # review 簇的 SQL 中 raw.orders 别名是 o
    review_expr = expr_template.format(col=f"o.{cols['raw']}")

    if metric_name in {
        "gmv", "order_count", "avg_order_value", "freight_value",
        "customer_count", "avg_items_per_order", "freight_rate",
    }:
        sql = f"""
        WITH dim_data AS (
            SELECT *, ({fact_expr}) AS dim_value
            FROM analytics.fact_orders
            WHERE purchase_date BETWEEN :start_date AND :end_date
              AND order_status = 'delivered'
        )
        SELECT
            dim_value AS dimension_value,
            SUM(item_gross_value + freight_value) AS gmv,
            COUNT(DISTINCT order_id) AS order_count,
            SUM(freight_value) AS freight_value,
            COUNT(DISTINCT customer_unique_id) AS customer_count,
            AVG(item_count) AS avg_items_per_order
        FROM dim_data
        GROUP BY dim_value
        """
        return sql, "analytics.fact_orders"

    if metric_name in {"repeat_customer_count", "repeat_purchase_rate"}:
        sql = f"""
        WITH dim_data AS (
            SELECT *, ({fact_expr}) AS dim_value
            FROM analytics.fact_orders
            WHERE purchase_date BETWEEN :start_date AND :end_date
        )
        SELECT
            dim_value AS dimension_value,
            COUNT(DISTINCT customer_unique_id) AS customer_count,
            COUNT(DISTINCT CASE WHEN EXISTS (
                SELECT 1 FROM analytics.fact_orders previous
                WHERE previous.customer_unique_id = dim_data.customer_unique_id
                  AND previous.purchase_date < :start_date
            ) THEN customer_unique_id END) AS repeat_customer_count
        FROM dim_data
        GROUP BY dim_value
        """
        return sql, "analytics.fact_orders"

    if metric_name in {
        "cancel_rate", "delivery_delay_rate", "on_time_delivery_rate", "avg_delivery_lead_time",
    }:
        sql = f"""
        WITH dim_data AS (
            SELECT *, ({raw_expr}) AS dim_value
            FROM raw.orders
            WHERE DATE(order_purchase_timestamp) BETWEEN :start_date AND :end_date
        )
        SELECT
            dim_value AS dimension_value,
            COUNT(*) AS total_orders,
            SUM(CASE WHEN order_status = 'canceled' THEN 1 ELSE 0 END) AS canceled_orders,
            SUM(CASE WHEN order_status = 'delivered'
                      AND order_delivered_customer_date IS NOT NULL
                      AND order_estimated_delivery_date IS NOT NULL THEN 1 ELSE 0 END) AS delivered_orders,
            SUM(CASE WHEN order_status = 'delivered'
                      AND order_delivered_customer_date > order_estimated_delivery_date THEN 1 ELSE 0 END) AS delayed_orders,
            SUM(CASE WHEN order_status = 'delivered'
                      AND order_delivered_customer_date <= order_estimated_delivery_date THEN 1 ELSE 0 END) AS on_time_orders,
            AVG(CASE WHEN order_status = 'delivered' AND order_delivered_customer_date IS NOT NULL
                     THEN EXTRACT(EPOCH FROM (order_delivered_customer_date - order_purchase_timestamp)) / 86400.0 END)
                AS avg_delivery_lead_time
        FROM dim_data
        GROUP BY dim_value
        """
        return sql, "raw.orders"

    # review_score_avg, bad_review_rate
    sql = f"""
    WITH dim_data AS (
        SELECT o.*, r.review_score, ({review_expr}) AS dim_value
        FROM raw.orders o
        LEFT JOIN raw.order_reviews r ON o.order_id = r.order_id
        WHERE DATE(o.order_purchase_timestamp) BETWEEN :start_date AND :end_date
    )
    SELECT
        dim_value AS dimension_value,
        AVG(review_score) AS review_score_avg,
        COUNT(review_score) AS reviewed_orders,
        SUM(CASE WHEN review_score <= 2 THEN 1 ELSE 0 END) AS bad_review_orders
    FROM dim_data
    GROUP BY dim_value
    """
    return sql, "raw.orders+raw.order_reviews"


#
# 作用：为月内日期分桶维度（月初/月中/月末）生成拆解 SQL。
# 参数：
# - metric_name: 指标名称。
# 返回：
# - 二元组 `(sql, source_table)`，SQL 使用 :start_date 和 :end_date 参数。
def _month_position_metric_sql(metric_name: str) -> tuple[str, str]:
    """返回月内日期分桶维度拆解所需的 SQL。"""

    month_segment_expr = """
        CASE
            WHEN EXTRACT(DAY FROM order_date) BETWEEN 1 AND 10 THEN '月初(1-10日)'
            WHEN EXTRACT(DAY FROM order_date) BETWEEN 11 AND 20 THEN '月中(11-20日)'
            ELSE '月末(21-31日)'
        END"""

    if metric_name in {
        "gmv", "order_count", "avg_order_value", "freight_value",
        "customer_count", "avg_items_per_order", "freight_rate",
    }:
        sql = f"""
        WITH month_data AS (
            SELECT *,
                {month_segment_expr.replace('order_date', 'purchase_date')} AS month_segment
            FROM analytics.fact_orders
            WHERE purchase_date BETWEEN :start_date AND :end_date
              AND order_status = 'delivered'
        )
        SELECT
            month_segment AS dimension_value,
            SUM(item_gross_value + freight_value) AS gmv,
            COUNT(DISTINCT order_id) AS order_count,
            SUM(freight_value) AS freight_value,
            COUNT(DISTINCT customer_unique_id) AS customer_count,
            AVG(item_count) AS avg_items_per_order
        FROM month_data
        GROUP BY month_segment
        """
        return sql, "analytics.fact_orders"

    if metric_name in {"repeat_customer_count", "repeat_purchase_rate"}:
        sql = f"""
        WITH month_data AS (
            SELECT *,
                {month_segment_expr.replace('order_date', 'purchase_date')} AS month_segment
            FROM analytics.fact_orders
            WHERE purchase_date BETWEEN :start_date AND :end_date
        )
        SELECT
            month_segment AS dimension_value,
            COUNT(DISTINCT customer_unique_id) AS customer_count,
            COUNT(DISTINCT CASE WHEN EXISTS (
                SELECT 1 FROM analytics.fact_orders previous
                WHERE previous.customer_unique_id = month_data.customer_unique_id
                  AND previous.purchase_date < :start_date
            ) THEN customer_unique_id END) AS repeat_customer_count
        FROM month_data
        GROUP BY month_segment
        """
        return sql, "analytics.fact_orders"

    if metric_name in {
        "cancel_rate", "delivery_delay_rate", "on_time_delivery_rate", "avg_delivery_lead_time",
    }:
        sql = f"""
        WITH month_data AS (
            SELECT *,
                {month_segment_expr.replace('order_date', 'order_purchase_timestamp')} AS month_segment
            FROM raw.orders
            WHERE DATE(order_purchase_timestamp) BETWEEN :start_date AND :end_date
        )
        SELECT
            month_segment AS dimension_value,
            COUNT(*) AS total_orders,
            SUM(CASE WHEN order_status = 'canceled' THEN 1 ELSE 0 END) AS canceled_orders,
            SUM(CASE WHEN order_status = 'delivered'
                      AND order_delivered_customer_date IS NOT NULL
                      AND order_estimated_delivery_date IS NOT NULL THEN 1 ELSE 0 END) AS delivered_orders,
            SUM(CASE WHEN order_status = 'delivered'
                      AND order_delivered_customer_date > order_estimated_delivery_date THEN 1 ELSE 0 END) AS delayed_orders,
            SUM(CASE WHEN order_status = 'delivered'
                      AND order_delivered_customer_date <= order_estimated_delivery_date THEN 1 ELSE 0 END) AS on_time_orders,
            AVG(CASE WHEN order_status = 'delivered' AND order_delivered_customer_date IS NOT NULL
                     THEN EXTRACT(EPOCH FROM (order_delivered_customer_date - order_purchase_timestamp)) / 86400.0 END)
                AS avg_delivery_lead_time
        FROM month_data
        GROUP BY month_segment
        """
        return sql, "raw.orders"

    # review_score_avg, bad_review_rate
    sql = f"""
    WITH month_data AS (
        SELECT o.*, r.review_score,
            {month_segment_expr.replace('order_date', 'o.order_purchase_timestamp')} AS month_segment
        FROM raw.orders o
        LEFT JOIN raw.order_reviews r ON o.order_id = r.order_id
        WHERE DATE(o.order_purchase_timestamp) BETWEEN :start_date AND :end_date
    )
    SELECT
        month_segment AS dimension_value,
        AVG(review_score) AS review_score_avg,
        COUNT(review_score) AS reviewed_orders,
        SUM(CASE WHEN review_score <= 2 THEN 1 ELSE 0 END) AS bad_review_orders
    FROM month_data
    GROUP BY month_segment
    """
    return sql, "raw.orders+raw.order_reviews"


#
# 作用：根据指标名与维度名选择正确的维度拆解 SQL 模板。
# 参数：
# - metric_name: 指标名称。
# - dimension: 拆解维度名称。
# 返回：
# - 二元组 `(sql, source_table)`，分别表示查询模板和来源表名。
def _breakdown_sql_for_dimension(metric_name: str, dimension: str) -> tuple[str, str]:
    """根据 metric 和维度，返回拆解查询所需的 SQL 模板和来源表。"""

    if dimension not in SUPPORTED_BREAKDOWN_DIMENSIONS:
        raise ValueError(f"Unsupported breakdown dimension: {dimension}")

    if dimension == "seller":
        # Seller identity must be resolved into CanonicalScope and executed by
        # explorer.breakdown_by_seller; this legacy platform-only repository
        # does not accept a seller authorization scope.
        raise ValueError("Seller breakdown requires the controlled seller-scope query path")

    if dimension == "month_position":
        return _month_position_metric_sql(metric_name)

    if dimension in {"seller_state", "product_category"}:
        dimension_expr = "s.seller_state" if dimension == "seller_state" else "p.product_category_name"
        return _item_dimension_metric_sql(metric_name, dimension_expr)

    if dimension == "payment_type":
        return _payment_dimension_metric_sql(metric_name)

    if metric_name in {"gmv", "order_count", "avg_order_value", "freight_value", "avg_items_per_order", "freight_rate"}:
        sql = f"""
        SELECT
            COALESCE({dimension}, 'UNKNOWN') AS dimension_value,
            SUM(item_gross_value + freight_value) AS gmv,
            COUNT(*) AS order_count,
            CASE
                WHEN COUNT(*) = 0 THEN 0
                ELSE ROUND((SUM(item_gross_value + freight_value) / COUNT(*))::numeric, 2)
            END AS avg_order_value,
            SUM(freight_value) AS freight_value,
            AVG(item_count) AS avg_items_per_order
        FROM analytics.fact_orders
        WHERE purchase_date = :metric_date AND order_status = 'delivered'
        GROUP BY COALESCE({dimension}, 'UNKNOWN')
        """
        return sql, "analytics.fact_orders"

    if metric_name == "customer_count":
        sql = f"""
        SELECT
            COALESCE({dimension}, 'UNKNOWN') AS dimension_value,
            COUNT(DISTINCT customer_unique_id) AS customer_count
        FROM analytics.fact_orders
        WHERE purchase_date = :metric_date
          AND customer_unique_id IS NOT NULL
        GROUP BY COALESCE({dimension}, 'UNKNOWN')
        """
        return sql, "analytics.fact_orders"

    if metric_name in {"repeat_customer_count", "repeat_purchase_rate"}:
        sql = f"""
        SELECT
            COALESCE(fo.{dimension}, 'UNKNOWN') AS dimension_value,
            COUNT(DISTINCT fo.customer_unique_id) AS customer_count,
            COUNT(
                DISTINCT CASE
                    WHEN EXISTS (
                        SELECT 1
                        FROM analytics.fact_orders prev
                        WHERE prev.customer_unique_id = fo.customer_unique_id
                          AND prev.purchase_date < :metric_date
                    )
                    THEN fo.customer_unique_id
                    ELSE NULL
                END
            ) AS repeat_customer_count
        FROM analytics.fact_orders fo
        WHERE fo.purchase_date = :metric_date
          AND fo.customer_unique_id IS NOT NULL
        GROUP BY COALESCE(fo.{dimension}, 'UNKNOWN')
        """
        return sql, "analytics.fact_orders"

    if metric_name == "cancel_rate":
        sql = f"""
        SELECT
            COALESCE(c.{dimension}, 'UNKNOWN') AS dimension_value,
            COUNT(*) AS total_orders,
            SUM(CASE WHEN o.order_status = 'canceled' THEN 1 ELSE 0 END) AS canceled_orders
        FROM raw.orders o
        LEFT JOIN raw.customers c
            ON o.customer_id = c.customer_id
        WHERE DATE(o.order_purchase_timestamp) = :metric_date
        GROUP BY COALESCE(c.{dimension}, 'UNKNOWN')
        """
        return sql, "raw.orders"

    if metric_name in {"delivery_delay_rate", "on_time_delivery_rate"}:
        sql = f"""
        SELECT
            COALESCE(c.{dimension}, 'UNKNOWN') AS dimension_value,
            SUM(
                CASE
                    WHEN o.order_status = 'delivered'
                         AND o.order_delivered_customer_date IS NOT NULL
                         AND o.order_estimated_delivery_date IS NOT NULL
                    THEN 1
                    ELSE 0
                END
            ) AS delivered_orders,
            SUM(
                CASE
                    WHEN o.order_status = 'delivered'
                         AND o.order_delivered_customer_date IS NOT NULL
                         AND o.order_estimated_delivery_date IS NOT NULL
                         AND o.order_delivered_customer_date > o.order_estimated_delivery_date
                    THEN 1
                    ELSE 0
                END
            ) AS delayed_orders,
            SUM(
                CASE
                    WHEN o.order_status = 'delivered'
                         AND o.order_delivered_customer_date IS NOT NULL
                         AND o.order_estimated_delivery_date IS NOT NULL
                         AND o.order_delivered_customer_date <= o.order_estimated_delivery_date
                    THEN 1
                    ELSE 0
                END
            ) AS on_time_orders
        FROM raw.orders o
        LEFT JOIN raw.customers c
            ON o.customer_id = c.customer_id
        WHERE DATE(o.order_purchase_timestamp) = :metric_date
        GROUP BY COALESCE(c.{dimension}, 'UNKNOWN')
        """
        return sql, "raw.orders"

    if metric_name == "avg_delivery_lead_time":
        sql = f"""
        SELECT
            COALESCE(c.{dimension}, 'UNKNOWN') AS dimension_value,
            AVG(EXTRACT(EPOCH FROM (o.order_delivered_customer_date - o.order_purchase_timestamp)) / 86400.0) AS avg_delivery_lead_time
        FROM raw.orders o
        LEFT JOIN raw.customers c
            ON o.customer_id = c.customer_id
        WHERE DATE(o.order_purchase_timestamp) = :metric_date
          AND o.order_status = 'delivered'
          AND o.order_delivered_customer_date IS NOT NULL
        GROUP BY COALESCE(c.{dimension}, 'UNKNOWN')
        """
        return sql, "raw.orders"

    if metric_name == "bad_review_rate":
        sql = f"""
        SELECT
            COALESCE(c.{dimension}, 'UNKNOWN') AS dimension_value,
            COUNT(r.review_score) AS reviewed_orders,
            SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END) AS bad_review_orders
        FROM raw.orders o
        LEFT JOIN raw.customers c
            ON o.customer_id = c.customer_id
        LEFT JOIN raw.order_reviews r
            ON o.order_id = r.order_id
        WHERE DATE(o.order_purchase_timestamp) = :metric_date
        GROUP BY COALESCE(c.{dimension}, 'UNKNOWN')
        """
        return sql, "raw.order_reviews"

    sql = f"""
    SELECT
        COALESCE(c.{dimension}, 'UNKNOWN') AS dimension_value,
        AVG(r.review_score) AS review_score_avg
    FROM raw.orders o
    LEFT JOIN raw.customers c
        ON o.customer_id = c.customer_id
    LEFT JOIN raw.order_reviews r
        ON o.order_id = r.order_id
    WHERE DATE(o.order_purchase_timestamp) = :metric_date
    GROUP BY COALESCE(c.{dimension}, 'UNKNOWN')
    """
    return sql, "raw.order_reviews"


#
# 作用：查询某天某指标在指定维度上的拆解结果，并按需要计算占比。
# 参数：
# - metric_name: 指标名称。
# - metric_date: 待查询日期。
# - dimension: 拆解维度，必须属于 `SUPPORTED_BREAKDOWN_DIMENSIONS`。
# - limit: 返回维度项数量上限。
# 返回：
# - 拆解结果字典，包含维度值列表、数值、占比、来源表和状态。
def breakdown_by_dimension(
    metric_name: str,
    metric_date: str | date | datetime,
    dimension: str,
    limit: int = 10,
) -> dict:
    """按维度拆解某一天的指标。"""

    if metric_name not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric: {metric_name}")
    if dimension == "seller":
        return _seller_breakdown_not_supported(metric_name)
    if dimension not in SUPPORTED_BREAKDOWN_DIMENSIONS:
        raise ValueError(f"Unsupported breakdown dimension: {dimension}")

    normalized = _normalize_date(metric_date)

    # 派生时间维度与实体维度共用同一拆解契约；传入单日时，按维度语义
    # 自动扩展到所在自然月或自然年，避免"单日的周几分布"失去意义。
    # 每个维度均通过 _derived_time_dimension_metric_sql 生成按指标簇
    # 分类的完整 SQL，覆盖全部 15 个指标（fact_orders / 复购 / 履约 / 评价）。
    if dimension in {"day_of_week", "hour_of_day", "quarter_of_year", "half_of_year"}:
        from calendar import monthrange

        if dimension in {"quarter_of_year", "half_of_year"}:
            start = normalized.replace(month=1, day=1)
            end = normalized.replace(month=12, day=31)
        else:
            start = normalized.replace(day=1)
            end = normalized.replace(day=monthrange(normalized.year, normalized.month)[1])
        sql, source_table = _derived_time_dimension_metric_sql(metric_name, dimension)
        result = _execute_breakdown(
            metric_name, dimension, sql, source_table,
            {"start_date": start, "end_date": end}, limit,
        )
        return result | {
            "metric_date": normalized.isoformat(),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        }

    # month_position 本质上是区间查询（需要整月数据），单日期自动展开为所在月份。
    if dimension == "month_position":
        from calendar import monthrange

        month_start = normalized.replace(day=1)
        month_end = normalized.replace(day=monthrange(normalized.year, normalized.month)[1])
        sql, source_table = _month_position_metric_sql(metric_name)
        result = _execute_breakdown(
            metric_name, dimension, sql, source_table,
            {"start_date": month_start, "end_date": month_end}, limit,
        )
        return result | {
            "metric_date": normalized.isoformat(),
            "start_date": month_start.isoformat(),
            "end_date": month_end.isoformat(),
        }

    sql, source_table = _breakdown_sql_for_dimension(metric_name, dimension)
    return _execute_breakdown(
        metric_name, dimension, sql, source_table, {"metric_date": normalized}, limit,
    ) | {"metric_date": normalized.isoformat()}


def _execute_breakdown(
    metric_name: str,
    dimension: str,
    sql: str,
    source_table: str,
    parameters: dict[str, date],
    limit: int,
) -> dict:
    wrapped_query = text(
        f"""
        WITH base AS (
            {sql}
        )
        SELECT *
        FROM base
        """
    )

    with repository_connection(engine) as conn:
        rows = conn.execute(wrapped_query, parameters).mappings().all()

    items: list[dict] = []
    total_value = 0.0
    value_rows: list[tuple[str, float | None]] = []

    for row in rows:
        value = _metric_value_from_row(metric_name, dict(row))

        value_rows.append((str(row["dimension_value"]), value))

        if value is not None and _is_share_supported(metric_name):
            total_value += value

    for dimension_value, value in value_rows:
        share = None
        if value is not None and total_value > 0 and _is_share_supported(metric_name):
            share = round(value * 100.0 / total_value, 2)
        items.append(
            {
                "dimension_value": dimension_value,
                "value": value,
                "share": share,
            }
        )
    items.sort(key=lambda item: abs(float(item.get("value") or 0)), reverse=True)
    items = items[:limit]

    result = {
        "metric_name": metric_name,
        "dimension": dimension,
        "items": items,
        "source_table": source_table,
        "status": "ok" if items else "not_found",
    }
    if metric_name == "gmv" and dimension == "payment_type":
        result["allocation_method"] = "order_payment_value_ratio"
        result["lineage_note"] = (
            "按每笔订单中各支付方式的 payment_value 占该订单支付总额的比例分摊 GMV；"
            "GMV 按商品金额加运费计算。缺少支付记录或支付总额为零时归入 UNKNOWN；"
            "分币舍入差额计入该订单支付金额最大的方式。"
        )
    return result


def breakdown_by_dimension_period(
    metric_name: str,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    dimension: str,
    limit: int = 10,
) -> dict:
    """Break down a metric across an inclusive period using the same entity SQL."""

    if metric_name not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric: {metric_name}")
    if dimension == "seller":
        return _seller_breakdown_not_supported(metric_name)
    start, end = _normalize_date(start_date), _normalize_date(end_date)
    if end < start:
        raise ValueError("end_date must be greater than or equal to start_date")
    sql, source_table = _breakdown_sql_for_dimension(metric_name, dimension)
    # month_position SQL 已直接使用 :start_date / :end_date，无需字符串替换。
    if dimension == "month_position":
        period_sql = sql
    else:
        period_sql = sql.replace("< :metric_date", "< :start_date").replace(
            "= :metric_date", "BETWEEN :start_date AND :end_date",
        )
    result = _execute_breakdown(
        metric_name, dimension, period_sql, source_table,
        {"start_date": start, "end_date": end}, limit,
    )
    return result | {"start_date": start.isoformat(), "end_date": end.isoformat()}


def _seller_breakdown_not_supported(metric_name: str) -> dict[str, object]:
    """Structured fail-closed response for the legacy unscoped breakdown API."""
    return {
        "metric_name": metric_name,
        "dimension": "seller",
        "items": [],
        "status": "unsupported",
        "reason_code": "seller_scope_required",
        "message": "Seller breakdown requires a server-authorized seller scope.",
    }


def compare_breakdown_by_dimension(
    metric_name: str,
    current_date: str | date | datetime,
    baseline_date: str | date | datetime,
    dimension: str,
    limit: int = 10,
) -> dict:
    """Return paired breakdown facts for change-contribution analysis."""

    if dimension == "seller":
        return _seller_breakdown_not_supported(metric_name)
    if metric_name == "order_count" and dimension == "product_category":
        return _compare_order_count_proportional_days(
            metric_name, current_date, baseline_date, limit,
        )

    current = breakdown_by_dimension(metric_name, current_date, dimension, 100000)
    baseline = breakdown_by_dimension(metric_name, baseline_date, dimension, 100000)
    current_by_value = {item["dimension_value"]: item for item in current["items"]}
    baseline_by_value = {item["dimension_value"]: item for item in baseline["items"]}
    current_total = sum(float(item.get("value") or 0) for item in current["items"])
    baseline_total = sum(float(item.get("value") or 0) for item in baseline["items"])
    ranked_values = sorted(
        current_by_value.keys() | baseline_by_value.keys(),
        key=lambda value: abs(
            float(current_by_value.get(value, {}).get("value") or 0)
            - float(baseline_by_value.get(value, {}).get("value") or 0)
        ),
        reverse=True,
    )[:limit]
    result = {
        "metric_name": metric_name,
        "dimension": dimension,
        "current_date": current["metric_date"],
        "baseline_date": baseline["metric_date"],
        "current_items": [current_by_value.get(value, {"dimension_value": value, "value": 0, "share": 0}) for value in ranked_values],
        "baseline_items": [baseline_by_value.get(value, {"dimension_value": value, "value": 0, "share": 0}) for value in ranked_values],
        "current_total": round(current_total, 2),
        "baseline_total": round(baseline_total, 2),
        "total_delta": round(current_total - baseline_total, 2),
        "source_table": current["source_table"],
        "status": "ok" if current["items"] or baseline["items"] else "not_found",
    }
    for key in ("allocation_method", "lineage_note"):
        if key in current:
            result[key] = current[key]
    return result


def compare_breakdown_by_dimension_period(
    metric_name: str,
    current_start: str | date | datetime,
    current_end: str | date | datetime,
    baseline_start: str | date | datetime,
    baseline_end: str | date | datetime,
    dimension: str,
    limit: int = 10,
) -> dict:
    """Return full-period paired facts ranked by absolute dimension delta."""

    if dimension == "seller":
        return _seller_breakdown_not_supported(metric_name)
    if metric_name == "order_count" and dimension == "product_category":
        return _compare_order_count_proportional_periods(
            metric_name, current_start, current_end,
            baseline_start, baseline_end, limit,
        )

    current = breakdown_by_dimension_period(metric_name, current_start, current_end, dimension, 100000)
    baseline = breakdown_by_dimension_period(metric_name, baseline_start, baseline_end, dimension, 100000)
    current_by_value = {item["dimension_value"]: item for item in current["items"]}
    baseline_by_value = {item["dimension_value"]: item for item in baseline["items"]}
    current_total = sum(float(item.get("value") or 0) for item in current["items"])
    baseline_total = sum(float(item.get("value") or 0) for item in baseline["items"])
    ranked_values = sorted(
        current_by_value.keys() | baseline_by_value.keys(),
        key=lambda value: abs(
            float(current_by_value.get(value, {}).get("value") or 0)
            - float(baseline_by_value.get(value, {}).get("value") or 0)
        ),
        reverse=True,
    )[:limit]
    result = {
        "metric_name": metric_name, "dimension": dimension,
        "current_period": {"start_date": current["start_date"], "end_date": current["end_date"]},
        "baseline_period": {"start_date": baseline["start_date"], "end_date": baseline["end_date"]},
        "current_items": [current_by_value.get(value, {"dimension_value": value, "value": 0, "share": 0}) for value in ranked_values],
        "baseline_items": [baseline_by_value.get(value, {"dimension_value": value, "value": 0, "share": 0}) for value in ranked_values],
        "current_total": round(current_total, 2), "baseline_total": round(baseline_total, 2),
        "total_delta": round(current_total - baseline_total, 2),
        "source_table": current["source_table"],
        "status": "ok" if current["items"] or baseline["items"] else "not_found",
    }
    for key in ("allocation_method", "lineage_note"):
        if key in current:
            result[key] = current[key]
    return result


def _order_count_allocation_rows(
    *,
    date_predicate: str,
    parameters: dict[str, object],
    dimension: str,
    reconciliation_label: str,
) -> tuple[list[tuple[str, float]], float]:
    """Allocate each delivered order equally across its product categories.

    The category set for one order is derived from the same raw item rows and
    the same delivered-order filter as the ordinary platform breakdown, so an
    order that touches several categories contributes ``1/k`` to each.
    """

    if dimension != "product_category":
        raise ValueError(f"Unsupported order allocation dimension: {dimension}")
    query = text(f"""
        WITH order_dimension AS (
            SELECT DISTINCT
                o.order_id,
                COALESCE(p.product_category_name, 'UNKNOWN') AS dimension_value
            FROM raw.orders o
            JOIN raw.order_items oi
                ON o.order_id = oi.order_id
            LEFT JOIN raw.products p
                ON oi.product_id = p.product_id
            WHERE {date_predicate}
              AND o.order_status = 'delivered'
        ),
        counted AS (
            SELECT
                order_id,
                dimension_value,
                COUNT(*) OVER (PARTITION BY order_id) AS dimension_count
            FROM order_dimension
        ),
        grouped AS (
            SELECT dimension_value, SUM(1.0 / dimension_count) AS allocated_value
            FROM counted
            GROUP BY dimension_value
        ),
        total AS (
            SELECT COUNT(DISTINCT order_id) AS total_orders FROM order_dimension
        )
        SELECT g.dimension_value, g.allocated_value, t.total_orders
        FROM grouped g CROSS JOIN total t
        ORDER BY g.allocated_value DESC, g.dimension_value ASC
    """)
    with repository_connection(engine) as conn:
        rows = conn.execute(query, parameters).mappings().all()
    values = [
        (str(row["dimension_value"]), float(row["allocated_value"] or 0))
        for row in rows
    ]
    total = float(rows[0]["total_orders"] or 0) if rows else 0.0
    require_reconciled(values, total, label=reconciliation_label)
    return values, total


def breakdown_order_count_proportional(
    metric_date: str | date | datetime,
    dimension: str = "product_category",
    limit: int = 10,
) -> dict:
    """Single-day order-count breakdown using proportional category allocation."""

    normalized = _normalize_date(metric_date)
    rows, total = _order_count_allocation_rows(
        date_predicate="DATE(o.order_purchase_timestamp) = :metric_date",
        parameters={"metric_date": normalized},
        dimension=dimension,
        reconciliation_label=f"platform {normalized.isoformat()}",
    )
    return {
        "metric_name": "order_count",
        "dimension": dimension,
        "metric_date": normalized.isoformat(),
        "items": allocated_items(rows, total=total, limit=limit),
        "population_total": round(total, 6),
        "source_table": "raw.order_items + raw.orders + raw.products",
        "status": "ok" if rows else "not_found",
        "method": ALLOCATION_METHOD,
        "allocation_method": ALLOCATION_METHOD,
        "allocation_note": ALLOCATION_NOTE,
    }


def breakdown_order_count_proportional_period(
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    dimension: str = "product_category",
    limit: int = 10,
) -> dict:
    """Period order-count breakdown using proportional category allocation."""

    start, end = _normalize_date(start_date), _normalize_date(end_date)
    if end < start:
        raise ValueError("end_date must be greater than or equal to start_date")
    rows, total = _order_count_allocation_rows(
        date_predicate="DATE(o.order_purchase_timestamp) BETWEEN :start_date AND :end_date",
        parameters={"start_date": start, "end_date": end},
        dimension=dimension,
        reconciliation_label=f"platform {start.isoformat()}..{end.isoformat()}",
    )
    return {
        "metric_name": "order_count",
        "dimension": dimension,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "items": allocated_items(rows, total=total, limit=limit),
        "population_total": round(total, 6),
        "source_table": "raw.order_items + raw.orders + raw.products",
        "status": "ok" if rows else "not_found",
        "method": ALLOCATION_METHOD,
        "allocation_method": ALLOCATION_METHOD,
        "allocation_note": ALLOCATION_NOTE,
    }


def _proportional_order_count_compare(
    metric_name: str,
    *,
    current_rows: list[tuple[str, float]],
    current_total: float,
    baseline_rows: list[tuple[str, float]],
    baseline_total: float,
    limit: int,
    current_period: dict[str, str],
    baseline_period: dict[str, str],
) -> dict:
    current_items, baseline_items = paired_allocated_items(
        current_rows, baseline_rows,
        current_total=current_total, baseline_total=baseline_total, limit=limit,
    )
    return {
        "metric_name": metric_name,
        "dimension": "product_category",
        "current_date": current_period["end_date"],
        "baseline_date": baseline_period["end_date"],
        "current_period": current_period,
        "baseline_period": baseline_period,
        "current_items": current_items,
        "baseline_items": baseline_items,
        "current_total": round(current_total, 6),
        "baseline_total": round(baseline_total, 6),
        "total_delta": round(current_total - baseline_total, 6),
        "source_table": "raw.order_items + raw.orders + raw.products",
        "status": "ok" if current_rows or baseline_rows else "not_found",
        "method": ALLOCATION_METHOD,
        "allocation_method": ALLOCATION_METHOD,
        "allocation_note": ALLOCATION_NOTE,
    }


def _compare_order_count_proportional_days(
    metric_name: str,
    current_date: str | date | datetime,
    baseline_date: str | date | datetime,
    limit: int,
) -> dict:
    current = _normalize_date(current_date)
    baseline = _normalize_date(baseline_date)
    current_rows, current_total = _order_count_allocation_rows(
        date_predicate="DATE(o.order_purchase_timestamp) = :metric_date",
        parameters={"metric_date": current},
        dimension="product_category",
        reconciliation_label=f"platform current {current.isoformat()}",
    )
    baseline_rows, baseline_total = _order_count_allocation_rows(
        date_predicate="DATE(o.order_purchase_timestamp) = :metric_date",
        parameters={"metric_date": baseline},
        dimension="product_category",
        reconciliation_label=f"platform baseline {baseline.isoformat()}",
    )
    return _proportional_order_count_compare(
        metric_name,
        current_rows=current_rows, current_total=current_total,
        baseline_rows=baseline_rows, baseline_total=baseline_total,
        limit=limit,
        current_period={"start_date": current.isoformat(), "end_date": current.isoformat()},
        baseline_period={"start_date": baseline.isoformat(), "end_date": baseline.isoformat()},
    )


def _compare_order_count_proportional_periods(
    metric_name: str,
    current_start: str | date | datetime,
    current_end: str | date | datetime,
    baseline_start: str | date | datetime,
    baseline_end: str | date | datetime,
    limit: int,
) -> dict:
    current_start, current_end = _normalize_date(current_start), _normalize_date(current_end)
    baseline_start, baseline_end = _normalize_date(baseline_start), _normalize_date(baseline_end)
    current_rows, current_total = _order_count_allocation_rows(
        date_predicate="DATE(o.order_purchase_timestamp) BETWEEN :start_date AND :end_date",
        parameters={"start_date": current_start, "end_date": current_end},
        dimension="product_category",
        reconciliation_label=f"platform current {current_start.isoformat()}..{current_end.isoformat()}",
    )
    baseline_rows, baseline_total = _order_count_allocation_rows(
        date_predicate="DATE(o.order_purchase_timestamp) BETWEEN :start_date AND :end_date",
        parameters={"start_date": baseline_start, "end_date": baseline_end},
        dimension="product_category",
        reconciliation_label=f"platform baseline {baseline_start.isoformat()}..{baseline_end.isoformat()}",
    )
    return _proportional_order_count_compare(
        metric_name,
        current_rows=current_rows, current_total=current_total,
        baseline_rows=baseline_rows, baseline_total=baseline_total,
        limit=limit,
        current_period={"start_date": current_start.isoformat(), "end_date": current_end.isoformat()},
        baseline_period={"start_date": baseline_start.isoformat(), "end_date": baseline_end.isoformat()},
    )


def compare_ratio_facts_by_dimension_period(
    numerator_metric: str,
    denominator_metric: str,
    current_start: str | date | datetime,
    current_end: str | date | datetime,
    baseline_start: str | date | datetime,
    baseline_end: str | date | datetime,
    dimension: str,
) -> dict:
    """Return complete numerator/denominator sufficient statistics by group.

    This deliberately supports only the registered freight_value / gmv ratio
    and one-row-per-order month-position buckets. It does not apply a Top-N
    limit or join item/payment tables, so callers can verify period reconciliation.
    """
    contract = SUPPORTED_RATIO_FACT_METRICS.get((numerator_metric, denominator_metric))
    if contract is None:
        raise ValueError("unsupported grouped ratio numerator/denominator pair")
    dimension_expression = SUPPORTED_RATIO_FACT_DIMENSIONS.get(dimension)
    if dimension_expression is None:
        raise ValueError(f"Unsupported grouped ratio dimension: {dimension}")

    periods = {
        "current": (_normalize_date(current_start), _normalize_date(current_end)),
        "baseline": (_normalize_date(baseline_start), _normalize_date(baseline_end)),
    }
    if any(end < start for start, end in periods.values()):
        raise ValueError("period end_date must be greater than or equal to start_date")

    groups_by_period: dict[str, dict[str, dict[str, float]]] = {}
    for label, (start, end) in periods.items():
        statement = text(f"""
            SELECT
                ({dimension_expression}) AS dimension_value,
                {contract['numerator_sql']} AS numerator,
                {contract['denominator_sql']} AS denominator
            FROM analytics.fact_orders
            WHERE purchase_date BETWEEN :start_date AND :end_date
              AND order_status = 'delivered'
            GROUP BY ({dimension_expression})
            ORDER BY ({dimension_expression})
        """)
        with repository_connection(engine) as conn:
            rows = conn.execute(statement, {"start_date": start, "end_date": end}).mappings().all()
        groups: dict[str, dict[str, float]] = {}
        for row in rows:
            group = str(row["dimension_value"])
            if group in groups:
                raise ValueError(f"duplicate grouped ratio key: {group}")
            numerator = float(row["numerator"] or 0.0)
            denominator = float(row["denominator"] or 0.0)
            groups[group] = {"numerator": numerator, "denominator": denominator}
        groups_by_period[label] = groups

    current = groups_by_period["current"]
    baseline = groups_by_period["baseline"]
    return {
        "metric_name": contract["metric_name"],
        "numerator_metric": numerator_metric,
        "denominator_metric": denominator_metric,
        "dimension": dimension,
        "current_period": {"start_date": periods["current"][0].isoformat(), "end_date": periods["current"][1].isoformat()},
        "baseline_period": {"start_date": periods["baseline"][0].isoformat(), "end_date": periods["baseline"][1].isoformat()},
        "current_groups": current,
        "baseline_groups": baseline,
        "current_numerator_total": sum(item["numerator"] for item in current.values()),
        "current_denominator_total": sum(item["denominator"] for item in current.values()),
        "baseline_numerator_total": sum(item["numerator"] for item in baseline.values()),
        "baseline_denominator_total": sum(item["denominator"] for item in baseline.values()),
        "complete": bool(current) and bool(baseline),
        "source_table": "analytics.fact_orders",
        "status": "ok" if current and baseline else "not_found",
    }


#
# 作用：查询指定日期的 GMV。
# 参数：
# - metric_date: 待查询日期。
# 返回：
# - GMV 指标结果字典。
def query_gmv(metric_date: str | date | datetime) -> dict:
    """查询某一天的 GMV。"""

    return query_metric("gmv", metric_date)


#
# 作用：查询指定日期的订单量。
# 参数：
# - metric_date: 待查询日期。
# 返回：
# - 订单量指标结果字典。
def query_order_count(metric_date: str | date | datetime) -> dict:
    """查询某一天的订单量。"""

    return query_metric("order_count", metric_date)


#
# 作用：查询指定日期的客单价。
# 参数：
# - metric_date: 待查询日期。
# 返回：
# - 客单价指标结果字典。
def query_avg_order_value(metric_date: str | date | datetime) -> dict:
    """查询某一天的客单价。"""

    return query_metric("avg_order_value", metric_date)


#
# 作用：查询指定日期的取消率。
# 参数：
# - metric_date: 待查询日期。
# 返回：
# - 取消率指标结果字典。
def query_cancel_rate(metric_date: str | date | datetime) -> dict:
    """查询某一天的取消率。"""

    return query_metric("cancel_rate", metric_date)


#
# 作用：查询指定日期的运费总额。
# 参数：
# - metric_date: 待查询日期。
# 返回：
# - 运费总额指标结果字典。
def query_freight_value(metric_date: str | date | datetime) -> dict:
    """查询某一天的运费总额。"""

    return query_metric("freight_value", metric_date)


#
# 作用：查询指定日期的平均评论分。
# 参数：
# - metric_date: 待查询日期。
# 返回：
# - 平均评论分指标结果字典。
def query_review_score_avg(metric_date: str | date | datetime) -> dict:
    """查询某一天订单对应的平均评论分。"""

    return query_metric("review_score_avg", metric_date)


#
# 作用：查询指定日期的延迟送达率。
# 参数：
# - metric_date: 待查询日期。
# 返回：
# - 延迟送达率指标结果字典。
def query_delivery_delay_rate(metric_date: str | date | datetime) -> dict:
    """查询某一天的延迟送达率。"""

    return query_metric("delivery_delay_rate", metric_date)


#
# 作用：查询指定日期的成交客户数。
# 参数：
# - metric_date: 待查询日期。
# 返回：
# - 成交客户数指标结果字典。
def query_customer_count(metric_date: str | date | datetime) -> dict:
    """查询某一天的成交客户数。"""

    return query_metric("customer_count", metric_date)


#
# 作用：查询指定日期的复购客户数。
# 参数：
# - metric_date: 待查询日期。
# 返回：
# - 复购客户数指标结果字典。
def query_repeat_customer_count(metric_date: str | date | datetime) -> dict:
    """查询某一天的复购客户数。"""

    return query_metric("repeat_customer_count", metric_date)


#
# 作用：查询指定日期的复购率。
# 参数：
# - metric_date: 待查询日期。
# 返回：
# - 复购率指标结果字典。
def query_repeat_purchase_rate(metric_date: str | date | datetime) -> dict:
    """查询某一天的复购率。"""

    return query_metric("repeat_purchase_rate", metric_date)


#
# 作用：查询指定日期的平均每单件数。
# 参数：
# - metric_date: 待查询日期。
# 返回：
# - 平均每单件数指标结果字典。
def query_avg_items_per_order(metric_date: str | date | datetime) -> dict:
    """查询某一天的平均每单件数。"""

    return query_metric("avg_items_per_order", metric_date)


#
# 作用：查询指定日期的平均送达时长。
# 参数：
# - metric_date: 待查询日期。
# 返回：
# - 平均送达时长指标结果字典。
def query_avg_delivery_lead_time(metric_date: str | date | datetime) -> dict:
    """查询某一天的平均送达时长。"""

    return query_metric("avg_delivery_lead_time", metric_date)


#
# 作用：查询指定日期的准时送达率。
# 参数：
# - metric_date: 待查询日期。
# 返回：
# - 准时送达率指标结果字典。
def query_on_time_delivery_rate(metric_date: str | date | datetime) -> dict:
    """查询某一天的准时送达率。"""

    return query_metric("on_time_delivery_rate", metric_date)


#
# 作用：查询指定日期的差评率。
# 参数：
# - metric_date: 待查询日期。
# 返回：
# - 差评率指标结果字典。
def query_bad_review_rate(metric_date: str | date | datetime) -> dict:
    """查询某一天的差评率。"""

    return query_metric("bad_review_rate", metric_date)


#
# 作用：查询指定日期的运费率。
# 参数：
# - metric_date: 待查询日期。
# 返回：
# - 运费率指标结果字典。
def query_freight_rate(metric_date: str | date | datetime) -> dict:
    """查询某一天的运费率。"""

    return query_metric("freight_rate", metric_date)

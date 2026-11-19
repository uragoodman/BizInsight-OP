"""经营指标查询导出模块。

该文件本身不承载复杂计算逻辑，主要作用是把 `daily_metrics` 中的常用查询函数
统一重导出，方便上层按单一入口导入指标能力。
"""

from app.data.repositories.daily_metrics import (
    breakdown_by_dimension,
    compare_breakdown_by_dimension,
    breakdown_by_dimension_period,
    compare_breakdown_by_dimension_period,
    compare_metric,
    query_avg_order_value,
    query_avg_delivery_lead_time,
    query_avg_items_per_order,
    query_bad_review_rate,
    query_cancel_rate,
    query_customer_count,
    query_delivery_delay_rate,
    query_freight_value,
    query_freight_rate,
    query_gmv,
    query_metric,
    query_metric_trend,
    query_metric_period,
    query_metric_statistics,
    query_on_time_delivery_rate,
    query_order_count,
    query_repeat_customer_count,
    query_repeat_purchase_rate,
    query_review_score_avg,
)

__all__ = [
    "query_metric",
    "query_metric_trend",
    "query_metric_period",
    "query_metric_statistics",
    "compare_metric",
    "breakdown_by_dimension",
    "compare_breakdown_by_dimension",
    "breakdown_by_dimension_period",
    "compare_breakdown_by_dimension_period",
    "query_gmv",
    "query_order_count",
    "query_avg_order_value",
    "query_customer_count",
    "query_repeat_customer_count",
    "query_repeat_purchase_rate",
    "query_avg_items_per_order",
    "query_avg_delivery_lead_time",
    "query_on_time_delivery_rate",
    "query_bad_review_rate",
    "query_freight_rate",
    "query_cancel_rate",
    "query_freight_value",
    "query_review_score_avg",
    "query_delivery_delay_rate",
]

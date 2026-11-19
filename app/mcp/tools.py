"""MCP 工具层。

这里负责两件事：
- 维护当前有哪些 MCP tools
- 把工具调用转发到 repository 层
"""

from __future__ import annotations

from typing import Literal, cast

from app.data.repositories.daily_metrics import (
    breakdown_by_dimension as repository_breakdown_by_dimension,
    compare_breakdown_by_dimension as repository_compare_breakdown_by_dimension,
    compare_breakdown_by_dimension_period as repository_compare_breakdown_by_dimension_period,
    compare_ratio_facts_by_dimension_period as repository_compare_ratio_facts_by_dimension_period,
    compare_metric as repository_compare_metric,
    query_avg_delivery_lead_time as repository_query_avg_delivery_lead_time,
    query_avg_items_per_order as repository_query_avg_items_per_order,
    query_avg_order_value as repository_query_avg_order_value,
    query_bad_review_rate as repository_query_bad_review_rate,
    query_cancel_rate as repository_query_cancel_rate,
    query_customer_count as repository_query_customer_count,
    query_delivery_delay_rate as repository_query_delivery_delay_rate,
    query_freight_rate as repository_query_freight_rate,
    query_freight_value as repository_query_freight_value,
    query_gmv as repository_query_gmv,
    query_metric_trend as repository_query_metric_trend,
    query_metric_statistics as repository_query_metric_statistics,
    query_metric_period as repository_query_metric_period,
    query_on_time_delivery_rate as repository_query_on_time_delivery_rate,
    query_order_count as repository_query_order_count,
    query_repeat_customer_count as repository_query_repeat_customer_count,
    query_repeat_purchase_rate as repository_query_repeat_purchase_rate,
    query_review_score_avg as repository_query_review_score_avg,
)
from app.data.repositories.multidimensional import (
    query_metric_by_dimensions as repository_query_metric_by_dimensions,
)
from app.mcp.schemas import (
    MetricBatchPeriodArguments,
    MetricBreakdownArguments,
    MetricMultiDimensionBreakdownArguments,
    MetricCompareArguments,
    MetricContributionArguments,
    MetricPeriodContributionArguments,
    FreightRateRatioFactsArguments,
    MetricToolArguments,
    MetricTrendArguments,
    MetricStatisticsArguments,
    RegisteredRatioFactsArguments,
)
from app.core.config import get_settings
from app.analysis.contracts import PeriodSpec
from app.analysis.periods import compare_periods as compare_metric_periods, previous_period, validate_natural_period, year_ago_period
from app.analysis.semantic import get_metric_definition, list_metric_definitions
from app.resilience import execute_with_resilience


TOOL_REGISTRY: dict[str, dict[str, str]] = {
    "query_gmv": {"metric_name": "gmv", "description": "查询指定日期的 GMV（成交总额）。"},
    "query_order_count": {"metric_name": "order_count", "description": "查询指定日期的订单量。"},
    "query_avg_order_value": {"metric_name": "avg_order_value", "description": "查询指定日期的客单价。"},
    "query_cancel_rate": {"metric_name": "cancel_rate", "description": "查询指定日期的取消率。"},
    "query_freight_value": {"metric_name": "freight_value", "description": "查询指定日期的运费总额。"},
    "query_review_score_avg": {"metric_name": "review_score_avg", "description": "查询指定日期订单对应的平均评论分。"},
    "query_delivery_delay_rate": {"metric_name": "delivery_delay_rate", "description": "查询指定日期的延迟送达率。"},
    "query_customer_count": {"metric_name": "customer_count", "description": "查询指定日期的成交客户数。"},
    "query_repeat_customer_count": {"metric_name": "repeat_customer_count", "description": "查询指定日期的复购客户数。"},
    "query_repeat_purchase_rate": {"metric_name": "repeat_purchase_rate", "description": "查询指定日期的复购率。"},
    "query_avg_items_per_order": {"metric_name": "avg_items_per_order", "description": "查询指定日期的平均每单件数。"},
    "query_avg_delivery_lead_time": {"metric_name": "avg_delivery_lead_time", "description": "查询指定日期的平均送达时长。"},
    "query_on_time_delivery_rate": {"metric_name": "on_time_delivery_rate", "description": "查询指定日期的准时送达率。"},
    "query_bad_review_rate": {"metric_name": "bad_review_rate", "description": "查询指定日期的差评率。"},
    "query_freight_rate": {"metric_name": "freight_rate", "description": "查询指定日期的运费率。"},
    "query_metric_trend": {"metric_name": "multi", "description": "查询指标在一段时间内的趋势。"},
    "query_metric_statistics": {"metric_name": "multi", "description": "查询指标的平均值、最大值、最小值、中位数等描述性统计。"},
    "query_scoped_order_freight_distribution": {"metric_name": "seller_avg_freight_per_order", "description": "在受控店铺和日期范围内按已交付订单实体汇总每单运费分布；不返回订单标识。"},
    "query_scoped_registered_association": {"metric_name": "multi", "description": "执行与当前授权指标匹配的服务端注册变量对；返回样本量、效应量和不确定区间，不作因果解释。"},
    "compare_metric": {"metric_name": "multi", "description": "比较同一指标在两个日期上的差异。"},
    "breakdown_by_dimension": {"metric_name": "multi", "description": "按维度拆解某一天的指标。"},
    "query_metric_by_dimensions": {"metric_name": "multi", "description": "按两个到四个精确可观测的分组维度交叉汇总指标；不做跨粒度分摊。"},
    "compare_breakdown_by_dimension": {"metric_name": "multi", "description": "比较两个日期的维度拆解，用于变化贡献度。"},
    "compare_breakdown_by_dimension_period": {"metric_name": "multi", "description": "比较两个自然周期的维度变化贡献。"},
    "compare_freight_rate_ratio_facts_by_dimension_period": {"metric_name": "freight_rate", "description": "获取平台运费率按月内日期位置拆解所需的完整分子/分母事实；不计算因果效应。"},
    "compare_avg_order_value_ratio_facts_by_dimension_period": {"metric_name": "avg_order_value", "description": "获取平台客单价按月内日期位置拆解所需的完整分子/分母事实；不计算因果效应。"},
    "query_metric_period": {"metric_name": "multi", "description": "查询指标的周/月周期聚合和血缘。"},
    "compare_metric_period": {"metric_name": "multi", "description": "查询指标的周期环比或同比。"},
    "query_metric_batch_period": {"metric_name": "multi", "description": "批量查询自然周或自然月指标。"},
    "list_metric_definitions": {"metric_name": "semantic", "description": "列出可执行指标语义定义。"},
    "get_metric_definition": {"metric_name": "semantic", "description": "查询一个指标的语义定义与字段血缘。"},
}


def get_supported_tools() -> list[str]:
    """按稳定顺序返回当前已经实现的 MCP 工具名。"""

    return list(TOOL_REGISTRY.keys())


def list_semantic_metric_definitions() -> list[dict[str, object]]:
    return list_metric_definitions()


def get_semantic_metric_definition(metric_name: str) -> dict[str, object]:
    definition = get_metric_definition(metric_name)
    return definition.__dict__ | {"lineage_ref": definition.lineage_ref}


def _normalize_metric_date(metric_date: str) -> MetricToolArguments:
    """校验单日指标查询的入参。"""

    return MetricToolArguments(metric_date=metric_date)


def _database_read(operation, *args, **kwargs):
    """All MCP data tools are read-only and therefore safe for bounded retry."""
    return execute_with_resilience(
        lambda: operation(*args, **kwargs), "postgresql", timeout_seconds=get_settings().database_timeout_seconds,
    )


def query_gmv(metric_date: str) -> dict:
    args = _normalize_metric_date(metric_date)
    return _database_read(repository_query_gmv, args.metric_date)


def query_order_count(metric_date: str) -> dict:
    args = _normalize_metric_date(metric_date)
    return _database_read(repository_query_order_count, args.metric_date)


def query_avg_order_value(metric_date: str) -> dict:
    args = _normalize_metric_date(metric_date)
    return _database_read(repository_query_avg_order_value, args.metric_date)


def query_cancel_rate(metric_date: str) -> dict:
    args = _normalize_metric_date(metric_date)
    return _database_read(repository_query_cancel_rate, args.metric_date)


def query_freight_value(metric_date: str) -> dict:
    args = _normalize_metric_date(metric_date)
    return _database_read(repository_query_freight_value, args.metric_date)


def query_review_score_avg(metric_date: str) -> dict:
    args = _normalize_metric_date(metric_date)
    return _database_read(repository_query_review_score_avg, args.metric_date)


def query_delivery_delay_rate(metric_date: str) -> dict:
    args = _normalize_metric_date(metric_date)
    return _database_read(repository_query_delivery_delay_rate, args.metric_date)


def query_customer_count(metric_date: str) -> dict:
    args = _normalize_metric_date(metric_date)
    return _database_read(repository_query_customer_count, args.metric_date)


def query_repeat_customer_count(metric_date: str) -> dict:
    args = _normalize_metric_date(metric_date)
    return _database_read(repository_query_repeat_customer_count, args.metric_date)


def query_repeat_purchase_rate(metric_date: str) -> dict:
    args = _normalize_metric_date(metric_date)
    return _database_read(repository_query_repeat_purchase_rate, args.metric_date)


def query_avg_items_per_order(metric_date: str) -> dict:
    args = _normalize_metric_date(metric_date)
    return _database_read(repository_query_avg_items_per_order, args.metric_date)


def query_avg_delivery_lead_time(metric_date: str) -> dict:
    args = _normalize_metric_date(metric_date)
    return _database_read(repository_query_avg_delivery_lead_time, args.metric_date)


def query_on_time_delivery_rate(metric_date: str) -> dict:
    args = _normalize_metric_date(metric_date)
    return _database_read(repository_query_on_time_delivery_rate, args.metric_date)


def query_bad_review_rate(metric_date: str) -> dict:
    args = _normalize_metric_date(metric_date)
    return _database_read(repository_query_bad_review_rate, args.metric_date)


def query_freight_rate(metric_date: str) -> dict:
    args = _normalize_metric_date(metric_date)
    return _database_read(repository_query_freight_rate, args.metric_date)


def query_metric_trend(metric_name: str, start_date: str, end_date: str, granularity: str = "day") -> dict:
    """MCP 工具：查询指标趋势。"""

    args = MetricTrendArguments.model_validate({
        "metric_name": metric_name, "start_date": start_date,
        "end_date": end_date, "granularity": granularity,
    })
    return _database_read(repository_query_metric_trend, args.metric_name, args.start_date, args.end_date, args.granularity)


def query_metric_statistics(
    metric_name: str,
    start_date: str,
    end_date: str,
    granularity: str = "day",
    statistics: list[str] | None = None,
) -> dict:
    """MCP 工具：对已限定范围内的指标时间点计算描述性统计。"""

    args = MetricStatisticsArguments.model_validate({
        "metric_name": metric_name,
        "start_date": start_date,
        "end_date": end_date,
        "granularity": granularity,
        "statistics": statistics or [],
    })
    return _database_read(
        repository_query_metric_statistics,
        args.metric_name,
        args.start_date,
        args.end_date,
        args.granularity,
        requested=args.statistics,
    )


def compare_metric(metric_name: str, left_date: str, right_date: str) -> dict:
    """MCP 工具：比较两个日期上的同一指标。"""

    args = MetricCompareArguments.model_validate({
        "metric_name": metric_name, "left_date": left_date, "right_date": right_date,
    })
    return _database_read(repository_compare_metric, args.metric_name, args.left_date, args.right_date)


def breakdown_by_dimension(metric_name: str, metric_date: str, dimension: str, limit: int = 10) -> dict:
    """MCP 工具：按维度拆解某日指标。"""

    args = MetricBreakdownArguments.model_validate({
        "metric_name": metric_name, "metric_date": metric_date,
        "dimension": dimension, "limit": limit,
    })
    return _database_read(repository_breakdown_by_dimension, args.metric_name, args.metric_date, args.dimension, args.limit)


def query_metric_by_dimensions(
    metric_name: str,
    start_date: str,
    end_date: str,
    dimensions: list[str],
    limit: int = 100,
) -> dict:
    """MCP tool: exactly group a registered metric by two to four dimensions."""
    args = MetricMultiDimensionBreakdownArguments.model_validate({
        "metric_name": metric_name,
        "start_date": start_date,
        "end_date": end_date,
        "dimensions": dimensions,
        "limit": limit,
    })
    return _database_read(
        repository_query_metric_by_dimensions,
        args.metric_name,
        args.start_date,
        args.end_date,
        args.dimensions,
        args.limit,
    )


def compare_breakdown_by_dimension(metric_name: str, current_date: str, baseline_date: str, dimension: str, limit: int = 10) -> dict:
    args = MetricContributionArguments.model_validate({
        "metric_name": metric_name, "current_date": current_date, "baseline_date": baseline_date,
        "dimension": dimension, "limit": limit,
    })
    return _database_read(
        repository_compare_breakdown_by_dimension, args.metric_name, args.current_date,
        args.baseline_date, args.dimension, args.limit,
    )


def compare_breakdown_by_dimension_period(
    metric_name: str, current_start: str, current_end: str,
    baseline_start: str, baseline_end: str, granularity: str,
    dimension: str, limit: int = 10,
) -> dict:
    args = MetricPeriodContributionArguments.model_validate({
        "metric_name": metric_name, "current_start": current_start, "current_end": current_end,
        "baseline_start": baseline_start, "baseline_end": baseline_end, "granularity": granularity,
        "dimension": dimension, "limit": limit,
    })
    current = PeriodSpec.model_validate({
        "start_date": args.current_start, "end_date": args.current_end, "granularity": args.granularity,
    })
    baseline = PeriodSpec.model_validate({
        "start_date": args.baseline_start, "end_date": args.baseline_end, "granularity": args.granularity,
    })
    validate_natural_period(current)
    validate_natural_period(baseline)
    return _database_read(
        repository_compare_breakdown_by_dimension_period, args.metric_name,
        current.start_date, current.end_date, baseline.start_date, baseline.end_date,
        args.dimension, args.limit,
    )


def compare_freight_rate_ratio_facts_by_dimension_period(
    metric_name: str, current_start: str, current_end: str,
    baseline_start: str, baseline_end: str, granularity: str,
    dimension: str,
) -> dict:
    """Return complete, registry-backed facts for the registered ratio method."""
    args = FreightRateRatioFactsArguments.model_validate({
        "metric_name": metric_name,
        "current_start": current_start,
        "current_end": current_end,
        "baseline_start": baseline_start,
        "baseline_end": baseline_end,
        "granularity": granularity,
        "dimension": dimension,
    })
    current = PeriodSpec.model_validate({
        "start_date": args.current_start, "end_date": args.current_end,
        "granularity": args.granularity,
    })
    baseline = PeriodSpec.model_validate({
        "start_date": args.baseline_start, "end_date": args.baseline_end,
        "granularity": args.granularity,
    })
    validate_natural_period(current)
    validate_natural_period(baseline)
    from app.metrics.registry import get_registry

    definition = get_registry().get(args.metric_name)
    capability = definition.attribution.for_scope("platform")
    aggregation = definition.period_aggregation
    if (
        "symmetric_mix_within_v1" not in capability.method_ids
        or args.dimension not in capability.dimensions
        or aggregation is None
        or aggregation.kind != "weighted_ratio"
        or (aggregation.numerator_metric, aggregation.denominator_metric) != ("freight_value", "gmv")
    ):
        raise ValueError("registered freight-rate grouped attribution capability is unavailable")
    return _database_read(
        repository_compare_ratio_facts_by_dimension_period,
        aggregation.numerator_metric, aggregation.denominator_metric,
        current.start_date, current.end_date,
        baseline.start_date, baseline.end_date, args.dimension,
    )


def compare_avg_order_value_ratio_facts_by_dimension_period(
    metric_name: str, current_start: str, current_end: str,
    baseline_start: str, baseline_end: str, granularity: str,
    dimension: str,
) -> dict:
    """Return complete, Registry-backed AOV sufficient facts by month position."""
    args = RegisteredRatioFactsArguments.model_validate({
        "metric_name": metric_name,
        "current_start": current_start,
        "current_end": current_end,
        "baseline_start": baseline_start,
        "baseline_end": baseline_end,
        "granularity": granularity,
        "dimension": dimension,
    })
    current = PeriodSpec.model_validate({
        "start_date": args.current_start, "end_date": args.current_end,
        "granularity": args.granularity,
    })
    baseline = PeriodSpec.model_validate({
        "start_date": args.baseline_start, "end_date": args.baseline_end,
        "granularity": args.granularity,
    })
    validate_natural_period(current)
    validate_natural_period(baseline)
    from app.metrics.registry import get_registry

    definition = get_registry().get(args.metric_name)
    capability = definition.attribution.for_scope("platform")
    aggregation = definition.period_aggregation
    if (
        args.metric_name != "avg_order_value"
        or "symmetric_mix_within_v1" not in capability.method_ids
        or args.dimension not in capability.dimensions
        or aggregation is None
        or aggregation.kind != "weighted_ratio"
        or (aggregation.numerator_metric, aggregation.denominator_metric) != ("gmv", "order_count")
    ):
        raise ValueError("registered average-order-value grouped attribution capability is unavailable")
    return _database_read(
        repository_compare_ratio_facts_by_dimension_period,
        aggregation.numerator_metric, aggregation.denominator_metric,
        current.start_date, current.end_date,
        baseline.start_date, baseline.end_date, args.dimension,
    )


def query_metric_period(metric_name: str, start_date: str, end_date: str, granularity: str) -> dict:
    period = PeriodSpec.model_validate({"start_date": start_date, "end_date": end_date, "granularity": granularity})
    validate_natural_period(period)
    definition = get_metric_definition(metric_name)
    result = _database_read(repository_query_metric_period, metric_name, period.start_date, period.end_date)
    return {
        "metric_name": metric_name, "period": period.model_dump(mode="json"),
        "value": result["value"], "unit": definition.unit,
        "source_refs": [str(result.get("source_table", definition.source_table))],
        "definition_version": definition.version, "lineage_ref": definition.lineage_ref,
        "status": "ok" if result["value"] is not None else "not_found",
    }


def query_metric_batch_period(metric_names: list[str], start_date: str, end_date: str, granularity: str) -> dict:
    args = MetricBatchPeriodArguments.model_validate({
        "metric_names": metric_names, "start_date": start_date,
        "end_date": end_date, "granularity": granularity,
    })
    period = PeriodSpec.model_validate({
        "start_date": args.start_date, "end_date": args.end_date, "granularity": args.granularity,
    })
    validate_natural_period(period)
    items = [
        query_metric_period(metric_name, args.start_date, args.end_date, args.granularity)
        for metric_name in args.metric_names
    ]
    return {
        "period": period.model_dump(mode="json"), "items": items,
        "status": "ok" if any(item["status"] == "ok" for item in items) else "not_found",
    }


def compare_metric_period(metric_name: str, start_date: str, end_date: str, granularity: str, comparison_type: str) -> dict:
    period = PeriodSpec.model_validate({"start_date": start_date, "end_date": end_date, "granularity": granularity})
    if comparison_type == "period_over_period":
        baseline = previous_period(period)
    elif comparison_type == "year_over_year":
        baseline = year_ago_period(period)
    else:
        raise ValueError("comparison_type must be period_over_period or year_over_year")
    resolved_type = cast(Literal["period_over_period", "year_over_year"], comparison_type)
    return compare_metric_periods(metric_name, period, baseline, resolved_type).model_dump(mode="json")

"""BizInsight 本地 MCP Server。"""

from __future__ import annotations

import argparse

from mcp.server.fastmcp import FastMCP

from app.mcp.tools import (
    breakdown_by_dimension,
    query_metric_by_dimensions,
    compare_breakdown_by_dimension,
    compare_breakdown_by_dimension_period,
    compare_avg_order_value_ratio_facts_by_dimension_period,
    compare_freight_rate_ratio_facts_by_dimension_period,
    compare_metric_period,
    compare_metric,
    query_avg_delivery_lead_time,
    query_avg_items_per_order,
    query_avg_order_value,
    query_bad_review_rate,
    query_cancel_rate,
    query_customer_count,
    query_delivery_delay_rate,
    query_freight_rate,
    query_freight_value,
    query_gmv,
    query_metric_trend,
    query_metric_statistics,
    query_metric_period,
    query_metric_batch_period,
    list_semantic_metric_definitions,
    get_semantic_metric_definition,
    query_on_time_delivery_rate,
    query_order_count,
    query_repeat_customer_count,
    query_repeat_purchase_rate,
    query_review_score_avg,
)


SERVER_NAME = "bizdata-mcp-server"


def create_mcp_server() -> FastMCP:
    """构建并注册 BizInsight 当前可用的 MCP 工具。"""

    server = FastMCP(
        name=SERVER_NAME,
        instructions="暴露基于 BizInsight analytics repository 的经营分析指标查询工具。",
    )

    @server.tool(name="query_gmv", description="查询指定日期的 GMV。")
    def query_gmv_tool(metric_date: str) -> dict:
        return query_gmv(metric_date)

    @server.tool(name="query_order_count", description="查询指定日期的订单量。")
    def query_order_count_tool(metric_date: str) -> dict:
        return query_order_count(metric_date)

    @server.tool(name="query_avg_order_value", description="查询指定日期的客单价。")
    def query_avg_order_value_tool(metric_date: str) -> dict:
        return query_avg_order_value(metric_date)

    @server.tool(name="query_cancel_rate", description="查询指定日期的取消率。")
    def query_cancel_rate_tool(metric_date: str) -> dict:
        return query_cancel_rate(metric_date)

    @server.tool(name="query_freight_value", description="查询指定日期的运费总额。")
    def query_freight_value_tool(metric_date: str) -> dict:
        return query_freight_value(metric_date)

    @server.tool(name="query_review_score_avg", description="查询指定日期的平均评论分。")
    def query_review_score_avg_tool(metric_date: str) -> dict:
        return query_review_score_avg(metric_date)

    @server.tool(name="query_delivery_delay_rate", description="查询指定日期的延迟送达率。")
    def query_delivery_delay_rate_tool(metric_date: str) -> dict:
        return query_delivery_delay_rate(metric_date)

    @server.tool(name="query_customer_count", description="查询指定日期的成交客户数。")
    def query_customer_count_tool(metric_date: str) -> dict:
        return query_customer_count(metric_date)

    @server.tool(name="query_repeat_customer_count", description="查询指定日期的复购客户数。")
    def query_repeat_customer_count_tool(metric_date: str) -> dict:
        return query_repeat_customer_count(metric_date)

    @server.tool(name="query_repeat_purchase_rate", description="查询指定日期的复购率。")
    def query_repeat_purchase_rate_tool(metric_date: str) -> dict:
        return query_repeat_purchase_rate(metric_date)

    @server.tool(name="query_avg_items_per_order", description="查询指定日期的平均每单件数。")
    def query_avg_items_per_order_tool(metric_date: str) -> dict:
        return query_avg_items_per_order(metric_date)

    @server.tool(name="query_avg_delivery_lead_time", description="查询指定日期的平均送达时长。")
    def query_avg_delivery_lead_time_tool(metric_date: str) -> dict:
        return query_avg_delivery_lead_time(metric_date)

    @server.tool(name="query_on_time_delivery_rate", description="查询指定日期的准时送达率。")
    def query_on_time_delivery_rate_tool(metric_date: str) -> dict:
        return query_on_time_delivery_rate(metric_date)

    @server.tool(name="query_bad_review_rate", description="查询指定日期的差评率。")
    def query_bad_review_rate_tool(metric_date: str) -> dict:
        return query_bad_review_rate(metric_date)

    @server.tool(name="query_freight_rate", description="查询指定日期的运费率。")
    def query_freight_rate_tool(metric_date: str) -> dict:
        return query_freight_rate(metric_date)

    @server.tool(name="query_metric_trend", description="查询指标在一段时间内的趋势。")
    def query_metric_trend_tool(metric_name: str, start_date: str, end_date: str, granularity: str = "day") -> dict:
        return query_metric_trend(metric_name, start_date, end_date, granularity)

    @server.tool(name="query_metric_statistics", description="查询指标的平均值、最大值、最小值、中位数等描述性统计。")
    def query_metric_statistics_tool(
        metric_name: str,
        start_date: str,
        end_date: str,
        granularity: str = "day",
        statistics: list[str] | None = None,
    ) -> dict:
        return query_metric_statistics(metric_name, start_date, end_date, granularity, statistics)

    @server.tool(name="compare_metric", description="比较指标在两个日期上的差异。")
    def compare_metric_tool(metric_name: str, left_date: str, right_date: str) -> dict:
        return compare_metric(metric_name, left_date, right_date)

    @server.tool(name="breakdown_by_dimension", description="按地域维度拆解某日指标。")
    def breakdown_by_dimension_tool(metric_name: str, metric_date: str, dimension: str, limit: int = 10) -> dict:
        return breakdown_by_dimension(metric_name, metric_date, dimension, limit)

    @server.tool(
        name="query_metric_by_dimensions",
        description=(
            "按两个到四个维度精确交叉汇总已注册指标。当前商品行口径支持 GMV、"
            "运费、订单数和客户数；不做支付方式与商品行之间的金额分摊，跨粒度组合会拒绝。"
        ),
    )
    def query_metric_by_dimensions_tool(
        metric_name: str,
        start_date: str,
        end_date: str,
        dimensions: list[str],
        limit: int = 100,
    ) -> dict:
        return query_metric_by_dimensions(metric_name, start_date, end_date, dimensions, limit)

    @server.tool(name="compare_breakdown_by_dimension", description="比较两个日期的维度拆解并返回贡献度计算事实。")
    def compare_breakdown_by_dimension_tool(metric_name: str, current_date: str, baseline_date: str, dimension: str, limit: int = 10) -> dict:
        return compare_breakdown_by_dimension(metric_name, current_date, baseline_date, dimension, limit)

    @server.tool(name="compare_breakdown_by_dimension_period", description="比较两个自然周期的维度变化贡献事实。")
    def compare_breakdown_by_dimension_period_tool(
        metric_name: str, current_start: str, current_end: str,
        baseline_start: str, baseline_end: str, granularity: str,
        dimension: str, limit: int = 10,
    ) -> dict:
        return compare_breakdown_by_dimension_period(
            metric_name, current_start, current_end, baseline_start, baseline_end,
            granularity, dimension, limit,
        )

    @server.tool(name="compare_freight_rate_ratio_facts_by_dimension_period", description="获取平台运费率按月内日期位置归因所需的完整分子/分母事实。")
    def compare_freight_rate_ratio_facts_by_dimension_period_tool(
        metric_name: str, current_start: str, current_end: str,
        baseline_start: str, baseline_end: str, granularity: str,
        dimension: str,
    ) -> dict:
        return compare_freight_rate_ratio_facts_by_dimension_period(
            metric_name, current_start, current_end, baseline_start, baseline_end,
            granularity, dimension,
        )

    @server.tool(name="compare_avg_order_value_ratio_facts_by_dimension_period", description="获取平台客单价按月内日期位置归因所需的完整分子/分母事实。")
    def compare_avg_order_value_ratio_facts_by_dimension_period_tool(
        metric_name: str, current_start: str, current_end: str,
        baseline_start: str, baseline_end: str, granularity: str,
        dimension: str,
    ) -> dict:
        return compare_avg_order_value_ratio_facts_by_dimension_period(
            metric_name, current_start, current_end, baseline_start, baseline_end,
            granularity, dimension,
        )

    @server.tool(name="query_metric_period", description="查询周/月指标聚合和血缘。")
    def query_metric_period_tool(metric_name: str, start_date: str, end_date: str, granularity: str) -> dict:
        return query_metric_period(metric_name, start_date, end_date, granularity)

    @server.tool(name="compare_metric_period", description="查询周/月指标环比或同比。")
    def compare_metric_period_tool(metric_name: str, start_date: str, end_date: str, granularity: str, comparison_type: str) -> dict:
        return compare_metric_period(metric_name, start_date, end_date, granularity, comparison_type)

    @server.tool(name="query_metric_batch_period", description="批量查询自然周或自然月指标。")
    def query_metric_batch_period_tool(metric_names: list[str], start_date: str, end_date: str, granularity: str) -> dict:
        return query_metric_batch_period(metric_names, start_date, end_date, granularity)

    @server.tool(name="list_metric_definitions", description="列出指标语义定义与字段血缘。")
    def list_metric_definitions_tool() -> list[dict[str, object]]:
        return list_semantic_metric_definitions()

    @server.tool(name="get_metric_definition", description="查询单个指标语义定义与字段血缘。")
    def get_metric_definition_tool(metric_name: str) -> dict[str, object]:
        return get_semantic_metric_definition(metric_name)

    return server


def main() -> None:
    """启动 MCP Server。"""

    parser = argparse.ArgumentParser(description="启动 bizdata-mcp-server。")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="MCP Server 使用的传输方式。",
    )
    args = parser.parse_args()
    create_mcp_server().run(transport=args.transport)


if __name__ == "__main__":
    main()

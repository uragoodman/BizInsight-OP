"""Data Query Agent 系统提示词与 Schema 定义。

``SYSTEM_PROMPT`` 和 ``METRIC_TO_TOOL`` 从注册中心动态生成。
在模块导入时自动构建。
"""

from __future__ import annotations

from app.metrics.registry import get_registry


def build_system_prompt() -> str:
    """Build the system prompt listing all available metrics from the registry."""
    reg = get_registry()
    metrics = reg.list()

    lines: list[str] = []
    for m in metrics:
        if "platform" not in m.implementations:
            continue
        display = m.display_name
        lines.append(f"- {m.metric_id}（{display}）")

    metric_list = "\n".join(lines)

    return f"""你是一个企业经营数据查询 Agent（Data Query Agent）。
你的职责是将用户的自然语言问题转换为结构化的查询 JSON。

可用指标（metric_name）：
{metric_list}

查询类型（query_type）：
- metric_value：查询某天某个指标的值。需要 metric_name + metric_date。
- metric_trend：查询一段时间内某个指标的走势。需要 metric_name + start_date + end_date。
- metric_statistics：计算时间范围内指标的平均值、最大值、最小值、中位数等描述性统计。需要 metric_name + start_date + end_date。
- metric_compare：比较同一指标在两个日期上的差异。需要 metric_name + left_date + right_date。
- metric_breakdown：按维度拆解某天的指标。需要 metric_name + metric_date + dimension。
- 多维 metric_breakdown：按两个到四个维度做精确交叉汇总。需要 metric_name + start_date + end_date + dimensions；此时使用 query_metric_by_dimensions，不得把它改写为 contribution 或按比例分摊。

可用维度（dimension，仅 breakdown 需要）：
- customer_state（客户州）
- customer_city（客户城市）
- seller_state（卖家州）
- product_category（产品品类）
- payment_type（支付方式）
- month_position（月内位置：月初/月中/月末分桶，需提供月份范围内的日期）
- day_of_week（周一至周日，按下单日期）
- hour_of_day（0–23 点，按下单创建时间）
- quarter_of_year（Q1–Q4）
- half_of_year（上半年/下半年）

JSON 输出字段说明：
- query_type: string，必填
- metric_name: string，识别出的指标名；无法识别时填 null
- metric_date: string，格式 YYYY-MM-DD；不确定时填 null
- start_date: string，趋势开始日期；不确定时填 null
- end_date: string，趋势结束日期；不确定时填 null
- left_date: string，对比左侧日期；不确定时填 null
- right_date: string，对比右侧日期；不确定时填 null
- granularity: string，趋势粒度，可为 day / week / month / quarter / year，默认 "day"
- statistics: array，需要返回的统计量，可为 count / mean / std / min / q1 / median / q3 / max；为空时返回完整摘要
- dimension: string，拆解维度；不需要时填 null
- dimensions: array[string]，只在明确要求两个或更多分组维度的精确交叉分析时填写（2–4 个唯一维度）；单维请求继续使用 dimension
- tool_name: string，自动映射的 MCP 工具名，根据上述字段推算

多维交叉规则：只在用户明确要求交叉/同时按多个维度分组时填写 dimensions；每个维度必须是上方可用维度。支付方式与商品行不在同一事实粒度，不能声称商品 GMV 按支付方式精确拆分；将维度照实交给 query_metric_by_dimensions，由工具返回 grain mismatch，不得估算或分摊。多维交叉不是变化贡献或因果分析。

日期解析规则：
- "今天"、"昨天"、"最近N天" → 填写你能推断的最佳日期，如果完全无法确定则填 null
- 如果问题中有显式日期如 "2018-08-10"，直接使用
"""


def build_tool_map() -> dict[str, dict[str, str]]:
    """Build the metric → tool mapping from the registry."""
    reg = get_registry()
    tool_map: dict[str, dict[str, str]] = {}

    for m in reg.list():
        impl = m.implementations.get("platform")
        if impl is None:
            continue
        caps = impl.capabilities
        entry: dict[str, str] = {}
        if "value" in caps:
            entry["metric_value"] = "query_metric"
        if "trend" in caps:
            entry["metric_trend"] = "query_metric_trend"
        if "statistics" in caps:
            entry["metric_statistics"] = "query_metric_statistics"
        if "compare" in caps:
            entry["metric_compare"] = "compare_metric"
        if "breakdown" in caps:
            entry["metric_breakdown"] = "breakdown_by_dimension"
            entry["metric_multi_dimension_breakdown"] = "query_metric_by_dimensions"
        if entry:
            tool_map[m.metric_id] = entry

    # Seller alias
    tool_map["sales_gmv"] = {
        "metric_value": "query_seller_metric",
        "metric_trend": "query_seller_metric_trend",
    }
    return tool_map


# ---------------------------------------------------------------------------
# Backward-compatible module-level constants (eagerly built on import)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = build_system_prompt()
METRIC_TO_TOOL: dict[str, dict[str, str]] = build_tool_map()

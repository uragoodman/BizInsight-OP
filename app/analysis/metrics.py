"""指标查询结果的自然语言摘要层。

这个文件的职责不是查数据，而是把已经拿到的结构化结果整理成更适合用户阅读
的业务表达。它处在：

`tool_result -> analysis summary -> answer`

这条链路的中间层。
"""


def _format_value(value: float | None) -> str:
    """把数值格式化成适合直接展示给用户的字符串。"""

    if value is None:
        return "暂无数据"
    return f"{value:.2f}"


def summarize_metric_result(
    question: str,
    tool_result: dict,
    metric_context: str,
    critic_result: dict | None = None,
) -> str:
    """把不同类型的指标查询结果整理成最终回答摘要。

    当前支持五类结果：
    - 单点查询：回答“某天某指标是多少”
    - 趋势查询：回答“一段时间里这个指标怎么变化”
    - 对比查询：回答“两个日期之间差了多少”
    - 拆解查询：回答“某个指标按某个维度如何分布”
    - 统计查询：回答平均值、最大值、最小值、中位数等描述性统计
    """

    result = tool_result.get("result", {})
    critic_status = critic_result.get("status") if critic_result else "unknown"

    if result.get("status") != "ok":
        compare_result = result.get("compare_result")
        if compare_result and (
            compare_result.get("status") != "ok"
            or compare_result.get("left_value") is None
            or compare_result.get("right_value") is None
        ):
            return (
                f"{compare_result.get('metric_name')} 无法完成比较："
                f"{compare_result.get('left_date')}={_format_value(compare_result.get('left_value'))}，"
                f"{compare_result.get('right_date')}={_format_value(compare_result.get('right_value'))}。"
                "至少一侧缺少有效数据，未计算或解释变化。"
            )
        reason = result.get("reason", "当前问题暂不支持。")
        return f"当前未能完成指标查询。原因：{reason}"

    metric_result = result.get("metric_result")
    if metric_result:
        return (
            f"{metric_result.get('metric_date')} 的 {metric_result.get('metric_name')} 为 "
            f"{_format_value(metric_result.get('value'))}。"
            f" 数据来源：{metric_result.get('source_table')}。"
            f" Critic 审核状态：{critic_status}。"
        )

    statistics_result = result.get("statistics_result")
    if statistics_result:
        labels = {
            "count": "样本数",
            "mean": "平均值",
            "average": "平均值",
            "std": "标准差",
            "min": "最小值",
            "q1": "P25",
            "median": "中位数",
            "q3": "P75",
            "max": "最大值",
        }
        stats = statistics_result.get("statistics") or {}
        summary = "，".join(
            f"{labels.get(name, name)}={_format_value(value)}"
            for name, value in stats.items()
            if value is not None
        ) or "暂无统计值"
        return (
            f"{statistics_result.get('metric_name')} 在 "
            f"{statistics_result.get('start_date')} 至 {statistics_result.get('end_date')} "
            f"按 {statistics_result.get('granularity')} 粒度统计：{summary}。"
            f" 最小值时间点：{statistics_result.get('min_bucket')}，"
            f"最大值时间点：{statistics_result.get('max_bucket')}。"
            f" 数据来源：{statistics_result.get('source_table')}。"
            f" Critic 审核状态：{critic_status}。"
        )

    trend_result = result.get("trend_result")
    if trend_result:
        points = trend_result.get("points", [])
        preview = ", ".join(
            f"{point.get('bucket')}={_format_value(point.get('value'))}" for point in points[:5]
        )
        return (
            f"{trend_result.get('metric_name')} 在 {trend_result.get('start_date')} 到 "
            f"{trend_result.get('end_date')} 的趋势已查询完成。"
            f" 前几个数据点：{preview}。"
            f" 数据来源：{trend_result.get('source_table')}。"
            f" Critic 审核状态：{critic_status}。"
        )

    compare_result = result.get("compare_result")
    if compare_result:
        if (
            compare_result.get("status") != "ok"
            or compare_result.get("left_value") is None
            or compare_result.get("right_value") is None
        ):
            return (
                f"{compare_result.get('metric_name')} 无法完成比较："
                f"{compare_result.get('left_date')}={_format_value(compare_result.get('left_value'))}，"
                f"{compare_result.get('right_date')}={_format_value(compare_result.get('right_value'))}。"
                "至少一侧缺少有效数据，未计算或解释变化。"
            )
        return (
            f"{compare_result.get('metric_name')} 对比结果："
            f"{compare_result.get('left_date')}={_format_value(compare_result.get('left_value'))}，"
            f"{compare_result.get('right_date')}={_format_value(compare_result.get('right_value'))}，"
            f"差值={_format_value(compare_result.get('delta_value'))}，"
            f"变化率={_format_value(compare_result.get('delta_ratio'))}% 。"
            f" 数据来源：{compare_result.get('source_table')}。"
            f" Critic 审核状态：{critic_status}。"
        )

    breakdown_result = result.get("breakdown_result")
    items = breakdown_result.get("items", []) if breakdown_result else []
    preview = ", ".join(
        f"{item.get('dimension_value')}={_format_value(item.get('value'))}" for item in items[:5]
    )
    return (
        f"{breakdown_result.get('metric_date')} 的 {breakdown_result.get('metric_name')} 已按 "
        f"{breakdown_result.get('dimension')} 拆解。"
        f" 前几项结果：{preview}。"
        f" 数据来源：{breakdown_result.get('source_table')}。"
        f" Critic 审核状态：{critic_status}。"
    )

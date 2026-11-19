"""报告草稿生成。

Report Agent 的核心逻辑：整合指标定义、数据结果和归因结论，生成结构化报告草案。
"""

from __future__ import annotations

from app.data.repositories.daily_metrics import get_daily_metrics
from app.data.repositories.daily_metrics import get_latest_metric_date
from app.analysis.contracts import PeriodSpec
from app.analysis.periodic_reports import build_periodic_report, report_to_markdown
from app.analysis.contracts import PeriodicReport
from app.analysis.targets import TargetStore
from app.core.config import get_settings
from typing import Literal
from app.rag.retriever import retrieve_knowledge


def _is_periodic_question(question: str) -> bool:
    lowered = question.lower()
    if any(keyword in lowered for keyword in ("weekly", "week report", "monthly", "month report")):
        return True
    return any(keyword in question for keyword in ("周报", "月报", "周度", "月度", "本周", "本月", "周经营", "月经营"))


def build_periodic_report_artifact(question: str) -> PeriodicReport | None:
    """Resolve a natural period from the latest business date and build the full artifact."""

    if not _is_periodic_question(question):
        return None
    from calendar import monthrange
    from datetime import date, timedelta

    latest_raw = get_latest_metric_date()
    if not latest_raw:
        return None
    latest = date.fromisoformat(latest_raw)
    granularity: Literal["week", "month"]
    if "月" in question:
        last_day = date(latest.year, latest.month, monthrange(latest.year, latest.month)[1])
        period_end = latest if latest == last_day else latest.replace(day=1) - timedelta(days=1)
        start = period_end.replace(day=1)
        granularity = "month"
    else:
        period_end = latest - timedelta(days=(latest.weekday() + 1) % 7)
        start = period_end - timedelta(days=6)
        granularity = "week"
    target_store = TargetStore(get_settings().phase6_store_path)
    return build_periodic_report(
        PeriodSpec(start_date=start, end_date=period_end, granularity=granularity),
        ["gmv", "order_count", "avg_order_value"], include_charts=True,
        target_lookup=lambda metric_name, period: target_store.get(metric_name, period),
        chart_output_dir=get_settings().report_chart_dir,
    )


def generate_report(question: str, tool_result: dict | None = None, metric_context: str = "") -> str:
    """生成经营分析报告草稿。

    参数：
    - `question`：用户原始问题（如"生成本周经营分析报告"）。
    - `tool_result`：MCP 工具查询结果（可能为空）。
    - `metric_context`：RAG 检索的指标定义上下文。

    返回值：
    - `str`：Markdown 格式的报告草案。
    """

    # Phase 6 periodic reports resolve against the latest business date, not the
    # machine clock. This keeps offline datasets and replay deterministic.
    if _is_periodic_question(question):
        try:
            report = build_periodic_report_artifact(question)
            if report:
                return report_to_markdown(report)
        except Exception:
            # Preserve the legacy draft fallback when the database or optional
            # reporting capability is unavailable.
            pass

    sections: list[str] = ["# 经营分析报告（草案）"]

    # 从 template 知识库获取报告模板
    try:
        rag = retrieve_knowledge("weekly report template", doc_types={"template"}, top_k=1)
        if rag.hits:
            sections.append(f"\n## 报告结构参考\n{rag.hits[0].text}")
    except Exception:
        pass

    # 从 metric_context 提取指标定义
    if metric_context:
        sections.append(f"\n## 指标口径\n{metric_context}")

    # 优先使用 Data Query Agent 已经通过 MCP 取回的指标，避免报告节点
    # 另起一套不可追踪的数据读取路径。
    queried_metric = (tool_result or {}).get("result", {}).get("metric_result")
    if queried_metric:
        sections.append(
            f"\n## 核心指标概览（{queried_metric.get('metric_date', '未知日期')}）\n"
            f"- {queried_metric.get('metric_name', '核心指标')}："
            f"**{queried_metric.get('value', 'N/A')}**\n"
            f"- 数据来源：{queried_metric.get('source_table', 'unknown')}"
        )

    # 当上游尚未提供数据时才尝试获取最近一天的核心指标数据，保证报告草案
    # 仍可生成；该回退路径在后续可替换为更多 MCP 批量查询工具。
    try:
        from datetime import date

        today_str = date.today().isoformat()
        daily = get_daily_metrics(today_str)
    except Exception:
        daily = None

    if daily and not queried_metric:
        sections.append(
            f"\n## 核心指标概览（{daily.get('metric_date', today_str)}）\n"
            f"- GMV：**{daily.get('gmv', 'N/A')}**\n"
            f"- 订单量：**{daily.get('order_count', 'N/A')}**\n"
            f"- 客单价：**{daily.get('avg_order_value', 'N/A')}**"
        )

    if tool_result:
        sections.append(f"\n## 查询数据\n```json\n{tool_result}\n```")

    sections.append("\n## 后续建议\n")
    sections.append("- 确认数据口径与指标定义一致。")
    sections.append("- 关注异常指标的维度下钻分析。")
    sections.append("- 由 Critic Agent 审核后发布正式版本。")
    sections.append(f"\n> 报告基于问题「{question}」自动生成，待审核。")

    return "\n".join(sections)

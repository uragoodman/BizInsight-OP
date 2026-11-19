"""Structured weekly/monthly report assembly and Markdown rendering."""

from __future__ import annotations

from hashlib import sha256
from datetime import timedelta
from datetime import datetime, time, timezone
from typing import Literal

from app.analysis.charts import render_contribution_chart, render_trend_chart
from app.analysis.contracts import AnomalyFinding, ContributionFinding, PeriodicReport, PeriodSpec, ReportMetric
from app.analysis.periods import aggregate_metric_period, compare_periods, previous_period, query_metric_series, validate_natural_period, year_ago_period
from app.analysis.semantic import get_metric_definition
from app.analysis.targets import compare_to_target
from app.analysis.statistical_anomaly import detect_series_anomalies
from app.analysis.contribution import calculate_change_contribution
from app.data.repositories.daily_metrics import compare_breakdown_by_dimension_period
from app.data.report_snapshot import consistent_report_reads


# These metrics are sums whose dimension deltas can be reconciled to the
# report-level delta. Ratio, average, and distinct-count metrics require a
# dedicated decomposition method and must not be presented as additive
# contribution.
ADDITIVE_CONTRIBUTION_METRICS = {"gmv", "freight_value"}

# Metrics implemented by the platform periodic-report query path.  Kept as a
# capability contract so the unified metric catalog can show the same union
# for platform and seller reports without making unsupported options usable.
PLATFORM_REPORT_SUPPORTED_METRICS = frozenset({
    "customer_count", "repeat_customer_count", "repeat_purchase_rate",
    "avg_items_per_order", "on_time_delivery_rate", "delivery_delay_rate",
    "avg_delivery_lead_time", "avg_approval_lead_hours",
    "avg_dispatch_lead_hours", "avg_delivery_transit_hours",
    "order_approval_rate", "order_shipment_rate", "order_final_delivery_rate",
    "payment_value", "installment_order_share", "review_score_avg",
    "bad_review_rate", "cancel_rate", "review_comment_rate", "gmv",
    "order_count", "avg_order_value", "freight_value", "freight_rate",
})


@consistent_report_reads
def build_periodic_report(
    period: PeriodSpec,
    metric_names: list[str],
    *,
    fetcher=None,
    include_charts: bool = True,
    target_lookup=None,
    anomalies: list[AnomalyFinding] | None = None,
    contributions: list[ContributionFinding] | None = None,
    auto_enrich: bool = True,
    chart_output_dir: str = "data/reports/charts",
) -> PeriodicReport:
    if period.granularity not in {"week", "month"}:
        raise ValueError("Periodic reports require week or month granularity")
    validate_natural_period(period)
    report_type: Literal["weekly", "monthly"] = "weekly" if period.granularity == "week" else "monthly"
    resolved_anomalies = list(anomalies or [])
    resolved_contributions = list(contributions or [])
    metrics: list[ReportMetric] = []
    charts = []
    lineage = []
    freshness_values = []
    evidence_records: dict[str, dict[str, object]] = {}
    enrichment_risks: list[str] = []
    for metric_name in metric_names:
        kwargs = {} if fetcher is None else {"fetcher": fetcher}
        series = query_metric_series(metric_name, period, **kwargs)
        definition = get_metric_definition(metric_name)
        if series.freshness_at:
            freshness_values.append(series.freshness_at)
        pop = compare_periods(metric_name, period, previous_period(period), "period_over_period", **kwargs)
        yoy = compare_periods(metric_name, period, year_ago_period(period), "year_over_year", **kwargs)
        target = target_lookup(metric_name, period) if target_lookup else None
        value = aggregate_metric_period(metric_name, period, daily_fetcher=fetcher)
        evidence_id = f"ev_series_{sha256(series.model_dump_json().encode()).hexdigest()[:16]}"
        evidence_ids = [evidence_id]
        if target:
            evidence_ids.append(f"target:{metric_name}:{target.scope}:{target.version}:{target.effective_at}")
        metrics.append(ReportMetric(
            metric_name=metric_name, value=value, unit=definition.unit,
            period_over_period=pop, year_over_year=yoy,
            target_comparison=compare_to_target(metric_name, value, period, target),
            evidence_ids=evidence_ids,
        ))
        evidence_records[evidence_id] = {
            "kind": "metric_series", "metric_name": metric_name,
            "series": series.model_dump(mode="json"), "value": value,
            "period_over_period": pop.model_dump(mode="json"),
            "year_over_year": yoy.model_dump(mode="json"),
            "definition": {"version": definition.version, "lineage_ref": definition.lineage_ref},
        }
        if target:
            evidence_records[evidence_ids[-1]] = {
                "kind": "metric_target", "metric_name": metric_name,
                "target_value": target.target_value, "version": target.version,
                "effective_at": target.effective_at.isoformat(), "scope": target.scope,
            }
        lineage.append(definition.lineage_ref)
        if target:
            lineage.append(f"target:{metric_name}:{target.scope}@{target.version} -> report:{report_type}")
        if include_charts:
            charts.append(render_trend_chart(series, chart_output_dir))
        if auto_enrich and fetcher is None:
            lookback_period = PeriodSpec(start_date=period.end_date - timedelta(days=29), end_date=period.end_date)
            lookback_series = query_metric_series(metric_name, lookback_period)
            resolved_anomalies.extend(
                item for item in detect_series_anomalies(lookback_series)
                if period.start_date <= item.metric_date <= period.end_date
            )
    if auto_enrich and fetcher is None and resolved_anomalies:
        primary = max(resolved_anomalies, key=lambda item: abs(item.score))
        definition = get_metric_definition(primary.metric_name)
        baseline_period = previous_period(period)
        if primary.metric_name not in ADDITIVE_CONTRIBUTION_METRICS:
            enrichment_risks.append(f"contribution_not_decomposable:{primary.metric_name}")
        else:
            report_metric = next(item for item in metrics if item.metric_name == primary.metric_name)
            comparison = report_metric.period_over_period
            for dimension in definition.allowed_dimensions:
                raw = compare_breakdown_by_dimension_period(
                    primary.metric_name, period.start_date, period.end_date,
                    baseline_period.start_date, baseline_period.end_date, dimension, 10,
                )
                if raw.get("status") != "ok" or comparison is None:
                    continue
                resolved_contributions.append(calculate_change_contribution(
                    primary.metric_name, dimension, period.end_date, baseline_period.end_date,
                    raw.get("current_items", []), raw.get("baseline_items", []), comparison.delta_value,
                    comparison.current_value, comparison.baseline_value, period, baseline_period,
                ))
    if include_charts:
        charts.extend(render_contribution_chart(item, chart_output_dir) for item in resolved_contributions)
    for anomaly in resolved_anomalies:
        for evidence_id in anomaly.evidence_ids:
            evidence_records[evidence_id] = {"kind": "anomaly", **anomaly.model_dump(mode="json")}
    for contribution in resolved_contributions:
        for evidence_id in contribution.evidence_ids:
            evidence_records[evidence_id] = {"kind": "contribution", **contribution.model_dump(mode="json")}
    risks = list(enrichment_risks)
    if any(metric.value is None for metric in metrics):
        risks.append("missing_metric_data")
    if not freshness_values:
        risks.append("freshness_unknown")
    if any(chart.status != "ok" for chart in charts):
        risks.append("chart_unavailable")
    if any(abs(item.score) >= item.threshold * 2 for item in resolved_anomalies):
        risks.append("critical_anomaly")
    signature = sha256(f"{report_type}|{period.model_dump_json()}|{','.join(metric_names)}".encode()).hexdigest()[:16]
    data_freshness = max(freshness_values) if freshness_values else datetime.combine(
        period.end_date, time.min, tzinfo=timezone.utc,
    )
    report = PeriodicReport(
        report_id=f"report_{signature}", report_type=report_type, period=period,
        title=f"经营分析{report_type == 'weekly' and '周报' or '月报'}（{period.start_date} 至 {period.end_date}）",
        summary=f"本报告覆盖 {len(metrics)} 个核心指标，识别 {len(resolved_anomalies)} 个统计异常。",
        metrics=metrics, anomalies=resolved_anomalies, contributions=resolved_contributions,
        charts=charts, risks=risks, lineage_refs=lineage,
        recommendations=["复核高影响异常对应的业务事件。", "确认指标口径、目标版本和数据新鲜度后再发布。"],
        data_freshness_at=data_freshness, generated_at=data_freshness,
        evidence_records=evidence_records,
    )
    if fetcher is None:
        from app.analysis.report_alignment import capture_report_attribution_snapshots

        report = capture_report_attribution_snapshots(report)
    return report


def report_to_markdown(report: PeriodicReport) -> str:
    lines = [f"# {report.title}", "", report.summary, "", "## 核心指标"]
    if report.scope.scope_type == "seller":
        lines[3:3] = [f"**报告范围**：单店｜{report.scope.seller_label}  ", f"**店铺 ID**：`{report.scope.seller_id}`", ""]
    else:
        lines[3:3] = ["**报告范围**：全平台", ""]
    for metric in report.metrics:
        pop = metric.period_over_period
        yoy = metric.year_over_year
        lines.append(f"- **{metric.metric_name}**：{metric.value if metric.value is not None else 'N/A'} {metric.unit}")
        if pop:
            lines.append(f"  - 环比：{pop.delta_ratio if pop.delta_ratio is not None else 'N/A'}%（{pop.status}）")
        if yoy:
            lines.append(f"  - 同比：{yoy.delta_ratio if yoy.delta_ratio is not None else 'N/A'}%（{yoy.status}）")
        if metric.target_comparison:
            target = metric.target_comparison
            lines.append(f"  - 目标差异：{target.delta_ratio if target.delta_ratio is not None else 'N/A'}%（{target.status}）")
            if target.target_version:
                lines.append(f"  - 目标版本：{target.target_version}；达成率：{target.attainment_rate if target.attainment_rate is not None else 'N/A'}%")
    if report.anomalies:
        lines.extend(["", "## 异常"])
        lines.extend(f"- {item.metric_name} {item.metric_date}: {item.direction}, score={item.score}" for item in report.anomalies)
    if report.contributions:
        lines.extend(["", "## 变化贡献度"])
        for finding in report.contributions:
            lines.append(f"- {finding.metric_name} / {finding.dimension}：总变化 {finding.total_delta:+.2f}")
            lines.extend(
                f"  - {item.dimension_value}: {item.delta_value:+.2f} ({item.contribution_ratio if item.contribution_ratio is not None else 'N/A'})"
                for item in finding.items[:5]
            )
    lines.extend(["", "## 图表"])
    for chart in report.charts:
        lines.append(f"- {chart.chart_type}: {chart.path if chart.status == 'ok' else chart.reason}")
    lines.extend(["", "## 数据口径与血缘"])
    lines.extend(f"- {item}" for item in report.lineage_refs)
    lines.extend(["", "## 风险与建议"])
    lines.extend(f"- 风险：{item}" for item in report.risks)
    lines.extend(f"- 建议：{item}" for item in report.recommendations)
    lines.append(f"- 数据新鲜度：{report.data_freshness_at.isoformat() if report.data_freshness_at else 'unknown'}")
    lines.append("\n> 本报告为自动生成草稿，正式发布前需经过风险审核。")
    return "\n".join(lines)

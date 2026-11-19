"""Periodic reports computed directly from one authorized seller's raw scope."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from hashlib import sha256

from app.analysis.charts import render_trend_chart
from app.analysis.contracts import AnomalyFinding, MetricPoint, MetricSeries, PeriodComparison, PeriodicReport, PeriodSpec, ReportMetric, ReportScope
from app.analysis.periods import previous_period, validate_natural_period, year_ago_period
from app.analysis.semantic import get_metric_definition
from app.analysis.statistical_anomaly import detect_series_anomalies
from app.analysis.targets import compare_to_target
from app.data.repositories import seller_extensions
from app.data.repositories.explorer import query_scoped_metric, query_scoped_trend
from app.data.report_snapshot import consistent_report_reads
from app.metrics.registry import get_registry
from app.schemas.explorer import ControlledAnalysisScope


# These metrics have a complete controlled seller value/trend implementation.
# The report builder routes them through seller_extensions below, while the
# five base KPIs continue to use the common explorer query path.
SELLER_REPORT_SUPPORTED_METRICS = frozenset({
    "gmv",
    "order_count",
    "avg_order_value",
    "customer_count",
    "freight_value",
    "seller_free_shipping_order_rate",
    "seller_avg_early_delivery_days",
    "seller_avg_late_delivery_days",
    "seller_avg_delivery_estimate_abs_error_days",
    "seller_p90_order_to_delivery_days",
    "seller_missing_delivery_timestamp_rate",
    "seller_single_item_order_product_review_score_avg",
    "seller_single_item_order_product_bad_review_rate",
    "seller_product_attributed_review_score_avg",
    "seller_product_attributed_bad_review_rate",
    "seller_historical_returning_customer_count",
    "seller_historical_returning_customer_rate",
    "seller_new_customer_count",
    "seller_new_customer_rate",
    "seller_top1_product_gross_share",
    "seller_top3_category_gross_share",
    "seller_new_product_gross_share",
    "seller_item_gross_value",
    "seller_avg_item_price",
    "seller_avg_freight_per_order",
    "seller_avg_payment_installments",
    "seller_credit_card_payment_share",
    "seller_boleto_payment_share",
    "seller_active_product_count",
    "seller_active_category_count",
})


def _scope(seller_id: str, period: PeriodSpec, metrics: list[str]) -> ControlledAnalysisScope:
    return ControlledAnalysisScope(
        seller_id=seller_id,
        default_start_date=period.start_date,
        default_end_date=period.end_date,
        metric_name=metrics[0],
        granularity="day",
        allowed_metrics=tuple(metrics),
    )


def _metric_value(scope: ControlledAnalysisScope, metric_name: str) -> dict[str, object]:
    if seller_extensions.supports(metric_name):
        return seller_extensions.query(scope, metric_name)
    return query_scoped_metric(scope, metric_name)


def _metric_trend(scope: ControlledAnalysisScope, metric_name: str, period: PeriodSpec) -> dict[str, object]:
    if seller_extensions.supports(metric_name):
        return seller_extensions.trend(scope, metric_name, period.start_date, period.end_date)
    return query_scoped_trend(scope, metric_name, start_date=period.start_date, end_date=period.end_date)


def _comparison(metric_name: str, seller_id: str, current: PeriodSpec, baseline: PeriodSpec, metrics: list[str], comparison_type: str) -> PeriodComparison:
    current_value = _metric_value(_scope(seller_id, current, metrics), metric_name).get("value")
    baseline_value = _metric_value(_scope(seller_id, baseline, metrics), metric_name).get("value")
    current_number = float(current_value) if isinstance(current_value, (int, float)) else None
    baseline_number = float(baseline_value) if isinstance(baseline_value, (int, float)) else None
    delta = round(current_number - baseline_number, 2) if current_number is not None and baseline_number is not None else None
    ratio = round(delta * 100 / baseline_number, 2) if delta is not None and baseline_number not in (None, 0) else None
    return PeriodComparison(
        comparison_type=comparison_type, current_period=current, baseline_period=baseline,
        current_value=current_number, baseline_value=baseline_number, delta_value=delta, delta_ratio=ratio,
        status="not_found" if current_number is None or baseline_number is None else "zero_baseline" if baseline_number == 0 else "ok",
    )


def _series(metric_name: str, seller_id: str, period: PeriodSpec, metrics: list[str]) -> MetricSeries:
    definition = get_metric_definition(metric_name, "seller")
    data = _metric_trend(_scope(seller_id, period, metrics), metric_name, period)
    points = [MetricPoint(bucket=item["bucket"], value=item.get("value")) for item in data.get("points", [])]
    freshness = datetime.combine(max((point.bucket for point in points), default=period.end_date), time.max, tzinfo=timezone.utc)
    return MetricSeries(
        metric_name=metric_name, unit=definition.unit, period=period, points=points,
        source_refs=[f"seller:{seller_id}", str(data.get("source_table", definition.source_table))],
        definition_version=definition.version, freshness_at=freshness,
    )


@consistent_report_reads
def build_seller_scoped_periodic_report(
    seller_id: str,
    seller_label: str,
    period: PeriodSpec,
    metric_names: list[str],
    *,
    target_lookup=None,
    include_charts: bool = True,
    chart_output_dir: str = "data/reports/charts",
) -> PeriodicReport:
    validate_natural_period(period)
    registry = get_registry()
    unsupported = [
        name for name in metric_names
        if name not in SELLER_REPORT_SUPPORTED_METRICS
        or not registry.supports(name, "value", "seller")
        or not registry.supports(name, "trend", "seller")
    ]
    if unsupported:
        raise ValueError(f"Metrics unavailable for seller reports: {', '.join(unsupported)}")
    metrics: list[ReportMetric] = []
    charts = []
    anomalies: list[AnomalyFinding] = []
    freshness_values: list[datetime] = []
    evidence_records: dict[str, dict[str, object]] = {}
    lineage: list[str] = []
    for metric_name in metric_names:
        definition = get_metric_definition(metric_name, "seller")
        series = _series(metric_name, seller_id, period, metric_names)
        if series.freshness_at:
            freshness_values.append(series.freshness_at)
        current = _metric_value(_scope(seller_id, period, metric_names), metric_name)
        value = current.get("value")
        numeric_value = float(value) if isinstance(value, (int, float)) else None
        evidence_id = f"ev_seller_series_{sha256((seller_id + series.model_dump_json()).encode()).hexdigest()[:16]}"
        target = target_lookup(metric_name, period) if target_lookup else None
        evidence_ids = [evidence_id]
        evidence_records[evidence_id] = {
            "kind": "seller_metric_series", "seller_id": seller_id, "seller_label": seller_label,
            "metric_name": metric_name, "series": series.model_dump(mode="json"), "value": numeric_value,
        }
        if target:
            target_id = f"target:{metric_name}:{target.scope}:{target.version}:{target.effective_at}"
            evidence_ids.append(target_id)
            evidence_records[target_id] = {"kind": "metric_target", "metric_name": metric_name, "target_value": target.target_value, "version": target.version, "scope": target.scope}
        metrics.append(ReportMetric(
            metric_name=metric_name, value=numeric_value, unit=definition.unit,
            period_over_period=_comparison(metric_name, seller_id, period, previous_period(period), metric_names, "period_over_period"),
            year_over_year=_comparison(metric_name, seller_id, period, year_ago_period(period), metric_names, "year_over_year"),
            target_comparison=compare_to_target(metric_name, numeric_value, period, target, scope="seller"), evidence_ids=evidence_ids,
        ))
        lineage.append(f"seller:{seller_id} -> {definition.lineage_ref}")
        if include_charts:
            charts.append(render_trend_chart(series, chart_output_dir))
        lookback = PeriodSpec(start_date=period.end_date - timedelta(days=29), end_date=period.end_date)
        for finding in detect_series_anomalies(_series(metric_name, seller_id, lookback, metric_names)):
            if period.start_date <= finding.metric_date <= period.end_date:
                anomalies.append(finding)
    for finding in anomalies:
        for evidence_id in finding.evidence_ids:
            evidence_records[evidence_id] = {"kind": "anomaly", "seller_id": seller_id, **finding.model_dump(mode="json")}
    signature = sha256(f"seller|{seller_id}|{period.model_dump_json()}|{','.join(metric_names)}".encode()).hexdigest()[:16]
    risks = ["chart_unavailable"] if any(chart.status != "ok" for chart in charts) else []
    if any(metric.value is None for metric in metrics):
        risks.append("missing_metric_data")
    if any(abs(item.score) >= item.threshold * 2 for item in anomalies):
        risks.append("critical_anomaly")
    report = PeriodicReport(
        report_id=f"seller_report_{signature}", report_type="weekly" if period.granularity == "week" else "monthly", period=period,
        scope=ReportScope(scope_type="seller", seller_id=seller_id, seller_label=seller_label),
        title=f"经营分析{'周报' if period.granularity == 'week' else '月报'}｜{seller_label}（{period.start_date} 至 {period.end_date}）",
        summary=f"本报告覆盖店铺 {seller_label} 的 {len(metrics)} 个核心指标，识别 {len(anomalies)} 个统计异常。",
        metrics=metrics, anomalies=anomalies, charts=charts, lineage_refs=lineage, evidence_records=evidence_records,
        recommendations=["复核高影响异常对应的店铺经营或履约事件。", "确认指标口径、目标版本和数据新鲜度后再发布。"],
        risks=risks, data_freshness_at=max(freshness_values, default=datetime.combine(period.end_date, time.max, tzinfo=timezone.utc)),
    )
    from app.analysis.report_alignment import capture_report_attribution_snapshots

    report = capture_report_attribution_snapshots(report)
    return report

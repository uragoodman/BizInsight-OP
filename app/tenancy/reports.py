"""Seller-only periodic report builder; never calls platform analytics."""
from datetime import datetime, time, timezone
from hashlib import sha256

from app.analysis.charts import render_trend_chart
from app.analysis.contracts import MetricPoint, MetricSeries, PeriodComparison, PeriodicReport, PeriodSpec, ReportMetric
from app.analysis.periods import previous_period, year_ago_period, validate_natural_period
from app.analysis.semantic import get_metric_definition
from app.analysis.targets import compare_to_target
from app.data.repositories.tenant_metrics import SELLER_METRICS, query_seller_metric, query_seller_trend
from app.tenancy.context import TenantContext


def build_seller_periodic_report(
    context: TenantContext, period: PeriodSpec, metric_names: list[str], *,
    target_lookup=None, include_charts: bool = True, chart_output_dir: str = "data/reports/charts",
) -> PeriodicReport:
    if context.scope_type != "seller" or not set(metric_names).issubset(SELLER_METRICS):
        raise ValueError("Seller report requested unsupported metric or scope")
    validate_natural_period(period)
    def value(metric, current): return query_seller_metric(context, metric, str(current.start_date), str(current.end_date))["value"]
    metrics, evidence, lineage, charts = [], {}, [], []
    for name in metric_names:
        current, previous, yearly = value(name, period), value(name, previous_period(period)), value(name, year_ago_period(period))
        definition = get_metric_definition(name, "seller")
        def comparison(base, kind, baseline):
            delta = round(current-base, 2) if base is not None else None
            return PeriodComparison(comparison_type=kind,current_period=period,baseline_period=baseline,current_value=current,baseline_value=base,delta_value=delta,delta_ratio=None if not base else round(delta*100/base,2),status="not_found" if base is None else ("zero_baseline" if base==0 else "ok"))
        ev=f"tenant:{context.tenant_id}:metric:{name}:{period.start_date}:{period.end_date}"
        evidence[ev]={"kind":"tenant_metric","tenant_id":context.tenant_id,"metric_name":name,"value":current,"source":"analytics.tenant_daily_metrics"}
        target=target_lookup(name,period) if target_lookup else None
        evidence_ids = [ev]
        if target:
            target_evidence = f"target:{name}:tenant:{target.version}:{target.effective_at}"
            evidence_ids.append(target_evidence)
            evidence[target_evidence] = {"kind":"metric_target", "metric_name":name, "target_value":target.target_value, "version":target.version, "effective_at":target.effective_at.isoformat(), "scope":"tenant"}
            lineage.append(f"target:{name}:tenant@{target.version} -> seller_report")
        metrics.append(ReportMetric(metric_name=name,value=current,unit=definition.unit,period_over_period=comparison(previous,"period_over_period",previous_period(period)),year_over_year=comparison(yearly,"year_over_year",year_ago_period(period)),target_comparison=compare_to_target(name, current, period, target, scope="seller"),evidence_ids=evidence_ids))
        lineage.append(definition.lineage_ref)
        if include_charts:
            trend = query_seller_trend(context, name, str(period.start_date), str(period.end_date))
            series = MetricSeries(metric_name=name, unit=definition.unit, period=period,
                points=[MetricPoint(bucket=item["metric_date"], value=item["value"]) for item in trend["points"]],
                source_refs=["analytics.tenant_daily_metrics"], definition_version=definition.version,
                freshness_at=datetime.combine(period.end_date, time.min, tzinfo=timezone.utc))
            charts.append(render_trend_chart(series, chart_output_dir))
    key=sha256(f"{context.tenant_id}:{period.model_dump_json()}:{metric_names}".encode()).hexdigest()[:16]
    risks = ["chart_unavailable"] if any(chart.status != "ok" for chart in charts) else []
    freshness = datetime.combine(period.end_date, time.min, tzinfo=timezone.utc)
    return PeriodicReport(report_id=f"seller_report_{key}",report_type="weekly" if period.granularity=="week" else "monthly",period=period,title=f"Seller report {period.start_date} to {period.end_date}",summary=f"Seller-scoped report for {len(metrics)} metrics.",metrics=metrics,charts=charts,lineage_refs=lineage,evidence_records=evidence,data_freshness_at=freshness,recommendations=["Review seller metric changes before publication."],risks=risks)

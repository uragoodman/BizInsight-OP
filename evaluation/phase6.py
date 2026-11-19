"""Deterministic Phase 6 business acceptance cases."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from app.analysis.contracts import MetricPoint, MetricSeries, PeriodSpec
from app.analysis.contribution import calculate_change_contribution
from app.analysis.events import LocalNotificationAdapter, process_pending_events
from app.analysis.periodic_reports import build_periodic_report
from app.analysis.publishing import ReportStore
from app.analysis.statistical_anomaly import detect_series_anomalies
from app.analysis.targets import MetricTarget, TargetStore
from evaluation.scorers import score_phase6_report


def _fetch(metric: str, day: date) -> dict:
    offset = (day - date(2024, 1, 1)).days
    values = {"gmv": 1000 + offset, "order_count": 100.0, "avg_order_value": (1000 + offset) / 100}
    return {"value": values.get(metric), "source_table": "golden.fixture"}


def run_phase6_acceptance() -> list[dict[str, object]]:
    period = PeriodSpec(start_date=date(2025, 1, 6), end_date=date(2025, 1, 12), granularity="week")
    anomaly_period = PeriodSpec(start_date=date(2025, 1, 1), end_date=date(2025, 1, 8))
    series = MetricSeries(
        metric_name="gmv", unit="BRL", period=anomaly_period, source_refs=["golden.fixture"], definition_version="1.0",
        points=[MetricPoint(bucket=anomaly_period.start_date + timedelta(days=index), value=value)
                for index, value in enumerate([100, 100, 100, 100, 100, 100, 100, 180])],
    )
    anomalies = detect_series_anomalies(series, min_samples=7)
    contribution = calculate_change_contribution(
        "gmv", "payment_type", date(2025, 1, 8), date(2025, 1, 7),
        [{"dimension_value": "card", "value": 160}, {"dimension_value": "voucher", "value": 20}],
        [{"dimension_value": "card", "value": 90}, {"dimension_value": "voucher", "value": 10}],
    )
    details: list[dict[str, object]] = [
        {"id": "phase6-anomaly", "passed": len(anomalies) == 1 and anomalies[0].metric_date == date(2025, 1, 8)},
        {"id": "phase6-contribution", "passed": round(sum(item.delta_value for item in contribution.items), 2) == contribution.total_delta},
    ]
    with TemporaryDirectory() as temporary:
        target_store = TargetStore(str(Path(temporary) / "phase6.sqlite"))
        target_store.put(MetricTarget("gmv", period.start_date, period.end_date, 12000, "v1", date(2025, 1, 1), "finance"))
        report = build_periodic_report(
            period, ["gmv", "order_count", "avg_order_value"], fetcher=_fetch,
            target_lookup=lambda metric, report_period: target_store.get(metric, report_period),
            anomalies=anomalies, contributions=[contribution], auto_enrich=False,
            chart_output_dir=str(Path(temporary) / "charts"),
        )
        month = PeriodSpec(start_date=date(2025, 1, 1), end_date=date(2025, 1, 31), granularity="month")
        monthly_report = build_periodic_report(
            month, ["gmv", "order_count", "avg_order_value"], fetcher=_fetch,
            anomalies=anomalies, contributions=[contribution], auto_enrich=False,
            chart_output_dir=str(Path(temporary) / "monthly-charts"),
        )
        weekly_score = score_phase6_report(report.model_dump(mode="json"))
        monthly_score = score_phase6_report(monthly_report.model_dump(mode="json"))
        natural_month_baseline = all(
            metric.period_over_period is not None
            and metric.period_over_period.baseline_period is not None
            and metric.period_over_period.baseline_period.start_date == date(2024, 12, 1)
            and metric.period_over_period.baseline_period.end_date == date(2024, 12, 31)
            for metric in monthly_report.metrics
        )
        details.append({
            "id": "phase6-report",
            "passed": all(weekly_score.values()) and all(monthly_score.values()) and natural_month_baseline,
        })
        details.append({"id": "phase6-target", "passed": report.metrics[0].target_comparison is not None and report.metrics[0].target_comparison.target_version == "v1"})
        store = ReportStore(str(Path(temporary) / "publish.sqlite"))
        draft = store.create_draft(report, "draft")
        approval = store.review(report.report_id, int(str(draft["version"])), "approve", "reviewer")
        published = store.publish(report.report_id, approval.report_version, "publisher")
        details.append({"id": "phase6-approval", "passed": published["status"] == "published"})
        adapter = LocalNotificationAdapter()
        first_count = process_pending_events(store.outbox, adapter)
        second_count = process_pending_events(store.outbox, adapter)
        details.append({"id": "phase6-outbox", "passed": first_count == 3 and second_count == 0})
        with TemporaryDirectory() as reject_tmp:
            reject_store = ReportStore(str(Path(reject_tmp) / "reject.sqlite"))
            reject_draft = reject_store.create_draft(report, "draft")
            reject_store.review(report.report_id, int(str(reject_draft["version"])), "reject", "reviewer")
            try:
                reject_store.publish(report.report_id, int(str(reject_draft["version"])), "publisher")
                rejected_blocked = False
            except ValueError:
                rejected_blocked = True
            details.append({"id": "phase6-approval-reject", "passed": rejected_blocked})
        plateau_series = MetricSeries(
            metric_name="gmv", unit="BRL", period=anomaly_period, source_refs=["golden.fixture"], definition_version="1.0",
            points=[MetricPoint(bucket=anomaly_period.start_date + timedelta(days=index), value=100.0) for index in range(8)],
        )
        details.append({"id": "phase6-anomaly-plateau", "passed": detect_series_anomalies(plateau_series, min_samples=7) == []})
    return details

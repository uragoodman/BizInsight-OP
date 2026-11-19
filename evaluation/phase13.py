"""Feature-path acceptance cases for reports and payment contribution analysis.

These cases exercise distinct outcomes through the current report builder,
report store, IAM permission check, and period contribution repository helper.
They are reported separately from the graph trajectory gate because most are
business-function contracts rather than agent-routing cases.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.analysis.contribution import calculate_change_contribution
from app.analysis.contracts import PeriodSpec
from app.analysis.periodic_reports import build_periodic_report, report_to_markdown
from app.analysis.publishing import ReportStore
from app.data.repositories import daily_metrics
from app.iam.service import require_permission
from app.tenancy import TenantContext
from evaluation.scorers import score_phase6_report


def _period(granularity: str) -> PeriodSpec:
    if granularity == "month":
        return PeriodSpec(start_date=date(2025, 1, 1), end_date=date(2025, 1, 31), granularity="month")
    return PeriodSpec(start_date=date(2025, 1, 6), end_date=date(2025, 1, 12), granularity="week")


def _fetch(metric: str, day: date) -> dict[str, object]:
    del day
    values = {"gmv": 120.0, "order_count": 12.0, "avg_order_value": 10.0}
    return {"value": values.get(metric), "source_table": "golden.fixture"}


def _case(case_id: str, feature: str, path: str, passed: bool, **checks: object) -> dict[str, object]:
    return {"id": case_id, "category": "feature_path", "feature": feature, "path": path,
            "passed": bool(passed), "checks": checks}


def run_phase13_acceptance() -> list[dict[str, object]]:
    details: list[dict[str, object]] = []

    # Report generation: weekly and monthly are separate supported products.
    for granularity in ("week", "month"):
        report = build_periodic_report(
            _period(granularity), ["gmv", "order_count"], fetcher=_fetch,
            include_charts=True, auto_enrich=False,
            contributions=[calculate_change_contribution(
                "gmv", "payment_type", _period(granularity).end_date,
                _period(granularity).end_date.replace(day=max(1, _period(granularity).end_date.day - 1)),
                [{"dimension_value": "card", "value": 90.0}],
                [{"dimension_value": "card", "value": 80.0}],
            )],
        )
        score = score_phase6_report(report.model_dump(mode="json"))
        expected_type = "weekly" if granularity == "week" else "monthly"
        details.append(_case(
            f"report-{granularity}-success", f"{granularity}_report", "success",
            report.report_type == expected_type and all(score.values()),
            report_type=report.report_type, report_checks=score,
        ))

    missing = build_periodic_report(
        _period("week"), ["gmv"],
        fetcher=lambda metric, day: {"value": None, "source_table": "golden.fixture"},
        include_charts=False, auto_enrich=False,
    )
    details.append(_case(
        "report-missing-data", "weekly_report", "missing_data",
        missing.metrics[0].value is None and "missing_metric_data" in missing.risks,
        metric_value=missing.metrics[0].value, risks=missing.risks,
    ))

    viewer = TenantContext(
        tenant_id="feature-path", user_id="viewer", role="viewer",
        scope_type="seller", seller_id="seller-feature-path",
    )
    try:
        require_permission(viewer, "report_export")
        denied = False
    except PermissionError:
        denied = True
    details.append(_case("report-permission-denied", "report", "permission_denied", denied,
                         role=viewer.role, required_permission="report_export"))

    def fail_fetch(_metric: str, _day: date) -> dict[str, object]:
        raise RuntimeError("fixture source unavailable")

    try:
        build_periodic_report(
            _period("week"), ["gmv"], fetcher=fail_fetch,
            include_charts=False, auto_enrich=False,
        )
        dependency_failed = False
    except RuntimeError as exc:
        dependency_failed = str(exc) == "fixture source unavailable"
    details.append(_case("report-dependency-failure", "weekly_report", "dependency_failure",
                         dependency_failed, failure="fixture source unavailable"))

    # Review rejection blocks publication. A revised draft becomes a new
    # version, can be reviewed, and can then be published.
    with TemporaryDirectory() as temporary:
        report = build_periodic_report(
            _period("week"), ["gmv"], fetcher=_fetch,
            include_charts=False, auto_enrich=False,
        )
        store = ReportStore(str(Path(temporary) / "report.sqlite"))
        draft = store.create_draft(report, report_to_markdown(report), actor="author")
        rejected = store.review(report.report_id, int(draft["version"]), "reject", "reviewer", "补充数据说明")
        try:
            store.publish(report.report_id, rejected.report_version, "publisher")
            rejection_blocks_publish = False
        except ValueError:
            rejection_blocks_publish = True
        revised = store.create_draft(report, report_to_markdown(report) + "\n\n修订：补充数据说明。", actor="author")
        approved = store.review(report.report_id, int(revised["version"]), "approve", "reviewer", "已复核")
        published = store.publish(report.report_id, approved.report_version, "publisher")
        resumed = (rejection_blocks_publish and int(revised["version"]) > rejected.report_version
                   and published["status"] == "published")
        details.append(_case("report-review-reject-recover", "report_review", "rejected_then_revised_and_published",
                             resumed, rejected_version=rejected.report_version,
                             revised_version=int(revised["version"]), final_status=published["status"]))

    # Payment contribution: reconcile real group changes; preserve an explicit
    # not_found result for empty periods; enforce analysis permission; surface
    # source failures instead of turning them into a fabricated zero result.
    contribution = calculate_change_contribution(
        "gmv", "payment_type", date(2025, 1, 12), date(2025, 1, 5),
        [{"dimension_value": "card", "value": 90.0}, {"dimension_value": "wallet", "value": 30.0}],
        [{"dimension_value": "card", "value": 70.0}, {"dimension_value": "wallet", "value": 30.0}],
        overall_total_delta=20.0, current_total=120.0, baseline_total=100.0,
    )
    details.append(_case("payment-contribution-success", "payment_contribution", "success",
                         round(sum(item.delta_value for item in contribution.items), 2) == contribution.total_delta,
                         total_delta=contribution.total_delta,
                         group_deltas={item.dimension_value: item.delta_value for item in contribution.items}))

    empty_breakdown = {"items": [], "start_date": "2025-01-01", "end_date": "2025-01-07",
                       "source_table": "golden.fixture"}
    with patch.object(daily_metrics, "breakdown_by_dimension_period", return_value=empty_breakdown):
        missing_result = daily_metrics.compare_breakdown_by_dimension_period(
            "gmv", date(2025, 1, 8), date(2025, 1, 14),
            date(2025, 1, 1), date(2025, 1, 7), "payment_type",
        )
    details.append(_case("payment-contribution-missing-data", "payment_contribution", "missing_data",
                         missing_result["status"] == "not_found" and not missing_result["current_items"],
                         status=missing_result["status"]))

    try:
        require_permission(viewer, "analyze")
        payment_denied = False
    except PermissionError:
        payment_denied = True
    details.append(_case("payment-contribution-permission-denied", "payment_contribution",
                         "permission_denied", payment_denied, role=viewer.role,
                         required_permission="analyze"))

    def dependency_down(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("payment breakdown source unavailable")

    try:
        with patch.object(daily_metrics, "breakdown_by_dimension_period", dependency_down):
            daily_metrics.compare_breakdown_by_dimension_period(
                "gmv", date(2025, 1, 8), date(2025, 1, 14),
                date(2025, 1, 1), date(2025, 1, 7), "payment_type",
            )
        payment_dependency_failed = False
    except RuntimeError as exc:
        payment_dependency_failed = str(exc) == "payment breakdown source unavailable"
    details.append(_case("payment-contribution-dependency-failure", "payment_contribution",
                         "dependency_failure", payment_dependency_failed,
                         failure="payment breakdown source unavailable"))

    return details

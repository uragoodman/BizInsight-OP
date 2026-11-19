from __future__ import annotations

from datetime import date, datetime, timezone
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

from app.analysis.attribution import _platform_dimension_finding
from app.analysis.contracts import (
    PeriodComparison,
    PeriodSpec,
    PeriodicReport,
    ReportMetric,
)
from app.analysis.report_alignment import (
    attribution_snapshot_key,
    capture_report_attribution_snapshots,
)
def _report() -> PeriodicReport:
    current = PeriodSpec(start_date=date(2025, 2, 1), end_date=date(2025, 2, 28), granularity="month")
    baseline = PeriodSpec(start_date=date(2025, 1, 1), end_date=date(2025, 1, 31), granularity="month")
    metric = ReportMetric(
        metric_name="gmv", value=100.0, unit="BRL",
        period_over_period=PeriodComparison(
            comparison_type="period_over_period", current_period=current,
            baseline_period=baseline, current_value=100.0, baseline_value=80.0,
            delta_value=20.0, delta_ratio=25.0, status="ok",
        ),
        evidence_ids=["ev-gmv"],
    )
    return PeriodicReport(
        report_id="report-snapshot", report_type="monthly", period=current,
        title="Monthly", summary="Snapshot fixture", metrics=[metric],
        evidence_records={"ev-gmv": {"kind": "metric_series"}},
        lineage_refs=["analytics.fact_orders"],
        data_freshness_at=datetime(2025, 2, 28, tzinfo=timezone.utc),
        scope={"scope_type": "platform"},
    )


def test_capture_pins_reconciled_dimension_facts_into_report():
    report = _report()
    capability = SimpleNamespace(method_ids=("additive_change_v1",), dimensions=("product_category",))
    metric_definition = SimpleNamespace(
        attribution=SimpleNamespace(for_scope=lambda _scope: capability),
        period_aggregation=None,
    )
    registry = SimpleNamespace(get=lambda _metric: metric_definition)
    raw = {
        "status": "ok", "current_total": 100.0, "baseline_total": 80.0,
        "total_delta": 20.0,
        "current_items": [{"dimension_value": "A", "value": 100.0}],
        "baseline_items": [{"dimension_value": "A", "value": 80.0}],
        "source_table": "analytics.fact_orders",
    }
    with patch("app.metrics.registry.get_registry", return_value=registry), \
         patch("app.analysis.semantic.get_metric_definition", return_value=SimpleNamespace(allowed_dimensions=("product_category",), period_aggregation=None)), \
         patch("app.data.repositories.daily_metrics.compare_breakdown_by_dimension_period", return_value=raw):
        pinned = capture_report_attribution_snapshots(report)

    snapshot = pinned.attribution_breakdowns[attribution_snapshot_key("gmv", "product_category")]
    assert snapshot["status"] == "ok"
    assert snapshot["current_items"] == raw["current_items"]
    assert snapshot["baseline_items"] == raw["baseline_items"]
    assert snapshot["snapshot_digest"]


def test_insight_attribution_uses_report_snapshot_not_live_source():
    report = _report()
    capability = SimpleNamespace(method_ids=("additive_change_v1",), dimensions=("product_category",))
    metric_definition = SimpleNamespace(
        attribution=SimpleNamespace(for_scope=lambda _scope: capability), period_aggregation=None,
    )
    registry = SimpleNamespace(get=lambda _metric: metric_definition)
    raw = {
        "status": "ok", "current_total": 100.0, "baseline_total": 80.0,
        "total_delta": 20.0,
        "current_items": [{"dimension_value": "A", "value": 100.0}],
        "baseline_items": [{"dimension_value": "A", "value": 80.0}],
        "source_table": "analytics.fact_orders",
    }
    with patch("app.metrics.registry.get_registry", return_value=registry), \
         patch("app.analysis.semantic.get_metric_definition", return_value=SimpleNamespace(allowed_dimensions=("product_category",), period_aggregation=None)), \
         patch("app.data.repositories.daily_metrics.compare_breakdown_by_dimension_period", return_value=raw):
        pinned = capture_report_attribution_snapshots(report)

    with patch(
        "app.analysis.attribution.compare_breakdown_by_dimension_period",
        side_effect=AssertionError("insight path must not query mutable live tables"),
    ):
        finding = _platform_dimension_finding(
            pinned, pinned.metrics[0], "product_category",
            pinned.metrics[0].period_over_period.baseline_period,
        )
    assert finding.total_delta == 20.0
    assert finding.items[0].dimension_value == "A"


def test_missing_snapshot_fails_closed_instead_of_requerying():
    report = _report()
    metric = report.metrics[0]
    with patch(
        "app.analysis.attribution.compare_breakdown_by_dimension_period",
        side_effect=AssertionError("must not query live source"),
    ):
        from app.analysis.attribution import AttributionDataInsufficient, _platform_dimension_finding

        try:
            _platform_dimension_finding(report, metric, "product_category", metric.period_over_period.baseline_period)
        except AttributionDataInsufficient as exc:
            assert exc.period_statuses == {"snapshot": "report_snapshot_missing"}
        else:
            raise AssertionError("missing report snapshot must fail closed")


def test_report_snapshot_reuses_one_connection_for_repository_reads():
    from app.data import report_snapshot

    connection = MagicMock()
    connection.begin.return_value = nullcontext()
    connection_context = MagicMock()
    connection_context.__enter__.return_value = connection
    with patch.object(report_snapshot.engine, "connect", return_value=connection_context) as connect:
        with report_snapshot.report_read_snapshot() as outer:
            assert outer is connection
            with report_snapshot.repository_connection() as inner:
                assert inner is outer
            assert report_snapshot.current_report_snapshot_connection() is connection
        assert report_snapshot.current_report_snapshot_connection() is None
    connect.assert_called_once_with()
    assert "REPEATABLE READ" in str(connection.execute.call_args.args[0])


def test_seller_authorization_transaction_joins_report_snapshot():
    from app.data import report_snapshot
    from app.data.rls import seller_query_transaction

    context = SimpleNamespace(
        tenant_id="tenant-a", is_platform_admin=False, scope_type="seller",
    )
    scope = SimpleNamespace(
        seller_id="seller-a", default_start_date=date(2025, 2, 1),
        default_end_date=date(2025, 2, 28),
    )
    connection = object()
    with patch("app.tenancy.current_tenant_context", return_value=context), \
         patch.object(report_snapshot, "current_report_snapshot_connection", return_value=connection), \
         patch("app.data.rls.set_local_context") as set_local, \
         patch.object(report_snapshot.engine, "begin") as begin:
        with seller_query_transaction(report_snapshot.engine, scope) as yielded:
            assert yielded is connection
        set_local.assert_called_once()
        begin.assert_not_called()


def test_ratio_snapshot_is_captured_even_when_metric_is_after_first_two_report_metrics():
    from app.analysis.report_alignment import ratio_attribution_snapshot_key
    from app.data.repositories import daily_metrics

    report = _report()
    template = report.metrics[0]

    def metric(name, current, baseline, unit):
        comparison = template.period_over_period.model_copy(update={
            "current_value": current,
            "baseline_value": baseline,
            "delta_value": current - baseline,
            "delta_ratio": (current - baseline) * 100 / baseline if baseline else None,
        })
        return template.model_copy(update={
            "metric_name": name,
            "value": current,
            "unit": unit,
            "period_over_period": comparison,
        })

    report = report.model_copy(update={"metrics": [
        metric("unknown_metric_a", 2.0, 1.0, "count"),
        metric("unknown_metric_b", 3.0, 2.0, "count"),
        metric("avg_order_value", 10.0, 10.0, "BRL/order"),
        metric("gmv", 100.0, 80.0, "BRL"),
        metric("order_count", 10.0, 8.0, "orders"),
    ]})
    ratio_facts = {
        "status": "ok", "complete": True, "source_table": "analytics.fact_orders",
        "baseline_groups": {"early": {"numerator": 80.0, "denominator": 8.0}},
        "current_groups": {"early": {"numerator": 100.0, "denominator": 10.0}},
        "baseline_numerator_total": 80.0, "current_numerator_total": 100.0,
        "baseline_denominator_total": 8.0, "current_denominator_total": 10.0,
    }
    with patch.object(daily_metrics, "compare_ratio_facts_by_dimension_period", return_value=ratio_facts):
        pinned = capture_report_attribution_snapshots(report)

    snapshot = pinned.attribution_breakdowns[
        ratio_attribution_snapshot_key("avg_order_value", "month_position")
    ]
    assert snapshot["status"] == "ok"
    assert snapshot["metric_id"] == "avg_order_value"
    assert snapshot["baseline_groups"] == ratio_facts["baseline_groups"]
    assert snapshot["current_groups"] == ratio_facts["current_groups"]
    assert snapshot["snapshot_digest"]

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.analysis.contracts import ChartArtifact, MetricPoint, MetricSeries, PeriodComparison, PeriodSpec, ReportScope
from app.analysis.contribution import calculate_change_contribution
from app.analysis.events import EventOutbox, LocalNotificationAdapter, process_pending_events
from app.analysis.charts import render_trend_chart
from app.analysis.periodic_reports import build_periodic_report, report_to_markdown
from app.analysis.periods import aggregate_series, compare_periods, previous_period, query_metric_series
from app.analysis.publishing import ReportStore
from app.analysis.semantic import METRIC_REGISTRY, get_metric_definition
from app.analysis.statistical_anomaly import detect_and_emit_anomalies, detect_series_anomalies
from app.analysis.targets import MetricTarget, TargetStore, compare_to_target
from app.agents.data_query_agent import build_metric_query_plan
from app.agents.alert_agent import AlertAgent
from app.mcp.tools import get_supported_tools, query_metric_batch_period
from app.data.repositories.daily_metrics import _breakdown_sql_for_dimension
from app.data.repositories.daily_metrics import query_metric_period as repository_query_metric_period
from app.main import app
from app.core.config import get_settings
from app.graph.nodes.human_review import human_review
from app.graph.nodes.critic import _review_periodic_report
from evaluation.scorers import score_phase6_report


def fixture_fetcher(metric_name: str, metric_date: date) -> dict:
    offset = (metric_date - date(2025, 1, 1)).days
    values = {
        "gmv": 100.0 + offset,
        "order_count": 10.0,
        "avg_order_value": (100.0 + offset) / 10,
    }
    return {"value": values.get(metric_name), "source_table": "fixture.metrics"}


def test_phase6_contract_rejects_inverted_period() -> None:
    with pytest.raises(ValueError, match="end_date"):
        PeriodSpec(start_date=date(2025, 2, 1), end_date=date(2025, 1, 1))


def test_phase6_contract_rejects_unknown_version_and_invalid_series() -> None:
    with pytest.raises(ValueError, match="contract_version"):
        PeriodSpec.model_validate({
            "contract_version": "2.0", "start_date": "2025-01-01", "end_date": "2025-01-01",
        })
    period = PeriodSpec(start_date=date(2025, 1, 1), end_date=date(2025, 1, 2))
    with pytest.raises(ValueError, match="ordered"):
        MetricSeries(
            metric_name="gmv", unit="BRL", period=period, source_refs=["fixture"], definition_version="1.0",
            points=[MetricPoint(bucket=date(2025, 1, 2), value=2), MetricPoint(bucket=date(2025, 1, 1), value=1)],
        )


def test_natural_month_comparison_uses_complete_previous_month() -> None:
    march = PeriodSpec(
        start_date=date(2025, 3, 1), end_date=date(2025, 3, 31), granularity="month",
        timezone="Asia/Shanghai",
    )
    baseline = previous_period(march)
    assert (baseline.start_date, baseline.end_date) == (date(2025, 2, 1), date(2025, 2, 28))
    assert baseline.timezone == "Asia/Shanghai"


def test_contract_rejects_invalid_timezone_and_unverifiable_chart() -> None:
    with pytest.raises(ValueError, match="IANA"):
        PeriodSpec(start_date=date(2025, 1, 1), end_date=date(2025, 1, 1), timezone="Mars/Olympus")
    with pytest.raises(ValueError, match="ok chart requires"):
        ChartArtifact(chart_type="trend", status="ok")
    with pytest.raises(ValueError, match="reason"):
        ChartArtifact(chart_type="trend", status="chart_unavailable")
    period = PeriodSpec(start_date=date(2025, 1, 1), end_date=date(2025, 1, 2))
    with pytest.raises(ValueError, match="declared period"):
        MetricSeries(
            metric_name="gmv", unit="BRL", period=period, source_refs=["fixture"], definition_version="1.0",
            points=[MetricPoint(bucket=date(2025, 1, 3), value=3)],
        )


def test_semantic_registry_covers_existing_metrics_and_lineage() -> None:
    assert len(METRIC_REGISTRY) >= 20
    # Phase 6 M1b 漏斗指标必须在注册表中
    for funnel_metric in ("order_approval_rate", "order_shipment_rate", "order_final_delivery_rate"):
        assert funnel_metric in METRIC_REGISTRY, f"{funnel_metric} missing from registry"
    # Phase 6 M3 评论指标必须在注册表中
    assert "review_comment_rate" in METRIC_REGISTRY, "review_comment_rate missing from registry"
    definition = get_metric_definition("gmv")
    assert "analytics.v_delivered_orders" in definition.lineage_ref
    assert "metric:gmv" in definition.lineage_ref
    assert "payment_type" in definition.allowed_dimensions


def test_every_advertised_metric_dimension_pair_has_executable_sql() -> None:
    for definition in METRIC_REGISTRY.values():
        for dimension in definition.allowed_dimensions:
            sql, source = _breakdown_sql_for_dimension(definition.name, dimension)
            assert "SELECT" in sql and source


def test_weighted_ratio_uses_components_not_average_of_daily_ratios() -> None:
    period = PeriodSpec(start_date=date(2025, 1, 1), end_date=date(2025, 1, 2))

    def uneven(metric: str, day: date) -> dict:
        values = {
            ("gmv", date(2025, 1, 1)): 100.0,
            ("order_count", date(2025, 1, 1)): 10.0,
            ("avg_order_value", date(2025, 1, 1)): 10.0,
            ("gmv", date(2025, 1, 2)): 900.0,
            ("order_count", date(2025, 1, 2)): 30.0,
            ("avg_order_value", date(2025, 1, 2)): 30.0,
        }
        return {"value": values[(metric, day)], "source_table": "fixture"}

    series = query_metric_series("avg_order_value", period, uneven)
    assert aggregate_series(series) == 25.0


@pytest.mark.parametrize("metric,expected", [
    ("gmv", 1000.0), ("order_count", 10.0), ("avg_order_value", 100.0),
    ("cancel_rate", 10.0), ("freight_value", 100.0), ("review_score_avg", 4.0),
    ("delivery_delay_rate", 25.0), ("customer_count", 8.0),
    ("repeat_customer_count", 2.0), ("repeat_purchase_rate", 25.0),
    ("avg_items_per_order", 2.5), ("avg_delivery_lead_time", 3.0),
    ("on_time_delivery_rate", 75.0), ("bad_review_rate", 20.0), ("freight_rate", 10.0),
])
def test_entity_level_period_repository_aggregates_all_metrics(monkeypatch, metric, expected) -> None:
    row = {
        "gmv": 1000, "order_count": 10, "freight_value": 100, "customer_count": 8,
        "repeat_customer_count": 2, "avg_items_per_order": 2.5, "total_orders": 10,
        "canceled_orders": 1, "delivered_orders": 8, "delayed_orders": 2,
        "on_time_orders": 6, "avg_delivery_lead_time": 3, "review_score_avg": 4,
        "reviewed_orders": 5, "bad_review_orders": 1,
    }

    class Result:
        def mappings(self): return self
        def first(self): return row

    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def execute(self, *args, **kwargs): return Result()

    class Engine:
        def connect(self): return Connection()

    monkeypatch.setattr("app.data.repositories.daily_metrics.engine", Engine())
    result = repository_query_metric_period(metric, "2025-01-01", "2025-01-31")
    assert result["value"] == expected


def test_period_comparison_handles_zero_baseline() -> None:
    current = PeriodSpec(start_date=date(2025, 1, 2), end_date=date(2025, 1, 2))
    baseline = PeriodSpec(start_date=date(2025, 1, 1), end_date=date(2025, 1, 1))

    def values(metric: str, day: date) -> dict:
        return {"value": 0.0 if day.day == 1 else 10.0, "source_table": "fixture"}

    result = compare_periods("gmv", current, baseline, fetcher=values)
    assert result.status == "zero_baseline"
    assert result.delta_value == 10.0
    assert result.delta_ratio is None


def test_mad_detector_finds_known_anomaly_and_skips_missing() -> None:
    period = PeriodSpec(start_date=date(2025, 1, 1), end_date=date(2025, 1, 10))
    values = [100, 101, None, 99, 100, 101, 100, 99, 100, 200]
    series = MetricSeries(
        metric_name="gmv", unit="BRL", period=period, source_refs=["fixture"], definition_version="1.0",
        points=[MetricPoint(bucket=period.start_date + timedelta(days=index), value=value) for index, value in enumerate(values)],
    )
    findings = detect_series_anomalies(series, min_samples=7)
    assert [item.metric_date for item in findings] == [date(2025, 1, 10)]
    assert findings[0].direction == "up"


def test_anomaly_duration_counts_consecutive_same_direction_days() -> None:
    period = PeriodSpec(start_date=date(2025, 1, 1), end_date=date(2025, 1, 9))
    series = MetricSeries(
        metric_name="gmv", unit="BRL", period=period, source_refs=["fixture"], definition_version="1.0",
        points=[MetricPoint(bucket=period.start_date + timedelta(days=index), value=value)
                for index, value in enumerate([10, 10, 10, 10, 10, 10, 10, 50, 60])],
    )
    findings = detect_series_anomalies(series, min_samples=7)
    assert [item.duration_days for item in findings] == [1, 2]


def test_anomaly_scan_emits_deduplicated_alert(tmp_path) -> None:
    period = PeriodSpec(start_date=date(2025, 1, 1), end_date=date(2025, 1, 8))
    series = MetricSeries(
        metric_name="gmv", unit="BRL", period=period, source_refs=["fixture"], definition_version="1.0",
        points=[MetricPoint(bucket=period.start_date + timedelta(days=i), value=value)
                for i, value in enumerate([10, 10, 10, 10, 10, 10, 10, 50])],
    )
    outbox = EventOutbox(str(tmp_path / "anomaly.sqlite"))
    assert len(detect_and_emit_anomalies(series, outbox, min_samples=7)) == 1
    detect_and_emit_anomalies(series, outbox, min_samples=7)
    assert len(outbox.pending()) == 1
    assert outbox.pending()[0].event_type == "AnomalyDetected"


def test_anomaly_outbox_applies_metric_direction_cooldown(tmp_path) -> None:
    outbox = EventOutbox(str(tmp_path / "cooldown.sqlite"))
    first = outbox.emit_anomaly(
        {"metric_name": "gmv", "direction": "down", "metric_date": "2025-01-01"},
        "anomaly:gmv:2025-01-01:mad", cooldown_hours=24,
    )
    suppressed = outbox.emit_anomaly(
        {"metric_name": "gmv", "direction": "down", "metric_date": "2025-01-02"},
        "anomaly:gmv:2025-01-02:mad", cooldown_hours=24,
    )
    distinct_direction = outbox.emit_anomaly(
        {"metric_name": "gmv", "direction": "up", "metric_date": "2025-01-02"},
        "anomaly:gmv:2025-01-02:up", cooldown_hours=24,
    )
    assert suppressed.event_id == first.event_id
    assert distinct_direction.event_id != first.event_id
    assert len(outbox.pending()) == 2


def test_alert_agent_only_consumes_structured_finding(tmp_path) -> None:
    period = PeriodSpec(start_date=date(2025, 1, 1), end_date=date(2025, 1, 8))
    series = MetricSeries(
        metric_name="gmv", unit="BRL", period=period, source_refs=["fixture"], definition_version="1.0",
        points=[MetricPoint(bucket=period.start_date + timedelta(days=i), value=value)
                for i, value in enumerate([10, 10, 10, 10, 10, 10, 10, 50])],
    )
    finding = detect_series_anomalies(series, min_samples=7)[0]
    summary, event = AlertAgent(EventOutbox(str(tmp_path / "agent.sqlite"))).handle(finding)
    assert "基线" in summary
    assert event.event_type == "AnomalyDetected"


def test_phase6_period_tools_are_allowlisted() -> None:
    tools = get_supported_tools()
    assert "query_metric_period" in tools
    assert "compare_metric_period" in tools
    assert "query_metric_batch_period" in tools
    assert "compare_breakdown_by_dimension" in tools
    assert "compare_breakdown_by_dimension_period" in tools
    assert "list_metric_definitions" in tools
    assert "get_metric_definition" in tools


def test_batch_period_tool_validates_natural_period_and_returns_versions(monkeypatch) -> None:
    monkeypatch.setattr("app.mcp.tools._database_read", lambda operation, *args: {
        "value": 42.0, "source_table": "fixture.period",
    })
    result = query_metric_batch_period(
        ["gmv", "order_count"], "2025-01-06", "2025-01-12", "week",
    )
    assert result["status"] == "ok"
    assert [item["metric_name"] for item in result["items"]] == ["gmv", "order_count"]
    assert all(item["definition_version"] == "1.0" for item in result["items"])
    with pytest.raises(ValueError, match="Monday-Sunday"):
        query_metric_batch_period(["gmv"], "2025-01-07", "2025-01-13", "week")


def test_change_contribution_reconciles_total_delta() -> None:
    result = calculate_change_contribution(
        "gmv", "payment_type", date(2025, 1, 2), date(2025, 1, 1),
        [{"dimension_value": "card", "value": 130}, {"dimension_value": "cash", "value": 20}],
        [{"dimension_value": "card", "value": 100}, {"dimension_value": "cash", "value": 30}],
    )
    assert result.total_delta == 20
    assert sum(item.delta_value for item in result.items) == result.total_delta
    assert sum(item.contribution_ratio or 0 for item in result.items) == pytest.approx(1)
    assert result.residual_explainable is True
    assert result.residual_tolerance == pytest.approx(0.03)
    assert result.residual_note


def test_truncated_contribution_adds_other_and_conserves_total() -> None:
    result = calculate_change_contribution(
        "gmv", "customer_state", date(2025, 2, 28), date(2025, 1, 31),
        [{"dimension_value": "SP", "value": 80}],
        [{"dimension_value": "SP", "value": 50}],
        overall_total_delta=50, current_total=150, baseline_total=100,
    )
    other = next(item for item in result.items if item.dimension_value == "Other")
    assert other.delta_value == 20
    assert sum(item.delta_value for item in result.items) == result.total_delta == 50


def test_contribution_rejects_overall_delta_that_disagrees_with_period_totals() -> None:
    from app.analysis.attribution_methods import AttributionDataError

    with pytest.raises(AttributionDataError, match="overall change"):
        calculate_change_contribution(
            "gmv", "product_category", date(2025, 2, 28), date(2025, 1, 31),
            [{"dimension_value": "A", "value": 70}],
            [{"dimension_value": "A", "value": 60}],
            overall_total_delta=25, current_total=120, baseline_total=100,
        )


@pytest.mark.parametrize(
    "rows, message",
    [
        ([{"dimension_value": "A", "value": 1}, {"dimension_value": "A", "value": 2}], "duplicate dimension"),
        ([{"dimension_value": "A", "value": float("nan")}], "non-finite"),
    ],
)
def test_change_contribution_rejects_duplicate_groups_and_non_finite_values(rows, message) -> None:
    from app.analysis.attribution_methods import AttributionDataError

    with pytest.raises(AttributionDataError, match=message):
        calculate_change_contribution(
            "gmv", "product_category", date(2025, 2, 28), date(2025, 1, 31),
            rows, [],
        )


def test_contribution_question_maps_to_single_paired_mcp_tool() -> None:
    plan = build_metric_query_plan("拆解 2018-08-10 与 2018-08-09 的 GMV 按支付方式贡献度")
    assert plan.query_type == "metric_contribution"
    assert plan.tool_name == "compare_breakdown_by_dimension"
    assert plan.current_date == "2018-08-10"
    assert plan.baseline_date == "2018-08-09"
    assert plan.dimension == "payment_type"


def test_periodic_report_contains_comparisons_and_lineage(tmp_path) -> None:
    period = PeriodSpec(start_date=date(2025, 1, 13), end_date=date(2025, 1, 19), granularity="week")
    contribution = calculate_change_contribution(
        "gmv", "payment_type", date(2025, 1, 19), date(2025, 1, 18),
        [{"dimension_value": "card", "value": 120}], [{"dimension_value": "card", "value": 100}],
    )
    report = build_periodic_report(
        period, ["gmv", "avg_order_value"], fetcher=fixture_fetcher,
        include_charts=True, contributions=[contribution], auto_enrich=False,
        chart_output_dir=str(tmp_path),
    )
    markdown = report_to_markdown(report)
    assert report.report_type == "weekly"
    assert report.metrics[0].period_over_period is not None
    assert report.metrics[0].year_over_year is not None
    assert len(report.lineage_refs) == 2
    assert "环比" in markdown and "同比" in markdown
    assert all(score_phase6_report(report.model_dump(mode="json")).values())


def test_periodic_report_is_reproducible_for_same_fixture_input() -> None:
    period = PeriodSpec(start_date=date(2025, 1, 13), end_date=date(2025, 1, 19), granularity="week")
    first = build_periodic_report(
        period, ["gmv", "avg_order_value"], fetcher=fixture_fetcher,
        include_charts=False, auto_enrich=False,
    )
    second = build_periodic_report(
        period, ["gmv", "avg_order_value"], fetcher=fixture_fetcher,
        include_charts=False, auto_enrich=False,
    )
    assert first.model_dump_json() == second.model_dump_json()


def test_critic_rejects_tampered_metric_and_detached_chart(tmp_path) -> None:
    period = PeriodSpec(start_date=date(2025, 1, 13), end_date=date(2025, 1, 19), granularity="week")
    report = build_periodic_report(
        period, ["gmv"], fetcher=fixture_fetcher,
        include_charts=True, chart_output_dir=str(tmp_path), auto_enrich=False,
    )
    tampered = report.model_copy(deep=True)
    tampered.metrics[0].value = 999999
    numeric_review = _review_periodic_report({"periodic_report": tampered.model_dump(mode="json")})
    assert numeric_review["status"] == "rejected"
    assert any("disagrees" in issue for issue in numeric_review["issues"])

    # 图表不再落盘 PNG；被篡改的证据绑定等价于"图表与报告事实失去关联"。
    detached = report.model_copy(deep=True)
    detached.charts[0].evidence_ids = ["ev_detached_from_report"]
    chart_review = _review_periodic_report({"periodic_report": detached.model_dump(mode="json")})
    assert chart_review["status"] == "rejected"
    assert any("references missing evidence" in issue for issue in chart_review["issues"])


def test_headless_trend_chart_is_content_addressed() -> None:
    period = PeriodSpec(start_date=date(2025, 1, 1), end_date=date(2025, 1, 3))
    series = query_metric_series("gmv", period, fixture_fetcher)
    first = render_trend_chart(series)
    second = render_trend_chart(series)
    assert first.status == "ok"
    assert first.data_points == second.data_points
    assert first.content_hash == second.content_hash
    assert first.path is None


def test_real_report_builder_auto_connects_anomaly_contribution_and_both_charts(tmp_path, monkeypatch) -> None:
    period = PeriodSpec(start_date=date(2025, 1, 6), end_date=date(2025, 1, 12), granularity="week")

    def series(metric_name, requested_period):
        days = (requested_period.end_date - requested_period.start_date).days + 1
        values = [100.0] * days
        if days >= 8:
            values[-1] = 180.0
        return MetricSeries(
            metric_name=metric_name, unit="BRL", period=requested_period,
            source_refs=["fixture"], definition_version="1.0",
            points=[MetricPoint(bucket=requested_period.start_date + timedelta(days=index), value=value)
                    for index, value in enumerate(values)],
        )

    def comparison(metric_name, current, baseline, comparison_type, **kwargs):
        return PeriodComparison(
            comparison_type=comparison_type, current_period=current, baseline_period=baseline,
            current_value=180, baseline_value=100, delta_value=80, delta_ratio=80, status="ok",
        )

    monkeypatch.setattr("app.analysis.periodic_reports.query_metric_series", series)
    monkeypatch.setattr("app.analysis.periodic_reports.aggregate_metric_period", lambda *args, **kwargs: 180.0)
    monkeypatch.setattr("app.analysis.periodic_reports.compare_periods", comparison)
    monkeypatch.setattr("app.analysis.periodic_reports.compare_breakdown_by_dimension_period", lambda metric, current_start, current_end, baseline_start, baseline_end, dimension, limit: {
        "status": "ok", "current_items": [{"dimension_value": "A", "value": 180}],
        "baseline_items": [{"dimension_value": "A", "value": 100}], "total_delta": 80,
        "current_total": 180, "baseline_total": 100,
    })
    report = build_periodic_report(
        period, ["gmv"], include_charts=True, chart_output_dir=str(tmp_path),
    )
    assert report.anomalies
    assert {item.dimension for item in report.contributions} == set(get_metric_definition("gmv").allowed_dimensions)
    assert {item.chart_type for item in report.charts} == {"trend", "contribution"}


def test_auto_enrichment_does_not_mislabel_ratio_shifts_as_additive_contribution(monkeypatch) -> None:
    period = PeriodSpec(start_date=date(2025, 1, 6), end_date=date(2025, 1, 12), granularity="week")

    def series(metric_name, requested_period):
        days = (requested_period.end_date - requested_period.start_date).days + 1
        values = [10.0] * days
        if days >= 8:
            values[-1] = 50.0
        return MetricSeries(
            metric_name=metric_name, unit="BRL", period=requested_period,
            source_refs=["fixture"], definition_version="1.0",
            points=[MetricPoint(bucket=requested_period.start_date + timedelta(days=index), value=value)
                    for index, value in enumerate(values)],
        )

    monkeypatch.setattr("app.analysis.periodic_reports.query_metric_series", series)
    monkeypatch.setattr("app.analysis.periodic_reports.aggregate_metric_period", lambda *args, **kwargs: 50.0)
    monkeypatch.setattr("app.analysis.periodic_reports.compare_periods", lambda metric, current, baseline, comparison_type, **kwargs: PeriodComparison(
        comparison_type=comparison_type, current_period=current, baseline_period=baseline,
        current_value=50, baseline_value=10, delta_value=40, delta_ratio=400, status="ok",
    ))
    monkeypatch.setattr(
        "app.analysis.periodic_reports.compare_breakdown_by_dimension_period",
        lambda *args, **kwargs: pytest.fail("ratio contribution query must not execute"),
    )
    report = build_periodic_report(period, ["avg_order_value"], include_charts=False)
    assert report.contributions == []
    assert "contribution_not_decomposable:avg_order_value" in report.risks


def test_lower_is_better_target_variance_has_favorable_sign() -> None:
    period = PeriodSpec(start_date=date(2025, 1, 1), end_date=date(2025, 1, 31), granularity="month")
    target = MetricTarget("cancel_rate", period.start_date, period.end_date, 5.0, "v1", date(2024, 12, 1), "finance")
    result = compare_to_target("cancel_rate", 3.0, period, target)
    assert result.delta_value == 2.0
    assert result.delta_ratio == 40.0


def test_target_store_preserves_versions_and_selects_latest_effective(tmp_path) -> None:
    store = TargetStore(str(tmp_path / "targets.sqlite"))
    period = PeriodSpec(start_date=date(2025, 1, 1), end_date=date(2025, 1, 31), granularity="month")
    store.put(MetricTarget("gmv", period.start_date, period.end_date, 100.0, "v1", date(2024, 12, 1), "finance"))
    store.put(MetricTarget("gmv", period.start_date, period.end_date, 120.0, "v2", date(2024, 12, 15), "finance"))
    selected = store.get("gmv", period)
    assert selected is not None
    assert selected.version == "v2"
    assert selected.target_value == 120.0
    with pytest.raises(ValueError, match="immutable"):
        store.put(MetricTarget("gmv", period.start_date, period.end_date, 999.0, "v2", date(2024, 12, 15), "other"))


def test_outbox_and_publish_workflow_are_idempotent(tmp_path) -> None:
    database = str(tmp_path / "phase6.sqlite")
    outbox = EventOutbox(database)
    first = outbox.emit("MetricWindowClosed", {"period": "2025-W01"}, "window:2025-W01")
    second = outbox.emit("MetricWindowClosed", {"period": "2025-W01"}, "window:2025-W01")
    assert first.event_id == second.event_id
    assert len(outbox.pending()) == 1

    period = PeriodSpec(start_date=date(2025, 1, 6), end_date=date(2025, 1, 12), granularity="week")
    report = build_periodic_report(period, ["gmv"], fetcher=fixture_fetcher, include_charts=False)
    store = ReportStore(database)
    draft = store.create_draft(report, "draft")
    with pytest.raises(ValueError, match="approved"):
        store.publish(report.report_id, int(draft["version"]), "publisher")
    approval = store.review(report.report_id, int(draft["version"]), "edit", "reviewer", edited_content="edited")
    assert store.get(report.report_id, int(draft["version"]))["status"] == "rejected"
    with pytest.raises(ValueError, match="cannot be reviewed"):
        store.review(report.report_id, approval.report_version, "reject", "other-reviewer")
    published = store.publish(report.report_id, approval.report_version, "publisher")
    repeated = store.publish(report.report_id, approval.report_version, "publisher")
    assert published["status"] == "published"
    assert repeated["content_hash"] == published["content_hash"]


def test_report_state_and_outbox_event_commit_atomically(tmp_path, monkeypatch) -> None:
    database = str(tmp_path / "atomic.sqlite")
    period = PeriodSpec(start_date=date(2025, 1, 6), end_date=date(2025, 1, 12), granularity="week")
    report = build_periodic_report(period, ["gmv"], fetcher=fixture_fetcher, include_charts=False)
    store = ReportStore(database)
    monkeypatch.setattr(store.outbox, "emit_in_transaction", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("outbox failed")))
    with pytest.raises(RuntimeError, match="outbox failed"):
        store.create_draft(report, "draft")
    with pytest.raises(KeyError):
        store.get(report.report_id)


def test_outbox_worker_delivers_each_pending_event_once(tmp_path) -> None:
    outbox = EventOutbox(str(tmp_path / "events.sqlite"))
    outbox.emit("MetricWindowClosed", {"period": "2025-W01"}, "window:2025-W01")
    adapter = LocalNotificationAdapter()
    assert process_pending_events(outbox, adapter) == 1
    assert process_pending_events(outbox, adapter) == 0
    assert len(adapter.delivered) == 1


def test_outbox_claim_prevents_second_worker_from_claiming_same_event(tmp_path) -> None:
    outbox = EventOutbox(str(tmp_path / "claims.sqlite"))
    expected = outbox.emit("MetricWindowClosed", {}, "only-once")
    claimed = outbox.claim_next()
    assert claimed is not None and claimed.event_id == expected.event_id
    assert outbox.claim_next() is None


def test_outbox_moves_repeated_delivery_failure_to_dead_letter(tmp_path) -> None:
    outbox = EventOutbox(str(tmp_path / "dead-letter.sqlite"))
    outbox.emit("MetricWindowClosed", {}, "failing-window")

    class FailingAdapter:
        def deliver(self, event) -> None:
            raise RuntimeError("channel unavailable")

    for _ in range(3):
        process_pending_events(outbox, FailingAdapter())
    assert outbox.pending() == []


def test_semantic_api_exposes_lineage() -> None:
    response = TestClient(app).get("/analysis/semantic/metrics/gmv")
    assert response.status_code == 200
    assert response.json()["lineage_ref"].endswith("metric:gmv@1.0")


def test_analysis_write_endpoints_fail_closed_in_production(monkeypatch, tmp_path) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "analysis_write_token", "secret-token")
    monkeypatch.setattr(settings, "phase6_store_path", str(tmp_path / "auth.sqlite"))
    client = TestClient(app)
    denied = client.put("/analysis/targets/gmv", json={
        "period_start": "2025-01-01", "period_end": "2025-01-31", "target_value": 10,
        "version": "v1", "effective_at": "2024-12-01", "created_by": "finance",
    })
    assert denied.status_code == 403
    allowed = client.put("/analysis/targets/gmv", headers={
        "X-Analysis-Token": "secret-token", "X-Actor-ID": "finance",
    }, json={
        "period_start": "2025-01-01", "period_end": "2025-01-31", "target_value": 10,
        "version": "auth-v1", "effective_at": "2024-12-01", "created_by": "finance",
    })
    assert allowed.status_code == 200


def test_chat_review_endpoint_is_protected_in_production(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "analysis_write_token", "secret-token")
    response = TestClient(app).post("/chat/nonexistent/review", json={
        "action": "approve", "actor": "reviewer", "reason": "verified",
    })
    assert response.status_code == 403


def test_report_api_enforces_review_before_publish(tmp_path, monkeypatch) -> None:
    period = PeriodSpec(start_date=date(2025, 1, 6), end_date=date(2025, 1, 12), granularity="week")
    fixture_report = build_periodic_report(period, ["gmv"], fetcher=fixture_fetcher, include_charts=False)
    monkeypatch.setattr("app.api.routes.analysis.build_periodic_report", lambda *args, **kwargs: fixture_report)
    settings = get_settings()
    monkeypatch.setattr(settings, "phase6_store_path", str(tmp_path / "api.sqlite"))
    client = TestClient(app)
    created = client.post("/analysis/reports", json={
        "start_date": "2025-01-06", "end_date": "2025-01-12",
        "granularity": "week", "metric_names": ["gmv"], "include_charts": False,
    })
    assert created.status_code == 200
    payload = created.json()
    report_id, version = payload["report"]["report_id"], payload["version"]
    denied = client.post(f"/analysis/reports/{report_id}/publish", json={"version": version, "actor": "publisher"})
    assert denied.status_code == 409
    approved = client.post(f"/analysis/reports/{report_id}/review", json={
        "version": version, "action": "approve", "actor": "reviewer", "reason": "verified",
    })
    assert approved.status_code == 200
    published = client.post(f"/analysis/reports/{report_id}/publish", json={"version": version, "actor": "publisher"})
    assert published.status_code == 200
    assert published.json()["status"] == "published"


def test_report_api_creates_seller_scoped_report(tmp_path, monkeypatch) -> None:
    period = PeriodSpec(start_date=date(2025, 1, 6), end_date=date(2025, 1, 12), granularity="week")
    fixture_report = build_periodic_report(period, ["gmv"], fetcher=fixture_fetcher, include_charts=False)
    fixture_report.scope = ReportScope(scope_type="seller", seller_id="seller-1", seller_label="Sao Paulo · SP｜seller-1")
    captured: dict[str, object] = {}

    def build_seller(seller_id, seller_label, report_period, metric_names, **kwargs):
        captured.update({"seller_id": seller_id, "seller_label": seller_label, "period": report_period, "metrics": metric_names})
        return fixture_report

    monkeypatch.setattr("app.api.routes.analysis.get_accessible_seller", lambda *_: {"seller_id": "seller-1", "label": "Sao Paulo · SP｜seller-1"})
    monkeypatch.setattr("app.api.routes.analysis.build_seller_scoped_periodic_report", build_seller)
    settings = get_settings()
    monkeypatch.setattr(settings, "phase6_store_path", str(tmp_path / "seller-report.sqlite"))
    response = TestClient(app).post("/analysis/reports", json={
        "start_date": "2025-01-06", "end_date": "2025-01-12", "granularity": "week",
        "scope_type": "seller", "seller_id": "seller-1", "metric_names": ["gmv"], "include_charts": False,
    })
    assert response.status_code == 200
    assert captured["seller_id"] == "seller-1"
    assert response.json()["report"]["scope"]["scope_type"] == "seller"


def test_periodic_report_request_requires_consistent_scope() -> None:
    from app.schemas.analysis import PeriodicReportRequest

    with pytest.raises(ValueError, match="seller_id is required"):
        PeriodicReportRequest(start_date=date(2025, 1, 1), end_date=date(2025, 1, 31), granularity="month", scope_type="seller")
    with pytest.raises(ValueError, match="only allowed"):
        PeriodicReportRequest(start_date=date(2025, 1, 1), end_date=date(2025, 1, 31), granularity="month", scope_type="platform", seller_id="seller-1")


def test_seller_tenant_report_scope_is_forced_to_its_own_seller(monkeypatch) -> None:
    from app.api.routes.analysis import _effective_report_scope
    from app.schemas.analysis import PeriodicReportRequest
    from app.tenancy import TenantContext

    monkeypatch.setattr("app.api.routes.analysis.get_accessible_seller", lambda *_: {"seller_id": "seller-1", "label": "Sao Paulo · SP｜seller-1"})
    context = TenantContext(tenant_id="tenant-1", user_id="user-1", role="analyst", scope_type="seller", seller_id="seller-1")
    payload = PeriodicReportRequest(start_date=date(2025, 1, 6), end_date=date(2025, 1, 12), granularity="week")
    assert _effective_report_scope(payload, context) == ("seller", "seller-1", "Sao Paulo · SP｜seller-1")
    payload = PeriodicReportRequest(start_date=date(2025, 1, 6), end_date=date(2025, 1, 12), granularity="week", scope_type="seller", seller_id="seller-2")
    with pytest.raises(ValueError, match="another seller"):
        _effective_report_scope(payload, context)


def test_seller_report_compares_date_grants_as_dates(monkeypatch) -> None:
    from types import SimpleNamespace

    from fastapi import HTTPException
    from app.api.routes import analysis as analysis_routes
    from app.schemas.analysis import PeriodicReportRequest
    from app.tenancy import SellerGrant, TenantContext

    period = PeriodSpec(start_date=date(2025, 1, 6), end_date=date(2025, 1, 12), granularity="week")
    fixture_report = build_periodic_report(period, ["gmv"], fetcher=fixture_fetcher, include_charts=False)
    fixture_report.scope = ReportScope(scope_type="seller", seller_id="seller-1", seller_label="Seller 1")
    context = TenantContext(
        tenant_id="tenant-1", user_id="owner-1", role="owner", scope_type="seller",
        seller_id="seller-1",
        seller_grants=(SellerGrant("seller-1", "2025-01-01", "2025-01-31"),),
    )
    monkeypatch.setattr(
        analysis_routes, "_effective_report_scope",
        lambda *_args: ("seller", "seller-1", "Seller 1"),
    )
    monkeypatch.setattr(analysis_routes, "build_seller_scoped_periodic_report", lambda *_args, **_kwargs: fixture_report)
    monkeypatch.setattr(
        analysis_routes, "_store",
        lambda _context: SimpleNamespace(create_draft=lambda *_args, **_kwargs: {"version": 1, "status": "draft"}),
    )

    allowed = PeriodicReportRequest(
        start_date=date(2025, 1, 6), end_date=date(2025, 1, 12), granularity="week",
        scope_type="seller", seller_id="seller-1", metric_names=["gmv"], include_charts=False,
    )
    result = analysis_routes.create_periodic_report(allowed, actor="owner-1", tenant_context=context)
    assert result["status"] == "draft"

    outside_grant = allowed.model_copy(update={"end_date": date(2025, 2, 2)})
    with pytest.raises(HTTPException) as exc_info:
        analysis_routes.create_periodic_report(outside_grant, actor="owner-1", tenant_context=context)
    assert exc_info.value.status_code == 400


def test_tenant_target_store_rejects_reversed_period_before_database(monkeypatch) -> None:
    from types import SimpleNamespace

    import app.tenancy.target_store as target_store_module
    from app.tenancy import TenantContext
    from app.tenancy.target_store import TenantTargetStore

    context = TenantContext(
        tenant_id="tenant-1", user_id="owner-1", role="owner", scope_type="seller", seller_id="seller-1",
    )
    target = SimpleNamespace(
        scope="tenant", seller_id="seller-1", metric_name="gmv",
        period_start=date(2025, 2, 1), period_end=date(2025, 1, 1),
    )
    monkeypatch.setattr(
        target_store_module.engine,
        "begin",
        lambda: (_ for _ in ()).throw(AssertionError("invalid target must be rejected before DB access")),
    )

    with pytest.raises(ValueError, match="period_end must be on or after period_start"):
        TenantTargetStore(context).put(target)


def test_multi_seller_report_scope_accepts_only_granted_seller(monkeypatch) -> None:
    """Adding multi-store grants must not reopen a client-supplied ID bypass."""
    from app.api.routes.analysis import _effective_report_scope
    from app.schemas.analysis import PeriodicReportRequest
    from app.tenancy import SellerGrant, TenantContext

    monkeypatch.setattr(
        "app.api.routes.analysis.get_accessible_seller",
        lambda _context, seller_id: {"seller_id": seller_id, "label": f"Seller {seller_id}"},
    )
    context = TenantContext(
        tenant_id="tenant-1",
        user_id="user-1",
        role="analyst",
        scope_type="seller",
        seller_grants=(SellerGrant("seller-1"), SellerGrant("seller-2")),
    )
    allowed = PeriodicReportRequest(
        start_date=date(2025, 1, 6), end_date=date(2025, 1, 12),
        granularity="week", scope_type="seller", seller_id="seller-2",
    )
    assert _effective_report_scope(allowed, context) == ("seller", "seller-2", "Seller seller-2")
    denied = allowed.model_copy(update={"seller_id": "seller-3"})
    with pytest.raises(ValueError, match="another seller"):
        _effective_report_scope(denied, context)


def test_langgraph_human_review_publishes_same_durable_report(tmp_path, monkeypatch) -> None:
    database = str(tmp_path / "hitl.sqlite")
    settings = get_settings()
    monkeypatch.setattr(settings, "phase6_store_path", database)
    period = PeriodSpec(start_date=date(2025, 1, 6), end_date=date(2025, 1, 12), granularity="week")
    report = build_periodic_report(period, ["gmv"], fetcher=fixture_fetcher, include_charts=False)
    store = ReportStore(database)
    draft = store.create_draft(report, "draft")
    monkeypatch.setattr("app.graph.nodes.human_review.interrupt", lambda _: {
        "action": "approve", "actor": "reviewer", "reason": "verified",
    })
    state = human_review({
        "thread_id": "thread", "trace_id": "trace", "answer": "draft",
        "report_id": report.report_id, "report_version": int(str(draft["version"])),
        "report_risks": [], "node_history": [],
    })
    assert state["critic_verdict"] == "approved"
    assert store.get(report.report_id, int(state["report_version"]))["status"] == "published"

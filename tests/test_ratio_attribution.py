from datetime import date, datetime, timezone

import pytest

from app.analysis.attribution import (
    build_decomposition,
    build_dimension_breakdowns,
    build_grouped_ratio_attributions,
)
from app.analysis.attribution_methods import AttributionDataError, symmetric_ratio_attribution
from app.analysis.contracts import (
    InsightAngle,
    InsightBrief,
    PeriodComparison,
    PeriodSpec,
    PeriodicReport,
    ReportMetric,
    ReportScope,
)
from app.analysis.insights import build_insight_context


def _metric(name: str, current: float, baseline: float, unit: str) -> ReportMetric:
    period = PeriodSpec(start_date=date(2025, 2, 1), end_date=date(2025, 2, 28), granularity="month")
    return ReportMetric(
        metric_name=name,
        value=current,
        unit=unit,
        period_over_period=PeriodComparison(
            comparison_type="period_over_period",
            current_period=period,
            baseline_period=PeriodSpec(
                start_date=date(2025, 1, 1), end_date=date(2025, 1, 31), granularity="month",
            ),
            current_value=current,
            baseline_value=baseline,
            delta_value=round(current - baseline, 2),
            delta_ratio=None,
            status="ok",
        ),
        evidence_ids=[f"ev_{name}"],
    )


def _report(metrics: list[ReportMetric]) -> PeriodicReport:
    period = PeriodSpec(start_date=date(2025, 2, 1), end_date=date(2025, 2, 28), granularity="month")
    return PeriodicReport(
        report_id="ratio-attribution",
        report_type="monthly",
        period=period,
        title="测试报告",
        summary="用于比率分解测试。",
        metrics=metrics,
        evidence_records={
            evidence_id: {"kind": "metric_series", "metric_name": metric.metric_name}
            for metric in metrics
            for evidence_id in metric.evidence_ids
        },
        lineage_refs=["registered_period_factors"],
        data_freshness_at=datetime(2025, 2, 28, tzinfo=timezone.utc),
        scope=ReportScope(scope_type="platform"),
    )


def test_symmetric_ratio_decomposition_reconciles_numerator_and_denominator_effects():
    result = symmetric_ratio_attribution(
        metric_id="avg_order_value",
        baseline_numerator=100,
        current_numerator=150,
        baseline_denominator=10,
        current_denominator=12,
        unit="BRL/order",
    )
    components = {item.component_id: item.value for item in result.raw_components}
    assert result.method_id == "symmetric_ratio_v1"
    assert result.baseline_value == pytest.approx(10)
    assert result.current_value == pytest.approx(12.5)
    assert components["numerator_effect"] == pytest.approx(4.5833333333)
    assert components["denominator_effect"] == pytest.approx(-2.0833333333)
    assert sum(components.values()) == pytest.approx(result.total_delta)
    assert result.residual_tolerance > 0
    assert "对账" in result.residual_note


def test_symmetric_mix_within_applies_ratio_multiplier_and_reconciles():
    from app.analysis.attribution_methods import symmetric_mix_within_attribution

    result = symmetric_mix_within_attribution(
        metric_id="freight_rate",
        baseline_groups={"A": {"numerator": 20, "denominator": 80},
                         "B": {"numerator": 5, "denominator": 20}},
        current_groups={"A": {"numerator": 20, "denominator": 100},
                        "B": {"numerator": 10, "denominator": 50}},
        multiplier=100,
        unit="%",
    )
    assert result.baseline_value == pytest.approx(25.0)
    assert result.current_value == pytest.approx(20.0)
    assert result.total_delta == pytest.approx(-5.0)
    assert sum(item.value for item in result.raw_components) == pytest.approx(-5.0)


@pytest.mark.parametrize("scale", [1e6, 1e9, 1e12, 1e15])
def test_symmetric_mix_within_scales_rounding_tolerance_for_large_valid_values(scale):
    from app.analysis.attribution_methods import symmetric_mix_within_attribution

    result = symmetric_mix_within_attribution(
        metric_id="weighted_rate",
        baseline_groups={
            "A": {"numerator": scale, "denominator": 1},
            "B": {"numerator": 3 * scale, "denominator": 3},
        },
        current_groups={
            "A": {"numerator": scale * 1.000001, "denominator": 1},
            "B": {"numerator": 3 * scale * 0.999999, "denominator": 3},
        },
    )

    assert result.residual_value != 0.0
    assert abs(result.residual_value) <= result.residual_tolerance
    assert result.residual_tolerance > 1e-9
    assert result.residual_type == "rounding"


def test_symmetric_mix_within_never_imputes_new_or_exited_group_rate_as_zero():
    from app.analysis.attribution_methods import symmetric_mix_within_attribution

    result = symmetric_mix_within_attribution(
        metric_id="freight_rate",
        baseline_groups={"A": {"numerator": 10, "denominator": 100},
                         "B": {"numerator": 4, "denominator": 20}},
        current_groups={"A": {"numerator": 8, "denominator": 80},
                        "C": {"numerator": 3, "denominator": 20}},
        multiplier=100,
    )
    components = {item.component_id: item for item in result.raw_components}
    assert "B:exit" in components
    assert "C:entry" in components
    assert components["B:exit"].component_type == "exit_structural_effect"
    assert components["C:entry"].component_type == "entry_structural_effect"
    assert not any(item.component_type == "within_effect" and item.component_id.startswith(("B:", "C:"))
                   for item in result.raw_components)
    assert sum(item.value for item in result.raw_components) == pytest.approx(result.total_delta)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_symmetric_mix_within_rejects_non_finite_group_facts(value):
    from app.analysis.attribution_methods import symmetric_mix_within_attribution

    with pytest.raises(AttributionDataError, match="non-finite"):
        symmetric_mix_within_attribution(
            metric_id="freight_rate",
            baseline_groups={"A": {"numerator": value, "denominator": 10}},
            current_groups={"A": {"numerator": 2, "denominator": 10}},
        )


@pytest.mark.parametrize("bad_value", [True, False])
def test_symmetric_mix_within_rejects_boolean_group_facts(bad_value):
    from app.analysis.attribution_methods import symmetric_mix_within_attribution

    with pytest.raises(AttributionDataError, match="boolean"):
        symmetric_mix_within_attribution(
            metric_id="freight_rate",
            baseline_groups={"A": {"numerator": bad_value, "denominator": 10}},
            current_groups={"A": {"numerator": 2, "denominator": 10}},
        )


@pytest.mark.parametrize("bad_groups", [None, [], "not-a-mapping"])
def test_symmetric_mix_within_rejects_malformed_group_containers(bad_groups):
    from app.analysis.attribution_methods import symmetric_mix_within_attribution

    with pytest.raises(AttributionDataError, match="mapping"):
        symmetric_mix_within_attribution(
            metric_id="freight_rate", baseline_groups=bad_groups,
            current_groups={"A": {"numerator": 1, "denominator": 2}},
        )


@pytest.mark.parametrize("bad_key", ["", 1, None])
def test_symmetric_mix_within_rejects_invalid_group_keys(bad_key):
    from app.analysis.attribution_methods import symmetric_mix_within_attribution

    with pytest.raises(AttributionDataError, match="group keys"):
        symmetric_mix_within_attribution(
            metric_id="freight_rate",
            baseline_groups={bad_key: {"numerator": 1, "denominator": 2}},
            current_groups={"A": {"numerator": 1, "denominator": 2}},
        )


def test_symmetric_mix_within_rejects_non_mapping_group_facts():
    from app.analysis.attribution_methods import symmetric_mix_within_attribution

    with pytest.raises(AttributionDataError, match="mapping"):
        symmetric_mix_within_attribution(
            metric_id="freight_rate", baseline_groups={"A": None},
            current_groups={"A": {"numerator": 1, "denominator": 2}},
        )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf"), True])
def test_symmetric_ratio_rejects_non_finite_and_boolean_components(bad_value):
    with pytest.raises(AttributionDataError):
        symmetric_ratio_attribution(
            metric_id="freight_rate", baseline_numerator=bad_value,
            current_numerator=20, baseline_denominator=10,
            current_denominator=12, multiplier=100,
        )


@pytest.mark.parametrize(
    "baseline_denominator,current_denominator",
    [(0, 10), (10, 0), (-1, 10)],
)
def test_symmetric_ratio_refuses_undefined_denominators(baseline_denominator, current_denominator):
    with pytest.raises(AttributionDataError):
        symmetric_ratio_attribution(
            metric_id="freight_rate",
            baseline_numerator=10,
            current_numerator=20,
            baseline_denominator=baseline_denominator,
            current_denominator=current_denominator,
            multiplier=100,
        )


def test_symmetric_ratio_rejects_change_below_endpoint_float_precision():
    with pytest.raises(AttributionDataError, match="below endpoint float precision"):
        symmetric_ratio_attribution(
            metric_id="large_ratio",
            baseline_numerator=1e18,
            current_numerator=1.001e18,
            baseline_denominator=1000,
            current_denominator=1001,
        )


def test_symmetric_ratio_records_scale_aware_rounding_residual():
    result = symmetric_ratio_attribution(
        metric_id="large_ratio",
        baseline_numerator=1e18,
        current_numerator=1.002e18,
        baseline_denominator=1000,
        current_denominator=1001,
    )
    explained = sum(item.value for item in result.raw_components)

    assert abs(result.residual_value) <= result.residual_tolerance
    assert result.residual_tolerance > 1e-8
    assert explained + result.residual_value == pytest.approx(result.total_delta)


def test_product_attribution_rejects_non_mapping_factor_inputs():
    from app.analysis.attribution_methods import product_shapley_attribution

    with pytest.raises(AttributionDataError, match="mapping"):
        product_shapley_attribution(
            metric_id="gmv", baseline_factors=["orders", "aov"],
            current_factors=["orders", "aov"],
        )


def test_report_snapshot_produces_registered_aov_ratio_decomposition():
    report = _report([
        _metric("avg_order_value", 12.5, 10.0, "BRL/order"),
        _metric("gmv", 150.0, 100.0, "BRL"),
        _metric("order_count", 12.0, 10.0, "orders"),
    ])
    brief = InsightBrief(
        angle=InsightAngle.CONTRIBUTION,
        focus_metrics=["avg_order_value"],
        min_words=1,
        max_words=200,
    )
    decompositions, evidence = build_decomposition(report, brief)
    assert len(decompositions) == 1
    decomposition = decompositions[0]
    assert decomposition.method_id == "symmetric_ratio_v1"
    assert decomposition.factor_metrics == ["gmv", "order_count"]
    assert decomposition.total_delta == pytest.approx(2.5)
    assert decomposition.display_method == "symmetric_ratio"
    assert sum(item.delta_value for item in decomposition.factors) == pytest.approx(2.5)
    audit = evidence[decomposition.evidence_ids[0]]["investigation_evidence"]
    assert audit["method_id"] == "symmetric_ratio_v1"
    assert audit["limitations"] == ["分子/分母对称分解只描述数值变化，不代表因果效应。"]


@pytest.mark.parametrize(
    "metric_name,ratio_current,ratio_baseline,numerator,numerator_current,numerator_baseline,denominator,denominator_current,denominator_baseline",
    [
        ("freight_rate", 20.0, 25.0, "freight_value", 30.0, 25.0, "gmv", 150.0, 100.0),
        ("repeat_purchase_rate", 20.0, 20.0, "repeat_customer_count", 4.0, 2.0, "customer_count", 20.0, 10.0),
    ],
)
def test_registered_ratio_metrics_use_snapshot_components(
    metric_name, ratio_current, ratio_baseline, numerator,
    numerator_current, numerator_baseline, denominator,
    denominator_current, denominator_baseline,
):
    unit = "%"
    report = _report([
        _metric(metric_name, ratio_current, ratio_baseline, unit),
        _metric(numerator, numerator_current, numerator_baseline, "BRL" if numerator == "freight_value" else "customers"),
        _metric(denominator, denominator_current, denominator_baseline, "BRL" if denominator == "gmv" else "customers"),
    ])
    brief = InsightBrief(
        angle=InsightAngle.CONTRIBUTION,
        focus_metrics=[metric_name],
        min_words=1,
        max_words=200,
    )

    decompositions, _ = build_decomposition(report, brief)

    assert len(decompositions) == 1
    assert decompositions[0].metric_name == metric_name
    assert decompositions[0].method_id == "symmetric_ratio_v1"
    assert decompositions[0].total_delta == pytest.approx(ratio_current - ratio_baseline)
    assert [item.factor_metric for item in decompositions[0].factors] == [numerator, denominator]


def test_freight_rate_grouped_attribution_uses_complete_registered_time_groups():

    report = _report([
        _metric("freight_rate", 20.0, 25.0, "%"),
        _metric("freight_value", 30.0, 25.0, "BRL"),
        _metric("gmv", 150.0, 100.0, "BRL"),
    ])
    brief = InsightBrief(
        angle=InsightAngle.CONTRIBUTION,
        focus_metrics=["freight_rate"],
        min_words=1,
        max_words=200,
    )

    def grouped_facts(numerator, denominator, *args):
        assert (numerator, denominator) == ("freight_value", "gmv")
        dimension = args[-1]
        assert dimension == "month_position"
        return {
            "status": "ok", "complete": True, "source_table": "analytics.fact_orders",
            "current_period": {"start_date": "2025-02-01", "end_date": "2025-02-28"},
            "baseline_period": {"start_date": "2025-01-01", "end_date": "2025-01-31"},
            "baseline_groups": {
                "月初(1-10日)": {"numerator": 20.0, "denominator": 80.0},
                "月中(11-20日)": {"numerator": 5.0, "denominator": 20.0},
            },
            "current_groups": {
                "月初(1-10日)": {"numerator": 20.0, "denominator": 100.0},
                "月中(11-20日)": {"numerator": 10.0, "denominator": 50.0},
            },
            "baseline_numerator_total": 25.0,
            "current_numerator_total": 30.0,
            "baseline_denominator_total": 100.0,
            "current_denominator_total": 150.0,
        }

    from report_test_helpers import pin_ratio_attribution_snapshot
    report = pin_ratio_attribution_snapshot(
        report, "freight_rate", "month_position",
        grouped_facts("freight_value", "gmv", "month_position")["baseline_groups"],
        grouped_facts("freight_value", "gmv", "month_position")["current_groups"],
    )
    evidence = build_grouped_ratio_attributions(report, brief)

    rows = [item for item in evidence.values() if item["kind"] == "grouped_ratio_attribution"]
    assert len(rows) == 1
    for row in rows:
        assert row["method_id"] == "symmetric_mix_within_v1"
        assert row["total_delta"] == pytest.approx(-5.0)
        assert sum(item["value"] for item in row["components"]) == pytest.approx(-5.0)
        assert row["investigation_evidence"]["data_quality"]["complete"] is True

    selected, kind_map, context = build_insight_context(
        report, brief, extra_attribution_evidence=evidence,
    )
    assert len(context["grouped_attributions"]) == 1
    assert {item["evidence_id"] for item in rows} <= set(selected)
    assert all(kind_map[item["evidence_id"]] == "grouped_ratio_attribution" for item in rows)


def test_average_order_value_grouped_attribution_uses_registered_ratio_facts(monkeypatch):

    report = _report([
        _metric("avg_order_value", 150 / 14, 10.0, "BRL/order"),
        _metric("gmv", 150.0, 100.0, "BRL"),
        _metric("order_count", 14.0, 10.0, "orders"),
    ])
    brief = InsightBrief(
        angle=InsightAngle.CONTRIBUTION,
        focus_metrics=["avg_order_value"],
        min_words=1,
        max_words=200,
    )

    def grouped_facts(numerator, denominator, *args):
        assert (numerator, denominator, args[-1]) == ("gmv", "order_count", "month_position")
        return {
            "status": "ok", "complete": True, "source_table": "analytics.fact_orders",
            "current_period": {"start_date": "2025-02-01", "end_date": "2025-02-28"},
            "baseline_period": {"start_date": "2025-01-01", "end_date": "2025-01-31"},
            "baseline_groups": {
                "月初(1-10日)": {"numerator": 60.0, "denominator": 6.0},
                "月中(11-20日)": {"numerator": 40.0, "denominator": 4.0},
            },
            "current_groups": {
                "月初(1-10日)": {"numerator": 90.0, "denominator": 9.0},
                "月中(11-20日)": {"numerator": 60.0, "denominator": 5.0},
            },
            "baseline_numerator_total": 100.0,
            "current_numerator_total": 150.0,
            "baseline_denominator_total": 10.0,
            "current_denominator_total": 14.0,
        }

    from report_test_helpers import pin_ratio_attribution_snapshot
    facts = grouped_facts("gmv", "order_count", "month_position")
    report = pin_ratio_attribution_snapshot(
        report, "avg_order_value", "month_position",
        facts["baseline_groups"], facts["current_groups"],
    )
    monkeypatch.setattr(
        "app.data.repositories.daily_metrics.compare_ratio_facts_by_dimension_period",
        lambda *_args: pytest.fail("insight path must only consume report-pinned ratio facts"),
    )

    evidence = build_grouped_ratio_attributions(report, brief)

    rows = [item for item in evidence.values() if item["kind"] == "grouped_ratio_attribution"]
    assert len(rows) == 1
    item = rows[0]
    assert item["metric_name"] == "avg_order_value"
    assert item["dimension"] == "month_position"
    assert item["method_id"] == "symmetric_mix_within_v1"
    assert item["total_delta"] == pytest.approx(150 / 14 - 10)
    assert sum(part["value"] for part in item["components"]) == pytest.approx(item["total_delta"])
    assert item["investigation_evidence"]["metric_id"] == "avg_order_value"


def test_grouped_ratio_report_does_not_use_generic_breakdown_dimensions():
    report = _report([
        _metric("avg_order_value", 150 / 14, 10.0, "BRL/order"),
        _metric("gmv", 150.0, 100.0, "BRL"),
        _metric("order_count", 14.0, 10.0, "orders"),
    ])
    brief = InsightBrief(
        angle=InsightAngle.CONTRIBUTION,
        focus_metrics=["avg_order_value"],
        dimension="product_category",
        min_words=1,
        max_words=200,
    )
    evidence = build_grouped_ratio_attributions(report, brief)

    assert evidence == {}


def test_grouped_ratio_missing_report_snapshot_is_an_explicit_data_gap():
    report = _report([
        _metric("avg_order_value", 150 / 14, 10.0, "BRL/order"),
        _metric("gmv", 150.0, 100.0, "BRL"),
        _metric("order_count", 14.0, 10.0, "orders"),
    ])
    brief = InsightBrief(
        angle=InsightAngle.CONTRIBUTION,
        focus_metrics=["avg_order_value"],
        min_words=1,
        max_words=200,
    )

    evidence = build_grouped_ratio_attributions(report, brief, trace_id="trace-snapshot-missing")
    assert evidence
    assert all(item["kind"] == "capability_gap" for item in evidence.values())
    assert all("report_ratio_snapshot_missing" in item["reason"] for item in evidence.values())
    assert all(
        item["investigation_evidence"]["trace_id"] == "trace-snapshot-missing"
        for item in evidence.values()
    )


def test_grouped_ratio_tampered_snapshot_fails_closed():
    report = _report([
        _metric("avg_order_value", 150 / 14, 10.0, "BRL/order"),
        _metric("gmv", 150.0, 100.0, "BRL"),
        _metric("order_count", 14.0, 10.0, "orders"),
    ])
    from report_test_helpers import pin_ratio_attribution_snapshot
    report = pin_ratio_attribution_snapshot(
        report, "avg_order_value", "month_position",
        {"early": {"numerator": 100.0, "denominator": 10.0}},
        {"early": {"numerator": 150.0, "denominator": 14.0}},
    )
    from app.analysis.report_alignment import ratio_attribution_snapshot_key
    snapshots = dict(report.attribution_breakdowns)
    snapshot_key = ratio_attribution_snapshot_key("avg_order_value", "month_position")
    snapshot = dict(snapshots[snapshot_key])
    snapshot["current_groups"] = {"early": {"numerator": 149.0, "denominator": 14.0}}
    snapshots[snapshot_key] = snapshot  # retain old digest to simulate corrupt persisted facts
    report = report.model_copy(update={"attribution_breakdowns": snapshots})
    brief = InsightBrief(
        angle=InsightAngle.CONTRIBUTION, focus_metrics=["avg_order_value"],
        min_words=1, max_words=200,
    )

    evidence = build_grouped_ratio_attributions(report, brief)
    assert evidence
    assert all(item["kind"] == "capability_gap" for item in evidence.values())
    assert all("report_ratio_snapshot_digest_invalid" in item["reason"] for item in evidence.values())


def test_additive_breakdown_without_report_snapshot_is_not_silently_skipped(monkeypatch):
    from app.analysis import attribution

    report = _report([_metric("gmv", 150.0, 100.0, "BRL")])
    brief = InsightBrief(
        angle=InsightAngle.CONTRIBUTION,
        focus_metrics=["gmv"],
        min_words=1,
        max_words=200,
    )

    monkeypatch.setattr(
        attribution, "compare_breakdown_by_dimension_period",
        lambda *_args, **_kwargs: pytest.fail("insight attribution must not query mutable source"),
    )
    findings, evidence = build_dimension_breakdowns(report, brief)
    assert findings == []
    assert evidence
    assert all(item["kind"] == "data_gap" for item in evidence.values())
    assert all(item["period_statuses"] == {"snapshot": "report_snapshot_missing"} for item in evidence.values())


def test_additive_breakdown_not_found_is_preserved_as_data_gap(monkeypatch):
    from app.analysis import attribution

    report = _report([_metric("gmv", 150.0, 100.0, "BRL")])
    brief = InsightBrief(
        angle=InsightAngle.CONTRIBUTION,
        focus_metrics=["gmv"],
        min_words=1,
        max_words=200,
    )
    monkeypatch.setattr(
        attribution, "compare_breakdown_by_dimension_period",
        lambda *_args, **_kwargs: pytest.fail("insight attribution must not query mutable source"),
    )

    findings, evidence = build_dimension_breakdowns(report, brief, trace_id="trace-data-gap")

    assert findings == []
    assert evidence
    assert all(item["kind"] == "data_gap" for item in evidence.values())
    assert all(item["gap_code"] == "DATA_INSUFFICIENT" for item in evidence.values())
    assert all(item["period_statuses"] == {"snapshot": "report_snapshot_missing"} for item in evidence.values())
    typed = next(iter(evidence.values()))["investigation_evidence"]
    assert typed["evidence_type"] == "attribution_data_gap"
    assert typed["trace_id"] == "trace-data-gap"
    assert typed["scope_fingerprint"]
    assert typed["source_action_id"]
    assert typed["data_quality"]["status"] == "snapshot_unavailable"

    gap_id, gap = next(iter(evidence.items()))
    selected, kinds, context = build_insight_context(
        report, brief, extra_attribution_evidence=evidence,
    )
    assert gap_id in selected
    assert kinds[gap_id] == "attribution_data_gap"
    assert context["attribution_data_gaps"][0]["reason"] == gap["reason"]
    assert context["attribution_capability_gaps"] == []


def test_grouped_ratio_attribution_fails_closed_when_group_totals_do_not_reconcile():
    report = _report([
        _metric("freight_rate", 20.0, 25.0, "%"),
        _metric("freight_value", 30.0, 25.0, "BRL"),
        _metric("gmv", 150.0, 100.0, "BRL"),
    ])
    brief = InsightBrief(
        angle=InsightAngle.CONTRIBUTION,
        focus_metrics=["freight_rate"],
        min_words=1,
        max_words=200,
    )
    from report_test_helpers import pin_ratio_attribution_snapshot
    report = pin_ratio_attribution_snapshot(
        report, "freight_rate", "month_position",
        {"early": {"numerator": 24.0, "denominator": 100.0}},
        {"early": {"numerator": 30.0, "denominator": 150.0}},
    )

    evidence = build_grouped_ratio_attributions(report, brief)

    assert evidence
    assert all(item["kind"] == "capability_gap" for item in evidence.values())
    assert all("report_ratio_snapshot_total_mismatch" in item["reason"] for item in evidence.values())


def test_ratio_fact_repository_returns_untruncated_partitioned_facts(monkeypatch):
    from app.data.repositories import daily_metrics

    class Result:
        def __init__(self, rows):
            self.rows = rows

        def mappings(self):
            return self

        def all(self):
            return self.rows

    class Connection:
        def __init__(self, owner):
            self.owner = owner

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, statement, params):
            self.owner.sql.append(str(statement))
            if params["start_date"].isoformat() == "2025-01-01":
                rows = [
                    {"dimension_value": "A", "numerator": 20, "denominator": 80},
                    {"dimension_value": "B", "numerator": 5, "denominator": 20},
                ]
            else:
                rows = [
                    {"dimension_value": "A", "numerator": 20, "denominator": 100},
                    {"dimension_value": "B", "numerator": 10, "denominator": 50},
                ]
            return Result(rows)

    class FakeEngine:
        sql = []

        def connect(self):
            return Connection(self)

    monkeypatch.setattr(daily_metrics, "engine", FakeEngine())
    result = daily_metrics.compare_ratio_facts_by_dimension_period(
        "freight_value", "gmv",
        date(2025, 2, 1), date(2025, 2, 28),
        date(2025, 1, 1), date(2025, 1, 31),
        "month_position",
    )

    assert result["status"] == "ok"
    assert result["complete"] is True
    assert result["baseline_numerator_total"] == pytest.approx(25)
    assert result["current_denominator_total"] == pytest.approx(150)
    assert set(result["current_groups"]) == {"A", "B"}
    assert all("LIMIT" not in sql.upper() for sql in FakeEngine.sql)
    assert all("EXTRACT(DAY FROM purchase_date)" in sql for sql in FakeEngine.sql)
    with pytest.raises(ValueError):
        daily_metrics.compare_ratio_facts_by_dimension_period(
            "freight_value", "gmv", date(2025, 2, 1), date(2025, 2, 28),
            date(2025, 1, 1), date(2025, 1, 31), "product_category",
        )


def test_ratio_fact_repository_supports_registered_average_order_value_contract(monkeypatch):
    from app.data.repositories import daily_metrics

    class Result:
        def mappings(self):
            return self

        def all(self):
            return [{"dimension_value": "月初(1-10日)", "numerator": 120, "denominator": 10}]

    class Connection:
        def __init__(self, owner):
            self.owner = owner

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, statement, params):
            self.owner.sql.append(str(statement))
            return Result()

    class FakeEngine:
        sql = []

        def connect(self):
            return Connection(self)

    monkeypatch.setattr(daily_metrics, "engine", FakeEngine())
    result = daily_metrics.compare_ratio_facts_by_dimension_period(
        "gmv", "order_count",
        date(2025, 2, 1), date(2025, 2, 28),
        date(2025, 1, 1), date(2025, 1, 31), "month_position",
    )

    assert result["metric_name"] == "avg_order_value"
    assert result["complete"] is True
    assert result["current_numerator_total"] == 120
    assert result["current_denominator_total"] == 10
    assert all("SUM(item_gross_value + freight_value) AS numerator" in sql for sql in FakeEngine.sql)
    assert all("COUNT(*) AS denominator" in sql for sql in FakeEngine.sql)
    assert all("FROM analytics.fact_orders" in sql for sql in FakeEngine.sql)


def test_missing_ratio_factors_are_recorded_as_gap_not_fabricated_effects():
    report = _report([_metric("avg_order_value", 12.5, 10.0, "BRL/order")])
    brief = InsightBrief(
        angle=InsightAngle.CONTRIBUTION,
        focus_metrics=["avg_order_value"],
        min_words=1,
        max_words=200,
    )
    decompositions, evidence = build_decomposition(report, brief)
    assert decompositions == []
    gap_id, gap = next(iter(evidence.items()))
    assert gap["kind"] == "capability_gap"
    assert "缺少已注册的分子或分母" in gap["reason"]

    selected, kinds, context = build_insight_context(
        report, brief, extra_attribution_evidence=evidence,
    )
    assert gap_id in selected
    assert kinds[gap_id] == "attribution_capability_gap"
    assert context["attribution_capability_gaps"] == [{
        "metric_name": "avg_order_value",
        "reason": gap["reason"],
        "evidence_id": gap_id,
    }]


def test_ratio_decomposition_rejects_factor_period_mismatch():
    report = _report([
        _metric("avg_order_value", 12.5, 10.0, "BRL/order"),
        _metric("gmv", 150.0, 100.0, "BRL"),
        _metric("order_count", 12.0, 10.0, "orders").model_copy(update={
            "period_over_period": _metric(
                "order_count", 12.0, 10.0, "orders",
            ).period_over_period.model_copy(update={
                "baseline_period": PeriodSpec(
                    start_date=date(2024, 12, 1), end_date=date(2024, 12, 31), granularity="month",
                ),
            }),
        }),
    ])
    brief = InsightBrief(
        angle=InsightAngle.CONTRIBUTION,
        focus_metrics=["avg_order_value"],
        min_words=1,
        max_words=200,
    )

    decompositions, evidence = build_decomposition(report, brief)

    assert decompositions == []
    gap = next(iter(evidence.values()))
    assert gap["kind"] == "capability_gap"
    assert "当前期/基期不一致" in gap["reason"]


def test_chat_ratio_evidence_is_deterministically_recomputed_before_acceptance():
    from app.agents.diagnostic_contracts import CallToolAction, DiagnosticComparison
    from app.graph.nodes.diagnostic import _typed_ratio_contribution_values
    from app.investigation.contracts import AttributionEvidence, BreakdownDimensionAction
    from app.investigation.engine import _evidence_failure

    query = CallToolAction(
        action_type="CALL_TOOL",
        tool_name="compare_freight_rate_ratio_facts_by_dimension_period",
        metric_name="freight_rate", dimension="month_position",
        comparison=DiagnosticComparison(
            current_start="2018-08-01", current_end="2018-08-31",
            baseline_start="2018-07-01", baseline_end="2018-07-31",
            granularity="month",
        ),
    )
    payload = {
        "status": "ok", "complete": True, "metric_name": "freight_rate",
        "dimension": "month_position",
        "current_period": {"start_date": "2018-08-01", "end_date": "2018-08-31"},
        "baseline_period": {"start_date": "2018-07-01", "end_date": "2018-07-31"},
        "baseline_groups": {
            "early": {"numerator": 2.0, "denominator": 100.0},
            "late": {"numerator": 1.0, "denominator": 100.0},
        },
        "current_groups": {
            "early": {"numerator": 3.0, "denominator": 150.0},
            "late": {"numerator": 1.0, "denominator": 50.0},
        },
        "baseline_numerator_total": 3.0, "baseline_denominator_total": 200.0,
        "current_numerator_total": 4.0, "current_denominator_total": 200.0,
        "source_table": "analytics.fact_orders",
    }
    values = _typed_ratio_contribution_values(
        {"result": {"status": "ok", "contribution_result": payload}}, query,
    )
    assert values["attribution_status"] == "ok"
    assert values["method_id"] == "symmetric_mix_within_v1"
    assert sum(item["delta_value"] for item in values["components"]) == pytest.approx(values["total_delta"])

    action = BreakdownDimensionAction(
        action_type="BREAKDOWN_DIMENSION", metric_id="freight_rate",
        dimension="month_position", method_id="symmetric_mix_within_v1",
    )
    evidence = AttributionEvidence(
        evidence_id="ev-ratio-chat", evidence_type="grouped_ratio_attribution",
        metric_id="freight_rate", source_action_id="ratio-action", source_ref="mcp",
        scope_fingerprint="scope", method_id="symmetric_mix_within_v1",
        method_version="1.0", authorized_metric_provenance="exact", unit="%",
        values=values,
    )
    assert _evidence_failure(action, evidence) is None

    tampered = evidence.model_copy(deep=True)
    tampered.values["components"][0]["delta_value"] += 0.1
    assert _evidence_failure(action, tampered)[0] == "INVALID_DATA"


def test_chat_average_order_value_ratio_evidence_uses_registered_period_aggregation():
    from app.agents.diagnostic_contracts import CallToolAction, DiagnosticComparison
    from app.graph.nodes.diagnostic import _typed_ratio_contribution_values
    from app.investigation.contracts import AttributionEvidence, BreakdownDimensionAction
    from app.investigation.engine import _evidence_failure

    query = CallToolAction(
        action_type="CALL_TOOL",
        tool_name="compare_avg_order_value_ratio_facts_by_dimension_period",
        metric_name="avg_order_value", dimension="month_position",
        comparison=DiagnosticComparison(
            current_start="2018-08-01", current_end="2018-08-31",
            baseline_start="2018-07-01", baseline_end="2018-07-31",
            granularity="month",
        ),
    )
    payload = {
        "status": "ok", "complete": True, "metric_name": "avg_order_value",
        "dimension": "month_position",
        "current_period": {"start_date": "2018-08-01", "end_date": "2018-08-31"},
        "baseline_period": {"start_date": "2018-07-01", "end_date": "2018-07-31"},
        "baseline_groups": {
            "early": {"numerator": 60.0, "denominator": 6.0},
            "late": {"numerator": 40.0, "denominator": 4.0},
        },
        "current_groups": {
            "early": {"numerator": 90.0, "denominator": 9.0},
            "late": {"numerator": 60.0, "denominator": 5.0},
        },
        "baseline_numerator_total": 100.0, "baseline_denominator_total": 10.0,
        "current_numerator_total": 150.0, "current_denominator_total": 14.0,
        "source_table": "analytics.fact_orders",
    }
    values = _typed_ratio_contribution_values(
        {"result": {"status": "ok", "contribution_result": payload}}, query,
    )
    assert values["attribution_status"] == "ok"
    assert values["method_id"] == "symmetric_mix_within_v1"
    assert values["baseline_value"] == pytest.approx(10.0)
    assert values["current_value"] == pytest.approx(150 / 14)

    action = BreakdownDimensionAction(
        action_type="BREAKDOWN_DIMENSION", metric_id="avg_order_value",
        dimension="month_position", method_id="symmetric_mix_within_v1",
    )
    evidence = AttributionEvidence(
        evidence_id="ev-aov-chat", evidence_type="grouped_ratio_attribution",
        metric_id="avg_order_value", source_action_id="aov-action", source_ref="mcp",
        scope_fingerprint="scope", method_id="symmetric_mix_within_v1",
        method_version="1.0", authorized_metric_provenance="exact", unit="BRL/order",
        values=values,
    )
    assert _evidence_failure(action, evidence) is None


def test_ratio_facts_mcp_tool_uses_only_registered_numerator_denominator_and_dimension(monkeypatch):
    from app.mcp.tools import compare_freight_rate_ratio_facts_by_dimension_period

    calls = []
    monkeypatch.setattr(
        "app.mcp.tools._database_read",
        lambda fn, *args: calls.append((fn.__name__, args)) or {"status": "ok", "complete": True},
    )
    result = compare_freight_rate_ratio_facts_by_dimension_period(
        "freight_rate", "2018-08-01", "2018-08-31", "2018-07-01", "2018-07-31",
        "month", "month_position",
    )
    assert result["complete"] is True
    assert calls == [(
        "compare_ratio_facts_by_dimension_period",
        ("freight_value", "gmv", date(2018, 8, 1), date(2018, 8, 31),
         date(2018, 7, 1), date(2018, 7, 31), "month_position"),
    )]


def test_avg_order_value_ratio_facts_tool_uses_registered_components(monkeypatch):
    from app.mcp.tools import compare_avg_order_value_ratio_facts_by_dimension_period

    calls = []
    monkeypatch.setattr(
        "app.mcp.tools._database_read",
        lambda fn, *args: calls.append((fn.__name__, args)) or {"status": "ok", "complete": True},
    )
    result = compare_avg_order_value_ratio_facts_by_dimension_period(
        "avg_order_value", "2018-08-01", "2018-08-31", "2018-07-01", "2018-07-31",
        "month", "month_position",
    )
    assert result["complete"] is True
    assert calls == [(
        "compare_ratio_facts_by_dimension_period",
        ("gmv", "order_count", date(2018, 8, 1), date(2018, 8, 31),
         date(2018, 7, 1), date(2018, 7, 31), "month_position"),
    )]

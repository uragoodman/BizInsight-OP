from datetime import date, datetime, timezone

import pytest

from app.agents.diagnostic_contracts import CallToolAction, DiagnosticComparison
from app.analysis.attribution import build_grouped_ratio_attributions
from app.analysis.contribution import calculate_change_contribution
from app.analysis.contracts import (
    InsightAngle,
    InsightBrief,
    PeriodComparison,
    PeriodSpec,
    PeriodicReport,
    ReportMetric,
    ReportScope,
)
from app.graph.nodes.diagnostic import _typed_contribution_values


@pytest.mark.parametrize(
    ("current_items", "baseline_items", "current_total", "baseline_total", "expected"),
    [
        (
            [
                {"dimension_value": "A", "value": 70.0},
                {"dimension_value": "B", "value": 10.0},
            ],
            [
                {"dimension_value": "A", "value": 50.0},
                {"dimension_value": "C", "value": 10.0},
            ],
            100.0,
            80.0,
            {"A": 20.0, "B": 10.0, "C": -10.0, "Other": 0.0},
        ),
        (
            [{"dimension_value": "A", "value": 45.0}],
            [{"dimension_value": "A", "value": 50.0}],
            60.0,
            80.0,
            {"A": -5.0, "Other": -15.0},
        ),
    ],
)
def test_report_and_chat_additive_attribution_have_identical_group_effects(
    current_items, baseline_items, current_total, baseline_total, expected,
):
    """Both entry points must preserve the same additive attribution contract."""
    total_delta = current_total - baseline_total
    report_finding = calculate_change_contribution(
        metric_name="gmv",
        dimension="product_category",
        current_date=date(2026, 2, 28),
        baseline_date=date(2026, 1, 31),
        current_items=current_items,
        baseline_items=baseline_items,
        overall_total_delta=total_delta,
        current_total=current_total,
        baseline_total=baseline_total,
    )

    query = CallToolAction(
        action_type="CALL_TOOL",
        tool_name="compare_breakdown_by_dimension",
        metric_name="gmv",
        dimension="product_category",
        comparison=DiagnosticComparison(
            current_start="2026-02-01",
            current_end="2026-02-28",
            baseline_start="2026-01-01",
            baseline_end="2026-01-31",
            granularity="month",
        ),
    )
    chat_values = _typed_contribution_values(
        {
            "result": {
                "status": "ok",
                "contribution_result": {
                    "status": "ok",
                    "current_total": current_total,
                    "baseline_total": baseline_total,
                    "total_delta": total_delta,
                    "current_items": current_items,
                    "baseline_items": baseline_items,
                },
            },
        },
        query,
    )

    report_effects = {item.dimension_value: item.delta_value for item in report_finding.items}
    chat_effects = {item["dimension_value"]: item["delta_value"] for item in chat_values["components"]}
    assert report_effects == pytest.approx(expected)
    assert chat_effects == pytest.approx(expected)
    assert chat_effects == pytest.approx(report_effects)
    assert report_finding.total_delta == pytest.approx(total_delta)
    assert chat_values["total_delta"] == pytest.approx(total_delta)
    assert chat_values["attribution_status"] == "ok"


def test_report_and_chat_grouped_ratio_attribution_have_identical_effects(monkeypatch):
    from app.analysis import attribution
    from app.graph.nodes.diagnostic import _typed_ratio_contribution_values

    current_period = PeriodSpec(
        start_date=date(2025, 2, 1), end_date=date(2025, 2, 28), granularity="month",
    )
    baseline_period = PeriodSpec(
        start_date=date(2025, 1, 1), end_date=date(2025, 1, 31), granularity="month",
    )

    def metric(metric_id, current, baseline, unit):
        return ReportMetric(
            metric_name=metric_id,
            value=current,
            unit=unit,
            period_over_period=PeriodComparison(
                comparison_type="period_over_period",
                current_period=current_period,
                baseline_period=baseline_period,
                current_value=current,
                baseline_value=baseline,
                delta_value=round(current - baseline, 2),
                delta_ratio=None,
                status="ok",
            ),
            evidence_ids=[f"ev_{metric_id}"],
        )

    report = PeriodicReport(
        report_id="ratio-path-parity",
        report_type="monthly",
        period=current_period,
        title="ratio parity",
        summary="ratio parity fixture",
        metrics=[
            metric("freight_rate", 20.0, 25.0, "%"),
            metric("freight_value", 30.0, 25.0, "BRL"),
            metric("gmv", 150.0, 100.0, "BRL"),
        ],
        evidence_records={"ev_freight_rate": {"kind": "metric_series"}},
        lineage_refs=["registered_period_factors"],
        data_freshness_at=datetime(2025, 2, 28, tzinfo=timezone.utc),
        scope=ReportScope(scope_type="platform"),
    )
    baseline_groups = {
        "early": {"numerator": 20.0, "denominator": 80.0},
        "late": {"numerator": 5.0, "denominator": 20.0},
    }
    current_groups = {
        "early": {"numerator": 20.0, "denominator": 100.0},
        "late": {"numerator": 10.0, "denominator": 50.0},
    }
    facts = {
        "status": "ok",
        "complete": True,
        "metric_name": "freight_rate",
        "dimension": "month_position",
        "source_table": "analytics.fact_orders",
        "current_period": {"start_date": "2025-02-01", "end_date": "2025-02-28"},
        "baseline_period": {"start_date": "2025-01-01", "end_date": "2025-01-31"},
        "baseline_groups": baseline_groups,
        "current_groups": current_groups,
        "baseline_numerator_total": 25.0,
        "current_numerator_total": 30.0,
        "baseline_denominator_total": 100.0,
        "current_denominator_total": 150.0,
    }
    from report_test_helpers import pin_ratio_attribution_snapshot
    report = pin_ratio_attribution_snapshot(
        report, "freight_rate", "month_position", baseline_groups, current_groups,
    )

    brief = InsightBrief(
        angle=InsightAngle.CONTRIBUTION,
        focus_metrics=["freight_rate"],
        dimension="month_position",
        min_words=1,
        max_words=200,
    )
    report_evidence = build_grouped_ratio_attributions(report, brief)
    report_row = next(
        item for item in report_evidence.values()
        if item["kind"] == "grouped_ratio_attribution"
    )

    query = CallToolAction(
        action_type="CALL_TOOL",
        tool_name="compare_freight_rate_ratio_facts_by_dimension_period",
        metric_name="freight_rate",
        dimension="month_position",
        comparison=DiagnosticComparison(
            current_start="2025-02-01",
            current_end="2025-02-28",
            baseline_start="2025-01-01",
            baseline_end="2025-01-31",
            granularity="month",
        ),
    )
    chat_values = _typed_ratio_contribution_values(
        {"result": {"status": "ok", "contribution_result": facts}}, query,
    )

    report_group_effects = {
        item["dimension_value"]: item["total_effect"]
        for item in report_row["groups"]
    }
    chat_group_effects = {
        item["dimension_value"]: item["delta_value"]
        for item in chat_values["components"]
    }
    assert report_group_effects == pytest.approx(chat_group_effects)
    assert chat_values["total_delta"] == pytest.approx(report_row["total_delta"])
    assert chat_values["baseline_value"] == pytest.approx(report_row["baseline_value"])
    assert chat_values["current_value"] == pytest.approx(report_row["current_value"])
    assert chat_values["method_id"] == report_row["method_id"] == "symmetric_mix_within_v1"
    assert chat_values["residual_value"] == pytest.approx(
        report_row["investigation_evidence"]["values"]["residual_value"], abs=1e-12,
    )
    report_raw = {
        item["component_id"]: item["value"]
        for item in report_row["components"]
    }
    chat_raw = {
        item["component_id"]: item["value"]
        for item in chat_values["raw_components"]
    }
    assert chat_raw == pytest.approx(report_raw)

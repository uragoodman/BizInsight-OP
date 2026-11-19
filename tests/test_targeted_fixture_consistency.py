from __future__ import annotations

from datetime import date

import pytest

from app.analysis.contracts import PeriodSpec
from evaluation.fixtures import (
    fixture_comparison,
    fixture_mcp,
    fixture_period_breakdown,
    fixture_period_value,
)


def test_incomplete_case_single_value_and_trend_share_the_same_missing_tail():
    day = "2018-04-25"
    value = fixture_mcp(
        "anomaly_analysis", f"查询 {day} 的 gmv",
    )["result"]
    trend = fixture_mcp(
        "anomaly_analysis", f"查询 gmv trend 趋势 2018-04-19 到 {day}",
    )["result"]["trend_result"]

    assert value["status"] == "not_found"
    assert value["metric_result"]["value"] is None
    assert trend["points"][-2:] == [
        {"bucket": "2018-04-24", "value": None},
        {"bucket": day, "value": None},
    ]
    assert all(point["value"] is not None for point in trend["points"][:-2])


def test_order_drop_100_trend_comparison_and_refuted_dimensions_reconcile():
    trend = fixture_mcp(
        "anomaly_analysis",
        "查询 gmv 趋势 2018-04-17 到 2018-04-18",
        plan={"query_type": "metric_trend", "start_date": "2018-04-17", "end_date": "2018-04-18"},
    )["result"]["trend_result"]
    comparison = fixture_mcp(
        "anomaly_analysis",
        "compare gmv 2018-04-18 with 2018-04-17",
        plan={"query_type": "metric_compare"},
    )["result"]["compare_result"]
    contribution = fixture_mcp(
        "anomaly_analysis",
        "compare 2018-04-18 with 2018-04-17 gmv by dimension customer_state",
        plan={"query_type": "metric_contribution"},
    )["result"]["contribution_result"]

    baseline_value = 1000.0
    current_value = 600.0
    expected_delta = current_value - baseline_value

    assert [(point["bucket"], point["value"]) for point in trend["points"]] == [
        ("2018-04-17", baseline_value),
        ("2018-04-18", current_value),
    ]
    assert comparison["left_value"] == current_value
    assert comparison["right_value"] == baseline_value
    assert comparison["delta_value"] == expected_delta

    baseline_groups = contribution["baseline_items"]
    current_groups = contribution["current_items"]
    assert sum(item["value"] for item in baseline_groups) == baseline_value
    assert sum(item["value"] for item in current_groups) == current_value
    group_deltas = {
        item["dimension_value"]: item["value"]
        - next(base["value"] for base in baseline_groups if base["dimension_value"] == item["dimension_value"])
        for item in current_groups
    }
    assert sum(group_deltas.values()) == contribution["total_delta"] == expected_delta
    assert max(abs(delta / expected_delta) for delta in group_deltas.values()) <= 0.01


@pytest.mark.parametrize(
    ("current_day", "baseline_value", "current_value", "concentrated", "missing"),
    [
        ("2018-04-18", 1000.0, 600.0, False, False),  # core drop, evenly spread
        ("2018-04-20", 1000.0, 1400.0, False, False),  # core rise
        ("2018-04-26", 1000.0, 500.0, True, False),  # concentrated core case, overlaps Olist calendar facts
        ("2018-04-25", 1000.0, None, False, True),  # core missing tail
    ],
)
def test_case_facts_reconcile_across_trend_compare_period_and_dimension(
    current_day: str, baseline_value: float, current_value: float | None,
    concentrated: bool, missing: bool,
):
    current = date.fromisoformat(current_day)
    baseline = current.fromordinal(current.toordinal() - 1)
    current_text, baseline_text = current.isoformat(), baseline.isoformat()

    trend = fixture_mcp(
        "anomaly_analysis", f"GMV trend {baseline_text} to {current_text}",
        plan={"query_type": "metric_trend", "start_date": baseline_text, "end_date": current_text},
    )["result"]["trend_result"]
    comparison = fixture_mcp(
        "metric_query", f"compare gmv {current_text} with {baseline_text}",
        plan={"query_type": "metric_compare", "left_date": current_text, "right_date": baseline_text},
    )["result"]["compare_result"]
    period_comparison = fixture_comparison(
        "gmv", PeriodSpec(start_date=current, end_date=current),
        PeriodSpec(start_date=baseline, end_date=baseline), "period_over_period",
    )
    dimension = fixture_period_breakdown(
        "gmv", current, current, baseline, baseline, "customer_state", 10,
    )
    contribution = fixture_mcp(
        "anomaly_analysis", f"GMV contribution {current_text} with {baseline_text}",
        plan={"query_type": "metric_contribution", "current_date": current_text, "baseline_date": baseline_text},
    )["result"]["contribution_result"]

    if missing:
        assert [point["value"] for point in trend["points"]] == [None, None]
        assert comparison["status"] == "not_found"
        assert comparison["delta_value"] is None
        assert period_comparison.status == "not_found"
        assert dimension["status"] == "incomplete"
        assert contribution["status"] == "incomplete"
        return

    delta = current_value - baseline_value
    assert [point["value"] for point in trend["points"]] == [baseline_value, current_value]
    assert (comparison["left_value"], comparison["right_value"], comparison["delta_value"]) == (
        current_value, baseline_value, delta,
    )
    assert (period_comparison.current_value, period_comparison.baseline_value, period_comparison.delta_value) == (
        current_value, baseline_value, delta,
    )
    for result in (dimension, contribution):
        assert sum(item["value"] for item in result["current_items"]) == current_value
        assert sum(item["value"] for item in result["baseline_items"]) == baseline_value
        group_delta = {
            item["dimension_value"]: item["value"] - next(
                base["value"] for base in result["baseline_items"]
                if base["dimension_value"] == item["dimension_value"]
            ) for item in result["current_items"]
        }
        assert sum(group_delta.values()) == result["total_delta"] == delta
        if concentrated:
            assert max(abs(value / delta) for value in group_delta.values()) >= 0.51
        else:
            assert max(abs(value / delta) for value in group_delta.values()) <= 0.01


def test_weighted_ratio_periods_use_registered_component_aggregation():
    start, end = date(2018, 4, 18), date(2018, 4, 20)
    period = PeriodSpec(start_date=start, end_date=end)
    from app.metrics.registry import get_registry

    for metric_name in ("avg_order_value", "freight_rate"):
        aggregation = get_registry().get(metric_name).period_aggregation
        assert aggregation is not None
        expected = (
            fixture_period_value(aggregation.numerator_metric, period)
            / fixture_period_value(aggregation.denominator_metric, period)
            * aggregation.multiplier
        )
        observed = fixture_period_value(metric_name, period)
        breakdown = fixture_period_breakdown(
            metric_name, start, end, start, end, "product_category", 10,
        )

        assert observed == pytest.approx(expected)
        assert breakdown["aggregation"] == "weighted_ratio"
        assert breakdown["current_total"] == observed
        assert breakdown["current_items"] == []  # ratio values are not additive groups


def test_olist_case_keeps_snapshot_facts_across_tools():
    from json import loads
    from pathlib import Path

    root = Path(__file__).parents[1]
    dataset = loads((root / "evaluation/datasets/provider_holdout_olist_v1.json").read_text(encoding="utf-8"))
    facts = loads((root / "evaluation/fixtures/provider_holdout_olist_v1.json").read_text(encoding="utf-8"))["facts_by_day"]
    ground_truth = dataset["cases"][0]["source_ground_truth"]
    current_text, baseline_text = ground_truth["current_date"], ground_truth["baseline_date"]
    current, baseline = date.fromisoformat(current_text), date.fromisoformat(baseline_text)

    trend = fixture_mcp(
        "anomaly_analysis", f"GMV trend {baseline_text} to {current_text}",
        plan={"query_type": "metric_trend", "start_date": baseline_text, "end_date": current_text},
    )["result"]["trend_result"]
    comparison = fixture_mcp(
        "metric_query", f"compare gmv {current_text} with {baseline_text}",
        plan={"query_type": "metric_compare", "left_date": current_text, "right_date": baseline_text},
    )["result"]["compare_result"]
    period = fixture_comparison(
        "gmv", PeriodSpec(start_date=current, end_date=current),
        PeriodSpec(start_date=baseline, end_date=baseline), "period_over_period",
    )
    breakdown = fixture_period_breakdown(
        "gmv", current, current, baseline, baseline, "customer_state", 20,
    )
    contribution = fixture_mcp(
        "anomaly_analysis", f"GMV contribution {current_text} with {baseline_text}",
        plan={"query_type": "metric_contribution", "current_date": current_text, "baseline_date": baseline_text},
    )["result"]["contribution_result"]

    expected_current, expected_baseline = facts[current_text]["gmv"], facts[baseline_text]["gmv"]
    assert trend["source_table"] == "olist-public.orders+order_items"
    assert [point["value"] for point in trend["points"]] == [expected_baseline, expected_current]
    assert (comparison["left_value"], comparison["right_value"]) == (expected_current, expected_baseline)
    assert (period.current_value, period.baseline_value) == (expected_current, expected_baseline)
    assert breakdown["source_table"] == "olist-public.order_items+orders+customers"
    assert breakdown["current_total"] == pytest.approx(expected_current, abs=0.02)
    assert breakdown["baseline_total"] == pytest.approx(expected_baseline, abs=0.02)
    assert contribution["source_table"] == "olist-public.order_items+orders+customers"
    assert sum(item["value"] for item in contribution["current_items"]) == pytest.approx(expected_current, abs=0.02)
    assert sum(item["value"] for item in contribution["baseline_items"]) == pytest.approx(expected_baseline, abs=0.02)

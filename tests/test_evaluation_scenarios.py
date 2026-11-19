"""Phase 0 scenario fixtures lock the deterministic "rule gives up" signal.

These tests prove the three golden scenarios are constructed so that the
current Workflow ALWAYS stops with a give-up signal on them — that is the
baseline grating-against which the Phase 4/5 Agent is measured.  They assert
fixture shape, the contribution verdict the rule computes, and the queue being
structurally unable to discover a multiplicative/cross-metric cause.
"""

import json
from datetime import date
from pathlib import Path

from app.analysis.contracts import MetricPoint, MetricSeries, PeriodSpec
from app.analysis.statistical_anomaly import detect_series_anomalies
from app.graph.nodes.root_cause import _build_follow_up_queue, _evaluate_step
from evaluation.fixtures import (
    SCENARIO_DATES, fixture_comparison, fixture_mcp, fixture_period_breakdown,
)
from evaluation.scorers import collect_anomaly_metrics, score_scenario, score_diagnostic_run


def _scenario_entry(question: str, purpose: str, metric_id: str = "gmv") -> dict:
    raw = fixture_mcp("metric_query", question)
    assert "contribution_result" in raw["result"], f"expected a contribution result, got {raw['result']}"
    return {
        "metric_id": metric_id, "query_id": "rcq_scenario_1",
        "purpose": purpose, "hypothesis": "scenario hypothesis",
        "result": raw,
    }


# ---------------------------------------------------------------------------
# scenario contribution is REFUTED (top ratio < 1%) → rule accumulates refuted
# ---------------------------------------------------------------------------

def test_scenario_contribution_is_refuted() -> None:
    # 只有冻结 gate 的三个场景（规则必然放弃组）保证全维度被驳回；
    # 长尾场景 2018-08-26 按设计是“单一品类集中致因”（supported）。
    # Keep the frozen baseline gate explicit.  SCENARIO_DATES now also contains
    # generated core/long-tail dates and is not ordered as the original gate.
    gate_dates = ("2018-08-20", "2018-08-21", "2018-08-22")
    for scenario_date in gate_dates:
        question = f"拆解 {scenario_date} 与 2018-08-19 的 gmv 按客户州贡献度"
        step = _evaluate_step(_scenario_entry(question, "dimension_contribution:customer_state"), 1, [])
        assert step["verdict"] == "refuted", (
            f"{scenario_date}: expected refuted contribution, got {step['verdict']}"
        )


# ---------------------------------------------------------------------------
# scenario trend is a real drop that anomaly detection flags
# ---------------------------------------------------------------------------

def test_scenario_trend_flags_drop() -> None:
    raw = fixture_mcp("metric_query", "查询最近 7 天 gmv 趋势")
    trend = raw["result"]["trend_result"]
    points = trend["points"]
    assert points[0]["value"] > points[-1]["value"], "GMV trend must end in a drop"
    series = MetricSeries(
        metric_name="gmv", unit="BRL",
        period=PeriodSpec(start_date=points[0]["bucket"], end_date=points[-1]["bucket"], granularity="day"),
        source_refs=["golden.fixture"], definition_version="1.0",
        points=[MetricPoint(bucket=p["bucket"], value=p["value"]) for p in points],
    )
    assert detect_series_anomalies(series, min_samples=6), "scenario trend drop must be detected as an anomaly"


# ---------------------------------------------------------------------------
# the rule's queue is structurally unable to find a multiplicative/cross-metric
# cause: it only issues single-attribute dimension contributions, never a
# factor-decomposition or another metric
# ---------------------------------------------------------------------------

def test_rule_queue_cannot_discover_factor_or_cross_metric() -> None:
    queue = _build_follow_up_queue("为什么 2018-08-20 的 GMV 下降了？")
    dimensions = [item for item in queue if item["purpose"].startswith("dimension_contribution:")]
    purposes = {item["purpose"] for item in dimensions}
    # GMV priority dimensions must be covered — the give-up stop requires them.
    assert "dimension_contribution:product_category" in purposes
    assert "dimension_contribution:customer_state" in purposes
    # Every queued question is an additive single-attribute contribution only.
    assert all(
        ("客单价" not in item["question"] and "avg_order_value" not in item["question"])
        for item in queue
    ), "the rule must never call a multiplicative-factor query"


def test_multiplicative_factor_scenario_has_flat_orders_and_halved_aov() -> None:
    aov = fixture_mcp("metric_query", "查询 2018-08-20 的 avg_order_value")["result"]["metric_result"]["value"]
    orders = fixture_mcp("metric_query", "查询 2018-08-20 的 order_count")["result"]["metric_result"]["value"]
    assert aov == 5.0
    assert orders == 100.0


def test_multi_metric_scenario_has_orders_rising_while_gmv_falls() -> None:
    gmv = fixture_mcp("metric_query", "查询 2018-08-22 的 gmv")["result"]["metric_result"]["value"]
    orders = fixture_mcp("metric_query", "查询 2018-08-22 的 order_count")["result"]["metric_result"]["value"]
    assert gmv < 1000.0
    assert orders > 100.0


def test_adjacent_synthetic_cases_keep_independent_baseline_across_evidence_paths() -> None:
    dataset_path = Path(__file__).parents[1] / "evaluation" / "datasets" / "golden_v5_agent_368.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    case = next(item for item in dataset["cases"] if item["id"] == "v5-dimension_driven-234")
    current = date.fromisoformat("2017-08-23")
    baseline = date.fromisoformat("2017-08-22")

    trend_raw = fixture_mcp(
        "anomaly_analysis", f"gmv trend {baseline.isoformat()} to {current.isoformat()}; {case['query']}",
        plan={"query_type": "metric_trend"},
    )
    points = trend_raw["result"]["trend_result"]["points"]
    assert [point["bucket"] for point in points] == [baseline.isoformat(), current.isoformat()]
    assert points[0]["value"] > points[-1]["value"]
    assert points[0]["value"] == 1000.0
    assert points[-1]["value"] == 500.0
    assert trend_raw["result"]["trend_result"]["source_table"] == "golden.fixture"

    compare_raw = fixture_mcp(
        "metric_query", f"compare gmv {current.isoformat()} {baseline.isoformat()}",
        plan={"query_type": "metric_compare"},
    )
    comparison = compare_raw["result"]["compare_result"]
    assert (comparison["left_value"], comparison["right_value"], comparison["delta_value"]) == (500.0, 1000.0, -500.0)

    period_current = PeriodSpec(start_date=current, end_date=current)
    period_baseline = PeriodSpec(start_date=baseline, end_date=baseline)
    period_comparison = fixture_comparison("gmv", period_current, period_baseline, "period_over_period")
    assert (period_comparison.current_value, period_comparison.baseline_value, period_comparison.delta_value) == (500.0, 1000.0, -500.0)

    dimension = fixture_period_breakdown(
        "gmv", current, current, baseline, baseline, "customer_state", 10,
    )
    assert (dimension["current_total"], dimension["baseline_total"], dimension["total_delta"]) == (500.0, 1000.0, -500.0)
    assert sum(item["value"] for item in dimension["current_items"]) == dimension["current_total"]
    assert sum(item["value"] for item in dimension["baseline_items"]) == dimension["baseline_total"]

    contribution_raw = fixture_mcp(
        "anomaly_analysis", f"{case['query']} {baseline.isoformat()}",
        plan={"query_type": "metric_contribution"},
    )
    contribution = contribution_raw["result"]["contribution_result"]
    assert contribution["source_table"] == "golden.fixture"
    assert contribution["total_delta"] == -500.0
    assert sum(item["value"] for item in contribution["current_items"]) == 500.0
    assert sum(item["value"] for item in contribution["baseline_items"]) == 1000.0

    holdout = json.loads((Path(__file__).parents[1] / "evaluation" / "datasets" / "provider_holdout_olist_v1.json").read_text(encoding="utf-8"))
    holdout_case = holdout["cases"][0]["source_ground_truth"]
    holdout_current, holdout_baseline = holdout_case["current_date"], holdout_case["baseline_date"]
    facts = json.loads((Path(__file__).parents[1] / "evaluation" / "fixtures" / "provider_holdout_olist_v1.json").read_text(encoding="utf-8"))["facts_by_day"]
    olist_trend = fixture_mcp(
        "anomaly_analysis", f"gmv trend {holdout_baseline} to {holdout_current}",
        plan={"query_type": "metric_trend"},
    )["result"]["trend_result"]
    assert olist_trend["source_table"] == "olist-public.orders+order_items"
    assert [point["value"] for point in olist_trend["points"]] == [facts[holdout_baseline]["gmv"], facts[holdout_current]["gmv"]]


# ---------------------------------------------------------------------------
# score_scenario marks a baseline give-up as failing the expectation
# ---------------------------------------------------------------------------

def _baseline_state() -> dict:
    return {
        "analysis_result": {
            "summary": "## 异常分析：gmv\n- 趋势基线：相较 2025-03-30 下降 400.00（-40.00%）。\n"
                       "- 证据来源：golden.fixture。\n- 原因假设：维度变化仅表示相关贡献，仍需结合业务事件验证。",
            "stop_reason": "no_material_contributor_in_examined_dimensions",
            "contribution": [],
            "anomalies": [{"metric_name": "gmv"}],
        },
        "claims": [{"value": "未发现显著贡献维度项"}],
        "root_cause_evidence": [
            {"purpose": "trend_baseline"},
            {"purpose": "dimension_contribution:customer_state"},
            {"purpose": "dimension_contribution:customer_city"},
            {"purpose": "dimension_contribution:seller_state"},
            {"purpose": "dimension_contribution:product_category"},
        ],
        "evidence_bundles": [{"evidence_id": "ev_1"}],
        "tool_calls": [{"arguments": {}}],
        "critic_verdict": "approved",
        "task_type": "anomaly_analysis",
    }


def test_score_scenario_baseline_fails_hidden_dimension_expectation() -> None:
    state = _baseline_state()
    verdict = score_scenario(state, {"found_factor": "avg_order_value", "abstain": False})
    assert verdict["evidence_examined"] is True, "the rule must have looked before giving up"
    assert verdict["found_avg_order_value"] is False, "baseline must not name the AOV factor"
    assert all(verdict.values()) is False


def test_score_scenario_baseline_fails_abstain_expectation() -> None:
    verdict = score_scenario(_baseline_state(), {"abstain": True})
    assert verdict["evidence_examined"] is True
    assert verdict["abstain"] is False, "the Workflow cannot refuse attribution, it forces a conclusion"
    assert all(verdict.values()) is False


def _traceable_missing_trend_state(*, call_status: str = "ok", abstain: bool = True) -> dict:
    evidence_id = "ev_missing_gmv_trend"
    return {
        "diagnostic_target_metric": "gmv",
        "analysis_result": {
            "abstain": abstain,
            "stop_reason": "insufficient_evidence",
            "summary": "GMV trend data is incomplete; no causal conclusion is supported.",
        },
        "root_cause_evidence": [],
        "tool_calls": [{
            "tool_name": "query_metric_trend",
            "arguments": {
                "metric_name": "gmv",
                "period": {"start_date": "2018-04-24", "end_date": "2018-04-25", "granularity": "day"},
            },
            "status": call_status,
            "evidence_id": evidence_id,
        }],
        "evidence_bundles": [{
            "evidence_id": evidence_id,
            "source_type": "mcp",
            "source_ref": "query_metric_trend",
            "trusted": False,
            "validation_status": "pending_typed_evidence_validation",
            "content": {"result": {
                "status": "ok",
                "trend_result": {
                    "metric_name": "gmv",
                    "start_date": "2018-04-24",
                    "end_date": "2018-04-25",
                    "points": [
                        {"bucket": "2018-04-24", "value": None},
                        {"bucket": "2018-04-25", "value": None},
                    ],
                },
            }},
        }],
        "investigation_state": {"target": {"metric_id": "gmv"}},
    }


def test_score_scenario_accepts_auditable_missing_trend_when_agent_abstains() -> None:
    state = _traceable_missing_trend_state()
    metrics = collect_anomaly_metrics(state)
    verdict = score_scenario(state, {"abstain": True})

    assert metrics["has_trend"] is False, "missing values must not be promoted to a usable trend baseline"
    assert metrics["has_traceable_missing_trend"] is True
    assert verdict["evidence_examined"] is True
    assert verdict["abstain"] is True


def test_score_scenario_recognizes_a_gap_in_a_partially_populated_trend() -> None:
    state = _traceable_missing_trend_state()
    points = state["evidence_bundles"][0]["content"]["result"]["trend_result"]["points"]
    points[0]["value"] = 1000.0

    metrics = collect_anomaly_metrics(state)
    verdict = score_scenario(state, {"abstain": True})

    assert metrics["has_trend"] is False
    assert metrics["has_traceable_missing_trend"] is True
    assert verdict["evidence_examined"] is True


def test_score_scenario_does_not_accept_abstention_without_executed_missing_trend() -> None:
    state = _traceable_missing_trend_state(call_status="failed")
    verdict = score_scenario(state, {"abstain": True})

    assert verdict["evidence_examined"] is False
    assert verdict["abstain"] is True


def test_score_scenario_does_not_accept_missing_trend_without_abstention() -> None:
    state = _traceable_missing_trend_state(abstain=False)
    verdict = score_scenario(state, {"abstain": True})

    assert verdict["evidence_examined"] is False
    assert verdict["abstain"] is False


def _typed_scoring_state(*, evidence_type: str = "metric_decomposition", evidence_scope: str = "scope-1", validation_status: str = "valid") -> dict:
    evidence_id = "typed-evidence-1"
    metric = "gmv"
    target = {
        "metric_id": metric, "scope_fingerprint": "scope-1",
        "current_start": "2018-04-25", "current_end": "2018-04-26",
        "baseline_start": "2018-04-23", "baseline_end": "2018-04-24",
    }
    values = {"factor_metrics": ["order_count", "avg_order_value"]}
    if evidence_type == "dimension_contribution":
        values = {
            "attribution_status": "ok", "query": {"dimension": "customer_state"},
            "result": {"total_delta": -100, "current_period": {"start_date": "2018-04-25", "end_date": "2018-04-26"},
                       "baseline_period": {"start_date": "2018-04-23", "end_date": "2018-04-24"}},
            "components": [{"dimension_value": "SP", "delta_value": -80}],
        }
    evidence = {
        "evidence_id": evidence_id, "evidence_type": evidence_type, "metric_id": metric,
        "scope_fingerprint": evidence_scope, "source_action_id": "candidate-1", "method_id": "method-1",
        "values": values, "comparison": {
            "current_period": {"start_date": "2018-04-25", "end_date": "2018-04-26"},
            "baseline_period": {"start_date": "2018-04-23", "end_date": "2018-04-24"},
        },
    }
    investigation = {
        "target": target, "stop_reason": {"code": "COMPLETED"}, "evidence": [evidence],
        "action_results": [{
            "candidate_id": "candidate-1", "metric_id": metric, "method_id": "method-1",
            "status": "completed", "validation_result": {"status": validation_status}, "evidence_ids": [evidence_id],
        }],
    }
    return {
        "trace_id": "trace-1", "diagnostic_target_metric": metric, "investigation_state": investigation,
        "finish_decision": {"action_type": "FINISH", "found_factor": "order_count", "evidence_ids": [evidence_id]},
        "analysis_result": {"abstain": False, "evidence_ids": [evidence_id], "summary": "order_count explains the numeric decomposition."},
    }


def test_score_factor_keyword_alone_does_not_count_as_finding() -> None:
    state = {"analysis_result": {"abstain": False, "summary": "order_count explains GMV"}}
    verdict = score_scenario(state, {"found_factor": "order_count"})
    assert verdict["found_order_count"] is False


def test_score_factor_requires_completed_valid_evidence_binding() -> None:
    state = _typed_scoring_state(validation_status="invalid_data")
    verdict = score_scenario(state, {"found_factor": "order_count"})
    assert verdict["found_order_count"] is False


def test_score_factor_rejects_scope_mismatch() -> None:
    state = _typed_scoring_state(evidence_scope="another-scope")
    verdict = score_scenario(state, {"found_factor": "order_count"})
    assert verdict["found_order_count"] is False


def test_score_factor_rejects_comparison_period_mismatch() -> None:
    state = _typed_scoring_state()
    state["investigation_state"]["evidence"][0]["comparison"]["current_period"]["end_date"] = "2018-04-27"
    verdict = score_scenario(state, {"found_factor": "order_count"})
    assert verdict["found_order_count"] is False


def test_score_valid_numeric_decomposition_is_structured_and_period_bound() -> None:
    state = _typed_scoring_state()
    verdict = score_scenario(state, {"found_factor": "order_count"})
    assert verdict["found_order_count"] is True


def test_cross_metric_link_is_not_numeric_decomposition() -> None:
    state = _typed_scoring_state()
    state["finish_decision"]["cross_metric_link"] = "delivery_delay_rate"
    state["analysis_result"]["cross_metric_links"] = ["delivery_delay_rate"]
    # A numeric decomposition mentioning another metric does not prove a link.
    verdict = score_scenario(state, {"cross_metric_link": "delivery_delay_rate", "abstain": False})
    assert verdict["found_cross_metric_link"] is False


def test_cross_metric_link_requires_registered_scoped_association_evidence() -> None:
    state = _typed_scoring_state(evidence_type="registered_association")
    state["investigation_state"]["evidence"][0]["values"]["related_metric_id"] = "delivery_delay_rate"
    state["finish_decision"]["cross_metric_link"] = "delivery_delay_rate"
    state["analysis_result"]["cross_metric_links"] = ["delivery_delay_rate"]
    verdict = score_scenario(state, {"cross_metric_link": "delivery_delay_rate", "abstain": False})
    assert verdict["found_cross_metric_link"] is True

    state["investigation_state"]["stop_reason"] = {"code": "ENOUGH_EVIDENCE"}
    state["finish_decision"]["action_type"] = "FINISH"
    state["analysis_presentation"] = {"status": "completed"}
    state["root_cause_evidence"] = [{"purpose": "trend_baseline"}]
    run = score_diagnostic_run(
        state, case_id="scored-cross-metric-link", arm="offline", replicate=1,
        expected_outcome="cross_metric_link",
        scenario_expectation={"cross_metric_link": "delivery_delay_rate", "abstain": False},
    )
    assert run["task_acceptance_pass"] is True
    assert run["effective_diagnosis_completed"] is False
    assert run["correct_safe_stop"] is False


def test_score_statistical_contribution_without_claiming_business_cause() -> None:
    state = _typed_scoring_state(evidence_type="dimension_contribution")
    eid = "typed-evidence-1"
    state["finish_decision"] = {"action_type": "ABSTAIN", "supporting_evidence_ids": [eid]}
    state["analysis_result"] = {
        "abstain": True, "evidence_ids": [eid], "stop_reason": "insufficient_evidence",
        "summary": "SP had the largest statistical contribution (80%); the business cause is unresolved.",
        "causal_readiness": {"causal_effect_estimated": False},
        "contribution": [{
            "dimension": "customer_state", "analysis_nature": "descriptive_contribution",
            "current_date": "2018-04-26", "baseline_date": "2018-04-24", "evidence_ids": [eid],
            "items": [{"dimension_value": "SP", "delta_value": -80, "contribution_ratio": 0.8}],
        }],
    }
    verdict = score_scenario(state, {
        "found_dimension": True,
        "reported_dimension_contribution": {"dimension": "customer_state", "dimension_value": "SP", "contribution_share": 0.8},
        "business_cause_unresolved": True,
    })
    assert verdict["found_dimension"] is True
    assert verdict["reported_dimension_contribution"] is True
    assert verdict["business_cause_unresolved"] is True


def test_per_run_scorer_counts_valid_statistical_contribution_separately() -> None:
    state = _typed_scoring_state(evidence_type="dimension_contribution")
    state["investigation_state"]["stop_reason"] = {"code": "ENOUGH_EVIDENCE"}
    eid = "typed-evidence-1"
    state["finish_decision"] = {"action_type": "ABSTAIN", "supporting_evidence_ids": [eid]}
    state["analysis_result"] = {
        "abstain": True, "evidence_ids": [eid], "contribution": [{
            "dimension": "customer_state", "analysis_nature": "descriptive_contribution",
            "current_date": "2018-04-26", "baseline_date": "2018-04-24", "evidence_ids": [eid],
            "items": [{"dimension_value": "SP", "delta_value": -80, "contribution_ratio": 0.8}],
        }],
    }
    state["analysis_presentation"] = {"status": "completed"}
    row = score_diagnostic_run(
        state, case_id="statistical-contribution", arm="offline", replicate=1,
        expected_outcome="statistical_contribution",
    )
    assert row["task_acceptance_pass"] is True
    assert row["effective_diagnosis_completed"] is True
    assert row["effective_diagnosis_type"] == "statistical_contribution"
    assert row["correct_safe_stop"] is False
    assert row["conclusion_evidence_valid"] is True


def test_unknown_expectation_field_fails_closed() -> None:
    verdict = score_scenario(_baseline_state(), {"found_dimension": True, "found_dimenson": True})
    assert verdict["expectation_schema_valid"] is False
    assert verdict["unknown_expectation_found_dimenson"] is False


def test_collect_anomaly_metrics_reports_stop_reason() -> None:
    metrics = collect_anomaly_metrics(_baseline_state())
    assert metrics["stop_reason"] == "no_material_contributor_in_examined_dimensions"
    assert metrics["examined_dimensions"] == 4
    assert metrics["has_trend"] is True
    assert metrics["contribution_count"] == 0


def test_collect_anomaly_metrics_separates_terminal_code_from_output_status() -> None:
    state = _traceable_missing_trend_state()
    state["investigation_state"]["stop_reason"] = {"code": "DATA_INSUFFICIENT", "detail": "missing buckets"}
    metrics = collect_anomaly_metrics(state)
    assert metrics["stop_reason"] == "insufficient_evidence"
    assert metrics["investigation_stop_code"] == "DATA_INSUFFICIENT"
    assert metrics["final_output_status"] == "abstained"


def test_collect_anomaly_metrics_captures_call_quality_latency_and_usage() -> None:
    state = _baseline_state() | {
        "tool_calls": [
            {"tool_name": "query_metric_trend", "arguments": {"metric": "gmv"}, "status": "ok"},
            {"tool_name": "query_metric_trend", "arguments": {"metric": "gmv"}, "status": "ok"},
            {"tool_name": "query_metric_trend", "arguments": {"metric": "bad"}, "status": "failed"},
        ],
        "node_timings": [{"duration_ms": 12.5}, {"duration_ms": 7.5}],
        "llm_usage": {"input_tokens": 100, "output_tokens": 25, "estimated_cost": 0.0123},
    }
    metrics = collect_anomaly_metrics(state)
    assert metrics["invalid_call_count"] == 1
    assert metrics["duplicate_call_count"] == 1
    assert metrics["total_latency_ms"] == 20.0
    assert metrics["total_tokens"] == 125
    assert metrics["estimated_cost"] == 0.0123

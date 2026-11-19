from __future__ import annotations

import copy

from evaluation.resume_bg_numeric_scoring import (
    compare_case_numeric_fields,
    extract_case_numeric_fields,
    registered_numeric_decomposition_projection,
)
from evaluation.resume_bg_observation import _matches_period, extract_formal_observation


def _registered_decomposition_case():
    return {
        "case_id": "registered-decomposition-observation",
        "operation": "registered_numeric_decomposition",
        "target_metric": "gmv",
        "method_id": "product_shapley_v1",
        "dependency_metric_ids": ["gmv", "order_count", "avg_order_value"],
        "metric_unit": "BRL",
        "scope": "platform",
        "period": {"kind": "point_pair", "baseline": "2017-04-07", "current": "2017-04-08"},
        "expected": {"outcome": "numeric_result", "truth": {
            "baseline_value": 100.0,
            "current_value": 120.0,
            "total_delta": 20.0,
            "display_components": {"order_count": 12.0, "avg_order_value": 8.0},
        }},
        "numeric_match_contract": {
            "status": "verified",
            "absolute_tolerance_by_path": {
                "baseline_value": 0.001,
                "current_value": 0.001,
                "total_delta": 0.001,
                "display_components.*": 0.001,
            },
            "exact_key_set_prefixes": ["display_components."],
        },
    }


def _registered_decomposition_state():
    target, baseline, current = "gmv", "2017-04-07", "2017-04-08"
    factor_ids = ["order_count", "avg_order_value"]
    input_metrics = [target, *factor_ids]
    input_ids = [f"trend-evidence-{metric}" for metric in input_metrics]
    request = {
        "status": "ready", "target_metric_id": target,
        "method_id": "product_shapley_v1", "factor_metric_ids": factor_ids,
        "input_metric_ids": input_metrics,
        "baseline_date": baseline, "current_date": current,
    }
    evidence = []
    calls = []
    for metric, evidence_id in zip(input_metrics, input_ids):
        request_id = f"request-{metric}"
        evidence.append({
            "evidence_id": evidence_id, "source_type": "mcp", "source_ref": "registry_trend",
            "trusted": True,
            "content": {"request_id": request_id, "metric_id": metric, "result": {
                "status": "ok",
                "parsed": {"authorized_metric": {"metric_id": metric}},
                "trend_result": {"status": "ok", "metric_name": metric, "points": [
                    {"bucket": baseline, "value": 100.0}, {"bucket": current, "value": 120.0},
                ]},
            }},
        })
        calls.append({
            "request_id": request_id, "evidence_id": evidence_id,
            "tool_name": "registry_trend", "status": "ok",
            "arguments": {"metric_name": metric, "period": {"start_date": baseline, "end_date": current}},
            "authorization_snapshot": {
                "decision": "allowed", "authorization_checked": True,
                "metric_id": metric, "scope": "platform",
            },
        })

    decomposition_id = "decomposition-evidence"
    evidence.append({
        "evidence_id": decomposition_id, "source_type": "deterministic_attribution",
        "source_ref": "authorized_complete_daily_trend_evidence", "trusted": True,
        "content": {"result": {
            "baseline_value": 100.0, "current_value": 120.0, "total_delta": 20.0,
            "factor_metrics": factor_ids, "factor_identities": [],
            "input_evidence_ids": input_ids, "residual_explainable": True,
            "residual_tolerance": 0.001,
            "display_components": [
                {"component_id": "order_count", "value": 12.0},
                {"component_id": "avg_order_value", "value": 8.0},
            ],
        }},
    })
    return {
        "scope_type": "platform",
        "registered_decomposition_request": request,
        "metric_planning_decision": {
            "operation_intent": "registered_numeric_decomposition",
            "registered_decomposition": request,
        },
        "analysis_presentation": {"status": "completed", "metric_decomposition": {
            "metric_id": target, "unit": "BRL", "method_id": "product_shapley_v1",
            "analysis_nature": "registered_numeric_decomposition", "business_cause_confirmed": False,
            "baseline_value": 100.0, "current_value": 120.0, "delta_value": 20.0,
            "baseline_period": {"start_date": baseline, "end_date": baseline},
            "current_period": {"start_date": current, "end_date": current},
            "factors": [
                {"metric_id": "order_count", "contribution": 12.0, "unit": "BRL"},
                {"metric_id": "avg_order_value", "contribution": 8.0, "unit": "BRL"},
            ],
            "evidence_ids": [decomposition_id],
        }},
        "evidence_bundles": evidence,
        "tool_calls": calls,
    }


def test_registered_decomposition_observation_uses_target_card_not_first_factor_query():
    case = _registered_decomposition_case()
    state = _registered_decomposition_state()
    projection = registered_numeric_decomposition_projection(case, state)
    assert projection["valid"] is True
    assert projection["metric_id"] == "gmv"
    assert projection["operation"] == "registered_numeric_decomposition"
    assert projection["period"] == {
        "kind": "point_pair", "baseline": "2017-04-07", "current": "2017-04-08",
    }
    actual, source = extract_case_numeric_fields(case, state, "G")
    assert source["registered_numeric_decomposition"] == "validated"
    assert actual == {
        "baseline_value": 100.0, "current_value": 120.0, "total_delta": 20.0,
        "display_components.order_count": 12.0,
        "display_components.avg_order_value": 8.0,
    }
    assert compare_case_numeric_fields(case, actual, arm="G")[0] is True
    observation = extract_formal_observation(case, "G", {"state_snapshot": state})
    assert observation["metric_id"] == "gmv"
    assert observation["operation"] == "registered_numeric_decomposition"
    assert observation["period_match"] is True
    assert observation["numeric_values_match"] is True


def test_registered_decomposition_does_not_pass_when_card_evidence_or_input_binding_breaks():
    case = _registered_decomposition_case()
    state = _registered_decomposition_state()
    bad_card = copy.deepcopy(state)
    bad_card["analysis_presentation"]["metric_decomposition"]["factors"][0]["contribution"] = 999
    assert registered_numeric_decomposition_projection(case, bad_card)["valid"] is False

    bad_input = copy.deepcopy(state)
    bad_input["tool_calls"][0]["authorization_snapshot"]["decision"] = "denied"
    assert registered_numeric_decomposition_projection(case, bad_input)["valid"] is False


def test_month_pair_matches_its_explicit_full_periods_and_month_labels():
    expected = {
        "kind": "month_pair", "baseline_month": "2017-12", "current_month": "2018-01",
        "baseline_start": "2017-12-01", "baseline_end": "2017-12-31",
        "current_start": "2018-01-01", "current_end": "2018-01-31",
    }
    actual = {
        "kind": "period_pair", "baseline_start": "2017-12-01", "baseline_end": "2017-12-31",
        "current_start": "2018-01-01", "current_end": "2018-01-31", "granularity": "month",
    }
    assert _matches_period(actual, expected) is True
    wrong_period = dict(actual, current_start="2018-02-01")
    assert _matches_period(wrong_period, expected) is False
    contradictory_label = dict(expected, current_month="2018-02")
    assert _matches_period(actual, contradictory_label) is False


def test_month_pair_tool_observation_does_not_turn_facts_into_a_completed_answer():
    case = {
        "case_id": "period-statistical-contribution",
        "operation": "dimension_contribution", "target_metric": "avg_order_value",
        "scope": "platform", "dimension": "month_position",
        "period": {
            "kind": "month_pair", "baseline_month": "2017-12", "current_month": "2018-01",
            "baseline_start": "2017-12-01", "baseline_end": "2017-12-31",
            "current_start": "2018-01-01", "current_end": "2018-01-31",
        },
        "expected": {"outcome": "numeric_result", "truth": {
            "baseline_value": 152.92550154, "current_value": 152.48089687,
        }},
    }
    arguments = {
        "metric_name": "avg_order_value", "dimension": "month_position", "granularity": "month",
        "baseline_start": "2017-12-01", "baseline_end": "2017-12-31",
        "current_start": "2018-01-01", "current_end": "2018-01-31",
    }
    state = {
        "scope_type": "platform",
        "tool_calls": [{
            "request_id": "ratio-period-call", "planned_metric_id": "avg_order_value",
            "planned_operation": "metric_contribution", "arguments": arguments,
            "tool_name": "compare_avg_order_value_ratio_facts_by_dimension_period", "status": "ok",
        }],
        "query_results": [{"request_id": "ratio-period-call", "arguments": arguments, "result": {
            "status": "ok", "contribution_result": {
                "metric_name": "avg_order_value", "status": "ok", "complete": True,
                "current_groups": {}, "baseline_groups": {},
            },
        }}],
        "analysis_presentation": {"status": "failed", "contributions": []},
    }
    observation = extract_formal_observation(case, "G", {"state_snapshot": state})
    assert observation["operation"] == "dimension_contribution"
    assert observation["metric_id"] == "avg_order_value"
    assert observation["period_match"] is True
    # Period-tool completion is not an answer or a computed contribution.
    assert observation["numeric_values"] is None
    assert observation["numeric_values_match"] is None

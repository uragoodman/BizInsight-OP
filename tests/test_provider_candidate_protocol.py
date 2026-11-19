from __future__ import annotations

import copy
import json
import socket
from pathlib import Path

import pytest

from evaluation.provider_candidate_protocol import (
    _actual_component_map,
    _structured_output_checks,
    _text_output_checks,
    score_provider_candidate_run,
    score_provider_candidate_run_v3,
)


ROOT = Path(__file__).parents[1]
AOV_TRACE = ROOT / "docs/testing/diagnostic_presentation_status_verify_20261011/traces/aov_ratio.json"
PROTOCOL_FILE = ROOT / "docs/testing/provider_candidate_protocol_17_20261013/candidate_protocol_17.json"
EXCLUSION_FILE = ROOT / "docs/testing/provider_candidate_protocol_17_20261013/exclusions_15.json"


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def deny(*args, **kwargs):
        raise AssertionError("network access is blocked in candidate protocol tests")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


def _aov_candidate() -> dict:
    return {
        "case_id": "SCORE-POSITIVE-AOV",
        "target_metric": "avg_order_value",
        "scope": "platform",
        "comparison_period": {"baseline": "2024-05-03", "current": "2024-05-04"},
        "registered_capability": {"method": "symmetric_ratio_v1"},
        "unit": "BRL/order",
        "tolerance": {"absolute": 0.03},
        "scoring_expectation": {
            "expected_outcome": "numeric_decomposition",
            "component_field": "display_components",
            "method": "symmetric_ratio_v1",
            "evidence_unit": "BRL/order",
            "values": {
                "baseline": 10.0, "current": 20.0, "delta": 10.0,
                "components": {"gmv": 2.6666666667, "order_count": 7.3333333333},
            },
        },
        "output_expectation": {
            "kind": "numeric_decomposition", "metric_labels": ["avg_order_value", "客单价"],
            "metric_unit": "BRL/order", "change_unit": "BRL/order", "component_unit": "BRL/order",
            "unit_text_aliases": ["BRL/order"], "component_unit_text_aliases": ["BRL/order"],
            "component_text_labels": {"gmv": ["gmv", "成交总额"], "order_count": ["order_count", "订单量"]},
            "period_text": {"baseline_text_tokens": ["2024-05-03"], "current_text_tokens": ["2024-05-04"]},
            "baseline_text_labels": ["基线"], "current_text_labels": ["当前"],
            "numeric_tolerance": 0.03, "causal_boundary_required": True,
        },
    }


def _full_trace() -> dict:
    trace = json.loads(AOV_TRACE.read_text(encoding="utf-8"))
    state = trace["workflow_state"]
    # Score-only fixture: the old development trace predates unit propagation;
    # set the registered unit to construct a valid expected-input example.
    for evidence in state["investigation_state"]["evidence"]:
        if evidence.get("evidence_type") == "metric_decomposition":
            evidence["unit"] = "BRL/order"
    decomposition = next(item for item in state["investigation_state"]["evidence"] if item.get("evidence_type") == "metric_decomposition")
    evidence_id = decomposition["evidence_id"]
    state["answer"] = (
        f"客单价：基线 2024-05-03 为 10.00 BRL/order；当前 2024-05-04 为 20.00 BRL/order；"
        f"变化 +10.00 BRL/order。成交总额贡献 2.67 BRL/order，订单量贡献 7.33 BRL/order。证据：{evidence_id}。"
    )
    presentation = state.setdefault("analysis_presentation", {})
    presentation["status"] = "completed"
    presentation["scope"] = {"seller_id": None, "start_date": "2024-05-03", "end_date": "2024-05-04", "granularity": "day"}
    presentation["metric_decomposition"] = {
        "metric_id": "avg_order_value", "unit": "BRL/order", "change_unit": "BRL/order",
        "baseline_value": 10.0, "current_value": 20.0, "delta_value": 10.0,
        "baseline_period": {"start_date": "2024-05-03", "end_date": "2024-05-03"},
        "current_period": {"start_date": "2024-05-04", "end_date": "2024-05-04"},
        "method_id": "symmetric_ratio_v1", "analysis_nature": "registered_numeric_decomposition",
        "business_cause_confirmed": False, "evidence_ids": [evidence_id],
        "factors": [
            {"metric_id": "gmv", "contribution": 2.6666666667, "unit": "BRL/order"},
            {"metric_id": "order_count", "contribution": 7.3333333333, "unit": "BRL/order"},
        ],
    }
    review = state["critic_result"]["llm_review"]
    review.update({
        "mode": "full", "status": "invoked",
        "groundedness": {"verdict": "passed", "issues": []},
        "completeness": {"verdict": "passed", "issues": []},
        "consistency": {"verdict": "passed", "issues": []},
    })
    state["critic_result"]["status"] = "approved"
    state["llm_calls"] = [
        {"node": "critic", "operation": "extract", "call_id": f"critic-call-{index}",
         "outcome": "success", "attempts": [{"attempt_id": f"critic-attempt-{index}"}]}
        for index in range(1, 4)
    ]
    return trace


def _statistical_input() -> tuple[dict, dict]:
    evidence_id = "ev_stat_score"
    scope = "scope-platform-score"
    baseline, current = "2024-06-01", "2024-06-02"
    values = {
        "query": {"dimension": "customer_state"},
        "baseline_value": 10.0, "current_value": 9.0, "total_delta": -1.0,
        "components": [{"dimension_value": "SP", "delta_value": -1.0}],
        "residual_value": 0.0, "residual_type": "rounding", "residual_explainable": True,
    }
    evidence = {
        "evidence_id": evidence_id, "evidence_type": "dimension_contribution",
        "metric_id": "order_count", "values": values, "unit": "orders",
        "source_action_id": "candidate-stat", "source_ref": "registered_dimension_contribution",
        "scope_fingerprint": scope, "method_id": "additive_change_v1", "method_version": "1.0",
        "comparison": {
            "current_period": {"start_date": current, "end_date": current},
            "baseline_period": {"start_date": baseline, "end_date": baseline},
        },
    }
    state = {
        "trace_id": "offline-score-only", "scope_type": "platform",
        "answer": f"订单量，客户州维度；基线 2024-06-01 为 10 orders；当前 2024-06-02 为 9 orders；变化 -1 orders。SP 贡献 -1 orders。该统计贡献不证明业务因果。证据：{evidence_id}。",
        "analysis_result": {
            "abstain": False, "evidence_ids": [evidence_id], "summary": "statistical contribution",
            "contribution": [{
                "analysis_nature": "descriptive_contribution", "metric_name": "order_count",
                "dimension": "customer_state", "current_date": current, "baseline_date": baseline,
                "items": [{"dimension_value": "SP", "delta_value": -1.0}], "evidence_ids": [evidence_id],
            }],
            "investigation_summary": {"stop_reason": {"code": "ENOUGH_EVIDENCE"}},
        },
        "analysis_presentation": {
            "status": "completed", "scope": {"seller_id": None, "start_date": baseline, "end_date": current, "granularity": "day"},
            "changes": [{"metric_name": "order_count", "unit": "orders", "change_unit": "orders",
                         "delta_ratio": -10.0, "delta_ratio_unit": "%", "baseline_value": 10.0, "current_value": 9.0, "delta_value": -1.0,
                         "baseline_period": baseline, "current_period": current, "evidence_ids": [evidence_id]}],
            "contributions": [{"metric_name": "order_count", "dimension": "customer_state", "analysis_nature": "descriptive_contribution",
                               "method": "additive_change_v1", "unit": "orders", "total_delta": -1.0,
                               "baseline_period": {"start_date": baseline, "end_date": baseline},
                               "current_period": {"start_date": current, "end_date": current},
                               "items": [{"dimension_value": "SP", "delta_value": -1.0}], "evidence_ids": [evidence_id]}],
        },
        "finish_decision": {"action_type": "FINISH", "evidence_ids": [evidence_id]},
        "investigation_state": {
            "target": {
                "metric_id": "order_count", "current_start": current, "current_end": current,
                "baseline_start": baseline, "baseline_end": baseline,
                "canonical_scope": {"scope_type": "platform"}, "scope_fingerprint": scope,
            },
            "stop_reason": {"code": "ENOUGH_EVIDENCE"},
            "evidence": [evidence],
            "action_results": [{
                "candidate_id": "candidate-stat", "metric_id": "order_count", "method_id": "additive_change_v1",
                "status": "completed", "validation_result": {"status": "valid"}, "evidence_ids": [evidence_id],
            }],
        },
        "critic_result": {
            "status": "approved", "llm_review": {
                "mode": "critic_llm_review_off", "status": "evaluation_ablation",
                "groundedness": {"verdict": "evaluation_ablation"},
                "completeness": {"verdict": "evaluation_ablation"},
                "consistency": {"verdict": "evaluation_ablation"},
            },
            "causal_review": {"status": "approved", "issues": []},
        },
        "llm_calls": [], "errors": [],
    }
    candidate = {
        "case_id": "SCORE-STATISTICAL", "target_metric": "order_count", "scope": "platform",
        "comparison_period": {"baseline": baseline, "current": current},
        "registered_capability": {"method": "additive_change_v1"}, "unit": "orders",
        "tolerance": {"absolute": 0.03},
        "scoring_expectation": {
            "expected_outcome": "statistical_contribution", "component_field": "components",
            "evidence_unit": "orders", "components": {"SP": -1.0},
            "values": {"baseline": 10.0, "current": 9.0, "delta": -1.0},
        },
        "output_expectation": {
            "kind": "statistical_contribution", "metric_labels": ["order_count", "订单量"], "metric_unit": "orders",
            "change_unit": "orders", "component_unit": "orders", "unit_text_aliases": ["orders"], "component_unit_text_aliases": ["orders"],
            "relative_change_unit": "%",
            "dimension_id": "customer_state", "component_text_labels": {"SP": ["SP"]},
            "period_text": {"baseline_text_tokens": [baseline], "current_text_tokens": [current]},
            "baseline_text_labels": ["基线"], "current_text_labels": ["当前"], "numeric_tolerance": 0.03,
            "causal_boundary_required": True,
        },
    }
    return state, candidate


def test_saved_decomposition_scores_full_only_when_all_critic_reviews_complete() -> None:
    result = score_provider_candidate_run(_full_trace(), candidate=_aov_candidate(), arm="full", replicate=1)
    assert result["scorer_version"] == "provider-candidate-run-v4"
    assert result["base_task_acceptance_pass"] is True
    assert result["evidence_valid_for_candidate"] is True
    assert result["numeric_values_correct"] is True
    assert result["critic"]["review_contract_pass"] is True
    assert result["structured_output"]["card_matches"] is True
    assert result["final_answer_output"]["text_matches_candidate"] is True
    assert result["task_acceptance_pass"] is True
    assert result["effective_expected_result_observed"] is True


def test_v1_score_remains_available_and_unchanged_by_output_checks() -> None:
    from evaluation.provider_candidate_protocol import score_provider_candidate_run_v1

    legacy_candidate = _aov_candidate()
    values = legacy_candidate["scoring_expectation"]["values"]
    legacy_candidate["scoring_expectation"]["components"] = values.pop("components")
    result = score_provider_candidate_run_v1(_full_trace(), candidate=legacy_candidate, arm="full", replicate=1)
    assert result["scorer_version"] == "provider-candidate-run-v1"
    assert result["task_acceptance_pass"] is True


def test_v3_scorer_remains_callable_for_historical_comparison() -> None:
    result = score_provider_candidate_run_v3(_full_trace(), candidate=_aov_candidate(), arm="full", replicate=1)
    assert result["scorer_version"] == "provider-candidate-run-v3"


def test_v4_maps_ratio_components_only_from_registered_order_and_contract() -> None:
    trace = _full_trace()
    item = next(
        row for row in trace["workflow_state"]["investigation_state"]["evidence"]
        if row.get("evidence_type") == "metric_decomposition"
    )
    candidate = _aov_candidate()
    assert _actual_component_map(item, candidate) == pytest.approx({
        "gmv": 2.6666666667, "order_count": 7.3333333333,
    })

    reversed_components = copy.deepcopy(item)
    reversed_components["values"]["display_components"].reverse()
    assert _actual_component_map(reversed_components, candidate) is None

    wrong_type = copy.deepcopy(item)
    wrong_type["values"]["display_components"][0]["component_type"] = "denominator_effect"
    assert _actual_component_map(wrong_type, candidate) is None

    wrong_factor_order = copy.deepcopy(item)
    wrong_factor_order["values"]["factor_metrics"] = ["order_count", "gmv"]
    assert _actual_component_map(wrong_factor_order, candidate) is None

    wrong_component_unit = copy.deepcopy(item)
    wrong_component_unit["values"]["display_components"][0]["unit"] = "%"
    assert _actual_component_map(wrong_component_unit, candidate) is None


def test_v4_rejects_missing_structured_scope_and_uses_nested_component_truth() -> None:
    trace = _full_trace()
    candidate = _aov_candidate()
    assert "components" not in candidate["scoring_expectation"]
    trace["workflow_state"]["analysis_presentation"]["scope"] = None
    result = score_provider_candidate_run(trace, candidate=candidate, arm="full", replicate=1)
    assert result["numeric_values_correct"] is True
    assert result["scope_matches_candidate"] is True
    assert result["structured_output"]["checks"]["scope"] is False
    assert result["task_acceptance_pass"] is False


def test_wrong_numeric_truth_fails_without_changing_base_structural_score() -> None:
    candidate = _aov_candidate()
    candidate["scoring_expectation"]["values"]["delta"] = 11.0
    result = score_provider_candidate_run(_full_trace(), candidate=candidate, arm="full", replicate=1)
    assert result["base_task_acceptance_pass"] is True
    assert result["evidence_valid_for_candidate"] is True
    assert result["numeric_values_correct"] is False
    assert result["task_acceptance_pass"] is False


def test_missing_registered_unit_fails_candidate_truth_binding() -> None:
    trace = _full_trace()
    state = trace["workflow_state"]
    for evidence in state["investigation_state"]["evidence"]:
        if evidence.get("evidence_type") == "metric_decomposition":
            evidence["unit"] = ""
    result = score_provider_candidate_run(trace, candidate=_aov_candidate(), arm="full", replicate=1)
    assert result["numeric_values_correct"] is True
    assert result["evidence_unit_matches"] is False
    assert result["task_acceptance_pass"] is False


@pytest.mark.parametrize("mismatch", ["scope", "period"])
def test_wrong_scope_or_period_fails_candidate_evidence_binding(mismatch: str) -> None:
    candidate = _aov_candidate()
    if mismatch == "period":
        candidate["comparison_period"]["current"] = "2024-05-05"
    else:
        candidate["scope"] = "seller"
    result = score_provider_candidate_run(_full_trace(), candidate=candidate, arm="full", replicate=1)
    assert result["evidence_valid_for_candidate"] is False
    assert result["task_acceptance_pass"] is False


def test_valid_decomposition_cannot_hide_execution_failure() -> None:
    trace = _full_trace()
    state = trace["workflow_state"]
    state["investigation_state"]["stop_reason"]["code"] = "EXECUTION_FAILED"
    state["investigation_state"]["stop_reason"]["detail"] = "later execution error"
    state["analysis_presentation"]["status"] = "failed"
    state["errors"] = [{"code": "execution_exception", "message": "later action failed"}]
    result = score_provider_candidate_run(trace, candidate=_aov_candidate(), arm="full", replicate=1)
    assert result["numeric_values_correct"] is True
    assert result["effective_expected_result_observed"] is True
    assert result["execution_error_free"] is False
    assert result["task_acceptance_pass"] is False


def test_full_arm_with_incomplete_critic_review_fails_overall_task() -> None:
    trace = _full_trace()
    state = trace["workflow_state"]
    state["critic_result"]["llm_review"]["consistency"] = {"verdict": "skipped", "issues": []}
    state["llm_calls"] = state["llm_calls"][:2]
    result = score_provider_candidate_run(trace, candidate=_aov_candidate(), arm="full", replicate=1)
    assert result["numeric_values_correct"] is True
    assert result["critic"]["review_contract_pass"] is False
    assert result["task_acceptance_pass"] is False


def test_partial_keeps_valid_expected_result_separate_from_full_task_pass() -> None:
    trace = _full_trace()
    state = trace["workflow_state"]
    state["investigation_state"]["stop_reason"]["code"] = "DATA_INSUFFICIENT"
    state["analysis_presentation"]["status"] = "partial"
    result = score_provider_candidate_run(trace, candidate=_aov_candidate(), arm="full", replicate=1)
    assert result["partial_output"] is True
    assert result["numeric_values_correct"] is True
    assert result["effective_expected_result_observed"] is True
    assert result["partial_complete_task_pass"] is False
    assert result["task_acceptance_pass"] is False


def test_correct_evidence_does_not_hide_answer_omitting_expected_result() -> None:
    trace = _full_trace()
    trace["workflow_state"]["answer"] = "客单价已核对。证据：" + trace["workflow_state"]["analysis_presentation"]["metric_decomposition"]["evidence_ids"][0]
    result = score_provider_candidate_run(trace, candidate=_aov_candidate(), arm="full", replicate=1)
    assert result["evidence_valid_for_candidate"] is True
    assert result["numeric_values_correct"] is True
    assert result["structured_output"]["card_matches"] is True
    assert result["final_answer_output"]["text_matches_candidate"] is False
    assert result["task_acceptance_pass"] is False


def test_correct_card_does_not_hide_wrong_text_value_or_direction() -> None:
    trace = _full_trace()
    answer = trace["workflow_state"]["answer"]
    trace["workflow_state"]["answer"] = answer.replace("变化 +10.00", "变化 -10.00")
    result = score_provider_candidate_run(trace, candidate=_aov_candidate(), arm="full", replicate=1)
    assert result["structured_output"]["card_matches"] is True
    assert result["final_answer_output"]["checks"]["delta_value_and_direction"] is False
    assert result["task_acceptance_pass"] is False


def test_valid_card_citation_can_ground_answer_without_inline_id() -> None:
    trace = _full_trace()
    trace["workflow_state"]["answer"] = trace["workflow_state"]["answer"].split("证据：")[0]
    result = score_provider_candidate_run(trace, candidate=_aov_candidate(), arm="full", replicate=1)
    assert result["structured_output"]["checks"]["evidence_reference"] is True
    assert result["final_answer_output"]["evidence_reference_source"] == "structured_presentation_card"
    assert result["task_acceptance_pass"] is True


def test_wrong_structured_card_evidence_reference_fails_output_quality() -> None:
    trace = _full_trace()
    trace["workflow_state"]["analysis_presentation"]["metric_decomposition"]["evidence_ids"] = ["ev_wrong_run"]
    result = score_provider_candidate_run(trace, candidate=_aov_candidate(), arm="full", replicate=1)
    assert result["numeric_values_correct"] is True
    assert result["structured_output"]["checks"]["evidence_reference"] is False
    assert result["output_quality_pass"] is False
    assert result["task_acceptance_pass"] is False


@pytest.mark.parametrize("change", ["period", "unit", "citation"])
def test_answer_period_unit_or_evidence_reference_must_match(change: str) -> None:
    trace = _full_trace()
    answer = trace["workflow_state"]["answer"]
    if change == "period":
        trace["workflow_state"]["answer"] = answer.replace("2024-05-03", "2024-05-02")
    elif change == "unit":
        trace["workflow_state"]["answer"] = answer.replace("BRL/order", "orders")
    else:
        trace["workflow_state"]["answer"] = answer.replace("ev_", "ev_bad_")
    result = score_provider_candidate_run(trace, candidate=_aov_candidate(), arm="full", replicate=1)
    assert result["structured_output"]["card_matches"] is True
    assert result["final_answer_output"]["text_matches_candidate"] is False
    assert result["task_acceptance_pass"] is False


def test_registered_display_rounding_within_tolerance_passes_text_check() -> None:
    trace = _full_trace()
    answer = trace["workflow_state"]["answer"]
    answer = answer.replace("10.00 BRL/order", "10.02 BRL/order", 1)
    trace["workflow_state"]["answer"] = answer
    result = score_provider_candidate_run(trace, candidate=_aov_candidate(), arm="full", replicate=1)
    assert result["numeric_values_correct"] is True
    assert result["final_answer_output"]["text_matches_candidate"] is True
    assert result["task_acceptance_pass"] is True


def test_statistical_contribution_is_accepted_without_causal_proof() -> None:
    state, candidate = _statistical_input()
    state["answer"] = state["answer"].replace("该统计贡献不证明业务因果。", "")
    result = score_provider_candidate_run(state, candidate=candidate, arm="off", replicate=1)
    assert result["base_effective_diagnosis_completed"] is True
    assert result["numeric_values_correct"] is True
    assert result["critic"]["review_contract_pass"] is True
    assert result["task_acceptance_pass"] is True


def test_statistical_output_requires_separate_metric_unit_in_change_card() -> None:
    state, candidate = _statistical_input()
    state["analysis_presentation"]["changes"][0].pop("unit")
    result = score_provider_candidate_run(state, candidate=candidate, arm="off", replicate=1)
    assert result["structured_output"]["checks"]["metric_unit"] is False
    assert result["output_quality_pass"] is False
    assert result["task_acceptance_pass"] is False


def test_statistical_output_requires_separate_change_and_relative_units() -> None:
    state, candidate = _statistical_input()
    state["analysis_presentation"]["changes"][0].pop("change_unit")
    result = score_provider_candidate_run(state, candidate=candidate, arm="off", replicate=1)
    assert result["structured_output"]["checks"]["change_unit"] is False
    assert result["output_quality_pass"] is False
    state, candidate = _statistical_input()
    state["analysis_presentation"]["changes"][0].pop("delta_ratio_unit")
    result = score_provider_candidate_run(state, candidate=candidate, arm="off", replicate=1)
    assert result["structured_output"]["checks"]["relative_change_unit"] is False


def test_unmapped_answer_alias_is_manual_review_not_automatic_pass() -> None:
    trace = _full_trace()
    candidate = _aov_candidate()
    candidate["output_expectation"]["component_text_labels"].pop("gmv")
    result = score_provider_candidate_run(trace, candidate=candidate, arm="full", replicate=1)
    assert result["final_answer_output"]["manual_review_status"] == "required"
    assert result["output_quality_pass"] is False
    assert result["task_acceptance_pass"] is False


def test_off_arm_requires_explicit_ablation_and_zero_critic_calls() -> None:
    trace = _full_trace()
    state = trace["workflow_state"]
    state["critic_result"]["llm_review"].update({
        "mode": "critic_llm_review_off", "status": "evaluation_ablation",
        "groundedness": {"verdict": "evaluation_ablation"},
        "completeness": {"verdict": "evaluation_ablation"},
        "consistency": {"verdict": "evaluation_ablation"},
    })
    state["llm_calls"] = [call for call in state["llm_calls"] if call.get("node") != "critic"]
    result = score_provider_candidate_run(trace, candidate=_aov_candidate(), arm="off", replicate=1)
    assert result["critic"]["review_contract_pass"] is True
    assert result["critic"]["critic_review_sdk_logical_calls"] == 0
    assert result["task_acceptance_pass"] is True


def test_v4_accepts_the_batch_executor_off_arm_enum_without_changing_v3() -> None:
    trace = _full_trace()
    state = trace["workflow_state"]
    state["critic_result"]["llm_review"].update({
        "mode": "critic_llm_review_off", "status": "evaluation_ablation",
        "groundedness": {"verdict": "evaluation_ablation"},
        "completeness": {"verdict": "evaluation_ablation"},
        "consistency": {"verdict": "evaluation_ablation"},
    })
    state["llm_calls"] = [call for call in state["llm_calls"] if call.get("node") != "critic"]

    # v3 retains its historical canonical enum; v4 adapts the batch runner's
    # persisted arm name before delegating to that unchanged base contract.
    historical = score_provider_candidate_run_v3(
        trace, candidate=_aov_candidate(), arm="off", replicate=1,
    )
    assert historical["scorer_version"] == "provider-candidate-run-v3"
    current = score_provider_candidate_run(
        trace, candidate=_aov_candidate(), arm="critic_llm_review_off", replicate=1,
    )
    assert current["arm"] == "critic_llm_review_off"
    assert current["scoring_arm"] == "off"
    assert current["critic"]["review_contract_pass"] is True
    assert current["critic"]["critic_review_sdk_logical_calls"] == 0


def test_v4_rejects_unknown_arm_instead_of_defaulting_to_off() -> None:
    with pytest.raises(ValueError, match="arm must be 'full'"):
        score_provider_candidate_run(
            _full_trace(), candidate=_aov_candidate(), arm="unexpected", replicate=1,
        )


def test_manifest_has_only_the_reviewed_17_cases_and_monthly_dc06_dc07() -> None:
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    excluded = json.loads(EXCLUSION_FILE.read_text(encoding="utf-8"))
    assert protocol["status"] == "data_and_scoring_protocol_candidate_not_frozen"
    assert protocol["scoring"]["scorer_version"] == "provider-candidate-run-v3"
    assert protocol["case_count"] == 17
    assert protocol["task_type_counts"] == {
        "registered_numeric_decomposition": 10,
        "dimension_statistical_contribution": 7,
    }
    assert protocol["gmv_count"] == 3
    assert protocol["planned_run_denominator"] == 102
    assert all("output_expectation" in item for item in protocol["cases"])
    by_id = {item["case_id"]: item for item in protocol["cases"]}
    assert by_id["DC06"]["comparison_period"]["baseline"] == {"start": "2017-01-01", "end": "2017-01-31"}
    assert by_id["DC06"]["comparison_period"]["current"] == {"start": "2017-02-01", "end": "2017-02-28"}
    assert by_id["DC07"]["comparison_period"]["baseline"] == {"start": "2017-03-01", "end": "2017-03-31"}
    assert by_id["DC07"]["comparison_period"]["current"] == {"start": "2017-04-01", "end": "2017-04-30"}
    assert by_id["DC07"]["scoring_expectation"]["evidence_unit"] == "percentage_point"
    assert {item["case_id"] for item in excluded["cases"]} == {
        "DC02", "NC11", "NC12", "LI01", "LI02", "LI03", "LI04", "LI05", "LI06", "LI07", "LI08",
        "CF01", "CF02", "CF03", "CF04",
    }


def _freight_rate_candidate() -> dict:
    return {
        "case_id": "UNIT-RATE-10-12", "target_metric": "freight_rate", "scope": "platform",
        "comparison_period": {"baseline": "2024-01-01", "current": "2024-01-02"},
        "registered_capability": {"method": "symmetric_ratio_v1"}, "unit": "%",
        "tolerance": {"absolute": 0.03},
        "scoring_expectation": {
            "expected_outcome": "numeric_decomposition", "method": "symmetric_ratio_v1",
            "evidence_unit": "%", "values": {"baseline": 10.0, "current": 12.0, "delta": 2.0,
                                                  "components": {"freight_value": 1.0, "gmv": 1.0}},
        },
        "output_expectation": {
            "kind": "numeric_decomposition", "metric_labels": ["freight_rate"],
            "metric_unit": "%", "change_unit": "percentage_points", "component_unit": "percentage_points",
            "metric_unit_text_aliases": ["%", "percent", "percentage"],
            "change_unit_text_aliases": ["percentage points", "percentage point", "pp"],
            "component_unit_text_aliases": ["percentage points", "percentage point", "pp"],
            "relative_change_unit": "%", "relative_change_unit_text_aliases": ["%", "percent", "percentage"],
            "expected_relative_change_percent": 20.0, "relative_change_text_markers": ["relative change", "relative growth", "delta_ratio"],
            "component_text_labels": {"freight_value": ["freight_value"], "gmv": ["gmv"]},
            "baseline_text_labels": ["baseline"], "current_text_labels": ["current"],
            "period_text": {"baseline_text_tokens": ["2024-01-01"], "current_text_tokens": ["2024-01-02"]},
            "numeric_tolerance": 0.03,
        },
    }


def _freight_rate_card(candidate: dict, *, metric_unit="%", change_unit="percentage_points", component_unit="percentage_points") -> dict:
    values = candidate["scoring_expectation"]["values"]
    return {
        "scope": {"seller_id": None, "start_date": "2024-01-01", "end_date": "2024-01-02"},
        "metric_decomposition": {
            "metric_id": "freight_rate", "unit": metric_unit, "change_unit": change_unit,
            "baseline_value": 10.0, "current_value": 12.0, "delta_value": 2.0,
            "baseline_period": {"start_date": "2024-01-01", "end_date": "2024-01-01"},
            "current_period": {"start_date": "2024-01-02", "end_date": "2024-01-02"},
            "method_id": "symmetric_ratio_v1", "analysis_nature": "registered_numeric_decomposition",
            "business_cause_confirmed": False, "evidence_ids": ["ev-rate"],
            "factors": [
                {"metric_id": "freight_value", "contribution": 1.0, "unit": component_unit},
                {"metric_id": "gmv", "contribution": 1.0, "unit": component_unit},
            ],
        },
    }


def _freight_rate_answer(candidate: dict, *, delta_unit="percentage points", relative=None) -> str:
    text = (
        "freight_rate baseline 2024-01-01: 10.00%; current 2024-01-02: 12.00%; "
        f"absolute change +2.00 {delta_unit}; freight_value contribution +1.00 percentage points; "
        "gmv contribution +1.00 percentage points."
    )
    if relative is not None:
        text += f" Relative growth {relative:.2f}%."
    return text


def test_percent_metric_separates_metric_change_and_component_units() -> None:
    candidate = _freight_rate_candidate()
    structured = _structured_output_checks(_freight_rate_card(candidate), candidate, {"ev-rate"})
    assert structured["checks"]["metric_unit"] is True
    assert structured["checks"]["change_unit"] is True
    assert structured["checks"]["factor_units"] is True
    assert structured["card_matches"] is True
    text = _text_output_checks(_freight_rate_answer(candidate, relative=20.0), candidate, {"ev-rate"})
    assert text["checks"]["baseline_value_context"] is True
    assert text["checks"]["current_value_context"] is True
    assert text["checks"]["delta_value_and_direction"] is True
    assert text["checks"]["relative_change_claim"] is True
    assert text["text_matches_candidate"] is True


def test_percent_delta_cannot_be_mislabeled_as_relative_percent_growth() -> None:
    candidate = _freight_rate_candidate()
    wrong_absolute_unit = _text_output_checks(_freight_rate_answer(candidate, delta_unit="%"), candidate, {"ev-rate"})
    assert wrong_absolute_unit["checks"]["delta_value_and_direction"] is False
    wrong_relative_claim = _text_output_checks(_freight_rate_answer(candidate, relative=2.0), candidate, {"ev-rate"})
    assert wrong_relative_claim["checks"]["relative_change_claim"] is False
    assert wrong_relative_claim["text_matches_candidate"] is False


def test_registered_equivalent_unit_names_are_accepted_without_cross_equivalence() -> None:
    candidate = _freight_rate_candidate()
    text = (
        "freight_rate baseline 2024-01-01: 10.00 percent; current 2024-01-02: 12.00 percent; "
        "absolute change +2.00 pp; freight_value contribution +1.00 percentage point; "
        "gmv contribution +1.00 percentage points."
    )
    result = _text_output_checks(text, candidate, {"ev-rate"})
    assert result["checks"]["baseline_value_context"] is True
    assert result["checks"]["current_value_context"] is True
    assert result["checks"]["delta_value_and_direction"] is True
    assert result["checks"]["component_claims_contextualized"] is True
    assert result["text_matches_candidate"] is True
    invalid = _text_output_checks(text.replace("+2.00 pp", "+2.00%"), candidate, {"ev-rate"})
    assert invalid["checks"]["delta_value_and_direction"] is False


def test_statistical_percent_change_units_must_be_explicit() -> None:
    state, candidate = _statistical_input()
    candidate["target_metric"] = "freight_rate"
    candidate["unit"] = "%"
    candidate["scoring_expectation"]["evidence_unit"] = "percentage_point"
    candidate["scoring_expectation"]["values"] = {
        "baseline": 10.0, "current": 12.0, "delta": 2.0, "components": {"SP": 2.0},
    }
    candidate["scoring_expectation"]["components"] = {"SP": 2.0}
    candidate["output_expectation"].update({
        "metric_unit": "%", "change_unit": "percentage_points", "component_unit": "percentage_point",
    })
    presentation = state["analysis_presentation"]
    presentation["changes"][0]["metric_name"] = "freight_rate"
    presentation["changes"][0]["baseline_value"] = 10.0
    presentation["changes"][0]["current_value"] = 12.0
    presentation["changes"][0]["delta_value"] = 2.0
    presentation["changes"][0].pop("unit", None)
    contribution = presentation["contributions"][0]
    contribution["metric_name"] = "freight_rate"
    contribution["unit"] = "percentage_point"
    contribution["total_delta"] = 2.0
    contribution["items"][0]["delta_value"] = 2.0
    checks = _structured_output_checks(presentation, candidate, {"ev_stat_score"})["checks"]
    assert checks["component_unit"] is True
    assert checks["metric_unit"] is False
    assert checks["change_unit"] is False


def test_v3_checks_units_on_a_registry_generated_dc07_change_card() -> None:
    from app.graph.nodes.answer import _make_changes_from_tool

    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    candidate = next(case for case in protocol["cases"] if case["case_id"] == "DC07")
    values = candidate["scoring_expectation"]["values"]
    periods = candidate["comparison_period"]
    current_period, baseline_period = periods["current"], periods["baseline"]
    current, baseline = values["current"], values["baseline"]
    compare_item = {
        "tool_name": "query_scoped_metric_compare",
        "result": {"compare_result": {
            "metric_name": "freight_rate", "left_value": current, "right_value": baseline,
            "delta_value": values["delta"],
            "delta_ratio": values["delta"] * 100.0 / baseline if baseline else None,
            "left_date": current_period["start"], "left_end_date": current_period["end"],
            "right_date": baseline_period["start"], "right_end_date": baseline_period["end"],
            "status": "ok",
        }},
    }
    evidence = {
        "evidence_id": "ev-dc07-compare", "source_type": "mcp",
        "source_ref": compare_item["tool_name"], "content": {"result": compare_item["result"]},
    }
    cards, _ = _make_changes_from_tool(compare_item, [evidence])
    presentation = {
        "scope": {"seller_id": None, "start_date": baseline_period["start"], "end_date": current_period["end"]},
        "changes": cards,
        "contributions": [{
            "metric_name": "freight_rate", "dimension": candidate["output_expectation"]["dimension_id"],
            "analysis_nature": "descriptive_contribution",
            "method": candidate["scoring_expectation"]["method"],
            "unit": candidate["output_expectation"]["component_unit"],
            "total_delta": values["delta"], "evidence_ids": ["ev-dc07-compare"],
            "items": [{"dimension_value": key, "delta_value": value}
                      for key, value in values["components"].items()],
        }],
    }
    checks = _structured_output_checks(presentation, candidate, {"ev-dc07-compare"})["checks"]
    assert cards[0]["unit"] == "%"
    assert cards[0]["change_unit"] == "percentage_points"
    assert cards[0]["delta_ratio_unit"] == "%"
    assert checks["metric_unit"] is True
    assert checks["change_unit"] is True
    assert checks["relative_change_unit"] is True


def test_raw_percent_evidence_does_not_validate_percentage_point_display() -> None:
    candidate = _freight_rate_candidate()
    wrong_change = _structured_output_checks(
        _freight_rate_card(candidate, change_unit="%"), candidate, {"ev-rate"}
    )
    assert wrong_change["checks"]["change_unit"] is False
    assert wrong_change["card_matches"] is False
    wrong_component = _structured_output_checks(
        _freight_rate_card(candidate, component_unit="%"), candidate, {"ev-rate"}
    )
    assert wrong_component["checks"]["factor_units"] is False
    assert wrong_component["card_matches"] is False


def test_candidate_nc07_to_nc10_have_separate_percent_unit_expectations() -> None:
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in protocol["cases"]}
    for case_id in ("NC07", "NC08", "NC09", "NC10"):
        case = cases[case_id]
        expectation = case["output_expectation"]
        assert case["unit"] == "%"
        assert case["scoring_expectation"]["evidence_unit"] == "%"
        assert expectation["metric_unit"] == "%"
        assert expectation["change_unit"] == "percentage_points"
        assert expectation["component_unit"] == "percentage_points"
        assert expectation["relative_change_unit"] == "%"
        assert expectation["unit_basis"]["raw_evidence_unit_is_not_display_unit_proof"] is True
        assert "%" not in expectation["change_unit_text_aliases"]


def test_money_and_aov_units_remain_registry_units() -> None:
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in protocol["cases"]}
    for case_id in ("NC01", "NC04", "NC05", "NC06"):
        case = cases[case_id]
        expectation = case["output_expectation"]
        assert expectation["metric_unit"] == case["unit"]
        assert expectation["change_unit"] == case["unit"]
        assert expectation["component_unit"] == case["unit"]


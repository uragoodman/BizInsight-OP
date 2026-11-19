"""Contract tests for the development-only baseline/governed protocol."""
from __future__ import annotations

import ipaddress
import json
import socket
from pathlib import Path

import pytest

from evaluation.resume_bg_protocol import ARMS, build_manifest, score_bg_run, validate_cases
from test_platform_metric_trend_acceptance import isolated_trend_db  # noqa: F401


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch):
    def blocked(sock, address):
        host = address[0] if isinstance(address, tuple) else address
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host == "localhost"
        if not loopback:
            raise AssertionError(f"external network is blocked by B/G protocol tests: {host}")
        return None

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", lambda address, *a, **k: blocked(None, address))


def _cases():
    return [
        {"case_id": "N1", "protocol_task_type": "registered_numeric_decomposition", "target_metric": "gmv"},
        {"case_id": "N2", "protocol_task_type": "registered_numeric_decomposition", "target_metric": "avg_order_value"},
        {"case_id": "D1", "protocol_task_type": "dimension_statistical_contribution", "target_metric": "freight_value"},
        {"case_id": "D2", "protocol_task_type": "dimension_statistical_contribution", "target_metric": "freight_rate"},
        {"case_id": "S1", "protocol_task_type": "expected_data_insufficiency", "target_metric": "gmv", "readiness": "blocked"},
        {"case_id": "L1", "protocol_task_type": "expected_capability_limit", "target_metric": "order_count", "readiness": "blocked"},
    ]


def _bound_public_evidence(evidence_id: str, metric_id: str, status: str = "ok", *, start=None, end=None, dimension=None):
    request_id = f"req:{evidence_id}"
    parsed = {"metric_name": metric_id, "authorized_metric": {"metric_id": metric_id}}
    if start is not None:
        parsed.update({"start_date": start, "end_date": end})
    if dimension is not None:
        parsed["dimension"] = dimension
    return {
        "tool_calls": [{"request_id": request_id, "call_status": "succeeded"}],
        "evidence_bundles": [{
            "evidence_id": evidence_id, "request_id": request_id,
            "content": {"request_id": request_id, "result": {
                "status": status,
                "parsed": parsed,
            }},
        }],
    }


def test_fixed_mix_and_manifest_are_exactly_six_pairs_with_new_unique_runs():
    cases = _cases()
    validate_cases(cases)
    manifest = build_manifest(cases, seed=20261016)
    assert manifest["planned_units"] == 12
    assert manifest["provider_sends"] == 0
    assert all(row["status"] == "not_started" for row in manifest["entries"])
    assert len({row["run_id"] for row in manifest["entries"]}) == 12
    assert {row["arm"] for row in manifest["entries"]} == set(ARMS)
    assert manifest["arm_order_pair_counts"] == {"basic_tool_loop": 3, "governed_workflow": 3}
    assert all(manifest["entries"][i]["pair_id"] == manifest["entries"][i + 1]["pair_id"]
               for i in range(0, 12, 2))


def test_g_li03_original_question_capability_stop_is_structured_without_sql(
    monkeypatch, isolated_trend_db, tmp_path,
):
    """Record the current G business-contract miss as a score failure, not xfail/pass."""
    from test_registered_numeric_decomposition_intent import _install_offline_workflow
    from evaluation.resume_bg_protocol import score_bg_run

    case_path = Path("docs/testing/resume_bg_ab_dev_20261016/cases_6.json")
    cases = json.loads(case_path.read_text(encoding="utf-8"))["cases"]
    case = next(row for row in cases if row["case_id"] == "LI03")
    invoke, trace, statements, _model_calls, external_attempts = _install_offline_workflow(
        monkeypatch, isolated_trend_db, "order_dimension",
    )
    statement_count_before = len(statements)
    response = invoke(case["user_question"])
    new_statements = statements[statement_count_before:]
    investigation = trace.get("investigation_state") or {}
    stop_reason = investigation.get("stop_reason") or {}
    decisions = trace.get("decision_records") or []
    tools = trace.get("tool_calls") or []
    planning = trace.get("metric_planning_decision") or {}
    batch_plan = trace.get("metric_query_batch_plan") or {}
    operation = (trace.get("context_compilation") or {}).get("operation_intent") or {}
    observed = {
        "task_type": trace.get("task_type"),
        "operation_intent": operation.get("kind"),
        "planning_status": planning.get("status"),
        "planning_intent": planning.get("operation_intent"),
        "confirmed_metrics": planning.get("confirmed_metrics"),
        "batch_query_type": (batch_plan.get("decision") or {}).get("query_type"),
        "batch_reason": (batch_plan.get("decision") or {}).get("reason"),
        "terminal_code": stop_reason.get("code") if isinstance(stop_reason, dict) else stop_reason,
        "presentation_status": (response.get("presentation") or {}).get("status"),
        "presentation_code": (response.get("presentation") or {}).get("limitation_code"),
        "errors": [row.get("code") for row in (trace.get("errors") or []) if isinstance(row, dict)],
        "decision_records": [{"dimension": row.get("dimension"), "code": row.get("limitation_code"), "sql_executed": row.get("sql_executed")} for row in decisions],
        "tool_calls": [{"tool": row.get("tool_name"), "status": row.get("call_status"), "limitation_code": row.get("limitation_code")} for row in tools],
        "new_sql_count": len(new_statements),
    }
    artifact = {
        **observed,
        "case_id": case["case_id"],
        "question_sha256": __import__("hashlib").sha256(case["user_question"].encode("utf-8")).hexdigest(),
        "execution_mode": "normal run_chat_workflow with offline unavailable-model stubs and isolated PostgreSQL",
        "provider_calls": 0,
        "external_network_attempts": external_attempts,
    }
    (tmp_path / "li03_g_workflow_diagnostic.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    assert external_attempts == []
    assert trace["terminal_code"] == "CAPABILITY_GAP", observed
    assert observed["presentation_status"] == "unsupported", observed
    assert observed["errors"] == [], observed
    assert observed["new_sql_count"] == 0, observed
    assert decisions and decisions[-1]["limitation_code"] == "CAPABILITY_GAP", observed
    assert all(row.get("tool") == "metric_clarification" for row in observed["tool_calls"]), observed
    score = score_bg_run(
        case, arm="governed_workflow",
        trace={"state_snapshot": trace, "workflow_result": response},
        lifecycle_status="completed",
    )
    assert score["task_acceptance_pass"] is False
    assert score["correct_safe_stop"] is True
    assert score["terminal_code"] == "CAPABILITY_GAP"


@pytest.mark.parametrize("bad_arm", ["full", "critic_llm_review_off"])
def test_critic_ablation_labels_are_not_b_g_arms(bad_arm):
    case = _cases()[0]
    with pytest.raises(ValueError):
        score_bg_run(case, arm=bad_arm, trace={}, lifecycle_status="not_started")


def test_wrong_case_mix_fails_closed():
    cases = _cases()
    cases[0] = {**cases[0], "protocol_task_type": "dimension_statistical_contribution"}
    with pytest.raises(ValueError, match="Case mix mismatch"):
        validate_cases(cases)


def test_completed_status_alone_does_not_pass_numeric_case():
    case = {
        "case_id": "N1", "protocol_task_type": "registered_numeric_decomposition",
        "target_metric": "gmv", "scope": "platform",
        "comparison_period": {"baseline": "2020-01-01", "current": "2020-01-02"},
        "expected_result": {"baseline": 10, "current": 12, "delta": 2, "components": {"orders": 2}},
        "tolerance": {"absolute": 0.03}, "readiness": "eligible",
    }
    trace = {"workflow_state": {"analysis_presentation": {"status": "completed"}, "answer": "done"}}
    result = score_bg_run(case, arm="governed_workflow", trace=trace, lifecycle_status="completed",
                          human_review={"decision": "pass"})
    assert result["presentation_status"] == "completed"
    assert result["task_acceptance_pass"] is False
    assert result["effective_diagnostic_completed"] is False
    assert result["evidence_valid"] is False


def test_valid_numeric_result_passes_only_with_bound_evidence_and_blinded_text_review():
    case = {
        "case_id": "N1", "protocol_task_type": "registered_numeric_decomposition",
        "target_metric": "gmv", "scope": "platform",
        "comparison_period": {"baseline": "2020-01-01", "current": "2020-01-02"},
        "expected_result": {"baseline": 10, "current": 12, "delta": 2, "components": {"orders": 2}},
        "unit": "BRL", "output_units": {"metric_unit": "BRL", "change_unit": "BRL", "component_unit": "BRL"},
        "tolerance": {"absolute": 0.03}, "readiness": "eligible",
    }
    trace_state = {
        "scope_type": "platform", "answer": "GMV 从 10 BRL 增至 12 BRL，变化 2 BRL；这只是数值分解。",
        "analysis_presentation": {"status": "completed", "metric_id": "gmv", "baseline_date": "2020-01-01",
            "current_date": "2020-01-02", "metric_decomposition": {
                "baseline_value": 10, "current_value": 12, "total_delta": 2, "unit": "BRL",
                "change_unit": "BRL", "component_unit": "BRL", "display_components": {"orders": 2},
            }},
        "final": {"claims": [{"text": "The registered decomposition is numeric, not causal.", "evidence_ids": ["ev-1"]}]},
    }
    trace_state.update(_bound_public_evidence("ev-1", "gmv", start="2020-01-01", end="2020-01-02"))
    trace = {"workflow_state": trace_state}
    result = score_bg_run(case, arm="governed_workflow", trace=trace, lifecycle_status="completed",
                          human_review={"decision": "pass"})
    assert result["effective_diagnostic_completed"] is True
    assert result["correct_safe_stop"] is False
    assert result["task_acceptance_pass"] is True
    assert result["manual_review_required"] is False


def test_partial_keeps_valid_decomposition_but_is_not_complete_task_pass():
    case = {
        "case_id": "N1", "protocol_task_type": "registered_numeric_decomposition",
        "target_metric": "gmv", "scope": "platform",
        "comparison_period": {"baseline": "2020-01-01", "current": "2020-01-02"},
        "expected_result": {"baseline": 10, "current": 12, "delta": 2, "components": {"orders": 2}},
        "unit": "BRL", "output_units": {"metric_unit": "BRL", "change_unit": "BRL", "component_unit": "BRL"},
        "tolerance": {"absolute": 0.03}, "readiness": "eligible",
    }
    trace_state = {
        "scope_type": "platform", "answer": "GMV changed by 2 BRL; deeper inspection remains unresolved.",
        "analysis_presentation": {"status": "partial", "metric_id": "gmv", "baseline_date": "2020-01-01",
            "current_date": "2020-01-02", "metric_decomposition": {
                "baseline_value": 10, "current_value": 12, "total_delta": 2, "unit": "BRL",
                "change_unit": "BRL", "component_unit": "BRL", "display_components": {"orders": 2},
            }},
        "final": {"claims": [{"text": "The registered decomposition is numeric, not causal.", "evidence_ids": ["ev-1"]}]},
    }
    trace_state.update(_bound_public_evidence("ev-1", "gmv", start="2020-01-01", end="2020-01-02"))
    trace = {"workflow_state": trace_state}
    result = score_bg_run(case, arm="governed_workflow", trace=trace, lifecycle_status="completed",
                          human_review={"decision": "pass"})
    assert result["effective_diagnostic_completed"] is True
    assert result["task_acceptance_pass"] is False


def test_dimension_contribution_is_not_root_cause_and_units_are_separate():
    case = {
        "case_id": "D1", "protocol_task_type": "dimension_statistical_contribution",
        "target_metric": "freight_rate", "scope": "platform",
        "comparison_period": {"baseline": "2020-01-01", "current": "2020-01-02"},
        "expected_result": {"baseline_metric": 10, "current_metric": 12, "delta": 2,
                             "expected_contributions": {"A:within": 2}},
        "unit": "%", "output_units": {"metric_unit": "%", "change_unit": "percentage_point",
                                         "component_unit": "percentage_point"},
        "tolerance": {"absolute": 0.03}, "readiness": "eligible",
    }
    trace_state = {
        "scope_type": "platform", "answer": "The freight rate rose by 2 percentage points. This is descriptive contribution, not business cause.",
        "analysis_presentation": {"status": "completed", "metric_id": "freight_rate", "baseline_date": "2020-01-01",
            "current_date": "2020-01-02", "dimension_contribution": {
                "baseline_value": 10, "current_value": 12, "total_delta": 2,
                "metric_unit": "%", "change_unit": "percentage_point", "component_unit": "percentage_point",
                "contributions": {"A:within": 2},
            }},
        "final": {"claims": [{"text": "Statistical contribution only.", "evidence_ids": ["ev-1"]}]},
    }
    trace_state.update(_bound_public_evidence("ev-1", "freight_rate", start="2020-01-01", end="2020-01-02"))
    trace = {"workflow_state": trace_state}
    result = score_bg_run(case, arm="basic_tool_loop", trace=trace, lifecycle_status="completed",
                          human_review={"decision": "pass"})
    assert result["effective_diagnostic_completed"] is True
    assert result["task_acceptance_pass"] is True


def test_expected_safe_stop_is_not_effective_diagnosis_and_budget_stop_does_not_pass():
    case = {
        "case_id": "S1", "protocol_task_type": "expected_data_insufficiency",
        "target_metric": "gmv", "scope": "platform", "readiness": "eligible",
        "expected_terminal_code": "DATA_INSUFFICIENT",
    }
    trace_state = {
        "scope_type": "platform", "terminal_code": "DATA_INSUFFICIENT",
        "answer": "缺少趋势桶，无法验证。",
        "analysis_presentation": {"status": "partial", "metric_id": "gmv"},
        "final": {"claims": [{"text": "趋势数据不足", "evidence_ids": ["ev-missing"]}]},
    }
    trace_state.update(_bound_public_evidence("ev-missing", "gmv", status="not_found"))
    trace = {"workflow_state": trace_state}
    result = score_bg_run(case, arm="basic_tool_loop", trace=trace, lifecycle_status="completed",
                          human_review={"decision": "pass"})
    assert result["correct_safe_stop"] is True
    assert result["effective_diagnostic_completed"] is False
    assert result["task_acceptance_pass"] is True
    trace["workflow_state"]["terminal_code"] = "BUDGET_EXHAUSTED"
    result = score_bg_run(case, arm="basic_tool_loop", trace=trace, lifecycle_status="completed",
                          human_review={"decision": "pass"})
    assert result["correct_safe_stop"] is False
    assert result["task_acceptance_pass"] is False


def test_invalid_or_unbound_evidence_and_execution_error_do_not_pass():
    case = {
        "case_id": "S1", "protocol_task_type": "expected_capability_limit",
        "target_metric": "order_count", "scope": "platform", "readiness": "eligible",
        "expected_terminal_code": "CAPABILITY_GAP",
    }
    trace = {"workflow_state": {
        "scope_type": "platform", "terminal_code": "CAPABILITY_GAP", "answer": "未注册该维度。",
        "analysis_presentation": {"status": "partial", "metric_id": "order_count"},
        "evidence_bundles": [{"evidence_id": "ev-other"}],
        "final": {"claims": [{"text": "能力限制", "evidence_ids": ["missing"]}]},
    }}
    result = score_bg_run(case, arm="governed_workflow", trace=trace, lifecycle_status="completed",
                          human_review={"decision": "pass"})
    assert result["evidence_valid"] is False
    assert result["task_acceptance_pass"] is False
    trace["workflow_state"]["terminal_code"] = "EXECUTION_FAILED"
    trace["workflow_state"]["errors"] = [{"code": "provider_timeout"}]
    result = score_bg_run(case, arm="governed_workflow", trace=trace, lifecycle_status="completed",
                          human_review={"decision": "pass"})
    assert result["correct_safe_stop"] is False
    assert result["execution_error_free"] is False


def test_registered_capability_refusal_can_be_a_traceable_safe_stop_without_diagnostic_completion():
    case = {
        "case_id": "L1", "protocol_task_type": "expected_capability_limit",
        "target_metric": "order_count", "scope": "platform", "readiness": "eligible",
        "expected_terminal_code": "CAPABILITY_GAP",
    }
    trace_state = {
        "scope_type": "platform", "terminal_code": "CAPABILITY_GAP", "answer": "该维度未注册，未执行替代维度查询。",
        "analysis_presentation": {"status": "unsupported", "metric_id": "order_count"},
        "decision_records": [{
            "limitation_code": "CAPABILITY_GAP", "metric_id": "order_count",
            "scope": "platform", "authorization_checked": True,
            "registry_capability_checked": True, "sql_executed": False,
        }],
        "final": {"claims": [{"text": "The requested capability is not registered.", "evidence_ids": ["ev-capability"]}]},
        "tool_calls": [{"request_id": "req-capability", "call_status": "succeeded"}],
        "evidence_bundles": [{
            "evidence_id": "ev-capability", "request_id": "req-capability",
            "content": {"request_id": "req-capability", "result": {
                "status": "unsupported", "parsed": {"metric_name": "order_count", "authorized_metric": {"metric_id": "order_count"}},
            }},
        }],
    }
    result = score_bg_run(case, arm="governed_workflow", trace={"workflow_state": trace_state},
                          lifecycle_status="completed", human_review={"decision": "pass"})
    assert result["status_consistent"] is True
    assert result["correct_safe_stop"] is True
    assert result["effective_diagnostic_completed"] is False
    assert result["task_acceptance_pass"] is True


def test_missing_data_safe_stop_requires_current_authorized_traceable_result():
    case = {
        "case_id": "S1", "protocol_task_type": "expected_data_insufficiency",
        "target_metric": "gmv", "scope": "platform", "readiness": "eligible",
        "expected_terminal_code": "DATA_INSUFFICIENT",
        "comparison_period": {"current": "2020-01-02"},
    }
    trace_state = {
        "scope_type": "platform", "terminal_code": "DATA_INSUFFICIENT",
        "answer": "该日期的GMV缺少有效观测，不能判断趋势。",
        "analysis_presentation": {"status": "partial", "metric_id": "gmv"},
        "final": {"claims": [{"text": "缺少该日数据", "evidence_ids": ["ev-gap"]}]},
    }
    trace_state.update(_bound_public_evidence(
        "ev-gap", "gmv", status="not_found", start="2020-01-02", end="2020-01-02",
    ))
    result = score_bg_run(
        case, arm="basic_tool_loop", trace={"workflow_state": trace_state},
        lifecycle_status="completed", human_review={"decision": "pass"},
    )
    assert result["correct_safe_stop"] is True
    assert result["effective_diagnostic_completed"] is False
    assert result["task_acceptance_pass"] is True

    parsed = trace_state["evidence_bundles"][0]["content"]["result"]["parsed"]
    parsed["start_date"] = parsed["end_date"] = "2020-01-01"
    result = score_bg_run(
        case, arm="basic_tool_loop", trace={"workflow_state": trace_state},
        lifecycle_status="completed", human_review={"decision": "pass"},
    )
    assert result["correct_safe_stop"] is False
    assert result["task_acceptance_pass"] is False


def test_governed_task_requires_all_applicable_critic_verdicts_when_preregistered():
    case = {
        "case_id": "S1", "protocol_task_type": "expected_data_insufficiency",
        "target_metric": "gmv", "scope": "platform", "readiness": "eligible",
        "expected_terminal_code": "DATA_INSUFFICIENT",
        "critic_review_required_for_governed_arm": True,
    }
    state = {
        "scope_type": "platform", "terminal_code": "DATA_INSUFFICIENT", "answer": "数据不足。",
        "analysis_presentation": {"status": "partial", "metric_id": "gmv"},
        "final": {"claims": [{"text": "数据不足", "evidence_ids": ["ev-gap"]}]},
        "critic_result": {"llm_review": {
            "groundedness": {"verdict": "passed"},
            "completeness": {"verdict": "not_completed"},
            "consistency": {"verdict": "not_completed"},
        }},
    }
    state.update(_bound_public_evidence("ev-gap", "gmv", status="not_found"))
    result = score_bg_run(
        case, arm="governed_workflow", trace={"workflow_state": state},
        lifecycle_status="completed", human_review={"decision": "pass"},
    )
    assert result["correct_safe_stop"] is True
    assert result["critic_review_applicable"] is True
    assert result["critic_review_complete"] is False
    assert result["task_acceptance_pass"] is False


def test_saved_candidate_manifest_has_12_unstarted_units_and_expected_blockers():
    root = Path(__file__).resolve().parents[1]
    folder = root / "docs/testing/resume_bg_ab_dev_20261016"
    cases = json.loads((folder / "cases_6.json").read_text(encoding="utf-8"))["cases"]
    manifest = json.loads((folder / "run_manifest_12.json").read_text(encoding="utf-8"))
    validate_cases(cases)
    assert manifest["planned_units"] == 12 and manifest["provider_sends"] == 0
    assert all(row["status"] == "not_started" for row in manifest["entries"])
    assert {case["case_id"] for case in cases if case["readiness"] == "blocked"} == {"LI03"}
    assert next(case for case in cases if case["case_id"] == "S01-core-incomplete-107")["readiness"] == "eligible"

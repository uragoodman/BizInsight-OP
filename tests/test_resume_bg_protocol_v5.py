import copy
import json
import socket
from pathlib import Path

import pytest

from evaluation.resume_bg_conditions import (
    validate_capability_refusal, validate_case_pre_run_condition, validate_missing_tail,
)
from evaluation.resume_bg_protocol import score_bg_run_v5


@pytest.fixture(autouse=True)
def block_network_and_database(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("v5 scorer and condition tests must remain offline")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)


def _cases():
    path = Path("docs/testing/resume_bg_ab_dev_20261016/cases_6.json")
    return {row["case_id"]: row for row in json.loads(path.read_text(encoding="utf-8"))["cases"]}


def _record(case_id, arm):
    for path in Path("docs/testing/resume_bg_ab_dev_20261016/results_v1/traces").glob("*.run_record.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["case_id"] == case_id and row["arm"] == arm:
            return row
    raise AssertionError(f"saved development record absent: {case_id}/{arm}")


def test_v5_preserves_v3_v4_and_maps_symmetric_ratio_roles_from_registry():
    case = _cases()["NC03"]
    raw = _record("NC03", "basic_tool_loop")
    before = copy.deepcopy(raw)
    result = score_bg_run_v5(case, arm=raw["arm"], trace=raw["trace"], lifecycle_status="completed")
    assert result["scorer_version"] == "resume-bg-score-v5"
    assert result["parent_scorer_version"] == "resume-bg-score-v4"
    assert result["factor_mapping"]["factor_aliases_from_registry"] == {
        "numerator_effect": "gmv", "denominator_effect": "order_count",
    }
    assert all(result["numeric_checks"].values())
    assert result["effective_diagnostic_completed"]
    assert result["task_acceptance_status"] == "pending_manual_review"
    assert not result["task_acceptance_pass"]
    assert raw == before


def test_initial_query_and_investigation_evidence_are_independently_merged():
    case = _cases()["NC01"]
    raw = _record("NC01", "governed_workflow")
    result = score_bg_run_v5(case, arm=raw["arm"], trace=raw["trace"], lifecycle_status="completed")
    binding = result["evidence_binding"]
    assert binding["valid_query_evidence_ids"]
    assert binding["valid_investigation_evidence_ids"]
    assert binding["claims_bound"]
    assert not binding["conflicted_evidence_ids"]
    assert result["effective_diagnostic_completed"]
    assert result["task_acceptance_status"] == "pending_manual_review"


def test_conflicting_duplicate_evidence_id_is_excluded():
    case = _cases()["NC01"]
    raw = _record("NC01", "governed_workflow")
    trace = copy.deepcopy(raw["trace"])
    duplicate = next(row for row in trace["state_snapshot"]["evidence_bundles"]
                     if row["evidence_id"] == "ev_5a30f9ba2d8fa90d")
    duplicate["content"]["result"]["total_delta"] += 1
    result = score_bg_run_v5(case, arm=raw["arm"], trace=trace, lifecycle_status="completed")
    assert "ev_5a30f9ba2d8fa90d" in result["evidence_binding"]["conflicted_evidence_ids"]
    assert not result["effective_diagnostic_completed"]
    assert "evidence_invalid_or_unbound" in result["automated_failure_reasons"]


@pytest.mark.parametrize("tamper", ["request", "authorization", "date", "call_link", "scope"])
def test_public_evidence_wrong_binding_is_not_accepted(tamper):
    case = _cases()["NC03"]
    raw = _record("NC03", "basic_tool_loop")
    trace = copy.deepcopy(raw["trace"])
    state = trace["state_snapshot"]
    bundle = state["evidence_bundles"][0]
    parsed = bundle["content"]["result"]["parsed"]
    if tamper == "request":
        bundle["request_id"] = "unrelated-request"
    elif tamper == "authorization":
        parsed["authorized_metric"]["metric_id"] = "gmv"
    elif tamper == "date":
        bundle["content"]["source_queries"][0]["envelope"]["result"]["parsed"]["start_date"] = "2017-02-02"
    elif tamper == "call_link":
        state["tool_calls"][0]["request_id"] = "unrelated-request"
    elif tamper == "scope":
        state["scope_type"] = "seller"
    result = score_bg_run_v5(case, arm=raw["arm"], trace=trace, lifecycle_status="completed")
    assert not result["effective_diagnostic_completed"]
    assert not result["evidence_valid"]
    assert "evidence_invalid_or_unbound" in result["automated_failure_reasons"]


def test_automation_failure_is_not_masked_by_human_pending():
    case = _cases()["NC03"]
    raw = _record("NC03", "basic_tool_loop")
    trace = copy.deepcopy(raw["trace"])
    trace["workflow_result"]["presentation"]["delta"] += 5
    result = score_bg_run_v5(case, arm=raw["arm"], trace=trace, lifecycle_status="completed")
    assert result["human_review_status"] == "pending"
    assert result["task_acceptance_status"] == "failed"
    assert "numeric_checks.delta" in result["automated_failure_reasons"]


def test_old_special_cases_keep_condition_failure_or_unverified_separate():
    cases = _cases()
    s01 = _record("S01-core-incomplete-107", "basic_tool_loop")
    s01_score = score_bg_run_v5(cases["S01-core-incomplete-107"], arm=s01["arm"], trace=s01["trace"], lifecycle_status="completed")
    assert s01_score["experiment_condition_status"] == "experiment_condition_invalid"
    assert s01_score["task_acceptance_status"] == "failed"
    li03 = _record("LI03", "governed_workflow")
    li03_score = score_bg_run_v5(cases["LI03"], arm=li03["arm"], trace=li03["trace"], lifecycle_status="failed")
    assert li03_score["experiment_condition_status"] == "experiment_condition_invalid"
    assert li03_score["task_acceptance_status"] == "failed"
    assert "execution_error_free" in li03_score["automated_failure_reasons"]


def test_missing_tail_condition_requires_exactly_two_missing_tail_buckets_and_controls():
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    result = validate_missing_tail(
        requested_dates=dates, missing_dates=dates[-2:],
        points=[{"date": dates[0], "value": 7}, {"date": dates[1], "value": 9},
                {"date": dates[2], "value": None}, {"date": dates[3], "value": None}],
    )
    assert result["status"] == "ready" and result["missing_not_zero_filled"]
    assert validate_missing_tail(
        requested_dates=dates, missing_dates=dates[-2:],
        points=[{"date": dates[0], "value": 7}, {"date": dates[1], "value": 9},
                {"date": dates[2], "value": 0}, {"date": dates[3], "value": None}],
    )["reason"] == "missing_bucket_filled_as_zero"
    assert validate_missing_tail(
        requested_dates=dates, missing_dates=dates[-2:],
        points=[{"date": dates[0], "value": 7}, {"date": dates[2], "value": None}, {"date": dates[3], "value": None}],
    )["control_buckets_present"] is False
    assert validate_missing_tail(
        requested_dates=dates, missing_dates=[dates[1], dates[3]],
        points=[{"date": dates[0], "value": 7}, {"date": dates[1], "value": None},
                {"date": dates[2], "value": 9}, {"date": dates[3], "value": None}],
    )["status"] == "experiment_condition_invalid"


def test_capability_condition_separates_unsupported_from_unauthorized():
    common = {"metric_id": "order_count", "scope": "platform", "operation": "metric_contribution", "dimension": "payment_type"}
    ready = validate_capability_refusal(
        **common, capability_snapshot={**common, "supported": False},
        authorization_snapshot={**common, "authorized": True},
    )
    assert ready["status"] == "ready"
    unauthorized = validate_capability_refusal(
        **common, capability_snapshot={**common, "supported": True},
        authorization_snapshot={**common, "authorized": False},
    )
    assert unauthorized["reason"] == "authorization_denial_is_not_capability_gap"
    supported = validate_capability_refusal(
        **common, capability_snapshot={**common, "supported": True},
        authorization_snapshot={**common, "authorized": True},
    )
    assert supported["reason"] == "requested_capability_is_supported_and_authorized"
    assert validate_capability_refusal(
        **common, capability_snapshot=None, authorization_snapshot={**common, "authorized": True},
    )["status"] == "experiment_condition_unverified"


def test_case_pre_run_gate_fails_closed_without_structured_observation():
    case = {"protocol_task_type": "expected_data_insufficiency", "case_id": "S01"}
    assert validate_case_pre_run_condition(case, observation=None)["status"] == "experiment_condition_unverified"
    case = {"protocol_task_type": "expected_capability_limit", "case_id": "LI03", "readiness": "blocked"}
    assert validate_case_pre_run_condition(case, observation=None)["status"] == "experiment_condition_unverified"
    case = {"protocol_task_type": "registered_numeric_decomposition"}
    assert validate_case_pre_run_condition(case, observation=None)["status"] == "experiment_condition_not_applicable"


def test_expected_safe_stop_can_be_pending_without_numeric_diagnosis():
    case = {
        "case_id": "SAFE-DATA",
        "protocol_task_type": "expected_data_insufficiency",
        "target_metric": "gmv", "scope": "platform",
        "comparison_period": {"current": "2024-05-04"},
        "expected_terminal_code": "DATA_INSUFFICIENT",
        "expected_limitation": "missing observation is not zero",
        "expected_result": {}, "unit": "BRL", "output_units": {"metric_unit": "BRL"},
        "readiness": "eligible", "tolerance": {"absolute": 0.03},
    }
    envelope = {
        "request_id": "r1", "tool_name": "registry_trend", "evidence_id": "e1", "status": "ok",
        "arguments": {"metric_name": "gmv", "metric_date": "2024-05-04"},
    }
    bundle = {
        "evidence_id": "e1", "request_id": "r1", "source_ref": "registry_trend",
        "content": {"request_id": "r1", "result": {
            "status": "not_found", "parsed": {"metric_name": "gmv", "metric_date": "2024-05-04",
                                                     "authorized_metric": {"metric_id": "gmv"}},
            "trend_result": {"points": [{"bucket": "2024-05-04", "value": None}]},
        }},
    }
    trace = {"state_snapshot": {
        "scope_type": "platform", "authorization_context": {"scope_type": "platform"},
        "answer": "The requested bucket has no data; I cannot calculate the comparison.",
        "terminal_code": "DATA_INSUFFICIENT", "evidence_bundles": [bundle],
        "tool_calls": [envelope], "claims": [{"evidence_ids": ["e1"]}],
        "decision_records": [{"limitation_code": "DATA_INSUFFICIENT", "metric_id": "gmv",
                              "scope": "platform", "authorization_checked": True,
                              "sql_executed": True, "evidence_id": "e1"}],
        "final": {"action": "FINAL", "answer": "The requested bucket has no data; I cannot calculate the comparison.",
                  "claims": [{"evidence_ids": ["e1"]}],
                  "presentation": {"metric_id": "gmv", "scope": "platform", "current_date": "2024-05-04",
                                   "status": "partial"}},
    }}
    result = score_bg_run_v5(case, arm="basic_tool_loop", trace=trace, lifecycle_status="completed",
                             experiment_condition_status="ready")
    assert result["correct_safe_stop"]
    assert not result["effective_diagnostic_completed"]
    assert result["task_acceptance_status"] == "pending_manual_review"
    assert not result["task_acceptance_pass"]
    assert "effective_diagnostic_not_completed" not in result["automated_failure_reasons"]

"""Per-run root-cause scoring over the persisted offline workflow traces."""

import json
import copy
from pathlib import Path

from evaluation.non_root_scorer import summarize_diagnostic_runs
from evaluation.scorers import DIAGNOSTIC_RUN_SCORER_VERSION, collect_anomaly_metrics, score_diagnostic_run


TRACE_DIR = Path(__file__).parents[1] / "docs/testing/diagnostic_presentation_status_verify_20261011/traces"


def _saved(case_id: str) -> dict:
    return json.loads((TRACE_DIR / f"{case_id}.json").read_text(encoding="utf-8"))


def _score(case_id: str, expected: str, *, stop_code: str | None = None) -> dict:
    return score_diagnostic_run(
        _saved(case_id), case_id=case_id, arm="offline-deterministic", replicate=1,
        expected_outcome=expected, expected_stop_code=stop_code,
    )


def test_saved_conflicts_are_accepted_safe_stops_not_diagnoses() -> None:
    for case_id in ("aov_conflict", "freight_conflict"):
        row = _score(case_id, "safe_stop", stop_code="INVALID_DATA")
        assert row["scorer_version"] == DIAGNOSTIC_RUN_SCORER_VERSION
        assert row["task_acceptance_pass"] is True
        assert row["correct_safe_stop"] is True
        assert row["effective_diagnosis_completed"] is False
        assert row["terminal_presentation_consistent"] is True
        assert row["conclusion_evidence_valid"] is True
        assert row["raw_metrics"]["investigation_stop_code"] == "INVALID_DATA"


def test_conflict_requires_consistent_status_and_all_in_scope_input_evidence() -> None:
    bad_status = _saved("aov_conflict")
    bad_status["workflow_state"]["analysis_presentation"]["status"] = "completed"
    row = score_diagnostic_run(
        bad_status, case_id="aov-conflict-wrong-status", arm="offline-deterministic", replicate=1,
        expected_outcome="safe_stop", expected_stop_code="INVALID_DATA",
    )
    assert row["terminal_presentation_consistent"] is False
    assert row["correct_safe_stop"] is False
    assert row["task_acceptance_pass"] is False

    missing_evidence = _saved("freight_conflict")
    failure = missing_evidence["workflow_state"]["diagnostic_data_reconciliation_failures"][0]
    removed_id = failure["input_evidence_ids"][0]
    missing_evidence["workflow_state"]["evidence_bundles"] = [
        item for item in missing_evidence["workflow_state"]["evidence_bundles"]
        if item.get("evidence_id") != removed_id
    ]
    row = score_diagnostic_run(
        missing_evidence, case_id="freight-conflict-missing-evidence", arm="offline-deterministic", replicate=1,
        expected_outcome="safe_stop", expected_stop_code="INVALID_DATA",
    )
    assert row["conclusion_evidence_valid"] is False
    assert row["correct_safe_stop"] is False
    assert row["task_acceptance_pass"] is False


def test_saved_ratio_runs_count_as_effective_numeric_decomposition() -> None:
    for case_id in ("aov_ratio", "freight_ratio"):
        row = _score(case_id, "numeric_decomposition")
        assert row["task_acceptance_pass"] is True
        assert row["effective_diagnosis_completed"] is True
        assert row["effective_diagnosis_type"] == "numeric_decomposition"
        assert row["correct_safe_stop"] is False
        assert row["terminal_presentation_consistent"] is True
        assert row["conclusion_evidence_valid"] is True


def test_saved_execution_error_is_not_a_safe_stop() -> None:
    row = _score("aov_execution_exception", "execution_failure")
    assert row["task_acceptance_pass"] is True
    assert row["terminal_code"] == "EXECUTION_FAILED"
    assert row["presentation_status"] == "failed"
    assert row["terminal_presentation_consistent"] is True
    assert row["correct_safe_stop"] is False
    assert row["effective_diagnosis_completed"] is False


def test_data_insufficient_stop_needs_scoped_executed_missing_trend() -> None:
    evidence_id = "ev-missing"
    state = {
        "diagnostic_target_metric": "gmv",
        "analysis_result": {"abstain": True, "evidence_ids": []},
        "analysis_presentation": {"status": "partial"},
        "finish_decision": {"action_type": "ABSTAIN", "supporting_evidence_ids": []},
        "tool_result": {"result": {"parsed": {
            "metric_name": "gmv", "start_date": "2024-01-01", "end_date": "2024-01-02",
            "authorized_metric": {"scope_fingerprint": "scope-current"},
        }}},
        "investigation_state": {
            "target": {
                "metric_id": "gmv", "scope_fingerprint": "scope-current",
                "baseline_start": "2024-01-01", "baseline_end": "2024-01-01",
                "current_start": "2024-01-02", "current_end": "2024-01-02",
            },
            "stop_reason": {"code": "DATA_INSUFFICIENT"},
        },
        "tool_calls": [{
            "tool_name": "registry_trend", "arguments": {"metric_name": "gmv"},
            "status": "ok", "evidence_id": evidence_id,
        }],
        "evidence_bundles": [{
            "evidence_id": evidence_id, "trusted": True,
            "validation_status": "accepted_by_investigation_engine",
            "content": {"result": {
                "status": "ok",
                "parsed": {
                    "metric_name": "gmv", "start_date": "2024-01-01", "end_date": "2024-01-02",
                    "authorized_metric": {"scope_fingerprint": "scope-current"},
                },
                "trend_result": {"metric_name": "gmv", "points": [{"bucket": "2024-01-01", "value": None}]},
            }},
        }],
    }
    accepted = score_diagnostic_run(
        state, case_id="missing-trend", arm="offline", replicate=1,
        expected_outcome="safe_stop", expected_stop_code="DATA_INSUFFICIENT",
    )
    assert accepted["task_acceptance_pass"] is True
    assert accepted["correct_safe_stop"] is True
    assert accepted["effective_diagnosis_completed"] is False
    assert accepted["conclusion_evidence_valid"] is True

    state["evidence_bundles"][0]["content"]["result"]["parsed"]["authorized_metric"]["scope_fingerprint"] = "other-scope"
    rejected = score_diagnostic_run(
        state, case_id="missing-trend-wrong-scope", arm="offline", replicate=1,
        expected_outcome="safe_stop", expected_stop_code="DATA_INSUFFICIENT",
    )
    assert rejected["task_acceptance_pass"] is False
    assert rejected["conclusion_evidence_valid"] is False


def test_data_insufficient_accepts_trusted_typed_evidence_without_duplicate_bundle_status() -> None:
    state = copy.deepcopy(_saved("aov_ratio")["workflow_state"])
    eid = state["finish_decision"]["evidence_ids"][0]
    state["investigation_state"]["stop_reason"] = {"code": "DATA_INSUFFICIENT"}
    state["analysis_result"]["abstain"] = True
    state["finish_decision"] = {"action_type": "ABSTAIN", "supporting_evidence_ids": [eid]}
    state["analysis_result"]["evidence_ids"] = [eid]
    state["analysis_presentation"] = {"status": "partial"}
    for bundle in state.get("evidence_bundles", []):
        if bundle.get("evidence_id") == eid:
            bundle["trusted"] = True
            bundle["validation_status"] = None
    row = score_diagnostic_run(
        state, case_id="typed-evidence-safe-stop", arm="offline", replicate=1,
        expected_outcome="safe_stop", expected_stop_code="DATA_INSUFFICIENT",
    )
    assert row["task_acceptance_pass"] is True
    assert row["correct_safe_stop"] is True
    assert row["effective_diagnosis_completed"] is False
    assert row["conclusion_evidence_valid"] is True
    wrong_expectation = score_diagnostic_run(
        state, case_id="typed-evidence-safe-stop-when-diagnosis-expected", arm="offline", replicate=1,
        expected_outcome="numeric_decomposition",
    )
    assert wrong_expectation["safe_stop_behavior_observed"] is True
    assert wrong_expectation["correct_safe_stop"] is False
    assert wrong_expectation["task_acceptance_pass"] is False


def test_saved_zero_denominator_diagnostic_attempt_is_not_scored_from_partial_status() -> None:
    for case_id in ("aov_zero_denominator", "freight_zero_denominator"):
        row = _score(case_id, "safe_stop")
        assert row["task_acceptance_pass"] is False
        assert row["correct_safe_stop"] is False
        assert row["terminal_code"] is None
        assert row["terminal_presentation_consistent"] is None
        # The saved direct query control is not an end-to-end diagnostic terminal record.
        direct = _saved(case_id)["zero_value_trace"]
        assert direct["workflow_result"]["presentation"]["status"] == "partial"
        assert direct["workflow_state"]["tool_result"]["result"]["status"] == "not_found"


def test_token_usage_unknown_stays_null_but_no_request_is_zero() -> None:
    saved_state = _saved("aov_ratio")["workflow_state"]
    no_request = collect_anomaly_metrics(saved_state)
    assert no_request["token_request_status"] == "no_model_request"
    assert no_request["total_tokens"] == 0

    unknown_state = dict(saved_state)
    unknown_state["llm_token_summary"] = {
        "request_status": "requested", "sdk_attempt_count": 1,
        "usage_complete": False, "known_tokens_subtotal": 17,
        "total_tokens": None,
    }
    unknown = collect_anomaly_metrics(unknown_state)
    assert unknown["total_tokens"] is None
    assert unknown["known_tokens_subtotal"] == 17


def test_aggregate_keeps_explicit_denominators_and_unknown_token_count() -> None:
    conflict = _score("aov_conflict", "safe_stop", stop_code="INVALID_DATA")
    zero = _score("aov_zero_denominator", "safe_stop")
    unknown_trace = _saved("aov_ratio")
    unknown_trace["workflow_state"]["llm_token_summary"] = {
        "request_status": "requested", "sdk_attempt_count": 1,
        "usage_complete": False, "known_tokens_subtotal": 17, "total_tokens": None,
    }
    unknown = score_diagnostic_run(
        unknown_trace, case_id="ratio-unknown-usage", arm="offline-deterministic", replicate=2,
        expected_outcome="numeric_decomposition",
    )
    summary = summarize_diagnostic_runs([conflict, zero, unknown])
    assert summary["run_count"] == 3
    assert summary["metrics"]["task_acceptance_pass"] == {
        "passed": 2, "denominator": 3, "unknown": 0, "rate": 0.6667,
    }
    assert summary["metrics"]["terminal_presentation_consistent"]["unknown"] == 1
    assert summary["tokens"]["unknown_usage_runs"] == 1
    assert summary["tokens"]["complete_total"] is None


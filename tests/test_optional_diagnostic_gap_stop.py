"""Regression for candidate-local insufficiency after a validated decomposition."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from app.graph.nodes.diagnostic import (
    _optional_data_gap_is_scoped_to_valid_decomposition,
    _record_candidate_data_gap,
    diagnostic_decide,
)
from app.investigation.candidates import generate_candidate_actions
from app.investigation.contracts import (
    ActionResult, AnalysisTarget, AttributionEvidence, DataValidationResult,
    InvestigationState, UnresolvedBranch, WorkflowSelection,
)
from app.investigation.engine import run_investigation


ROOT = Path(__file__).resolve().parents[1]
AOV_OFF_TRACE = (
    ROOT / "docs/testing/provider_aov_token_stop_rerun_20261012/traces/"
    "provider-trial-aov_ratio-critic_llm_review_off-c5179b4ee02a.json"
)
AOV_CONFLICT_RESULT = ROOT / "docs/testing/non_gmv_ratio_conflict_rerun_20261011/aov_conflict.json"


def _aov_off_state() -> dict:
    return json.loads(AOV_OFF_TRACE.read_text(encoding="utf-8"))["state_snapshot"]


def _aov_failed_candidate(state: dict):
    investigation = InvestigationState.model_validate(state["investigation_state"])
    failed = investigation.action_results[-1]
    candidate = next(item for item in investigation.candidate_actions if item.candidate_id == failed.candidate_id)
    return investigation, failed, candidate


def test_optional_seven_bucket_failure_is_local_and_preserves_aov_decomposition():
    state = _aov_off_state()
    investigation, failed, candidate = _aov_failed_candidate(state)

    assert failed.action_type == "CHECK_ANOMALY"
    assert failed.metric_id == "order_count"
    assert failed.failure_code == "DATA_INSUFFICIENT"
    assert len(next(
        item for item in investigation.evidence
        if item.evidence_id == "ev_639399c82810d225"
    ).values["result"]["points"]) == 7
    eligible, decomposition_id = _optional_data_gap_is_scoped_to_valid_decomposition(state, failed)
    assert eligible is True
    assert decomposition_id == "ev_3d1455e29f398e78"

    # The archived run used its last investigation action. Keep the local
    # failure, but preserve the independent budget stop rather than promoting
    # the check's 7-vs-8 gap to a global DATA_INSUFFICIENT stop.
    state["diagnostic_candidate_data_gaps"] = []
    _record_candidate_data_gap(
        state, investigation, failed, fallback_candidate=candidate,
        failure_detail=failed.failure_detail,
    )
    state["investigation_state"] = investigation.model_dump(mode="json")

    assert investigation.action_results[-1].failure_code == "DATA_INSUFFICIENT"
    assert investigation.action_results[-1].evidence_ids == []
    assert investigation.stop_reason.code == "BUDGET_EXHAUSTED"
    assert state["diagnostic_candidate_data_gaps"][0]["failure_scope"] == "optional_deep_check"
    assert state["diagnostic_candidate_data_gaps"][0]["related_decomposition_evidence_id"] == decomposition_id
    assert state["diagnostic_step_state"]["target_trend"] == "completed"
    assert state["diagnostic_step_state"]["related_metrics"] == "completed"
    assert "ev_3d1455e29f398e78" in {
        item.evidence_id for item in investigation.evidence
    }

    prior_tool_count = len(state.get("tool_calls", []))
    prior_action_count = len(investigation.action_results)
    terminal = diagnostic_decide(state)
    terminal_investigation = InvestigationState.model_validate(terminal["investigation_state"])
    assert terminal["diagnostic_action"]["action_type"] == "ABSTAIN"
    assert terminal_investigation.stop_reason.code == "BUDGET_EXHAUSTED"
    assert len(terminal.get("tool_calls", [])) == prior_tool_count
    assert len(terminal_investigation.action_results) == prior_action_count


def test_required_trend_gap_is_not_relaxed_even_with_a_valid_decomposition():
    state = _aov_off_state()
    investigation, failed, _candidate = _aov_failed_candidate(state)
    state["diagnostic_step_state"]["target_trend"] = "data_insufficient"
    state["diagnostic_required_action"] = True
    failed = failed.model_copy(update={
        "action_type": "ANALYZE_TREND", "metric_id": "avg_order_value",
        "branch_id": next(item.branch_id for item in investigation.branches if item.branch_type == "root_target"),
    })
    state["diagnostic_required_action"] = True
    failed = failed.model_copy(update={
        "action_type": "ANALYZE_TREND", "metric_id": "avg_order_value",
        "branch_id": next(item.branch_id for item in investigation.branches if item.branch_type == "root_target"),
    })

    eligible, evidence_id = _optional_data_gap_is_scoped_to_valid_decomposition(state, failed)

    assert eligible is False
    assert evidence_id is None
    assert investigation.stop_reason.code == "DATA_INSUFFICIENT"
    terminal = diagnostic_decide(state)
    assert terminal["diagnostic_action"]["action_type"] == "ABSTAIN"
    assert terminal["investigation_state"]["stop_reason"]["code"] == "DATA_INSUFFICIENT"


def test_input_conflict_keeps_invalid_data_stop():
    state = _aov_off_state()
    investigation, failed, _candidate = _aov_failed_candidate(state)
    conflict = json.loads(AOV_CONFLICT_RESULT.read_text(encoding="utf-8"))["workflow_state"]
    state["investigation_state"] = copy.deepcopy(conflict["investigation_state"])
    conflicted_investigation = InvestigationState.model_validate(state["investigation_state"])
    failed = failed.model_copy(update={"failure_code": "INVALID_DATA"})

    eligible, evidence_id = _optional_data_gap_is_scoped_to_valid_decomposition(state, failed)

    assert eligible is False
    assert evidence_id is None
    assert conflicted_investigation.stop_reason.code == "INVALID_DATA"
    terminal = diagnostic_decide(state)
    assert terminal["diagnostic_action"]["action_type"] == "ABSTAIN"
    assert terminal["investigation_state"]["stop_reason"]["code"] == "INVALID_DATA"


def test_optional_gap_without_valid_decomposition_remains_global_insufficiency():
    state = _aov_off_state()
    investigation, failed, _candidate = _aov_failed_candidate(state)
    investigation.evidence = [
        item for item in investigation.evidence
        if item.evidence_id != "ev_3d1455e29f398e78"
    ]
    investigation.action_results = [
        item for item in investigation.action_results
        if item.action_type != "DECOMPOSE_METRIC"
    ]
    investigation.branches = [
        item for item in investigation.branches
        if item.parent_evidence_id != "ev_3d1455e29f398e78"
    ]
    investigation.remaining_budget = 0
    state["investigation_state"] = investigation.model_dump(mode="json")
    state["diagnostic_candidate_data_gaps"] = []

    eligible, evidence_id = _optional_data_gap_is_scoped_to_valid_decomposition(state, failed)

    assert eligible is False
    assert evidence_id is None
    _record_candidate_data_gap(
        state, investigation, failed,
        fallback_candidate=next(
            item for item in investigation.candidate_actions
            if item.candidate_id == failed.candidate_id
        ),
        failure_detail=failed.failure_detail,
    )
    assert investigation.stop_reason.code == "DATA_INSUFFICIENT"


def test_no_alternative_candidate_terminates_without_repeating_the_failed_action():
    target = AnalysisTarget(
        metric_id="avg_order_value",
        current_start="2026-02-02", current_end="2026-02-02",
        baseline_start="2026-02-01", baseline_end="2026-02-01",
        canonical_scope={"scope_type": "platform", "tenant_id": "offline-test"},
        scope_fingerprint="offline-aov-scope", requested_analysis_type="diagnostic",
    )
    decomposition = AttributionEvidence(
        evidence_id="ev-offline-decomposition",
        evidence_type="metric_decomposition", metric_id="avg_order_value",
        values={
            "factor_metrics": ["gmv", "order_count"],
            "baseline_value": 10.0, "current_value": 20.0, "total_delta": 10.0,
        },
        source_action_id="prior-decomposition", source_ref="registered_decomposition",
        scope_fingerprint=target.scope_fingerprint,
        method_id="symmetric_ratio_v1", method_version="1.0",
        authorized_metric_provenance="evaluation_fixture",
        comparison={
            "current_period": {"start_date": "2026-02-02", "end_date": "2026-02-02"},
            "baseline_period": {"start_date": "2026-02-01", "end_date": "2026-02-01"},
        },
        data_quality={"status": "valid", "input_series_complete": True},
    )
    state = InvestigationState(
        target=target,
        remaining_budget=3,
        evidence=[decomposition],
        action_results=[ActionResult(
            action_id="action_1", candidate_id="prior-decomposition",
            action_type="DECOMPOSE_METRIC", metric_id="avg_order_value",
            branch_id="root-aov",
            method_id="symmetric_ratio_v1", status="completed",
            validation_result=DataValidationResult(status="valid"),
            evidence_ids=[decomposition.evidence_id],
        )],
        branches=[
            UnresolvedBranch(
                branch_id="root-aov", branch_type="root_target",
                metric_id="avg_order_value", depth=0,
            ),
            UnresolvedBranch(
                branch_id="branch-order-count", parent_branch_id="root-aov",
                parent_evidence_id=decomposition.evidence_id,
                branch_type="metric_effect", metric_id="order_count",
                dimension="__metric_factor__", dimension_value="order_count",
                depth=0,
            ),
        ],
    )
    generated = generate_candidate_actions(
        target,
        allowed_methods={"symmetric_ratio_v1"}, allowed_dimensions=(),
        branches=state.branches, focus_branch_id="branch-order-count",
        supported_action_types={"CHECK_ANOMALY"},
    )
    assert len(generated) == 1
    calls = []

    result = run_investigation(
        state,
        candidate_factory=lambda _current: generated,
        selector=lambda legal, _current: WorkflowSelection(candidate_id=legal[0].candidate_id),
        executor=lambda candidate, _current: (calls.append(candidate.candidate_id) or []),
        current_period={"start": "2026-02-02", "end": "2026-02-02"},
        baseline_period={"start": "2026-02-01", "end": "2026-02-01"},
        max_actions=3,
    )

    assert calls == [generated[0].candidate_id]
    assert len(result.action_results) == 2
    assert result.action_results[-1].failure_code == "DATA_INSUFFICIENT"
    assert result.stop_reason.code == "NO_HIGH_VALUE_ACTION"
    assert result.remaining_budget == 2
    assert result.evidence[0].evidence_id == decomposition.evidence_id

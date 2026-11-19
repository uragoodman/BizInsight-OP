"""Terminal-to-answer projection for validated numeric decompositions."""

from __future__ import annotations

import json
from pathlib import Path

from app.graph.nodes.answer import generate_answer
from app.graph.nodes.diagnostic import (
    _record_candidate_data_gap,
    diagnostic_finalize,
)
from app.investigation.contracts import InvestigationState
from app.schemas.analysis_presentation import AnalysisPresentation
from app.tenancy import TenantContext, activate_tenant_context


ROOT = Path(__file__).resolve().parents[1]
AOV_OFF_TRACE = (
    ROOT / "docs/testing/provider_aov_token_stop_rerun_20261012/traces/"
    "provider-trial-aov_ratio-critic_llm_review_off-c5179b4ee02a.json"
)
AOV_NORMAL_TRACE = ROOT / "docs/testing/diagnostic_presentation_status_verify_20261011/traces/aov_ratio.json"
AOV_DIMENSION_TRACE = ROOT / "docs/testing/non_gmv_diagnostic_20261011_after/order_dimension.json"
DECOMPOSITION_ID = "ev_3d1455e29f398e78"


def _saved_aov_state() -> dict:
    return json.loads(AOV_OFF_TRACE.read_text(encoding="utf-8"))["state_snapshot"]


def _saved_normal_aov_state() -> dict:
    return json.loads(AOV_NORMAL_TRACE.read_text(encoding="utf-8"))["workflow_state"]


def _partial_aov_state() -> dict:
    state = _saved_aov_state()
    # Re-apply the public stop classifier to the saved failed action, so the
    # archived pre-fix DATA_INSUFFICIENT stop is represented by the current
    # candidate-local optional gap and independent budget stop.
    state["errors"] = []
    investigation = InvestigationState.model_validate(state["investigation_state"])
    failed = investigation.action_results[-1]
    candidate = next(item for item in investigation.candidate_actions if item.candidate_id == failed.candidate_id)
    state["diagnostic_candidate_data_gaps"] = []
    _record_candidate_data_gap(
        state, investigation, failed,
        fallback_candidate=candidate,
        failure_detail=failed.failure_detail,
    )
    state["investigation_state"] = investigation.model_dump(mode="json")
    state["errors"] = [{"code": "token_budget_exceeded", "message": "offline saved-run replay"}]
    return state


def _run_terminal_and_answer(state: dict, monkeypatch) -> dict:
    monkeypatch.setattr("app.core.execution_context.enqueue_sse_event", lambda _event: None)
    context = TenantContext(
        tenant_id="test-tenant", user_id="test-user", role="admin", scope_type="platform",
    )
    with activate_tenant_context(context):
        finalized = diagnostic_finalize(state)
        answered = generate_answer(finalized)
    return answered


def test_optional_gap_keeps_valid_aov_decomposition_in_partial_terminal_and_answer(monkeypatch):
    state = _partial_aov_state()

    answered = _run_terminal_and_answer(state, monkeypatch)
    result = answered["analysis_result"]
    card = AnalysisPresentation.model_validate(answered["analysis_presentation"]).metric_decomposition

    assert result["abstain"] is True
    assert result["investigation_summary"]["stop_reason"]["code"] == "BUDGET_EXHAUSTED"
    assert answered["investigation_state"]["action_results"][-1]["failure_code"] == "DATA_INSUFFICIENT"
    assert answered["diagnostic_candidate_data_gaps"][0]["failure_scope"] == "optional_deep_check"
    assert result["metric_decomposition"]["baseline_value"] == 10.0
    assert result["metric_decomposition"]["current_value"] == 20.0
    assert result["metric_decomposition"]["delta_value"] == 10.0
    assert [item["contribution"] for item in result["metric_decomposition"]["factors"]] == [
        2.6666666666666665, 7.333333333333332,
    ]
    assert result["metric_decomposition"]["business_cause_confirmed"] is False
    assert DECOMPOSITION_ID in result["evidence_ids"]
    assert DECOMPOSITION_ID in answered["finish_decision"]["supporting_evidence_ids"]
    assert answered["analysis_presentation"]["status"] == "partial"
    assert card is not None and card.evidence_ids == [DECOMPOSITION_ID]
    assert any(item["evidence_id"] == DECOMPOSITION_ID for item in answered["analysis_presentation"]["evidence"])
    assert "基线值 10.00" in answered["answer"]
    assert "当前值 20.00" in answered["answer"]
    assert "+2.67" in answered["answer"] and "+7.33" in answered["answer"]
    assert "不证明业务因果" in answered["answer"] and "业务原因仍未确认" in answered["answer"]
    assert "调查动作预算已用尽" in answered["answer"]
    assert "Optional deep check CHECK_ANOMALY order_count was not completed" in answered["answer"]
    assert "Optional deep check CHECK_ANOMALY order_count was not completed" in answered["answer"]
    assert DECOMPOSITION_ID in answered["answer"]
    assert "7" in answered["answer"] and "8" in answered["answer"]


def test_partial_projection_rejects_decomposition_that_fails_registered_validation(monkeypatch):
    state = _partial_aov_state()
    evidence = next(
        item for item in state["investigation_state"]["evidence"]
        if item.get("evidence_id") == DECOMPOSITION_ID
    )
    evidence["values"]["residual_explainable"] = False

    answered = _run_terminal_and_answer(state, monkeypatch)

    assert "metric_decomposition" not in answered["analysis_result"]
    assert answered["analysis_presentation"]["metric_decomposition"] is None
    assert "基线值 10.00" not in answered["answer"]
    assert answered["analysis_presentation"]["status"] == "partial"


def test_partial_projection_rejects_scope_mismatch_and_unvalidated_action(monkeypatch):
    wrong_scope = _partial_aov_state()
    evidence = next(
        item for item in wrong_scope["investigation_state"]["evidence"]
        if item.get("evidence_id") == DECOMPOSITION_ID
    )
    evidence["scope_fingerprint"] = "different-authorized-scope"
    scope_answer = _run_terminal_and_answer(wrong_scope, monkeypatch)
    assert "metric_decomposition" not in scope_answer["analysis_result"]
    assert scope_answer["analysis_presentation"]["metric_decomposition"] is None
    assert "基线值 10.00" not in scope_answer["answer"]

    unvalidated = _partial_aov_state()
    action_result = next(
        item for item in unvalidated["investigation_state"]["action_results"]
        if item.get("action_type") == "DECOMPOSE_METRIC"
    )
    action_result["validation_result"]["status"] = "invalid"
    action_answer = _run_terminal_and_answer(unvalidated, monkeypatch)
    assert "metric_decomposition" not in action_answer["analysis_result"]
    assert action_answer["analysis_presentation"]["metric_decomposition"] is None
    assert "基线值 10.00" not in action_answer["answer"]


def test_partial_projection_rejects_period_mismatch_and_invalid_data_stop(monkeypatch):
    mismatched = _partial_aov_state()
    evidence = next(item for item in mismatched["investigation_state"]["evidence"] if item.get("evidence_id") == DECOMPOSITION_ID)
    evidence["comparison"]["current_period"]["end_date"] = "2024-05-05"
    answered_mismatch = _run_terminal_and_answer(mismatched, monkeypatch)
    assert "metric_decomposition" not in answered_mismatch["analysis_result"]
    assert answered_mismatch["analysis_presentation"]["metric_decomposition"] is None

    conflict = _partial_aov_state()
    conflict["investigation_state"]["stop_reason"] = {
        "code": "INVALID_DATA", "detail": "Conflicting input evidence.", "initiator": "backend",
    }
    answered_conflict = _run_terminal_and_answer(conflict, monkeypatch)
    assert "metric_decomposition" not in answered_conflict["analysis_result"]
    assert answered_conflict["analysis_presentation"]["metric_decomposition"] is None
    assert answered_conflict["analysis_presentation"]["status"] == "partial"
    assert answered_conflict["analysis_result"]["investigation_summary"]["stop_reason"]["code"] == "INVALID_DATA"


def test_required_trend_data_gap_blocks_partial_decomposition_projection(monkeypatch):
    state = _partial_aov_state()
    state["errors"] = []
    state["diagnostic_step_state"]["target_trend"] = "data_insufficient"
    state["investigation_state"]["stop_reason"] = {
        "code": "DATA_INSUFFICIENT", "detail": "Required target trend is incomplete.", "initiator": "backend",
    }
    state["diagnostic_candidate_data_gaps"] = [{
        "failure_scope": "required_step", "required_step_id": "target_trend",
        "failure_code": "DATA_INSUFFICIENT",
    }]

    answered = _run_terminal_and_answer(state, monkeypatch)

    assert "metric_decomposition" not in answered["analysis_result"]
    assert answered["analysis_presentation"]["metric_decomposition"] is None
    assert answered["analysis_presentation"]["status"] == "partial"
    assert answered["analysis_result"]["investigation_summary"]["stop_reason"]["code"] == "DATA_INSUFFICIENT"


def test_unvalidated_partial_decomposition_never_surfaces_factor_effect(monkeypatch):
    state = _partial_aov_state()
    state["errors"] = []
    state["diagnostic_step_state"]["target_trend"] = "data_insufficient"
    state["investigation_state"]["stop_reason"] = {
        "code": "DATA_INSUFFICIENT", "detail": "Required target trend is incomplete.", "initiator": "backend",
    }
    state["diagnostic_candidate_data_gaps"] = [{
        "failure_scope": "required_step", "required_step_id": "target_trend",
        "failure_code": "DATA_INSUFFICIENT",
    }]
    state["investigation_state"]["branches"] = [{
        "branch_id": "branch-order-count",
        "branch_type": "metric_effect",
        "metric_id": "order_count",
        "parent_evidence_id": DECOMPOSITION_ID,
        "effect_value": 7.333333333333332,
        "effect_unit": "BRL/order",
        "root_effect_share": 0.7333333333333332,
        "status": "open",
    }]

    answered = _run_terminal_and_answer(state, monkeypatch)

    assert "metric_decomposition" not in answered["analysis_result"]
    assert answered["analysis_presentation"]["metric_decomposition"] is None
    limitations = "\n".join(answered["analysis_result"]["limitations"])
    presentation_text = json.dumps(answered["analysis_presentation"], ensure_ascii=False)
    assert "尚无可引用的有效贡献证据" in limitations
    assert "数值贡献 +7.33 BRL/order" not in limitations
    assert "+7.33" not in answered["answer"]
    assert "+7.33" not in presentation_text


def test_inconsistent_duplicate_decomposition_for_same_scope_and_period_is_not_projected(monkeypatch):
    state = _partial_aov_state()
    original = next(
        item for item in state["investigation_state"]["evidence"]
        if item.get("evidence_id") == DECOMPOSITION_ID
    )
    conflicting = json.loads(json.dumps(original))
    conflicting["evidence_id"] = "ev_inconsistent_duplicate_decomposition"
    conflicting["values"]["current_value"] = 21.0
    conflicting["values"]["total_delta"] = 11.0
    state["investigation_state"]["evidence"].append(conflicting)

    answered = _run_terminal_and_answer(state, monkeypatch)

    assert "metric_decomposition" not in answered["analysis_result"]
    assert answered["analysis_presentation"]["metric_decomposition"] is None
    assert "基线值 10.00" not in answered["answer"]


def test_normal_validated_decomposition_remains_completed_and_structured(monkeypatch):
    state = _saved_normal_aov_state()
    answered = _run_terminal_and_answer(state, monkeypatch)

    assert answered["analysis_result"]["abstain"] is False
    assert answered["analysis_presentation"]["status"] == "completed"
    normal_evidence_id = answered["analysis_presentation"]["metric_decomposition"]["evidence_ids"][0]
    assert normal_evidence_id in answered["analysis_result"]["evidence_ids"]
    assert any(item["evidence_id"] == normal_evidence_id for item in answered["analysis_presentation"]["evidence"])
    assert "+2.67" in answered["answer"] and "+7.33" in answered["answer"]


def test_saved_dimension_contribution_still_reaches_answer_without_provider(monkeypatch):
    state = json.loads(AOV_DIMENSION_TRACE.read_text(encoding="utf-8"))["workflow_state"]
    monkeypatch.setattr("app.llm.client.llm_chat", lambda **_kwargs: None)

    answered = _run_terminal_and_answer(state, monkeypatch)

    cards = answered["analysis_presentation"]["contributions"]
    assert len(cards) == 1
    assert cards[0]["analysis_nature"] == "descriptive_contribution"
    assert cards[0]["evidence_ids"]
    assert answered["analysis_presentation"]["metric_decomposition"] is None
    assert any(
        claim.get("source_evidence_ids") == cards[0]["evidence_ids"]
        for claim in answered["claims"]
    )

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.graph.critic_review_policy import decide_optional_review_cascade
from app.graph.nodes.critic import (
    _completed_review_budget_reserve,
    _optional_review_fits_with_required_followup_reserve,
)


def _low_risk_state():
    answer = "GMV 分解结果显示两个组成因素均已核验；数值分解本身不证明业务因果。"
    ledger = {
        "schema_version": "registered_decomposition_answer_contract_v1",
        "metric_id": "gmv",
        "method_id": "additive_change_v1",
        "evidence_ids": ["ev-decomposition"],
        "business_cause_confirmed": False,
    }
    return {
        "answer": answer,
        "errors": [],
        "adaptive_answer_contract": {
            "ledger": ledger,
            "ledger_sha256": hashlib.sha256(
                json.dumps(ledger, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
                .encode("utf-8")
            ).hexdigest(),
            "rendered_answer": answer,
            "rendered_answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        },
    }


def _passed_deterministic_reviews():
    return {
        "request_contract_review": {"status": "passed"},
        "data_review": {
            "status": "passed", "target_metric_id": "gmv",
            "method_id": "additive_change_v1", "evidence_ids": ["ev-decomposition"],
        },
        "claim_review": {"status": "approved"},
        "numeric_review": {"status": "not_applicable"},
        "completeness_review": {"status": "skipped"},
        "causal_review": {"status": "approved"},
    }


def _decide(*, task_type="metric_query", state=None, reviews=None, terminal=False):
    return decide_optional_review_cascade(
        task_type=task_type,
        registered_decomposition_review=True,
        terminal_execution_failure=terminal,
        state=state or _low_risk_state(),
        deterministic_reviews=reviews or _passed_deterministic_reviews(),
    )


def test_cascade_keeps_groundedness_and_skips_only_advisory_reviews_for_low_risk_case():
    decision = _decide()

    assert decision["tier"] == "low"
    assert decision["action"] == "groundedness_only"
    assert decision["required_review_names"] == ["groundedness"]
    assert decision["skip_review_names"] == ["completeness", "consistency"]


def test_cascade_fails_closed_for_diagnostic_tasks_workflow_errors_and_failed_gates():
    diagnostic = _decide(task_type="anomaly_analysis")
    errored = _decide(state={**_low_risk_state(), "errors": [{"code": "database_error"}]})
    failed_gate = _decide(reviews={
        **_passed_deterministic_reviews(),
        "numeric_review": {"status": "rejected"},
    })

    assert diagnostic["action"] == "full_review"
    assert diagnostic["tier"] == "high"
    assert errored["action"] == "full_review"
    assert errored["reason_code"] == "workflow_errors_present"
    assert failed_gate["action"] == "full_review"
    assert failed_gate["reason_code"] == "deterministic_gate_not_passed"


def test_cascade_does_not_run_after_authoritative_terminal_failure():
    decision = _decide(terminal=True)

    assert decision["action"] == "not_run"
    assert decision["tier"] == "terminal"
    assert decision["skip_review_names"] == []


def test_optional_review_budget_guard_preserves_one_required_followup_call():
    state = {
        "runtime_budget": {"max_tokens": 1000},
        "llm_usage": {
            "input_tokens": 600, "output_tokens": 0,
            "preflight_reserved_tokens": 100,
        },
        "llm_calls": [{
            "call_id": "call-groundedness",
            "token_budget_preflight": {"allowed": True, "required_tokens": 250},
        }],
    }
    review = {"call_ids": ["call-groundedness"]}

    assert _completed_review_budget_reserve(state, review) == 250
    assert _optional_review_fits_with_required_followup_reserve(
        state, optional_call_reserve=150, required_call_reserve=250,
    ) is False
    assert _optional_review_fits_with_required_followup_reserve(
        {**state, "llm_usage": {"input_tokens": 600, "output_tokens": 0}},
        optional_call_reserve=150, required_call_reserve=250,
    ) is True
    assert _optional_review_fits_with_required_followup_reserve(
        {**state, "llm_usage": {"input_tokens": 600, "output_tokens": 0}},
        optional_call_reserve=151, required_call_reserve=250,
    ) is False


@pytest.mark.parametrize(
    ("mode", "expected_calls", "expected_status", "applied"),
    [
        ("shadow", ["groundedness", "completeness", "consistency"], "passed", False),
        (
            "active", ["groundedness"],
            "required_passed_optional_skipped_by_risk_cascade", True,
        ),
    ],
)
def test_review_cascade_shadow_is_observational_and_active_keeps_groundedness(
    monkeypatch, mode, expected_calls, expected_status, applied,
):
    from app.core.config import get_settings
    from app.core.execution_context import activate_state, reset_state
    from app.graph.critic_review_mode import CriticLLMReviewMode
    import app.graph.nodes.critic as critic_module

    trace_path = (
        Path(__file__).resolve().parents[1]
        / "docs/testing/registered_numeric_decomposition_intent_20261014/decomposition_avg_order_value.json"
    )
    state = json.loads(trace_path.read_text(encoding="utf-8"))
    for key in ("critic_result", "critic_verdict", "critic_revision_target", "critic_decisions"):
        state.pop(key, None)
    state["errors"] = []
    state["current_agent"] = "critic"
    state["llm_calls"] = []
    state["llm_usage"] = {"input_tokens": 0, "output_tokens": 0, "estimated_cost": 0.0}
    settings = get_settings().model_copy(update={"critic_optional_review_cascade_mode": mode})
    monkeypatch.setattr(critic_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        critic_module,
        "decide_optional_review_cascade",
        lambda **_kwargs: {
            "tier": "low", "action": "groundedness_only",
            "reason_code": "test_low_risk", "skip_review_names": ["completeness", "consistency"],
        },
    )
    calls = []
    monkeypatch.setattr(
        critic_module,
        "llm_review_groundedness",
        lambda *_args, **_kwargs: calls.append("groundedness") or {"verdict": "passed", "issues": []},
    )
    monkeypatch.setattr(
        critic_module, "llm_review_evidence_completeness",
        lambda *_args, **_kwargs: calls.append("completeness") or {"verdict": "passed", "issues": []},
    )
    monkeypatch.setattr(
        critic_module, "llm_review_logic_consistency",
        lambda *_args, **_kwargs: calls.append("consistency") or {"verdict": "passed", "issues": []},
    )

    token = activate_state(state)
    try:
        reviewed = critic_module.critic(state, llm_review_mode=CriticLLMReviewMode.FULL)
    finally:
        reset_state(token)

    llm_review = reviewed["critic_result"]["llm_review"]
    assert calls == expected_calls
    assert llm_review["groundedness"]["verdict"] == "passed"
    assert llm_review["status"] == expected_status
    assert reviewed["critic_result"]["optional_review_cascade"]["applied"] is applied
    if applied:
        assert llm_review["completeness"]["failure_kind"] == "risk_cascade_skip"
        assert llm_review["consistency"]["failure_kind"] == "risk_cascade_skip"
        assert llm_review["context"]["planned_reviewers"] == ["groundedness"]

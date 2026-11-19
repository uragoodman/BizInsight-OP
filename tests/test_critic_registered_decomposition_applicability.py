"""Offline Critic applicability checks for Registry numeric decompositions."""

from __future__ import annotations

import copy
import json
import socket
from pathlib import Path
from types import SimpleNamespace

import httpx


TRACE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/testing/registered_numeric_decomposition_intent_20261014/decomposition_avg_order_value.json"
)


def _block_network(monkeypatch):
    attempts = []

    def blocked(*args, **kwargs):
        attempts.append("blocked")
        raise AssertionError("external network is blocked in this offline Critic test")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(httpx.Client, "send", blocked)
    monkeypatch.setattr(httpx.AsyncClient, "send", blocked)
    try:
        socket.create_connection(("203.0.113.1", 443), timeout=0.01)
    except AssertionError:
        pass
    else:  # pragma: no cover
        raise AssertionError("network block did not intercept the probe")
    attempts.clear()
    return attempts


def _sdk_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
    )


def test_registered_numeric_decomposition_full_off_sdk_review_and_rework(monkeypatch):
    """Full reviews once per check; off is explicit, and text rework keeps facts."""
    from app.agents import critic_agent
    from app.core.config import get_settings
    from app.core.execution_context import activate_state, reset_state
    from app.graph.critic_review_mode import CriticLLMReviewMode
    from app.graph.nodes.answer import generate_answer
    from app.graph.nodes.critic import critic, _registered_decomposition_review_context
    from app.graph.workflow import _route_from_critic
    from app.llm import client as llm

    network_attempts = _block_network(monkeypatch)
    saved = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    assert saved["metric_planning_decision"]["operation_intent"] == "registered_numeric_decomposition"
    assert saved["registered_decomposition_request"]["status"] == "ready"

    settings = get_settings().model_copy(update={
        "deepseek_api_key": "offline-critic-sdk-key",
        "deepseek_base_url": "https://offline-critic.invalid/v1",
        "llm_providers_json": "[]",
        "llm_resilience_attempts": 1,
        "llm_timeout_seconds": 5.0,
    })
    monkeypatch.setattr(llm, "get_settings", lambda: settings)
    monkeypatch.setattr(llm, "_quota_available", lambda: True)
    monkeypatch.setattr(llm, "record_usage", lambda **_kwargs: None)

    sdk_requests = []

    class FakeCompletions:
        def create(self, **kwargs):
            sdk_requests.append(copy.deepcopy(kwargs))
            index = len(sdk_requests)
            if index == 2:
                content = json.dumps({
                    "verdict": "failed",
                    "issues": ["Please state the verified comparison-period change explicitly."],
                })
            else:
                content = json.dumps({"verdict": "passed", "issues": []})
            return _sdk_response(content)

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=FakeCompletions().create)),
    )
    monkeypatch.setattr(llm, "OpenAI", lambda **_kwargs: client)
    # Exercise the actual llm_extract wrapper and SDK send boundary used by all
    # three Critic review functions; only the external SDK response is replaced.
    monkeypatch.setattr(critic_agent, "llm_extract", llm.llm_extract)

    base = copy.deepcopy(saved)
    for key in ("critic_result", "critic_verdict", "critic_revision_target", "critic_decisions"):
        base.pop(key, None)
    base["critic_llm_review_mode"] = "full"
    base["_revision_counts"] = {"generate_answer": 0}
    base["llm_calls"] = []
    base["llm_usage"] = {"input_tokens": 0, "output_tokens": 0, "estimated_cost": 0.0}
    base["llm_token_summary"] = {"logical_call_count": 0, "sdk_attempt_count": 0}

    full_input = copy.deepcopy(base)
    full_input["current_agent"] = "critic"
    full_initial_snapshot = copy.deepcopy(full_input)
    full_token = activate_state(full_input)
    try:
        full_reviewed = critic(full_input, llm_review_mode=CriticLLMReviewMode.FULL)
    finally:
        reset_state(full_token)

    assert len(sdk_requests) == 3
    first_cycle_messages = [request["messages"] for request in sdk_requests]
    assert "registered_numeric_decomposition" in str(first_cycle_messages)
    assert all(request.get("model") for request in sdk_requests)
    assert all("max_tokens" in request or "max_completion_tokens" in request for request in sdk_requests)
    request_settings = [
        {key: value for key, value in request.items() if key != "messages"}
        for request in sdk_requests
    ]
    assert request_settings[0] == request_settings[1] == request_settings[2]
    full_llm_review = full_reviewed["critic_result"]["llm_review"]
    assert full_llm_review["status"] == "failed"
    assert [full_llm_review[name]["verdict"] for name in ("groundedness", "completeness", "consistency")] == [
        "passed", "failed", "passed",
    ]
    assert full_reviewed["critic_verdict"] == "needs_revision"
    assert full_reviewed["critic_revision_target"] == "generate_answer"
    assert _route_from_critic(full_reviewed) == "generate_answer"
    assert "not_established_by_numeric_decomposition" in str(first_cycle_messages)
    expected_answer = base["answer"]
    decomposition_evidence_id = base["analysis_result"]["metric_decomposition"]["evidence_ids"][0]
    assert all(decomposition_evidence_id in str(messages) for messages in first_cycle_messages)
    assert all("avg_order_value" in str(messages) and "10.0" in str(messages) and "20.0" in str(messages)
               for messages in first_cycle_messages)

    plan_before = copy.deepcopy(full_reviewed["diagnostic_plan"])
    action_results_before = copy.deepcopy(full_reviewed["investigation_state"]["action_results"])
    decomposition_before = copy.deepcopy(full_reviewed["analysis_result"]["metric_decomposition"])
    calls_before_revision = copy.deepcopy(full_reviewed["tool_calls"])
    answer_before_revision = full_reviewed["answer"]
    revised = generate_answer(full_reviewed)
    assert revised["answer"] != answer_before_revision
    assert revised["answer_revision"]["status"] == "revised"
    assert revised["answer_revision"]["feedback_categories"] == ["completeness"]
    assert revised["analysis_result"]["metric_decomposition"] == decomposition_before
    assert revised["analysis_presentation"]["metric_decomposition"]["evidence_ids"] == decomposition_before["evidence_ids"]
    assert revised["diagnostic_plan"] == plan_before
    assert revised["investigation_state"]["action_results"] == action_results_before
    assert revised["tool_calls"] == calls_before_revision
    assert "2024-05-03" in revised["answer"] and "2024-05-04" in revised["answer"]
    assert "10.00" in revised["answer"] and "20.00" in revised["answer"]
    assert "+2.67" in revised["answer"] and "+7.33" in revised["answer"]
    assert "业务原因仍未确认" in revised["answer"]

    # The second Critic pass follows the actual answer-revision state path.
    second_token = activate_state(revised)
    try:
        full_final = critic(revised, llm_review_mode=CriticLLMReviewMode.FULL)
    finally:
        reset_state(second_token)
    assert len(sdk_requests) == 6
    assert full_final["critic_result"]["llm_review"]["status"] == "passed"
    assert all(full_final["critic_result"]["llm_review"][name]["verdict"] == "passed"
               for name in ("groundedness", "completeness", "consistency"))
    assert full_final["critic_verdict"] == "approved"
    assert full_final["analysis_result"]["metric_decomposition"] == decomposition_before
    assert full_final["tool_calls"] == calls_before_revision

    off_input = copy.deepcopy(base)
    off_input["current_agent"] = "critic"
    off_input["critic_llm_review_mode"] = "critic_llm_review_off"
    common_review_inputs = (
        "task_type", "context_compilation", "metric_planning_decision", "registered_decomposition_request",
        "answer", "analysis_result", "analysis_presentation", "investigation_state", "evidence_bundles",
        "authorized_metrics", "authorization_boundary", "controlled_analysis_scope", "runtime_budget",
        "critic_llm_thinking_config", "llm_call_config", "llm_call_config_summary",
        "tenant_id", "user_id", "user_role", "scope_type",
    )
    assert {key: full_initial_snapshot.get(key) for key in common_review_inputs} == {
        key: off_input.get(key) for key in common_review_inputs
    }
    off_token = activate_state(off_input)
    try:
        off_reviewed = critic(
            off_input,
            llm_review_mode=CriticLLMReviewMode.CRITIC_LLM_REVIEW_OFF,
        )
    finally:
        reset_state(off_token)
    assert len(sdk_requests) == 6
    off_llm_review = off_reviewed["critic_result"]["llm_review"]
    assert off_llm_review["status"] == "evaluation_ablation"
    assert [off_llm_review[name]["verdict"] for name in ("groundedness", "completeness", "consistency")] == [
        "evaluation_ablation", "evaluation_ablation", "evaluation_ablation",
    ]
    assert off_reviewed["critic_result"]["data_review"] == full_reviewed["critic_result"]["data_review"]
    assert off_reviewed["critic_result"]["claim_review"] == full_reviewed["critic_result"]["claim_review"]
    full_review_context = _registered_decomposition_review_context(
        full_initial_snapshot, full_initial_snapshot["registered_decomposition_request"],
        full_reviewed["critic_result"]["data_review"],
    )
    off_review_context = _registered_decomposition_review_context(
        off_input, off_input["registered_decomposition_request"],
        off_reviewed["critic_result"]["data_review"],
    )
    assert full_review_context == off_review_context
    assert full_review_context["operation_intent"] == "registered_numeric_decomposition"
    assert full_review_context["task_type"] == "metric_query"
    assert full_review_context["target_metric_id"] == "avg_order_value"
    assert full_review_context["business_causality"] == "not_established_by_numeric_decomposition"
    assert [item["evidence_id"] for item in full_review_context["verified_numeric_evidence"]] == [
        decomposition_evidence_id,
    ]
    assert off_reviewed["analysis_result"]["metric_decomposition"] == decomposition_before
    assert off_reviewed["tool_calls"] == calls_before_revision
    assert off_reviewed["answer"] == expected_answer
    assert network_attempts == []

    ordinary = copy.deepcopy(base)
    ordinary["metric_planning_decision"]["operation_intent"] = "ordinary_metric_query"
    ordinary["context_compilation"]["operation_intent"]["kind"] = "ordinary_metric_query"
    ordinary["registered_decomposition_request"] = {}
    ordinary["current_agent"] = "critic"
    ordinary_token = activate_state(ordinary)
    try:
        ordinary_reviewed = critic(ordinary, llm_review_mode=CriticLLMReviewMode.FULL)
    finally:
        reset_state(ordinary_token)
    assert len(sdk_requests) == 6
    assert ordinary_reviewed["critic_result"]["llm_review"]["status"] == "not_applicable"
    assert all(ordinary_reviewed["critic_result"]["llm_review"][name]["verdict"] == "skipped"
               for name in ("groundedness", "completeness", "consistency"))


def test_adaptive_contract_mismatch_regenerates_without_optional_llm(monkeypatch):
    """A post-render text change is repaired from the verified card, not reviewed by an LLM."""
    from app.core.execution_context import activate_state, reset_state
    from app.graph.critic_review_mode import CriticLLMReviewMode
    from app.graph.nodes.answer import generate_answer
    from app.graph.nodes.critic import critic
    from app.graph.workflow import _route_from_critic

    network_attempts = _block_network(monkeypatch)
    state = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    for key in ("critic_result", "critic_verdict", "critic_revision_target", "critic_decisions"):
        state.pop(key, None)
    state["critic_llm_review_mode"] = "adaptive"
    state["_revision_counts"] = {"generate_answer": 0}
    state["llm_calls"] = []
    state["llm_usage"] = {"input_tokens": 0, "output_tokens": 0, "estimated_cost": 0.0}
    state["llm_token_summary"] = {"logical_call_count": 0, "sdk_attempt_count": 0}

    rendered = generate_answer(state)["answer"]
    assert state.get("adaptive_answer_contract")
    state["answer"] = rendered + "\n订单量是客单价上涨的主要推动因素。"
    state["final_answer"] = state["answer"]

    token = activate_state(state)
    try:
        reviewed = critic(state, llm_review_mode=CriticLLMReviewMode.ADAPTIVE)
    finally:
        reset_state(token)
    assert reviewed["critic_verdict"] == "needs_revision"
    assert reviewed["critic_revision_target"] == "generate_answer"
    assert reviewed["critic_result"]["llm_review"]["status"] == "not_run_due_answer_integrity_repair"
    assert reviewed["llm_calls"] == []
    assert _route_from_critic(reviewed) == "generate_answer"
    assert network_attempts == []

    repaired = generate_answer(reviewed)
    assert repaired["answer"] == rendered
    assert repaired.get("adaptive_answer_contract")

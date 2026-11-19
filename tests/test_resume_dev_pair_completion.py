"""Offline completion-boundary and versioned-score regressions for the dev pair."""
from __future__ import annotations

import copy
import json
import socket
from pathlib import Path

import pytest

from app.tenancy import TenantContext, activate_tenant_context
from scripts import run_resume_dev_pair as runner


ROOT = Path(__file__).resolve().parents[1]
G_TRACE = (
    ROOT / "docs/testing/resume_dev_pair_order_dimension_20261015_9e522499/traces/"
    "432bd34e-630d-45a8-b3c2-0d6a69ad7011.json"
)
PAIR_DIR = G_TRACE.parent.parent


def _saved_g_state() -> dict:
    return json.loads(G_TRACE.read_text(encoding="utf-8"))["state_snapshot"]


def _required_evidence_floor_state() -> dict:
    state = copy.deepcopy(_saved_g_state())
    investigation = state["investigation_state"]
    investigation["stop_reason"] = None
    required_results = [
        item for item in investigation["action_results"]
        if item.get("action_type") in {"ANALYZE_TREND", "BREAKDOWN_DIMENSION"}
        and item.get("status") == "completed"
        and (item.get("validation_result") or {}).get("status") == "valid"
    ]
    assert len(required_results) == 2
    evidence_ids = {str(eid) for item in required_results for eid in item.get("evidence_ids", [])}
    investigation["action_results"] = required_results
    investigation["evidence"] = [
        item for item in investigation["evidence"]
        if str(item.get("evidence_id")) in evidence_ids
    ]
    state["root_cause_evidence"] = [
        item for item in state.get("root_cause_evidence", [])
        if str(item.get("query_id") or "") in evidence_ids
    ]
    query_evidence_ids = {
        str(item.get("evidence_id") or "") for item in state.get("query_results", [])
    }
    state["evidence_bundles"] = [
        item for item in state.get("evidence_bundles", [])
        if str(item.get("evidence_id") or "") in evidence_ids | query_evidence_ids
    ]
    state["diagnostic_step_state"] = {
        "trend": "completed",
        "dimension_contribution": "completed",
        "related_metric_optional": "pending",
    }
    state["diagnostic_candidate_data_gaps"] = []
    state["errors"] = []
    # Rewind only the saved node counter to the point immediately after its
    # two required actions; the original stored state is never modified.
    state["diagnostic_round"] = 2
    state.pop("diagnostic_action", None)
    return state


def _block_network(*_args, **_kwargs):
    raise AssertionError("external network must remain blocked in offline completion tests")


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    from app.llm import client

    monkeypatch.setattr(socket.socket, "connect", _block_network)
    monkeypatch.setattr(socket.socket, "connect_ex", _block_network)
    monkeypatch.setattr(socket, "create_connection", _block_network)
    for name in ("llm_chat", "llm_extract", "llm_classify", "llm_agent_action"):
        monkeypatch.setattr(client, name, lambda *a, **k: pytest.fail("unexpected model entry"))


def test_required_statistical_contribution_finishes_without_optional_exploration(monkeypatch):
    from app.graph.nodes import diagnostic, answer

    monkeypatch.setattr(socket.socket, "connect", _block_network)
    monkeypatch.setattr(socket.socket, "connect_ex", _block_network)
    state = _required_evidence_floor_state()
    tool_count = len(state.get("tool_calls") or [])
    action_count = len(state["investigation_state"]["action_results"])

    def unexpected_model_decision(_state):
        pytest.fail("server completion at the required evidence floor must not call an optional planner")

    monkeypatch.setattr(diagnostic, "_llm_decide_action", unexpected_model_decision)
    context = TenantContext(
        tenant_id="platform", user_id="resume-dev-pair-order-dimension",
        role="admin", scope_type="platform",
    )
    with activate_tenant_context(context):
        diagnostic.diagnostic_decide(state)
        action = state["diagnostic_action"]
        assert action["action_type"] == "FINISH"
        assert state["diagnostic_backend_terminal"] == "validated_statistical_contribution"
        assert set(action["evidence_ids"]) == {
            "ev_b6124cad3c790f00", "ev_7810507b75c6c16a",
        }
        assert len(state["tool_calls"]) == tool_count
        assert len(state["investigation_state"]["action_results"]) == action_count
        assert state["diagnostic_step_state"]["related_metric_optional"] == "pending"

        monkeypatch.setattr(
            diagnostic, "_audit_soft_finish",
            lambda *args, **kwargs: pytest.fail("validated task completion must not reopen optional candidates"),
        )
        diagnostic.policy_execute(state)
        diagnostic.diagnostic_finalize(state)

    assert state["analysis_result"]["abstain"] is False
    assert state["investigation_state"]["stop_reason"]["code"] == "ENOUGH_EVIDENCE"
    assert state["analysis_result"]["evidence_ids"] == action["evidence_ids"]
    assert "business cause remains unconfirmed" in state["analysis_result"]["summary"]

    from app.llm import client as llm_client

    monkeypatch.setattr(llm_client, "llm_chat", lambda *_args, **_kwargs: '{"recommendations": []}')
    answer.generate_answer(state)
    presentation = state["analysis_presentation"]
    assert presentation["status"] == "completed"
    assert presentation["contributions"]
    assert presentation["contributions"][0]["baseline_value"] == pytest.approx(10.0)
    assert presentation["contributions"][0]["current_value"] == pytest.approx(6.0)
    assert presentation["contributions"][0]["total_delta"] == pytest.approx(-4.0)
    assert "2024-05-03" in state["answer"] and "2024-05-04" in state["answer"]
    assert "业务原因仍未确认" in state["answer"]
    assert "统计贡献" in state["answer"]
    assert "10.00 单" in state["answer"] and "6.00 单" in state["answer"]
    assert "-4.00 单" in state["answer"]

    # Continue through the real deterministic Critic and graph routing, which
    # the previous test stopped before. Optional pending must not restart work.
    import importlib
    critic_node = importlib.import_module("app.graph.nodes.critic")
    from app.graph.workflow import _route_from_critic

    calls = []
    def passed(*args, **kwargs):
        calls.append(args)
        return {"verdict": "passed", "issues": []}
    for name in ("llm_review_groundedness", "llm_review_evidence_completeness", "llm_review_logic_consistency"):
        monkeypatch.setattr(critic_node, name, passed)
    with activate_tenant_context(context):
        critic_node.critic(state, llm_review_mode="full")
    assert len(calls) == 3
    assert state["critic_result"]["diagnostic_review"]["status"] == "approved"
    assert state["critic_verdict"] == "approved"
    assert _route_from_critic(state) == "persist_trace"
    assert len(state["tool_calls"]) == tool_count
    assert len(state["investigation_state"]["action_results"]) == action_count
    score = runner._score(state, {"answer": state["answer"], "presentation": presentation}, "governed_workflow")
    assert score["business_completed"] and score["review_completed"]
    assert score["task_pass"]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("required_pending", None),
        ("global_budget_stop", None),
        ("evidence_conflict", None),
        ("causal_intent", None),
    ],
)
def test_statistical_completion_fails_closed_for_other_obligations(mutation, expected):
    from app.graph.nodes.diagnostic import _validated_statistical_contribution_terminal

    state = _required_evidence_floor_state()
    if mutation == "required_pending":
        state["diagnostic_step_state"]["trend"] = "pending"
    elif mutation == "global_budget_stop":
        state["investigation_state"]["stop_reason"] = {
            "code": "BUDGET_EXHAUSTED", "detail": "budget exhausted", "initiator": "budget",
        }
    elif mutation == "evidence_conflict":
        result = state["investigation_state"]["action_results"][1]
        result["validation_result"] = {
            "status": "invalid",
            "checks": [{"status": "failed", "conflicting_evidence_ids": [result["evidence_ids"][0]]}],
        }
    elif mutation == "causal_intent":
        state["context_compilation"]["operation_intent"]["kind"] = "business_causal_investigation"

    context = TenantContext(tenant_id="platform", user_id="resume-dev-pair-order-dimension", role="admin", scope_type="platform")
    with activate_tenant_context(context):
        assert _validated_statistical_contribution_terminal(state) is expected
        assert _validated_statistical_contribution_terminal(state, allow_completed_stop=True) is expected


@pytest.mark.parametrize("verdict", ["rejected", "approved"])
def test_terminal_critic_verdict_does_not_restart_triage(verdict):
    from app.graph.workflow import _route_from_critic

    assert _route_from_critic({"critic_verdict": verdict, "errors": []}) == "persist_trace"


@pytest.mark.parametrize("mutation", ["required_pending", "budget", "scope", "unit", "citation", "causal_intent"])
def test_statistical_critic_revalidates_instead_of_trusting_finish_marker(mutation):
    import importlib
    from app.graph.nodes import diagnostic

    state = _required_evidence_floor_state()
    context = TenantContext(tenant_id="platform", user_id="resume-dev-pair-order-dimension", role="admin", scope_type="platform")
    with activate_tenant_context(context):
        diagnostic.diagnostic_decide(state)
        diagnostic.policy_execute(state)
        diagnostic.diagnostic_finalize(state)
        if mutation == "required_pending":
            state["diagnostic_step_state"]["trend"] = "pending"
        elif mutation == "budget":
            state["investigation_state"]["stop_reason"]["code"] = "BUDGET_EXHAUSTED"
        elif mutation == "scope":
            state["investigation_state"]["evidence"][1]["scope_fingerprint"] = "wrong-scope"
        elif mutation == "unit":
            state["investigation_state"]["evidence"][1]["unit"] = "BRL"
        elif mutation == "citation":
            state["finish_decision"]["evidence_ids"] = []
        else:
            state["context_compilation"]["operation_intent"]["kind"] = "business_causal_investigation"
        review = importlib.import_module("app.graph.nodes.critic")._review_diagnostic_trajectory(state)
    assert review["status"] == "rejected"
    assert review["issues"]


@pytest.mark.parametrize("mode,missing_review", [("full", False), ("full", True), ("critic_llm_review_off", False)])
def test_latest_failed_run_replayed_before_first_terminal(monkeypatch, mode, missing_review):
    """Reconstruct the evidence floor, not rescore the failed historical run."""
    import importlib
    from app.graph.nodes import diagnostic, answer
    from app.graph.workflow import _route_from_critic

    path = ROOT / "docs/testing/resume_dev_pair_order_dimension_20261015_7fff6ca4/traces/32db5367-ac6a-4e19-b7ac-c22b2a45e56a.json"
    original_bytes = path.read_bytes()
    state = json.loads(original_bytes)["state_snapshot"]
    # Remove only consequences of later rejected/restarted cycles. Keep the
    # actual required action outcomes, evidence, authorization and dates.
    state["investigation_state"]["stop_reason"] = None
    state["errors"] = []
    state["diagnostic_round"] = 2
    state["diagnostic_step_state"] = {
        step["step_id"]: "completed" if step["required"] else "pending"
        for step in state["diagnostic_plan"]["steps"]
    }
    unique_calls = {}
    for call in state["tool_calls"]:
        unique_calls[(call["tool_name"], repr(call["arguments"]))] = call
    state["tool_calls"] = list(unique_calls.values())
    count = len(state["tool_calls"])
    action_count = len(state["investigation_state"]["action_results"])
    critic_node = importlib.import_module("app.graph.nodes.critic")
    from app.llm import client
    monkeypatch.setattr(client, "llm_chat", lambda *a, **k: '{"recommendations": []}')
    reviews = []
    def passed(*args, **kwargs):
        reviews.append(args)
        return None if missing_review else {"verdict": "passed", "issues": []}
    for name in ("llm_review_groundedness", "llm_review_evidence_completeness", "llm_review_logic_consistency"):
        monkeypatch.setattr(critic_node, name, passed)
    monkeypatch.setattr(diagnostic, "_llm_decide_action", lambda *a: pytest.fail("must not explore"))
    context = TenantContext(tenant_id="platform", user_id="resume-dev-pair-order-dimension", role="admin", scope_type="platform")
    with activate_tenant_context(context):
        diagnostic.diagnostic_decide(state)
        assert state["diagnostic_action"]["action_type"] == "FINISH"
        diagnostic.policy_execute(state)
        diagnostic.diagnostic_finalize(state)
        answer.generate_answer(state)
        critic_node.critic(state, llm_review_mode=mode)
    assert state["critic_result"]["diagnostic_review"]["status"] == "approved"
    assert state["critic_verdict"] == ("rejected" if missing_review else "approved"), state["critic_result"]
    assert len(reviews) == (0 if mode != "full" else 1 if missing_review else 3)
    assert _route_from_critic(state) == "persist_trace"
    assert len(state["tool_calls"]) == count
    assert len(state["investigation_state"]["action_results"]) == action_count
    assert state["analysis_presentation"]["status"] == "completed"
    assert state["analysis_presentation"]["contributions"][0]["total_delta"] == -4
    assert path.read_bytes() == original_bytes


def test_completion_v3_separates_content_business_and_review_completion():
    governed_path = G_TRACE.with_suffix(".run_record.json")
    basic_path = G_TRACE.parent / "9e522499-0b05-4dd4-a7df-9d6f271cc69a.run_record.json"
    governed = json.loads(governed_path.read_text(encoding="utf-8"))
    basic = json.loads(basic_path.read_text(encoding="utf-8"))

    g_score = runner._score(
        governed["trace"]["state_snapshot"], governed["response"], "governed_workflow",
    )
    b_score = runner._score(
        basic["trace"]["state_snapshot"], basic["response"], "basic_tool_loop",
    )

    assert g_score["content_valid"] is True
    assert g_score["effective_statistical_contribution"] is True
    assert g_score["business_completed"] is False
    assert g_score["business_stop_code"] == "BUDGET_EXHAUSTED"
    assert g_score["presentation_status"] == "partial"
    assert g_score["review_completed"] is True
    assert g_score["review_approved"] is True
    assert g_score["task_pass"] is False

    assert b_score["content_valid"] is True
    assert b_score["business_completed"] is True
    assert b_score["review_applicability"] == "not_applicable"
    assert b_score["review_completed"] is None
    assert b_score["task_pass"] is True

    completed_state = copy.deepcopy(governed["trace"]["state_snapshot"])
    completed_response = copy.deepcopy(governed["response"])
    completed_state["analysis_result"]["abstain"] = False
    completed_state["investigation_state"]["stop_reason"] = {
        "code": "ENOUGH_EVIDENCE", "detail": "required evidence completed", "initiator": "agent",
    }
    completed_state["analysis_presentation"] = {"status": "completed"}
    completed_response["presentation"]["status"] = "completed"
    completed_score = runner._score(completed_state, completed_response, "governed_workflow")
    assert completed_score["content_valid"] is True
    assert completed_score["business_completed"] is True
    assert completed_score["review_completed"] is True
    assert completed_score["review_approved"] is True
    assert completed_score["task_pass"] is True

    incomplete = copy.deepcopy(governed["trace"]["state_snapshot"])
    incomplete["critic_result"]["llm_review"]["consistency"]["completion_status"] = "truncated"
    incomplete_score = runner._score(incomplete, governed["response"], "governed_workflow")
    assert incomplete_score["content_valid"] is True
    assert incomplete_score["business_completed"] is False
    assert incomplete_score["review_completed"] is False
    assert incomplete_score["task_pass"] is False


def test_v3_replay_does_not_mutate_saved_v2_scores():
    for run_record in PAIR_DIR.glob("traces/*.run_record.json"):
        record = json.loads(run_record.read_text(encoding="utf-8"))
        old_score_path = PAIR_DIR / f"{record['run_id']}.score.json"
        old_bytes = old_score_path.read_bytes()
        trace_state = record["trace"]["state_snapshot"]
        result = runner._score(trace_state, record["response"], record["arm"])
        assert result["rubric_version"] == "resume_dev_pair_completion_v3"
        assert old_score_path.read_bytes() == old_bytes

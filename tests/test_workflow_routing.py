from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.graph import workflow
from app.graph.nodes import intent_router as intent_router_node


@pytest.mark.parametrize("mode", ["off", "shadow"])
def test_off_and_shadow_continue_ordinary_chat_without_plan_routing(monkeypatch, mode: str) -> None:
    settings = SimpleNamespace(
        complex_route_mode=mode,
        semantic_interpreter_mode="active",
        complex_selector_mode="active",
    )
    monkeypatch.setattr(intent_router_node, "get_settings", lambda: settings)
    monkeypatch.setattr(workflow, "get_settings", lambda: settings)
    state = intent_router_node.intent_router({"user_message": "RFM 客户分群", "node_history": []})
    assert workflow._route_after_intent_router(state) == "triage"
    assert not state.get("complex_plan")
    assert state["semantic_interpreter_observation"]["status"] == "reserved_phase0_noop"


def test_shadow_graph_runs_triage_before_router_to_capture_actual_baseline(monkeypatch) -> None:
    settings = SimpleNamespace(complex_route_mode="shadow")
    monkeypatch.setattr(workflow, "get_settings", lambda: settings)

    assert workflow._route_after_semantic_parse({}) == "triage"
    assert workflow._route_after_triage_for_shadow({"task_type": "metric_query"}) == "intent_router"
    assert workflow._route_after_intent_router({
        "task_type": "metric_query",
        "next_agent": "data_query",
        "triage_unresolved": False,
    }) == "retrieve_analysis_memories"


def test_active_canary_runs_triage_first_and_skips_router_for_non_enrolled_requests(monkeypatch) -> None:
    settings = SimpleNamespace(complex_route_mode="active")
    monkeypatch.setattr(workflow, "get_settings", lambda: settings)

    assert workflow._route_after_semantic_parse({}) == "triage"
    assert workflow._route_after_triage_for_shadow({
        "task_type": "metric_query",
        "next_agent": "data_query",
        "triage_unresolved": False,
        "complex_route_canary_selected": False,
    }) == "retrieve_analysis_memories"
    assert workflow._route_after_triage_for_shadow({
        "task_type": "metric_query",
        "next_agent": "data_query",
        "triage_unresolved": False,
        "complex_route_canary_selected": True,
    }) == "intent_router"


def test_active_canary_cannot_bypass_triage_stop_conditions(monkeypatch) -> None:
    settings = SimpleNamespace(complex_route_mode="active")
    monkeypatch.setattr(workflow, "get_settings", lambda: settings)

    token_budget_state = {
        "complex_route_canary_selected": True,
        "task_type": "metric_query",
        "errors": [{"code": "token_budget_exceeded"}],
    }
    assert workflow._route_after_triage_for_shadow(token_budget_state) == "generate_answer"

    router_error_state = {
        "complex_route_canary_observation": {"eligible": True, "action": "complex_plan"},
        "complex_route_decision": object(),
        "errors": [{"code": "triage_provider_error"}],
    }
    assert workflow._route_after_intent_router(router_error_state) == "critic"

    router_budget_state = {
        "complex_route_canary_observation": {"eligible": True, "action": "complex_plan"},
        "errors": [{"code": "token_budget_exceeded"}],
        "task_type": "metric_query",
    }
    assert workflow._route_after_intent_router(router_budget_state) == "generate_answer"


def test_active_phase0_always_clarifies_and_never_routes_to_plan_node(monkeypatch) -> None:
    settings = SimpleNamespace(
        complex_route_mode="active",
        semantic_interpreter_mode="off",
        complex_selector_mode="off",
    )
    monkeypatch.setattr(intent_router_node, "get_settings", lambda: settings)
    monkeypatch.setattr(workflow, "get_settings", lambda: settings)
    result = intent_router_node.intent_router({"user_message": "客户留存分析", "node_history": []})
    assert result["complex_route_decision"]["evidence"]["clarification_reason"] == "semantic_router_not_ready"
    assert workflow._route_after_intent_router(result) == "complex_clarification"
    assert workflow._route_after_complex_plan({"complex_clarification": {"reason_code": "semantic_router_not_ready"}}) == "complex_clarification"


def test_active_mode_does_not_trust_legacy_template_dict(monkeypatch) -> None:
    monkeypatch.setattr(
        workflow,
        "get_settings",
        lambda: SimpleNamespace(complex_route_mode="active"),
    )
    assert workflow._route_after_intent_router({
        "complex_route_decision": {
            "kind": "template",
            "route_decision_status": "SELECTED",
            "templates": [{"template_id": "cohort_retention"}],
        },
    }) == "complex_clarification"


def test_compiled_graph_keeps_plan_node_behind_the_closed_routing_boundary() -> None:
    graph = workflow.build_chat_graph(checkpointed=False)
    assert {
        "semantic_parse",
        "intent_router",
        "complex_plan",
        "complex_clarification",
        "triage",
        "persist_trace",
    } <= set(graph.get_graph().nodes)

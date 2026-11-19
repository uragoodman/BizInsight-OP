from time import perf_counter

import pytest
from pydantic import ValidationError

from app.agents.contracts import CriticDecision, CriticRevisionTarget
from app.graph.runtime import _validate_boundary, instrument_node
from app.graph.workflow import CRITIC_REVISION_ROUTES, _route_from_critic, build_chat_graph


def _critic_state(verdict: str, target):
    return {
        "critic_verdict": verdict,
        "critic_revision_target": target,
        "critic_decisions": [{"llm_issues": []}],
        "errors": [],
        "runtime_steps": 0,
        "runtime_started_at": perf_counter(),
        "runtime_budget": {"max_graph_steps": 32, "max_latency_ms": 60_000},
        "node_timings": [],
    }


def test_diagnostic_plan_target_passes_runtime_boundary_and_routes_to_node():
    state = _critic_state("needs_revision", "diagnostic_plan")

    result = instrument_node("critic", lambda current: current)(state)

    assert result["errors"] == []
    assert result["critic_verdict"] == "needs_revision"
    assert result["critic_revision_target"] == "diagnostic_plan"
    assert _route_from_critic(result) == "diagnostic_plan"
    assert _route_from_critic({**result, "critic_revision_target": "root_cause"}) == "diagnostic_plan"


def test_invalid_target_fails_runtime_boundary_and_cannot_become_approved():
    state = _critic_state("needs_revision", "invented_node")

    result = instrument_node("critic", lambda current: current)(state)

    assert result["critic_verdict"] == "rejected"
    assert result["errors"][-1]["cause_type"] == "ValidationError"
    assert result["critic_revision_target"] == "invented_node"
    assert _route_from_critic(result) == "persist_trace"
    assert _route_from_critic(_critic_state("needs_revision", "invented_node")) == "triage"
    with pytest.raises(ValidationError):
        _validate_boundary("critic", _critic_state("needs_revision", "invented_node"))


@pytest.mark.parametrize(
    ("verdict", "expected_route"),
    [("approved", "persist_trace"), ("rejected", "persist_trace")],
)
def test_terminal_verdicts_accept_no_revision_target(verdict: str, expected_route: str):
    state = _critic_state(verdict, None)

    result = instrument_node("critic", lambda current: current)(state)

    assert result["errors"] == []
    assert result["critic_revision_target"] is None
    _validate_boundary("critic", result)
    assert _route_from_critic(result) == expected_route


def test_revision_target_enum_is_explicit_and_includes_only_registered_correction_nodes():
    assert {target.value for target in CriticRevisionTarget} == {
        "triage",
        "metric_definition",
        "data_query",
        "diagnostic_plan",
        "report",
        "generate_answer",
    }
    with pytest.raises(ValidationError):
        CriticDecision(verdict="needs_revision", revision_target="persist_trace")


def test_every_whitelisted_target_is_a_real_critic_graph_edge():
    graph_targets = {edge.target for edge in build_chat_graph().get_graph().edges if edge.source == "critic"}

    assert set(CRITIC_REVISION_ROUTES.values()) <= graph_targets


def test_real_critic_answer_revision_crosses_runtime_and_routes(monkeypatch):
    from app.graph.nodes import critic as critic_module

    issue = {"verdict": "failed", "issues": ["state the verified trend explicitly"]}
    monkeypatch.setattr(critic_module, "llm_review_groundedness", lambda *_args, **_kwargs: issue)
    monkeypatch.setattr(critic_module, "llm_review_evidence_completeness", lambda *_args, **_kwargs: {"verdict": "passed", "issues": []})
    monkeypatch.setattr(critic_module, "llm_review_logic_consistency", lambda *_args, **_kwargs: {"verdict": "passed", "issues": []})
    monkeypatch.setattr(critic_module, "review_data_query_result", lambda *_: {"status": "approved"})

    result = instrument_node("critic", critic_module.critic)({
        **_critic_state("needs_revision", None),
        "task_type": "anomaly_analysis",
        "answer": "draft",
        "rag_result": {},
        "rag_review": {"status": "approved"},
        "tool_result": {},
    })

    assert result["critic_revision_target"] == CriticRevisionTarget.GENERATE_ANSWER
    assert result["errors"] == []
    assert _route_from_critic(result) == "generate_answer"

"""AGENT_PLAN Phase 10: shared runtime protocol tests."""

from __future__ import annotations

import pytest

from app.agents.diagnostic_contracts import DiagnosticBudget, Observation as DiagnosticObservation, PolicyDecision as DiagnosticPolicyDecision
from app.agents.insight_contracts import InsightBudget, InsightObservation, PolicyVerdict
from app.agents.runtime import AgentRun, Budget, Observation, PolicyDecision, Termination, TraceEvent


def test_domain_budgets_inherit_shared_protocol_but_keep_own_vocabulary():
    diagnostic = DiagnosticBudget(max_rounds=6, max_tool_calls=8)
    insight = InsightBudget(max_rounds=3, max_queries=6)
    # Shared protocol fields
    assert diagnostic.max_rounds == 6 and insight.max_rounds == 3
    assert diagnostic.max_tokens == 12000 and insight.max_tokens == 12000
    assert diagnostic.max_cost == insight.max_cost == 1.0
    # Domain vocabulary stays distinct — cross-domain construction fails
    assert hasattr(diagnostic, "max_tool_calls") and not hasattr(insight, "max_tool_calls")
    assert hasattr(insight, "max_queries") and not hasattr(diagnostic, "max_queries")
    assert not hasattr(diagnostic, "max_actions")  # base does not leak vocabulary
    with pytest.raises(Exception):
        DiagnosticBudget(max_queries=6)  # insight vocabulary rejected (extra=forbid)
    with pytest.raises(Exception):
        InsightBudget(max_tool_calls=8)  # diagnostic vocabulary rejected


def test_domain_observations_narrow_the_shared_action_type():
    diagnostic = DiagnosticObservation(observation_id="o1", action_type="CALL_TOOL")
    insight = InsightObservation(observation_id="o2", action_type="QUERY_EVIDENCE")
    assert diagnostic.summary == "" and insight.evidence_ids == []
    with pytest.raises(Exception):
        DiagnosticObservation(observation_id="o3", action_type="QUERY_EVIDENCE")
    with pytest.raises(Exception):
        InsightObservation(observation_id="o4", action_type="CALL_TOOL")


def test_policy_decisions_narrow_action_type_to_domain_vocabulary():
    assert DiagnosticPolicyDecision(decision="allow", reason="ok", action_type="CALL_TOOL").decision == "allow"
    assert PolicyVerdict(decision="deny", reason="outside allowlist", action_type="QUERY_EVIDENCE").decision == "deny"
    with pytest.raises(Exception):
        DiagnosticPolicyDecision(decision="allow", reason="ok", action_type="QUERY_EVIDENCE")


def test_agent_run_records_termination_and_trace_for_audit():
    run = AgentRun(
        run_id="trace-1", domain="diagnostic", decision_driver="llm", status="finished",
        rounds=3, tool_calls=2, input_tokens=100, output_tokens=50, estimated_cost=0.01,
        termination=Termination(initiator="agent", reason="evidence sufficient"),
        trace=[TraceEvent(node="diagnostic_plan", summary="Plan with 2 steps.")],
    )
    assert run.termination.initiator == "agent"
    assert run.trace[0].node == "diagnostic_plan"
    with pytest.raises(Exception):
        AgentRun(
            run_id="t", domain="diagnostic", decision_driver="llm", status="finished",
            termination=Termination(initiator="spontaneous"),  # not a legal initiator
        )


def test_diagnostic_loop_writes_agent_run_record():
    from app.graph.nodes.diagnostic import diagnostic_finalize

    state = {
        "diagnostic_action": {"action_type": "FINISH", "conclusion": "supported", "evidence_ids": ["ev1"]},
        "evidence_bundles": [{"evidence_id": "ev1", "trusted": True}],
        "finish_decision": {},
        "diagnostic_plan": {"goal": "g"},
        "diagnostic_round": 2, "tool_calls": [], "node_history": [{"node": "observe", "summary": "x"}],
        "decision_driver": "llm", "termination_initiator": "agent",
        "trace_id": "run-record-test", "llm_usage": {"input_tokens": 10, "output_tokens": 5, "estimated_cost": 0.001},
        "diagnostic_target_metric": "gmv",
        "investigation_state": {
            "target": {"metric_id": "gmv", "scope_fingerprint": "scope-run-record"},
            "evidence": [{
                "evidence_id": "ev1", "source_action_id": "candidate-1",
                "scope_fingerprint": "scope-run-record", "metric_id": "gmv",
                "method_id": "observed_trend_v1", "trace_id": "run-record-test",
            }],
            "action_results": [{
                "candidate_id": "candidate-1", "status": "completed",
                "validation_result": {"status": "valid"},
                "evidence_ids": ["ev1"], "metric_id": "gmv",
                "method_id": "observed_trend_v1",
            }],
            "branches": [], "progress": {}, "stop_reason": None,
        },
        "analysis_result": {"summary": "supported", "abstain": False},
    }
    result = diagnostic_finalize(state)
    run = result["agent_run"]
    assert run["domain"] == "diagnostic"
    assert run["status"] == "finished"
    assert run["decision_driver"] == "llm"
    assert run["rounds"] == 2
    assert run["termination"]["initiator"] == "agent"
    assert run["input_tokens"] == 10 and run["output_tokens"] == 5
    assert run["trace"][0]["node"] == "observe"


def test_code_agent_run_can_be_recorded_at_worker_trace_boundary():
    from app.agents.runtime import make_agent_run

    run = make_agent_run(
        run_id="trace-code-1", domain="code", decision_driver="workflow_fallback",
        status="finished", termination_initiator="agent", rounds=1,
        tool_calls=1, trace=[{"node": "sandbox_execute", "summary": "completed"}],
    )
    assert run.domain == "code"
    assert run.trace[0].node == "sandbox_execute"

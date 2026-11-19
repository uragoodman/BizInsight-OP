"""Terminal workflow behavior after the shared LLM token budget closes."""

from types import SimpleNamespace

from app.graph import workflow
from app.graph.nodes.answer import generate_answer
from app.graph.nodes.data_query import data_query
from app.graph.nodes.diagnostic import diagnostic_finalize
from app.core.execution_context import activate_state, reset_state
from app.llm import client as llm
from app.investigation.contracts import (
    ActionResult,
    AnalysisTarget,
    AttributionEvidence,
    InvestigationState,
)
from app.mcp.schemas import MetricPlanningDecision, MetricQueryBatchPlan


def test_default_request_token_budget_supports_multi_step_diagnostics():
    from app.core.config import Settings

    assert Settings(_env_file=None).max_tokens == 60_000


def _state_with_verified_evidence():
    state = workflow.init_state({
        "trace_id": "trace-token-budget-stop",
        "tenant_id": "tenant-test",
        "user_id": "user-test",
        "user_message": "Why did GMV change?",
        "task_type": "anomaly_analysis",
        "runtime_budget": {"max_tokens": 10, "max_cost": 0, "max_graph_steps": 48},
        "errors": [{
            "code": "token_budget_exceeded",
            "message": "LLM token budget exceeded.",
            "retryable": False,
            "agent_name": "diagnostic_plan",
        }],
        # Simulate a stale FINISH from the node that made the over-budget call.
        # The terminal budget policy must replace it before validating citations.
        "diagnostic_action": {
            "action_type": "FINISH",
            "conclusion": "A conclusion that cannot be completed after the budget stop.",
            "evidence_ids": ["not-generated-after-budget"],
            "residual_uncertainty": "",
        },
        "diagnostic_target_metric": "gmv",
        "tool_calls": [{"tool_name": "query_scoped_metric", "status": "ok"}],
        "llm_calls": [{"call_id": "existing-call", "outcome": "processing_error", "attempts": []}],
    })
    scope_fingerprint = "scope-fingerprint-test"
    target = AnalysisTarget(
        metric_id="gmv",
        current_start="2026-10-01",
        current_end="2026-10-01",
        canonical_scope={"tenant_id": "tenant-test"},
        scope_fingerprint=scope_fingerprint,
        requested_analysis_type="diagnostic",
    )
    evidence = AttributionEvidence(
        evidence_id="ev-prior-verified",
        trace_id="trace-token-budget-stop",
        evidence_type="trend",
        metric_id="gmv",
        claim="Previously verified in-scope GMV trend.",
        values={"current_value": 80, "baseline_value": 100},
        unit="BRL",
        source_action_id="candidate-prior-query",
        source_ref="query_scoped_trend",
        scope_fingerprint=scope_fingerprint,
        method_id="registered_trend_v1",
        method_version="1.0",
        authorized_metric_provenance="registered_metric",
    )
    investigation = InvestigationState(
        target=target,
        evidence=[evidence],
        action_results=[ActionResult(
            action_id="action-prior-query",
            candidate_id="candidate-prior-query",
            action_type="trend_query",
            metric_id="gmv",
            method_id="registered_trend_v1",
            status="completed",
            validation_result={"status": "valid"},
            evidence_ids=["ev-prior-verified"],
        )],
    )
    state["investigation_state"] = investigation.model_dump(mode="json")
    state["evidence_bundles"] = [{
        "evidence_id": "ev-prior-verified",
        "source_type": "root_cause",
        "source_ref": "root_cause:query_scoped_trend",
        "trusted": True,
        "content": {"summary": "Previously verified trend evidence."},
    }]
    return state


def test_budget_exhaustion_routes_to_terminal_abstention_and_preserves_verified_evidence(monkeypatch):
    state = _state_with_verified_evidence()
    model_call_count = len(state["llm_calls"])
    tool_call_count = len(state["tool_calls"])

    assert workflow._route_after_diagnostic_plan(state) == "diagnostic_finalize"
    assert workflow._route_after_observe(state) == "diagnostic_finalize"
    assert workflow._route_after_diagnostic_decide(state) == "diagnostic_finalize"
    assert workflow._route_after_data_query(state) == "diagnostic_finalize"
    assert workflow._route_after_premise_check(state) == "diagnostic_finalize"

    finalized = diagnostic_finalize(state)
    assert finalized["diagnostic_action"]["action_type"] == "ABSTAIN"
    assert finalized["diagnostic_action"]["reason"] == "budget_exhausted"
    assert finalized["diagnostic_action"]["supporting_evidence_ids"] == ["ev-prior-verified"]
    assert finalized["analysis_result"]["abstain"] is True
    assert finalized["analysis_result"]["evidence_ids"] == ["ev-prior-verified"]
    assert finalized["analysis_result"]["investigation_summary"]["stop_reason"]["code"] == "BUDGET_EXHAUSTED"
    assert not any(error["code"] == "diagnostic_evidence_reference_invalid" for error in finalized["errors"])
    assert finalized["llm_calls"] and len(finalized["llm_calls"]) == model_call_count
    assert len(finalized["tool_calls"]) == tool_call_count

    answered = generate_answer(finalized)
    presentation = answered["analysis_presentation"]
    assert presentation["status"] == "partial"
    assert "Token" in answered["answer"] or "token" in answered["answer"]
    assert len(answered["llm_calls"]) == model_call_count
    assert len(answered["tool_calls"]) == tool_call_count
    assert answered["analysis_result"]["evidence_ids"] == ["ev-prior-verified"]


def test_budget_route_ends_non_diagnostic_work_before_critic_or_more_queries():
    state = workflow.init_state({
        "task_type": "metric_query",
        "errors": [{"code": "token_budget_exceeded", "retryable": False}],
    })
    assert workflow._route_from_triage(state) == "generate_answer"
    assert workflow._route_after_metric_definition(state) == "generate_answer"
    assert workflow._route_after_data_query(state) == "generate_answer"
    assert workflow._route_after_premise_check(state) == "generate_answer"


def test_data_query_does_not_execute_sql_when_planning_exhausts_token_budget(monkeypatch):
    state = {
        "trace_id": "trace-budget-before-sql",
        "user_message": "查询 2018-08-10 的 GMV",
        "task_type": "metric_query",
        "runtime_budget": {"max_tokens": 10, "max_tool_calls": 5},
        "errors": [],
        "tool_calls": [],
        "node_history": [],
        "controlled_analysis_scope": None,
        "use_current_explorer_context": False,
        "tenant_id": "tenant-test",
        "user_id": "user-test",
        "user_role": "admin",
    }
    sql_calls = []

    sdk_calls = []

    class OfflineSdk:
        def create(self, **_kwargs):
            sdk_calls.append("sent")
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content='{"fallback":"ignored"}'),
                    finish_reason="stop",
                )],
                usage=SimpleNamespace(prompt_tokens=7, completion_tokens=4),
            )

    sdk = OfflineSdk()
    provider_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=sdk.create)),
    )
    provider = llm._Provider(
        "offline", provider_client,
        lambda _operation, _explicit, _quality: ("offline-model", "standard"),
    )
    monkeypatch.setattr(llm, "_providers", lambda: [provider])
    monkeypatch.setattr(llm, "_quota_available", lambda: True)
    monkeypatch.setattr(llm, "execute_with_resilience", lambda operation, *_args, **_kwargs: operation())
    monkeypatch.setattr(llm, "record_usage", lambda **_kwargs: None)

    def planner(*_args, **_kwargs):
        # Exercise the public SDK path. The call must be refused before SDK
        # entry because the prompt + reserved output cannot fit this turn cap.
        assert llm.llm_extract("system", "planner input") is None
        return MetricQueryBatchPlan(decision=MetricPlanningDecision(
            status="confirmed", confirmed_metrics=["gmv"],
        ))

    monkeypatch.setattr("app.graph.nodes.data_query.plan_confirmed_metric_queries", planner)
    monkeypatch.setattr("app.graph.nodes.data_query.query_business_data", lambda **kwargs: sql_calls.append(kwargs))
    monkeypatch.setattr("app.graph.nodes.data_query.query_scoped_business_data", lambda **kwargs: sql_calls.append(kwargs))

    context_token = activate_state(state)
    try:
        result = data_query(state)
    finally:
        reset_state(context_token)

    assert sdk_calls == []
    assert result["llm_calls"][0]["attempts"] == []
    assert result["llm_calls"][0]["outcome"] == "not_sent"
    assert result["llm_calls"][0]["not_sent_reason"] == "token_budget_preflight"
    assert sql_calls == []
    assert result["tool_calls"] == []
    assert result["metric_planning_decision"]["status"] == "confirmed"
    assert result["node_history"][-1]["summary"] == "Stopped before database execution: LLM token budget exhausted."

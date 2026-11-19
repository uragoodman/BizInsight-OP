from __future__ import annotations

from datetime import date

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.core.config import Settings
from app.graph.workflow import run_chat_workflow
from app.scope.seller_scope import (
    SellerScopeIntent,
    issue_seller_scope_payload,
    resolve_seller_scope,
    verify_seller_scope_payload,
)
from app.tenancy import SellerGrant, TenantContext


@pytest.mark.parametrize(
    "planned_query_type,planned_dimension,expected_handler_calls",
    [
        ("metric_breakdown", "seller", 1),
        ("metric_value", None, 1),
        ("metric_breakdown", "customer_city", 0),
    ],
)
def test_run_chat_workflow_executes_real_data_query_with_signed_seller_scope(
    monkeypatch, planned_query_type, planned_dimension, expected_handler_calls,
) -> None:
    """Keep the graph and data_query real; replace only the external query boundary."""
    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: Settings(
            tenant_auth_secret="real-data-query-workflow-secret",
            seller_dimension_enabled=True,
            seller_multi_scope_enabled=True,
        ),
    )
    context = TenantContext(
        tenant_id="tenant-real-data-query", user_id="user-real-data-query",
        role="admin", scope_type="platform",
        seller_grants=(SellerGrant("A", "2024-01-01", "2024-01-31"),),
    )
    intent = SellerScopeIntent(
        seller_scope_type="selected", seller_ids=("A",),
        start_date=date(2024, 1, 1), end_date=date(2024, 1, 31),
        metric_ids=("gmv",), dimensions=("seller",),
    )
    signed_payload = issue_seller_scope_payload(intent, tenant=context)
    question = (
        "GMV"
        if planned_query_type == "metric_value"
        else "按客户城市拆解 GMV"
        if planned_dimension == "customer_city"
        else "按 seller 拆解 GMV"
    )
    observed: dict[str, object] = {
        "planner_scope": None,
        "planner_scope_data": None,
        "handler_calls": [],
        "platform_calls": 0,
        "report_calls": 0,
    }

    def passthrough(state):
        return state

    def compile_context(state):
        state["context_compilation"] = {
            "needs_clarification": False,
            "standalone_question": state["user_message"],
            "known_slots": {},
            "operation_intent": {"kind": "ordinary_metric_query", "source": "current_turn"},
        }
        return state

    def classify(state):
        state["task_type"] = "metric_query"
        state["next_agent"] = "data_query"
        state.setdefault("handoffs", []).append({
            "from": "triage", "to": "data_query", "reason": "test seller metric query",
        })
        state.setdefault("node_history", []).append({"node": "triage", "summary": "metric_query"})
        return state

    def fake_extract(_prompt, _content, **_kwargs):
        return {
            "status": "confirmed",
            "confirmed_metrics": ["gmv"],
            "query_type": planned_query_type,
            "dimension": planned_dimension,
            "granularity": "day",
            "confidence": 1.0,
            "reason_code": "exact_match",
            "reason": "Deterministic test planner response for the actual registry planner.",
        }

    import app.graph.workflow as workflow
    import app.graph.nodes.data_query as data_query_module

    actual_planner = data_query_module.plan_confirmed_metric_queries

    def record_planner_input(question, **kwargs):
        observed["planner_scope"] = kwargs.get("scope")
        observed["planner_scope_data"] = dict(kwargs.get("scope_data") or {})
        return actual_planner(question, **kwargs)

    def fake_scoped_query(question, received_payload, *, request_id=None, plan=None, **_kwargs):
        from app.auth.context import build_authorization_context

        assert received_payload == signed_payload
        verified_intent = verify_seller_scope_payload(received_payload, tenant=context)
        authz = build_authorization_context(context, metric_scope_type="platform")
        canonical_scope = resolve_seller_scope(verified_intent, tenant=context, authz=authz)
        calls = observed["handler_calls"]
        assert isinstance(calls, list)
        calls.append({
            "question": question,
            "payload": received_payload,
            "plan": plan.model_dump(mode="json") if plan is not None else None,
            "scope": canonical_scope.as_query_dict(),
        })
        boundary = {
            "permission_version": authz.permission_version,
            "policy_set_version": authz.policy_set_version,
            "authorization_fingerprint": authz.authz_digest,
            "query_scope_fingerprint": canonical_scope.query_scope_fingerprint,
        }
        if plan is None or plan.query_type != "metric_breakdown" or plan.dimension != "seller":
            return {
                "server": "seller-scope-fake", "tool_name": "breakdown_scoped_seller",
                "arguments": {"dimension": getattr(plan, "dimension", None)},
                "request_id": request_id or "fake-request",
                "seller_boundary": boundary,
                "seller_scope_fingerprint": canonical_scope.query_scope_fingerprint,
                "result": {"status": "unsupported", "parsed": {}, "reason_code": "unsupported_seller_operation"},
            }
        return {
            "server": "seller-scope-fake", "tool_name": "breakdown_scoped_seller",
            "arguments": {"dimension": "seller", "metric_name": plan.metric_name},
            "request_id": request_id or "fake-request",
            "seller_boundary": boundary,
            "seller_scope_fingerprint": canonical_scope.query_scope_fingerprint,
            "result": {
                "status": "ok",
                "parsed": {
                    "metric_name": plan.metric_name,
                    "dimension": "seller",
                    "total": {"gmv": 100, "order_count": 1},
                    "items": [{"seller_id": "A", "value": 100, "order_count": 1}],
                    "unattributed": None,
                },
            },
        }

    def unexpected_platform_query(**_kwargs):
        observed["platform_calls"] = int(observed["platform_calls"]) + 1
        pytest.fail("seller-scoped data_query must not call platform query_business_data")

    def unexpected_report(state):
        observed["report_calls"] = int(observed["report_calls"]) + 1
        pytest.fail("metric query must not enter the platform report builder")

    def answer(state):
        state["answer"] = "seller scoped controlled result"
        state["claims"] = []
        return state

    def persist_trace(state):
        observed["final_state"] = dict(state)
        return state

    def approve(state, **_kwargs):
        state["critic_verdict"] = "approved"
        state["critic_revision_target"] = None
        state["critic_decisions"] = [{"llm_issues": []}]
        return state

    class EmptyContextOrchestrator:
        def build_specialist_bundle(self, *_args, **_kwargs):
            return object()

        def packer_items(self, _bundle):
            return []

    monkeypatch.setattr(workflow, "get_checkpointer", lambda: MemorySaver())
    monkeypatch.setattr(workflow, "resolve_explorer_context", passthrough)
    monkeypatch.setattr(workflow, "load_session_memory", passthrough)
    monkeypatch.setattr(workflow, "compile_turn_context", compile_context)
    monkeypatch.setattr(workflow, "triage", classify)
    monkeypatch.setattr(workflow, "retrieve_analysis_memories", passthrough)
    monkeypatch.setattr(workflow, "generate_answer", answer)
    monkeypatch.setattr(workflow, "critic", approve)
    monkeypatch.setattr(workflow, "report", unexpected_report)
    monkeypatch.setattr(workflow, "persist_trace", persist_trace)
    monkeypatch.setattr(workflow, "persist_memory_p0", passthrough)
    monkeypatch.setattr(data_query_module, "ContextOrchestrator", EmptyContextOrchestrator)
    monkeypatch.setattr(data_query_module, "plan_confirmed_metric_queries", record_planner_input)
    monkeypatch.setattr("app.llm.client.llm_extract", fake_extract)
    monkeypatch.setattr(data_query_module, "query_scoped_business_data", fake_scoped_query)
    monkeypatch.setattr(data_query_module, "query_business_data", unexpected_platform_query)

    result = run_chat_workflow(
        question,
        tenant_context=context,
        memory_enabled=False,
        seller_scope_payload=signed_payload,
    )

    assert result["error"] is False
    assert result["task_type"] == "metric_query"
    assert int(observed["platform_calls"]) == 0
    assert int(observed["report_calls"]) == 0
    scope_data = observed["planner_scope_data"]
    assert isinstance(scope_data, dict)
    assert observed["planner_scope"] == "platform"
    assert scope_data["seller_scope_type"] == "selected"
    assert scope_data["start_date"] == "2024-01-01"
    assert scope_data["end_date"] == "2024-01-31"
    assert scope_data["allowed_metrics"] == ["gmv"]
    assert scope_data["allowed_dimensions"] == ["seller"]
    assert isinstance(scope_data["query_scope_fingerprint"], str)
    assert len(scope_data["query_scope_fingerprint"]) == 64
    assert "seller_ids" not in scope_data
    assert "seller_scope_payload" not in scope_data
    assert signed_payload["provenance"]["token"] not in str(scope_data)

    handler_calls = observed["handler_calls"]
    assert isinstance(handler_calls, list)
    assert len(handler_calls) == expected_handler_calls
    if planned_query_type == "metric_breakdown" and planned_dimension == "seller":
        assert handler_calls[0]["payload"] == signed_payload
        assert handler_calls[0]["plan"]["query_type"] == "metric_breakdown"
        assert handler_calls[0]["plan"]["dimension"] == "seller"
        assert handler_calls[0]["scope"]["seller_scope_type"] == "selected"
        assert result["answer"] == "seller scoped controlled result"
    else:
        final_state = observed["final_state"]
        assert isinstance(final_state, dict)
        assert final_state["tool_result"]["result"]["status"] in {"unsupported", "clarification_required"}

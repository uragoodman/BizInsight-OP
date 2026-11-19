from __future__ import annotations

from datetime import date

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.core.config import Settings
from app.graph.nodes.seller_scope_guard import SELLER_SCOPE_UNSUPPORTED_REASON
from app.graph.workflow import run_chat_workflow
from app.scope.seller_scope import SellerScopeIntent, issue_seller_scope_payload
from app.tenancy import SellerGrant, TenantContext


@pytest.mark.parametrize("task_type", ["anomaly_analysis", "report_generation"])
def test_run_chat_workflow_signed_seller_scope_cannot_reach_platform_fallback(
    monkeypatch, task_type: str,
) -> None:
    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: Settings(
            tenant_auth_secret="workflow-seller-scope-secret",
            seller_dimension_enabled=True,
            seller_multi_scope_enabled=True,
        ),
    )
    context = TenantContext(
        tenant_id="tenant-workflow-seller", user_id="user-workflow-seller",
        role="admin", scope_type="platform",
        seller_grants=(SellerGrant("A", "2024-01-01", "2024-01-31"),),
    )
    intent = SellerScopeIntent(
        seller_scope_type="selected", seller_ids=("A",),
        start_date=date(2024, 1, 1), end_date=date(2024, 1, 31),
        metric_ids=("gmv",), dimensions=("seller",),
    )
    signed_payload = issue_seller_scope_payload(intent, tenant=context)
    observed = {"memory": 0, "query": 0, "report": 0}

    def passthrough(state):
        return state

    def compile_context(state):
        state["context_compilation"] = {}
        return state

    def classify(state):
        state["task_type"] = task_type
        state["next_agent"] = "data_query"
        state.setdefault("handoffs", []).append({
            "from": "triage", "to": "data_query", "reason": "test classification",
        })
        state.setdefault("node_history", []).append({"node": "triage", "summary": task_type})
        return state

    def unexpected_memory(state):
        observed["memory"] += 1
        pytest.fail("seller scope must bypass generic analysis memory retrieval")

    def unexpected_query(state):
        observed["query"] += 1
        pytest.fail("unsupported seller operation reached data_query")

    def unexpected_report(state):
        observed["report"] += 1
        pytest.fail("seller scope reached the platform report builder")

    import app.graph.workflow as workflow

    monkeypatch.setattr(workflow, "get_checkpointer", lambda: MemorySaver())
    monkeypatch.setattr(workflow, "load_session_memory", passthrough)
    monkeypatch.setattr(workflow, "compile_turn_context", compile_context)
    monkeypatch.setattr(workflow, "triage", classify)
    monkeypatch.setattr(workflow, "retrieve_analysis_memories", unexpected_memory)
    monkeypatch.setattr(workflow, "data_query", unexpected_query)
    monkeypatch.setattr(workflow, "report", unexpected_report)
    monkeypatch.setattr(workflow, "persist_trace", passthrough)
    monkeypatch.setattr(workflow, "persist_memory_p0", passthrough)

    result = run_chat_workflow(
        "分析 seller GMV 并生成报告",
        tenant_context=context,
        memory_enabled=False,
        seller_scope_payload=signed_payload,
    )

    assert result["answer"] == SELLER_SCOPE_UNSUPPORTED_REASON
    assert result["error"] is False
    assert observed == {"memory": 0, "query": 0, "report": 0}


@pytest.mark.parametrize(
    "reason,answer_fragment",
    [
        ("ambiguous_dimension", "维度"),
        ("ambiguous_metric", "指标"),
    ],
)
def test_run_chat_workflow_signed_seller_ambiguity_stops_at_clarification(
    monkeypatch, reason, answer_fragment,
) -> None:
    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: Settings(tenant_auth_secret="workflow-seller-scope-secret"),
    )
    context = TenantContext(
        tenant_id="tenant-workflow-ambiguous", user_id="user-workflow-ambiguous",
        role="admin", scope_type="platform",
        seller_grants=(SellerGrant("A", "2024-01-01", "2024-01-31"),),
    )
    payload = issue_seller_scope_payload(SellerScopeIntent(
        seller_scope_type="selected", seller_ids=("A",),
        start_date=date(2024, 1, 1), end_date=date(2024, 1, 31),
        metric_ids=("gmv",), dimensions=("seller",),
    ), tenant=context)

    def passthrough(state):
        return state

    def compile_ambiguous(state):
        state["context_compilation"] = {
            "needs_clarification": True,
            "clarification_reason": reason,
            "missing_slots": ["dimension" if reason == "ambiguous_dimension" else "metric"],
        }
        return state

    def classify(state):
        state["task_type"] = "metric_query"
        state["next_agent"] = "data_query"
        state.setdefault("handoffs", []).append({
            "from": "triage", "to": "data_query", "reason": "ambiguous seller query",
        })
        state.setdefault("node_history", []).append({"node": "triage", "summary": "metric_query"})
        return state

    def unexpected_query(_state):
        pytest.fail("ambiguous seller query must not invoke data_query")

    import app.graph.workflow as workflow

    monkeypatch.setattr(workflow, "get_checkpointer", lambda: MemorySaver())
    monkeypatch.setattr(workflow, "load_session_memory", passthrough)
    monkeypatch.setattr(workflow, "compile_turn_context", compile_ambiguous)
    monkeypatch.setattr(workflow, "triage", classify)
    monkeypatch.setattr(workflow, "retrieve_analysis_memories", unexpected_query)
    monkeypatch.setattr(workflow, "data_query", unexpected_query)
    monkeypatch.setattr(workflow, "persist_trace", passthrough)
    monkeypatch.setattr(workflow, "persist_memory_p0", passthrough)

    result = run_chat_workflow(
        "按不明确的维度查询指标",
        tenant_context=context,
        memory_enabled=False,
        seller_scope_payload=payload,
    )

    assert result["task_type"] == "clarification"
    assert answer_fragment in result["answer"]
    assert result["error"] is False


def test_run_chat_workflow_signed_seller_metric_query_reaches_scoped_handler(monkeypatch) -> None:
    """Exercise the positive graph route with the signed scope still server-bound."""
    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: Settings(
            tenant_auth_secret="workflow-seller-scope-secret",
            seller_dimension_enabled=True,
            seller_multi_scope_enabled=True,
        ),
    )
    context = TenantContext(
        tenant_id="tenant-workflow-positive", user_id="user-workflow-positive",
        role="admin", scope_type="platform",
        seller_grants=(SellerGrant("A", "2024-01-01", "2024-01-31"),),
    )
    intent = SellerScopeIntent(
        seller_scope_type="selected", seller_ids=("A",),
        start_date=date(2024, 1, 1), end_date=date(2024, 1, 31),
        metric_ids=("gmv",), dimensions=("seller",),
    )
    signed_payload = issue_seller_scope_payload(intent, tenant=context)
    observed = {"scope_type": None, "fingerprint": None, "seller_ids_exposed": False}

    def passthrough(state):
        return state

    def compile_context(state):
        state["context_compilation"] = {}
        return state

    def classify(state):
        state["task_type"] = "metric_query"
        state["next_agent"] = "data_query"
        state.setdefault("handoffs", []).append({
            "from": "triage", "to": "data_query", "reason": "test classification",
        })
        state.setdefault("node_history", []).append({"node": "triage", "summary": "metric_query"})
        return state

    def execute_verified_scope(state):
        from app.agents.contracts import ToolInvocation
        from app.graph.nodes.data_query import _seller_scope_planning_context

        scope_type, planning_context, resolved_tenant = _seller_scope_planning_context(
            state["seller_scope_payload"],
        )
        observed["scope_type"] = scope_type
        observed["fingerprint"] = planning_context["query_scope_fingerprint"]
        observed["seller_ids_exposed"] = "seller_ids" in planning_context
        assert resolved_tenant is context
        invocation = ToolInvocation(
            server="tenant-mcp", tool_name="query_seller_breakdown",
            planned_metric_id="gmv", planned_operation="breakdown",
            arguments={"dimension": "seller", "metric_name": "gmv"}, status="ok",
        )
        state["tool_calls"] = [invocation.model_dump(mode="json")]
        state["tool_result"] = {
            "server": "tenant-mcp", "tool_name": "query_seller_breakdown",
            "arguments": {"dimension": "seller", "metric_name": "gmv"},
            "request_id": invocation.request_id,
            "result": {"status": "ok", "parsed": {"total": {"value": 100}, "items": []}},
        }
        return state

    def answer(state):
        state["answer"] = "seller-scoped result"
        state["claims"] = []
        return state

    def approve(state, **_kwargs):
        state["critic_verdict"] = "approved"
        state["critic_revision_target"] = None
        state["critic_decisions"] = [{"llm_issues": []}]
        return state

    import app.graph.workflow as workflow

    monkeypatch.setattr(workflow, "get_checkpointer", lambda: MemorySaver())
    monkeypatch.setattr(workflow, "load_session_memory", passthrough)
    monkeypatch.setattr(workflow, "compile_turn_context", compile_context)
    monkeypatch.setattr(workflow, "triage", classify)
    monkeypatch.setattr(workflow, "retrieve_analysis_memories", passthrough)
    monkeypatch.setattr(workflow, "data_query", execute_verified_scope)
    monkeypatch.setattr(workflow, "generate_answer", answer)
    monkeypatch.setattr(workflow, "critic", approve)
    monkeypatch.setattr(workflow, "resolve_explorer_context", passthrough)
    monkeypatch.setattr(workflow, "persist_trace", passthrough)
    monkeypatch.setattr(workflow, "persist_memory_p0", passthrough)
    monkeypatch.setattr("app.graph.nodes.data_query.current_tenant_context", lambda: context)

    result = run_chat_workflow(
        "查询所选商户的 GMV",
        tenant_context=context,
        memory_enabled=False,
        seller_scope_payload=signed_payload,
    )

    assert result["task_type"] == "metric_query"
    assert result["answer"] == "seller-scoped result"
    assert result["error"] is False
    assert observed["scope_type"] == "platform"
    assert isinstance(observed["fingerprint"], str) and len(observed["fingerprint"]) == 64
    assert observed["seller_ids_exposed"] is False

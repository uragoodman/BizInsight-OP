from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.graph import workflow
from app.tenancy import TenantContext


def test_stream_graph_interrupt_is_reported_as_suspended_not_failed(monkeypatch):
    builder = StateGraph(dict)

    def premise_check(_state):
        return {
            "task_type": "anomaly_analysis",
            "answer": "核对数据后，趋势方向与问题前提不一致。需要按实际方向继续分析吗？",
            "trace_id": "trace-premise-interrupt",
            "awaiting_user_confirmation": True,
            "premise_check": {"status": "mismatch"},
        }

    def premise_confirmation(_state):
        interrupt({"type": "premise_mismatch", "question": "确认后继续"})
        return {}

    builder.add_node("premise_check", premise_check)
    builder.add_node("premise_confirmation", premise_confirmation)
    builder.add_edge(START, "premise_check")
    builder.add_edge("premise_check", "premise_confirmation")
    builder.add_edge("premise_confirmation", END)
    compiled_graph = builder.compile(checkpointer=MemorySaver())

    monkeypatch.setattr(workflow, "build_chat_graph", lambda **_kwargs: compiled_graph)
    monkeypatch.setattr(workflow, "_pending_premise", lambda *_args: None)
    monkeypatch.setattr(workflow, "_task_progress_context", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(workflow, "_previous_turn_context", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(workflow, "_persist_chat_turn", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(workflow, "chat_context_scope_fingerprint", lambda *_args: "scope-fingerprint")
    monkeypatch.setattr(workflow, "activate_tenant_context", lambda _context: nullcontext())
    monkeypatch.setattr(workflow, "span", lambda *_args, **_kwargs: nullcontext())
    persisted_traces: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        workflow,
        "write_trace",
        lambda trace_id, payload: persisted_traces.append((trace_id, payload)),
    )

    events = list(workflow.stream_chat_workflow(
        "为什么 GMV 方向与预期不同？",
        thread_id="premise-interrupt-thread",
        tenant_context=TenantContext(
            tenant_id="platform-test", user_id="admin-test", role="admin", scope_type="platform",
        ),
        trace_id="trace-premise-interrupt",
    ))

    assert [event["event"] for event in events] == ["started", "node_completed", "suspended"]
    assert events[-1]["interaction_state"] == "awaiting_human_input"
    assert events[-1]["interaction_reason_code"] == "premise_direction_mismatch"
    assert events[-1]["task_type"] == "anomaly_analysis"
    assert events[-1]["answer"].startswith("核对数据后")
    assert events[-1]["trace_id"] == "trace-premise-interrupt"
    assert events[-1]["scope_fingerprint"] == "scope-fingerprint"
    assert len(persisted_traces) == 1
    assert persisted_traces[0][0] == "trace-premise-interrupt"
    assert persisted_traces[0][1]["_status"] == "suspended"
    assert persisted_traces[0][1]["answer"] == events[-1]["answer"]


def test_stream_unsupported_capability_is_completed_without_clarification(monkeypatch):
    builder = StateGraph(dict)

    def unsupported_capability(_state):
        return {
            "task_type": "anomaly_analysis",
            "answer": "该比较期间的品类贡献方法尚未注册，未执行分组查询。",
            "trace_id": "trace-capability-boundary",
            "metric_planning_decision": {
                "status": "unsupported",
                "reason_code": "period_contribution_not_registered",
            },
        }

    builder.add_node("unsupported_capability", unsupported_capability)
    builder.add_edge(START, "unsupported_capability")
    builder.add_edge("unsupported_capability", END)
    compiled_graph = builder.compile(checkpointer=MemorySaver())

    monkeypatch.setattr(workflow, "build_chat_graph", lambda **_kwargs: compiled_graph)
    monkeypatch.setattr(workflow, "_pending_premise", lambda *_args: None)
    monkeypatch.setattr(workflow, "_task_progress_context", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(workflow, "_previous_turn_context", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(workflow, "_persist_chat_turn", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(workflow, "_notify_result_context", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workflow, "chat_context_scope_fingerprint", lambda *_args: "scope-fingerprint")
    monkeypatch.setattr(workflow, "activate_tenant_context", lambda _context: nullcontext())
    monkeypatch.setattr(workflow, "span", lambda *_args, **_kwargs: nullcontext())

    events = list(workflow.stream_chat_workflow(
        "再按商品品类解释这个变化的统计贡献。",
        thread_id="unsupported-capability-thread",
        tenant_context=TenantContext(
            tenant_id="platform-test", user_id="admin-test", role="admin", scope_type="platform",
        ),
        trace_id="trace-capability-boundary",
    ))

    assert events[-1]["event"] == "completed"
    assert events[-1]["clarification"] is None
    assert events[-1]["terminal_status"] == "abstained"
    assert events[-1]["answer"].startswith("该比较期间")


def test_stream_completed_clarification_includes_only_safe_template_choices(monkeypatch):
    from app.analysis.templates import registry

    builder = StateGraph(dict)

    def ambiguous_route(_state):
        return {
            "task_type": "complex_analysis",
            "answer": "请选择一种分析方法。",
            "trace_id": "trace-retention-choice",
            "complex_clarification": {
                "status": "complex_template_required",
                "reason_code": "multiple_capabilities",
                "candidate_templates": [
                    {"template_id": "cohort_retention", "display_name": "伪造名称"},
                    {"template_id": "customer_lifecycle", "display_name": "伪造名称"},
                    {"template_id": "unknown_template", "display_name": "不可见模板"},
                ],
                "original_question": "must not be emitted",
            },
        }

    builder.add_node("ambiguous_route", ambiguous_route)
    builder.add_edge(START, "ambiguous_route")
    builder.add_edge("ambiguous_route", END)
    compiled_graph = builder.compile(checkpointer=MemorySaver())
    monkeypatch.setattr(workflow, "build_chat_graph", lambda **_kwargs: compiled_graph)
    monkeypatch.setattr(workflow, "_pending_premise", lambda *_args: None)
    monkeypatch.setattr(workflow, "_task_progress_context", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(workflow, "_previous_turn_context", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(workflow, "_persist_chat_turn", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(workflow, "_notify_result_context", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workflow, "chat_context_scope_fingerprint", lambda *_args: "scope-fingerprint")
    monkeypatch.setattr(workflow, "activate_tenant_context", lambda _context: nullcontext())
    monkeypatch.setattr(workflow, "span", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(registry, "list_chat_visible_templates", lambda: [
        {"name": "cohort_retention", "display_name": "Cohort 留存分析"},
        {"name": "customer_lifecycle", "display_name": "客户生命周期分析"},
    ])

    events = list(workflow.stream_chat_workflow(
        "客户留存分析",
        thread_id="retention-choice-thread",
        tenant_context=TenantContext(
            tenant_id="platform-test", user_id="admin-test", role="admin", scope_type="platform",
        ),
        trace_id="trace-retention-choice",
    ))

    final = events[-1]
    assert final["event"] == "completed"
    assert final["clarification"]["status"] == "complex_template_required"
    assert final["clarification"]["candidate_templates"] == [
        {"template_id": "cohort_retention", "display_name": "Cohort 留存分析"},
        {"template_id": "customer_lifecycle", "display_name": "客户生命周期分析"},
    ]
    assert "original_question" not in final["clarification"]


def test_pending_premise_is_bound_to_same_identity_scope_and_authz(monkeypatch):
    context = TenantContext(
        tenant_id="tenant-a", user_id="user-a", role="admin", scope_type="platform",
    )
    boundary = {
        "permission_version": "permission-v1",
        "policy_set_version": "policy-v1",
        "authorization_fingerprint": "authz-v1",
        "query_scope_fingerprint": "scope-v1",
    }
    saved = {
        "awaiting_user_confirmation": True,
        "premise_check": {"status": "mismatch"},
        "tenant_id": "tenant-a",
        "user_id": "user-a",
        "chat_session_id": "session-a",
        "current_query_scope_fingerprint": "scope-v1",
        "authorization_boundary": dict(boundary),
    }

    class FakeGraph:
        def get_state(self, _config):
            return SimpleNamespace(values=saved)

    graph = FakeGraph()
    config = {"configurable": {"thread_id": "thread-a"}}
    assert workflow._pending_premise(
        graph, config, context, "session-a", "scope-v1", boundary,
    ) is saved

    for session_id, scope, current_boundary, tenant in (
        ("session-b", "scope-v1", boundary, context),
        ("session-a", "scope-v2", boundary, context),
        ("session-a", "scope-v1", {**boundary, "permission_version": "permission-v2"}, context),
        ("session-a", "scope-v1", boundary,
         TenantContext(tenant_id="tenant-b", user_id="user-a", role="admin", scope_type="platform")),
        ("session-a", "scope-v1", boundary,
         TenantContext(tenant_id="tenant-a", user_id="user-b", role="admin", scope_type="platform")),
    ):
        with pytest.raises(PermissionError, match="another session or scope"):
            workflow._pending_premise(
                graph, config, tenant, session_id, scope, current_boundary,
            )


def test_legacy_suspended_turn_is_terminal_and_awaits_human_input(monkeypatch):
    saved: dict = {}

    class FakeHistoryRepository:
        def __init__(self, _context):
            pass

        def save_turn(self, *_args, **kwargs):
            saved.update(kwargs)

    monkeypatch.setattr(workflow, "ChatHistoryRepository", FakeHistoryRepository)
    context = TenantContext(
        tenant_id="tenant-a", user_id="user-a", role="admin", scope_type="platform",
    )

    assert workflow._persist_chat_turn(
        context, "session-a", "turn-a", "为什么 GMV 下降？",
        {"_status": "suspended", "task_type": "anomaly_analysis", "answer": ""},
        None,
    ) is True

    assert saved["status"] == "interrupted"
    assert saved["interaction_state"] == "awaiting_human_input"


def test_unsupported_capability_turn_is_persisted_as_abstained(monkeypatch):
    saved: dict = {}

    class FakeHistoryRepository:
        def __init__(self, _context):
            pass

        def save_turn(self, *_args, **kwargs):
            saved.update(kwargs)

    monkeypatch.setattr(workflow, "ChatHistoryRepository", FakeHistoryRepository)
    context = TenantContext(
        tenant_id="tenant-a", user_id="user-a", role="admin", scope_type="platform",
    )

    assert workflow._persist_chat_turn(
        context, "session-a", "turn-a", "按支付方式拆分订单量。",
        {
            "task_type": "metric_query",
            "answer": "当前查询暂不支持按支付方式拆分。未执行数据查询。",
            "metric_planning_decision": {"status": "unsupported", "reason_code": "unsupported_dimension"},
            "analysis_presentation": {"status": "unsupported"},
        },
        None,
    ) is True

    assert saved["status"] == "abstained"
    assert saved["interaction_state"] == "none"

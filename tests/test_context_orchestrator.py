"""WP1 Context Orchestrator contract tests.

These tests do not call an LLM, database, Mem0 or Qdrant.  They exercise the
source selection, trust tiers, per-agent minimality, evidence refresh and
metadata-only trace behavior.
"""

from __future__ import annotations

import json

import pytest

from app.context.orchestrator import (
    SOURCE_TIER,
    ContextOrchestrator,
    OrchestratedItem,
    Source,
    TrustTier,
)
from app.context.packer import ContextBudgetExceeded, estimate_tokens


def _state(**overrides):
    state = {
        "trace_id": "trace-wp1",
        "task_id": "task-wp1",
        "current_query_scope_fingerprint": "scope-wp1",
        "user_message": "分析一下最近的情况",
        "context_compilation": {
            "standalone_question": "分析一下最近的情况",
            "known_slots": {"metric": {"value": "gmv"}},
            "needs_clarification": False,
            "relation": "independent",
            "task_progress_context_used": True,
        },
        "memory_context": "SECRET_MEMORY_TEXT 上一轮查看了 GMV",
        "session_memory_context": "SECRET_MEMORY_TEXT 上一轮查看了 GMV",
        "previous_turn_context": {
            "source_turn_id": "turn-prev",
            "context_slots": {"metric": {"value": "gmv"}, "date": {"value": "2018-08-10"}},
        },
        "task_progress_context": {
            "objective": "查看 GMV",
            "completed_steps": ["data_query"],
            "pending_questions": [],
            "scope_fingerprint": "scope-wp1",
            "latest_trace_id": "trace-prev",
        },
        "analysis_memories": [
            {
                "memory_id": "m1", "insight_text": "HISTORY_MEMORY",
                "limitations": "old", "evidence_ids": ["ev-1"],
                "source_trace_id": "trace-prev", "scope_fingerprint": "scope-wp1",
            }
        ],
        "metric_authoritative_definition": {
            "metric_id": "gmv", "definition_version": "v1", "formula": "sum(price)",
        },
        "rag_result": {"context": "RAG_DOCUMENT"},
        "rag_citations": [{"chunk_id": "chunk-1"}],
        "evidence_bundles": [
            {"evidence_id": "ev-1", "source_type": "mcp", "source_ref": "query_gmv", "trusted": True}
        ],
        "answer": "draft answer",
    }
    state.update(overrides)
    return state


def test_session_summary_does_not_mix_with_analysis_memory_text() -> None:
    state = _state(
        memory_context="COMBINED analysis-memory text",
        session_memory_context="SESSION SUMMARY ONLY",
    )
    bundle = ContextOrchestrator().build_specialist_bundle(
        state, "answer", instruction="q", model="test-model",
    )
    assert "SESSION SUMMARY ONLY" in bundle.text
    assert "COMBINED analysis-memory text" not in bundle.text


def test_session_open_questions_are_included_as_historical_context():
    state = _state(previous_turn_context={
        "source_turn_id": "turn-prev",
        "open_questions": [{"field": "date", "reason": "date_required", "source_turn_id": "turn-older"}],
        "open_questions_updated_at": "2026-11-01T00:00:00+00:00",
    })
    bundle = ContextOrchestrator().build_specialist_bundle(
        state, "triage", instruction="继续分析", model="test-model",
    )
    assert "ctx.session_open_questions" in bundle.included_item_ids
    assert '"field": "date"' in bundle.text
    assert "turn-older" in bundle.metadata()["items"][-1]["source_refs"] or any(
        "turn-older" in item.get("source_refs", []) for item in bundle.metadata()["items"]
    )


def test_turn_resolution_payload_is_persistable_and_referenced() -> None:
    from app.context.compiler import compile_turn_context

    state = _state(chat_session_id="session-wp1")
    compiled = compile_turn_context(state)
    resolution = compiled["turn_resolution"]
    assert resolution["schema_version"] == "turn-resolution-v1"
    assert resolution["turn_id"] == "trace-wp1"
    assert resolution["session_id"] == "session-wp1"
    assert resolution["resolved_text"]
    bundle = ContextOrchestrator().build_specialist_bundle(
        compiled, "triage", instruction="q", model="test-model",
    )
    assert bundle.resolved_request["storage"] == "chat_turns.context_snapshot.turn_resolution"
    assert bundle.resolved_request["turn_id"] == resolution["turn_id"]


def test_chat_turn_persistence_includes_turn_resolution(monkeypatch) -> None:
    import app.graph.workflow as workflow
    from app.tenancy import TenantContext

    captured = {}

    class FakeRepository:
        def __init__(self, _context):
            pass

        def save_turn(self, *args, **kwargs):
            captured["context_snapshot"] = kwargs.get("context_snapshot")

    monkeypatch.setattr(workflow, "ChatHistoryRepository", FakeRepository)
    context = TenantContext(tenant_id="tenant", user_id="user", role="admin", scope_type="platform")
    workflow._persist_chat_turn(
        context, "session", "client-turn", "question",
        {
            "task_type": "metric_query", "answer": "a", "trace_id": "trace",
            "turn_resolution": {"schema_version": "turn-resolution-v1", "turn_id": "turn1"},
        },
        {"scope": "x"}, None, authorization_boundary={}, context_scope_fingerprint="fp",
    )
    assert captured["context_snapshot"]["turn_resolution"]["turn_id"] == "turn1"


def test_legacy_scope_clarification_and_presentation_survive_history(monkeypatch) -> None:
    import app.graph.workflow as workflow
    from app.tenancy import TenantContext

    saved: dict[str, object] = {}

    class RecordingRepository:
        def __init__(self, _context):
            pass

        def save_turn(self, *args, **kwargs):
            saved.update(kwargs)

    monkeypatch.setattr(workflow, "ChatHistoryRepository", RecordingRepository)
    context = TenantContext(tenant_id="tenant", user_id="user", role="admin", scope_type="platform")

    assert workflow._persist_chat_turn(
        context, "session", "turn", "按 RFM 做分群",
        {
            "task_type": "complex_analysis",
            "answer": "请先选择店铺、开始日期和结束日期，再确认复杂分析计划。",
            "trace_id": "trace",
            "clarification": {
                "status": "complex_scope_required",
                "reason_code": "scope_required",
                "reason": "不得把任意 provider 文本保存到历史。",
                "original_message": "private question",
            },
            "presentation": {
                "status": "completed",
                "facts": [{"metric_name": "gmv", "value": 12, "date_or_range": "2018-08"}],
                "raw_sql": "SELECT private_data",
            },
        },
        None,
    )

    assert saved["status"] == "interrupted"
    assert saved["interaction_state"] == "awaiting_scope"
    snapshot = saved["result_snapshot"]
    assert snapshot["clarification"] == {
        "status": "complex_scope_required",
        "reason_code": "scope_required",
    }
    assert snapshot["presentation"]["facts"][0]["value"] == 12
    assert "raw_sql" not in snapshot["presentation"]


def test_durable_turn_commit_skips_the_legacy_best_effort_writer(monkeypatch) -> None:
    import app.graph.workflow as workflow
    from app.tenancy import TenantContext

    class ForbiddenRepository:
        def __init__(self, _context):
            raise AssertionError("durable turn must have a single lifecycle writer")

    monkeypatch.setattr(workflow, "ChatHistoryRepository", ForbiddenRepository)
    context = TenantContext(tenant_id="tenant", user_id="user", role="admin", scope_type="platform")

    assert workflow._persist_chat_turn(
        context, "session", "client-turn", "question",
        {"task_type": "metric_query", "answer": "a", "trace_id": "trace"},
        {"scope": "x"}, None, persistence_turn_id="turn1",
        authorization_boundary={"execution_id": "exec-1"},
    )


def test_direct_caller_without_lifecycle_still_writes_history(monkeypatch) -> None:
    """Durability is marked by a bound execution, not by a server turn id alone."""
    import app.graph.workflow as workflow
    from app.tenancy import TenantContext

    saved: dict[str, object] = {}

    class RecordingRepository:
        def __init__(self, context):
            saved["tenant"] = context.tenant_id

        def save_turn(self, *args, **kwargs):
            saved["turn_id"] = args[1]
            saved["client_turn_id"] = kwargs.get("client_turn_id")

    monkeypatch.setattr(workflow, "ChatHistoryRepository", RecordingRepository)
    context = TenantContext(tenant_id="tenant", user_id="user", role="admin", scope_type="platform")

    assert workflow._persist_chat_turn(
        context, "session", "client-turn", "question",
        {"task_type": "metric_query", "answer": "a", "trace_id": "trace"},
        {"scope": "x"}, None, persistence_turn_id="turn1",
    )
    assert saved == {"tenant": "tenant", "turn_id": "turn1", "client_turn_id": "client-turn"}


def test_scope_clarification_and_presentation_survive_legacy_history(monkeypatch) -> None:
    import app.graph.workflow as workflow
    from app.tenancy import TenantContext

    saved: dict[str, object] = {}

    class RecordingRepository:
        def __init__(self, _context):
            pass

        def save_turn(self, *args, **kwargs):
            saved.update(kwargs)

    monkeypatch.setattr(workflow, "ChatHistoryRepository", RecordingRepository)
    context = TenantContext(tenant_id="tenant", user_id="user", role="admin", scope_type="platform")

    assert workflow._persist_chat_turn(
        context, "session", "turn", "按 RFM 做分群",
        {
            "task_type": "complex_analysis",
            "answer": "请先选择店铺、开始日期和结束日期，再确认复杂分析计划。",
            "trace_id": "trace",
            "complex_clarification": {
                "status": "complex_scope_required",
                "reason_code": "scope_required",
                "reason": "不能把任意 provider 文本存入历史。",
                "original_message": "private question",
            },
            "analysis_presentation": {
                "status": "completed",
                "facts": [{"metric_name": "gmv", "value": 12, "date_or_range": "2018-08"}],
                "raw_sql": "SELECT private_data",
            },
        },
        None,
    )

    assert saved["status"] == "interrupted"
    assert saved["interaction_state"] == "awaiting_scope"
    snapshot = saved["result_snapshot"]
    assert snapshot["clarification"] == {
        "status": "complex_scope_required",
        "reason_code": "scope_required",
    }
    assert snapshot["presentation"]["facts"][0]["value"] == 12
    assert "raw_sql" not in snapshot["presentation"]


def test_complex_clarification_is_not_persisted_as_completed(monkeypatch) -> None:
    import app.graph.workflow as workflow
    from app.tenancy import TenantContext

    saved: dict[str, object] = {}

    class RecordingRepository:
        def __init__(self, _context):
            pass

        def save_turn(self, *args, **kwargs):
            saved.update(kwargs)

    monkeypatch.setattr(workflow, "ChatHistoryRepository", RecordingRepository)
    context = TenantContext(tenant_id="tenant", user_id="user", role="admin", scope_type="platform")

    assert workflow._persist_chat_turn(
        context, "session", "turn", "synthetic complex request",
        {
            "task_type": "complex_analysis",
            "answer": "请补充分析对象或分析操作，以便继续处理请求。",
            "trace_id": "trace",
            "complex_clarification": {
                "status": "needs_context",
                "reason_code": "needs_context",
                "original_message": "must not be persisted",
            },
        },
        None,
    )

    assert saved["status"] == "abstained"
    assert saved["interaction_state"] == "awaiting_human_input"
    assert saved["result_snapshot"]["clarification"] == {
        "status": "needs_context",
        "reason_code": "needs_context",
    }


def test_critic_bundle_carries_verified_investigation_as_tool_evidence() -> None:
    state = _state(
        task_type="anomaly_analysis",
        critic_investigation_context={"verified_contribution_evidence": [{"evidence_id": "ev-1"}]},
        evidence_refresh={"available": True, "evidence_ids": ["ev-1"]},
    )
    bundle = ContextOrchestrator().build_specialist_bundle(
        state, "critic", instruction="q", model="test-model",
    )
    investigation_items = [
        item for item in bundle.items if item.item_id == "evidence.investigation_context"
    ]
    assert investigation_items
    assert investigation_items[0].source is Source.TOOL_EVIDENCE
    assert investigation_items[0].trust_tier is TrustTier.CURRENT_EVIDENCE
    assert bundle.metadata()["task_type"] == "anomaly_analysis"


def test_critic_bundle_can_replace_raw_tool_evidence_with_compact_projection() -> None:
    state = _state(
        task_type="anomaly_analysis",
        evidence_refresh={"available": True, "evidence_ids": ["ev-1"]},
        evidence_bundles=[{
            "evidence_id": "ev-1", "source_type": "mcp", "source_ref": "query_gmv",
            "trusted": True, "content": {"result": {"raw": "RAW_EVIDENCE_CONTENT"}},
        }],
    )
    summary_item = OrchestratedItem(
        item_id="evidence.ev-1",
        source=Source.TOOL_EVIDENCE,
        trust_tier=TrustTier.CURRENT_EVIDENCE,
        content='{"evidence_id":"ev-1","value":12}',
        source_refs=("query_gmv",),
        scope_fingerprint="scope-wp1",
        version="summary-v1",
        required=True,
        priority=90,
        evidence_ids=("ev-1",),
    )
    bundle = ContextOrchestrator().build_specialist_bundle(
        state,
        "critic",
        instruction="q",
        model="test-model",
        additional_items=[summary_item],
        replace_sources={Source.TOOL_EVIDENCE},
    )

    assert "RAW_EVIDENCE_CONTENT" not in bundle.text
    assert '"value":12' in bundle.text
    assert "evidence.ev-1" in bundle.included_item_ids
    assert Source.METRIC_REGISTRY in {item.source for item in bundle.items}


def test_critic_review_does_not_reintroduce_task_or_question_items(monkeypatch) -> None:
    from app.agents import critic_agent

    captured = {}

    def fake_extract(_system, user_message, **_kwargs):
        captured["user"] = user_message
        return {"verdict": "passed", "issues": []}

    monkeypatch.setattr(critic_agent, "llm_extract", fake_extract)
    critic_agent.llm_review_evidence_completeness(
        "answer", "metric_query", context_text="CONTROLLED_CONTEXT",
    )
    assert "任务类型" not in captured["user"]
    assert "metric_query" not in captured["user"]
    assert "CONTROLLED_CONTEXT" in captured["user"]


def test_diagnostic_helper_passes_system_prompt_to_budget(monkeypatch) -> None:
    from app.graph.nodes import diagnostic

    captured = {}

    class FakeBundle:
        text = "DIAGNOSTIC_CONTEXT"

    def fake_build(self, state, agent, **kwargs):
        captured["agent"] = agent
        captured["system_prompt"] = kwargs.get("system_prompt")
        return FakeBundle()

    monkeypatch.setattr(ContextOrchestrator, "build_specialist_bundle", fake_build)
    message = diagnostic._diagnostic_context_user_message(
        _state(), system_prompt="DIAGNOSTIC_SYSTEM_PROMPT",
    )
    assert captured["agent"] == "diagnostic"
    assert captured["system_prompt"] == "DIAGNOSTIC_SYSTEM_PROMPT"
    assert "DIAGNOSTIC_CONTEXT" in message


def test_item_token_estimate_covers_rendered_block() -> None:
    state = _state()
    bundle = ContextOrchestrator().build_specialist_bundle(
        state, "answer", instruction="q", model="test-model",
    )
    item = next(item for item in bundle.items if item.item_id == "memory.m1")
    assert item.token_estimate > estimate_tokens(item.content, "test-model")


def test_triage_context_is_minimal_and_traceable() -> None:
    state = _state()
    context = ContextOrchestrator().build_triage_context(
        state, instruction="分析一下最近的情况", system_prompt="system", model="test-model",
    )

    assert context.bundle.agent == "triage"
    assert context.bundle.bundle_schema_version == "context-bundle-v1"
    assert context.bundle.resolved_request["ref"].startswith("turn-resolution:")
    assert context.bundle.resolved_request["text"] == "分析一下最近的情况"
    assert "text" not in context.bundle.metadata()["resolved_request"]
    assert "text" not in state["context_bundles"]["triage"]["resolved_request"]
    assert all(item.token_estimate > 0 for item in context.bundle.items)
    assert {item.source for item in context.bundle.items} <= {
        Source.CONVERSATION, Source.TASK, Source.MEM0,
    }
    assert all(item.source not in {
        Source.ANALYSIS_MEMORY, Source.RAG, Source.METRIC_REGISTRY, Source.TOOL_EVIDENCE,
    } for item in context.bundle.items)
    assert context.route_slots == {"metric": "gmv"}
    assert state["triage_context"]["bundle"]["assembly_policy_version"] == "context-orchestrator-v1"
    assert state["context_bundles"]["triage"]["agent"] == "triage"
    assert state["context_orchestration_events"]
    assert state["context_orchestration_summary"]["bundles"] == 1
    assert any(
        item["source"] == "mem0" and item["reason"] == "not_configured"
        for item in state["context_source_availability"]["triage"]
    )


def test_specialist_bundles_differ_by_agent() -> None:
    state = _state()
    state["evidence_refresh"] = {"available": True, "evidence_ids": ["ev-1"]}
    orchestrator = ContextOrchestrator()
    data_query = orchestrator.build_specialist_bundle(state, "data_query", instruction="q", model="test-model")
    answer = orchestrator.build_specialist_bundle(state, "answer", instruction="q", model="test-model")
    critic = orchestrator.build_specialist_bundle(state, "critic", instruction="q", model="test-model")

    assert Source.TOOL_EVIDENCE not in {item.source for item in data_query.items}
    assert Source.TOOL_EVIDENCE in {item.source for item in answer.items}
    assert Source.ANALYSIS_MEMORY in {item.source for item in answer.items}
    assert {item.source for item in critic.items} <= {Source.TOOL_EVIDENCE, Source.METRIC_REGISTRY}
    assert Source.CONVERSATION not in {item.source for item in critic.items}
    assert critic.answer_under_review is not None
    assert critic.answer_under_review["content"] == "draft answer"
    assert critic.metadata()["answer_under_review"]["present"] is True
    assert "content" not in critic.metadata()["answer_under_review"]


def test_evidence_refresh_gates_tool_evidence() -> None:
    state = _state()
    state.pop("evidence_refresh", None)
    orchestrator = ContextOrchestrator()
    before = orchestrator.build_specialist_bundle(state, "answer", instruction="q", model="test-model")
    assert Source.TOOL_EVIDENCE not in {item.source for item in before.items}

    refresh = orchestrator.refresh_evidence(state)
    assert refresh["available"] is True
    assert refresh["evidence_ids"] == ["ev-1"]

    after = orchestrator.build_specialist_bundle(state, "answer", instruction="q", model="test-model")
    assert Source.TOOL_EVIDENCE in {item.source for item in after.items}


def test_metadata_never_contains_item_content() -> None:
    state = _state()
    ContextOrchestrator().build_specialist_bundle(state, "answer", instruction="q", model="test-model")
    serialized = json.dumps(state["context_bundles"], ensure_ascii=False)
    events = json.dumps(state["context_orchestration_events"], ensure_ascii=False)
    assert "SECRET_MEMORY_TEXT" not in serialized
    assert "HISTORY_MEMORY" not in serialized
    assert "SECRET_MEMORY_TEXT" not in events
    assert "HISTORY_MEMORY" not in events


def test_every_item_matches_source_tier_mapping() -> None:
    state = _state()
    state["evidence_refresh"] = {"available": True, "evidence_ids": ["ev-1"]}
    bundle = ContextOrchestrator().build_specialist_bundle(
        state, "answer", instruction="q", model="test-model",
    )
    for item in bundle.items:
        assert item.trust_tier is SOURCE_TIER[item.source]


def test_required_instruction_overflow_fails_explicitly() -> None:
    state = _state(user_message="x" * 200_000)
    state["context_compilation"] = {"standalone_question": state["user_message"], "known_slots": {}}
    with pytest.raises(ContextBudgetExceeded):
        ContextOrchestrator().build_triage_context(
            state, instruction=state["user_message"], system_prompt="system", model="test-model",
        )


def test_triage_node_uses_orchestrator_bundle(monkeypatch) -> None:
    import app.graph.nodes.triage as triage_module

    captured = {}

    def fake_classify(_system, user_message, _allowed, **kwargs):
        captured["user"] = user_message
        captured.update(kwargs)
        return ""

    monkeypatch.setattr(triage_module, "llm_classify", fake_classify)
    state = _state()
    result = triage_module.triage(state)

    assert result["task_type"] == "clarification"
    assert result["next_agent"] == "context_clarification"
    assert result["triage_unresolved"] is True
    assert result["triage_unresolved_reason"] == "task_type_classifier_unavailable"
    assert "execution_plan" not in result
    from app.context.compiler import context_clarification
    clarified = context_clarification(result)
    assert clarified["metric_planning_decision"]["reason_code"] == "task_type_unresolved"
    assert "无法可靠判断" in clarified["answer"]
    assert captured["user"]
    assert captured["operation"] == "classify.task_type"
    assert captured["interaction_policy"] == "nonstream_fast"
    assert "上一轮查看了 GMV" in captured["user"]
    assert "HISTORY_MEMORY" not in captured["user"]
    assert result["triage_context"]["bundle"]["agent"] == "triage"
    assert result["context_bundles"]["triage"]["items"]


@pytest.mark.parametrize(
    ("classifier_result", "expected_reason"),
    [
        ("", "task_type_classifier_unavailable"),
        (None, "task_type_classifier_unavailable"),
        ("unregistered_route", "task_type_classifier_invalid_output"),
        ("raise", "task_type_classifier_unavailable"),
    ],
)
def test_triage_classifier_failure_never_uses_keyword_fallback(
    monkeypatch, classifier_result, expected_reason: str,
) -> None:
    import inspect
    import app.graph.nodes.triage as triage_module
    from app.graph import workflow

    def classify(*_args, **_kwargs):
        if classifier_result == "raise":
            raise RuntimeError("simulated classifier failure")
        return classifier_result

    monkeypatch.setattr(triage_module, "llm_classify", classify)
    state = _state()
    result = triage_module.triage(state)

    assert result["task_type"] == "clarification"
    assert result["triage_unresolved"] is True
    assert result["triage_unresolved_reason"] == expected_reason
    assert result["context_compilation"]["clarification_reason"] == "task_type_unresolved"
    assert result["context_compilation"]["missing_slots"][-1] == "task_type"
    assert "execution_plan" not in result
    assert workflow._route_after_triage(result) == "context_clarification"
    assert workflow._route_from_triage(result) == "context_clarification"
    assert "_keyword_fallback" not in inspect.getsource(triage_module)
    from app.graph.runtime import _validate_boundary
    _validate_boundary("triage", result)


def test_answer_grounded_narrative_uses_orchestrator_context(monkeypatch) -> None:
    import app.graph.nodes.answer as answer_module

    captured = {}

    def fake_chat(**kwargs):
        captured["user"] = kwargs.get("user_message")
        return "答案已生成"

    monkeypatch.setattr("app.llm.client.llm_chat", fake_chat)
    state = _state()
    text, status = answer_module._grounded_narrative(state, {"value": 1}, "fallback")

    assert text == "答案已生成"
    assert status == "llm_grounded"
    assert "HISTORY_MEMORY" in captured["user"]
    assert "context_bundles" in state and state["context_bundles"].get("answer")


def test_critic_review_receives_orchestrated_context(monkeypatch) -> None:
    from app.agents import critic_agent

    captured = {}

    def fake_extract(_system, user_message, **_kwargs):
        captured["user"] = user_message
        return {"verdict": "passed", "issues": []}

    monkeypatch.setattr(critic_agent, "llm_extract", fake_extract)
    result = critic_agent.llm_review_groundedness(
        "answer", "rag evidence", context_text="HISTORY_FOR_CRITIC",
    )

    assert result["verdict"] == "passed"
    assert "HISTORY_FOR_CRITIC" in captured["user"]
    assert "[orchestrated_context" in captured["user"]
    assert "rag evidence" not in captured["user"]


def test_data_query_planner_receives_context_text(monkeypatch) -> None:
    from app.agents import data_query_agent

    captured = {}

    def fake_extract(_system, user_message, **_kwargs):
        captured["user"] = user_message
        return None

    monkeypatch.setattr("app.llm.client.llm_extract", fake_extract)
    plan = data_query_agent.plan_confirmed_metric_queries(
        "看一下营业额 2018-08-10 是多少",
        scope="platform",
        allowed_metrics={"gmv"},
        context_text="HISTORY_FOR_PLANNER",
    )

    assert plan.decision.status in {"clarification_required", "confirmed"}
    assert "HISTORY_FOR_PLANNER" in captured["user"]
    assert "[受控上下文参考" in captured["user"]


def test_stale_task_progress_is_not_revived() -> None:
    state = _state()
    state["context_compilation"]["task_progress_context_used"] = False
    state["data_version_manifest_hash"] = "manifest-current"
    state["task_progress_context"]["data_version_manifest_hash"] = "manifest-old"
    bundle = ContextOrchestrator().build_specialist_bundle(
        state, "triage", instruction="q", model="test-model",
    )
    assert Source.TASK not in {item.source for item in bundle.items}


def test_failed_tool_evidence_is_not_refreshed() -> None:
    state = _state(evidence_bundles=[
        {"evidence_id": "ev-failed", "source_type": "mcp", "source_ref": "query_gmv",
         "content": {"result": {"status": "failed"}}, "trusted": True},
    ])
    refresh = ContextOrchestrator().refresh_evidence(state)
    assert refresh["available"] is False
    assert refresh["evidence_ids"] == []


def test_rag_evidence_is_not_labelled_current_tool_evidence() -> None:
    state = _state(evidence_bundles=[
        {"evidence_id": "ev-rag", "source_type": "rag", "source_ref": "doc", "trusted": True},
    ])
    refresh = ContextOrchestrator().refresh_evidence(state)
    assert refresh["available"] is False
    bundle = ContextOrchestrator().build_specialist_bundle(
        state, "answer", instruction="q", model="test-model",
    )
    assert Source.TOOL_EVIDENCE not in {item.source for item in bundle.items}


def test_policy_rejected_item_is_recorded() -> None:
    state = _state(historical_claims_context=[{"value": "historical claim"}])
    bundle = ContextOrchestrator().build_specialist_bundle(
        state, "triage", instruction="q", model="test-model",
    )
    assert any(
        item.get("item_id") == "ctx.historical_claims"
        and item.get("reason") == "policy_source_or_tier"
        for item in bundle.omitted
    )


def test_max_items_truncation_is_recorded() -> None:
    state = _state(
        evidence_bundles=[
            {"evidence_id": f"ev-{index}", "source_type": "mcp",
             "source_ref": "query_gmv", "content": {"result": {"status": "ok"}},
             "trusted": True}
            for index in range(20)
        ],
        evidence_refresh={"available": True, "evidence_ids": [f"ev-{index}" for index in range(20)]},
    )
    bundle = ContextOrchestrator().build_specialist_bundle(
        state, "answer", instruction="q", model="test-model",
    )
    assert any(item.get("reason") == "max_items_exceeded" for item in bundle.omitted)


def test_missing_trusted_field_is_not_current_evidence() -> None:
    state = _state(evidence_bundles=[
        {"evidence_id": "ev-missing", "source_type": "mcp", "source_ref": "query_gmv",
         "content": {"result": {"status": "ok"}}},
    ])
    refresh = ContextOrchestrator().refresh_evidence(state)
    assert refresh["available"] is False
    assert refresh["evidence_ids"] == []


def test_diagnostic_refresh_happens_after_validation() -> None:
    from app.graph.workflow import _with_evidence_refresh

    def untrusted_tool(state):
        state["evidence_bundles"].append({
            "evidence_id": "ev-diagnostic", "source_type": "mcp",
            "source_ref": "query_gmv", "content": {"result": {"status": "ok"}},
            "trusted": False,
        })
        return state

    def validated_observe(state):
        state["evidence_bundles"][-1]["trusted"] = True
        return state

    state = _state(evidence_bundles=[])
    _with_evidence_refresh(untrusted_tool)(state)
    assert state["evidence_refresh"]["available"] is False
    _with_evidence_refresh(validated_observe)(state)
    assert state["evidence_refresh"]["available"] is True
    assert state["evidence_refresh"]["evidence_ids"] == ["ev-diagnostic"]


def test_real_observe_validation_refreshes_evidence_for_answer_bundle() -> None:
    """Exercise the actual observe node and shared evidence validator without DB/LLM."""
    from app.agents.diagnostic_contracts import CallToolAction, DiagnosticPeriod
    from app.graph.nodes.diagnostic import observe
    from app.graph.workflow import _with_evidence_refresh

    state = {
        "trace_id": "trace-real-observe-refresh",
        "task_id": "task-real-observe-refresh",
        "tenant_id": "tenant-real-observe",
        "user_id": "user-real-observe",
        "user_message": "分析 2018-08-20 的 GMV",
        "diagnostic_target_metric": "gmv",
        "controlled_analysis_scope": {
            "tenant_id": "tenant-real-observe",
            "user_id": "user-real-observe",
            "default_start_date": "2018-08-01",
            "default_end_date": "2018-08-31",
            "granularity": "day",
            "allowed_metrics": ["gmv"],
            "allowed_dimensions": [],
        },
        "diagnostic_plan": {
            "goal": "observe current metric evidence",
            "hypotheses": ["metric changed"],
            "steps": [{
                "step_id": "trend",
                "objective": "read metric trend",
                "allowed_actions": ["CALL_TOOL", "FINISH"],
                "completion_condition": "validated trend evidence exists",
                "kind": "trend",
            }],
            "stop_conditions": ["done"],
            "budget": {"max_rounds": 3, "max_tool_calls": 2},
        },
        "diagnostic_step_state": {"trend": "pending"},
        "diagnostic_action": CallToolAction(
            action_type="CALL_TOOL",
            tool_name="query_metric_trend",
            metric_name="gmv",
            period=DiagnosticPeriod(
                start_date="2018-08-20", end_date="2018-08-20", granularity="day",
            ),
        ).model_dump(mode="json"),
        "diagnostic_tool_results": [{
            "tool_name": "query_metric_trend",
            "result": {
                "status": "ok",
                "points": [{"bucket": "2018-08-20", "value": 100.0}],
            },
        }],
        "diagnostic_decision_source": "backend",
        "diagnostic_investigation_action_count": 0,
        "tool_calls": [],
        "evidence_bundles": [],
        "root_cause_evidence": [],
        "observations": [],
        "node_history": [],
        "errors": [],
        "current_query_scope_fingerprint": "scope-real-observe",
    }

    observed = _with_evidence_refresh(observe)(state)

    accepted = [item for item in observed["evidence_bundles"] if item.get("trusted") is True]
    assert len(accepted) == 1
    evidence = accepted[0]
    assert evidence["validation_status"] == "accepted_by_investigation_engine"
    assert observed["investigation_state"]["action_results"][-1]["status"] == "completed"
    assert observed["evidence_refresh"]["available"] is True
    assert observed["evidence_refresh"]["evidence_ids"] == [evidence["evidence_id"]]

    answer_bundle = ContextOrchestrator().build_specialist_bundle(
        observed, "answer", instruction="Summarize the validated metric result.", model="test-model",
    )
    assert f"evidence.{evidence['evidence_id']}" in answer_bundle.included_item_ids


def test_diagnostic_context_message_uses_diagnostic_policy() -> None:
    from app.graph.nodes.diagnostic import _diagnostic_context_user_message

    state = _state(evidence_refresh={"available": True, "evidence_ids": ["ev-1"]})
    message = _diagnostic_context_user_message(state)
    assert "objective=查看 GMV" in message
    assert "[metric_registry" in message
    assert "[tool_evidence" in message
    assert "SECRET_MEMORY_TEXT" not in message
    assert "HISTORY_MEMORY" not in message


def test_answer_final_prompt_budget_falls_back_explicitly(monkeypatch) -> None:
    import app.graph.nodes.answer as answer_module

    monkeypatch.setattr(
        "app.llm.client.llm_chat",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM must not be called")),
    )
    state = _state(
        evidence_bundles=[{
            "evidence_id": "ev-large", "source_type": "mcp", "source_ref": "query_gmv",
            "content": {"blob": "测" * 30000}, "trusted": True,
        }],
        evidence_refresh={"available": True, "evidence_ids": ["ev-large"]},
    )
    text, status = answer_module._grounded_narrative(
        state, {"blob": "测" * 30000}, "fallback",
    )
    assert text == "fallback"
    assert status == "context_budget_exceeded"


def test_evidence_refresh_wrapper_runs_after_tool_node() -> None:
    from app.graph.workflow import _with_evidence_refresh

    def fake_tool_node(state):
        state["evidence_bundles"].append({
            "evidence_id": "ev-new", "source_type": "mcp",
            "source_ref": "query_gmv", "trusted": True,
        })
        return state

    state = _state(evidence_bundles=[])
    result = _with_evidence_refresh(fake_tool_node)(state)
    assert result["evidence_refresh"]["available"] is True
    assert result["evidence_refresh"]["evidence_ids"] == ["ev-new"]

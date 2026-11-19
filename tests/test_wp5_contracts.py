"""WP5 unit contracts for durable conversation state."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import date
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from app.api.routes import chat as chat_routes
from app.analysis.publishing import ReportStore
from app.chat_lifecycle.service import LifecycleService
from app.api.routes.chat import _lifecycle_result_status
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.explorer import ExplorerQuery
from app.schemas.events import SseCompletedEvent, SseSuspendedEvent
from app.graph.nodes import human_review as human_review_node
from app.graph import workflow as workflow_module
from app.tenancy import TenantContext


def test_turn_task_type_resolution_uses_snapshot_when_column_has_pending_sentinel() -> None:
    from app.chat_lifecycle.integration import resolve_turn_task_type

    assert resolve_turn_task_type("pending", "metric_query") == "metric_query"
    assert resolve_turn_task_type("metric_query", "complex_analysis") == "metric_query"
    assert resolve_turn_task_type("pending", "pending") == "unknown"


def test_summary_coverage_matches_every_rendered_turn() -> None:
    from app.context.packer import estimate_tokens

    rows = [
        {"turn_id": f"turn-{i}", "commit_version": i, "question": f"question-{i}", "answer": "答" * 640}
        for i in range(1, 31)
    ]

    summary, covered = LifecycleService._render_summary_rows(rows)

    covered_ids = [row["turn_id"] for row in covered]
    assert covered_ids[-1] == "turn-30"
    assert covered_ids == [f"turn-{i}" for i in range(31 - len(covered), 31)]
    assert summary.startswith(f"用户：question-{31 - len(covered)}")
    assert summary.endswith("答" * 640)
    assert estimate_tokens(summary) <= 4000


def test_empty_turn_is_still_inside_the_reported_summary_coverage() -> None:
    assert LifecycleService._render_summary_rows([]) == ("", [])
    summary, covered = LifecycleService._render_summary_rows(
        [{"turn_id": "empty", "commit_version": 1, "question": "", "answer": ""}],
    )
    assert summary == "用户：（本轮未记录问题文本）"
    assert covered[0]["turn_id"] == "empty"


def test_clarification_and_graph_interrupt_are_nonterminal_suspensions() -> None:
    clarification = {"event": "completed", "clarification": {"status": "confirmation_required"}}
    assert _lifecycle_result_status(clarification)[:2] == (
        "suspended", "awaiting_metric_confirmation",
    )
    assert _lifecycle_result_status({"__interrupt__": [{"value": "confirm"}]})[:2] == (
        "suspended", "awaiting_human_input",
    )


def test_sse_terminal_schemas_distinguish_commit_from_suspend() -> None:
    completed = SseCompletedEvent(
        event="completed", sequence_number=5, answer="done", durable_turn_id="turn-1",
        execution_id="exec-1", commit_version=8,
    )
    suspended = SseSuspendedEvent(
        event="suspended", sequence_number=5, answer="pending",
        interaction_state="awaiting_human_input",
        clarification={"status": "awaiting_human_review"},
    )

    assert completed.commit_version == 8
    assert suspended.interaction_state == "awaiting_human_input"
    assert suspended.model_dump(mode="json")["clarification"] == {"status": "awaiting_human_review"}
    assert completed.event != suspended.event


def test_sync_chat_response_exposes_a_suspended_interaction() -> None:
    response = ChatResponse(
        task_type="clarification", answer="请确认", trace_id="trace-1",
        interaction_state="awaiting_metric_confirmation",
    )

    assert response.interaction_state == "awaiting_metric_confirmation"


def test_durable_chat_keeps_legacy_memory_session_out_of_canonical_identity() -> None:
    durable = ChatRequest(
        message="继续分析", chat_session_id="chat-1",
        use_current_explorer_context=True,
        explorer_context=ExplorerQuery(start_date=date(2026, 10, 1), end_date=date(2026, 10, 31)),
    )
    # WP6: durable chat reads/writes chat_sessions/chat_turns only; the legacy
    # memory_sessions identity is intentionally not aliased from chat_session_id.
    assert chat_routes._memory_session_for_request(durable) is None

    matching = durable.model_copy(update={"memory_session_id": "chat-1"})
    assert chat_routes._memory_session_for_request(matching) is None

    mismatched = durable.model_copy(update={"memory_session_id": "other-session"})
    with pytest.raises(HTTPException, match="memory_session_id_must_match_chat_session_id"):
        chat_routes._memory_session_for_request(mismatched)


def test_legacy_memory_only_chat_keeps_its_existing_session_identity() -> None:
    legacy = ChatRequest(message="继续", memory_session_id="legacy-memory-session")
    assert chat_routes._memory_session_for_request(legacy) == "legacy-memory-session"


def test_finalization_projects_suspension_and_commit_metadata(monkeypatch) -> None:
    calls = {}
    service = SimpleNamespace(commit_metadata=lambda _turn: {
        "turn_id": "durable-1", "final_execution_id": "exec-1", "commit_version": 4,
    })
    binding = SimpleNamespace(
        service=service,
        turn=SimpleNamespace(turn_id="durable-1"),
        execution=SimpleNamespace(execution_id="exec-1"),
    )

    def fake_finalize(_binding, **kwargs):
        calls.update(kwargs)
        return True

    monkeypatch.setattr(chat_routes, "finalize_chat_lifecycle", fake_finalize)
    event = {
        "event": "completed",
        "clarification": {"status": "confirmation_required", "reason_code": "comparison_dimension_not_supported"},
    }

    assert chat_routes._finalize_binding(binding, event, task_id="task-1")

    assert event["event"] == "suspended"
    assert event["interaction_state"] == "awaiting_metric_confirmation"
    assert event["clarification"]["reason_code"] == "comparison_dimension_not_supported"
    assert event["commit_version"] == 4
    assert calls["task_status"] == "awaiting_confirmation"


def test_finalization_persists_internal_turn_resolution_without_returning_it(monkeypatch):
    calls = {}
    binding = SimpleNamespace(
        service=SimpleNamespace(commit_metadata=lambda _turn: {
            "turn_id": "durable-1", "final_execution_id": "exec-1", "commit_version": 4,
        }),
        turn=SimpleNamespace(turn_id="durable-1"),
        execution=SimpleNamespace(execution_id="exec-1"),
    )

    def fake_finalize(_binding, **kwargs):
        calls.update(kwargs)
        return True

    monkeypatch.setattr(chat_routes, "finalize_chat_lifecycle", fake_finalize)
    response = {
        "event": "completed", "answer": "done", "trace_id": "trace-1",
        "turn_resolution": {"unresolved_fields": ["date"]},
    }
    assert chat_routes._finalize_binding(binding, response)
    assert calls["result"]["turn_resolution"]["unresolved_fields"] == ["date"]
    assert "turn_resolution" not in response


def test_failed_durable_turn_commit_cannot_return_completed_event(monkeypatch) -> None:
    binding = SimpleNamespace(
        service=SimpleNamespace(commit_metadata=lambda _turn: {}),
        turn=SimpleNamespace(turn_id="durable-1"),
        execution=SimpleNamespace(execution_id="exec-1"),
    )
    monkeypatch.setattr(chat_routes, "finalize_chat_lifecycle", lambda *_args, **_kwargs: False)
    completed = {"event": "completed", "answer": "answer"}

    with pytest.raises(RuntimeError, match="chat_turn_commit_not_persisted"):
        chat_routes._finalize_binding(binding, completed)


def test_chat_stream_emits_failed_when_durable_commit_fails(monkeypatch) -> None:
    from datetime import date
    from fastapi.testclient import TestClient

    from app.api.auth import require_tenant_context
    from app.main import app
    from app.schemas.explorer import ExplorerQuery

    context = TenantContext(tenant_id="platform", user_id="wp5-stream-commit", role="admin", scope_type="platform")
    binding = SimpleNamespace(
        is_replay=False,
        active_reconnect=False,
        execution=SimpleNamespace(thread_id="thread-wp5", trace_id="trace-wp5", execution_id="exec-wp5"),
        query_scope_fingerprint="scope-fingerprint",
        persistence_turn_id="turn-wp5",
        authorization_boundary={},
        service=SimpleNamespace(),
        turn=SimpleNamespace(turn_id="turn-wp5"),
    )

    def fake_stream(*_args, **_kwargs):
        yield {"event": "completed", "task_type": "metric_query", "answer": "not committed", "trace_id": "trace-wp5"}

    app.dependency_overrides[require_tenant_context] = lambda: context
    try:
        monkeypatch.setattr(chat_routes, "begin_chat_lifecycle", lambda *_args, **_kwargs: binding)
        monkeypatch.setattr(chat_routes, "_expected_data_manifest_hash", lambda *_args, **_kwargs: "data-version")
        monkeypatch.setattr(chat_routes, "relation", lambda *_args, **_kwargs: {"action": "new", "resolved_message": "question"})
        monkeypatch.setattr(chat_routes, "create_task", lambda *_args, **_kwargs: {"task_id": "task-wp5"})
        monkeypatch.setattr(chat_routes, "update_task", lambda *_args, **_kwargs: {})
        monkeypatch.setattr(chat_routes, "stream_chat_workflow", fake_stream)
        monkeypatch.setattr(chat_routes, "finalize_chat_lifecycle", lambda *_args, **_kwargs: False)
        response = TestClient(app).post("/chat/stream", json={
            "message": "question", "chat_session_id": "session-wp5", "chat_turn_id": "client-turn-wp5",
            "use_current_explorer_context": True,
            "explorer_context": ExplorerQuery(
                start_date=date(2025, 1, 1), end_date=date(2025, 1, 2),
            ).model_dump(mode="json"),
        })
    finally:
        app.dependency_overrides.pop(require_tenant_context, None)

    assert response.status_code == 200, response.text
    assert '"event": "failed"' in response.text
    assert '"event": "completed"' not in response.text


def test_chat_stream_does_not_select_template_without_a_v2_route(monkeypatch) -> None:
    import json

    from fastapi.testclient import TestClient

    from app.api.auth import require_tenant_context
    from app.main import app
    from app.core.config import Settings
    from app.graph import workflow
    from app.graph.nodes import intent_router as intent_router_node

    context = TenantContext(tenant_id="platform", user_id="wp5-no-template-route", role="admin", scope_type="platform")
    updates: list[dict] = []
    settings = Settings(_env_file=None, complex_route_mode="active")
    monkeypatch.setattr(workflow, "get_settings", lambda: settings)
    monkeypatch.setattr(intent_router_node, "get_settings", lambda: settings)
    # This no-db runtime contract test does not exercise chat-history storage.
    monkeypatch.setattr(workflow.ChatHistoryRepository, "save_turn", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workflow.ChatHistoryRepository, "latest_completed_context", lambda *_args, **_kwargs: {})

    app.dependency_overrides[require_tenant_context] = lambda: context
    try:
        monkeypatch.setattr(chat_routes, "relation", lambda *_args, **_kwargs: {"action": "new", "resolved_message": "客户留存分析"})
        monkeypatch.setattr(chat_routes, "create_task", lambda *_args, **_kwargs: {"task_id": "task-wp5-template"})
        monkeypatch.setattr(chat_routes, "update_task", lambda *_args, **kwargs: updates.append(kwargs) or {})
        response = TestClient(app).post("/chat/stream", json={
            "message": "客户留存分析",
            "chat_session_id": "session-wp5-template",
            "chat_turn_id": "turn-wp5-template",
        })
    finally:
        app.dependency_overrides.pop(require_tenant_context, None)

    assert response.status_code == 200, response.text
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    terminal = next(event for event in reversed(events) if event["event"] in {"completed", "suspended", "failed"})
    assert terminal["task_type"] == "complex_analysis"
    assert terminal["clarification"]["status"] == "complex_analysis_unavailable"
    assert terminal["clarification"]["reason_code"] == "semantic_router_not_ready"
    assert terminal.get("complex_plan") is None
    assert updates[-1]["status"] == "awaiting_confirmation"


def test_chat_stream_reconnect_waits_for_the_existing_turn_and_replays_commit(monkeypatch) -> None:
    from datetime import date
    from fastapi.testclient import TestClient

    from app.api.auth import require_tenant_context
    from app.main import app
    from app.schemas.explorer import ExplorerQuery

    snapshots = iter([
        {"status": "running", "result_snapshot": {}},
        {
            "status": "completed", "task_type": "pending", "answer": "saved answer",
            "trace_id": "trace-original",
            "result_snapshot": {"task_type": "metric_query", "answer": "saved answer"},
        },
    ])
    service = SimpleNamespace(
        current_turn_snapshot=lambda _turn: next(snapshots),
        commit_metadata=lambda _turn: {
            "turn_id": "turn-existing", "final_execution_id": "exec-existing", "commit_version": 7,
        },
    )
    binding = SimpleNamespace(
        is_replay=False, active_reconnect=True, service=service,
        persistence_turn_id="turn-existing", execution=SimpleNamespace(execution_id="exec-existing"),
    )
    context = TenantContext(tenant_id="platform", user_id="wp5-stream-reconnect", role="admin", scope_type="platform")

    app.dependency_overrides[require_tenant_context] = lambda: context
    try:
        monkeypatch.setattr(chat_routes, "begin_chat_lifecycle", lambda *_args, **_kwargs: binding)
        monkeypatch.setattr(chat_routes.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(
            chat_routes, "stream_chat_workflow",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not rerun graph")),
        )
        response = TestClient(app).post("/chat/stream", json={
            "message": "question", "chat_session_id": "session-wp5", "chat_turn_id": "client-turn-wp5",
            "use_current_explorer_context": True,
            "explorer_context": ExplorerQuery(
                start_date=date(2025, 1, 1), end_date=date(2025, 1, 2),
            ).model_dump(mode="json"),
        })
        assert '"task_type": "metric_query"' in response.text
    finally:
        app.dependency_overrides.pop(require_tenant_context, None)

    assert response.status_code == 200, response.text
    assert '"event": "heartbeat"' in response.text
    assert '"answer": "saved answer"' in response.text
    assert '"durable_turn_id": "turn-existing"' in response.text
    assert '"commit_version": 7' in response.text


def test_task_scope_clarification_keeps_task_awaiting_scope() -> None:
    assert chat_routes._task_status_from_result({
        "clarification": {"status": "complex_scope_required"},
    }) == "awaiting_scope"


def test_human_review_report_side_effect_is_idempotent_after_checkpoint_replay(tmp_path, monkeypatch) -> None:
    database = str(tmp_path / "phase6.sqlite")
    store = ReportStore(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """INSERT INTO report_versions
               (report_id,version,status,content,content_hash,report_json,created_at,actor,reason)
               VALUES ('report-hitl',1,'draft','draft text','draft-hash','{}','2026-10-31T00:00:00+00:00','system','')"""
        )

    monkeypatch.setattr(human_review_node, "write_trace", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(human_review_node, "interrupt", lambda _payload: {
        "action": "approve", "actor": "reviewer", "reason": "checked", "_interaction_index": "0",
    })
    monkeypatch.setattr(human_review_node, "get_settings", lambda: SimpleNamespace(phase6_store_path=database))
    original = {
        "thread_id": "thread-hitl-replay", "trace_id": "trace-hitl-replay",
        "report_id": "report-hitl", "report_version": 1, "answer": "draft text",
        "critic_result": {}, "report_risks": [],
    }

    first = human_review_node.human_review(dict(original))
    replay = human_review_node.human_review(dict(original))

    assert first["critic_verdict"] == replay["critic_verdict"] == "approved"
    assert first["report_version"] == replay["report_version"] == 1
    with sqlite3.connect(database) as connection:
        versions = connection.execute(
            "SELECT count(*) FROM report_versions WHERE report_id='report-hitl'"
        ).fetchone()[0]
        approvals = connection.execute(
            "SELECT count(*) FROM report_approvals WHERE report_id='report-hitl'"
        ).fetchone()[0]
        published = connection.execute(
            "SELECT status FROM report_versions WHERE report_id='report-hitl' AND version=1"
        ).fetchone()[0]
    assert (versions, approvals, published) == (1, 1, "published")
    idempotency_key = "hitl-review:" + hashlib.sha256(
        b"trace-hitl-replay:report-hitl:1:0"
    ).hexdigest()
    with pytest.raises(ValueError, match="different decision"):
        store.review(
            "report-hitl", 1, "approve", "another-reviewer", "checked",
            idempotency_key=idempotency_key,
        )


def test_completed_checkpoint_recovers_turn_commit_without_reinvoking_graph(monkeypatch) -> None:
    saved = {
        "tenant_id": "tenant", "authorization_boundary": {
            "permission_version": "p1", "policy_set_version": "s1",
            "authorization_fingerprint": "a1", "query_scope_fingerprint": "q1",
        },
        "human_decision": {"action": "approve", "interaction_index": "0"},
        "current_agent": "persist_trace", "task_type": "report", "answer": "approved",
        "trace_id": "trace-recover", "task_id": "task-recover", "errors": [],
        "turn_relation_decision": {"relation": "continue"},
    }

    class FakeGraph:
        def get_state(self, _config):
            return SimpleNamespace(values=saved, next=())

        def invoke(self, *_args, **_kwargs):
            raise AssertionError("completed checkpoint recovery must not invoke the graph")

    calls = {}

    class FakeLifecycle:
        def __init__(self, *_args):
            pass

        def execution_for_thread(self, _thread_id):
            return SimpleNamespace(execution_id="exec-recover", turn_id="turn-recover")

        def claim_review_interaction(self, *_args, **_kwargs):
            return "processing", "review-interaction-1", None

        def finalize(self, _execution, **kwargs):
            calls.update(kwargs)
            return True

    monkeypatch.setattr(workflow_module, "build_chat_graph", lambda **_kwargs: FakeGraph())
    auth_context_module = __import__("app.auth.context", fromlist=["build_authorization_context"])
    auth_enforcement_module = __import__("app.auth.enforcement", fromlist=["AuthorizationBoundary"])
    lifecycle_module = __import__("app.chat_lifecycle.service", fromlist=["LifecycleService"])
    monkeypatch.setattr(auth_context_module, "build_authorization_context", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(auth_enforcement_module, "assert_authorization_boundary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(lifecycle_module, "LifecycleService", FakeLifecycle)

    response = workflow_module.resume_human_review(
        "thread-recover", {"action": "approve", "actor": "reviewer"},
        TenantContext(tenant_id="tenant", user_id="user", role="admin", scope_type="tenant"),
    )

    assert response == {"task_type": "report", "answer": "approved", "trace_id": "trace-recover"}
    assert calls["review_interaction_id"] == "review-interaction-1"
    assert calls["review_interaction_result"] == response
    assert calls["status"] == "completed"
    assert calls["task_id"] == "task-recover"

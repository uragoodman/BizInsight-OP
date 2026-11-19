"""Lifecycle HTTP tests with the graph replaced by a deterministic fake.

This deliberately does not invoke a model.  It verifies the API/lifecycle
boundary, execution finalization, and duplicate client-turn idempotency.
"""

from uuid import UUID, uuid4
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import text

from app.data.database import engine
from app.data.rls import set_local_context
from app.main import app
from app.auth.context import build_authorization_context
from app.api.auth import require_tenant_context
from app.chat_lifecycle.integration import begin_chat_lifecycle
from app.api.routes.chat import _is_suspended_premise_resume
from app.schemas.explorer import ExplorerQuery
from app.data.repositories.explorer import controlled_scope, preview_explorer
from app.scope.canonical import CanonicalScope
from app.tenancy import TenantContext
from types import SimpleNamespace
from fastapi import HTTPException
from app.schemas.chat import ChatRequest


def test_premise_resume_requires_server_bound_same_thread_and_scope() -> None:
    scope_fingerprint = "scope-fingerprint"
    turn = SimpleNamespace(
        status="running",
        result_snapshot={"interaction_reason_code": "premise_direction_mismatch"},
        scope_fingerprint=scope_fingerprint,
    )
    execution = SimpleNamespace(
        thread_id="durable-thread",
        query_scope_fingerprint=scope_fingerprint,
    )
    binding = SimpleNamespace(
        turn=turn,
        execution=execution,
        active_reconnect=False,
        query_scope_fingerprint=scope_fingerprint,
    )

    assert _is_suspended_premise_resume(
        binding, ChatRequest(message="按实际数据纠正方向", thread_id="durable-thread"),
    )
    assert not _is_suspended_premise_resume(
        binding, ChatRequest(message="另外查询一下七月的 GMV", thread_id="durable-thread"),
    )
    assert not _is_suspended_premise_resume(
        SimpleNamespace(**{**binding.__dict__, "active_reconnect": True}),
        ChatRequest(message="按实际数据纠正方向", thread_id="durable-thread"),
    )
    with pytest.raises(HTTPException) as thread_error:
        _is_suspended_premise_resume(
            binding, ChatRequest(message="按实际数据纠正方向", thread_id="other-thread"),
        )
    assert thread_error.value.status_code == 409

    binding.execution.query_scope_fingerprint = "stale-scope"
    with pytest.raises(HTTPException) as scope_error:
        _is_suspended_premise_resume(
            binding, ChatRequest(message="按实际数据纠正方向", thread_id="durable-thread"),
        )
    assert scope_error.value.status_code == 409


def _ensure_platform_test_identity(context: TenantContext) -> None:
    """Give the trusted platform test context the IAM rows production requires."""
    with engine.begin() as connection:
        connection.execute(text("""
            INSERT INTO iam.tenants(tenant_id, tenant_name, scope_type, status)
            VALUES (:tenant, 'platform lifecycle contract tests', 'platform', 'active')
            ON CONFLICT (tenant_id) DO NOTHING
        """), {"tenant": context.tenant_id})
        connection.execute(text("""
            INSERT INTO iam.users(user_id, display_name, email, status)
            VALUES (:user, 'platform lifecycle contract test', :email, 'active')
            ON CONFLICT (user_id) DO NOTHING
        """), {"user": context.user_id, "email": f"{context.user_id}@example.invalid"})
        connection.execute(text("""
            INSERT INTO iam.tenant_memberships(tenant_id, user_id, role)
            VALUES (:tenant, :user, 'admin')
            ON CONFLICT (tenant_id, user_id) DO NOTHING
        """), {"tenant": context.tenant_id, "user": context.user_id})


def test_scoped_chat_creates_and_finalizes_execution_without_llm() -> None:
    context = TenantContext(
        tenant_id="platform", user_id="lifecycle-http-test", role="admin",
        scope_type="platform",
    )
    _ensure_platform_test_identity(context)
    session_id = f"lifecycle-test-{uuid4().hex}"
    client_turn_id = f"client-{uuid4().hex}"
    trace_id = str(uuid4())
    body = {
        "message": "请查询 GMV",
        "use_current_explorer_context": True,
        "explorer_context": {
            "seller_id": "1f50f920176a81dab994f9023523100",
            "metric_name": "gmv",
            "start_date": "2018-08-01",
            "end_date": "2018-08-03",
            "granularity": "day",
        },
        "memory_enabled": False,
        "chat_session_id": session_id,
        "chat_turn_id": client_turn_id,
    }
    fake = {
        "task_type": "metric_query",
        "answer": "mock lifecycle answer",
        "trace_id": trace_id,
        "error": False,
        "errors": [],
        "clarification": None,
        "presentation": None,
    }
    app.dependency_overrides[require_tenant_context] = lambda: context
    try:
        with patch("app.api.routes.chat.run_chat_workflow", return_value=fake) as run:
            with patch("app.api.routes.chat.relation", return_value={"action": "none"}), patch(
                "app.api.routes.chat.uuid4", return_value=UUID(trace_id)
            ):
                response = TestClient(app).post("/chat", json=body)
            assert response.status_code == 200, response.text
            assert response.json()["answer"] == "mock lifecycle answer"

            # The same browser request is an idempotent replay, not a second graph
            # invocation and not a second conversation execution.
            with patch("app.api.routes.chat.relation", return_value={"action": "none"}), patch(
                "app.api.routes.chat.uuid4", return_value=UUID(trace_id)
            ):
                replay = TestClient(app).post("/chat", json=body)
            assert replay.status_code == 200, replay.text
            assert replay.json()["answer"] == "mock lifecycle answer"
            assert replay.json()["task_type"] == "metric_query"
            assert run.call_count == 1
    finally:
        app.dependency_overrides.pop(require_tenant_context, None)

    with engine.begin() as connection:
        set_local_context(connection, context, scope_type="platform")
        row = connection.execute(text("""
        SELECT t.status AS turn_status, t.task_type, t.result_snapshot->>'task_type' AS snapshot_task_type,
                   e.status AS execution_status,
                   t.final_execution_id, t.final_trace_id,
                   e.langgraph_thread_id
            FROM governance.chat_turns t
            JOIN governance.conversation_executions e ON e.turn_id=t.turn_id
            WHERE t.client_turn_id=:client
        """), {"client": client_turn_id}).mappings().one()
    assert row["turn_status"] == "completed"
    assert row["task_type"] == "metric_query"
    assert row["snapshot_task_type"] == "metric_query"
    assert row["execution_status"] == "completed"
    assert row["final_execution_id"]
    assert row["final_trace_id"] == trace_id
    assert row["langgraph_thread_id"].startswith("turn:")


def test_suspended_clarification_round_trips_through_turn_history_safely() -> None:
    context = TenantContext(
        tenant_id="platform", user_id="lifecycle-http-test", role="admin",
        scope_type="platform",
    )
    _ensure_platform_test_identity(context)
    session_id = f"clarification-history-{uuid4().hex}"
    client_turn_id = f"client-{uuid4().hex}"
    body = {
        "message": "请按店铺和商品品类拆解订单量",
        "use_current_explorer_context": True,
        "explorer_context": {
            "seller_id": "1f50f920176a81dab994f9023523100",
            "metric_name": "order_count",
            "start_date": "2018-08-01",
            "end_date": "2018-08-31",
            "granularity": "day",
        },
        "chat_session_id": session_id,
        "chat_turn_id": client_turn_id,
    }
    fake = {
        "task_type": "clarification",
        "answer": "问题中包含多个可能的维度，请明确一个。",
        "trace_id": str(uuid4()),
        "error": False,
        "errors": [],
        "clarification": {
            "status": "clarification_required",
            "reason_code": "ambiguous_dimension",
            "missing_slots": ["dimension"],
            "dimension_candidates": ["seller", "product_category"],
            "original_question": "must not be nested in clarification projection",
            "sql": "SELECT secret",
            "raw_state": {"token": "secret"},
        },
        "presentation": None,
    }
    app.dependency_overrides[require_tenant_context] = lambda: context
    try:
        with patch("app.api.routes.chat.run_chat_workflow", return_value=fake), patch(
            "app.api.routes.chat.relation", return_value={"action": "none"}
        ):
            client = TestClient(app)
            response = client.post("/chat", json=body)
        assert response.status_code == 200, response.text

        history = client.get(f"/chat/sessions/{session_id}/turns")
        assert history.status_code == 200, history.text
        turn = history.json()["items"][0]
        assert turn["status"] == "running"
        assert turn["interaction_state"] == "awaiting_metric_confirmation"
        assert turn["answer"] == fake["answer"]
        assert turn["clarification"] == {
            "status": "clarification_required",
            "reason_code": "ambiguous_dimension",
            "reason": "问题中有多个候选分组维度，需要明确选择。",
            "missing_slots": ["dimension"],
            "dimension_candidates": ["seller", "product_category"],
        }
        assert not {"sql", "raw_state", "original_question", "token"} & set(str(turn["clarification"]).split())
        assert "result_snapshot" not in turn
        assert not {"thread_id", "client_turn_id", "scope_fingerprint", "interaction_reason_code"} & set(turn)
    finally:
        app.dependency_overrides.pop(require_tenant_context, None)


def test_premise_checkpoint_and_metric_candidates_round_trip_after_history_read() -> None:
    context = TenantContext(
        tenant_id="platform", user_id="lifecycle-http-test", role="admin",
        scope_type="platform",
    )
    _ensure_platform_test_identity(context)
    session_id = f"premise-history-{uuid4().hex}"
    client_turn_id = f"client-{uuid4().hex}"
    payload = {
        "seller_id": "1f50f920176a81dab994f9023523100",
        "metric_name": "gmv",
        "start_date": "2018-08-01",
        "end_date": "2018-08-31",
        "granularity": "day",
    }
    binding = begin_chat_lifecycle(
        context,
        session_id=session_id,
        client_turn_id=client_turn_id,
        question="帮我分析这两项指标",
        explorer_payload=payload,
        thread_id=f"requested-thread-{uuid4().hex}",
        trace_id=str(uuid4()),
    )
    assert binding is not None and binding.execution is not None
    result = {
        "event": "suspended",
        "interaction_state": "awaiting_human_input",
        "interaction_reason_code": "premise_direction_mismatch",
        "task_type": "clarification",
        "answer": "请确认分析方向。",
        "clarification": {
            "status": "clarification_required",
            "reason_code": "ambiguous_metric",
            "missing_slots": ["metric"],
            "ambiguous_candidates": [
                {"metric_id": "gmv", "display_name": "伪造标签不得返回", "sql": "SELECT secret"},
                {"metric_id": "not_registered", "display_name": "不存在"},
            ],
            "raw_state": {"token": "secret"},
        },
    }
    assert binding.service.finalize(
        binding.execution,
        status="suspended",
        interaction_state="awaiting_human_input",
        answer=result["answer"],
        result=result,
    )

    app.dependency_overrides[require_tenant_context] = lambda: context
    try:
        client = TestClient(app)
        response = client.get(f"/chat/sessions/{session_id}/turns")
        assert response.status_code == 200, response.text
        turn = response.json()["items"][0]
        sessions = client.get("/chat/sessions")
        assert sessions.status_code == 200, sessions.text
        session = next(item for item in sessions.json()["items"] if item["id"] == session_id)
        assert turn["thread_id"] == binding.execution.thread_id
        assert turn["client_turn_id"] == client_turn_id
        assert turn["interaction_reason_code"] == "premise_direction_mismatch"
        assert turn["interaction_state"] == "awaiting_human_input"
        assert turn["scope_fingerprint"]
        assert session["scope_fingerprint"] == turn["scope_fingerprint"]
        assert "scope_key" not in turn
        assert turn["clarification"]["ambiguous_candidates"] == [
            {"metric_id": "gmv", "display_name": "成交总额"},
        ]
        assert not {"raw_state", "sql", "token"} & set(str(turn))
        assert "result_snapshot" not in turn

        resumed = begin_chat_lifecycle(
            context,
            session_id=session_id,
            client_turn_id=turn["client_turn_id"],
            question="按实际数据纠正方向",
            explorer_payload=payload,
            thread_id=turn["thread_id"],
            trace_id=str(uuid4()),
        )
        assert resumed is not None and resumed.execution is not None
        assert resumed.turn.turn_id == binding.turn.turn_id
        assert resumed.execution.execution_id == binding.execution.execution_id
        assert resumed.execution.thread_id == binding.execution.thread_id
    finally:
        app.dependency_overrides.pop(require_tenant_context, None)


def test_history_resume_credentials_are_hidden_for_terminal_changed_scope_and_other_owner() -> None:
    context = TenantContext(
        tenant_id="platform", user_id=f"history-owner-{uuid4().hex}", role="admin",
        scope_type="platform",
    )
    other_user = TenantContext(
        tenant_id="platform", user_id=f"history-other-{uuid4().hex}", role="admin",
        scope_type="platform",
    )
    _ensure_platform_test_identity(context)
    _ensure_platform_test_identity(other_user)
    payload = {
        "seller_id": "1f50f920176a81dab994f9023523100",
        "metric_name": "gmv", "start_date": "2018-08-01",
        "end_date": "2018-08-31", "granularity": "day",
    }

    def create_turn(session_id: str, client_turn_id: str):
        binding = begin_chat_lifecycle(
            context, session_id=session_id, client_turn_id=client_turn_id,
            question="为什么 GMV 下降？", explorer_payload=payload,
            thread_id=f"requested-{uuid4().hex}", trace_id=str(uuid4()),
        )
        assert binding is not None and binding.execution is not None
        return binding

    suspended_session = f"resume-visible-{uuid4().hex}"
    suspended = create_turn(suspended_session, f"client-{uuid4().hex}")
    assert suspended.service.finalize(
        suspended.execution, status="suspended", interaction_state="awaiting_human_input",
        answer="请确认实际方向。", result={
            "event": "suspended", "interaction_state": "awaiting_human_input",
            "interaction_reason_code": "premise_direction_mismatch",
            "task_type": "anomaly_analysis", "answer": "请确认实际方向。",
        },
    )

    terminal_sessions = []
    for status in ("completed", "failed"):
        session_id = f"resume-{status}-{uuid4().hex}"
        binding = create_turn(session_id, f"client-{uuid4().hex}")
        result = {"task_type": "metric_query", "answer": "完成" if status == "completed" else ""}
        assert binding.service.finalize(
            binding.execution, status=status, answer=result["answer"], result=result,
            error_code="test_failure" if status == "failed" else None,
        )
        terminal_sessions.append(session_id)

    app.dependency_overrides[require_tenant_context] = lambda: context
    try:
        client = TestClient(app)
        visible = client.get(f"/chat/sessions/{suspended_session}/turns").json()["items"][0]
        assert visible["thread_id"] == suspended.execution.thread_id
        assert visible["client_turn_id"] == suspended.turn.client_turn_id
        assert visible["scope_fingerprint"]

        other_session = client.get(f"/chat/sessions/not-{suspended_session}/turns")
        assert other_session.status_code == 200
        assert other_session.json()["items"] == []
        for session_id in terminal_sessions:
            item = client.get(f"/chat/sessions/{session_id}/turns").json()["items"][0]
            assert not {"thread_id", "client_turn_id", "scope_fingerprint", "interaction_reason_code"} & set(item)

        with engine.begin() as connection:
            set_local_context(connection, context, scope_type="platform")
            connection.execute(text("""
                UPDATE governance.chat_sessions
                SET scope_data=jsonb_set(scope_data, '{_scope_fingerprint}', '"changed-scope"'::jsonb)
                WHERE tenant_id=:tenant AND user_id=:user AND chat_session_id=:session
            """), {"tenant": context.tenant_id, "user": context.user_id, "session": suspended_session})
        mismatched = client.get(f"/chat/sessions/{suspended_session}/turns").json()["items"][0]
        assert not {"thread_id", "client_turn_id", "scope_fingerprint", "interaction_reason_code"} & set(mismatched)

        app.dependency_overrides[require_tenant_context] = lambda: other_user
        cross_user = client.get(f"/chat/sessions/{suspended_session}/turns")
        assert cross_user.status_code == 200
        assert cross_user.json()["items"] == []
    finally:
        app.dependency_overrides.pop(require_tenant_context, None)


def test_concurrent_same_client_turn_retry_creates_one_turn_and_execution() -> None:
    context = TenantContext(
        tenant_id="platform", user_id=f"concurrent-retry-{uuid4().hex}", role="admin",
        scope_type="platform",
    )
    _ensure_platform_test_identity(context)
    session_id = f"concurrent-session-{uuid4().hex}"
    client_turn_id = f"client-{uuid4().hex}"
    payload = {
        "seller_id": "1f50f920176a81dab994f9023523100",
        "metric_name": "gmv", "start_date": "2018-08-01",
        "end_date": "2018-08-31", "granularity": "day",
    }
    barrier = Barrier(2)

    def start_same_turn(trace_id: str):
        barrier.wait(timeout=10)
        return begin_chat_lifecycle(
            context, session_id=session_id, client_turn_id=client_turn_id,
            question="并发幂等性验收", explorer_payload=payload,
            thread_id=f"requested-{trace_id}", trace_id=trace_id,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        bindings = list(pool.map(lambda _: start_same_turn(str(uuid4())), range(2)))

    assert all(binding is not None and binding.execution is not None for binding in bindings)
    assert len({binding.turn.turn_id for binding in bindings}) == 1
    assert len({binding.execution.execution_id for binding in bindings}) == 1
    assert sum(bool(binding.active_reconnect) for binding in bindings) == 1
    with engine.begin() as connection:
        set_local_context(connection, context, scope_type="platform")
        turns = connection.execute(text("""
            SELECT turn_id FROM governance.chat_turns
            WHERE tenant_id=:tenant AND user_id=:user
              AND chat_session_id=:session AND client_turn_id=:client
        """), {"tenant": context.tenant_id, "user": context.user_id,
              "session": session_id, "client": client_turn_id}).scalars().all()
        executions = connection.execute(text("""
            SELECT execution_id FROM governance.conversation_executions
            WHERE tenant_id=:tenant AND user_id=:user AND turn_id=:turn
        """), {"tenant": context.tenant_id, "user": context.user_id,
              "turn": bindings[0].turn.turn_id}).scalars().all()
    assert turns == [bindings[0].turn.turn_id]
    assert executions == [bindings[0].execution.execution_id]


def test_system_repair_creates_second_execution_on_same_turn() -> None:
    """A bounded repair changes Execution, never creates a second Turn."""
    context = TenantContext(
        tenant_id="platform", user_id="system", role="admin", scope_type="platform",
    )
    _ensure_platform_test_identity(context)
    session_id = f"repair-test-{uuid4().hex}"
    client_turn_id = f"client-{uuid4().hex}"
    payload = {
        "seller_id": "1f50f920176a81dab994f9023523100",
        "metric_name": "gmv",
        "start_date": "2018-08-01",
        "end_date": "2018-08-03",
        "granularity": "day",
    }
    binding = begin_chat_lifecycle(
        context,
        session_id=session_id,
        client_turn_id=client_turn_id,
        question="系统修复测试",
        explorer_payload=payload,
        thread_id=f"thread-{uuid4().hex}",
        trace_id=str(uuid4()),
    )
    assert binding is not None and binding.execution is not None
    snapshot = preview_explorer(context, ExplorerQuery.model_validate(payload))
    controlled = controlled_scope(snapshot)
    authz = build_authorization_context(
        context,
        allowed_metric_ids=controlled.allowed_metrics,
        allowed_dimensions=controlled.allowed_dimensions,
        metric_scope_type="seller",
    )
    canonical = CanonicalScope.resolve(
        authz,
        seller_ids=[controlled.seller_id],
        start_date=controlled.default_start_date,
        end_date=controlled.default_end_date,
        metric_ids=controlled.allowed_metrics,
        dimensions=controlled.allowed_dimensions,
        granularity=controlled.granularity,
    )
    repaired = binding.service.create_repair_execution(
        binding.execution,
        trace_id=str(uuid4()),
        thread_id=f"repair-thread-{uuid4().hex}",
        query_scope=canonical.narrow(),
    )
    assert repaired.turn_id == binding.turn.turn_id
    assert repaired.attempt_no == 2
    assert binding.service.finalize(
        repaired, status="completed", answer="repaired", result={"answer": "repaired"}
    )
    with engine.begin() as connection:
        set_local_context(connection, context, scope_type="platform")
        rows = connection.execute(text("""
            SELECT e.attempt_no, e.status, t.status AS turn_status,
                   t.final_execution_id
            FROM governance.conversation_executions e
            JOIN governance.chat_turns t ON t.turn_id=e.turn_id
            WHERE t.client_turn_id=:client
            ORDER BY e.attempt_no
        """), {"client": client_turn_id}).mappings().all()
    assert [(row["attempt_no"], row["status"]) for row in rows] == [(1, "failed"), (2, "completed")]
    assert rows[-1]["turn_status"] == "completed"
    assert rows[-1]["final_execution_id"]


def test_task_progress_round_trips_across_chat_turns(monkeypatch, tmp_path) -> None:
    """Exercise HTTP → lifecycle/task DB → trace → next-turn context compilation.

    The graph/provider boundary is deterministic here; lifecycle, relation
    resolution, task-state persistence/readback, trace storage, and the real
    context compiler remain active against the test database.
    """
    import json

    from app.core.config import get_settings
    from app.graph import workflow
    from app.tasks.service import task_context

    context = TenantContext(
        tenant_id="platform", user_id="lifecycle-http-test", role="admin",
        scope_type="platform",
    )
    _ensure_platform_test_identity(context)
    session_id = f"task-progress-e2e-{uuid4().hex}"
    client = TestClient(app)
    previous_override = app.dependency_overrides.get(require_tenant_context)
    app.dependency_overrides[require_tenant_context] = lambda: context

    settings = get_settings()
    monkeypatch.setattr(settings, "relation_resolver_llm_enabled", True)
    monkeypatch.setattr(settings, "task_continuity_enabled", True)
    monkeypatch.setattr(settings, "trace_backend", "file")
    monkeypatch.setattr(settings, "trace_dir", str(tmp_path / "traces"))
    monkeypatch.setattr(workflow, "get_checkpointer", lambda: __import__(
        "langgraph.checkpoint.memory", fromlist=["MemorySaver"],
    ).MemorySaver())

    manifest = {"value": "manifest-a"}
    compiled_turns: list[dict] = []
    persisted_traces: list[str] = []

    def deterministic_relation(_system, raw_payload):
        payload = json.loads(raw_payload)
        candidates = payload.get("task_candidates") or []
        return {
            "action": "CONTINUE",
            "task_id": candidates[0]["task_id"],
            "reusable_evidence_ids": (payload.get("available_evidence_ids") or [])[:1],
            "reason": "same session follow-up",
        }

    def fake_workflow(*args, **kwargs):
        message = args[0]
        task_id = args[10]
        boundary = kwargs["authorization_boundary"]
        trace_id = kwargs["trace_id"]
        scope_fp = boundary["query_scope_fingerprint"]

        # On later turns, use the real task loader and compiler, with the
        # manifest that the request router resolved for this test turn.
        progress = workflow._task_progress_context(
            context, task_id, scope_fp, trusted_query_scope=True,
        )
        compiled = workflow.compile_turn_context({
            "user_message": message,
            "current_query_scope_fingerprint": scope_fp,
            "data_version_manifest_hash": manifest["value"],
            "task_progress_context": progress,
            "node_history": [],
        })
        compiled_turns.append({
            "used": compiled["context_compilation"]["task_progress_context_used"],
            "claims": list(compiled.get("historical_claims_context") or []),
            "task_id": task_id,
        })

        evidence_id = f"ev-{len(persisted_traces) + 1}"
        trace = {
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
            "controlled_analysis_scope": {"seller_id": "1f50f920176a81dab994f9023523100"},
            "current_query_scope_fingerprint": scope_fp,
            "data_version_manifest_hash": manifest["value"],
            "critic_verdict": "approved",
            "claims": [{"value": f"approved claim {evidence_id}", "source_evidence_ids": [evidence_id]}],
            "evidence_bundles": [{
                "evidence_id": evidence_id, "source_type": "test", "source_ref": "fixture",
                "trusted": True,
            }],
            "node_history": [{"node": "triage"}, {"node": "data_query"}],
            "task_type": "metric_query",
        }
        from app.harness.storage import write_trace
        write_trace(trace_id, trace)
        persisted_traces.append(trace_id)
        return {
            "task_type": "metric_query", "answer": "deterministic test answer",
            "trace_id": trace_id, "error": False, "errors": [],
            "clarification": None, "presentation": None,
            "turn_context": {"context_slots": {"metric": {"value": "gmv"}}},
        }

    monkeypatch.setattr("app.api.routes.chat._expected_data_manifest_hash", lambda *_: manifest["value"])
    monkeypatch.setattr("app.api.routes.chat.run_chat_workflow", fake_workflow)
    monkeypatch.setattr("app.llm.client.llm_extract", deterministic_relation)

    base = {
        "use_current_explorer_context": True,
        "explorer_context": {
            "seller_id": "1f50f920176a81dab994f9023523100",
            "metric_name": "gmv", "start_date": "2018-08-01",
            "end_date": "2018-08-03", "granularity": "day",
        },
        "memory_enabled": False,
        "chat_session_id": session_id,
    }
    try:
        first = client.post("/chat", json={
            **base, "message": "请分析 GMV", "chat_turn_id": f"turn-{uuid4().hex}",
        })
        assert first.status_code == 200, first.text
        first_tasks = task_context(context, session_id)["tasks"]
        assert len(first_tasks) == 1
        task_id = first_tasks[0]["task_id"]
        first_state = first_tasks[0]["task_state"]
        assert first_state["approved_claims"] == [{
            "value": "approved claim ev-1", "source_evidence_ids": ["ev-1"],
        }]
        assert first_state["evidence_refs"][0]["evidence_id"] == "ev-1"

        second = client.post("/chat", json={
            **base, "message": "按品类拆开看看", "chat_turn_id": f"turn-{uuid4().hex}",
        })
        assert second.status_code == 200, second.text
        assert compiled_turns[-1]["task_id"] == task_id
        assert compiled_turns[-1]["used"] is True
        assert compiled_turns[-1]["claims"][0]["value"] == "approved claim ev-1"

        # A new sealed data version keeps task identity but makes old findings
        # unavailable to relation and context compilation.
        manifest["value"] = "manifest-b"
        third = client.post("/chat", json={
            **base, "message": "继续分析这个变化", "chat_turn_id": f"turn-{uuid4().hex}",
        })
        assert third.status_code == 200, third.text
        assert compiled_turns[-1]["used"] is False
        assert compiled_turns[-1]["claims"] == []
        assert task_context(context, session_id)["tasks"][0]["task_state"]["evidence_refs"] == [
            {"evidence_id": "ev-3", "source_type": "test", "source_ref": "fixture", "trusted": True},
        ]
    finally:
        if previous_override is None:
            app.dependency_overrides.pop(require_tenant_context, None)
        else:
            app.dependency_overrides[require_tenant_context] = previous_override


def test_task_progress_reaches_real_graph_context_compiler(monkeypatch, tmp_path) -> None:
    """Run two real LangGraph turns; stub only model/retrieval provider edges."""
    import json

    from app.core.config import get_settings
    from app.graph import workflow
    from app.rag.schemas import (
        RagCitation, RagRetrievalSignals, RagSearchHit, RagSearchResult,
    )
    from app.tasks.service import task_context

    context = TenantContext(
        tenant_id="platform", user_id="lifecycle-http-test", role="admin",
        scope_type="platform",
    )
    _ensure_platform_test_identity(context)
    session_id = f"task-progress-graph-{uuid4().hex}"
    client = TestClient(app)
    previous_override = app.dependency_overrides.get(require_tenant_context)
    app.dependency_overrides[require_tenant_context] = lambda: context

    settings = get_settings()
    monkeypatch.setattr(settings, "relation_resolver_llm_enabled", True)
    monkeypatch.setattr(settings, "task_continuity_enabled", True)
    monkeypatch.setattr(settings, "data_versioning_enabled", False)
    monkeypatch.setattr(settings, "rag_release_controller_enabled", False)
    monkeypatch.setattr(settings, "trace_backend", "file")
    monkeypatch.setattr(settings, "trace_dir", str(tmp_path / "graph-traces"))
    monkeypatch.setattr(workflow, "get_checkpointer", lambda: __import__(
        "langgraph.checkpoint.memory", fromlist=["MemorySaver"],
    ).MemorySaver())

    from app.graph.nodes import triage as triage_node
    from app.graph.nodes import metric_definition as definition_node

    monkeypatch.setattr(triage_node, "llm_classify", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(definition_node, "published_rag_version", lambda: "continuity-e2e")
    original_resolve_scope = workflow.resolve_explorer_context

    def resolve_scope_with_test_manifest(state):
        state = original_resolve_scope(state)
        state["data_version_manifest_hash"] = "manifest-e2e"
        return state

    monkeypatch.setattr(workflow, "resolve_explorer_context", resolve_scope_with_test_manifest)

    def retrieval(query, **_kwargs):
        citation = RagCitation(
            source_path="test://gmv-definition", title="GMV 定义测试材料",
            doc_type="metric", metric_name="gmv", chunk_id="gmv-definition-e2e",
        )
        hit = RagSearchHit(
            chunk_id="gmv-definition-e2e", text="GMV 是成交商品金额与运费之和。",
            score=1.0, citation=citation,
            signals=RagRetrievalSignals(rerank_score=1.0),
        )
        return RagSearchResult(
            query=query, backend="local", strategy="local_hybrid", top_k=5,
            hits=[hit], context=hit.text, index_version="continuity-e2e",
        )

    monkeypatch.setattr(definition_node, "retrieve_knowledge", retrieval)
    monkeypatch.setattr(
        "app.llm.client.llm_chat",
        lambda **_kwargs: "GMV 是已支付成交商品金额与运费的合计。",
    )

    def relation_extract(_system, raw_payload):
        payload = json.loads(raw_payload)
        if "task_candidates" not in payload:
            return None
        candidate = payload["task_candidates"][0]
        return {
            "action": "CONTINUE", "task_id": candidate["task_id"],
            "reusable_evidence_ids": payload.get("available_evidence_ids", [])[:1],
            "reason": "same-session ordinary follow-up",
        }

    monkeypatch.setattr("app.llm.client.llm_extract", relation_extract)
    monkeypatch.setattr("app.api.routes.chat._expected_data_manifest_hash", lambda *_: "manifest-e2e")

    from app.harness.storage import read_trace, write_trace as persist_trace_file
    trace_count = {"value": 0}

    def write_trace_with_approved_fixture(trace_id, state):
        trace_count["value"] += 1
        if trace_count["value"] == 1:
            state = dict(state)
            state["critic_verdict"] = "approved"
            state["claims"] = [{
                "value": "fixture approved conclusion",
                "source_evidence_ids": ["ev-continuity-fixture"],
            }]
            state["evidence_bundles"] = list(state.get("evidence_bundles") or []) + [{
                "evidence_id": "ev-continuity-fixture", "source_type": "test",
                "source_ref": "approved-fixture", "trusted": True,
            }]
        persist_trace_file(trace_id, state)

    monkeypatch.setattr(workflow, "write_trace", write_trace_with_approved_fixture)

    base = {
        "use_current_explorer_context": True,
        "explorer_context": {
            "seller_id": "1f50f920176a81dab994f9023523100",
            "metric_name": "gmv", "start_date": "2018-08-01",
            "end_date": "2018-08-03", "granularity": "day",
        },
        "memory_enabled": False,
        "chat_session_id": session_id,
    }
    try:
        first = client.post("/chat", json={
            **base, "message": "GMV 是什么？", "chat_turn_id": f"turn-{uuid4().hex}",
        })
        assert first.status_code == 200, first.text
        first_trace = read_trace(first.json()["trace_id"], context)["state_snapshot"]
        first_nodes = {item["node"] for item in first_trace["node_history"]}
        assert {"context_compiler", "triage", "metric_definition", "generate_answer", "critic", "persist_trace"} <= first_nodes

        second = client.post("/chat", json={
            **base, "message": "这个指标口径呢？", "chat_turn_id": f"turn-{uuid4().hex}",
        })
        assert second.status_code == 200, second.text
        second_trace = read_trace(second.json()["trace_id"], context)["state_snapshot"]
        assert second_trace["context_compilation"]["task_progress_context_used"] is True
        assert second_trace["historical_claims_context"][0]["value"] == "fixture approved conclusion"
        second_nodes = {item["node"] for item in second_trace["node_history"]}
        assert {"context_compiler", "triage", "metric_definition", "generate_answer", "critic", "persist_trace"} <= second_nodes
        assert task_context(context, session_id)["tasks"][0]["task_state"]["recent_turns"][-1]["status"] == "completed"
    finally:
        if previous_override is None:
            app.dependency_overrides.pop(require_tenant_context, None)
        else:
            app.dependency_overrides[require_tenant_context] = previous_override


def test_relation_version_preflight_uses_validated_explorer_query(monkeypatch) -> None:
    from app.api.routes.chat import _expected_data_manifest_hash
    from app.core.config import get_settings
    from app.schemas.explorer import ExplorerQuery

    context = TenantContext(
        tenant_id="platform", user_id="lifecycle-http-test", role="admin",
        scope_type="platform",
    )
    _ensure_platform_test_identity(context)
    monkeypatch.setattr(get_settings(), "task_continuity_enabled", True)
    monkeypatch.setattr(get_settings(), "data_versioning_enabled", True)
    monkeypatch.setattr(
        "app.data_versions.service.current_gmv_manifest_hash",
        lambda *_args, **_kwargs: "manifest-test",
    )
    query = ExplorerQuery.model_validate({
        "seller_id": "1f50f920176a81dab994f9023523100",
        "metric_name": "gmv", "start_date": "2018-08-01",
        "end_date": "2018-08-03", "granularity": "day",
    })
    assert _expected_data_manifest_hash(context, query) == "manifest-test"

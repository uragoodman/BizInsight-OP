"""Run WP5 Turn Commit, focus, summary and HITL contracts on an isolated clone.

Required environment: ISOLATED_TEST_POSTGRES_ADMIN_URL. The template defaults to
WP2_TEST_POSTGRES_TEMPLATE or ``bizinsight``. The configured application DB is
never used; the clone is migrated and dropped by this script.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import tempfile
import traceback
import uuid
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from typing import TypedDict

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    raw_admin_url = os.getenv("ISOLATED_TEST_POSTGRES_ADMIN_URL", "").strip()
    template = os.getenv("WP5_TEST_POSTGRES_TEMPLATE") or os.getenv("WP2_TEST_POSTGRES_TEMPLATE", "bizinsight")
    if not raw_admin_url:
        print("WP5_TURN_COMMIT_SKIPPED: ISOLATED_TEST_POSTGRES_ADMIN_URL is not set")
        return 2
    admin_url = make_url(raw_admin_url)
    admin = create_engine(admin_url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    database = f"wp5_turn_commit_{uuid.uuid4().hex[:10]}"
    try:
        with admin.connect() as conn:
            if not conn.execute(text("SELECT 1 FROM pg_database WHERE datname=:name"), {"name": template}).scalar():
                print(f"WP5_TURN_COMMIT_SKIPPED: template database {template!r} does not exist")
                return 2
            try:
                conn.execute(text(f'CREATE DATABASE "{database}" TEMPLATE "{template}"'))
            except Exception as exc:  # noqa: BLE001 - template may have an active app connection
                docker_container = os.getenv("WP5_TEST_POSTGRES_DOCKER_CONTAINER", "").strip()
                if not docker_container:
                    raise
                # PostgreSQL cannot clone a template while any app connection
                # is open. A custom-format dump is consistent under MVCC and
                # avoids terminating the project's live worker connection.
                conn.execute(text(f'CREATE DATABASE "{database}" TEMPLATE template0'))
                dump = subprocess.run(
                    ["docker", "exec", docker_container, "pg_dump", "-U", "postgres", "-Fc", "-d", template],
                    capture_output=True, timeout=300,
                )
                if dump.returncode:
                    raise RuntimeError(f"pg_dump failed: {dump.stderr[-2000:]!r}") from exc
                restore = subprocess.run(
                    ["docker", "exec", "-i", docker_container, "pg_restore", "-U", "postgres",
                     "--no-owner", "--no-privileges", "-d", database],
                    input=dump.stdout, capture_output=True, timeout=300,
                )
                if restore.returncode:
                    raise RuntimeError(f"pg_restore failed: {restore.stderr[-3000:]!r}") from exc
        db_url = admin_url.set(database=database)
        env = {
            **os.environ,
            "POSTGRES_URL": db_url.render_as_string(hide_password=False),
            "TENANT_AUTH_REQUIRED": "false",
            "ABAC_DATABASE_ENABLED": "false",
            "TRACE_BACKEND": "file",
            "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        }
        migrated = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=300,
        )
        if migrated.returncode:
            raise RuntimeError(f"isolated Alembic upgrade failed\n{migrated.stdout[-3000:]}\n{migrated.stderr[-3000:]}")
        os.environ.update({key: env[key] for key in (
            "POSTGRES_URL", "TENANT_AUTH_REQUIRED", "ABAC_DATABASE_ENABLED", "TRACE_BACKEND",
        )})
        return _exercise(db_url)
    finally:
        try:
            with admin.connect() as conn:
                conn.execute(text(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)'))
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        admin.dispose()


def _exercise(db_url) -> int:
    from app.auth.context import build_authorization_context
    from app.chat_history.repository import ChatHistoryRepository
    from app.chat_lifecycle.service import LifecycleService
    from app.data.database import engine
    from app.data.repositories.explorer import controlled_scope, preview_explorer
    from app.schemas.explorer import ExplorerQuery
    from app.scope.canonical import CanonicalScope
    from app.tasks.service import create_task, task_context
    from app.tenancy import TenantContext

    context = TenantContext(tenant_id="platform", user_id=f"wp5-{uuid.uuid4().hex[:10]}", role="admin", scope_type="platform")
    with engine.begin() as conn:
        conn.execute(text("""INSERT INTO iam.tenants(tenant_id,tenant_name,scope_type,status)
            VALUES ('platform','WP5 isolated verification','platform','active') ON CONFLICT DO NOTHING"""))
        conn.execute(text("""INSERT INTO iam.users(user_id,display_name,email,status)
            VALUES (:user,'WP5 isolated verification',:email,'active') ON CONFLICT DO NOTHING"""),
            {"user": context.user_id, "email": f"{context.user_id}@example.invalid"})
        conn.execute(text("""INSERT INTO iam.tenant_memberships(tenant_id,user_id,role)
            VALUES ('platform',:user,'admin') ON CONFLICT DO NOTHING"""), {"user": context.user_id})
        seller, start, end = conn.execute(text("""SELECT seller_id, MIN(purchase_date), MAX(purchase_date)
            FROM analytics.v_delivered_order_items GROUP BY seller_id ORDER BY COUNT(*) DESC LIMIT 1""")).one()
    payload = {"seller_id": seller, "metric_name": "gmv", "start_date": str(start), "end_date": str(end), "granularity": "day"}
    controlled = controlled_scope(preview_explorer(context, ExplorerQuery.model_validate(payload)))
    authz = build_authorization_context(context, allowed_metric_ids=controlled.allowed_metrics,
        allowed_dimensions=controlled.allowed_dimensions, metric_scope_type="seller")
    scope = CanonicalScope.resolve(authz, seller_ids=[controlled.seller_id],
        start_date=controlled.default_start_date, end_date=controlled.default_end_date,
        metric_ids=controlled.allowed_metrics, dimensions=controlled.allowed_dimensions,
        granularity=controlled.granularity)
    service = LifecycleService(context, authz)
    session_id = f"wp5-session-{uuid.uuid4().hex}"
    tasks = [create_task(context, session_id, f"task-{i}", payload, status="running")["task_id"] for i in range(2)]
    turns = []
    executions = []
    for i in range(22):
        turn = service.create_turn(session_id=session_id, question=f"question-{i+1}", scope=scope,
            client_turn_id=f"client-{i}-{uuid.uuid4().hex[:6]}")
        execution = service.create_execution(turn, trace_id=f"trace-{i}", thread_id=f"thread-{i}", query_scope=scope.narrow())
        turns.append(turn)
        executions.append(execution)
        if i == 0:
            old_execution = execution
            assert task_context(context, session_id)["current_task"] is None, "task context ignored the cleared session focus"
        if i == 1:
            new_execution = execution
            assert not service.bind_active_task(old_execution, tasks[0]), "stale execution reclaimed task focus"
            assert service.bind_active_task(new_execution, tasks[1]), "current execution could not bind its task"
            assert task_context(context, session_id)["current_task"]["task_id"] == tasks[1]
            assert service.finalize(
                old_execution, status="completed", answer="answer-1",
                result={"answer": "answer-1", "turn_resolution": {
                    "schema_version": "turn-resolution-v1",
                    "unresolved_fields": ["date"],
                }},
            )
            with engine.begin() as conn:
                session_row = conn.execute(text("""SELECT focus_owner_execution_id,active_task_id,open_questions
                    FROM governance.chat_sessions
                    WHERE tenant_id=:tenant AND user_id=:user AND chat_session_id=:session"""),
                    {"tenant": context.tenant_id, "user": context.user_id, "session": session_id}).mappings().one()
            assert session_row["focus_owner_execution_id"] == new_execution.execution_id
            assert session_row["active_task_id"] == tasks[1]
            assert session_row["open_questions"] == [{
                "field": "date", "reason": "unresolved", "source_turn_id": old_execution.turn_id,
            }]
            with engine.begin() as conn:
                resolution_snapshot = conn.execute(text("""SELECT context_snapshot->'turn_resolution'
                    FROM governance.chat_turns WHERE turn_id=:turn"""), {"turn": old_execution.turn_id}).scalar_one()
            assert resolution_snapshot["unresolved_fields"] == ["date"]
        if i != 0:
            assert service.finalize(
                execution, status="completed", answer=f"answer-{i+1}", result={"answer": f"answer-{i+1}"},
                task_id=tasks[1] if i == 1 else None,
                task_status="completed" if i == 1 else None,
            )
    assert service.finalize(old_execution, status="completed", answer="answer-1", result={"answer": "answer-1"})
    with engine.begin() as conn:
        commits = conn.execute(text("""SELECT count(*), min(commit_version), max(commit_version)
            FROM governance.chat_turns WHERE tenant_id=:tenant AND user_id=:user AND chat_session_id=:session"""),
            {"tenant": context.tenant_id, "user": context.user_id, "session": session_id}).one()
    assert tuple(commits) == (22, 1, 22), commits
    prior = ChatHistoryRepository(context).latest_completed_context(session_id)
    assert prior and len(prior["recent_committed_turns"]) == 20
    assert prior["summary_covered_from_version"] == 1
    assert prior["summary_covered_to_version"] == 2
    assert prior["summary_generated_at"] is not None
    assert "question-1" in prior["conversation_summary"] and "question-2" in prior["conversation_summary"]
    assert [row["commit_version"] for row in prior["recent_committed_turns"]] == list(range(3, 23))
    assert "question-3" not in prior["conversation_summary"]
    with engine.begin() as conn:
        task_row = conn.execute(text("""SELECT status,summary_data FROM governance.conversation_tasks
            WHERE task_id=:task AND tenant_id=:tenant AND user_id=:user"""),
            {"task": tasks[1], "tenant": context.tenant_id, "user": context.user_id}).mappings().one()
    assert task_row["status"] == "completed"
    assert task_row["summary_data"]["last_committed_turn_id"] == turns[1].turn_id

    # HITL claim -> Turn Commit -> replay is persisted as one transaction.
    turn = service.create_turn(session_id=session_id, question="review", scope=scope,
        client_turn_id=f"review-{uuid.uuid4().hex}")
    execution = service.create_execution(turn, trace_id="review-trace", thread_id="review-thread", query_scope=scope.narrow())
    review_task = create_task(context, session_id, "review task", payload, status="running")["task_id"]
    assert service.bind_active_task(execution, review_task)
    assert service.finalize(execution, status="suspended", result={"event": "suspended"},
        interaction_state="awaiting_human_input", task_id=review_task, task_status="awaiting_confirmation")
    decision = {"action": "approve", "actor": "reviewer"}
    claim, interaction_id, replay = service.claim_review_interaction(execution, node_name="human_review", interrupt_index="0", decision=decision)
    assert claim == "claimed" and interaction_id and replay is None
    claim, pending_id, replay = service.claim_review_interaction(execution, node_name="human_review", interrupt_index="0", decision=decision)
    assert claim == "processing" and pending_id == interaction_id and replay is None
    stale_claim_token = service._review_claim_tokens[interaction_id]
    with engine.begin() as conn:
        conn.execute(text("""UPDATE governance.chat_review_interactions
            SET lease_expires_at=now()-interval '1 second' WHERE interaction_id=:id"""),
            {"id": interaction_id})
    claim, reclaimed_id, replay = service.claim_review_interaction(
        execution, node_name="human_review", interrupt_index="0", decision=decision,
    )
    assert claim == "claimed" and reclaimed_id == interaction_id and replay is None
    active_claim_token = service._review_claim_tokens[interaction_id]
    assert active_claim_token != stale_claim_token
    assert not service.renew_review_interaction(interaction_id, stale_claim_token)
    assert service.renew_review_interaction(interaction_id, active_claim_token)
    claim, pending_id, replay = service.claim_review_interaction(execution, node_name="human_review", interrupt_index="0", decision=decision)
    assert claim == "processing" and pending_id == interaction_id and replay is None
    claim, _, _ = service.claim_review_interaction(execution, node_name="human_review", interrupt_index="0", decision={"action": "reject"})
    assert claim == "conflict"
    result = {"answer": "approved"}
    try:
        service.finalize(
            execution, status="completed", answer="approved", result=result,
            task_id=review_task, task_status="completed",
            review_interaction_id="missing-review-interaction", review_interaction_result=result,
        )
    except RuntimeError as exc:
        assert str(exc) == "human_review_interaction_commit_mismatch"
    else:
        raise AssertionError("missing HITL receipt must roll back the Turn Commit")
    assert service.commit_metadata(turn.turn_id).get("commit_version") is None
    claim, retry_id, replay = service.claim_review_interaction(
        execution, node_name="human_review", interrupt_index="0", decision=decision,
    )
    assert claim == "processing" and retry_id == interaction_id and replay is None
    assert service.finalize(execution, status="completed", answer="approved", result=result,
        task_id=review_task, task_status="completed",
        review_interaction_id=interaction_id, review_interaction_result=result)
    claim, replay_id, replay = service.claim_review_interaction(execution, node_name="human_review", interrupt_index="0", decision=decision)
    assert claim == "replay" and replay_id == interaction_id and replay == result
    assert service.commit_metadata(turn.turn_id)["commit_version"] == 23

    _exercise_checkpointed_hitl_recovery(context, authz, scope, engine)

    # Deleting the canonical conversation also removes its legacy L1 memory
    # snapshot; the summary itself lives on chat_sessions and cascades with it.
    with engine.begin() as conn:
        conn.execute(text("""INSERT INTO governance.memory_sessions
            (tenant_id,user_id,memory_session_id,session_summary,last_scope_fingerprint,
             last_seller_id,last_start_date,last_end_date,last_trace_id,expires_at)
            VALUES (:tenant,:user,:session,'legacy snapshot',:fingerprint,:seller,:start,:end,'legacy-trace',now()+interval '1 day')"""), {
            "tenant": context.tenant_id, "user": context.user_id, "session": session_id,
            "fingerprint": "a" * 64, "seller": seller, "start": start, "end": end,
        })
    assert ChatHistoryRepository(context).delete_session(session_id)
    with engine.begin() as conn:
        remaining = conn.execute(text("""SELECT
            (SELECT count(*) FROM governance.chat_sessions WHERE tenant_id=:tenant AND user_id=:user AND chat_session_id=:session),
            (SELECT count(*) FROM governance.chat_turns WHERE tenant_id=:tenant AND user_id=:user AND chat_session_id=:session),
            (SELECT count(*) FROM governance.memory_sessions WHERE tenant_id=:tenant AND user_id=:user AND memory_session_id=:session)"""), {
            "tenant": context.tenant_id, "user": context.user_id, "session": session_id,
        }).one()
    assert tuple(remaining) == (0, 0, 0), remaining
    print("WP5_TURN_COMMIT_OK: migration, focus CAS, summary window/deletion, PostgreSQL+LangGraph HITL recovery/replay")
    engine.dispose()
    return 0


def _exercise_checkpointed_hitl_recovery(context, authz, scope, engine) -> None:
    """Exercise the real interrupt/checkpoint and PG recovery boundary together."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    from app.analysis.publishing import ReportStore
    from app.chat_history.repository import ChatHistoryRepository
    from app.chat_lifecycle.service import LifecycleService
    from app.api.auth import require_analysis_write, require_tenant_context
    from app.api.routes.chat import router as chat_router
    from app.graph.nodes import human_review as human_review_node
    from app.graph import workflow as workflow_module

    class ReviewState(TypedDict, total=False):
        thread_id: str
        trace_id: str
        tenant_id: str
        authorization_boundary: dict
        current_query_scope_fingerprint: str
        controlled_analysis_scope: dict
        report_id: str
        report_version: int
        answer: str
        final_answer: str
        critic_result: dict
        report_risks: list
        task_type: str
        errors: list
        node_history: list
        human_decision: dict
        critic_verdict: str
        current_agent: str

    session_id = ChatHistoryRepository(context).create_session("WP5 checkpoint recovery")["id"]
    service = LifecycleService(context, authz)
    turn = service.create_turn(
        session_id=session_id, question="review report", scope=scope,
        client_turn_id=f"wp5-hitl-{uuid.uuid4().hex}",
    )
    thread_id = f"wp5-hitl-thread-{uuid.uuid4().hex}"
    execution = service.create_execution(
        turn, trace_id=f"wp5-hitl-trace-{uuid.uuid4().hex}", thread_id=thread_id,
        query_scope=scope.narrow(),
    )
    raw_scope = {
        "allowed_metrics": list(authz.allowed_metric_ids),
        "allowed_dimensions": list(authz.allowed_dimensions),
    }
    boundary = {
        "permission_version": authz.permission_version,
        "policy_set_version": authz.policy_set_version,
        "authorization_fingerprint": authz.authz_digest,
        "query_scope_fingerprint": scope.query_scope_fingerprint,
    }
    state: ReviewState = {
        "thread_id": thread_id, "trace_id": execution.trace_id,
        "tenant_id": context.tenant_id, "authorization_boundary": boundary,
        "current_query_scope_fingerprint": scope.query_scope_fingerprint,
        "controlled_analysis_scope": raw_scope,
        "report_id": f"report-{uuid.uuid4().hex}", "report_version": 1,
        "answer": "draft report answer", "final_answer": "draft report answer",
        "critic_result": {}, "report_risks": [], "task_type": "report",
        "errors": [], "node_history": [],
    }

    with tempfile.TemporaryDirectory(prefix="wp5_report_") as temp_dir:
        report_db = str(Path(temp_dir) / "reports.sqlite")
        store = ReportStore(report_db)
        with closing(sqlite3.connect(report_db)) as conn:
            with conn:
                conn.execute("""INSERT INTO report_versions
                    (report_id,version,status,content,content_hash,report_json,created_at,actor,reason)
                    VALUES (?,1,'draft','draft report answer','draft-hash','{}','2026-10-31T00:00:00+00:00','system','')""",
                    (state["report_id"],))

        builder = StateGraph(ReviewState)
        builder.add_node("human_review", human_review_node.human_review)
        builder.add_node("persist_trace", lambda current: current)
        builder.add_edge(START, "human_review")
        builder.add_edge("human_review", "persist_trace")
        builder.add_edge("persist_trace", END)
        graph = builder.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": thread_id}}
        app = FastAPI()
        app.include_router(chat_router)
        app.dependency_overrides[require_analysis_write] = lambda: "reviewer"
        app.dependency_overrides[require_tenant_context] = lambda: context
        client = TestClient(app)
        with patch.object(human_review_node, "write_trace", lambda *_args, **_kwargs: None), \
             patch.object(human_review_node, "get_settings", lambda: SimpleNamespace(phase6_store_path=report_db)):
            first = graph.invoke(state, config=config)
            assert first.get("__interrupt__"), "real LangGraph node did not pause at interrupt"
            assert tuple(graph.get_state(config).next) == ("human_review",)
            service.finalize(
                execution, status="suspended", result={"answer": state["answer"]},
                interaction_state="awaiting_human_input",
            )

            decision = {"action": "approve", "actor": "reviewer", "reason": "verified"}
            # Simulate a request process dying after it claims the interaction
            # but before it can resume the still-pending graph checkpoint.
            with patch.object(workflow_module, "build_chat_graph", lambda **_kwargs: graph), \
                 patch.object(workflow_module, "get_settings", lambda: SimpleNamespace(checkpoint_durability="sync")), \
                 patch.object(graph, "invoke", side_effect=RuntimeError("injected_preinvoke_crash")):
                abandoned_response = client.post(f"/chat/{thread_id}/review", json=decision)
            assert abandoned_response.status_code == 409, abandoned_response.text
            assert tuple(graph.get_state(config).next) == ("human_review",)
            with engine.begin() as conn:
                conn.execute(text("""UPDATE governance.chat_review_interactions
                    SET lease_expires_at=now()-interval '1 second'
                    WHERE execution_id=:execution AND node_name='human_review' AND interrupt_index='0'"""),
                    {"execution": execution.execution_id})

            real_finalize = LifecycleService.finalize
            injected = {"failed": False}

            def fail_after_graph_checkpoint(self, target_execution, **kwargs):
                if kwargs.get("review_interaction_id") and not injected["failed"]:
                    injected["failed"] = True
                    raise RuntimeError("injected_turn_commit_failure")
                return real_finalize(self, target_execution, **kwargs)

            with patch.object(workflow_module, "build_chat_graph", lambda **_kwargs: graph), \
                 patch.object(workflow_module, "get_settings", lambda: SimpleNamespace(checkpoint_durability="sync")), \
                 patch.object(LifecycleService, "finalize", fail_after_graph_checkpoint):
                failed_response = client.post(f"/chat/{thread_id}/review", json=decision)
                assert failed_response.status_code == 409, failed_response.text
                assert "injected_turn_commit_failure" in failed_response.text

            checkpoint = graph.get_state(config)
            assert not checkpoint.next
            assert checkpoint.values["human_decision"]["interaction_index"] == "0"
            with engine.begin() as conn:
                interaction = conn.execute(text("""SELECT status FROM governance.chat_review_interactions
                    WHERE execution_id=:execution AND node_name='human_review' AND interrupt_index='0'"""),
                    {"execution": execution.execution_id}).scalar_one()
            assert interaction == "processing"

            # Same request sees a completed graph checkpoint; recovery commits
            # from saved state and must not invoke the graph or node again.
            with patch.object(workflow_module, "build_chat_graph", lambda **_kwargs: graph), \
                 patch.object(workflow_module, "get_settings", lambda: SimpleNamespace(checkpoint_durability="sync")):
                recovered_response = client.post(f"/chat/{thread_id}/review", json=decision)
            assert recovered_response.status_code == 200, recovered_response.text
            response = recovered_response.json()
            assert {key: response[key] for key in ("task_type", "answer", "trace_id")} == {
                "task_type": "report", "answer": "draft report answer", "trace_id": execution.trace_id,
            }
            assert response["durable_turn_id"] == turn.turn_id
            assert response["execution_id"] == execution.execution_id
            assert service.commit_metadata(turn.turn_id)["final_execution_id"] == execution.execution_id
            with closing(sqlite3.connect(report_db)) as conn:
                counts = conn.execute("""SELECT
                    (SELECT count(*) FROM report_approvals WHERE report_id=?),
                    (SELECT count(*) FROM report_versions WHERE report_id=?),
                    (SELECT status FROM report_versions WHERE report_id=? AND version=1)""",
                    (state["report_id"], state["report_id"], state["report_id"])).fetchone()
            assert tuple(counts) == (1, 1, "published"), counts


if __name__ == "__main__":
    raise SystemExit(main())

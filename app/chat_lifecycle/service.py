"""Durable Turn/Execution lifecycle operations.

The graph is an execution engine; this service is the authority for durable
chat state.  It intentionally uses conditional SQL updates for terminal
transitions so a late worker or an SSE retry cannot overwrite the winner.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from secrets import token_hex
from typing import Any

from sqlalchemy import text

from app.auth.context import AuthorizationContext
from app.data.database import engine
from app.data.rls import set_local_context
from app.scope.canonical import CanonicalScope, QueryScope
from app.tenancy import TenantContext
from app.observability import record_event, record_measurement


def _id(prefix: str) -> str:
    return f"{prefix}_{token_hex(12)}"


def _normalise_open_questions(value: Any, *, source_turn_id: str | None = None) -> list[dict[str, str]]:
    """Persist only bounded unresolved-field metadata, never user-authored text."""
    if not isinstance(value, list):
        return []
    questions: list[dict[str, str]] = []
    for item in value[:20]:
        if isinstance(item, str):
            field_name, reason = item.strip(), "unresolved"
        elif isinstance(item, dict):
            field_name = str(item.get("field") or item.get("name") or "").strip()
            reason = str(item.get("reason") or "unresolved").strip()
        else:
            continue
        if not field_name:
            continue
        questions.append({
            "field": field_name[:80],
            "reason": reason[:80] or "unresolved",
            **({"source_turn_id": source_turn_id[:128]} if source_turn_id else {}),
        })
    return questions


@dataclass(frozen=True, slots=True)
class TurnHandle:
    turn_id: str
    session_id: str
    client_turn_id: str | None
    authz_ref: str
    permission_version: str
    policy_set_version: str
    authorization_fingerprint: str
    scope_fingerprint: str
    status: str = "queued"
    task_type: str = "pending"
    answer: str | None = None
    trace_id: str = ""
    result_snapshot: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ExecutionHandle:
    execution_id: str
    turn_id: str
    attempt_no: int
    trace_id: str
    thread_id: str
    authorization_fingerprint: str
    query_scope_fingerprint: str


class LifecycleService:
    def __init__(self, context: TenantContext, authz: AuthorizationContext) -> None:
        if context.tenant_id != authz.tenant_id or context.user_id != authz.user_id:
            raise PermissionError("TenantContext and AuthorizationContext do not match")
        self.context = context
        self.authz = authz

    def _scope(self, conn) -> None:
        set_local_context(conn, self.context)

    def _save_authorization_snapshot(self, conn) -> None:
        from app.metrics.registry import get_registry

        conn.execute(text("""
            INSERT INTO governance.authorization_snapshots
              (authz_ref, tenant_id, user_id, permission_version,
               policy_set_version, snapshot_digest, seller_scope_type,
               metric_policy_ref, dimension_policy_ref, tool_policy_ref)
            VALUES (:ref, :tenant, :user, :permission, :policy, :digest,
                    :scope_type, :metric_ref, :dimension_ref, :tool_ref)
            ON CONFLICT (authz_ref) DO NOTHING
        """), {
            "ref": self.authz.authz_ref,
            "tenant": self.authz.tenant_id,
            "user": self.authz.user_id,
            "permission": self.authz.permission_version,
            "policy": self.authz.policy_set_version,
            "digest": self.authz.authz_digest,
            "scope_type": self.authz.scope_type,
            "metric_ref": f"metrics:{self.authz.permission_version}",
            "dimension_ref": f"dimensions:{self.authz.policy_set_version}",
            "tool_ref": f"tools:{self.authz.policy_set_version}",
        })
        conn.execute(text("""
            INSERT INTO governance.policy_versions
              (tenant_id, user_id, permission_version, policy_set_version,
               catalog_digest, authz_digest, policy_json)
            VALUES (:tenant, :user, :permission, :policy, :catalog, :digest,
                    CAST(:policy_json AS jsonb))
            ON CONFLICT (tenant_id, user_id, permission_version, policy_set_version)
            DO UPDATE SET authz_digest=EXCLUDED.authz_digest,
                          catalog_digest=EXCLUDED.catalog_digest,
                          policy_json=EXCLUDED.policy_json
        """), {
            "tenant": self.authz.tenant_id,
            "user": self.authz.user_id,
            "permission": self.authz.permission_version,
            "policy": self.authz.policy_set_version,
            "catalog": get_registry().catalog_digest,
            "digest": self.authz.authz_digest,
            "policy_json": json.dumps(self.authz.as_audit_dict(), ensure_ascii=False),
        })

    def _audit(self, event_name: str, *, turn_id: str | None = None,
               execution_id: str | None = None, trace_id: str | None = None,
               payload: dict[str, Any] | None = None) -> None:
        """Write durable audit evidence without breaking rolling migrations."""
        attributes = {
            "tenant_id": self.context.tenant_id,
            "user_id": self.context.user_id,
            "authz_ref": self.authz.authz_ref,
            "permission_version": self.authz.permission_version,
            "policy_set_version": self.authz.policy_set_version,
            "turn_id": turn_id,
            "execution_id": execution_id,
            "trace_id": trace_id,
            **(payload or {}),
        }
        record_event(event_name, attributes)
        try:
            with engine.begin() as conn:
                self._scope(conn)
                conn.execute(text("""
                    INSERT INTO governance.lifecycle_audit_events
                      (event_name,tenant_id,user_id,authz_ref,permission_version,
                       policy_set_version,turn_id,execution_id,trace_id,payload)
                    VALUES (:name,:tenant,:user,:authz,:permission,:policy,
                            :turn,:execution,:trace,CAST(:payload AS jsonb))
                """), {
                    "name": event_name, "tenant": self.context.tenant_id,
                    "user": self.context.user_id, "authz": self.authz.authz_ref,
                    "permission": self.authz.permission_version,
                    "policy": self.authz.policy_set_version, "turn": turn_id,
                    "execution": execution_id, "trace": trace_id,
                    "payload": json.dumps(payload or {}, ensure_ascii=False, default=str),
                })
        except Exception:
            # During a rolling migration the in-process event still provides a
            # signal; the database table becomes authoritative after upgrade.
            return

    def save_authorization_snapshot(self) -> None:
        """Persist the reference/digest, never a second permission database."""

        with engine.begin() as conn:
            self._scope(conn)
            self._save_authorization_snapshot(conn)

    def create_turn(
        self,
        *,
        session_id: str,
        question: str,
        scope: CanonicalScope,
        client_turn_id: str | None = None,
        inherited_from_turn_id: str | None = None,
        context_snapshot: dict[str, Any] | None = None,
    ) -> TurnHandle:
        """Create the user-visible turn before invoking LangGraph.

        A repeated client id returns the existing row, making browser retries
        safe and preventing duplicate user messages.
        """

        self.authz.require_sellers(scope.seller_ids)
        scope_fingerprint = scope.query_scope_fingerprint
        turn_id = _id("turn")
        with engine.begin() as conn:
            self._scope(conn)
            self._save_authorization_snapshot(conn)
            # Serialize client-turn idempotency checks within the session.
            # The upsert takes a row lock, so concurrent retries cannot both
            # observe a missing client_turn_id and race into the unique index.
            conn.execute(text("""
                INSERT INTO governance.chat_sessions
                  (tenant_id, user_id, chat_session_id, title, scope_data)
                VALUES (:tenant, :user, :session, :title, CAST(:scope AS jsonb))
                ON CONFLICT (tenant_id, user_id, chat_session_id) DO UPDATE
                  SET updated_at=now()
            """), {
                "tenant": self.context.tenant_id, "user": self.context.user_id,
                "session": session_id, "title": question.strip()[:160] or "新会话",
                "scope": json.dumps(
                    {**scope.as_query_dict(), "_scope_fingerprint": scope_fingerprint},
                    ensure_ascii=False,
                ),
            })
            existing = None
            if client_turn_id:
                existing = conn.execute(text("""
                    SELECT turn_id, chat_session_id, client_turn_id, authz_ref,
                           permission_version, policy_set_version, authz_digest,
                           scope_fingerprint, status, task_type, answer, trace_id,
                           result_snapshot
                    FROM governance.chat_turns
                    WHERE tenant_id=:tenant AND user_id=:user
                      AND chat_session_id=:session AND client_turn_id=:client
                """), {
                    "tenant": self.context.tenant_id, "user": self.context.user_id,
                    "session": session_id, "client": client_turn_id,
                }).mappings().one_or_none()
            if existing:
                if (
                    existing["authz_digest"] != self.authz.authz_digest
                    or existing["permission_version"] != self.authz.permission_version
                    or existing["policy_set_version"] != self.authz.policy_set_version
                    or existing["scope_fingerprint"] != scope_fingerprint
                ):
                    # A browser retry is only idempotent within the same
                    # authorization/scope snapshot.  Never replay an old
                    # result after permissions or the requested scope change.
                    raise ValueError("existing client turn has stale authorization or scope")
                return TurnHandle(
                    turn_id=existing["turn_id"], session_id=existing["chat_session_id"],
                    client_turn_id=existing["client_turn_id"], authz_ref=existing["authz_ref"],
                    permission_version=existing["permission_version"],
                    policy_set_version=existing["policy_set_version"],
                    authorization_fingerprint=existing["authz_digest"],
                    scope_fingerprint=existing["scope_fingerprint"],
                    status=existing["status"] or "queued",
                    task_type=existing["task_type"] or "pending",
                    answer=existing["answer"],
                    trace_id=existing["trace_id"] or "",
                    result_snapshot=dict(existing["result_snapshot"] or {}),
                )
            if inherited_from_turn_id:
                parent = conn.execute(text("""
                    SELECT status, scope_fingerprint, authz_digest
                    FROM governance.chat_turns
                    WHERE turn_id=:parent AND tenant_id=:tenant
                      AND user_id=:user AND chat_session_id=:session
                    FOR SHARE
                """), {
                    "parent": inherited_from_turn_id,
                    "tenant": self.context.tenant_id,
                    "user": self.context.user_id,
                    "session": session_id,
                }).mappings().one_or_none()
                if parent is None:
                    raise ValueError("inherited turn does not belong to the current user and session")
                if parent["status"] != "completed":
                    raise ValueError("inherited turn is not completed")
                if (
                    parent["scope_fingerprint"] != scope_fingerprint
                    or parent["authz_digest"] != self.authz.authz_digest
                ):
                    raise ValueError("inherited turn has stale authorization or scope")
            conn.execute(text("""
                INSERT INTO governance.chat_turns
                  (turn_id, tenant_id, user_id, chat_session_id, question, answer,
                   task_type, trace_id, client_turn_id, status, interaction_state,
                   inherited_from_turn_id, context_snapshot, permission_version,
                   policy_set_version, authz_ref, authz_digest, scope_fingerprint)
                VALUES (:turn, :tenant, :user, :session, :question, NULL,
                        'pending', '', :client, 'queued', 'none', :inherited,
                        CAST(:context AS jsonb), :permission, :policy, :authz_ref,
                        :digest, :scope)
            """), {
                "turn": turn_id, "tenant": self.context.tenant_id,
                "user": self.context.user_id, "session": session_id,
                "question": question, "client": client_turn_id,
                "inherited": inherited_from_turn_id,
                "context": json.dumps(context_snapshot or {}, ensure_ascii=False),
                "permission": self.authz.permission_version,
                "policy": self.authz.policy_set_version,
                "authz_ref": self.authz.authz_ref,
                "digest": self.authz.authz_digest,
                "scope": scope_fingerprint,
            })
        handle = TurnHandle(
            turn_id=turn_id, session_id=session_id, client_turn_id=client_turn_id,
            authz_ref=self.authz.authz_ref,
            permission_version=self.authz.permission_version,
            policy_set_version=self.authz.policy_set_version,
            authorization_fingerprint=self.authz.authz_digest,
            scope_fingerprint=scope_fingerprint,
            status="queued",
        )
        self._audit("chat.turn.created", turn_id=turn_id, payload={"client_turn_id": client_turn_id})
        return handle

    def create_execution(
        self,
        turn: TurnHandle,
        *,
        trace_id: str,
        thread_id: str,
        query_scope: QueryScope,
        execution_request_id: str | None = None,
    ) -> ExecutionHandle:
        with engine.begin() as conn:
            self._scope(conn)
            turn_row = conn.execute(text("""
                SELECT turn_id, status, authz_ref, permission_version,
                       policy_set_version, authz_digest, scope_fingerprint
                FROM governance.chat_turns
                WHERE turn_id=:turn AND tenant_id=:tenant AND user_id=:user
                FOR UPDATE
            """), {
                "turn": turn.turn_id, "tenant": self.context.tenant_id,
                "user": self.context.user_id,
            }).mappings().one_or_none()
            if turn_row is None:
                raise ValueError("turn does not belong to the current user")
            if turn_row["authz_ref"] != self.authz.authz_ref or turn_row["authz_digest"] != self.authz.authz_digest:
                raise ValueError("turn authorization snapshot is stale")
            if turn_row["status"] not in {"queued", "running"}:
                raise ValueError("terminal turn cannot create another execution")
            row = conn.execute(text("""
                SELECT COALESCE(MAX(attempt_no), 0) + 1 AS next_attempt
                FROM governance.conversation_executions
                WHERE turn_id=:turn
            """), {"turn": turn.turn_id}).mappings().one()
            attempt_no = int(row["next_attempt"])
            active = conn.execute(text("""
                SELECT execution_id FROM governance.conversation_executions
                WHERE turn_id=:turn AND status IN ('queued','running','suspended')
                LIMIT 1
            """), {"turn": turn.turn_id}).first()
            if active is not None:
                raise ValueError("turn already has an active execution")
            execution_id = _id("exec")
            conn.execute(text("""
                INSERT INTO governance.conversation_executions
                  (execution_id, turn_id, tenant_id, user_id, attempt_no, status,
                   langgraph_thread_id, trace_id, authorized_permission_version,
                   authorized_policy_set_version, authorization_fingerprint,
                   query_scope_fingerprint, heartbeat_at, execution_request_id)
                VALUES (:execution, :turn, :tenant, :user, :attempt, 'queued',
                        :thread, :trace, :permission, :policy, :authz, :scope,
                        now(), :request)
            """), {
                "execution": execution_id, "turn": turn.turn_id,
                "tenant": self.context.tenant_id, "user": self.context.user_id,
                "attempt": attempt_no, "thread": thread_id, "trace": trace_id,
                "permission": self.authz.permission_version,
                "policy": self.authz.policy_set_version,
                "authz": self.authz.authz_digest,
                "scope": query_scope.query_scope_fingerprint,
                "request": execution_request_id,
            })
            conn.execute(text("""
                UPDATE governance.chat_turns
                   SET status='running', final_trace_id=:trace
                 WHERE turn_id=:turn AND status IN ('queued','running')
            """), {"turn": turn.turn_id, "trace": trace_id})
            # Focus follows the newest started execution. A late completion
            # can update its own task, but cannot reclaim this session focus.
            # Keep active_task_id: the new execution must still be able to
            # inherit the session's current task when the relation resolver
            # decides "continue". Only bind_active_task (guarded by the focus
            # owner CAS below) may move the active task forward, so a late
            # execution cannot reclaim focus by finishing later.
            conn.execute(text("""
                UPDATE governance.chat_sessions
                   SET focus_owner_execution_id=:execution,
                       session_state_version=session_state_version+1,
                       updated_at=now()
                 WHERE tenant_id=:tenant AND user_id=:user
                   AND chat_session_id=:session
            """), {
                "execution": execution_id, "tenant": self.context.tenant_id,
                "user": self.context.user_id, "session": turn.session_id,
            })
        handle = ExecutionHandle(
            execution_id=execution_id, turn_id=turn.turn_id, attempt_no=attempt_no,
            trace_id=trace_id, thread_id=thread_id,
            authorization_fingerprint=self.authz.authz_digest,
            query_scope_fingerprint=query_scope.query_scope_fingerprint,
        )
        self._audit("chat.execution.created", turn_id=turn.turn_id, execution_id=execution_id, trace_id=trace_id, payload={"attempt_no": attempt_no})
        return handle

    def create_repair_execution(
        self,
        previous: ExecutionHandle,
        *,
        trace_id: str,
        thread_id: str,
        query_scope: QueryScope,
        execution_request_id: str | None = None,
        max_repair_attempts: int = 1,
    ) -> ExecutionHandle:
        """Start one bounded system repair on the same Turn.

        The failed attempt is terminalized, but its parent Turn remains
        ``running`` while the replacement Execution is queued.  This is the
        lifecycle distinction between an internal repair and a user retry:
        repair keeps the Turn and provenance; user retry must create a new
        Turn (with ``inherited_from_turn_id``).
        """

        if max_repair_attempts < 1:
            raise ValueError("max_repair_attempts must be positive")
        with engine.begin() as conn:
            self._scope(conn)
            turn_row = conn.execute(text("""
                SELECT turn_id, status, authz_ref, authz_digest
                FROM governance.chat_turns
                WHERE turn_id=:turn AND tenant_id=:tenant AND user_id=:user
                FOR UPDATE
            """), {
                "turn": previous.turn_id, "tenant": self.context.tenant_id,
                "user": self.context.user_id,
            }).mappings().one_or_none()
            if turn_row is None:
                raise ValueError("turn does not belong to the current user")
            if turn_row["status"] != "running":
                raise ValueError("only a running turn can be system-repaired")
            if turn_row["authz_ref"] != self.authz.authz_ref or turn_row["authz_digest"] != self.authz.authz_digest:
                raise ValueError("turn authorization snapshot is stale")
            old = conn.execute(text("""
                SELECT execution_id, attempt_no, status
                FROM governance.conversation_executions
                WHERE execution_id=:execution AND turn_id=:turn
                  AND tenant_id=:tenant AND user_id=:user
                FOR UPDATE
            """), {
                "execution": previous.execution_id, "turn": previous.turn_id,
                "tenant": self.context.tenant_id, "user": self.context.user_id,
            }).mappings().one_or_none()
            if old is None:
                raise ValueError("execution does not belong to the current turn")
            if int(old["attempt_no"]) > max_repair_attempts:
                raise ValueError("maximum system repair attempts exceeded")
            if old["status"] not in {"queued", "running"}:
                raise ValueError("only an active execution can be repaired")
            conn.execute(text("""
                UPDATE governance.conversation_executions
                   SET status='failed', terminal_event='repair_requested',
                       error_code='TOOL_VALIDATION_FAILED',
                       error_detail=CAST(:detail AS jsonb), finished_at=now(), heartbeat_at=now()
                 WHERE execution_id=:execution AND status IN ('queued','running')
            """), {
                "execution": previous.execution_id,
                "detail": json.dumps({"reason": "bounded_system_repair"}, ensure_ascii=False),
            })
            next_attempt = int(old["attempt_no"]) + 1
            execution_id = _id("exec")
            conn.execute(text("""
                INSERT INTO governance.conversation_executions
                  (execution_id, turn_id, tenant_id, user_id, attempt_no, status,
                   langgraph_thread_id, trace_id, authorized_permission_version,
                   authorized_policy_set_version, authorization_fingerprint,
                   query_scope_fingerprint, heartbeat_at, execution_request_id)
                VALUES (:execution, :turn, :tenant, :user, :attempt, 'queued',
                        :thread, :trace, :permission, :policy, :authz, :scope,
                        now(), :request)
            """), {
                "execution": execution_id, "turn": previous.turn_id,
                "tenant": self.context.tenant_id, "user": self.context.user_id,
                "attempt": next_attempt, "thread": thread_id, "trace": trace_id,
                "permission": self.authz.permission_version,
                "policy": self.authz.policy_set_version,
                "authz": self.authz.authz_digest,
                "scope": query_scope.query_scope_fingerprint,
                "request": execution_request_id,
            })
            conn.execute(text("""
                UPDATE governance.chat_turns
                   SET final_trace_id=:trace, interaction_state='none'
                 WHERE turn_id=:turn AND status='running'
            """), {"turn": previous.turn_id, "trace": trace_id})
        handle = ExecutionHandle(
            execution_id=execution_id, turn_id=previous.turn_id,
            attempt_no=next_attempt, trace_id=trace_id, thread_id=thread_id,
            authorization_fingerprint=self.authz.authz_digest,
            query_scope_fingerprint=query_scope.query_scope_fingerprint,
        )
        self._audit("chat.execution.repair_created", turn_id=previous.turn_id, execution_id=execution_id, trace_id=trace_id, payload={"attempt_no": next_attempt})
        return handle

    def load_suspended_execution(self, turn: TurnHandle) -> ExecutionHandle | None:
        """Return the current suspended execution for a HITL resume.

        Resuming a human/metric confirmation is deliberately different from a
        retry: it reuses the persisted LangGraph thread and execution attempt.
        The caller must still perform the normal authorization check before
        invoking the graph.
        """

        with engine.begin() as conn:
            self._scope(conn)
            row = conn.execute(text("""
                SELECT execution_id, turn_id, attempt_no, trace_id,
                       langgraph_thread_id, authorization_fingerprint,
                       query_scope_fingerprint, status
                FROM governance.conversation_executions
                WHERE execution_id = (
                    SELECT execution_id
                    FROM governance.conversation_executions
                    WHERE turn_id=:turn AND tenant_id=:tenant AND user_id=:user
                      AND status='suspended'
                    ORDER BY attempt_no DESC
                    LIMIT 1
                )
            """), {
                "turn": turn.turn_id,
                "tenant": self.context.tenant_id,
                "user": self.context.user_id,
            }).mappings().one_or_none()
        if row is None:
            return None
        return ExecutionHandle(
            execution_id=row["execution_id"],
            turn_id=row["turn_id"],
            attempt_no=int(row["attempt_no"]),
            trace_id=row["trace_id"] or "",
            thread_id=row["langgraph_thread_id"] or "",
            authorization_fingerprint=row["authorization_fingerprint"],
            query_scope_fingerprint=row["query_scope_fingerprint"],
        )

    def load_active_execution(self, turn: TurnHandle) -> ExecutionHandle | None:
        """Find an in-flight execution for an idempotent SSE reconnect."""
        with engine.begin() as conn:
            self._scope(conn)
            row = conn.execute(text("""
                SELECT execution_id, turn_id, attempt_no, trace_id,
                       langgraph_thread_id, authorization_fingerprint,
                       query_scope_fingerprint
                FROM governance.conversation_executions
                WHERE turn_id=:turn AND tenant_id=:tenant AND user_id=:user
                  AND status IN ('queued','running')
                ORDER BY attempt_no DESC LIMIT 1
            """), {
                "turn": turn.turn_id, "tenant": self.context.tenant_id,
                "user": self.context.user_id,
            }).mappings().one_or_none()
        if row is None:
            return None
        return ExecutionHandle(
            execution_id=row["execution_id"], turn_id=row["turn_id"],
            attempt_no=int(row["attempt_no"]), trace_id=row["trace_id"] or "",
            thread_id=row["langgraph_thread_id"] or "",
            authorization_fingerprint=row["authorization_fingerprint"],
            query_scope_fingerprint=row["query_scope_fingerprint"],
        )

    def current_turn_snapshot(self, turn_id: str) -> dict[str, Any] | None:
        """Read a scoped Turn snapshot for a reconnecting stream."""
        with engine.begin() as conn:
            self._scope(conn)
            row = conn.execute(text("""
                SELECT turn_id, status, task_type, answer, trace_id, result_snapshot
                FROM governance.chat_turns
                WHERE turn_id=:turn AND tenant_id=:tenant AND user_id=:user
            """), {
                "turn": turn_id, "tenant": self.context.tenant_id,
                "user": self.context.user_id,
            }).mappings().one_or_none()
        if row is None:
            return None
        return dict(row)

    def heartbeat(self, execution_id: str) -> bool:
        with engine.begin() as conn:
            self._scope(conn)
            row = conn.execute(text("""
                UPDATE governance.conversation_executions
                   SET heartbeat_at=now()
                 WHERE execution_id=:execution AND status IN ('queued','running')
                 RETURNING execution_id
            """), {"execution": execution_id}).first()
        if row is None:
            try:
                record_measurement("context_focus_cas_conflicts")
            except Exception:  # noqa: BLE001 - telemetry must not change lifecycle behavior
                pass
        return row is not None

    def bind_active_task(self, execution: ExecutionHandle, task_id: str) -> bool:
        """Bind a task to this session only while this execution owns focus."""
        with engine.begin() as conn:
            self._scope(conn)
            row = conn.execute(text("""
                UPDATE governance.chat_sessions s
                   SET active_task_id=:task,
                       session_state_version=session_state_version+1,
                       updated_at=now()
                 WHERE s.tenant_id=:tenant AND s.user_id=:user
                   AND s.focus_owner_execution_id=:execution
                   AND EXISTS (
                       SELECT 1 FROM governance.conversation_tasks t
                       JOIN governance.chat_turns c
                         ON c.chat_session_id=t.chat_session_id
                        AND c.tenant_id=t.tenant_id AND c.user_id=t.user_id
                       WHERE c.turn_id=:turn AND t.task_id=:task
                         AND t.tenant_id=:tenant AND t.user_id=:user
                   )
                RETURNING s.chat_session_id
            """), {
                "task": task_id, "tenant": self.context.tenant_id,
                "user": self.context.user_id, "execution": execution.execution_id,
                "turn": execution.turn_id,
            }).first()
        return row is not None

    def execution_for_thread(self, thread_id: str) -> ExecutionHandle | None:
        with engine.begin() as conn:
            self._scope(conn)
            row = conn.execute(text("""
                SELECT execution_id, turn_id, attempt_no, trace_id,
                       langgraph_thread_id, authorization_fingerprint,
                       query_scope_fingerprint
                FROM governance.conversation_executions
                WHERE tenant_id=:tenant AND user_id=:user
                  AND langgraph_thread_id=:thread
                ORDER BY attempt_no DESC LIMIT 1
            """), {
                "tenant": self.context.tenant_id, "user": self.context.user_id,
                "thread": thread_id,
            }).mappings().one_or_none()
        if row is None:
            return None
        return ExecutionHandle(
            execution_id=row["execution_id"], turn_id=row["turn_id"],
            attempt_no=int(row["attempt_no"]), trace_id=row["trace_id"] or "",
            thread_id=row["langgraph_thread_id"] or "",
            authorization_fingerprint=row["authorization_fingerprint"],
            query_scope_fingerprint=row["query_scope_fingerprint"],
        )

    def commit_metadata(self, turn_id: str) -> dict[str, Any]:
        with engine.begin() as conn:
            self._scope(conn)
            row = conn.execute(text("""
                SELECT t.turn_id, t.final_execution_id, t.commit_version,
                       t.status, e.trace_id
                FROM governance.chat_turns t
                LEFT JOIN governance.conversation_executions e
                  ON e.execution_id=t.final_execution_id
                WHERE t.turn_id=:turn AND t.tenant_id=:tenant AND t.user_id=:user
            """), {
                "turn": turn_id, "tenant": self.context.tenant_id,
                "user": self.context.user_id,
            }).mappings().one_or_none()
        return dict(row) if row else {}

    def claim_review_interaction(
        self, execution: ExecutionHandle, *, node_name: str,
        interrupt_index: str, decision: dict[str, Any], force_reclaim: bool = False,
    ) -> tuple[str, str | None, dict[str, Any] | None]:
        digest = hashlib.sha256(json.dumps(decision, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        with engine.begin() as conn:
            self._scope(conn)
            conn.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"), {
                "key": f"{execution.execution_id}:{node_name}:{interrupt_index}",
            })
            existing = conn.execute(text("""
                SELECT interaction_id, decision_digest, status, result_snapshot,
                       lease_expires_at
                FROM governance.chat_review_interactions
                WHERE execution_id=:execution AND node_name=:node AND interrupt_index=:index
                FOR UPDATE
            """), {
                "execution": execution.execution_id, "node": node_name,
                "index": interrupt_index,
            }).mappings().one_or_none()
            if existing:
                if existing["decision_digest"] != digest:
                    return "conflict", None, None
                if existing["status"] == "completed":
                    return "replay", str(existing["interaction_id"]), dict(existing["result_snapshot"] or {})
                if not force_reclaim and existing["status"] == "processing" and existing["lease_expires_at"] is not None \
                        and existing["lease_expires_at"] > datetime.now(timezone.utc):
                    return "processing", str(existing["interaction_id"]), None
                interaction_id = str(existing["interaction_id"])
                claim_token = token_hex(24)
                conn.execute(text("""
                    UPDATE governance.chat_review_interactions
                       SET status='processing', claim_token=:token,
                           lease_expires_at=now()+interval '90 seconds', updated_at=now()
                     WHERE interaction_id=:id
                """), {"id": interaction_id, "token": claim_token})
                self._review_claim_tokens = getattr(self, "_review_claim_tokens", {})
                self._review_claim_tokens[interaction_id] = claim_token
                return "claimed", interaction_id, None
            active = conn.execute(text("""
                SELECT execution_id FROM governance.conversation_executions
                WHERE execution_id=:execution AND turn_id=:turn
                  AND tenant_id=:tenant AND user_id=:user AND status='suspended'
                FOR UPDATE
            """), {
                "execution": execution.execution_id, "turn": execution.turn_id,
                "tenant": self.context.tenant_id, "user": self.context.user_id,
            }).scalar_one_or_none()
            if active is None:
                return "not_suspended", None, None
            interaction_id = _id("review")
            claim_token = token_hex(24)
            conn.execute(text("""
                INSERT INTO governance.chat_review_interactions
                  (interaction_id,tenant_id,user_id,execution_id,node_name,
                   interrupt_index,decision_digest,status,claim_token,lease_expires_at)
                VALUES (:id,:tenant,:user,:execution,:node,:index,:digest,'processing',:token,
                        now()+interval '90 seconds')
            """), {
                "id": interaction_id, "tenant": self.context.tenant_id,
                "user": self.context.user_id, "execution": execution.execution_id,
                "node": node_name, "index": interrupt_index, "digest": digest,
                "token": claim_token,
            })
            self._review_claim_tokens = getattr(self, "_review_claim_tokens", {})
            self._review_claim_tokens[interaction_id] = claim_token
        return "claimed", interaction_id, None

    def renew_review_interaction(self, interaction_id: str, claim_token: str) -> bool:
        """Extend a live HITL request lease; claim_token fences stale request owners."""
        with engine.begin() as conn:
            self._scope(conn)
            return conn.execute(text("""
                UPDATE governance.chat_review_interactions
                   SET lease_expires_at=now()+interval '90 seconds', updated_at=now()
                 WHERE interaction_id=:id AND tenant_id=:tenant AND user_id=:user
                   AND status='processing' AND claim_token=:token
                RETURNING interaction_id
            """), {
                "id": interaction_id, "tenant": self.context.tenant_id,
                "user": self.context.user_id, "token": claim_token,
            }).first() is not None

    @staticmethod
    def _render_summary_rows(rows: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """Render the newest contiguous excerpt window within the frozen 4k-token budget."""
        from app.context.packer import estimate_tokens

        max_tokens = 4000
        selected: list[tuple[dict[str, Any], str, int]] = []
        used_tokens = 0
        for row in reversed(rows):
            question = " ".join(str(row.get("question") or "").split())[:320]
            answer = " ".join(str(row.get("answer") or "").split())[:640]
            question = question or "（本轮未记录问题文本）"
            rendered = f"用户：{question}；答复：{answer}" if answer else f"用户：{question}"
            cost = estimate_tokens(rendered)
            if selected and used_tokens + cost + estimate_tokens("\n") > max_tokens:
                break
            if not selected and cost > max_tokens:
                # Per-turn clipping should make this unreachable for supported
                # tokenizers; fail closed rather than claim coverage for text
                # that cannot fit the summary budget.
                break
            selected.append((row, rendered, cost))
            used_tokens += cost + (estimate_tokens("\n") if len(selected) > 1 else 0)
        selected.reverse()
        while selected and estimate_tokens("\n".join(item[1] for item in selected)) > max_tokens:
            selected.pop(0)
        covered_rows = [item[0] for item in selected]
        return "\n".join(item[1] for item in selected), covered_rows

    def finalize(
        self,
        execution: ExecutionHandle,
        *,
        status: str,
        answer: str | None = None,
        result: dict[str, Any] | None = None,
        error_code: str | None = None,
        error_detail: dict[str, Any] | None = None,
        interaction_state: str = "none",
        final_node: str | None = None,
        task_id: str | None = None,
        task_status: str | None = None,
        relation_decision: dict[str, Any] | None = None,
        review_interaction_id: str | None = None,
        review_interaction_result: dict[str, Any] | None = None,
    ) -> bool:
        """Atomically finalize execution and its parent turn exactly once."""

        if status not in {"completed", "abstained", "failed", "cancelled", "interrupted", "suspended"}:
            raise ValueError(f"invalid terminal execution status: {status}")
        terminal = status != "suspended"
        final_trace_id = str((result or {}).get("trace_id") or execution.trace_id)
        task_state_patch: dict[str, Any] = {}
        if terminal and task_id:
            try:
                from app.tasks.service import _task_state_patch_from_trace

                task_state_patch = _task_state_patch_from_trace(
                    self.context, final_trace_id, execution.query_scope_fingerprint,
                )
            except Exception:  # noqa: BLE001 - Turn commit still records a fail-closed task snapshot
                task_state_patch = {"scope_fingerprint": "", "data_version_manifest_hash": ""}
        result_task_type = str((result or {}).get("task_type") or "").strip()
        if not result_task_type or result_task_type.lower() == "pending":
            result_task_type = ""
        with engine.begin() as conn:
            self._scope(conn)
            updated = conn.execute(text("""
                UPDATE governance.conversation_executions
                   SET status=:status, interaction_state=:interaction,
                       trace_id=:trace,
                       result_snapshot=CAST(:result AS jsonb),
                       error_code=:error_code,
                       error_detail=CAST(:error_detail AS jsonb),
                       terminal_event=:terminal_event, final_node=:final_node,
                       finished_at=CASE WHEN :terminal THEN now() ELSE finished_at END,
                       heartbeat_at=now()
                 WHERE execution_id=:execution
                   AND status IN ('queued','running','suspended')
                RETURNING execution_id
            """), {
                "execution": execution.execution_id, "status": status,
                "trace": final_trace_id,
                "interaction": interaction_state,
                "result": json.dumps(result or {}, ensure_ascii=False),
                "error_code": error_code,
                "error_detail": json.dumps(error_detail or {}, ensure_ascii=False),
                "terminal_event": status if terminal else None,
                "final_node": final_node, "terminal": terminal,
            }).first()
            if updated is None:
                current = conn.execute(text("""
                    SELECT e.status, t.status AS turn_status, t.final_execution_id
                    FROM governance.conversation_executions e
                    JOIN governance.chat_turns t ON t.turn_id=e.turn_id
                    WHERE e.execution_id=:execution AND e.turn_id=:turn
                      AND e.tenant_id=:tenant AND e.user_id=:user
                """), {
                    "execution": execution.execution_id, "turn": execution.turn_id,
                    "tenant": self.context.tenant_id, "user": self.context.user_id,
                }).mappings().one_or_none()
                return bool(current and terminal and current["status"] == status and current["turn_status"] in
                            {"completed", "abstained", "failed", "cancelled", "interrupted"}
                            and current["final_execution_id"] == execution.execution_id)
            if terminal:
                session = conn.execute(text("""
                    SELECT s.chat_session_id, s.turn_commit_version,
                           s.summary_covered_to_version, s.conversation_summary,
                           s.summary_covered_from_version, s.summary_covered_from_turn_id,
                           s.summary_covered_to_turn_id, s.open_questions
                    FROM governance.chat_sessions s
                    JOIN governance.chat_turns t
                      ON t.tenant_id=s.tenant_id AND t.user_id=s.user_id
                     AND t.chat_session_id=s.chat_session_id
                    WHERE t.turn_id=:turn AND s.tenant_id=:tenant AND s.user_id=:user
                    FOR UPDATE OF s
                """), {
                    "turn": execution.turn_id, "tenant": self.context.tenant_id,
                    "user": self.context.user_id,
                }).mappings().one_or_none()
                if session is None:
                    raise RuntimeError("turn_commit_session_missing")
                commit_version = int(session["turn_commit_version"] or 0) + 1
                if review_interaction_result is not None:
                    review_interaction_result.update({
                        "durable_turn_id": execution.turn_id,
                        "execution_id": execution.execution_id,
                        "commit_version": commit_version,
                    })
                resolution = result.get("turn_resolution") if isinstance(result, dict) else None
                has_resolution = isinstance(resolution, dict)
                unresolved = resolution.get("unresolved_fields") if has_resolution else None
                open_questions = _normalise_open_questions(
                    unresolved, source_turn_id=execution.turn_id,
                ) if has_resolution else None
                conn.execute(text("""
                    UPDATE governance.chat_sessions
                       SET turn_commit_version=:version,
                           last_committed_turn_id=:turn, committed_at=now(),
                           open_questions=CASE WHEN :has_resolution
                               THEN CAST(:open_questions AS jsonb) ELSE open_questions END,
                           open_questions_updated_at=CASE WHEN :has_resolution
                               THEN now() ELSE open_questions_updated_at END,
                           session_state_version=session_state_version+1,
                           updated_at=now()
                     WHERE tenant_id=:tenant AND user_id=:user
                       AND chat_session_id=:session
                """), {
                    "version": commit_version, "turn": execution.turn_id,
                    "tenant": self.context.tenant_id, "user": self.context.user_id,
                    "session": session["chat_session_id"],
                    "has_resolution": has_resolution,
                    "open_questions": json.dumps(open_questions or [], ensure_ascii=False),
                })
                turn_update = conn.execute(text("""
                    UPDATE governance.chat_turns
                       SET status=:status, interaction_state=:interaction,
                           answer=:answer,
                           task_type=COALESCE(
                               NULLIF(:task_type,''), NULLIF(task_type,'pending'),
                               NULLIF(task_type,''), 'unknown'
                           ),
                           task_id=COALESCE(:task_id, task_id),
                           final_execution_id=:execution, final_trace_id=:trace,
                           result_snapshot=CAST(:result AS jsonb), error_code=:error_code,
                           context_snapshot=CASE WHEN :has_resolution THEN
                               COALESCE(context_snapshot, '{}'::jsonb)
                               || jsonb_build_object('turn_resolution', CAST(:turn_resolution AS jsonb))
                               ELSE COALESCE(context_snapshot, '{}'::jsonb) END,
                           commit_version=:commit_version,
                           error_detail=CAST(:error_detail AS jsonb), finalized_at=now()
                     WHERE turn_id=:turn AND status IN ('queued','running')
                    RETURNING turn_id
                """), {
                    "turn": execution.turn_id,
                    "task_id": task_id,
                    "commit_version": commit_version,
                    "status": status if status != "interrupted" else "interrupted",
                    "interaction": interaction_state,
                    "answer": answer,
                    "task_type": result_task_type or None,
                    "execution": execution.execution_id, "trace": final_trace_id,
                    "result": json.dumps(result or {}, ensure_ascii=False),
                    "error_code": error_code,
                    "error_detail": json.dumps(error_detail or {}, ensure_ascii=False),
                    "has_resolution": has_resolution,
                    "turn_resolution": json.dumps(resolution or {}, ensure_ascii=False),
                }).first()
                if turn_update is None:
                    raise RuntimeError("turn_commit_parent_not_active")
                if task_id:
                    task_row = conn.execute(text("""
                        SELECT * FROM governance.conversation_tasks
                        WHERE task_id=:task AND tenant_id=:tenant AND user_id=:user
                          AND chat_session_id=:session FOR UPDATE
                    """), {
                        "task": task_id, "tenant": self.context.tenant_id,
                        "user": self.context.user_id, "session": session["chat_session_id"],
                    }).mappings().one_or_none()
                    committed_task_status = task_status or status
                    task_is_terminal = task_row and task_row["status"] in {
                        "completed", "abstained", "failed", "cancelled", "interrupted", "superseded",
                    }
                    if task_row is not None and (not task_is_terminal or task_row["status"] == committed_task_status):
                        from app.tasks.service import _bounded_task_state

                        task_state = _bounded_task_state(
                            dict(task_row), task_state_patch, status=committed_task_status,
                            trace_id=final_trace_id, turn_id=execution.turn_id,
                            relation_decision=relation_decision,
                        )
                        conn.execute(text("""
                            UPDATE governance.conversation_tasks
                               SET status=CAST(:status AS varchar), latest_trace_id=:trace,
                                   task_state=CAST(:task_state AS jsonb),
                                   task_state_version=task_state_version+1,
                                   summary_data=COALESCE(summary_data, '{}'::jsonb)
                                       || CAST(:summary AS jsonb),
                                   updated_at=now(),
                                   completed_at=CASE WHEN CAST(:status AS varchar) IN
                                       ('completed','abstained','failed','cancelled','interrupted','superseded')
                                       THEN now() ELSE completed_at END
                             WHERE task_id=:task AND tenant_id=:tenant AND user_id=:user
                        """), {
                            "status": committed_task_status, "trace": final_trace_id,
                            "task_state": json.dumps(task_state, ensure_ascii=False),
                            "summary": json.dumps({"last_committed_turn_id": execution.turn_id}, ensure_ascii=False),
                            "task": task_id, "tenant": self.context.tenant_id,
                            "user": self.context.user_id,
                        })
                # Build a deterministic, rebuildable summary from committed
                # turns older than the latest 20. Coverage is versioned so it
                # cannot overlap the recent raw-turn window.
                summary_rows = conn.execute(text("""
                    SELECT turn_id, commit_version, question, answer
                    FROM governance.chat_turns
                    WHERE tenant_id=:tenant AND user_id=:user
                      AND chat_session_id=:session AND commit_version IS NOT NULL
                      AND commit_version <= :cutoff
                    ORDER BY commit_version
                """), {
                    "tenant": self.context.tenant_id, "user": self.context.user_id,
                    "session": session["chat_session_id"],
                    "cutoff": max(commit_version - 20, 0),
                }).mappings().all()
                if summary_rows:
                    summary_text, covered_rows = self._render_summary_rows([dict(row) for row in summary_rows])
                    if not covered_rows:
                        summary_text = ""
                    conn.execute(text("""
                        UPDATE governance.chat_sessions
                           SET conversation_summary=:summary,
                               summary_covered_from_version=:first_version,
                               summary_covered_to_version=:last_version,
                               summary_covered_from_turn_id=:first_turn,
                               summary_covered_to_turn_id=:last_turn,
                               summary_policy_version='summary-policy-v1',
                               summary_generated_at=now()
                         WHERE tenant_id=:tenant AND user_id=:user
                           AND chat_session_id=:session
                    """), {
                        "summary": summary_text,
                        "first_version": covered_rows[0]["commit_version"] if covered_rows else None,
                        "last_version": covered_rows[-1]["commit_version"] if covered_rows else None,
                        "first_turn": covered_rows[0]["turn_id"] if covered_rows else None,
                        "last_turn": covered_rows[-1]["turn_id"] if covered_rows else None,
                        "tenant": self.context.tenant_id, "user": self.context.user_id,
                        "session": session["chat_session_id"],
                    })
            else:
                # Keep the durable Turn open while input is pending, but store
                # only a small safe snapshot so the history API can restore the
                # clarification after refresh.
                from app.chat_history.safe_projection import safe_suspended_turn_snapshot

                turn_snapshot = safe_suspended_turn_snapshot(result, answer)
                conn.execute(text("""
                    UPDATE governance.chat_turns
                       SET interaction_state=:interaction,
                           answer=:answer,
                           result_snapshot=CAST(:turn_result AS jsonb)
                     WHERE turn_id=:turn AND status='running'
                """), {
                    "turn": execution.turn_id,
                    "interaction": interaction_state,
                    "answer": answer,
                    "turn_result": json.dumps(turn_snapshot, ensure_ascii=False),
                })
                if task_id and task_status:
                    conn.execute(text("""
                        UPDATE governance.conversation_tasks
                           SET status=:status, updated_at=now(), completed_at=NULL
                         WHERE task_id=:task AND tenant_id=:tenant AND user_id=:user
                           AND chat_session_id=(
                               SELECT chat_session_id FROM governance.chat_turns
                               WHERE turn_id=:turn AND tenant_id=:tenant AND user_id=:user
                           )
                           AND status NOT IN ('completed','abstained','failed','cancelled','interrupted','superseded')
                    """), {
                        "status": task_status, "task": task_id,
                        "tenant": self.context.tenant_id, "user": self.context.user_id,
                        "turn": execution.turn_id,
                    })
            if review_interaction_id:
                interaction_update = conn.execute(text("""
                    UPDATE governance.chat_review_interactions
                       SET status='completed',
                           result_snapshot=CAST(:result AS jsonb),
                           lease_expires_at=NULL, updated_at=now()
                     WHERE interaction_id=:id AND tenant_id=:tenant AND user_id=:user
                       AND execution_id=:execution AND status='processing'
                       AND claim_token=:claim_token
                    RETURNING interaction_id
                """), {
                    "id": review_interaction_id,
                    "tenant": self.context.tenant_id,
                    "user": self.context.user_id,
                    "execution": execution.execution_id,
                    "claim_token": getattr(self, "_review_claim_tokens", {}).get(review_interaction_id),
                    "result": json.dumps(review_interaction_result or {}, ensure_ascii=False, default=str),
                }).first()
                if interaction_update is None:
                    raise RuntimeError("human_review_interaction_commit_mismatch")
        self._audit(
            "chat.execution.finalized",
            turn_id=execution.turn_id,
            execution_id=execution.execution_id,
            trace_id=final_trace_id,
            payload={"status": status, "error_code": error_code},
        )
        record_event(f"chat.execution.status.{status}", {
            "tenant_id": self.context.tenant_id,
            "user_id": self.context.user_id,
            "turn_id": execution.turn_id,
            "execution_id": execution.execution_id,
            "error_code": error_code,
        })
        return True

    def reconcile_stale_executions(self, *, stale_seconds: int = 600) -> int:
        """Close abandoned workers so neither task nor UI remains ``running``."""

        if stale_seconds < 1:
            raise ValueError("stale_seconds must be positive")
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=stale_seconds)
        with engine.begin() as conn:
            self._scope(conn)
            rows = conn.execute(text("""
                UPDATE governance.conversation_executions
                   SET status='interrupted', terminal_event='interrupted',
                       error_code='EXECUTION_INTERRUPTED',
                       error_detail=CAST(:detail AS jsonb), finished_at=now()
                 WHERE tenant_id=:tenant AND user_id=:user
                   AND status IN ('queued','running')
                   AND COALESCE(heartbeat_at, created_at) < :cutoff
                RETURNING execution_id, turn_id, trace_id
            """), {
                "tenant": self.context.tenant_id,
                "user": self.context.user_id,
                "cutoff": cutoff,
                "detail": json.dumps({"reason": "heartbeat_timeout", "stale_seconds": stale_seconds}),
            }).mappings().all()
            for row in rows:
                conn.execute(text("""
                    UPDATE governance.chat_turns
                       SET status='interrupted', error_code='EXECUTION_INTERRUPTED',
                           error_detail=CAST(:detail AS jsonb),
                           final_execution_id=:execution, final_trace_id=:trace,
                           finalized_at=now()
                     WHERE turn_id=:turn AND status IN ('queued','running')
                """), {
                    "turn": row["turn_id"],
                    "execution": row["execution_id"],
                    "trace": row.get("trace_id"),
                    "detail": json.dumps({"execution_id": row["execution_id"]}),
                })
        count = len(rows)
        if count:
            self._audit("chat.execution.reconciled", payload={"count": count, "reason": "heartbeat_timeout"})
        return count

    def reconcile_stale_orphan_tasks(self, *, stale_seconds: int = 86_400) -> int:
        """Interrupt legacy running tasks that have no durable execution evidence.

        This intentionally targets only old task rows with no associated Turn
        and no persisted Trace. Modern running work is represented by a Turn
        and Execution and is handled by ``reconcile_stale_executions`` instead.
        The tenant/user predicates and transaction-local RLS context are both
        required; rows with any linked Turn or Trace are left untouched.
        """

        if stale_seconds < 1:
            raise ValueError("stale_seconds must be positive")
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=stale_seconds)
        with engine.begin() as conn:
            self._scope(conn)
            rows = conn.execute(text("""
                WITH stale AS MATERIALIZED (
                    SELECT t.task_id
                    FROM governance.conversation_tasks t
                    WHERE t.tenant_id=:tenant AND t.user_id=:user
                      AND t.status='running' AND t.updated_at < :cutoff
                      AND NOT EXISTS (
                          SELECT 1 FROM governance.chat_turns c
                          WHERE c.tenant_id=t.tenant_id AND c.user_id=t.user_id
                            AND c.task_id=t.task_id
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM governance.tenant_traces tr
                          WHERE tr.tenant_id=t.tenant_id
                            AND tr.trace_id=t.latest_trace_id
                      )
                    FOR UPDATE OF t SKIP LOCKED
                )
                UPDATE governance.conversation_tasks t
                   SET status='interrupted', completed_at=now(), updated_at=now(),
                       summary_data=COALESCE(t.summary_data, '{}'::jsonb)
                         || jsonb_build_object(
                              'lifecycle_reconciliation',
                              jsonb_build_object(
                                'reason', 'stale_legacy_task_without_turn_or_trace',
                                'stale_seconds', :stale_seconds,
                                'reconciled_at', now()
                              )
                         )
                  FROM stale
                 WHERE t.task_id=stale.task_id AND t.tenant_id=:tenant
                   AND t.user_id=:user AND t.status='running'
                RETURNING t.task_id
            """), {
                "tenant": self.context.tenant_id,
                "user": self.context.user_id,
                "cutoff": cutoff,
                "stale_seconds": stale_seconds,
            }).all()
        count = len(rows)
        if count:
            self._audit("chat.task.reconciled", payload={
                "count": count,
                "reason": "stale_legacy_task_without_turn_or_trace",
                "stale_seconds": stale_seconds,
            })
        return count

    def reconcile_integrity(self) -> dict[str, int]:
        """Check lifecycle invariants for this tenant/user.

        It reports violations rather than guessing a winner. State repair
        remains CAS-controlled by ``finalize`` and stale reconciliation.
        """
        with engine.begin() as conn:
            self._scope(conn)
            active_parent = conn.execute(text("""
                SELECT count(*) FROM governance.conversation_executions e
                JOIN governance.chat_turns t ON t.turn_id=e.turn_id
                WHERE e.tenant_id=:tenant AND e.user_id=:user
                  AND e.status IN ('queued','running','suspended')
                  AND t.status IN ('completed','abstained','failed','cancelled','interrupted')
            """), {"tenant": self.context.tenant_id, "user": self.context.user_id}).scalar_one()
            active_count = conn.execute(text("""
                SELECT count(*) FROM governance.conversation_executions
                WHERE tenant_id=:tenant AND user_id=:user
                  AND status IN ('queued','running','suspended')
            """), {"tenant": self.context.tenant_id, "user": self.context.user_id}).scalar_one()
            missing_final = conn.execute(text("""
                SELECT count(*) FROM governance.chat_turns
                WHERE tenant_id=:tenant AND user_id=:user
                  AND status IN ('completed','abstained','failed','cancelled','interrupted')
                  AND (final_execution_id IS NULL OR final_trace_id IS NULL)
            """), {"tenant": self.context.tenant_id, "user": self.context.user_id}).scalar_one()
            final_execution_mismatch = conn.execute(text("""
                SELECT count(*)
                FROM governance.chat_turns t
                LEFT JOIN governance.conversation_executions e
                  ON e.execution_id=t.final_execution_id
                 AND e.turn_id=t.turn_id
                 AND e.tenant_id=t.tenant_id
                 AND e.user_id=t.user_id
                WHERE t.tenant_id=:tenant AND t.user_id=:user
                  AND t.status IN ('completed','abstained','failed','cancelled','interrupted')
                  AND (t.final_execution_id IS NULL OR e.execution_id IS NULL)
            """), {"tenant": self.context.tenant_id, "user": self.context.user_id}).scalar_one()
            final_trace_missing = conn.execute(text("""
                SELECT count(*)
                FROM governance.chat_turns t
                LEFT JOIN governance.tenant_traces tr
                  ON tr.tenant_id=t.tenant_id AND tr.trace_id=t.final_trace_id
                WHERE t.tenant_id=:tenant AND t.user_id=:user
                  AND t.status IN ('completed','abstained','failed','cancelled','interrupted')
                  AND (t.final_trace_id IS NULL OR tr.trace_id IS NULL)
            """), {"tenant": self.context.tenant_id, "user": self.context.user_id}).scalar_one()
        result = {
            "terminal_with_active_execution": int(active_parent or 0),
            "active_executions": int(active_count or 0),
            "terminal_missing_final_refs": int(missing_final or 0),
            "terminal_final_execution_mismatch": int(final_execution_mismatch or 0),
            "terminal_final_trace_missing": int(final_trace_missing or 0),
        }
        violations = sum(value for key, value in result.items() if key != "active_executions")
        record_event("chat.lifecycle.reconciliation", {**result, "violations": violations, "tenant_id": self.context.tenant_id, "user_id": self.context.user_id})
        return result


def stale_execution_cutoff(seconds: int) -> datetime:
    if seconds < 1:
        raise ValueError("seconds must be positive")
    return datetime.now(timezone.utc) - timedelta(seconds=seconds)

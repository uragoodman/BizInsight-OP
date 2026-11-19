"""Database access for the intentionally compact chat-history model."""

from __future__ import annotations

import json
from uuid import uuid4

from sqlalchemy import text

from app.data.database import engine
from app.data.rls import set_local_context
from app.tenancy import TenantContext
from app.code_analysis.storage import get_artifact_store
from app.chat_history.safe_projection import safe_analysis_presentation, safe_clarification


_LEGACY_COMPLEX_SCOPE_ANSWER = "请先选择店铺、开始日期和结束日期，再确认复杂分析计划。"


class ChatHistoryRepository:
    def __init__(self, context: TenantContext) -> None:
        self.context = context

    def _scope(self, conn) -> None:
        # 每个事务都重新设置租户和用户上下文，供 PostgreSQL RLS 使用；这里的值
        # 来自 JWT 解析后的 TenantContext，而不是请求体里的 session/user 字段。
        set_local_context(conn, self.context)

    def create_session(self, title: str = "新会话") -> dict:
        """Create an empty durable session for workflows that need a session ID."""
        session_id = f"session-{uuid4()}"
        with engine.begin() as conn:
            self._scope(conn)
            row = conn.execute(text("""
                INSERT INTO governance.chat_sessions
                    (tenant_id, user_id, chat_session_id, title)
                VALUES (:tenant_id, :user_id, :session_id, :title)
                RETURNING chat_session_id AS id, title, scope_data AS scope,
                          created_at, updated_at
            """), {
                "tenant_id": self.context.tenant_id,
                "user_id": self.context.user_id,
                "session_id": session_id,
                "title": title.strip()[:160] or "新会话",
            }).mappings().one()
        return dict(row)

    def set_session_scope(
        self, session_id: str, scope: dict | None, *, scope_fingerprint: str | None = None,
    ) -> dict:
        """Persist the selected explorer scope for this authenticated user's session."""
        with engine.begin() as conn:
            self._scope(conn)
            row = conn.execute(text("""
                INSERT INTO governance.chat_sessions
                    (tenant_id, user_id, chat_session_id, title, scope_data)
                VALUES (:tenant_id, :user_id, :session_id, '新会话', CAST(:scope AS jsonb))
                ON CONFLICT (tenant_id, user_id, chat_session_id) DO UPDATE SET
                    scope_data = EXCLUDED.scope_data,
                    updated_at = now()
                RETURNING chat_session_id AS id, title,
                          scope_data - '_scope_fingerprint' AS scope,
                          scope_data->>'_scope_fingerprint' AS scope_fingerprint,
                          created_at, updated_at
            """), {
                "tenant_id": self.context.tenant_id,
                "user_id": self.context.user_id,
                "session_id": session_id,
                "scope": json.dumps(
                    {**scope, "_scope_fingerprint": scope_fingerprint}
                    if scope is not None and scope_fingerprint else scope,
                    ensure_ascii=False,
                ) if scope is not None else None,
            }).mappings().one()
        return dict(row)

    def save_turn(self, session_id: str, turn_id: str, question: str, answer: str | None,
                  task_type: str, trace_id: str, scope: dict | None = None,
                  task_id: str | None = None, *, status: str = "completed",
                  interaction_state: str = "none", error_code: str | None = None,
                  error_detail: dict | None = None, result_snapshot: dict | None = None,
                  context_snapshot: dict | None = None, client_turn_id: str | None = None,
                  permission_version: str | None = None, policy_set_version: str | None = None,
                  authz_ref: str | None = None, authz_digest: str | None = None,
                  scope_fingerprint: str | None = None) -> None:
        # 持久化一轮聊天：chat_sessions 保存会话索引，chat_turns 保存问答和 trace_id。
        # 它是聊天历史，不是 LangGraph checkpoint，也不是分析记忆；三者用于不同的
        # 恢复/展示/跨轮上下文场景。写入前仍必须在同一事务内执行 _scope。
        title = question.strip()[:80] or "新会话"
        with engine.begin() as conn:
            self._scope(conn)
            conn.execute(text("""
                INSERT INTO governance.chat_sessions
                    (tenant_id, user_id, chat_session_id, title, scope_data)
                VALUES (:tenant_id, :user_id, :session_id, :title, CAST(:scope AS jsonb))
                ON CONFLICT (tenant_id, user_id, chat_session_id) DO UPDATE SET
                    title = CASE WHEN governance.chat_sessions.title = '新会话'
                                 THEN EXCLUDED.title ELSE governance.chat_sessions.title END,
                    scope_data = COALESCE(EXCLUDED.scope_data, governance.chat_sessions.scope_data),
                    updated_at = now()
            """), {"tenant_id": self.context.tenant_id, "user_id": self.context.user_id,
                    "session_id": session_id, "title": title,
                    "scope": json.dumps(scope, ensure_ascii=False) if scope else None})
            conn.execute(text("""
                INSERT INTO governance.chat_turns
                    (turn_id, tenant_id, user_id, chat_session_id, question, answer,
                     task_type, trace_id, task_id, status, interaction_state,
                     error_code, error_detail, result_snapshot, context_snapshot,
                     client_turn_id, permission_version, policy_set_version,
                     authz_ref, authz_digest, scope_fingerprint)
                VALUES (:turn_id, :tenant_id, :user_id, :session_id, :question, :answer,
                        :task_type, :trace_id, :task_id, :status, :interaction_state,
                        :error_code, CAST(:error_detail AS jsonb),
                        CAST(:result_snapshot AS jsonb), CAST(:context_snapshot AS jsonb),
                        :client_turn_id, :permission_version, :policy_set_version,
                        :authz_ref, :authz_digest, :scope_fingerprint)
                ON CONFLICT (turn_id) DO UPDATE SET
                    answer=EXCLUDED.answer, task_type=EXCLUDED.task_type,
                    trace_id=EXCLUDED.trace_id, status=EXCLUDED.status,
                    interaction_state=EXCLUDED.interaction_state,
                    error_code=EXCLUDED.error_code, error_detail=EXCLUDED.error_detail,
                    result_snapshot=EXCLUDED.result_snapshot,
                    context_snapshot=EXCLUDED.context_snapshot,
                    finalized_at=CASE WHEN EXCLUDED.status IN
                        ('completed','abstained','failed','cancelled','interrupted')
                        THEN now() ELSE governance.chat_turns.finalized_at END
            """), {"turn_id": turn_id, "tenant_id": self.context.tenant_id,
                    "user_id": self.context.user_id, "session_id": session_id,
                    "question": question, "answer": answer, "task_type": task_type, "trace_id": trace_id,
                    "task_id": task_id, "status": status, "interaction_state": interaction_state,
                    "error_code": error_code,
                    "error_detail": json.dumps(error_detail or {}, ensure_ascii=False),
                    "result_snapshot": json.dumps(result_snapshot or {}, ensure_ascii=False),
                    "context_snapshot": json.dumps(context_snapshot or {}, ensure_ascii=False),
                    "client_turn_id": client_turn_id,
                    "permission_version": permission_version, "policy_set_version": policy_set_version,
                    "authz_ref": authz_ref, "authz_digest": authz_digest,
                    "scope_fingerprint": scope_fingerprint})

    def list_sessions(self, offset: int, limit: int) -> tuple[list[dict], bool]:
        limit = min(max(limit, 1), 50)
        with engine.begin() as conn:
            self._scope(conn)
            rows = conn.execute(text("""
                SELECT s.chat_session_id AS id, s.title,
                       s.scope_data - '_scope_fingerprint' AS scope,
                       s.scope_data->>'_scope_fingerprint' AS scope_fingerprint,
                       s.created_at, s.updated_at, COUNT(t.turn_id)::int AS turn_count
                FROM governance.chat_sessions s
                LEFT JOIN governance.chat_turns t ON t.chat_session_id=s.chat_session_id
                    AND t.tenant_id=s.tenant_id AND t.user_id=s.user_id
                WHERE s.tenant_id=:tenant_id AND s.user_id=:user_id
                GROUP BY s.tenant_id, s.user_id, s.chat_session_id
                ORDER BY s.updated_at DESC
                OFFSET :offset LIMIT :limit_plus_one
            """), {"tenant_id": self.context.tenant_id, "user_id": self.context.user_id,
                    "offset": max(offset, 0), "limit_plus_one": limit + 1}).mappings().all()
        items = [dict(row) for row in rows[:limit]]
        return items, len(rows) > limit

    def list_turns(self, session_id: str, offset: int, limit: int) -> tuple[list[dict], bool]:
        limit = min(max(limit, 1), 100)
        with engine.begin() as conn:
            self._scope(conn)
            rows = conn.execute(text("""
                SELECT t.turn_id AS id, t.client_turn_id, t.question AS message, t.answer, t.task_type,
                       COALESCE(NULLIF(t.final_trace_id, ''), t.trace_id) AS trace_id,
                       t.status, t.interaction_state, t.error_code, t.error_detail,
                       t.result_snapshot, t.context_snapshot, t.scope_fingerprint,
                       t.created_at,
                       e.status AS resume_execution_status,
                       e.interaction_state AS resume_execution_interaction_state,
                       e.langgraph_thread_id AS resume_thread_id,
                       e.query_scope_fingerprint AS resume_scope_fingerprint,
                       s.scope_data->>'_scope_fingerprint' AS session_scope_fingerprint
                FROM governance.chat_turns t
                JOIN governance.chat_sessions s
                  ON s.chat_session_id=t.chat_session_id
                 AND s.tenant_id=t.tenant_id AND s.user_id=t.user_id
                LEFT JOIN LATERAL (
                    SELECT status, interaction_state, langgraph_thread_id,
                           query_scope_fingerprint
                    FROM governance.conversation_executions
                    WHERE turn_id=t.turn_id AND tenant_id=t.tenant_id
                      AND user_id=t.user_id
                    ORDER BY attempt_no DESC
                    LIMIT 1
                ) e ON true
                WHERE t.tenant_id=:tenant_id AND t.user_id=:user_id AND t.chat_session_id=:session_id
                ORDER BY t.created_at DESC OFFSET :offset LIMIT :limit_plus_one
            """), {"tenant_id": self.context.tenant_id, "user_id": self.context.user_id,
                    "session_id": session_id, "offset": max(offset, 0), "limit_plus_one": limit + 1}).mappings().all()
        items = []
        for row in rows[:limit]:
            item = dict(row)
            snapshot = item.pop("result_snapshot", None)
            snapshot = snapshot if isinstance(snapshot, dict) else {}
            clarification = safe_clarification(snapshot.get("clarification"))
            if (
                clarification is None
                and item.get("task_type") == "complex_analysis"
                and item.get("answer") == _LEGACY_COMPLEX_SCOPE_ANSWER
            ):
                # Older best-effort history writes persisted this server-owned
                # prompt as a completed answer and dropped its structured
                # scope status. Restore only this exact legacy shape so the
                # history card can render the safe scope-selection action.
                clarification = safe_clarification({
                    "status": "complex_scope_required",
                    "reason_code": "scope_required",
                })
            item["clarification"] = clarification
            presentation = safe_analysis_presentation(snapshot.get("presentation"))
            if presentation is not None:
                item["presentation"] = presentation
            item.pop("error_detail", None)
            context_snapshot = item.pop("context_snapshot", None)
            context_snapshot = context_snapshot if isinstance(context_snapshot, dict) else {}
            scope = context_snapshot.get("scope")
            turn_scope_fingerprint = item.get("scope_fingerprint")
            reason_code = snapshot.get("interaction_reason_code")
            resumable = (
                item.get("status") in {"running", "suspended"}
                and item.get("interaction_state") == "awaiting_human_input"
                and reason_code == "premise_direction_mismatch"
                and item.get("resume_execution_status") == "suspended"
                and item.get("resume_execution_interaction_state") == "awaiting_human_input"
                and item.get("resume_thread_id")
                and item.get("resume_scope_fingerprint")
                and item.get("resume_scope_fingerprint") == turn_scope_fingerprint
                and item.get("session_scope_fingerprint") == turn_scope_fingerprint
                and isinstance(scope, dict)
            )
            if resumable:
                item["thread_id"] = item.pop("resume_thread_id")
                item["client_turn_id"] = item.get("client_turn_id")
                item["interaction_reason_code"] = reason_code
                item["scope_fingerprint"] = item.pop("resume_scope_fingerprint")
            else:
                item.pop("client_turn_id", None)
                item.pop("resume_thread_id", None)
                item.pop("resume_scope_fingerprint", None)
                item.pop("scope_fingerprint", None)
            item.pop("resume_execution_status", None)
            item.pop("resume_execution_interaction_state", None)
            item.pop("session_scope_fingerprint", None)
            items.append(item)
        return items, len(rows) > limit

    def latest_completed_context(
        self,
        session_id: str,
        *,
        exclude_turn_id: str | None = None,
        source_turn_id: str | None = None,
        scope_fingerprint: str | None = None,
    ) -> dict | None:
        """Return only the previous turn's structured context for follow-ups.

        The answer text is not treated as authorization.  This method returns
        a small, same-user/same-tenant context snapshot only when the previous
        turn was completed under the current query scope.  The caller still
        performs the current authorization-boundary check before execution.
        """
        with engine.begin() as conn:
            self._scope(conn)
            session_summary = conn.execute(text("""
                SELECT conversation_summary, summary_policy_version, summary_generated_at,
                       summary_covered_from_turn_id, summary_covered_to_turn_id,
                       summary_covered_from_version, summary_covered_to_version,
                       open_questions, open_questions_updated_at
                FROM governance.chat_sessions
                WHERE tenant_id=:tenant_id AND user_id=:user_id
                  AND chat_session_id=:session_id
            """), {
                "tenant_id": self.context.tenant_id,
                "user_id": self.context.user_id,
                "session_id": session_id,
            }).mappings().one_or_none()
            recent_turns = conn.execute(text("""
                SELECT turn_id, commit_version, question, answer, status
                FROM governance.chat_turns
                WHERE tenant_id=:tenant_id AND user_id=:user_id
                  AND chat_session_id=:session_id AND commit_version IS NOT NULL
                  AND commit_version > COALESCE(:summary_to, 0)
                ORDER BY commit_version DESC LIMIT 20
            """), {
                "tenant_id": self.context.tenant_id,
                "user_id": self.context.user_id,
                "session_id": session_id,
                "summary_to": session_summary["summary_covered_to_version"] if session_summary else None,
            }).mappings().all()
            row = conn.execute(text("""
                SELECT turn_id, result_snapshot, context_snapshot,
                       scope_fingerprint, authz_digest, permission_version,
                       policy_set_version
                FROM governance.chat_turns
                WHERE tenant_id=:tenant_id AND user_id=:user_id
                  AND chat_session_id=:session_id
                  AND status='completed'
                  AND (CAST(:exclude_turn_id AS varchar) IS NULL OR turn_id <> CAST(:exclude_turn_id AS varchar))
                  AND (CAST(:source_turn_id AS varchar) IS NULL OR turn_id=CAST(:source_turn_id AS varchar))
                  AND (CAST(:scope_fingerprint AS varchar) IS NULL OR scope_fingerprint=CAST(:scope_fingerprint AS varchar))
                ORDER BY created_at DESC
                LIMIT 1
            """), {
                "tenant_id": self.context.tenant_id,
                "user_id": self.context.user_id,
                "session_id": session_id,
                "exclude_turn_id": exclude_turn_id,
                "source_turn_id": source_turn_id,
                "scope_fingerprint": scope_fingerprint,
            }).mappings().one_or_none()
        if row is None:
            return ({
                "conversation_summary": session_summary["conversation_summary"],
                "summary_policy_version": session_summary["summary_policy_version"],
                "summary_covered_from_turn_id": session_summary["summary_covered_from_turn_id"],
                "summary_covered_to_turn_id": session_summary["summary_covered_to_turn_id"],
                "summary_covered_from_version": session_summary["summary_covered_from_version"],
                "summary_covered_to_version": session_summary["summary_covered_to_version"],
                "summary_generated_at": session_summary["summary_generated_at"],
                "open_questions": list(session_summary["open_questions"] or []),
                "open_questions_updated_at": session_summary["open_questions_updated_at"],
                "recent_committed_turns": [dict(item) for item in reversed(recent_turns)],
            } if (session_summary and (
                session_summary["conversation_summary"] or session_summary["open_questions"]
            )) or recent_turns else None)
        snapshot = dict(row["result_snapshot"] or {})
        context = dict(snapshot.get("turn_context") or {})
        result_context = snapshot.get("result_context")
        if isinstance(result_context, dict):
            # Result follow-ups are a separate server-built projection from
            # semantic turn slots. Keep it available to the compiler, which
            # validates version, scope, and data version before use.
            context["result_context"] = dict(result_context)
        if context:
            context.update({
                "source_turn_id": row["turn_id"],
                "scope_fingerprint": row["scope_fingerprint"],
                "authz_digest": row["authz_digest"],
                "permission_version": row["permission_version"],
                "policy_set_version": row["policy_set_version"],
                "source_context_snapshot": dict(row["context_snapshot"] or {}),
            })
        if session_summary and session_summary["conversation_summary"]:
            context.update({
                "conversation_summary": session_summary["conversation_summary"],
                "summary_policy_version": session_summary["summary_policy_version"],
                "summary_covered_from_turn_id": session_summary["summary_covered_from_turn_id"],
                "summary_covered_to_turn_id": session_summary["summary_covered_to_turn_id"],
                "summary_covered_from_version": session_summary["summary_covered_from_version"],
                "summary_covered_to_version": session_summary["summary_covered_to_version"],
            })
        if session_summary:
            context.update({
                "summary_generated_at": session_summary["summary_generated_at"],
                "open_questions": list(session_summary["open_questions"] or []),
                "open_questions_updated_at": session_summary["open_questions_updated_at"],
            })
        if recent_turns:
            context["recent_committed_turns"] = [dict(item) for item in reversed(recent_turns)]
        return context

    def delete_session(self, session_id: str) -> bool:
        with engine.begin() as conn:
            self._scope(conn)
            keys = conn.execute(text("""SELECT a.object_key FROM governance.code_analysis_artifacts a
              JOIN governance.code_analysis_jobs j ON j.job_id=a.job_id WHERE j.chat_session_id=:session_id"""), {"session_id":session_id}).scalars().all()
            removable_files = conn.execute(text("""
              SELECT DISTINCT f.file_id,f.object_key,f.purged_at
              FROM governance.code_analysis_files f
              JOIN governance.code_analysis_jobs target ON target.file_id=f.file_id
              WHERE target.chat_session_id=:session_id
                AND NOT EXISTS (
                  SELECT 1 FROM governance.code_analysis_jobs other
                  WHERE other.file_id=f.file_id
                    AND other.chat_session_id IS DISTINCT FROM :session_id
                )
            """), {"session_id": session_id}).mappings().all()
            # Storage deletion happens before the database commit. A failure
            # keeps the conversation and references intact; a retry is safe
            # because object deletion is idempotent.
            store = get_artifact_store()
            for key in [
                *keys,
                *(row["object_key"] for row in removable_files if not row["purged_at"]),
            ]:
                store.delete(key)
            conn.execute(text("DELETE FROM governance.code_analysis_jobs WHERE chat_session_id=:session_id"), {"session_id":session_id})
            for row in removable_files:
                conn.execute(text("""DELETE FROM governance.code_analysis_files
                  WHERE file_id=:file_id
                    AND NOT EXISTS (
                      SELECT 1 FROM governance.code_analysis_jobs
                      WHERE file_id=:file_id
                    )"""), {"file_id": row["file_id"]})
            # The legacy analysis-session snapshot uses the same durable
            # session ID. Remove it with the canonical chat session so a
            # deleted conversation cannot be rehydrated from the old L1 row.
            conn.execute(text("""
                DELETE FROM governance.memory_sessions
                WHERE tenant_id=:tenant_id AND user_id=:user_id
                  AND memory_session_id=:session_id
            """), {
                "tenant_id": self.context.tenant_id,
                "user_id": self.context.user_id,
                "session_id": session_id,
            })
            result = conn.execute(text("DELETE FROM governance.chat_sessions WHERE tenant_id=:tenant_id AND user_id=:user_id AND chat_session_id=:session_id"),
                                  {"tenant_id": self.context.tenant_id, "user_id": self.context.user_id, "session_id": session_id})
        return bool(result.rowcount)

    def delete_turn(self, session_id: str, turn_id: str) -> bool:
        with engine.begin() as conn:
            self._scope(conn)
            # A confirmed complex analysis belongs to this exact question.
            # Remove its result snapshot and any dependent action draft before
            # deleting the visible chat turn, so partial-history deletion also
            # releases the associated storage.
            conn.execute(text("""DELETE FROM governance.complex_analysis_plans
                WHERE chat_session_id=:session_id AND chat_turn_id=:turn_id"""),
                {"session_id": session_id, "turn_id": turn_id})
            keys = conn.execute(text("""SELECT a.object_key FROM governance.code_analysis_artifacts a
              JOIN governance.code_analysis_jobs j ON j.job_id=a.job_id WHERE j.chat_session_id=:session_id
              AND j.source_turn_id=:turn_id"""), {"session_id":session_id,"turn_id":turn_id}).scalars().all()
            removable_files = conn.execute(text("""
              SELECT DISTINCT f.file_id,f.object_key,f.purged_at
              FROM governance.code_analysis_files f
              JOIN governance.code_analysis_jobs target ON target.file_id=f.file_id
              WHERE target.chat_session_id=:session_id
                AND target.source_turn_id=:turn_id
                AND NOT EXISTS (
                  SELECT 1 FROM governance.code_analysis_jobs other
                  WHERE other.file_id=f.file_id
                    AND (
                      other.chat_session_id IS DISTINCT FROM :session_id
                      OR other.source_turn_id IS DISTINCT FROM :turn_id
                    )
                )
            """), {"session_id": session_id, "turn_id": turn_id}).mappings().all()
            store = get_artifact_store()
            for key in [
                *keys,
                *(row["object_key"] for row in removable_files if not row["purged_at"]),
            ]:
                store.delete(key)
            conn.execute(text("""DELETE FROM governance.code_analysis_jobs WHERE chat_session_id=:session_id
              AND source_turn_id=:turn_id"""), {"session_id":session_id,"turn_id":turn_id})
            for row in removable_files:
                conn.execute(text("""DELETE FROM governance.code_analysis_files
                  WHERE file_id=:file_id
                    AND NOT EXISTS (
                      SELECT 1 FROM governance.code_analysis_jobs
                      WHERE file_id=:file_id
                    )"""), {"file_id": row["file_id"]})
            result = conn.execute(text("""DELETE FROM governance.chat_turns
                WHERE chat_session_id=:session_id AND turn_id=:turn_id"""),
                {"session_id": session_id, "turn_id": turn_id})
        return bool(result.rowcount)

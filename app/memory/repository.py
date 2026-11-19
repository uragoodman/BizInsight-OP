"""PostgreSQL 记忆存取层。所有查询都绑定可信 TenantContext，不接受浏览器传入的租户或用户标识。"""

from __future__ import annotations

import json
from datetime import date
from uuid import uuid4

from sqlalchemy import text

from app.data.database import engine
from app.memory.contracts import AnalysisMemory, SessionMemory
from app.tenancy import TenantContext


class MemoryRepository:
    def __init__(self, context: TenantContext) -> None:
        self.context = context

    def _scope(self, conn) -> None:
        # RLS 需要每个事务都写入可信上下文；user_id 不从 HTTP payload 读取。
        # 这是所有记忆读写的安全边界；缺少 scope 时，即使 SQL 条件正确，也不能
        # 认为租户隔离已经成立。
        conn.execute(text("SELECT set_config('app.tenant_id', :tenant_id, true)"), {"tenant_id": self.context.tenant_id})
        conn.execute(text("SELECT set_config('app.user_id', :user_id, true)"), {"user_id": self.context.user_id})
        conn.execute(text("SELECT set_config('app.scope_type', :scope_type, true)"), {"scope_type": self.context.scope_type})

    def _record_decision(
        self, conn, *, action: str, actor: str, memory_id: str | None = None,
        reason: str = "", rule_scores: dict | None = None, source_memory_ids: list[str] | None = None,
    ) -> str:
        decision_id = f"mdec_{uuid4().hex}"
        conn.execute(text("""INSERT INTO governance.memory_decisions
            (decision_id,tenant_id,user_id,memory_id,action,actor,reason,rule_scores,source_memory_ids)
            VALUES (:id,:tenant,:user,:memory_id,:action,:actor,:reason,
                    CAST(:rule_scores AS jsonb),CAST(:source_ids AS jsonb))"""), {
                "id": decision_id, "tenant": self.context.tenant_id, "user": self.context.user_id,
                "memory_id": memory_id, "action": action, "actor": actor, "reason": reason,
                "rule_scores": json.dumps(rule_scores or {}), "source_ids": json.dumps(source_memory_ids or []),
            })
        return decision_id

    def record_memory_decision(
        self, *, action: str, actor: str, memory_id: str | None = None, reason: str = "",
        rule_scores: dict | None = None, source_memory_ids: list[str] | None = None,
    ) -> str:
        with engine.begin() as conn:
            self._scope(conn)
            return self._record_decision(
                conn, action=action, actor=actor, memory_id=memory_id, reason=reason,
                rule_scores=rule_scores, source_memory_ids=source_memory_ids,
            )

    def upsert_source_snapshot(self, snapshot: dict) -> bool:
        with engine.begin() as conn:
            self._scope(conn)
            inserted = conn.execute(text("""INSERT INTO governance.memory_source_snapshots
                (snapshot_id,tenant_id,user_id,source_data_version,scope_fingerprint,source_trace_id,
                 metric_definition_version,data_version_manifest_id,snapshot_payload,payload_bytes)
                VALUES (:id,:tenant,:user,:version,:scope,:trace,:metric_version,:manifest_id,
                        CAST(:payload AS jsonb),:bytes)
                ON CONFLICT (tenant_id,user_id,source_data_version) DO NOTHING RETURNING snapshot_id"""), {
                    "id": snapshot["snapshot_id"], "tenant": self.context.tenant_id, "user": self.context.user_id,
                    "version": snapshot["source_data_version"], "scope": snapshot["scope_fingerprint"],
                    "trace": snapshot["source_trace_id"], "metric_version": snapshot["metric_definition_version"], "manifest_id": snapshot.get("data_version_manifest_id"),
                    "payload": json.dumps(snapshot["snapshot_payload"], ensure_ascii=False, default=str),
                    "bytes": snapshot["payload_bytes"],
                }).scalar_one_or_none() is not None
        return inserted

    def get_source_snapshot(self, source_data_version: str) -> dict | None:
        with engine.begin() as conn:
            self._scope(conn)
            row = conn.execute(text("""SELECT * FROM governance.memory_source_snapshots
                WHERE source_data_version=:version"""), {"version": source_data_version}).mappings().first()
        return dict(row) if row else None

    def stale_memories_for_data_version(
        self, scope_fingerprint: str, task_type: str, source_data_version: str, actor: str, reason: str,
    ) -> list[str]:
        """Mark only versioned memories in the same deterministic group as stale."""
        with engine.begin() as conn:
            self._scope(conn)
            rows = conn.execute(text("""UPDATE governance.analysis_memories
                SET status='stale', reviewed_by=:actor, reviewed_at=now(), review_reason=:reason
                WHERE scope_fingerprint=:scope AND task_type=:task AND status IN ('active','pending_review')
                  AND source_data_version IS NOT NULL AND source_data_version <> :version
                RETURNING memory_id"""), {
                    "scope": scope_fingerprint, "task": task_type, "version": source_data_version,
                    "actor": actor, "reason": reason,
                }).scalars().all()
            for memory_id in rows:
                self._record_decision(
                    conn, action="staled", actor=actor, memory_id=str(memory_id), reason=reason,
                    rule_scores={"source_data_version": source_data_version},
                )
        return [str(memory_id) for memory_id in rows]

    def get_session(self, memory_session_id: str) -> dict | None:
        with engine.begin() as conn:
            self._scope(conn)
            row = conn.execute(text("""SELECT * FROM governance.memory_sessions
                WHERE memory_session_id=:session_id AND expires_at > now()"""), {"session_id": memory_session_id}).mappings().first()
        return dict(row) if row else None

    def upsert_session(self, memory: SessionMemory) -> None:
        # P0 会话记忆写入：按 tenant_id + user_id + memory_session_id 幂等更新，保存
        # 摘要、未决问题、最近受控范围和 source trace_id，供下一轮 load_session_memory 使用。
        with engine.begin() as conn:
            self._scope(conn)
            conn.execute(text("""INSERT INTO governance.memory_sessions
                (tenant_id,user_id,memory_session_id,session_summary,open_questions,
                 last_scope_fingerprint,last_seller_id,last_start_date,last_end_date,
                 last_trace_id,snapshot_data,version,updated_at,expires_at)
                VALUES (:tenant_id,:user_id,:session_id,:summary,CAST(:questions AS jsonb),
                 :fingerprint,:seller_id,:start_date,:end_date,
                 :trace_id,CAST(:snapshot AS jsonb),:version,now(),now() + interval '30 days')
                ON CONFLICT (tenant_id,user_id,memory_session_id) DO UPDATE SET
                  session_summary=EXCLUDED.session_summary,
                  open_questions=EXCLUDED.open_questions,
                  last_scope_fingerprint=EXCLUDED.last_scope_fingerprint,
                  last_seller_id=EXCLUDED.last_seller_id,
                  last_start_date=EXCLUDED.last_start_date,
                  last_end_date=EXCLUDED.last_end_date,
                  last_trace_id=EXCLUDED.last_trace_id,
                  snapshot_data=EXCLUDED.snapshot_data,
                  version=EXCLUDED.version,
                  updated_at=now(), expires_at=now() + interval '30 days'
            """), {
                "tenant_id": self.context.tenant_id, "user_id": self.context.user_id,
                "session_id": memory.memory_session_id, "summary": memory.session_summary,
                "questions": json.dumps(memory.open_questions, ensure_ascii=False),
                "fingerprint": memory.last_scope_fingerprint, "seller_id": memory.last_seller_id,
                "start_date": memory.last_start_date, "end_date": memory.last_end_date,
                "trace_id": memory.last_trace_id,
                "snapshot": json.dumps(memory.snapshot_data, ensure_ascii=False, default=str)
                    if memory.snapshot_data else "null",
                "version": memory.version,
            })

    def insert_analysis_memory(
        self,
        memory: AnalysisMemory,
        embedding: list[float] | None = None,
        embedding_model: str | None = None,
        embedding_version: str | None = None,
        consolidation_job_id: str | None = None,
        consolidation_lease_token: str | None = None,
    ) -> bool:
        # P1/P2 分析记忆写入：保存带 evidence、trace、数据版本和审核状态的长期结论。
        # 该方法还负责检查 consolidation lease，避免并发归并任务重复提交或越权写入。
        with engine.begin() as conn:
            self._scope(conn)
            if consolidation_job_id and consolidation_lease_token:
                live = conn.execute(text("""SELECT job_id FROM governance.memory_consolidation_jobs
                    WHERE job_id=:job_id AND lease_token=:lease_token AND status='generating'
                      AND lease_expires_at > clock_timestamp() FOR UPDATE"""), {
                        "job_id": consolidation_job_id, "lease_token": consolidation_lease_token,
                    }).scalar_one_or_none()
                if not live:
                    raise ValueError("Memory consolidation lease was lost")
            result = conn.execute(text("""INSERT INTO governance.analysis_memories
                (memory_id,tenant_id,user_id,seller_id,start_date,end_date,scope_fingerprint,metric_tags,task_type,insight_text,limitations,evidence_ids,source_trace_id,status,
                 embedding,embedding_model,embedding_version,valid_from,valid_to,superseded_by,source_data_version,metric_definition_version,reviewed_by,reviewed_at,normalized_claim_hash,review_reason)
                VALUES (:memory_id,:tenant_id,:user_id,:seller_id,:start_date,:end_date,:fingerprint,CAST(:metrics AS jsonb),:task_type,:insight,:limitations,CAST(:evidence_ids AS jsonb),:trace_id,:status,
                 :embedding,:embedding_model,:embedding_version,:valid_from,:valid_to,:superseded_by,:source_data_version,:metric_definition_version,:reviewed_by,:reviewed_at,:normalized_claim_hash,:review_reason)
                ON CONFLICT DO NOTHING RETURNING memory_id"""), {
                "memory_id": memory.memory_id, "tenant_id": self.context.tenant_id, "user_id": self.context.user_id,
                "seller_id": memory.seller_id, "start_date": memory.start_date, "end_date": memory.end_date,
                "fingerprint": memory.scope_fingerprint, "metrics": json.dumps(memory.metric_tags),
                "task_type": memory.task_type, "insight": memory.insight_text, "limitations": memory.limitations,
                "evidence_ids": json.dumps(memory.evidence_ids), "trace_id": memory.source_trace_id,
                "embedding": embedding, "embedding_model": embedding_model, "embedding_version": embedding_version,
                "valid_from": memory.valid_from, "valid_to": memory.valid_to, "superseded_by": memory.superseded_by,
                "source_data_version": memory.source_data_version,
                "metric_definition_version": memory.metric_definition_version,
                "reviewed_by": memory.reviewed_by, "reviewed_at": memory.reviewed_at,
                "status": memory.status, "normalized_claim_hash": memory.normalized_claim_hash,
                "review_reason": memory.review_reason,
            })
            inserted = result.scalar_one_or_none() is not None
        return inserted

    def find_memory_by_hash(
        self, seller_id: str, scope_fingerprint: str, claim_hash: str | None, source_data_version: str | None = None,
    ) -> dict | None:
        if not claim_hash:
            return None
        with engine.begin() as conn:
            self._scope(conn)
            row = conn.execute(text("""SELECT * FROM governance.analysis_memories
                WHERE seller_id=:seller_id AND scope_fingerprint=:fingerprint
                  AND normalized_claim_hash=:claim_hash
                  AND source_data_version IS NOT DISTINCT FROM :source_data_version
                  AND status IN ('active','pending_review')
                ORDER BY created_at DESC LIMIT 1"""), {
                "seller_id": seller_id, "fingerprint": scope_fingerprint, "claim_hash": claim_hash,
                "source_data_version": source_data_version,
            }).mappings().first()
        return dict(row) if row else None

    def list_analysis_memories(
        self,
        seller_id: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        task_type: str | None = None,
        metric_tags: list[str] | None = None,
        limit: int = 50,
        query_embedding: list[float] | None = None,
    ) -> list[dict]:
        """语义召回：提供 query_embedding 时对候选计算点积相似度（归一化向量 → 余弦）。

        status 只取 'active'（superseded/stale/revoked 默认排除，满足生命周期语义）。
        未提供向量时退化为现有确定性过滤 + created_at 排序。
        """
        clauses = ["status='active'"]
        params: dict[str, object] = {"limit": min(max(limit, 1), 50)}
        if seller_id:
            clauses.append("seller_id=:seller_id"); params["seller_id"] = seller_id
        if start_date and end_date:
            clauses.append("start_date <= :end_date AND end_date >= :start_date")
            params.update(start_date=start_date, end_date=end_date)
        if task_type:
            clauses.append("task_type=:task_type"); params["task_type"] = task_type
        if metric_tags:
            clauses.append("metric_tags ?| CAST(:metric_tags AS text[])"); params["metric_tags"] = metric_tags
        select_extra = ""
        if query_embedding is not None:
            params["query_embedding"] = query_embedding
            select_extra = ", (SELECT sum(x*y) FROM unnest(m.embedding, :query_embedding) AS t(x,y)) AS semantic_score"
        with engine.begin() as conn:
            self._scope(conn)
            rows = conn.execute(
                text(f"SELECT m.*{select_extra} FROM governance.analysis_memories m WHERE {' AND '.join(clauses)} ORDER BY created_at DESC LIMIT :limit"),
                params,
            ).mappings().all()
        return [dict(row) for row in rows]

    def list_memories_by_status(self, status: str, limit: int = 50) -> list[dict]:
        with engine.begin() as conn:
            self._scope(conn)
            rows = conn.execute(text("""SELECT * FROM governance.analysis_memories
                WHERE status=:status ORDER BY created_at DESC LIMIT :limit"""),
                {"status": status, "limit": min(max(limit, 1), 50)}).mappings().all()
        return [dict(row) for row in rows]

    def list_supersede_candidates(
        self,
        scope_fingerprint: str,
        metric_tags: list[str] | None = None,
        task_type: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """P3.2：同一 Scope 下可被新结论取代的 active 候选（供服务端决定 supersede）。

        只允许在同一 tenant/user（RLS）+ 相同 scope_fingerprint + 同指标候选集内取代。
        """
        clauses = ["status='active'", "scope_fingerprint=:fingerprint"]
        params: dict[str, object] = {"fingerprint": scope_fingerprint, "limit": min(max(limit, 1), 20)}
        if metric_tags:
            clauses.append("metric_tags ?| CAST(:metric_tags AS text[])"); params["metric_tags"] = metric_tags
        if task_type:
            clauses.append("task_type=:task_type"); params["task_type"] = task_type
        with engine.begin() as conn:
            self._scope(conn)
            rows = conn.execute(
                text(f"SELECT * FROM governance.analysis_memories WHERE {' AND '.join(clauses)} ORDER BY created_at DESC LIMIT :limit"),
                params,
            ).mappings().all()
        return [dict(row) for row in rows]

    def list_active_memories_for_scope(self, scope_fingerprint: str, limit: int = 5) -> list[dict]:
        with engine.begin() as conn:
            self._scope(conn)
            rows = conn.execute(text("""SELECT * FROM governance.analysis_memories
                WHERE scope_fingerprint=:fingerprint AND status='active'
                ORDER BY created_at DESC LIMIT :limit"""), {
                    "fingerprint": scope_fingerprint, "limit": min(max(limit, 2), 20),
                }).mappings().all()
        return [dict(row) for row in rows]

    def list_memory_decisions(self, memory_id: str, limit: int = 50) -> list[dict]:
        with engine.begin() as conn:
            self._scope(conn)
            rows = conn.execute(text("""SELECT * FROM governance.memory_decisions
                WHERE memory_id=:memory_id ORDER BY created_at DESC LIMIT :limit"""), {
                    "memory_id": memory_id, "limit": min(max(limit, 1), 100),
                }).mappings().all()
        return [dict(row) for row in rows]

    def create_consolidation_job(
        self, job_id: str, actor: str, scope_fingerprint: str, input_payload: dict, input_hash: str,
    ) -> dict:
        with engine.begin() as conn:
            self._scope(conn)
            row = conn.execute(text("""INSERT INTO governance.memory_consolidation_jobs
                (job_id,tenant_id,user_id,actor,scope_fingerprint,input_payload,input_hash)
                VALUES (:id,:tenant,:user,:actor,:scope,CAST(:payload AS jsonb),:hash)
                ON CONFLICT DO NOTHING RETURNING *"""), {
                    "id": job_id, "tenant": self.context.tenant_id, "user": self.context.user_id,
                    "actor": actor, "scope": scope_fingerprint,
                    "payload": json.dumps(input_payload, ensure_ascii=False, default=str), "hash": input_hash,
                }).mappings().first()
            if not row:
                row = conn.execute(text("""SELECT * FROM governance.memory_consolidation_jobs
                    WHERE scope_fingerprint=:scope AND input_hash=:hash AND status IN ('queued','generating')
                    ORDER BY created_at DESC LIMIT 1"""), {"scope": scope_fingerprint, "hash": input_hash}).mappings().one()
            item = dict(row)
            self._record_decision(
                conn, action="proposal_queued", actor=actor, reason="durable LLM consolidation proposal queued",
                source_memory_ids=[str(item) for item in input_payload.get("source_memory_ids", [])],
            )
        return item

    def get_consolidation_job(self, job_id: str) -> dict | None:
        with engine.begin() as conn:
            self._scope(conn)
            row = conn.execute(text("SELECT * FROM governance.memory_consolidation_jobs WHERE job_id=:id"), {"id": job_id}).mappings().first()
        return dict(row) if row else None

    def list_consolidation_jobs(self, limit: int = 20) -> list[dict]:
        with engine.begin() as conn:
            self._scope(conn)
            rows = conn.execute(text("""SELECT * FROM governance.memory_consolidation_jobs
                ORDER BY created_at DESC LIMIT :limit"""), {"limit": min(max(limit, 1), 100)}).mappings().all()
        return [dict(row) for row in rows]

    def supersede_analysis_memory(self, old_memory_id: str, new_memory_id: str, actor: str, reason: str = "") -> bool:
        """P3.2：把旧结论标为 superseded，并记录取代它的新记忆（审计可回放谱系）。"""
        with engine.begin() as conn:
            self._scope(conn)
            result = conn.execute(
                text("""UPDATE governance.analysis_memories
                    SET status='superseded', superseded_by=:new_id, reviewed_by=:actor, reviewed_at=now()
                    WHERE memory_id=:old_id AND status='active'"""),
                {"new_id": new_memory_id, "old_id": old_memory_id, "actor": actor},
            )
        return bool(result.rowcount)

    def get_analysis_memory(self, memory_id: str) -> dict | None:
        with engine.begin() as conn:
            self._scope(conn)
            row = conn.execute(text("SELECT * FROM governance.analysis_memories WHERE memory_id=:memory_id"), {"memory_id": memory_id}).mappings().first()
        return dict(row) if row else None

    def review_pending_memory(
        self, memory_id: str, action: str, actor: str, reason: str = "", old_memory_id: str | None = None,
    ) -> dict:
        """Atomically approve, reject, or supersede a pending memory under RLS."""
        if action not in {"supersede", "approve_independent", "reject"}:
            raise ValueError("Unsupported memory review action")
        with engine.begin() as conn:
            self._scope(conn)
            pending = conn.execute(text("SELECT * FROM governance.analysis_memories WHERE memory_id=:id FOR UPDATE"), {"id": memory_id}).mappings().first()
            if not pending:
                raise KeyError("Memory not found")
            if pending["status"] != "pending_review":
                raise ValueError("Memory is not pending review")
            if action == "reject":
                result = conn.execute(text("""UPDATE governance.analysis_memories
                    SET status='rejected', reviewed_by=:actor, reviewed_at=now(), review_reason=:reason
                    WHERE memory_id=:id AND status='pending_review' RETURNING *"""),
                    {"id": memory_id, "actor": actor, "reason": reason}).mappings().one()
                item = dict(result)
                self._record_decision(conn, action="rejected", actor=actor, memory_id=memory_id, reason=reason)
                return item
            if action == "approve_independent":
                result = conn.execute(text("""UPDATE governance.analysis_memories
                    SET status='active', reviewed_by=:actor, reviewed_at=now(), review_reason=:reason
                    WHERE memory_id=:id AND status='pending_review' RETURNING *"""),
                    {"id": memory_id, "actor": actor, "reason": reason}).mappings().one()
                item = dict(result)
                self._record_decision(conn, action="approved_independent", actor=actor, memory_id=memory_id, reason=reason)
                return item
            if not old_memory_id:
                raise ValueError("supersede requires old_memory_id")
            old = conn.execute(text("SELECT * FROM governance.analysis_memories WHERE memory_id=:id FOR UPDATE"), {"id": old_memory_id}).mappings().first()
            if not old or old["status"] != "active":
                raise ValueError("Superseded memory must be active")
            if old["seller_id"] != pending["seller_id"] or old["scope_fingerprint"] != pending["scope_fingerprint"]:
                raise ValueError("Superseded memory must share seller and scope")
            if old["task_type"] != pending["task_type"] or not (set(old["metric_tags"]) & set(pending["metric_tags"])):
                raise ValueError("Superseded memory must share task type and a metric")
            activated = conn.execute(text("""UPDATE governance.analysis_memories
                SET status='active', reviewed_by=:actor, reviewed_at=now(), review_reason=:reason
                WHERE memory_id=:id AND status='pending_review' RETURNING *"""),
                {"id": memory_id, "actor": actor, "reason": reason}).mappings().one()
            changed = conn.execute(text("""UPDATE governance.analysis_memories
                SET status='superseded', superseded_by=:new_id, reviewed_by=:actor, reviewed_at=now(), review_reason=:reason
                WHERE memory_id=:old_id AND status='active'"""),
                {"old_id": old_memory_id, "new_id": memory_id, "actor": actor, "reason": reason})
            if changed.rowcount != 1:
                raise ValueError("Active memory changed during review")
            item = dict(activated)
            self._record_decision(
                conn, action="superseded", actor=actor, memory_id=memory_id, reason=reason,
                source_memory_ids=[old_memory_id],
            )
            return item

    def revoke_analysis_memory(self, memory_id: str) -> bool:
        with engine.begin() as conn:
            self._scope(conn)
            result = conn.execute(text("UPDATE governance.analysis_memories SET status='revoked', revoked_at=now() WHERE memory_id=:memory_id AND status='active'"), {"memory_id": memory_id})
        return bool(result.rowcount)

    def clear_session(self, memory_session_id: str) -> bool:
        with engine.begin() as conn:
            self._scope(conn)
            result = conn.execute(text("UPDATE governance.memory_sessions SET expires_at=now() WHERE memory_session_id=:session_id AND expires_at > now()"), {"session_id": memory_session_id})
        return bool(result.rowcount)

"""PostgreSQL tenant-scoped report and outbox persistence (RLS protected)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Literal

from sqlalchemy import text

from app.analysis.contracts import InsightArtifact, PeriodicReport, ReportApproval
from app.data.database import engine
from app.tenancy.context import TenantContext


class TenantReportStore:
    def __init__(self, context: TenantContext) -> None:
        self.context = context

    def _scope(self, conn) -> None:
        conn.execute(text("SELECT set_config('app.tenant_id', :tenant_id, true)"), {"tenant_id": self.context.tenant_id})
        conn.execute(text("SELECT set_config('app.scope_type', :scope, true)"), {"scope": self.context.scope_type})

    def _event(self, conn, event_type: str, payload: dict, key: str) -> None:
        event_id = hashlib.sha256(f"{self.context.tenant_id}:{key}".encode()).hexdigest()[:32]
        conn.execute(text("""INSERT INTO governance.tenant_outbox(tenant_id,event_id,idempotency_key,event_type,payload_json)
            VALUES (:tenant_id,:event_id,:key,:event_type,CAST(:payload AS jsonb)) ON CONFLICT (tenant_id,idempotency_key) DO NOTHING"""), {"tenant_id": self.context.tenant_id, "event_id": event_id, "key": key, "event_type": event_type, "payload": json.dumps(payload)})

    def create_draft(self, report: PeriodicReport, content: str, actor: str, data_version_manifest_id: str | None = None, data_version_manifest_hash: str | None = None) -> dict:
        digest = hashlib.sha256(content.encode()).hexdigest()
        with engine.begin() as conn:
            self._scope(conn)
            existing = conn.execute(text("SELECT * FROM governance.tenant_reports WHERE report_id=:id AND content_hash=:hash"), {"id": report.report_id, "hash": digest}).mappings().first()
            if existing:
                return dict(existing)
            version = conn.execute(text("SELECT COALESCE(MAX(version),0)+1 FROM governance.tenant_reports WHERE report_id=:id"), {"id": report.report_id}).scalar_one()
            conn.execute(text("""INSERT INTO governance.tenant_reports(tenant_id,report_id,version,status,content,content_hash,report_json,actor,data_version_manifest_id,data_version_manifest_hash)
                VALUES (:tenant_id,:id,:version,'draft',:content,:hash,CAST(:payload AS jsonb),:actor,:manifest_id,:manifest_hash)"""), {"tenant_id": self.context.tenant_id, "id": report.report_id, "version": version, "content": content, "hash": digest, "payload": report.model_dump_json(), "actor": actor, "manifest_id": data_version_manifest_id, "manifest_hash": data_version_manifest_hash})
            self._event(conn, "ReportDrafted", {"report_id": report.report_id, "version": version}, f"report-drafted:{report.report_id}:{version}")
        return self.get(report.report_id, int(version))

    def get(self, report_id: str, version: int | None = None) -> dict:
        with engine.begin() as conn:
            self._scope(conn)
            sql = "SELECT * FROM governance.tenant_reports WHERE report_id=:id" + (" AND version=:version" if version else "") + " ORDER BY version DESC LIMIT 1"
            row = conn.execute(text(sql), {"id": report_id, "version": version}).mappings().first()
        if not row:
            raise KeyError("Report not found")
        return dict(row)

    def list(self, limit: int = 50) -> list[dict]:
        with engine.begin() as conn:
            self._scope(conn)
            rows = conn.execute(text("""SELECT r.* FROM governance.tenant_reports r
                INNER JOIN (SELECT report_id, MAX(version) AS version
                            FROM governance.tenant_reports GROUP BY report_id) latest
                  ON latest.report_id=r.report_id AND latest.version=r.version
                ORDER BY r.created_at DESC LIMIT :limit"""), {"limit": limit}).mappings()
            return [dict(row) for row in rows]

    def review(self, report_id: str, version: int, action: Literal["approve", "edit", "reject"], actor: str, reason: str = "", edited_content: str | None = None) -> ReportApproval:
        if action not in {"approve", "edit", "reject"}:
            raise ValueError("invalid review action")
        with engine.begin() as conn:
            self._scope(conn)
            row = conn.execute(text("SELECT * FROM governance.tenant_reports WHERE report_id=:id AND version=:version FOR UPDATE"), {"id": report_id, "version": version}).mappings().first()
            if not row:
                raise KeyError("Report not found")
            if row["status"] != "draft":
                raise ValueError("Only draft reports can be reviewed")
            content = edited_content if action == "edit" else row["content"]
            if action == "edit" and content is None:
                raise ValueError("edited_content is required")
            status = "rejected" if action == "reject" else "approved"
            conn.execute(text("UPDATE governance.tenant_reports SET status=:status,content=:content,actor=:actor,reason=:reason WHERE report_id=:id AND version=:version"), {"status": status, "content": content, "actor": actor, "reason": reason, "id": report_id, "version": version})
            if action != "reject": self._event(conn, "ReportApproved", {"report_id": report_id, "version": version}, f"report-approved:{report_id}:{version}")
        return ReportApproval(report_id=report_id, report_version=version, action=action, actor=actor, reason=reason, original_hash=row["content_hash"], reviewed_hash=hashlib.sha256(str(content).encode()).hexdigest())

    def publish(self, report_id: str, version: int, actor: str) -> dict:
        with engine.begin() as conn:
            self._scope(conn)
            row = conn.execute(text("SELECT * FROM governance.tenant_reports WHERE report_id=:id AND version=:version FOR UPDATE"), {"id": report_id, "version": version}).mappings().first()
            if not row: raise KeyError("Report not found")
            if row["status"] != "approved": raise ValueError("Only approved reports can be published")
            # 发布时钉死洞察版本：取最新一条人工审核通过（status=approved）的洞察。
            # 发布物从此恒定；已发布报告不再展示后来生成的洞察。无合格洞察时钉 null。
            pinned = conn.execute(text(
                """SELECT insight_version FROM governance.tenant_report_insights
                   WHERE report_id=:id AND report_version=:rv AND status='approved'
                   ORDER BY insight_version DESC LIMIT 1"""
            ), {"id": report_id, "rv": version}).scalar_one_or_none()
            conn.execute(text("UPDATE governance.tenant_reports SET status='published',actor=:actor,pinned_insight_version=:pin WHERE report_id=:id AND version=:version"), {"actor": actor, "pin": pinned, "id": report_id, "version": version})
            # 同事务把被钉选的洞察标记为 published，状态机闭环：draft → approved → published。
            if pinned is not None:
                conn.execute(text(
                    """UPDATE governance.tenant_report_insights SET status='published'
                       WHERE report_id=:id AND report_version=:rv AND insight_version=:iv"""
                ), {"id": report_id, "rv": version, "iv": pinned})
            self._event(conn, "ReportPublished", {"report_id": report_id, "version": version}, f"report-published:{report_id}:{version}")
        return self.get(report_id, version)

    def pending_events(self, limit: int = 100) -> list[dict]:
        with engine.begin() as conn:
            self._scope(conn)
            return [dict(row) for row in conn.execute(text("SELECT * FROM governance.tenant_outbox WHERE status='pending' ORDER BY occurred_at LIMIT :limit"), {"limit": limit}).mappings()]

    # ================================================================
    # AI 洞察制品（版本绑定在 report_version 之上，RLS 同 schema 保护）
    # ================================================================

    @staticmethod
    def _artifact_to_row(artifact: InsightArtifact, insight_version: int) -> dict:
        """把 InsightArtifact 序列化为 tenant_report_insights 的行字典（JSON 字段转文本，Postgres 存 jsonb）。"""
        return {
            "report_id": artifact.report_id, "report_version": artifact.report_version,
            "insight_version": insight_version, "status": artifact.status,
            "brief_json": json.dumps(artifact.brief.model_dump(mode="json"), ensure_ascii=False),
            "input_snapshot_hash": artifact.input_snapshot_hash,
            "selected_evidence_ids": json.dumps(artifact.selected_evidence_ids, ensure_ascii=False),
            "insights_json": json.dumps([item.model_dump(mode="json") for item in artifact.insights], ensure_ascii=False),
            "model": artifact.model, "prompt_version": artifact.prompt_version,
            "critic_verdict": artifact.critic_verdict.value if artifact.critic_verdict else None,
            "critic_result": json.dumps(artifact.critic_result, ensure_ascii=False),
            "validation_errors": json.dumps(artifact.validation_errors, ensure_ascii=False),
            "attribution_evidence": json.dumps(artifact.attribution_evidence, ensure_ascii=False),
            "actor": artifact.actor,
            "reviewed_by": artifact.reviewed_by,
            "reviewed_at": artifact.reviewed_at.isoformat() if artifact.reviewed_at else None,
            "review_reason": artifact.review_reason,
            "duration_ms": artifact.duration_ms,
            "input_tokens": artifact.input_tokens,
            "output_tokens": artifact.output_tokens,
            "estimated_cost": artifact.estimated_cost,
            "data_version_manifest_id": artifact.data_version_manifest_id,
            "data_version_manifest_hash": artifact.data_version_manifest_hash,
        }

    def create_insight(self, artifact: InsightArtifact) -> dict:
        """同步落库一份洞察（sync 路径）：同 (报告版本, 输入哈希) 的 draft 幂等复用，RLS 隔离租户。"""
        with engine.begin() as conn:
            self._scope(conn)
            # 幂等：同报告版本 + 同输入哈希的 draft 直接复用，不重复落库。
            existing = conn.execute(text(
                "SELECT * FROM governance.tenant_report_insights WHERE report_id=:id AND report_version=:rv AND input_snapshot_hash=:hash AND status='draft'"
            ), {"id": artifact.report_id, "rv": artifact.report_version, "hash": artifact.input_snapshot_hash}).mappings().first()
            if existing:
                return self._insight_dict(existing)
            version = conn.execute(text(
                "SELECT COALESCE(MAX(insight_version),0)+1 FROM governance.tenant_report_insights WHERE report_id=:id AND report_version=:rv"
            ), {"id": artifact.report_id, "rv": artifact.report_version}).scalar_one()
            conn.execute(text(
                """INSERT INTO governance.tenant_report_insights
                   (tenant_id, report_id, report_version, insight_version, status, brief_json,
                    input_snapshot_hash, selected_evidence_ids, insights_json, model, prompt_version,
                    critic_verdict, critic_result, validation_errors, attribution_evidence, actor,
                    reviewed_by, reviewed_at, review_reason, data_version_manifest_id, data_version_manifest_hash)
                   VALUES (:tenant_id,:report_id,:report_version,:insight_version,:status,
                           CAST(:brief_json AS jsonb),:input_snapshot_hash,
                           CAST(:selected_evidence_ids AS jsonb),CAST(:insights_json AS jsonb),
                           :model,:prompt_version,:critic_verdict,
                           CAST(:critic_result AS jsonb),CAST(:validation_errors AS jsonb),
                           CAST(:attribution_evidence AS jsonb),:actor,
                           :reviewed_by,:reviewed_at,:review_reason,
                           :data_version_manifest_id,:data_version_manifest_hash)"""
            ), {"tenant_id": self.context.tenant_id, **self._artifact_to_row(artifact, int(version))})
        return self.get_insight(artifact.report_id, artifact.report_version, int(version))

    @staticmethod
    def _insight_dict(row) -> dict:
        """把 Postgres 行转成 dict，并把存成文本的 JSON 字段解析回对象（jsonb 已是对象则跳过）。"""
        result = dict(row)
        for key in ("brief_json", "selected_evidence_ids", "insights_json", "critic_result", "validation_errors", "attribution_evidence"):
            if isinstance(result.get(key), str):
                result[key] = json.loads(result[key])
        return result

    def review_insight(
        self,
        report_id: str,
        report_version: int,
        insight_version: int,
        action: Literal["approve", "reject", "edit"],
        actor: str,
        reason: str = "",
        edited_insights: list[dict] | None = None,
    ) -> dict:
        """人工审核洞察（RLS 隔离）：approve → approved；reject → rejected；edit → 替换内容并 approved。

        仅 draft / pending_review 状态可审核；FOR UPDATE 行锁保证并发下状态判断一致。
        """
        if action not in {"approve", "reject", "edit"}:
            raise ValueError("action must be approve, reject, or edit")
        if action == "edit" and edited_insights is None:
            raise ValueError("edited_insights is required for edit")
        with engine.begin() as conn:
            self._scope(conn)
            row = conn.execute(text(
                "SELECT * FROM governance.tenant_report_insights WHERE report_id=:id AND report_version=:rv AND insight_version=:iv FOR UPDATE"
            ), {"id": report_id, "rv": report_version, "iv": insight_version}).mappings().first()
            if not row:
                raise KeyError(f"Insight not found: {report_id} v{report_version} #{insight_version}")
            if row["status"] not in {"draft", "pending_review"}:
                raise ValueError(f"Insight in {row['status']} state cannot be reviewed")
            reviewed_at = datetime.now(timezone.utc).isoformat()
            if action == "edit":
                conn.execute(text(
                    """UPDATE governance.tenant_report_insights
                       SET status='approved', insights_json=CAST(:insights AS jsonb),
                           reviewed_by=:reviewer, reviewed_at=:reviewed_at, review_reason=:reason
                       WHERE report_id=:id AND report_version=:rv AND insight_version=:iv"""
                ), {"insights": json.dumps(edited_insights, ensure_ascii=False), "reviewer": actor,
                    "reviewed_at": reviewed_at, "reason": reason, "id": report_id, "rv": report_version, "iv": insight_version})
            else:
                status = "approved" if action == "approve" else "rejected"
                conn.execute(text(
                    """UPDATE governance.tenant_report_insights
                       SET status=:status, reviewed_by=:reviewer, reviewed_at=:reviewed_at, review_reason=:reason
                       WHERE report_id=:id AND report_version=:rv AND insight_version=:iv"""
                ), {"status": status, "reviewer": actor, "reviewed_at": reviewed_at, "reason": reason,
                    "id": report_id, "rv": report_version, "iv": insight_version})
        return self.get_insight(report_id, report_version, insight_version)

    def get_insight(self, report_id: str, report_version: int, insight_version: int | None = None) -> dict:
        """按报告版本读取洞察（RLS 隔离）；insight_version 缺省时返回最新一条。未找到抛 KeyError。"""
        with engine.begin() as conn:
            self._scope(conn)
            sql = "SELECT * FROM governance.tenant_report_insights WHERE report_id=:id AND report_version=:rv" + (
                " AND insight_version=:iv" if insight_version else ""
            ) + " ORDER BY insight_version DESC LIMIT 1"
            row = conn.execute(text(sql), {"id": report_id, "rv": report_version, "iv": insight_version}).mappings().first()
        if not row:
            raise KeyError(f"Insight not found: {report_id} v{report_version}")
        return self._insight_dict(row)

    def list_insights(self, report_id: str, report_version: int, limit: int = 20) -> list[dict]:
        """列出某报告版本的全部洞察（RLS 隔离），按 insight_version 降序（最新在前）。"""
        with engine.begin() as conn:
            self._scope(conn)
            rows = conn.execute(text(
                """SELECT * FROM governance.tenant_report_insights WHERE report_id=:id AND report_version=:rv
                   ORDER BY insight_version DESC LIMIT :limit"""
            ), {"id": report_id, "rv": report_version, "limit": limit}).mappings()
            return [self._insight_dict(row) for row in rows]

    def latest_insight_by_hash(self, report_id: str, report_version: int, input_snapshot_hash: str) -> dict | None:
        """按输入快照哈希查最新一条洞察（RLS 隔离）；用于同步路径的幂等复用。"""
        with engine.begin() as conn:
            self._scope(conn)
            row = conn.execute(text(
                """SELECT * FROM governance.tenant_report_insights
                   WHERE report_id=:id AND report_version=:rv AND input_snapshot_hash=:hash
                   ORDER BY insight_version DESC LIMIT 1"""
            ), {"id": report_id, "rv": report_version, "hash": input_snapshot_hash}).mappings().first()
        return self._insight_dict(row) if row else None

    def create_generating(self, report_id: str, report_version: int, brief, actor: str, data_version_manifest_id: str | None = None, data_version_manifest_hash: str | None = None) -> dict:
        """插入 generating 占位行（POST 秒回），后台任务完成后再 finalize。

        同 (报告版本, brief) 已有 generating 或 draft → 幂等复用，防双击重复起任务。
        """
        brief_json = json.dumps(brief.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        with engine.begin() as conn:
            self._scope(conn)
            existing = conn.execute(text(
                """SELECT * FROM governance.tenant_report_insights
                   WHERE report_id=:id AND report_version=:rv AND brief_json=CAST(:brief AS jsonb) AND status IN ('generating','draft')
                   ORDER BY insight_version DESC LIMIT 1"""
            ), {"id": report_id, "rv": report_version, "brief": brief_json}).mappings().first()
            if existing:
                return self._insight_dict(existing)
            version = conn.execute(text(
                "SELECT COALESCE(MAX(insight_version),0)+1 FROM governance.tenant_report_insights WHERE report_id=:id AND report_version=:rv"
            ), {"id": report_id, "rv": report_version}).scalar_one()
            conn.execute(text(
                """INSERT INTO governance.tenant_report_insights
                   (tenant_id, report_id, report_version, insight_version, status, brief_json,
                    input_snapshot_hash, selected_evidence_ids, insights_json, model, prompt_version,
                    critic_verdict, critic_result, validation_errors, attribution_evidence, actor,
                    data_version_manifest_id, data_version_manifest_hash)
                   VALUES (:tenant_id,:report_id,:report_version,:insight_version,'generating',
                           CAST(:brief_json AS jsonb),'',CAST('[]' AS jsonb),CAST('[]' AS jsonb),
                           '','pending',NULL,CAST('[]' AS jsonb),CAST('[]' AS jsonb),
                           CAST('{}' AS jsonb),:actor,:manifest_id,:manifest_hash)"""
            ), {"tenant_id": self.context.tenant_id, "report_id": report_id, "report_version": report_version,
                "insight_version": int(version), "brief_json": brief_json, "actor": actor, "manifest_id": data_version_manifest_id, "manifest_hash": data_version_manifest_hash})
        return self.get_insight(report_id, report_version, int(version))

    def finalize_insight(self, report_id: str, report_version: int, insight_version: int, artifact) -> dict:
        """后台任务完成后，把 generating 占位行更新为最终制品（含观测字段）。仅 generating 可 finalize。"""
        row = self._artifact_to_row(artifact, insight_version)
        with engine.begin() as conn:
            self._scope(conn)
            current = conn.execute(text(
                "SELECT status FROM governance.tenant_report_insights WHERE report_id=:id AND report_version=:rv AND insight_version=:iv"
            ), {"id": report_id, "rv": report_version, "iv": insight_version}).scalar_one_or_none()
            if current != "generating":
                raise ValueError(f"Insight in {current or 'missing'} state cannot be finalized")
            conn.execute(text(
                """UPDATE governance.tenant_report_insights
                   SET status=:status, brief_json=CAST(:brief_json AS jsonb), input_snapshot_hash=:input_snapshot_hash,
                       selected_evidence_ids=CAST(:selected_evidence_ids AS jsonb), insights_json=CAST(:insights_json AS jsonb),
                       model=:model, prompt_version=:prompt_version, critic_verdict=:critic_verdict,
                       critic_result=CAST(:critic_result AS jsonb), validation_errors=CAST(:validation_errors AS jsonb),
                       attribution_evidence=CAST(:attribution_evidence AS jsonb),
                       duration_ms=:duration_ms, input_tokens=:input_tokens, output_tokens=:output_tokens,
                       estimated_cost=:estimated_cost, data_version_manifest_id=:data_version_manifest_id,
                       data_version_manifest_hash=:data_version_manifest_hash
                   WHERE report_id=:id AND report_version=:rv AND insight_version=:iv"""
            ), {**row, "id": report_id, "rv": report_version, "iv": insight_version})
        return self.get_insight(report_id, report_version, insight_version)

    def fail_insight(self, report_id: str, report_version: int, insight_version: int, reason: str) -> dict:
        """后台任务崩溃兜底：把 generating 行标记为 rejected，不留在 generating 卡死。"""
        with engine.begin() as conn:
            self._scope(conn)
            conn.execute(text(
                """UPDATE governance.tenant_report_insights SET status='rejected', validation_errors=CAST(:errors AS jsonb)
                   WHERE report_id=:id AND report_version=:rv AND insight_version=:iv AND status='generating'"""
            ), {"errors": json.dumps([reason], ensure_ascii=False), "id": report_id, "rv": report_version, "iv": insight_version})
        return self.get_insight(report_id, report_version, insight_version)

    def update_insight_progress(self, report_id: str, report_version: int, insight_version: int, stage: str, message: str) -> None:
        """记录生成阶段进度（attribution/llm_generation/critic），供 SSE 端点轮询推送；仅 generating 状态写入。"""
        with engine.begin() as conn:
            self._scope(conn)
            conn.execute(text(
                """UPDATE governance.tenant_report_insights SET progress=CAST(:progress AS jsonb)
                   WHERE report_id=:id AND report_version=:rv AND insight_version=:iv AND status='generating'"""
            ), {"progress": json.dumps({"stage": stage, "message": message}, ensure_ascii=False),
                "id": report_id, "rv": report_version, "iv": insight_version})

    def list_all_insights(self, limit: int = 500) -> list[dict]:
        """跨报告读取当前租户的全部洞察（RLS 隔离，质量仪表盘聚合用），按创建时间降序。"""
        with engine.begin() as conn:
            self._scope(conn)
            rows = conn.execute(text(
                "SELECT * FROM governance.tenant_report_insights ORDER BY created_at DESC LIMIT :limit"
            ), {"limit": limit}).mappings()
            return [self._insight_dict(row) for row in rows]

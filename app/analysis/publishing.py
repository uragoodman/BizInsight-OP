"""Versioned, idempotent report approval and publishing service."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from app.analysis.contracts import InsightArtifact, PeriodicReport, ReportApproval
from app.analysis.events import EventOutbox


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


class ReportStore:
    def __init__(self, path: str = "data/phase6.sqlite") -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.outbox = EventOutbox(path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS report_versions (
                    report_id TEXT NOT NULL, version INTEGER NOT NULL, status TEXT NOT NULL,
                    content TEXT NOT NULL, content_hash TEXT NOT NULL, report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL, actor TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '',
                    pinned_insight_version INTEGER,
                    PRIMARY KEY(report_id, version)
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS report_approvals (
                    report_id TEXT NOT NULL, version INTEGER NOT NULL, action TEXT NOT NULL,
                    actor TEXT NOT NULL, reason TEXT NOT NULL, original_hash TEXT NOT NULL,
                    reviewed_hash TEXT NOT NULL, decided_at TEXT NOT NULL,
                    idempotency_key TEXT,
                    UNIQUE(report_id, version, action, reviewed_hash)
                )
            """)
            self._ensure_column(
                connection, "report_approvals", "idempotency_key",
                "ALTER TABLE report_approvals ADD COLUMN idempotency_key TEXT",
            )
            connection.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS uq_report_approval_idempotency_key
                ON report_approvals(idempotency_key) WHERE idempotency_key IS NOT NULL
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS report_insights (
                    report_id TEXT NOT NULL, report_version INTEGER NOT NULL, insight_version INTEGER NOT NULL,
                    status TEXT NOT NULL, brief_json TEXT NOT NULL, input_snapshot_hash TEXT NOT NULL,
                    selected_evidence_ids TEXT NOT NULL, insights_json TEXT NOT NULL,
                    model TEXT NOT NULL, prompt_version TEXT NOT NULL, critic_verdict TEXT,
                    critic_result TEXT NOT NULL DEFAULT '[]', validation_errors TEXT NOT NULL DEFAULT '[]',
                    attribution_evidence TEXT NOT NULL DEFAULT '{}',
                    investigation_summary TEXT,
                    actor TEXT NOT NULL DEFAULT 'system',
                    reviewed_by TEXT, reviewed_at TEXT, review_reason TEXT NOT NULL DEFAULT '',
                    duration_ms INTEGER, input_tokens INTEGER, output_tokens INTEGER, estimated_cost REAL,
                    progress TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (report_id, report_version, insight_version)
                )
            """)
            # 存量库补列：CREATE TABLE IF NOT EXISTS 不会给已有表加列。
            self._ensure_column(
                connection, "report_versions", "pinned_insight_version",
                "ALTER TABLE report_versions ADD COLUMN pinned_insight_version INTEGER",
            )
            for column, ddl in (
                ("attribution_evidence", "ALTER TABLE report_insights ADD COLUMN attribution_evidence TEXT NOT NULL DEFAULT '{}'"),
                ("investigation_summary", "ALTER TABLE report_insights ADD COLUMN investigation_summary TEXT"),
                ("reviewed_by", "ALTER TABLE report_insights ADD COLUMN reviewed_by TEXT"),
                ("reviewed_at", "ALTER TABLE report_insights ADD COLUMN reviewed_at TEXT"),
                ("review_reason", "ALTER TABLE report_insights ADD COLUMN review_reason TEXT NOT NULL DEFAULT ''"),
                ("duration_ms", "ALTER TABLE report_insights ADD COLUMN duration_ms INTEGER"),
                ("input_tokens", "ALTER TABLE report_insights ADD COLUMN input_tokens INTEGER"),
                ("output_tokens", "ALTER TABLE report_insights ADD COLUMN output_tokens INTEGER"),
                ("estimated_cost", "ALTER TABLE report_insights ADD COLUMN estimated_cost REAL"),
                ("progress", "ALTER TABLE report_insights ADD COLUMN progress TEXT NOT NULL DEFAULT '{}'"),
            ):
                self._ensure_column(connection, "report_insights", column, ddl)

    @staticmethod
    def _ensure_column(connection: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        columns = [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]
        if column not in columns:
            connection.execute(ddl)

    def create_draft(self, report: PeriodicReport, content: str, actor: str = "system") -> dict[str, object]:
        digest = content_hash(content)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM report_versions WHERE report_id=? AND content_hash=?", (report.report_id, digest)
            ).fetchone()
            if existing:
                return dict(existing)
            row = connection.execute(
                "SELECT COALESCE(MAX(version),0)+1 AS next_version FROM report_versions WHERE report_id=?",
                (report.report_id,),
            ).fetchone()
            version = int(row["next_version"])
            connection.execute(
                """INSERT INTO report_versions
                   (report_id,version,status,content,content_hash,report_json,created_at,actor)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (report.report_id, version, "draft", content, digest, report.model_dump_json(),
                 datetime.now(timezone.utc).isoformat(), actor),
            )
            self.outbox.emit_in_transaction(
                connection, "ReportDrafted", {"report_id": report.report_id, "version": version},
                f"report-drafted:{report.report_id}:{version}",
            )
        return self.get(report.report_id, version)

    def get(self, report_id: str, version: int | None = None) -> dict[str, object]:
        query = "SELECT * FROM report_versions WHERE report_id=?"
        params: tuple[object, ...] = (report_id,)
        if version is not None:
            query += " AND version=?"
            params += (version,)
        query += " ORDER BY version DESC LIMIT 1"
        with self._connection() as connection:
            row = connection.execute(query, params).fetchone()
        if row is None:
            raise KeyError(f"Report not found: {report_id}")
        return dict(row)

    def list(self, limit: int = 50) -> list[dict[str, object]]:
        """Return the most recent version of each report for the console."""
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT rv.* FROM report_versions rv
                   INNER JOIN (SELECT report_id, MAX(version) AS version
                               FROM report_versions GROUP BY report_id) latest
                     ON latest.report_id=rv.report_id AND latest.version=rv.version
                   ORDER BY rv.created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def review(
        self,
        report_id: str,
        version: int,
        action: Literal["approve", "edit", "reject"],
        actor: str,
        reason: str = "",
        edited_content: str | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> ReportApproval:
        if action not in {"approve", "edit", "reject"}:
            raise ValueError("action must be approve, edit, or reject")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            original_row = connection.execute(
                "SELECT * FROM report_versions WHERE report_id=? AND version=?", (report_id, version),
            ).fetchone()
            if original_row is None:
                raise KeyError(f"Report not found: {report_id}")
            original = dict(original_row)
            reviewed = edited_content if action == "edit" else str(original["content"])
            if action == "edit" and edited_content is None:
                raise ValueError("edited_content is required for edit")
            reviewed_hash = content_hash(str(reviewed))
            if idempotency_key:
                prior = connection.execute(
                    "SELECT * FROM report_approvals WHERE idempotency_key=?", (idempotency_key,),
                ).fetchone()
                if prior is not None:
                    prior = dict(prior)
                    if (
                        prior["report_id"] != report_id
                        or prior["action"] != action
                        or prior["actor"] != actor
                        or prior["reason"] != reason
                        or prior["original_hash"] != str(original["content_hash"])
                        or prior["reviewed_hash"] != reviewed_hash
                    ):
                        raise ValueError("review idempotency key reused with different decision")
                    return ReportApproval(
                        report_id=report_id, report_version=int(prior["version"]),
                        action=action, actor=actor, reason=reason,
                        original_hash=str(prior["original_hash"]),
                        reviewed_hash=str(prior["reviewed_hash"]),
                        decided_at=datetime.fromisoformat(str(prior["decided_at"])),
                    )
            if original["status"] != "draft":
                raise ValueError(f"Report in {original['status']} state cannot be reviewed")
            next_version = version
            status = "rejected" if action == "reject" else "approved"
            if action == "edit":
                next_version = int(connection.execute(
                    "SELECT COALESCE(MAX(version),0)+1 AS v FROM report_versions WHERE report_id=?", (report_id,)
                ).fetchone()["v"])
                connection.execute(
                    """UPDATE report_versions
                       SET status='rejected', actor=?, reason=?
                       WHERE report_id=? AND version=?""",
                    (actor, f"superseded by edited version {next_version}", report_id, version),
                )
                connection.execute(
                    """INSERT INTO report_versions
                       (report_id,version,status,content,content_hash,report_json,created_at,actor,reason)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (report_id, next_version, "approved", reviewed, reviewed_hash, original["report_json"],
                     datetime.now(timezone.utc).isoformat(), actor, reason),
                )
            else:
                connection.execute(
                    "UPDATE report_versions SET status=?, actor=?, reason=? WHERE report_id=? AND version=?",
                    (status, actor, reason, report_id, version),
                )
            approval = ReportApproval(
                report_id=report_id, report_version=next_version, action=action, actor=actor, reason=reason,
                original_hash=str(original["content_hash"]), reviewed_hash=reviewed_hash,
            )
            connection.execute(
                """INSERT OR IGNORE INTO report_approvals
                   (report_id,version,action,actor,reason,original_hash,reviewed_hash,decided_at,idempotency_key)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (report_id, next_version, action, actor, reason, approval.original_hash,
                 approval.reviewed_hash, approval.decided_at.isoformat(), idempotency_key),
            )
            if action != "reject":
                self.outbox.emit_in_transaction(
                    connection, "ReportApproved", {"report_id": report_id, "version": next_version},
                    f"report-approved:{report_id}:{next_version}",
                )
        return approval

    def publish(self, report_id: str, version: int, actor: str) -> dict[str, object]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            report_row = connection.execute(
                "SELECT * FROM report_versions WHERE report_id=? AND version=?", (report_id, version),
            ).fetchone()
            if report_row is None:
                raise KeyError(f"Report not found: {report_id}")
            report = dict(report_row)
            if report["status"] == "published":
                return report
            if report["status"] != "approved":
                raise ValueError("Only approved reports can be published")
            latest = connection.execute(
                "SELECT version FROM report_versions WHERE report_id=? ORDER BY version DESC LIMIT 1", (report_id,),
            ).fetchone()
            if latest is None or int(latest["version"]) != version:
                raise ValueError("Only the latest approved report version can be published")
            # 发布时钉死洞察版本：取最新一条人工审核通过（status=approved）的洞察。
            # 发布物从此恒定；已发布报告不再展示后来生成的洞察。无合格洞察时钉 null（纯确定性报告）。
            pinned_row = connection.execute(
                """SELECT insight_version FROM report_insights
                   WHERE report_id=? AND report_version=? AND status='approved'
                   ORDER BY insight_version DESC LIMIT 1""",
                (report_id, version),
            ).fetchone()
            pinned = int(pinned_row[0]) if pinned_row else None
            connection.execute(
                "UPDATE report_versions SET status='published', pinned_insight_version=?, actor=? WHERE report_id=? AND version=?",
                (pinned, actor, report_id, version),
            )
            # 同事务把被钉选的洞察标记为 published，状态机闭环：draft → approved → published。
            if pinned is not None:
                connection.execute(
                    "UPDATE report_insights SET status='published' WHERE report_id=? AND report_version=? AND insight_version=?",
                    (report_id, version, pinned),
                )
            self.outbox.emit_in_transaction(
                connection, "ReportPublished", {"report_id": report_id, "version": version},
                f"report-published:{report_id}:{version}",
            )
        return self.get(report_id, version)


    # ================================================================
    # AI 洞察制品（版本绑定在 report_version 之上）
    # ================================================================

    @staticmethod
    def _artifact_to_row(artifact: InsightArtifact, insight_version: int) -> dict[str, object]:
        """把 InsightArtifact 序列化为 report_insights 的行字典（JSON 字段转文本）。"""
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
            "investigation_summary": (
                json.dumps(artifact.investigation_summary, ensure_ascii=False)
                if artifact.investigation_summary is not None else None
            ),
            "actor": artifact.actor,
            "reviewed_by": artifact.reviewed_by,
            "reviewed_at": artifact.reviewed_at.isoformat() if artifact.reviewed_at else None,
            "review_reason": artifact.review_reason,
            "duration_ms": artifact.duration_ms,
            "input_tokens": artifact.input_tokens,
            "output_tokens": artifact.output_tokens,
            "estimated_cost": artifact.estimated_cost,
        }

    @staticmethod
    def _insight_dict(row: sqlite3.Row) -> dict[str, object]:
        """把 SQLite 行转成 dict，并把存成 TEXT 的 JSON 字段解析回对象。"""
        result = dict(row)
        for key in ("brief_json", "selected_evidence_ids", "insights_json", "critic_result", "validation_errors", "attribution_evidence", "investigation_summary", "progress"):
            value = result.get(key)
            if isinstance(value, str):
                try:
                    result[key] = json.loads(value)
                except (TypeError, ValueError):
                    pass
        return result

    def create_insight(self, artifact: InsightArtifact) -> dict[str, object]:
        """同步落库一份洞察（sync 路径）：同 (报告版本, 输入哈希) 的 draft 幂等复用。

        参数：`artifact` 为已生成完成的洞察制品（含观测字段）。

        返回值：新落库（或复用的既有 draft）的行 dict。
        """
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            # 幂等：同报告版本 + 同输入哈希的 draft 直接复用，不重复落库。
            existing = connection.execute(
                "SELECT * FROM report_insights WHERE report_id=? AND report_version=? AND input_snapshot_hash=? AND status='draft'",
                (artifact.report_id, artifact.report_version, artifact.input_snapshot_hash),
            ).fetchone()
            if existing:
                return self._insight_dict(existing)
            version = int(connection.execute(
                "SELECT COALESCE(MAX(insight_version),0)+1 FROM report_insights WHERE report_id=? AND report_version=?",
                (artifact.report_id, artifact.report_version),
            ).fetchone()[0])
            row = self._artifact_to_row(artifact, version)
            row["created_at"] = datetime.now(timezone.utc).isoformat()
            connection.execute(
                """INSERT INTO report_insights
                   (report_id, report_version, insight_version, status, brief_json, input_snapshot_hash,
                    selected_evidence_ids, insights_json, model, prompt_version, critic_verdict,
                    critic_result, validation_errors, attribution_evidence, investigation_summary, actor, reviewed_by,
                    reviewed_at, review_reason, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                tuple(row[key] for key in (
                    "report_id", "report_version", "insight_version", "status", "brief_json",
                    "input_snapshot_hash", "selected_evidence_ids", "insights_json", "model",
                    "prompt_version", "critic_verdict", "critic_result", "validation_errors",
                    "attribution_evidence", "investigation_summary", "actor", "reviewed_by", "reviewed_at", "review_reason",
                    "created_at",
                )),
            )
        return self.get_insight(artifact.report_id, artifact.report_version, version)

    def review_insight(
        self,
        report_id: str,
        report_version: int,
        insight_version: int,
        action: Literal["approve", "reject", "edit"],
        actor: str,
        reason: str = "",
        edited_insights: list[dict] | None = None,
    ) -> dict[str, object]:
        """人工审核洞察：approve → approved；reject → rejected；edit → 替换内容并 approved。

        参数：`report_id` / `report_version` / `insight_version` 定位洞察；
        `action` 审核动作；`actor` / `reason` 审核人及备注；`edited_insights` 为 edit 时的新内容列表。

        返回值：审核后的行 dict。仅 draft / pending_review 状态可审核。
        """
        if action not in {"approve", "reject", "edit"}:
            raise ValueError("action must be approve, reject, or edit")
        if action == "edit" and edited_insights is None:
            raise ValueError("edited_insights is required for edit")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM report_insights WHERE report_id=? AND report_version=? AND insight_version=?",
                (report_id, report_version, insight_version),
            ).fetchone()
            if row is None:
                raise KeyError(f"Insight not found: {report_id} v{report_version} #{insight_version}")
            if row["status"] not in {"draft", "pending_review"}:
                raise ValueError(f"Insight in {row['status']} state cannot be reviewed")
            reviewed_at = datetime.now(timezone.utc).isoformat()
            if action == "edit":
                connection.execute(
                    """UPDATE report_insights SET status='approved', insights_json=?, reviewed_by=?, reviewed_at=?, review_reason=?
                       WHERE report_id=? AND report_version=? AND insight_version=?""",
                    (json.dumps(edited_insights, ensure_ascii=False), actor, reviewed_at, reason,
                     report_id, report_version, insight_version),
                )
            else:
                status = "approved" if action == "approve" else "rejected"
                connection.execute(
                    """UPDATE report_insights SET status=?, reviewed_by=?, reviewed_at=?, review_reason=?
                       WHERE report_id=? AND report_version=? AND insight_version=?""",
                    (status, actor, reviewed_at, reason, report_id, report_version, insight_version),
                )
        return self.get_insight(report_id, report_version, insight_version)

    def get_insight(self, report_id: str, report_version: int, insight_version: int | None = None) -> dict[str, object]:
        """按报告版本读取洞察；insight_version 缺省时返回最新一条。未找到抛 KeyError。"""
        query = "SELECT * FROM report_insights WHERE report_id=? AND report_version=?"
        params: tuple[object, ...] = (report_id, report_version)
        if insight_version is not None:
            query += " AND insight_version=?"
            params += (insight_version,)
        query += " ORDER BY insight_version DESC LIMIT 1"
        with self._connection() as connection:
            row = connection.execute(query, params).fetchone()
        if row is None:
            raise KeyError(f"Insight not found: {report_id} v{report_version}")
        return self._insight_dict(row)

    def list_insights(self, report_id: str, report_version: int, limit: int = 20) -> list[dict[str, object]]:
        """列出某报告版本的全部洞察，按 insight_version 降序（最新在前）。"""
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT * FROM report_insights WHERE report_id=? AND report_version=?
                   ORDER BY insight_version DESC LIMIT ?""",
                (report_id, report_version, limit),
            ).fetchall()
        return [self._insight_dict(row) for row in rows]

    def latest_insight_by_hash(self, report_id: str, report_version: int, input_snapshot_hash: str) -> dict[str, object] | None:
        """按输入快照哈希查最新一条洞察；用于同步路径的幂等复用。"""
        with self._connection() as connection:
            row = connection.execute(
                """SELECT * FROM report_insights WHERE report_id=? AND report_version=? AND input_snapshot_hash=?
                   ORDER BY insight_version DESC LIMIT 1""",
                (report_id, report_version, input_snapshot_hash),
            ).fetchone()
        return self._insight_dict(row) if row else None

    def create_generating(self, report_id: str, report_version: int, brief, actor: str) -> dict[str, object]:
        """插入 generating 占位行（POST 秒回），后台任务完成后再 finalize。

        参数：
        - `brief`：InsightBrief，其 JSON 序列化作为去重键。
        - `actor`：触发者。

        返回值：占位行 dict（含真实 insight_version）。

        同 (report_id, rv, brief) 已有 generating 或 draft → 幂等复用，防双击重复起任务。
        """
        brief_json = json.dumps(brief.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """SELECT * FROM report_insights WHERE report_id=? AND report_version=? AND brief_json=? AND status IN ('generating','draft')
                   ORDER BY insight_version DESC LIMIT 1""",
                (report_id, report_version, brief_json),
            ).fetchone()
            if existing:
                return self._insight_dict(existing)
            version = int(connection.execute(
                "SELECT COALESCE(MAX(insight_version),0)+1 FROM report_insights WHERE report_id=? AND report_version=?",
                (report_id, report_version),
            ).fetchone()[0])
            connection.execute(
                """INSERT INTO report_insights
                   (report_id, report_version, insight_version, status, brief_json, input_snapshot_hash,
                    selected_evidence_ids, insights_json, model, prompt_version, critic_verdict,
                    critic_result, validation_errors, attribution_evidence, actor, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (report_id, report_version, version, "generating", brief_json, "",
                 "[]", "[]", "", "pending", None, "[]", "[]", "{}", actor,
                 datetime.now(timezone.utc).isoformat()),
            )
        return self.get_insight(report_id, report_version, version)

    def finalize_insight(self, report_id: str, report_version: int, insight_version: int, artifact) -> dict[str, object]:
        """后台任务完成后，把 generating 占位行更新为最终制品（含观测字段）。

        参数：`artifact` 为生成完成的 InsightArtifact（含 status/insights/观测字段）。

        返回值：finalize 后的行 dict。仅 generating 状态可 finalize（幂等防重复收尾）。
        """
        row = self._artifact_to_row(artifact, insight_version)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT status FROM report_insights WHERE report_id=? AND report_version=? AND insight_version=?",
                (report_id, report_version, insight_version),
            ).fetchone()
            if current is None or current["status"] != "generating":
                raise ValueError(f"Insight in {current['status'] if current else 'missing'} state cannot be finalized")
            connection.execute(
                """UPDATE report_insights
                   SET status=?, brief_json=?, input_snapshot_hash=?, selected_evidence_ids=?, insights_json=?,
                       model=?, prompt_version=?, critic_verdict=?, critic_result=?, validation_errors=?,
                       attribution_evidence=?, investigation_summary=?, duration_ms=?, input_tokens=?, output_tokens=?, estimated_cost=?
                   WHERE report_id=? AND report_version=? AND insight_version=?""",
                tuple(row[key] for key in (
                    "status", "brief_json", "input_snapshot_hash", "selected_evidence_ids", "insights_json",
                    "model", "prompt_version", "critic_verdict", "critic_result", "validation_errors",
                    "attribution_evidence", "investigation_summary", "duration_ms", "input_tokens", "output_tokens", "estimated_cost",
                )) + (report_id, report_version, insight_version),
            )
        return self.get_insight(report_id, report_version, insight_version)

    def fail_insight(self, report_id: str, report_version: int, insight_version: int, reason: str) -> dict[str, object]:
        """后台任务崩溃兜底：把 generating 行标记为 rejected，不留在 generating 卡死。

        参数：`reason` 为失败原因，写入 validation_errors 作为审计记录。
        """
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """UPDATE report_insights SET status='rejected', validation_errors=?
                   WHERE report_id=? AND report_version=? AND insight_version=? AND status='generating'""",
                (json.dumps([reason], ensure_ascii=False), report_id, report_version, insight_version),
            )
        return self.get_insight(report_id, report_version, insight_version)

    def update_insight_progress(self, report_id: str, report_version: int, insight_version: int, stage: str, message: str) -> None:
        """记录生成阶段进度，供 SSE 端点轮询推送（只在 generating 状态时写入）。

        参数：`stage` 阶段名（attribution/llm_generation/critic），`message` 用户可见文案。
        """
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """UPDATE report_insights SET progress=?
                   WHERE report_id=? AND report_version=? AND insight_version=? AND status='generating'""",
                (json.dumps({"stage": stage, "message": message}, ensure_ascii=False), report_id, report_version, insight_version),
            )

    def list_all_insights(self, limit: int = 500) -> list[dict[str, object]]:
        """跨报告读取全部洞察（质量仪表盘聚合用），按创建时间降序。"""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM report_insights ORDER BY created_at DESC LIMIT ?", (limit,),
            ).fetchall()
        return [self._insight_dict(row) for row in rows]


def report_risks(report: PeriodicReport) -> list[str]:
    risks = list(report.risks)
    if any(metric.value is None for metric in report.metrics):
        risks.append("missing_metric_data")
    if any(chart.status != "ok" for chart in report.charts):
        risks.append("chart_unavailable")
    if not report.lineage_refs:
        risks.append("missing_lineage")
    if any(abs(item.score) >= item.threshold * 2 for item in report.anomalies):
        risks.append("critical_anomaly")
    if any(
        metric.target_comparison and metric.target_comparison.delta_ratio is not None
        and abs(metric.target_comparison.delta_ratio) >= 20
        for metric in report.metrics
    ):
        risks.append("material_target_variance")
    if report.data_freshness_at and (datetime.now(timezone.utc) - report.data_freshness_at).total_seconds() > 48 * 3600:
        risks.append("stale_data")
    return sorted(set(risks))

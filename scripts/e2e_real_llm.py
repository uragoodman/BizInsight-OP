"""真实 LLM E2E：构建真实周报 → 真实 DeepSeek 生成洞察 → Critic → 人审 → 发布钉死。

验证 P1 四类场景中"正常归因洞察 + 完整发布链路"，使用本地真实数据与真实 LLM。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from app.analysis.contracts import InsightAngle, InsightBrief
from app.analysis.insights import generate_insight_artifact, InsightUnavailableError
from app.analysis.publishing import ReportStore
from app.analysis.reports import build_periodic_report_artifact


def main() -> int:
    """执行真实 LLM E2E：构建周报 → 生成洞察 → 人审 → 发布钉死；失败返回非零退出码。"""
    report = build_periodic_report_artifact("本周经营周报")
    if report is None:
        print("E2E FAILED: 无法从真实数据构建周报（build_periodic_report_artifact 返回 None）")
        return 1
    print(f"[1] 真实周报构建成功: report_id={report.report_id} 期间={report.period.start_date}~{report.period.end_date} "
          f"指标={[m.metric_name for m in report.metrics]} 异常={len(report.anomalies)}")

    brief = InsightBrief(
        angle=InsightAngle.CONTRIBUTION, max_insights=3, min_words=30, max_words=160,
        include_recommendations=True, focus_metrics=["gmv"],
    )
    try:
        artifact = generate_insight_artifact(report, 1, brief, "e2e-real-llm")
    except InsightUnavailableError as exc:
        print(f"E2E FAILED: 生成洞察时 LLM 不可用: {exc}")
        return 1
    print(f"[2] 洞察生成完成: status={artifact.status} critic_verdict={artifact.critic_verdict} "
          f"model={artifact.model} prompt_version={artifact.prompt_version}")
    print(f"    归因补查证据数: {len(artifact.attribution_evidence)}")
    print(f"    可见证据白名单数: {len(artifact.selected_evidence_ids)}")
    for i, ins in enumerate(artifact.insights, start=1):
        print(f"    insight[{i}] type={ins.statement_type.value} conf={ins.confidence} ev={ins.evidence_ids}")
        print(f"            {ins.title}: {ins.finding[:80]}")
        if ins.recommendation:
            print(f"            建议: {ins.recommendation[:80]}")
    if artifact.validation_errors:
        print(f"    校验错误: {artifact.validation_errors}")

    if artifact.status == "rejected":
        print("E2E PARTIAL: 洞察被服务端/Critic 拒绝（已保存审计），确定性报告仍可发布。")
    if artifact.status == "pending_review":
        print("E2E PARTIAL: Critic 不可用，洞察 pending_review（fail-closed 生效）。")

    # 存储 + 人审 + 发布（SQLite platform 路径）
    tmp = Path(tempfile.mkdtemp(prefix="e2e-real-"))
    store = ReportStore(str(tmp / "phase6.sqlite"))
    content = f"# {report.title}\n自动生成草稿。"
    store.create_draft(report, content, "e2e")
    version = 1
    row = store.create_insight(artifact)
    iv = int(row["insight_version"])
    print(f"[3] 洞察已存储: insight_version={iv} status={row['status']}")

    if artifact.status in {"draft", "pending_review"}:
        store.review_insight(report.report_id, version, iv, "approve", "e2e-reviewer", "真实 E2E 人工审核通过")
        print(f"[4] 人工审核通过: {store.get_insight(report.report_id, version, iv)['status']}")
    store.review(report.report_id, version, "approve", "e2e-reviewer", "真实 E2E 审核")
    published = store.publish(report.report_id, version, "e2e-publisher")
    pinned = published.get("pinned_insight_version")
    print(f"[5] 报告已发布: status={published['status']} pinned_insight_version={pinned}")
    if pinned is not None:
        pin_status = store.get_insight(report.report_id, version, int(pinned))["status"]
        print(f"    被钉选洞察状态: {pin_status}（应为 published）")
        assert pin_status == "published"
    print("E2E PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

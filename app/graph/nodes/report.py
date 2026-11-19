"""Report Agent — 报告草稿生成。

Target 阶段：整合指标定义、数据结果和归因结论，生成报告草案。
与 Critic Agent 形成 Draft + Critic 工作模式。

MECE 写入字段：analysis_result, current_agent, node_history
"""

from app.analysis.reports import build_periodic_report_artifact, generate_report
from app.analysis.periodic_reports import report_to_markdown
from app.analysis.publishing import ReportStore, report_risks
from app.agents.contracts import EvidenceBundle, stable_evidence_id
from app.core.config import get_settings
from app.graph.state import AgentState


def report(state: AgentState) -> AgentState:
    """生成经营分析报告草稿。

    参数：
    - `state`：当前工作流状态，包含 `user_message`、`tool_result`、`metric_context`。

    返回值：
    - `AgentState`：写入 `analysis_result` 后的状态。
    """

    if state.get("seller_scope_payload"):
        from app.graph.nodes.seller_scope_guard import reject_unsupported_seller_scope

        return reject_unsupported_seller_scope(state)

    # 打点入口：report 只消费前面已经绑定到 state 的指标定义、工具结果和归因结论，
    # 负责生成草稿并把 report_id/report_version 写回 state；它不应绕过 data_query
    # 直接访问业务数据库。若进入 Critic 修订循环，还要读取上一轮的 revision_feedback。
    question = state.get("user_message", "")
    tool_result = state.get("tool_result", {})
    metric_context = state.get("metric_context", "")

    artifact = None
    try:
        artifact = build_periodic_report_artifact(question)
    except Exception as exc:
        if get_settings().environment.lower() in {"production", "prod"}:
            raise
        state.setdefault("errors", []).append({
            "code": "periodic_report_unavailable",
            "node": "report",
            "message": str(exc),
            "retryable": True,
        })
        artifact = None
    summary = report_to_markdown(artifact) if artifact else generate_report(question, tool_result, metric_context)
    if artifact:
        artifact.risks = report_risks(artifact)
        summary = report_to_markdown(artifact)
        stored = ReportStore(get_settings().phase6_store_path).create_draft(
            artifact, summary, actor=state.get("audit_actor", "system"),
        )
        state["report_id"] = artifact.report_id
        state["report_version"] = int(str(stored["version"]))
        state["report_risks"] = artifact.risks
        state["periodic_report"] = artifact.model_dump(mode="json")
        content = artifact.model_dump(mode="json")
        state.setdefault("evidence_bundles", []).append(EvidenceBundle(
            evidence_id=stable_evidence_id("report", artifact.report_id, content),
            source_type="report", source_ref=artifact.report_id, content=content,
        ).model_dump(mode="json"))
        for evidence_id, evidence_content in artifact.evidence_records.items():
            state.setdefault("evidence_bundles", []).append(EvidenceBundle(
                evidence_id=evidence_id, source_type="report",
                source_ref=f"{artifact.report_id}:{evidence_id}", content=evidence_content,
            ).model_dump(mode="json"))

    # Critic 打回后，本节点读取最近一次审核意见并明确写入修订稿。
    # 这让 Draft + Critic 循环保留反馈上下文，而不是无差别地重复同一份草案。
    revision_feedback: list[str] = []
    if state.get("_revision_counts", {}).get("report", 0) > 0:
        for decision in reversed(state.get("critic_decisions", [])):
            if decision.get("revision_target") == "report":
                revision_feedback = decision.get("llm_issues", [])
                break
    if revision_feedback:
        summary += "\n\n## Critic 修订记录\n" + "\n".join(
            f"- 已针对审核意见复核：{issue}" for issue in revision_feedback
        )

    # 打点出口：analysis_result 是后续 answer/critic 共同消费的报告摘要，report artifact
    # 和 evidence_bundles 则用于持久化、审核和最终答案中的可追溯引用。
    state["analysis_result"] = {
        "summary": summary,
        "task_type": "report_generation",
        "revision_feedback": revision_feedback,
    }
    state["current_agent"] = "report"
    state.setdefault("_revision_counts", {})
    state["_revision_counts"]["report"] = state["_revision_counts"].get("report", 0) + 1
    state.setdefault("node_history", []).append(
        {
            "node": "report",
            "summary": (
                f"Report generated ({len(summary)} chars, "
                f"critic_feedback={len(revision_feedback)})"
            ),
        }
    )
    return state

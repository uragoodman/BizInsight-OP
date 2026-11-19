"""Durable approval gate for high-risk or unresolved Critic decisions."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from langgraph.types import interrupt

from app.graph.state import AgentState
from app.analysis.publishing import ReportStore
from app.core.config import get_settings
from app.harness.storage import write_trace


def human_review(state: AgentState) -> AgentState:
    """执行 Critic 触发的人工审核中断与恢复。

    先把中断前的完整 state 写入 trace，再通过 LangGraph interrupt 把审核载荷
    交给外部；恢复后校验 action，按 approve/edit/reject 更新 ReportStore，最后
    把 human_decision 和 critic_verdict 写回 state，供条件边继续或结束工作流。
    """
    # 这是一个持久化边界：后续 interrupt 可能长时间等待，所以必须先写 trace，
    # 否则“已暂停但未恢复”的现场无法回放。
    # 在 interrupt 暂停前先写入 Trace，确保即使审核永不到来也保留了执行现场
    write_trace(state.get("trace_id", "unknown"), dict(state))
    payload = {
        "thread_id": state.get("thread_id"), "trace_id": state.get("trace_id"),
        "answer": state.get("answer", ""), "critic": state.get("critic_result", {}),
        "report_id": state.get("report_id"), "report_version": state.get("report_version"),
        "report_risks": state.get("report_risks", []),
    }
    # interrupt 返回外部审核系统提交的决定，是从暂停态恢复到业务态的入口。
    decision: dict[str, Any] = interrupt(payload)
    action = decision.get("action")
    if action not in {"approve", "edit", "reject"}:
        raise ValueError("Human review action must be approve, edit, or reject.")
    if action == "edit":
        state["answer"] = str(decision.get("answer", state.get("answer", "")))
        state["final_answer"] = state["answer"]
    if state.get("report_id") and state.get("report_version"):
        store = ReportStore(get_settings().phase6_store_path)
        interaction_index = decision.get("_interaction_index")
        review_identity = state.get("execution_id") or state.get("trace_id") or state.get("thread_id") or "legacy-review"
        idempotency_key = "hitl-review:" + hashlib.sha256(
            f"{review_identity}:{state['report_id']}:{state['report_version']}:{interaction_index if interaction_index is not None else '0'}".encode("utf-8")
        ).hexdigest()
        approval = store.review(
            state["report_id"], state["report_version"], action,
            str(decision.get("actor", "unknown")), str(decision.get("reason", "")),
            state.get("answer") if action == "edit" else None,
            idempotency_key=idempotency_key,
        )
        if action in {"approve", "edit"}:
            published = store.publish(state["report_id"], approval.report_version, str(decision.get("actor", "unknown")))
            state["report_version"] = int(str(published["version"]))
    # 打点出口：human_decision 记录审计信息，critic_verdict 决定后续条件边如何走。
    state["human_decision"] = {
        "action": action, "actor": str(decision.get("actor", "unknown")),
        "reason": str(decision.get("reason", "")), "timestamp": datetime.now(timezone.utc).isoformat(),
        "interaction_index": decision.get("_interaction_index"),
    }
    state["critic_verdict"] = "approved" if action in {"approve", "edit"} else "rejected"
    state["current_agent"] = "human_review"
    state.setdefault("node_history", []).append({"node": "human_review", "summary": f"Human action={action}"})
    return state

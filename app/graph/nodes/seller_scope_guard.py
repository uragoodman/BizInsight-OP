"""Fail-closed response for seller-scope operations without scoped handlers."""

from __future__ import annotations

from typing import Any

from app.graph.state import AgentState


SELLER_SCOPE_UNSUPPORTED_REASON = (
    "当前 seller scope 仅支持已注册的按 seller 分组查询；异常诊断和报告尚无 seller-scoped 实现，"
    "本次未执行平台范围查询。"
)


def has_seller_scope(state: AgentState | dict[str, Any]) -> bool:
    """Return whether a signed seller scope was supplied to this graph turn."""

    return bool(state.get("seller_scope_payload"))


def reject_unsupported_seller_scope(state: AgentState) -> AgentState:
    """Write a deterministic terminal refusal without invoking any data handler."""

    state["seller_scope_terminal"] = True
    state["query_results"] = []
    state["tool_result"] = {
        "server": "seller-scope-guard",
        "tool_name": None,
        "arguments": {},
        "request_id": f"{state.get('trace_id') or state.get('thread_id') or 'seller-scope'}:unsupported",
        "result": {
            "status": "unsupported",
            "reason_code": "seller_scope_operation_unsupported",
            "reason": SELLER_SCOPE_UNSUPPORTED_REASON,
            "parsed": {},
        },
    }
    state["analysis_result"] = {
        "summary": SELLER_SCOPE_UNSUPPORTED_REASON,
        "task_type": str(state.get("task_type") or "metric_query"),
    }
    state["answer"] = SELLER_SCOPE_UNSUPPORTED_REASON
    state["terminal_code"] = "CAPABILITY_GAP"
    state["termination_initiator"] = "policy"
    state["current_agent"] = "seller_scope_guard"
    state.setdefault("node_history", []).append({
        "node": "seller_scope_guard",
        "summary": "Stopped an operation that has no seller-scoped execution handler.",
    })
    return state


def seller_scope_guard(state: AgentState) -> AgentState:
    """Graph node used when routing detects an unsupported seller operation."""

    return reject_unsupported_seller_scope(state)

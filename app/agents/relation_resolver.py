"""Conversation relation resolver (AGENT_PLAN Phase 7 / A2).

A small, controlled LLM decision — not a marketing "agent": given the current
conversation task and the new message, decide whether the turn continues the
existing task (optionally reusing its evidence), starts a new/parallel task, or
needs clarification.  The deterministic keyword heuristic in
``app.tasks.service.relation`` remains the degradation path when the LLM is
unavailable or returns an invalid decision.

Evidence governance: ``reusable_evidence_ids`` may only reference evidence
bundles that exist in the prior task's persisted trace — the whitelist is
enforced here, never trusted from model output.
"""

from __future__ import annotations

import json
from typing import Any

ALLOWED_DECISIONS = ("CONTINUE", "PARALLEL_NEW_TASK", "NEW_TASK", "CLARIFY")

_SYSTEM_PROMPT = (
    "你是多轮分析会话的关系解析器。根据当前任务与新消息，只输出 JSON 对象："
    '{"action":"CONTINUE|PARALLEL_NEW_TASK|NEW_TASK|CLARIFY","task_id":"仅 CONTINUE 需要",'
    '"reusable_evidence_ids":["可复用的既有证据ID，仅 CONTINUE 且确需复用时给"],"reason":"简短理由"}。'
    "先读取 context_resolution 和 resolved_message：它们是服务器基于同一授权范围上一轮结构化上下文生成的语义补全，"
    "只用于理解省略和指代，不是权限或数据事实；再判定任务关系。若 needs_clarification 为 true，不要假装补全成功。"
    "CONTINUE 的 task_id 必须来自 task_candidates；若短消息可能指向多个任务，返回 CLARIFY。"
    "判定规则：短追问（如“那这是为什么”“继续看另一维度”）→ CONTINUE 并复用上轮证据；"
    "明确的新问题但可并行推进 → PARALLEL_NEW_TASK；全新独立问题 → NEW_TASK；"
    "无法判断意图 → CLARIFY。不得编造 evidence_id，只能从可用证据清单中选择。"
)


def build_decision_payload(
    current_task: dict[str, Any] | None,
    message: str,
    scope: dict[str, Any] | None,
    available_evidence_ids: list[str],
    context_resolution: dict[str, Any] | None = None,
) -> str:
    """Compact, deterministic JSON context handed to the resolver."""
    task = current_task or {}
    resolution = context_resolution or {}
    turn_resolution = resolution.get("turn_resolution") or {}
    source_needs = resolution.get("source_needs") or turn_resolution.get("source_needs") or []
    progress = task.get("task_state") or {}
    return json.dumps({
        "current_task": {
            "task_id": task.get("task_id"), "status": task.get("status"),
            "objective": task.get("objective"), "latest_trace_id": task.get("latest_trace_id"),
            "scope": task.get("scope_data"),
            "progress": {
                "schema_version": progress.get("schema_version"),
                "state_version": progress.get("state_version"),
                "scope_fingerprint": progress.get("scope_fingerprint"),
                "data_version_manifest_hash": progress.get("data_version_manifest_hash"),
                "completed_steps": progress.get("completed_steps", [])[-12:],
                "approved_claims": progress.get("approved_claims", [])[:5],
                "hypothesis_chain_summary": progress.get("hypothesis_chain_summary", {}),
                "pending_questions": progress.get("pending_questions", [])[:5],
                "latest_trace_id": progress.get("latest_trace_id"),
            },
        },
        "task_candidates": [
            {
                "task_id": task.get("task_id"),
                "status": task.get("status"),
                "title": task.get("title"),
                "objective": str(task.get("objective") or "")[:500],
                "latest_trace_id": task.get("latest_trace_id"),
                "scope": task.get("scope_data"),
            }
            for task in (resolution.get("task_candidates") or [])[:10]
            if isinstance(task, dict) and task.get("task_id")
        ],
        "new_message": message,
        "resolved_message": resolution.get("standalone_question") or message,
        "context_resolution": {
            "relation": resolution.get("relation"),
            "resolved_references": resolution.get("resolved_references", []),
            "known_slots": resolution.get("known_slots", {}),
            "context_sources": resolution.get("context_sources", []),
            "source_turn_id": resolution.get("source_turn_id"),
            "user_changes": resolution.get("user_changes", []),
            "scope_candidates": resolution.get("scope_candidates", []),
            "task_relation": resolution.get("task_relation", {}),
            "source_needs": [
                {
                    "source": need.get("source"),
                    "trigger_ids": need.get("trigger_ids", []),
                    "memory_types": need.get("memory_types", []),
                    "status": need.get("status"),
                }
                for need in source_needs
                if isinstance(need, dict)
            ],
            "needs_clarification": resolution.get("needs_clarification", False),
            "clarification_reason": resolution.get("clarification_reason"),
        } if resolution else {},
        "new_scope": scope,
        "available_evidence_ids": available_evidence_ids,
    }, ensure_ascii=False)


def resolve_relation_llm(
    current_task: dict[str, Any] | None,
    message: str,
    scope: dict[str, Any] | None,
    available_evidence_ids: list[str],
    context_resolution: dict[str, Any] | None = None,
    available_tasks: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Return a validated decision dict, or None when the LLM path is unusable.

    Validation is server-side: unknown actions, missing CONTINUE targets and
    evidence ids outside the prior trace whitelist all degrade to None so the
    keyword fallback takes over.
    """
    from app.llm.client import llm_extract

    resolution = dict(context_resolution or {})
    if available_tasks is not None:
        resolution["task_candidates"] = available_tasks
    raw = llm_extract(
        _SYSTEM_PROMPT,
        build_decision_payload(current_task, message, scope, available_evidence_ids, resolution),
        operation="extract.relation_resolution",
        interaction_policy="nonstream_fast",
    )
    if not isinstance(raw, dict):
        return None
    action = str(raw.get("action", "")).upper()
    if action not in ALLOWED_DECISIONS:
        return None
    reason = str(raw.get("reason") or "")[:200]
    raw_ids = raw.get("reusable_evidence_ids")
    reusable = [str(eid) for eid in raw_ids if str(eid) in set(available_evidence_ids)] if isinstance(raw_ids, list) else []
    if action == "CONTINUE":
        task_id = str(raw.get("task_id") or (current_task or {}).get("task_id") or "")
        if not task_id:
            return None
        allowed_task_ids = {
            str(item.get("task_id")) for item in (available_tasks or [])
            if isinstance(item, dict) and item.get("task_id")
        }
        if available_tasks is not None and task_id not in allowed_task_ids:
            return None
        return {"action": action, "task_id": task_id, "reusable_evidence_ids": reusable, "reason": reason}
    if action == "CLARIFY":
        return {"action": action, "task_id": None, "reusable_evidence_ids": [], "reason": reason}
    return {"action": action, "task_id": None, "reusable_evidence_ids": reusable, "reason": reason}


def prior_evidence_ids(
    latest_trace_id: str | None,
    limit: int = 10,
    expected_scope_fingerprint: str | None = None,
    expected_data_version: str | None = None,
) -> list[str]:
    """Evidence bundle ids from the prior task's persisted trace (best-effort)."""
    # Evidence is reusable only after both identities are known. Without a
    # current data manifest (for example, a non-versioned query path), keep
    # the prior artifacts as history and do not offer them as reusable input.
    if not latest_trace_id or not expected_scope_fingerprint or not expected_data_version:
        return []
    try:
        from app.harness.storage import read_trace

        trace = read_trace(latest_trace_id)
    except Exception:  # noqa: BLE001 - trace storage unavailable → no reuse evidence
        return []
    snapshot = (trace or {}).get("state_snapshot", trace or {})
    if str(snapshot.get("current_query_scope_fingerprint") or "") != expected_scope_fingerprint:
        return []
    if str(snapshot.get("data_version_manifest_hash") or "") != expected_data_version:
        return []
    bundles = snapshot.get("evidence_bundles") or []
    ids = [
        str(bundle.get("evidence_id")) for bundle in bundles
        if isinstance(bundle, dict) and bundle.get("evidence_id") and bundle.get("trusted", True)
    ]
    return ids[:limit]

"""Immutable-by-record capture of the answer that enters the Critic.

The persisted trace normally contains the *final* state.  When Critic asks
for a revision, that final state can no longer tell an offline evaluator which
answer Critic actually reviewed.  This module records that input at the Critic
boundary without changing any answer, verdict, routing or model-call policy.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

from app.graph.state import AgentState


SNAPSHOT_SCHEMA_VERSION = "pre_critic_answer_snapshot_v2"


def _json_copy(value: Any) -> Any:
    """Return a JSON-safe deep copy suitable for a durable trace artifact."""

    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _snapshot_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(canonical.encode("utf-8")).hexdigest()


def capture_pre_critic_answer_snapshot(state: AgentState) -> dict[str, Any]:
    """Append the exact state visible to Critic before it can revise anything.

    A run may enter Critic more than once.  Each entry receives its own record;
    nothing here overwrites an earlier draft.  The snapshot is JSON-copied so
    later in-memory mutations cannot alter its captured contents.
    """

    snapshots = state.setdefault("pre_critic_answer_snapshots", [])
    context_compilation = state.get("context_compilation") or {}
    request_contract = context_compilation.get("request_contract") if isinstance(context_compilation, dict) else {}
    evidence_bundles = state.get("evidence_bundles") or []
    llm_calls = state.get("llm_calls") or []

    payload: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_index": len(snapshots) + 1,
        "task_type": state.get("task_type", ""),
        "trace_id": state.get("trace_id", ""),
        "workflow_version": state.get("workflow_version", ""),
        "prompt_version": state.get("prompt_version", ""),
        "user_message": state.get("user_message", ""),
        "request_contract": request_contract or {},
        "answer": state.get("answer", ""),
        "final_answer": state.get("final_answer"),
        "claims": state.get("claims") or [],
        "evidence_bundles": evidence_bundles,
        # Operational limitations are not numerical business evidence, but
        # reviewers need them to assess safe stops and unfinished branches.
        "operational_context": {
            "terminal_code": state.get("terminal_code"),
            "decision_records": state.get("decision_records") or [],
            "diagnostic_plan": state.get("diagnostic_plan"),
            "diagnostic_step_state": state.get("diagnostic_step_state") or {},
            "investigation_state": state.get("investigation_state") or {},
            "analysis_result": state.get("analysis_result") or {},
        },
        "evidence_bundle_ids": [
            bundle.get("evidence_id")
            for bundle in evidence_bundles
            if isinstance(bundle, dict) and bundle.get("evidence_id")
        ],
        "critic_llm_review_mode": state.get("critic_llm_review_mode", ""),
        "critic_llm_thinking_config": state.get("critic_llm_thinking_config") or {},
        "llm_call_offset": len(llm_calls),
        "llm_token_summary_before_critic": state.get("llm_token_summary") or {},
        "revision_counts": state.get("_revision_counts") or {},
        "answer_present": bool(str(state.get("answer") or "").strip()),
    }
    immutable_payload = _json_copy(payload)
    snapshot = {
        **immutable_payload,
        "snapshot_sha256": _snapshot_hash(immutable_payload),
    }
    snapshots.append(snapshot)
    return snapshot

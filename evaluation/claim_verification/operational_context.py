"""Minimal pre-Critic operational facts; never expose original answer summaries."""
from __future__ import annotations
from typing import Any

DECISION_FIELDS = ("authorization_checked", "registry_capability_checked", "limitation_code", "metric_id", "dimension", "operation", "scope", "comparison", "sql_executed", "evidence_id")
BRANCH_FIELDS = ("branch_id", "branch_type", "metric_id", "dimension", "dimension_value", "status", "parent_evidence_id")
ACTION_FIELDS = ("action_id", "action_type", "branch_id", "metric_id", "dimension", "status", "failure_code", "evidence_ids")


def _project(items: Any, fields: tuple[str, ...]) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    return [{k: item[k] for k in fields if k in item} for item in items if isinstance(item, dict)]


def for_review(context: Any) -> dict[str, Any] | None:
    if not isinstance(context, dict):
        return None
    investigation = context.get("investigation_state") or {}
    if not isinstance(investigation, dict):
        investigation = {}
    stop = investigation.get("stop_reason") or {}
    steps = context.get("diagnostic_step_state") or {}
    return {"terminal_code": context.get("terminal_code"),
            "decision_records": _project(context.get("decision_records"), DECISION_FIELDS),
            "diagnostic_step_state": {str(k): v for k, v in steps.items() if isinstance(k, str) and isinstance(v, str)} if isinstance(steps, dict) else {},
            "investigation_stop": {k: stop[k] for k in ("code", "initiator") if k in stop} if isinstance(stop, dict) else {},
            "branches": _project(investigation.get("branches"), BRANCH_FIELDS),
            "actions": _project(investigation.get("action_results"), ACTION_FIELDS)}

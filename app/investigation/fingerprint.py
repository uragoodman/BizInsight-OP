"""Canonical fingerprints for executed query semantics."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from typing import Any

from app.investigation.contracts import AnalysisAction, BreakdownDimensionAction


_UNORDERED_KEYS = {
    "seller_ids", "allowed_seller_ids", "metric_ids", "allowed_metrics",
    "allowed_dimensions", "allowed_tools", "tool_ids", "filters",
}


def canonicalize(value: Any, *, key: str | None = None, _unordered_context: bool = False) -> Any:
    if isinstance(value, dict):
        return {
            str(name): canonicalize(
                value[name], key=str(name),
                _unordered_context=_unordered_context or str(name) == "filters",
            )
            for name in sorted(value, key=str)
        }
    if isinstance(value, (list, tuple)):
        items = [canonicalize(item, _unordered_context=_unordered_context) for item in value]
        if _unordered_context or key in _UNORDERED_KEYS:
            return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False, default=str))
        return items
    if isinstance(value, (set, frozenset)):
        return sorted((canonicalize(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False, default=str))
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat() if value.tzinfo else value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def executor_semantics(action: AnalysisAction) -> dict[str, Any]:
    payload = action.model_dump(mode="json")
    if isinstance(action, BreakdownDimensionAction):
        # DRILL_DOWN and BREAKDOWN_DIMENSION share one executor. Parent evidence
        # is lineage, while inherited filters determine whether the query differs.
        payload["action_type"] = "DIMENSION_BREAKDOWN"
        payload.pop("parent_evidence_id", None)
    return canonicalize(payload)


def action_fingerprint(
    action: AnalysisAction,
    *,
    current_period: dict[str, Any],
    baseline_period: dict[str, Any] | None,
    canonical_scope: dict[str, Any],
    metric_definition_version: str = "1.0",
    method_version: str,
) -> str:
    envelope = canonicalize({
        "executor_action": executor_semantics(action),
        "current_period": current_period,
        "baseline_period": baseline_period,
        "scope": canonical_scope,
        "metric_definition_version": metric_definition_version,
        "method_version": method_version,
    })
    serialized = json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

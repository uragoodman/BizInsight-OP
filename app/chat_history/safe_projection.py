"""Small allowlisted projections for persisted chat history."""

from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

_CODE = re.compile(r"^[a-z][a-z0-9_]{0,79}$")
_DIMENSION_CANDIDATES = {
    "product_category", "product", "seller", "customer_state", "payment_type",
    "month_position", "customer_city", "seller_state",
}
_SAFE_MISSING_SLOTS = {
    "metric", "dimension", "operation", "date", "comparison_period",
    "comparison_dimension", "scope_filter", "comparison_dates_incomplete",
    "comparison_dates_ambiguous", "comparison_date_labels_conflict",
    "comparison_periods_incomplete", "comparison_periods_ambiguous",
    "comparison_period_labels_conflict", "comparison_shape_conflict",
    "invalid_comparison_date", "invalid_comparison_period",
    "exact_date_context_unavailable", "date_outside_locked_scope",
}
_SAFE_REASONS = {
    "ambiguous_metric": "问题中有多个可能的指标匹配，请选择一个指标。",
    "ambiguous_dimension": "问题中有多个候选分组维度，需要明确选择。",
    "dimension_required": "请说明希望按哪个业务维度分组。",
    "date_outside_locked_scope": "请求日期超出当前已锁定的分析范围。",
    "comparison_dimension_not_supported": "当前请求组合的比较维度暂不支持。",
}


def safe_clarification(value: Any) -> dict[str, Any] | None:
    """Return only display-safe clarification fields; never expose graph state."""
    if not isinstance(value, dict):
        return None
    status = value.get("status")
    reason_code = value.get("reason_code") or value.get("reason")
    if not isinstance(status, str) or not _CODE.fullmatch(status):
        status = "clarification_required"
    if not isinstance(reason_code, str) or not _CODE.fullmatch(reason_code):
        reason_code = ""
    projected: dict[str, Any] = {"status": status}
    if reason_code:
        projected["reason_code"] = reason_code
    # Never echo arbitrary graph/LLM prose through history. A reason is emitted
    # only from a server-owned reason-code mapping; older free-form values may
    # contain the original question, SQL fragments, or provider diagnostics.
    if reason_code in _SAFE_REASONS:
        projected["reason"] = _SAFE_REASONS[reason_code]
    missing = value.get("missing_slots")
    if isinstance(missing, list):
        projected["missing_slots"] = [
            item for item in missing[:20]
            if isinstance(item, str) and item in _SAFE_MISSING_SLOTS
        ]
    dimensions = value.get("dimension_candidates") or value.get("candidate_dimensions")
    if isinstance(dimensions, list):
        projected["dimension_candidates"] = [
            item for item in dimensions[:12]
            if isinstance(item, str) and item in _DIMENSION_CANDIDATES
        ]
    candidates = value.get("ambiguous_candidates") or value.get("candidates")
    if isinstance(candidates, list):
        safe_candidates: list[dict[str, str]] = []
        for item in candidates[:12]:
            if not isinstance(item, dict):
                continue
            metric_id = item.get("metric_id")
            if not isinstance(metric_id, str) or not _CODE.fullmatch(metric_id):
                continue
            try:
                from app.metrics.registry import get_registry

                registered = get_registry().get(metric_id)
            except Exception:  # noqa: BLE001 - an unavailable registry yields no candidate
                registered = None
            if registered is not None:
                # Never trust arbitrary graph-provided display/template text.
                safe_candidates.append({
                    "metric_id": metric_id,
                    "display_name": str(registered.display_name),
                })
        if safe_candidates:
            projected["ambiguous_candidates"] = safe_candidates
    template_candidates = value.get("candidate_templates")
    if isinstance(template_candidates, list):
        try:
            from app.analysis.templates.registry import list_chat_visible_templates
            from app.context.semantic_catalog_v2 import DEFAULT_SEMANTIC_CATALOG

            visible = {
                str(item.get("name")): str(item.get("display_name") or item.get("name"))
                for item in list_chat_visible_templates()
                if item.get("name")
            }
            registered = {
                capability.template_id
                for capability in DEFAULT_SEMANTIC_CATALOG.capabilities
                if capability.template_id
            }
        except Exception:  # noqa: BLE001 - unavailable registries expose no choices
            visible, registered = {}, set()
        safe_templates: list[dict[str, str]] = []
        seen_templates: set[str] = set()
        for item in template_candidates[:8]:
            if not isinstance(item, dict):
                continue
            template_id = item.get("template_id")
            if (
                not isinstance(template_id, str)
                or not _CODE.fullmatch(template_id)
                or template_id not in registered
                or template_id not in visible
                or template_id in seen_templates
            ):
                continue
            seen_templates.add(template_id)
            safe_templates.append({
                "template_id": template_id,
                "display_name": visible[template_id],
            })
        if len(safe_templates) >= 2:
            projected["candidate_templates"] = safe_templates
    return projected


def safe_analysis_presentation(value: Any) -> dict[str, Any] | None:
    """Return only the declared result-card schema from a persisted turn snapshot."""
    if not isinstance(value, dict):
        return None
    try:
        from app.schemas.analysis_presentation import AnalysisPresentation

        return AnalysisPresentation.model_validate(value).model_dump(mode="json")
    except (ValidationError, TypeError, ValueError):
        return None


def safe_suspended_turn_snapshot(result: Any, answer: str | None) -> dict[str, Any]:
    """Build a minimal Turn snapshot for readback of a nonterminal pause."""
    result = result if isinstance(result, dict) else {}
    clarification = safe_clarification(
        result.get("clarification") or result.get("metric_planning_decision")
    )
    snapshot: dict[str, Any] = {
        "task_type": str(result.get("task_type") or "clarification")[:80],
        "answer": (answer or "")[:4000],
    }
    if clarification:
        snapshot["clarification"] = clarification
    reason_code = result.get("interaction_reason_code")
    if reason_code == "premise_direction_mismatch":
        snapshot["interaction_reason_code"] = reason_code
    return snapshot

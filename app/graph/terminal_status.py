"""Shared durable status classification for completed chat graph results."""

from __future__ import annotations

from typing import Any


_BUDGET_AND_REVIEW_ERRORS = {
    "token_budget_exceeded",
    "cost_budget_exceeded",
    "critic_review_incomplete",
}


def result_terminal_status(result: dict[str, Any], *, default_error_status: str = "abstained") -> str:
    """Classify graph outcome without claiming incomplete review as success.

    A validated partial presentation can be preserved as an abstention when
    review or a budget boundary stops the run. The same stop without a
    deliverable is a failure. Legacy errors retain their caller's existing
    default classification.
    """
    requested = str(result.get("_status") or "")
    if requested in {"suspended", "interrupted", "queued", "running"}:
        return requested
    if requested in {"failed", "abstained"}:
        return requested

    errors = result.get("errors") or []
    # SSE/lifecycle adapters may preserve only the stable error code instead
    # of the graph's full ``errors`` array. A non-empty code still represents
    # an error and must not fall through to the success default.
    if (result.get("error") or result.get("error_code")) and not errors:
        errors = [{"code": str(result.get("error_code") or "EXECUTION_ABSTAINED")}]
    codes = {
        str(error.get("code") or "")
        for error in errors if isinstance(error, dict)
    }
    # A completed negative review is stronger than the planner's capability
    # refusal. Otherwise an unsupported presentation could mask critic_rejected
    # and be persisted as a normal abstention.
    if "critic_rejected" in codes:
        return "failed"

    planning = result.get("metric_planning_decision") or result.get("clarification") or {}
    if isinstance(planning, dict) and planning.get("status") in {
        "clarification_required", "confirmation_required", "unsupported",
    }:
        return "abstained"
    presentation = result.get("analysis_presentation") or result.get("presentation") or {}
    if isinstance(presentation, dict) and presentation.get("status") == "unsupported":
        return "abstained"
    if not errors:
        return "completed"

    presentation_status = str(presentation.get("status") or "") if isinstance(presentation, dict) else ""
    if codes & _BUDGET_AND_REVIEW_ERRORS:
        if presentation_status in {"partial", "completed"}:
            return "abstained"
        return "failed"
    return default_error_status

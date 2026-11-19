"""Shared presentation status mapping for terminal diagnostic outcomes."""

from __future__ import annotations

from collections.abc import Mapping


_DIAGNOSTIC_TERMINAL_PRESENTATION_STATUS = {
    # The investigation completed and can explain why it cannot safely make a
    # numerical claim. This is a partial result, not an execution failure.
    "INVALID_DATA": "partial",
    "DATA_INSUFFICIENT": "partial",
    "BUDGET_EXHAUSTED": "partial",
    "EXECUTION_FAILED": "failed",
}


def diagnostic_terminal_presentation_status(analysis_result: object) -> str | None:
    """Return the presentation status implied by a diagnostic terminal code."""

    if not isinstance(analysis_result, Mapping):
        return None
    investigation = analysis_result.get("investigation_summary")
    if not isinstance(investigation, Mapping):
        return None
    stop_reason = investigation.get("stop_reason")
    if not isinstance(stop_reason, Mapping):
        return None
    code = str(stop_reason.get("code") or "").strip().upper()
    return _DIAGNOSTIC_TERMINAL_PRESENTATION_STATUS.get(code)

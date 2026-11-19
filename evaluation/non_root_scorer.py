"""Common result scorer for the five non-root evaluation subsets.

This scorer intentionally does not convert skipped environment checks into
passes and does not call a unit-test result a Provider accuracy result.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def score_case_outcomes(outcomes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize case-level outcomes without hiding skipped or failed cases."""

    items = list(outcomes)
    passed = sum(item.get("status") == "passed" for item in items)
    skipped = sum(item.get("status") == "skipped" for item in items)
    failed = sum(item.get("status") == "failed" for item in items)
    unknown = len(items) - passed - skipped - failed
    return {
        "total": len(items),
        "passed": passed,
        "skipped": skipped,
        "failed": failed,
        "unknown": unknown,
        "pass_rate": round(passed / len(items), 4) if items else 0.0,
        "acceptance_complete": bool(items) and failed == 0 and skipped == 0 and unknown == 0,
    }


def score_subset(*, planned: int, passed: int, skipped: int = 0, failed: int = 0) -> dict[str, Any]:
    """Summarize an already aggregated subset with explicit denominator."""

    accounted = passed + skipped + failed
    return {
        "planned": planned,
        "passed": passed,
        "skipped": skipped,
        "failed": failed,
        "unaccounted": max(planned - accounted, 0),
        "pass_rate": round(passed / planned, 4) if planned else 0.0,
        "acceptance_complete": planned > 0 and passed == planned and skipped == 0 and failed == 0,
    }


def summarize_diagnostic_runs(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate versioned root-cause run scores with explicit per-metric denominators.

    Unknown values are excluded from that metric's denominator and counted as unknown;
    raw counts (calls, duplicates, invalid calls, and tokens) remain separately visible.
    """
    rows = list(records)
    keys = [
        (str(row.get("case_id") or ""), str(row.get("arm") or ""), row.get("replicate"))
        for row in rows
    ]
    if any(not case_id or not arm or isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 1 for case_id, arm, replicate in keys):
        raise ValueError("Every diagnostic run needs case_id, arm and a positive integer replicate")
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate case × arm × replicate diagnostic run record")
    boolean_metrics = (
        "task_acceptance_pass",
        "effective_diagnosis_completed",
        "safe_stop_behavior_observed",
        "correct_safe_stop",
        "terminal_presentation_consistent",
        "conclusion_evidence_valid",
    )
    summaries: dict[str, Any] = {}
    for name in boolean_metrics:
        observed = [row.get(name) for row in rows if row.get(name) is not None]
        passed = sum(value is True for value in observed)
        summaries[name] = {
            "passed": passed,
            "denominator": len(observed),
            "unknown": len(rows) - len(observed),
            "rate": round(passed / len(observed), 4) if observed else None,
        }
    token_rows = [row.get("raw_metrics", {}) for row in rows]
    known_subtotals = [int(item["known_tokens_subtotal"]) for item in token_rows if isinstance(item, Mapping) and item.get("known_tokens_subtotal") is not None]
    total_tokens = [item.get("total_tokens") for item in token_rows if isinstance(item, Mapping)]
    unknown_total_count = sum(value is None for value in total_tokens)
    summary = {
        "scorer_version": next((row.get("scorer_version") for row in rows if row.get("scorer_version")), None),
        "run_count": len(rows),
        "metrics": summaries,
        "raw_counts": {
            key: sum(int(item.get(key, 0) or 0) for item in token_rows if isinstance(item, Mapping))
            for key in ("tool_call_count", "invalid_call_count", "duplicate_call_count", "unauthorized_call_count")
        },
        "tokens": {
            "known_subtotal": sum(known_subtotals),
            "known_subtotal_denominator": len(known_subtotals),
            "complete_total": sum(int(value) for value in total_tokens if value is not None) if rows and unknown_total_count == 0 else None,
            "complete_total_denominator": len(rows) if rows and unknown_total_count == 0 else 0,
            "unknown_usage_runs": unknown_total_count,
        },
    }
    return summary

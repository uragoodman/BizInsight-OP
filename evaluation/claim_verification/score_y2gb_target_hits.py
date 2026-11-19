"""Summarize independently adjudicated target-hit labels for Y2GB.

This deliberately does not infer a target hit from `record.verdict ==
issue_found`. The adjudication must identify the predeclared target issue in
the actual deterministic checks or Full review issues. This script only
aggregates a separate append-only adjudication JSONL and run result JSONL.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(results_path: Path, adjudications_path: Path) -> dict[str, Any]:
    results = _read(results_path)
    adjudications = _read(adjudications_path)
    result_keys = {(row.get("case_id"), row.get("arm")): row for row in results}
    annotations: dict[tuple[str, str], dict[str, Any]] = {}
    errors: list[str] = []
    for item in adjudications:
        key = (item.get("case_id"), item.get("arm"))
        if key in annotations:
            errors.append(f"duplicate adjudication: {key}")
        annotations[key] = item
        if item.get("target_hit") not in {True, False, None}:
            errors.append(f"{key}: target_hit must be true, false, or null")
        if not item.get("target_issue_id") or not item.get("basis"):
            errors.append(f"{key}: target_issue_id and basis are required")
        if item.get("review_status") not in {"adjudicated", "needs_manual_review"}:
            errors.append(f"{key}: invalid review_status")
    if set(annotations) != set(result_keys):
        missing = sorted(set(result_keys) - set(annotations))
        extra = sorted(set(annotations) - set(result_keys))
        errors.append(f"result/adjudication key mismatch; missing={missing}, extra={extra}")
    by_arm: dict[str, Counter[str]] = {}
    for key, result in result_keys.items():
        item = annotations.get(key)
        if not item:
            continue
        arm = str(key[1])
        counts = by_arm.setdefault(arm, Counter())
        if item.get("review_status") == "needs_manual_review" or item.get("target_hit") is None:
            counts["unknown_or_manual"] += 1
        else:
            counts["adjudicated"] += 1
            counts["target_hit"] += int(item["target_hit"] is True)
            counts["target_not_hit"] += int(item["target_hit"] is False)
        counts["review_incomplete_runs"] += int(result.get("verdict") == "review_incomplete")
        counts["provider_failures_or_errors"] += int(any(
            review.get("error_type") or not isinstance(review.get("result"), dict)
            for review in (result.get("reviews") or [])
        ))
    return {
        "valid": not errors,
        "arms": {arm: dict(counts) for arm, counts in sorted(by_arm.items())},
        "denominator_rule": "Only independently adjudicated, target-specific decisions count in hit-rate denominator; unknown/manual rows remain reported separately.",
        "verdict_is_not_target_hit": True,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--adjudications", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.results, args.adjudications)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

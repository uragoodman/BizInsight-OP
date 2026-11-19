"""Audit pilot input integrity; never send construction labels to reviewers."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from evaluation.claim_verification.annotation_schema import CLAIM_TYPES


def audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cases = []
    for row in rows:
        reasons = []
        answer = row.get("answer") or ""
        claims = row.get("atomic_claims") or []
        if not claims:
            reasons.append("missing_claims")
        for claim in claims:
            span = claim.get("text_span")
            if not span or span not in answer:
                reasons.append("claim_span_not_verbatim")
            elif answer.count(span) != 1:
                reasons.append("claim_span_not_unique")
            if claim.get("claim_type") is not None and claim.get("claim_type") not in CLAIM_TYPES:
                reasons.append("claim_type_invalid")
            if "start" in claim or "end" in claim:
                start, end = claim.get("start"), claim.get("end")
                if not isinstance(start, int) or not isinstance(end, int):
                    reasons.append("claim_span_offsets_missing")
                elif not (0 <= start < end <= len(answer)) or answer[start:end] != span:
                    reasons.append("claim_span_offsets_inconsistent")
        if row.get("provenance") == "targeted_mutation":
            reasons.append("mutation_claims_do_not_cover_source_answer")
        if any(c.get("correctness") == "injected_known_error" or c.get("expected_route") for c in claims):
            reasons.append("ground_truth_in_reviewer_input")
        if any((coverage.get("status") if isinstance(coverage, dict) else None) != "verified" for coverage in (row.get("requirement_coverage") or {}).values()):
            reasons.append("requirement_coverage_unverified")
        if row.get("human_verified") is not True or row.get("oracle_claim_labels") is not True:
            reasons.append("oracle_labels_unverified")
        cases.append({"pilot_id": row.get("pilot_id"), "reasons": reasons})
    counts = collections.Counter(reason for case in cases for reason in case["reasons"])
    return {"total": len(rows), "ready": sum(not case["reasons"] for case in cases), "issue_counts": dict(counts), "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Refusing to overwrite readiness report")
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = audit(rows)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("total", "ready", "issue_counts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

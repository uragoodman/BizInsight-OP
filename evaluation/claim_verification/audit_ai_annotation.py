"""Conservative, non-oracle checks on an AI annotation against fixed inputs."""
from __future__ import annotations

from typing import Any
from evaluation.claim_verification.deterministic_probe import probe


def audit(case: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    validation = record.get("validation") or {}
    checks = probe(case)["checks"]
    claims = validation.get("claims") or []
    issues = list(validation.get("issues") or [])
    if validation.get("status") != "structurally_valid":
        issues.append("invalid_structure")
    if validation.get("coverage_uncertain") is not False:
        issues.append("coverage_uncertain")
    for check in checks:
        if check.get("matched") is not False:
            continue
        for claim in claims:
            if (claim.get("correctness") == "supported" and isinstance(claim.get("start"), int)
                and isinstance(claim.get("end"), int) and claim["start"] < check["end"]
                and check["start"] < claim["end"]):
                issues.append("supported_claim_overlaps_deterministic_mismatch")
    if any(item.get("status") == "uncertain" for item in validation.get("requirements") or []):
        issues.append("requirement_uncertain")
    if any(item.get("correctness") == "uncertain" or item.get("type") == "unknown" for item in claims):
        issues.append("claim_uncertain")
    return {"case_id": case.get("case_id"), "issues": sorted(set(issues)), "numeric_checks": len(checks),
            "numeric_mismatches": sum(c.get("matched") is False for c in checks),
            "review_status": "needs_manual_or_ai_review", "human_verified": False, "oracle_claim_labels": False}


def main() -> None:
    import argparse
    import collections
    import hashlib
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Refusing to overwrite audit")
    inputs = [json.loads(line) for line in args.inputs.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [json.loads(line) for line in args.results.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = {row["case_id"]: row for row in inputs}
    if len(cases) != len(inputs) or len({row["case_id"] for row in records}) != len(records) or any(row["case_id"] not in cases for row in records):
        raise ValueError("Duplicate or unknown case id")
    rows = [audit(cases[row["case_id"]], row) for row in records]
    result = {"input_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
              "results_sha256": hashlib.sha256(args.results.read_bytes()).hexdigest(),
              "planned": len(inputs), "records_reviewed": len(records),
              "missing_or_failed": sorted(set(cases) - {row["case_id"] for row in records}),
              "issue_counts": dict(collections.Counter(issue for row in rows for issue in row["issues"])),
              "cases": rows, "independent_review": False}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("planned", "records_reviewed", "issue_counts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

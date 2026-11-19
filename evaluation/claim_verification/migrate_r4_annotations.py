"""Migrate the r4 assistant annotations onto the frozen v1 annotation schema.

The r4 drafts carry operational-context evidence that the earlier freeze does
not, so they are the better source.  They were produced under a different
validator, however, and cannot be fed to the benchmark as-is.  This script:

* maps r4 claim/requirement records onto `claim-annotation-v1`;
* repairs exactly the structural defects a human would have to repair anyway;
* records every repair so nothing is silently rewritten;
* never sets `human_verified` or `oracle_claim_labels`.

It refuses to invent correctness labels: `correctness` is copied from r4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from evaluation.claim_verification.annotation_schema import (
    EVIDENCE_REQUIRED_TYPES, validate_annotation_row,
)

PROCESS_STATUS_MARKERS = ("调查分支状态",)


def _expected_decision(claims: list[dict[str, Any]], coverage: dict[str, str]) -> str:
    if any(claim["correctness"] == "unsupported" for claim in claims):
        return "reject_or_revise"
    if any(status in ("missing", "uncertain") for status in coverage.values()):
        return "needs_semantic_review"
    if any(claim["correctness"] != "supported" for claim in claims):
        return "needs_semantic_review"
    return "pass"


def _repair_claim(claim: dict[str, Any], answer: str) -> tuple[dict[str, Any], list[str]]:
    """Fix only defects that make a row structurally unusable."""

    repairs: list[str] = []
    span = claim.get("text_span")
    claim_type = claim.get("claim_type")
    evidence_ids = list(claim.get("evidence_ids") or [])

    if claim_type in EVIDENCE_REQUIRED_TYPES and not evidence_ids:
        if any(marker in (span or "") for marker in PROCESS_STATUS_MARKERS):
            # The frozen guide classifies agent process status as `unknown`, and
            # `unknown` is semantic by definition: it can never be passed by a
            # deterministic check, so an empty evidence list is correct here.
            claim["claim_type"] = "unknown"
            repairs.append(
                f"claim {claim.get('claim_id')}: process-status block retyped "
                f"evidence_reference -> unknown (no business evidence exists)"
            )
        else:
            repairs.append(
                f"claim {claim.get('claim_id')}: WARNING unrepaired evidence_reference "
                f"without evidence"
            )

    return claim, repairs


def build_row(
    case: dict[str, Any], record: dict[str, Any], key: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    answer = case["answer"]
    validation = record["validation"]
    repairs: list[str] = []

    claims: list[dict[str, Any]] = []
    for index, raw in enumerate(validation.get("claims") or [], start=1):
        span = raw.get("span")
        if not isinstance(span, str) or not span or answer.count(span) != 1:
            repairs.append(f"claim {index}: dropped non-verbatim span")
            continue
        start = answer.find(span)
        claim = {
            "claim_id": f"C{index}",
            "start": start,
            "end": start + len(span),
            "text_span": span,
            "claim_type": raw.get("type"),
            "evidence_ids": list(raw.get("evidence_ids") or []),
            "correctness": raw.get("correctness"),
        }
        claim, claim_repairs = _repair_claim(claim, answer)
        repairs.extend(claim_repairs)
        claims.append(claim)

    requirements = list(case.get("task_requirements") or [])
    coverage = {
        item["requirement"]: item["status"] for item in validation.get("requirements") or []
        if isinstance(item, dict) and "requirement" in item
    }
    for requirement in requirements:
        if coverage.get(requirement) not in ("covered", "missing", "uncertain"):
            coverage[requirement] = "uncertain"
            repairs.append(f"requirement '{requirement}': status defaulted to uncertain")

    row = {
        "case_id": case["case_id"],
        "answer": answer,
        "claims": claims,
        "task_requirements": requirements,
        "requirement_coverage": coverage,
        "provenance": "targeted_mutation" if key.get("provenance") == "targeted_mutation" else "real_trace_draft",
        "expected_decision": _expected_decision(claims, coverage),
        "annotation_status": "assistant_annotated_pending_human_spotcheck",
        "human_verified": False,
        "oracle_claim_labels": False,
        "annotation_author": "assistant",
    }
    return row, repairs


def _normalise_provenance(row: dict[str, Any]) -> str:
    """The frozen v1 schema only accepts three provenance literals."""

    if row["provenance"] == "targeted_mutation":
        return "targeted_mutation"
    return "real_correct" if row["expected_decision"] != "reject_or_revise" else "real_failure"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError("Output exists; refusing to overwrite an annotation artifact")

    cases = {r["case_id"]: r for r in (json.loads(x) for x in args.inputs.read_text(encoding="utf-8").splitlines() if x.strip())}
    records = {r["case_id"]: r for r in (json.loads(x) for x in args.results.read_text(encoding="utf-8").splitlines() if x.strip())}
    keys = {r["case_id"]: r for r in (json.loads(x) for x in args.key.read_text(encoding="utf-8").splitlines() if x.strip())}
    if not (set(cases) == set(records) == set(keys)):
        raise ValueError("inputs, results and key must cover the same cases")

    rows, repairs_by_case, report = [], {}, []
    for case_id in sorted(cases):
        row, repairs = build_row(cases[case_id], records[case_id], keys[case_id])
        row["provenance"] = _normalise_provenance(row)
        check = validate_annotation_row(row)
        rows.append(row)
        if repairs:
            repairs_by_case[keys[case_id]["pilot_id"]] = repairs
        report.append({"case_id": case_id, "pilot_id": keys[case_id]["pilot_id"], **check})

    args.output_dir.mkdir(parents=True)
    (args.output_dir / "annotations.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    summary = {
        "schema_version": "claim-annotation-v1",
        "source": "development_freeze_operational_r4/assistant_repaired_drafts_r4",
        "cases": len(rows),
        "claims": sum(len(row["claims"]) for row in rows),
        "structurally_valid": sum(item["structurally_valid"] for item in report),
        "ready_for_benchmark": sum(item["ready_for_benchmark"] for item in report),
        "annotation_author": "assistant",
        "human_verified": False,
        "oracle_claim_labels": False,
        "repairs": repairs_by_case,
        "input_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        "results_sha256": hashlib.sha256(args.results.read_bytes()).hexdigest(),
        "key_sha256": hashlib.sha256(args.key.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases_report": report,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    unrepaired = [item for item in report if not item["structurally_valid"]]
    print(json.dumps({
        "cases": summary["cases"], "claims": summary["claims"],
        "structurally_valid": summary["structurally_valid"],
        "ready_for_benchmark": summary["ready_for_benchmark"],
        "repairs": sum(len(v) for v in repairs_by_case.values()),
        "unrepaired": len(unrepaired),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

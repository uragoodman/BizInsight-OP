"""Materialize a Y2GB reviewer corpus from independently qualified inputs.

This is a deterministic local file transformer only. It does not create facts,
answers, labels, evidence, or model outputs. The qualification file must carry
the exact reviewer-facing case fields and a separate truth audit. Scoring keys
are stripped before output and the deterministic probe must not already catch
the intended issue.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.claim_verification.deterministic_probe import probe
from evaluation.claim_verification.validate_y2gb_plan import FORBIDDEN_REVIEW_KEYS, _read_jsonl

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = ROOT / "docs/testing/claim_verification_pilot_20261027/y1gb_y2gb_20261028/Y2GB_CASE_PLAN.jsonl"
REVIEW_FIELDS = (
    "case_id", "user_query", "task_requirements", "request_contract",
    "evidence_bundles", "operational_context", "operational_context_provenance",
    "registry_evidence", "answer", "task_type",
)


def materialize(plan_path: Path, qualified_sources_path: Path, output_path: Path,
                scoring_key_path: Path | None = None) -> dict[str, Any]:
    if output_path.exists() or (scoring_key_path is not None and scoring_key_path.exists()):
        raise RuntimeError("Refusing to overwrite a Y2GB corpus, scoring key, or prior run input")
    plan = _read_jsonl(plan_path)
    source_rows = _read_jsonl(qualified_sources_path)
    plan_by_id = {row["case_id"]: row for row in plan}
    source_by_id = {row.get("case_id"): row for row in source_rows}
    if len(plan_by_id) != 23 or len(source_by_id) != len(source_rows):
        raise ValueError("plan must have 23 unique cases and qualification rows must have unique IDs")
    if set(source_by_id) != set(plan_by_id):
        raise ValueError("qualification ledger must cover all 23 planned case IDs exactly")

    output: list[dict[str, Any]] = []
    scoring_key: list[dict[str, Any]] = []
    for case_id, planned in plan_by_id.items():
        source = source_by_id[case_id]
        status = source.get("qualification_status")
        if status == "blocked":
            if not isinstance(source.get("blocker"), str) or not source["blocker"].strip():
                raise ValueError(f"{case_id}: blocked row needs a concrete blocker")
            continue
        if status != "qualified":
            raise ValueError(f"{case_id}: qualification status must be qualified or blocked")
        truth_audit = source.get("truth_audit")
        if not isinstance(truth_audit, dict) or truth_audit.get("verified") is not True:
            raise ValueError(f"{case_id}: missing verified independent truth audit")
        if truth_audit.get("target_issue_id") != case_id or not truth_audit.get("basis_ref"):
            raise ValueError(f"{case_id}: truth audit must identify target issue and evidence basis")
        reviewer_case = source.get("review_case")
        if not isinstance(reviewer_case, dict):
            raise ValueError(f"{case_id}: review_case must contain exact fixed answer and evidence")
        leaked = FORBIDDEN_REVIEW_KEYS & reviewer_case.keys()
        if leaked:
            raise ValueError(f"{case_id}: scoring labels leaked into reviewer input: {sorted(leaked)}")
        case = {key: reviewer_case[key] for key in REVIEW_FIELDS if key in reviewer_case}
        case["case_id"] = case_id
        if (case.get("request_contract") or {}).get("metric_id") != planned.get("metric_id"):
            raise ValueError(f"{case_id}: reviewer request contract metric differs from the candidate plan")
        if case.get("expected") not in (None, "should_be_flagged"):
            raise ValueError(f"{case_id}: unexpected evaluation label")
        case["expected"] = "should_be_flagged"
        if not case.get("answer") or not case.get("user_query"):
            raise ValueError(f"{case_id}: exact user query and fixed answer are required")
        trusted = [bundle for bundle in case.get("evidence_bundles", [])
                   if isinstance(bundle, dict) and bundle.get("trusted") is True]
        if not trusted and not case.get("operational_context"):
            raise ValueError(f"{case_id}: no trusted evidence or operational stop context")
        probe_result = probe(case)
        if probe_result.get("status") == "mismatch_found":
            raise ValueError(f"{case_id}: deterministic probe already detects a mismatch; not an LLM-incremental case")
        output.append(case)
        scoring_key.append({
            "case_id": case_id,
            "target_issue_id": truth_audit["target_issue_id"],
            "target_hit_rule": planned.get("target_hit_rule"),
            "planned_faulty_statement": planned.get("planned_faulty_statement"),
            "source_case_id": planned.get("source_case_id"),
            "source_run_id": planned.get("source_run_id"),
            "source_trace": planned.get("source_trace"),
            "source_trace_sha256": truth_audit.get("source_trace_sha256"),
            "original_answer_sha256": truth_audit.get("original_answer_sha256"),
            "faulty_answer_sha256": truth_audit.get("faulty_answer_sha256"),
            "basis_ref": truth_audit["basis_ref"],
            "expected": "should_be_flagged",
            "truth_scope": truth_audit.get("truth_scope"),
        })

    if not output:
        raise ValueError("no qualified Y2 cases; refusing to create an empty corpus")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output),
        encoding="utf-8",
    )
    if scoring_key_path is not None:
        scoring_key_path.parent.mkdir(parents=True, exist_ok=True)
        scoring_key_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in scoring_key),
            encoding="utf-8",
        )
    return {"written": len(output), "blocked": len(plan_by_id) - len(output),
            "path": str(output_path), "scoring_key_path": str(scoring_key_path) if scoring_key_path else None,
            "provider_calls": 0, "reviewer_calls": 0,
            "mode": "offline_materialization_after_truth_qualification"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--qualified-sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scoring-key-output", type=Path)
    args = parser.parse_args()
    result = materialize(args.plan, args.qualified_sources, args.output, args.scoring_key_output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

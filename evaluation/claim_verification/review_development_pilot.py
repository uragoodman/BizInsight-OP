"""Produce an AI-assisted, provenance-aware review ledger for the pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Review ledger already exists; do not overwrite a review artifact.")
    workspace = {
        str(row["case_id"]): row
        for row in (json.loads(line) for line in args.workspace.read_text(encoding="utf-8").splitlines() if line.strip())
    }
    pilot = [json.loads(line) for line in args.pilot.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(pilot) != 30 or len({row.get("pilot_id") for row in pilot}) != 30:
        raise RuntimeError("Pilot must contain exactly 30 uniquely identified rows.")
    ledger = []
    for row in pilot:
        case = workspace.get(str(row.get("source_case_id")))
        if case is None:
            raise RuntimeError(f"Missing frozen workspace source for {row.get('pilot_id')}.")
        provenance = row.get("provenance")
        if provenance == "real_trace_draft":
            decision, review_status = "pass", "ai_assisted_review_complete"
            basis = "Frozen source-case expected outcome, pre-Critic answer, trace evidence and task requirements were reviewed together."
            notes = []
            if row["source_case_id"] in {"FBG015", "FBG016"}:
                notes.append("Descriptive contribution is not treated as established business causality; retain semantic review for any causal extension.")
        elif row.get("mutation_type") == "numeric_fact_error":
            if row.get("original_text") not in case["original_answer_seen_by_critic"] or row.get("changed_text") not in row.get("answer", ""):
                raise RuntimeError(f"Numeric mutation does not preserve its declared source/change anchor: {row['pilot_id']}.")
            decision, review_status = "reject_or_revise", "ai_assisted_review_complete"
            basis = "Declared numeric anchor was replaced; the changed value conflicts with the frozen source answer/evidence."
            notes = []
        elif provenance == "targeted_mutation":
            if row.get("changed_text") not in row.get("answer", ""):
                raise RuntimeError(f"Semantic mutation text is absent: {row['pilot_id']}.")
            decision, review_status = "needs_semantic_review", "ai_assisted_review_complete"
            basis = "Injected causal, recommendation, data-insufficiency, or cross-claim statement cannot be deterministically accepted from the supplied evidence."
            notes = []
        else:
            raise RuntimeError(f"Unsupported pilot row: {row['pilot_id']}.")
        ledger.append({
            "pilot_id": row["pilot_id"], "source_case_id": row["source_case_id"],
            "provenance": provenance, "mutation_type": row.get("mutation_type"),
            "review_status": review_status, "reviewer_kind": "ai_assisted_delegated_review",
            "expected_decision": decision, "review_basis": basis, "review_notes": notes,
            "source_trace_id": row["source_trace_id"], "source_trace_sha256": case["source_trace_sha256"],
            "task_requirements_count": len(row.get("task_requirements") or []),
            "oracle_claim_labels": False,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ledger), encoding="utf-8")
    print(json.dumps({
        "reviewed": len(ledger), "pass": sum(row["expected_decision"] == "pass" for row in ledger),
        "reject_or_revise": sum(row["expected_decision"] == "reject_or_revise" for row in ledger),
        "needs_semantic_review": sum(row["expected_decision"] == "needs_semantic_review" for row in ledger),
        "review_ledger_sha256": _sha256(ledger), "human_verified": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

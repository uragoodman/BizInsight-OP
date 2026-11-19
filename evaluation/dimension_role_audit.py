#!/usr/bin/env python3
"""Run the P0 dimension-role audit over the frozen dataset.

The runner is offline and deterministic: it compiles each case with the real
compiler, records the current contract hash and the observe-only ``p0_audit``
block, and compares the 12 core hashes with the pre-change baseline.

Usage:
    python -m evaluation.dimension_role_audit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "evaluation" / "datasets" / "dimension_role_cases_v1.jsonl"
DEFAULT_BASELINE = ROOT / "tests" / "fixtures" / "p0_contract_hashes.json"
DEFAULT_OUT = ROOT / "data" / "quality" / "semantic-compiler-p0-latest.json"


def _load_cases(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _state_for(case: dict[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {
        "user_message": case["input"],
        "trace_id": f"p0-{case['case_id']}",
        # Skip the preference provider in the offline audit; the compiler
        # still treats this as an explicitly empty candidate set.
        "preference_context": {"source_needs": [], "candidate_analysis_plan": None},
    }
    if case.get("scope"):
        state["controlled_analysis_scope"] = case["scope"]
        state["use_current_explorer_context"] = True
        state["current_query_scope_fingerprint"] = "p0-scope-fingerprint"
    previous = (case.get("context") or {}).get("previous_turn_context")
    if previous:
        state["previous_turn_context"] = previous
    return state


def run(cases_path: Path, baseline_path: Path, out_path: Path) -> dict[str, Any]:
    from app.context.compiler import compile_turn_context

    rows = _load_cases(cases_path)
    results: list[dict[str, Any]] = []
    for case in rows:
        compiled = compile_turn_context(_state_for(case))["context_compilation"]
        contract = compiled.get("request_contract") or {}
        audit = compiled.get("p0_audit") or {}
        serialized_audit = json.dumps(audit, ensure_ascii=False)
        results.append(
            {
                "case_id": case["case_id"],
                "tier": case["tier"],
                "category": case["category"],
                "contract_hash": compiled.get("request_contract_hash"),
                "parse_status": contract.get("parse_status"),
                "metric_id": contract.get("metric_id"),
                "dimension": contract.get("dimension"),
                "operation_intent": contract.get("operation_intent"),
                "needs_clarification": compiled.get("needs_clarification"),
                "clarification_reason": compiled.get("clarification_reason"),
                "audit_present": bool(audit),
                "raw_question_in_audit": bool(case["input"])
                and case["input"] in serialized_audit,
                "p0_audit": audit,
            }
        )

    core = [item for item in results if item["tier"] == "core"]
    baseline_cases = (
        json.loads(baseline_path.read_text(encoding="utf-8")).get("cases", [])
        if baseline_path.exists()
        else []
    )
    baseline_by_id = {
        item["case_id"]: item.get("contract_hash") for item in baseline_cases
    }
    comparisons = []
    for item in core:
        baseline_hash = baseline_by_id.get(item["case_id"])
        comparisons.append(
            {
                "case_id": item["case_id"],
                "baseline_hash": baseline_hash,
                "current_hash": item["contract_hash"],
                "match": baseline_hash == item["contract_hash"],
            }
        )

    summary = {
        "version": "p0-1",
        "dataset": cases_path.name,
        "total": len(results),
        "core": len(core),
        "extended": len(results) - len(core),
        "audit_present": sum(1 for item in results if item["audit_present"]),
        "raw_question_in_audit": sum(
            1 for item in results if item["raw_question_in_audit"]
        ),
        "core_hash_matches": sum(1 for item in comparisons if item["match"]),
        "core_hash_total": len(comparisons),
        "core_hash_mismatches": [
            item["case_id"] for item in comparisons if not item["match"]
        ],
    }
    payload = {
        "summary": summary,
        "contract_hash_comparison": comparisons,
        "cases": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="P0 dimension-role audit")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    payload = run(args.cases, args.baseline, args.out)
    summary = payload["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"evidence={args.out}")
    return (
        0
        if not summary["core_hash_mismatches"] and summary["raw_question_in_audit"] == 0
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

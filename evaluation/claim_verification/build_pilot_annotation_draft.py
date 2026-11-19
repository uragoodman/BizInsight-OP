"""Merge real-answer drafts and declared mutations into a 30-row review fixture.

Mutation labels describe how the fixture was constructed.  They are useful for
measuring detector behavior in a development pilot, but they are not a sample
of production error prevalence and are never marked human/oracle verified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _mutation_claim(row: dict[str, Any], source: dict[str, Any] | None) -> dict[str, Any]:
    mutation_type = str(row.get("mutation_type") or "")
    changed = str(row.get("changed_text") or "")
    if mutation_type == "numeric_fact_error":
        if not source or str(row.get("original_text") or "") not in str(source.get("original_answer_seen_by_critic") or ""):
            raise RuntimeError(f"Numeric mutation lacks its original frozen anchor: {row.get('pilot_id')}")
        claim_type, route = "numeric_fact", "deterministic"
    elif mutation_type == "cross_claim_contradiction":
        claim_type, route = "unknown", "semantic"
    else:
        claim_type, route = "causal_claim", "semantic"
    return {
        "claim_id": "M1",
        "text_span": changed,
        "claim_type": claim_type,
        "evidence_ids": [],
        "correctness": "injected_known_error",
        "expected_route": route,
        "label_basis": "Declared mutation retained in frozen pilot answer; not a human/oracle annotation.",
        "annotation_confidence": "construction_derived_draft",
    }


def build(pilot: list[dict[str, Any]], real_drafts: list[dict[str, Any]], source_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    real_by_case = {str(row["case_id"]): row for row in real_drafts}
    source_by_case = {str(row["case_id"]): row for row in source_cases}
    output = []
    for row in pilot:
        case_id = str(row["source_case_id"])
        if row.get("provenance") == "real_trace_draft":
            real = real_by_case.get(case_id)
            if real is None:
                raise RuntimeError(f"Real pilot row lacks an annotation draft: {row['pilot_id']}")
            claims = real["atomic_claims"]
            requirements = real["requirement_coverage"]
            label_origin = "ai_assisted_real_answer_draft"
        else:
            source = source_by_case.get(case_id)
            claims = [_mutation_claim(row, source)]
            requirements = {
                requirement: {"status": "pending_independent_review", "linked_claim_ids": ["M1"]}
                for requirement in row.get("task_requirements") or []
            }
            label_origin = "construction_derived_mutation_draft"
        output.append({
            "pilot_id": row["pilot_id"],
            "source_case_id": case_id,
            "provenance": row["provenance"],
            "mutation_type": row.get("mutation_type"),
            "answer": row["answer"],
            "atomic_claims": claims,
            "requirement_coverage": requirements,
            "annotation_status": "ai_assisted_draft_pending_independent_review",
            "label_origin": label_origin,
            "human_verified": False,
            "oracle_claim_labels": False,
        })
    if len(output) != 30 or len({row["pilot_id"] for row in output}) != 30:
        raise RuntimeError("Expected 30 uniquely identified pilot annotations.")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--real-drafts", type=Path, required=True)
    parser.add_argument("--source-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Annotation artifact already exists; do not overwrite it.")
    result = build(_read(args.pilot), _read(args.real_drafts), _read(args.source_cases))
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in result), encoding="utf-8")
    print(json.dumps({
        "cases": len(result),
        "real_answer_drafts": sum(row["label_origin"] == "ai_assisted_real_answer_draft" for row in result),
        "construction_derived_mutations": sum(row["label_origin"] == "construction_derived_mutation_draft" for row in result),
        "claims": sum(len(row["atomic_claims"]) for row in result),
        "human_verified": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

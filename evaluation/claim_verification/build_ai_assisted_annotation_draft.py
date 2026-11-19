"""Create conservative, reviewable Claim drafts from frozen pre-Critic inputs.

The output intentionally remains an AI-assisted *draft*.  It is not an oracle
label set and must never be used to report evaluator accuracy without a later
independent annotation/review pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _claim(
    claim_id: str,
    text: str,
    claim_type: str,
    evidence_ids: list[str],
    rationale: str,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "text_span": text,
        "claim_type": claim_type,
        "evidence_ids": evidence_ids,
        "label_basis": rationale,
        "correctness": "pending_independent_review",
        "annotation_confidence": "ai_assisted_draft",
    }


def _first_nonempty_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def annotate_case(case: dict[str, Any]) -> dict[str, Any]:
    answer = str(case.get("original_answer_seen_by_critic") or "")
    evidence_ids = [str(item.get("evidence_id")) for item in case.get("evidence_bundles") or [] if item.get("evidence_id")]
    runtime_claims = case.get("runtime_claims_unverified") or []
    claims: list[dict[str, Any]] = []

    for index, runtime_claim in enumerate(runtime_claims, start=1):
        text = str(runtime_claim.get("value") or _first_nonempty_line(answer))
        raw_type = str(runtime_claim.get("claim_type") or "")
        # Runtime "analysis" is not sufficiently specific to be a semantic
        # claim; the request contract determines whether it is a comparison.
        if raw_type == "metric":
            claim_type = "numeric_fact"
            rationale = "Runtime metric claim has one or more evidence bundles."
        elif case.get("request_contract", {}).get("operation_intent") == "dimensional_statistical_contribution":
            claim_type = "comparison"
            rationale = "Contribution result is treated as descriptive comparison, not business causality."
        else:
            claim_type = "comparison"
            rationale = "Runtime analysis claim is tied to a comparison/trend evidence bundle."
        claims.append(_claim(f"C{index}", text, claim_type, evidence_ids, rationale))

    lowered = answer.lower()
    if not evidence_ids and ("未执行数据查询" in answer or "不支持" in answer or "未改用" in answer):
        claims.append(_claim(
            f"C{len(claims) + 1}",
            _first_nonempty_line(answer),
            "limitation_or_refusal",
            [],
            "Safe-stop wording: no evidence is asserted and no unsupported substitute was used.",
        ))
    if "不证明业务因果" in answer or "business causality remains unconfirmed" in lowered:
        span = "不证明业务因果" if "不证明业务因果" in answer else "Business causality remains unconfirmed."
        claims.append(_claim(
            f"C{len(claims) + 1}",
            span,
            "limitation_or_refusal",
            evidence_ids,
            "Explicit limitation: descriptive statistics are not business-causality evidence.",
        ))
    if not claims:
        claims.append(_claim(
            "C1", _first_nonempty_line(answer), "unknown", evidence_ids,
            "No reliable runtime claim structure was available; require semantic review.",
        ))

    requirement_coverage = {
        requirement: {
            "status": "pending_independent_review",
            "linked_claim_ids": [item["claim_id"] for item in claims],
        }
        for requirement in case.get("task_requirements") or []
    }
    return {
        **case,
        "annotation_status": "ai_assisted_draft_pending_independent_review",
        "atomic_claims": claims,
        "requirement_coverage": requirement_coverage,
        "oracle_claim_labels": False,
        "provenance_label": "ai_assisted_draft",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Annotation artifact already exists; do not overwrite it.")
    rows = [json.loads(line) for line in args.workspace.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [row for row in rows if row.get("case_id") != "FBG023"]
    if len(selected) != 10:
        raise RuntimeError(f"Expected 10 eligible real-answer pilot cases, found {len(selected)}.")
    annotated = [annotate_case(row) for row in selected]
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in annotated), encoding="utf-8")
    print(json.dumps({
        "cases": len(annotated),
        "claims": sum(len(row["atomic_claims"]) for row in annotated),
        "human_verified": False,
        "oracle_claim_labels": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

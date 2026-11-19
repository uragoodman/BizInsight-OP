# -*- coding: utf-8 -*-
"""Score E3-real answers with deterministic fact/evidence keypoints.

Fact recall uses per-case aliases and optional directional regex patterns.
Evidence recall checks required evidence IDs; evidence precision checks that
every cited evidence ID exists in the context actually sent to the provider.
It does not judge whether a citation semantically supports the answer.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from app.context.packer import (
    ContextBudget,
    ContextItem,
    ContextSource,
    pack_context,
)


def _items(case: dict[str, Any]) -> list[ContextItem]:
    return [
        ContextItem(
            item_id=str(row["item_id"]),
            source=ContextSource(str(row["source"])),
            content=str(row["content"]),
            priority=int(row.get("priority", 50)),
            required=bool(row.get("required")),
            evidence_ids=tuple(str(value) for value in (row.get("evidence_ids") or [])),
        )
        for row in case.get("items") or []
    ]


def _available_evidence(case: dict[str, Any]) -> dict[str, set[str]]:
    items = _items(case)
    budget = ContextBudget(
        agent=str(case["agent"]),
        model=str(case.get("model") or "test-model"),
        **case["budget"],
    )
    included_ids = set(pack_context(items, budget).included_item_ids)
    return {
        "raw": {
            evidence_id
            for item in items
            for evidence_id in item.evidence_ids
        },
        "packed": {
            evidence_id
            for item in items
            if item.item_id in included_ids
            for evidence_id in item.evidence_ids
        },
    }


def _load_cases(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["case_id"]: row for row in rows}


def _matches(answer: str, fact: dict[str, Any]) -> bool:
    folded = answer.casefold()
    aliases = [str(item).casefold() for item in (fact.get("aliases") or [])]
    if any(alias in folded for alias in aliases):
        return True
    return any(
        re.search(str(pattern), answer, re.IGNORECASE | re.DOTALL)
        for pattern in (fact.get("patterns") or [])
    )


def _evidence_ids(answer: str) -> set[str]:
    match = re.search(r"EVIDENCE_IDS\s*:\s*(.+)", answer, re.IGNORECASE)
    if not match:
        return set()
    return {
        token.strip().strip(".。")
        for token in re.split(r"[,，\s]+", match.group(1))
        if token.strip().strip(".。")
    }


def _score(case: dict[str, Any], answer: str, available_evidence: set[str]) -> dict[str, Any]:
    facts = case.get("expected_facts") or []
    covered = [fact["id"] for fact in facts if _matches(answer, fact)]
    required_evidence = {
        str(value)
        for item in case.get("items") or []
        if item.get("required")
        for value in item.get("evidence_ids") or []
    }
    found_evidence = _evidence_ids(answer)
    valid_citations = found_evidence & available_evidence
    return {
        "fact_total": len(facts),
        "fact_covered": len(covered),
        "fact_recall": len(covered) / len(facts) if facts else None,
        "covered_fact_ids": covered,
        "required_evidence_ids": sorted(required_evidence),
        "found_evidence_ids": sorted(found_evidence),
        "available_evidence_ids": sorted(available_evidence),
        "valid_citation_ids": sorted(valid_citations),
        "unsupported_evidence_ids": sorted(found_evidence - available_evidence),
        "evidence_recall": (
            len(required_evidence & found_evidence) / len(required_evidence)
            if required_evidence else None
        ),
        "evidence_precision": (
            len(valid_citations) / len(found_evidence) if found_evidence else None
        ),
        "answer_nonempty": bool(answer.strip()),
    }


def _aggregate(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    facts = sum(row[mode]["fact_total"] for row in rows)
    covered = sum(row[mode]["fact_covered"] for row in rows)
    evidence = sum(len(row[mode]["required_evidence_ids"]) for row in rows)
    evidence_hit = sum(
        len(set(row[mode]["required_evidence_ids"]) & set(row[mode]["found_evidence_ids"]))
        for row in rows
    )
    cited = sum(len(row[mode]["found_evidence_ids"]) for row in rows)
    valid = sum(
        len(set(row[mode]["found_evidence_ids"]) & set(row[mode]["available_evidence_ids"]))
        for row in rows
    )
    return {
        "cases": len(rows),
        "fact_total": facts,
        "fact_covered": covered,
        "fact_recall": covered / facts if facts else None,
        "required_evidence": evidence,
        "evidence_retained": evidence_hit,
        "evidence_recall": evidence_hit / evidence if evidence else None,
        "cited_evidence": cited,
        "valid_citations": valid,
        "evidence_precision": valid / cited if cited else None,
        "unsupported_citations": sum(
            len(row[mode]["unsupported_evidence_ids"]) for row in rows
        ),
        "answer_nonempty": sum(int(row[mode]["answer_nonempty"]) for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="evaluation/datasets/e3_context_cases_v1.jsonl")
    parser.add_argument("--raw-result", default="docs/testing/metric_seed_20261102/e3_real_result.json")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()
    cases = _load_cases(Path(args.cases))
    source = json.loads(Path(args.raw_result).read_text(encoding="utf-8"))
    rows = []
    for result in source.get("results") or []:
        case = cases[result["case_id"]]
        available = _available_evidence(case)
        rows.append({
            "case_id": result["case_id"],
            "difficulty": result["difficulty"],
            "question": result["question"],
            "expected_fact_ids": [fact["id"] for fact in case.get("expected_facts") or []],
            "raw": _score(case, result["raw"]["answer"], available["raw"]),
            "packed": _score(case, result["packed"]["answer"], available["packed"]),
            "input_token_reduction": result.get("input_token_reduction"),
        })
    report = {
        "schema_version": "e3-fact-score-v1",
        "cases": len(rows),
        "raw": _aggregate(rows, "raw"),
        "packed": _aggregate(rows, "packed"),
        "by_difficulty": {
            difficulty: {
                "raw": _aggregate([row for row in rows if row["difficulty"] == difficulty], "raw"),
                "packed": _aggregate([row for row in rows if row["difficulty"] == difficulty], "packed"),
            }
            for difficulty in ("easy", "medium", "hard")
        },
        "details": rows,
        "limits": [
            "Deterministic alias/keypoint matching, not an LLM judge.",
            f"{len(rows)}-case real-provider sample; not a production generalization claim.",
            "Facts are derived from the supplied context and frozen per case.",
            "Evidence precision only checks that cited IDs exist in the context actually sent; it does not verify semantic support.",
        ],
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.json_out:
        target = Path(args.json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output + "\n", encoding="utf-8")
    complete = all(
        row[mode]["answer_nonempty"] for row in rows for mode in ("raw", "packed")
    ) and all(
        report[mode]["fact_recall"] is not None
        and report[mode]["evidence_recall"] is not None
        and report[mode]["evidence_precision"] is not None
        for mode in ("raw", "packed")
    )
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())

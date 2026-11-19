# -*- coding: utf-8 -*-
"""Deterministic E3 context packing evaluation.

Compares all available context items (raw baseline) with the governed Packer
output. It measures token reduction, required item retention, required evidence
retention and optional omission. It is deterministic and does not call an LLM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.context.packer import (
    ContextBudget,
    ContextItem,
    ContextSource,
    pack_context,
    token_counting_method,
)


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _eval_case(case: dict[str, Any]) -> dict[str, Any]:
    items = [
        ContextItem(
            item_id=str(row["item_id"]),
            source=ContextSource(str(row["source"])),
            content=str(row["content"]),
            priority=int(row.get("priority", 50)),
            required=bool(row.get("required")),
            evidence_ids=tuple(str(value) for value in (row.get("evidence_ids") or [])),
        )
        for row in case["items"]
    ]
    budget = ContextBudget(
        agent=str(case["agent"]), model=str(case.get("model") or "test-model"),
        **case["budget"],
    )
    packed = pack_context(items, budget)
    raw_budget = ContextBudget(
        agent=str(case["agent"]), model=str(case.get("model") or "test-model"),
        **{**case["budget"], "available_context_tokens": 1_000_000},
    )
    raw_tokens = pack_context(items, raw_budget).estimated_tokens
    packed_tokens = packed.estimated_tokens
    included_ids = set(packed.included_item_ids)
    included_evidence = {
        evidence_id
        for item in items
        if item.item_id in included_ids
        for evidence_id in item.evidence_ids
    }
    required_ids = set(case.get("expected_required_item_ids") or [])
    optional_ids = set(case.get("expected_optional_item_ids") or [])
    required_evidence = {
        evidence_id
        for item in items
        if item.item_id in required_ids
        for evidence_id in item.evidence_ids
    }
    return {
        "case_id": case["case_id"],
        "difficulty": case["difficulty"],
        "agent": case["agent"],
        "raw_tokens": raw_tokens,
        "packed_tokens": packed_tokens,
        "compression": 1 - packed_tokens / raw_tokens if raw_tokens else 0.0,
        "included_item_ids": sorted(included_ids),
        "included_evidence_ids": sorted(included_evidence),
        "omitted_item_ids": sorted(
            str(row.get("item_id")) for row in packed.omitted if row.get("item_id")
        ),
        "required_item_ids": sorted(required_ids),
        "required_retention": (
            len(required_ids & included_ids) / len(required_ids) if required_ids else None
        ),
        "required_evidence_ids": sorted(required_evidence),
        "evidence_retention": (
            len(required_evidence & included_evidence) / len(required_evidence)
            if required_evidence else None
        ),
        "optional_ids": sorted(optional_ids),
        "optional_omitted": len(optional_ids - included_ids),
        "optional_omission_rate": (
            len(optional_ids - included_ids) / len(optional_ids) if optional_ids else None
        ),
        "over_budget_required": packed.over_budget_required,
    }


def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw = sum(row["raw_tokens"] for row in rows)
    packed = sum(row["packed_tokens"] for row in rows)
    required = sum(len(row["required_item_ids"]) for row in rows)
    required_hit = sum(
        len(set(row["required_item_ids"]) & set(row["included_item_ids"])) for row in rows
    )
    evidence = sum(len(row["required_evidence_ids"]) for row in rows)
    evidence_hit = sum(
        len(set(row["required_evidence_ids"]) & set(row["included_evidence_ids"]))
        for row in rows
    )
    return {
        "cases": len(rows),
        "raw_tokens": raw,
        "packed_tokens": packed,
        "compression": 1 - packed / raw if raw else 0.0,
        "required_items": required,
        "required_items_retained": required_hit,
        "required_retention": required_hit / required if required else None,
        "required_evidence": evidence,
        "required_evidence_retained": evidence_hit,
        "required_evidence_retention": evidence_hit / evidence if evidence else None,
        "required_overflow_cases": sum(1 for row in rows if row["over_budget_required"]),
        "optional_items": sum(len(row["optional_ids"]) for row in rows),
        "optional_omitted": sum(row["optional_omitted"] for row in rows),
        "optional_omission_rate": (
            sum(row["optional_omitted"] for row in rows)
            / sum(len(row["optional_ids"]) for row in rows)
            if sum(len(row["optional_ids"]) for row in rows) else None
        ),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _agg(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="evaluation/datasets/e3_context_cases_v1.jsonl")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()
    cases = _load(Path(args.cases))
    rows = [_eval_case(case) for case in cases]
    by_difficulty = {
        name: _summary([row for row in rows if row["difficulty"] == name])
        for name in ("easy", "medium", "hard")
    }
    report = {
        "schema_version": "e3-context-packing-v1",
        "dataset_version": "e3-context-cases-v1",
        "token_counting_method": token_counting_method("test-model"),
        "cases": len(rows),
        "by_difficulty": by_difficulty,
        "overall": _summary(rows),
        "details": rows,
        "limits": [
            "Deterministic Packer evaluation; no LLM call was made.",
            "raw_tokens is the sum of all candidate items; packed_tokens is the Packer estimate.",
            "required retention is the primary gate; optional omission is reported separately.",
        ],
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.json_out:
        target = Path(args.json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output + "\n", encoding="utf-8")
    return 0 if all(
        (summary["required_retention"] in (None, 1.0))
        and not summary["required_overflow_cases"]
        for summary in by_difficulty.values()
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())

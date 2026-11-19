"""Create a traceable review queue from blind inputs and partial numeric checks.

Hints identify text to inspect; they are not gold labels or routing decisions.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


RISK_HINTS = {
    "causal_language": re.compile(r"原因|根因|导致|因为|证明|caus(?:e|al)", re.I),
    "recommendation_language": re.compile(r"建议|应当|应该|应立即|应停止|recommend", re.I),
    "limitation_language": re.compile(r"未确认|不证明|无法|不支持|未执行|缺少|unconfirmed|not.*caus", re.I),
}


def create(case: dict[str, Any], numeric: dict[str, Any]) -> dict[str, Any]:
    if case.get("case_id") != numeric.get("case_id"):
        raise ValueError("case id mismatch")
    answer = case["answer"]
    checks = numeric.get("checks") or []
    candidates = []
    for span in case.get("candidate_spans") or []:
        start, end = span["start"], span["end"]
        if not (0 <= start <= end <= len(answer)) or answer[start:end] != span["text"]:
            raise ValueError("span offsets do not match answer")
        hints = [name for name, pattern in RISK_HINTS.items() if pattern.search(span["text"])]
        overlapping = []
        for check in checks:
            if check["start"] < end and start < check["end"]:
                overlapping.append({key: check[key] for key in ("start", "end", "evidence_id", "evidence_field", "matched")})
        candidates.append({
            "claim_id": span["span_id"], "start": start, "end": end, "text_span": span["text"],
            "hints": hints, "numeric_checks": overlapping,
            "claim_type": "unknown", "evidence_ids": [], "correctness": "uncertain",
            "review_status": "pending",
        })
    if "".join(item["text_span"] for item in candidates) != answer:
        raise ValueError("candidate spans do not cover answer")
    return {
        "case_id": case["case_id"], "candidates": candidates,
        "task_requirements": [{"requirement": req, "status": "unresolved", "evidence": []} for req in case.get("task_requirements") or []],
        "annotation_status": "pending", "human_verified": False, "oracle_claim_labels": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--numeric", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Refusing to overwrite review queue")
    cases = [json.loads(x) for x in args.inputs.read_text(encoding="utf-8").splitlines() if x.strip()]
    records = [json.loads(x) for x in args.numeric.read_text(encoding="utf-8").splitlines() if x.strip()]
    index = {row["case_id"]: row for row in records}
    if len(index) != len(records) or len(index) != len(cases) or {row["case_id"] for row in cases} != set(index):
        raise ValueError("input and numeric records do not match one-to-one")
    output = [create(case, index[case["case_id"]]) for case in cases]
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output), encoding="utf-8")
    print(json.dumps({"cases": len(output), "candidate_spans": sum(len(x["candidates"]) for x in output), "numeric_linked_spans": sum(bool(c["numeric_checks"]) for x in output for c in x["candidates"]), "risk_hint_spans": sum(bool(c["hints"]) for x in output for c in x["candidates"]), "verified_cases": 0}))


if __name__ == "__main__":
    main()

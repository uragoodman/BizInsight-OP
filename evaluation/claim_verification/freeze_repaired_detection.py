"""Freeze the prior development cases with four repaired omission variants.

The query explicitly asks for units. Numerical values remain unchanged, while
all answer-side unit expressions are removed. This is an injected defect, not
a naturally occurring production failure. No provider calls occur here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/testing/claim_verification_pilot_20261027/unified_benchmark/hard_variants_v5.jsonl"
REQUIREMENT = "必须明确说明各数值及变化率的单位。"


def repair(case: dict) -> dict:
    result = dict(case)
    result["case_id"] = case["case_id"] + "-unit-required"
    result["user_query"] = case["user_query"].rstrip("。") + "；请明确说明数值、差值及变化率的单位。"
    result["task_requirements"] = [*(case.get("task_requirements") or []), REQUIREMENT]
    answer = case["answer"]
    # Preserve the user's original quoted question in the answer, but remove
    # explicit numerical units from the answer proper only.
    answer = re.sub(r"(?<=\d)\s*(?:雷亚尔|百分点|人|%|％)", "", answer)
    if re.search(r"(?<=\d)\s*(?:雷亚尔|百分点|人|%|％)", answer):
        raise ValueError(f"unit still present: {case['case_id']}")
    if answer == case["answer"] and case["case_id"] not in {"H4-FBG046", "H4-FBG052"}:
        raise ValueError(f"expected unit removal: {case['case_id']}")
    result["answer"] = answer
    result["mutation"] = "显式要求说明数值及变化率单位，答案缺失该说明"
    result["error_family"] = "explicit_unit_requirement_omission"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite frozen data")
    cases = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(cases) != 20 or len({c["case_id"] for c in cases}) != 20:
        raise ValueError("source must have 20 unique cases")
    repaired = [repair(c) if c["case_id"].startswith("H4-") else c for c in cases]
    if sum(c["error_family"] == "explicit_unit_requirement_omission" for c in repaired) != 4:
        raise ValueError("expected exactly four omission cases")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        for case in repaired:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"count": len(repaired), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Run the deterministic Off-path qualification over a materialized Y2 corpus.

This calls fixed_review.review(..., arm='off'), whose implementation does not
invoke any LLM review function. It is a qualification check, not a Y2 outcome.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.claim_verification.fixed_review import review
from evaluation.claim_verification.validate_y2gb_plan import _read_jsonl


def preflight(corpus_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise RuntimeError("Refusing to overwrite a prior Y2 preflight record")
    cases = _read_jsonl(corpus_path)
    if not cases or len({row.get("case_id") for row in cases}) != len(cases):
        raise ValueError("corpus must contain unique, non-empty candidate rows")
    rows = []
    deterministic_statuses: Counter[str] = Counter()
    verdicts: Counter[str] = Counter()
    for case in cases:
        result = review(case, "off", evaluation_run_id=f"y2-preflight-{case['case_id']}")
        deterministic_status = (result.get("deterministic") or {}).get("status", "unknown")
        deterministic_statuses[deterministic_status] += 1
        verdicts[result.get("verdict", "unknown")] += 1
        summary = result.get("llm_token_summary") or {}
        if result.get("llm_calls") or summary.get("sdk_attempt_count", 0) != 0:
            raise RuntimeError(f"{case['case_id']}: Off preflight unexpectedly made an LLM/SDK call")
        rows.append({
            "case_id": case["case_id"], "arm": "off",
            "qualification_only": True,
            "deterministic_probe": result.get("deterministic"),
            "review_verdict": result.get("verdict"),
            "llm_call_count": 0, "sdk_attempt_count": 0,
            "request_status": summary.get("request_status", "no_model_request"),
        })
    if deterministic_statuses.get("mismatch_found", 0):
        raise RuntimeError("deterministic Off probe already detects at least one seeded target error")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return {"valid": True, "cases": len(cases), "deterministic_statuses": dict(sorted(deterministic_statuses.items())),
            "off_review_verdicts": dict(sorted(verdicts.items())), "llm_calls": 0,
            "sdk_attempts": 0, "provider_calls": 0, "qualification_only": True,
            "output": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(preflight(args.corpus, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Produce a read-only review table for the 37 real-answer cases.

Writes one row per case with everything a human needs to adjudicate, and
assigns **no** verdicts.  The only judgement field is `dispute_status`, which
starts as `pending` for every row.

Fields per row:
  case_id, cohort, answer_origin, answer, user_query, task_requirements,
  evidence_summary, registry_unit, critic_verdict, verdict_applies_to,
  cited_numbers_supported (mechanical), unit_claim_source (mechanical),
  dimension_claim_supported (mechanical), notes, dispute_status
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from evaluation.claim_verification.annotate_assisted import _explainable, _numbers

NUMBER = re.compile(r"[+\-−]?\d[\d,]*(?:\.\d+)?")


def _payload_of(bundle: dict[str, Any]) -> dict[str, Any]:
    result = (bundle.get("content") or {}).get("result") or {}
    for key in ("contribution_result", "compare_result", "metric_result", "trend_result"):
        payload = result.get(key)
        if isinstance(payload, dict) and payload.get("status") == "ok":
            return payload
    return {}


def _evidence_summary(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    out = {"bundles": len(bundles), "items": []}
    for bundle in bundles:
        result = (bundle.get("content") or {}).get("result") or {}
        parsed = result.get("parsed") or {}
        payload = _payload_of(bundle)
        out["items"].append({
            "evidence_id": bundle.get("evidence_id"),
            "status": result.get("status"),
            "metric_name": parsed.get("metric_name") or payload.get("metric_name"),
            "query_type": parsed.get("query_type"),
            "dimension": parsed.get("dimension"),
            "filters": parsed.get("filters"),
            "unit_field": payload.get("unit"),
            "source_table": payload.get("source_table"),
            "values": {k: payload.get(k) for k in
                       ("baseline_value", "current_value", "delta_value", "delta_ratio", "value", "total_delta", "metric_date")
                       if payload.get(k) is not None},
        })
    return out


def _cited_numbers_supported(answer: str, bundles: list[dict[str, Any]]) -> dict[str, Any]:
    """Which numbers in the answer can be matched to evidence values."""

    numbers = _numbers(answer)
    known: set[float] = set()
    for bundle in bundles:
        values, _unit = _explainable(bundle)
        known |= values
    if not numbers:
        return {"numbers": [], "unsupported": [], "status": "no_numbers"}
    magnitudes = {abs(v) for v in known}
    unsupported = [v for v in numbers if abs(v) not in magnitudes]
    return {"total": len(numbers), "unsupported": sorted(set(unsupported)),
            "status": "all_supported" if not unsupported else "some_unmatched"}


def _unit_claim_source(answer: str, bundles: list[dict[str, Any]], registry: dict[str, Any] | None) -> dict[str, Any]:
    """Where could any unit statement in the answer come from?"""

    mentions = [s.strip() for s in re.split(r"[。\n]", answer) if "单位" in s]
    if not mentions:
        return {"mentions": [], "query_payload_unit": None, "registry_unit": (registry or {}).get("unit"),
                "traceable_source": "none_needed"}
    payload_units = [_payload_of(b).get("unit") for b in bundles]
    payload_units = [u for u in payload_units if u]
    registry_unit = (registry or {}).get("unit")
    if payload_units:
        source = "query_result"
    elif registry_unit:
        source = "registry"
    else:
        source = "no_recorded_source"
    return {"mentions": mentions, "query_payload_unit": payload_units or None,
            "registry_unit": registry_unit, "traceable_source": source}


def _dimension_claim_supported(answer: str, bundles: list[dict[str, Any]]) -> dict[str, Any]:
    claims = bool(re.search(r"维度[^\n]{0,20}证据(?:显示|表明)", answer))
    if not claims:
        return {"answer_claims_evidenced_dimension": False, "status": "not_applicable"}
    dims = [((b.get("content") or {}).get("result") or {}).get("parsed", {}).get("dimension") for b in bundles]
    return {"answer_claims_evidenced_dimension": True, "parsed_dimensions": dims,
            "status": "supported" if any(d for d in dims) else "parsed_dimension_empty"}


def build_row(case: dict[str, Any]) -> dict[str, Any]:
    answer = case["answer"]
    bundles = case.get("evidence_bundles") or []
    registry = case.get("registry_evidence")
    origin = case.get("answer_origin", "unknown")
    return {
        "case_id": case["case_id"],
        "cohort": case["cohort"],
        "answer_origin": origin,
        "user_query": case.get("user_query"),
        "task_requirements": case.get("task_requirements"),
        "task_type": case.get("task_type"),
        "critic_verdict": case.get("critic_verdict"),
        "verdict_applies_to": case.get("verdict_applies_to"),
        "answer": answer,
        "evidence_summary": _evidence_summary(bundles),
        "registry_evidence": registry,
        "mechanical": {
            "cited_numbers": _cited_numbers_supported(answer, bundles),
            "unit_claim": _unit_claim_source(answer, bundles, registry),
            "dimension_claim": _dimension_claim_supported(answer, bundles),
        },
        "notes": "",
        "dispute_status": "pending",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Refusing to overwrite a review table")

    cases = [json.loads(x) for x in args.corpus.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [build_row(case) for case in cases]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
    )
    from collections import Counter
    print(json.dumps({
        "rows": len(rows),
        "cohort": dict(Counter(r["cohort"] for r in rows)),
        "answer_origin": dict(Counter(r["answer_origin"] for r in rows)),
        "critic_verdict": dict(Counter(str(r["critic_verdict"]) for r in rows)),
        "dispute_status": dict(Counter(r["dispute_status"] for r in rows)),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

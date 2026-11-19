"""Conservative evidence-based numeric probe for fixed-answer development cases.

This is a partial verifier: unchecked prose never becomes a pass. The probe
accepts reviewer inputs only and never reads the mutation/ground-truth key.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import re
from pathlib import Path
from typing import Any

NUMBER = r"[+\-−]?\d[\d,]*(?:\.\d+)?"
TOTAL = re.compile(rf"总变化\s*({NUMBER})")
TABLE = re.compile(rf"\|\s*(基线日|当前日)\s*\|\s*(\d{{4}}-\d{{2}}-\d{{2}})\s*\|\s*({NUMBER})")
DATE_VALUE = re.compile(rf"(\d{{4}}-\d{{2}}-\d{{2}})(?:\*\*)?\s*(?:的(?:数值)?(?:为)?|[=：:])\s*(?:\*\*)?({NUMBER})")
ROLE_LINE = re.compile(rf"(?:^|\n)\s*(?:-\s*)?(基线日|当前日)\s+(\d{{4}}-\d{{2}}-\d{{2}})\s*[：:]\s*({NUMBER})")
_PERCENT_DELTA_CLAIM = re.compile(
    rf"(?P<label>绝对差值|绝对变化(?:值)?|变化值|数值贡献|贡献值|贡献)(?!率)"
    rf"(?:的)?\s*(?:单位\s*(?:为|是|[:：=])\s*|(?:为|是|[:：=])?\s*[+\-−]?{NUMBER}\s*)"
    r"(?P<unit>%|％|百分比|百分数)"
)


def _num(value: str) -> float:
    return float(value.replace(",", "").replace("−", "-"))


def _authorized_comparison_units(case: dict[str, Any], metric: str | None) -> tuple[str, str, str, str] | None:
    """Return units only when registry metadata is tied to trusted evidence.

    At least one successful trusted result for the requested metric must carry
    a single authorization catalog digest. Units then come from the current
    registry contract; saved free text or unbound unit labels are not enough.
    """
    if not metric:
        return None
    registry_evidence = case.get("registry_evidence")
    matching_evidence_ids: list[str] = []
    authorized_digests: set[str] = set()
    for bundle in case.get("evidence_bundles") or []:
        if not isinstance(bundle, dict) or bundle.get("trusted") is not True:
            continue
        content = bundle.get("content") or {}
        if not isinstance(content, dict):
            continue
        result = content.get("result") or {}
        if not isinstance(result, dict):
            continue
        parsed = result.get("parsed") or {}
        if not isinstance(parsed, dict):
            continue
        authorized = parsed.get("authorized_metric") or {}
        if not isinstance(authorized, dict):
            continue
        payload = (result.get("compare_result") or result.get("contribution_result")
                   or result.get("metric_result") or result.get("trend_result"))
        if (content.get("metric_id") == metric
                and result.get("status") == "ok"
                and isinstance(payload, dict) and payload.get("status") == "ok"
                and payload.get("metric_name") == metric
                and parsed.get("metric_name") == metric
                and authorized.get("metric_id") == metric
                and isinstance(authorized.get("catalog_digest"), str)
                and authorized.get("catalog_digest")):
            authorized_digests.add(authorized["catalog_digest"])
            evidence_id = bundle.get("evidence_id")
            if isinstance(evidence_id, str) and evidence_id:
                matching_evidence_ids.append(evidence_id)
    if not matching_evidence_ids or len(authorized_digests) != 1:
        return None

    # Reuse the project's registry-backed comparison contract. Unknown metrics
    # or unavailable registry metadata remain unchecked instead of guessed.
    from app.metrics.comparison_units import comparison_unit_contract
    from app.metrics.registry import get_registry

    try:
        registry = get_registry()
    except Exception:
        return None
    if authorized_digests != {registry.catalog_digest}:
        return None
    registered_metric_unit, change_unit, relative_unit = comparison_unit_contract(metric)
    if not registered_metric_unit or not change_unit or not relative_unit:
        return None
    if isinstance(registry_evidence, dict):
        if (registry_evidence.get("source") != "authorized_registry_catalog"
                or registry_evidence.get("role") != "metric_definition_not_business_numeric_evidence"
                or registry_evidence.get("metric_id") != metric
                or registry_evidence.get("catalog_digest") not in authorized_digests
                or registry_evidence.get("unit") != registered_metric_unit):
            return None
    return registered_metric_unit, change_unit, relative_unit, matching_evidence_ids[0]


def probe(case: dict[str, Any]) -> dict[str, Any]:
    answer = str(case.get("answer") or "")
    metric = (case.get("request_contract") or {}).get("metric_id")
    records = []
    date_records = []
    bundles = case.get("evidence_bundles") or []
    counts = Counter(b.get("evidence_id") for b in bundles if isinstance(b, dict) and b.get("evidence_id"))
    for bundle in bundles:
        if not isinstance(bundle, dict) or bundle.get("trusted") is not True:
            continue
        evidence_id = bundle.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id or counts[evidence_id] != 1:
            continue
        content = bundle.get("content") or {}
        if not isinstance(content, dict) or content.get("metric_id") != metric:
            continue
        result = content.get("result") or {}
        parsed = result.get("parsed") or {}
        payload = result.get("compare_result") or result.get("contribution_result") or result.get("metric_result") or result.get("trend_result")
        authorized = parsed.get("authorized_metric") or {}
        if not isinstance(payload, dict) or result.get("status") != "ok" or payload.get("status") != "ok":
            continue
        if not metric or parsed.get("metric_name") != metric or authorized.get("metric_id") != metric or payload.get("metric_name") != metric:
            continue
        if result.get("compare_result") or result.get("contribution_result"):
            records.append((bundle.get("evidence_id"), payload))
        if result.get("metric_result") or result.get("trend_result"):
            date_records.append((bundle.get("evidence_id"), payload))
    checks = []
    if len(records) == 1:
        evidence_id, payload = records[0]
        for match in TOTAL.finditer(answer):
            if re.search(r"占\s*$", answer[max(0, match.start() - 8):match.start()]):
                continue
            expected = payload.get("total_delta")
            if isinstance(expected, (int, float)) and not isinstance(expected, bool) and math.isfinite(expected):
                actual = _num(match.group(1))
                checks.append({"text": match.group(0), "start": match.start(), "end": match.end(), "evidence_id": evidence_id, "evidence_field": "total_delta", "actual": actual, "expected": expected, "matched": math.isclose(actual, expected, rel_tol=0, abs_tol=0.005)})
        for match in list(TABLE.finditer(answer)) + list(ROLE_LINE.finditer(answer)):
            role = "baseline" if match.group(1) == "基线日" else "current"
            expected, date = payload.get(role + "_value"), payload.get(role + "_date")
            if date == match.group(2) and isinstance(expected, (int, float)) and not isinstance(expected, bool) and math.isfinite(expected):
                actual = _num(match.group(3))
                checks.append({"text": match.group(0), "start": match.start(), "end": match.end(), "evidence_id": evidence_id, "evidence_field": role + "_value", "actual": actual, "expected": expected, "matched": math.isclose(actual, expected, rel_tol=0, abs_tol=0.005)})
    if len(date_records) == 1:
        evidence_id, payload = date_records[0]
        values = {}
        seen_dates = set()
        if payload.get("metric_date") and isinstance(payload.get("value"), (int, float)) and not isinstance(payload.get("value"), bool):
            values[payload["metric_date"]] = payload["value"]
        for point in payload.get("points") or []:
            if not isinstance(point, dict) or not point.get("bucket") or not isinstance(point.get("value"), (int, float)) or isinstance(point.get("value"), bool):
                continue
            # Duplicate dates are ambiguous, even when values happen to match.
            date = point["bucket"]
            values[date] = None if date in seen_dates else point["value"]
            seen_dates.add(date)
        for match in DATE_VALUE.finditer(answer):
            expected = values.get(match.group(1))
            if expected is None or not math.isfinite(expected):
                continue
            actual = _num(match.group(2))
            checks.append({"text": match.group(0), "start": match.start(), "end": match.end(), "evidence_id": evidence_id, "evidence_field": "date_value:" + match.group(1), "actual": actual, "expected": expected, "matched": math.isclose(actual, expected, rel_tol=0, abs_tol=0.005)})
    unit_contract = _authorized_comparison_units(case, metric)
    if unit_contract:
        _metric_unit, change_unit, _relative_unit, evidence_id = unit_contract
        if change_unit == "percentage_points":
            for match in _PERCENT_DELTA_CLAIM.finditer(answer):
                checks.append({
                    "text": match.group(0), "start": match.start(), "end": match.end(),
                    "evidence_id": evidence_id, "evidence_field": "absolute_change_unit",
                    "actual": match.group("unit"), "expected": "percentage_points", "matched": False,
                })
    return {"case_id": case.get("case_id"), "status": "mismatch_found" if any(not check["matched"] for check in checks) else "partial_checks_only" if checks else "unverified", "checks": checks, "whole_answer_verified": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Refusing to overwrite probe output")
    rows = [json.loads(line) for line in args.inputs.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = [probe(row) for row in rows]
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in results), encoding="utf-8")
    print(json.dumps({"cases": len(results), "mismatch_found": sum(r["status"] == "mismatch_found" for r in results), "partial_checks_only": sum(r["status"] == "partial_checks_only" for r in results), "unverified": sum(r["status"] == "unverified" for r in results)}))


if __name__ == "__main__":
    main()

"""Compare emitted comparison values with their cited query evidence.

This supplements provenance checks; it never changes evidence or repairs text.
Only explicit comparison fields and unambiguous total-change statements are
parsed. It is not a general natural-language fact checker.
"""
from __future__ import annotations

import math
import re
from typing import Any

_NUMBER = r"[+\-−]?\d[\d,]*(?:\.\d+)?"
_TOTAL_CHANGE = re.compile(
    rf"(?:总变化(?:量|额)?|总差值|差值|净变化|total\s+(?:change|delta))"
    rf"\s*(?:为|是|[:：=]|of|is)?\s*({_NUMBER})", re.I,
)
_DIRECTION_CHANGE = re.compile(
    rf"(?:总额|总量|客单价|订单量|GMV|AOV)[^。;；\n]{{0,100}}?"
    rf"(下降|减少|增加|上升)(?:了)?\s*({_NUMBER})\s*"
    rf"(BRL/order|BRL|orders|雷亚尔/单|雷亚尔|单|百分点|percentage_points|%)", re.I,
)


def _same(actual: Any, expected: Any) -> bool:
    if actual is None or expected is None:
        return actual is expected
    if isinstance(actual, bool) or isinstance(expected, bool):
        return False
    try:
        a, e = float(actual), float(expected)
        return math.isfinite(a) and math.isfinite(e) and abs(a - e) <= 0.00500001
    except (TypeError, ValueError):
        return False


def _unit(value: Any) -> str:
    return {"百分点": "percentage_points", "pp": "percentage_points",
            "%": "%", "percent": "%"}.get(str(value), str(value))


def _period_label(start: Any, end: Any) -> str | None:
    """Normalize a cited period to the label used by comparison cards."""
    if not isinstance(start, str) or not start:
        return None
    if not isinstance(end, str) or not end:
        end = start
    return start if start == end else f"{start} 至 {end}"


def _comparison(bundle: dict) -> dict | None:
    content = bundle.get("content") or {}
    result = content.get("result") or {}
    if result.get("status") != "ok":
        return None
    payload = result.get("contribution_result") or result.get("compare_result")
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        return None
    # Do not guess legacy left/right roles. The public adapter must have
    # supplied canonical roles before this check can use a comparison.
    baseline = payload.get("baseline_value", payload.get("baseline_total"))
    current = payload.get("current_value", payload.get("current_total"))
    if baseline is None or current is None:
        return None
    metric = payload.get("metric_name")
    parsed = result.get("parsed") or {}
    authorized = parsed.get("authorized_metric") or {}
    if not metric or parsed.get("metric_name") != metric or authorized.get("metric_id") != metric:
        return None
    try:
        baseline, current = float(baseline), float(current)
        if not all(math.isfinite(v) for v in (baseline, current)):
            return None
    except (ValueError, TypeError):
        return None
    unit = payload.get("unit")
    if not unit:
        try:
            from app.metrics.registry import get_registry
            unit = get_registry().get(metric).unit
        except (KeyError, ValueError):
            unit = None
    baseline_date = payload.get("baseline_date") or payload.get("right_date")
    current_date = payload.get("current_date") or payload.get("left_date")
    baseline_end_date = payload.get("right_end_date") or baseline_date
    current_end_date = payload.get("left_end_date") or current_date
    return {
        "metric": metric, "baseline_value": baseline, "current_value": current,
        "delta": current - baseline,
        "delta_ratio": None if baseline == 0 else (current - baseline) * 100 / baseline,
        "baseline_date": baseline_date,
        "current_date": current_date,
        "baseline_period": _period_label(baseline_date, baseline_end_date),
        "current_period": _period_label(current_date, current_end_date),
        "unit": unit, "change_unit": "percentage_points" if unit == "%" else unit,
        "delta_ratio_unit": "%",
    }


def review_numeric_output(state: dict) -> dict:
    evidence = {
        b.get("evidence_id"): record
        for b in state.get("evidence_bundles", []) if isinstance(b, dict)
        if (record := _comparison(b)) is not None
    }
    issues: list[str] = []
    checked = 0

    def candidates(refs, metric=None):
        values = [evidence[r] for r in refs if r in evidence and (not metric or evidence[r]["metric"] == metric)]
        # Conflicting cited comparisons cannot serve as an arbitrary choice
        # of whichever number happens to match the generated answer.
        unique = {repr(sorted(v.items())): v for v in values}
        return list(unique.values())

    presentation = state.get("analysis_presentation") or {}
    cards = []
    if isinstance(presentation, dict):
        for key in ("changes", "contributions"):
            cards.extend((f"{key}[{i}]", c) for i, c in enumerate(presentation.get(key) or []) if isinstance(c, dict))
        if "baseline_value" in presentation and "current_value" in presentation:
            cards.append(("comparison", presentation))
    aliases = {"baseline_value": "baseline_value", "current_value": "current_value",
               "delta": "delta", "delta_value": "delta", "total_delta": "delta",
               "delta_ratio": "delta_ratio"}
    for label, card in cards:
        metric = card.get("metric_name") or card.get("metric_id")
        refs = card.get("evidence_ids") or []
        # Legacy flat presentations cite evidence on their item/claim objects.
        if not refs and label == "comparison":
            refs = list(dict.fromkeys(r for c in state.get("claims", []) for r in c.get("source_evidence_ids", [])))
        records = candidates(refs, metric)
        if not records:
            if candidates(refs) and metric:
                issues.append(f"{label}: output metric differs from cited comparison evidence.")
            continue  # Existing provenance/diagnostic checks own other shapes.
        checked += 1
        if len(records) != 1:
            issues.append(f"{label}: cited comparisons are ambiguous or conflicting.")
            continue
        record = records[0]
        for field, role in aliases.items():
            if field in card and not _same(card[field], record[role]):
                issues.append(f"{label}.{field}: output differs from cited comparison evidence.")
        for field in ("unit", "change_unit", "delta_ratio_unit"):
            if card.get(field) is not None and record.get(field) is not None and _unit(card[field]) != _unit(record[field]):
                issues.append(f"{label}.{field}: output unit differs from comparison contract.")
        for field, role in (("baseline_date", "baseline_date"), ("current_date", "current_date"),
                            ("baseline_period", "baseline_period"), ("current_period", "current_period")):
            if isinstance(card.get(field), str) and record.get(role) and card[field] != record[role]:
                issues.append(f"{label}.{field}: output period differs from cited comparison evidence.")

    def review_text(label, text, records):
        nonlocal checked
        matches = list(_TOTAL_CHANGE.finditer(text))
        directed = list(_DIRECTION_CHANGE.finditer(text))
        if not (matches or directed) or not records:
            return
        # Multi-metric prose requires explicit metric association; do not
        # compare a number with the union of unrelated metrics' values.
        if len(records) != 1:
            return
        checked += len(matches)
        for match in matches:
            prefix = text[max(0, match.start() - 24):match.start()]
            if re.search(r"(?:占|share\s+of)\s*$", prefix, re.I):
                # A component's share of total change is neither the total
                # absolute change nor the metric's relative change rate.
                checked -= 1
                continue
            value = float(match.group(1).replace(",", "").replace("−", "-"))
            role = "delta_ratio" if text[match.end():].lstrip().startswith('%') else "delta"
            if not _same(value, records[0][role]):
                issues.append(f"{label}: stated total change differs from cited comparison evidence.")
        for match in directed:
            checked += 1
            value = float(match.group(2).replace(",", "").replace("−", "-"))
            value = -abs(value) if match.group(1) in {"下降", "减少"} else abs(value)
            # Percent relative change must not be compared to an absolute
            # amount, nor percentage points to a relative percent.
            stated_unit = {"雷亚尔": "BRL", "雷亚尔/单": "BRL/order", "单": "orders"}.get(match.group(3), match.group(3))
            role = "delta_ratio" if stated_unit == "%" else "delta"
            if not _same(value, records[0][role]):
                issues.append(f"{label}: stated directional change differs from cited comparison evidence.")
            expected_unit = "%" if role == "delta_ratio" else records[0]["change_unit"]
            if expected_unit and _unit(stated_unit) != _unit(expected_unit):
                issues.append(f"{label}: stated change unit differs from cited comparison contract.")

    for i, claim in enumerate(state.get("claims", [])):
        review_text(f"claim[{i}]", str(claim.get("text") or claim.get("value") or ""),
                    candidates(claim.get("source_evidence_ids") or []))
    # Only bind unambiguous single-metric prose to references actually emitted
    # by the answer, never to arbitrary evidence present elsewhere in state.
    answer_refs = list(dict.fromkeys(r for c in state.get("claims", []) for r in c.get("source_evidence_ids", [])))
    review_text("answer", str(state.get("answer") or state.get("final_answer") or ""), candidates(answer_refs))
    return {"status": "rejected" if issues else "approved" if checked else "not_applicable",
            "issues": issues, "checked_values": checked,
            "coverage": "canonical cited comparisons, output cards and explicit unambiguous total-change text"}

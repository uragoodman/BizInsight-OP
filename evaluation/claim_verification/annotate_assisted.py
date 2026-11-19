"""Build assistant-authored Claim annotations under the frozen v1 schema.

This is *not* an independent human adjudication.  The assistant may correct
span hallucinations, arithmetic that only becomes checkable once group totals
are included, and requirement coverage.  It may not invent evidence, and it may
not mark an assertion supported merely because a neighbouring one is.

Every row is written with ``human_verified=false``.  Promoting the set requires
a separate human spot-check pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from evaluation.claim_verification.annotation_schema import validate_annotation_row
from evaluation.claim_verification.deterministic_probe import probe

ASSISTANT = "assistant"
NUMBER = re.compile(r"[+\-−]?\d[\d,]*(?:\.\d+)?")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
PROCESS_MARKERS = ("调查分支状态", "尚未完成深入调查", "受数据覆盖限制")
LIMITATION_MARKERS = ("未执行数据查询", "不支持", "未改用", "不证明业务因果", "未确认业务因果", "无法")
RECOMMENDATION_MARKERS = ("建议", "应当", "应该", "推荐", "应立即", "应停止", "应增加", "应削减")
DECLARED_UNIT = re.compile(r"单位(?:说明)?[：:]\s*([^。；\n]{0,40})")
EVIDENCE_TABLE = re.compile(r"(?:数据来源|证据来源|source)[：:]\s*([A-Za-z0-9_.]+)")
CAUSAL_MARKERS = re.compile(
    r"导致|引起|造成|因为|由于|根因|原因(?:已经|已)?确定|证明|这表明|这说明|说明.{0,6}(?:已经|正在|是)|"
    r"表明|意味着|确认.{0,12}(?:已经|已)|caus(?:e|ed|al)"
)
NUMERIC_TYPES = {"numeric_fact", "calculation", "comparison"}


def _num(value: str) -> float:
    return float(value.replace(",", "").replace("−", "-"))


STRUCTURAL_COUNT = re.compile(r"\d+\s*(?:个时间点|个数据点|个维度|天|条记录|个分组)")
DIRECTION_WORDS = re.compile(r"下降|下滑|减少|上升|增加|增长|降低")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


EVIDENCE_ID = re.compile(r"\bev_[0-9a-f]{8,}\b")


def _numbers(span: str) -> list[float]:
    # Dates are not metrics, "共 2 个时间点" counts structure, and evidence IDs
    # contain hex digits that would otherwise parse as numbers.
    cleaned = DATE.sub(" ", span)
    cleaned = EVIDENCE_ID.sub(" ", cleaned)
    cleaned = STRUCTURAL_COUNT.sub(" ", cleaned)
    return [_num(match.group(0)) for match in NUMBER.finditer(cleaned)]


def _values_close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0, abs_tol=0.02)


def _magnitude_close(actual: float, expected: float) -> bool:
    return math.isclose(abs(actual), abs(expected), rel_tol=0, abs_tol=0.02)


def _explainable(bundle: dict[str, Any]) -> tuple[set[float], str | None]:
    """Every number this bundle can justify, plus its declared unit.

    Two shapes matter.  A contribution payload gives per-dimension values and
    a total, so a legitimate "Other" bucket is only justified as
    ``total - sum(groups)``.  A scalar payload gives named values directly.
    """

    result = (bundle.get("content") or {}).get("result") or {}
    if result.get("status") != "ok":
        return set(), None
    unit = ((result.get("parsed") or {}).get("authorized_metric") or {}).get("unit")
    known: set[float] = set()

    contribution = result.get("contribution_result") or {}
    if contribution.get("status") == "ok":
        totals: dict[str, float] = {}
        groups: dict[str, dict[str, float]] = {}
        for role in ("baseline", "current"):
            values = {}
            for item in contribution.get(f"{role}_items") or []:
                value = item.get("value")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    values[str(item.get("dimension_value"))] = float(value)
            groups[role] = values
            total = contribution.get(f"{role}_total")
            if isinstance(total, (int, float)) and not isinstance(total, bool):
                totals[role] = float(total)
                known.add(float(total))
                known.add(round(float(total) - sum(values.values()), 2))
            known.update(values.values())
        total_delta = contribution.get("total_delta")
        if isinstance(total_delta, (int, float)) and not isinstance(total_delta, bool):
            known.add(float(total_delta))
        for dimension in set(groups["baseline"]) | set(groups["current"]):
            delta = groups["current"].get(dimension, 0.0) - groups["baseline"].get(dimension, 0.0)
            known.add(round(delta, 2))
            if isinstance(total_delta, (int, float)) and total_delta:
                known.add(round(abs(delta) / abs(float(total_delta)) * 100, 1))
        if "baseline" in totals and "current" in totals:
            known.add(round(totals["current"] - totals["baseline"], 2))
            # A derived "Other" bucket is legitimate when it is exactly the part
            # of each total not covered by the listed dimensions.
            before = round(totals["baseline"] - sum(groups["baseline"].values()), 2)
            after = round(totals["current"] - sum(groups["current"].values()), 2)
            known |= {before, after, round(after - before, 2)}
        return known, unit

    scalar = result.get("compare_result") or result.get("metric_result") or result.get("trend_result") or {}
    if scalar.get("status") == "ok":
        for key, value in scalar.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                known.add(float(value))
        for point in scalar.get("points") or []:
            value = point.get("value") if isinstance(point, dict) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                known.add(float(value))
        return known, unit

    return known, unit


def _unexplained(
    span: str, bundles: list[dict[str, Any]], evidence_ids: list[str],
) -> tuple[bool, bool]:
    """Split unverifiable numbers into (a cited value, a derived share).

    A *value* the cited evidence provably does not contain is a contradiction.
    A percentage is different: the answer may quote a rounding or a derived
    rate, so a share that does not match is only a weaker signal.
    """

    numbers = _numbers(span)
    if not numbers:
        return False, False
    # Judge against the bundle this claim actually cites, not whichever bundle
    # happened to be seen first.
    candidate = next(
        (b for b in bundles if b.get("evidence_id") in evidence_ids and b.get("trusted") is True),
        None,
    )
    if candidate is None:
        trusted = [b for b in bundles if b.get("trusted") is True and b.get("evidence_id")]
        candidate = trusted[0] if len(trusted) == 1 else None
    if candidate is None:
        return False, False
    known, _unit = _explainable(candidate)
    baseline_total = _contribution_total(candidate, "baseline")
    current_total = _contribution_total(candidate, "current")
    total_delta = _contribution_delta(candidate)
    unexplained_value = unexplained_share = False
    shares: set[float] = set()
    if baseline_total and current_total is not None:
        shares.add(abs(round((current_total - baseline_total) / baseline_total * 100, 2)))
    if total_delta:
        # Share of the total movement: |delta| / |total_delta|.
        shares |= {abs(round(value / total_delta * 100, 1)) for value in known}
    magnitudes = {abs(value) for value in known}
    known_deltas = _known_deltas(candidate)
    for value in numbers:
        if any(_values_close(value, k) for k in known) or abs(value) in magnitudes:
            continue
        # "整体下降 0.59" states a movement between two evidenced values.
        if any(_values_close(value, k) for k in known_deltas):
            continue
        # A share that reproduces the evidence's own arithmetic is justified.
        if any(_values_close(value, k) for k in shares):
            continue
        if bool(SHARE_CONTEXT.search(span)) and abs(value) <= 100:
            unexplained_share = True
        else:
            unexplained_value = True
    return unexplained_value, unexplained_share


def _known_deltas(bundle: dict[str, Any]) -> set[float]:
    """Signed and absolute differences the evidence actually supports."""

    result = (bundle.get("content") or {}).get("result") or {}
    payload = result.get("compare_result") or {}
    deltas: set[float] = set()
    if payload.get("status") == "ok":
        before, after = payload.get("baseline_value"), payload.get("current_value")
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            deltas.add(round(float(after) - float(before), 2))
    trend = result.get("trend_result") or {}
    points = [p.get("value") for p in trend.get("points") or [] if isinstance(p, dict)]
    points = [p for p in points if isinstance(p, (int, float)) and not isinstance(p, bool)]
    if len(points) >= 2:
        deltas.add(round(float(points[-1]) - float(points[0]), 2))
    return deltas | {-d for d in deltas}


def _probe_mismatch(span: str, start: int, end: int, checks: list[dict[str, Any]]) -> bool:
    """A deterministic check inside the span already says the value is wrong."""

    return any(
        isinstance(check, dict)
        and check.get("matched") is False
        and isinstance(check.get("start"), int) and isinstance(check.get("end"), int)
        and check["start"] < end and start < check["end"]
        for check in checks
    )


def _lacks_value(span: str, bundles: list[dict[str, Any]], evidence_ids: list[str]) -> bool:
    """A cited business value that no trusted bundle contains.

    Only values in a business-value position count: a bare number inside a
    "占总变化 x%" phrase is a share, and shares are judged separately.
    """

    numbers = _numbers(span)
    if not numbers:
        return False
    trusted = [b for b in bundles if b.get("trusted") is True and b.get("evidence_id")]
    if not trusted:
        return False
    known: set[float] = set()
    deltas: set[float] = set()
    for bundle in trusted:
        values, _unit = _explainable(bundle)
        known |= values
        deltas |= _known_deltas(bundle)
    magnitudes = {abs(value) for value in known}
    for value in numbers:
        if any(_values_close(value, k) for k in known) or abs(value) in magnitudes:
            continue
        if any(_values_close(value, k) for k in deltas):
            continue
        if _is_share(value, span):
            continue
        match = _enclosing_number_regex(value, span)
        if match and re.search(r"占|占比|变化率|%", span[max(0, match.start() - 12):match.end() + 4]):
            continue
        return True
    return False


def _is_share(value: float, span: str) -> bool:
    return abs(value) <= 100 and bool(SHARE_CONTEXT.search(span))


def _enclosing_number_regex(value: float, span: str):
    for match in NUMBER.finditer(DATE.sub(" ", span)):
        try:
            if math.isclose(_num(match.group(0)), value, rel_tol=0, abs_tol=0.02):
                return match
        except ValueError:
            continue
    return None


def _contribution_delta(bundle: dict[str, Any]) -> float | None:
    payload = ((bundle.get("content") or {}).get("result") or {}).get("contribution_result") or {}
    delta = payload.get("total_delta")
    return float(delta) if isinstance(delta, (int, float)) and not isinstance(delta, bool) else None


def _contribution_total(bundle: dict[str, Any], role: str) -> float | None:
    payload = ((bundle.get("content") or {}).get("result") or {}).get("contribution_result") or {}
    total = payload.get(f"{role}_total")
    return float(total) if isinstance(total, (int, float)) and not isinstance(total, bool) else None


def _looks_like_share(span: str, value: float) -> bool:
    return bool(re.search(r"占|变化率|占比|%", span)) and 0 <= abs(value) <= 100


def _binding(
    span: str, bundles: list[dict[str, Any]], used: set[str],
) -> tuple[list[str], bool]:
    """Return (evidence ids, whether every number in the span is justified).

    `evidence_reference` claims name their own source, either as an evidence ID
    or by naming the table the trusted bundles were read from.
    """

    numbers = _numbers(span)
    named = [b["evidence_id"] for b in bundles
             if b.get("trusted") is True and b.get("evidence_id") and b["evidence_id"] in span]
    if named:
        return named[:1], True
    if not numbers:
        declared = EVIDENCE_TABLE.search(span)
        if declared:
            table = declared.group(1).rstrip("。")
            for bundle in bundles:
                source = (((bundle.get("content") or {}).get("result") or {}).get("parse_source")
                          or _source_table(bundle))
                if bundle.get("trusted") is True and source == table:
                    return [bundle["evidence_id"]], True
            return [], False
        return [], False
    best: tuple[int, str] | None = None
    for bundle in bundles:
        evidence_id = bundle.get("evidence_id")
        if bundle.get("trusted") is not True or not evidence_id:
            continue
        known, _unit = _explainable(bundle)
        if not known:
            continue
        matched = sum(
            1 for value in numbers
            # A percentage in "占总变化 x%" is a derivation, not a value to be
            # matched against the bundle. Counting it as unmatched would make
            # every correct contribution sentence look uncertain.
            if any(_values_close(value, k) for k in known) or _is_share(value, span)
        )
        # Pick strictly by coverage. Preferring an unused bundle would assign a
        # trend bundle to a contribution claim just because its ID was free.
        if matched and (best is None or matched > best[0]
                        or (matched == best[0] and evidence_id < best[1])):
            best = (matched, evidence_id)
    if best is None:
        return [], False
    used.add(best[1])
    return [best[1]], best[0] == len(numbers)


def _source_table(bundle: dict[str, Any]) -> str | None:
    result = (bundle.get("content") or {}).get("result") or {}
    for key in ("contribution_result", "compare_result", "metric_result", "trend_result"):
        table = (result.get(key) or {}).get("source_table")
        if table:
            return table
    return None


def _retype(span: str, claim_type: str) -> str:
    if any(marker in span for marker in PROCESS_MARKERS):
        return "unknown"
    # A stated unit or scope definition is a definition claim even when the AI
    # filed it as `unknown`.
    if DECLARED_UNIT.search(span) and claim_type in NUMERIC_TYPES | {"unknown"}:
        return "definition_or_scope"
    # Agent-authored prose over-triggers these two labels: "按支付方式分析…"
    # is scope, and "未改用其他维度代替" is a limitation, not advice.
    if claim_type == "recommendation" and not any(marker in span for marker in RECOMMENDATION_MARKERS):
        return "limitation_or_refusal" if any(m in span for m in LIMITATION_MARKERS) else "definition_or_scope"
    if claim_type == "causal_claim" and not CAUSAL_MARKERS.search(span):
        return "limitation_or_refusal" if any(m in span for m in LIMITATION_MARKERS) else "definition_or_scope"
    if claim_type == "evidence_reference":
        names_evidence = bool(EVIDENCE_TABLE.search(span)) or "ev_" in span
        if not names_evidence:
            # A sentence that asserts scope or validation without naming a
            # source is not an evidence reference.
            return "definition_or_scope"
    return claim_type


DIRECTION = (
    (("低于", "下降", "减少", "下滑"), "down"),
    (("高于", "上升", "增加", "增长"), "up"),
    (("持平", "不变"), "flat"),
)


def _direction_consistent(span: str, bundles: list[dict[str, Any]]) -> tuple[bool, str | None]:
    """Check a verbal comparison against the bundle's two dated values.

    A sentence like "整体下降 0.59" states a *delta*, and a delta's sign is
    read directly; "当前值低于基线" states a level, which must be compared the
    other way round.  Treating both identically flags correct answers.
    """

    stated = next((direction for words, direction in DIRECTION if any(word in span for word in words)), None)
    if stated is None:
        return True, None
    delta_style = bool(DIRECTION_WORDS.search(span)) and bool(NUMBER.search(DATE.sub(" ", span)))
    for bundle in bundles:
        if bundle.get("trusted") is not True or not bundle.get("evidence_id"):
            continue
        result = (bundle.get("content") or {}).get("result") or {}
        payload = result.get("compare_result") or {}
        if payload.get("status") != "ok":
            continue
        before, after = payload.get("baseline_value"), payload.get("current_value")
        if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
            continue
        if delta_style:
            # The numbers carry the sign; only flag an explicit contradiction.
            numbers = _numbers(span)
            if not numbers:
                continue
            observed = "down" if numbers[0] < 0 else "up" if numbers[0] > 0 else "flat"
            return observed == stated or after == before, bundle["evidence_id"]
        observed = "up" if after > before else "down" if after < before else "flat"
        return observed == stated, bundle["evidence_id"]
    return True, None


def _declared_unit_supported(span: str, bundles: list[dict[str, Any]]) -> bool:
    declared = DECLARED_UNIT.search(span)
    if not declared:
        return True
    units = {_explainable(bundle)[1] for bundle in bundles if bundle.get("trusted") is True}
    units.discard(None)
    if not units:
        return True  # nothing in the bundle contradicts the statement
    words = {
        "orders": ("单", "个", "笔"),
        "BRL": ("雷亚尔", "BRL", "元"),
        "%": ("%", "百分点"),
    }
    return any(word in declared.group(1) for unit in units for word in words.get(unit, (unit,)))


def _comparison_roles(span: str, bundles: list[dict[str, Any]], metric_id: str | None) -> str:
    """Only compare explicit baseline/current values to matching evidence roles."""
    number = r"(?!\d{4}-\d{2}-\d{2})([+\-−]?\d[\d,]*(?:\.\d+)?)"
    baseline = re.search(r"基线(?:值|日)?\s*[：:=]?\s*" + number, span)
    current = re.search(r"当前(?:值|日)?\s*[：:=]?\s*" + number, span)
    if not baseline or not current:
        return "unknown"
    candidates = []
    for bundle in bundles:
        if bundle.get("trusted") is not True:
            continue
        result = (bundle.get("content") or {}).get("result") or {}
        payload = result.get("compare_result") or {}
        if result.get("status") != "ok" or payload.get("status") != "ok":
            continue
        parsed = result.get("parsed") or {}
        if not metric_id or (bundle.get("content") or {}).get("metric_id") != metric_id or parsed.get("metric_name") != metric_id or (parsed.get("authorized_metric") or {}).get("metric_id") != metric_id or payload.get("metric_name") != metric_id:
            continue
        before, after = payload.get("baseline_value"), payload.get("current_value")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (before, after)):
            continue
        candidates.append((float(before), float(after)))
    if len(candidates) != 1:
        return "unknown"
    before, after = candidates[0]
    matched = math.isclose(_num(baseline.group(1)), before, rel_tol=0, abs_tol=0.00500001) and math.isclose(_num(current.group(1)), after, rel_tol=0, abs_tol=0.00500001)
    if not matched:
        return "mismatch"
    # Other claims in the same span require independent checks, even if the
    # two role-labelled numbers happen to be correct.
    whole = re.fullmatch(r"\s*基线(?:值|日)?\s*[：:=]?\s*" + number + r"\s*[,，;；]\s*当前(?:值|日)?\s*[：:=]?\s*" + number + r"\s*[。.]?\s*", span)
    return "matched" if whole else "unknown"


def build_row(case: dict[str, Any], record: dict[str, Any], key: dict[str, Any]) -> dict[str, Any]:
    answer = case["answer"]
    validation = record.get("validation") or {}
    bundles = [b for b in case.get("evidence_bundles") or [] if isinstance(b, dict)]
    trusted = [b for b in bundles if b.get("trusted") is True and b.get("evidence_id")]
    fallback = trusted[0]["evidence_id"] if trusted else None
    used: set[str] = set()
    probe_checks = probe(case).get("checks") or []

    claims: list[dict[str, Any]] = []
    for raw in validation.get("claims") or []:
        span = raw.get("span")
        if not isinstance(span, str) or not span or answer.count(span) != 1:
            continue  # span does not exist verbatim; never silently repair it
        start = answer.find(span)
        claim_type = _retype(span, raw.get("type") or "unknown")
        ids, fully_explained = _binding(span, bundles, used)
        direction_ok, direction_source = _direction_consistent(span, bundles)
        if direction_source and not ids:
            ids, fully_explained = [direction_source], True
        unexplained_value, unexplained_share = _unexplained(span, bundles, ids)
        probe_mismatch = _probe_mismatch(span, start, start + len(span), probe_checks)
        lacks_value = _lacks_value(span, bundles, ids)
        numbers = _numbers(span)
        if not numbers and claim_type in NUMERIC_TYPES and fallback:
            # No business value left after removing dates and structural counts;
            # the sentence still refers to the same series.
            ids, fully_explained = ids or [fallback], True

        role_check = _comparison_roles(span, bundles, (case.get("request_contract") or {}).get("metric_id"))
        if claim_type in ("causal_claim", "recommendation"):
            correctness = "uncertain"
        elif any(marker in span for marker in PROCESS_MARKERS):
            correctness = "uncertain"
        elif claim_type in NUMERIC_TYPES or claim_type == "evidence_reference":
            if not direction_ok or probe_mismatch or role_check == "mismatch":
                # A covering deterministic check, or a business value the cited
                # evidence provably lacks, is proof of an error.
                correctness = "unsupported"
            elif not ids:
                correctness = "uncertain"
            elif claim_type == "evidence_reference" or not numbers:
                correctness = "supported"
            elif role_check == "matched" and fully_explained and not unexplained_share:
                correctness = "supported"
            else:
                # The numbers could not all be matched. That is not proof of an
                # error, so it must not become a false reject.
                correctness = "uncertain"
        elif claim_type == "definition_or_scope":
            correctness = "uncertain"
        else:
            correctness = "uncertain"
        if claim_type == "limitation_or_refusal":
            ids = []
            correctness = "uncertain"
        if not ids and claim_type in NUMERIC_TYPES and fallback and _numbers(span):
            ids = [fallback]

        claims.append({
            "claim_id": f"C{len(claims) + 1}",
            "start": start,
            "end": start + len(span),
            "text_span": span,
            "claim_type": claim_type,
            "evidence_ids": ids,
            "correctness": correctness,
        })

    # The injected sentence is the row's reason to exist; a reviewer must never
    # receive a mutation row whose claim set omits it.  A sentence-level claim
    # that already covers the injection is kept rather than duplicated.
    changed = (key.get("changed_text") or "").strip()
    if key.get("provenance") == "targeted_mutation" and changed and answer.count(changed) == 1:
        start = answer.find(changed)
        end = start + len(changed)
        covered = any(start < claim["end"] and claim["start"] < end for claim in claims)
        if not covered:
            claims.append({
                "claim_id": f"C{len(claims) + 1}", "start": start, "end": end,
                "text_span": changed, "claim_type": "unknown", "evidence_ids": [], "correctness": "uncertain",
            })

    requirements = list(case.get("task_requirements") or [])
    coverage = _coverage(requirements, answer, key.get("expected_outcome"))
    return {
        "case_id": case["case_id"],
        "answer": answer,
        "claims": claims,
        "task_requirements": requirements,
        "requirement_coverage": coverage,
        "provenance": _provenance(key),
        "expected_decision": _expected_decision(claims, coverage, key.get("expected_outcome")),
        "annotation_status": "assistant_annotated_pending_human_spotcheck",
        "human_verified": False,
        "oracle_claim_labels": False,
        "annotation_author": ASSISTANT,
    }


def _provenance(key: dict[str, Any]) -> str:
    if key.get("provenance") == "targeted_mutation":
        return "targeted_mutation"
    return "real_correct" if key.get("expected_outcome") != "safe_stop" else "real_failure"


NEGATED_CAUSALITY = re.compile(r"未确认|不证明|无法确认|不能确认|未建立|not (?:confirmed|established)|remains unconfirmed")
NEGATED_CAUSALITY_MARKERS = ("未确认", "不证明", "无法确认", "不能确认", "未建立")
QUOTED = re.compile(r"[“「][^”」]*[”」]")
SHARE_CONTEXT = re.compile(r"占|占比|变化率|%")


def _unsupported_causality(answer: str) -> bool:
    """True when a sentence asserts causality without hedging it.

    The answer usually quotes the user's request, which may itself contain
    "说明未确认业务因果" or "说明单位".  Quoted text is not the answer's claim.
    """

    for sentence in re.split(r"[。\n]", QUOTED.sub(" ", answer)):
        if CAUSAL_MARKERS.search(sentence) and not NEGATED_CAUSALITY.search(sentence):
            return True
    return False


def _causality_boundary(answer: str, expected_outcome: str | None) -> str:
    """Coverage of "no business-causality claim unless independently supported".

    Satisfied when the answer either stays descriptive or explicitly declines
    to assert causality.  `missing` is reserved for asserted causality and for
    answers that make no statement at all.
    """

    if _unsupported_causality(answer):
        return "missing"
    lowered = answer.lower()
    if any(marker in answer for marker in NEGATED_CAUSALITY_MARKERS) or "unconfirmed" in lowered:
        return "covered"
    if all(marker not in answer for marker in LIMITATION_MARKERS):
        return "covered"
    return "uncertain"


def _coverage(requirements: list[str], answer: str, expected_outcome: str | None) -> dict[str, str]:
    # These requirements depend on authorization, scope, operational state and
    # evidence; answer wording or the expected outcome cannot establish them.
    return {requirement: "uncertain" for requirement in requirements}


def _expected_decision(
    claims: list[dict[str, Any]], coverage: dict[str, str], expected_outcome: str | None,
) -> str:
    if any(claim["correctness"] == "unsupported" for claim in claims):
        return "reject_or_revise"
    if expected_outcome == "safe_stop":
        # The answer correctly declined.  Not producing a tool result is the
        # intended behaviour, so it is not a task failure.
        coverage = {k: ("covered" if k == "actual authorized tool result" else v) for k, v in coverage.items()}
    if any(status in ("missing", "uncertain") for status in coverage.values()):
        return "needs_semantic_review"
    if any(claim["correctness"] != "supported" for claim in claims):
        return "needs_semantic_review"
    return "pass"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError("Output exists; refusing overwrite of an annotation artifact")

    cases = {r["case_id"]: r for r in (json.loads(x) for x in args.inputs.read_text(encoding="utf-8").splitlines() if x.strip())}
    records = {r["case_id"]: r for r in (json.loads(x) for x in args.results.read_text(encoding="utf-8").splitlines() if x.strip())}
    keys = {r["case_id"]: r for r in (json.loads(x) for x in args.key.read_text(encoding="utf-8").splitlines() if x.strip())}
    if set(cases) != set(records) or set(cases) != set(keys):
        raise ValueError("inputs, results and key must cover the same cases")

    rows, report = [], []
    for case_id in sorted(cases):
        row = build_row(cases[case_id], records[case_id], keys[case_id])
        rows.append(row)
        report.append({"case_id": case_id, "pilot_id": keys[case_id].get("pilot_id"), **validate_annotation_row(row)})

    args.output_dir.mkdir(parents=True)
    (args.output_dir / "annotations.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    summary = {
        "schema_version": "claim-annotation-v1",
        "cases": len(rows),
        "claims": sum(len(row["claims"]) for row in rows),
        "structurally_valid": sum(item["structurally_valid"] for item in report),
        "ready_for_benchmark": sum(item["ready_for_benchmark"] for item in report),
        "annotation_author": ASSISTANT,
        "human_verified": False,
        "oracle_claim_labels": False,
        "input_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        "results_sha256": hashlib.sha256(args.results.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases_report": report,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in ("cases", "claims", "structurally_valid", "ready_for_benchmark")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

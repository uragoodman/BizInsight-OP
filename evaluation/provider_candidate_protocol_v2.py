"""Versioned machine scoring for the 17 public-data provider candidates.

This protocol composes the existing root-cause run scorer with candidate-truth,
scope/period, execution, and Critic-completion checks. It does not replace or
silently change ``root-cause-run-v2`` semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from evaluation.scorers import DIAGNOSTIC_RUN_SCORER_VERSION, _cited_valid_evidence, score_diagnostic_run


PROVIDER_CANDIDATE_SCORER_VERSION = "provider-candidate-run-v2"
PROVIDER_CANDIDATE_SCORER_V1 = "provider-candidate-run-v1"
FULL_REVIEW_NAMES = ("groundedness", "completeness", "consistency")
VALID_REVIEW_VERDICTS = {"passed", "failed"}


def _state(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("workflow_state", "state_snapshot", "state"):
        value = trace.get(key)
        if isinstance(value, Mapping):
            return value
    return trace


def _period_expectation(candidate: Mapping[str, Any]) -> dict[str, str]:
    period = candidate.get("comparison_period") or {}
    baseline, current = period.get("baseline"), period.get("current")
    if isinstance(baseline, str) and isinstance(current, str):
        return {
            "baseline_start": baseline, "baseline_end": baseline,
            "current_start": current, "current_end": current,
        }
    if isinstance(baseline, Mapping) and isinstance(current, Mapping):
        return {
            "baseline_start": str(baseline.get("start") or ""),
            "baseline_end": str(baseline.get("end") or ""),
            "current_start": str(current.get("start") or ""),
            "current_end": str(current.get("end") or ""),
        }
    return {}


def _target_matches(state: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[bool, bool, bool]:
    investigation = state.get("investigation_state") or {}
    target = investigation.get("target") or {} if isinstance(investigation, Mapping) else {}
    expected_period = _period_expectation(candidate)
    metric_ok = str(target.get("metric_id") or "").casefold() == str(candidate.get("target_metric") or "").casefold()
    period_ok = bool(expected_period) and all(str(target.get(key) or "") == value for key, value in expected_period.items())
    canonical_scope = target.get("canonical_scope") or {}
    scope_type = str(canonical_scope.get("scope_type") or canonical_scope.get("scope") or state.get("scope_type") or "").casefold()
    # Current platform target construction has used tenant_id=platform as its
    # canonical marker; accept that exact existing representation as well.
    platform_marker = str(canonical_scope.get("tenant_id") or "").casefold() == "platform"
    expected_scope = str(candidate.get("scope") or "").casefold()
    scope_ok = expected_scope == "platform" and (scope_type == "platform" or platform_marker)
    return metric_ok, period_ok, scope_ok


def _component_map(value: Any) -> dict[str, float] | None:
    if isinstance(value, Mapping):
        parsed: dict[str, float] = {}
        for key, raw in value.items():
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                return None
            parsed[str(key)] = float(raw)
        return parsed
    if not isinstance(value, list):
        return None
    parsed = {}
    for item in value:
        if not isinstance(item, Mapping):
            return None
        key = item.get("dimension_value", item.get("component_id"))
        raw = item.get("delta_value", item.get("value"))
        if key is None or isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        key = str(key)
        if key in parsed:
            return None
        parsed[key] = float(raw)
    return parsed


def _actual_component_map(evidence: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float] | None:
    values = evidence.get("values") or {}
    scoring = candidate.get("scoring_expectation") or {}
    wanted = scoring.get("component_field")
    fields = [wanted] if wanted else []
    fields += [field for field in ("display_components", "components", "raw_components") if field not in fields]
    factors = values.get("factor_metrics")
    for field in fields:
        parsed = _component_map(values.get(field))
        if parsed is None:
            continue
        expected = scoring.get("components") or {}
        if set(expected) <= set(parsed):
            return parsed
        # Some registered ratio decompositions use positional
        # numerator_effect/denominator_effect IDs. Their registered factor order
        # is carried with the evidence and is the only permitted name mapping.
        raw = values.get(field)
        if (
            isinstance(raw, list) and isinstance(factors, list)
            and len(raw) == len(factors)
            and all(isinstance(item, Mapping) and isinstance(item.get("value"), (int, float)) for item in raw)
            and set(expected) <= {str(item) for item in factors}
        ):
            return {str(factor): float(item["value"]) for factor, item in zip(factors, raw)}
    return None


def _expected_evidence(
    state: Mapping[str, Any], candidate: Mapping[str, Any], expected_kind: str,
) -> list[dict[str, Any]]:
    typed = _cited_valid_evidence(dict(state), {expected_kind})
    expected_metric = str(candidate.get("target_metric") or "").casefold()
    expected_method = str((candidate.get("registered_capability") or {}).get("method") or "")
    expected_period = _period_expectation(candidate)
    inv = state.get("investigation_state") or {}
    target = inv.get("target") or {} if isinstance(inv, Mapping) else {}
    target_scope = str(target.get("scope_fingerprint") or "")
    action_results = {
        str(result.get("candidate_id")): result
        for result in (inv.get("action_results") or []) if isinstance(result, Mapping)
    }
    evidence = {str(item.get("evidence_id")): item for item in (inv.get("evidence") or []) if isinstance(item, Mapping)}
    valid: list[dict[str, Any]] = []
    for evidence_id, item in typed.items():
        if str(item.get("metric_id") or "").casefold() != expected_metric:
            continue
        if str(item.get("method_id") or "") != expected_method:
            continue
        if not target_scope or str(item.get("scope_fingerprint") or "") != target_scope:
            continue
        comparison = item.get("comparison") or {}
        current = comparison.get("current_period") or {}
        baseline = comparison.get("baseline_period") or {}
        if not all((
            str(current.get("start_date") or "") == expected_period.get("current_start"),
            str(current.get("end_date") or "") == expected_period.get("current_end"),
            str(baseline.get("start_date") or "") == expected_period.get("baseline_start"),
            str(baseline.get("end_date") or "") == expected_period.get("baseline_end"),
        )):
            continue
        action = action_results.get(str(item.get("source_action_id") or ""))
        if not action or evidence_id not in evidence or evidence_id not in {str(eid) for eid in action.get("evidence_ids", [])}:
            continue
        if action.get("status") != "completed" or (action.get("validation_result") or {}).get("status") != "valid":
            continue
        valid.append(dict(item))
    return valid


def _review_result(state: Mapping[str, Any], arm: str) -> dict[str, Any]:
    critic = state.get("critic_result") or {}
    llm_review = critic.get("llm_review") or {}
    critic_node_calls = [
        call for call in (state.get("llm_calls") or [])
        if isinstance(call, Mapping) and call.get("node") == "critic"
    ]
    review_calls = [call for call in critic_node_calls if call.get("operation") == "extract"]
    if arm == "full":
        outputs = {name: (llm_review.get(name) or {}).get("verdict") for name in FULL_REVIEW_NAMES}
        sent_calls = [
            call for call in review_calls
            if call.get("outcome") == "success"
            and isinstance(call.get("attempts"), list)
            and bool(call.get("attempts"))
            and all(isinstance(attempt, Mapping) and attempt.get("attempt_id") for attempt in call["attempts"])
        ]
        calls_complete = len(sent_calls) >= len(FULL_REVIEW_NAMES)
        verdicts_complete = all(value in VALID_REVIEW_VERDICTS for value in outputs.values())
        all_passed = all(value == "passed" for value in outputs.values())
        return {
            "mode_matches": llm_review.get("mode") == "full" and llm_review.get("status") == "invoked",
            "review_outputs": outputs,
            "verdicts_complete": verdicts_complete,
            "critic_review_sdk_logical_calls": len(review_calls),
            "critic_node_logical_calls": len(critic_node_calls),
            "critic_review_sent_successfully": len(sent_calls),
            "call_records_complete": calls_complete,
            "all_reviews_passed": all_passed,
            "final_critic_approved": critic.get("status") == "approved",
            "review_contract_pass": bool(
                llm_review.get("mode") == "full" and llm_review.get("status") == "invoked"
                and verdicts_complete and calls_complete and all_passed
                and critic.get("status") == "approved"
            ),
        }
    if arm == "off":
        outputs = {name: (llm_review.get(name) or {}).get("verdict") for name in FULL_REVIEW_NAMES}
        ablated = (
            llm_review.get("mode") == "critic_llm_review_off"
            and llm_review.get("status") == "evaluation_ablation"
            and all(value == "evaluation_ablation" for value in outputs.values())
        )
        return {
            "mode_matches": ablated,
            "review_outputs": outputs,
            "verdicts_complete": ablated,
            "critic_review_sdk_logical_calls": len(review_calls),
            "critic_node_logical_calls": len(critic_node_calls),
            "call_records_complete": not critic_node_calls,
            "all_reviews_passed": None,
            "final_critic_approved": critic.get("status") == "approved",
            "review_contract_pass": bool(ablated and not review_calls and critic.get("status") == "approved"),
        }
        return {"mode_matches": False, "review_contract_pass": False, "error": "arm must be 'full' or 'off'"}


def score_provider_candidate_run_v1(
    trace: dict[str, Any], *, candidate: dict[str, Any], arm: str, replicate: int,
) -> dict[str, Any]:
    """Score one candidate × arm × replicate while retaining the v2 base score."""
    case_id = str(candidate.get("case_id") or "")
    expected_outcome = str((candidate.get("scoring_expectation") or {}).get("expected_outcome") or "")
    normalized_trace = dict(trace)
    normalized_trace.setdefault("workflow_state", dict(_state(trace)))
    base = score_diagnostic_run(
        normalized_trace, case_id=case_id, arm=arm, replicate=replicate,
        expected_outcome=expected_outcome,
    )
    state = _state(trace)
    expected_kind = "metric_decomposition" if expected_outcome == "numeric_decomposition" else "dimension_contribution"
    evidence = _expected_evidence(state, candidate, expected_kind)
    metric_ok, period_ok, scope_ok = _target_matches(state, candidate)
    scoring = candidate.get("scoring_expectation") or {}
    expected = scoring.get("values") or {}
    tolerance = float((candidate.get("tolerance") or {}).get("absolute", 0.03))
    unit = str(scoring.get("evidence_unit") or candidate.get("unit") or "")
    evidence_unit_ok = bool(evidence) and all(str(item.get("unit") or "") == unit for item in evidence)

    numeric_details: list[dict[str, Any]] = []
    for item in evidence:
        values = item.get("values") or {}
        comparisons = {
            "baseline": (values.get("baseline_value"), expected.get("baseline")),
            "current": (values.get("current_value"), expected.get("current")),
            "delta": (values.get("total_delta"), expected.get("delta")),
        }
        scalar_checks = {
            name: isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and isinstance(wanted, (int, float)) and not isinstance(wanted, bool)
            and abs(float(actual) - float(wanted)) <= tolerance
            for name, (actual, wanted) in comparisons.items()
        }
        actual_components = _actual_component_map(item, candidate)
        expected_components = scoring.get("components") or {}
        component_errors = {
            key: (None if actual_components is None or key not in actual_components else round(actual_components[key] - float(wanted), 10))
            for key, wanted in expected_components.items()
        }
        components_ok = bool(actual_components is not None) and all(
            error is not None and abs(error) <= tolerance for error in component_errors.values()
        )
        extra_nonzero = [] if actual_components is None else [
            key for key, value in actual_components.items()
            if key not in expected_components and abs(value) > tolerance
        ]
        numeric_details.append({
            "evidence_id": item.get("evidence_id"),
            "scalar_checks": scalar_checks,
            "actual_components": actual_components,
            "component_errors": component_errors,
            "components_match": components_ok and not extra_nonzero,
            "unexpected_nonzero_components": extra_nonzero,
        })
    scalar_correct = bool(numeric_details) and all(all(row["scalar_checks"].values()) for row in numeric_details)
    components_correct = bool(numeric_details) and all(row["components_match"] for row in numeric_details)
    numbers_correct = scalar_correct and components_correct
    evidence_valid = bool(evidence) and metric_ok and period_ok and scope_ok and evidence_unit_ok

    code = base.get("terminal_code")
    execution_errors = state.get("errors") or []
    execution_error_free = not execution_errors and code != "EXECUTION_FAILED"
    partial = base.get("presentation_status") == "partial" or base.get("trace_presentation_status") == "partial"
    critic = _review_result(state, arm)
    causal_review = (state.get("critic_result") or {}).get("causal_review") or {}
    causal_boundary_pass = causal_review.get("status") == "approved"
    workflow_result = trace.get("workflow_result") or {}
    answer = str(
        state.get("answer") or state.get("final_answer")
        or (workflow_result.get("answer") if isinstance(workflow_result, Mapping) else "")
        or ""
    ).strip()
    presentation_complete = (
        base.get("presentation_status") == "completed"
        and base.get("trace_presentation_status") == "completed"
    )
    valid_expected_analysis = evidence_valid and numbers_correct
    task_pass = bool(
        base.get("task_acceptance_pass")
        and valid_expected_analysis
        and execution_error_free
        and critic.get("review_contract_pass")
        and causal_boundary_pass
        and presentation_complete
        and bool(answer)
        and not partial
    )
    return {
        "scorer_version": PROVIDER_CANDIDATE_SCORER_V1,
        "base_scorer_version": DIAGNOSTIC_RUN_SCORER_VERSION,
        "case_id": case_id,
        "arm": arm,
        "replicate": replicate,
        "expected_outcome": expected_outcome,
        "terminal_code": code,
        "terminal_presentation_consistent": base.get("terminal_presentation_consistent"),
        "base_task_acceptance_pass": base.get("task_acceptance_pass"),
        "base_effective_diagnosis_completed": base.get("effective_diagnosis_completed"),
        "base_safe_stop_behavior_observed": base.get("safe_stop_behavior_observed"),
        "base_conclusion_evidence_valid": base.get("conclusion_evidence_valid"),
        "expected_analysis_evidence_count": len(evidence),
        "target_metric_matches": metric_ok,
        "scope_matches_candidate": scope_ok,
        "period_matches_candidate": period_ok,
        "evidence_unit_matches": evidence_unit_ok,
        "evidence_valid_for_candidate": evidence_valid,
        "numeric_values_correct": numbers_correct,
        "numeric_comparison_details": numeric_details,
        "effective_expected_result_observed": valid_expected_analysis,
        "task_completed": task_pass,
        "task_acceptance_pass": task_pass,
        "task_acceptance_denominator": 1,
        "partial_output": partial,
        "partial_complete_task_pass": False if partial else None,
        "execution_error_free": execution_error_free,
        "execution_errors": list(execution_errors),
        "answer_present": bool(answer),
        "presentation_completed": presentation_complete,
        "critic": critic,
        "unsupported_causal_language_review": causal_review.get("status"),
        "causal_boundary_pass": causal_boundary_pass,
        "raw_metrics": base.get("raw_metrics"),
    }


_NUMBER_RE = re.compile(r"(?<![\w.])([+\-−]?\s*\d[\d,]*(?:\.\d+)?)(?![\w.])")
_EVIDENCE_RE = re.compile(r"(?<![A-Za-z0-9])ev_[A-Za-z0-9_-]+")


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _period_matches(value: Any, start: str, end: str) -> bool:
    if isinstance(value, str):
        if start == end:
            return value == start or start in value
        return start in value and end in value
    if not isinstance(value, Mapping):
        return False
    observed_start = str(value.get("start_date") or value.get("start") or value.get("date") or "")
    observed_end = str(value.get("end_date") or value.get("end") or value.get("date") or "")
    return observed_start == start and observed_end == end


def _number_tokens(text: str) -> list[float]:
    values: list[float] = []
    for match in _NUMBER_RE.finditer(text.replace("−", "-")):
        token = match.group(1).replace(" ", "").replace(",", "")
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _near_alias_value(text: str, aliases: list[str], expected: float, tolerance: float) -> bool:
    folded = text.casefold()
    for alias in aliases:
        if not alias:
            continue
        needle = alias.casefold()
        start = 0
        while (pos := folded.find(needle, start)) >= 0:
            context = text[max(0, pos - 35): min(len(text), pos + len(alias) + 100)]
            if any(abs(value - expected) <= tolerance for value in _number_tokens(context)):
                return True
            start = pos + len(needle)
    return False


def _near_alias_value_with_unit(
    text: str, aliases: list[str], expected: float, tolerance: float, unit_aliases: list[str],
) -> bool:
    folded = text.casefold()
    for alias in aliases:
        if not alias:
            continue
        needle = alias.casefold()
        start = 0
        while (pos := folded.find(needle, start)) >= 0:
            context = text[max(0, pos - 35): min(len(text), pos + len(alias) + 120)]
            if (any(abs(value - expected) <= tolerance for value in _number_tokens(context))
                    and any(unit.casefold() in context.casefold() for unit in unit_aliases if unit)):
                return True
            start = pos + len(needle)
    return False


def _labeled_period_present(text: str, labels: list[str], tokens: list[str]) -> bool:
    folded = text.casefold()
    for label in labels:
        start = 0
        while (pos := folded.find(label.casefold(), start)) >= 0:
            context = text[max(0, pos - 30): min(len(text), pos + len(label) + 160)]
            if any(token and token.casefold() in context.casefold() for token in tokens):
                return True
            start = pos + len(label)
    return False


def _direction_text_matches(text: str, expected_delta: float, tolerance: float, unit_aliases: list[str]) -> bool:
    labels = ["变化", "变动", "change", "delta", "increase", "decrease", "增加", "上升", "下降", "减少"]
    folded = text.casefold()
    for label in labels:
        pos = folded.find(label.casefold())
        if pos < 0:
            continue
        context = text[max(0, pos - 30): min(len(text), pos + 100)]
        nums = _number_tokens(context)
        has_unit = any(unit and unit.casefold() in context.casefold() for unit in unit_aliases)
        if has_unit and any(abs(value - expected_delta) <= tolerance for value in nums):
            return True
        if has_unit and expected_delta > 0 and any(abs(abs(value) - expected_delta) <= tolerance for value in nums):
            if any(word in context for word in ("增加", "上升", "增长", "increase", "rose", "上漲")):
                return True
        if has_unit and expected_delta < 0 and any(abs(abs(value) - abs(expected_delta)) <= tolerance for value in nums):
            if any(word in context for word in ("下降", "减少", "降低", "decrease", "decline", "fell")):
                return True
    return False


def _output_evidence_ids(item: Mapping[str, Any] | None) -> set[str]:
    if not isinstance(item, Mapping):
        return set()
    raw = item.get("evidence_ids") or []
    return {str(value) for value in raw if value}


def _presentation_scope_matches(presentation: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    scope = presentation.get("scope")
    if not isinstance(scope, Mapping) or candidate.get("scope") != "platform" or scope.get("seller_id"):
        return False
    periods = _period_expectation(candidate)
    return (
        str(scope.get("start_date") or "") == periods.get("baseline_start")
        and str(scope.get("end_date") or "") == periods.get("current_end")
    )


def _structured_output_checks(
    presentation: Mapping[str, Any], candidate: Mapping[str, Any], valid_evidence_ids: set[str],
) -> dict[str, Any]:
    scoring = candidate.get("scoring_expectation") or {}
    out = candidate.get("output_expectation") or {}
    expected = scoring.get("values") or {}
    components = expected.get("components") or scoring.get("components") or {}
    tol = float(out.get("numeric_tolerance", candidate.get("tolerance", {}).get("absolute", 0.03)))
    expected_unit = str(out.get("metric_unit") or candidate.get("unit") or "")
    component_unit = str(out.get("component_unit") or scoring.get("evidence_unit") or expected_unit)
    metric = str(candidate.get("target_metric") or "").casefold()
    periods = _period_expectation(candidate)
    kind = str(scoring.get("expected_outcome") or "")

    if kind == "numeric_decomposition":
        card = presentation.get("metric_decomposition")
        if not isinstance(card, Mapping):
            return {"card_present": False, "card_matches": False, "failure_reasons": ["metric_decomposition card missing"]}
        factor_rows = card.get("factors") or []
        actual_factors = {
            str(row.get("metric_id")): row for row in factor_rows if isinstance(row, Mapping) and row.get("metric_id")
        }
        factor_values = {
            key: _as_float(actual_factors[key].get("contribution")) for key in components if key in actual_factors
        }
        expected_ids = set(valid_evidence_ids)
        cited = _output_evidence_ids(card)
        checks = {
            "scope": _presentation_scope_matches(presentation, candidate),
            "metric": str(card.get("metric_id") or "").casefold() == metric,
            "unit": card.get("unit") == expected_unit and card.get("change_unit") == expected_unit,
            "baseline": _as_float(card.get("baseline_value")) is not None and abs(float(card["baseline_value"]) - float(expected.get("baseline"))) <= tol,
            "current": _as_float(card.get("current_value")) is not None and abs(float(card["current_value"]) - float(expected.get("current"))) <= tol,
            "delta": _as_float(card.get("delta_value")) is not None and abs(float(card["delta_value"]) - float(expected.get("delta"))) <= tol,
            "baseline_period": _period_matches(card.get("baseline_period"), periods.get("baseline_start", ""), periods.get("baseline_end", "")),
            "current_period": _period_matches(card.get("current_period"), periods.get("current_start", ""), periods.get("current_end", "")),
            "method": str(card.get("method_id") or "") == str(scoring.get("method") or (candidate.get("registered_capability") or {}).get("method") or ""),
            "factor_set": set(actual_factors) == set(components),
            "factor_values": set(factor_values) == set(components) and all(abs(float(factor_values[key]) - float(value)) <= tol for key, value in components.items()),
            "factor_units": all(actual_factors[key].get("unit") == component_unit for key in components if key in actual_factors),
            "evidence_reference": bool(cited) and cited <= expected_ids and bool(expected_ids & cited),
            "descriptive_not_causal": card.get("analysis_nature") == "registered_numeric_decomposition" and card.get("business_cause_confirmed") is False,
        }
        return {"card_present": True, "kind": kind, "checks": checks, "card_matches": all(checks.values()), "failure_reasons": [key for key, passed in checks.items() if not passed]}

    cards = presentation.get("contributions") or []
    matching = [row for row in cards if isinstance(row, Mapping) and str(row.get("metric_name") or "").casefold() == metric]
    if len(matching) != 1:
        return {"card_present": bool(matching), "kind": kind, "card_matches": False, "failure_reasons": ["expected exactly one matching contribution card"]}
    card = matching[0]
    expected_dimension = str(out.get("dimension_id") or "")
    expected_items = set(components)
    actual_rows = [row for row in (card.get("items") or []) if isinstance(row, Mapping)]
    actual = {str(row.get("dimension_value")): row for row in actual_rows if row.get("dimension_value") is not None}
    cited = _output_evidence_ids(card)
    expected_ids = set(valid_evidence_ids)
    change_cards = [row for row in (presentation.get("changes") or []) if isinstance(row, Mapping) and str(row.get("metric_name") or "").casefold() == metric]
    change = change_cards[0] if len(change_cards) == 1 else {}
    checks = {
        "scope": _presentation_scope_matches(presentation, candidate),
        "dimension": str(card.get("dimension") or "") == expected_dimension,
        "analysis_nature": card.get("analysis_nature") == "descriptive_contribution",
        "method": str(card.get("method") or "") == str(scoring.get("method") or (candidate.get("registered_capability") or {}).get("method") or ""),
        "unit": card.get("unit") == component_unit,
        "change_card_present": len(change_cards) == 1,
        "metric_unit": bool(change) and change.get("unit") == expected_unit,
        "baseline": bool(change) and _as_float(change.get("baseline_value")) is not None and abs(float(change["baseline_value"]) - float(expected.get("baseline"))) <= tol,
        "current": bool(change) and _as_float(change.get("current_value")) is not None and abs(float(change["current_value"]) - float(expected.get("current"))) <= tol,
        "change_card_delta": bool(change) and _as_float(change.get("delta_value")) is not None and abs(float(change["delta_value"]) - float(expected.get("delta"))) <= tol,
        "delta": _as_float(card.get("total_delta")) is not None and abs(float(card["total_delta"]) - float(expected.get("delta"))) <= tol,
        "baseline_period": bool(change) and _period_matches(change.get("baseline_period"), periods.get("baseline_start", ""), periods.get("baseline_end", "")),
        "current_period": bool(change) and _period_matches(change.get("current_period"), periods.get("current_start", ""), periods.get("current_end", "")),
        "component_set": set(actual) == expected_items,
        "component_values": set(actual) == expected_items and all(
            _as_float(actual[key].get("delta_value")) is not None
            and abs(float(actual[key]["delta_value"]) - float(components[key])) <= tol
            for key in expected_items
        ),
        "component_unit": str(out.get("component_unit") or scoring.get("evidence_unit") or "") == str(scoring.get("evidence_unit") or ""),
        "evidence_reference": bool(cited) and cited <= expected_ids and bool(expected_ids & cited),
    }
    return {"card_present": True, "kind": kind, "checks": checks, "card_matches": all(checks.values()), "failure_reasons": [key for key, passed in checks.items() if not passed]}


def _text_output_checks(
    answer: str, candidate: Mapping[str, Any], valid_evidence_ids: set[str],
) -> dict[str, Any]:
    scoring = candidate.get("scoring_expectation") or {}
    out = candidate.get("output_expectation") or {}
    expected = scoring.get("values") or {}
    components = expected.get("components") or scoring.get("components") or {}
    tol = float(out.get("numeric_tolerance", candidate.get("tolerance", {}).get("absolute", 0.03)))
    text = answer.strip()
    labels = out.get("metric_labels") or []
    period_rules = out.get("period_text") or {}
    baseline_tokens = period_rules.get("baseline_text_tokens") or []
    current_tokens = period_rules.get("current_text_tokens") or []
    answer_citations = set(_EVIDENCE_RE.findall(text))
    # Citations are user-visible in the structured presentation card. Inline
    # answer citations are optional, but if present must resolve to this run's
    # candidate-valid evidence.
    checks: dict[str, bool] = {
        "nonempty": bool(text),
        "target_metric_named": any(label.casefold() in text.casefold() for label in labels if label),
        "baseline_period_named": _labeled_period_present(text, out.get("baseline_text_labels") or [], baseline_tokens),
        "current_period_named": _labeled_period_present(text, out.get("current_text_labels") or [], current_tokens),
        "evidence_reference_present": bool(valid_evidence_ids),
        "explicit_answer_citations_valid": not answer_citations or answer_citations <= valid_evidence_ids,
    }
    baseline_aliases = out.get("baseline_text_labels") or ["baseline", "基线", "基准期", "对比期"]
    current_aliases = out.get("current_text_labels") or ["current", "当前", "本期"]
    baseline = _as_float(expected.get("baseline"))
    current = _as_float(expected.get("current"))
    delta = _as_float(expected.get("delta"))
    unit_aliases = out.get("unit_text_aliases") or []
    checks["baseline_value_context"] = baseline is not None and _near_alias_value_with_unit(text, baseline_aliases, baseline, tol, unit_aliases)
    checks["current_value_context"] = current is not None and _near_alias_value_with_unit(text, current_aliases, current, tol, unit_aliases)
    checks["delta_value_and_direction"] = delta is not None and _direction_text_matches(text, delta, tol, unit_aliases)
    checks["unit_text_present"] = any(alias and alias.casefold() in text.casefold() for alias in unit_aliases)
    component_aliases = out.get("component_text_labels") or {}
    component_units = out.get("component_unit_text_aliases") or unit_aliases
    component_checks = {
        key: _near_alias_value_with_unit(text, list(component_aliases.get(key) or [key]), float(value), tol, component_units)
        for key, value in components.items()
    }
    checks["component_claims_contextualized"] = bool(component_checks) and all(component_checks.values())
    causal_review_required = bool(out.get("causal_boundary_required", True))
    # The separate deterministic causal-language review checks unsupported
    # causal claims; this scorer never demands causal proof for contributions.
    manual_review_required: list[str] = []
    folded_text = text.casefold()
    if labels and not any(label.casefold() in folded_text for label in labels if label):
        manual_review_required.append("target metric wording is not covered by registered display labels")
    if not any(label.casefold() in folded_text for label in (out.get("baseline_text_labels") or []) if label):
        manual_review_required.append("baseline/current wording cannot be identified deterministically")
    if not any(label.casefold() in folded_text for label in (out.get("current_text_labels") or []) if label):
        manual_review_required.append("current-period wording cannot be identified deterministically")
    if not any(label.casefold() in folded_text for label in ("变化", "变动", "change", "delta", "increase", "decrease", "增加", "上升", "下降", "减少")):
        manual_review_required.append("change direction wording is outside the deterministic vocabulary")
    for key, aliases in component_aliases.items():
        if not any(alias and alias.casefold() in folded_text for alias in aliases):
            manual_review_required.append(f"component wording for {key} is not covered by candidate labels")
    if not labels or not unit_aliases or set(component_aliases) != set(components):
        manual_review_required.append("candidate lacks complete registered display aliases for deterministic text checking")
    if not baseline_tokens or not current_tokens:
        manual_review_required.append("period wording cannot be mapped to baseline/current without candidate-specific aliases")
    automatable = all(checks.values()) and not manual_review_required
    return {
        "checks": checks,
        "component_checks": component_checks,
        "manual_review_required": manual_review_required,
        "manual_review_status": "required" if manual_review_required else "not_required_for_declared_checks",
        "text_matches_candidate": automatable,
        "cited_evidence_ids": sorted(answer_citations),
        "evidence_reference_source": "answer_text_and_presentation" if answer_citations else "structured_presentation_card",
        "causal_boundary_review_source": "deterministic_causal_language_review" if causal_review_required else None,
    }


def _presentation_from_trace(trace: Mapping[str, Any], state: Mapping[str, Any]) -> Mapping[str, Any]:
    presentation = state.get("analysis_presentation")
    if isinstance(presentation, Mapping):
        return presentation
    result = trace.get("workflow_result") or {}
    presentation = result.get("presentation") if isinstance(result, Mapping) else None
    return presentation if isinstance(presentation, Mapping) else {}


def score_provider_candidate_run(
    trace: dict[str, Any], *, candidate: dict[str, Any], arm: str, replicate: int,
) -> dict[str, Any]:
    """Version 2: add independent structured-card and user-answer fidelity checks."""
    result = score_provider_candidate_run_v1(trace, candidate=candidate, arm=arm, replicate=replicate)
    state = _state(trace)
    workflow_result = trace.get("workflow_result") or {}
    answer = str(
        state.get("answer") or state.get("final_answer")
        or (workflow_result.get("answer") if isinstance(workflow_result, Mapping) else "")
        or ""
    )
    presentation = _presentation_from_trace(trace, state)
    valid_ids = {
        str(item.get("evidence_id")) for item in _expected_evidence(
            state, candidate,
            "metric_decomposition" if (candidate.get("scoring_expectation") or {}).get("expected_outcome") == "numeric_decomposition" else "dimension_contribution",
        ) if item.get("evidence_id")
    }
    structured = _structured_output_checks(presentation, candidate, valid_ids)
    text = _text_output_checks(answer, candidate, valid_ids)
    output_quality_pass = bool(
        structured.get("card_matches") is True
        and text.get("text_matches_candidate") is True
        and not text.get("manual_review_required")
    )
    result.update({
        "scorer_version": PROVIDER_CANDIDATE_SCORER_VERSION,
        "v1_task_acceptance_pass": result.get("task_acceptance_pass"),
        "structured_output": structured,
        "final_answer_output": text,
        "output_quality_pass": output_quality_pass,
        "task_acceptance_pass": bool(result.get("task_acceptance_pass") and output_quality_pass),
        "task_completed": bool(result.get("task_acceptance_pass") and output_quality_pass),
        "output_quality_manual_review_required": bool(text.get("manual_review_required")),
    })
    return result

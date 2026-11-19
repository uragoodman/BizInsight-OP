"""Pre-registered B/G development comparison protocol.

This protocol is deliberately separate from the historical Critic full/off
candidate scorer. It validates the fixed six-case mix, creates a paired B/G
schedule, and scores a small, arm-neutral observation contract. It does not
start workflows or contact a Provider.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import random
import uuid
from typing import Any
import copy


PROTOCOL_VERSION = "resume-bg-score-v2"  # immutable schedule's version
SCORER_VERSION = "resume-bg-score-v3"
OFFLINE_SCORER_VERSION = "resume-bg-score-v4"
OFFLINE_SCORER_V5_VERSION = "resume-bg-score-v5"
EXPERIMENT_ID = "resume-bg-dev-ab-20261016-v1"
SEED = 20261016
ARMS = ("basic_tool_loop", "governed_workflow")
TASK_COUNTS = {
    "registered_numeric_decomposition": 2,
    "dimension_statistical_contribution": 2,
    "expected_data_insufficiency": 1,
    "expected_capability_limit": 1,
}


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_cases(cases: Sequence[Mapping[str, Any]]) -> None:
    if len(cases) != 6:
        raise ValueError("B/G protocol requires exactly six preselected cases.")
    ids = [str(case.get("case_id") or "") for case in cases]
    if any(not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("Case IDs must be nonempty and unique.")
    counts: dict[str, int] = {}
    for case in cases:
        kind = str(case.get("protocol_task_type") or "")
        counts[kind] = counts.get(kind, 0) + 1
    if counts != TASK_COUNTS:
        raise ValueError(f"Case mix mismatch: expected {TASK_COUNTS}, got {counts}.")
    metrics = {str(case.get("target_metric") or "") for case in cases}
    if not {"gmv", "avg_order_value", "freight_value", "freight_rate"}.issubset(metrics):
        raise ValueError("The fixed candidate mix must cover GMV, AOV, freight value and freight rate.")
    if sum(case.get("protocol_task_type") == "registered_numeric_decomposition" for case in cases) != 2:
        raise ValueError("Exactly two registered numeric decomposition cases are required.")
    if sum(case.get("protocol_task_type") == "dimension_statistical_contribution" for case in cases) != 2:
        raise ValueError("Exactly two dimension contribution cases are required.")
    # Blocked cases remain visible in the preflight set; they cannot be passed
    # off as executable or silently replaced with easier cases.


def build_manifest(cases: Sequence[Mapping[str, Any]], *, seed: int = SEED) -> dict[str, Any]:
    """Create six paired cases, balanced B/G first, with new deterministic IDs."""
    validate_cases(cases)
    rng = random.Random(seed)
    case_order = list(cases)
    rng.shuffle(case_order)
    first_arms = [ARMS[0]] * 3 + [ARMS[1]] * 3
    rng.shuffle(first_arms)
    namespace = uuid.uuid5(uuid.NAMESPACE_URL, EXPERIMENT_ID)
    entries: list[dict[str, Any]] = []
    for case, first in zip(case_order, first_arms):
        case_id = str(case["case_id"])
        pair_id = f"{case_id}-bg-r1"
        ordered = (first, ARMS[1] if first == ARMS[0] else ARMS[0])
        for arm in ordered:
            key = f"{EXPERIMENT_ID}:{case_id}:{arm}:1"
            entries.append({
                "case_id": case_id,
                "arm": arm,
                "replicate": 1,
                "pair_id": pair_id,
                "run_id": str(uuid.uuid5(namespace, key)),
                "schedule_index": len(entries) + 1,
                "status": "not_started",
                "provider_sends": 0,
                "score": None,
            })
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "seed": seed,
        "schedule_mode": "serial_paired; seeded case and arm order; three pairs start in each arm",
        "planned_units": 12,
        "started_units": 0,
        "provider_sends": 0,
        "arm_order_pair_counts": {
            arm: sum(entries[i]["arm"] == arm for i in range(0, 12, 2)) for arm in ARMS
        },
        "entries": entries,
    }


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _value_close(actual: Any, expected: Any, tolerance: float) -> bool:
    return (
        isinstance(actual, (int, float)) and not isinstance(actual, bool)
        and isinstance(expected, (int, float)) and not isinstance(expected, bool)
        and abs(float(actual) - float(expected)) <= tolerance
    )


def _component_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    result: dict[str, Any] = {}
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, Mapping):
                return {}
            key = item.get("component_id", item.get("metric_id", item.get("dimension_value", item.get("key"))))
            number = item.get("delta_value", item.get("contribution", item.get("delta", item.get("value"))))
            if key is None or str(key) in result:
                return {}
            result[str(key)] = number
    return result


def _common_observation(trace: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    state = next((trace.get(key) for key in ("workflow_state", "state_snapshot", "state")
                  if isinstance(trace.get(key), Mapping)), trace)
    workflow_result = _as_mapping(trace.get("workflow_result"))
    investigation = _as_mapping(state.get("investigation_state"))
    target = _as_mapping(investigation.get("target"))
    final = _as_mapping(state.get("final"))
    presentation = dict(_as_mapping(state.get("analysis_presentation")) or _as_mapping(state.get("presentation"))
                        or _as_mapping(workflow_result.get("presentation")) or _as_mapping(final.get("presentation")))
    # Adapt existing serialized contracts, without changing values or units.
    card = dict(_as_mapping(presentation.get("metric_decomposition") or presentation.get("decomposition")))
    if not card and case.get("protocol_task_type") == "registered_numeric_decomposition" and final.get("action") == "FINAL":
        card = {
            "baseline_value": presentation.get("baseline_value"),
            "current_value": presentation.get("current_value"),
            "total_delta": presentation.get("delta"),
            "components": presentation.get("items"),
            "unit": presentation.get("unit"),
            "change_unit": presentation.get("change_unit", presentation.get("unit")),
            "component_unit": presentation.get("component_unit", presentation.get("unit")),
        }
        # This legacy flat B card declares one unit for its values. Percentage
        # differences cannot borrow % as a pp unit and remain a failed check.
        if not state.get("errors"):
            presentation.setdefault("status", "completed")
    if card:
        if "total_delta" not in card:
            card["total_delta"] = card.get("delta_value")
        if "components" not in card:
            card["components"] = card.get("factors")
        factors = card.get("factors") or []
        factor_units = {item.get("unit") for item in factors if isinstance(item, Mapping)}
        if "component_unit" not in card and len(factor_units) == 1:
            card["component_unit"] = next(iter(factor_units))
        presentation["metric_decomposition"] = card
    answer = str(state.get("answer") or state.get("final_answer") or workflow_result.get("answer") or final.get("answer") or "")
    evidence_ids: set[str] = set()
    evidence_rows = investigation.get("evidence") or state.get("evidence_bundles") or []
    if isinstance(evidence_rows, list):
        for item in evidence_rows:
            if isinstance(item, Mapping) and item.get("evidence_id"):
                evidence_ids.add(str(item["evidence_id"]))
    # A G trace has typed evidence linked to completed, valid actions. A B
    # trace has public-tool envelopes linked by request ID. Neither arm passes
    # merely because an arbitrary evidence ID exists.
    valid_evidence_ids: set[str] = set()
    if investigation.get("evidence"):
        try:
            from evaluation.provider_candidate_protocol import _expected_evidence, _validated_typed_evidence
            task_kind = str(case.get("protocol_task_type") or "")
            if task_kind in {"registered_numeric_decomposition", "dimension_statistical_contribution"}:
                expected_kind = "metric_decomposition" if task_kind == "registered_numeric_decomposition" else "dimension_contribution"
                valid_evidence_ids = {
                    str(item.get("evidence_id"))
                    for item in _expected_evidence(dict(state), dict(case), expected_kind)
                    if item.get("evidence_id")
                }
            else:
                valid_evidence_ids = set(_validated_typed_evidence(dict(state)))
        except Exception:
            valid_evidence_ids = set()
        # The shared investigation engine is the authoritative validation
        # boundary for governed-workflow Evidence. Retain evidence IDs only
        # when the engine ledger says the associated action completed and its
        # validation passed, and the Evidence matches the current target and
        # scope. This avoids confusing a protocol-specific extractor mismatch
        # with invalid business evidence.
        engine_target_scope = _as_mapping(target.get("canonical_scope"))
        engine_scope_fingerprint = str(
            engine_target_scope.get("scope_fingerprint")
            or target.get("scope_fingerprint") or ""
        )
        completed_engine_evidence = {
            str(evidence_id)
            for action_result in (investigation.get("action_results") or [])
            if isinstance(action_result, Mapping)
            and action_result.get("status") == "completed"
            and _as_mapping(action_result.get("validation_result")).get("status") == "valid"
            for evidence_id in (action_result.get("evidence_ids") or [])
        }
        for item in investigation.get("evidence") or []:
            if not isinstance(item, Mapping):
                continue
            evidence_id = str(item.get("evidence_id") or "")
            if (
                evidence_id in completed_engine_evidence
                and str(item.get("scope_fingerprint") or "") == engine_scope_fingerprint
            ):
                valid_evidence_ids.add(evidence_id)
    else:
        calls_by_request = {
            str(item.get("request_id")): item
            for item in (state.get("tool_calls") or [])
            if isinstance(item, Mapping) and item.get("request_id")
        }
        for item in state.get("evidence_bundles") or []:
            if not isinstance(item, Mapping) or not item.get("evidence_id"):
                continue
            request_id = str(item.get("request_id") or "")
            content = _as_mapping(item.get("content"))
            result = _as_mapping(content.get("result"))
            parsed = _as_mapping(result.get("parsed"))
            call = calls_by_request.get(request_id, {})
            if (
                content.get("tool_name") == "registered_numeric_decomposition"
                and request_id and content.get("request_id") == request_id
                and call.get("call_status") == "succeeded"
                and call.get("response") == content
                and result.get("status") == "ok"
                and result.get("target_metric_id") == case.get("target_metric")
                and parsed.get("metric_name") == case.get("target_metric")
                and _as_mapping(parsed.get("authorized_metric")).get("metric_id") == case.get("target_metric")
            ):
                sources = content.get("source_queries") or []
                checked_ids = set()
                expected_period = _as_mapping(case.get("comparison_period"))
                allowed_ids = set(_as_mapping(case.get("same_identity_qualification")).get("authorized_dependency_metric_ids") or [])
                allowed_ids.add(case.get("target_metric"))
                for source in sources:
                    envelope = _as_mapping(source.get("envelope"))
                    child_result = _as_mapping(envelope.get("result"))
                    child_plan = _as_mapping(child_result.get("parsed"))
                    child_request = source.get("request_id")
                    if (
                        source.get("metric_id") in allowed_ids
                        and envelope.get("request_id") == child_request
                        and child_request == f"{request_id}:input:{source.get('metric_id')}"
                        and source.get("evidence_id") == f"{request_id}:source:{source.get('metric_id')}"
                        and child_result.get("status") == "ok"
                        and child_plan.get("metric_name") == source.get("metric_id")
                        and _as_mapping(child_plan.get("authorized_metric")).get("metric_id") == source.get("metric_id")
                        and (child_plan.get("start_date"), child_plan.get("end_date")) == (expected_period.get("baseline"), expected_period.get("current"))
                    ):
                        checked_ids.add(source.get("evidence_id"))
                declared_ids = set(result.get("source_evidence_ids") or [])
                if sources and len(checked_ids) == len(sources) and checked_ids == declared_ids:
                    valid_evidence_ids.update(checked_ids)
                    valid_evidence_ids.add(str(item["evidence_id"]))
                continue
            metric_id = str(parsed.get("metric_name") or "")
            allowed_metrics = {str(case.get("target_metric") or "")}
            qualification = _as_mapping(case.get("same_identity_qualification"))
            allowed_metrics.update(str(item) for item in qualification.get("authorized_dependency_metric_ids", []) if item)
            period = _as_mapping(case.get("comparison_period"))
            baseline, current = period.get("baseline"), period.get("current")
            period_matches = True
            if isinstance(baseline, str) and isinstance(current, str):
                observed_pair = (parsed.get("baseline_date"), parsed.get("current_date"))
                observed_range = (parsed.get("start_date"), parsed.get("end_date"))
                period_matches = observed_pair == (baseline, current) or observed_range == (baseline, current)
            elif isinstance(baseline, Mapping) and isinstance(current, Mapping):
                expected_ranges = (
                    str(baseline.get("start") or baseline.get("start_date") or ""),
                    str(baseline.get("end") or baseline.get("end_date") or ""),
                    str(current.get("start") or current.get("start_date") or ""),
                    str(current.get("end") or current.get("end_date") or ""),
                )
                observed_ranges = tuple(str(parsed.get(key) or "") for key in (
                    "baseline_start", "baseline_end", "current_start", "current_end",
                ))
                period_matches = observed_ranges == expected_ranges
            dimension_matches = (
                not case.get("dimension") or str(parsed.get("dimension") or "") == str(case.get("dimension"))
            )
            limitation = case.get("expected_limitation")
            limitation_matches = not limitation or str(parsed.get("dimension") or "") == str(limitation)
            if (
                request_id and str(content.get("request_id") or "") == request_id
                and call.get("call_status") == "succeeded"
                and result.get("status") in {"ok", "not_found", "unsupported"}
                and parsed.get("authorized_metric")
                and metric_id in allowed_metrics
                and period_matches and dimension_matches and limitation_matches
            ):
                valid_evidence_ids.add(str(item["evidence_id"]))
    claims = final.get("claims") or state.get("claims") or []
    claim_ids: set[str] = set()
    if isinstance(claims, list):
        for claim in claims:
            if isinstance(claim, Mapping):
                claim_ids.update(str(item) for item in claim.get("evidence_ids", []) if item)
                claim_ids.update(str(item) for item in claim.get("source_evidence_ids", []) if item)
    analysis = _as_mapping(state.get("analysis_result"))
    finish = _as_mapping(state.get("finish_decision"))
    for container in (analysis, finish):
        claim_ids.update(str(item) for item in (container.get("evidence_ids") or []) if item)
        claim_ids.update(str(item) for item in (container.get("supporting_evidence_ids") or []) if item)
        for contribution in container.get("contribution") or []:
            if isinstance(contribution, Mapping):
                claim_ids.update(str(item) for item in (contribution.get("evidence_ids") or []) if item)
    terminal = (
        state.get("terminal_code") or state.get("stop_code")
        or investigation.get("stop_reason") or state.get("stop_reason")
    )
    if isinstance(terminal, Mapping):
        terminal = terminal.get("code")
    status = presentation.get("status")
    expected_terminal = str(case.get("expected_terminal_code") or "").upper()
    limitation_record_valid = False
    for record in state.get("decision_records") or []:
        if not isinstance(record, Mapping):
            continue
        code = str(record.get("limitation_code") or "").upper()
        evidence_id = str(record.get("evidence_id") or "")
        identity_matches = (
            str(record.get("metric_id") or "") == str(case.get("target_metric") or "")
            and str(record.get("scope") or "") == str(case.get("scope") or "")
            and bool(record.get("authorization_checked"))
        )
        if code != expected_terminal or not identity_matches:
            continue
        if code == "CAPABILITY_GAP":
            limitation_record_valid = (
                record.get("sql_executed") is False
                and bool(record.get("registry_capability_checked"))
                and (not case.get("dimension") or str(record.get("dimension") or "") == str(case.get("dimension")))
            )
        elif code in {"DATA_INSUFFICIENT", "INVALID_DATA"}:
            limitation_record_valid = (
                record.get("sql_executed") is True
                and bool(evidence_id)
                and evidence_id in valid_evidence_ids
            )
        break
    # A safe-stop may be supported by a bound, authorized query showing that
    # the requested data is absent/incomplete. Keep this separate from valid
    # analytical evidence: it can support abstention, never numeric diagnosis.
    valid_limitation_evidence_ids: set[str] = set()
    calls_by_evidence = {
        str(item.get("evidence_id")): item
        for item in (state.get("tool_calls") or [])
        if isinstance(item, Mapping) and item.get("evidence_id")
    }
    calls_by_request = {
        str(item.get("request_id")): item
        for item in (state.get("tool_calls") or [])
        if isinstance(item, Mapping) and item.get("request_id")
    }
    expected_period = case.get("comparison_period") or {}
    expected_current = expected_period.get("current")
    for bundle in state.get("evidence_bundles") or []:
        if not isinstance(bundle, Mapping) or not bundle.get("evidence_id"):
            continue
        evidence_id = str(bundle.get("evidence_id"))
        content = _as_mapping(bundle.get("content"))
        result = _as_mapping(content.get("result"))
        parsed = _as_mapping(result.get("parsed"))
        request_id = str(bundle.get("request_id") or content.get("request_id") or "")
        call = calls_by_evidence.get(evidence_id) or calls_by_request.get(request_id, {})
        if (
            not request_id
            or str(call.get("request_id") or "") != request_id
            or (call.get("evidence_id") and str(call.get("evidence_id")) != evidence_id)
            or str(call.get("tool_name") or "") != str(bundle.get("source_ref") or call.get("tool_name") or "")
            or parsed.get("metric_name") != case.get("target_metric")
            or not parsed.get("authorized_metric")
        ):
            continue
        scope_context = _as_mapping(state.get("authorization_context"))
        observed_scope = state.get("scope_type") or scope_context.get("scope_type")
        if observed_scope and str(observed_scope) != str(case.get("scope") or ""):
            continue
        result_status = str(result.get("status") or "").casefold()
        indicates_gap = result_status in {"not_found", "incomplete", "insufficient_data"}
        if not indicates_gap and isinstance(result.get("trend_result"), Mapping):
            points = result["trend_result"].get("points")
            indicates_gap = isinstance(points, list) and any(
                isinstance(point, Mapping) and point.get("value") is None for point in points
            )
        if not indicates_gap:
            continue
        if isinstance(expected_current, str):
            point_date = str(parsed.get("metric_date") or "")[:10]
            start_date = str(parsed.get("start_date") or "")[:10]
            end_date = str(parsed.get("end_date") or "")[:10]
            date_matches = point_date == expected_current or (
                start_date and end_date and start_date <= expected_current <= end_date
            )
            if not date_matches:
                continue
        elif isinstance(expected_current, Mapping):
            expected_ranges = (
                str(expected_period.get("baseline", {}).get("start") or expected_period.get("baseline", {}).get("start_date") or ""),
                str(expected_period.get("baseline", {}).get("end") or expected_period.get("baseline", {}).get("end_date") or ""),
                str(expected_current.get("start") or expected_current.get("start_date") or ""),
                str(expected_current.get("end") or expected_current.get("end_date") or ""),
            )
            observed_ranges = tuple(str(parsed.get(key) or "") for key in (
                "baseline_start", "baseline_end", "current_start", "current_end",
            ))
            if observed_ranges != expected_ranges:
                continue
        valid_limitation_evidence_ids.add(evidence_id)
    cited_limitation_evidence = bool(claim_ids) and claim_ids.issubset(valid_evidence_ids | valid_limitation_evidence_ids)
    return {
        "state": state,
        "investigation": investigation,
        "target": target,
        "presentation": presentation,
        "answer": answer,
        "terminal_code": str(terminal or "").upper() or None,
        "presentation_status": str(status or "").lower() or None,
        "valid_evidence_ids": valid_evidence_ids,
        "observed_evidence_ids": evidence_ids,
        "cited_evidence_ids": claim_ids,
        "claims_bound": bool(claim_ids) and claim_ids.issubset(valid_evidence_ids | valid_limitation_evidence_ids),
        "limitation_record_valid": limitation_record_valid,
        "valid_limitation_evidence_ids": valid_limitation_evidence_ids,
        "limitation_evidence_valid": bool(valid_limitation_evidence_ids) and cited_limitation_evidence,
        "errors": state.get("errors") or [],
    }


def score_bg_run(
    case: Mapping[str, Any], *, arm: str, trace: Mapping[str, Any], lifecycle_status: str,
    human_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a B or G trace without using Critic approval as task truth.

    The observation is intentionally conservative: unsupported answer text is
    marked for human review. A structurally valid card or ``completed`` status
    alone cannot produce an automatic task pass.
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown B/G arm: {arm}")
    obs = _common_observation(trace, case)
    presentation = obs["presentation"]
    target = obs["target"]
    expected_period = case.get("comparison_period") or {}
    planning = _as_mapping(obs["state"].get("metric_planning_decision"))
    confirmed = planning.get("confirmed_metrics") or []
    metric_ok = str(presentation.get("metric_id") or target.get("metric_id")
                    or (confirmed[0] if len(confirmed) == 1 else "")) == str(case.get("target_metric") or "")
    scope = target.get("canonical_scope") or {}
    presentation_scope = presentation.get("scope")
    observed_scope = (
        scope.get("scope_type") or scope.get("scope")
        or (_as_mapping(presentation_scope).get("scope_type") if isinstance(presentation_scope, Mapping) else presentation_scope)
        or _as_mapping(obs["state"].get("authorization_context")).get("scope_type")
        or obs["state"].get("scope_type")
    )
    scope_ok = str(observed_scope or "") == str(case.get("scope") or "")
    baseline = expected_period.get("baseline")
    current = expected_period.get("current")
    period_ok = True
    if isinstance(baseline, str) and isinstance(current, str):
        period_ok = (
            str(presentation.get("baseline_date") or target.get("baseline_start") or "") == baseline
            and str(presentation.get("current_date") or target.get("current_start") or "") == current
        )
    elif isinstance(baseline, Mapping) and isinstance(current, Mapping):
        period_ok = all((
            str(presentation.get("baseline_start") or target.get("baseline_start") or "") == str(baseline.get("start") or baseline.get("start_date") or ""),
            str(presentation.get("baseline_end") or target.get("baseline_end") or "") == str(baseline.get("end") or baseline.get("end_date") or ""),
            str(presentation.get("current_start") or target.get("current_start") or "") == str(current.get("start") or current.get("start_date") or ""),
            str(presentation.get("current_end") or target.get("current_end") or "") == str(current.get("end") or current.get("end_date") or ""),
        ))
    expected_presentations = {
        "EXECUTION_FAILED": "failed",
        "INVALID_DATA": "partial",
        "DATA_INSUFFICIENT": "partial",
        "CAPABILITY_GAP": {"partial", "unsupported"},
        "AUTHORIZATION_BLOCKED": "partial",
        "BUDGET_EXHAUSTED": "partial",
    }.get(obs["terminal_code"])
    if isinstance(expected_presentations, str):
        expected_presentations = {expected_presentations}
    status_consistent = (
        True if expected_presentations is None or obs["presentation_status"] is None
        else obs["presentation_status"] in expected_presentations
    )
    evidence_valid = bool(obs["valid_evidence_ids"]) and obs["claims_bound"]
    execution_error_free = not obs["errors"] and obs["terminal_code"] != "EXECUTION_FAILED"
    result = _as_mapping(case.get("expected_result"))
    task_kind = str(case.get("protocol_task_type") or "")
    # A pre-query capability refusal has no executed time window. Verify the
    # exact requested task instead; never pretend it queried the expected dates.
    if task_kind == "expected_capability_limit" and obs["limitation_record_valid"]:
        if obs["state"].get("user_message") and obs["state"]["user_message"] == case.get("user_question"):
            period_ok = True
    tolerance = float(_as_mapping(case.get("tolerance")).get("absolute") or 0.03)
    numeric_checks: dict[str, bool] = {}
    effective_diagnostic = False
    if task_kind == "registered_numeric_decomposition":
        card = _as_mapping(presentation.get("metric_decomposition") or presentation.get("decomposition"))
        numeric_checks = {
            name: _value_close(card.get(actual), result.get(expected), tolerance)
            for name, actual, expected in (
                ("baseline", "baseline_value", "baseline"),
                ("current", "current_value", "current"),
                ("delta", "total_delta", "delta"),
            )
        }
        expected_components = _component_mapping(result.get("components") or result.get("display_components"))
        actual_components = _component_mapping(card.get("display_components") or card.get("components"))
        units = _as_mapping(case.get("output_units"))
        expected_metric_unit = str(units.get("metric_unit") or case.get("unit") or "")
        if expected_components:
            numeric_checks["components"] = (
                set(actual_components) == set(expected_components)
                and all(_value_close(actual_components.get(k), v, tolerance) for k, v in expected_components.items())
            )
        numeric_checks["metric_unit"] = str(card.get("metric_unit") or card.get("unit") or presentation.get("metric_unit") or presentation.get("unit") or "") == expected_metric_unit
        numeric_checks["change_unit"] = str(card.get("change_unit") or presentation.get("change_unit") or "") == str(units.get("change_unit") or expected_metric_unit)
        if expected_components:
            numeric_checks["component_unit"] = str(card.get("component_unit") or presentation.get("component_unit") or "") == str(units.get("component_unit") or expected_metric_unit)
        effective_diagnostic = bool(numeric_checks) and all(numeric_checks.values()) and evidence_valid
    elif task_kind == "dimension_statistical_contribution":
        contribution = _as_mapping(presentation.get("dimension_contribution") or presentation.get("contribution"))
        expected_contribution_value = _as_mapping(result.get("expected_contributions"))
        expected_components = _component_mapping(
            expected_contribution_value.get("group_components") or expected_contribution_value
        )
        actual_components = _component_mapping(contribution.get("contributions") or contribution.get("items"))
        units = _as_mapping(case.get("output_units"))
        numeric_checks = {
            "baseline": _value_close(contribution.get("baseline_value"), result.get("baseline_metric"), tolerance),
            "current": _value_close(contribution.get("current_value"), result.get("current_metric"), tolerance),
            "delta": _value_close(contribution.get("total_delta"), result.get("delta"), tolerance),
            "components": set(actual_components) == set(expected_components) and all(
                _value_close(actual_components.get(key), value, tolerance) for key, value in expected_components.items()
            ),
            "metric_unit": str(contribution.get("metric_unit") or contribution.get("unit") or presentation.get("metric_unit") or presentation.get("unit") or "") == str(units.get("metric_unit") or case.get("unit") or ""),
            "change_unit": str(contribution.get("change_unit") or presentation.get("change_unit") or "") == str(units.get("change_unit") or units.get("metric_unit") or case.get("unit") or ""),
            "component_unit": str(contribution.get("component_unit") or presentation.get("component_unit") or "") == str(units.get("component_unit") or units.get("change_unit") or case.get("unit") or ""),
        }
        effective_diagnostic = all(numeric_checks.values()) and evidence_valid
    expected_stop = case.get("expected_terminal_code")
    safe_stop_statuses = {"partial", "unsupported"} if obs["terminal_code"] == "CAPABILITY_GAP" else {"partial"}
    safe_stop_support_valid = (
        obs["limitation_evidence_valid"]
        if task_kind == "expected_data_insufficiency"
        else obs["limitation_record_valid"] if task_kind == "expected_capability_limit"
        else evidence_valid
    )
    correct_safe_stop = bool(
        task_kind in {"expected_data_insufficiency", "expected_capability_limit"}
        and expected_stop and obs["terminal_code"] == expected_stop
        and obs["presentation_status"] in safe_stop_statuses
        and safe_stop_support_valid and status_consistent
    )
    critic_result = _as_mapping(obs["state"].get("critic_result"))
    critic = _as_mapping(critic_result.get("llm_review"))
    review_required = bool(case.get("critic_review_required_for_governed_arm"))
    review_applicable = arm == "governed_workflow" and (review_required or (bool(critic_result) and critic.get("status") != "not_applicable"))
    review_names = ("groundedness", "completeness", "consistency")
    review_records = {name: _as_mapping(critic.get(name)) for name in review_names}
    review_complete = None if not review_applicable else all(
        review_records[name].get("verdict") in {"passed", "failed"}
        for name in review_names
    )
    limitation_contract_verified = case.get("readiness") != "blocked"
    human_review = _as_mapping(human_review)
    manual_review_required = bool(obs["answer"].strip()) and human_review.get("decision") not in {"pass", "fail"}
    text_output_correct = human_review.get("decision") == "pass"
    # Natural-language truthfulness remains a blinded human check. No
    # classifier or Critic verdict is used as a substitute oracle.
    task_acceptance_pass = bool(
        lifecycle_status == "completed" and limitation_contract_verified
        and status_consistent and execution_error_free
        and (evidence_valid or (task_kind == "expected_data_insufficiency" and obs["limitation_evidence_valid"])
                                or (task_kind == "expected_capability_limit" and obs["limitation_record_valid"]))
        and bool(obs["answer"].strip()) and metric_ok and scope_ok and period_ok and text_output_correct and not manual_review_required
        and ((task_kind in {"registered_numeric_decomposition", "dimension_statistical_contribution"}
              and effective_diagnostic and obs["presentation_status"] == "completed")
             or (task_kind in {"expected_data_insufficiency", "expected_capability_limit"} and correct_safe_stop))
        and (not review_applicable or review_complete is True)
    )
    return {
        "scorer_version": SCORER_VERSION,
        "case_id": case.get("case_id"), "arm": arm,
        "lifecycle_status": lifecycle_status,
        "task_acceptance_pass": task_acceptance_pass,
        "task_acceptance_status": "passed" if task_acceptance_pass else "pending_manual_review" if manual_review_required and effective_diagnostic else "failed",
        "task_acceptance_denominator": 1,
        "effective_diagnostic_completed": effective_diagnostic,
        "correct_safe_stop": correct_safe_stop,
        "expected_limitation_contract_verified": limitation_contract_verified,
        "terminal_code": obs["terminal_code"],
        "presentation_status": obs["presentation_status"],
        "status_consistent": status_consistent,
        "metric_matches": metric_ok, "scope_matches": scope_ok, "period_matches": period_ok,
        "evidence_valid": evidence_valid, "execution_error_free": execution_error_free,
        "numeric_checks": numeric_checks,
        "critic_review_applicable": review_applicable,
        "critic_review_record": dict(critic) if critic else None,
        "critic_review_complete": review_complete,
        "manual_review_required": manual_review_required,
        "human_review_decision": human_review.get("decision"),
        "task_pass_reason": (
            "manual_review_required" if manual_review_required
            else None if task_acceptance_pass else "one_or_more_pre_registered_checks_failed_or_unknown"
        ),
    }


def score_bg_run_v4(
    case: Mapping[str, Any], *, arm: str, trace: Mapping[str, Any], lifecycle_status: str,
    human_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Opt-in offline schema adaptation; v3 execution and old results stay intact.

    Normalize only one explicitly identified ContributionCard. Keep the old
    exact group coverage requirement, tolerance, and independent text review.
    Other is reported as an omitted-group net aggregate, never as full coverage.
    """
    adapted = copy.deepcopy(dict(trace))
    state = next((adapted[key] for key in ('workflow_state', 'state_snapshot', 'state')
                  if isinstance(adapted.get(key), Mapping)), adapted)
    task = case.get('protocol_task_type')
    presentation = state.get('analysis_presentation') or state.get('presentation') or {}
    selected = None
    adaptation = 'not_applicable'
    if task == 'dimension_statistical_contribution' and isinstance(presentation, Mapping) and 'contributions' in presentation:
        presentation = copy.deepcopy(dict(presentation))
        raw_cards = presentation.get('contributions')
        cards = [c for c in raw_cards if isinstance(c, Mapping)
                 and c.get('metric_name') == case.get('target_metric')
                 and c.get('dimension') == case.get('dimension')] if isinstance(raw_cards, list) else []
        if len(cards) != 1:
            adaptation = 'missing_or_ambiguous_matching_card'
        elif presentation.get('dimension_contribution') or presentation.get('contribution'):
            adaptation = 'conflicting_legacy_and_list_representations'
        else:
            adaptation = 'formal_contribution_card_list'
            selected = copy.deepcopy(dict(cards[0]))
            declared_unit = selected.get('unit')
            # The monetary/count ContributionCard has a shared unit. Percent
            # levels cannot lend % to absolute percentage-point contributions.
            if declared_unit and declared_unit != '%':
                selected.setdefault('change_unit', declared_unit)
                selected.setdefault('component_unit', declared_unit)
            units = _as_mapping(case.get('output_units'))
            for field in ('change_unit', 'component_unit'):
                if selected.get(field) in {'percentage_points', 'percentage_point', 'pp', '百分点'} and units.get(field) in {'percentage_points', 'percentage_point', 'pp', '百分点'}:
                    selected[field] = units[field]
            presentation['dimension_contribution'] = selected
            presentation['metric_id'] = selected['metric_name']
            for role in ('baseline', 'current'):
                period = selected.get(role + '_period')
                if isinstance(period, Mapping):
                    presentation[role + '_start'] = period.get('start_date')
                    presentation[role + '_end'] = period.get('end_date')
                if selected.get(role + '_date'):
                    presentation[role + '_date'] = selected[role + '_date']
            state['analysis_presentation'] = presentation
    score = score_bg_run(case, arm=arm, trace=adapted, lifecycle_status=lifecycle_status, human_review=human_review)
    score['scorer_version'] = OFFLINE_SCORER_VERSION
    score['parent_scorer_version'] = SCORER_VERSION
    score['schema_adaptation'] = adaptation
    if adaptation in {'missing_or_ambiguous_matching_card', 'conflicting_legacy_and_list_representations'}:
        score.update(task_acceptance_pass=False, task_acceptance_status='failed', effective_diagnostic_completed=False,
                     task_pass_reason=adaptation)
    if selected is not None:
        obs = _common_observation(adapted, case)
        references = selected.get('evidence_ids') or []
        card_bound = bool(references) and set(references).issubset(obs['valid_evidence_ids'])
        score['contribution_card_evidence_valid'] = card_bound
        if not card_bound:
            score.update(task_acceptance_pass=False, task_acceptance_status='failed', effective_diagnostic_completed=False,
                         evidence_valid=False, task_pass_reason='contribution_card_evidence_invalid')
        expected_result = _as_mapping(case.get('expected_result'))
        expected_raw = _as_mapping(expected_result.get('expected_contributions'))
        expected = _component_mapping(expected_raw.get('group_components') or expected_raw)
        actual = _component_mapping(selected.get('items') or selected.get('contributions'))
        tol = float(_as_mapping(case.get('tolerance')).get('absolute') or 0.03)
        shown = set(actual) - ({'Other'} if 'Other' not in expected else set())
        omitted = set(expected) - shown
        known_values_correct = bool(actual) and shown.issubset(expected) and all(
            _value_close(actual[k], expected[k], tol) for k in shown)
        aggregate_correct = ('Other' not in actual or 'Other' in expected or
                             _value_close(actual['Other'], sum(expected[k] for k in omitted), tol))
        score['dimension_coverage'] = {
            'expected_group_count': len(expected), 'displayed_named_group_count': len(shown),
            'omitted_group_count': len(omitted), 'has_omitted_net_aggregate': 'Other' in actual and 'Other' not in expected,
            'displayed_net_values_match': known_values_correct,
            'omitted_net_aggregate_matches': aggregate_correct,
            'exact_group_coverage': set(actual) == set(expected),
            'full_task_coverage_required': True,
        }
    score['automated_failure_reasons'] = [
        key for key in ('metric_matches', 'scope_matches', 'period_matches', 'evidence_valid',
                        'execution_error_free', 'status_consistent') if score.get(key) is False
    ] + ['numeric_checks.' + key for key, value in score.get('numeric_checks', {}).items() if not value]
    if score.get('critic_review_applicable') and score.get('critic_review_complete') is not True:
        score['automated_failure_reasons'].append('critic_review_incomplete')
    if adaptation in {'missing_or_ambiguous_matching_card', 'conflicting_legacy_and_list_representations'}:
        score['automated_failure_reasons'].append(adaptation)
    score['human_text_review_status'] = 'pending' if score['manual_review_required'] else score.get('human_review_decision')
    return score


def _v5_registered_factor_aliases(case: Mapping[str, Any], trace: Mapping[str, Any]) -> tuple[dict[str, str], str | None]:
    """Resolve public ratio-role labels from the registered method, never case key order."""
    task = str(case.get("protocol_task_type") or "")
    if task != "registered_numeric_decomposition":
        return {}, None
    state = next((trace.get(key) for key in ("workflow_state", "state_snapshot", "state")
                  if isinstance(trace.get(key), Mapping)), trace)
    method = str(_as_mapping(case.get("registered_capability")).get("method") or "")
    for call in state.get("tool_calls") or []:
        if not isinstance(call, Mapping):
            continue
        response = _as_mapping(call.get("response"))
        result = _as_mapping(response.get("result"))
        method = str(result.get("method_id") or result.get("method") or method)
        if method:
            break
    if method != "symmetric_ratio_v1":
        return {}, method or None
    metric = str(case.get("target_metric") or "")
    try:
        from app.metrics.registry import get_registry
        registered = get_registry().get(metric)
        contract = registered.period_aggregation
        numerator = str(contract.numerator_metric or "")
        denominator = str(contract.denominator_metric or "")
    except Exception:
        return {}, method
    if not numerator or not denominator or numerator == denominator:
        return {}, method
    return {"numerator_effect": numerator, "denominator_effect": denominator}, method


def _v5_prepare_trace(case: Mapping[str, Any], trace: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Adapt only explicit serialized aliases for the opt-in v5 scorer."""
    adapted = copy.deepcopy(dict(trace))
    state = next((adapted.get(key) for key in ("workflow_state", "state_snapshot", "state")
                  if isinstance(adapted.get(key), Mapping)), adapted)
    aliases, method = _v5_registered_factor_aliases(case, adapted)
    if aliases:
        workflow_result = adapted.get("workflow_result") if isinstance(adapted.get("workflow_result"), dict) else {}
        for container in (state.get("presentation"), state.get("analysis_presentation"),
                          _as_mapping(state.get("final")).get("presentation"), workflow_result.get("presentation")):
            if not isinstance(container, Mapping):
                continue
            for field in ("items", "components", "factors"):
                for row in container.get(field) or []:
                    if isinstance(row, dict) and row.get("key") in aliases:
                        row["key"] = aliases[row["key"]]
            for card_field in ("metric_decomposition", "decomposition"):
                card = container.get(card_field)
                if isinstance(card, dict):
                    for field in ("items", "components", "factors"):
                        for row in card.get(field) or []:
                            if isinstance(row, dict) and row.get("key") in aliases:
                                row["key"] = aliases[row["key"]]
    return adapted, {"registered_method": method, "factor_aliases_from_registry": aliases}


def _v5_observation(case: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
    """Validate initial-query and investigation evidence independently, then merge safely."""
    clean_case = copy.deepcopy(dict(case))
    # expected_limitation is prose, not a dimension identifier. Structured limitation
    # checks remain in _common_observation's decision/evidence validation.
    clean_case.pop("expected_limitation", None)
    query_trace = copy.deepcopy(dict(trace))
    query_state = next((query_trace.get(key) for key in ("workflow_state", "state_snapshot", "state")
                        if isinstance(query_trace.get(key), Mapping)), query_trace)
    query_inv = query_state.get("investigation_state")
    if isinstance(query_inv, dict):
        query_inv["evidence"] = []
        query_inv["action_results"] = []
    investigation_trace = copy.deepcopy(dict(trace))
    investigation_state = next((investigation_trace.get(key) for key in ("workflow_state", "state_snapshot", "state")
                                if isinstance(investigation_trace.get(key), Mapping)), investigation_trace)
    investigation_state["evidence_bundles"] = []
    query_obs = _common_observation(query_trace, clean_case)
    investigation_obs = _common_observation(investigation_trace, clean_case)
    query_ids = _v5_public_evidence_ids(clean_case, query_trace)
    investigation_ids = set(investigation_obs["valid_evidence_ids"])
    query_limitation_ids = set(query_obs.get("valid_limitation_evidence_ids") or [])
    investigation_limitation_ids = set(investigation_obs.get("valid_limitation_evidence_ids") or [])

    # An ID with conflicting serialized payloads is tainted across both channels.
    payloads: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    state = next((trace.get(key) for key in ("workflow_state", "state_snapshot", "state")
                  if isinstance(trace.get(key), Mapping)), trace)
    inv = _as_mapping(state.get("investigation_state"))
    for source_name, collection in (("query", state.get("evidence_bundles") or []),
                                    ("investigation", inv.get("evidence") or [])):
        for row in collection:
            if isinstance(row, Mapping) and row.get("evidence_id"):
                evidence_id = str(row["evidence_id"])
                payloads.setdefault(evidence_id, []).append((source_name, row))
    def semantic_signature(source_name: str, row: Mapping[str, Any]) -> str | None:
        metric = str(row.get("metric_id") or "")
        if source_name == "query":
            content = _as_mapping(row.get("content"))
            result = _as_mapping(content.get("result"))
            parsed = _as_mapping(result.get("parsed"))
            metric = str(parsed.get("metric_name") or result.get("metric_name") or result.get("metric_id")
                         or result.get("target_metric_id")
                         or _as_mapping(content.get("data")).get("metric_id")
                         or _as_mapping(content.get("arguments")).get("metric_id")
                         or _as_mapping(content.get("arguments")).get("metric_name") or metric)
            trend = _as_mapping(result.get("trend_result"))
            if isinstance(trend.get("points"), list):
                facts = {"dates": [trend.get("start_date"), trend.get("end_date")],
                         "points": [(str(p.get("bucket") or ""), p.get("value")) for p in trend["points"] if isinstance(p, Mapping)]}
                return canonical_sha256(facts)
            fact = next((result.get(key) for key in ("contribution_result", "metric_result", "compare_result", "breakdown_result")
                         if isinstance(result.get(key), Mapping) and any(k in result[key] for k in ("baseline_value", "current_value", "total_delta"))), None)
            if fact is None and any(k in result for k in ("baseline_value", "current_value", "total_delta")):
                fact = result
        else:
            values = _as_mapping(row.get("values"))
            result = _as_mapping(values.get("result"))
            if isinstance(result.get("points"), list):
                facts = {"dates": [result.get("start_date"), result.get("end_date")],
                         "points": [(str(p.get("bucket") or ""), p.get("value")) for p in result["points"] if isinstance(p, Mapping)]}
                return canonical_sha256(facts)
            fact = result if any(k in result for k in ("baseline_value", "current_value", "baseline_total", "current_total", "total_delta")) else (
                values if any(k in values for k in ("baseline_value", "current_value", "total_delta")) else None
            )
        if isinstance(fact, Mapping):
            components = fact.get("display_components") or fact.get("components") or fact.get("items") or []
            if isinstance(components, list):
                components = sorted((str(item.get("component_id") or item.get("key") or item.get("dimension_value") or ""),
                                     item.get("value", item.get("delta", item.get("delta_value"))))
                                    for item in components if isinstance(item, Mapping))
            def dimension_rows(field: str) -> list[tuple[str, Any, Any]]:
                raw = fact.get(field) or []
                if not isinstance(raw, list):
                    return []
                return sorted((str(item.get("dimension_value") or item.get("key") or ""),
                               item.get("value", item.get("delta_value")), item.get("share"))
                              for item in raw if isinstance(item, Mapping))
            return canonical_sha256({"dimension": fact.get("dimension"),
                                     "baseline": fact.get("baseline_value", fact.get("baseline_total")),
                                     "current": fact.get("current_value", fact.get("current_total")),
                                     "delta": fact.get("total_delta"), "unit": fact.get("unit"),
                                     "components": components,
                                     "baseline_items": dimension_rows("baseline_items"),
                                     "current_items": dimension_rows("current_items")})
    conflicted: set[str] = set()
    for evidence_id, rows in payloads.items():
        # Identical IDs across the MCP and typed-evidence adapters are expected
        # when their normalized facts agree; an uncomparable or differing pair
        # is treated as a conflict and excluded.
        signatures = [semantic_signature(name, row) for name, row in rows]
        metrics = set()
        for source_name, row in rows:
            if source_name == "investigation":
                metric = str(row.get("metric_id") or "")
            else:
                content = _as_mapping(row.get("content")); result = _as_mapping(content.get("result")); parsed = _as_mapping(result.get("parsed"))
                metric = str(parsed.get("metric_name") or result.get("metric_name") or result.get("metric_id")
                             or result.get("target_metric_id") or _as_mapping(content.get("data")).get("metric_id")
                             or _as_mapping(content.get("arguments")).get("metric_id") or "")
            if metric:
                metrics.add(metric)
        if len(rows) > 1 and (any(sig is None for sig in signatures) or len(set(signatures)) != 1 or len(metrics) > 1):
            conflicted.add(evidence_id)
    valid = (query_ids | investigation_ids) - conflicted
    limitation_valid = (query_limitation_ids | investigation_limitation_ids) - conflicted
    claim_ids = set(query_obs["cited_evidence_ids"]) | set(investigation_obs["cited_evidence_ids"])
    claim_bound = bool(claim_ids) and claim_ids.issubset(valid | limitation_valid)
    base = investigation_obs if inv.get("evidence") else query_obs
    merged = dict(base)
    merged.update({
        "valid_evidence_ids": valid,
        "valid_limitation_evidence_ids": limitation_valid,
        "observed_evidence_ids": set(query_obs["observed_evidence_ids"]) | set(investigation_obs["observed_evidence_ids"]),
        "cited_evidence_ids": claim_ids,
        "claims_bound": claim_bound,
        "limitation_evidence_valid": bool(limitation_valid) and claim_bound,
        "evidence_conflicts": sorted(conflicted),
        "query_evidence_valid_ids": sorted(query_ids - conflicted),
        "investigation_evidence_valid_ids": sorted(investigation_ids - conflicted),
        "limitation_record_valid": bool(query_obs["limitation_record_valid"] or investigation_obs["limitation_record_valid"]),
    })
    return merged


def _v5_public_evidence_ids(case: Mapping[str, Any], trace: Mapping[str, Any]) -> set[str]:
    """Revalidate public query envelopes against their executed plan/call binding."""
    state = next((trace.get(key) for key in ("workflow_state", "state_snapshot", "state")
                  if isinstance(trace.get(key), Mapping)), trace)
    scope_context = _as_mapping(state.get("authorization_context"))
    observed_scope = str(state.get("scope_type") or scope_context.get("scope_type") or "")
    if observed_scope and observed_scope != str(case.get("scope") or ""):
        return set()
    calls = {str(row.get("request_id")): row for row in (state.get("tool_calls") or [])
             if isinstance(row, Mapping) and row.get("request_id")}
    calls_by_evidence = {str(row.get("evidence_id")): row for row in (state.get("tool_calls") or [])
                         if isinstance(row, Mapping) and row.get("evidence_id")}
    allowed = {str(case.get("target_metric") or "")}
    allowed.update(str(item) for item in _as_mapping(case.get("same_identity_qualification")).get(
        "authorized_dependency_metric_ids", []) if item)
    period = _as_mapping(case.get("comparison_period"))
    baseline, current = period.get("baseline"), period.get("current")
    valid: set[str] = set()
    for bundle in state.get("evidence_bundles") or []:
        if not isinstance(bundle, Mapping) or not bundle.get("evidence_id"):
            continue
        evidence_id = str(bundle["evidence_id"])
        call = calls_by_evidence.get(evidence_id, {})
        request_id = str(bundle.get("request_id") or _as_mapping(bundle.get("content")).get("request_id")
                         or call.get("request_id") or "")
        content = _as_mapping(bundle.get("content"))
        result = _as_mapping(_as_mapping(content.get("result")))
        parsed = _as_mapping(result.get("parsed"))
        call = calls.get(request_id, call)
        authorized = _as_mapping(parsed.get("authorized_metric"))
        metric_id = str(parsed.get("metric_name") or "")
        call_tool = str(call.get("tool_name") or "")
        bundle_tool = str(bundle.get("source_ref") or content.get("tool_name") or "")
        if not (
            request_id and (not content.get("request_id") or content.get("request_id") == request_id)
            and str(call.get("request_id") or "") == request_id
            and (not call.get("evidence_id") or str(call.get("evidence_id")) == evidence_id)
            and (call.get("call_status") == "succeeded" or call.get("status") == "ok")
            and (call.get("response") == content if call.get("response") is not None else str(call.get("evidence_id") or "") == evidence_id)
            and authorized.get("metric_id") == metric_id and metric_id in allowed
            and call_tool and (not bundle_tool or bundle_tool == call_tool)
            and result.get("status") in {"ok", "not_found", "unsupported", "incomplete", "insufficient_data"}
        ):
            continue
        declared_scope = str(parsed.get("scope_type") or parsed.get("scope") or "")
        if declared_scope and declared_scope != str(case.get("scope") or ""):
            continue
        dimension = case.get("dimension")
        if dimension and str(parsed.get("dimension") or "") != str(dimension):
            continue
        if baseline is not None and current is not None:
            if isinstance(baseline, str) and isinstance(current, str):
                observed = (parsed.get("baseline_date"), parsed.get("current_date"))
                observed_range = (parsed.get("start_date"), parsed.get("end_date"))
                if observed != (baseline, current) and observed_range != (baseline, current):
                    # Single-date queries may bind to the expected current bucket.
                    if not (baseline == current and str(parsed.get("metric_date") or "")[:10] == current):
                        continue
            elif isinstance(baseline, Mapping) and isinstance(current, Mapping):
                expected_ranges = (
                    str(baseline.get("start") or baseline.get("start_date") or ""),
                    str(baseline.get("end") or baseline.get("end_date") or ""),
                    str(current.get("start") or current.get("start_date") or ""),
                    str(current.get("end") or current.get("end_date") or ""),
                )
                observed_ranges = tuple(str(parsed.get(key) or "") for key in (
                    "baseline_start", "baseline_end", "current_start", "current_end"))
                if observed_ranges != expected_ranges:
                    # Some public comparison envelopes preserve explicit role dates.
                    if (parsed.get("baseline_date"), parsed.get("current_date")) != (baseline, current):
                        continue
        valid.add(evidence_id)
        # A registered decomposition envelope cites child query evidence. Add
        # those IDs only when every nested request is explicitly bound to the
        # parent, allowed dependency, exact dates and matching authorization.
        if content.get("tool_name") == "registered_numeric_decomposition":
            result = _as_mapping(content.get("result"))
            parsed = _as_mapping(result.get("parsed"))
            if (result.get("status") != "ok"
                    or str(result.get("target_metric_id") or parsed.get("metric_name") or "") != str(case.get("target_metric") or "")
                    or _as_mapping(parsed.get("authorized_metric")).get("metric_id") != case.get("target_metric")):
                valid.discard(evidence_id)
                continue
            sources = content.get("source_queries") or []
            verified_source_ids: set[str] = set()
            expected_period = _as_mapping(case.get("comparison_period"))
            for source in sources:
                if not isinstance(source, Mapping):
                    continue
                source_metric = str(source.get("metric_id") or "")
                source_request = str(source.get("request_id") or "")
                envelope = _as_mapping(source.get("envelope"))
                child_result = _as_mapping(_as_mapping(envelope.get("result")))
                child_plan = _as_mapping(child_result.get("parsed"))
                child_authorized = _as_mapping(child_plan.get("authorized_metric"))
                expected_ranges = (expected_period.get("baseline"), expected_period.get("current"))
                child_ranges = (child_plan.get("start_date"), child_plan.get("end_date"))
                child_role_dates = (child_plan.get("baseline_date"), child_plan.get("current_date"))
                period_ok = child_ranges == expected_ranges or child_role_dates == expected_ranges
                if (source_metric in allowed and source_request == request_id + ":input:" + source_metric
                        and envelope.get("request_id") == source_request
                        and source.get("evidence_id") == request_id + ":source:" + source_metric
                        and child_result.get("status") == "ok"
                        and child_plan.get("metric_name") == source_metric
                        and child_authorized.get("metric_id") == source_metric and period_ok):
                    verified_source_ids.add(str(source["evidence_id"]))
            declared = {str(item) for item in (result.get("source_evidence_ids") or []) if item}
            if sources and verified_source_ids == declared and len(verified_source_ids) == len(sources):
                valid.update(verified_source_ids)
            else:
                valid.discard(evidence_id)
    return valid


def score_bg_run_v5(
    case: Mapping[str, Any], *, arm: str, trace: Mapping[str, Any], lifecycle_status: str,
    human_review: Mapping[str, Any] | None = None, experiment_condition_status: str | None = None,
) -> dict[str, Any]:
    """Explicit offline v5 replay scorer; historical v3/v4 results are never mutated."""
    adapted, mapping = _v5_prepare_trace(case, trace)
    clean_case = copy.deepcopy(dict(case))
    clean_case.pop("expected_limitation", None)
    score = score_bg_run_v4(clean_case, arm=arm, trace=adapted, lifecycle_status=lifecycle_status,
                            human_review=human_review)
    obs = _v5_observation(case, adapted)
    task = str(case.get("protocol_task_type") or "")
    numeric_ok = bool(score.get("numeric_checks")) and all(score["numeric_checks"].values())
    evidence_ok = bool(obs["valid_evidence_ids"]) and obs["claims_bound"] and not obs["evidence_conflicts"]
    if task == "dimension_statistical_contribution":
        card_bound = bool(score.get("contribution_card_evidence_valid"))
        # The v4 card check used one evidence channel; v5 checks the merged set.
        presentation = _as_mapping(obs["presentation"])
        raw_cards = presentation.get("contributions") or []
        matching = [card for card in raw_cards if isinstance(card, Mapping)
                    and card.get("metric_name") == case.get("target_metric")
                    and card.get("dimension") == case.get("dimension")]
        if len(matching) == 1:
            refs = {str(item) for item in (matching[0].get("evidence_ids") or []) if item}
            card_bound = bool(refs) and refs.issubset(obs["valid_evidence_ids"])
        score["contribution_card_evidence_valid"] = card_bound
        evidence_ok = evidence_ok and card_bound
    score["evidence_valid"] = evidence_ok
    score["effective_diagnostic_completed"] = numeric_ok and evidence_ok
    score["evidence_binding"] = {
        "valid_query_evidence_ids": obs["query_evidence_valid_ids"],
        "valid_investigation_evidence_ids": obs["investigation_evidence_valid_ids"],
        "conflicted_evidence_ids": obs["evidence_conflicts"],
        "claims_bound": obs["claims_bound"],
    }
    score["factor_mapping"] = mapping

    human = _as_mapping(human_review)
    human_status = "pending" if human.get("decision") not in {"pass", "fail"} else human.get("decision")
    score["human_review_status"] = human_status
    score["human_text_review_status"] = human_status
    condition_detail: dict[str, Any] = {}
    condition_status = experiment_condition_status
    if experiment_condition_status is None and task == "expected_data_insufficiency":
        points = []
        for bundle in _as_mapping(obs["state"]).get("evidence_bundles") or []:
            result = _as_mapping(_as_mapping(_as_mapping(bundle).get("content")).get("result"))
            trend = _as_mapping(result.get("trend_result"))
            if isinstance(trend.get("points"), list):
                points.extend(item for item in trend["points"] if isinstance(item, Mapping))
        if any(item.get("value") is not None for item in points):
            condition_status = "experiment_condition_invalid"
            condition_detail = {"reason": "historical_run_returned_numeric_trend_value; missing-tail condition not established"}
        else:
            condition_status = "experiment_condition_unverified"
            condition_detail = {"reason": "missing-tail contract and bucket dates were not saved as structured run condition"}
    if experiment_condition_status is None and task == "expected_capability_limit":
        if case.get("readiness") == "blocked":
            condition_status = "experiment_condition_invalid"
            condition_detail = {"reason": "historical_case_readiness_blocked; execution qualification was not established"}
        else:
            condition_status = "experiment_condition_unverified"
            condition_detail = {"reason": "historical capability and authorization snapshots are incomplete"}
    qualification = str(_as_mapping(case.get("same_identity_qualification")).get("qualification") or "")
    automated_failures: list[str] = []
    for key in ("metric_matches", "scope_matches", "period_matches", "execution_error_free", "status_consistent"):
        if score.get(key) is False:
            automated_failures.append(key)
    if task in {"registered_numeric_decomposition", "dimension_statistical_contribution"}:
        task_evidence_ok = evidence_ok
    elif task == "expected_data_insufficiency":
        task_evidence_ok = bool(obs["limitation_evidence_valid"])
    elif task == "expected_capability_limit":
        task_evidence_ok = bool(obs["limitation_record_valid"])
    else:
        task_evidence_ok = False
    if not task_evidence_ok:
        automated_failures.append("evidence_invalid_or_unbound")
    automated_failures.extend("numeric_checks." + key for key, passed in score.get("numeric_checks", {}).items() if not passed)
    if score.get("critic_review_applicable") and score.get("critic_review_complete") is not True:
        automated_failures.append("critic_review_incomplete")
    if lifecycle_status != "completed":
        automated_failures.append("lifecycle_not_completed")
    if task in {"registered_numeric_decomposition", "dimension_statistical_contribution"}:
        if not score.get("effective_diagnostic_completed"):
            automated_failures.append("effective_diagnostic_not_completed")
        if score.get("presentation_status") != "completed":
            automated_failures.append("presentation_not_completed")
    elif task in {"expected_data_insufficiency", "expected_capability_limit"}:
        if not score.get("correct_safe_stop"):
            automated_failures.append("expected_safe_stop_not_verified")

    if condition_status is None:
        if case.get("readiness") == "blocked" or qualification.startswith("not_independently_qualified"):
            condition_status = "experiment_condition_invalid"
        elif task in {"expected_data_insufficiency", "expected_capability_limit"}:
            condition_status = "experiment_condition_unverified"
        else:
            condition_status = "experiment_condition_not_applicable"
    if condition_status == "experiment_condition_invalid":
        automated_failures.append("experiment_condition_invalid")
    elif condition_status == "experiment_condition_unverified":
        automated_failures.append("experiment_condition_unverified")
    automated_pass = not automated_failures
    score["automated_checks_pass"] = automated_pass
    score["automated_failure_reasons"] = list(dict.fromkeys(automated_failures))
    score["experiment_condition_status"] = condition_status
    score["experiment_condition_details"] = condition_detail
    score["task_acceptance_pass"] = bool(automated_pass and human_status == "pass")
    if score["task_acceptance_pass"]:
        task_status = "passed"
    elif not automated_pass:
        task_status = "failed"
    else:
        task_status = "pending_manual_review"
    score["task_acceptance_status"] = task_status
    score["task_pass_reason"] = None if task_status == "passed" else (
        ";".join(score["automated_failure_reasons"]) if score["automated_failure_reasons"]
        else "manual_review_pending"
    )
    score["scorer_version"] = OFFLINE_SCORER_V5_VERSION
    score["parent_scorer_version"] = OFFLINE_SCORER_VERSION
    return score

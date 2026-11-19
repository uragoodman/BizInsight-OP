"""Case-aware, tri-state observation extraction for formal B/G traces."""
from __future__ import annotations

from typing import Any, Mapping
from evaluation.resume_bg_numeric_scoring import (
    compare_case_numeric_fields,
    extract_case_numeric_fields,
    registered_numeric_decomposition_projection,
)


def _state(trace: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("state_snapshot", "workflow_state", "state"):
        value = trace.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _result_body(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _metric_and_operation(
    state: Mapping[str, Any], arm: str, case: Mapping[str, Any] | None = None,
) -> tuple[str | None, str | None, list[str]]:
    decomposition = state.get("registered_decomposition_request")
    decision = state.get("metric_planning_decision")
    decision = decision if isinstance(decision, Mapping) else {}
    if not isinstance(decomposition, Mapping) or not decomposition:
        nested = decision.get("registered_decomposition")
        decomposition = nested if isinstance(nested, Mapping) else {}
    if decision.get("operation_intent") == "registered_numeric_decomposition":
        target = str(decomposition.get("target_metric_id") or "")
        if target:
            projection = registered_numeric_decomposition_projection(case or {}, state)
            if projection.get("valid") is True:
                return target, "registered_numeric_decomposition", ["registered_numeric_decomposition"]
            return target, None, []

    tools = [item for item in (state.get("tool_calls") or []) if isinstance(item, dict)]
    qresults = [item for item in (state.get("query_results") or []) if isinstance(item, dict)]
    observed_ops: list[str] = []
    metric = None
    for call in tools:
        args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        metric = metric or call.get("planned_metric_id") or args.get("metric_id") or args.get("metric_name")
        op = call.get("planned_operation") or call.get("tool") or call.get("tool_name")
        if op:
            observed_ops.append(str(op))
    decision = state.get("metric_planning_decision") or {}
    if isinstance(decision, dict):
        metric = metric or next(iter(decision.get("confirmed_metrics") or []), None)
        if decision.get("query_type"):
            observed_ops.append(str(decision["query_type"]))
    if metric is None:
        for item in qresults:
            args = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            parsed = ((item.get("result") or {}).get("parsed") or {}) if isinstance(item.get("result"), dict) else {}
            metric = args.get("metric_name") or args.get("metric_id") or parsed.get("metric_name")
            if metric:
                break
    normalized = []
    for op in observed_ops:
        if op in {"metric_contribution", "compare_breakdown_by_dimension", "compare_breakdown_by_dimension_period"}:
            normalized.append("metric_contribution")
        elif op in {"metric_compare", "compare_metric"}:
            normalized.append("metric_compare")
        elif op in {"metric_trend", "query_trend", "query_metric_trend", "registry_trend"}:
            normalized.append("metric_trend")
        elif op in {"metric_value", "query_value"}:
            normalized.append("metric_value")
        else:
            normalized.append(op)
    return (str(metric) if metric else None, normalized[-1] if normalized else None, normalized)


def _actual_period(
    state: Mapping[str, Any], case: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    decision = state.get("metric_planning_decision")
    decision = decision if isinstance(decision, Mapping) else {}
    if decision.get("operation_intent") == "registered_numeric_decomposition":
        projection = registered_numeric_decomposition_projection(case or {}, state)
        if projection.get("valid") is not True:
            return None
        request = state.get("registered_decomposition_request")
        if not isinstance(request, Mapping) or not request:
            nested = decision.get("registered_decomposition")
            request = nested if isinstance(nested, Mapping) else {}
        if request.get("status") != "ready":
            return None
        baseline, current = request.get("baseline_date"), request.get("current_date")
        if baseline and current and baseline != current:
            return {"kind": "point_pair", "baseline": baseline, "current": current}
        # Do not fall through to an internal factor query: its date range is
        # supporting evidence, not a replacement for the target comparison.
        return None

    calls = [item for item in (state.get("tool_calls") or []) if isinstance(item, dict)]
    plan = state.get("metric_query_batch_plan") or {}
    plans = plan.get("plans") if isinstance(plan, dict) else None
    for call in calls:
        args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        period = args.get("period") if isinstance(args.get("period"), dict) else {}
        comparison = args.get("comparison") if isinstance(args.get("comparison"), dict) else {}
        baseline_date = args.get("baseline_date") or comparison.get("baseline_date")
        current_date = args.get("current_date") or comparison.get("current_date")
        baseline_start = args.get("baseline_start") or comparison.get("baseline_start")
        baseline_end = args.get("baseline_end") or comparison.get("baseline_end")
        current_start = args.get("current_start") or comparison.get("current_start")
        current_end = args.get("current_end") or comparison.get("current_end")
        if baseline_date and current_date:
            return {"kind": "point_pair", "baseline": baseline_date, "current": current_date}
        if baseline_start and current_start:
            return {"kind": "period_pair", "baseline_start": baseline_start,
                    "baseline_end": baseline_end, "current_start": current_start,
                    "current_end": current_end, "granularity": args.get("granularity") or comparison.get("granularity")}
        start_date = args.get("start_date") or period.get("start_date")
        end_date = args.get("end_date") or period.get("end_date")
        if start_date and end_date:
            return {"kind": "range", "start": start_date, "end": end_date}
        if args.get("metric_date"):
            return {"kind": "point", "date": args.get("metric_date")}
    for item in plans or []:
        if not isinstance(item, dict):
            continue
        if item.get("baseline_date") and item.get("current_date"):
            return {"kind": "point_pair", "baseline": item["baseline_date"], "current": item["current_date"]}
        if item.get("metric_date"):
            return {"kind": "point", "date": item["metric_date"]}
        if item.get("start_date") and item.get("end_date"):
            return {"kind": "range", "start": item["start_date"], "end": item["end_date"]}
    compilation = state.get("context_compilation") or {}
    contract = compilation.get("request_contract") if isinstance(compilation, dict) else None
    comparison = contract.get("comparison") if isinstance(contract, dict) else None
    if isinstance(comparison, dict):
        if comparison.get("kind") == "day_pair":
            return {"kind": "point_pair", "baseline": comparison.get("baseline_date"), "current": comparison.get("current_date")}
        if comparison.get("kind") == "period_pair":
            return {"kind": "period_pair", **{key: comparison.get(key) for key in (
                "baseline_start", "baseline_end", "current_start", "current_end", "granularity",
            )}}
    slots = compilation.get("known_slots") if isinstance(compilation, dict) else None
    if isinstance(slots, dict):
        baseline = slots.get("baseline_date") or {}
        current = slots.get("current_date") or {}
        if baseline.get("value") and current.get("value"):
            return {"kind": "point_pair", "baseline": baseline["value"], "current": current["value"]}
    return None


def _evidence_status(state: Mapping[str, Any]) -> bool | None:
    bundles = [item for item in (state.get("evidence_bundles") or []) if isinstance(item, dict)]
    calls = [item for item in (state.get("tool_calls") or []) if isinstance(item, dict)]
    records = [item for item in (state.get("decision_records") or []) if isinstance(item, dict)]
    if (
        not bundles and records
        and all(_valid_evaluation_surface_refusal(state, item) for item in records)
    ):
        return True
    if not bundles and records and all(item.get("authorization_checked") is True for item in records):
        if all(item.get("sql_executed") is False for item in records):
            return True
    if not calls:
        # A legitimate pre-query safe stop can have decision evidence instead.
        return True if records and all(item.get("authorization_checked") is True for item in records) else None
    if not bundles:
        return None
    ids = {str(item.get("evidence_id")) for item in bundles if item.get("evidence_id")}
    requests = {str(item.get("request_id")) for item in calls if item.get("request_id")}
    bundle_requests = {str(item.get("request_id")) for item in bundles if item.get("request_id")}
    query_result_requests = {
        str(item.get("request_id")) for item in (state.get("query_results") or [])
        if isinstance(item, dict) and item.get("request_id")
    }
    response_requests = {
        str((item.get("response") or {}).get("request_id"))
        for item in calls if isinstance(item.get("response"), dict)
        and (item.get("response") or {}).get("request_id")
    }
    if not requests:
        return False
    if bundle_requests and not requests.issubset(bundle_requests):
        if not requests.issubset(query_result_requests | response_requests):
            return False
    elif not bundle_requests and not requests.issubset(query_result_requests | response_requests):
        return None
    if not ids:
        return False
    for claim in state.get("claims") or []:
        if isinstance(claim, dict):
            refs = claim.get("evidence_ids")
            if isinstance(refs, list) and any(str(ref) not in ids for ref in refs):
                return False
    return True


def _valid_evaluation_surface_refusal(state: Mapping[str, Any], record: Mapping[str, Any]) -> bool:
    """Validate a B-arm pre-execution refusal without pretending SQL ran."""
    surface = state.get("evaluation_tool_surface")
    if not isinstance(surface, Mapping):
        return False
    surface_hash = str(surface.get("surface_sha256") or "")
    allowed = set(str(item) for item in (surface.get("allowed_operation_names") or []))
    requested_tool = str(record.get("requested_tool") or "")
    if not (
        record.get("source") == "evaluation_tool_surface"
        and record.get("evaluation_tool_surface_checked") is True
        and record.get("evaluation_tool_surface_sha256") == surface_hash
        and record.get("sql_executed") is False
        and record.get("limitation_code") == "CAPABILITY_GAP"
        and requested_tool
        and requested_tool not in allowed
    ):
        return False
    matching_call = next((
        item for item in (state.get("tool_calls") or [])
        if isinstance(item, Mapping) and item.get("request_id") == record.get("request_id")
    ), None)
    if matching_call is None:
        return False
    arguments = matching_call.get("arguments")
    arguments = arguments if isinstance(arguments, Mapping) else {}
    return bool(
        matching_call.get("call_status") == "refused_before_execution"
        and matching_call.get("tool") == requested_tool
        and str(arguments.get("metric_id") or arguments.get("metric_name") or "") == str(record.get("metric_id") or "")
        and arguments.get("dimension") == record.get("dimension")
    )


def _presentation(state: Mapping[str, Any]) -> dict[str, Any] | None:
    for key in ("analysis_presentation", "final_display", "presentation"):
        val = state.get(key)
        if isinstance(val, dict) and val:
            return val.get("presentation") if isinstance(val.get("presentation"), dict) else val
    final = state.get("final")
    if isinstance(final, dict) and isinstance(final.get("presentation"), dict):
        return final["presentation"]
    return None


def _unit(state: Mapping[str, Any]) -> str | None:
    presentation = _presentation(state) or {}
    units: list[str] = []
    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in {"unit", "metric_unit"} and isinstance(child, str) and child:
                    units.append(child)
                else:
                    collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
    collect(presentation)
    if units:
        return units[0] if len(set(units)) == 1 else None
    for item in state.get("query_results") or []:
        body = item.get("result") if isinstance(item, dict) and isinstance(item.get("result"), dict) else {}
        parsed = body.get("parsed") if isinstance(body.get("parsed"), dict) else {}
        val = parsed.get("unit")
        if isinstance(val, str) and val:
            return val
    return None


def _matches_period(actual: dict[str, Any] | None, expected: Mapping[str, Any]) -> bool | None:
    if actual is None:
        return None
    kind = expected.get("kind")
    if kind == "point":
        return actual.get("kind") == "point" and actual.get("date") == expected.get("date")
    if kind == "range":
        return actual.get("kind") == "range" and actual.get("start") == expected.get("start") and actual.get("end") == expected.get("end")
    if kind == "point_pair":
        return actual.get("kind") == "point_pair" and actual.get("baseline") == expected.get("baseline") and actual.get("current") == expected.get("current")
    if kind == "period_pair":
        return actual.get("kind") == "period_pair" and all(actual.get(key) == expected.get(key) for key in ("baseline_start", "baseline_end", "current_start", "current_end", "granularity"))
    if kind == "month_pair":
        baseline_month = expected.get("baseline_month")
        current_month = expected.get("current_month")
        baseline_start, baseline_end = expected.get("baseline_start"), expected.get("baseline_end")
        current_start, current_end = expected.get("current_start"), expected.get("current_end")
        # The month labels are part of the frozen case contract, not hints.
        # Reject a contradictory label/range instead of comparing only dates.
        if not (
            isinstance(baseline_month, str) and isinstance(current_month, str)
            and isinstance(baseline_start, str) and isinstance(baseline_end, str)
            and isinstance(current_start, str) and isinstance(current_end, str)
            and baseline_start[:7] == baseline_month == baseline_end[:7]
            and current_start[:7] == current_month == current_end[:7]
        ):
            return False
        return actual.get("kind") == "period_pair" and all(
            actual.get(key) == value for key, value in (
                ("baseline_start", baseline_start), ("baseline_end", baseline_end),
                ("current_start", current_start), ("current_end", current_end),
                ("granularity", "month"),
            )
        )
    return None


def _multi_step_period_match(
    state: Mapping[str, Any], expected: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Validate trend and contribution periods independently for a multi-step case.

    A single ``period`` projection cannot represent both the trend range and
    the point-pair contribution request. Validate the actual call arguments
    for both operations instead of comparing only the first call's shape.
    """
    baseline = expected.get("baseline")
    current = expected.get("current")
    observed: dict[str, list[dict[str, Any]]] = {"metric_trend": [], "metric_contribution": []}
    for call in state.get("tool_calls") or []:
        if not isinstance(call, Mapping):
            continue
        args = call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
        raw_operation = call.get("planned_operation") or call.get("tool") or call.get("tool_name")
        operation = str(raw_operation or "")
        if operation in {"metric_trend", "query_trend", "query_metric_trend", "registry_trend"}:
            period = args.get("period") if isinstance(args.get("period"), Mapping) else {}
            observed["metric_trend"].append({
                "start": args.get("start_date") or period.get("start_date"),
                "end": args.get("end_date") or period.get("end_date"),
            })
        elif operation in {
            "metric_contribution", "compare_breakdown_by_dimension",
            "compare_breakdown_by_dimension_period", "registry_contribution",
        }:
            comparison = args.get("comparison") if isinstance(args.get("comparison"), Mapping) else {}
            observed["metric_contribution"].append({
                "baseline": args.get("baseline_date") or comparison.get("baseline_date"),
                "current": args.get("current_date") or comparison.get("current_date"),
            })
    trend_ok = len(observed["metric_trend"]) == 1 and observed["metric_trend"][0] == {
        "start": baseline, "end": current,
    }
    contribution_ok = len(observed["metric_contribution"]) == 1 and observed["metric_contribution"][0] == {
        "baseline": baseline, "current": current,
    }
    detail = {
        "expected": {"baseline": baseline, "current": current},
        "observed": observed,
        "trend_match": trend_ok,
        "contribution_match": contribution_ok,
    }
    return trend_ok and contribution_ok, detail


def extract_formal_observation(case: Mapping[str, Any], arm: str, trace: Mapping[str, Any]) -> dict[str, Any]:
    """Extract what can be proven from an actual trace; absent fields stay unknown."""
    state = _state(trace)
    metric, actual_op, ops = _metric_and_operation(state, arm, case)
    period = _actual_period(state, case)
    scope = state.get("scope_type")
    if scope not in {"platform", "seller"}:
        scope = None
    expected_operation = str(case.get("operation") or "")
    expected_op_map = {
        "registered_numeric_decomposition": {"registered_numeric_decomposition"},
        "metric_value": {"metric_value"}, "metric_compare": {"metric_compare"},
        "metric_trend": {"metric_trend"}, "dimension_contribution": {"metric_contribution"},
        "multi_step_diagnostic": {"metric_trend", "metric_contribution"},
        "expected_data_insufficiency": {"metric_compare", "metric_value", "metric_trend"},
    }
    if expected_operation == "multi_step_diagnostic":
        operation = expected_operation if {"metric_trend", "metric_contribution"}.issubset(set(ops)) else actual_op
    elif expected_operation == "expected_data_insufficiency":
        result_statuses = [
            ((item.get("result") or {}).get("status"))
            for item in state.get("query_results") or [] if isinstance(item, dict)
        ]
        operation = expected_operation if actual_op in expected_op_map[expected_operation] and any(
            status in {"not_found", "insufficient_data"} for status in result_statuses
        ) else actual_op
    elif actual_op in expected_op_map.get(expected_operation, set()):
        operation = expected_operation
    else:
        operation = actual_op

    expected_period = case.get("period") if isinstance(case.get("period"), Mapping) else {}
    period_details = None
    if expected_operation == "multi_step_diagnostic":
        period_ok, period_details = _multi_step_period_match(state, expected_period)
    else:
        period_ok = _matches_period(period, expected_period)
    metric_ok: bool | None = None if metric is None else metric == case.get("target_metric")
    scope_ok: bool | None = None if scope is None else scope == case.get("scope")
    operation_ok: bool | None = None if operation is None else operation == expected_operation
    dimension = None
    for call in (state.get("tool_calls") or []):
        if isinstance(call, dict):
            args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
            if args.get("dimension"):
                dimension = args.get("dimension")
                break
    decision = state.get("metric_planning_decision") or {}
    if dimension is None and isinstance(decision, dict):
        dimension = decision.get("dimension")
    expected_dimension = case.get("dimension")
    dimension_match: bool | None = None if dimension is None and expected_dimension is not None else dimension == expected_dimension
    evidence_ok = _evidence_status(state)
    presentation = _presentation(state)
    numeric, numeric_result_sources = extract_case_numeric_fields(case, state, arm)
    numeric_match: bool | None = None
    numeric_field_comparisons: list[dict[str, Any]] = []
    if (case.get("expected") or {}).get("outcome") == "numeric_result":
        numeric_match, numeric_field_comparisons = compare_case_numeric_fields(case, numeric, arm=arm)
    decomposition_projection = registered_numeric_decomposition_projection(case, state)
    unit = (
        decomposition_projection.get("unit")
        if decomposition_projection.get("valid") is True
        else _unit(state)
    )
    expected = case.get("expected") or {}
    truth = expected.get("truth")
    unit_match: bool | None = None
    if expected.get("outcome") == "numeric_result" and isinstance(truth, Mapping):
        expected_unit = truth.get("unit") or case.get("metric_unit")
        unit_match = None if not unit or not expected_unit else unit == expected_unit

    terminal_code = state.get("terminal_code")
    if terminal_code is None:
        errors = [item for item in (state.get("errors") or []) if isinstance(item, dict)]
        terminal_code = next((item.get("code") for item in reversed(errors) if item.get("code")), None)
    limitation_disclosed: bool | None = None
    if expected.get("outcome") == "safe_stop":
        limitation = str(expected.get("terminal_code") or "")
        limitation_disclosed = bool(
            (presentation and presentation.get("limitation_code") == limitation)
            or any(isinstance(item, dict) and item.get("limitation_code") == limitation for item in state.get("decision_records") or [])
            or (isinstance(state.get("investigation_state"), dict) and (state["investigation_state"].get("stop_reason") or {}).get("code") == limitation)
        )
        if not any((presentation, state.get("decision_records"), state.get("investigation_state"))):
            limitation_disclosed = None
    precondition_verified: bool | None = None
    if expected.get("outcome") == "safe_stop":
        body_statuses = []
        for item in state.get("query_results") or []:
            body = item.get("result") if isinstance(item, dict) and isinstance(item.get("result"), dict) else {}
            body_statuses.append(body.get("status"))
        records = [item for item in (state.get("decision_records") or []) if isinstance(item, dict)]
        if expected.get("terminal_code") == "CAPABILITY_GAP":
            precondition_verified = True if any(
                item.get("limitation_code") == "CAPABILITY_GAP"
                and (
                    (item.get("authorization_checked") is True and item.get("sql_executed") is False)
                    or _valid_evaluation_surface_refusal(state, item)
                )
                for item in records
            ) else None
        elif expected.get("terminal_code") == "DATA_INSUFFICIENT":
            from evaluation.resume_formal_bg_v1 import verify_data_insufficiency_precondition

            expected_truth = expected.get("truth") if isinstance(expected.get("truth"), Mapping) else {}
            baseline_value = expected_truth.get("baseline_value", expected_truth.get("baseline"))
            observed_gaps = _data_gap_query_observations(state)
            verification = verify_data_insufficiency_precondition(baseline_value, observed_gaps)
            precondition_verified = verification["verified"] if observed_gaps else (
                True if "not_found" in body_statuses or "insufficient_data" in body_statuses else None
            )

    answer = state.get("answer") or state.get("final_answer")
    if not isinstance(answer, str):
        final = state.get("final")
        answer = final.get("answer") if isinstance(final, dict) else ""
    manual_reasons = []
    if isinstance(answer, str) and answer.strip():
        manual_reasons.append("Human review required for semantic fidelity of free-form answer wording.")
    return {
        "operation": operation,
        "metric_id": metric,
        "scope": scope,
        "dimension": dimension,
        "dimension_match": dimension_match,
        "period": period if period is not None else None,
        "period_match": period_ok,
        "period_match_details": period_details,
        "evidence_valid": evidence_ok,
        "numeric_values": numeric,
        "numeric_values_match": numeric_match,
        "numeric_field_comparisons": numeric_field_comparisons,
        "numeric_result_sources": numeric_result_sources,
        "unit": unit,
        "unit_match": unit_match,
        "answer_card": presentation,
        "answer_card_consistent": (
            None if presentation is None or metric_ok is None or period_ok is None or evidence_ok is None
            else not (metric_ok is False or period_ok is False or evidence_ok is False)
        ),
        "answer": answer if isinstance(answer, str) else None,
        "terminal_code": terminal_code,
        "precondition_verified": precondition_verified,
        "expected_stop": None if terminal_code is None else terminal_code == expected.get("terminal_code"),
        "limitation_disclosed": limitation_disclosed,
        "effective_diagnostic_completed": bool(numeric_match is True and evidence_ok is True),
        "governance_execution": "not_applicable" if arm == "B" else (
            "critic_completed" if _critic_completed(state) else "critic_incomplete"
        ),
        "manual_review_required": bool(manual_reasons),
        "manual_review_template": {
            "review_fields": ["answer_semantically_matches_verified_evidence", "no_unsupported_causal_claim", "limitation_wording_is_not_contradictory"],
            "decision": None,
            "reviewer": None,
            "rationale": None,
            "reasons": manual_reasons,
        },
        "extraction_status": "partial_unknowns" if any(value is None for value in (operation, metric, scope, period_ok, evidence_ok)) else "complete",
    }


def _critic_completed(state: Mapping[str, Any]) -> bool:
    result = state.get("critic_result") or {}
    if result.get("status") not in {"approved", "rejected", "needs_revision"}:
        return False
    return result.get("deterministic_review", {}).get("status") is not None or result.get("data_review", {}).get("status") is not None


def _data_gap_query_observations(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Collect explicit missing-side comparison evidence from either arm's trace shape."""
    found: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            if (
                value.get("status") == "not_found"
                and "baseline_value" in value and "current_value" in value
            ):
                found.append({
                    "status": value.get("status"),
                    "baseline_value": value.get("baseline_value"),
                    "current_value": value.get("current_value"),
                })
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(state.get("query_results") or [])
    visit(state.get("tool_calls") or [])
    # Avoid duplicate copies of the same serialized result in response/result envelopes.
    unique = {
        (row["status"], row["baseline_value"], row["current_value"]): row
        for row in found
    }
    return list(unique.values())

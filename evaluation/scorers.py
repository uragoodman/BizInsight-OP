from __future__ import annotations

from typing import Any
from hashlib import sha256
from math import ceil, isfinite
from pathlib import Path
import re


SCENARIO_SCORER_VERSION = "root-cause-v4"
DIAGNOSTIC_RUN_SCORER_VERSION = "root-cause-run-v2"
_SCENARIO_EXPECT_KEYS = {
    "abstain", "found_factor", "cross_metric_link", "found_dimension",
    "required_evidence_purposes", "reported_dimension_contribution",
    "business_cause_unresolved", "text_compatibility",
}


def rate(passed: int, total: int) -> float:
    return round(passed / total, 4) if total else 0.0


def _has_traceable_missing_target_trend(state: dict[str, Any]) -> bool:
    """Return whether an executed target-trend query has an auditable gap.

    Incomplete trend responses are intentionally rejected as typed attribution
    evidence, so they do not appear in ``root_cause_evidence``.  The raw MCP
    bundle and its tool-call record still provide an execution audit trail;
    accept that trail as an examined trend only when both share an evidence ID
    and the target metric's returned trend has no usable values.
    """
    target = str(state.get("diagnostic_target_metric") or "")
    investigation = state.get("investigation_state") or {}
    if not target and isinstance(investigation, dict):
        target = str((investigation.get("target") or {}).get("metric_id") or "")
    if not target:
        return False

    bundles = {
        str(item.get("evidence_id")): item
        for item in (state.get("evidence_bundles") or [])
        if isinstance(item, dict) and item.get("evidence_id")
    }
    for call in state.get("tool_calls") or []:
        if not isinstance(call, dict) or call.get("tool_name") not in {"query_metric_trend", "registry_trend"}:
            continue
        if str(call.get("status") or "").casefold() not in {"ok", "success", "completed"}:
            continue
        arguments = call.get("arguments") or {}
        if str(arguments.get("metric_name") or arguments.get("metric") or "") != target:
            continue
        evidence_id = str(call.get("evidence_id") or "")
        bundle = bundles.get(evidence_id)
        if not evidence_id or not isinstance(bundle, dict):
            continue
        if bundle.get("source_ref") not in {"query_metric_trend", "registry_trend"}:
            continue
        content = bundle.get("content") or {}
        response = content.get("result") if isinstance(content, dict) else None
        if not isinstance(response, dict):
            continue
        envelope = response.get("result") if isinstance(response.get("result"), dict) else response
        trend = envelope.get("trend_result") if isinstance(envelope, dict) else None
        if not isinstance(trend, dict) or str(trend.get("metric_name") or "") != target:
            continue
        points = trend.get("points")
        if not isinstance(points, list) or not points or any(
            not isinstance(point, dict)
            or not isinstance(point.get("value"), (int, float))
            or isinstance(point.get("value"), bool)
            or not isfinite(float(point["value"]))
            for point in points
        ):
            return True
    return False


# Phase 0: cross-metric ground-truth hints, used to verify a diagnosis actually
# names the intended cause (or abstains) rather than only "completed".
_FINDING_TEXTS: dict[str, tuple[str, ...]] = {
    "avg_order_value": ("客单价", "avg_order_value", "aov"),
    "delivery_delay_rate": ("延迟送达率", "延误送达率", "延迟率", "delivery_delay_rate"),
    "order_count": ("订单量", "订单数", "order_count"),
    "gmv": ("gmv", "成交额"),
}


def collect_anomaly_metrics(state: dict[str, Any]) -> dict[str, Any]:
    """Numeric trajectory metrics for a root-cause run (baseline + agent alike)."""
    analysis = state.get("analysis_result", {}) or {}
    rc_evidence = state.get("root_cause_evidence", [])
    dimensions = [
        item for item in rc_evidence
        if str(item.get("purpose", "")).startswith("dimension_contribution:")
    ]
    calls = state.get("tool_calls", []) or []
    call_keys = [
        (str(call.get("tool_name") or ""), repr(call.get("arguments") or {}))
        for call in calls
    ]
    duplicate_call_count = len(call_keys) - len(set(call_keys))
    invalid_call_count = sum(
        1 for call in calls
        if str(call.get("status") or "ok").casefold() not in {"ok", "success", "completed"}
        or call.get("arguments") is None
    )
    timings = [
        float(item.get("duration_ms", 0) or 0)
        for item in (state.get("node_timings", []) or [])
        if item.get("duration_ms") is not None
    ]
    total_latency_ms = round(sum(timings), 3)
    usage = state.get("llm_usage", {}) or {}
    token_summary = state.get("llm_token_summary")
    token_total: int | None
    token_usage_complete: bool | None
    token_request_status: str | None
    if isinstance(token_summary, dict):
        token_request_status = str(token_summary.get("request_status") or "unknown")
        token_usage_complete = bool(token_summary.get("usage_complete"))
        if token_request_status == "no_model_request" and int(token_summary.get("sdk_attempt_count", 0) or 0) == 0:
            token_total = 0
            token_usage_complete = True
        elif token_usage_complete and isinstance(token_summary.get("total_tokens"), int):
            token_total = int(token_summary["total_tokens"])
        elif token_usage_complete and all(
            isinstance(token_summary.get(key), int) for key in ("input_tokens", "output_tokens")
        ):
            token_total = int(token_summary["input_tokens"]) + int(token_summary["output_tokens"])
        else:
            token_total = None
    elif isinstance(usage, dict) and ("input_tokens" in usage or "output_tokens" in usage):
        token_request_status = "requested"
        token_usage_complete = all(isinstance(usage.get(key), int) for key in ("input_tokens", "output_tokens"))
        token_total = int(usage["input_tokens"]) + int(usage["output_tokens"]) if token_usage_complete else None
    else:
        token_request_status = "unknown"
        token_usage_complete = None
        token_total = None
    investigation = state.get("investigation_state") or {}
    stop = investigation.get("stop_reason") if isinstance(investigation, dict) else None
    if not stop:
        stop = (analysis.get("investigation_summary") or {}).get("stop_reason")
    if isinstance(stop, dict):
        investigation_stop_code = str(stop.get("code") or "")
    else:
        investigation_stop_code = str(getattr(stop, "code", "") or "")
    agent_run = state.get("agent_run") or {}
    output_status = str(agent_run.get("status") or "") if isinstance(agent_run, dict) else ""
    if not output_status:
        if state.get("errors"):
            output_status = "failed"
        elif analysis.get("abstain") is True:
            output_status = "abstained"
        elif (state.get("finish_decision") or {}).get("action_type") == "FINISH":
            output_status = "answered"
        elif analysis:
            output_status = "incomplete"
        else:
            output_status = "missing"
    return {
        "evidence_count": len(state.get("evidence_bundles", [])),
        "tool_call_count": len(calls),
        "invalid_call_count": invalid_call_count,
        "duplicate_call_count": duplicate_call_count,
        "unauthorized_call_count": sum(
            1 for call in calls
            if str(call.get("status") or "").casefold() in {"unauthorized", "permission_denied"}
        ) + sum(
            1 for error in state.get("errors", [])
            if str(error.get("code") or "").casefold() in {"unauthorized", "permission_denied", "scope_violation"}
        ),
        "examined_dimensions": len(dimensions),
        "has_trend": any(str(item.get("purpose", "")) == "trend_baseline" for item in rc_evidence),
        "has_traceable_missing_trend": _has_traceable_missing_target_trend(state),
        "stop_reason": str(analysis.get("stop_reason") or ""),
        "investigation_stop_code": investigation_stop_code,
        "final_output_status": output_status,
        "abstain": bool(analysis.get("abstain")),
        "contribution_count": len(analysis.get("contribution") or []),
        "total_latency_ms": total_latency_ms,
        "total_tokens": token_total,
        "token_usage_complete": token_usage_complete,
        "token_request_status": token_request_status,
        "known_tokens_subtotal": (
            int(token_summary["known_tokens_subtotal"])
            if isinstance(token_summary, dict) and isinstance(token_summary.get("known_tokens_subtotal"), int)
            else None
        ),
        "estimated_cost": float(usage["estimated_cost"]) if isinstance(usage, dict) and usage.get("estimated_cost") is not None else None,
        "critic_verdict": state.get("critic_verdict"),
        "task_type": state.get("task_type"),
        "termination_initiator": state.get("termination_initiator"),
    }


def _validated_typed_evidence(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return typed evidence linked to a completed, valid action in target scope."""
    investigation = state.get("investigation_state") or {}
    if not isinstance(investigation, dict):
        return {}
    target = investigation.get("target") or {}
    if not isinstance(target, dict):
        return {}
    scope = str(target.get("scope_fingerprint") or "")
    metric = str(target.get("metric_id") or state.get("diagnostic_target_metric") or "")
    if not scope or not metric:
        return {}
    typed: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for evidence in investigation.get("evidence", []) or []:
        if not isinstance(evidence, dict) or not evidence.get("evidence_id"):
            continue
        eid = str(evidence["evidence_id"])
        if eid in typed:
            duplicates.add(eid)
        typed[eid] = evidence
    bound: dict[str, dict[str, Any]] = {}
    for result in investigation.get("action_results", []) or []:
        if not isinstance(result, dict):
            continue
        validation = result.get("validation_result") or {}
        valid = (
            result.get("status") == "completed"
            and isinstance(validation, dict)
            and validation.get("status") == "valid"
        )
        if not valid:
            continue
        for eid in result.get("evidence_ids", []) or []:
            eid = str(eid)
            evidence = typed.get(eid)
            if not evidence or eid in duplicates:
                continue
            if str(evidence.get("source_action_id") or "") != str(result.get("candidate_id") or ""):
                continue
            if str(evidence.get("scope_fingerprint") or "") != scope:
                continue
            if str(evidence.get("metric_id") or "").casefold() != metric.casefold():
                continue
            if result.get("metric_id") and str(result["metric_id"]).casefold() != metric.casefold():
                continue
            if result.get("method_id") and str(evidence.get("method_id") or "") != str(result["method_id"]):
                continue
            if evidence.get("trace_id") and state.get("trace_id") and str(evidence["trace_id"]) != str(state["trace_id"]):
                continue
            bound[eid] = evidence
    return bound


def _evidence_period_matches_target(evidence: dict[str, Any], target: dict[str, Any]) -> bool:
    comparison = evidence.get("comparison") or {}
    current = comparison.get("current_period") or comparison.get("period") or {}
    baseline = comparison.get("baseline_period") or {}
    if not isinstance(current, dict):
        return False
    for key in ("current_start", "current_end", "baseline_start", "baseline_end"):
        expected = target.get(key)
        if expected is None:
            continue
        source = baseline if key.startswith("baseline_") else current
        period_key = "start_date" if key.endswith("start") else "end_date"
        if str(source.get(period_key) or "") != str(expected):
            return False
    return bool(current.get("start_date") and current.get("end_date"))


def _cited_valid_evidence(state: dict[str, Any], allowed_types: set[str]) -> dict[str, dict[str, Any]]:
    valid = _validated_typed_evidence(state)
    analysis = state.get("analysis_result") or {}
    finish = state.get("finish_decision") or {}
    cited = {str(item) for item in (analysis.get("evidence_ids") or [])}
    cited.update(str(item) for item in (finish.get("evidence_ids") or []))
    target = ((state.get("investigation_state") or {}).get("target") or {})
    return {
        eid: evidence for eid, evidence in valid.items()
        if eid in cited
        and evidence.get("evidence_type") in allowed_types
        and _evidence_period_matches_target(evidence, target)
    }


_SAFE_DIAGNOSTIC_STOP_CODES = {"INVALID_DATA", "DATA_INSUFFICIENT", "CAPABILITY_GAP"}
_TERMINAL_PRESENTATION_STATUS: dict[str, set[str]] = {
    "ENOUGH_EVIDENCE": {"completed"},
    "COMPLETED": {"completed"},
    "INVALID_DATA": {"partial"},
    "DATA_INSUFFICIENT": {"partial"},
    # A capability limit may retain valid trend facts (partial) or have no usable
    # facts at all (unsupported); both are explicit limitation states.
    "CAPABILITY_GAP": {"partial", "unsupported"},
    "EXECUTION_FAILED": {"failed"},
}


def _terminal_code(state: dict[str, Any]) -> str | None:
    investigation = state.get("investigation_state") or {}
    analysis = state.get("analysis_result") or {}
    stop = investigation.get("stop_reason") if isinstance(investigation, dict) else None
    if not stop:
        stop = (analysis.get("investigation_summary") or {}).get("stop_reason") if isinstance(analysis, dict) else None
    if isinstance(stop, dict):
        return str(stop.get("code") or "").strip().upper() or None
    return str(getattr(stop, "code", "") or "").strip().upper() or None


def _presentation_status(state: dict[str, Any], response: dict[str, Any] | None = None) -> str | None:
    presentation = (response or {}).get("presentation") if isinstance(response, dict) else None
    if not isinstance(presentation, dict):
        presentation = state.get("analysis_presentation")
    return str(presentation.get("status") or "").strip().lower() or None if isinstance(presentation, dict) else None


def _trace_presentation_status(state: dict[str, Any]) -> str | None:
    presentation = state.get("analysis_presentation")
    return str(presentation.get("status") or "").strip().lower() or None if isinstance(presentation, dict) else None


def _cited_ids(state: dict[str, Any]) -> set[str]:
    analysis = state.get("analysis_result") or {}
    finish = state.get("finish_decision") or {}
    ids = {str(item) for item in analysis.get("evidence_ids", []) if item}
    ids.update(str(item) for item in finish.get("evidence_ids", []) if item)
    ids.update(str(item) for item in finish.get("supporting_evidence_ids", []) if item)
    for contribution in analysis.get("contribution", []) or []:
        if isinstance(contribution, dict):
            ids.update(str(item) for item in contribution.get("evidence_ids", []) if item)
    return ids


def _scope_valid_reconciliation_evidence(state: dict[str, Any]) -> bool:
    """Audit all raw inputs supporting an INVALID_DATA stop against the active query scope."""
    failures = state.get("diagnostic_data_reconciliation_failures") or []
    if not failures:
        return False
    bundles = {
        str(item.get("evidence_id")): item
        for item in state.get("evidence_bundles", [])
        if isinstance(item, dict) and item.get("evidence_id")
    }
    initial_result = state.get("tool_result") or {}
    initial_parsed = ((initial_result.get("result") or {}).get("parsed") or {}) if isinstance(initial_result, dict) else {}
    initial_scope = ((initial_parsed.get("authorized_metric") or {}).get("scope_fingerprint") or "")
    if not initial_scope:
        return False
    tool_calls = state.get("tool_calls") or []
    target = ((state.get("investigation_state") or {}).get("target") or {})
    required_dates = [
        str(target.get(key)) for key in ("baseline_start", "baseline_end", "current_start", "current_end")
        if target.get(key)
    ]
    for failure in failures:
        input_ids = [str(item) for item in failure.get("input_evidence_ids", [])]
        expected_metrics = {str(item).casefold() for item in failure.get("input_metric_ids", [])}
        if not input_ids or not expected_metrics:
            return False
        observed_scopes: set[str] = set()
        for evidence_id in input_ids:
            bundle = bundles.get(evidence_id)
            if not isinstance(bundle, dict) or bundle.get("trusted") is not True:
                return False
            status = bundle.get("validation_status")
            if status is not None and status != "accepted_by_investigation_engine":
                return False
            content = bundle.get("content") or {}
            response = content.get("result") if isinstance(content, dict) else None
            parsed = response.get("parsed") if isinstance(response, dict) else None
            auth = parsed.get("authorized_metric") if isinstance(parsed, dict) else None
            metric = str((parsed or {}).get("metric_name") or "").casefold() if isinstance(parsed, dict) else ""
            scope = str((auth or {}).get("scope_fingerprint") or "") if isinstance(auth, dict) else ""
            if not isinstance(response, dict) or response.get("status") != "ok" or metric not in expected_metrics or not scope:
                return False
            start = str((parsed or {}).get("start_date") or "") if isinstance(parsed, dict) else ""
            end = str((parsed or {}).get("end_date") or "") if isinstance(parsed, dict) else ""
            if required_dates and (not start or not end or any(not (start <= day <= end) for day in required_dates)):
                return False
            observed_scopes.add(scope)
            # Every non-primary item must have a successful tool-call record. The primary trend
            # is recorded as the workflow's initial tool_result by this graph path.
            matching_calls = [call for call in tool_calls if isinstance(call, dict) and str(call.get("evidence_id") or "") == evidence_id]
            is_initial_target = (
                metric == str(target.get("metric_id") or "").casefold()
                and evidence_id not in {str(call.get("evidence_id") or "") for call in tool_calls if isinstance(call, dict)}
                and isinstance(initial_result, dict)
                and str(initial_parsed.get("metric_name") or "").casefold() == metric
            )
            if matching_calls:
                if not any(str(call.get("status") or "").casefold() in {"ok", "success", "completed"} for call in matching_calls):
                    return False
            elif not is_initial_target:
                return False
        if len(observed_scopes) != 1 or initial_scope not in observed_scopes:
            return False
    return True


def _safe_stop_evidence_valid(state: dict[str, Any], code: str) -> bool:
    if code == "INVALID_DATA":
        return _scope_valid_reconciliation_evidence(state)
    if code == "DATA_INSUFFICIENT":
        target = str(
            state.get("diagnostic_target_metric")
            or ((state.get("investigation_state") or {}).get("target") or {}).get("metric_id")
            or ""
        )
        target_spec = ((state.get("investigation_state") or {}).get("target") or {})
        required_dates = [
            str(target_spec.get(key)) for key in ("baseline_start", "baseline_end", "current_start", "current_end")
            if target_spec.get(key)
        ]
        initial = state.get("tool_result") or {}
        initial_payload = initial.get("result") or {} if isinstance(initial, dict) else {}
        initial_parsed = initial_payload.get("parsed") or {} if isinstance(initial_payload, dict) else {}
        initial_authorized = initial_parsed.get("authorized_metric") or {} if isinstance(initial_parsed, dict) else {}
        active_scope = str(initial_authorized.get("scope_fingerprint") or "")
        candidates: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = []
        if isinstance(initial, dict) and initial_parsed:
            candidates.append((str(initial.get("evidence_id") or ""), initial_payload, None))
        bundles = {
            str(item.get("evidence_id")): item
            for item in state.get("evidence_bundles", [])
            if isinstance(item, dict) and item.get("evidence_id")
        }
        for call in state.get("tool_calls") or []:
            if not isinstance(call, dict) or call.get("tool_name") not in {"query_metric_trend", "registry_trend"}:
                continue
            if str(call.get("status") or "").casefold() not in {"ok", "success", "completed"}:
                continue
            if str((call.get("arguments") or {}).get("metric_name") or "").casefold() != target.casefold():
                continue
            evidence_id = str(call.get("evidence_id") or "")
            bundle = bundles.get(evidence_id)
            if not bundle or bundle.get("trusted") is not True:
                continue
            raw = ((bundle.get("content") or {}).get("result") or {})
            if not isinstance(raw, dict):
                continue
            envelope = raw.get("result") if isinstance(raw.get("result"), dict) else raw
            candidates.append((evidence_id, envelope, bundle))
        for _evidence_id, envelope, bundle in candidates:
            parsed = envelope.get("parsed") or {}
            auth = parsed.get("authorized_metric") or {}
            metric = str(parsed.get("metric_name") or "").casefold()
            scope = str(auth.get("scope_fingerprint") or "")
            start, end = str(parsed.get("start_date") or ""), str(parsed.get("end_date") or "")
            if not target or metric != target.casefold() or not active_scope or not scope or scope != active_scope:
                continue
            if required_dates and (not start or not end or any(not (start <= day <= end) for day in required_dates)):
                continue
            if bundle and bundle.get("validation_status") not in {None, "accepted_by_investigation_engine", "validated"}:
                continue
            trend = envelope.get("trend_result")
            metric_result = envelope.get("metric_result")
            missing = False
            if isinstance(trend, dict) and str(trend.get("metric_name") or "").casefold() == target.casefold():
                points = trend.get("points")
                missing = not isinstance(points, list) or not points or all(
                    not isinstance(point, dict)
                    or not isinstance(point.get("value"), (int, float))
                    or isinstance(point.get("value"), bool)
                    for point in points
                )
            if isinstance(metric_result, dict) and str(metric_result.get("metric_name") or "").casefold() == target.casefold():
                missing = missing or str(metric_result.get("status") or "").casefold() in {"not_found", "insufficient_data"}
            if missing:
                return True
        cited = _cited_ids(state)
        if cited:
            bundles = {str(item.get("evidence_id")): item for item in state.get("evidence_bundles", []) if isinstance(item, dict)}
            validated_typed = _validated_typed_evidence(state)
            return all(
                eid in bundles and bundles[eid].get("trusted") is True
                and (
                    bundles[eid].get("validation_status") in {"accepted_by_investigation_engine", "validated"}
                    or eid in validated_typed
                )
                for eid in cited
            )
        return False
    cited = _cited_ids(state)
    if not cited:
        # A structured capability/data limitation can be correct without making a factual claim.
        analysis = state.get("analysis_result") or {}
        finish = state.get("finish_decision") or {}
        return bool(analysis.get("abstain") or finish.get("action_type") == "ABSTAIN")
    bundles = {
        str(item.get("evidence_id")): item
        for item in state.get("evidence_bundles", [])
        if isinstance(item, dict) and item.get("evidence_id")
    }
    valid_typed = _validated_typed_evidence(state)
    return all(
        evidence_id in bundles
        and bundles[evidence_id].get("trusted") is True
        and (
            evidence_id in valid_typed
            or bundles[evidence_id].get("validation_status") in {"accepted_by_investigation_engine", "validated"}
        )
        for evidence_id in cited
    )


def score_diagnostic_run(
    state_or_trace: dict[str, Any],
    *,
    case_id: str,
    arm: str,
    replicate: int,
    expected_outcome: str,
    expected_stop_code: str | None = None,
    scenario_expectation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Versioned score for one case × arm × replicate, preserving unknowns and denominators.

    ``expected_outcome`` is one of ``numeric_decomposition``,
    ``statistical_contribution``, ``safe_stop`` or ``execution_failure``.
    """
    if not case_id.strip() or not arm.strip() or isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 1:
        raise ValueError("case_id and arm are required; replicate must be a positive integer")
    if expected_outcome not in {
        "numeric_decomposition", "statistical_contribution", "cross_metric_link",
        "safe_stop", "execution_failure",
    }:
        raise ValueError(f"Unsupported expected_outcome: {expected_outcome}")
    trace = state_or_trace
    state = trace.get("workflow_state", trace) if isinstance(trace, dict) else {}
    response = trace.get("workflow_result") if isinstance(trace, dict) else None
    analysis = state.get("analysis_result") or {}
    code = _terminal_code(state)
    presentation = _presentation_status(state, response)
    trace_presentation = _trace_presentation_status(state)
    expected_statuses = _TERMINAL_PRESENTATION_STATUS.get(code or "")
    observed_statuses = [item for item in (presentation, trace_presentation) if item]
    status_consistent: bool | None = (
        None if not code or not observed_statuses or not expected_statuses
        else all(item in expected_statuses for item in observed_statuses) and len(set(observed_statuses)) == 1
    )
    metrics = collect_anomaly_metrics(state)
    typed_decomposition = _cited_valid_evidence(state, {"metric_decomposition"})
    typed_dimension = _cited_valid_evidence(state, {"dimension_contribution"})
    finish = state.get("finish_decision") or {}
    numeric = bool(
        code == "ENOUGH_EVIDENCE"
        and finish.get("action_type") == "FINISH"
        and analysis.get("abstain") is False
        and typed_decomposition
    )
    has_statistical = any(
        isinstance(item, dict)
        and item.get("analysis_nature") == "descriptive_contribution"
        and item.get("items")
        and _contribution_has_matching_source(item, typed_dimension, ((state.get("investigation_state") or {}).get("target") or {}))
        for item in analysis.get("contribution", []) or []
    )
    statistical = bool(code == "ENOUGH_EVIDENCE" and typed_dimension and has_statistical)
    effective_type = "numeric_decomposition" if numeric else "statistical_contribution" if statistical else None
    effective = effective_type is not None
    safe_stop_behavior = bool(
        code in _SAFE_DIAGNOSTIC_STOP_CODES
        and analysis.get("abstain") is True
        and status_consistent is True
        and _safe_stop_evidence_valid(state, code)
    )
    evidence_valid: bool | None
    if effective_type == "numeric_decomposition":
        evidence_valid = bool(typed_decomposition)
    elif effective_type == "statistical_contribution":
        evidence_valid = bool(typed_dimension)
    elif code == "INVALID_DATA":
        evidence_valid = _scope_valid_reconciliation_evidence(state)
    elif code in {"DATA_INSUFFICIENT", "CAPABILITY_GAP"}:
        evidence_valid = _safe_stop_evidence_valid(state, code)
    elif _cited_ids(state):
        evidence_valid = _safe_stop_evidence_valid(state, code or "")
    else:
        evidence_valid = None
    checks = score_scenario(state, scenario_expectation) if scenario_expectation is not None else None
    if expected_outcome == "cross_metric_link":
        evidence_valid = bool(checks and checks.get("found_cross_metric_link") is True)
    correct_safe_stop = bool(expected_outcome == "safe_stop" and safe_stop_behavior)
    if expected_outcome == "numeric_decomposition":
        task_acceptance = numeric and status_consistent is True and evidence_valid is True
    elif expected_outcome == "statistical_contribution":
        task_acceptance = statistical and status_consistent is True and evidence_valid is True
    elif expected_outcome == "cross_metric_link":
        task_acceptance = bool(
            checks and checks.get("found_cross_metric_link") is True
            and status_consistent is True and evidence_valid is True
        )
    elif expected_outcome == "safe_stop":
        task_acceptance = correct_safe_stop and (expected_stop_code is None or code == expected_stop_code.upper())
    else:
        # An explicitly expected execution exception can meet the case contract, but never counts
        # as a correct safe stop or an effective diagnosis.
        task_acceptance = bool(code == "EXECUTION_FAILED" and status_consistent is True)
    if scenario_expectation is not None:
        task_acceptance = bool(task_acceptance and all(checks.values()))
    return {
        "scorer_version": DIAGNOSTIC_RUN_SCORER_VERSION,
        "case_id": case_id,
        "arm": arm,
        "replicate": replicate,
        "expected_outcome": expected_outcome,
        "expected_stop_code": expected_stop_code.upper() if expected_stop_code else None,
        "terminal_code": code,
        "presentation_status": presentation,
        "trace_presentation_status": trace_presentation,
        "expected_presentation_status": sorted(expected_statuses) if expected_statuses else None,
        "terminal_presentation_consistent": status_consistent,
        "task_acceptance_pass": bool(task_acceptance),
        "task_acceptance_denominator": 1,
        "effective_diagnosis_completed": effective,
        "effective_diagnosis_type": effective_type,
        "effective_diagnosis_denominator": 1,
        "safe_stop_behavior_observed": safe_stop_behavior,
        "correct_safe_stop": correct_safe_stop,
        "correct_safe_stop_denominator": 1,
        "conclusion_evidence_valid": evidence_valid,
        "evidence_validity_denominator": int(evidence_valid is not None),
        "scenario_checks": checks,
        "raw_metrics": metrics,
    }


def _contribution_has_matching_source(
    contribution: dict[str, Any], evidence_by_id: dict[str, dict[str, Any]], target: dict[str, Any],
) -> bool:
    dimension = str(contribution.get("dimension") or "")
    metric = str(contribution.get("metric_name") or "")
    cited = {str(item) for item in (contribution.get("evidence_ids") or [])}
    matching_evidence = [evidence_by_id[eid] for eid in cited if eid in evidence_by_id]
    if not dimension or not matching_evidence:
        return False
    if not any(
        str(((item.get("values") or {}).get("query") or {}).get("dimension") or "") == dimension
        and (not metric or str(item.get("metric_id") or "").casefold() == metric.casefold())
        for item in matching_evidence
    ):
        return False
    if target.get("current_end") and str(contribution.get("current_date") or "") != str(target["current_end"]):
        return False
    if target.get("baseline_end") and str(contribution.get("baseline_date") or "") != str(target["baseline_end"]):
        return False
    return True


def score_scenario(state: dict[str, Any], expect: dict[str, Any]) -> dict[str, bool]:
    """Score a root-cause run against the golden scenario's ground truth.

    ``expect`` keys: ``abstain`` (bool), ``found_factor`` / ``cross_metric_link``
    (a metric id, or a list of acceptable ids for multi-factor scenarios), and
    optional ``required_evidence_purposes`` (exact or prefix-matched evidence
    purposes that must be present).
    The mandatory ``evidence_examined`` bar keeps "gave up" from scoring as
    success — the rule must at least have looked at trend + a dimension.

    Substantive findings require a structured terminal decision and cited,
    validated evidence bound to the target metric, scope, and period. Optional
    text compatibility checks are supplementary and never replace that bar.
    """
    metrics = collect_anomaly_metrics(state)
    analysis = state.get("analysis_result", {}) or {}
    haystack = " ".join(filter(None, [
        str(analysis.get("summary") or ""),
        " ".join(str(claim.get("value") or "") for claim in state.get("claims", [])),
        " ".join(str(item.get("top_contributor") or "") for item in (analysis.get("contribution") or [])),
    ])).casefold()

    checks: dict[str, bool] = {
        "expectation_schema_valid": isinstance(expect, dict) and set(expect) <= _SCENARIO_EXPECT_KEYS,
        # Missing or flat-data controls correctly stop after checking the
        # trend; they need not fabricate a dimension drill-down.
        # A missing response is not a usable trend baseline. It counts as
        # examined only when the target query and its traceable result prove
        # the gap, and the expected outcome is an actual structured abstention.
        "evidence_examined": (
            metrics["has_trend"]
            or (
                bool(expect.get("abstain"))
                and bool(analysis.get("abstain"))
                and metrics["has_traceable_missing_trend"]
            )
        ) and (
            bool(expect.get("abstain"))
            or bool(expect.get("cross_metric_link"))
            or metrics["examined_dimensions"] >= 1
        ),
    }
    required_purposes = [str(item) for item in expect.get("required_evidence_purposes", [])]
    if required_purposes:
        purposes = [str(item.get("purpose") or "") for item in state.get("root_cause_evidence", [])]
        checks["required_evidence"] = all(
            any(purpose == required or purpose.startswith(required + ":") for purpose in purposes)
            for required in required_purposes
        )
    for unknown in set(expect) - _SCENARIO_EXPECT_KEYS:
        checks[f"unknown_expectation_{unknown}"] = False
    if expect.get("abstain"):
        checks["abstain"] = bool(analysis.get("abstain"))
    else:
        raw_target = expect.get("found_factor") or expect.get("cross_metric_link")
        acceptable = [str(item) for item in raw_target] if isinstance(raw_target, list) else ([str(raw_target)] if raw_target else [])
        finish = state.get("finish_decision") or {}
        structured = finish.get("found_factor")
        if expect.get("found_factor") and acceptable:
            decomposition = _cited_valid_evidence(state, {"metric_decomposition"})
            valid_factors = {
                str(factor).casefold()
                for evidence in decomposition.values()
                for factor in (evidence.get("values") or {}).get("factor_metrics", [])
            }
            checks[f"found_{acceptable[0]}"] = (
                finish.get("action_type") == "FINISH"
                and analysis.get("abstain") is False
                and str(structured or "").strip() in acceptable
                and bool(set(acceptable).intersection(valid_factors))
            )
            compatibility = expect.get("text_compatibility") or {}
            if compatibility.get("found_factor") is True:
                hints = tuple(h for factor in acceptable for h in _FINDING_TEXTS.get(factor, (factor,)))
                checks["text_compatibility_found_factor"] = any(hint.casefold() in haystack for hint in hints)
        elif expect.get("cross_metric_link") and acceptable:
            # A related trend is not itself a link, and a numeric decomposition
            # is not a cross-metric relationship. Require an explicit structured
            # terminal link backed by cited, registered-association evidence.
            finish = state.get("finish_decision") or {}
            target_metric = str(
                ((state.get("investigation_state") or {}).get("target") or {}).get("metric_id")
                or state.get("diagnostic_target_metric") or ""
            ).casefold()
            associations = _cited_valid_evidence(state, {"registered_association"})
            expected_links = {item.casefold() for item in acceptable}
            checks["found_cross_metric_link"] = bool(
                finish.get("action_type") == "FINISH"
                and analysis.get("abstain") is False
                and str(finish.get("cross_metric_link") or "").casefold() in expected_links
                and any(
                    str(evidence.get("metric_id") or "").casefold() == target_metric
                    and str(
                        (evidence.get("values") or {}).get("related_metric_id")
                        or ((evidence.get("values") or {}).get("result") or {}).get("related_metric_id")
                        or ""
                    ).casefold() in expected_links
                    for evidence in associations.values()
                )
            )
    if "found_dimension" in expect:
        wanted = bool(expect.get("found_dimension"))
        dimension_evidence = _cited_valid_evidence(state, {"dimension_contribution"})
        contributions = analysis.get("contribution") or []
        target = ((state.get("investigation_state") or {}).get("target") or {})
        structured_contribution = any(
            item.get("analysis_nature") == "descriptive_contribution"
            and item.get("items")
            and _contribution_has_matching_source(item, dimension_evidence, target)
            for item in contributions if isinstance(item, dict)
        )
        checks["found_dimension"] = bool(dimension_evidence) and structured_contribution if wanted else not structured_contribution
    requested_dimension = expect.get("reported_dimension_contribution")
    if requested_dimension:
        expected_dimension = (
            str(requested_dimension.get("dimension") or "")
            if isinstance(requested_dimension, dict) else str(requested_dimension)
        )
        expected_value = str(requested_dimension.get("dimension_value") or "") if isinstance(requested_dimension, dict) else ""
        expected_share = requested_dimension.get("contribution_share") if isinstance(requested_dimension, dict) else None
        share_tolerance = float(requested_dimension.get("share_tolerance", 0.02)) if isinstance(requested_dimension, dict) else 0.02
        contributions = analysis.get("contribution") or []
        dimension_evidence = _cited_valid_evidence(state, {"dimension_contribution"})
        target = ((state.get("investigation_state") or {}).get("target") or {})
        matching = [
            contribution for contribution in contributions
            if str(contribution.get("dimension") or "") == expected_dimension
            and contribution.get("analysis_nature") == "descriptive_contribution"
            and contribution.get("items")
            and _contribution_has_matching_source(contribution, dimension_evidence, target)
        ]
        answer_text = " ".join(str(state.get(key) or "") for key in ("answer", "final_answer")).strip()
        if not answer_text:
            answer_text = str(analysis.get("summary") or "")
        top_named = False
        share_matches = expected_share is None
        for contribution in matching:
            top = max(
                contribution["items"],
                key=lambda item: abs(float(item.get("delta_value") or 0)),
            )
            label = str(top.get("dimension_value") or "")
            correct_value = not expected_value or label.casefold() == expected_value.casefold()
            if label and correct_value and label.casefold() in answer_text.casefold():
                top_named = True
                if expected_share is not None:
                    actual_share = top.get("contribution_ratio")
                    if isinstance(actual_share, (int, float)):
                        share_matches = abs(abs(float(actual_share)) - float(expected_share)) <= share_tolerance
                        rendered_shares = [float(value) / 100 for value in re.findall(r"(\d+(?:\.\d+)?)\s*%", answer_text)]
                        share_matches = share_matches and any(
                            abs(value - float(expected_share)) <= share_tolerance
                            for value in rendered_shares
                        )
                break
        checks["reported_dimension_contribution"] = bool(matching and top_named)
        if expected_share is not None:
            checks["dimension_share_matches_source"] = bool(top_named and share_matches)
        if expect.get("business_cause_unresolved"):
            readiness = analysis.get("causal_readiness") or {}
            checks["business_cause_unresolved"] = (
                bool(analysis.get("abstain"))
                and readiness.get("causal_effect_estimated") is False
                and bool(matching)
                and bool(dimension_evidence)
            )
    return checks


def score_trajectory(state: dict[str, Any]) -> dict[str, Any]:
    history = [item.get("node") for item in state.get("node_history", [])]
    claims = state.get("claims", [])
    bundles = state.get("evidence_bundles", [])
    checks = {
        "has_handoff": bool(state.get("handoffs")),
        "has_critic": "critic" in history,
        "terminates": state.get("critic_verdict") in {"approved", "rejected"},
        # A knowledge-only clarification may intentionally make no factual
        # claim.  It is still evidence-grounded when it carries a reviewed
        # RAG/root-cause bundle; do not force it to manufacture a metric
        # claim merely to satisfy the evaluator.
        "evidence_bound": (
            bool(claims) and all(claim.get("source_evidence_ids") for claim in claims)
        ) or (not claims and bool(bundles)),
        "tool_parameters": not state.get("tool_calls") or all(call.get("arguments") is not None for call in state.get("tool_calls", [])),
        "citation_coverage": bool(state.get("rag_citations") or state.get("tool_calls")),
    }
    metrics = collect_anomaly_metrics(state)
    checks["metrics"] = {
        "evidence_coverage": (
            round(
                sum(bool(claim.get("source_evidence_ids")) for claim in claims) / len(claims),
                4,
            )
            if claims else (1.0 if bundles else 0.0)
        ),
        "invalid_call_count": metrics["invalid_call_count"],
        "duplicate_call_count": metrics["duplicate_call_count"],
        "latency_ms": metrics["total_latency_ms"],
        "tokens": metrics["total_tokens"],
        "estimated_cost": metrics["estimated_cost"],
    }
    return checks


def summarize_anomaly_trajectories(
    entries: list[dict[str, Any]],
    *,
    category: str = "root_cause_scenario",
) -> dict[str, Any]:
    """Aggregate Phase 0's diagnostic metrics without hiding the control-arm failures.

    ``category`` selects which entries aggregate together — the promotion gate
    uses the frozen ``root_cause_scenario`` set, the long-tail report aggregates
    ``root_cause_longtail`` separately (review follow-up #7: rule path and
    long-tail path are never mixed into one number).
    """
    scenarios = [item for item in entries if item.get("category") == category]
    if not scenarios:
        return {
            "completion_rate": 0.0,
            "evidence_coverage": 0.0,
            "correct_abstain_rate": 0.0,
            "invalid_call_count": 0,
            "duplicate_call_count": 0,
            "unauthorized_call_count": 0,
            "trajectory_branch_rate": 0.0,
            "p95_latency_ms": 0.0,
            "total_tokens": None,
            "token_runs": 0,
            "token_unknown_runs": 0,
            "total_cost": 0.0,
            "termination_initiator_counts": {},
        }
    metric_rows = [item.get("scenario_metrics", {}) for item in scenarios]
    passed = sum(bool(item.get("passed")) for item in scenarios)
    abstain_cases = [item for item in scenarios if item.get("expected", {}).get("abstain")]
    correct_abstain = sum(
        bool(item.get("scenario_checks", {}).get("abstain")) for item in abstain_cases
    )
    paths = {
        tuple(
            (event.get("node"), event.get("summary") if event.get("node") == "diagnostic_decide" else "")
            for event in item.get("node_history", [])
        )
        for item in scenarios
        if item.get("node_history")
    }
    latencies = sorted(float(row.get("total_latency_ms", 0) or 0) for row in metric_rows)
    p95_index = max(0, min(len(latencies) - 1, ceil(len(latencies) * 0.95) - 1))
    termination_counts: dict[str, int] = {}
    for item in scenarios:
        initiator = str(item.get("termination_initiator") or item.get("scenario_metrics", {}).get("termination_initiator") or "unknown")
        termination_counts[initiator] = termination_counts.get(initiator, 0) + 1
    return {
        "completion_rate": rate(passed, len(scenarios)),
        "evidence_coverage": rate(
            sum(bool(item.get("scenario_checks", {}).get("evidence_examined")) for item in scenarios),
            len(scenarios),
        ),
        "correct_abstain_rate": rate(correct_abstain, len(abstain_cases)),
        "invalid_call_count": sum(int(row.get("invalid_call_count", 0) or 0) for row in metric_rows),
        "duplicate_call_count": sum(int(row.get("duplicate_call_count", 0) or 0) for row in metric_rows),
        "unauthorized_call_count": sum(int(row.get("unauthorized_call_count", 0) or 0) for row in metric_rows),
        "trajectory_branch_rate": round(max(0, len(paths) - 1) / len(scenarios), 4),
        "p95_latency_ms": round(latencies[p95_index], 3),
        "total_tokens": (
            sum(int(row["total_tokens"]) for row in metric_rows)
            if all(row.get("total_tokens") is not None for row in metric_rows) else None
        ),
        "token_runs": sum(row.get("total_tokens") is not None for row in metric_rows),
        "token_unknown_runs": sum(row.get("total_tokens") is None for row in metric_rows),
        "total_cost": round(sum(float(row.get("estimated_cost", 0) or 0) for row in metric_rows), 8),
        "termination_initiator_counts": termination_counts,
    }


def score_phase6_report(report: dict[str, Any]) -> dict[str, bool]:
    metrics = report.get("metrics", [])
    charts = report.get("charts", [])
    evidence = report.get("evidence_records", {})
    references = [
        evidence_id
        for section in (metrics, report.get("anomalies", []), report.get("contributions", []), charts)
        for item in section
        for evidence_id in item.get("evidence_ids", [])
    ]
    chart_payloads_valid = all(
        item.get("status") == "chart_unavailable"
        or (
            item.get("path") and Path(item["path"]).is_file()
            and sha256(Path(item["path"]).read_bytes()).hexdigest() == item.get("content_hash")
        )
        # 新契约：图表是纯结构化数据（无 path 落盘），ok 状态必须携带
        # content_hash、evidence_ids 和 data_points，与 ChartArtifact 校验器一致。
        or (
            not item.get("path") and item.get("content_hash")
            and item.get("evidence_ids") and item.get("data_points")
        )
        for item in charts
    )
    return {
        "periodic": report.get("report_type") in {"weekly", "monthly"},
        "comparisons": bool(metrics) and all(item.get("period_over_period") and item.get("year_over_year") for item in metrics),
        "evidence_bound": bool(evidence) and bool(references) and all(item in evidence for item in references),
        "lineage": bool(report.get("lineage_refs")),
        "chart_contract": (
            {item.get("chart_type") for item in charts} >= {"trend", "contribution"}
            and all(item.get("status") in {"ok", "chart_unavailable"} for item in charts)
            and chart_payloads_valid
        ),
    }

"""Granular, case-bound audit records for future B/G run records.

This is an evaluation projection only: it does not change agent decisions or
retroactively fill missing historical fields. Any fact that cannot be proven
from the run's own trace remains ``None``/``unknown``.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Mapping


AUDIT_SCHEMA_VERSION = "resume-bg-companion-audit-v4"


def _inner_trace(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    value = trace.get("trace")
    return value if isinstance(value, Mapping) else trace


def _state(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    inner = _inner_trace(trace)
    for key in ("state_snapshot", "workflow_state", "state"):
        value = inner.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _pointer(path: str, key: Any) -> str:
    token = str(key).replace("~", "~0").replace("/", "~1")
    return f"{path}/{token}"


def _fingerprint(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) == 64 and all(char in "0123456789abcdefABCDEF" for char in text):
        return text.lower()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _reference_occurrences(value: Any, path: str) -> list[dict[str, str]]:
    """Enumerate explicit evidence references, preserving duplicate occurrences."""
    found: list[dict[str, str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = _pointer(path, key)
            if key in {"evidence_ids", "source_evidence_ids"} and isinstance(child, list):
                found.extend(
                    {"evidence_id": str(item), "location": f"{child_path}/{index}"}
                    for index, item in enumerate(child) if item
                )
            elif key == "evidence_id" and isinstance(child, str) and child:
                found.append({"evidence_id": child, "location": child_path})
            else:
                found.extend(_reference_occurrences(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_reference_occurrences(child, f"{path}/{index}"))
    return found


def _all_refs(state: Mapping[str, Any]) -> list[dict[str, str]]:
    presentation = state.get("analysis_presentation") or state.get("final_display") or state.get("presentation")
    if isinstance(presentation, Mapping) and isinstance(presentation.get("presentation"), Mapping):
        presentation = presentation["presentation"]
    refs: list[dict[str, str]] = []
    if isinstance(presentation, Mapping):
        refs.extend(_reference_occurrences(presentation, "/analysis_presentation"))
    refs.extend(_reference_occurrences(state.get("claims") or [], "/claims"))
    # The bare loop intentionally does not construct governed ``claims`` or
    # claim-evidence bindings.  Its raw model final is nevertheless retained,
    # so audit exactly the IDs the model supplied rather than treating its
    # citations as structurally unobservable.  This read does not validate,
    # add, or promote those IDs into bindings.
    model_final = state.get("model_authored_final")
    if isinstance(model_final, Mapping):
        refs.extend(_reference_occurrences(model_final.get("claims") or [], "/model_authored_final/claims"))
    return refs


def _metric_from_bundle(bundle: Mapping[str, Any]) -> str | None:
    content = bundle.get("content") if isinstance(bundle.get("content"), Mapping) else {}
    for candidate in (content.get("metric_id"), content.get("metric_name")):
        if candidate:
            return str(candidate)
    result = content.get("result") if isinstance(content.get("result"), Mapping) else {}
    parsed = result.get("parsed") if isinstance(result.get("parsed"), Mapping) else {}
    for candidate in (parsed.get("metric_name"), parsed.get("metric_id"), result.get("metric_name"), result.get("metric_id")):
        if candidate:
            return str(candidate)
    values = content.get("values") if isinstance(content.get("values"), Mapping) else {}
    query = values.get("query") if isinstance(values.get("query"), Mapping) else {}
    return str(query.get("metric_name")) if query.get("metric_name") else None


def _call_arguments(call: Mapping[str, Any]) -> Mapping[str, Any]:
    args = call.get("arguments")
    if isinstance(args, Mapping):
        # Registry trend calls persist their temporal contract under ``period``.
        # Flatten that *recorded* nested object for auditing so a complete
        # request is not incorrectly reported as unknown merely because its
        # transport shape differs from other query tools.
        result = dict(args)
        period = args.get("period")
        if isinstance(period, Mapping):
            for key in (
                "metric_date", "start_date", "end_date", "baseline_date", "current_date",
                "baseline_start", "baseline_end", "current_start", "current_end", "granularity",
            ):
                if result.get(key) is None and period.get(key) is not None:
                    result[key] = period[key]
        return result
    response = call.get("response")
    if isinstance(response, Mapping) and isinstance(response.get("arguments"), Mapping):
        return response["arguments"]
    return {}


def _is_capability_stop_without_query(
    decision_record: Mapping[str, Any] | None,
) -> bool:
    """Whether this is a deliberate capability stop rather than a data call.

    Such a control event has already passed the authorization gate, but no
    business-data query is applicable or executed.  It must remain visible in
    the audit while staying out of authorization-violation and query-validity
    denominators.
    """
    return bool(
        decision_record
        and decision_record.get("authorization_checked") is True
        and decision_record.get("sql_executed") is False
        and str(decision_record.get("limitation_code") or "") == "CAPABILITY_GAP"
    )


def _call_response_status(call: Mapping[str, Any]) -> str:
    status = call.get("status")
    if status in {"ok", "not_found", "unsupported", "failed", "error", "execution_failed", "refused_before_execution"}:
        return str(status)
    response = call.get("response")
    result = response.get("result") if isinstance(response, Mapping) else None
    if isinstance(result, Mapping) and result.get("status"):
        return str(result["status"])
    if call.get("call_status"):
        return str(call["call_status"])
    if status:
        return str(status)
    return "unknown"


def _period_matches(case: Mapping[str, Any], call: Mapping[str, Any]) -> bool | None:
    expected = case.get("period") if isinstance(case.get("period"), Mapping) else {}
    args = dict(_call_arguments(call))
    comparison = args.get("comparison") if isinstance(args.get("comparison"), Mapping) else {}
    for key in ("baseline_date", "current_date", "baseline_start", "baseline_end", "current_start", "current_end", "granularity", "start_date", "end_date", "metric_date"):
        if args.get(key) is None and comparison.get(key) is not None:
            args[key] = comparison[key]
    kind = expected.get("kind")
    operation = str(call.get("planned_operation") or call.get("tool_name") or call.get("tool") or "")
    if case.get("operation") == "multi_step_diagnostic":
        if "trend" in operation:
            start, end = args.get("start_date"), args.get("end_date")
            return None if start is None or end is None else str(start) == str(expected.get("baseline")) and str(end) == str(expected.get("current"))
        if "contribution" in operation or "breakdown" in operation:
            base = args.get("baseline_date") or args.get("left_date")
            current = args.get("current_date") or args.get("right_date")
            if base is None or current is None:
                return None
            return str(base) == str(expected.get("baseline")) and str(current) == str(expected.get("current"))
    if kind == "point":
        actual = args.get("metric_date")
        return None if actual is None else str(actual) == str(expected.get("date"))
    if kind == "range":
        start, end = args.get("start_date"), args.get("end_date")
        return None if start is None or end is None else str(start) == str(expected.get("start")) and str(end) == str(expected.get("end"))
    if kind == "point_pair":
        base = args.get("baseline_date") or args.get("left_date")
        current = args.get("current_date") or args.get("right_date")
        return None if base is None or current is None else str(base) == str(expected.get("baseline")) and str(current) == str(expected.get("current"))
    if kind == "period_pair":
        keys = ("baseline_start", "baseline_end", "current_start", "current_end", "granularity")
        if any(args.get(key) is None for key in keys):
            return None
        return all(str(args.get(key)) == str(expected.get(key)) for key in keys)
    return None


def _operation_matches(case: Mapping[str, Any], call: Mapping[str, Any]) -> bool | None:
    actual = str(call.get("planned_operation") or call.get("tool") or call.get("tool_name") or "")
    expected = str(case.get("operation") or "")
    if expected == "multi_step_diagnostic":
        return _normalize_operation(actual) in {"metric_trend", "metric_contribution"}
    if expected == "expected_data_insufficiency":
        return _normalize_operation(actual) in {"metric_compare", "metric_value", "metric_trend"}
    return None if not actual else _normalize_operation(actual) == _normalize_operation(expected)


def _normalize_operation(value: str) -> str:
    return {
        "compare_breakdown_by_dimension": "metric_contribution",
        "compare_breakdown_by_dimension_period": "metric_contribution",
        "dimension_contribution": "metric_contribution",
        "registry_compare": "metric_compare",
        "compare_metric": "metric_compare",
        "registry_value": "metric_value",
        "query_value": "metric_value",
        "registry_trend": "metric_trend",
        "query_trend": "metric_trend",
    }.get(value, value)


def _bundle_call_map(state: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(item.get("request_id")): item
        for item in state.get("tool_calls") or []
        if isinstance(item, Mapping) and item.get("request_id")
    }


def _call_authorized_metric(call: Mapping[str, Any], state: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, str | None]:
    """Find the server AuthorizedMetric tied to this exact request ID.

    Prefer an execution-time snapshot. For older traces, only use an envelope
    or evidence bundle whose request ID matches the tool call. Never borrow an
    unrelated run-level AuthorizedMetric to make a call look authorized.
    """
    request_id = str(call.get("request_id") or "")
    direct = call.get("authorization_snapshot")
    if isinstance(direct, Mapping):
        authorized = direct.get("authorized_metric")
        if isinstance(authorized, Mapping) and str(direct.get("request_id") or "") == request_id:
            return authorized, "tool_call.authorization_snapshot"
    response = call.get("response") if isinstance(call.get("response"), Mapping) else {}
    candidates: list[tuple[Mapping[str, Any], str]] = []
    for label, envelope in (("tool_call.response", response),):
        result = envelope.get("result") if isinstance(envelope.get("result"), Mapping) else {}
        parsed = result.get("parsed") if isinstance(result.get("parsed"), Mapping) else {}
        authorized = parsed.get("authorized_metric")
        if isinstance(authorized, Mapping) and str(envelope.get("request_id") or "") == request_id:
            candidates.append((authorized, label + ".result.parsed"))
    for index, bundle in enumerate(state.get("evidence_bundles") or []):
        if not isinstance(bundle, Mapping):
            continue
        content = bundle.get("content") if isinstance(bundle.get("content"), Mapping) else {}
        if str(content.get("request_id") or "") != request_id:
            continue
        result = content.get("result") if isinstance(content.get("result"), Mapping) else {}
        parsed = result.get("parsed") if isinstance(result.get("parsed"), Mapping) else {}
        authorized = parsed.get("authorized_metric")
        if isinstance(authorized, Mapping):
            candidates.append((authorized, f"evidence_bundles/{index}.content.result.parsed"))
    if candidates:
        return candidates[0]
    # A run-level authorized-metric list cannot prove which individual call
    # received the grant. Keep the call-level snapshot unknown instead of
    # borrowing a same-metric object from another call in this run.
    return None, None


def _authorization_snapshot(
    case: Mapping[str, Any], state: Mapping[str, Any], call: Mapping[str, Any],
    decision_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    args = _call_arguments(call)
    request_id = str(call.get("request_id") or "")
    metric_id = str(call.get("planned_metric_id") or args.get("metric_id") or args.get("metric_name") or "")
    status = _call_response_status(call)
    authorized, authorized_source = _call_authorized_metric(call, state)
    authorization_request_bound = bool(
        authorized_source and authorized_source != "run_state.authorized_metrics_legacy_fallback"
    )
    authorized_metric_id = str((authorized or {}).get("metric_id") or "")
    metric_match = None if not metric_id or not authorized_metric_id else metric_id == authorized_metric_id
    scope = str(args.get("scope") or state.get("scope_type") or "")
    scope_source = "tool_arguments" if args.get("scope") else "run_identity.scope_type" if state.get("scope_type") else None
    expected_scope = str(case.get("scope") or "")
    scope_match = None if not scope or not expected_scope else scope == expected_scope
    embedded_snapshot = call.get("authorization_snapshot") if isinstance(call.get("authorization_snapshot"), Mapping) else None
    embedded_snapshot_request_match = (
        None if embedded_snapshot is None
        else bool(request_id and str(embedded_snapshot.get("request_id") or "") == request_id)
    )
    explicit_decision = None
    decision_source = None
    if embedded_snapshot_request_match is True:
        explicit_decision = embedded_snapshot.get("decision")
        decision_source = embedded_snapshot.get("decision_source") or "tool_call.authorization_snapshot"
    if embedded_snapshot_request_match is False:
        decision = "unknown"
        decision_source = "tool_call_authorization_snapshot_request_id_mismatch"
    elif explicit_decision in {"allowed", "denied"}:
        decision = explicit_decision
    elif decision_record and decision_record.get("authorization_decision") in {"allowed", "denied"}:
        decision = decision_record.get("authorization_decision")
        decision_source = "request_correlated_decision_record"
    elif decision_record and decision_record.get("authorization_checked") is True:
        limitation = str(decision_record.get("limitation_code") or "")
        if limitation in {"AUTH_DENIED", "SCOPE_REJECTED", "AUTHORIZATION_BLOCKED"} and decision_record.get("sql_executed") is False:
            decision = "denied"
            decision_source = "request_correlated_pre_execution_denial"
        elif _is_capability_stop_without_query(decision_record):
            decision = "not_applicable"
            decision_source = "authorization_checked_capability_stop_without_query"
        elif decision_record.get("sql_executed") is True and authorized and metric_match is True and scope_match is True and authorization_request_bound:
            decision = "allowed"
            decision_source = "request_correlated_authorized_execution"
        else:
            decision = "unknown"
            decision_source = "request_correlated_check_without_decisive_outcome"
    elif status in {"ok", "not_found"} and authorized and metric_match is True and scope_match is True and authorization_request_bound:
        decision = "allowed"
        decision_source = "same_request_authorized_metric_and_successful_query_result"
    elif status in {"unsupported", "refused_before_execution"} and decision_record:
        limitation = str(decision_record.get("limitation_code") or "")
        decision = "denied" if limitation in {"AUTH_DENIED", "SCOPE_REJECTED", "AUTHORIZATION_BLOCKED"} else "unknown"
        decision_source = "request_correlated_controlled_refusal"
    else:
        decision = "unknown"
        decision_source = "insufficient_request_correlated_authorization_evidence"
    identity = {
        "tenant_id": state.get("tenant_id"), "user_id": state.get("user_id"),
        "role": state.get("user_role"), "scope_type": state.get("scope_type"),
    }
    identity_hash = hashlib.sha256(json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")).hexdigest() if any(identity.values()) else None
    boundary = state.get("authorization_boundary") if isinstance(state.get("authorization_boundary"), Mapping) else {}
    return {
        "schema_version": "per-tool-authorization-snapshot-v1",
        "request_id": request_id or None,
        "decision": decision,
        "decision_source": decision_source,
        "embedded_snapshot_request_match": embedded_snapshot_request_match,
        "authorization_checked": bool(
            (decision_record or {}).get("authorization_checked") is True
            or (embedded_snapshot_request_match is True
                and isinstance(call.get("authorization_snapshot"), Mapping)
                and call["authorization_snapshot"].get("authorization_checked") is True)
            or (status in {"ok", "not_found"} and authorized is not None and metric_match is True and scope_match is True and authorization_request_bound)
        ),
        "metric_id": metric_id or None,
        "authorized_metric_id": authorized_metric_id or None,
        "metric_match": metric_match,
        "scope": scope or None,
        "scope_source": scope_source,
        "expected_scope": expected_scope or None,
        "scope_match": scope_match,
        "operation": call.get("planned_operation") or call.get("tool_name") or call.get("tool"),
        "dimension": args.get("dimension"),
        "period": {key: args[key] for key in (
            "metric_date", "baseline_date", "current_date", "start_date", "end_date",
            "baseline_start", "baseline_end", "current_start", "current_end", "granularity",
        ) if args.get(key) is not None},
        "tool": call.get("tool_name") or call.get("tool"),
        "tool_result_status": status,
        "authorized_metric_source": authorized_source,
        "authorized_metric_present": authorized is not None,
        "authorization_request_bound": authorization_request_bound,
        "authorized_metric": ({
            "metric_id": authorized_metric_id,
            "provenance_type": authorized.get("provenance_type"),
            "authorization_fingerprint": _fingerprint(authorized.get("authorization_fingerprint")),
            "query_scope_fingerprint": _fingerprint(authorized.get("query_scope_fingerprint") or authorized.get("scope_fingerprint")),
            "catalog_digest": _fingerprint(authorized.get("catalog_digest")),
        } if authorized else None),
        "identity_sha256": identity_hash,
        "identity_role": state.get("user_role"),
        "identity_scope_type": state.get("scope_type"),
        "permission_version": boundary.get("permission_version"),
        "authorization_boundary_scope_sha256": _fingerprint(boundary.get("query_scope_fingerprint")),
        "decision_record_id": (decision_record or {}).get("decision_id"),
        "sql_executed": (decision_record or {}).get("sql_executed") if decision_record else status in {"ok", "not_found"},
    }


def _citation_audit(case: Mapping[str, Any], state: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
    bundles = {
        str(item.get("evidence_id")): item
        for item in state.get("evidence_bundles") or []
        if isinstance(item, Mapping) and item.get("evidence_id")
    }
    calls = _bundle_call_map(state)
    refs = _all_refs(state)
    occurrences: list[dict[str, Any]] = []
    for index, ref in enumerate(refs, 1):
        evidence_id = ref["evidence_id"]
        bundle = bundles.get(evidence_id)
        content = bundle.get("content") if isinstance(bundle, Mapping) and isinstance(bundle.get("content"), Mapping) else {}
        request_id = content.get("request_id")
        call = calls.get(str(request_id)) if request_id else None
        metric = _metric_from_bundle(bundle) if bundle else None
        expected_metric = str(case.get("target_metric") or "")
        metric_match = None if metric is None or not expected_metric else metric == expected_metric
        call_args = _call_arguments(call) if call else {}
        scope = call_args.get("scope") or observation.get("scope") or state.get("scope_type")
        expected_scope = case.get("scope")
        scope_match = None if scope is None or expected_scope is None else scope == expected_scope
        period_match = _period_matches(case, call) if call else None
        # For a multi-step run, the exact evidence may be from either required
        # operation; match its own call period rather than the first call only.
        trusted = bundle.get("trusted") if isinstance(bundle, Mapping) else None
        if bundle is not None and trusted is None:
            trusted = True  # EvidenceBundle's schema default is trusted=True.
        request_match = None if not request_id else bool(call and str(call.get("request_id")) == str(request_id))
        invalid = bundle is None or metric_match is False or scope_match is False or period_match is False or trusted is False or request_match is False
        unknown = bundle is not None and any(v is None for v in (metric_match, scope_match, period_match, trusted, request_match))
        valid = not invalid and not unknown
        occurrences.append({
            "case_id": case.get("case_id"), "arm": case.get("arm"), "run_id": state.get("trace_id"),
            "object_id": evidence_id, "reference_index": index, "result": "valid" if valid else "invalid" if invalid else "unknown",
            "reason": "same-run evidence exists and metric/scope/period/trust checks matched" if valid else "evidence ID was not found in this run" if bundle is None else "one or more citation checks failed" if invalid else "trace lacks fields needed to verify metric/scope/period",
            "evidence_location": ref["location"], "bundle_location": f"/evidence_bundles/{evidence_id}" if bundle else None,
            "request_id": request_id, "metric_id": metric, "metric_match": metric_match,
            "scope_match": scope_match, "period_match": period_match, "request_match": request_match, "trusted": trusted,
        })
    counts = Counter(item["result"] for item in occurrences)
    return {
        "denominator": len(occurrences), "numerator": counts["valid"],
        "rate": counts["valid"] / len(occurrences) if occurrences else None,
        "unknown_count": counts["unknown"], "invalid_count": counts["invalid"],
        "occurrences": occurrences,
        "no_enumerable_references": not occurrences,
    }


def _authorization_audit(case: Mapping[str, Any], state: Mapping[str, Any]) -> list[dict[str, Any]]:
    explicit = [item for item in state.get("decision_records") or [] if isinstance(item, Mapping)]
    scope_type = state.get("scope_type")
    result: list[dict[str, Any]] = []
    for index, call in enumerate(state.get("tool_calls") or [], 1):
        if not isinstance(call, Mapping):
            continue
        args = _call_arguments(call)
        metric_id = str(call.get("planned_metric_id") or args.get("metric_id") or args.get("metric_name") or "")
        request_id = str(call.get("request_id") or "")
        decision_record = next((item for item in explicit if str(item.get("request_id") or "") == request_id), None)
        snapshot = _authorization_snapshot(case, state, call, decision_record)
        decision = snapshot["decision"]
        basis = snapshot["decision_source"]
        auth = snapshot["authorized_metric"]
        snapshot_integrity = (
            "not_applicable" if decision == "not_applicable"
            else "invalid" if not request_id or snapshot.get("embedded_snapshot_request_match") is False or snapshot.get("metric_match") is False or snapshot.get("scope_match") is False
            else "valid" if snapshot.get("metric_match") is True and snapshot.get("scope_match") is True and snapshot.get("authorization_request_bound") is True
            else "unknown"
        )
        result.append({
            "case_id": case.get("case_id"), "arm": case.get("arm"), "run_id": state.get("trace_id"),
            "object_id": request_id or f"tool_call:{index}", "request_id": request_id,
            "result": decision, "reason": basis,
            "evidence_location": f"/tool_calls/{index-1}",
            "tool": call.get("tool_name") or call.get("tool"), "metric_id": metric_id or None,
            "operation": call.get("planned_operation"), "dimension": args.get("dimension"),
            "scope": args.get("scope") or scope_type,
            "scope_fingerprint": (auth or {}).get("query_scope_fingerprint"),
            "authorization_fingerprint": (auth or {}).get("authorization_fingerprint"),
            "catalog_digest": (auth or {}).get("catalog_digest"),
            "policy_version": snapshot.get("permission_version"),
            "authorized_metric_present": snapshot["authorized_metric_present"],
            "snapshot": snapshot,
            "snapshot_integrity": snapshot_integrity,
        })
    call_request_ids = {str(item.get("request_id")) for item in state.get("tool_calls") or [] if isinstance(item, Mapping) and item.get("request_id")}
    for index, decision_record in enumerate(explicit, 1):
        request_id = str(decision_record.get("request_id") or "")
        if request_id and request_id in call_request_ids:
            continue
        limitation = str(decision_record.get("limitation_code") or "")
        if decision_record.get("authorization_decision") in {"allowed", "denied"}:
            decision, basis = decision_record["authorization_decision"], "explicit persisted authorization_decision"
        elif decision_record.get("authorization_checked") is True and limitation in {"AUTH_DENIED", "SCOPE_REJECTED"} and decision_record.get("sql_executed") is False:
            decision, basis = "denied", "explicit authorization/scope denial before SQL"
        elif _is_capability_stop_without_query(decision_record):
            # This decision record represents a checked control-plane stop,
            # not an attempted business-data call.  There is no authorization
            # decision to score in the call-rate denominator.
            decision, basis = "not_applicable", "authorization checked; capability unavailable; no data query executed"
        elif decision_record.get("authorization_checked") is True and decision_record.get("sql_executed") is True:
            decision, basis = "allowed", "authorization check recorded before SQL execution"
        else:
            decision, basis = "unknown", "decision record does not establish authorization outcome"
        result.append({
            "case_id": case.get("case_id"), "arm": case.get("arm"), "run_id": state.get("trace_id"),
            "object_id": request_id or f"decision_record:{index}", "request_id": request_id or None,
            "result": decision, "reason": basis, "evidence_location": f"/decision_records/{index-1}",
            "tool": decision_record.get("requested_tool"), "metric_id": decision_record.get("metric_id"),
            "operation": decision_record.get("operation"), "dimension": decision_record.get("dimension"),
            "scope": decision_record.get("scope") or scope_type,
            "limitation_code": limitation or None,
            "authorization_checked": decision_record.get("authorization_checked"),
            "sql_executed": decision_record.get("sql_executed"),
            "authorized_metric_present": None,
            # Decision records without a ToolInvocation do not carry a tool
            # authorization snapshot.  A verified capability stop is
            # deliberately outside that denominator rather than an unknown
            # snapshot-quality result.
            "snapshot_integrity": "not_applicable" if decision == "not_applicable" else "unknown",
        })
    return result


def _call_audit(case: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    calls = [item for item in state.get("tool_calls") or [] if isinstance(item, Mapping)]
    expected_metric = str(case.get("target_metric") or "")
    expected_dimension = case.get("dimension")
    expected_stop = (case.get("expected") or {}).get("terminal_code")
    expected_outcome = (case.get("expected") or {}).get("outcome")
    safe_stop = expected_outcome == "safe_stop"
    seen: dict[tuple[Any, ...], str | None] = {}
    rows: list[dict[str, Any]] = []
    for index, call in enumerate(calls, 1):
        args = _call_arguments(call)
        metric = str(call.get("planned_metric_id") or args.get("metric_id") or args.get("metric_name") or "")
        operation = str(call.get("planned_operation") or call.get("tool") or call.get("tool_name") or "")
        dimension = args.get("dimension")
        request_id = str(call.get("request_id") or "")
        status = _call_response_status(call)
        signature = (call.get("tool_name") or call.get("tool"), metric, operation, dimension,
                     *(str(args.get(k) or "") for k in ("metric_date", "start_date", "end_date", "baseline_date", "current_date", "baseline_start", "baseline_end", "current_start", "current_end")))
        duplicate = signature in seen
        duplicate_of = seen.get(signature) if duplicate else None
        if not duplicate:
            seen[signature] = request_id or None
        metric_match = None if not metric else metric == expected_metric
        dimension_required = expected_dimension is not None and not (
            case.get("operation") == "multi_step_diagnostic"
            and _normalize_operation(operation) == "metric_trend"
        )
        dimension_match = None if not dimension_required else dimension == expected_dimension
        operation_match = _operation_matches(case, call)
        period_match = _period_matches(case, call)
        decision_record = next(
            (
                item for item in state.get("decision_records") or []
                if isinstance(item, Mapping) and str(item.get("request_id") or "") == request_id
            ),
            None,
        )
        capability_stop = _is_capability_stop_without_query(decision_record)
        alignment_checks = [metric_match, operation_match]
        if dimension_required:
            alignment_checks.append(dimension_match)
        alignment_checks.append(period_match)
        alignment = False if any(value is False for value in alignment_checks) else None if any(value is None for value in alignment_checks) else True
        if status == "ok":
            information = "provided" if call.get("evidence_id") or call.get("response") or any(
                isinstance(row, Mapping) and str(row.get("request_id") or "") == request_id
                for row in state.get("query_results") or []
            ) else "unknown"
        elif status == "not_found" and safe_stop and expected_stop == "DATA_INSUFFICIENT":
            information = "expected_limitation"
        elif status == "unsupported" and safe_stop and expected_stop == "CAPABILITY_GAP":
            information = "expected_limitation"
        elif status == "not_found":
            information = "not_found_result"
        elif status == "unsupported":
            information = "controlled_unsupported_result"
        else:
            information = "execution_error" if status in {"failed", "error", "execution_failed"} else "unknown"
        protocol_validity = (
            "not_applicable" if capability_stop
            else "invalid" if alignment is False else "valid" if alignment is True else "unknown"
        )
        execution_outcome = (
            "execution_error" if status in {"failed", "error", "execution_failed"}
            else "provided" if information == "provided"
            else "expected_limitation" if information == "expected_limitation"
            else "not_found" if information == "not_found_result"
            else "unsupported" if information == "controlled_unsupported_result"
            else "unknown"
        )
        reason = (
            "authorization checked; registered capability unavailable; no data query executed"
            if capability_stop
            else "tool parameters differ from frozen case contract" if alignment is False
            else "protocol parameters match frozen case contract" if alignment is True
            else "required fields are missing from persisted call"
        )
        rows.append({
            "case_id": case.get("case_id"), "arm": case.get("arm"), "run_id": state.get("trace_id"),
            "object_id": request_id or f"tool_call:{index}", "request_id": request_id,
            "result": protocol_validity,
            "reason": reason, "evidence_location": f"/tool_calls/{index-1}",
            "tool": call.get("tool_name") or call.get("tool"), "status": status,
            "metric_id": metric or None, "operation": operation or None, "dimension": dimension,
            "period": {k: args.get(k) for k in ("metric_date", "start_date", "end_date", "baseline_date", "current_date", "baseline_start", "baseline_end", "current_start", "current_end") if args.get(k) is not None},
            "metric_match": metric_match, "operation_match": operation_match,
            "dimension_match": dimension_match, "period_match": period_match,
            "duplicate_signature": duplicate, "duplicate_of_request_id": duplicate_of,
            "duplicate_disposition": "detected_but_intent_unknown" if duplicate else "not_detected",
            "information_result": information, "execution_outcome": execution_outcome,
        })
    counts = Counter(item["result"] for item in rows)
    duplicate_count = sum(row["duplicate_signature"] for row in rows)
    applicable = [row for row in rows if row["result"] != "not_applicable"]
    return {
        "denominator": len(applicable), "invalid_count": counts["invalid"],
        "valid_count": counts["valid"], "unknown_count": counts["unknown"],
        "not_applicable_count": counts["not_applicable"],
        "invalid_rate": counts["invalid"] / len(applicable) if applicable else None,
        "duplicate_signature_count": duplicate_count,
        "execution_error_count": sum(row["execution_outcome"] == "execution_error" for row in rows),
        "calls": rows,
    }


def _presentation_value_for_path(
    case: Mapping[str, Any], path: str, presentation: Any,
) -> tuple[bool, Any, list[str], str | None]:
    """Resolve a scored numeric field to its exact structured-card field.

    For additive contribution cards, each numeric field must carry its own
    source evidence IDs. A parent-card citation is not reused to imply field
    binding; that would conceal missing per-field capture.
    """
    if not isinstance(presentation, Mapping):
        return False, None, [], None
    if isinstance(presentation.get("presentation"), Mapping):
        presentation = presentation["presentation"]
    cards = presentation.get("contributions") or []
    if path in {"trend_points.baseline", "trend_points.current"}:
        # Contribution totals are the same registered period totals exposed by
        # the trend endpoint, and the card carries their explicit field-level
        # evidence binding.
        key = "baseline_value" if path.endswith("baseline") else "current_value"
        for card_index, card in enumerate(cards):
            if isinstance(card, Mapping) and key in card:
                field_map = card.get("field_evidence_ids") if isinstance(card.get("field_evidence_ids"), Mapping) else {}
                ids = [str(value) for value in (field_map.get(key) or []) if value]
                return True, card.get(key), ids, f"/analysis_presentation/contributions/{card_index}/{key}"
        return False, None, [], None
    if path.startswith("baseline_groups.") or path.startswith("current_groups."):
        side, dimension = path.split(".", 1)
        value_key = "baseline_value" if side == "baseline_groups" else "current_value"
        for card_index, card in enumerate(cards):
            if not isinstance(card, Mapping):
                continue
            for item_index, item in enumerate(card.get("items") or []):
                if isinstance(item, Mapping) and str(item.get("dimension_value")) == dimension and value_key in item:
                    field_map = item.get("field_evidence_ids") if isinstance(item.get("field_evidence_ids"), Mapping) else {}
                    ids = [str(value) for value in (field_map.get(value_key) or []) if value]
                    pointer = f"/analysis_presentation/contributions/{card_index}/items/{item_index}/{value_key}"
                    return True, item.get(value_key), ids, pointer
        return False, None, [], None
    card_fields = {
        "baseline_total": "baseline_value",
        "current_total": "current_value",
        "total_delta": "total_delta",
    }
    if path in card_fields:
        key = card_fields[path]
        for card_index, card in enumerate(cards):
            if isinstance(card, Mapping) and key in card:
                field_map = card.get("field_evidence_ids") if isinstance(card.get("field_evidence_ids"), Mapping) else {}
                ids = [str(value) for value in (field_map.get(key) or []) if value]
                return True, card.get(key), ids, f"/analysis_presentation/contributions/{card_index}/{key}"
    # Other operation types require their own explicit field mapping. Do not
    # infer a match from numeric equality alone.
    return False, None, [], None


def _core_claim_audit(case: Mapping[str, Any], observation: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    comparisons = observation.get("numeric_field_comparisons") or []
    if not comparisons:
        return {
            "status": "not_applicable_or_unavailable",
            "numeric_match": {"numerator": None, "denominator": None, "rate": None},
            "field_evidence_binding": {"numerator": None, "denominator": None, "rate": None},
            "claims": [],
        }
    bundles = {
        str(item.get("evidence_id")): item for item in state.get("evidence_bundles") or []
        if isinstance(item, Mapping) and item.get("evidence_id")
    }
    claim_rows = [item for item in state.get("claims") or [] if isinstance(item, Mapping)]
    refs = _all_refs(state)
    citation_ids = {item["evidence_id"] for item in refs}
    presentation = state.get("analysis_presentation") or state.get("final_display") or state.get("presentation") or {}
    if isinstance(presentation, Mapping) and isinstance(presentation.get("presentation"), Mapping):
        presentation = presentation["presentation"]
    card_ids = {item["evidence_id"] for item in _reference_occurrences(presentation, "/analysis_presentation")} if isinstance(presentation, Mapping) else set()
    result: list[dict[str, Any]] = []
    for comparison in comparisons:
        path = str(comparison.get("path") or "")
        source_ids: set[str] = set()
        source_value_mismatches: list[str] = []
        metric = str(case.get("target_metric") or "")
        expected_scope = case.get("scope")
        for bundle_id, bundle in bundles.items():
            if _metric_from_bundle(bundle) != metric:
                continue
            content = bundle.get("content") if isinstance(bundle.get("content"), Mapping) else {}
            request_id = content.get("request_id")
            call = _bundle_call_map(state).get(str(request_id)) if request_id else None
            if call is None:
                continue
            trusted = bundle.get("trusted")
            if trusted is None:
                trusted = True  # EvidenceBundle contract default.
            result_status = _bundle_result(bundle).get("status")
            if trusted is not True or result_status not in {"ok", "not_found"}:
                continue
            args = _call_arguments(call)
            actual_scope = args.get("scope") or state.get("scope_type")
            scope_match = None if actual_scope is None or expected_scope is None else actual_scope == expected_scope
            period_match = _period_matches(case, call)
            has_value, source_value = _bundle_value_for_path(bundle, path, case, call)
            if scope_match is not True or period_match is not True or not has_value:
                continue
            source_ids.add(bundle_id)
            if _values_agree(source_value, comparison.get("actual"), comparison.get("absolute_tolerance")) is False:
                source_value_mismatches.append(bundle_id)
        referenced_claims = []
        for i, claim in enumerate(claim_rows):
            ids = [str(value) for value in (claim.get("source_evidence_ids") or []) if value]
            if source_ids.intersection(ids):
                referenced_claims.append({"claim_index": i, "claim_type": claim.get("claim_type"), "source_evidence_ids": ids, "value_sha256": hashlib.sha256(str(claim.get("value") or "").encode()).hexdigest()})
        bound_ids = sorted(source_ids & citation_ids)
        # The existing Claim contract has no field_path. Do not claim that a
        # generic answer claim supports a particular numeric atom unless the
        # presentation contains an explicit atom-to-evidence binding.
        field_bindings = [
            item for item in state.get("claim_evidence_bindings") or []
            if isinstance(item, Mapping) and str(item.get("field_path") or "") == path
        ]
        if field_bindings:
            ids = {str(eid) for item in field_bindings for eid in (item.get("evidence_ids") or [])}
            claim_refs = {
                str(eid) for claim in claim_rows
                if isinstance(claim, Mapping)
                for eid in (claim.get("source_evidence_ids") or []) if eid
            }
            evidence_status = (
                "valid" if ids and ids.issubset(source_ids) and ids.issubset(citation_ids) and ids.intersection(claim_refs)
                else "invalid" if ids else "unknown"
            )
            evidence_ids = sorted(ids)
        else:
            evidence_status = "unknown"
            evidence_ids = bound_ids
        matched = comparison.get("matched") is True
        eligible_card_ids = card_ids & source_ids
        card_association = "valid" if eligible_card_ids else "unknown" if not card_ids else "invalid"
        # A citation somewhere in a long answer does not bind every numeric
        # atom. Only an explicit field_path-to-evidence record can make this
        # check valid; absent records remain unknown, never an inferred pass or
        # a false invalid caused by an unrelated citation.
        answer_binding = evidence_status
        presentation_found, presentation_value, presentation_ids, presentation_path = _presentation_value_for_path(
            case, path, presentation,
        )
        if not presentation_found:
            presentation_binding = "unknown"
        elif _values_agree(presentation_value, comparison.get("actual"), comparison.get("absolute_tolerance")) is False:
            presentation_binding = "invalid"
        elif not presentation_ids:
            presentation_binding = "unknown"
        elif not set(presentation_ids).issubset(source_ids) or not set(presentation_ids).issubset(card_ids):
            presentation_binding = "invalid"
        else:
            presentation_binding = "valid"
        numeric_result = "pass" if matched else "fail" if comparison.get("matched") is False or comparison.get("actual_field_present") is False else "unknown"
        result.append({
            "case_id": case.get("case_id"), "arm": case.get("arm"), "run_id": state.get("trace_id"),
            "object_id": path, "claim_path": path,
            "expected": comparison.get("expected"), "actual": comparison.get("actual"),
            "numeric_match": comparison.get("matched"), "absolute_error": comparison.get("absolute_error"),
            "tolerance": comparison.get("absolute_tolerance"),
            "field_present": comparison.get("actual_field_present"),
            "result": numeric_result,
            "reason": "numeric field matches frozen truth" if numeric_result == "pass" else "numeric field differs from frozen truth or is missing" if numeric_result == "fail" else "numeric comparison unavailable",
            "evidence_location": "/observation/numeric_field_comparisons/" + path,
            "numeric_match": numeric_result,
            "field_evidence_binding": "invalid" if source_value_mismatches else "valid" if source_ids else evidence_status,
            "answer_evidence_binding": answer_binding,
            "presentation_field_evidence_binding": presentation_binding,
            "presentation_field_path": presentation_path,
            "presentation_level_evidence_association": card_association,
            "source_evidence_ids": sorted(source_ids),
            "source_value_mismatch_evidence_ids": source_value_mismatches,
            "answer_citation_evidence_ids": evidence_ids,
            "presentation_citation_evidence_ids": sorted(card_ids),
            "related_claims": referenced_claims,
        })
    numeric_counts = Counter(item["numeric_match"] for item in result)
    binding_counts = Counter(item["field_evidence_binding"] for item in result)
    answer_binding_counts = Counter(item["answer_evidence_binding"] for item in result)
    return {
        "status": "recorded",
        "numeric_match": {
            "numerator": numeric_counts["pass"], "denominator": len(result),
            "rate": numeric_counts["pass"] / len(result) if result else None,
            "fail_count": numeric_counts["fail"], "unknown_count": numeric_counts["unknown"],
        },
        "field_evidence_binding": {
            "numerator": binding_counts["valid"], "denominator": len(result),
            "rate": binding_counts["valid"] / len(result) if result else None,
            "invalid_count": binding_counts["invalid"], "unknown_count": binding_counts["unknown"],
            "note": "Checks each scored numeric observation path against the request-correlated tool evidence; presentation-field and prose-claim bindings are reported separately.",
        },
        "answer_evidence_binding": {
            "numerator": answer_binding_counts["valid"], "denominator": len(result),
            "rate": answer_binding_counts["valid"] / len(result) if result and (answer_binding_counts["valid"] + answer_binding_counts["invalid"]) else None,
            "invalid_count": answer_binding_counts["invalid"], "unknown_count": answer_binding_counts["unknown"],
        },
        "presentation_field_evidence_binding": {
            "numerator": sum(item["presentation_field_evidence_binding"] == "valid" for item in result),
            "denominator": len(result),
            "rate": (
                sum(item["presentation_field_evidence_binding"] == "valid" for item in result) / len(result)
                if result and any(item["presentation_field_evidence_binding"] != "unknown" for item in result)
                else None
            ),
            "invalid_count": sum(item["presentation_field_evidence_binding"] == "invalid" for item in result),
            "unknown_count": sum(item["presentation_field_evidence_binding"] == "unknown" for item in result),
            "note": "Checks the exact structured presentation field against the request-correlated source value and the card's evidence IDs. When every field is unknown, rate is null rather than implying zero observed coverage; prose-level atom binding remains separate.",
        },
        "claims": result,
    }


def _safe_stop_audit(case: Mapping[str, Any], observation: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    expected = case.get("expected") if isinstance(case.get("expected"), Mapping) else {}
    if expected.get("outcome") != "safe_stop":
        return {"applicability": "not_applicable", "automated_status": "not_applicable", "semantic_content_status": "not_applicable"}
    expected_code = expected.get("terminal_code")
    actual_code = state.get("terminal_code") or observation.get("terminal_code")
    precondition = observation.get("precondition_verified")
    code_match = None if actual_code is None or expected_code is None else actual_code == expected_code
    presentation = state.get("presentation")
    if not isinstance(presentation, Mapping):
        final_display = state.get("final_display")
        presentation = final_display.get("presentation") if isinstance(final_display, Mapping) and isinstance(final_display.get("presentation"), Mapping) else {}
    presentation_status = presentation.get("status") if isinstance(presentation, Mapping) else None
    expected_statuses = {
        "DATA_INSUFFICIENT": {"partial"},
        "CAPABILITY_GAP": {"unsupported", "partial"},
    }
    allowed_statuses = expected_statuses.get(str(expected_code))
    presentation_match = None if presentation_status is None or allowed_statuses is None else presentation_status in allowed_statuses
    if code_match is False or precondition is False or presentation_match is False:
        automated = "fail"
    elif code_match is True and precondition is True and presentation_match is True:
        automated = "pass"
    else:
        automated = "unknown"
    return {
        "applicability": "expected_safe_stop",
        "expected_terminal_code": expected_code,
        "observed_terminal_code": actual_code,
        "terminal_code_match": code_match,
        "precondition_verified": precondition,
        "presentation_status": presentation_status,
        "presentation_status_match": presentation_match,
        "automated_status": automated,
        "semantic_content_status": "manual_review_required",
        "note": "Structured stop/precondition can be checked mechanically; whether the free-form explanation is semantically correct remains a human judgment.",
    }


def _bundle_result(bundle: Mapping[str, Any]) -> Mapping[str, Any]:
    content = bundle.get("content") if isinstance(bundle.get("content"), Mapping) else {}
    result = content.get("result") if isinstance(content.get("result"), Mapping) else {}
    return result


def _bundle_value_for_path(
    bundle: Mapping[str, Any], path: str, case: Mapping[str, Any], call: Mapping[str, Any] | None = None,
) -> tuple[bool, Any]:
    """Read the exact registered result field corresponding to a scored path."""
    result = _bundle_result(bundle)
    if path in {"trend_points.baseline", "trend_points.current"}:
        period = case.get("period") if isinstance(case.get("period"), Mapping) else {}
        role = "baseline" if path.endswith("baseline") else "current"
        bucket = period.get(role)
        trend = result.get("trend_result") if isinstance(result.get("trend_result"), Mapping) else {}
        for row in trend.get("points") or []:
            if isinstance(row, Mapping) and str(row.get("bucket")) == str(bucket) and "value" in row:
                return True, row.get("value")
        # A registered contribution result carries the same validated
        # baseline/current aggregate totals. This supports tasks whose
        # structured output presents those totals on the contribution card
        # without a separate trend card.
        contribution = result.get("contribution_result") if isinstance(result.get("contribution_result"), Mapping) else {}
        date_key = "baseline_date" if role == "baseline" else "current_date"
        total_key = "baseline_total" if role == "baseline" else "current_total"
        if str(contribution.get(date_key) or "") == str(bucket) and total_key in contribution:
            return True, contribution.get(total_key)
        return False, None
    if path.startswith("points."):
        bucket = path.split(".", 1)[1]
        trend = result.get("trend_result") if isinstance(result.get("trend_result"), Mapping) else {}
        for row in trend.get("points") or []:
            if isinstance(row, Mapping) and str(row.get("bucket")) == bucket and "value" in row:
                return True, row.get("value")
        return False, None
    for prefix, result_key, list_key, value_keys in (
        ("baseline_groups.", "contribution_result", "baseline_items", ("baseline_value", "value")),
        ("current_groups.", "contribution_result", "current_items", ("current_value", "value")),
    ):
        if path.startswith(prefix):
            dimension_value = path[len(prefix):]
            contribution = result.get(result_key) if isinstance(result.get(result_key), Mapping) else {}
            for row in contribution.get(list_key) or []:
                if isinstance(row, Mapping) and str(row.get("dimension_value")) == dimension_value:
                    for value_key in value_keys:
                        if value_key in row:
                            return True, row.get(value_key)
            # "Other" is a residual derived from complete totals minus the
            # returned top-N rows. Bind it only when the actual call's limit
            # matches both returned lists, making the residual reproducible
            # from this exact tool result rather than inferred from display.
            if dimension_value.casefold() == "other" and isinstance(call, Mapping):
                tool_name = str(call.get("tool_name") or call.get("name") or "")
                args = _call_arguments(call)
                # The MCP contribution tools define limit=10. Calls that omit
                # the optional argument therefore have this explicit public
                # default; no case-specific value is inferred here.
                raw_limit = args.get("limit", 10 if tool_name in {
                    "compare_breakdown_by_dimension", "compare_breakdown_by_dimension_period",
                } else None)
                try:
                    limit = int(raw_limit)
                except (TypeError, ValueError):
                    limit = 0
                rows = contribution.get(list_key) or []
                other_side = "baseline_items" if list_key == "current_items" else "current_items"
                other_rows = contribution.get(other_side) or []
                total_key = "current_total" if list_key == "current_items" else "baseline_total"
                total = contribution.get(total_key)
                if (
                    tool_name in {"compare_breakdown_by_dimension", "compare_breakdown_by_dimension_period"}
                    and limit > 0 and len(rows) == limit and len(other_rows) == limit
                    and isinstance(total, (int, float))
                    and all(isinstance(item, Mapping) and isinstance(item.get("value"), (int, float)) for item in rows)
                ):
                    return True, float(total) - sum(float(item["value"]) for item in rows)
            return False, None
    aliases = {
        "baseline_total": ("baseline_total", "baseline_value"),
        "current_total": ("current_total", "current_value"),
        "total_delta": ("total_delta", "delta_value"),
    }
    for result_key in ("metric_result", "compare_result", "contribution_result"):
        payload = result.get(result_key)
        if isinstance(payload, Mapping):
            for key in aliases.get(path, (path,)):
                if key in payload:
                    return True, payload.get(key)
    return False, None


def _values_agree(left: Any, right: Any, tolerance: Any) -> bool | None:
    if left is None or right is None:
        return left is None and right is None
    try:
        return abs(float(left) - float(right)) <= float(tolerance or 0)
    except (TypeError, ValueError):
        return left == right


def build_companion_quality_audit(
    case: Mapping[str, Any], arm: str, trace: Mapping[str, Any], observation: Mapping[str, Any],
    *, expected_run_id: str | None = None,
) -> dict[str, Any]:
    """Build appendable, per-object evidence/authorization/call quality facts."""
    state = _state(trace)
    case_context = {**dict(case), "arm": arm}
    citation = _citation_audit(case_context, state, observation)
    auth = _authorization_audit(case_context, state)
    calls = _call_audit(case_context, state)
    core = _core_claim_audit(case_context, observation, state)
    observed_run_id = state.get("trace_id") or _inner_trace(trace).get("trace_id")
    run_id_match = None if not expected_run_id or not observed_run_id else str(expected_run_id) == str(observed_run_id)
    if run_id_match is False:
        for occurrence in citation["occurrences"]:
            occurrence["result"] = "invalid"
            occurrence["reason"] = "trace run ID does not match the run record ID"
        citation_counts = Counter(item["result"] for item in citation["occurrences"])
        citation.update({
            "numerator": citation_counts["valid"],
            "rate": citation_counts["valid"] / citation["denominator"] if citation["denominator"] else None,
            "unknown_count": citation_counts["unknown"],
            "invalid_count": citation_counts["invalid"],
        })
    numeric_contract = case.get("numeric_match_contract") if isinstance(case.get("numeric_match_contract"), Mapping) else {}
    expected = case.get("expected") if isinstance(case.get("expected"), Mapping) else {}
    truth = expected.get("truth") if isinstance(expected.get("truth"), Mapping) else {}
    truth_sha256 = hashlib.sha256(json.dumps(
        truth, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")).hexdigest() if truth else None
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "case_id": case.get("case_id"), "arm": arm,
        "run_id": observed_run_id,
        "run_identity": {"expected_run_id": expected_run_id, "observed_run_id": observed_run_id, "match": run_id_match},
        "frozen_case_contract": {
            "target_metric": case.get("target_metric"), "operation": case.get("operation"),
            "scope": case.get("scope"), "dimension": case.get("dimension"),
            "period": case.get("period"), "expected_outcome": expected.get("outcome"),
            "expected_terminal_code": expected.get("terminal_code"),
            "truth_sha256": truth_sha256,
            "numeric_match_contract_version": numeric_contract.get("version"),
            "numeric_match_contract_status": numeric_contract.get("status"),
        },
        "core_conclusion_evidence_coverage": core,
        "valid_evidence_citations": citation,
        "correct_refusal_content": _safe_stop_audit(case_context, observation, state),
        "authorization_decisions": {
            "denominator": sum(row["result"] != "not_applicable" for row in auth),
            "allowed_count": sum(row["result"] == "allowed" for row in auth),
            "denied_count": sum(row["result"] == "denied" for row in auth),
            "unknown_count": sum(row["result"] == "unknown" for row in auth),
            "not_applicable_count": sum(row["result"] == "not_applicable" for row in auth),
            "calls": auth,
        },
        "tool_call_validity": calls,
    }


def summarize_companion_quality_records(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate only within explicit run-record cohorts; preserve every denominator."""
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        audit = record.get("companion_quality_audit")
        run_id = str(record.get("run_id") or (audit or {}).get("run_id") or "unknown-run")
        cohort = str(record.get("cohort_id") or record.get("manifest_sha256") or f"unversioned:{run_id}")
        arm = str(record.get("arm") or (audit or {}).get("arm") or "unknown-arm")
        groups.setdefault((cohort, arm), []).append(record)

    summaries = []
    for (cohort, arm), rows in sorted(groups.items()):
        audits = [row.get("companion_quality_audit") for row in rows if isinstance(row.get("companion_quality_audit"), Mapping)]
        numeric = [claim for audit in audits for claim in ((audit.get("core_conclusion_evidence_coverage") or {}).get("claims") or [])]
        citations = [item for audit in audits for item in ((audit.get("valid_evidence_citations") or {}).get("occurrences") or [])]
        auth = [item for audit in audits for item in ((audit.get("authorization_decisions") or {}).get("calls") or [])]
        calls = [item for audit in audits for item in ((audit.get("tool_call_validity") or {}).get("calls") or [])]
        stops = [audit.get("correct_refusal_content") or {} for audit in audits]
        def counts(values: list[str], keys: tuple[str, ...]) -> dict[str, int]:
            counter = Counter(values)
            return {key: counter[key] for key in keys}
        numeric_counts = counts([str(item.get("numeric_match") or "unknown") for item in numeric], ("pass", "fail", "unknown"))
        citation_counts = counts([str(item.get("result") or "unknown") for item in citations], ("valid", "invalid", "unknown"))
        auth_counts = counts([str(item.get("result") or "unknown") for item in auth], ("allowed", "denied", "unknown", "not_applicable"))
        auth_snapshot_counts = counts([str(item.get("snapshot_integrity") or "unknown") for item in auth], ("valid", "invalid", "unknown", "not_applicable"))
        call_counts = counts([str(item.get("result") or "unknown") for item in calls], ("valid", "invalid", "unknown", "not_applicable"))
        applicable_stops = [item for item in stops if item.get("applicability") == "expected_safe_stop"]
        stop_counts = counts([str(item.get("automated_status") or "unknown") for item in applicable_stops], ("pass", "fail", "unknown"))
        summaries.append({
            "cohort_id": cohort, "arm": arm, "run_count": len(rows),
            "core_numeric_fields": {
                "numerator": numeric_counts["pass"], "denominator": len(numeric),
                "rate": numeric_counts["pass"] / len(numeric) if numeric else None,
                "failed": numeric_counts["fail"], "unknown": numeric_counts["unknown"],
            },
            "evidence_citations": {
                "numerator": citation_counts["valid"], "denominator": len(citations),
                "rate": citation_counts["valid"] / len(citations) if citations else None,
                "invalid": citation_counts["invalid"], "unknown": citation_counts["unknown"],
            },
            "authorization_decisions": {
                "denominator": len([item for item in auth if item.get("result") != "not_applicable"]), **auth_counts,
                "snapshot_integrity": {
                    "denominator": len([item for item in auth if item.get("snapshot_integrity") != "not_applicable"]),
                    **auth_snapshot_counts,
                },
            },
            "tool_call_protocol_validity": {
                "denominator": len([item for item in calls if item.get("result") != "not_applicable"]), **call_counts,
                "duplicate_signatures": sum(bool(item.get("duplicate_signature")) for item in calls),
                "execution_errors": sum(item.get("execution_outcome") == "execution_error" for item in calls),
            },
            "safe_stop_structural_review": {"denominator": len(applicable_stops), **stop_counts,
                "not_applicable": len(stops) - len(applicable_stops)},
            "safe_stop_semantic_manual_review_required": sum(item.get("semantic_content_status") == "manual_review_required" for item in stops),
        })
    return {"schema_version": AUDIT_SCHEMA_VERSION, "groups": summaries}

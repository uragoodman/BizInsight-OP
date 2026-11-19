"""Field-aligned numeric assertions for the v7 original-definition B/G set.

The score adapter intentionally compares named result fields and keyed buckets
or dimensions. It never matches a truth value against an arbitrary number
found elsewhere in a response.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping


def _registered_result_rows(state: Mapping[str, Any], arm: str) -> list[dict[str, Any]]:
    # G duplicates the tool response into query_results and tool_calls. Use
    # one authoritative representation; B only persists tool_calls.
    rows: list[dict[str, Any]] = []
    if arm == "G":
        for item in state.get("query_results") or []:
            if isinstance(item, Mapping) and isinstance(item.get("result"), Mapping):
                rows.append(dict(item["result"]))
        if rows:
            return rows
    for call in state.get("tool_calls") or []:
        if not isinstance(call, Mapping):
            continue
        response = call.get("response") if isinstance(call.get("response"), Mapping) else {}
        result = response.get("result") if isinstance(response.get("result"), Mapping) else {}
        if result:
            rows.append(dict(result))
    return rows


def _validated_diagnostic_numeric_results(
    case: Mapping[str, Any], state: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Project only engine-validated G diagnostic evidence into scorer shapes.

    Governed diagnostics persist their facts in the investigation ledger rather
    than ``query_results``. Reuse the runtime trust boundary and additionally
    bind the projection to this case's metric, dimension, operation and dates.
    """
    if case.get("operation") != "multi_step_diagnostic":
        return {}
    investigation = state.get("investigation_state")
    if not isinstance(investigation, Mapping):
        return {}
    try:
        from app.graph.nodes.diagnostic import (
            _completed_valid_evidence_ids,
            validated_dimension_contribution_evidence,
        )

        valid_ids = set(_completed_valid_evidence_ids(state))
        validated_contributions = validated_dimension_contribution_evidence(state)
    except Exception:
        # Malformed/unavailable engine validation cannot promote raw values.
        return {}

    period = case.get("period") if isinstance(case.get("period"), Mapping) else {}
    baseline, current = period.get("baseline"), period.get("current")
    target_metric = str(case.get("target_metric") or "")
    target_dimension = str(case.get("dimension") or "")
    evidence_rows = {
        str(item.get("evidence_id")): item
        for item in (investigation.get("evidence") or [])
        if isinstance(item, Mapping) and item.get("evidence_id")
    }
    projected: dict[str, dict[str, Any]] = {}

    for evidence_id in valid_ids:
        evidence = evidence_rows.get(evidence_id)
        if not evidence or str(evidence.get("metric_id") or "") != target_metric:
            continue
        values = evidence.get("values") if isinstance(evidence.get("values"), Mapping) else {}
        query = values.get("query") if isinstance(values.get("query"), Mapping) else {}
        result = values.get("result") if isinstance(values.get("result"), Mapping) else {}
        if evidence.get("evidence_type") == "trend":
            trend_period = query.get("period") if isinstance(query.get("period"), Mapping) else {}
            if (
                str(query.get("metric_name") or "") == target_metric
                and str(result.get("metric_name") or "") == target_metric
                and str(trend_period.get("start_date") or "") == str(baseline)
                and str(trend_period.get("end_date") or "") == str(current)
                and result.get("status") == "ok"
                and isinstance(result.get("points"), list)
            ):
                projected["trend_result"] = {
                    "metric_name": target_metric,
                    "points": result["points"],
                }

    # This shared helper checks trusted bundle linkage, completed action,
    # validation, scope, method, units, period and additive reconciliation.
    for item in validated_contributions:
        if (
            str(item.get("evidence_id") or "") not in valid_ids
            or str(item.get("metric_id") or "") != target_metric
            or str(item.get("dimension") or "") != target_dimension
            or str(item.get("baseline_date") or "") != str(baseline)
            or str(item.get("current_date") or "") != str(current)
        ):
            continue
        components = item.get("items") or []
        if not isinstance(components, list) or not components:
            continue
        # Runtime evidence carries a summarized top-N plus an explicit Other
        # residual. Preserve that representation so scoring can compare the
        # visible keys and independently reconcile the omitted net aggregate.
        projected["contribution_result"] = {
            "metric_name": target_metric,
            "dimension": target_dimension,
            "baseline_total": item.get("baseline_value"),
            "current_total": item.get("current_value"),
            "total_delta": item.get("total_delta"),
            "baseline_items": [
                {"dimension_value": row.get("dimension_value"), "value": row.get("baseline_value")}
                for row in components
            ],
            "current_items": [
                {"dimension_value": row.get("dimension_value"), "value": row.get("current_value")}
                for row in components
            ],
        }
    return projected


def registered_numeric_decomposition_projection(
    case: Mapping[str, Any], state: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a governed registered decomposition only when its trace chain binds.

    This observes the target-level decomposition, not the first internal factor
    query. The projection is accepted only when the ready request, typed card,
    deterministic evidence, supporting trend evidence, and authorized tool
    calls all agree on target, factors, scope, and point-pair dates.
    """
    if case.get("operation") != "registered_numeric_decomposition":
        return {"valid": False, "reason": "case_operation_mismatch"}
    decision = state.get("metric_planning_decision")
    decision = decision if isinstance(decision, Mapping) else {}
    request = state.get("registered_decomposition_request")
    if not isinstance(request, Mapping) or not request:
        nested = decision.get("registered_decomposition")
        request = nested if isinstance(nested, Mapping) else {}
    if not request or request.get("status") != "ready":
        return {"valid": False, "reason": "registered_request_missing_or_not_ready"}

    target = str(request.get("target_metric_id") or "")
    method = str(request.get("method_id") or "")
    factors = request.get("factor_metric_ids")
    factors = [str(item) for item in factors] if isinstance(factors, list) else []
    expected_factors: list[str] = []
    try:
        from app.metrics.registry import get_registry

        target_definition = get_registry().get(target)
        scope_capability = target_definition.attribution.for_scope(str(case.get("scope") or "platform"))
        if method not in scope_capability.method_ids:
            return {"valid": False, "reason": "method_not_in_registered_scope_contract"}
        if method == "symmetric_ratio_v1" and target_definition.period_aggregation is not None:
            expected_factors = [
                target_definition.period_aggregation.numerator_metric,
                target_definition.period_aggregation.denominator_metric,
            ]
        else:
            expected_factors = [item.factor_id for item in scope_capability.factor_specs.get(method, ())]
    except Exception:
        return {"valid": False, "reason": "registered_factor_contract_unavailable"}
    expected_period = case.get("period") if isinstance(case.get("period"), Mapping) else {}
    baseline, current = expected_period.get("baseline"), expected_period.get("current")
    if not (
        target == str(case.get("target_metric") or "")
        and state.get("scope_type") == case.get("scope")
        and method == str(case.get("method_id") or "")
        and factors and factors == expected_factors
        and request.get("baseline_date") == baseline
        and request.get("current_date") == current
        and decision.get("operation_intent") == "registered_numeric_decomposition"
    ):
        return {"valid": False, "reason": "request_does_not_match_case_contract"}

    presentation = state.get("analysis_presentation")
    presentation = presentation if isinstance(presentation, Mapping) else {}
    card = presentation.get("metric_decomposition")
    if not isinstance(card, Mapping) or not card:
        return {"valid": False, "reason": "decomposition_card_missing"}
    if (
        card.get("metric_id") != target
        or card.get("method_id") != method
        or (case.get("metric_unit") and card.get("unit") != case.get("metric_unit"))
        or card.get("analysis_nature") != "registered_numeric_decomposition"
        or card.get("business_cause_confirmed") is not False
        or not isinstance(card.get("evidence_ids"), list)
        or len(card["evidence_ids"]) != 1
    ):
        return {"valid": False, "reason": "decomposition_card_contract_mismatch"}

    def finite(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    def close(left: Any, right: Any, tolerance: float = 1e-8) -> bool:
        return finite(left) and finite(right) and math.isclose(float(left), float(right), rel_tol=0, abs_tol=tolerance)

    baseline_period = card.get("baseline_period")
    current_period = card.get("current_period")
    if not (
        isinstance(baseline_period, Mapping) and isinstance(current_period, Mapping)
        and baseline_period.get("start_date") == baseline and baseline_period.get("end_date") == baseline
        and current_period.get("start_date") == current and current_period.get("end_date") == current
    ):
        return {"valid": False, "reason": "card_period_mismatch"}

    evidence = [item for item in (state.get("evidence_bundles") or []) if isinstance(item, Mapping)]
    evidence_by_id = {str(item.get("evidence_id")): item for item in evidence if item.get("evidence_id")}
    decomposition_evidence = evidence_by_id.get(str(card["evidence_ids"][0]))
    if not (
        isinstance(decomposition_evidence, Mapping)
        and decomposition_evidence.get("trusted") is True
        and decomposition_evidence.get("source_type") == "deterministic_attribution"
        and decomposition_evidence.get("source_ref") == "authorized_complete_daily_trend_evidence"
    ):
        return {"valid": False, "reason": "trusted_decomposition_evidence_missing"}
    evidence_content = decomposition_evidence.get("content")
    evidence_content = evidence_content if isinstance(evidence_content, Mapping) else {}
    result = evidence_content.get("result")
    result = result if isinstance(result, Mapping) else {}
    if (
        result.get("factor_metrics") != factors
        or result.get("residual_explainable") is not True
        or not all(close(card.get(field), result.get(field)) for field in ("baseline_value", "current_value"))
    ):
        return {"valid": False, "reason": "decomposition_evidence_values_or_factors_mismatch"}
    if not close(card.get("delta_value"), result.get("total_delta")):
        return {"valid": False, "reason": "card_delta_mismatch"}

    card_factors = card.get("factors")
    if not isinstance(card_factors, list) or not all(isinstance(item, Mapping) for item in card_factors):
        return {"valid": False, "reason": "card_factors_missing"}
    card_factor_map: dict[str, Mapping[str, Any]] = {}
    for item in card_factors:
        factor_id = str(item.get("metric_id") or "")
        if not factor_id or factor_id in card_factor_map:
            return {"valid": False, "reason": "card_factor_identity_invalid"}
        card_factor_map[factor_id] = item
    if set(card_factor_map) != set(factors):
        return {"valid": False, "reason": "card_factor_set_mismatch"}

    identity_rows = result.get("factor_identities")
    role_to_metric = {
        str(item.get("role_id")): str(item.get("metric_id"))
        for item in identity_rows if isinstance(item, Mapping) and item.get("role_id") and item.get("metric_id")
    } if isinstance(identity_rows, list) else {}
    evidence_components = result.get("display_components")
    evidence_component_map = {
        str(item.get("component_id")): item.get("value")
        for item in evidence_components if isinstance(item, Mapping) and item.get("component_id")
    } if isinstance(evidence_components, list) else {}
    for factor_id, factor in card_factor_map.items():
        if not finite(factor.get("contribution")):
            return {"valid": False, "reason": "card_factor_value_invalid"}
        evidence_component_id = factor_id
        if evidence_component_id not in evidence_component_map:
            evidence_component_id = next((role for role, metric_id in role_to_metric.items() if metric_id == factor_id), "")
        if not evidence_component_id or not close(factor.get("contribution"), evidence_component_map.get(evidence_component_id)):
            return {"valid": False, "reason": "card_factor_not_bound_to_evidence"}

    input_ids = result.get("input_evidence_ids")
    if not isinstance(input_ids, list) or len(input_ids) != len(set(input_ids)):
        return {"valid": False, "reason": "input_evidence_ids_invalid"}
    input_bundles = [evidence_by_id.get(str(evidence_id)) for evidence_id in input_ids]
    if any(not isinstance(item, Mapping) for item in input_bundles):
        return {"valid": False, "reason": "input_evidence_missing"}
    input_by_metric: dict[str, Mapping[str, Any]] = {}
    for item in input_bundles:
        if not (
            item.get("trusted") is True
            and item.get("source_type") == "mcp"
            and item.get("source_ref") == "registry_trend"
        ):
            return {"valid": False, "reason": "input_evidence_not_trusted_trend"}
        content = item.get("content") if isinstance(item.get("content"), Mapping) else {}
        metric_id = str(content.get("metric_id") or "")
        body = content.get("result") if isinstance(content.get("result"), Mapping) else {}
        trend = body.get("trend_result") if isinstance(body.get("trend_result"), Mapping) else {}
        parsed = body.get("parsed") if isinstance(body.get("parsed"), Mapping) else {}
        authorized_metric = parsed.get("authorized_metric")
        authorized_metric = authorized_metric if isinstance(authorized_metric, Mapping) else {}
        if (
            not metric_id or metric_id in input_by_metric
            or body.get("status") != "ok"
            or trend.get("status") != "ok"
            or trend.get("metric_name") != metric_id
            or authorized_metric.get("metric_id") != metric_id
        ):
            return {"valid": False, "reason": "input_trend_contract_invalid"}
        points = trend.get("points")
        point_by_date = {
            str(point.get("bucket")): point.get("value")
            for point in points if isinstance(point, Mapping) and point.get("bucket") is not None
        } if isinstance(points, list) else {}
        if not (finite(point_by_date.get(str(baseline))) and finite(point_by_date.get(str(current)))):
            return {"valid": False, "reason": "input_trend_dates_or_values_missing"}
        input_by_metric[metric_id] = item
    expected_inputs = set(str(item) for item in (request.get("input_metric_ids") or []))
    if (
        set(input_by_metric) != expected_inputs
        or expected_inputs != {target, *factors}
        or not expected_inputs
    ):
        return {"valid": False, "reason": "input_metric_set_mismatch"}

    calls = [item for item in (state.get("tool_calls") or []) if isinstance(item, Mapping)]
    for metric_id, item in input_by_metric.items():
        evidence_id = str(item.get("evidence_id") or "")
        matching = [call for call in calls if call.get("evidence_id") == evidence_id]
        if len(matching) != 1:
            return {"valid": False, "reason": "input_tool_call_binding_missing_or_ambiguous"}
        call = matching[0]
        args = call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
        period = args.get("period") if isinstance(args.get("period"), Mapping) else {}
        auth = call.get("authorization_snapshot") if isinstance(call.get("authorization_snapshot"), Mapping) else {}
        if not (
            call.get("tool_name") == "registry_trend"
            and call.get("status") == "ok"
            and args.get("metric_name") == metric_id
            and period.get("start_date") == baseline and period.get("end_date") == current
            and auth.get("decision") == "allowed" and auth.get("authorization_checked") is True
            and auth.get("metric_id") == metric_id and auth.get("scope") == case.get("scope")
        ):
            return {"valid": False, "reason": "input_tool_call_or_authorization_mismatch"}

    displayed_sum = sum(float(item["contribution"]) for item in card_factor_map.values())
    residual_tolerance = result.get("residual_tolerance")
    if not finite(residual_tolerance) or residual_tolerance < 0:
        return {"valid": False, "reason": "decomposition_residual_tolerance_invalid"}
    if abs(displayed_sum - float(card.get("delta_value"))) > max(1e-8, float(residual_tolerance)):
        return {"valid": False, "reason": "displayed_factors_do_not_reconcile"}

    values: dict[str, Any] = {
        "baseline_value": card.get("baseline_value"),
        "current_value": card.get("current_value"),
        "total_delta": card.get("delta_value"),
    }
    expected_components = ((case.get("expected") or {}).get("truth") or {}).get("display_components")
    expected_components = expected_components if isinstance(expected_components, Mapping) else {}
    for expected_id in expected_components:
        metric_id = str(expected_id)
        if metric_id not in card_factor_map:
            metric_id = role_to_metric.get(str(expected_id), "")
        factor = card_factor_map.get(metric_id)
        if not factor:
            return {"valid": False, "reason": "expected_factor_identity_not_bound"}
        values[f"display_components.{expected_id}"] = factor.get("contribution")
    return {
        "valid": True,
        "metric_id": target,
        "operation": "registered_numeric_decomposition",
        "period": {"kind": "point_pair", "baseline": baseline, "current": current},
        "unit": card.get("unit"),
        "values": values,
        "evidence_ids": list(card.get("evidence_ids") or []),
        "reason": None,
    }


def _canonical_values(case: Mapping[str, Any], state: Mapping[str, Any], arm: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    expected = case.get("expected") if isinstance(case.get("expected"), Mapping) else {}
    if expected.get("outcome") not in {"numeric_result", "safe_stop"}:
        return None, {"status": "not_applicable"}
    op = str(case.get("operation") or "")
    key_by_op = {
        "registered_numeric_decomposition": (),
        "metric_value": ("metric_result",),
        "metric_compare": ("compare_result",),
        "expected_data_insufficiency": ("compare_result",),
        "metric_trend": ("trend_result",),
        "dimension_contribution": ("contribution_result",),
        "multi_step_diagnostic": ("trend_result", "contribution_result"),
    }
    result_keys = key_by_op.get(op, ())
    if op == "registered_numeric_decomposition":
        projection = registered_numeric_decomposition_projection(case, state)
        if projection.get("valid") is True:
            return dict(projection["values"]), {
                "status": "present",
                "registered_numeric_decomposition": "validated",
                "evidence_ids": projection["evidence_ids"],
            }
        return None, {"status": "missing_or_invalid", "registered_numeric_decomposition": projection.get("reason")}
    found: dict[str, list[dict[str, Any]]] = {key: [] for key in result_keys}
    for row in _registered_result_rows(state, arm):
        for key in result_keys:
            payload = row.get(key)
            metric_name = payload.get("metric_name", payload.get("metric_id")) if isinstance(payload, Mapping) else None
            if isinstance(payload, Mapping) and metric_name == case.get("target_metric"):
                found[key].append(dict(payload))

    # G's multi-step diagnostic graph stores validated query facts in typed
    # investigation evidence. Use this only to fill result kinds absent from
    # the regular registered-query envelope; never duplicate competing facts.
    if arm == "G" and op == "multi_step_diagnostic" and any(not found[key] for key in result_keys):
        diagnostic_results = _validated_diagnostic_numeric_results(case, state)
        for key in result_keys:
            if not found[key] and key in diagnostic_results:
                found[key].append(diagnostic_results[key])

    values: dict[str, Any] = {}
    result_status: dict[str, Any] = {}
    for key in result_keys:
        payloads = found[key]
        result_status[key] = "present" if len(payloads) == 1 else (
            "missing" if not payloads else "duplicate_result"
        )
        # Require exactly one actual result for every planned operation.
        if len(payloads) != 1:
            continue
        payload = payloads[0]
        if key == "metric_result":
            if "value" in payload:
                values["value"] = payload["value"]
        elif key == "compare_result":
            for field in ("baseline_value", "current_value", "delta_value", "delta_ratio"):
                if field in payload:
                    values[field] = payload[field]
        elif key == "trend_result":
            points = payload.get("points")
            if isinstance(points, list):
                buckets: set[str] = set()
                for point in points:
                    if not isinstance(point, Mapping) or point.get("bucket") is None or "value" not in point:
                        continue
                    bucket = str(point["bucket"])
                    if bucket in buckets:
                        result_status[key] = "duplicate_bucket"
                        values.pop(f"points.{bucket}", None)
                        continue
                    buckets.add(bucket)
                    values[f"points.{bucket}"] = point["value"]
        elif key == "contribution_result":
            for field in ("baseline_total", "current_total", "total_delta"):
                if field in payload:
                    values[field] = payload[field]
            for source_field, prefix in (("baseline_items", "baseline_groups"), ("current_items", "current_groups")):
                items = payload.get(source_field)
                if not isinstance(items, list):
                    continue
                identities: set[str] = set()
                malformed = False
                for item in items:
                    if not isinstance(item, Mapping) or item.get("dimension_value") is None or "value" not in item:
                        malformed = True
                        continue
                    identity = str(item["dimension_value"])
                    if identity in identities:
                        malformed = True
                    identities.add(identity)
                    values[f"{prefix}.{identity}"] = item["value"]
                result_status[source_field] = "malformed_or_duplicate_key" if malformed else "present"

    # Multi-step cases require the daily trend to agree with the corresponding
    # registered contribution totals, by each exact bucket.
    if op == "multi_step_diagnostic":
        period = case.get("period") if isinstance(case.get("period"), Mapping) else {}
        values["trend_points.baseline"] = values.get(f"points.{period.get('baseline')}")
        values["trend_points.current"] = values.get(f"points.{period.get('current')}")
    return values if values else None, {"status": "present" if any(v == "present" for v in result_status.values()) else "missing", "results": result_status}


def _expected_values(case: Mapping[str, Any]) -> dict[str, Any]:
    expected = case.get("expected") if isinstance(case.get("expected"), Mapping) else {}
    truth = expected.get("truth") if isinstance(expected.get("truth"), Mapping) else {}
    result: dict[str, Any] = {}
    scalar_fields = ("value", "baseline_value", "current_value", "delta_value", "delta_ratio",
                     "baseline_total", "current_total", "total_delta")
    for field in scalar_fields:
        if field in truth:
            result[field] = truth[field]
    for field in ("baseline_groups", "current_groups"):
        groups = truth.get(field)
        if isinstance(groups, Mapping):
            for identity, value in groups.items():
                result[f"{field}.{identity}"] = value
    components = truth.get("display_components")
    if isinstance(components, Mapping):
        for identity, value in components.items():
            result[f"display_components.{identity}"] = value
    points = truth.get("points")
    if isinstance(points, list):
        for point in points:
            if isinstance(point, Mapping) and point.get("bucket") is not None and "value" in point:
                result[f"points.{point['bucket']}"] = point["value"]
    derived_fields = (case.get("numeric_match_contract") or {}).get("derived_fields") or {}
    for path, source_path in derived_fields.items():
        if source_path == "baseline_total" and "baseline_total" in truth:
            result[str(path)] = truth["baseline_total"]
        elif source_path == "current_total" and "current_total" in truth:
            result[str(path)] = truth["current_total"]
    return result


def compare_case_numeric_fields(
    case: Mapping[str, Any], actual: dict[str, Any] | None, *, arm: str | None = None,
) -> tuple[bool | None, list[dict[str, Any]]]:
    contract = case.get("numeric_match_contract")
    if not isinstance(contract, Mapping) or contract.get("status") != "verified":
        return None, []
    expected = _expected_values(case)
    if not expected:
        return None, []
    if actual is None:
        return None, []

    # Governed multi-step answers may show the registered top-N groups plus an
    # explicit net ``Other`` aggregate. Compare each displayed group exactly
    # and compute Other's expected value from the independently frozen truth.
    if arm == "G" and case.get("operation") == "multi_step_diagnostic":
        truth = ((case.get("expected") or {}).get("truth") or {})
        for prefix, truth_field in (
            ("baseline_groups.", "baseline_groups"),
            ("current_groups.", "current_groups"),
        ):
            actual_keys = {path[len(prefix):] for path in actual if path.startswith(prefix)}
            if "Other" not in actual_keys:
                continue
            truth_groups = truth.get(truth_field)
            if not isinstance(truth_groups, Mapping):
                continue
            shown = actual_keys - {"Other"}
            normalized: dict[str, Any] = {
                key: value for key, value in expected.items()
                if not key.startswith(prefix)
            }
            for key in shown:
                if key in truth_groups:
                    normalized[prefix + key] = truth_groups[key]
            omitted = set(truth_groups) - shown
            normalized[prefix + "Other"] = sum(
                float(truth_groups[key]) for key in omitted
                if isinstance(truth_groups[key], (int, float)) and not isinstance(truth_groups[key], bool)
            )
            expected = normalized
    tolerances = contract.get("absolute_tolerance_by_path") or {}
    null_exact = set(contract.get("null_exact_paths") or [])
    comparisons: list[dict[str, Any]] = []
    for path, expected_value in sorted(expected.items()):
        present = path in actual
        actual_value = actual.get(path)
        tolerance = tolerances.get(path)
        if tolerance is None:
            for rule, rule_value in tolerances.items():
                if str(rule).endswith("*") and path.startswith(str(rule)[:-1]):
                    tolerance = rule_value
                    break
        if path in null_exact or expected_value is None:
            matched = present and actual_value is None
            error = None
            tolerance = None
        elif (not present or not isinstance(expected_value, (int, float)) or isinstance(expected_value, bool)
              or not isinstance(actual_value, (int, float)) or isinstance(actual_value, bool) or tolerance is None):
            matched = False
            error = None
        else:
            error = abs(float(actual_value) - float(expected_value))
            matched = error <= float(tolerance)
        comparisons.append({
            "path": path, "expected": expected_value,
            "actual": actual_value if present else None,
            "actual_field_present": present,
            "absolute_error": error, "absolute_tolerance": tolerance,
            "matched": matched,
        })
    for prefix in contract.get("exact_key_set_prefixes") or []:
        expected_keys = {path for path in expected if path.startswith(prefix)}
        actual_keys = {path for path in actual if path.startswith(prefix)}
        if expected_keys != actual_keys:
            comparisons.append({
                "path": str(prefix).rstrip("."),
                "expected_key_count": len(expected_keys), "actual_key_count": len(actual_keys),
                "missing_keys": sorted(expected_keys - actual_keys),
                "unexpected_keys": sorted(actual_keys - expected_keys), "matched": False,
            })
    return all(row.get("matched") is True for row in comparisons), comparisons


def extract_case_numeric_fields(case: Mapping[str, Any], state: Mapping[str, Any], arm: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    return _canonical_values(case, state, arm)


def validate_numeric_match_contracts(
    cases: list[Mapping[str, Any]], *, csv_truth_document: Mapping[str, Any], csv_truth_sha256: str,
) -> dict[str, Any]:
    source_rows = {
        str(row.get("case_id")): row
        for row in (csv_truth_document.get("cases") or []) if isinstance(row, Mapping)
    }
    errors: list[str] = []
    checked = 0
    numeric_cases = [case for case in cases if (case.get("expected") or {}).get("outcome") == "numeric_result"]
    for case in numeric_cases:
        cid = str(case.get("case_id"))
        source = source_rows.get(cid)
        contract = case.get("numeric_match_contract")
        if source is None or not isinstance(contract, Mapping):
            errors.append(f"{cid}: missing independent CSV row or field-match contract")
            continue
        if source.get("candidate_semantic_status") != contract.get("source_status"):
            errors.append(f"{cid}: source reconciliation status differs")
        if contract.get("source_artifact_sha256") != csv_truth_sha256:
            errors.append(f"{cid}: source artifact hash differs")
        if (case.get("expected") or {}).get("truth") != source.get("truth"):
            errors.append(f"{cid}: selected truth differs from independent CSV truth")
        source_truth_hash = hashlib.sha256(json.dumps(
            source.get("truth"), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        if contract.get("source_truth_sha256") != source_truth_hash:
            errors.append(f"{cid}: source truth fingerprint differs")
        if contract.get("status") != "verified":
            errors.append(f"{cid}: field-match contract is not verified")
        expected_paths = set(_expected_values(case))
        tolerance_rules = contract.get("absolute_tolerance_by_path") or {}
        null_paths = set(contract.get("null_exact_paths") or [])
        declared_fields = (case.get("tolerance") or {}).get("fields") or {}
        declared_nulls = set((case.get("tolerance") or {}).get("null_exact_fields") or [])
        if declared_fields != tolerance_rules:
            errors.append(f"{cid}: public tolerance fields differ from executable field contract")
        if declared_nulls != null_paths:
            errors.append(f"{cid}: public exact-null fields differ from executable field contract")
        for path in expected_paths - null_paths:
            if path not in tolerance_rules and not any(
                str(rule).endswith("*") and path.startswith(str(rule)[:-1]) for rule in tolerance_rules
            ):
                errors.append(f"{cid}: no field tolerance for {path}")
        checked += 1
    safe_cases = [case for case in cases if (case.get("expected") or {}).get("outcome") == "safe_stop"]
    for case in safe_cases:
        tol = case.get("tolerance") or {}
        if case.get("case_id") in {"FBG017", "FBG020"} and tol.get("status") != "not_applicable_expected_capability_stop":
            errors.append(f"{case.get('case_id')}: capability stop must declare numeric tolerance not applicable")
        if case.get("case_id") == "FBG023":
            if tol.get("fields", {}).get("baseline_value") != 0.011 or "current_value" not in (tol.get("null_exact_fields") or []):
                errors.append("FBG023: data-gap baseline tolerance and explicit null assertion are required")
    return {
        "status": "verified" if checked == len(numeric_cases) and not errors else "blocked",
        "numeric_cases_checked": checked,
        "safe_stop_cases_checked": len(safe_cases),
        "source_artifact_sha256": csv_truth_sha256,
        "errors": errors,
    }

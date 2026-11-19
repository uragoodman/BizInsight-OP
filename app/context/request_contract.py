"""Canonical, traceable request contract shared by workflow layers.

The compiler is the only layer that interprets user wording.  Later layers
consume this JSON-safe snapshot and its digest; they may validate it against
registry, authorization, plans and evidence, but must not reinterpret the
message to change the requested operation, dimension or comparison roles.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any


CONTRACT_VERSION = "analysis-request-contract-v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def contract_digest(contract: dict[str, Any]) -> str:
    payload = {key: value for key, value in contract.items() if key != "contract_hash"}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def seal_request_contract(payload: dict[str, Any]) -> dict[str, Any]:
    contract = deepcopy(payload)
    contract["schema_version"] = CONTRACT_VERSION
    contract["contract_hash"] = contract_digest(contract)
    return contract


def validate_request_contract(contract: object) -> tuple[bool, str | None]:
    if not isinstance(contract, dict):
        return False, "request_contract_missing"
    if contract.get("schema_version") != CONTRACT_VERSION:
        return False, "request_contract_version_invalid"
    supplied = contract.get("contract_hash")
    if not isinstance(supplied, str) or supplied != contract_digest(contract):
        return False, "request_contract_hash_mismatch"
    intent = contract.get("operation_intent")
    allowed_intents = {
        "ordinary_metric_query", "dimensional_statistical_contribution",
        "registered_numeric_decomposition", "business_causal_investigation",
    }
    if not isinstance(intent, str) or intent not in allowed_intents:
        return False, "request_contract_operation_missing"
    comparison = contract.get("comparison")
    if not isinstance(comparison, dict):
        return False, "request_contract_comparison_missing"
    dimensions = contract.get("dimensions")
    if dimensions is not None:
        if (
            not isinstance(dimensions, list)
            or not 2 <= len(dimensions) <= 4
            or any(not isinstance(item, str) or not item.strip() for item in dimensions)
        ):
            return False, "request_contract_dimensions_invalid"
        if len(dimensions) != len(set(dimensions)):
            return False, "request_contract_dimensions_duplicate"
        if intent != "ordinary_metric_query" or contract.get("dimension"):
            return False, "request_contract_multi_dimension_operation_invalid"
        if contract.get("required_tool") != "query_metric_by_dimensions":
            return False, "request_contract_required_tool_drift"
        if contract.get("required_capability") != "breakdown":
            return False, "request_contract_required_capability_drift"
        resolved_comparison = (
            comparison.get("kind") == "day_pair"
            and comparison.get("baseline_date")
            and comparison.get("current_date")
        ) or (
            comparison.get("kind") == "period_pair"
            and comparison.get("baseline")
            and comparison.get("current")
        )
        if resolved_comparison:
            return False, "request_contract_multi_dimension_comparison_unsupported"
    if intent == "dimensional_statistical_contribution":
        evidence_sequence = contract.get("required_evidence_sequence")
        if evidence_sequence not in (None, ["trend", "dimension_contribution"]):
            return False, "request_contract_evidence_sequence_invalid"
        expected_tool = (
            "compare_breakdown_by_dimension_period"
            if comparison.get("kind") == "period_pair"
            else "compare_breakdown_by_dimension"
        )
        if contract.get("required_tool") != expected_tool:
            return False, "request_contract_required_tool_drift"
        if contract.get("required_capability") != "compare":
            return False, "request_contract_required_capability_drift"
        if contract.get("causal_boundary") != "statistical_contribution_not_business_causation":
            return False, "request_contract_causal_boundary_drift"
        # A correctly sealed but unresolved request is a valid clarification
        # outcome. Preserve its parser reason before checking fields that are
        # intentionally absent (metric/dimension/date roles).
        if contract.get("parse_status") == "clarification_required":
            return False, str(contract.get("uncertainty_reason") or "request_clarification_required")
        if not contract.get("metric_id"):
            return False, "metric_required"
        if not contract.get("dimension"):
            return False, "ambiguous_dimension"
        if contract.get("parse_status") != "resolved":
            return False, str(contract.get("uncertainty_reason") or "comparison_dates_incomplete")
        kind = comparison.get("kind")
        if kind == "day_pair" and not (comparison.get("baseline_date") and comparison.get("current_date")):
            return False, "comparison_dates_incomplete"
        if kind == "period_pair":
            for role in ("baseline", "current"):
                period = comparison.get(role)
                if not isinstance(period, dict) or not period.get("start_date") or not period.get("end_date"):
                    return False, "comparison_periods_incomplete"
        if kind not in {"day_pair", "period_pair"}:
            return False, "comparison_dates_incomplete"
    return True, None


def validate_compiled_request_contract(
    compilation: object,
    *,
    state_contract_hash: str | None = None,
    require_state_anchor: bool = False,
) -> tuple[bool, str | None]:
    if not isinstance(compilation, dict):
        return False, "request_contract_missing"
    contract = compilation.get("request_contract")
    valid, reason = validate_request_contract(contract)
    if not valid:
        return valid, reason
    assert isinstance(contract, dict)
    if compilation.get("request_contract_hash") != contract.get("contract_hash"):
        return False, "request_contract_compilation_hash_drift"
    if require_state_anchor and not state_contract_hash:
        return False, "request_contract_state_anchor_missing"
    if state_contract_hash is not None and state_contract_hash != contract.get("contract_hash"):
        return False, "request_contract_state_anchor_drift"
    # An unresolved contract is a valid clarification outcome: surface the
    # compiler's own uncertainty reason instead of failing later field-identity
    # checks that are intentionally incomplete (e.g. period_pair without roles).
    if contract.get("parse_status") != "resolved":
        return False, str(contract.get("uncertainty_reason") or "request_clarification_required")
    mirrored_intent = (compilation.get("operation_intent") or {}).get("kind")
    if mirrored_intent != contract.get("operation_intent"):
        return False, "request_contract_operation_drift"
    operation_intent = compilation.get("operation_intent") or {}
    if contract.get("operation_intent") == "dimensional_statistical_contribution":
        if operation_intent.get("required_evidence_sequence") != contract.get("required_evidence_sequence"):
            return False, "request_contract_evidence_sequence_drift"
        if operation_intent.get("dimension") != contract.get("dimension"):
            return False, "request_contract_operation_dimension_drift"
        comparison = contract.get("comparison") or {}
        if comparison.get("kind") == "period_pair":
            expected_periods = {key: value for key, value in comparison.items() if key != "kind"}
            if operation_intent.get("comparison_periods") != expected_periods:
                return False, "request_contract_operation_period_drift"
    if contract.get("dimensions") is not None:
        if operation_intent.get("dimensions") != contract.get("dimensions"):
            return False, "request_contract_operation_dimensions_drift"
    if contract.get("operation_intent") == "registered_numeric_decomposition":
        if operation_intent.get("target_metric_id") != contract.get("metric_id"):
            return False, "request_contract_target_metric_drift"
    slots = compilation.get("known_slots") or {}
    for slot_name, contract_key in (("metric", "metric_id"), ("dimension", "dimension")):
        slot = slots.get(slot_name) or {}
        slot_value = slot.get("value") if isinstance(slot, dict) else None
        contract_value = contract.get(contract_key)
        # An unresolved value is valid only when the contract itself requires
        # clarification; it can never be upgraded by a later layer.
        if slot_value != contract_value:
            if not (contract.get("parse_status") != "resolved" and not slot_value and not contract_value):
                return False, f"request_contract_{slot_name}_drift"
    dimensions_slot = slots.get("dimensions") or {}
    slot_dimensions = dimensions_slot.get("value") if isinstance(dimensions_slot, dict) else None
    contract_dimensions = contract.get("dimensions")
    if slot_dimensions != contract_dimensions:
        if not (
            contract.get("parse_status") != "resolved"
            and not slot_dimensions and not contract_dimensions
        ):
            return False, "request_contract_dimensions_drift"
    comparison = contract.get("comparison") or {}
    if comparison.get("kind") == "day_pair":
        for role in ("baseline_date", "current_date"):
            slot = slots.get(role) or {}
            slot_value = slot.get("value") if isinstance(slot, dict) else None
            if slot_value != comparison.get(role):
                return False, f"request_contract_{role}_drift"
    if comparison.get("kind") == "period_pair":
        intent_periods = (compilation.get("operation_intent") or {}).get("comparison_periods")
        expected_periods = {key: value for key, value in comparison.items() if key != "kind"}
        if intent_periods != expected_periods:
            return False, "request_contract_comparison_periods_drift"
    return True, None


def contribution_plan_mismatch(contract: dict[str, Any], decision: object, plans: object) -> str | None:
    """Return the first mismatch between a contribution contract and plan."""
    model_dump = getattr(decision, "model_dump", None)
    decision_dict = decision if isinstance(decision, dict) else model_dump(mode="json") if callable(model_dump) else {}
    plan_rows = plans if isinstance(plans, list) else []
    if not isinstance(decision_dict, dict):
        return "contract_violation_planner"
    if decision_dict.get("operation_intent") != "dimensional_statistical_contribution":
        return "contract_violation_planner_operation"
    if decision_dict.get("query_type") not in {None, "metric_contribution"}:
        return "contract_violation_planner_query_type"
    if decision_dict.get("dimension") not in {None, contract.get("dimension")}:
        return "contract_violation_planner_dimension"
    if decision_dict.get("status") != "confirmed":
        # A controlled refusal remains valid only when it does not carry an
        # executable plan or claim a different operation/target.
        if plan_rows:
            return "contract_violation_refusal_has_plan"
        metrics = set(str(item) for item in (decision_dict.get("confirmed_metrics") or []))
        if metrics and metrics != {str(contract.get("metric_id") or "")}:
            return "contract_violation_refusal_metric"
        return None
    if len(plan_rows) != 1 or not isinstance(plan_rows[0], dict):
        return "contract_violation_planner_plan_count"
    plan = plan_rows[0]
    comparison = contract.get("comparison") or {}
    if plan.get("metric_name") != contract.get("metric_id"):
        return "contract_violation_planner_metric"
    if plan.get("query_type") != "metric_contribution" or plan.get("dimension") != contract.get("dimension"):
        return "contract_violation_planner_plan_operation_or_dimension"
    if comparison.get("kind") == "day_pair":
        if plan.get("baseline_date") != comparison.get("baseline_date"):
            return "contract_violation_planner_baseline_date"
        if plan.get("current_date") != comparison.get("current_date"):
            return "contract_violation_planner_current_date"
        if plan.get("tool_name") != "compare_breakdown_by_dimension":
            return "contract_violation_planner_tool"
    elif comparison.get("kind") == "period_pair":
        baseline = comparison.get("baseline") or {}
        current = comparison.get("current") or {}
        expected = {
            "baseline_start": baseline.get("start_date"),
            "baseline_end": baseline.get("end_date"),
            "current_start": current.get("start_date"),
            "current_end": current.get("end_date"),
        }
        if any(plan.get(key) != value for key, value in expected.items()):
            return "contract_violation_planner_period"
        if plan.get("granularity") != comparison.get("granularity"):
            return "contract_violation_planner_granularity"
        if plan.get("tool_name") != "compare_breakdown_by_dimension_period":
            return "contract_violation_planner_tool"
    return None


def contribution_execution_reservation(
    contract: dict[str, Any], plan: object, state: dict[str, Any], *, request_id: str,
) -> dict[str, Any]:
    """Build a deterministic, privacy-conscious reservation for one tool execution.

    The signature binds the sealed request contract to the actual validated
    plan and trusted scope identity. The reservation itself contains only
    audit identifiers and role-bearing query parameters, never raw question
    text or credentials.
    """
    model_dump = getattr(plan, "model_dump", None)
    plan_data = plan if isinstance(plan, dict) else model_dump(mode="json") if callable(model_dump) else {}
    comparison_keys = (
        "baseline_date", "current_date", "baseline_start", "baseline_end",
        "current_start", "current_end", "granularity",
    )
    params = {key: plan_data.get(key) for key in comparison_keys if key in plan_data}
    query_parameters = {
        key: value for key, value in plan_data.items()
        if key not in {"authorized_metric", "association_pair_id", "schema_version"}
    }
    identity = {
        key: state.get(key)
        for key in ("tenant_id", "user_role", "scope_type", "current_query_scope_fingerprint")
        if state.get(key) is not None
    }
    controlled_scope = state.get("controlled_analysis_scope")
    if controlled_scope is not None:
        identity["controlled_scope_sha256"] = hashlib.sha256(
            _canonical_json(controlled_scope).encode("utf-8")
        ).hexdigest()
    authorization_boundary = state.get("authorization_boundary")
    if authorization_boundary is not None:
        identity["authorization_boundary_sha256"] = hashlib.sha256(
            _canonical_json(authorization_boundary).encode("utf-8")
        ).hexdigest()
    scope_digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    signature_payload = {
        "contract_hash": contract.get("contract_hash"),
        "tool_name": plan_data.get("tool_name"),
        "metric_id": plan_data.get("metric_name"),
        "operation": plan_data.get("query_type"),
        "dimension": plan_data.get("dimension"),
        "query_parameters_sha256": hashlib.sha256(
            _canonical_json(query_parameters).encode("utf-8")
        ).hexdigest(),
        "comparison": params,
        "scope_sha256": scope_digest,
    }
    return {
        **signature_payload,
        "execution_signature": hashlib.sha256(_canonical_json(signature_payload).encode("utf-8")).hexdigest(),
        "request_id": request_id,
        "status": "reserved",
    }


_PERIOD_CONTRIBUTION_EXECUTION_ALIASES = {
    # The planner exposes one logical period-contribution operation. The
    # registry adapter dispatches registered ratio metrics to their dedicated
    # implementations; these aliases preserve that distinction without
    # weakening metric/date/dimension/evidence checks.
    ("avg_order_value", "compare_breakdown_by_dimension_period"):
        "compare_avg_order_value_ratio_facts_by_dimension_period",
    ("freight_rate", "compare_breakdown_by_dimension_period"):
        "compare_freight_rate_ratio_facts_by_dimension_period",
}


def _contribution_execution_tool_matches(
    contract: dict[str, Any], plan: dict[str, Any], call: dict[str, Any],
) -> bool:
    planned_tool = str(plan.get("tool_name") or "")
    actual_tool = str(call.get("tool_name") or "")
    if actual_tool == planned_tool:
        return True
    comparison = contract.get("comparison") or {}
    if comparison.get("kind") != "period_pair":
        return False
    expected_adapter_tool = _PERIOD_CONTRIBUTION_EXECUTION_ALIASES.get(
        (str(contract.get("metric_id") or ""), planned_tool),
    )
    return bool(expected_adapter_tool and actual_tool == expected_adapter_tool)


def contribution_execution_state_mismatch(contract: dict[str, Any], state: dict[str, Any]) -> str | None:
    """Prove a contribution is ready for the answer route, or return a violation.

    Explicit planning refusals may proceed to a bounded answer explaining the
    limitation. A confirmed contribution must have exactly one contract-bound
    plan, reservation, tool invocation and evidence-bearing result.
    """
    decision = state.get("metric_planning_decision") or {}
    batch = state.get("metric_query_batch_plan") or {}
    plans = batch.get("plans") if isinstance(batch, dict) else None
    plans = plans if isinstance(plans, list) else []
    result_envelope = state.get("tool_result") or {}
    result_body = result_envelope.get("result") if isinstance(result_envelope, dict) else None
    result_body = result_body if isinstance(result_body, dict) else {}
    contribution_calls = [
        item for item in (state.get("tool_calls") or [])
        if isinstance(item, dict) and item.get("planned_operation") == "metric_contribution"
    ]
    contract_hash = contract.get("contract_hash")
    reservations = [
        item for item in (state.get("request_contract_query_reservations") or [])
        if isinstance(item, dict) and item.get("contract_hash") == contract_hash
    ]

    if decision.get("status") != "confirmed":
        if (
            decision.get("status") in {"unsupported", "needs_confirmation"}
            and not plans and not reservations and not contribution_calls
            and result_body.get("status") in {"unsupported", "clarification_required"}
        ):
            return None
        return "contract_violation_route_without_confirmed_plan"

    plan_mismatch = contribution_plan_mismatch(contract, decision, plans)
    if plan_mismatch:
        return f"contract_violation_route_{plan_mismatch.removeprefix('contract_violation_')}"
    if len(reservations) != 1:
        return "contract_violation_route_reservation_count"
    if len(contribution_calls) != 1:
        return "contract_violation_route_tool_call_count"
    plan = plans[0]
    call = contribution_calls[0]
    expected_reservation = contribution_execution_reservation(
        contract, plan, state, request_id=str(reservations[0].get("request_id") or ""),
    )
    if expected_reservation.get("execution_signature") != reservations[0].get("execution_signature"):
        return "contract_violation_route_execution_signature"
    if (
        not _contribution_execution_tool_matches(contract, plan, call)
        or call.get("request_id") != reservations[0].get("request_id")
        or call.get("planned_metric_id") != contract.get("metric_id")
        or call.get("planned_operation") != "metric_contribution"
    ):
        return "contract_violation_route_tool_binding"
    arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    expected_arguments = {
        "metric_name": contract.get("metric_id"),
        "dimension": contract.get("dimension"),
    }
    comparison = contract.get("comparison") or {}
    if comparison.get("kind") == "day_pair":
        expected_arguments.update({
            "baseline_date": comparison.get("baseline_date"),
            "current_date": comparison.get("current_date"),
        })
    else:
        baseline = comparison.get("baseline") or {}
        current = comparison.get("current") or {}
        expected_arguments.update({
            "baseline_start": baseline.get("start_date"),
            "baseline_end": baseline.get("end_date"),
            "current_start": current.get("start_date"),
            "current_end": current.get("end_date"),
        })
    if any(arguments.get(key) != value for key, value in expected_arguments.items()):
        return "contract_violation_route_tool_arguments"
    if result_envelope.get("request_id") != reservations[0].get("request_id"):
        return "contract_violation_route_result_request_id"
    matched_results = [
        item for item in (state.get("query_results") or [])
        if isinstance(item, dict) and item.get("request_id") == reservations[0].get("request_id")
    ]
    if len(matched_results) != 1 or not matched_results[0].get("evidence_id"):
        return "contract_violation_route_evidence_binding"
    if result_body.get("status") not in {"ok", "not_found", "unsupported", "partial", "invalid_data"}:
        return "contract_violation_route_result_status"
    return None


def record_contract_violation(state: dict[str, Any], layer: str, reason: str) -> None:
    code = reason if reason.startswith("contract_violation_") else f"contract_violation_{layer}"
    state.setdefault("request_contract_failures", []).append({
        "layer": layer, "reason": reason, "contract_hash": (
            ((state.get("context_compilation") or {}).get("request_contract") or {}).get("contract_hash")
        ),
    })
    if not any(isinstance(item, dict) and item.get("code") == code for item in state.get("errors", [])):
        state.setdefault("errors", []).append({
            "code": code,
            "message": "请求契约与下游执行不一致，已停止以避免静默更换任务。",
            "retryable": False,
            "agent_name": layer,
        })


def record_contract_observation(
    state: dict[str, Any], layer: str, observed: dict[str, Any] | None = None,
) -> None:
    """Append a compact audit observation without copying raw user data."""
    compilation = state.get("context_compilation") or {}
    contract = compilation.get("request_contract") or {}
    state.setdefault("request_contract_observations", []).append({
        "layer": layer,
        "contract_hash": contract.get("contract_hash"),
        "operation_intent": contract.get("operation_intent"),
        "metric_id": contract.get("metric_id"),
        "dimension": contract.get("dimension"),
        "comparison": deepcopy(contract.get("comparison")),
        "observed": deepcopy(observed or {}),
    })

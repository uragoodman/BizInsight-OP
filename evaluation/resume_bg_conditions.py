"""Fail-closed, offline-testable run-condition checks for B/G cases.

These checks are evidence validators, not data loaders or permission bypasses.
Production callers must supply observations obtained through the normal
authorized query/registry path after loading the isolated dataset.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

CONDITION_CONTRACT_VERSION = "resume-bg-conditions-v1"


def validate_missing_tail(
    *, requested_dates: Sequence[str], missing_dates: Sequence[str], points: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    requested = [str(item) for item in requested_dates]
    missing = [str(item) for item in missing_dates]
    point_by_date = {str(point.get("date") or point.get("bucket") or "")[:10]: point for point in points
                     if isinstance(point, Mapping)}
    if not requested or len(set(requested)) != len(requested) or len(set(missing)) != len(missing):
        return {"status": "experiment_condition_unverified", "reason": "invalid_or_ambiguous_date_contract"}
    if not set(missing).issubset(set(requested)):
        return {"status": "experiment_condition_invalid", "reason": "missing_dates_outside_requested_window"}
    observed_missing = {date for date in missing if date not in point_by_date or point_by_date[date].get("value") is None}
    fabricated_zero = {date for date in missing if date in point_by_date and point_by_date[date].get("value") == 0}
    controls = [date for date in requested if date not in set(missing)]
    controls_present = bool(controls) and all(
        date in point_by_date and point_by_date[date].get("value") is not None for date in controls
    )
    if fabricated_zero:
        return {"status": "experiment_condition_invalid", "reason": "missing_bucket_filled_as_zero",
                "missing_dates": sorted(missing), "zero_filled_dates": sorted(fabricated_zero),
                "control_buckets_present": controls_present}
    if observed_missing != set(missing) or not controls_present:
        return {"status": "experiment_condition_invalid", "reason": "missing_tail_or_control_bucket_mismatch",
                "expected_missing_dates": sorted(missing), "observed_missing_dates": sorted(observed_missing),
                "control_buckets_present": controls_present}
    if len(missing) != 2 or requested[-2:] != missing:
        return {"status": "experiment_condition_invalid", "reason": "missing_tail_contract_not_exactly_two_terminal_buckets",
                "missing_dates": sorted(missing), "requested_dates": requested}
    return {"status": "ready", "condition_contract_version": CONDITION_CONTRACT_VERSION,
            "missing_dates": sorted(missing), "control_buckets_present": True,
            "missing_not_zero_filled": True}


def validate_capability_refusal(
    *, metric_id: str, scope: str, operation: str, dimension: str,
    capability_snapshot: Mapping[str, Any] | None,
    authorization_snapshot: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Separate unsupported capability from unauthorized access and supported use."""
    if not isinstance(capability_snapshot, Mapping) or not isinstance(authorization_snapshot, Mapping):
        return {"status": "experiment_condition_unverified", "reason": "capability_or_authorization_snapshot_missing"}
    expected = (str(metric_id), str(scope), str(operation), str(dimension))
    observed_cap = tuple(str(capability_snapshot.get(key) or "") for key in
                         ("metric_id", "scope", "operation", "dimension"))
    observed_auth = tuple(str(authorization_snapshot.get(key) or "") for key in
                          ("metric_id", "scope", "operation", "dimension"))
    if observed_cap != expected or observed_auth != expected:
        return {"status": "experiment_condition_unverified", "reason": "snapshot_request_contract_mismatch",
                "expected": expected, "capability_observed": observed_cap, "authorization_observed": observed_auth}
    capability_supported = capability_snapshot.get("supported")
    authorized = authorization_snapshot.get("authorized")
    if not isinstance(capability_supported, bool) or not isinstance(authorized, bool):
        return {"status": "experiment_condition_unverified", "reason": "snapshot_boolean_missing"}
    if capability_supported and not authorized:
        return {"status": "experiment_condition_invalid", "reason": "authorization_denial_is_not_capability_gap"}
    if capability_supported and authorized:
        return {"status": "experiment_condition_invalid", "reason": "requested_capability_is_supported_and_authorized"}
    if not capability_supported and not authorized:
        return {"status": "experiment_condition_unverified", "reason": "unsupported_and_unauthorized_are_confounded"}
    return {"status": "ready", "condition_contract_version": CONDITION_CONTRACT_VERSION,
            "capability_supported": False, "authorized": True,
            "request": {"metric_id": metric_id, "scope": scope, "operation": operation, "dimension": dimension}}


def validate_case_pre_run_condition(
    case: Mapping[str, Any], *, observation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate special-case preconditions after data load, before run reservation/start.

    ``observation`` must be produced through the normal authorized registry/query
    path by the caller. Missing evidence fails closed; case prose is not proof.
    """
    task = str(case.get("protocol_task_type") or "")
    if task == "expected_data_insufficiency":
        contract = case.get("condition_contract")
        if not isinstance(contract, Mapping) or not isinstance(observation, Mapping):
            return {"status": "experiment_condition_unverified", "reason": "structured_condition_contract_or_query_observation_missing"}
        result = validate_missing_tail(
            requested_dates=contract.get("requested_dates") or [],
            missing_dates=contract.get("missing_dates") or [],
            points=observation.get("points") or [],
        )
        result["metric_id"] = contract.get("metric_id")
        result["scope"] = contract.get("scope")
        result["comparison_window"] = contract.get("comparison_window")
        return result
    if task == "expected_capability_limit":
        contract = case.get("limitation_contract")
        if not isinstance(contract, Mapping) or not isinstance(observation, Mapping):
            return {"status": "experiment_condition_unverified", "reason": "structured_limitation_contract_or_capability_snapshot_missing"}
        result = validate_capability_refusal(
            metric_id=str(contract.get("metric_id") or ""),
            scope=str(contract.get("scope") or ""),
            operation=str(contract.get("operation") or ""),
            dimension=str(contract.get("dimension") or ""),
            capability_snapshot=observation.get("capability_snapshot"),
            authorization_snapshot=observation.get("authorization_snapshot"),
        )
        if case.get("readiness") == "blocked" and result.get("status") == "ready":
            return {"status": "experiment_condition_invalid", "reason": "case_readiness_metadata_conflicts_with_verified_qualification",
                    "qualification": result}
        return result
    return {"status": "experiment_condition_not_applicable", "condition_contract_version": CONDITION_CONTRACT_VERSION}

"""Fail-closed result governance at the output boundary.

The query layer remains responsible for authorization and correctness.  This
module protects the *returned artifact*: small-cell suppression, classification
filtering, field masking, and export approval.  It intentionally does not
guess a sample size for legacy payloads; a producer must provide one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from app.core.config import get_settings


class ResultPolicyDenied(PermissionError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ResultPolicyDecision:
    payload: dict[str, Any]
    suppressed_cells: int = 0
    masked_fields: int = 0
    classification: str = "internal"
    reidentification_risk_score: int = 0


def _json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _allowed_classifications() -> set[str]:
    try:
        value = json.loads(get_settings().result_policy_allowed_classifications_json)
    except (TypeError, ValueError):
        return {"public", "internal"}
    return {str(item) for item in value} if isinstance(value, list) else {"public", "internal"}


def _mask_value(value: Any, replacement: str = "[MASKED]") -> Any:
    if isinstance(value, dict):
        return {key: _mask_value(item, replacement) for key, item in value.items()}
    if isinstance(value, list):
        return [_mask_value(item, replacement) for item in value]
    return replacement


def _apply_masking(value: Any, masks: Mapping[str, Any]) -> tuple[Any, int]:
    if isinstance(value, dict):
        count = 0
        result: dict[str, Any] = {}
        for key, item in value.items():
            rule = masks.get(key)
            if rule:
                result[key] = _mask_value(item, str(rule) if isinstance(rule, str) else "[MASKED]")
                count += 1
            else:
                result[key], nested = _apply_masking(item, masks)
                count += nested
        return result, count
    if isinstance(value, list):
        items = []
        count = 0
        for item in value:
            masked, nested = _apply_masking(item, masks)
            items.append(masked)
            count += nested
        return items, count
    return value, 0


def _suppress_small_cells(value: Any, minimum: int) -> tuple[Any, int]:
    """Suppress only explicit count/sample_size cells below the threshold."""

    if isinstance(value, list):
        result = []
        count = 0
        for item in value:
            masked, nested = _suppress_small_cells(item, minimum)
            result.append(masked)
            count += nested
        return result, count
    if not isinstance(value, dict):
        return value, 0
    result: dict[str, Any] = {}
    count = 0
    sample = value.get("sample_size", value.get("sample_count"))
    if sample is None and isinstance(value.get("count"), (int, float)):
        sample = value.get("count")
    small = isinstance(sample, (int, float)) and sample < minimum
    for key, item in value.items():
        if small and key in {"value", "mean", "median", "min", "max", "q1", "q3", "points", "items"}:
            result[key] = None
            count += 1
        else:
            result[key], nested = _suppress_small_cells(item, minimum)
            count += nested
    if small:
        result["suppressed"] = True
        result["suppression_reason"] = "small_cell"
    return result, count


def _suppress_high_dimensional_detail(value: Any, maximum_dimensions: int) -> tuple[Any, int]:
    """Limit derived output that can make small groups re-identifiable.

    Producers may declare the number of grouping dimensions using
    ``dimension_count``/``dimensions``/``group_by``.  The policy does not
    infer a risk from arbitrary prose; it only acts on explicit provenance
    supplied by the query producer.  Detail cells are suppressed rather than
    silently changing the query semantics.
    """

    if isinstance(value, list):
        result = []
        count = 0
        for item in value:
            protected, nested = _suppress_high_dimensional_detail(item, maximum_dimensions)
            result.append(protected)
            count += nested
        return result, count
    if not isinstance(value, dict):
        return value, 0
    result: dict[str, Any] = {}
    count = 0
    dimensions = value.get("dimensions", value.get("group_by"))
    declared_count = value.get("dimension_count")
    if declared_count is None and isinstance(dimensions, list):
        declared_count = len(dimensions)
    too_many = isinstance(declared_count, (int, float)) and declared_count > maximum_dimensions
    quasi = value.get("quasi_identifiers")
    reidentification_risk = isinstance(quasi, list) and bool(quasi)
    for key, item in value.items():
        if (too_many or reidentification_risk) and key in {
            "value", "mean", "median", "min", "max", "q1", "q3", "points", "items", "rows", "evidence"
        }:
            result[key] = None
            count += 1
        else:
            result[key], nested = _suppress_high_dimensional_detail(item, maximum_dimensions)
            count += nested
    if too_many or reidentification_risk:
        result["suppressed"] = True
        result["suppression_reason"] = "derived_output_privacy"
    return result, count


def _reidentification_risk(value: Any, *, maximum_quasi_identifiers: int) -> int:
    """Return a conservative, explainable risk score for derived output.

    This is intentionally not a statistical identity model.  It is a bounded
    policy check for producer-declared quasi-identifiers and uniqueness hints.
    A score is only computed from explicit metadata, never inferred from
    arbitrary business text.  This makes the policy reproducible and safe to
    run at the API boundary.
    """

    if isinstance(value, list):
        return max(
            (_reidentification_risk(item, maximum_quasi_identifiers=maximum_quasi_identifiers) for item in value),
            default=0,
        )
    if not isinstance(value, dict):
        return 0
    score = 0
    quasi = value.get("quasi_identifiers")
    if isinstance(quasi, list):
        count = len(quasi)
        if count > maximum_quasi_identifiers:
            score += min(3, count - maximum_quasi_identifiers)
    if value.get("unique_combination") is True or value.get("contains_direct_identifier") is True:
        score += 3
    if isinstance(value.get("distinct_count"), (int, float)) and value.get("distinct_count") <= 5:
        score += 1
    nested = max(
        (_reidentification_risk(item, maximum_quasi_identifiers=maximum_quasi_identifiers) for item in value.values()),
        default=0,
    )
    return max(score, nested)


def _suppress_reidentification_risk(value: Any, *, threshold: int, maximum_quasi_identifiers: int) -> tuple[Any, int, int]:
    """Suppress detail when an explicit derived-output risk score is high."""

    if isinstance(value, list):
        result = []
        suppressed = 0
        score = 0
        for item in value:
            protected, count, item_score = _suppress_reidentification_risk(
                item, threshold=threshold, maximum_quasi_identifiers=maximum_quasi_identifiers
            )
            result.append(protected)
            suppressed += count
            score = max(score, item_score)
        return result, suppressed, score
    if not isinstance(value, dict):
        return value, 0, 0
    score = _reidentification_risk(value, maximum_quasi_identifiers=maximum_quasi_identifiers)
    if score >= threshold and any(key in value for key in ("rows", "items", "points", "evidence", "value")):
        result = dict(value)
        count = 0
        for key in ("rows", "items", "points", "evidence", "value", "mean", "median", "min", "max"):
            if key in result:
                result[key] = None
                count += 1
        result["suppressed"] = True
        result["suppression_reason"] = "reidentification_risk"
        result["reidentification_risk_score"] = score
        return result, count, score
    result: dict[str, Any] = {}
    suppressed = 0
    nested_score = score
    for key, item in value.items():
        protected, count, item_score = _suppress_reidentification_risk(
            item, threshold=threshold, maximum_quasi_identifiers=maximum_quasi_identifiers
        )
        result[key] = protected
        suppressed += count
        nested_score = max(nested_score, item_score)
    return result, suppressed, nested_score


def apply_result_policy(
    payload: Mapping[str, Any],
    *,
    action: str = "analyze",
    classification: str | None = None,
    export_approved: bool = False,
    authorization_context: Any | None = None,
) -> ResultPolicyDecision:
    """Apply output controls without changing the source query semantics."""

    settings = get_settings()
    classification = str(classification or payload.get("classification") or "internal")
    allowed = set(getattr(authorization_context, "allowed_classifications", ()) or _allowed_classifications())
    if classification not in allowed:
        raise ResultPolicyDenied("RESULT_CLASSIFICATION_DENIED", "Result classification is not visible to this caller.")
    if action == "export" and settings.result_policy_export_approval_required and not export_approved:
        raise ResultPolicyDenied("EXPORT_APPROVAL_REQUIRED", "Export requires an approved report artifact.")
    result, suppressed = _suppress_small_cells(dict(payload), max(0, int(settings.result_policy_min_cell_size)))
    result, derived_suppressed = _suppress_high_dimensional_detail(
        result, max(1, int(settings.result_policy_max_derived_dimensions))
    )
    result, risk_suppressed, risk_score = _suppress_reidentification_risk(
        result,
        threshold=max(1, int(settings.result_policy_reidentification_threshold)),
        maximum_quasi_identifiers=max(0, int(settings.result_policy_reidentification_max_quasi_identifiers)),
    )
    masks = _json_object(settings.result_policy_mask_fields_json)
    result, masked = _apply_masking(result, masks)
    return ResultPolicyDecision(
        payload=result,
        suppressed_cells=suppressed + derived_suppressed + risk_suppressed,
        masked_fields=masked,
        classification=classification,
        reidentification_risk_score=risk_score,
    )


def enforce_export_approval(*, approved: bool) -> None:
    settings = get_settings()
    if settings.result_policy_export_approval_required and not approved:
        raise ResultPolicyDenied("EXPORT_APPROVAL_REQUIRED", "Export requires an approved report artifact.")

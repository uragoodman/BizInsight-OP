"""Deterministic, versioned attribution formulas with explicit reconciliation."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from app.investigation.contracts import AttributionComponent, AttributionResult


class AttributionDataError(ValueError):
    """Raised when data cannot support a mathematically defined attribution."""


class DataReconciliationError(AttributionDataError):
    """Complete, comparable evidence disagrees on the same metric result."""


SUPPORTED_ATTRIBUTION_METHODS = frozenset({
    "additive_change_v1",
    "symmetric_mix_within_v1",
    "symmetric_ratio_v1",
    "product_shapley_v1",
    "proportional_overlap_v1",
})

# Registry declarations are executable capability contracts, not free-form
# documentation. Keep the required sufficient-statistic fields beside the
# deterministic methods so startup can reject misspelled or stale contracts.
ATTRIBUTION_REQUIRED_DATA_FIELDS = {
    "additive_change_v1": frozenset({
        "baseline_total", "current_total", "baseline_group_totals", "current_group_totals",
    }),
    "symmetric_mix_within_v1": frozenset({
        "grouped_numerator", "grouped_denominator", "complete_group_population",
    }),
    "symmetric_ratio_v1": frozenset({
        "baseline_numerator", "current_numerator", "baseline_denominator", "current_denominator",
    }),
    "product_shapley_v1": frozenset({
        "baseline_factor_values", "current_factor_values",
    }),
    "proportional_overlap_v1": frozenset({
        "baseline_total", "current_total", "baseline_group_totals",
        "current_group_totals", "allocation_method",
    }),
}


def _finite_number(value: Any, label: str) -> float:
    """Coerce a numeric input while rejecting booleans and non-finite values."""
    if isinstance(value, bool):
        raise AttributionDataError(f"{label} must be a finite number, not a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AttributionDataError(f"{label} must be a finite number; non-finite or non-numeric input") from exc
    if not math.isfinite(number):
        raise AttributionDataError(f"{label} must be a finite number; non-finite or non-numeric input")
    return number


def _validate_tolerance(tolerance: Any) -> float:
    value = _finite_number(tolerance, "tolerance")
    if value < 0:
        raise AttributionDataError("tolerance must be non-negative")
    return value


def _require_mapping(value: Any, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise AttributionDataError(f"{label} must be a mapping")
    return value


def _validate_metric_id(metric_id: Any) -> str:
    if not isinstance(metric_id, str) or not metric_id.strip():
        raise AttributionDataError("metric_id must be a non-empty string")
    return metric_id


def _validate_unit(unit: Any) -> str:
    if not isinstance(unit, str):
        raise AttributionDataError("unit must be a string")
    return unit


def _validate_evidence_ids(evidence_ids: Any) -> list[str]:
    if evidence_ids is None:
        return []
    if (not isinstance(evidence_ids, list)
            or any(not isinstance(item, str) or not item.strip() for item in evidence_ids)):
        raise AttributionDataError("evidence_ids must be a list of non-empty strings")
    return evidence_ids


def _finite_sum(values: Any, label: str) -> float:
    try:
        total = math.fsum(values)
    except (OverflowError, ValueError) as exc:
        raise AttributionDataError(f"{label} exceeds finite numeric range") from exc
    if not math.isfinite(total):
        raise AttributionDataError(f"{label} exceeds finite numeric range")
    return total


def _ratio(explained: float, delta: float) -> float:
    if abs(delta) <= 1e-12:
        return 1.0 if abs(explained) <= 1e-12 else 0.0
    return max(0.0, min(1.0, 1.0 - abs(delta - explained) / abs(delta)))


def additive_change_attribution(
    *, metric_id: str, baseline_value: float, current_value: float,
    baseline_groups: Mapping[str, float], current_groups: Mapping[str, float],
    scope_fingerprint: str = "", evidence_ids: list[str] | None = None,
    tolerance: float = 0.01,
) -> AttributionResult:
    """Attribute an additive total only when groups fully reconcile to totals."""
    metric_id = _validate_metric_id(metric_id)
    evidence_ids = _validate_evidence_ids(evidence_ids)
    baseline_value = _finite_number(baseline_value, "baseline_value")
    current_value = _finite_number(current_value, "current_value")
    tolerance = _validate_tolerance(tolerance)
    baseline_groups = _require_mapping(baseline_groups, "baseline_groups")
    current_groups = _require_mapping(current_groups, "current_groups")
    for label, groups in (("baseline_groups", baseline_groups), ("current_groups", current_groups)):
        if any(not isinstance(group, str) or not group for group in groups):
            raise AttributionDataError(f"{label} keys must be non-empty strings")
    baseline_values = {group: _finite_number(value, f"baseline group {group!r}")
                       for group, value in baseline_groups.items()}
    current_values = {group: _finite_number(value, f"current group {group!r}")
                      for group, value in current_groups.items()}
    delta = current_value - baseline_value
    if not math.isfinite(delta):
        raise AttributionDataError("period delta exceeds finite numeric range")
    baseline_sum = sum(baseline_values.values())
    current_sum = sum(current_values.values())
    if not math.isfinite(baseline_sum) or not math.isfinite(current_sum):
        raise AttributionDataError("group totals exceed finite numeric range")
    if abs(baseline_sum - baseline_value) > tolerance or abs(current_sum - current_value) > tolerance:
        raise AttributionDataError("dimension coverage does not reconcile to period totals")
    component_values = {
        group: current_values.get(group, 0.0) - baseline_values.get(group, 0.0)
        for group in sorted(set(baseline_values) | set(current_values))
    }
    if any(not math.isfinite(value) for value in component_values.values()):
        raise AttributionDataError("group contribution exceeds finite numeric range")
    components = [
        AttributionComponent(
            component_id=str(group),
            value=component_values[group],
            component_type="group_delta",
        )
        for group in sorted(set(baseline_groups) | set(current_groups))
    ]
    explained = sum(component.value for component in components)
    residual = delta - explained
    if not math.isfinite(explained) or not math.isfinite(residual):
        raise AttributionDataError("additive reconciliation exceeds finite numeric range")
    if abs(residual) > tolerance:
        raise AttributionDataError("group contributions fail reconciliation")
    return AttributionResult(
        metric_id=metric_id, method_id="additive_change_v1", method_version="1.0",
        baseline_value=float(baseline_value), current_value=float(current_value), total_delta=delta,
        raw_components=components, display_method="raw", display_components=components,
        reconciliation_ratio=_ratio(explained, delta), residual_value=residual,
        residual_type="rounding" if abs(residual) > 1e-12 else "unknown",
        residual_explainable=True,
        residual_tolerance=tolerance,
        residual_note="分组增量与整体变化在加总容差内对账。",
        evidence_ids=evidence_ids,
    )


def symmetric_mix_within_attribution(
    *, metric_id: str,
    baseline_groups: Mapping[str, Mapping[str, float]],
    current_groups: Mapping[str, Mapping[str, float]],
    numerator_key: str = "numerator", denominator_key: str = "denominator",
    multiplier: float = 1.0, unit: str = "", evidence_ids: list[str] | None = None,
    tolerance: float = 1e-9,
) -> AttributionResult:
    """Decompose a weighted rate/average into within, mix, and structural effects.

    A group present in only one period is reported as an entry/exit structural
    effect. Its undefined within-group rate is never imputed as zero.
    """
    metric_id = _validate_metric_id(metric_id)
    unit = _validate_unit(unit)
    evidence_ids = _validate_evidence_ids(evidence_ids)
    scale = _finite_number(multiplier, "ratio multiplier")
    tolerance = _validate_tolerance(tolerance)
    if scale <= 0:
        raise AttributionDataError("ratio multiplier must be finite and positive")

    def checked(data: Mapping[str, Mapping[str, float]], label: str) -> tuple[dict[str, tuple[float, float]], float]:
        data = _require_mapping(data, f"{label}_groups")
        rows: dict[str, tuple[float, float]] = {}
        for group, values in data.items():
            if not isinstance(group, str) or not group:
                raise AttributionDataError(f"{label} group keys must be non-empty strings")
            values = _require_mapping(values, f"{label} group {group!r} facts")
            try:
                numerator = _finite_number(values[numerator_key], f"{label} group {group!r} numerator")
                denominator = _finite_number(values[denominator_key], f"{label} group {group!r} denominator")
            except KeyError as exc:
                raise AttributionDataError(f"{label} group {group!r} is missing numerator/denominator") from exc
            if denominator < 0:
                raise AttributionDataError(f"{label} group {group!r} has negative denominator")
            if abs(denominator) <= tolerance and abs(numerator) > tolerance:
                raise AttributionDataError(f"{label} group {group!r} has a nonzero numerator with zero denominator")
            if denominator > 0:
                rows[group] = (numerator, denominator)
        denominator_total = sum(denominator for _, denominator in rows.values())
        if not math.isfinite(denominator_total) or denominator_total <= tolerance:
            raise AttributionDataError(f"{label} period has zero effective denominator")
        return rows, denominator_total

    base, base_den = checked(baseline_groups, "baseline")
    current, current_den = checked(current_groups, "current")
    base_value = scale * sum(num for num, _ in base.values()) / base_den
    current_value = scale * sum(num for num, _ in current.values()) / current_den
    if not math.isfinite(base_value) or not math.isfinite(current_value):
        raise AttributionDataError("ratio values exceed finite numeric range")
    components: list[AttributionComponent] = []
    for group in sorted(set(base) | set(current)):
        in_base, in_current = group in base, group in current
        if in_base and in_current:
            n0, d0 = base[group]
            n1, d1 = current[group]
            r0, r1 = scale * n0 / d0, scale * n1 / d1
            w0, w1 = d0 / base_den, d1 / current_den
            components.extend((
                AttributionComponent(component_id=f"{group}:within", value=(r1-r0)*(w1+w0)/2,
                    unit=unit, component_type="within_effect"),
                AttributionComponent(component_id=f"{group}:mix", value=(w1-w0)*(r1+r0)/2,
                    unit=unit, component_type="mix_effect"),
            ))
        elif in_current:
            n1, d1 = current[group]
            components.append(AttributionComponent(component_id=f"{group}:entry", value=(d1/current_den)*(scale*n1/d1),
                unit=unit, component_type="entry_structural_effect"))
        else:
            n0, d0 = base[group]
            components.append(AttributionComponent(component_id=f"{group}:exit", value=-(d0/base_den)*(scale*n0/d0),
                unit=unit, component_type="exit_structural_effect"))

    delta = current_value - base_value
    explained = sum(component.value for component in components)
    residual = delta - explained
    if (not math.isfinite(delta) or not math.isfinite(explained) or not math.isfinite(residual)
            or any(not math.isfinite(component.value) for component in components)):
        raise AttributionDataError("mix-within effects exceed finite numeric range")
    # Cancellation in ``current_value - base_value`` and summation of many
    # group effects can exceed a tiny fixed absolute epsilon even when the
    # inputs satisfy the exact identity. Retain the caller's absolute
    # tolerance, but never demand precision below a conservative float-roundoff
    # bound derived from the magnitudes actually added/subtracted.
    arithmetic_scale = max(
        1.0, abs(base_value), abs(current_value),
        sum(abs(component.value) for component in components),
    )
    rounding_floor = math.ulp(arithmetic_scale) * 4 * max(1, len(components))
    if not math.isfinite(rounding_floor):
        raise AttributionDataError("mix-within rounding tolerance exceeds finite numeric range")
    allowed_error = max(tolerance, rounding_floor)
    if abs(residual) > allowed_error:
        raise AttributionDataError("symmetric mix-within components fail reconciliation")
    return AttributionResult(
        metric_id=metric_id, method_id="symmetric_mix_within_v1", method_version="1.0",
        baseline_value=base_value, current_value=current_value, total_delta=delta,
        raw_components=components, display_method="raw", display_components=components,
        reconciliation_ratio=_ratio(explained, delta), residual_value=residual,
        residual_type="rounding" if abs(residual) > 1e-12 else "unknown",
        residual_explainable=True,
        residual_tolerance=allowed_error,
        residual_note="组内、构成及进入/退出结构效应与加权比率变化在容差内对账。",
        evidence_ids=evidence_ids,
    )


def symmetric_ratio_attribution(
    *,
    metric_id: str,
    baseline_numerator: float,
    current_numerator: float,
    baseline_denominator: float,
    current_denominator: float,
    multiplier: float = 1.0,
    unit: str = "",
    evidence_ids: list[str] | None = None,
    tolerance: float = 1e-8,
) -> AttributionResult:
    """Exactly decompose a scalar ratio change into numerator/denominator effects.

    For R = multiplier * N / D, the symmetric two-factor path allocation
    splits the finite change evenly across the baseline/current endpoints. It
    is descriptive decomposition, not a causal effect.
    """
    metric_id = _validate_metric_id(metric_id)
    unit = _validate_unit(unit)
    evidence_ids = _validate_evidence_ids(evidence_ids)
    values = (
        baseline_numerator, current_numerator, baseline_denominator,
        current_denominator, multiplier,
    )
    n0, n1, d0, d1, scale = (
        _finite_number(value, "ratio component") for value in values
    )
    tolerance = _validate_tolerance(tolerance)
    if d0 <= tolerance or d1 <= tolerance:
        raise AttributionDataError("ratio decomposition requires positive denominators in both periods")
    if scale <= 0:
        raise AttributionDataError("ratio multiplier must be positive")

    base = scale * n0 / d0
    current = scale * n1 / d1
    numerator_effect = scale * (n1 - n0) * (1.0 / d0 + 1.0 / d1) / 2.0
    denominator_effect = scale * (n0 + n1) / 2.0 * (1.0 / d1 - 1.0 / d0)
    delta = current - base
    explained = _finite_sum((numerator_effect, denominator_effect), "ratio component sum")
    residual = delta - explained
    if not all(math.isfinite(value) for value in (
        base, current, numerator_effect, denominator_effect, delta, residual,
    )):
        raise AttributionDataError("ratio effects exceed finite numeric range")
    endpoint_resolution = max(math.ulp(abs(base)), math.ulp(abs(current)))
    component_scale = max(abs(numerator_effect), abs(denominator_effect), abs(explained))
    component_rounding_floor = math.ulp(component_scale) * 4
    if (delta == 0.0 and explained != 0.0
            and component_rounding_floor < abs(explained) <= endpoint_resolution):
        raise AttributionDataError(
            "ratio delta is below endpoint float precision; attribution is not reliably resolvable"
        )
    arithmetic_scale = max(1.0, abs(base), abs(current), abs(numerator_effect) + abs(denominator_effect))
    rounding_floor = math.ulp(arithmetic_scale) * 2
    allowed_error = max(tolerance, abs(delta) * 1e-10, rounding_floor)
    if not math.isfinite(allowed_error):
        raise AttributionDataError("ratio rounding tolerance exceeds finite numeric range")
    if abs(residual) > allowed_error:
        raise AttributionDataError("symmetric ratio effects fail reconciliation")
    components = [
        AttributionComponent(
            component_id="numerator_effect", value=numerator_effect,
            unit=unit, component_type="numerator_effect",
        ),
        AttributionComponent(
            component_id="denominator_effect", value=denominator_effect,
            unit=unit, component_type="denominator_effect",
        ),
    ]
    return AttributionResult(
        metric_id=metric_id, method_id="symmetric_ratio_v1", method_version="1.0",
        baseline_value=base, current_value=current, total_delta=delta,
        raw_components=components, display_method="raw", display_components=components,
        reconciliation_ratio=_ratio(explained, delta),
        residual_value=residual,
        residual_type="rounding" if residual != 0.0 else "unknown",
        residual_explainable=True,
        residual_tolerance=allowed_error,
        residual_note="分子效应与分母效应之和按对称分解公式与比率变化对账；有限精度残差单独保留。",
        evidence_ids=evidence_ids,
    )


def bind_registered_ratio_factor_identities(result: AttributionResult, *, scope: str) -> AttributionResult:
    """Attach exact ratio-role identities from the registered metric contract.

    This never infers factors from values, order, or labels. A malformed or
    conflicting registration fails closed before the result is presented.
    """
    from app.metrics.registry import get_registry

    if result.method_id != "symmetric_ratio_v1":
        raise ValueError("ratio factor identity binding requires symmetric_ratio_v1")
    definition = get_registry().get(result.metric_id)
    capability = definition.attribution.for_scope(scope)
    aggregation = definition.period_aggregation
    if (
        "symmetric_ratio_v1" not in capability.method_ids
        or aggregation is None
        or aggregation.kind != "weighted_ratio"
    ):
        raise ValueError("registered weighted-ratio decomposition is unavailable in this scope")
    expected = {
        "numerator_effect": aggregation.numerator_metric,
        "denominator_effect": aggregation.denominator_metric,
    }
    display_component_ids = [item.component_id for item in result.display_components]
    raw_component_ids = [item.component_id for item in result.raw_components]
    if (
        result.metric_id != definition.metric_id
        or set(display_component_ids) != set(expected)
        or len(display_component_ids) != len(expected)
        or set(raw_component_ids) != set(expected)
        or len(raw_component_ids) != len(expected)
    ):
        raise ValueError("ratio component roles do not match the registered factor contract")
    identities = [
        {
            "role_id": role_id,
            "metric_id": metric_id,
            "display_name": get_registry().get(metric_id).display_name or metric_id,
        }
        for role_id, metric_id in expected.items()
    ]
    if result.factor_identities:
        expected_identity_map = {
            item["role_id"]: (item["metric_id"], item["display_name"])
            for item in identities
        }
        actual = {
            item.role_id: (item.metric_id, item.display_name)
            for item in result.factor_identities
        }
        if actual != expected_identity_map:
            raise ValueError("existing ratio factor identity mapping conflicts with the registry")
    return AttributionResult.model_validate({
        **result.model_dump(mode="python"),
        "factor_identities": identities,
    })


def product_shapley_attribution(
    *, metric_id: str, baseline_factors: Mapping[str, float], current_factors: Mapping[str, float],
    unit: str = "", evidence_ids: list[str] | None = None,
) -> AttributionResult:
    """Exact product decomposition; retain interaction and Shapley display split."""
    metric_id = _validate_metric_id(metric_id)
    unit = _validate_unit(unit)
    evidence_ids = _validate_evidence_ids(evidence_ids)
    baseline_factors = _require_mapping(baseline_factors, "baseline_factors")
    current_factors = _require_mapping(current_factors, "current_factors")
    if (not baseline_factors or set(baseline_factors) != set(current_factors)
            or any(not isinstance(key, str) or not key for key in baseline_factors)):
        raise AttributionDataError("baseline/current product factors must have identical nonempty keys")
    if len(baseline_factors) != 2:
        raise AttributionDataError("product_shapley_v1 currently supports exactly two factors")
    names = sorted(baseline_factors)
    a0, b0 = (_finite_number(baseline_factors[name], f"baseline factor {name!r}") for name in names)
    a1, b1 = (_finite_number(current_factors[name], f"current factor {name!r}") for name in names)
    da, db = a1-a0, b1-b0
    effect_a, effect_b, interaction = da*b0, db*a0, da*db
    baseline_value, current_value = a0*b0, a1*b1
    delta = current_value - baseline_value
    if not all(math.isfinite(value) for value in (
        baseline_value, current_value, delta, effect_a, effect_b, interaction,
    )):
        raise AttributionDataError("product effects exceed finite numeric range")
    raw = [
        AttributionComponent(component_id=f"{names[0]}:baseline_factor_effect", value=effect_a,
            unit=unit, component_type="factor_effect"),
        AttributionComponent(component_id=f"{names[1]}:baseline_factor_effect", value=effect_b,
            unit=unit, component_type="factor_effect"),
        AttributionComponent(component_id="interaction", value=interaction, unit=unit,
            component_type="interaction_effect"),
    ]
    display_values = (effect_a + interaction / 2, effect_b + interaction / 2)
    if not all(math.isfinite(value) for value in display_values):
        raise AttributionDataError("product display effects exceed finite numeric range")
    displayed = [
        AttributionComponent(component_id=names[0], value=display_values[0], unit=unit,
            component_type="shapley_factor_effect"),
        AttributionComponent(component_id=names[1], value=display_values[1], unit=unit,
            component_type="shapley_factor_effect"),
    ]
    raw_sum = _finite_sum((item.value for item in raw), "product raw component sum")
    display_sum = _finite_sum((item.value for item in displayed), "product display component sum")
    arithmetic_scale = max(
        1.0, abs(baseline_value), abs(current_value),
        _finite_sum((abs(item.value) for item in raw), "product raw component magnitude"),
        _finite_sum((abs(item.value) for item in displayed), "product display component magnitude"),
    )
    rounding_floor = math.ulp(arithmetic_scale) * max(1, len(raw))
    if not math.isfinite(rounding_floor):
        raise AttributionDataError("product rounding tolerance exceeds finite numeric range")
    endpoint_resolution = max(math.ulp(abs(baseline_value)), math.ulp(abs(current_value)))
    if delta == 0.0 and raw_sum != 0.0 and abs(raw_sum) <= endpoint_resolution:
        raise AttributionDataError(
            "product delta is below endpoint float precision; attribution is not reliably resolvable"
        )
    residual = delta - raw_sum
    allowed_error = max(1e-12, rounding_floor)
    if abs(residual) > allowed_error or abs(display_sum - raw_sum) > allowed_error:
        raise AttributionDataError("product raw/display effects fail finite-precision reconciliation")
    return AttributionResult(
        metric_id=metric_id, method_id="product_shapley_v1", method_version="1.0",
        baseline_value=baseline_value, current_value=current_value, total_delta=delta,
        raw_components=raw, display_method="shapley", display_components=displayed,
        reconciliation_ratio=_ratio(raw_sum, delta), residual_value=residual,
        residual_type="rounding" if residual != 0.0 else "unknown",
        residual_explainable=True, evidence_ids=evidence_ids,
        residual_tolerance=allowed_error,
        residual_note="原始因子效应与交互项按乘积恒等式对账；展示项为 Shapley 分摊，有限精度残差单独保留。",
    )

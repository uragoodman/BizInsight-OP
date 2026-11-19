"""Resolve configured product factors from additive sufficient statistics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from app.analysis.attribution_methods.methods import AttributionDataError, _finite_number
from app.metrics.contracts import AttributionFactorSpec


def resolve_product_factor_values(
    factor_specs: Iterable[AttributionFactorSpec],
    period_totals: Mapping[str, tuple[Any, Any]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return baseline/current factor values from ``(baseline, current)`` totals.

    ``sum`` factors read one additive metric. ``ratio_of_sums`` factors divide
    additive numerator and denominator totals within each period. No groupwise
    averaging or implicit missing-value substitution is performed.
    """
    baseline: dict[str, float] = {}
    current: dict[str, float] = {}

    def pair(metric_id: str | None, factor_id: str) -> tuple[float, float]:
        if not metric_id or metric_id not in period_totals:
            raise AttributionDataError(
                f"factor {factor_id!r} is missing additive source {metric_id!r}"
            )
        before, after = period_totals[metric_id]
        return (
            _finite_number(before, f"{factor_id} baseline source"),
            _finite_number(after, f"{factor_id} current source"),
        )

    specs = list(factor_specs)
    if len(specs) != 2 or len({item.factor_id for item in specs}) != 2:
        raise AttributionDataError("product decomposition requires exactly two uniquely named factors")

    for spec in specs:
        if spec.kind == "sum":
            before, after = pair(spec.metric_id, spec.factor_id)
        elif spec.kind == "ratio_of_sums":
            before_num, after_num = pair(spec.numerator_metric, spec.factor_id)
            before_den, after_den = pair(spec.denominator_metric, spec.factor_id)
            if before_den <= 0 or after_den <= 0:
                raise AttributionDataError(f"factor {spec.factor_id!r} requires positive denominators")
            before, after = before_num / before_den, after_num / after_den
        else:  # Runtime guard for malformed/custom capability objects.
            raise AttributionDataError(f"unsupported product factor kind: {spec.kind!r}")
        before = _finite_number(before, f"{spec.factor_id} baseline factor")
        after = _finite_number(after, f"{spec.factor_id} current factor")
        baseline[spec.factor_id] = before
        current[spec.factor_id] = after
    return baseline, current

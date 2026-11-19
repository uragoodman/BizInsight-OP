"""Centralized period-pair contribution capability checks.

Period support is narrower than metric-level attribution support: declaring an
attribution method does not make it valid for every time comparison. Keep the
small runtime matrix shared by planner preflight and answer projection so an
unsupported request is stopped before SQL instead of failing after execution.
"""

from __future__ import annotations

from typing import Any


def registered_period_contribution_method(
    definition: Any,
    *,
    metric_id: str,
    dimension: str,
    scope: str,
    granularity: str | None,
) -> str | None:
    """Return the explicitly executable period method, or ``None``.

    ``additive_change_v1`` is limited to the validated platform freight/GMV
    category monthly paths. Proportional overlap and ratio/mix paths retain
    their existing method-specific constraints.
    """

    if scope not in definition.implementations:
        return None
    capability = definition.attribution.for_scope(scope)
    if dimension not in capability.dimensions:
        return None

    if (
        "proportional_overlap_v1" in capability.method_ids
        and metric_id in {"order_count", "customer_count", "repeat_customer_count"}
        and dimension in {"product_category", "seller", "payment_type"}
    ):
        return "proportional_overlap_v1"

    if (
        scope == "platform"
        and metric_id in {"freight_value", "gmv"}
        and dimension == "product_category"
        and granularity == "month"
        and "additive_change_v1" in capability.method_ids
    ):
        return "additive_change_v1"

    if (
        "symmetric_mix_within_v1" in capability.method_ids
        and definition.period_aggregation is not None
    ):
        return "symmetric_mix_within_v1"

    return None

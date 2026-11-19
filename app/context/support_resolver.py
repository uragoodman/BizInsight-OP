"""Unified dimension support resolution for the Semantic Compiler.

P1 replaces the scattered dimension checks with one deterministic resolver.
It deliberately keeps five independent checks instead of collapsing them into a
single boolean:

- ``schema_registered``: the dimension exists anywhere in the metric registry;
- ``capability_enabled``: the metric/scope declares the requested operation;
- ``scope_allowed``: the locked Explorer scope permits the dimension;
- ``metric_compatible``: the metric declares the dimension for this operation;
- ``user_permitted``: the authorization snapshot permits the dimension.

Ordinary queries read implementation metadata; contribution reads attribution
metadata.  That distinction is intentional: ``order_count`` can support
attribution on ``product_category`` while its ordinary seller implementation
does not declare a breakdown capability.  The resolver never invents a
capability and never substitutes another dimension.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

SupportStatus = Literal["ok", "missing", "denied", "unknown"]

_OPERATION_ALIASES = {
    "metric_value": "value",
    "metric_trend": "trend",
    "metric_compare": "compare",
    "metric_breakdown": "breakdown",
    "metric_contribution": "contribution",
}

# These dimensions can put one entity into more than one group.  For
# distinct-count metrics, group counts overlap and cannot be summed into an
# additive contribution.  The registry still declares the additive capability
# for historical reasons; this resolver is the execution gate.
_OVERLAPPING_CONTRIBUTION_DIMENSIONS = {"product_category", "seller", "payment_type"}
_DISTINCT_COUNT_METRICS = {
    "order_count",
    "customer_count",
    "repeat_customer_count",
}


def _is_distinct_overlap(
    definition: Any,
    dimension: str | None,
    scope_type: str,
) -> bool:
    if scope_type != "seller" or dimension not in _OVERLAPPING_CONTRIBUTION_DIMENSIONS:
        return False
    metric_id = str(getattr(definition, "metric_id", "") or "")
    aggregation = str(getattr(definition, "aggregation", "") or "")
    return metric_id in _DISTINCT_COUNT_METRICS or aggregation == "distinct_count"


@dataclass(frozen=True)
class SupportResult:
    metric_id: str | None
    dimension: str | None
    operation: str
    scope_type: str
    schema_registered: SupportStatus
    capability_enabled: SupportStatus
    scope_allowed: SupportStatus
    metric_compatible: SupportStatus
    user_permitted: SupportStatus
    reason_code: str | None = None
    allowed_values: tuple[str, ...] = ()

    @property
    def executable(self) -> bool:
        """Whether execution may proceed; unknown permission defers to MCP."""

        blocking = (
            self.schema_registered,
            self.capability_enabled,
            self.scope_allowed,
            self.metric_compatible,
        )
        if any(status != "ok" for status in blocking):
            return False
        return self.user_permitted in {"ok", "unknown"}

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["executable"] = self.executable
        return value


def resolve_support(
    *,
    metric_id: str | None,
    dimension: str | None,
    operation: str,
    scope: Any = None,
    authz: Any = None,
    registry: Any = None,
) -> SupportResult:
    """Resolve one metric/operation/dimension/scope tuple.

    ``scope`` accepts a mapping or a ControlledAnalysisScope-like model.
    ``authz`` accepts an AuthorizationContext-like object.  Missing inputs
    produce ``unknown`` rather than a guessed capability.
    """

    registry = registry or _registry()
    scope_data = _as_mapping(scope)
    scope_type = "seller" if scope_data.get("seller_id") else "platform"
    normalized_operation = _normalize_operation(operation)
    dimension_value = str(dimension).strip() if dimension else None

    all_dimensions: set[str] = set()
    schema_status: SupportStatus = "unknown"
    try:
        all_dimensions = _registry_dimensions(registry)
        schema_status = (
            "ok"
            if dimension_value is None or dimension_value in all_dimensions
            else "missing"
        )
    except Exception:  # noqa: BLE001 - registry failure fails closed
        schema_status = "unknown"

    capability_status: SupportStatus = "ok" if dimension_value is None else "unknown"
    compatibility_status: SupportStatus = "ok" if dimension_value is None else "unknown"
    definition = None
    if metric_id:
        try:
            definition = registry.get(metric_id)
        except Exception:  # noqa: BLE001 - unknown metric fails closed
            definition = None

    if metric_id and definition is None:
        capability_status = "unknown"
        compatibility_status = "unknown"
    elif metric_id and definition is not None:
        capability_status, compatibility_status = _metric_checks(
            definition,
            dimension=dimension_value,
            operation=normalized_operation,
            scope_type=scope_type,
            registry=registry,
        )
    elif not metric_id:
        # No metric to evaluate yet (for example a preference candidate
        # applied before metric resolution).  Schema and scope still apply.
        capability_status = "ok"
        compatibility_status = "ok"

    scope_status: SupportStatus = "ok"
    if dimension_value is not None and scope_data:
        allowed = {str(item) for item in (scope_data.get("allowed_dimensions") or [])}
        if allowed and dimension_value not in allowed:
            scope_status = "denied"

    permission_status: SupportStatus = "ok" if dimension_value is None else "unknown"
    if dimension_value is not None and authz is not None:
        allowed_dimensions = {
            str(item) for item in (getattr(authz, "allowed_dimensions", ()) or ())
        }
        permission_status = "ok" if dimension_value in allowed_dimensions else "denied"

    distinct_overlap = (
        normalized_operation == "contribution"
        and definition is not None
        and _is_distinct_overlap(definition, dimension_value, scope_type)
        and not _has_proportional_overlap_method(definition, scope_type)
    )
    reason_code = _reason_code(
        operation=normalized_operation,
        schema_status=schema_status,
        capability_status=capability_status,
        compatibility_status=compatibility_status,
        scope_status=scope_status,
        permission_status=permission_status,
        contribution_non_additive=(
            normalized_operation == "contribution"
            and definition is not None
            and str(getattr(definition, "aggregation", "")) != "sum"
            and not _has_symmetric_ratio_method(definition, scope_type)
        ),
        contribution_distinct=distinct_overlap,
    )
    allowed_values = _allowed_values(
        scope_data=scope_data,
        definition=definition,
        scope_type=scope_type,
        registry_dimensions=all_dimensions,
    )
    return SupportResult(
        metric_id=metric_id,
        dimension=dimension_value,
        operation=normalized_operation,
        scope_type=scope_type,
        schema_registered=schema_status,
        capability_enabled=capability_status,
        scope_allowed=scope_status,
        metric_compatible=compatibility_status,
        user_permitted=permission_status,
        reason_code=reason_code,
        allowed_values=allowed_values,
    )


def _metric_checks(
    definition: Any,
    *,
    dimension: str | None,
    operation: str,
    scope_type: str,
    registry: Any,
) -> tuple[SupportStatus, SupportStatus]:
    if operation == "contribution":
        try:
            attribution = definition.attribution.for_scope(scope_type)
            methods = {str(item) for item in (attribution.method_ids or ())}
            dimensions = {str(item) for item in (attribution.dimensions or ())}
        except Exception:  # noqa: BLE001 - missing attribution fails closed
            return "unknown", "unknown"
        if _is_distinct_overlap(definition, dimension, scope_type):
            # A distinct entity (for example an order) can belong to several
            # groups. Group deltas overlap and cannot be presented as additive
            # contribution without an explicit allocation method.
            if "proportional_overlap_v1" not in methods:
                return "missing", "missing"
        additive = "additive_change_v1" in methods
        if scope_type == "platform":
            additive = additive or "product_shapley_v1" in methods
        symmetric_ratio = bool(
            {"symmetric_ratio_v1", "symmetric_mix_within_v1"} & methods
        )
        proportional = "proportional_overlap_v1" in methods
        capability = (
            "ok"
            if (additive and str(getattr(definition, "aggregation", "")) == "sum")
            or symmetric_ratio
            or proportional
            else "missing"
        )
        if dimension is None:
            return capability, "ok"
        return capability, "ok" if dimension in dimensions else "missing"

    if scope_type == "seller" and dimension == "seller" and operation == "breakdown":
        # Seller users can request a degenerate one-group seller breakdown.
        # This does not inherit other platform grouping dimensions into the
        # seller implementation.
        try:
            platform_impl = definition.implementations.get("platform")
            allowed = (
                platform_impl is not None
                and "breakdown" in platform_impl.capabilities
                and registry.is_dimension_allowed(definition.metric_id, "seller", "platform")
            )
        except Exception:  # noqa: BLE001
            allowed = False
        return ("ok", "ok") if allowed else ("missing", "missing")

    try:
        implementation = definition.implementations.get(scope_type)
    except Exception:  # noqa: BLE001 - missing implementation fails closed
        implementation = None
    if implementation is None:
        return "missing", "missing"
    capabilities = {str(item) for item in (implementation.capabilities or ())}
    dimensions = {str(item) for item in (implementation.dimensions or ())}
    capability = "ok" if operation in capabilities else "missing"
    if dimension is None:
        return capability, "ok"
    # Derived time dimensions are executable when the implementation declares
    # breakdown, even though they are not listed as stored dimensions.
    derived_time_dimensions = {
        "month_position",
        "day_of_week",
        "hour_of_day",
        "quarter_of_year",
        "half_of_year",
    }
    if (
        dimension not in dimensions
        and dimension in derived_time_dimensions
        and "breakdown" in capabilities
    ):
        return capability, "ok"
    try:
        allowed = registry.is_dimension_allowed(definition.metric_id, dimension, scope_type)
    except Exception:  # noqa: BLE001
        allowed = dimension in dimensions
    return capability, "ok" if allowed else "missing"


def _reason_code(
    *,
    operation: str,
    schema_status: SupportStatus,
    capability_status: SupportStatus,
    compatibility_status: SupportStatus,
    scope_status: SupportStatus,
    permission_status: SupportStatus,
    contribution_non_additive: bool = False,
    contribution_distinct: bool = False,
) -> str | None:
    if schema_status == "missing":
        return "dimension_not_registered"
    if scope_status == "denied":
        return "dimension_outside_locked_scope"
    if operation == "contribution" and contribution_distinct:
        return "contribution_distinct_not_additive"
    if operation == "contribution" and contribution_non_additive:
        return "contribution_not_additive"
    if operation == "contribution" and capability_status == "missing":
        return "contribution_not_supported"
    if compatibility_status == "missing":
        return "dimension_not_supported_by_metric"
    if capability_status == "missing":
        return "capability_not_enabled"
    if permission_status == "denied":
        return "dimension_not_permitted"
    if "unknown" in {
        schema_status,
        capability_status,
        compatibility_status,
        scope_status,
    }:
        return "support_unknown"
    # Unknown permission defers to the MCP authorization boundary; it is not
    # a blocking reason at compile time.
    return None


def _has_symmetric_ratio_method(definition: Any, scope_type: str) -> bool:
    methods = _attribution_methods(definition, scope_type)
    return bool({"symmetric_ratio_v1", "symmetric_mix_within_v1"} & methods)


def _has_proportional_overlap_method(definition: Any, scope_type: str) -> bool:
    return "proportional_overlap_v1" in _attribution_methods(definition, scope_type)


def _attribution_methods(definition: Any, scope_type: str) -> set[str]:
    try:
        return {
            str(item) for item in (
                definition.attribution.for_scope(scope_type).method_ids or ()
            )
        }
    except Exception:  # noqa: BLE001 - missing attribution declares no method
        return set()


def _allowed_values(
    *,
    scope_data: dict[str, Any],
    definition: Any,
    scope_type: str,
    registry_dimensions: set[str],
) -> tuple[str, ...]:
    scoped = sorted(
        str(item) for item in (scope_data.get("allowed_dimensions") or []) if item
    )
    if scoped:
        return tuple(scoped)
    if definition is not None:
        try:
            implementation = definition.implementations.get(scope_type)
            if implementation is not None:
                return tuple(sorted(str(item) for item in implementation.dimensions))
        except Exception:  # noqa: BLE001 - best-effort display mirror
            pass
    return tuple(sorted(registry_dimensions))


def _registry_dimensions(registry: Any) -> set[str]:
    dimensions: set[str] = {
        str(item.dimension_id)
        for item in registry.dimensions()
        if registry.dimension_enabled(item.dimension_id)
    }
    for definition in registry.list():
        for implementation in definition.implementations.values():
            dimensions.update(
                str(item) for item in implementation.dimensions
                if registry.dimension_enabled(str(item))
            )
        for capability in definition.attribution.scopes.values():
            dimensions.update(str(item) for item in capability.dimensions)
    return dimensions


def _normalize_operation(operation: str) -> str:
    value = str(operation or "value").strip()
    return _OPERATION_ALIASES.get(value, value)


def _as_mapping(scope: Any) -> dict[str, Any]:
    if scope is None:
        return {}
    if isinstance(scope, dict):
        return scope
    model_dump = getattr(scope, "model_dump", None)
    if callable(model_dump):
        value = model_dump(mode="json")
        return value if isinstance(value, dict) else {}
    return {}


def _registry() -> Any:
    from app.metrics.registry import get_registry

    return get_registry()

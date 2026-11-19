"""Executable metric semantics — now backed by the config-driven registry.

``METRIC_REGISTRY`` and ``SELLER_METRIC_REGISTRY`` remain as frozen snapshots
for backward-compatible imports.  ``get_metric_definition()`` and
``list_metric_definitions()`` read from the live registry.

To be removed in Phase 4 after all consumers migrate to the new
``app.metrics.registry.MetricRegistry`` API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from app.metrics.registry import get_registry

Aggregation = Literal["sum", "weighted_ratio", "average", "distinct_count"]


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    aliases: tuple[str, ...]
    unit: str
    base_grain: Literal["order", "customer", "review", "day"]
    aggregation: Aggregation
    formula: str
    source_table: str
    source_columns: tuple[str, ...]
    allowed_dimensions: tuple[str, ...]
    direction: Literal["higher_is_better", "lower_is_better", "neutral"] = "neutral"
    owner: str = "analytics"
    version: str = "1.0"
    freshness_sla_hours: int = 24
    scope: Literal["platform", "seller"] = "platform"

    @property
    def lineage_ref(self) -> str:
        # Registry ``lineage_source`` identifies source relations (and can
        # contain more than one relation), not a column name.  Keep the
        # registry value intact instead of suffixing a guessed column.
        source = self.source_table
        if self.source_columns:
            source = ",".join(
                f"{self.source_table}.{column}" for column in self.source_columns
            )
        return f"{source} -> metric:{self.name}@{self.version}"


# ---------------------------------------------------------------------------
# Dimension exclusions (backward compat with old SQL code path)
# ---------------------------------------------------------------------------

_DIMENSION_EXCLUSIONS: dict[str, set[str]] = {
    "cancel_rate": {"seller_state", "product_category", "payment_type"},
    "freight_value": {"payment_type"},
    "avg_items_per_order": {"payment_type"},
    "freight_rate": {"payment_type"},
    # Domain metrics: no global breakdown SQL in daily_metrics.py
    "payment_value": {"customer_state", "customer_city", "seller_state", "product_category", "payment_type"},
    "installment_order_share": {"customer_state", "customer_city", "seller_state", "product_category", "payment_type"},
    "avg_approval_lead_hours": {"customer_state", "customer_city", "seller_state", "product_category", "payment_type"},
    "avg_dispatch_lead_hours": {"customer_state", "customer_city", "seller_state", "product_category", "payment_type"},
    "avg_delivery_transit_hours": {"customer_state", "customer_city", "seller_state", "product_category", "payment_type"},
}

_COMMON_DIMENSIONS = (
    "customer_state", "customer_city", "seller_state", "product_category", "payment_type"
)


def _allowed_dimensions(metric_id: str, registry_dimensions: tuple[str, ...]) -> tuple[str, ...]:
    """Return dimensions supported by the legacy semantic/SQL path.

    Seller is executed only through the controlled seller-scope query path;
    exposing it from these legacy definitions would make callers advertise a
    dimension that ``daily_metrics`` cannot safely execute without a canonical
    seller authorization scope.
    """
    excluded = _DIMENSION_EXCLUSIONS.get(metric_id, set())
    return tuple(d for d in registry_dimensions if d not in excluded and d != "seller")


# ---------------------------------------------------------------------------
# Convert registry MetricDefinition → legacy MetricDefinition
# ---------------------------------------------------------------------------

_DISPLAY_NAME_FALLBACK: dict[str, str] = {
    "gmv": "GMV", "order_count": "订单量", "avg_order_value": "客单价",
    "cancel_rate": "取消率", "freight_value": "运费总额",
    "review_score_avg": "平均评分", "delivery_delay_rate": "延迟送达率",
    "customer_count": "客户数", "repeat_customer_count": "复购客户数",
    "repeat_purchase_rate": "复购率", "avg_items_per_order": "平均每单件数",
    "avg_delivery_lead_time": "平均送达时长",
    "on_time_delivery_rate": "准时送达率", "bad_review_rate": "差评率",
    "freight_rate": "运费率",
    "payment_value": "支付金额", "installment_order_share": "分期支付金额占比",
    "avg_approval_lead_hours": "平均审批时长",
    "avg_dispatch_lead_hours": "平均发货时长",
    "avg_delivery_transit_hours": "平均配送时长",
    "sales_gmv": "销售额",
}

# The tenant mart predates the registry and stores seller GMV as ``sales_gmv``.
# Keep that storage-facing name readable by reports while resolving semantics
# through the canonical, registry-owned ``gmv`` definition.
_LEGACY_METRIC_ALIASES: dict[str, str] = {"sales_gmv": "gmv"}


def _from_registry(name: str, scope: str = "platform") -> MetricDefinition:
    """Build a legacy MetricDefinition from the live registry."""
    reg = get_registry()
    try:
        canonical = reg.resolve(_LEGACY_METRIC_ALIASES.get(name, name))
    except Exception:
        raise ValueError(f"Unsupported metric: {name}") from None

    new_defn = reg.get(canonical)
    impl = new_defn.implementations.get(scope)  # type: ignore[arg-type]
    if impl is None:
        raise ValueError(f"{canonical} has no {scope} implementation")

    source_table = impl.lineage_source if impl else ""
    # The registry currently provides relation-level lineage, not source
    # columns.  Inferring a column from the final token duplicated the
    # relation in lineage_ref (for example ``view.view``).
    source_columns: tuple[str, ...] = ()

    # Build formula from available metadata
    agg_map = {"sum": "sum", "count": "count", "average": "avg", "distinct_count": "distinct"}
    raw_agg = agg_map.get(new_defn.aggregation, new_defn.aggregation)
    formula = new_defn.aggregation
    if impl and impl.lineage_source:
        formula = f"{raw_agg}({impl.lineage_source})"

    dimensions = _allowed_dimensions(canonical, tuple(impl.dimensions)) if impl else ()

    return MetricDefinition(
        name=canonical,
        aliases=new_defn.aliases,
        unit=new_defn.unit,
        base_grain=new_defn.base_grain,
        aggregation=new_defn.aggregation,
        formula=formula,
        source_table=impl.lineage_source if impl else "",
        source_columns=source_columns,
        allowed_dimensions=dimensions,
        direction=new_defn.direction,
        version=new_defn.version,
        scope=scope,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Backward-compatible frozen snapshots
# ---------------------------------------------------------------------------

METRIC_REGISTRY: dict[str, MetricDefinition] = {}
SELLER_METRIC_REGISTRY: dict[str, MetricDefinition] = {}


def _warm_registries() -> None:
    """Lazily populate the backward-compat snapshots on first access."""
    if METRIC_REGISTRY:
        return
    reg = get_registry()
    for defn in reg.list():
        if "platform" in (defn.implementations or {}):
            METRIC_REGISTRY[defn.metric_id] = _from_registry(defn.metric_id, "platform")
    # Seller snapshot
    for metric_id in ["sales_gmv", "order_count", "customer_count",
                       "avg_order_value", "repeat_purchase_rate"]:
        try:
            SELLER_METRIC_REGISTRY[metric_id] = _from_registry(metric_id, "seller")
        except ValueError:
            pass


# Trigger warm on module import
_warm_registries()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_metric_definition(metric_name: str, scope: Literal["platform", "seller"] = "platform") -> MetricDefinition:
    """Look up a metric definition.  Delegates to the registry."""
    try:
        return _from_registry(metric_name, scope)
    except ValueError:
        raise ValueError(f"Unsupported metric: {metric_name}")


def list_metric_definitions(scope: Literal["platform", "seller"] = "platform") -> list[dict[str, object]]:
    """List all metric definitions as dicts.  Delegates to the registry."""
    reg = get_registry()
    result = []
    for defn in reg.list():
        try:
            d = _from_registry(defn.metric_id, scope)
        except ValueError:
            # Seller-only metrics not in platform scope — skip unless explorer-visible
            if scope == "platform" and "explorer" in defn.surfaces:
                # Include for Explorer display even without platform impl
                d = _from_registry(defn.metric_id, "seller")
            else:
                continue
        entry = asdict(d) | {"lineage_ref": d.lineage_ref}
        # New-style fields for frontend consumption
        registry_defn = reg.get(defn.metric_id)
        entry["metric_name"] = d.name
        entry["display_name"] = registry_defn.display_name
        entry["display_format"] = registry_defn.display_format
        impl = registry_defn.implementations.get(scope)  # type: ignore[arg-type]
        entry["capabilities"] = list(impl.capabilities) if impl else []
        entry["surfaces"] = list(registry_defn.surfaces)
        entry["attribution"] = {
            "support_by_scope": {
                scope_name: {
                    "status": (
                        "supported" if registry_defn.attribution.for_scope(scope_name).method_ids
                        else "unsupported" if scope_name in registry_defn.implementations
                        else "scope_unavailable"
                    ),
                    "methods": list(registry_defn.attribution.for_scope(scope_name).method_ids),
                    "dimensions": list(registry_defn.attribution.for_scope(scope_name).dimensions),
                    "required_data": {
                        method: list(fields)
                        for method, fields in registry_defn.attribution.for_scope(scope_name).required_data.items()
                    },
                    "factor_specs": {
                        method: [
                            {
                                "factor_id": factor.factor_id,
                                "kind": factor.kind,
                                "metric_id": factor.metric_id,
                                "numerator_metric": factor.numerator_metric,
                                "denominator_metric": factor.denominator_metric,
                            }
                            for factor in factors
                        ]
                        for method, factors in registry_defn.attribution.for_scope(scope_name).factor_specs.items()
                    },
                    "limitation": registry_defn.attribution.limitation,
                }
                for scope_name in ("platform", "seller")
            },
            "limitation": registry_defn.attribution.limitation,
        }
        result.append(entry)
    return result

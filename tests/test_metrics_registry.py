"""Phase 1 tests: registry loading, validation, and query API."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from app.metrics.contracts import MetricDefinition
from app.metrics.exceptions import (
    DimensionNotAllowed,
    MetricNotFound,
    RegistryValidationError,
    ScopeNotAllowed,
)
from app.metrics.handler_catalog import HANDLER_REGISTRY, get_handler
from app.metrics.registry import MetricRegistry, get_registry

YAML_DIR = str(
    Path(__file__).resolve().parent.parent / "config" / "metrics"
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def registry() -> MetricRegistry:
    reg = MetricRegistry(yaml_dir=YAML_DIR)
    reg.load()
    return reg


# =============================================================================
# Loading
# =============================================================================


def test_registry_loads_all_metrics(registry: MetricRegistry) -> None:
    """All YAML files are parsed; platform + seller metrics are present."""
    all_metrics = registry.list()
    ids = {m.metric_id for m in all_metrics}
    # Core 5
    assert "gmv" in ids
    assert "order_count" in ids
    assert "avg_order_value" in ids
    assert "customer_count" in ids
    assert "freight_value" in ids
    # Payment + fulfillment
    assert "payment_value" in ids
    assert "installment_order_share" in ids
    assert "on_time_delivery_rate" in ids
    assert "avg_approval_lead_hours" in ids
    assert "avg_dispatch_lead_hours" in ids
    assert "avg_delivery_transit_hours" in ids
    # Rest
    assert "cancel_rate" in ids
    assert "review_score_avg" in ids
    assert "delivery_delay_rate" in ids
    assert "repeat_customer_count" in ids
    assert "repeat_purchase_rate" in ids
    assert "avg_items_per_order" in ids
    assert "avg_delivery_lead_time" in ids
    assert "bad_review_rate" in ids
    assert "freight_rate" in ids
    # Count
    assert len(all_metrics) >= 20, f"expected ≥20 platform metrics, got {len(all_metrics)}"


def test_registry_rejects_unregistered_product_factor_source(tmp_path: Path) -> None:
    import shutil

    copied_metrics = tmp_path / "metrics"
    shutil.copytree(YAML_DIR, copied_metrics)
    attribution_path = copied_metrics / "attribution.yaml"
    catalog = yaml.safe_load(attribution_path.read_text(encoding="utf-8"))
    catalog["metrics"]["gmv"]["scopes"]["platform"]["factor_specs"][
        "product_shapley_v1"
    ][0]["metric_id"] = "unregistered_source_metric"
    attribution_path.write_text(
        yaml.safe_dump(catalog, allow_unicode=True, sort_keys=False), encoding="utf-8",
    )

    registry = MetricRegistry(yaml_dir=str(copied_metrics))
    with pytest.raises(RegistryValidationError, match="factor source.*is not registered"):
        registry.load()


def test_registry_rejects_required_data_that_does_not_match_method_contract(tmp_path: Path) -> None:
    import shutil

    copied_metrics = tmp_path / "metrics"
    shutil.copytree(YAML_DIR, copied_metrics)
    attribution_path = copied_metrics / "attribution.yaml"
    catalog = yaml.safe_load(attribution_path.read_text(encoding="utf-8"))
    catalog["metrics"]["gmv"]["scopes"]["platform"]["required_data"][
        "additive_change_v1"
    ] = ["baseline_total", "current_total", "baseline_group_totals", "current_group_total"]
    attribution_path.write_text(
        yaml.safe_dump(catalog, allow_unicode=True, sort_keys=False), encoding="utf-8",
    )

    registry = MetricRegistry(yaml_dir=str(copied_metrics))
    with pytest.raises(RegistryValidationError, match="does not match the deterministic method contract"):
        registry.load()


def test_seller_metrics_have_seller_implementation(registry: MetricRegistry) -> None:
    """Seller-scoped implementations exist for seller-accessible metrics."""
    seller_metrics = registry.list(scope="seller")
    seller_ids = {m.metric_id for m in seller_metrics}
    assert "gmv" in seller_ids
    assert "order_count" in seller_ids
    assert "customer_count" in seller_ids
    assert "avg_order_value" in seller_ids
    assert "repeat_purchase_rate" in seller_ids
    assert "payment_value" in seller_ids
    assert "installment_order_share" in seller_ids
    assert "on_time_delivery_rate" in seller_ids
    assert "avg_approval_lead_hours" in seller_ids
    assert "avg_dispatch_lead_hours" in seller_ids
    assert "avg_delivery_transit_hours" in seller_ids


# =============================================================================
# Validation
# =============================================================================


def test_validate_succeeds(registry: MetricRegistry) -> None:
    """All handlers, docs, and capabilities pass validation."""
    # Should not raise
    registry.validate()


def test_validate_duplicate_metric_id_raises() -> None:
    """Duplicate metric_id in YAML must fail."""
    reg = MetricRegistry(yaml_dir=YAML_DIR)
    reg.load()
    # Inject duplicate into internal state
    reg._metrics["gmv_dup_test"] = reg._metrics["gmv"]
    # Can't inject into YAML, so verify that load() catches it
    bad_dir = Path(__file__).parent / "fixtures" / "bad_registry"
    bad_dir.mkdir(parents=True, exist_ok=True)
    duplicate_yaml = bad_dir / "duplicate.yaml"
    duplicate_yaml.write_text(
        yaml.dump({
            "metrics": [
                {
                    "metric_id": "test_dup",
                    "display_name": "First",
                    "aliases": ["first"],
                    "unit": "count",
                    "display_format": "number",
                    "base_grain": "order",
                    "aggregation": "sum",
                    "direction": "neutral",
                    "version": "1.0",
                    "documentation": "",
                    "display_order": 99,
                    "surfaces": ["chat"],
                    "implementations": {
                        "platform": {
                            "handler_key": "sales.gmv",
                            "capabilities": ["value"],
                            "dimensions": [],
                            "lineage_source": "test",
                        }
                    },
                },
                {
                    "metric_id": "test_dup",
                    "display_name": "Duplicate",
                    "aliases": ["duplicate"],
                    "unit": "count",
                    "display_format": "number",
                    "base_grain": "order",
                    "aggregation": "sum",
                    "direction": "neutral",
                    "version": "1.0",
                    "documentation": "",
                    "display_order": 99,
                    "surfaces": ["chat"],
                    "implementations": {
                        "platform": {
                            "handler_key": "sales.gmv",
                            "capabilities": ["value"],
                            "dimensions": [],
                            "lineage_source": "test",
                        }
                    },
                },
            ],
        }),
        encoding="utf-8",
    )
    try:
        dup_reg = MetricRegistry(yaml_dir=str(bad_dir))
        with pytest.raises(RegistryValidationError, match="Duplicate metric_id"):
            dup_reg.load()
    finally:
        import shutil
        shutil.rmtree(bad_dir, ignore_errors=True)


def test_duplicate_alias_raises() -> None:
    """Two metrics sharing an alias must fail validation."""
    bad_dir = Path(__file__).parent / "fixtures" / "bad_alias"
    bad_dir.mkdir(parents=True, exist_ok=True)
    alias_yaml = bad_dir / "alias_dup.yaml"
    alias_yaml.write_text(
        yaml.dump({
            "metrics": [
                {
                    "metric_id": "metric_a",
                    "display_name": "A",
                    "aliases": ["shared_alias"],
                    "unit": "count", "display_format": "number",
                    "base_grain": "order", "aggregation": "sum",
                    "direction": "neutral", "version": "1.0",
                    "documentation": "", "display_order": 99,
                    "surfaces": ["chat"],
                    "implementations": {
                        "platform": {
                            "handler_key": "sales.gmv", "capabilities": ["value"],
                            "dimensions": [], "lineage_source": "t",
                        }
                    },
                },
                {
                    "metric_id": "metric_b",
                    "display_name": "B",
                    "aliases": ["shared_alias"],
                    "unit": "count", "display_format": "number",
                    "base_grain": "order", "aggregation": "sum",
                    "direction": "neutral", "version": "1.0",
                    "documentation": "", "display_order": 99,
                    "surfaces": ["chat"],
                    "implementations": {
                        "platform": {
                            "handler_key": "sales.gmv", "capabilities": ["value"],
                            "dimensions": [], "lineage_source": "t",
                        }
                    },
                },
            ],
        }),
        encoding="utf-8",
    )
    try:
        dup_reg = MetricRegistry(yaml_dir=str(bad_dir))
        with pytest.raises(RegistryValidationError, match="Duplicate alias"):
            dup_reg.load()
    finally:
        import shutil
        shutil.rmtree(bad_dir, ignore_errors=True)


def test_unknown_handler_key_raises(registry: MetricRegistry) -> None:
    """Metric with handler_key not in HANDLER_REGISTRY fails validation."""
    reg = MetricRegistry(yaml_dir=YAML_DIR)
    reg.load()
    # Tamper a handler_key
    gmv = reg._metrics["gmv"]
    bad_impl = gmv.implementations["platform"]
    reg._metrics["gmv"] = MetricDefinition(
        metric_id=gmv.metric_id,
        display_name=gmv.display_name,
        aliases=gmv.aliases,
        unit=gmv.unit,
        display_format=gmv.display_format,
        base_grain=gmv.base_grain,
        aggregation=gmv.aggregation,
        direction=gmv.direction,
        version=gmv.version,
        documentation=gmv.documentation,
        display_order=gmv.display_order,
        surfaces=gmv.surfaces,
        implementations={
            "platform": type(bad_impl)(
                handler_key="nonexistent.handler",
                capabilities=bad_impl.capabilities,
                dimensions=bad_impl.dimensions,
                lineage_source=bad_impl.lineage_source,
                lineage_description=bad_impl.lineage_description,
            ),
        },
    )
    with pytest.raises(RegistryValidationError, match="not in HANDLER_REGISTRY"):
        reg.validate()


def test_missing_documentation_raises(registry: MetricRegistry) -> None:
    """Metric referencing a nonexistent doc path fails validation."""
    reg = MetricRegistry(yaml_dir=YAML_DIR)
    reg.load()
    # Tamper a doc path
    gmv = reg._metrics["gmv"]
    reg._metrics["gmv"] = MetricDefinition(
        metric_id=gmv.metric_id,
        display_name=gmv.display_name,
        aliases=gmv.aliases,
        unit=gmv.unit,
        display_format=gmv.display_format,
        base_grain=gmv.base_grain,
        aggregation=gmv.aggregation,
        direction=gmv.direction,
        version=gmv.version,
        documentation="docs/knowledge/metrics/NONEXISTENT_FILE.md",
        display_order=gmv.display_order,
        surfaces=gmv.surfaces,
        implementations=gmv.implementations,
    )
    with pytest.raises(RegistryValidationError, match="doc not found"):
        reg.validate()


def test_empty_documentation_raises(registry: MetricRegistry) -> None:
    """Every registered metric must point to a knowledge document."""
    reg = MetricRegistry(yaml_dir=YAML_DIR)
    reg.load()
    reg._metrics["gmv"] = replace(reg._metrics["gmv"], documentation="")
    with pytest.raises(RegistryValidationError, match="documentation is required"):
        reg.validate()


def test_invalid_surface_raises(registry: MetricRegistry) -> None:
    """A metric cannot silently advertise an unsupported product surface."""
    reg = MetricRegistry(yaml_dir=YAML_DIR)
    reg.load()
    reg._metrics["gmv"] = replace(reg._metrics["gmv"], surfaces=("chat", "unknown"))
    with pytest.raises(RegistryValidationError, match="invalid surface"):
        reg.validate()


def test_breakdown_without_dimensions_raises() -> None:
    """Metric declaring breakdown capability but no dimensions must fail."""
    bad_dir = Path(__file__).parent / "fixtures" / "bad_breakdown"
    bad_dir.mkdir(parents=True, exist_ok=True)
    bad_yaml = bad_dir / "bad.yaml"
    bad_yaml.write_text(
        yaml.dump({
            "metrics": [
                {
                    "metric_id": "test_bad_breakdown",
                    "display_name": "Bad",
                    "aliases": ["bad"],
                    "unit": "count",
                    "display_format": "number",
                    "base_grain": "order",
                    "aggregation": "sum",
                    "direction": "neutral",
                    "version": "1.0",
                    "documentation": "",
                    "display_order": 99,
                    "surfaces": ["chat"],
                    "implementations": {
                        "platform": {
                            "handler_key": "sales.gmv",
                            "capabilities": ["value", "breakdown"],
                            "dimensions": [],
                            "lineage_source": "test",
                        }
                    },
                },
            ],
        }),
        encoding="utf-8",
    )
    try:
        reg = MetricRegistry(yaml_dir=str(bad_dir))
        reg.load()
        with pytest.raises(RegistryValidationError, match="breakdown capability requires"):
            reg.validate()
    finally:
        import shutil
        shutil.rmtree(bad_dir, ignore_errors=True)


# =============================================================================
# Alias resolution
# =============================================================================


def test_resolve_by_metric_id(registry: MetricRegistry) -> None:
    assert registry.resolve("gmv") == "gmv"


@pytest.mark.parametrize("alias,expected", [
    ("GMV", "gmv"),
    ("成交总额", "gmv"),
    ("客单价", "avg_order_value"),
    ("AOV", "avg_order_value"),
    ("准时送达率", "on_time_delivery_rate"),
    ("支付金额", "payment_value"),
])
def test_resolve_by_alias(
    registry: MetricRegistry, alias: str, expected: str,
) -> None:
    assert registry.resolve(alias) == expected


@pytest.mark.parametrize("bad_name", [
    "nonexistent_metric",
    "foobar",
    "",
])
def test_resolve_unknown_raises(
    registry: MetricRegistry, bad_name: str,
) -> None:
    with pytest.raises(MetricNotFound):
        registry.resolve(bad_name)


# =============================================================================
# get / list
# =============================================================================


def test_get_returns_definition(registry: MetricRegistry) -> None:
    defn = registry.get("gmv")
    assert isinstance(defn, MetricDefinition)
    assert defn.metric_id == "gmv"
    assert defn.display_name == "成交总额"
    assert defn.unit == "BRL"
    assert defn.display_format == "currency"


def test_period_aggregation_is_registry_owned_and_unit_checked(registry: MetricRegistry) -> None:
    ratio = registry.get_period_aggregation("avg_order_value")
    assert ratio is not None
    assert ratio.numerator_metric == "gmv"
    assert ratio.denominator_metric == "order_count"
    assert ratio.multiplier == 1.0


def test_get_unknown_raises(registry: MetricRegistry) -> None:
    with pytest.raises(MetricNotFound, match="Unknown metric"):
        registry.get("not_a_metric")


def test_list_scope_platform(registry: MetricRegistry) -> None:
    metrics = registry.list(scope="platform")
    ids = {m.metric_id for m in metrics}
    assert "gmv" in ids
    assert "cancel_rate" in ids


def test_list_scope_seller(registry: MetricRegistry) -> None:
    metrics = registry.list(scope="seller")
    ids = {m.metric_id for m in metrics}
    assert "gmv" in ids
    assert "cancel_rate" not in ids  # not in seller scope


# =============================================================================
# Capabilities & dimensions
# =============================================================================


def test_supports_value(registry: MetricRegistry) -> None:
    assert registry.supports("gmv", "value") is True


def test_supports_breakdown_on_payment_value(registry: MetricRegistry) -> None:
    """payment_value supports breakdown in platform scope."""
    assert registry.supports("payment_value", "breakdown") is True


def test_does_not_support_breakdown_on_cancel_rate(registry: MetricRegistry) -> None:
    assert registry.supports("cancel_rate", "breakdown") is False


def test_supports_unknown_capability(registry: MetricRegistry) -> None:
    assert registry.supports("gmv", "fly") is False


def test_is_dimension_allowed(registry: MetricRegistry) -> None:
    assert registry.is_dimension_allowed("gmv", "customer_state") is True
    assert registry.is_dimension_allowed("gmv", "payment_type") is True


def test_is_dimension_not_allowed(registry: MetricRegistry) -> None:
    assert registry.is_dimension_allowed("gmv", "hour_of_day") is False


def test_is_dimension_raises_for_wrong_scope(registry: MetricRegistry) -> None:
    """cancel_rate has no seller implementation → ScopeNotAllowed."""
    with pytest.raises(ScopeNotAllowed):
        registry.is_dimension_allowed("cancel_rate", "customer_state", scope="seller")


# =============================================================================
# Handler catalog
# =============================================================================


def test_all_handler_keys_registered(registry: MetricRegistry) -> None:
    """Every handler_key referenced in YAML exists in HANDLER_REGISTRY."""
    for defn in registry.list():
        for scope_key in ("platform", "seller"):
            impl = defn.implementations.get(scope_key)
            if impl is None:
                continue
            # Should not raise
            fn = get_handler(impl.handler_key)
            assert callable(fn), f"{impl.handler_key} is not callable"


def test_get_handler_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_handler("does.not.exist")

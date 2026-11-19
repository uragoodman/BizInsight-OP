"""P1 tests for the unified dimension support resolver."""

from __future__ import annotations

from types import SimpleNamespace

from app.context.support_resolver import resolve_support

SELLER_SCOPE = {
    "seller_id": "seller-1",
    "default_start_date": "2018-08-01",
    "default_end_date": "2018-08-31",
    "metric_name": "gmv",
    "granularity": "day",
    "allowed_metrics": ["gmv", "order_count", "freight_value"],
    "allowed_dimensions": ["product_category", "payment_type", "month_position"],
}


def _resolve(
    metric: str, dimension: str, operation: str, *, scope=SELLER_SCOPE, authz=None
):
    return resolve_support(
        metric_id=metric,
        dimension=dimension,
        operation=operation,
        scope=scope,
        authz=authz,
    )


def test_ordinary_breakdown_uses_implementation_not_attribution() -> None:
    result = _resolve("order_count", "product_category", "breakdown")
    assert result.schema_registered == "ok"
    assert result.scope_allowed == "ok"
    assert result.metric_compatible == "missing"
    assert result.capability_enabled == "missing"
    assert result.executable is False
    assert result.reason_code == "dimension_not_supported_by_metric"


def test_contribution_uses_attribution_not_ordinary_implementation() -> None:
    product_category = _resolve("order_count", "product_category", "contribution")
    assert product_category.schema_registered == "ok"
    assert product_category.capability_enabled == "ok"
    assert product_category.metric_compatible == "ok"
    assert product_category.scope_allowed == "ok"
    assert product_category.user_permitted == "unknown"
    assert product_category.executable is True
    assert product_category.reason_code is None

    month_position = _resolve("order_count", "month_position", "contribution")
    assert month_position.capability_enabled == "ok"
    assert month_position.metric_compatible == "ok"
    assert month_position.executable is True
    assert month_position.reason_code is None


def test_gmv_contribution_is_supported_while_ordinary_breakdown_is_not() -> None:
    breakdown = _resolve("gmv", "product_category", "breakdown")
    contribution = _resolve("gmv", "product_category", "contribution")
    assert breakdown.executable is False
    assert breakdown.reason_code == "dimension_not_supported_by_metric"
    assert contribution.executable is True
    assert contribution.reason_code is None


def test_ratio_metric_contribution_uses_symmetric_ratio_method() -> None:
    result = _resolve("freight_rate", "month_position", "contribution", scope=None)
    assert result.capability_enabled == "ok"
    assert result.metric_compatible == "ok"
    assert result.executable is True
    assert result.reason_code is None


def test_non_additive_metric_cannot_use_contribution() -> None:
    result = _resolve("avg_order_value", "product_category", "contribution")
    assert result.reason_code == "contribution_not_additive"
    assert result.executable is False


def test_unregistered_seller_dimension_fails_closed() -> None:
    result = _resolve("gmv", "seller", "breakdown")
    assert result.schema_registered == "missing"
    assert result.executable is False
    assert result.reason_code == "dimension_not_registered"


def test_out_of_scope_dimension_is_denied_before_metric_compatibility() -> None:
    result = _resolve("gmv", "customer_city", "breakdown")
    assert result.schema_registered == "ok"
    assert result.scope_allowed == "denied"
    assert result.reason_code == "dimension_outside_locked_scope"


def test_authorization_denial_is_reported() -> None:
    authz = SimpleNamespace(allowed_dimensions=("product_category",))
    denied = _resolve("gmv", "month_position", "contribution", authz=authz)
    allowed = _resolve("gmv", "product_category", "contribution", authz=authz)
    assert denied.user_permitted == "denied"
    assert denied.reason_code == "dimension_not_permitted"
    assert allowed.user_permitted == "ok"


def test_unknown_metric_fails_closed() -> None:
    result = _resolve("not_a_registered_metric", "product_category", "breakdown")
    assert result.capability_enabled == "unknown"
    assert result.metric_compatible == "unknown"
    assert result.executable is False
    assert result.reason_code == "support_unknown"


def test_platform_scope_uses_platform_implementation() -> None:
    result = _resolve("gmv", "product_category", "breakdown", scope=None)
    assert result.scope_type == "platform"
    assert result.capability_enabled == "ok"
    assert result.metric_compatible == "ok"
    assert result.executable is True


def test_resolver_exposes_five_independent_statuses() -> None:
    result = _resolve("gmv", "product_category", "breakdown")
    payload = result.as_dict()
    for key in (
        "schema_registered",
        "capability_enabled",
        "scope_allowed",
        "metric_compatible",
        "user_permitted",
        "reason_code",
        "allowed_values",
        "executable",
    ):
        assert key in payload
    assert "product_category" in payload["allowed_values"]

"""Authorization and privacy boundary tests for exact multi-axis queries."""

from __future__ import annotations

import pytest

from app.auth.context import AuthorizationDenied, build_authorization_context
from app.metrics.authorization import authorize_metric
from app.mcp import client as mcp_client
from app.mcp.schemas import MetricQueryPlan
from app.tenancy import TenantContext, activate_tenant_context


PLATFORM_ADMIN = TenantContext(
    tenant_id="platform",
    user_id="multi-dimension-auth-test",
    role="admin",
    scope_type="platform",
)
SCOPE_DATA = {
    "tenant_id": PLATFORM_ADMIN.tenant_id,
    "user_id": PLATFORM_ADMIN.user_id,
    "permissions": PLATFORM_ADMIN.role,
    "scope_type": PLATFORM_ADMIN.scope_type,
}


def _authorized_gmv():
    return authorize_metric(
        "gmv",
        provenance_type="exact",
        provenance_id="test-exact-gmv",
        scope="platform",
        scope_data=SCOPE_DATA,
    )


def _multi_plan(dimensions: list[str]) -> MetricQueryPlan:
    return MetricQueryPlan(
        query_type="metric_breakdown",
        metric_name="gmv",
        start_date="2017-10-01",
        end_date="2017-10-31",
        dimensions=dimensions,
        tool_name="query_metric_by_dimensions",
        authorized_metric=_authorized_gmv(),
    )


def _use_static_tool_policy(monkeypatch) -> None:
    # Keep the test on the checked-in policy file and avoid any database read.
    monkeypatch.setattr("app.security.abac.database_rules", lambda *_args, **_kwargs: None)


def test_multi_dimension_grant_is_separate_and_bounded_by_query_axes(monkeypatch):
    _use_static_tool_policy(monkeypatch)
    authz = build_authorization_context(PLATFORM_ADMIN, metric_scope_type="platform")

    # The old single-dimension/contribution boundary remains narrow.
    assert "customer_state" not in authz.allowed_dimensions
    with pytest.raises(AuthorizationDenied):
        authz.require_dimensions(["customer_state"])

    # The exact aggregate tool has its own grant, bounded by its SQL capability.
    assert "customer_state" in authz.allowed_multi_dimension_dimensions
    assert "product_category" in authz.allowed_multi_dimension_dimensions
    assert "customer_city" not in authz.allowed_multi_dimension_dimensions
    assert "seller_state" not in authz.allowed_multi_dimension_dimensions
    assert "payment_type" not in authz.allowed_multi_dimension_dimensions
    assert "query_metric_by_dimensions" in authz.allowed_tools
    authz.require_multi_dimension_dimensions(["customer_state", "product_category"])


def test_multi_dimension_dispatch_allows_authorized_pair_and_suppresses_small_cells(monkeypatch):
    _use_static_tool_policy(monkeypatch)
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_tool(tool_name, arguments, **_kwargs):
        calls.append((tool_name, arguments))
        return {
            "status": "ok",
            "metric_name": "gmv",
            "dimensions": ["customer_state", "product_category"],
            "items": [
                {
                    "dimensions": {"customer_state": "CA", "product_category": "beauty"},
                    "value": 10.0,
                    "item_rows": 2,
                    "order_count": 2,
                    "customer_count": 4,
                },
                {
                    "dimensions": {"customer_state": "SP", "product_category": "health"},
                    "value": 30.0,
                    "item_rows": 8,
                    "order_count": 7,
                    "customer_count": 5,
                },
            ],
        }

    monkeypatch.setattr(mcp_client, "call_local_mcp_tool", fake_tool)
    monkeypatch.setattr(mcp_client, "resolve_expression", lambda *_args, **_kwargs: None)
    with activate_tenant_context(PLATFORM_ADMIN):
        response = mcp_client.query_business_data(
            "metric_query",
            "按客户州和商品品类交叉汇总 GMV",
            plan=_multi_plan(["customer_state", "product_category"]),
            request_id="multi-axis-authorized",
        )

    assert len(calls) == 1
    assert calls[0][0] == "query_metric_by_dimensions"
    assert calls[0][1]["dimensions"] == ["customer_state", "product_category"]
    result = response["result"]["multi_dimension_breakdown_result"]
    assert result["status"] == "ok"
    assert len(result["items"]) == 1
    assert result["items"][0]["dimensions"]["customer_state"] == "SP"
    assert result["suppressed_cell_count"] == 1
    assert "sample_size" not in result["items"][0]


def test_multi_dimension_dispatch_denies_axis_outside_authorization_before_tool(monkeypatch):
    _use_static_tool_policy(monkeypatch)

    def unexpected_tool(*_args, **_kwargs):
        raise AssertionError("unauthorized multidimensional plan reached the tool")

    monkeypatch.setattr(mcp_client, "call_local_mcp_tool", unexpected_tool)
    monkeypatch.setattr(mcp_client, "resolve_expression", lambda *_args, **_kwargs: None)
    with activate_tenant_context(PLATFORM_ADMIN):
        response = mcp_client.query_business_data(
            "metric_query",
            "按客户城市和商品品类交叉汇总 GMV",
            plan=_multi_plan(["customer_city", "product_category"]),
            request_id="multi-axis-unauthorized",
        )

    assert response["result"]["status"] == "unsupported"
    assert response["tool_name"] is None
    assert response["authorization_snapshot"]["decision"] == "denied"
    assert response["authorization_snapshot"]["execution_attempted"] is False

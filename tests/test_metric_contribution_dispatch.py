"""Fail-closed routing tests for point and period metric contributions.

All transport, database, and external network boundaries are blocked. The
tests exercise the real plan builder, query dispatcher, tool permission gate,
and registry operation selection without reading business data.
"""

from __future__ import annotations

import socket
import ipaddress
from datetime import date

import pytest
from sqlalchemy.engine import Engine

from app.agents.data_query_agent import build_query_plan
from app.analysis.contribution import calculate_change_contribution
from app.metrics.authorization import authorize_metric
from app.mcp import client as mcp_client
from app.mcp.schemas import MetricPlanningDecision, MetricQueryPlan
from app.tenancy import TenantContext, activate_tenant_context


TENANT = TenantContext(
    tenant_id="platform", user_id="contribution-dispatch-test",
    role="admin", scope_type="platform",
)
SCOPE_DATA = {
    "tenant_id": TENANT.tenant_id,
    "user_id": TENANT.user_id,
    "permissions": TENANT.role,
    "scope_type": TENANT.scope_type,
}


@pytest.fixture(autouse=True)
def block_network_and_database(monkeypatch):
    """A regression must fail if it attempts network or SQL access."""

    original_connect = socket.socket.connect
    original_create_connection = socket.create_connection

    def _is_loopback(address):
        host = address[0] if isinstance(address, tuple) else address
        if host == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except (TypeError, ValueError):
            return False

    def blocked_connect(sock, address):
        if _is_loopback(address):
            return original_connect(sock, address)
        raise AssertionError("external network access is blocked in contribution dispatch tests")

    def blocked_create_connection(address, *args, **kwargs):
        if _is_loopback(address):
            return original_create_connection(address, *args, **kwargs)
        raise AssertionError("external network access is blocked in contribution dispatch tests")

    def blocked_database(*_args, **_kwargs):
        raise AssertionError("database access is blocked in contribution dispatch tests")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)
    monkeypatch.setattr(socket, "create_connection", blocked_create_connection)
    monkeypatch.setattr(Engine, "connect", blocked_database)
    with activate_tenant_context(TENANT):
        yield


def _authorized(metric_name: str):
    return authorize_metric(
        metric_name,
        provenance_type="exact",
        provenance_id=f"test-mention:{metric_name}",
        scope="platform",
        scope_data=SCOPE_DATA,
    )


def _block_unexpected_execution(monkeypatch):
    async def unexpected_mcp(*_args, **_kwargs):
        pytest.fail("invalid plan reached the MCP data transport")

    monkeypatch.setattr(mcp_client, "_call_local_mcp_tool", unexpected_mcp)
    monkeypatch.setattr(
        "app.metrics.registry.MetricRegistry.execute",
        lambda *_args, **_kwargs: pytest.fail("invalid plan reached Registry execution"),
    )


@pytest.mark.parametrize("metric_name", ["order_count", "gmv"])
def test_point_contribution_built_plan_routes_to_two_date_tool(monkeypatch, metric_name):
    question = (
        f"Compare {metric_name} contribution by product_category on "
        "2018-08-23 and 2018-08-22"
    )
    decision = MetricPlanningDecision(
        status="confirmed", confirmed_metrics=[metric_name],
        query_type="metric_contribution", dimension="product_category",
        confidence=1,
    )
    plan = build_query_plan(
        question, decision, metrics=[_authorized(metric_name)],
        comparison_dates={"baseline_date": "2018-08-22", "current_date": "2018-08-23"},
    )[0]
    assert plan.current_date == "2018-08-23"
    assert plan.baseline_date == "2018-08-22"

    observed: list[tuple[str, dict[str, object]]] = []

    async def fake_mcp(tool_name, arguments):
        observed.append((tool_name, dict(arguments)))
        return {"status": "ok", "metric_name": metric_name, "items": []}

    monkeypatch.setattr(mcp_client, "_call_local_mcp_tool", fake_mcp)
    monkeypatch.setattr(
        "app.metrics.registry.MetricRegistry.execute",
        lambda *_args, **_kwargs: pytest.fail("contribution must not call single-day Registry execution"),
    )
    monkeypatch.setattr(
        mcp_client, "resolve_expression",
        lambda *_args, **_kwargs: pytest.fail("typed contribution dates must bypass date resolution"),
    )
    permission_checks: list[str] = []
    original_permission_check = mcp_client.ensure_tool_allowed

    def capture_permission(agent_name, tool_name, **kwargs):
        permission_checks.append(tool_name)
        return original_permission_check(agent_name, tool_name, **kwargs)

    monkeypatch.setattr(mcp_client, "ensure_tool_allowed", capture_permission)
    request_id = f"point-request-{metric_name}"
    result = mcp_client.query_business_data(
        "metric_query", question, plan=plan, request_id=request_id,
    )

    assert observed == [(
        "compare_breakdown_by_dimension",
        {
            "metric_name": metric_name,
            "current_date": "2018-08-23",
            "baseline_date": "2018-08-22",
            "dimension": "product_category",
            "limit": 100,
        },
    )]
    assert permission_checks == ["compare_breakdown_by_dimension"]
    assert result["tool_name"] == "compare_breakdown_by_dimension"
    assert result["arguments"] == observed[0][1]
    assert result["request_id"] == request_id
    assert result["idempotency_key"] == request_id
    assert result["result"]["status"] == "ok"
    assert result["result"]["parsed"]["current_date"] == "2018-08-23"
    assert result["result"]["parsed"]["baseline_date"] == "2018-08-22"
    assert result["result"]["parsed"]["authorized_metric"]["metric_id"] == metric_name
    snapshot = result["authorization_snapshot"]
    assert snapshot["request_id"] == request_id
    assert snapshot["decision"] == "allowed"
    assert snapshot["authorization_checked"] is True
    assert snapshot["authorized_metric_id"] == metric_name
    assert snapshot["scope"] == "platform"
    assert snapshot["operation"] == "metric_contribution"
    assert snapshot["dimension"] == "product_category"
    assert snapshot["period"]["baseline_date"] == "2018-08-22"
    assert snapshot["period"]["current_date"] == "2018-08-23"
    assert snapshot["permission_version"]
    assert snapshot["policy_set_version"]
    assert snapshot["identity_sha256"]


def test_period_contribution_keeps_period_contract_and_normalizes_legacy_tool_hint(monkeypatch):
    plan = MetricQueryPlan(
        query_type="metric_contribution", metric_name="gmv",
        tool_name="compare_breakdown_by_dimension", dimension="product_category",
        granularity="month", baseline_start="2018-07-01", baseline_end="2018-07-31",
        current_start="2018-08-01", current_end="2018-08-31",
        authorized_metric=_authorized("gmv"),
    )
    observed = []

    async def fake_mcp(tool_name, arguments):
        observed.append((tool_name, dict(arguments)))
        return {"status": "ok", "metric_name": "gmv", "items": []}

    monkeypatch.setattr(mcp_client, "_call_local_mcp_tool", fake_mcp)
    monkeypatch.setattr(
        "app.metrics.registry.MetricRegistry.execute",
        lambda *_args, **_kwargs: pytest.fail("period contribution must not call single-day Registry execution"),
    )
    monkeypatch.setattr(
        mcp_client, "resolve_expression",
        lambda *_args, **_kwargs: pytest.fail("typed period dates must bypass date resolution"),
    )
    result = mcp_client.query_business_data(
        "metric_query", "compare monthly GMV contribution", plan=plan,
        request_id="period-request-1",
    )

    expected_args = {
        "metric_name": "gmv", "current_start": "2018-08-01",
        "current_end": "2018-08-31", "baseline_start": "2018-07-01",
        "baseline_end": "2018-07-31", "granularity": "month",
        "dimension": "product_category", "limit": 100,
    }
    assert observed == [("compare_breakdown_by_dimension_period", expected_args)]
    assert result["tool_name"] == "compare_breakdown_by_dimension_period"
    assert result["arguments"] == expected_args
    assert result["request_id"] == "period-request-1"
    assert result["result"]["parsed"]["current_start"] == "2018-08-01"
    assert result["result"]["parsed"]["baseline_start"] == "2018-07-01"


def _period_additive_projection_state(metric_name="freight_value"):
    comparison = {
        "kind": "period_pair",
        "baseline": {"start_date": "2018-07-01", "end_date": "2018-07-31"},
        "current": {"start_date": "2018-08-01", "end_date": "2018-08-31"},
        "granularity": "month",
    }
    payload = {
        "status": "ok", "metric_name": metric_name, "dimension": "product_category",
        "baseline_period": {"start_date": "2018-07-01", "end_date": "2018-07-31"},
        "current_period": {"start_date": "2018-08-01", "end_date": "2018-08-31"},
        "baseline_total": 200.0, "current_total": 220.0, "total_delta": 20.0,
        "baseline_items": [
            {"dimension_value": "books", "value": 120.0},
            {"dimension_value": "home", "value": 80.0},
        ],
        "current_items": [
            {"dimension_value": "books", "value": 150.0},
            {"dimension_value": "home", "value": 70.0},
        ],
        "source_table": "analytics.daily_metrics",
    }
    parsed = {
        "metric_name": metric_name, "dimension": "product_category",
        "query_type": "metric_contribution",
        "baseline_start": "2018-07-01", "baseline_end": "2018-07-31",
        "current_start": "2018-08-01", "current_end": "2018-08-31",
    }
    return {
        "controlled_analysis_scope": {},
        "context_compilation": {"request_contract": {
            "metric_id": metric_name,
            "dimension": "product_category",
            "operation_intent": "dimensional_statistical_contribution",
            "comparison": comparison,
        }},
        "evidence_bundles": [{
            "trusted": True, "evidence_id": "period-contribution-evidence",
            "content": {"result": {
                "status": "ok", "parsed": parsed,
                "contribution_result": payload,
            }},
        }],
    }


def test_period_additive_projection_is_limited_to_platform_freight_category(monkeypatch):
    from app.graph.nodes.answer import _validated_request_contribution_cards

    state = _period_additive_projection_state()
    monkeypatch.setattr(
        "app.graph.nodes.critic._review_request_contract",
        lambda *_args, **_kwargs: {
            "status": "passed", "contribution_evidence_ids": ["period-contribution-evidence"],
        },
    )
    cards = _validated_request_contribution_cards(state)

    assert len(cards) == 1
    card = cards[0]
    assert card["method"] == "additive_change_v1"
    assert card["baseline_period"] == {
        "start_date": "2018-07-01", "end_date": "2018-07-31", "granularity": "month",
    }
    assert card["current_period"] == {
        "start_date": "2018-08-01", "end_date": "2018-08-31", "granularity": "month",
    }
    assert card["total_delta"] == pytest.approx(20.0)
    assert sum(item["delta_value"] for item in card["items"]) == pytest.approx(20.0)


def test_period_additive_projection_rejects_mismatched_period_evidence(monkeypatch):
    from app.graph.nodes.answer import _validated_request_contribution_cards

    state = _period_additive_projection_state()
    state["evidence_bundles"][0]["content"]["result"]["contribution_result"]["current_period"]["end_date"] = "2018-08-30"
    monkeypatch.setattr(
        "app.graph.nodes.critic._review_request_contract",
        lambda *_args, **_kwargs: {
            "status": "passed", "contribution_evidence_ids": ["period-contribution-evidence"],
        },
    )

    assert _validated_request_contribution_cards(state) == []
    assert any(
        item.get("code") == "contribution_projection_failed"
        for item in state.get("errors", [])
    )


def test_period_additive_projection_does_not_open_gmv_period_contribution(monkeypatch):
    from app.graph.nodes.answer import _validated_request_contribution_cards

    state = _period_additive_projection_state(metric_name="gmv")
    monkeypatch.setattr(
        "app.graph.nodes.critic._review_request_contract",
        lambda *_args, **_kwargs: {
            "status": "passed", "contribution_evidence_ids": ["period-contribution-evidence"],
        },
    )
    monkeypatch.setattr(
        "app.graph.nodes.answer._validated_additive_request_cards",
        lambda *_args, **_kwargs: pytest.fail("GMV period contribution must remain closed"),
    )

    assert _validated_request_contribution_cards(state) == []
    assert state["contribution_projection"]["terminal_code"] == "CAPABILITY_GAP"
    assert state["contribution_projection"]["status"] == "unsupported"
    assert state.get("errors", []) == []


def test_capability_gap_renders_as_unsupported_without_workflow_error(monkeypatch):
    from app.graph.nodes.answer import (
        _record_contribution_projection_failure,
        generate_answer,
    )
    from app.graph.terminal_status import result_terminal_status

    state = {
        "task_type": "anomaly_analysis",
        "analysis_result": {
            "summary": "", "facts": [], "changes": [],
            "contribution": [], "limitations": [],
        },
        "context_compilation": {},
        "evidence_bundles": [],
        "errors": [],
        "trace_id": "capability-boundary-test",
    }
    _record_contribution_projection_failure(
        state, "CAPABILITY_GAP",
        "metric=gmv dimension=product_category period_pair not registered",
    )
    monkeypatch.setattr("app.graph.nodes.answer._validated_contribution_cards", lambda _state: [])

    with activate_tenant_context(TENANT):
        answered = generate_answer(state)

    assert answered["errors"] == []
    assert answered["analysis_presentation"]["status"] == "unsupported"
    assert answered["analysis_presentation"]["contributions"] == []
    assert "尚未注册可验证的统计贡献方法" in answered["answer"]
    assert "metric=gmv" not in answered["answer"]
    assert result_terminal_status(answered) == "completed"


def test_proportional_period_method_precedes_additive_projection(monkeypatch):
    from app.graph.nodes.answer import _validated_request_contribution_cards

    state = _period_additive_projection_state(metric_name="order_count")
    state["context_compilation"]["request_contract"]["metric_id"] = "order_count"
    state["evidence_bundles"][0]["content"]["result"]["parsed"]["metric_name"] = "order_count"
    state["evidence_bundles"][0]["content"]["result"]["contribution_result"]["metric_name"] = "order_count"
    called = []
    monkeypatch.setattr(
        "app.graph.nodes.critic._review_request_contract",
        lambda *_args, **_kwargs: {
            "status": "passed", "contribution_evidence_ids": ["period-contribution-evidence"],
        },
    )
    monkeypatch.setattr(
        "app.graph.nodes.answer._validated_proportional_request_cards",
        lambda *_args, **_kwargs: called.append("proportional") or [{"method": "proportional_overlap_v1"}],
    )
    monkeypatch.setattr(
        "app.graph.nodes.answer._validated_additive_request_cards",
        lambda *_args, **_kwargs: pytest.fail("additive must not intercept proportional overlap"),
    )

    cards = _validated_request_contribution_cards(state)

    assert called == ["proportional"]
    assert cards == [{"method": "proportional_overlap_v1"}]


def test_explicit_contribution_limit_is_preserved(monkeypatch):
    plan = MetricQueryPlan(
        query_type="metric_contribution", metric_name="order_count",
        current_date="2018-08-23", baseline_date="2018-08-22",
        dimension="product_category", limit=10,
        authorized_metric=_authorized("order_count"),
    )
    observed = []

    async def fake_mcp(tool_name, arguments):
        observed.append((tool_name, dict(arguments)))
        return {"status": "ok", "metric_name": "order_count", "items": []}

    monkeypatch.setattr(mcp_client, "_call_local_mcp_tool", fake_mcp)
    result = mcp_client.query_business_data(
        "metric_query", "compare order count contribution by product category",
        plan=plan, request_id="explicit-contribution-limit",
    )
    assert observed[0][1]["limit"] == 10
    assert result["arguments"]["limit"] == 10


def test_full_contribution_default_preserves_all_38_groups_without_other():
    baseline_items = [
        {"dimension_value": f"category_{index:02d}", "value": 100 + index}
        for index in range(38)
    ]
    current_items = [
        {"dimension_value": f"category_{index:02d}", "value": 90 + index}
        for index in range(38)
    ]
    baseline_total = sum(item["value"] for item in baseline_items)
    current_total = sum(item["value"] for item in current_items)

    finding = calculate_change_contribution(
        "freight_value", "product_category", date.fromisoformat("2017-11-21"),
        date.fromisoformat("2017-11-20"), current_items, baseline_items,
        overall_total_delta=current_total - baseline_total,
        current_total=current_total, baseline_total=baseline_total,
    )

    assert len(finding.items) == 38
    assert {item.dimension_value for item in finding.items} == {
        f"category_{index:02d}" for index in range(38)
    }
    assert all(item.dimension_value != "Other" for item in finding.items)
    assert sum(item.delta_value for item in finding.items) == pytest.approx(-380.0)


def test_month_position_additive_contribution_reconciles():
    baseline_items = [
        {"dimension_value": "月初(1-10日)", "value": 100.0},
        {"dimension_value": "月中(11-20日)", "value": 90.0},
        {"dimension_value": "月末(21-31日)", "value": 40.0},
    ]
    current_items = [
        {"dimension_value": "月初(1-10日)", "value": 120.0},
        {"dimension_value": "月中(11-20日)", "value": 80.0},
        {"dimension_value": "月末(21-31日)", "value": 50.0},
    ]
    finding = calculate_change_contribution(
        "freight_value", "month_position", date.fromisoformat("2017-04-30"),
        date.fromisoformat("2017-03-31"), current_items, baseline_items,
        overall_total_delta=20.0, current_total=250.0, baseline_total=230.0,
    )

    by_label = {item.dimension_value: item.delta_value for item in finding.items}
    assert finding.total_delta == pytest.approx(20.0)
    assert set(by_label) == {"月初(1-10日)", "月中(11-20日)", "月末(21-31日)"}
    assert by_label["月初(1-10日)"] == pytest.approx(20.0)
    assert by_label["月中(11-20日)"] == pytest.approx(-10.0)
    assert by_label["月末(21-31日)"] == pytest.approx(10.0)
    assert sum(by_label.values()) == pytest.approx(20.0)


def test_period_ratio_facts_route_remains_unchanged(monkeypatch):
    plan = MetricQueryPlan(
        query_type="metric_contribution", metric_name="freight_rate",
        tool_name="compare_freight_rate_ratio_facts_by_dimension_period",
        dimension="month_position", granularity="month",
        baseline_start="2018-07-01", baseline_end="2018-07-31",
        current_start="2018-08-01", current_end="2018-08-31",
        authorized_metric=_authorized("freight_rate"),
    )
    observed = []

    async def fake_mcp(tool_name, arguments):
        observed.append((tool_name, dict(arguments)))
        return {"status": "ok", "metric_name": "freight_rate", "complete": True}

    monkeypatch.setattr(mcp_client, "_call_local_mcp_tool", fake_mcp)
    monkeypatch.setattr(
        "app.metrics.registry.MetricRegistry.execute",
        lambda *_args, **_kwargs: pytest.fail("ratio contribution must not call single-day Registry execution"),
    )
    result = mcp_client.query_business_data(
        "metric_query", "compare monthly freight rate contribution", plan=plan,
        request_id="ratio-period-request",
    )
    assert observed == [(
        "compare_freight_rate_ratio_facts_by_dimension_period",
        {
            "metric_name": "freight_rate", "current_start": "2018-08-01",
            "current_end": "2018-08-31", "baseline_start": "2018-07-01",
            "baseline_end": "2018-07-31", "granularity": "month",
            "dimension": "month_position",
        },
    )]
    assert result["result"]["status"] == "ok"


def test_single_day_breakdown_still_uses_registry_breakdown(monkeypatch):
    plan = MetricQueryPlan(
        query_type="metric_breakdown", metric_name="order_count",
        metric_date="2018-08-23", start_date="2018-08-23", end_date="2018-08-23",
        dimension="product_category", tool_name="breakdown_by_dimension",
        authorized_metric=_authorized("order_count"),
    )
    calls = []

    def fake_execute(_self, metric_name, *, operation, **kwargs):
        calls.append((metric_name, operation, kwargs))
        return {"status": "ok", "items": [{"dimension_value": "books", "value": 8}]}

    monkeypatch.setattr("app.metrics.registry.MetricRegistry.execute", fake_execute)
    monkeypatch.setattr(mcp_client, "resolve_expression", lambda *_args, **_kwargs: None)
    result = mcp_client.query_business_data(
        "metric_query", "order count breakdown on 2018-08-23",
        plan=plan, request_id="single-breakdown-request",
    )
    assert calls and calls[0][0:2] == ("order_count", "breakdown")
    assert calls[0][2]["metric_date"].isoformat() == "2018-08-23"
    assert result["server"] == "metric-registry"
    assert result["result"]["status"] == "ok"
    assert result["request_id"] == "single-breakdown-request"


@pytest.mark.parametrize(
    "plan_updates",
    [
        {},
        {"current_date": "2018-08-23"},
        {"current_date": "2018-08-23", "baseline_date": "2018-08-22", "left_date": "2018-08-23", "right_date": "2018-08-22"},
        {"current_date": "2018-08-23", "baseline_date": "2018-08-22", "current_start": "2018-08-01"},
        {"current_date": "2018-08-23", "baseline_date": "2018-08-22", "metric_date": "2018-08-23"},
        {"left_date": "2018-08-23", "right_date": "2018-08-22"},
    ],
)
def test_invalid_contribution_shapes_are_rejected_before_execution(monkeypatch, plan_updates):
    _block_unexpected_execution(monkeypatch)
    base = {
        "query_type": "metric_contribution", "metric_name": "order_count",
        "dimension": "product_category", "authorized_metric": _authorized("order_count"),
    }
    plan = MetricQueryPlan(**{**base, **plan_updates})
    result = mcp_client.query_business_data(
        "metric_query", "compare order count contribution", plan=plan,
        request_id="invalid-contribution-request",
    )
    assert result["result"]["status"] == "unsupported"
    assert result["result"]["parsed"]["query_type"] == "metric_contribution"
    assert result["request_id"] == "invalid-contribution-request"
    assert result["idempotency_key"] == "invalid-contribution-request"
    assert result["tool_name"] is None


def test_unsupported_dimension_is_rejected_before_execution(monkeypatch):
    _block_unexpected_execution(monkeypatch)
    plan = MetricQueryPlan(
        query_type="metric_contribution", metric_name="order_count",
        current_date="2018-08-23", baseline_date="2018-08-22",
        dimension="payment_type", tool_name="compare_breakdown_by_dimension",
        authorized_metric=_authorized("order_count"),
    )
    result = mcp_client.query_business_data(
        "metric_query", "compare order count by payment_type", plan=plan,
        request_id="unsupported-dimension-request",
    )
    assert result["result"]["status"] == "unsupported"
    assert "dimension" in result["result"]["reason"].lower()
    assert result["request_id"] == "unsupported-dimension-request"


def test_registered_dimension_is_still_denied_when_identity_policy_withholds_it(monkeypatch):
    """Registry support cannot substitute for the caller's dimension grant."""
    from app.auth.context import build_authorization_context
    from app.metrics.registry import get_registry

    _block_unexpected_execution(monkeypatch)
    registry = get_registry()
    assert registry.is_dimension_allowed("order_count", "customer_state", "platform")
    authz = build_authorization_context(TENANT, metric_scope_type="platform")
    assert "customer_state" not in authz.allowed_dimensions
    assert "product_category" in authz.allowed_dimensions

    plan = MetricQueryPlan(
        query_type="metric_contribution", metric_name="order_count",
        current_date="2018-08-23", baseline_date="2018-08-22",
        dimension="customer_state", tool_name="compare_breakdown_by_dimension",
        authorized_metric=_authorized("order_count"),
    )
    result = mcp_client.query_business_data(
        "metric_query", "compare order_count contribution by customer_state",
        plan=plan, request_id="registered-but-denied-dimension",
    )
    assert result["result"]["status"] == "unsupported"
    assert "Dimension scope is not authorized" in result["result"]["reason"]
    assert result["tool_name"] is None
    assert result["request_id"] == "registered-but-denied-dimension"
    snapshot = result["authorization_snapshot"]
    assert snapshot["request_id"] == "registered-but-denied-dimension"
    assert snapshot["decision"] == "denied"
    assert snapshot["authorization_checked"] is True
    assert snapshot["reason_code"] == "AUTH_DENIED"


def test_seller_scope_does_not_inherit_platform_contribution_operation(monkeypatch):
    """Seller value/trend support cannot create a seller contribution route."""
    from app.metrics.registry import get_registry

    _block_unexpected_execution(monkeypatch)
    seller = TenantContext(
        tenant_id="seller-tenant", user_id="seller-dispatch-test", role="analyst",
        scope_type="seller", seller_id="seller-1",
    )
    with activate_tenant_context(seller):
        seller_metric = authorize_metric(
            "gmv", provenance_type="exact", provenance_id="seller-exact-gmv",
            scope="seller", scope_data={
                "tenant_id": seller.tenant_id, "user_id": seller.user_id,
                "permissions": seller.role, "scope_type": seller.scope_type,
            },
        )
        assert get_registry().is_dimension_allowed("gmv", "customer_state", "seller")
        assert not get_registry().supports("gmv", "breakdown", "seller")
        plan = MetricQueryPlan(
            query_type="metric_contribution", metric_name="gmv",
            current_date="2018-08-23", baseline_date="2018-08-22",
            dimension="customer_state", tool_name="compare_breakdown_by_dimension",
            authorized_metric=seller_metric,
        )
        result = mcp_client.query_business_data(
            "metric_query", "compare seller gmv by customer_state",
            tenant_id=seller.tenant_id, role=seller.role, plan=plan,
            request_id="seller-scope-no-platform-expansion",
        )
    assert result["result"]["status"] == "unsupported"
    assert result["request_id"] == "seller-scope-no-platform-expansion"
    assert result["tool_name"] is None


def test_registry_timeout_is_typed_serializable_and_never_retried_through_mcp(monkeypatch):
    from app.metrics.registry import MetricRegistry

    plan = MetricQueryPlan(
        query_type="metric_value", metric_name="gmv", metric_date="2024-05-04",
        authorized_metric=_authorized("gmv"),
    )
    monkeypatch.setattr(mcp_client, "resolve_expression", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        MetricRegistry, "execute",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("private query text")),
    )
    monkeypatch.setattr(
        mcp_client, "call_local_mcp_tool",
        lambda *_args, **_kwargs: pytest.fail("a known Registry timeout must not repeat through MCP"),
    )

    result = mcp_client.query_business_data(
        "metric_query", "查询 2024-05-04 GMV", plan=plan,
        request_id="timeout-contract-request",
    )
    inner = result["result"]
    assert inner["status"] == "execution_failed"
    assert inner["failure_context"] == {
        "category": "tool_timeout", "service": "metric_registry",
        "code": "tool_timeout", "cause_type": "TimeoutError",
        "request_id": "timeout-contract-request", "retryable": None,
    }
    assert "超时" in inner["reason"]
    assert "private query text" not in str(result)
    assert result["request_id"] == "timeout-contract-request"
    assert inner["parsed"]["metric_name"] == "gmv"


@pytest.mark.parametrize(
    ("exception", "expected_status", "expected_category", "expected_code"),
    [
        (
            __import__("sqlalchemy").exc.OperationalError("SELECT private", {}, RuntimeError("secret")),
            "execution_failed", "database_error", "database_execution_failed",
        ),
        (
            __import__("app.metrics.exceptions", fromlist=["CapabilityNotSupported"]).CapabilityNotSupported("not supported"),
            "unsupported", "capability_gap", "capability_not_supported",
        ),
        (
            __import__("app.metrics.exceptions", fromlist=["ScopeNotAllowed"]).ScopeNotAllowed("scope denied"),
            "unsupported", "authorization_denied", "scope_not_allowed",
        ),
    ],
)
def test_registry_failures_keep_distinct_structured_categories(
    monkeypatch, exception, expected_status, expected_category, expected_code,
):
    from app.metrics.registry import MetricRegistry

    plan = MetricQueryPlan(
        query_type="metric_value", metric_name="gmv", metric_date="2024-05-04",
        authorized_metric=_authorized("gmv"),
    )
    monkeypatch.setattr(mcp_client, "resolve_expression", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        MetricRegistry, "execute",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(exception),
    )
    result = mcp_client.query_business_data(
        "metric_query", "查询 2024-05-04 GMV", plan=plan,
        request_id="typed-failure-request",
    )
    inner = result["result"]
    assert inner["status"] == expected_status
    assert inner["failure_context"]["category"] == expected_category
    assert inner["failure_context"]["code"] == expected_code
    assert inner["failure_context"]["request_id"] == "typed-failure-request"
    assert "secret" not in str(result)
    assert "private" not in str(result)


def test_mcp_transport_timeout_is_returned_as_tool_timeout(monkeypatch):
    from app.metrics.registry import MetricRegistry
    from app.resilience import ExternalServiceError

    plan = MetricQueryPlan(
        query_type="metric_trend", metric_name="gmv",
        start_date="2024-05-01", end_date="2024-05-04",
        tool_name="query_metric_trend", authorized_metric=_authorized("gmv"),
    )
    monkeypatch.setattr(mcp_client, "resolve_expression", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        MetricRegistry, "execute",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("registry unavailable")),
    )
    monkeypatch.setattr(
        mcp_client, "call_local_mcp_tool",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ExternalServiceError("mcp", "timeout", "private transport detail")
        ),
    )

    result = mcp_client.query_business_data(
        "metric_query", "查询GMV趋势", plan=plan,
        request_id="mcp-timeout-request",
    )
    assert result["result"]["status"] == "execution_failed"
    assert result["result"]["failure_context"]["category"] == "tool_timeout"
    assert result["result"]["failure_context"]["service"] == "mcp"
    assert result["request_id"] == "mcp-timeout-request"
    assert "private transport detail" not in str(result)


@pytest.mark.parametrize("period", [False, True])
def test_contribution_mcp_deadline_returns_correlated_typed_failure(monkeypatch, period):
    import asyncio
    from app.resilience import policy

    fields = (
        {"baseline_start": "2018-07-01", "baseline_end": "2018-07-31",
         "current_start": "2018-08-01", "current_end": "2018-08-31", "granularity": "month"}
        if period else {"baseline_date": "2018-08-22", "current_date": "2018-08-23"}
    )
    plan = MetricQueryPlan(
        query_type="metric_contribution", metric_name="order_count",
        dimension="product_category", authorized_metric=_authorized("order_count"), **fields,
    )
    attempts, cancellations = [], []

    async def blocked_transport(tool, arguments):
        attempts.append((tool, arguments))
        try:
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            cancellations.append(tool)
            raise
        pytest.fail("timed out tool must not return success")

    settings = mcp_client.get_settings().model_copy(update={
        "mcp_transport": "stdio", "mcp_timeout_seconds": 0.001,
    })
    monkeypatch.setattr(mcp_client, "get_settings", lambda: settings)
    monkeypatch.setattr(mcp_client, "_call_local_mcp_tool", blocked_transport)
    monkeypatch.setattr(policy, "_BREAKERS", {})
    monkeypatch.setattr(
        "app.metrics.registry.MetricRegistry.execute",
        lambda *_a, **_k: pytest.fail("contribution failure must not fall back to Registry"),
    )
    result = mcp_client.query_business_data(
        "metric_query", "贡献查询", plan=plan,
        request_id="contribution-timeout", idempotency_key="contribution-key",
    )
    assert result["result"]["status"] == "execution_failed"
    context = result["result"]["failure_context"]
    assert context["category"] == context["code"] == "tool_timeout"
    assert context["service"] == "mcp"
    assert context["request_id"] == result["request_id"] == "contribution-timeout"
    assert result["idempotency_key"] == "contribution-key"
    assert result["arguments"]["dimension"] == "product_category"
    assert len(attempts) == len(cancellations) == 2  # Existing normal-mode retry policy.
    assert "超时" in result["result"]["reason"]


@pytest.mark.parametrize("kind,category", [
    ("database", "database_error"), ("permission", "authorization_denied"),
    ("unexpected", "execution_error"),
])
def test_contribution_failure_does_not_lose_type_or_leak_exception(monkeypatch, kind, category):
    from sqlalchemy.exc import OperationalError

    exc = {
        "database": OperationalError("SELECT private_payload", {}, RuntimeError("secret_payload")),
        "permission": PermissionError("secret_payload"),
        "unexpected": RuntimeError("secret_payload"),
    }[kind]
    plan = MetricQueryPlan(
        query_type="metric_contribution", metric_name="order_count",
        dimension="product_category", baseline_date="2018-08-22", current_date="2018-08-23",
        authorized_metric=_authorized("order_count"),
    )
    def fail(*_a, **_k):
        raise exc
    monkeypatch.setattr(mcp_client, "call_local_mcp_tool", fail)
    settings = mcp_client.get_settings().model_copy(update={"mcp_transport": "stdio"})
    monkeypatch.setattr(mcp_client, "get_settings", lambda: settings)
    result = mcp_client.query_business_data("metric_query", "贡献查询", plan=plan, request_id="failure-request")
    assert result["result"]["failure_context"]["category"] == category
    assert result["result"]["failure_context"]["request_id"] == "failure-request"
    assert result["result"]["status"] == ("unsupported" if kind == "permission" else "execution_failed")
    assert "secret_payload" not in str(result)
    assert "private_payload" not in str(result)


def test_missing_authorization_is_rejected_before_execution(monkeypatch):
    _block_unexpected_execution(monkeypatch)
    plan = MetricQueryPlan(
        query_type="metric_contribution", metric_name="order_count",
        current_date="2018-08-23", baseline_date="2018-08-22",
        dimension="product_category", tool_name="compare_breakdown_by_dimension",
    )
    result = mcp_client.query_business_data(
        "metric_query", "compare order count by product category", plan=plan,
        request_id="unauthorized-contribution-request",
    )
    assert result["result"]["status"] == "unsupported"
    assert result["request_id"] == "unauthorized-contribution-request"
    assert result["idempotency_key"] == "unauthorized-contribution-request"
    assert result["result"]["parsed"]["authorized_metric"] is None


def test_left_right_dates_are_not_inferred_from_question(monkeypatch):
    _block_unexpected_execution(monkeypatch)
    plan = MetricQueryPlan(
        query_type="metric_contribution", metric_name="order_count",
        left_date="2018-08-23", right_date="2018-08-22",
        dimension="product_category", authorized_metric=_authorized("order_count"),
    )
    result = mcp_client.query_business_data(
        "metric_query",
        "Compare order_count contribution by product_category on 2018-08-23 and 2018-08-22",
        plan=plan, request_id="left-right-rejected-request",
    )
    assert result["result"]["status"] == "unsupported"
    assert "left/right" in result["result"]["reason"]
    assert result["request_id"] == "left-right-rejected-request"

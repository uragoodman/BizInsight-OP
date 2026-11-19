from __future__ import annotations

from contextlib import contextmanager
from datetime import date

import pytest

from app.auth.context import AuthorizationContext
from app.auth.enforcement import AuthorizationBoundary, StaleAuthorization, assert_authorization_boundary
from app.data.repositories import explorer
from app.metrics.registry import get_registry
from app.scope.canonical import ScopeError
from app.scope.seller_scope import SellerScopeIntent, issue_seller_scope_payload, resolve_seller_scope
from app.tenancy import SellerGrant, TenantContext, activate_tenant_context


def _authz(*, scope_type="platform", role="admin", sellers=(), grants=(), dims=("seller",)):
    return AuthorizationContext(
        tenant_id="tenant-test", user_id="user-test", role=role, scope_type=scope_type,
        permission_version="permission-v1", policy_set_version="policy-v1",
        authz_ref="authz-test", authz_digest="authz-digest-v1",
        allowed_seller_ids=tuple(sellers), seller_grants=tuple(grants),
        allowed_metric_ids=("gmv", "order_count"), allowed_dimensions=tuple(dims),
        allowed_actions=("read", "analyze"), allowed_tools=(),
    )


def _intent(scope_type, sellers=(), *, include_unattributed=False):
    return SellerScopeIntent(
        seller_scope_type=scope_type, seller_ids=tuple(sellers),
        start_date=date(2024, 1, 1), end_date=date(2024, 1, 31),
        metric_ids=("gmv", "order_count"), dimensions=("seller",),
        include_unattributed=include_unattributed,
    )


def _enable_flags(monkeypatch, *, dimension=True, multi=True):
    from app.core.config import Settings
    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: Settings(
            seller_dimension_enabled=dimension,
            seller_multi_scope_enabled=multi,
        ),
    )


def _platform_admin_with_grants(*seller_ids):
    grants = tuple(SellerGrant(seller_id, "2024-01-01", "2024-01-31") for seller_id in seller_ids)
    return TenantContext(
        "tenant-test", "user-test", "admin", "platform", seller_grants=grants,
    )


def _signed_scope_payload(monkeypatch, tenant, intent):
    from app.core.config import Settings

    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: Settings(
            seller_dimension_enabled=True,
            seller_multi_scope_enabled=True,
            tenant_auth_secret="test-seller-scope-signing-secret",
        ),
    )
    return issue_seller_scope_payload(intent, tenant=tenant)


@contextmanager
def _fake_transaction(connection):
    yield connection


def test_seller_dimension_registry_and_metric_semantics(monkeypatch):
    registry = get_registry()
    _enable_flags(monkeypatch, dimension=False)
    assert registry.dimension("seller").source == "order_items.seller_id"
    assert not registry.is_dimension_allowed("gmv", "seller", "platform")

    _enable_flags(monkeypatch, dimension=True)
    assert registry.is_dimension_allowed("gmv", "seller", "platform")
    assert registry.is_dimension_allowed("order_count", "seller", "platform")
    semantics = registry.get("order_count").implementations["platform"].dimension_semantics["seller"]
    assert semantics.additive is False
    assert semantics.semantics == "distinct_orders_per_seller"
    assert not registry.is_dimension_allowed("avg_order_value", "seller", "platform")


def test_registered_seller_dimension_still_requires_metric_capability_and_permission(monkeypatch):
    from app.context.support_resolver import resolve_support

    _enable_flags(monkeypatch, dimension=True)
    denied = _authz(dims=())
    result = resolve_support(
        metric_id="avg_order_value", dimension="seller", operation="metric_breakdown",
        scope={}, authz=denied, registry=get_registry(),
    )
    assert result.schema_registered == "ok"
    assert result.metric_compatible == "missing"
    assert result.user_permitted == "denied"
    assert result.executable is False


def test_seller_order_count_contribution_is_separate_from_ordinary_breakdown_capability():
    """Attribution YAML owns contribution support; sales YAML owns ordinary query support."""
    from app.context.support_resolver import resolve_support

    registry = get_registry()
    definition = registry.get("order_count")
    assert "breakdown" not in definition.implementations["seller"].capabilities
    assert definition.implementations["seller"].dimensions == ()
    assert "product_category" in definition.attribution.for_scope("seller").dimensions
    assert "additive_change_v1" in definition.attribution.for_scope("seller").method_ids
    assert "proportional_overlap_v1" in definition.attribution.for_scope("seller").method_ids

    authz = _authz(scope_type="seller", dims=("product_category", "month_position"))
    scope = {
        "seller_id": "seller-a",
        "allowed_dimensions": ["product_category", "month_position"],
    }
    overlap_contribution = resolve_support(
        metric_id="order_count", dimension="product_category",
        operation="contribution", scope=scope, authz=authz, registry=registry,
    )
    time_contribution = resolve_support(
        metric_id="order_count", dimension="month_position",
        operation="contribution", scope=scope, authz=authz, registry=registry,
    )
    ordinary_breakdown = resolve_support(
        metric_id="order_count", dimension="product_category",
        operation="breakdown", scope=scope, authz=authz, registry=registry,
    )
    assert overlap_contribution.executable is True
    assert overlap_contribution.reason_code is None
    assert time_contribution.executable is True
    assert ordinary_breakdown.executable is False


def test_scope_resolver_supports_explicit_single_selected_all_and_fingerprints(monkeypatch):
    _enable_flags(monkeypatch)
    tenant = _platform_admin_with_grants("A", "B")
    authz = _authz(grants=tenant.seller_grants)
    single = resolve_seller_scope(_intent("single", ("A",)), tenant=tenant, authz=authz)
    selected = resolve_seller_scope(_intent("selected", ("A", "B")), tenant=tenant, authz=authz)
    all_scope = resolve_seller_scope(_intent("all"), tenant=tenant, authz=authz)
    all_unknown = resolve_seller_scope(
        _intent("all", include_unattributed=True), tenant=tenant, authz=authz,
    )
    assert single.seller_scope_type == "single" and single.seller_ids == ("A",)
    assert selected.seller_scope_type == "selected" and selected.seller_ids == ("A", "B")
    assert all_scope.seller_scope_type == "all" and all_scope.seller_ids == ()
    assert all_scope.query_scope_fingerprint != all_unknown.query_scope_fingerprint


@pytest.mark.parametrize("scope_type", ["selected", "all"])
def test_non_platform_identity_cannot_request_multi_scope(monkeypatch, scope_type):
    _enable_flags(monkeypatch)
    tenant = TenantContext(
        "tenant-test", "user-test", "analyst", "seller", seller_id="A",
        seller_grants=(SellerGrant("A", "2024-01-01", "2024-01-31"),),
    )
    authz = _authz(scope_type="seller", role="analyst", sellers=("A",),
                   grants=(SellerGrant("A", "2024-01-01", "2024-01-31"),), dims=())
    with pytest.raises(ScopeError, match="Only platform administrators"):
        resolve_seller_scope(_intent(scope_type, ("A",) if scope_type == "selected" else ()), tenant=tenant, authz=authz)


def test_single_seller_identity_is_bound_to_its_server_grant(monkeypatch):
    _enable_flags(monkeypatch)
    grant = SellerGrant("A", "2024-01-01", "2024-01-31")
    tenant = TenantContext("tenant-test", "user-test", "analyst", "seller", seller_id="A", seller_grants=(grant,))
    authz = _authz(scope_type="seller", role="analyst", sellers=("A",), grants=(grant,), dims=("seller",))
    scope = resolve_seller_scope(_intent("single", ("A",)), tenant=tenant, authz=authz)
    assert scope.seller_scope_type == "single"
    with pytest.raises(ScopeError, match="cannot access"):
        resolve_seller_scope(_intent("single", ("B",)), tenant=tenant, authz=authz)


def test_selected_scope_checks_each_seller_grant_date(monkeypatch):
    _enable_flags(monkeypatch)
    tenant = TenantContext(
        "tenant-test", "user-test", "admin", "platform",
        seller_grants=(SellerGrant("A", "2024-01-01", "2024-01-31"),
                       SellerGrant("B", "2024-01-10", "2024-01-31")),
    )
    authz = _authz(grants=tenant.seller_grants)
    with pytest.raises(ScopeError, match="exceeds seller grant"):
        resolve_seller_scope(_intent("selected", ("A", "B")), tenant=tenant, authz=authz)


def test_include_unattributed_is_all_admin_only(monkeypatch):
    _enable_flags(monkeypatch)
    tenant = TenantContext("tenant-test", "user-test", "admin", "platform")
    authz = _authz()
    with pytest.raises(ValueError, match="only valid for ALL"):
        _intent("selected", ("A", "B"), include_unattributed=True)
    with pytest.raises(ScopeError, match="Multi-seller scopes are disabled"):
        _enable_flags(monkeypatch, multi=False)
        resolve_seller_scope(_intent("all", include_unattributed=True), tenant=tenant, authz=authz)


def test_unattributed_scope_change_stales_authorization_boundary(monkeypatch):
    _enable_flags(monkeypatch)
    tenant = TenantContext("tenant-test", "user-test", "admin", "platform")
    authz = _authz()
    plain = resolve_seller_scope(_intent("all"), tenant=tenant, authz=authz)
    included = resolve_seller_scope(_intent("all", include_unattributed=True), tenant=tenant, authz=authz)
    boundary = AuthorizationBoundary(
        permission_version=authz.permission_version,
        policy_set_version=authz.policy_set_version,
        authorization_fingerprint=authz.authz_digest,
        query_scope_fingerprint=plain.query_scope_fingerprint,
    )
    with pytest.raises(StaleAuthorization):
        assert_authorization_boundary(authz, boundary, current_scope_fingerprint=included.query_scope_fingerprint)


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows

    def one(self):
        return self._rows[0]


class _FakeConnection:
    def __init__(self, *, items=None, total=None, unknown=None):
        self.items = [
            {"seller_id": "A", "value": 250, "group_order_count": 2},
            {"seller_id": "B", "value": 200, "group_order_count": 1},
        ] if items is None else items
        self.total = {"gmv": 450, "order_count": 2} if total is None else total
        self.unknown = {"row_count": 1, "gmv": 50, "order_count": 1} if unknown is None else unknown
        self.calls = []

    def execute(self, statement, params):
        sql = str(statement)
        self.calls.append((sql, params))
        if "GROUP BY oi.seller_id" in sql:
            return _FakeResult(self.items)
        if "AS row_count" in sql:
            return _FakeResult([self.unknown])
        return _FakeResult([self.total])


def test_controlled_selected_query_binds_seller_ids_and_returns_distinct_total(monkeypatch):
    _enable_flags(monkeypatch)
    from app.auth.context import build_authorization_context

    tenant = _platform_admin_with_grants("A", "B")
    authz = build_authorization_context(tenant, metric_scope_type="platform")
    scope = resolve_seller_scope(_intent("selected", ("A", "B")), tenant=tenant, authz=authz)
    connection = _FakeConnection()
    monkeypatch.setattr(explorer, "canonical_seller_query_transaction", lambda _engine, _scope: _fake_transaction(connection))
    with activate_tenant_context(tenant):
        result = explorer.breakdown_by_seller(scope, "gmv", limit=20)

    grouped_sql, grouped_params = connection.calls[0]
    total_sql, total_params = connection.calls[1]
    assert "oi.seller_id = ANY(:seller_ids)" in grouped_sql
    assert grouped_params["seller_ids"] == ["A", "B"]
    assert "COUNT(DISTINCT oi.order_id)" in total_sql
    assert total_params["seller_ids"] == ["A", "B"]
    assert result["total"] == {
        "value": 450.0, "gmv": 450.0, "order_count": 2,
        "total_includes_unattributed": False,
    }
    assert result["items"][0]["order_count"] == 2
    assert result["unattributed"] == {
        "gmv": 50.0, "order_count": 1,
        "reason": "seller_id_is_null; order_count may overlap seller-attributed orders and is not additive",
        "scope": "date_range_all_unattributed",
    }


@pytest.mark.parametrize("scope_type,seller_ids", [
    ("single", ("A",)), ("selected", ("A", "B")), ("all", ()),
])
def test_scope_and_repository_rebuild_the_same_real_authz_snapshot(
    monkeypatch, scope_type, seller_ids,
):
    from app.auth.context import build_authorization_context

    _enable_flags(monkeypatch)
    tenant = _platform_admin_with_grants("A", "B")
    authz = build_authorization_context(tenant, metric_scope_type="platform")
    scope = resolve_seller_scope(
        _intent(scope_type, seller_ids), tenant=tenant, authz=authz,
    )
    connection = _FakeConnection()
    monkeypatch.setattr(
        explorer, "canonical_seller_query_transaction",
        lambda _engine, _scope: _fake_transaction(connection),
    )
    assert scope.authorization_fingerprint == authz.authz_digest
    assert scope.permission_version == authz.permission_version
    assert scope.policy_set_version == authz.policy_set_version
    with activate_tenant_context(tenant):
        result = explorer.breakdown_by_seller(scope, "gmv")
    assert result["status"] == "ok"


def test_scope_rejects_stale_authz_snapshot(monkeypatch):
    from app.auth.context import build_authorization_context

    _enable_flags(monkeypatch)
    tenant = _platform_admin_with_grants("A", "B")
    authz = build_authorization_context(tenant, metric_scope_type="platform")
    scope = resolve_seller_scope(
        _intent("selected", ("A", "B")), tenant=tenant, authz=authz,
    )
    changed = _platform_admin_with_grants("A")
    with activate_tenant_context(changed), pytest.raises(StaleAuthorization):
        explorer.breakdown_by_seller(scope, "gmv")


def test_canonical_scope_cannot_narrow_finite_seller_scopes_to_empty(monkeypatch):
    _enable_flags(monkeypatch)
    tenant = _platform_admin_with_grants("A", "B")
    authz = _authz(grants=tenant.seller_grants)
    single = resolve_seller_scope(_intent("single", ("A",)), tenant=tenant, authz=authz)
    selected = resolve_seller_scope(_intent("selected", ("A", "B")), tenant=tenant, authz=authz)
    all_scope = resolve_seller_scope(_intent("all"), tenant=tenant, authz=authz)
    with pytest.raises(ScopeError, match="empty seller set"):
        single.narrow(seller_ids=[])
    with pytest.raises(ScopeError, match="empty seller set"):
        selected.narrow(seller_ids=[])
    with pytest.raises(ScopeError, match="cannot be narrowed"):
        all_scope.narrow(seller_ids=[])


def test_platform_admin_selected_requires_explicit_grants_and_date_entitlement(monkeypatch):
    _enable_flags(monkeypatch)
    tenant = TenantContext("tenant-test", "user-test", "admin", "platform")
    authz = _authz()
    with pytest.raises(ScopeError, match="no explicit seller grant"):
        resolve_seller_scope(_intent("selected", ("A",)), tenant=tenant, authz=authz)

    grant = SellerGrant("A", "2024-01-01", "2024-01-31")
    tenant = TenantContext("tenant-test", "user-test", "admin", "platform", seller_grants=(grant,))
    authz = _authz(grants=(grant,))
    assert resolve_seller_scope(_intent("selected", ("A",)), tenant=tenant, authz=authz).seller_ids == ("A",)
    with pytest.raises(ScopeError, match="exceeds seller grant"):
        resolve_seller_scope(
            SellerScopeIntent(
                seller_scope_type="selected", seller_ids=("A",),
                start_date=date(2023, 12, 31), end_date=date(2024, 1, 31),
                metric_ids=("gmv",), dimensions=("seller",),
            ), tenant=tenant, authz=authz,
        )


def test_single_seller_grant_date_and_scope_resource_limits(monkeypatch):
    _enable_flags(monkeypatch)
    grant = SellerGrant("A", "2024-01-01", "2024-01-31")
    tenant = TenantContext(
        "tenant-test", "user-test", "analyst", "seller", seller_id="A",
        seller_grants=(grant,),
    )
    authz = _authz(
        scope_type="seller", role="analyst", sellers=("A",), grants=(grant,),
    )
    too_wide = SellerScopeIntent(
        seller_scope_type="single", seller_ids=("A",),
        start_date=date(2024, 1, 1), end_date=date(2024, 2, 1),
        metric_ids=("gmv",), dimensions=("seller",),
    )
    with pytest.raises(ScopeError, match="seller entitlement"):
        resolve_seller_scope(too_wide, tenant=tenant, authz=authz)
    with pytest.raises(ScopeError, match="date range"):
        resolve_seller_scope(
            SellerScopeIntent(
                seller_scope_type="single", seller_ids=("A",),
                start_date=date(2020, 1, 1), end_date=date(2024, 1, 31),
                metric_ids=("gmv",), dimensions=("seller",),
            ), tenant=tenant, authz=authz,
        )
    with pytest.raises(ScopeError, match="limit of 100 sellers"):
        resolve_seller_scope(SellerScopeIntent(
            seller_scope_type="selected", seller_ids=tuple(f"S{i}" for i in range(101)),
            start_date=date(2024, 1, 1), end_date=date(2024, 1, 31),
            metric_ids=("gmv",), dimensions=("seller",),
        ), tenant=TenantContext("tenant-test", "user-test", "admin", "platform"), authz=_authz())


def test_all_uses_platform_date_policy_without_seller_grants(monkeypatch):
    _enable_flags(monkeypatch)
    tenant = TenantContext("tenant-test", "user-test", "admin", "platform")
    scope = resolve_seller_scope(_intent("all"), tenant=tenant, authz=_authz())
    assert scope.seller_scope_type == "all"
    too_wide = SellerScopeIntent(
        seller_scope_type="all", start_date=date(2020, 1, 1), end_date=date(2024, 1, 1),
        metric_ids=("gmv",), dimensions=("seller",),
    )
    with pytest.raises(ScopeError, match="date range"):
        resolve_seller_scope(too_wide, tenant=tenant, authz=_authz())


def test_seller_scope_operation_is_consistent_with_registry_support(monkeypatch):
    from app.context.support_resolver import resolve_support
    from app.auth.context import build_authorization_context

    _enable_flags(monkeypatch)
    grant = SellerGrant("A", "2024-01-01", "2024-01-31")
    tenant = TenantContext(
        "tenant-test", "user-test", "analyst", "seller", seller_id="A",
        seller_grants=(grant,),
    )
    seller_authz = build_authorization_context(tenant, metric_scope_type="seller")
    result = resolve_support(
        metric_id="gmv", dimension="seller", operation="metric_breakdown",
        scope={"seller_id": "A", "allowed_dimensions": ["seller"]},
        authz=seller_authz, registry=get_registry(),
    )
    assert result.executable is True
    assert result.reason_code is None


def test_seller_scope_flag_combinations_do_not_create_unauthorized_multi_plan(monkeypatch):
    from app.metrics.registry import get_registry

    tenant = _platform_admin_with_grants("A")
    authz = _authz(grants=tenant.seller_grants)
    _enable_flags(monkeypatch, dimension=False, multi=False)
    assert not get_registry().is_dimension_allowed("gmv", "seller", "platform")
    assert all(
        "seller" not in metric["dimensions"]
        for metric in get_registry().planning_catalog(scope="platform")
    )
    _enable_flags(monkeypatch, dimension=True, multi=False)
    assert get_registry().is_dimension_allowed("gmv", "seller", "platform")
    assert resolve_seller_scope(_intent("single", ("A",)), tenant=tenant, authz=authz).seller_scope_type == "single"
    with pytest.raises(ScopeError, match="Multi-seller scopes are disabled"):
        resolve_seller_scope(_intent("selected", ("A",)), tenant=tenant, authz=authz)
    with pytest.raises(ScopeError, match="Multi-seller scopes are disabled"):
        resolve_seller_scope(_intent("all"), tenant=tenant, authz=authz)


def test_unscoped_daily_metrics_rejects_seller_with_structured_result():
    from app.data.repositories.daily_metrics import (
        SUPPORTED_BREAKDOWN_DIMENSIONS, breakdown_by_dimension,
    )

    assert "seller" not in SUPPORTED_BREAKDOWN_DIMENSIONS
    result = breakdown_by_dimension("gmv", date(2024, 1, 1), "seller")
    assert result["status"] == "unsupported"
    assert result["reason_code"] == "seller_scope_required"


def test_null_only_all_result_is_ok_and_marks_unattributed(monkeypatch):
    from app.auth.context import build_authorization_context

    _enable_flags(monkeypatch)
    tenant = TenantContext("tenant-test", "user-test", "admin", "platform")
    authz = build_authorization_context(tenant, metric_scope_type="platform")
    scope = resolve_seller_scope(_intent("all", include_unattributed=True), tenant=tenant, authz=authz)
    connection = _FakeConnection(
        items=[], total={"gmv": 0, "order_count": 0},
        unknown={"row_count": 1, "gmv": 50, "order_count": 1},
    )
    monkeypatch.setattr(
        explorer, "canonical_seller_query_transaction",
        lambda _engine, _scope: _fake_transaction(connection),
    )
    with activate_tenant_context(tenant):
        result = explorer.breakdown_by_seller(scope, "gmv")
    assert result["items"] == []
    assert result["status"] == "ok"
    assert result["total"]["total_includes_unattributed"] is True
    assert result["unattributed"]["scope"] == "date_range_all_unattributed"


def _single_seller_mcp_plan(tenant):
    from app.metrics.authorization import authorize_metric
    from app.mcp.schemas import MetricQueryPlan
    from app.schemas.explorer import ControlledAnalysisScope

    controlled = ControlledAnalysisScope(
        seller_id="A", default_start_date=date(2024, 1, 1),
        default_end_date=date(2024, 1, 31), metric_name="gmv", granularity="day",
        allowed_metrics=("gmv", "order_count"),
        allowed_dimensions=(
            "seller", "product_category", "payment_type", "month_position",
            "day_of_week", "hour_of_day", "quarter_of_year", "half_of_year",
        ),
    )
    old_metric = authorize_metric(
        "gmv", provenance_type="exact", provenance_id="mention:test",
        scope="seller", scope_data=controlled.model_dump(mode="json"),
    )
    return controlled, MetricQueryPlan(
        query_type="metric_breakdown", metric_name="gmv", dimension="seller",
        authorized_metric=old_metric, limit=10,
    )


def test_mcp_legacy_single_scope_produces_seller_intent_and_remints_metric(monkeypatch):
    from app.auth.context import build_authorization_context
    from app.auth.enforcement import AuthorizationBoundary
    from app.core.config import Settings
    from app.mcp.client import query_scoped_business_data
    from app.scope.canonical import CanonicalScope

    _enable_flags(monkeypatch)
    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: Settings(seller_dimension_enabled=True, seller_multi_scope_enabled=True),
    )
    grant = SellerGrant("A", "2024-01-01", "2024-01-31")
    tenant = TenantContext(
        "tenant-test", "user-test", "analyst", "seller", seller_id="A",
        seller_grants=(grant,),
    )
    controlled, plan = _single_seller_mcp_plan(tenant)
    old_authorized_metric = plan.authorized_metric
    lifecycle_authz = build_authorization_context(
        tenant,
        allowed_metric_ids=controlled.allowed_metrics,
        allowed_dimensions=controlled.allowed_dimensions,
        metric_scope_type="seller",
    )
    lifecycle_scope = CanonicalScope.resolve(
        lifecycle_authz,
        seller_ids=(controlled.seller_id,),
        start_date=controlled.default_start_date,
        end_date=controlled.default_end_date,
        metric_ids=controlled.allowed_metrics,
        dimensions=controlled.allowed_dimensions,
        granularity=controlled.granularity,
    )
    lifecycle_boundary = AuthorizationBoundary(
        permission_version=lifecycle_authz.permission_version,
        policy_set_version=lifecycle_authz.policy_set_version,
        authorization_fingerprint=lifecycle_authz.authz_digest,
        query_scope_fingerprint=lifecycle_scope.query_scope_fingerprint,
    )
    connection = _FakeConnection(
        items=[{"seller_id": "A", "value": 250, "group_order_count": 2}],
        total={"gmv": 250, "order_count": 2},
    )
    monkeypatch.setattr(
        explorer, "canonical_seller_query_transaction",
        lambda _engine, _scope: _fake_transaction(connection),
    )
    with activate_tenant_context(tenant):
        response = query_scoped_business_data(
            "按店铺拆解 GMV", controlled.model_dump(mode="json"), plan=plan,
            authorization_boundary=lifecycle_boundary,
            current_query_scope_fingerprint=lifecycle_scope.query_scope_fingerprint,
        )
    assert response["tool_name"] == "breakdown_scoped_seller"
    assert response["result"]["status"] == "ok"
    assert "只有一个 seller 分组" in response["result"]["breakdown_result"]["message"]
    fresh = response["result"]["parsed"]["authorized_metric"]
    assert fresh["query_scope_fingerprint"] != old_authorized_metric.query_scope_fingerprint
    assert "authorization_boundary" not in response
    assert "authorization_boundary" not in response["arguments"]

    # The boundary minted in MCP is intentionally local to that call. A later
    # lifecycle using the prior boundary still fails against the new scope.
    from app.auth.enforcement import AuthorizationBoundary, assert_authorization_boundary
    from app.auth.context import build_authorization_context
    from app.auth.enforcement import StaleAuthorization

    authz = build_authorization_context(tenant, metric_scope_type="seller")
    with pytest.raises(StaleAuthorization):
        assert_authorization_boundary(
            authz, lifecycle_boundary,
            current_scope_fingerprint=response["arguments"]["query_scope_fingerprint"],
        )


def test_structured_scope_payload_routes_selected_breakdown_end_to_end(monkeypatch):
    from app.core.config import Settings
    from app.mcp.client import query_scoped_business_data

    _enable_flags(monkeypatch)
    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: Settings(seller_dimension_enabled=True, seller_multi_scope_enabled=True),
    )
    tenant = _platform_admin_with_grants("A", "B")
    _controlled, plan = _single_seller_mcp_plan(tenant)
    intent = SellerScopeIntent(
        seller_scope_type="selected", seller_ids=("A", "B"),
        start_date=date(2024, 1, 1), end_date=date(2024, 1, 31),
        metric_ids=("gmv",), dimensions=("seller",),
    )
    signed_payload = _signed_scope_payload(monkeypatch, tenant, intent)
    connection = _FakeConnection()
    monkeypatch.setattr(
        explorer, "canonical_seller_query_transaction",
        lambda _engine, _scope: _fake_transaction(connection),
    )
    with activate_tenant_context(tenant):
        response = query_scoped_business_data(
            "按店铺拆解 GMV", signed_payload, plan=plan,
        )
    assert response["tool_name"] == "breakdown_scoped_seller"
    assert response["arguments"]["seller_scope_type"] == "selected"
    assert response["arguments"]["seller_ids"] == ["A", "B"]
    assert response["seller_boundary"]["query_scope_fingerprint"] == response["seller_scope_fingerprint"]
    assert response["seller_boundary"]["authorization_fingerprint"]
    assert response["result"]["status"] == "ok"
    assert response["result"]["breakdown_result"]["total"]["order_count"] == 2


@pytest.mark.parametrize("scope_type", ["single", "selected", "all"])
def test_reminted_seller_metric_is_bound_to_canonical_metric_and_dimension_set(monkeypatch, scope_type):
    from app.auth.context import build_authorization_context
    from app.core.config import Settings
    from app.metrics.authorization import (
        AuthorizedMetric, authorize_metric, scope_fingerprint,
        validate_authorized_metric,
    )
    from app.mcp.client import query_scoped_business_data

    _enable_flags(monkeypatch)
    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: Settings(seller_dimension_enabled=True, seller_multi_scope_enabled=True),
    )
    if scope_type == "single":
        tenant = TenantContext(
            "tenant-test", "user-test", "analyst", "seller", seller_id="A",
            seller_grants=(SellerGrant("A", "2024-01-01", "2024-01-31"),),
        )
        seller_ids = ("A",)
        auth_scope = "seller"
    else:
        tenant = _platform_admin_with_grants("A", "B")
        seller_ids = ("A", "B") if scope_type == "selected" else ()
        auth_scope = "platform"
    _controlled, plan = _single_seller_mcp_plan(tenant)
    intent = SellerScopeIntent(
        seller_scope_type=scope_type, seller_ids=seller_ids,
        start_date=date(2024, 1, 1), end_date=date(2024, 1, 31),
        metric_ids=("gmv",), dimensions=("seller",),
    )
    signed_payload = _signed_scope_payload(monkeypatch, tenant, intent)
    connection = _FakeConnection()
    monkeypatch.setattr(
        explorer, "canonical_seller_query_transaction",
        lambda _engine, _scope: _fake_transaction(connection),
    )
    with activate_tenant_context(tenant):
        response = query_scoped_business_data(
            "按店铺拆解 GMV", signed_payload,
            plan=plan,
        )
    assert response["result"]["status"] == "ok", response["result"].get("reason")
    assert response["seller_boundary"]["query_scope_fingerprint"] == response["seller_scope_fingerprint"]
    seller_metric = AuthorizedMetric.model_validate(
        response["result"]["parsed"]["authorized_metric"]
    )
    authz = build_authorization_context(tenant, metric_scope_type=auth_scope)
    canonical_scope = resolve_seller_scope(intent, tenant=tenant, authz=authz)
    validation_scope = canonical_scope.as_query_dict() | {
        "tenant_id": tenant.tenant_id,
        "user_id": tenant.user_id,
        "permissions": tenant.role,
        "scope_type": auth_scope,
        "metric_name": "gmv",
        "allowed_metrics": list(canonical_scope.metric_ids),
        "allowed_dimensions": list(canonical_scope.dimensions),
        "authorization_fingerprint": authz.authz_digest,
        "permission_version": authz.permission_version,
        "policy_set_version": authz.policy_set_version,
        "authz_ref": authz.authz_ref,
    }
    assert validation_scope["allowed_metrics"] == ["gmv"]
    assert validation_scope["allowed_dimensions"] == ["seller"]
    assert seller_metric.query_scope_fingerprint == scope_fingerprint(
        validation_scope | {"scope_type": auth_scope}
    )
    validate_authorized_metric(
        seller_metric, scope=auth_scope, scope_data=validation_scope,
    )

    # A previously minted broad credential for another metric does not
    # validate against this canonical one-metric seller scope.
    broad_scope = validation_scope | {"allowed_metrics": list(authz.allowed_metric_ids)}
    outside_metric = authorize_metric(
        "order_count", provenance_type="exact", provenance_id="test:outside",
        scope=auth_scope, scope_data=broad_scope,
    )
    with pytest.raises(ValueError, match="scope fingerprint is stale|outside the server-authorized metric scope"):
        validate_authorized_metric(
            outside_metric, scope=auth_scope, scope_data=validation_scope,
        )


def test_structured_server_scope_payload_routes_selected_breakdown_end_to_end(monkeypatch):
    from app.core.config import Settings
    from app.mcp.client import query_scoped_business_data

    _enable_flags(monkeypatch)
    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: Settings(seller_dimension_enabled=True, seller_multi_scope_enabled=True),
    )
    tenant = _platform_admin_with_grants("A", "B")
    _controlled, plan = _single_seller_mcp_plan(tenant)
    intent = SellerScopeIntent(
        seller_scope_type="selected", seller_ids=("A", "B"),
        start_date=date(2024, 1, 1), end_date=date(2024, 1, 31),
        metric_ids=("gmv",), dimensions=("seller",),
    )
    signed_payload = _signed_scope_payload(monkeypatch, tenant, intent)
    connection = _FakeConnection()
    monkeypatch.setattr(
        explorer, "canonical_seller_query_transaction",
        lambda _engine, _scope: _fake_transaction(connection),
    )
    with activate_tenant_context(tenant):
        response = query_scoped_business_data(
            "按店铺拆解 GMV", signed_payload, plan=plan,
        )
    assert response["tool_name"] == "breakdown_scoped_seller"
    assert response["arguments"]["seller_scope_type"] == "selected"
    assert response["arguments"]["seller_ids"] == ["A", "B"]
    assert response["seller_boundary"]["query_scope_fingerprint"] == response["seller_scope_fingerprint"]
    assert response["result"]["status"] == "ok"
    assert response["result"]["breakdown_result"]["total"]["order_count"] == 2


def test_old_metric_and_boundary_do_not_authorize_new_seller_scope(monkeypatch):
    from app.auth.context import build_authorization_context
    from app.auth.enforcement import AuthorizationBoundary
    from app.core.config import Settings
    from app.metrics.authorization import validate_authorized_metric
    from app.mcp.client import query_scoped_business_data

    _enable_flags(monkeypatch)
    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: Settings(seller_dimension_enabled=True, seller_multi_scope_enabled=True),
    )
    grant = SellerGrant("A", "2024-01-01", "2024-01-31")
    tenant = TenantContext(
        "tenant-test", "user-test", "analyst", "seller", seller_id="A",
        seller_grants=(grant,),
    )
    controlled, plan = _single_seller_mcp_plan(tenant)
    authz = build_authorization_context(tenant, metric_scope_type="seller")
    intent = SellerScopeIntent(
        seller_scope_type="single", seller_ids=("A",),
        start_date=date(2024, 1, 1), end_date=date(2024, 1, 31),
        metric_ids=("gmv",), dimensions=("seller",), granularity="day",
    )
    seller_scope = resolve_seller_scope(intent, tenant=tenant, authz=authz)
    new_validation_scope = seller_scope.as_query_dict() | {
        "tenant_id": tenant.tenant_id, "user_id": tenant.user_id,
        "permissions": tenant.role, "scope_type": "seller",
        "metric_name": "gmv", "allowed_metrics": authz.allowed_metric_ids,
        "allowed_dimensions": authz.allowed_dimensions,
        "authorization_fingerprint": authz.authz_digest,
        "permission_version": authz.permission_version,
        "policy_set_version": authz.policy_set_version,
        "authz_ref": authz.authz_ref,
    }
    with pytest.raises(ValueError, match="fingerprint is stale"):
        validate_authorized_metric(plan.authorized_metric, scope="seller", scope_data=new_validation_scope)

    connection = _FakeConnection(
        items=[{"seller_id": "A", "value": 250, "group_order_count": 2}],
        total={"gmv": 250, "order_count": 2},
    )
    monkeypatch.setattr(
        explorer, "canonical_seller_query_transaction",
        lambda _engine, _scope: _fake_transaction(connection),
    )
    auth_fields = {
        "permission_version": authz.permission_version,
        "policy_set_version": authz.policy_set_version,
        "authorization_fingerprint": authz.authz_digest,
    }
    old_boundary = AuthorizationBoundary(
        **auth_fields, query_scope_fingerprint="legacy-single-scope-fingerprint",
    )
    with activate_tenant_context(tenant):
        rejected = query_scoped_business_data(
            "按店铺拆解 GMV", controlled.model_dump(mode="json"), plan=plan,
            authorization_boundary=old_boundary,
        )
    assert rejected["result"]["status"] == "unsupported"
    assert "scope fingerprint changed" in rejected["result"]["reason"]

    new_boundary = AuthorizationBoundary(
        **auth_fields, query_scope_fingerprint=seller_scope.query_scope_fingerprint,
    )
    with activate_tenant_context(tenant):
        accepted = query_scoped_business_data(
            "按店铺拆解 GMV", controlled.model_dump(mode="json"), plan=plan,
            authorization_boundary=new_boundary,
            current_query_scope_fingerprint=seller_scope.query_scope_fingerprint,
        )
    assert accepted["result"]["status"] == "ok", accepted["result"].get("reason")


def test_fixture_math_keeps_seller_order_counts_non_additive():
    rows = [
        ("O1", "A", 100), ("O1", "B", 200), ("O2", "A", 150),
        ("O3", "C", 300), ("O4", None, 50),
        ("O6", "A", 60), ("O6", None, 40),
    ]
    selected = {"A", "B"}
    base_rows = rows[:5]
    assert {seller: sum(row[2] for row in base_rows if row[1] == seller) for seller in ("A", "B", "C")} == {
        "A": 250, "B": 200, "C": 300,
    }
    assert sum(row[2] for row in base_rows if row[1] is not None) == 750
    assert len({row[0] for row in base_rows if row[1] in selected}) == 2
    selected_rows = [row for row in rows if row[1] in selected]
    assert sum(row[2] for row in selected_rows) == 510
    assert len({row[0] for row in selected_rows}) == 3
    assert sum(row[2] for row in rows if row[1] is None) == 90
    # Seller group counts overlap on O1; they are not a platform total.
    assert len({row[0] for row in rows if row[1] == "A"}) + len(
        {row[0] for row in rows if row[1] == "B"}
    ) == 4
    assert len({row[0] for row in rows if row[1] is not None}) == 4
    assert len({row[0] for row in rows}) == 5

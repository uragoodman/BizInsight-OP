from dataclasses import replace
from datetime import date

import pytest

from app.auth.context import AuthorizationContext, AuthorizationDenied
from app.auth.enforcement import AuthorizationBoundary, StaleAuthorization, assert_authorization_boundary
from app.scope.canonical import CanonicalScope, ScopeError
from app.tenancy import SellerGrant


def _authz() -> AuthorizationContext:
    return AuthorizationContext(
        tenant_id="tenant-a",
        user_id="user-a",
        role="analyst",
        scope_type="seller",
        permission_version="p1",
        policy_set_version="policy1",
        authz_ref="authz:p1",
        authz_digest="digest1",
        allowed_seller_ids=("seller-a", "seller-b"),
        seller_grants=(
            SellerGrant("seller-a", "2024-01-01", "2024-12-31"),
            SellerGrant("seller-b", "2024-06-01", "2024-06-30"),
        ),
        allowed_metric_ids=("gmv", "order_count"),
        allowed_dimensions=("product_category",),
        allowed_actions=("read", "analyze"),
        allowed_tools=("query_gmv",),
    )


def test_scope_is_a_strict_subset_of_authorization() -> None:
    scope = CanonicalScope.resolve(
        _authz(),
        seller_ids=["seller-a"],
        start_date=date(2024, 2, 1),
        end_date=date(2024, 2, 29),
        metric_ids=["gmv"],
        dimensions=["product_category"],
    )
    query = scope.narrow(metric_ids=["gmv"], dimensions=["product_category"])
    assert query.seller_ids == ("seller-a",)
    assert query.query_scope_fingerprint
    assert query.permission_version == "p1"
    assert query.actions == ("analyze",)


def test_scope_cannot_expand_seller_metric_or_date() -> None:
    scope = CanonicalScope.resolve(
        _authz(),
        seller_ids=["seller-a"],
        start_date="2024-02-01",
        end_date="2024-02-29",
        metric_ids=["gmv"],
    )
    with pytest.raises(ScopeError, match="expand seller"):
        scope.narrow(seller_ids=["seller-b"])
    with pytest.raises(ScopeError, match="expand metric"):
        scope.narrow(metric_ids=["order_count"])
    with pytest.raises(ScopeError, match="expand date"):
        scope.narrow(start_date="2024-01-01")


def test_scope_carries_action_and_metric_definition_versions() -> None:
    scope = CanonicalScope.resolve(
        _authz(),
        seller_ids=["seller-a"],
        start_date="2024-02-01",
        end_date="2024-02-29",
        metric_ids=["gmv"],
        actions=["analyze"],
        metric_definition_versions={"gmv": "3"},
    )
    assert scope.actions == ("analyze",)
    assert scope.metric_definition_versions == (("gmv", "3"),)
    query = scope.narrow(actions=["analyze"], metric_definition_versions={"gmv": "3"})
    assert query.as_dict()["actions"] == ["analyze"]
    with pytest.raises(ScopeError, match="actions"):
        scope.narrow(actions=["read"])


def test_scope_rejects_unlisted_dimension_and_date_entitlement() -> None:
    with pytest.raises(ScopeError, match="not authorized"):
        CanonicalScope.resolve(
            _authz(),
            seller_ids=["seller-a"],
            start_date="2024-02-01",
            end_date="2024-02-29",
            metric_ids=["gmv"],
            dimensions=["customer_city"],
        )
    with pytest.raises(ScopeError, match="after seller entitlement"):
        CanonicalScope.resolve(
            _authz(),
            seller_ids=["seller-b"],
            start_date="2024-06-01",
            end_date="2024-07-01",
            metric_ids=["gmv"],
        )


def test_authorization_checks_fail_closed() -> None:
    with pytest.raises(AuthorizationDenied) as exc:
        _authz().require_tool("delete_everything")
    assert exc.value.code == "TOOL_DENIED"


def test_security_boundary_rejects_permission_or_scope_changes() -> None:
    authz = _authz()
    boundary = AuthorizationBoundary(
        permission_version=authz.permission_version,
        policy_set_version=authz.policy_set_version,
        authorization_fingerprint=authz.authz_digest,
        query_scope_fingerprint="q1",
    )
    assert_authorization_boundary(authz, boundary, current_scope_fingerprint="q1")
    with pytest.raises(StaleAuthorization):
        assert_authorization_boundary(authz, boundary, current_scope_fingerprint="q2")

    stale = replace(authz, permission_version="revoked")
    with pytest.raises(StaleAuthorization):
        assert_authorization_boundary(stale, boundary, current_scope_fingerprint="q1")


def test_rag_boundary_rechecks_current_scope_without_model_call() -> None:
    from app.rag.retriever import retrieve_knowledge
    from app.tenancy import TenantContext

    authz = _authz()
    context = TenantContext(
        tenant_id=authz.tenant_id,
        user_id=authz.user_id,
        role=authz.role,
        scope_type=authz.scope_type,
        seller_grants=authz.seller_grants,
    )
    boundary = {
        "permission_version": authz.permission_version,
        "policy_set_version": authz.policy_set_version,
        "authorization_fingerprint": authz.authz_digest,
        "query_scope_fingerprint": "q1",
        "execution_id": "ignored-lifecycle-field",
    }
    with pytest.raises(StaleAuthorization):
        retrieve_knowledge(
            "指标定义",
            tenant_context=context,
            authorization_context=authz,
            authorization_boundary=boundary,
            current_scope_fingerprint="q2",
            backend_override="local_hybrid",
        )

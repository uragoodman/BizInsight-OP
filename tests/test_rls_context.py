from datetime import date

import pytest

from app.data.rls import set_local_context
from app.tenancy import SellerGrant, TenantContext


class FakeConnection:
    def __init__(self) -> None:
        self.statements = []

    def execute(self, statement, params):
        self.statements.append((str(statement), params))


def test_rls_context_is_transaction_local_and_contains_identity() -> None:
    connection = FakeConnection()
    context = TenantContext(
        tenant_id="tenant-a",
        user_id="user-a",
        role="analyst",
        scope_type="seller",
        seller_grants=(SellerGrant("seller-a"),),
    )

    set_local_context(connection, context)

    assert len(connection.statements) == 4
    assert all("true" in sql for sql, _ in connection.statements)
    values = {params["setting"]: params["value"] for _, params in connection.statements}
    assert values == {
        "app.tenant_id": "tenant-a",
        "app.user_id": "user-a",
        "app.role": "analyst",
        "app.scope_type": "seller",
    }


class _Rows:
    def __init__(self, row):
        self.row = row

    def mappings(self):
        return self

    def first(self):
        return self.row


class _AuthorizationLeaseConnection(FakeConnection):
    def execute(self, statement, params):
        sql = str(statement)
        self.statements.append((sql, params))
        if "set_config" in sql:
            return _Rows(None)
        if "FROM iam.tenants" in sql:
            return _Rows({"status": "active", "scope_type": "seller"})
        if "FROM iam.users" in sql:
            return _Rows({"status": "active"})
        if "FROM iam.tenant_memberships" in sql:
            return _Rows({"role": "analyst"})
        if "FROM iam.membership_seller_grants" in sql:
            return _Rows({"access_start_date": date(2024, 1, 1), "access_end_date": date(2024, 12, 31)})
        raise AssertionError(f"unexpected SQL: {sql}")


def test_seller_query_holds_authorization_rows_for_transaction() -> None:
    connection = _AuthorizationLeaseConnection()
    context = TenantContext(
        tenant_id="tenant-a",
        user_id="user-a",
        role="analyst",
        scope_type="seller",
        seller_grants=(SellerGrant("seller-a"),),
    )

    set_local_context(
        connection,
        context,
        seller_access={"seller-a": (date(2024, 2, 1), date(2024, 3, 31))},
    )

    lock_queries = [sql for sql, _ in connection.statements if "FROM iam." in sql]
    assert len(lock_queries) == 4
    assert all("FOR SHARE" in sql for sql in lock_queries)
    grant_params = next(
        params for sql, params in connection.statements
        if "FROM iam.membership_seller_grants" in sql
    )
    assert grant_params["seller_id"] == "seller-a"


def test_platform_admin_can_access_explicit_seller_scope_without_seller_grant() -> None:
    context = TenantContext(
        tenant_id="platform",
        user_id="platform-admin",
        role="admin",
        scope_type="platform",
    )

    class PlatformConnection(_AuthorizationLeaseConnection):
        def execute(self, statement, params):
            sql = str(statement)
            if "FROM iam.tenants" in sql:
                self.statements.append((sql, params))
                return _Rows({"status": "active", "scope_type": "platform"})
            if "FROM iam.users" in sql:
                self.statements.append((sql, params))
                return _Rows({"status": "active"})
            if "FROM iam.tenant_memberships" in sql:
                self.statements.append((sql, params))
                return _Rows({"role": "admin"})
            return super().execute(statement, params)

    connection = PlatformConnection()
    set_local_context(
        connection,
        context,
        scope_type="platform",
        seller_access={"seller-a": (date(2024, 2, 1), date(2024, 3, 31))},
    )

    values = {
        params["setting"]: params["value"]
        for sql, params in connection.statements
        if "set_config" in sql
    }
    assert values["app.scope_type"] == "platform"
    assert not any("membership_seller_grants" in sql for sql, _ in connection.statements)


def test_seller_query_fails_closed_when_grant_is_missing_or_stale() -> None:
    class RevokedGrantConnection(_AuthorizationLeaseConnection):
        def execute(self, statement, params):
            if "FROM iam.membership_seller_grants" in str(statement):
                self.statements.append((str(statement), params))
                return _Rows(None)
            return super().execute(statement, params)

    context = TenantContext(
        tenant_id="tenant-a",
        user_id="user-a",
        role="analyst",
        scope_type="seller",
        seller_grants=(SellerGrant("seller-a"),),
    )
    with pytest.raises(PermissionError, match="Seller grant was revoked"):
        set_local_context(
            RevokedGrantConnection(),
            context,
            seller_access={"seller-a": (date(2024, 2, 1), date(2024, 3, 31))},
        )

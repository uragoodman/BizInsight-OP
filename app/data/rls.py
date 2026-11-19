"""Transaction-local PostgreSQL tenant context.

The third argument of set_config is true, giving the same transaction scope as
SET LOCAL while still allowing safely bound values.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from typing import Any, Iterator, Mapping

from sqlalchemy import text

from app.tenancy import TenantContext


def set_local_context(
    connection: Any,
    context: TenantContext,
    *,
    scope_type: str | None = None,
    allow_sensitive: bool | None = None,
    seller_access: Mapping[str, tuple[date, date]] | None = None,
    require_explicit_seller_grants: bool = False,
) -> None:
    if context is None:
        raise PermissionError("TenantContext is required before database access")
    values = {
        "app.tenant_id": context.tenant_id,
        "app.user_id": context.user_id,
        "app.role": context.role,
        "app.scope_type": scope_type or context.scope_type,
        # Database-native masking is opt-in per transaction.  A role alone
        # never reveals raw identifiers; callers must pass an explicit,
        # separately authorized data-governance decision.
    }
    if allow_sensitive is not None:
        values["app.allow_sensitive"] = "true" if allow_sensitive else "false"
    for setting, value in values.items():
        connection.execute(
            text("SELECT set_config(:setting, :value, true)"),
            {"setting": setting, "value": str(value)},
        )

    if seller_access:
        _lock_authorization_rows(
            connection, context, seller_access,
            require_explicit_seller_grants=require_explicit_seller_grants,
        )


def _lock_authorization_rows(
    connection: Any,
    context: TenantContext,
    seller_access: Mapping[str, tuple[date, date]],
    *,
    require_explicit_seller_grants: bool = False,
) -> None:
    """Hold the authorization rows stable for the lifetime of this query transaction.

    This gives seller-grant, membership, tenant, session, and API-key revocation
    a clear linearization point: a revoke committed before this check is denied;
    a revoke arriving after the locks are acquired waits for the bounded data
    transaction to finish.  Callers must execute the protected SELECT in this
    same transaction (normally via ``engine.begin()``).
    """
    if not seller_access:
        raise PermissionError("Seller-scoped database access requires an explicit seller/date scope")
    _lock_active_identity(connection, context)
    is_seller = context.scope_type == "seller"
    is_platform_admin = context.is_platform_admin
    if not (is_seller or is_platform_admin):
        raise PermissionError("Seller data access requires a seller grant or platform administrator identity")

    for seller_id, (requested_start, requested_end) in sorted(seller_access.items()):
        if requested_end < requested_start:
            raise PermissionError("Seller query date range is invalid")
        if is_platform_admin and context.seller_grant(seller_id) is None and not require_explicit_seller_grants:
            # Platform administrators are authorized through their active
            # platform membership when no seller grant is attached. If an
            # explicit seller grant exists, lock and revalidate it below.
            continue
        grant = connection.execute(text("""
            SELECT access_start_date, access_end_date
            FROM iam.membership_seller_grants
            WHERE tenant_id=:tenant_id AND user_id=:user_id AND seller_id=:seller_id
              AND revoked_at IS NULL
            FOR SHARE
        """), {
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
            "seller_id": str(seller_id),
        }).mappings().first()
        if grant is None:
            raise PermissionError("Seller grant was revoked before the database query")
        grant_start = grant["access_start_date"]
        grant_end = grant["access_end_date"]
        if (grant_start is not None and requested_start < grant_start) or (
            grant_end is not None and requested_end > grant_end
        ):
            raise PermissionError("Seller grant date range changed before the database query")


def _lock_active_identity(connection: Any, context: TenantContext) -> None:
    """Lock the active tenant/user/membership and credential rows for this query."""
    tenant = connection.execute(text("""
        SELECT status, scope_type FROM iam.tenants
        WHERE tenant_id=:tenant_id
        FOR SHARE
    """), {"tenant_id": context.tenant_id}).mappings().first()
    if (
        tenant is None
        or tenant["status"] != "active"
        or tenant["scope_type"] != context.scope_type
    ):
        raise PermissionError("Tenant is missing or inactive at the database access boundary")

    user = connection.execute(text("""
        SELECT status FROM iam.users
        WHERE user_id=:user_id
        FOR SHARE
    """), {"user_id": context.user_id}).mappings().first()
    if user is None or user["status"] != "active":
        raise PermissionError("User is missing or inactive at the database access boundary")

    membership = connection.execute(text("""
        SELECT role FROM iam.tenant_memberships
        WHERE tenant_id=:tenant_id AND user_id=:user_id AND revoked_at IS NULL
        FOR SHARE
    """), {"tenant_id": context.tenant_id, "user_id": context.user_id}).mappings().first()
    if membership is None or membership["role"] != context.role:
        raise PermissionError("Tenant membership changed before the database query")

    if context.auth_session_id:
        session = connection.execute(text("""
            SELECT 1 FROM iam.auth_sessions
            WHERE session_id=:session_id AND tenant_id=:tenant_id AND user_id=:user_id
              AND revoked_at IS NULL AND expires_at > now()
            FOR SHARE
        """), {
            "session_id": context.auth_session_id,
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
        }).first()
        if session is None:
            raise PermissionError("Login session was revoked or expired before the database query")

    if context.auth_credential_id:
        credential = connection.execute(text("""
            SELECT 1 FROM iam.api_credentials
            WHERE credential_id=:credential_id AND tenant_id=:tenant_id AND user_id=:user_id
              AND revoked_at IS NULL AND (expires_at IS NULL OR expires_at > now())
            FOR SHARE
        """), {
            "credential_id": context.auth_credential_id,
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
        }).first()
        if credential is None:
            raise PermissionError("API credential was revoked or expired before the database query")

@contextmanager
def seller_query_transaction(
    engine: Any,
    scope: Any,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> Iterator[Any]:
    """Open a scoped query transaction and lease its current seller authorization."""
    from app.tenancy import current_tenant_context

    tenant = current_tenant_context()
    if tenant is None:
        raise PermissionError("Trusted tenant context is required for seller-scoped database access")
    effective_start = start_date or scope.default_start_date
    effective_end = end_date or scope.default_end_date
    if effective_start < scope.default_start_date or effective_end > scope.default_end_date:
        raise PermissionError("Seller query date range exceeds the locked analysis scope")
    if effective_end < effective_start:
        raise PermissionError("Seller query date range is invalid")
    from app.data.report_snapshot import current_report_snapshot_connection

    shared_connection = current_report_snapshot_connection()
    if shared_connection is not None:
        database_scope_type = "platform" if tenant.is_platform_admin else "seller"
        set_local_context(
            shared_connection,
            tenant,
            scope_type=database_scope_type,
            seller_access={scope.seller_id: (effective_start, effective_end)},
        )
        yield shared_connection
        return

    with engine.begin() as connection:
        database_scope_type = "platform" if tenant.is_platform_admin else "seller"
        set_local_context(
            connection,
            tenant,
            scope_type=database_scope_type,
            seller_access={scope.seller_id: (effective_start, effective_end)},
        )
        yield connection


@contextmanager
def canonical_seller_query_transaction(engine: Any, scope: Any) -> Iterator[Any]:
    """Open a bounded transaction for a server-resolved single/selected/ALL scope."""
    from app.data.report_snapshot import current_report_snapshot_connection
    from app.tenancy import current_tenant_context

    tenant = current_tenant_context()
    if tenant is None:
        raise PermissionError("Trusted TenantContext is required for seller breakdown")
    scope_type = getattr(scope, "seller_scope_type", None)
    if scope_type not in {"single", "selected", "all"}:
        raise PermissionError("A canonical seller scope is required")
    if scope_type in {"selected", "all"} and not tenant.is_platform_admin:
        raise PermissionError("Selected and ALL seller access requires a platform administrator")
    if scope_type == "all" and not tenant.is_platform_admin:
        raise PermissionError("ALL seller access requires a platform administrator")

    seller_access = {
        str(seller_id): (scope.start_date, scope.end_date)
        for seller_id in scope.seller_ids
    }
    database_scope_type = "platform" if tenant.is_platform_admin else "seller"

    def configure(connection: Any) -> None:
        set_local_context(
            connection, tenant, scope_type=database_scope_type,
            seller_access=seller_access or None,
            require_explicit_seller_grants=scope.seller_scope_type == "selected",
        )
        if scope.seller_scope_type == "all":
            # ALL has no finite seller grant rows to lock. Hold the same
            # active identity rows stable for the data SELECT instead.
            _lock_active_identity(connection, tenant)

    shared = current_report_snapshot_connection()
    if shared is not None:
        configure(shared)
        yield shared
        return
    with engine.begin() as connection:
        configure(connection)
        yield connection

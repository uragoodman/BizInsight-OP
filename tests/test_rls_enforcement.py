"""RLS 强制测试：非 superuser 角色下，行级安全真正按租户拦截。

需要 ``bizinsight_app`` 角色（由 scripts/setup_rls_role.py 创建并授权）；缺失时跳过。
每个用例用独立连接 + finally RESET ROLE，避免角色泄漏到连接池。
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from app.data.database import engine
from app.data.rls import set_local_context
from app.tenancy import SellerGrant, TenantContext

RLS_ROLE = "bizinsight_app"


def _role_provisioned() -> bool:
    try:
        with engine.connect() as connection:
            return bool(connection.execute(
                text("SELECT 1 FROM pg_roles WHERE rolname=:r"), {"r": RLS_ROLE}
            ).scalar_one_or_none())
    except SQLAlchemyError:
        # RLS tests are integration-only. An unavailable database should skip
        # them rather than abort pytest collection for the whole repository.
        return False


@pytest.mark.skipif(
    not _role_provisioned(),
    reason="database unavailable or bizinsight_app role not provisioned",
)
class TestRlsEnforcement:
    def test_cross_tenant_rows_are_invisible(self) -> None:
        connection = engine.connect()
        try:
            with connection.begin():
                connection.execute(text("SET ROLE bizinsight_app"))
                connection.execute(text("SELECT set_config('app.scope_type', 'seller', true)"))
                connection.execute(text("SELECT set_config('app.tenant_id', 'dv-a-1598be', true)"))
                seen_a = connection.execute(
                    text("SELECT DISTINCT tenant_id FROM analytics.tenant_fact_order_items")
                ).scalars().all()
                assert seen_a == ["dv-a-1598be"], seen_a
                connection.execute(text("SELECT set_config('app.tenant_id', 'dv-a-173be4', true)"))
                seen_b = connection.execute(
                    text("SELECT DISTINCT tenant_id FROM analytics.tenant_fact_order_items")
                ).scalars().all()
                assert seen_b == ["dv-a-173be4"], seen_b
        finally:
            connection.execute(text("RESET ROLE"))
            connection.close()

    def test_unset_context_is_fail_closed(self) -> None:
        connection = engine.connect()
        try:
            with connection.begin():
                connection.execute(text("SET ROLE bizinsight_app"))
                # 不设置 app.tenant_id：策略恒不成立，必须返回 0 行（宁可误伤不可泄露）。
                count = connection.execute(
                    text("SELECT COUNT(*) FROM analytics.tenant_fact_order_items")
                ).scalar_one()
                assert count == 0
        finally:
            connection.execute(text("RESET ROLE"))
            connection.close()

    def test_platform_scope_sees_all_tenants(self) -> None:
        connection = engine.connect()
        try:
            with connection.begin():
                connection.execute(text("SET ROLE bizinsight_app"))
                connection.execute(text("SELECT set_config('app.scope_type', 'platform', true)"))
                tenants = connection.execute(
                    text("SELECT COUNT(DISTINCT tenant_id) FROM analytics.tenant_fact_order_items")
                ).scalar_one()
                assert tenants > 1, tenants
        finally:
            connection.execute(text("RESET ROLE"))
            connection.close()

    def test_transaction_local_tenant_context_does_not_leak_after_pool_reuse(self) -> None:
        """A returned DBAPI connection must not carry tenant A into tenant B's request."""
        # An isolated one-connection pool makes the check-in/check-out reuse
        # deterministic and prevents unrelated tests from selecting a
        # different idle connection. No rows are written by this test.
        probe_engine = create_engine(
            engine.url,
            future=True,
            pool_size=1,
            max_overflow=0,
            pool_pre_ping=True,
        )
        tenant_a = "dv-a-1598be"
        tenant_b = "dv-a-173be4"
        context_a = TenantContext(
            tenant_id=tenant_a, user_id="pool-reuse-user-a", role="analyst",
            scope_type="seller", seller_grants=(SellerGrant("test-seller-a"),),
        )
        context_b = TenantContext(
            tenant_id=tenant_b, user_id="pool-reuse-user-b", role="analyst",
            scope_type="seller", seller_grants=(SellerGrant("test-seller-b"),),
        )
        try:
            with probe_engine.connect() as connection:
                with connection.begin():
                    connection.execute(text("SET LOCAL ROLE bizinsight_app"))
                    set_local_context(connection, context_a, scope_type="seller")
                    pid_a = connection.execute(text("SELECT pg_backend_pid()")).scalar_one()
                    visible_a = connection.execute(text(
                        "SELECT DISTINCT tenant_id FROM analytics.tenant_fact_order_items"
                    )).scalars().all()
                    assert visible_a == [tenant_a], visible_a

            # Closing the SQLAlchemy connection returns the same DBAPI
            # connection to the one-slot pool. Transaction-local GUCs and role
            # must have been reset before this next checkout.
            with probe_engine.connect() as connection:
                with connection.begin():
                    connection.execute(text("SET LOCAL ROLE bizinsight_app"))
                    pid_b = connection.execute(text("SELECT pg_backend_pid()")).scalar_one()
                    assert pid_b == pid_a, "test did not reuse the same PostgreSQL backend"
                    no_context_count = connection.execute(text(
                        "SELECT COUNT(*) FROM analytics.tenant_fact_order_items"
                    )).scalar_one()
                    assert no_context_count == 0, no_context_count

                    set_local_context(connection, context_b, scope_type="seller")
                    visible_b = connection.execute(text(
                        "SELECT DISTINCT tenant_id FROM analytics.tenant_fact_order_items"
                    )).scalars().all()
                    assert visible_b == [tenant_b], visible_b
        finally:
            probe_engine.dispose()

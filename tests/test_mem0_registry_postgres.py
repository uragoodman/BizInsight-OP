"""Real PostgreSQL WP4-Registry prerequisite check for WP3.

Requires a migrated disposable database (Alembic >= 20261030_57).  The test
only inserts and removes its own synthetic tenant/user/registry rows.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text

from app.memory.preference_registry import PostgresPreferenceRegistry
from app.tenancy import TenantContext


pytestmark = pytest.mark.skipif(
    not (os.getenv("WP3_REGISTRY_TEST_DATABASE_URL") or os.getenv("POSTGRES_URL")),
    reason="Set WP3_REGISTRY_TEST_DATABASE_URL to a migrated disposable PostgreSQL database",
)


def test_registry_active_boundary_against_real_postgres():
    engine = create_engine(os.getenv("WP3_REGISTRY_TEST_DATABASE_URL") or os.environ["POSTGRES_URL"])
    suffix = uuid.uuid4().hex[:10]
    tenant = f"wp3-reg-{suffix}"
    user = f"wp3-reg-user-{suffix}"
    context = TenantContext(tenant_id=tenant, user_id=user, role="admin", scope_type="platform")
    registry = PostgresPreferenceRegistry(context, engine_override=engine)
    try:
        with engine.begin() as connection:
            connection.execute(text("""
                INSERT INTO iam.tenants(tenant_id, tenant_name, scope_type, status)
                VALUES (:tenant, 'wp3 registry test', 'platform', 'active')
            """), {"tenant": tenant})
            connection.execute(text("""
                INSERT INTO iam.users(user_id, display_name, email, status)
                VALUES (:user, 'wp3 registry user', :email, 'active')
            """), {"user": user, "email": f"{user}@example.invalid"})
            connection.execute(text("""
                INSERT INTO iam.tenant_memberships(tenant_id, user_id, role)
                VALUES (:tenant, :user, 'admin')
            """), {"tenant": tenant, "user": user})
            connection.execute(text("""
                INSERT INTO governance.memory_registry
                    (registry_id, tenant_id, user_id, seller_id, memory_type,
                     memory_key, mem0_memory_id, status, version, operation_id,
                     user_confirmed)
                VALUES (:registry_id, :tenant, :user, NULL, 'presentation',
                        'response_style', 'mem0-real', 'active', 1, :operation_id, true)
            """), {
                "registry_id": f"mreg_{suffix}", "tenant": tenant,
                "user": user, "operation_id": f"op-{suffix}",
            })

        registry.ensure_available()
        active = registry.validate(
            context=context, entity_scope="mem0_scope", memory_id="mem0-real",
            operation_id=f"op-{suffix}", memory_type="presentation",
            memory_key="response_style",
        )
        assert active.active is True

        # Both identifiers must identify the same Registry row. A duplicate
        # Mem0 point carrying this operation_id must not become authoritative
        # merely because its operation_id matches.
        unclaimed_duplicate = registry.validate(
            context=context, entity_scope="mem0_scope", memory_id="mem0-duplicate",
            operation_id=f"op-{suffix}", memory_type="presentation",
            memory_key="response_style",
        )
        assert unclaimed_duplicate.active is False
        assert unclaimed_duplicate.reason == "registry_record_missing"

        mismatched_operation = registry.validate(
            context=context, entity_scope="mem0_scope", memory_id="mem0-real",
            operation_id=f"op-other-{suffix}", memory_type="presentation",
            memory_key="response_style",
        )
        assert mismatched_operation.active is False
        assert mismatched_operation.reason == "registry_record_missing"

        with engine.begin() as connection:
            connection.execute(text("""
                UPDATE governance.memory_registry SET status='superseded'
                WHERE registry_id=:registry_id
            """), {"registry_id": f"mreg_{suffix}"})
        inactive = registry.validate(
            context=context, entity_scope="mem0_scope", memory_id="mem0-real",
            operation_id=f"op-{suffix}", memory_type="presentation",
            memory_key="response_style",
        )
        assert inactive.active is False
        assert inactive.reason == "registry_not_active"

        with engine.begin() as connection:
            connection.execute(text("""
                UPDATE governance.memory_registry
                   SET status='active',
                       expires_at=:expired
                 WHERE registry_id=:registry_id
            """), {
                "expired": datetime.now(timezone.utc) - timedelta(seconds=1),
                "registry_id": f"mreg_{suffix}",
            })
        expired = registry.validate(
            context=context, entity_scope="mem0_scope", memory_id="mem0-real",
            operation_id=f"op-{suffix}", memory_type="presentation",
            memory_key="response_style",
        )
        assert expired.active is False
        assert expired.reason == "registry_expired"
    finally:
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM governance.memory_registry WHERE tenant_id=:tenant"), {"tenant": tenant})
            connection.execute(text("DELETE FROM iam.tenant_memberships WHERE tenant_id=:tenant"), {"tenant": tenant})
            connection.execute(text("DELETE FROM iam.users WHERE user_id=:user"), {"user": user})
            connection.execute(text("DELETE FROM iam.tenants WHERE tenant_id=:tenant"), {"tenant": tenant})
        engine.dispose()


def test_memory_registry_rls_isolates_users_under_non_bypass_role():
    """The table policy must enforce user isolation, not only SQL predicates."""
    from sqlalchemy.exc import SQLAlchemyError

    engine = create_engine(os.getenv("WP3_REGISTRY_TEST_DATABASE_URL") or os.environ["POSTGRES_URL"])
    suffix = uuid.uuid4().hex[:10]
    tenant = f"wp3-rls-{suffix}"
    user_a = f"wp3-rls-a-{suffix}"
    user_b = f"wp3-rls-b-{suffix}"
    role = f"wp3_memreg_{suffix}"
    rows = [
        (user_a, f"mreg-a-{suffix}", f"op-a-{suffix}"),
        (user_b, f"mreg-b-{suffix}", f"op-b-{suffix}"),
    ]
    try:
        with engine.begin() as connection:
            connection.execute(text(f'CREATE ROLE "{role}" NOLOGIN NOSUPERUSER NOBYPASSRLS'))
            connection.execute(text(f'GRANT USAGE ON SCHEMA governance TO "{role}"'))
            connection.execute(text(f'GRANT SELECT ON governance.memory_registry TO "{role}"'))
            connection.execute(text("""
                INSERT INTO iam.tenants(tenant_id, tenant_name, scope_type, status)
                VALUES (:tenant, 'wp3 memory RLS test', 'platform', 'active')
            """), {"tenant": tenant})
            for user, registry_id, operation_id in rows:
                connection.execute(text("""
                    INSERT INTO iam.users(user_id, display_name, email, status)
                    VALUES (:user, 'wp3 registry RLS user', :email, 'active')
                """), {"user": user, "email": f"{user}@example.invalid"})
                connection.execute(text("""
                    INSERT INTO iam.tenant_memberships(tenant_id, user_id, role)
                    VALUES (:tenant, :user, 'admin')
                """), {"tenant": tenant, "user": user})
                connection.execute(text("""
                    INSERT INTO governance.memory_registry
                        (registry_id, tenant_id, user_id, seller_id, memory_type,
                         memory_key, mem0_memory_id, status, version, operation_id,
                         user_confirmed)
                    VALUES (:registry_id, :tenant, :user, NULL, 'presentation',
                            'response_style', :memory_id, 'active', 1, :operation_id, true)
                """), {
                    "registry_id": registry_id, "tenant": tenant, "user": user,
                    "memory_id": f"mem0-{user}", "operation_id": operation_id,
                })

        # This connection is intentionally privileged only to SET ROLE. The
        # effective query role is non-superuser and has no BYPASSRLS attribute.
        with engine.connect() as connection:
            with connection.begin():
                connection.execute(text(f'SET LOCAL ROLE "{role}"'))
                for user, expected_id in ((user_a, f"mreg-a-{suffix}"), (user_b, f"mreg-b-{suffix}")):
                    connection.execute(text("SELECT set_config('app.tenant_id', :tenant, true)"), {"tenant": tenant})
                    connection.execute(text("SELECT set_config('app.user_id', :user, true)"), {"user": user})
                    visible = connection.execute(text(
                        "SELECT registry_id FROM governance.memory_registry ORDER BY registry_id"
                    )).scalars().all()
                    assert visible == [expected_id]

                connection.execute(text("SELECT set_config('app.user_id', '', true)"))
                assert connection.execute(text(
                    "SELECT COUNT(*) FROM governance.memory_registry"
                )).scalar_one() == 0
    except SQLAlchemyError as exc:
        pytest.skip(f"isolated database lacks role/privilege setup for RLS probe: {exc}")
    finally:
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM governance.memory_registry WHERE tenant_id=:tenant"), {"tenant": tenant})
            connection.execute(text("DELETE FROM iam.tenant_memberships WHERE tenant_id=:tenant"), {"tenant": tenant})
            connection.execute(text("DELETE FROM iam.users WHERE user_id IN (:user_a, :user_b)"), {"user_a": user_a, "user_b": user_b})
            connection.execute(text("DELETE FROM iam.tenants WHERE tenant_id=:tenant"), {"tenant": tenant})
            role_exists = connection.execute(
                text("SELECT 1 FROM pg_roles WHERE rolname=:role"), {"role": role}
            ).scalar_one_or_none()
            if role_exists:
                connection.execute(text(f'REVOKE SELECT ON governance.memory_registry FROM "{role}"'))
                connection.execute(text(f'REVOKE USAGE ON SCHEMA governance FROM "{role}"'))
                connection.execute(text(f'DROP ROLE "{role}"'))
        engine.dispose()

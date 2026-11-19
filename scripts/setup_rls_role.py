"""Create the non-superuser application role used for RLS enforcement.

The API normally connects as ``postgres`` (superuser), which bypasses row-level
security. This script provisions ``bizinsight_app`` (NOLOGIN, no bypassrls) and
grants business-schema access so the application can ``SET ROLE bizinsight_app``
per connection, making the configured RLS policies actually enforce tenant
isolation. Idempotent: safe to run repeatedly.

用法：python -m scripts.setup_rls_role
"""
from __future__ import annotations

from sqlalchemy import create_engine, text

from app.core.config import get_settings

ROLE = "bizinsight_app"
LIFECYCLE_APP_ROLE = "bizinsight_lifecycle_app"
LIFECYCLE_LOGIN_ROLE = "bizinsight_lifecycle_login"
READ_SCHEMAS = ("raw",)
RW_SCHEMAS = ("analytics", "governance", "iam")


def main() -> None:
    # Do not import ``app.data.database.engine`` here.  That engine may have
    # DB_USE_RLS_ROLE=true and its checkout hook would try to SET ROLE before
    # this bootstrap script has created the role.  Bootstrap with a direct
    # administrative connection, then let normal application connections use
    # the RLS-bound role.
    settings = get_settings()
    admin_engine = create_engine(
        settings.postgres_url,
        future=True,
        pool_pre_ping=True,
        connect_args={"connect_timeout": int(settings.database_timeout_seconds)},
    )
    with admin_engine.begin() as conn:
        conn.execute(text(f"""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{ROLE}') THEN
                    CREATE ROLE {ROLE} NOLOGIN;  -- NOLOGIN: 仅通过 SET ROLE 使用
                END IF;
            END
            $$"""))
        # 角色显式不 bypass RLS（默认即 false），确保策略生效。
        conn.execute(text(f"ALTER ROLE {ROLE} NOBYPASSRLS"))
        for schema in (*READ_SCHEMAS, *RW_SCHEMAS):
            conn.execute(text(f"GRANT USAGE ON SCHEMA {schema} TO {ROLE}"))
        conn.execute(text(f"GRANT SELECT ON ALL TABLES IN SCHEMA {', '.join(READ_SCHEMAS)} TO {ROLE}"))
        conn.execute(text(f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {', '.join(RW_SCHEMAS)} TO {ROLE}"))
        conn.execute(text(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA {', '.join((*READ_SCHEMAS, *RW_SCHEMAS))} TO {ROLE}"))
        for schema in RW_SCHEMAS:
            conn.execute(text(f"ALTER DEFAULT PRIVILEGES IN SCHEMA {schema} GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {ROLE}"))
        conn.execute(text(f"""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{LIFECYCLE_APP_ROLE}') THEN
                    CREATE ROLE {LIFECYCLE_APP_ROLE} NOLOGIN NOBYPASSRLS;
                END IF;
            END
            $$"""))
        conn.execute(text(f"ALTER ROLE {LIFECYCLE_APP_ROLE} NOLOGIN NOBYPASSRLS"))
        # Reconcile this dedicated role's ACL on every run; a previous/manual
        # grant must not silently widen the lifecycle worker's authority.
        conn.execute(text(
            f"REVOKE ALL ON ALL TABLES IN SCHEMA raw, analytics, governance, iam FROM {LIFECYCLE_APP_ROLE}"
        ))
        conn.execute(text(
            f"REVOKE ALL ON ALL SEQUENCES IN SCHEMA raw, analytics, governance, iam FROM {LIFECYCLE_APP_ROLE}"
        ))
        conn.execute(text(
            f"REVOKE ALL ON SCHEMA raw, analytics, governance, iam FROM {LIFECYCLE_APP_ROLE}"
        ))
        conn.execute(text(f"GRANT USAGE ON SCHEMA governance, iam TO {LIFECYCLE_APP_ROLE}"))
        # IAM roster/context reads are column-scoped: no password, email, or
        # unrelated membership data is available to the periodic reconciler.
        conn.execute(text(f"""
            GRANT SELECT (tenant_id, scope_type, status) ON iam.tenants TO {LIFECYCLE_APP_ROLE};
            GRANT SELECT (user_id, status) ON iam.users TO {LIFECYCLE_APP_ROLE};
            GRANT SELECT (tenant_id, user_id, role, revoked_at)
              ON iam.tenant_memberships TO {LIFECYCLE_APP_ROLE};
            GRANT SELECT (tenant_id, user_id, seller_id, access_start_date,
                          access_end_date, revoked_at)
              ON iam.membership_seller_grants TO {LIFECYCLE_APP_ROLE}
        """))
        # Only lifecycle-owned rows are queryable/updatable, always subject to
        # their existing tenant/user RLS policies. No raw or analytics access.
        conn.execute(text(f"""
            GRANT SELECT (execution_id, turn_id, tenant_id, user_id, status,
                          heartbeat_at, created_at, trace_id)
              ON governance.conversation_executions TO {LIFECYCLE_APP_ROLE};
            GRANT UPDATE (status, terminal_event, error_code, error_detail, finished_at)
              ON governance.conversation_executions TO {LIFECYCLE_APP_ROLE};
            GRANT SELECT (turn_id, tenant_id, user_id, status, task_id,
                          final_execution_id, final_trace_id)
              ON governance.chat_turns TO {LIFECYCLE_APP_ROLE};
            GRANT UPDATE (status, error_code, error_detail, final_execution_id,
                          final_trace_id, finalized_at)
              ON governance.chat_turns TO {LIFECYCLE_APP_ROLE};
            GRANT SELECT (task_id, tenant_id, user_id, status, updated_at,
                          latest_trace_id, summary_data)
              ON governance.conversation_tasks TO {LIFECYCLE_APP_ROLE};
            GRANT UPDATE (status, completed_at, updated_at, summary_data)
              ON governance.conversation_tasks TO {LIFECYCLE_APP_ROLE};
            GRANT SELECT (tenant_id, trace_id) ON governance.tenant_traces TO {LIFECYCLE_APP_ROLE};
            GRANT INSERT (event_name, tenant_id, user_id, authz_ref,
                          permission_version, policy_set_version, turn_id,
                          execution_id, trace_id, payload)
              ON governance.lifecycle_audit_events TO {LIFECYCLE_APP_ROLE};
            GRANT USAGE, SELECT ON SEQUENCE governance.lifecycle_audit_events_event_id_seq
              TO {LIFECYCLE_APP_ROLE}
        """))
        conn.execute(text(f"""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{LIFECYCLE_LOGIN_ROLE}') THEN
                    CREATE ROLE {LIFECYCLE_LOGIN_ROLE} LOGIN NOINHERIT NOBYPASSRLS;
                END IF;
            END
            $$"""))
        conn.execute(text(
            f"ALTER ROLE {LIFECYCLE_LOGIN_ROLE} LOGIN NOINHERIT NOBYPASSRLS"
        ))
        password_statement = conn.execute(text("""
            SELECT format(
                'ALTER ROLE %I PASSWORD %L',
                CAST(:role_name AS text),
                CAST(:password AS text)
            )
        """), {
            "role_name": LIFECYCLE_LOGIN_ROLE,
            "password": settings.lifecycle_reconcile_db_password,
        }).scalar_one()
        conn.exec_driver_sql(password_statement)
        conn.execute(text(f"GRANT {LIFECYCLE_APP_ROLE} TO {LIFECYCLE_LOGIN_ROLE}"))
        conn.execute(text(f"REVOKE {ROLE} FROM {LIFECYCLE_LOGIN_ROLE}"))
        connect_statement = conn.execute(text("""
            SELECT format(
                'GRANT CONNECT ON DATABASE %I TO %I',
                current_database(),
                CAST(:role_name AS text)
            )
        """), {"role_name": LIFECYCLE_LOGIN_ROLE}).scalar_one()
        conn.exec_driver_sql(connect_statement)
        attrs = conn.execute(text(
            "SELECT rolname, rolsuper, rolbypassrls, rolcanlogin FROM pg_roles WHERE rolname=:r"
        ), {"r": ROLE}).mappings().one()
        lifecycle_attrs = conn.execute(text(
            "SELECT rolname, rolsuper, rolbypassrls, rolcanlogin, rolinherit FROM pg_roles WHERE rolname=:r"
        ), {"r": LIFECYCLE_LOGIN_ROLE}).mappings().one()
        lifecycle_app_attrs = conn.execute(text(
            "SELECT rolname, rolsuper, rolbypassrls, rolcanlogin FROM pg_roles WHERE rolname=:r"
        ), {"r": LIFECYCLE_APP_ROLE}).mappings().one()
        print("role ready:", dict(attrs))
        print("lifecycle login ready (password omitted):", dict(lifecycle_attrs))
        print("lifecycle app role ready:", dict(lifecycle_app_attrs))
    admin_engine.dispose()


if __name__ == "__main__":
    main()

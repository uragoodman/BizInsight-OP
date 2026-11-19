"""Create/rotate the narrowly granted cross-tenant preference worker role.

Run after Alembic migrations with MEMORY_PREFERENCE_ROLE_ADMIN_URL and
MEMORY_PREFERENCE_WORKER_PASSWORD set. The worker itself must use only
MEMORY_PREFERENCE_DATABASE_URL, never the admin URL.
"""

from __future__ import annotations

import os

import psycopg
from psycopg import sql


ROLE = "bizinsight_memory_worker"


def main() -> None:
    admin_url = os.environ.get("MEMORY_PREFERENCE_ROLE_ADMIN_URL", "").strip()
    password = os.environ.get("MEMORY_PREFERENCE_WORKER_PASSWORD", "")
    if not admin_url:
        raise SystemExit("MEMORY_PREFERENCE_ROLE_ADMIN_URL is required")
    if len(password) < 32:
        raise SystemExit("MEMORY_PREFERENCE_WORKER_PASSWORD must be at least 32 characters")
    if admin_url.startswith("postgresql+psycopg://"):
        admin_url = "postgresql://" + admin_url.removeprefix("postgresql+psycopg://")

    with psycopg.connect(admin_url, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                DO $$ BEGIN
                    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='bizinsight_memory_worker') THEN
                        EXECUTE 'CREATE ROLE bizinsight_memory_worker LOGIN BYPASSRLS '
                            || 'NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT';
                    END IF;
                END $$
            """)
            cursor.execute(
                sql.SQL("ALTER ROLE {} LOGIN BYPASSRLS NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT PASSWORD {}").format(
                    sql.Identifier(ROLE), sql.Literal(password),
                )
            )
            cursor.execute("GRANT USAGE ON SCHEMA governance TO bizinsight_memory_worker")
            cursor.execute("""
                GRANT SELECT, UPDATE, DELETE ON governance.memory_registry,
                    governance.memory_preference_overlays TO bizinsight_memory_worker
            """)
            cursor.execute("""
                GRANT SELECT, INSERT, UPDATE, DELETE ON governance.memory_outbox
                    TO bizinsight_memory_worker
            """)
            cursor.execute("""
                GRANT INSERT ON governance.memory_preference_audit
                    TO bizinsight_memory_worker
            """)
            cursor.execute("""
                GRANT SELECT ON governance.memory_preference_settings
                    TO bizinsight_memory_worker
            """)
            cursor.execute("""
                GRANT SELECT, UPDATE ON governance.memory_membership_erasure_jobs
                    TO bizinsight_memory_worker
            """)
            cursor.execute("""
                GRANT SELECT, UPDATE ON governance.account_erasure_jobs
                    TO bizinsight_memory_worker
            """)
            cursor.execute("""
                GRANT SELECT, DELETE ON governance.memory_sessions,
                    governance.analysis_memories TO bizinsight_memory_worker
            """)
            cursor.execute("""
                GRANT DELETE ON governance.memory_preference_settings
                    TO bizinsight_memory_worker
            """)
    print("memory preference worker role provisioned; credential value not displayed")


if __name__ == "__main__":
    main()

"""Add least-privilege roles for the Code Analysis API and queue worker."""

from alembic import op


revision = "20260803_15"
down_revision = "20260802_14"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Login credentials are provisioned by the deployment environment, never
    # embedded in a migration.  Creating NOLOGIN roles here keeps grants
    # deterministic on databases that are migrated before provisioning.
    op.execute(
        """
        DO $$
        BEGIN
          IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='bizinsight_code_api') THEN
            CREATE ROLE bizinsight_code_api NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS;
          END IF;
          IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='bizinsight_code_worker') THEN
            CREATE ROLE bizinsight_code_worker NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT BYPASSRLS;
          END IF;
        END
        $$;
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
          EXECUTE format(
            'GRANT CONNECT ON DATABASE %I TO bizinsight_code_api, bizinsight_code_worker',
            current_database()
          );
        END
        $$;
        """
    )
    op.execute("GRANT USAGE ON SCHEMA governance TO bizinsight_code_api, bizinsight_code_worker")
    op.execute(
        """GRANT SELECT, INSERT, UPDATE, DELETE
           ON governance.code_analysis_files,
              governance.code_analysis_jobs,
              governance.code_analysis_artifacts,
              governance.code_analysis_audits
           TO bizinsight_code_api, bizinsight_code_worker"""
    )
    # The API validates an optional source-turn link under that table's own
    # tenant/user RLS policy; it never receives broader chat-history access.
    op.execute(
        "GRANT SELECT ON governance.chat_turns TO bizinsight_code_api"
    )
    op.execute(
        "GRANT SELECT, INSERT, UPDATE ON governance.tenant_traces TO bizinsight_code_worker"
    )


def downgrade() -> None:
    op.execute(
        """REVOKE ALL PRIVILEGES
           ON governance.code_analysis_files,
              governance.code_analysis_jobs,
              governance.code_analysis_artifacts,
              governance.code_analysis_audits
           FROM bizinsight_code_api, bizinsight_code_worker"""
    )
    op.execute("REVOKE SELECT ON governance.chat_turns FROM bizinsight_code_api")
    op.execute(
        "REVOKE SELECT, INSERT, UPDATE ON governance.tenant_traces FROM bizinsight_code_worker"
    )
    op.execute("REVOKE USAGE ON SCHEMA governance FROM bizinsight_code_api, bizinsight_code_worker")
    op.execute(
        """
        DO $$
        BEGIN
          EXECUTE format(
            'REVOKE CONNECT ON DATABASE %I FROM bizinsight_code_api, bizinsight_code_worker',
            current_database()
          );
        END
        $$;
        """
    )
    # Roles are cluster-scoped and may have been provisioned independently.
    # Deliberately do not drop them during a schema downgrade.

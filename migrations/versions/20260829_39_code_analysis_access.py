"""Add explicit Code Analysis sharing and execution audit provenance."""

from alembic import op


revision = "20260829_39"
down_revision = "20260826_38"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """CREATE TABLE governance.code_analysis_access_grants (
          grant_id varchar(64) PRIMARY KEY,
          tenant_id varchar(128) NOT NULL,
          resource_type varchar(32) NOT NULL,
          resource_id varchar(128) NOT NULL,
          grantee_user_id varchar(128) NOT NULL,
          permission varchar(32) NOT NULL,
          granted_by varchar(128) NOT NULL,
          expires_at timestamptz,
          revoked_at timestamptz,
          created_at timestamptz NOT NULL DEFAULT now(),
          CONSTRAINT code_analysis_grant_resource_type CHECK (resource_type IN ('file','job','artifact','audit')),
          CONSTRAINT code_analysis_grant_permission CHECK (permission IN ('read','execute','audit')),
          CONSTRAINT code_analysis_grant_not_self CHECK (grantee_user_id <> granted_by)
        )"""
    )
    op.execute(
        """CREATE TABLE governance.code_analysis_access_events (
          event_id varchar(64) PRIMARY KEY,
          tenant_id varchar(128) NOT NULL,
          user_id varchar(128) NOT NULL,
          resource_type varchar(32) NOT NULL,
          resource_id varchar(128) NOT NULL,
          action varchar(32) NOT NULL,
          allowed boolean NOT NULL,
          metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
          created_at timestamptz NOT NULL DEFAULT now(),
          CONSTRAINT code_analysis_access_event_resource_type CHECK (resource_type IN ('file','job','artifact','audit'))
        )"""
    )
    op.execute("CREATE INDEX idx_code_analysis_access_events_lookup ON governance.code_analysis_access_events(tenant_id,resource_id,created_at DESC)")
    op.execute(
        """CREATE UNIQUE INDEX uq_code_analysis_active_grant
           ON governance.code_analysis_access_grants
              (tenant_id,resource_type,resource_id,grantee_user_id,permission)
           WHERE revoked_at IS NULL"""
    )
    op.execute(
        """CREATE INDEX idx_code_analysis_grants_lookup
           ON governance.code_analysis_access_grants
              (tenant_id,resource_type,resource_id,grantee_user_id)
           WHERE revoked_at IS NULL"""
    )
    op.execute("ALTER TABLE governance.code_analysis_access_grants ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE governance.code_analysis_access_grants FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE governance.code_analysis_access_events ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE governance.code_analysis_access_events FORCE ROW LEVEL SECURITY")
    op.execute(
        """CREATE POLICY code_analysis_access_grants_isolation
           ON governance.code_analysis_access_grants
           USING (tenant_id=current_setting('app.tenant_id',true))
           WITH CHECK (tenant_id=current_setting('app.tenant_id',true))"""
    )
    op.execute(
        """CREATE POLICY code_analysis_access_events_isolation
           ON governance.code_analysis_access_events
           USING (tenant_id=current_setting('app.tenant_id',true)
                  AND (user_id=current_setting('app.user_id',true)
                       OR current_setting('app.user_role',true) IN ('owner','admin')))
           WITH CHECK (tenant_id=current_setting('app.tenant_id',true)
                       AND user_id=current_setting('app.user_id',true))"""
    )
    op.execute(
        """GRANT SELECT, INSERT, UPDATE, DELETE
           ON governance.code_analysis_access_grants
           TO bizinsight_code_api, bizinsight_code_worker"""
    )
    op.execute("GRANT SELECT, INSERT ON governance.code_analysis_access_events TO bizinsight_code_api, bizinsight_code_worker")
    op.execute("GRANT SELECT ON iam.tenant_memberships TO bizinsight_code_api")

    op.execute(
        """ALTER TABLE governance.code_analysis_audits
           ADD COLUMN actor_id varchar(128),
           ADD COLUMN initiator_type varchar(32),
           ADD COLUMN permission_decision varchar(32),
           ADD COLUMN provider_request_id varchar(128),
           ADD COLUMN input_sha256 varchar(64),
           ADD COLUMN input_tokens integer NOT NULL DEFAULT 0,
           ADD COLUMN output_tokens integer NOT NULL DEFAULT 0,
           ADD COLUMN estimated_cost double precision NOT NULL DEFAULT 0,
           ADD COLUMN error_category varchar(80)"""
    )

    # Existing owner-only policies are upgraded to owner-or-explicit-grant.
    for table, resource_type, resource_column, grant_permission in (
        ("code_analysis_files", "file", "file_id", "read"),
        ("code_analysis_jobs", "job", "job_id", "read"),
    ):
        op.execute(f"DROP POLICY IF EXISTS {table}_isolation ON governance.{table}")
        op.execute(
            f"""CREATE POLICY {table}_isolation ON governance.{table}
            USING (
              tenant_id=current_setting('app.tenant_id',true)
              AND (
                user_id=current_setting('app.user_id',true)
                OR current_setting('app.user_role',true) IN ('owner','admin')
                OR EXISTS (
                  SELECT 1 FROM governance.code_analysis_access_grants g
                   WHERE g.tenant_id={table}.tenant_id
                     AND g.resource_type='{resource_type}'
                     AND g.resource_id={table}.{resource_column}
                     AND g.grantee_user_id=current_setting('app.user_id',true)
                    AND g.permission IN ('{grant_permission}','execute','audit')
                     AND g.revoked_at IS NULL
                     AND (g.expires_at IS NULL OR g.expires_at > now())
                )
              )
            )
            WITH CHECK (tenant_id=current_setting('app.tenant_id',true)
                        AND user_id=current_setting('app.user_id',true))"""
        )

    # Artifacts and audits can be shared directly or through a shared job.
    for table, resource_type, resource_column in (
        ("code_analysis_artifacts", "artifact", "artifact_id"),
        ("code_analysis_audits", "audit", "audit_id"),
    ):
        op.execute(f"DROP POLICY IF EXISTS {table}_isolation ON governance.{table}")
        op.execute(
            f"""CREATE POLICY {table}_isolation ON governance.{table}
            USING (
              tenant_id=current_setting('app.tenant_id',true)
              AND (
                user_id=current_setting('app.user_id',true)
                OR current_setting('app.user_role',true) IN ('owner','admin')
                OR EXISTS (
                  SELECT 1 FROM governance.code_analysis_access_grants g
                   WHERE g.tenant_id={table}.tenant_id
                     AND ((g.resource_type='{resource_type}' AND g.resource_id={table}.{resource_column})
                       OR (g.resource_type='job' AND g.resource_id={table}.job_id))
                     AND g.grantee_user_id=current_setting('app.user_id',true)
                     AND g.permission IN ('read','audit')
                     AND g.revoked_at IS NULL
                     AND (g.expires_at IS NULL OR g.expires_at > now())
                )
              )
            )
            WITH CHECK (tenant_id=current_setting('app.tenant_id',true)
                        AND user_id=current_setting('app.user_id',true))"""
        )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS governance.code_analysis_access_events")
    op.execute("DROP TABLE IF EXISTS governance.code_analysis_access_grants")
    op.execute("REVOKE SELECT ON iam.tenant_memberships FROM bizinsight_code_api")
    op.execute(
        """ALTER TABLE governance.code_analysis_audits
           DROP COLUMN error_category, DROP COLUMN estimated_cost,
           DROP COLUMN output_tokens, DROP COLUMN input_tokens,
           DROP COLUMN input_sha256, DROP COLUMN provider_request_id,
           DROP COLUMN permission_decision, DROP COLUMN initiator_type,
           DROP COLUMN actor_id"""
    )

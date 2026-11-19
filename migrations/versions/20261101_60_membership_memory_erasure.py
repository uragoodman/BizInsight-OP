"""Add a durable memory erasure coordinator for tenant membership revocation."""

from alembic import op


revision = "20261101_60"
down_revision = "20261031_59"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE governance.memory_membership_erasure_jobs (
            job_id varchar(64) PRIMARY KEY,
            tenant_id varchar(128) NOT NULL,
            subject_user_id varchar(128) NOT NULL,
            requested_by_user_id varchar(128) NOT NULL,
            idempotency_key varchar(128) NOT NULL,
            status varchar(16) NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','leased','completed','dead_letter')),
            attempts integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
            max_attempts integer NOT NULL DEFAULT 20 CHECK (max_attempts > 0),
            next_attempt_at timestamptz NOT NULL DEFAULT now(),
            lease_token varchar(64),
            lease_expires_at timestamptz,
            last_error_code varchar(128),
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            completed_at timestamptz,
            expires_at timestamptz NOT NULL DEFAULT (now() + interval '180 days'),
            UNIQUE (tenant_id, requested_by_user_id, idempotency_key),
            CHECK (
                (status = 'leased' AND lease_token IS NOT NULL AND lease_expires_at IS NOT NULL)
                OR (status <> 'leased' AND lease_token IS NULL AND lease_expires_at IS NULL)
            )
        )
    """)
    op.execute("""
        CREATE UNIQUE INDEX idx_membership_erasure_active_subject
        ON governance.memory_membership_erasure_jobs(tenant_id, subject_user_id)
        WHERE status IN ('pending','leased')
    """)
    op.execute("""
        CREATE INDEX idx_membership_erasure_claim
        ON governance.memory_membership_erasure_jobs(status, next_attempt_at, created_at)
        WHERE status IN ('pending','leased')
    """)
    op.execute("ALTER TABLE governance.memory_membership_erasure_jobs ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE governance.memory_membership_erasure_jobs FORCE ROW LEVEL SECURITY")
    op.execute("""
        CREATE POLICY memory_membership_erasure_jobs_owner
        ON governance.memory_membership_erasure_jobs
        USING (tenant_id = current_setting('app.tenant_id', true)
               AND requested_by_user_id = current_setting('app.user_id', true))
        WITH CHECK (tenant_id = current_setting('app.tenant_id', true)
                    AND requested_by_user_id = current_setting('app.user_id', true))
    """)
    # Tenant administrators need to detect an in-flight erasure before
    # reactivating a membership or creating a duplicate job. The table contains
    # only job metadata; status lookup endpoints still constrain the requester.
    op.execute("""
        CREATE POLICY memory_membership_erasure_jobs_admin_read
        ON governance.memory_membership_erasure_jobs
        FOR SELECT
        USING (tenant_id = current_setting('app.tenant_id', true)
               AND current_setting('app.role', true) IN ('owner','admin'))
    """)
    op.execute("""
        CREATE POLICY memory_membership_erasure_jobs_admin_update
        ON governance.memory_membership_erasure_jobs
        FOR UPDATE
        USING (tenant_id = current_setting('app.tenant_id', true)
               AND current_setting('app.role', true) IN ('owner','admin'))
        WITH CHECK (tenant_id = current_setting('app.tenant_id', true)
                    AND current_setting('app.role', true) IN ('owner','admin'))
    """)
    op.execute("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='bizinsight_app') THEN
                GRANT SELECT, INSERT ON governance.memory_membership_erasure_jobs TO bizinsight_app;
            END IF;
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='bizinsight_memory_worker') THEN
                GRANT SELECT, UPDATE ON governance.memory_membership_erasure_jobs TO bizinsight_memory_worker;
                GRANT SELECT, DELETE ON governance.memory_sessions,
                    governance.analysis_memories TO bizinsight_memory_worker;
                GRANT DELETE ON governance.memory_preference_settings TO bizinsight_memory_worker;
            END IF;
        END $$;
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS governance.memory_membership_erasure_jobs")

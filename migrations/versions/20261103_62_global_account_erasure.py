"""Add global account erasure coordination and its immediate access revocation."""

from alembic import op


revision = "20261103_62"
down_revision = "20261102_61"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE governance.account_erasure_jobs (
            job_id varchar(64) PRIMARY KEY,
            subject_user_id varchar(128) NOT NULL,
            anonymous_user_id varchar(128) NOT NULL UNIQUE,
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
            UNIQUE (subject_user_id,idempotency_key),
            CHECK ((status='leased' AND lease_token IS NOT NULL AND lease_expires_at IS NOT NULL)
                OR (status<>'leased' AND lease_token IS NULL AND lease_expires_at IS NULL))
        )
    """)
    op.execute("CREATE INDEX idx_account_erasure_claim ON governance.account_erasure_jobs(status,next_attempt_at,created_at) WHERE status IN ('pending','leased')")
    op.execute("ALTER TABLE governance.account_erasure_jobs ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE governance.account_erasure_jobs FORCE ROW LEVEL SECURITY")
    op.execute("""
        CREATE POLICY account_erasure_jobs_self ON governance.account_erasure_jobs
        USING (subject_user_id=current_setting('app.user_id',true))
        WITH CHECK (subject_user_id=current_setting('app.user_id',true))
    """)
    # The API can only invoke a function that erases the authenticated principal.
    # The SECURITY DEFINER body is fixed to this schema and validates the caller's
    # database-local identity before touching cross-tenant IAM rows.
    op.execute("""
        CREATE FUNCTION governance.request_account_erasure(p_idempotency_key varchar)
        RETURNS TABLE(job_id varchar, status varchar)
        LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, governance, iam
        AS $$
        DECLARE
            caller_id varchar(128) := current_setting('app.user_id', true);
            existing governance.account_erasure_jobs%ROWTYPE;
            generated_job_id varchar(64) := 'aerase_' || replace(gen_random_uuid()::text,'-','');
            generated_anonymous_id varchar(128) := 'anon_' || replace(gen_random_uuid()::text,'-','');
        BEGIN
            IF caller_id IS NULL OR caller_id = '' THEN
                RAISE EXCEPTION 'account_erasure_identity_required' USING ERRCODE='42501';
            END IF;
            IF p_idempotency_key IS NULL OR p_idempotency_key='' OR length(p_idempotency_key)>128 THEN
                RAISE EXCEPTION 'account_erasure_idempotency_key_invalid' USING ERRCODE='22023';
            END IF;
            SELECT * INTO existing FROM governance.account_erasure_jobs j
             WHERE j.subject_user_id=caller_id
               AND j.idempotency_key=p_idempotency_key FOR UPDATE;
            IF FOUND THEN
                IF existing.subject_user_id <> caller_id THEN
                    RAISE EXCEPTION 'account_erasure_idempotency_conflict' USING ERRCODE='23505';
                END IF;
                RETURN QUERY SELECT existing.job_id, existing.status;
                RETURN;
            END IF;
            IF NOT EXISTS (SELECT 1 FROM iam.users u
                           WHERE u.user_id=caller_id AND u.status='active' FOR UPDATE) THEN
                RAISE EXCEPTION 'active_account_not_found' USING ERRCODE='P0002';
            END IF;
            INSERT INTO governance.account_erasure_jobs(job_id,subject_user_id,
                anonymous_user_id,idempotency_key)
            VALUES (generated_job_id,caller_id,generated_anonymous_id,p_idempotency_key);
            INSERT INTO governance.memory_membership_erasure_jobs
                (job_id,tenant_id,subject_user_id,requested_by_user_id,idempotency_key)
            SELECT 'merase_' || substr(md5(generated_job_id || tenants.tenant_id),1,32),
                   tenants.tenant_id,caller_id,caller_id,
                   'account-erasure:' || generated_job_id || ':' || tenants.tenant_id
              FROM (
                  SELECT tenant_id FROM iam.tenant_memberships WHERE user_id=caller_id
                  UNION SELECT tenant_id FROM governance.memory_registry WHERE user_id=caller_id
                  UNION SELECT tenant_id FROM governance.memory_preference_overlays WHERE user_id=caller_id
                  UNION SELECT tenant_id FROM governance.memory_sessions WHERE user_id=caller_id
                  UNION SELECT tenant_id FROM governance.analysis_memories WHERE user_id=caller_id
                  UNION SELECT tenant_id FROM governance.memory_preference_settings WHERE user_id=caller_id
                  UNION SELECT tenant_id FROM governance.memory_outbox WHERE user_id=caller_id
                  UNION SELECT tenant_id FROM governance.memory_decisions WHERE user_id=caller_id
                  UNION SELECT tenant_id FROM governance.memory_consolidation_jobs WHERE user_id=caller_id
                  UNION SELECT tenant_id FROM governance.memory_source_snapshots WHERE user_id=caller_id
              ) tenants
            ON CONFLICT (tenant_id,requested_by_user_id,idempotency_key) DO NOTHING;
            UPDATE iam.users SET status='erasure_pending', updated_at=now()
             WHERE user_id=caller_id;
            UPDATE iam.auth_sessions SET revoked_at=COALESCE(revoked_at,now())
             WHERE user_id=caller_id;
            UPDATE iam.api_credentials SET revoked_at=COALESCE(revoked_at,now())
             WHERE user_id=caller_id;
            UPDATE iam.tenant_memberships SET revoked_at=COALESCE(revoked_at,now())
             WHERE user_id=caller_id;
            UPDATE governance.code_analysis_jobs AS jobs
               SET cancel_requested=true,updated_at=now()
             WHERE jobs.user_id=caller_id
               AND jobs.status IN ('awaiting_confirmation','queued','generating','validating','running');
            RETURN QUERY SELECT generated_job_id, 'pending'::varchar;
        END $$
    """)
    op.execute("REVOKE ALL ON FUNCTION governance.request_account_erasure(varchar) FROM PUBLIC")
    op.execute("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='bizinsight_app') THEN
                GRANT USAGE ON SCHEMA governance TO bizinsight_app;
                GRANT SELECT ON governance.account_erasure_jobs TO bizinsight_app;
                GRANT EXECUTE ON FUNCTION governance.request_account_erasure(varchar) TO bizinsight_app;
            END IF;
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='bizinsight_memory_worker') THEN
                GRANT SELECT, UPDATE ON governance.account_erasure_jobs TO bizinsight_memory_worker;
                GRANT SELECT, UPDATE, DELETE ON iam.users, iam.tenant_memberships,
                    iam.membership_seller_grants, iam.auth_sessions, iam.api_credentials
                    TO bizinsight_memory_worker;
                GRANT SELECT, UPDATE, DELETE ON governance.chat_sessions,
                    governance.conversation_tasks, governance.tenant_traces,
                    governance.authorization_snapshots, governance.lifecycle_audit_events,
                    governance.audit_events, governance.memory_registry,
                    governance.memory_preference_audit, governance.memory_membership_erasure_jobs,
                    governance.analysis_execution_audits, governance.data_replay_audits,
                    governance.permission_event_outbox, governance.memory_preference_overlays,
                    governance.memory_sessions, governance.analysis_memories,
                    governance.memory_preference_settings, governance.memory_outbox,
                    governance.memory_decisions, governance.memory_consolidation_jobs,
                    governance.memory_source_snapshots, governance.code_analysis_files,
                    governance.code_analysis_jobs, governance.code_analysis_artifacts,
                    governance.llm_usage_events,
                    governance.tenant_report_insights, governance.tenant_reports,
                    governance.abac_policy_rules
                    TO bizinsight_memory_worker;
                GRANT USAGE ON SCHEMA analytics TO bizinsight_memory_worker;
                GRANT SELECT, UPDATE ON analytics.tenant_metric_targets TO bizinsight_memory_worker;
            END IF;
        END $$
    """)


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS governance.request_account_erasure(varchar)")
    op.execute("DROP TABLE IF EXISTS governance.account_erasure_jobs")

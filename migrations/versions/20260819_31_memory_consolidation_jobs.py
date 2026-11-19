"""Durable lease queue for opt-in LLM memory consolidation proposals."""
from alembic import op

revision = "20260819_31"
down_revision = "20260818_30"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE governance.memory_decisions DROP CONSTRAINT IF EXISTS memory_decisions_action_check")
    op.execute("""ALTER TABLE governance.memory_decisions ADD CONSTRAINT memory_decisions_action_check
        CHECK (action IN ('created_active','deduplicated','pending_review_created','approved_independent',
                          'superseded','rejected','proposal_created','proposal_skipped','proposal_failed',
                          'staled','proposal_queued','proposal_completed'))""")
    op.execute("""CREATE TABLE governance.memory_consolidation_jobs (
        job_id varchar(128) PRIMARY KEY,
        tenant_id varchar(128) NOT NULL,
        user_id varchar(128) NOT NULL,
        actor varchar(128) NOT NULL,
        scope_fingerprint char(64) NOT NULL,
        input_payload jsonb NOT NULL,
        input_hash char(64) NOT NULL,
        status varchar(16) NOT NULL DEFAULT 'queued'
          CHECK (status IN ('queued','generating','completed','failed','cancelled')),
        attempts integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
        lease_token varchar(64),
        lease_expires_at timestamptz,
        proposed_memory_id varchar(128),
        error_code varchar(64),
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        completed_at timestamptz
    )""")
    op.execute("""CREATE INDEX idx_memory_consolidation_jobs_owner
        ON governance.memory_consolidation_jobs(tenant_id,user_id,status,created_at DESC)""")
    op.execute("""CREATE UNIQUE INDEX uq_memory_consolidation_live_input
        ON governance.memory_consolidation_jobs(tenant_id,user_id,scope_fingerprint,input_hash)
        WHERE status IN ('queued','generating')""")
    op.execute("ALTER TABLE governance.memory_consolidation_jobs ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE governance.memory_consolidation_jobs FORCE ROW LEVEL SECURITY")
    op.execute("""CREATE POLICY memory_consolidation_jobs_isolation ON governance.memory_consolidation_jobs
        USING (tenant_id=current_setting('app.tenant_id',true) AND user_id=current_setting('app.user_id',true))
        WITH CHECK (tenant_id=current_setting('app.tenant_id',true) AND user_id=current_setting('app.user_id',true))""")


def downgrade() -> None:
    op.execute("ALTER TABLE governance.memory_decisions DROP CONSTRAINT IF EXISTS memory_decisions_action_check")
    op.execute("""ALTER TABLE governance.memory_decisions ADD CONSTRAINT memory_decisions_action_check
        CHECK (action IN ('created_active','deduplicated','pending_review_created','approved_independent',
                          'superseded','rejected','proposal_created','proposal_skipped','proposal_failed','staled'))""")
    op.execute("DROP TABLE IF EXISTS governance.memory_consolidation_jobs")

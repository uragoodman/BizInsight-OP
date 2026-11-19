"""Phase C: replayable governed source snapshots and data-version invalidation."""
from alembic import op

revision = "20260818_30"
down_revision = "20260817_29"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""CREATE TABLE governance.memory_source_snapshots (
        snapshot_id varchar(128) PRIMARY KEY,
        tenant_id varchar(128) NOT NULL,
        user_id varchar(128) NOT NULL,
        source_data_version char(64) NOT NULL,
        scope_fingerprint char(64) NOT NULL,
        source_trace_id varchar(128) NOT NULL,
        metric_definition_version char(64) NOT NULL,
        snapshot_payload jsonb NOT NULL,
        payload_bytes integer NOT NULL CHECK (payload_bytes > 0),
        captured_at timestamptz NOT NULL DEFAULT now(),
        UNIQUE (tenant_id,user_id,source_data_version)
    )""")
    op.execute("""CREATE INDEX idx_memory_source_snapshots_owner_scope
        ON governance.memory_source_snapshots(tenant_id,user_id,scope_fingerprint,captured_at DESC)""")
    op.execute("ALTER TABLE governance.memory_source_snapshots ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE governance.memory_source_snapshots FORCE ROW LEVEL SECURITY")
    op.execute("""CREATE POLICY memory_source_snapshots_isolation ON governance.memory_source_snapshots
        USING (tenant_id=current_setting('app.tenant_id',true) AND user_id=current_setting('app.user_id',true))
        WITH CHECK (tenant_id=current_setting('app.tenant_id',true) AND user_id=current_setting('app.user_id',true))""")
    op.execute("ALTER TABLE governance.memory_decisions DROP CONSTRAINT IF EXISTS memory_decisions_action_check")
    op.execute("""ALTER TABLE governance.memory_decisions ADD CONSTRAINT memory_decisions_action_check
        CHECK (action IN ('created_active','deduplicated','pending_review_created','approved_independent',
                          'superseded','rejected','proposal_created','proposal_skipped','proposal_failed','staled'))""")


def downgrade() -> None:
    op.execute("ALTER TABLE governance.memory_decisions DROP CONSTRAINT IF EXISTS memory_decisions_action_check")
    op.execute("""ALTER TABLE governance.memory_decisions ADD CONSTRAINT memory_decisions_action_check
        CHECK (action IN ('created_active','deduplicated','pending_review_created','approved_independent',
                          'superseded','rejected','proposal_created','proposal_skipped','proposal_failed'))""")
    op.execute("DROP TABLE IF EXISTS governance.memory_source_snapshots")

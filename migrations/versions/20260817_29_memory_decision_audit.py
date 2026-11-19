"""Phase B append-only memory decision audit."""
from alembic import op

revision = "20260817_29"
down_revision = "20260816_28"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""CREATE TABLE governance.memory_decisions (
        decision_id varchar(128) PRIMARY KEY,
        tenant_id varchar(128) NOT NULL,
        user_id varchar(128) NOT NULL,
        memory_id varchar(128),
        action varchar(64) NOT NULL,
        actor varchar(128) NOT NULL,
        reason text NOT NULL DEFAULT '',
        rule_scores jsonb NOT NULL DEFAULT '{}'::jsonb,
        source_memory_ids jsonb NOT NULL DEFAULT '[]'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        CHECK (action IN ('created_active','deduplicated','pending_review_created','approved_independent',
                          'superseded','rejected','proposal_created','proposal_skipped','proposal_failed'))
    )""")
    op.execute("""CREATE INDEX idx_memory_decisions_owner_memory
        ON governance.memory_decisions(tenant_id, user_id, memory_id, created_at DESC)""")
    op.execute("ALTER TABLE governance.memory_decisions ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE governance.memory_decisions FORCE ROW LEVEL SECURITY")
    op.execute("""CREATE POLICY memory_decisions_isolation ON governance.memory_decisions
        USING (tenant_id=current_setting('app.tenant_id',true)
               AND user_id=current_setting('app.user_id',true))
        WITH CHECK (tenant_id=current_setting('app.tenant_id',true)
                    AND user_id=current_setting('app.user_id',true))""")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS governance.memory_decisions")

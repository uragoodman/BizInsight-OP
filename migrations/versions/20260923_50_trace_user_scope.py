"""Bind persisted traces to both tenant and user execution identity."""

from alembic import op


revision = "20260923_50"
down_revision = "20260923_49"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE governance.tenant_traces ADD COLUMN IF NOT EXISTS user_id varchar(128)")
    op.execute("""
        UPDATE governance.tenant_traces
           SET user_id=COALESCE(trace_json->'state_snapshot'->>'user_id', 'legacy')
         WHERE user_id IS NULL
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_tenant_traces_user ON governance.tenant_traces(tenant_id,user_id,created_at DESC)")
    op.execute("DROP POLICY IF EXISTS tenant_traces_isolation ON governance.tenant_traces")
    op.execute("""
        CREATE POLICY tenant_traces_isolation
        ON governance.tenant_traces
        USING (current_setting('app.scope_type', true) = 'platform'
            OR (tenant_id=current_setting('app.tenant_id', true)
                AND user_id=current_setting('app.user_id', true)))
        WITH CHECK (tenant_id=current_setting('app.tenant_id', true)
            AND user_id=current_setting('app.user_id', true))
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS governance.idx_tenant_traces_user")
    op.execute("ALTER TABLE governance.tenant_traces DROP COLUMN IF EXISTS user_id")

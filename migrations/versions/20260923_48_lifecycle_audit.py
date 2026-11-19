"""Durable lifecycle and authorization audit events."""

from alembic import op


revision = "20260923_48"
down_revision = "20260923_47"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS governance.lifecycle_audit_events (
          event_id bigserial PRIMARY KEY,
          event_name varchar(96) NOT NULL,
          tenant_id varchar(128) NOT NULL,
          user_id varchar(128) NOT NULL,
          authz_ref varchar(160),
          permission_version varchar(128),
          policy_set_version varchar(128),
          turn_id varchar(128),
          execution_id varchar(128),
          trace_id varchar(160),
          payload jsonb NOT NULL DEFAULT '{}'::jsonb,
          created_at timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_lifecycle_audit_scope ON governance.lifecycle_audit_events(tenant_id,user_id,created_at DESC)")
    op.execute("ALTER TABLE governance.lifecycle_audit_events ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE governance.lifecycle_audit_events FORCE ROW LEVEL SECURITY")
    op.execute("DROP POLICY IF EXISTS lifecycle_audit_events_isolation ON governance.lifecycle_audit_events")
    op.execute("""
        CREATE POLICY lifecycle_audit_events_isolation
        ON governance.lifecycle_audit_events
        USING (tenant_id=current_setting('app.tenant_id',true)
           AND user_id=current_setting('app.user_id',true))
        WITH CHECK (tenant_id=current_setting('app.tenant_id',true)
           AND user_id=current_setting('app.user_id',true))
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS governance.lifecycle_audit_events")

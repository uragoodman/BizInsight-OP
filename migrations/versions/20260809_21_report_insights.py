"""Report AI insight artifacts, version-bound to a report fact version."""
from alembic import op

revision = "20260809_21"
down_revision = "20260808_20"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS governance.tenant_report_insights (
            tenant_id varchar(128) NOT NULL REFERENCES iam.tenants(tenant_id),
            report_id varchar(128) NOT NULL, report_version integer NOT NULL,
            insight_version integer NOT NULL,
            status varchar(16) NOT NULL CHECK (status IN ('generating','draft','rejected','approved','published')),
            brief_json jsonb NOT NULL, input_snapshot_hash varchar(64) NOT NULL,
            selected_evidence_ids jsonb NOT NULL, insights_json jsonb NOT NULL,
            model varchar(128) NOT NULL, prompt_version varchar(64) NOT NULL,
            critic_verdict varchar(16), critic_result jsonb NOT NULL DEFAULT '[]'::jsonb,
            validation_errors jsonb NOT NULL DEFAULT '[]'::jsonb,
            actor varchar(128) NOT NULL DEFAULT 'system',
            created_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (tenant_id, report_id, report_version, insight_version)
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_tenant_report_insights_hash
        ON governance.tenant_report_insights(tenant_id, report_id, report_version, input_snapshot_hash)
    """)
    op.execute("ALTER TABLE governance.tenant_report_insights ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE governance.tenant_report_insights FORCE ROW LEVEL SECURITY")
    op.execute("""CREATE POLICY tenant_report_insights_isolation ON governance.tenant_report_insights
        USING (current_setting('app.scope_type', true) = 'platform' OR tenant_id = current_setting('app.tenant_id', true))
        WITH CHECK (tenant_id = current_setting('app.tenant_id', true))""")


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS tenant_report_insights_isolation ON governance.tenant_report_insights")
    op.execute("DROP TABLE IF EXISTS governance.tenant_report_insights")

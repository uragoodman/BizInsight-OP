"""Add auditable execution governance for whitelist complex analyses."""

from alembic import op

revision = "20260730_12"
down_revision = "20260729_11"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
    CREATE TABLE governance.analysis_execution_audits (
        execution_id varchar(64) PRIMARY KEY,
        tenant_id varchar(128) NOT NULL,
        user_id varchar(128) NOT NULL,
        complex_plan_id varchar(64) NOT NULL,
        plan_revision integer NOT NULL DEFAULT 1,
        template_name varchar(80) NOT NULL,
        status varchar(32) NOT NULL,
        plan_hash varchar(128) NOT NULL,
        input_fingerprint varchar(64) NOT NULL,
        template_code_hash varchar(64) NOT NULL,
        result_hash varchar(64),
        trace_id varchar(128),
        error_code varchar(80),
        started_at timestamptz NOT NULL DEFAULT now(),
        finished_at timestamptz,
        duration_ms integer,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb
    )
    """)
    op.execute("CREATE INDEX idx_execution_audit_owner ON governance.analysis_execution_audits(tenant_id,user_id,started_at DESC)")
    op.execute("CREATE INDEX idx_execution_audit_plan ON governance.analysis_execution_audits(complex_plan_id,started_at DESC)")
    op.execute("""CREATE UNIQUE INDEX uq_complex_plan_one_running_per_user
      ON governance.complex_analysis_plans(tenant_id,user_id) WHERE status='running'""")
    op.execute("ALTER TABLE governance.analysis_execution_audits ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE governance.analysis_execution_audits FORCE ROW LEVEL SECURITY")
    op.execute("""CREATE POLICY analysis_execution_audits_isolation ON governance.analysis_execution_audits
      USING (tenant_id=current_setting('app.tenant_id',true) AND user_id=current_setting('app.user_id',true))
      WITH CHECK (tenant_id=current_setting('app.tenant_id',true) AND user_id=current_setting('app.user_id',true))""")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS governance.uq_complex_plan_one_running_per_user")
    op.execute("DROP TABLE IF EXISTS governance.analysis_execution_audits")

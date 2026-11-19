"""Persist confirmed complex-analysis plans and action drafts."""

from alembic import op

revision = "20260728_10"
down_revision = "20260727_09"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
    CREATE TABLE governance.complex_analysis_plans (
        plan_id varchar(64) PRIMARY KEY,
        tenant_id varchar(128) NOT NULL,
        user_id varchar(128) NOT NULL,
        chat_session_id varchar(128),
        chat_turn_id varchar(128),
        question text NOT NULL,
        template_name varchar(80) NOT NULL,
        scope_data jsonb NOT NULL,
        params_data jsonb NOT NULL,
        plan_data jsonb NOT NULL,
        plan_hash varchar(128) NOT NULL,
        status varchar(32) NOT NULL DEFAULT 'pending_confirmation',
        expires_at timestamptz NOT NULL,
        confirmed_at timestamptz,
        completed_at timestamptz,
        trace_id varchar(128),
        result_data jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        FOREIGN KEY (tenant_id, user_id, chat_session_id)
          REFERENCES governance.chat_sessions(tenant_id, user_id, chat_session_id) ON DELETE CASCADE
    )
    """)
    op.execute("""
    CREATE TABLE governance.action_plans (
        action_plan_id varchar(64) PRIMARY KEY,
        tenant_id varchar(128) NOT NULL,
        user_id varchar(128) NOT NULL,
        complex_plan_id varchar(64) NOT NULL REFERENCES governance.complex_analysis_plans(plan_id) ON DELETE CASCADE,
        title varchar(200) NOT NULL,
        content jsonb NOT NULL,
        status varchar(32) NOT NULL DEFAULT 'draft',
        version integer NOT NULL DEFAULT 1,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now()
    )
    """)
    op.execute("CREATE INDEX idx_complex_plan_owner ON governance.complex_analysis_plans(tenant_id,user_id,created_at DESC)")
    op.execute("CREATE INDEX idx_action_plan_owner ON governance.action_plans(tenant_id,user_id,updated_at DESC)")
    for table in ("complex_analysis_plans", "action_plans"):
        op.execute(f"ALTER TABLE governance.{table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE governance.{table} FORCE ROW LEVEL SECURITY")
        op.execute(f"""CREATE POLICY {table}_isolation ON governance.{table}
          USING (tenant_id=current_setting('app.tenant_id',true) AND user_id=current_setting('app.user_id',true))
          WITH CHECK (tenant_id=current_setting('app.tenant_id',true) AND user_id=current_setting('app.user_id',true))""")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS governance.action_plans")
    op.execute("DROP TABLE IF EXISTS governance.complex_analysis_plans")

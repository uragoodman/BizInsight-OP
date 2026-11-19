"""Conversation tasks and governed model-routing metadata."""
from alembic import op

revision = "20260808_20"
down_revision = "20260807_19"
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.execute("""CREATE TABLE governance.conversation_tasks (
      task_id varchar(64) PRIMARY KEY, tenant_id varchar(128) NOT NULL, user_id varchar(128) NOT NULL,
      chat_session_id varchar(128) NOT NULL, title varchar(160) NOT NULL, objective text NOT NULL,
      status varchar(32) NOT NULL, scope_data jsonb, task_type varchar(64), route_data jsonb NOT NULL DEFAULT '{}'::jsonb,
      pending_data jsonb NOT NULL DEFAULT '{}'::jsonb, summary_data jsonb NOT NULL DEFAULT '{}'::jsonb,
      latest_trace_id varchar(128), created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(), completed_at timestamptz,
      FOREIGN KEY (tenant_id,user_id,chat_session_id) REFERENCES governance.chat_sessions(tenant_id,user_id,chat_session_id) ON DELETE CASCADE,
      CHECK (status IN ('draft','awaiting_scope','clarification_required','awaiting_confirmation','running','completed','failed','cancelled','superseded'))
    )""")
    op.execute("CREATE INDEX idx_conversation_tasks_session ON governance.conversation_tasks(tenant_id,user_id,chat_session_id,updated_at DESC)")
    op.execute("ALTER TABLE governance.chat_turns ADD COLUMN IF NOT EXISTS task_id varchar(64)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_chat_turns_task ON governance.chat_turns(tenant_id,user_id,task_id,created_at DESC)")
    op.execute("ALTER TABLE governance.conversation_tasks ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE governance.conversation_tasks FORCE ROW LEVEL SECURITY")
    op.execute("""CREATE POLICY conversation_tasks_isolation ON governance.conversation_tasks
      USING (tenant_id=current_setting('app.tenant_id',true) AND user_id=current_setting('app.user_id',true))
      WITH CHECK (tenant_id=current_setting('app.tenant_id',true) AND user_id=current_setting('app.user_id',true))""")
    op.execute("ALTER TABLE governance.llm_usage_events ADD COLUMN IF NOT EXISTS task_id varchar(64)")
    op.execute("ALTER TABLE governance.llm_usage_events ADD COLUMN IF NOT EXISTS route_profile varchar(32) NOT NULL DEFAULT 'standard'")
    op.execute("ALTER TABLE governance.llm_usage_events ADD COLUMN IF NOT EXISTS fallback_from varchar(32)")
    op.execute("ALTER TABLE governance.llm_usage_events ADD COLUMN IF NOT EXISTS price_version varchar(64) NOT NULL DEFAULT 'configured-v1'")
    op.execute("ALTER TABLE governance.llm_usage_events ADD COLUMN IF NOT EXISTS cost_source varchar(32) NOT NULL DEFAULT 'configured_price'")

def downgrade() -> None:
    op.execute("ALTER TABLE governance.llm_usage_events DROP COLUMN IF EXISTS cost_source")
    op.execute("ALTER TABLE governance.llm_usage_events DROP COLUMN IF EXISTS price_version")
    op.execute("ALTER TABLE governance.llm_usage_events DROP COLUMN IF EXISTS fallback_from")
    op.execute("ALTER TABLE governance.llm_usage_events DROP COLUMN IF EXISTS route_profile")
    op.execute("ALTER TABLE governance.llm_usage_events DROP COLUMN IF EXISTS task_id")
    op.execute("ALTER TABLE governance.chat_turns DROP COLUMN IF EXISTS task_id")
    op.execute("DROP TABLE governance.conversation_tasks")

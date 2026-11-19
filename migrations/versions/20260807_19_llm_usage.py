"""Persist user-visible LLM token usage."""

from alembic import op


revision = "20260807_19"
down_revision = "20260806_18"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
      CREATE TABLE IF NOT EXISTS governance.llm_usage_events (
        event_id varchar(64) PRIMARY KEY,
        tenant_id varchar(128) NOT NULL,
        user_id varchar(128) NOT NULL,
        trace_id varchar(128),
        chat_session_id varchar(128),
        operation varchar(32) NOT NULL,
        model varchar(160) NOT NULL,
        input_tokens integer NOT NULL CHECK (input_tokens >= 0),
        output_tokens integer NOT NULL CHECK (output_tokens >= 0),
        total_tokens integer NOT NULL CHECK (total_tokens = input_tokens + output_tokens),
        estimated_cost numeric(18,8) NOT NULL DEFAULT 0 CHECK (estimated_cost >= 0),
        usage_source varchar(32) NOT NULL DEFAULT 'provider_reported',
        created_at timestamptz NOT NULL DEFAULT now(),
        CHECK (usage_source IN ('provider_reported', 'estimated'))
      )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_llm_usage_events_owner_time ON governance.llm_usage_events(tenant_id,user_id,created_at DESC)")
    op.execute("ALTER TABLE governance.llm_usage_events ENABLE ROW LEVEL SECURITY")
    op.execute("DROP POLICY IF EXISTS llm_usage_events_owner ON governance.llm_usage_events")
    op.execute("""
      CREATE POLICY llm_usage_events_owner ON governance.llm_usage_events
      USING (tenant_id = current_setting('app.tenant_id', true)
             AND user_id = current_setting('app.user_id', true))
      WITH CHECK (tenant_id = current_setting('app.tenant_id', true)
                  AND user_id = current_setting('app.user_id', true))
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS governance.llm_usage_events")

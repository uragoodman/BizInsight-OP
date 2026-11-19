"""Lifecycle, immutable authorization references, and execution attempts."""

from alembic import op


revision = "20260923_45"
down_revision = "20260921_44"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS governance.authorization_snapshots (
          authz_ref varchar(160) PRIMARY KEY,
          tenant_id varchar(128) NOT NULL,
          user_id varchar(128) NOT NULL,
          permission_version varchar(128) NOT NULL,
          policy_set_version varchar(128) NOT NULL,
          snapshot_digest varchar(128) NOT NULL,
          seller_scope_type varchar(32) NOT NULL DEFAULT 'explicit',
          metric_policy_ref varchar(160),
          dimension_policy_ref varchar(160),
          tool_policy_ref varchar(160),
          created_at timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS governance.conversation_executions (
          execution_id varchar(128) PRIMARY KEY,
          turn_id varchar(128) NOT NULL,
          tenant_id varchar(128) NOT NULL,
          user_id varchar(128) NOT NULL,
          attempt_no integer NOT NULL,
          status varchar(32) NOT NULL DEFAULT 'queued',
          interaction_state varchar(64) NOT NULL DEFAULT 'none',
          langgraph_thread_id varchar(160),
          trace_id varchar(160),
          authorized_permission_version varchar(128) NOT NULL,
          authorized_policy_set_version varchar(128) NOT NULL,
          authorization_fingerprint varchar(128) NOT NULL,
          query_scope_fingerprint varchar(128) NOT NULL,
          heartbeat_at timestamptz,
          result_snapshot jsonb,
          error_code varchar(64),
          error_detail jsonb,
          terminal_event varchar(128),
          final_node varchar(160),
          execution_request_id varchar(160),
          created_at timestamptz NOT NULL DEFAULT now(),
          started_at timestamptz,
          finished_at timestamptz,
          FOREIGN KEY (turn_id) REFERENCES governance.chat_turns(turn_id) ON DELETE CASCADE,
          UNIQUE(turn_id, attempt_no),
          CHECK (status IN ('queued','running','suspended','completed','abstained','failed','cancelled','interrupted'))
        )
    """)
    op.execute("ALTER TABLE governance.chat_turns ALTER COLUMN answer DROP NOT NULL")
    for statement in (
        "ALTER TABLE governance.chat_turns ADD COLUMN IF NOT EXISTS client_turn_id varchar(160)",
        "ALTER TABLE governance.chat_turns ADD COLUMN IF NOT EXISTS status varchar(32) NOT NULL DEFAULT 'completed'",
        "ALTER TABLE governance.chat_turns ADD COLUMN IF NOT EXISTS interaction_state varchar(64) NOT NULL DEFAULT 'none'",
        "ALTER TABLE governance.chat_turns ADD COLUMN IF NOT EXISTS final_execution_id varchar(128)",
        "ALTER TABLE governance.chat_turns ADD COLUMN IF NOT EXISTS final_trace_id varchar(160)",
        "ALTER TABLE governance.chat_turns ADD COLUMN IF NOT EXISTS inherited_from_turn_id varchar(128)",
        "ALTER TABLE governance.chat_turns ADD COLUMN IF NOT EXISTS result_snapshot jsonb",
        "ALTER TABLE governance.chat_turns ADD COLUMN IF NOT EXISTS context_snapshot jsonb",
        "ALTER TABLE governance.chat_turns ADD COLUMN IF NOT EXISTS warning_codes jsonb NOT NULL DEFAULT '[]'::jsonb",
        "ALTER TABLE governance.chat_turns ADD COLUMN IF NOT EXISTS error_code varchar(64)",
        "ALTER TABLE governance.chat_turns ADD COLUMN IF NOT EXISTS error_detail jsonb",
        "ALTER TABLE governance.chat_turns ADD COLUMN IF NOT EXISTS finalized_at timestamptz",
        "ALTER TABLE governance.chat_turns ADD COLUMN IF NOT EXISTS permission_version varchar(128)",
        "ALTER TABLE governance.chat_turns ADD COLUMN IF NOT EXISTS policy_set_version varchar(128)",
        "ALTER TABLE governance.chat_turns ADD COLUMN IF NOT EXISTS authz_ref varchar(160)",
        "ALTER TABLE governance.chat_turns ADD COLUMN IF NOT EXISTS authz_digest varchar(128)",
        "ALTER TABLE governance.chat_turns ADD COLUMN IF NOT EXISTS scope_fingerprint varchar(128)",
    ):
        op.execute(statement)
    op.execute("ALTER TABLE governance.chat_turns DROP CONSTRAINT IF EXISTS chat_turns_status_check")
    op.execute("""
        ALTER TABLE governance.chat_turns
        ADD CONSTRAINT chat_turns_status_check
        CHECK (status IN ('queued','running','completed','abstained','failed','cancelled','interrupted'))
    """)
    op.execute("ALTER TABLE governance.chat_turns ADD CONSTRAINT chat_turns_final_execution_fk FOREIGN KEY (final_execution_id) REFERENCES governance.conversation_executions(execution_id) DEFERRABLE INITIALLY DEFERRED")
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_chat_turn_client_id ON governance.chat_turns(tenant_id,user_id,chat_session_id,client_turn_id) WHERE client_turn_id IS NOT NULL")
    op.execute("CREATE INDEX IF NOT EXISTS idx_chat_turns_status ON governance.chat_turns(tenant_id,user_id,status,created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_executions_turn_status ON governance.conversation_executions(tenant_id,user_id,turn_id,status)")
    op.execute("ALTER TABLE governance.conversation_tasks DROP CONSTRAINT IF EXISTS conversation_tasks_status_check")
    op.execute("""
        ALTER TABLE governance.conversation_tasks
        ADD CONSTRAINT conversation_tasks_status_check
        CHECK (status IN ('draft','awaiting_scope','clarification_required','awaiting_confirmation','running','completed','abstained','failed','cancelled','interrupted','superseded'))
    """)
    op.execute("ALTER TABLE governance.authorization_snapshots ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE governance.authorization_snapshots FORCE ROW LEVEL SECURITY")
    op.execute("""
        CREATE POLICY authorization_snapshots_isolation ON governance.authorization_snapshots
        USING (tenant_id=current_setting('app.tenant_id',true) AND user_id=current_setting('app.user_id',true))
        WITH CHECK (tenant_id=current_setting('app.tenant_id',true) AND user_id=current_setting('app.user_id',true))
    """)
    op.execute("ALTER TABLE governance.conversation_executions ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE governance.conversation_executions FORCE ROW LEVEL SECURITY")
    op.execute("""
        CREATE POLICY conversation_executions_isolation ON governance.conversation_executions
        USING (tenant_id=current_setting('app.tenant_id',true) AND user_id=current_setting('app.user_id',true))
        WITH CHECK (tenant_id=current_setting('app.tenant_id',true) AND user_id=current_setting('app.user_id',true))
    """)
    op.execute("""
        ALTER TABLE governance.chat_turns
        ADD CONSTRAINT chat_turns_status_interaction_check
        CHECK (interaction_state IN ('none','awaiting_scope','awaiting_metric_confirmation','awaiting_human_input'))
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_executions_heartbeat ON governance.conversation_executions(status,heartbeat_at)")
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_active_execution_per_turn
        ON governance.conversation_executions(turn_id)
        WHERE status IN ('queued','running','suspended')
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE governance.chat_turns DROP CONSTRAINT IF EXISTS chat_turns_status_interaction_check")
    op.execute("ALTER TABLE governance.chat_turns DROP CONSTRAINT IF EXISTS chat_turns_final_execution_fk")
    op.execute("ALTER TABLE governance.chat_turns DROP CONSTRAINT IF EXISTS chat_turns_status_check")
    op.execute("DROP TABLE IF EXISTS governance.conversation_executions")
    op.execute("DROP TABLE IF EXISTS governance.authorization_snapshots")
    for column in (
        "client_turn_id", "status", "interaction_state", "final_execution_id",
        "final_trace_id", "inherited_from_turn_id", "result_snapshot",
        "context_snapshot", "warning_codes", "error_code", "error_detail",
        "finalized_at", "permission_version", "policy_set_version",
        "authz_ref", "authz_digest", "scope_fingerprint",
    ):
        op.execute(f"ALTER TABLE governance.chat_turns DROP COLUMN IF EXISTS {column}")
    op.execute("DROP INDEX IF EXISTS uq_chat_turn_client_id")
    op.execute("DROP INDEX IF EXISTS uq_active_execution_per_turn")

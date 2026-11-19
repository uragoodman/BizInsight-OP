"""Persist compact user-question/final-answer chat history."""

from alembic import op

revision = "20260727_09"
down_revision = "20260727_08"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE governance.chat_sessions (
            tenant_id varchar(128) NOT NULL,
            user_id varchar(128) NOT NULL,
            chat_session_id varchar(128) NOT NULL,
            title varchar(160) NOT NULL DEFAULT '新会话',
            scope_data jsonb,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (tenant_id, user_id, chat_session_id)
        )
    """)
    op.execute("""
        CREATE TABLE governance.chat_turns (
            turn_id varchar(128) PRIMARY KEY,
            tenant_id varchar(128) NOT NULL,
            user_id varchar(128) NOT NULL,
            chat_session_id varchar(128) NOT NULL,
            question text NOT NULL,
            answer text NOT NULL,
            task_type varchar(64) NOT NULL,
            trace_id varchar(128) NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            FOREIGN KEY (tenant_id, user_id, chat_session_id)
                REFERENCES governance.chat_sessions(tenant_id, user_id, chat_session_id) ON DELETE CASCADE
        )
    """)
    op.execute("CREATE INDEX idx_chat_sessions_updated ON governance.chat_sessions(tenant_id, user_id, updated_at DESC)")
    op.execute("CREATE INDEX idx_chat_turns_session ON governance.chat_turns(tenant_id, user_id, chat_session_id, created_at DESC)")
    for table in ("chat_sessions", "chat_turns"):
        op.execute(f"ALTER TABLE governance.{table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE governance.{table} FORCE ROW LEVEL SECURITY")
        op.execute(f"""CREATE POLICY {table}_isolation ON governance.{table}
            USING (tenant_id = current_setting('app.tenant_id', true)
                   AND user_id = current_setting('app.user_id', true))
            WITH CHECK (tenant_id = current_setting('app.tenant_id', true)
                        AND user_id = current_setting('app.user_id', true))""")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS governance.chat_turns")
    op.execute("DROP TABLE IF EXISTS governance.chat_sessions")

"""P0 controlled business-analysis memory tables."""

from alembic import op


revision = "20260722_04"
down_revision = "20260717_03"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE governance.memory_sessions (
            -- 开发态允许受信的临时调试用户；生产环境的 TenantContext 仍由 IAM 认证生成。
            tenant_id varchar(128) NOT NULL,
            user_id varchar(128) NOT NULL,
            memory_session_id varchar(128) NOT NULL,
            session_summary varchar(1000) NOT NULL,
            open_questions jsonb NOT NULL DEFAULT '[]'::jsonb,
            last_scope_fingerprint char(64) NOT NULL,
            last_seller_id varchar(64) NOT NULL,
            last_start_date date NOT NULL,
            last_end_date date NOT NULL,
            last_trace_id varchar(128) NOT NULL,
            updated_at timestamptz NOT NULL DEFAULT now(),
            expires_at timestamptz NOT NULL,
            PRIMARY KEY (tenant_id, user_id, memory_session_id),
            CHECK (last_end_date >= last_start_date)
        )
    """)
    op.execute("""
        CREATE TABLE governance.analysis_memories (
            memory_id varchar(128) PRIMARY KEY,
            tenant_id varchar(128) NOT NULL,
            user_id varchar(128) NOT NULL,
            seller_id varchar(64) NOT NULL,
            start_date date NOT NULL,
            end_date date NOT NULL,
            scope_fingerprint char(64) NOT NULL,
            metric_tags jsonb NOT NULL DEFAULT '[]'::jsonb,
            task_type varchar(64) NOT NULL,
            insight_text varchar(500) NOT NULL,
            limitations varchar(500) NOT NULL DEFAULT '',
            evidence_ids jsonb NOT NULL,
            source_trace_id varchar(128) NOT NULL,
            status varchar(16) NOT NULL DEFAULT 'active' CHECK (status IN ('active','revoked')),
            created_at timestamptz NOT NULL DEFAULT now(),
            revoked_at timestamptz,
            UNIQUE (tenant_id, user_id, source_trace_id, insight_text),
            CHECK (end_date >= start_date)
        )
    """)
    op.execute("CREATE INDEX idx_memory_sessions_expiry ON governance.memory_sessions(tenant_id, user_id, expires_at)")
    op.execute("CREATE INDEX idx_analysis_memories_retrieval ON governance.analysis_memories(tenant_id, user_id, seller_id, status, start_date, end_date, created_at DESC)")
    for table in ("memory_sessions", "analysis_memories"):
        op.execute(f"ALTER TABLE governance.{table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE governance.{table} FORCE ROW LEVEL SECURITY")
        op.execute(f"""CREATE POLICY {table}_isolation ON governance.{table}
            USING (tenant_id = current_setting('app.tenant_id', true)
                   AND user_id = current_setting('app.user_id', true))
            WITH CHECK (tenant_id = current_setting('app.tenant_id', true)
                        AND user_id = current_setting('app.user_id', true))""")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS governance.analysis_memories")
    op.execute("DROP TABLE IF EXISTS governance.memory_sessions")

"""Add versioned session focus and idempotent HITL interaction records."""

from alembic import op


revision = "20261102_61"
down_revision = "20261101_60"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE governance.chat_sessions ADD COLUMN IF NOT EXISTS session_state_version bigint NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE governance.chat_sessions ADD COLUMN IF NOT EXISTS active_task_id varchar(128)")
    op.execute("ALTER TABLE governance.chat_sessions ADD COLUMN IF NOT EXISTS focus_owner_execution_id varchar(128)")
    op.execute("ALTER TABLE governance.chat_sessions ADD COLUMN IF NOT EXISTS last_committed_turn_id varchar(128)")
    op.execute("ALTER TABLE governance.chat_sessions ADD COLUMN IF NOT EXISTS committed_at timestamptz")
    op.execute("ALTER TABLE governance.chat_sessions ADD COLUMN IF NOT EXISTS turn_commit_version bigint NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE governance.chat_sessions ADD COLUMN IF NOT EXISTS conversation_summary text NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE governance.chat_sessions ADD COLUMN IF NOT EXISTS summary_covered_from_version bigint")
    op.execute("ALTER TABLE governance.chat_sessions ADD COLUMN IF NOT EXISTS summary_covered_to_version bigint")
    op.execute("ALTER TABLE governance.chat_sessions ADD COLUMN IF NOT EXISTS summary_covered_from_turn_id varchar(128)")
    op.execute("ALTER TABLE governance.chat_sessions ADD COLUMN IF NOT EXISTS summary_covered_to_turn_id varchar(128)")
    op.execute("ALTER TABLE governance.chat_sessions ADD COLUMN IF NOT EXISTS summary_policy_version varchar(64) NOT NULL DEFAULT 'summary-policy-v1'")
    op.execute("ALTER TABLE governance.chat_turns ADD COLUMN IF NOT EXISTS commit_version bigint")
    op.execute("""
        WITH ranked AS (
            SELECT turn_id,
                   row_number() OVER (
                       PARTITION BY tenant_id,user_id,chat_session_id
                       ORDER BY created_at,turn_id
                   ) AS commit_version
            FROM governance.chat_turns
            WHERE status IN ('completed','abstained','failed','cancelled','interrupted')
        )
        UPDATE governance.chat_turns t
           SET commit_version=r.commit_version
          FROM ranked r
         WHERE t.turn_id=r.turn_id AND t.commit_version IS NULL
    """)
    op.execute("""
        UPDATE governance.chat_sessions s
           SET turn_commit_version=COALESCE(x.max_version,0),
               last_committed_turn_id=x.latest_turn_id,
               committed_at=CASE WHEN x.max_version IS NULL THEN NULL ELSE now() END
          FROM (
              SELECT tenant_id,user_id,chat_session_id,
                     max(commit_version) AS max_version,
                     (array_agg(turn_id ORDER BY commit_version DESC))[1] AS latest_turn_id
              FROM governance.chat_turns
              WHERE commit_version IS NOT NULL
              GROUP BY tenant_id,user_id,chat_session_id
          ) x
         WHERE s.tenant_id=x.tenant_id AND s.user_id=x.user_id
           AND s.chat_session_id=x.chat_session_id
    """)
    op.execute("""
        WITH grouped AS (
            SELECT tenant_id,user_id,chat_session_id,
                   max(commit_version) AS max_version
            FROM governance.chat_turns
            WHERE commit_version IS NOT NULL
            GROUP BY tenant_id,user_id,chat_session_id
        ), rendered AS (
            SELECT t.tenant_id,t.user_id,t.chat_session_id,
                   t.commit_version,t.turn_id,
                   '用户：' || left(regexp_replace(t.question, '\\s+', ' ', 'g'),320)
                   || CASE WHEN coalesce(t.answer,'')='' THEN ''
                           ELSE '；答复：' || left(regexp_replace(t.answer, '\\s+', ' ', 'g'),640)
                      END AS line
            FROM governance.chat_turns t
            JOIN grouped g USING (tenant_id,user_id,chat_session_id)
            WHERE t.commit_version <= g.max_version - 20
        ), budgeted AS (
            -- Keep the newest contiguous suffix that fits the frozen summary
            -- budget.  The runtime rebuild uses the configured tokenizer and a
            -- 4k-token budget; this migration uses a conservative character
            -- bound so a migrated session cannot exceed that budget before
            -- its next Turn Commit rebuilds the summary exactly.
            SELECT *,
                   sum(length(line)+1) OVER (
                       PARTITION BY tenant_id,user_id,chat_session_id
                       ORDER BY commit_version DESC
                   ) AS trailing_chars
            FROM rendered
        ), summaries AS (
            SELECT tenant_id,user_id,chat_session_id,
                   min(commit_version) AS covered_from_version,
                   max(commit_version) AS covered_to_version,
                   (array_agg(turn_id ORDER BY commit_version))[1] AS covered_from_turn_id,
                   (array_agg(turn_id ORDER BY commit_version DESC))[1] AS covered_to_turn_id,
                   string_agg(line, E'\\n' ORDER BY commit_version) AS summary_text
            FROM budgeted
            WHERE trailing_chars <= 2400
            GROUP BY tenant_id,user_id,chat_session_id
        )
        UPDATE governance.chat_sessions s
           SET conversation_summary=x.summary_text,
               summary_covered_from_version=x.covered_from_version,
               summary_covered_to_version=x.covered_to_version,
               summary_covered_from_turn_id=x.covered_from_turn_id,
               summary_covered_to_turn_id=x.covered_to_turn_id,
               summary_policy_version='summary-policy-v1'
          FROM summaries x
         WHERE s.tenant_id=x.tenant_id AND s.user_id=x.user_id
           AND s.chat_session_id=x.chat_session_id
    """)
    op.execute("""
        UPDATE governance.chat_sessions s
           SET active_task_id=(
              SELECT t.task_id FROM governance.conversation_tasks t
              WHERE t.tenant_id=s.tenant_id AND t.user_id=s.user_id
                AND t.chat_session_id=s.chat_session_id
              ORDER BY t.updated_at DESC,t.task_id DESC LIMIT 1
           )
         WHERE EXISTS (
              SELECT 1 FROM governance.conversation_tasks t
              WHERE t.tenant_id=s.tenant_id AND t.user_id=s.user_id
                AND t.chat_session_id=s.chat_session_id
         )
    """)
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_chat_turn_commit_version ON governance.chat_turns(tenant_id,user_id,chat_session_id,commit_version) WHERE commit_version IS NOT NULL")

    op.execute("""
        CREATE TABLE governance.chat_review_interactions (
            interaction_id varchar(128) PRIMARY KEY,
            tenant_id varchar(128) NOT NULL,
            user_id varchar(128) NOT NULL,
            execution_id varchar(128) NOT NULL REFERENCES governance.conversation_executions(execution_id) ON DELETE CASCADE,
            node_name varchar(160) NOT NULL,
            interrupt_index varchar(160) NOT NULL,
            decision_digest varchar(64) NOT NULL,
            status varchar(16) NOT NULL DEFAULT 'processing'
                CHECK (status IN ('processing','completed','failed')),
            result_snapshot jsonb,
            claim_token varchar(96),
            lease_expires_at timestamptz,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            UNIQUE (execution_id, node_name, interrupt_index)
        )
    """)
    op.execute("ALTER TABLE governance.chat_review_interactions ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE governance.chat_review_interactions FORCE ROW LEVEL SECURITY")
    op.execute("""
        CREATE POLICY chat_review_interactions_isolation
        ON governance.chat_review_interactions
        USING (tenant_id=current_setting('app.tenant_id',true)
               AND user_id=current_setting('app.user_id',true))
        WITH CHECK (tenant_id=current_setting('app.tenant_id',true)
                    AND user_id=current_setting('app.user_id',true))
    """)
    op.execute("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='bizinsight_app') THEN
                GRANT SELECT, INSERT, UPDATE ON governance.chat_review_interactions TO bizinsight_app;
                GRANT SELECT, UPDATE ON governance.chat_sessions TO bizinsight_app;
            END IF;
        END $$;
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS governance.chat_review_interactions")
    op.execute("DROP INDEX IF EXISTS governance.uq_chat_turn_commit_version")
    op.execute("ALTER TABLE governance.chat_turns DROP COLUMN IF EXISTS commit_version")
    for column in ("summary_policy_version", "summary_covered_to_turn_id", "summary_covered_from_turn_id", "summary_covered_to_version", "summary_covered_from_version", "conversation_summary", "turn_commit_version", "committed_at", "last_committed_turn_id", "focus_owner_execution_id", "active_task_id", "session_state_version"):
        op.execute(f"ALTER TABLE governance.chat_sessions DROP COLUMN IF EXISTS {column}")

"""Strengthen final execution/trace ownership invariants."""

from alembic import op


revision = "20260923_46"
down_revision = "20260923_45"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # A foreign key on execution_id alone proves only that *some* execution
    # exists.  The composite key below proves it belongs to this turn.
    op.execute("""
        ALTER TABLE governance.conversation_executions
        ADD CONSTRAINT conversation_executions_execution_turn_key
        UNIQUE (execution_id, turn_id)
    """)
    op.execute("""
        ALTER TABLE governance.conversation_executions
        ADD CONSTRAINT conversation_executions_execution_trace_key
        UNIQUE (execution_id, trace_id)
    """)
    op.execute("""
        ALTER TABLE governance.chat_turns
        DROP CONSTRAINT IF EXISTS chat_turns_final_execution_fk
    """)
    op.execute("""
        ALTER TABLE governance.chat_turns
        ADD CONSTRAINT chat_turns_final_execution_fk
        FOREIGN KEY (final_execution_id, turn_id)
        REFERENCES governance.conversation_executions (execution_id, turn_id)
        DEFERRABLE INITIALLY DEFERRED
    """)
    op.execute("""
        ALTER TABLE governance.chat_turns
        ADD CONSTRAINT chat_turns_final_trace_required
        CHECK (final_execution_id IS NULL OR final_trace_id IS NOT NULL)
    """)
    op.execute("""
        ALTER TABLE governance.chat_turns
        ADD CONSTRAINT chat_turns_final_trace_fk
        FOREIGN KEY (final_execution_id, final_trace_id)
        REFERENCES governance.conversation_executions (execution_id, trace_id)
        DEFERRABLE INITIALLY DEFERRED
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE governance.chat_turns DROP CONSTRAINT IF EXISTS chat_turns_final_trace_fk")
    op.execute("ALTER TABLE governance.chat_turns DROP CONSTRAINT IF EXISTS chat_turns_final_trace_required")
    op.execute("ALTER TABLE governance.chat_turns DROP CONSTRAINT IF EXISTS chat_turns_final_execution_fk")
    op.execute("""
        ALTER TABLE governance.chat_turns
        ADD CONSTRAINT chat_turns_final_execution_fk
        FOREIGN KEY (final_execution_id)
        REFERENCES governance.conversation_executions (execution_id)
        DEFERRABLE INITIALLY DEFERRED
    """)
    op.execute("ALTER TABLE governance.conversation_executions DROP CONSTRAINT IF EXISTS conversation_executions_execution_trace_key")
    op.execute("ALTER TABLE governance.conversation_executions DROP CONSTRAINT IF EXISTS conversation_executions_execution_turn_key")

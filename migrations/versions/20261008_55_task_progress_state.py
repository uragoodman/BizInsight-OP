"""Versioned, bounded continuity state for conversation tasks."""

from alembic import op


revision = "20261008_55"
down_revision = "20261006_54"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE governance.conversation_tasks
        ADD COLUMN task_state_version bigint NOT NULL DEFAULT 0,
        ADD COLUMN task_state jsonb NOT NULL DEFAULT
          jsonb_build_object(
            'schema_version', 'task-state-v1',
            'version', 1,
            'scope_fingerprint', '',
            'latest_trace_id', NULL,
            'completed_steps', '[]'::jsonb,
            'approved_claims', '[]'::jsonb,
            'hypothesis_chain_summary', '{}'::jsonb,
            'evidence_refs', '[]'::jsonb,
            'recent_turns', '[]'::jsonb
          )
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE governance.conversation_tasks DROP COLUMN task_state")
    op.execute("ALTER TABLE governance.conversation_tasks DROP COLUMN task_state_version")

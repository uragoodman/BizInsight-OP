"""Add snapshot_data and version for multi-turn context carryover (Phase 7).

These columns enable SessionMemory v2: rolling snapshots of evidence refs,
approved claims, and hypothesis chains that survive across analysis turns.
"""

from alembic import op


revision = "20260727_08"
down_revision = "20260726_07"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE governance.memory_sessions "
               "ADD COLUMN IF NOT EXISTS version integer NOT NULL DEFAULT 1")
    op.execute("ALTER TABLE governance.memory_sessions "
               "ADD COLUMN IF NOT EXISTS snapshot_data jsonb DEFAULT NULL")
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_mem_sessions_v2
        ON governance.memory_sessions(tenant_id, user_id, version)
        WHERE version >= 2
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_mem_sessions_v2")
    op.execute("ALTER TABLE governance.memory_sessions DROP COLUMN IF EXISTS snapshot_data")
    op.execute("ALTER TABLE governance.memory_sessions DROP COLUMN IF EXISTS version")

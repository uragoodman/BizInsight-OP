"""Phase A memory governance: deterministic deduplication and review states."""
from alembic import op

revision = "20260816_28"
down_revision = "20260815_27"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE governance.analysis_memories DROP CONSTRAINT IF EXISTS analysis_memories_status_check")
    op.execute("""ALTER TABLE governance.analysis_memories ADD CONSTRAINT analysis_memories_status_check
        CHECK (status IN ('active','pending_review','superseded','stale','revoked','rejected'))""")
    op.execute("ALTER TABLE governance.analysis_memories ADD COLUMN IF NOT EXISTS normalized_claim_hash varchar(64)")
    op.execute("ALTER TABLE governance.analysis_memories ADD COLUMN IF NOT EXISTS review_reason text NOT NULL DEFAULT ''")
    op.execute("""CREATE UNIQUE INDEX IF NOT EXISTS uq_analysis_memory_active_pending_hash
        ON governance.analysis_memories(tenant_id, user_id, seller_id, scope_fingerprint, normalized_claim_hash)
        WHERE status IN ('active','pending_review') AND normalized_claim_hash IS NOT NULL""")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS governance.uq_analysis_memory_active_pending_hash")
    op.execute("ALTER TABLE governance.analysis_memories DROP COLUMN IF EXISTS review_reason")
    op.execute("ALTER TABLE governance.analysis_memories DROP COLUMN IF EXISTS normalized_claim_hash")
    op.execute("ALTER TABLE governance.analysis_memories DROP CONSTRAINT IF EXISTS analysis_memories_status_check")
    op.execute("""ALTER TABLE governance.analysis_memories ADD CONSTRAINT analysis_memories_status_check
        CHECK (status IN ('active','superseded','stale','revoked'))""")

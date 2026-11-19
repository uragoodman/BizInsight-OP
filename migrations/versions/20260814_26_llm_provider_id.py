"""Provider id column for LLM usage (multi-provider fallback ledger)."""
from alembic import op

revision = "20260814_26"
down_revision = "20260813_25"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE governance.llm_usage_events ADD COLUMN IF NOT EXISTS provider_id varchar(32) NOT NULL DEFAULT ''")


def downgrade() -> None:
    op.execute("ALTER TABLE governance.llm_usage_events DROP COLUMN IF EXISTS provider_id")

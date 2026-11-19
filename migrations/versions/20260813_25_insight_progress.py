"""Insight generation stage progress column for SSE progress push."""
from alembic import op

revision = "20260813_25"
down_revision = "20260812_24"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE governance.tenant_report_insights ADD COLUMN IF NOT EXISTS progress jsonb NOT NULL DEFAULT '{}'::jsonb")


def downgrade() -> None:
    op.execute("ALTER TABLE governance.tenant_report_insights DROP COLUMN IF EXISTS progress")

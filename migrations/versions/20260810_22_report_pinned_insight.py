"""Pin the AI insight version at report publish time for audit consistency."""
from alembic import op

revision = "20260810_22"
down_revision = "20260809_21"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE governance.tenant_reports ADD COLUMN IF NOT EXISTS pinned_insight_version integer")


def downgrade() -> None:
    op.execute("ALTER TABLE governance.tenant_reports DROP COLUMN IF EXISTS pinned_insight_version")

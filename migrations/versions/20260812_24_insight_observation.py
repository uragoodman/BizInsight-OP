"""Insight observation columns for async generation quality metrics."""
from alembic import op

revision = "20260812_24"
down_revision = "20260811_23"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE governance.tenant_report_insights ADD COLUMN IF NOT EXISTS duration_ms integer")
    op.execute("ALTER TABLE governance.tenant_report_insights ADD COLUMN IF NOT EXISTS input_tokens integer")
    op.execute("ALTER TABLE governance.tenant_report_insights ADD COLUMN IF NOT EXISTS output_tokens integer")
    op.execute("ALTER TABLE governance.tenant_report_insights ADD COLUMN IF NOT EXISTS estimated_cost double precision")


def downgrade() -> None:
    op.execute("ALTER TABLE governance.tenant_report_insights DROP COLUMN IF EXISTS estimated_cost")
    op.execute("ALTER TABLE governance.tenant_report_insights DROP COLUMN IF EXISTS output_tokens")
    op.execute("ALTER TABLE governance.tenant_report_insights DROP COLUMN IF EXISTS input_tokens")
    op.execute("ALTER TABLE governance.tenant_report_insights DROP COLUMN IF EXISTS duration_ms")

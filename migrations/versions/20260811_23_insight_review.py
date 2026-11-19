"""Insight lifecycle: pending_review status + human review fields + attribution snapshot."""
from alembic import op

revision = "20260811_23"
down_revision = "20260810_22"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # status 增加 pending_review（Critic 不可用时的 fail-closed 状态）。
    op.execute("ALTER TABLE governance.tenant_report_insights DROP CONSTRAINT IF EXISTS tenant_report_insights_status_check")
    op.execute("""ALTER TABLE governance.tenant_report_insights ADD CONSTRAINT tenant_report_insights_status_check
        CHECK (status IN ('generating','draft','rejected','approved','published','pending_review'))""")
    # 洞察输入证据快照（受控补查产生的归因证据）。
    op.execute("ALTER TABLE governance.tenant_report_insights ADD COLUMN IF NOT EXISTS attribution_evidence jsonb NOT NULL DEFAULT '{}'::jsonb")
    # 人工审核轨迹。
    op.execute("ALTER TABLE governance.tenant_report_insights ADD COLUMN IF NOT EXISTS reviewed_by varchar(128)")
    op.execute("ALTER TABLE governance.tenant_report_insights ADD COLUMN IF NOT EXISTS reviewed_at timestamptz")
    op.execute("ALTER TABLE governance.tenant_report_insights ADD COLUMN IF NOT EXISTS review_reason text NOT NULL DEFAULT ''")


def downgrade() -> None:
    op.execute("ALTER TABLE governance.tenant_report_insights DROP COLUMN IF EXISTS review_reason")
    op.execute("ALTER TABLE governance.tenant_report_insights DROP COLUMN IF EXISTS reviewed_at")
    op.execute("ALTER TABLE governance.tenant_report_insights DROP COLUMN IF EXISTS reviewed_by")
    op.execute("ALTER TABLE governance.tenant_report_insights DROP COLUMN IF EXISTS attribution_evidence")
    op.execute("ALTER TABLE governance.tenant_report_insights DROP CONSTRAINT IF EXISTS tenant_report_insights_status_check")

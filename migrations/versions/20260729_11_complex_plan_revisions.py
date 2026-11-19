"""Add revision chain and controlled human preferences to complex plans."""

from alembic import op

revision = "20260729_11"
down_revision = "20260728_10"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE governance.complex_analysis_plans ADD COLUMN parent_plan_id varchar(64) REFERENCES governance.complex_analysis_plans(plan_id)")
    op.execute("ALTER TABLE governance.complex_analysis_plans ADD COLUMN revision integer NOT NULL DEFAULT 1")
    op.execute("ALTER TABLE governance.complex_analysis_plans ADD COLUMN preferences_data jsonb NOT NULL DEFAULT '{}'::jsonb")
    op.execute("CREATE INDEX idx_complex_plan_revision_chain ON governance.complex_analysis_plans(tenant_id,user_id,parent_plan_id,revision DESC)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS governance.idx_complex_plan_revision_chain")
    op.execute("ALTER TABLE governance.complex_analysis_plans DROP COLUMN IF EXISTS preferences_data")
    op.execute("ALTER TABLE governance.complex_analysis_plans DROP COLUMN IF EXISTS revision")
    op.execute("ALTER TABLE governance.complex_analysis_plans DROP COLUMN IF EXISTS parent_plan_id")

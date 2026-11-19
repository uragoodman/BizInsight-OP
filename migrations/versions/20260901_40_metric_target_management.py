"""Add explicit seller ownership to tenant metric targets."""

from alembic import op


revision = "20260901_40"
down_revision = "20260829_39"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE analytics.tenant_metric_targets ADD COLUMN IF NOT EXISTS seller_id varchar(64)")
    op.execute("""
        UPDATE analytics.tenant_metric_targets t
        SET seller_id = a.seller_id
        FROM (
          SELECT tenant_id, min(seller_id) AS seller_id
          FROM iam.tenant_seller_accounts GROUP BY tenant_id
        ) a
        WHERE t.tenant_id = a.tenant_id AND t.seller_id IS NULL
    """)
    op.execute("ALTER TABLE analytics.tenant_metric_targets ALTER COLUMN seller_id SET NOT NULL")
    op.execute("ALTER TABLE analytics.tenant_metric_targets DROP CONSTRAINT IF EXISTS tenant_metric_targets_pkey")
    op.execute("ALTER TABLE analytics.tenant_metric_targets ADD PRIMARY KEY (tenant_id,seller_id,metric_name,period_start,period_end,version)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_tenant_metric_targets_filters ON analytics.tenant_metric_targets(tenant_id,seller_id,metric_name,period_start,effective_at)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS analytics.idx_tenant_metric_targets_filters")
    op.execute("ALTER TABLE analytics.tenant_metric_targets DROP CONSTRAINT IF EXISTS tenant_metric_targets_pkey")
    op.execute("ALTER TABLE analytics.tenant_metric_targets ADD PRIMARY KEY (tenant_id,metric_name,period_start,period_end,version)")
    op.execute("ALTER TABLE analytics.tenant_metric_targets DROP COLUMN IF EXISTS seller_id")

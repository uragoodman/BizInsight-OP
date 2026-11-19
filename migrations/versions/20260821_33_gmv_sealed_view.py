"""Expose GMV version rows only through a sealed-version controlled view."""
from alembic import op

revision = "20260821_33"
down_revision = "20260820_32"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""CREATE VIEW analytics.v_tenant_gmv_sealed_rows WITH (security_barrier=true) AS
      SELECT rows.* FROM analytics.tenant_gmv_version_rows rows
      JOIN governance.data_versions versions
        ON versions.version_id=rows.ingestion_version AND versions.tenant_id=rows.tenant_id
      WHERE versions.dataset_id='tenant_gmv_items' AND versions.status='sealed'""")


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS analytics.v_tenant_gmv_sealed_rows")

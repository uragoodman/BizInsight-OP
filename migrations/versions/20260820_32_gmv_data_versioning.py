"""GMV vertical-slice data versions, manifests, and replay audit."""
from alembic import op

revision = "20260820_32"
down_revision = "20260819_31"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""CREATE TABLE governance.data_versions (
      version_id varchar(80) PRIMARY KEY, tenant_id varchar(128) NOT NULL,
      dataset_id varchar(64) NOT NULL, status varchar(16) NOT NULL CHECK(status IN ('writing','sealed','superseded','expired')),
      content_hash char(64) NOT NULL, schema_hash char(64) NOT NULL, row_count bigint NOT NULL,
      min_business_date date, max_business_date date, affected_sellers jsonb NOT NULL DEFAULT '[]'::jsonb,
      storage_ref varchar(160) NOT NULL, parent_version_id varchar(80), sealed_at timestamptz, created_at timestamptz NOT NULL DEFAULT now(),
      UNIQUE(tenant_id,dataset_id,content_hash)
    )""")
    op.execute("""CREATE TABLE governance.data_version_manifests (
      manifest_id varchar(80) PRIMARY KEY, tenant_id varchar(128) NOT NULL, status varchar(16) NOT NULL DEFAULT 'sealed' CHECK(status IN ('sealed','expired')),
      versions jsonb NOT NULL, metric_definition_version char(64) NOT NULL, scope_fingerprint char(64) NOT NULL,
      manifest_hash char(64) NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
      UNIQUE(tenant_id,manifest_hash)
    )""")
    op.execute("""CREATE TABLE governance.data_replay_audits (
      replay_id varchar(80) PRIMARY KEY, tenant_id varchar(128) NOT NULL, user_id varchar(128) NOT NULL,
      artifact_type varchar(32) NOT NULL CHECK(artifact_type IN ('analysis_memory','complex_execution')),
      artifact_id varchar(128) NOT NULL, mode varchar(16) NOT NULL CHECK(mode IN ('exact','compare_latest')),
      status varchar(16) NOT NULL CHECK(status IN ('matched','drifted','unavailable')),
      old_manifest_id varchar(80) NOT NULL, compared_manifest_id varchar(80), result jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now()
    )""")
    op.execute("""CREATE TABLE analytics.tenant_gmv_version_rows (
      tenant_id varchar(128) NOT NULL, ingestion_version varchar(80) NOT NULL, order_id varchar(64) NOT NULL,
      order_item_id integer NOT NULL, seller_id varchar(64) NOT NULL, purchase_date date NOT NULL,
      price numeric NOT NULL, freight_value numeric NOT NULL, order_status varchar(32) NOT NULL,
      PRIMARY KEY(tenant_id,ingestion_version,order_id,order_item_id)
    )""")
    for table in ('data_versions','data_version_manifests','data_replay_audits'):
        op.execute(f"ALTER TABLE governance.{table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE governance.{table} FORCE ROW LEVEL SECURITY")
        op.execute(f"CREATE POLICY {table}_isolation ON governance.{table} USING (tenant_id=current_setting('app.tenant_id',true) AND user_id IS NOT DISTINCT FROM current_setting('app.user_id',true)) WITH CHECK (tenant_id=current_setting('app.tenant_id',true))" if table == 'data_replay_audits' else f"CREATE POLICY {table}_isolation ON governance.{table} USING (tenant_id=current_setting('app.tenant_id',true)) WITH CHECK (tenant_id=current_setting('app.tenant_id',true))")
    op.execute("ALTER TABLE analytics.tenant_gmv_version_rows ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE analytics.tenant_gmv_version_rows FORCE ROW LEVEL SECURITY")
    op.execute("""CREATE POLICY tenant_gmv_version_rows_isolation ON analytics.tenant_gmv_version_rows
      USING (current_setting('app.scope_type',true)='platform' OR tenant_id=current_setting('app.tenant_id',true))
      WITH CHECK (tenant_id=current_setting('app.tenant_id',true))""")
    op.execute("CREATE INDEX idx_gmv_version_scope ON analytics.tenant_gmv_version_rows(tenant_id,ingestion_version,seller_id,purchase_date)")
    op.execute("ALTER TABLE governance.analysis_execution_audits ADD COLUMN IF NOT EXISTS data_version_manifest_id varchar(80)")
    op.execute("ALTER TABLE governance.analysis_execution_audits ADD COLUMN IF NOT EXISTS data_version_manifest_hash char(64)")
    op.execute("ALTER TABLE governance.memory_source_snapshots ADD COLUMN IF NOT EXISTS data_version_manifest_id varchar(80)")


def downgrade() -> None:
    op.execute("ALTER TABLE governance.memory_source_snapshots DROP COLUMN IF EXISTS data_version_manifest_id")
    op.execute("ALTER TABLE governance.analysis_execution_audits DROP COLUMN IF EXISTS data_version_manifest_hash")
    op.execute("ALTER TABLE governance.analysis_execution_audits DROP COLUMN IF EXISTS data_version_manifest_id")
    op.execute("DROP TABLE IF EXISTS analytics.tenant_gmv_version_rows")
    op.execute("DROP TABLE IF EXISTS governance.data_replay_audits")
    op.execute("DROP TABLE IF EXISTS governance.data_version_manifests")
    op.execute("DROP TABLE IF EXISTS governance.data_versions")

"""Add tenant-isolated inventory snapshots and verified experiment facts.

These tables establish source contracts only. Inventory measurements must be
ingested from a source that declares snapshot semantics and units. Experiment
rows are not eligible for causal estimation until a trusted verifier marks the
design randomized and records verifier provenance. The application role is
read-only for experiment data: verification and observation ingestion must be
performed by a separately governed process, not by ordinary API credentials.
"""

from alembic import op


revision = "20260925_53"
down_revision = "20260923_52"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS analytics")
    op.execute("CREATE SCHEMA IF NOT EXISTS governance")
    op.execute("""
        CREATE TABLE IF NOT EXISTS analytics.inventory_snapshots (
          snapshot_row_id bigserial PRIMARY KEY,
          tenant_id varchar(128) NOT NULL,
          seller_id varchar(128) NOT NULL,
          warehouse_id varchar(128) NOT NULL,
          product_id varchar(128) NOT NULL,
          sku varchar(256),
          snapshot_at timestamptz NOT NULL,
          snapshot_semantics varchar(24) NOT NULL
            CHECK (snapshot_semantics IN ('daily_close','point_in_time','latest_observed')),
          on_hand_quantity numeric,
          reserved_quantity numeric,
          available_quantity numeric,
          quantity_unit varchar(64) NOT NULL,
          source_system varchar(128) NOT NULL,
          source_batch_id varchar(128) NOT NULL,
          source_row_number integer,
          ingested_by varchar(128) NOT NULL,
          created_at timestamptz NOT NULL DEFAULT now(),
          CHECK (on_hand_quantity IS NOT NULL OR available_quantity IS NOT NULL),
          CHECK (on_hand_quantity IS NULL OR (on_hand_quantity >= 0 AND on_hand_quantity <> 'NaN'::numeric)),
          CHECK (reserved_quantity IS NULL OR (reserved_quantity >= 0 AND reserved_quantity <> 'NaN'::numeric)),
          CHECK (available_quantity IS NULL OR (available_quantity >= 0 AND available_quantity <> 'NaN'::numeric)),
          UNIQUE (tenant_id, seller_id, warehouse_id, product_id, snapshot_at, source_system)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_inventory_scope_time ON analytics.inventory_snapshots(tenant_id,seller_id,snapshot_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_inventory_product_time ON analytics.inventory_snapshots(tenant_id,seller_id,product_id,snapshot_at DESC)")
    op.execute("ALTER TABLE analytics.inventory_snapshots ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE analytics.inventory_snapshots FORCE ROW LEVEL SECURITY")
    op.execute("DROP POLICY IF EXISTS inventory_snapshots_tenant_isolation ON analytics.inventory_snapshots")
    op.execute("""
        CREATE POLICY inventory_snapshots_tenant_isolation
        ON analytics.inventory_snapshots
        USING (tenant_id = current_setting('app.tenant_id', true))
        WITH CHECK (tenant_id = current_setting('app.tenant_id', true))
    """)
    op.execute("COMMENT ON TABLE analytics.inventory_snapshots IS 'Immutable, tenant-scoped inventory measurements. Missing snapshots are not zero stock.'")
    op.execute("COMMENT ON COLUMN analytics.inventory_snapshots.available_quantity IS 'Source-reported available stock; do not infer from on-hand unless the source contract explicitly defines the formula.'")

    op.execute("""
        CREATE TABLE IF NOT EXISTS governance.experiment_registry (
          tenant_id varchar(128) NOT NULL,
          experiment_id varchar(128) NOT NULL,
          seller_id varchar(128) NOT NULL,
          outcome_metric_id varchar(128) NOT NULL,
          assignment_unit varchar(24) NOT NULL
            CHECK (assignment_unit IN ('user','customer','order','session')),
          assignment_method varchar(24) NOT NULL
            CHECK (assignment_method IN ('randomized','observational')),
          status varchar(16) NOT NULL DEFAULT 'draft'
            CHECK (status IN ('draft','active','closed')),
          starts_at timestamptz NOT NULL,
          ends_at timestamptz NOT NULL,
          randomization_verified boolean NOT NULL DEFAULT false,
          verification_ref varchar(256),
          verified_by varchar(128),
          verified_at timestamptz,
          created_by varchar(128) NOT NULL,
          created_at timestamptz NOT NULL DEFAULT now(),
          PRIMARY KEY (tenant_id, experiment_id),
          CHECK (ends_at >= starts_at),
          CHECK (NOT randomization_verified OR
                 (assignment_method = 'randomized' AND verification_ref IS NOT NULL
                  AND verified_by IS NOT NULL AND verified_at IS NOT NULL))
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_experiment_registry_seller_time ON governance.experiment_registry(tenant_id,seller_id,starts_at,ends_at)")
    op.execute("ALTER TABLE governance.experiment_registry ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE governance.experiment_registry FORCE ROW LEVEL SECURITY")
    op.execute("DROP POLICY IF EXISTS experiment_registry_tenant_isolation ON governance.experiment_registry")
    op.execute("""
        CREATE POLICY experiment_registry_tenant_isolation
        ON governance.experiment_registry
        USING (tenant_id = current_setting('app.tenant_id', true))
        WITH CHECK (tenant_id = current_setting('app.tenant_id', true))
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS analytics.experiment_observations (
          observation_id bigserial PRIMARY KEY,
          tenant_id varchar(128) NOT NULL,
          experiment_id varchar(128) NOT NULL,
          seller_id varchar(128) NOT NULL,
          randomization_unit_hash varchar(128) NOT NULL,
          assigned_group varchar(16) NOT NULL CHECK (assigned_group IN ('treatment','control')),
          outcome_metric_id varchar(128) NOT NULL,
          outcome_value numeric NOT NULL,
          observed_at timestamptz NOT NULL,
          source_batch_id varchar(128) NOT NULL,
          created_at timestamptz NOT NULL DEFAULT now(),
          UNIQUE (tenant_id, experiment_id, randomization_unit_hash),
          FOREIGN KEY (tenant_id, experiment_id)
            REFERENCES governance.experiment_registry(tenant_id, experiment_id)
            ON DELETE CASCADE
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_experiment_observations_scope ON analytics.experiment_observations(tenant_id,seller_id,experiment_id,observed_at)")
    op.execute("ALTER TABLE analytics.experiment_observations ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE analytics.experiment_observations FORCE ROW LEVEL SECURITY")
    op.execute("DROP POLICY IF EXISTS experiment_observations_tenant_isolation ON analytics.experiment_observations")
    op.execute("""
        CREATE POLICY experiment_observations_tenant_isolation
        ON analytics.experiment_observations
        USING (tenant_id = current_setting('app.tenant_id', true))
        WITH CHECK (tenant_id = current_setting('app.tenant_id', true))
    """)
    op.execute("COMMENT ON COLUMN analytics.experiment_observations.randomization_unit_hash IS 'Pseudonymous, stable per-experiment unit key; raw customer identifiers are prohibited.'")
    op.execute("COMMENT ON TABLE analytics.experiment_observations IS 'One outcome row per randomized unit. Causal analysis requires a verified experiment_registry row.'")
    op.execute("""
        DO $$ BEGIN
          IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='bizinsight_app') THEN
            GRANT SELECT, INSERT ON analytics.inventory_snapshots TO bizinsight_app;
            GRANT USAGE, SELECT ON SEQUENCE analytics.inventory_snapshots_snapshot_row_id_seq TO bizinsight_app;
            GRANT SELECT ON governance.experiment_registry, analytics.experiment_observations TO bizinsight_app;
          END IF;
        END $$
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS analytics.experiment_observations")
    op.execute("DROP TABLE IF EXISTS governance.experiment_registry")
    op.execute("DROP TABLE IF EXISTS analytics.inventory_snapshots")

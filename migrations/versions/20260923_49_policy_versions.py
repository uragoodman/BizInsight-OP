"""Versioned policy metadata used for authorization audit and invalidation."""

from alembic import op


revision = "20260923_49"
down_revision = "20260923_48"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS governance.policy_versions (
          tenant_id varchar(128) NOT NULL,
          user_id varchar(128) NOT NULL,
          permission_version varchar(128) NOT NULL,
          policy_set_version varchar(128) NOT NULL,
          catalog_digest varchar(128),
          authz_digest varchar(128) NOT NULL,
          policy_json jsonb NOT NULL DEFAULT '{}'::jsonb,
          status varchar(32) NOT NULL DEFAULT 'active',
          created_at timestamptz NOT NULL DEFAULT now(),
          PRIMARY KEY (tenant_id, user_id, permission_version, policy_set_version)
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_policy_versions_scope
        ON governance.policy_versions(tenant_id, user_id, created_at DESC)
    """)
    op.execute("ALTER TABLE governance.policy_versions ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE governance.policy_versions FORCE ROW LEVEL SECURITY")
    op.execute("DROP POLICY IF EXISTS policy_versions_isolation ON governance.policy_versions")
    op.execute("""
        CREATE POLICY policy_versions_isolation
        ON governance.policy_versions
        USING (tenant_id=current_setting('app.tenant_id',true)
           AND user_id=current_setting('app.user_id',true))
        WITH CHECK (tenant_id=current_setting('app.tenant_id',true)
           AND user_id=current_setting('app.user_id',true))
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS governance.policy_versions")

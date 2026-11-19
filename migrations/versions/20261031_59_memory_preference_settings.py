"""Add user-scoped personalization setting for WP4."""

from alembic import op


revision = "20261031_59"
down_revision = "20261030_58"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE governance.memory_preference_settings (
            tenant_id varchar(128) NOT NULL,
            user_id varchar(128) NOT NULL,
            personalization_enabled boolean NOT NULL DEFAULT true,
            updated_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (tenant_id, user_id)
        )
    """)
    op.execute("ALTER TABLE governance.memory_preference_settings ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE governance.memory_preference_settings FORCE ROW LEVEL SECURITY")
    op.execute("""
        CREATE POLICY memory_preference_settings_isolation
        ON governance.memory_preference_settings
        USING (tenant_id = current_setting('app.tenant_id', true)
               AND user_id = current_setting('app.user_id', true))
        WITH CHECK (tenant_id = current_setting('app.tenant_id', true)
                    AND user_id = current_setting('app.user_id', true))
    """)
    op.execute("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='bizinsight_app') THEN
                GRANT SELECT, INSERT, UPDATE ON governance.memory_preference_settings TO bizinsight_app;
            END IF;
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='bizinsight_memory_worker') THEN
                GRANT SELECT ON governance.memory_preference_settings TO bizinsight_memory_worker;
                GRANT DELETE ON governance.memory_registry,
                    governance.memory_preference_overlays TO bizinsight_memory_worker;
            END IF;
        END $$;
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS governance.memory_preference_settings")

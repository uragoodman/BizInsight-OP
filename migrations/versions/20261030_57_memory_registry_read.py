"""Minimal read model for the WP4 Memory Registry prerequisite.

WP3 needs an authoritative ``active`` row before a Mem0 hit may enter a Bundle.
This migration creates only the minimal Registry table and RLS boundary; it
does not add Outbox, Writer state transitions, Overlay, or preference APIs.
Those remain WP4 work.
"""

from alembic import op


revision = "20261030_57"
down_revision = "20261009_56"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE governance.memory_registry (
            registry_id varchar(64) PRIMARY KEY,
            tenant_id varchar(128) NOT NULL,
            user_id varchar(128) NOT NULL,
            seller_id varchar(64),
            memory_type varchar(32) NOT NULL
                CHECK (memory_type IN ('presentation','analysis_method','scope_default')),
            memory_key varchar(128) NOT NULL,
            mem0_memory_id varchar(128),
            status varchar(32) NOT NULL
                CHECK (status IN ('pending_create','pending_update','pending_delete',
                                  'active','deleted','failed','superseded')),
            version integer NOT NULL DEFAULT 1,
            operation_id varchar(128) NOT NULL,
            source_turn_id varchar(128),
            source_trace_id varchar(128),
            user_confirmed boolean NOT NULL DEFAULT false,
            last_error text,
            attempts integer NOT NULL DEFAULT 0,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            expires_at timestamptz,
            UNIQUE (operation_id)
        )
    """)
    op.execute("""
        CREATE INDEX idx_memory_registry_lookup
        ON governance.memory_registry(tenant_id, user_id, seller_id, memory_key, status)
    """)
    op.execute("""
        CREATE UNIQUE INDEX idx_memory_registry_key_version
        ON governance.memory_registry(tenant_id, user_id, COALESCE(seller_id, ''), memory_key, version)
    """)
    op.execute("""
        CREATE UNIQUE INDEX idx_memory_registry_active
        ON governance.memory_registry(tenant_id, user_id, COALESCE(seller_id, ''), memory_key)
        WHERE status='active'
    """)
    op.execute("""
        CREATE INDEX idx_memory_registry_mem0_id
        ON governance.memory_registry(mem0_memory_id)
    """)
    op.execute("ALTER TABLE governance.memory_registry ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE governance.memory_registry FORCE ROW LEVEL SECURITY")
    op.execute("""
        CREATE POLICY memory_registry_isolation ON governance.memory_registry
        USING (tenant_id = current_setting('app.tenant_id', true)
               AND user_id = current_setting('app.user_id', true))
        WITH CHECK (tenant_id = current_setting('app.tenant_id', true)
                    AND user_id = current_setting('app.user_id', true))
    """)
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='bizinsight_app') THEN
                GRANT USAGE ON SCHEMA governance TO bizinsight_app;
                GRANT SELECT ON governance.memory_registry TO bizinsight_app;
            END IF;
        END $$;
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS governance.memory_registry")

"""Add durable preference Outbox and same-session overlay for WP4.

The Outbox stores references and operation metadata only. Preference values
remain in the short-lived Overlay (24h contract); Mem0 remains the long-term
preference store and the Registry remains its authoritative index.
"""

from alembic import op


revision = "20261030_58"
down_revision = "20261030_57"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE governance.memory_registry ADD COLUMN chat_session_id varchar(128)")
    op.execute("ALTER TABLE governance.memory_registry ADD COLUMN value_hash varchar(64)")
    op.execute("ALTER TABLE governance.memory_registry ADD COLUMN supersedes_registry_id varchar(64)")
    op.execute("""
        CREATE UNIQUE INDEX idx_memory_registry_active_value
        ON governance.memory_registry(
            tenant_id, user_id, COALESCE(seller_id, ''), memory_key, value_hash
        ) WHERE status='active' AND value_hash IS NOT NULL
    """)
    op.execute("""
        ALTER TABLE governance.memory_registry
        ADD CONSTRAINT fk_memory_registry_supersedes
        FOREIGN KEY (supersedes_registry_id)
        REFERENCES governance.memory_registry(registry_id)
    """)
    op.execute("""
        CREATE TABLE governance.memory_preference_overlays (
            overlay_id varchar(64) PRIMARY KEY,
            tenant_id varchar(128) NOT NULL,
            user_id varchar(128) NOT NULL,
            seller_id varchar(64),
            chat_session_id varchar(128) NOT NULL,
            registry_id varchar(64) NOT NULL
                REFERENCES governance.memory_registry(registry_id),
            memory_key varchar(128) NOT NULL,
            memory_type varchar(32) NOT NULL
                CHECK (memory_type IN ('presentation','analysis_method','scope_default')),
            value jsonb NOT NULL,
            source_turn_id varchar(128),
            operation_id varchar(128) NOT NULL,
            version integer NOT NULL CHECK (version > 0),
            status varchar(16) NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','failed','cleared','expired','superseded')),
            error_code varchar(128),
            created_at timestamptz NOT NULL DEFAULT now(),
            expires_at timestamptz NOT NULL DEFAULT (now() + interval '24 hours'),
            cleared_at timestamptz,
            UNIQUE (tenant_id, user_id, chat_session_id, memory_key, version),
            UNIQUE (operation_id),
            FOREIGN KEY (tenant_id, user_id, chat_session_id)
                REFERENCES governance.chat_sessions(tenant_id, user_id, chat_session_id)
                ON DELETE CASCADE
        )
    """)
    op.execute("""
        CREATE INDEX idx_memory_preference_overlays_session
        ON governance.memory_preference_overlays
            (tenant_id, user_id, chat_session_id, memory_key, created_at DESC)
        WHERE status IN ('pending','failed')
    """)
    op.execute("""
        CREATE TABLE governance.memory_outbox (
            event_id varchar(64) PRIMARY KEY,
            operation_id varchar(128) NOT NULL UNIQUE,
            idempotency_key varchar(256) NOT NULL,
            tenant_id varchar(128) NOT NULL,
            user_id varchar(128) NOT NULL,
            seller_id varchar(64),
            event_type varchar(16) NOT NULL
                CHECK (event_type IN ('add','update','delete','reconcile')),
            registry_id varchar(64) NOT NULL
                REFERENCES governance.memory_registry(registry_id),
            overlay_id varchar(64)
                REFERENCES governance.memory_preference_overlays(overlay_id)
                ON DELETE SET NULL,
            target_memory_id varchar(128),
            target_operation_id varchar(128),
            payload_ciphertext bytea,
            payload_nonce bytea,
            payload_key_version varchar(32),
            registry_version integer NOT NULL CHECK (registry_version > 0),
            status varchar(16) NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','leased','succeeded','failed','dead_letter')),
            attempts integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
            max_attempts integer NOT NULL DEFAULT 5 CHECK (max_attempts > 0),
            next_attempt_at timestamptz NOT NULL DEFAULT now(),
            lease_token varchar(64),
            lease_expires_at timestamptz,
            last_error_code varchar(128),
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            completed_at timestamptz,
            expires_at timestamptz NOT NULL DEFAULT (now() + interval '30 days'),
            CHECK (
                (status = 'leased' AND lease_token IS NOT NULL AND lease_expires_at IS NOT NULL)
                OR (status <> 'leased' AND lease_token IS NULL AND lease_expires_at IS NULL)
            ),
            CHECK (
                (payload_ciphertext IS NULL AND payload_nonce IS NULL AND payload_key_version IS NULL)
                OR (payload_ciphertext IS NOT NULL AND payload_nonce IS NOT NULL
                    AND payload_key_version IS NOT NULL)
            )
        )
    """)
    op.execute("""
        CREATE UNIQUE INDEX idx_memory_outbox_idempotency
        ON governance.memory_outbox(
            tenant_id, user_id, COALESCE(seller_id, ''), idempotency_key
        )
    """)
    op.execute("""
        CREATE INDEX idx_memory_outbox_claim
        ON governance.memory_outbox(status, next_attempt_at, created_at)
        WHERE status IN ('pending','leased')
    """)
    op.execute("""
        CREATE TABLE governance.memory_preference_audit (
            audit_id varchar(64) PRIMARY KEY,
            tenant_id varchar(128) NOT NULL,
            user_id varchar(128) NOT NULL,
            seller_id varchar(64),
            registry_id varchar(64),
            operation_id varchar(128) NOT NULL,
            event_type varchar(48) NOT NULL,
            actor_user_id varchar(128) NOT NULL,
            status varchar(24) NOT NULL,
            reason_code varchar(128),
            created_at timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE INDEX idx_memory_preference_audit_lookup
        ON governance.memory_preference_audit(tenant_id, user_id, registry_id, created_at DESC)
    """)

    for table in (
        "memory_registry", "memory_preference_overlays", "memory_outbox",
        "memory_preference_audit",
    ):
        op.execute(f"ALTER TABLE governance.{table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE governance.{table} FORCE ROW LEVEL SECURITY")
        op.execute(f"DROP POLICY IF EXISTS {table}_isolation ON governance.{table}")
        op.execute(f"""
            CREATE POLICY {table}_isolation ON governance.{table}
            USING (tenant_id = current_setting('app.tenant_id', true)
                   AND user_id = current_setting('app.user_id', true))
            WITH CHECK (tenant_id = current_setting('app.tenant_id', true)
                        AND user_id = current_setting('app.user_id', true))
        """)
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='bizinsight_app') THEN
                GRANT SELECT, INSERT, UPDATE, DELETE ON governance.memory_registry TO bizinsight_app;
                GRANT SELECT, INSERT, UPDATE, DELETE ON governance.memory_preference_overlays TO bizinsight_app;
                GRANT SELECT, INSERT, UPDATE ON governance.memory_outbox TO bizinsight_app;
                GRANT SELECT, INSERT ON governance.memory_preference_audit TO bizinsight_app;
            END IF;
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='bizinsight_memory_worker') THEN
                GRANT SELECT, UPDATE ON governance.memory_registry TO bizinsight_memory_worker;
                GRANT SELECT, UPDATE ON governance.memory_preference_overlays TO bizinsight_memory_worker;
                GRANT SELECT, INSERT, UPDATE, DELETE ON governance.memory_outbox TO bizinsight_memory_worker;
                GRANT INSERT ON governance.memory_preference_audit TO bizinsight_memory_worker;
            END IF;
        END $$;
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS governance.memory_preference_audit")
    op.execute("DROP TABLE IF EXISTS governance.memory_outbox")
    op.execute("DROP TABLE IF EXISTS governance.memory_preference_overlays")
    op.execute("DROP INDEX IF EXISTS governance.idx_memory_registry_active_value")
    op.execute("ALTER TABLE governance.memory_registry DROP CONSTRAINT IF EXISTS fk_memory_registry_supersedes")
    op.execute("ALTER TABLE governance.memory_registry DROP COLUMN IF EXISTS supersedes_registry_id")
    op.execute("ALTER TABLE governance.memory_registry DROP COLUMN IF EXISTS value_hash")
    op.execute("ALTER TABLE governance.memory_registry DROP COLUMN IF EXISTS chat_session_id")

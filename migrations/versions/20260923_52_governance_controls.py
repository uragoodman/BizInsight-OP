"""Complete the v1 governance controls.

This migration adds database-native masking views, a tenant-scoped ABAC
policy registry, and a durable permission-event outbox.  The application
still performs the first authorization check; these objects are the database
and integration safety net when an upstream layer is bypassed.
"""

from alembic import op


revision = "20260923_52"
down_revision = "20260923_51"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS governance")
    op.execute("CREATE SCHEMA IF NOT EXISTS analytics")
    op.execute(
        """
        CREATE OR REPLACE FUNCTION governance.mask_identifier(value text)
        RETURNS text
        LANGUAGE sql
        IMMUTABLE
        AS $$
          SELECT CASE
            WHEN current_setting('app.allow_sensitive', true) IN ('true','1')
              THEN value
            WHEN value IS NULL THEN NULL
            ELSE 'masked_' || md5(value)
          END
        $$
        """
    )
    # Safe views preserve stable masked identifiers for aggregate joins while
    # preventing raw customer identifiers from being returned by default.
    op.execute(
        """
        CREATE OR REPLACE VIEW analytics.v_delivered_orders_safe
        WITH (security_barrier=true) AS
        SELECT order_id,
               governance.mask_identifier(customer_id) AS customer_id,
               governance.mask_identifier(customer_unique_id) AS customer_unique_id,
               customer_city, customer_state, order_status,
               order_purchase_timestamp, purchase_date, payment_value,
               item_count, item_gross_value, freight_value
        FROM analytics.fact_orders
        WHERE order_status = 'delivered'
        """
    )
    op.execute(
        """
        CREATE OR REPLACE VIEW analytics.v_delivered_order_items_safe
        WITH (security_barrier=true) AS
        SELECT order_id, order_item_id, seller_id, product_id,
               governance.mask_identifier(customer_unique_id) AS customer_unique_id,
               customer_state, purchase_date, order_status, price, freight_value,
               shipping_limit_date
        FROM analytics.fact_order_items
        WHERE order_status = 'delivered'
        """
    )
    op.execute(
        """
        DO $$ BEGIN
          IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='bizinsight_app') THEN
            GRANT SELECT ON analytics.v_delivered_orders_safe, analytics.v_delivered_order_items_safe TO bizinsight_app;
          END IF;
        END $$;
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS governance.abac_policy_rules (
          tenant_id varchar(128) NOT NULL,
          policy_id varchar(128) NOT NULL,
          policy_version bigint NOT NULL DEFAULT 1,
          status varchar(16) NOT NULL DEFAULT 'active'
            CHECK (status IN ('active','disabled')),
          rule_json jsonb NOT NULL,
          created_by varchar(128) NOT NULL,
          updated_by varchar(128) NOT NULL,
          created_at timestamptz NOT NULL DEFAULT now(),
          updated_at timestamptz NOT NULL DEFAULT now(),
          PRIMARY KEY (tenant_id, policy_id)
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_abac_policy_rules_active ON governance.abac_policy_rules(tenant_id,status)")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS governance.permission_event_outbox (
          event_id varchar(128) PRIMARY KEY,
          tenant_id varchar(128) NOT NULL,
          actor_user_id varchar(128) NOT NULL,
          subject_user_id varchar(128),
          event_type varchar(64) NOT NULL,
          idempotency_key varchar(256) NOT NULL UNIQUE,
          payload_json jsonb NOT NULL,
          status varchar(16) NOT NULL DEFAULT 'pending'
            CHECK (status IN ('pending','processing','delivered','failed')),
          attempts integer NOT NULL DEFAULT 0,
          created_at timestamptz NOT NULL DEFAULT now(),
          claimed_at timestamptz,
          delivered_at timestamptz,
          last_error text
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_permission_event_outbox_pending ON governance.permission_event_outbox(status,created_at)")
    for table in ("abac_policy_rules", "permission_event_outbox"):
        op.execute(f"ALTER TABLE governance.{table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE governance.{table} FORCE ROW LEVEL SECURITY")
        op.execute(f"DROP POLICY IF EXISTS {table}_isolation ON governance.{table}")
        op.execute(
            f"""
            CREATE POLICY {table}_isolation ON governance.{table}
            USING (tenant_id = current_setting('app.tenant_id', true))
            WITH CHECK (tenant_id = current_setting('app.tenant_id', true))
            """
        )
    op.execute(
        """
        DO $$ BEGIN
          IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='bizinsight_app') THEN
            GRANT SELECT, INSERT, UPDATE, DELETE ON governance.abac_policy_rules TO bizinsight_app;
            GRANT SELECT, INSERT, UPDATE ON governance.permission_event_outbox TO bizinsight_app;
          END IF;
          IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='bizinsight_code_worker') THEN
            GRANT SELECT, UPDATE ON governance.permission_event_outbox TO bizinsight_code_worker;
            GRANT SELECT, UPDATE ON governance.chat_turns, governance.conversation_executions TO bizinsight_code_worker;
            GRANT INSERT, SELECT ON governance.lifecycle_audit_events TO bizinsight_code_worker;
          END IF;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS governance.permission_event_outbox")
    op.execute("DROP TABLE IF EXISTS governance.abac_policy_rules")
    op.execute("DROP VIEW IF EXISTS analytics.v_delivered_order_items_safe")
    op.execute("DROP VIEW IF EXISTS analytics.v_delivered_orders_safe")
    op.execute("DROP FUNCTION IF EXISTS governance.mask_identifier(text)")

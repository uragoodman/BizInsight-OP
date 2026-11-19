"""Phase 7: Olist single-seller tenants and database-enforced isolation."""

from alembic import op

revision = "20260717_03"
down_revision = "20260717_02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS iam")
    op.execute("CREATE SCHEMA IF NOT EXISTS governance")
    op.execute("CREATE SCHEMA IF NOT EXISTS analytics")
    op.execute("""
        CREATE TABLE IF NOT EXISTS iam.tenants (
            tenant_id varchar(128) PRIMARY KEY,
            tenant_name varchar(256) NOT NULL,
            scope_type varchar(16) NOT NULL CHECK (scope_type IN ('platform','seller')),
            status varchar(16) NOT NULL DEFAULT 'active' CHECK (status IN ('active','suspended','offboarded')),
            created_at timestamptz NOT NULL DEFAULT now(), offboarded_at timestamptz
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS iam.users (
            user_id varchar(128) PRIMARY KEY, display_name varchar(256) NOT NULL,
            status varchar(16) NOT NULL DEFAULT 'active', created_at timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS iam.tenant_memberships (
            tenant_id varchar(128) NOT NULL REFERENCES iam.tenants(tenant_id),
            user_id varchar(128) NOT NULL REFERENCES iam.users(user_id),
            role varchar(16) NOT NULL CHECK (role IN ('owner','analyst','reviewer','admin')),
            created_at timestamptz NOT NULL DEFAULT now(), revoked_at timestamptz,
            PRIMARY KEY (tenant_id, user_id)
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS iam.tenant_seller_accounts (
            tenant_id varchar(128) PRIMARY KEY REFERENCES iam.tenants(tenant_id),
            seller_id varchar(64) NOT NULL UNIQUE, source_name varchar(64) NOT NULL DEFAULT 'olist',
            created_at timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS iam.tenant_data_sources (
            source_id bigserial PRIMARY KEY, tenant_id varchar(128) NOT NULL REFERENCES iam.tenants(tenant_id),
            source_type varchar(64) NOT NULL, external_account_id varchar(256) NOT NULL,
            status varchar(16) NOT NULL DEFAULT 'active', created_at timestamptz NOT NULL DEFAULT now(),
            UNIQUE (tenant_id, source_type, external_account_id)
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS iam.api_credentials (
            credential_id varchar(128) PRIMARY KEY, tenant_id varchar(128) NOT NULL REFERENCES iam.tenants(tenant_id),
            user_id varchar(128) NOT NULL REFERENCES iam.users(user_id), key_hash varchar(64) NOT NULL UNIQUE,
            expires_at timestamptz, revoked_at timestamptz, created_at timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS iam.tenant_notification_channels (
            tenant_id varchar(128) NOT NULL REFERENCES iam.tenants(tenant_id), channel varchar(16) NOT NULL CHECK(channel IN ('slack','email','teams')),
            secret_reference varchar(512) NOT NULL, enabled boolean NOT NULL DEFAULT true, created_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY(tenant_id, channel)
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS governance.audit_events (
            event_id bigserial PRIMARY KEY, tenant_id varchar(128), user_id varchar(128),
            event_type varchar(64) NOT NULL, outcome varchar(16) NOT NULL, detail text NOT NULL DEFAULT '',
            request_path varchar(512), correlation_id varchar(128), occurred_at timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS governance.tenant_knowledge_chunks (
            tenant_id varchar(128) NOT NULL REFERENCES iam.tenants(tenant_id), chunk_id varchar(256) NOT NULL,
            title varchar(512) NOT NULL, content text NOT NULL, source_ref varchar(512) NOT NULL,
            doc_type varchar(32) NOT NULL DEFAULT 'unknown', metric_name varchar(128), created_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (tenant_id, chunk_id)
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS governance.tenant_reports (
            tenant_id varchar(128) NOT NULL REFERENCES iam.tenants(tenant_id), report_id varchar(128) NOT NULL,
            version integer NOT NULL, status varchar(32) NOT NULL CHECK (status IN ('draft','approved','rejected','published')),
            content text NOT NULL, content_hash varchar(64) NOT NULL, report_json jsonb NOT NULL,
            actor varchar(128) NOT NULL, reason text NOT NULL DEFAULT '', created_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (tenant_id, report_id, version), UNIQUE (tenant_id, report_id, content_hash)
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS governance.tenant_outbox (
            tenant_id varchar(128) NOT NULL REFERENCES iam.tenants(tenant_id), event_id varchar(128) NOT NULL,
            idempotency_key varchar(256) NOT NULL, event_type varchar(64) NOT NULL, payload_json jsonb NOT NULL,
            status varchar(32) NOT NULL DEFAULT 'pending', attempts integer NOT NULL DEFAULT 0, occurred_at timestamptz NOT NULL DEFAULT now(),
            processed_at timestamptz, last_error text, PRIMARY KEY (tenant_id, event_id), UNIQUE (tenant_id, idempotency_key)
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS governance.tenant_delivery_audit (
            tenant_id varchar(128) NOT NULL REFERENCES iam.tenants(tenant_id), event_id varchar(128) NOT NULL,
            channel varchar(16) NOT NULL, delivery_key varchar(256) NOT NULL, status varchar(16) NOT NULL,
            detail text NOT NULL DEFAULT '', delivered_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY(tenant_id,event_id,channel), UNIQUE(tenant_id,delivery_key)
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS governance.tenant_delegated_tasks (
            tenant_id varchar(128) NOT NULL REFERENCES iam.tenants(tenant_id), task_id varchar(128) NOT NULL,
            idempotency_key varchar(256) NOT NULL, correlation_id varchar(128) NOT NULL, status varchar(16) NOT NULL CHECK(status IN ('pending','running','completed','cancelled','failed')),
            payload_json jsonb NOT NULL, expires_at timestamptz NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY(tenant_id,task_id), UNIQUE(tenant_id,idempotency_key)
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS governance.tenant_traces (
            tenant_id varchar(128) NOT NULL REFERENCES iam.tenants(tenant_id), trace_id varchar(128) NOT NULL,
            trace_json jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(tenant_id,trace_id)
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS analytics.tenant_metric_targets (
            tenant_id varchar(128) NOT NULL REFERENCES iam.tenants(tenant_id), metric_name varchar(128) NOT NULL,
            period_start date NOT NULL, period_end date NOT NULL, target_value numeric NOT NULL, version varchar(64) NOT NULL,
            effective_at date NOT NULL, created_by varchar(128) NOT NULL, PRIMARY KEY(tenant_id,metric_name,period_start,period_end,version)
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS analytics.fact_order_items (
            order_id varchar(64) NOT NULL, order_item_id integer NOT NULL, seller_id varchar(64) NOT NULL,
            customer_unique_id varchar(64), customer_state varchar(8), purchase_date date NOT NULL,
            price numeric NOT NULL, freight_value numeric NOT NULL, order_status varchar(32) NOT NULL,
            PRIMARY KEY (order_id, order_item_id)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_fact_order_items_seller_date ON analytics.fact_order_items(seller_id, purchase_date)")
    op.execute("""
        CREATE TABLE IF NOT EXISTS analytics.tenant_fact_order_items (
            tenant_id varchar(128) NOT NULL REFERENCES iam.tenants(tenant_id), order_id varchar(64) NOT NULL,
            order_item_id integer NOT NULL, seller_id varchar(64) NOT NULL, customer_unique_id varchar(64),
            customer_state varchar(8), purchase_date date NOT NULL, price numeric NOT NULL, freight_value numeric NOT NULL,
            order_status varchar(32) NOT NULL, PRIMARY KEY (tenant_id, order_id, order_item_id)
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS analytics.tenant_fact_orders (
            tenant_id varchar(128) NOT NULL REFERENCES iam.tenants(tenant_id), order_id varchar(64) NOT NULL,
            customer_unique_id varchar(64), purchase_date date NOT NULL, sales_gmv numeric NOT NULL,
            PRIMARY KEY (tenant_id, order_id)
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS analytics.tenant_daily_metrics (
            tenant_id varchar(128) NOT NULL REFERENCES iam.tenants(tenant_id), metric_date date NOT NULL,
            sales_gmv numeric NOT NULL, order_count integer NOT NULL, customer_count integer NOT NULL,
            avg_order_value numeric NOT NULL, repeat_purchase_rate numeric NOT NULL,
            PRIMARY KEY (tenant_id, metric_date)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_tenant_items_date ON analytics.tenant_fact_order_items(tenant_id, purchase_date)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_tenant_metrics_date ON analytics.tenant_daily_metrics(tenant_id, metric_date)")
    for table in ("tenant_fact_order_items", "tenant_fact_orders", "tenant_daily_metrics"):
        op.execute(f"ALTER TABLE analytics.{table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE analytics.{table} FORCE ROW LEVEL SECURITY")
        op.execute(f"DROP POLICY IF EXISTS {table}_isolation ON analytics.{table}")
        op.execute(f"""CREATE POLICY {table}_isolation ON analytics.{table}
            USING (current_setting('app.scope_type', true) = 'platform'
                   OR tenant_id = current_setting('app.tenant_id', true))
            WITH CHECK (tenant_id = current_setting('app.tenant_id', true))""")
    op.execute("ALTER TABLE governance.tenant_knowledge_chunks ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE governance.tenant_knowledge_chunks FORCE ROW LEVEL SECURITY")
    op.execute("""CREATE POLICY tenant_knowledge_isolation ON governance.tenant_knowledge_chunks
        USING (current_setting('app.scope_type', true) = 'platform' OR tenant_id = current_setting('app.tenant_id', true))
        WITH CHECK (tenant_id = current_setting('app.tenant_id', true))""")
    for schema, table in (("governance", "tenant_reports"), ("governance", "tenant_outbox"), ("governance", "tenant_delivery_audit"), ("governance", "tenant_delegated_tasks"), ("governance", "tenant_traces"), ("iam", "tenant_notification_channels"), ("analytics", "tenant_metric_targets")):
        op.execute(f"ALTER TABLE {schema}.{table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {schema}.{table} FORCE ROW LEVEL SECURITY")
        op.execute(f"""CREATE POLICY {table}_isolation ON {schema}.{table}
            USING (current_setting('app.scope_type', true) = 'platform' OR tenant_id = current_setting('app.tenant_id', true))
            WITH CHECK (tenant_id = current_setting('app.tenant_id', true))""")
    op.execute("""
        CREATE OR REPLACE FUNCTION analytics.refresh_tenant_marts() RETURNS void LANGUAGE sql AS $$
          TRUNCATE analytics.tenant_fact_order_items, analytics.tenant_fact_orders, analytics.tenant_daily_metrics;
          INSERT INTO analytics.tenant_fact_order_items
          SELECT a.tenant_id, f.order_id, f.order_item_id, f.seller_id, f.customer_unique_id, f.customer_state,
                 f.purchase_date, f.price, f.freight_value, f.order_status
          FROM analytics.fact_order_items f JOIN iam.tenant_seller_accounts a ON a.seller_id=f.seller_id
          JOIN iam.tenants t ON t.tenant_id=a.tenant_id AND t.status='active';
          INSERT INTO analytics.tenant_fact_orders
          SELECT tenant_id, order_id, max(customer_unique_id), max(purchase_date), sum(price + freight_value)
          FROM analytics.tenant_fact_order_items GROUP BY tenant_id, order_id;
          INSERT INTO analytics.tenant_daily_metrics
          WITH first_purchase AS (SELECT tenant_id, customer_unique_id, min(purchase_date) first_date
                                  FROM analytics.tenant_fact_orders WHERE customer_unique_id IS NOT NULL GROUP BY 1,2)
          SELECT f.tenant_id, f.purchase_date, sum(f.sales_gmv), count(*), count(DISTINCT f.customer_unique_id),
                 CASE WHEN count(*)=0 THEN 0 ELSE round(sum(f.sales_gmv)/count(*), 2) END,
                 CASE WHEN count(DISTINCT f.customer_unique_id)=0 THEN 0 ELSE round(100.0 * count(DISTINCT CASE WHEN p.first_date < f.purchase_date THEN f.customer_unique_id END) / count(DISTINCT f.customer_unique_id), 2) END
          FROM analytics.tenant_fact_orders f LEFT JOIN first_purchase p ON p.tenant_id=f.tenant_id AND p.customer_unique_id=f.customer_unique_id
          GROUP BY f.tenant_id, f.purchase_date;
        $$
    """)


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS analytics.refresh_tenant_marts()")
    op.execute("DROP TABLE IF EXISTS analytics.tenant_daily_metrics")
    op.execute("DROP TABLE IF EXISTS analytics.tenant_fact_orders")
    op.execute("DROP TABLE IF EXISTS analytics.tenant_fact_order_items")
    op.execute("DROP TABLE IF EXISTS analytics.fact_order_items")
    op.execute("DROP TABLE IF EXISTS governance.audit_events")
    op.execute("DROP TABLE IF EXISTS governance.tenant_knowledge_chunks")
    op.execute("DROP TABLE IF EXISTS governance.tenant_reports")
    op.execute("DROP TABLE IF EXISTS governance.tenant_outbox")
    op.execute("DROP TABLE IF EXISTS governance.tenant_delivery_audit")
    op.execute("DROP TABLE IF EXISTS governance.tenant_delegated_tasks")
    op.execute("DROP TABLE IF EXISTS governance.tenant_traces")
    op.execute("DROP TABLE IF EXISTS analytics.tenant_metric_targets")
    op.execute("DROP TABLE IF EXISTS iam.api_credentials")
    op.execute("DROP TABLE IF EXISTS iam.tenant_notification_channels")
    op.execute("DROP TABLE IF EXISTS iam.tenant_data_sources")
    op.execute("DROP TABLE IF EXISTS iam.tenant_seller_accounts")
    op.execute("DROP TABLE IF EXISTS iam.tenant_memberships")
    op.execute("DROP TABLE IF EXISTS iam.users")
    op.execute("DROP TABLE IF EXISTS iam.tenants")

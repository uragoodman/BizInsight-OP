"""Phase 6 semantic targets, durable reports, and event outbox."""

from alembic import op

revision = "20260717_02"
down_revision = "20260716_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS analytics")
    op.execute("CREATE SCHEMA IF NOT EXISTS governance")
    op.execute("""
        CREATE TABLE IF NOT EXISTS analytics.metric_targets (
            id bigserial PRIMARY KEY,
            metric_name varchar(128) NOT NULL,
            period_start date NOT NULL,
            period_end date NOT NULL,
            scope varchar(128) NOT NULL DEFAULT 'global',
            target_value numeric NOT NULL,
            version varchar(64) NOT NULL,
            effective_at date NOT NULL,
            created_by varchar(128) NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            UNIQUE(metric_name, period_start, period_end, scope, version)
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS governance.report_versions (
            report_id varchar(128) NOT NULL,
            version integer NOT NULL,
            status varchar(32) NOT NULL CHECK (status IN ('draft','approved','rejected','published')),
            content text NOT NULL,
            content_hash varchar(64) NOT NULL,
            report_json jsonb NOT NULL,
            actor varchar(128) NOT NULL,
            reason text NOT NULL DEFAULT '',
            created_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY(report_id, version),
            UNIQUE(report_id, content_hash)
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS governance.event_outbox (
            event_id varchar(128) PRIMARY KEY,
            idempotency_key varchar(256) NOT NULL UNIQUE,
            event_type varchar(64) NOT NULL,
            severity varchar(32) NOT NULL,
            payload_json jsonb NOT NULL,
            status varchar(32) NOT NULL DEFAULT 'pending',
            attempts integer NOT NULL DEFAULT 0,
            occurred_at timestamptz NOT NULL,
            processed_at timestamptz,
            claimed_at timestamptz,
            last_error text
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS governance.report_approvals (
            report_id varchar(128) NOT NULL,
            version integer NOT NULL,
            action varchar(32) NOT NULL CHECK (action IN ('approve','edit','reject')),
            actor varchar(128) NOT NULL,
            reason text NOT NULL DEFAULT '',
            original_hash varchar(64) NOT NULL,
            reviewed_hash varchar(64) NOT NULL,
            decided_at timestamptz NOT NULL,
            UNIQUE(report_id, version, action, reviewed_hash)
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS governance.report_approvals")
    op.execute("DROP TABLE IF EXISTS governance.event_outbox")
    op.execute("DROP TABLE IF EXISTS governance.report_versions")
    op.execute("DROP TABLE IF EXISTS analytics.metric_targets")

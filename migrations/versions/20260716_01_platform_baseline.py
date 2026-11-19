"""Phase 5 platform governance baseline."""

from alembic import op

revision = "20260716_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS governance")
    op.execute("""
        CREATE TABLE IF NOT EXISTS governance.schema_release (
            version varchar(64) PRIMARY KEY,
            applied_at timestamptz NOT NULL DEFAULT now(),
            app_version varchar(64) NOT NULL,
            rag_release_version varchar(128) NOT NULL
        )
    """)
    op.execute("""
        INSERT INTO governance.schema_release(version, app_version, rag_release_version)
        VALUES ('20260716_01', '0.1.0', 'v1') ON CONFLICT (version) DO NOTHING
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS governance.schema_release")

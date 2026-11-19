"""Durable document reconciliation queue and indexed source state."""

from alembic import op


revision = "20261009_56"
down_revision = "20261008_55"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE governance.rag_document_sources (
          collection_name varchar(160) NOT NULL,
          source_id text NOT NULL,
          source_revision varchar(128) NOT NULL,
          content_hash char(64),
          indexed_revision varchar(128),
          indexed_hash char(64),
          status varchar(16) NOT NULL,
          last_synced_at timestamptz,
          last_error text,
          updated_at timestamptz NOT NULL DEFAULT now(),
          PRIMARY KEY (collection_name, source_id),
          CHECK (status IN ('indexed','failed','deleted'))
        )
    """)
    op.execute("""
        CREATE TABLE governance.rag_document_sync_jobs (
          job_id char(64) PRIMARY KEY,
          collection_name varchar(160) NOT NULL,
          corpus_hash char(64) NOT NULL,
          source_revision varchar(128) NOT NULL,
          status varchar(16) NOT NULL DEFAULT 'queued',
          attempts integer NOT NULL DEFAULT 0,
          max_attempts integer NOT NULL DEFAULT 8,
          available_at timestamptz NOT NULL DEFAULT now(),
          lease_owner varchar(128),
          lease_expires_at timestamptz,
          last_error text,
          created_at timestamptz NOT NULL DEFAULT now(),
          updated_at timestamptz NOT NULL DEFAULT now(),
          completed_at timestamptz,
          UNIQUE (collection_name, corpus_hash),
          CHECK (status IN ('queued','running','retry','completed','failed','superseded')),
          CHECK (attempts >= 0 AND max_attempts > 0)
        )
    """)
    op.execute("""
        CREATE INDEX idx_rag_document_sync_jobs_claim
        ON governance.rag_document_sync_jobs(status, available_at, created_at)
    """)
    op.execute("""
        DO $$ BEGIN
          IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='bizinsight_app') THEN
            GRANT USAGE ON SCHEMA governance TO bizinsight_app;
            GRANT SELECT, INSERT, UPDATE, DELETE ON
              governance.rag_document_sources, governance.rag_document_sync_jobs
              TO bizinsight_app;
          END IF;
          IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='bizinsight_code_worker') THEN
            GRANT USAGE ON SCHEMA governance TO bizinsight_code_worker;
            GRANT SELECT, INSERT, UPDATE, DELETE ON
              governance.rag_document_sources, governance.rag_document_sync_jobs
              TO bizinsight_code_worker;
          END IF;
        END $$;
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS governance.rag_document_sync_jobs")
    op.execute("DROP TABLE IF EXISTS governance.rag_document_sources")

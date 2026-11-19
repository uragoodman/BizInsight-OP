"""Persist fail-closed semantic metric mappings and confirmations."""

from alembic import op


revision = "20260921_44"
down_revision = "20260921_43"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS governance.semantic_metric_mappings (
          mapping_id varchar(128) PRIMARY KEY,
          resolution_id varchar(128) NOT NULL,
          mention_id varchar(128) NOT NULL,
          candidate_no integer,
          metric_id varchar(128),
          tenant_id varchar(128) NOT NULL,
          user_id varchar(128) NOT NULL,
          session_id varchar(128),
          turn_id varchar(128),
          scope_fingerprint varchar(128) NOT NULL,
          catalog_digest varchar(128) NOT NULL,
          relation varchar(32) NOT NULL,
          reason text NOT NULL DEFAULT '',
          evidence text NOT NULL DEFAULT '',
          status varchar(32) NOT NULL DEFAULT 'pending',
          user_decision varchar(32),
          trace_id varchar(128),
          execution_request_id varchar(128),
          created_at timestamptz NOT NULL DEFAULT now(),
          confirmed_at timestamptz,
          expires_at timestamptz NOT NULL,
          CHECK (relation IN ('equivalent', 'related_but_different', 'no_match')),
          CHECK (status IN ('pending', 'confirmed', 'cancelled', 'expired')),
          CHECK (user_decision IS NULL OR user_decision IN ('skipped_by_user')),
          CHECK ((relation = 'no_match' AND candidate_no IS NULL AND metric_id IS NULL)
                 OR (relation <> 'no_match' AND candidate_no IS NOT NULL AND metric_id IS NOT NULL))
        )
        """
    )
    op.execute("ALTER TABLE governance.semantic_metric_mappings ADD COLUMN IF NOT EXISTS user_decision varchar(32)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_semantic_metric_mapping_decision ON governance.semantic_metric_mappings(user_decision, status)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_semantic_metric_mapping_owner ON governance.semantic_metric_mappings(tenant_id, user_id, session_id, created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_semantic_metric_mapping_status ON governance.semantic_metric_mappings(status, expires_at)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS governance.semantic_metric_mappings")

"""Async RAG release-contract validation jobs."""
from alembic import op

revision = "20260825_37"
down_revision = "20260824_36"
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.execute("""CREATE TABLE governance.rag_contract_validation_jobs (
      job_id varchar(80) PRIMARY KEY, release_id varchar(80) NOT NULL REFERENCES governance.rag_releases(release_id),
      evaluation_run_id varchar(80) NOT NULL REFERENCES governance.evaluation_runs(run_id),
      status varchar(16) NOT NULL DEFAULT 'queued', attempts integer NOT NULL DEFAULT 0,
      lease_token varchar(80), lease_expires_at timestamptz, error_code varchar(64), created_at timestamptz NOT NULL DEFAULT now(), completed_at timestamptz,
      CHECK(status IN ('queued','running','completed','failed')), UNIQUE(evaluation_run_id)
    )""")

def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS governance.rag_contract_validation_jobs")

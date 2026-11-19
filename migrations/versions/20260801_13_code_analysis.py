"""Add governed uploaded-file Code Agent analysis jobs."""

from alembic import op

revision = "20260801_13"
down_revision = "20260730_12"
branch_labels = None
depends_on = None


def _rls(table: str) -> None:
    op.execute(f"ALTER TABLE governance.{table} ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE governance.{table} FORCE ROW LEVEL SECURITY")
    op.execute(f"""CREATE POLICY {table}_isolation ON governance.{table}
      USING (tenant_id=current_setting('app.tenant_id',true) AND user_id=current_setting('app.user_id',true))
      WITH CHECK (tenant_id=current_setting('app.tenant_id',true) AND user_id=current_setting('app.user_id',true))""")


def upgrade() -> None:
    op.execute("""CREATE TABLE governance.code_analysis_files (
      file_id varchar(64) PRIMARY KEY, tenant_id varchar(128) NOT NULL, user_id varchar(128) NOT NULL,
      original_name varchar(255) NOT NULL, file_type varchar(16) NOT NULL, content_type varchar(255) NOT NULL,
      object_key text NOT NULL, sha256 varchar(64) NOT NULL, size_bytes bigint NOT NULL,
      contract_data jsonb NOT NULL, expires_at timestamptz NOT NULL, created_at timestamptz NOT NULL DEFAULT now()
    )""")
    op.execute("""CREATE TABLE governance.code_analysis_jobs (
      job_id varchar(64) PRIMARY KEY, tenant_id varchar(128) NOT NULL, user_id varchar(128) NOT NULL,
      chat_session_id varchar(128), file_id varchar(64) NOT NULL REFERENCES governance.code_analysis_files(file_id) ON DELETE CASCADE,
      question text NOT NULL, plan_data jsonb NOT NULL, plan_hash varchar(64) NOT NULL,
      status varchar(32) NOT NULL, cancel_requested boolean NOT NULL DEFAULT false, attempts integer NOT NULL DEFAULT 0,
      lease_token varchar(64), lease_expires_at timestamptz, expires_at timestamptz NOT NULL,
      confirmed_at timestamptz, started_at timestamptz, completed_at timestamptz, result_data jsonb,
      error_code varchar(80), created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now()
    )""")
    op.execute("""CREATE TABLE governance.code_analysis_artifacts (
      artifact_id varchar(64) PRIMARY KEY, tenant_id varchar(128) NOT NULL, user_id varchar(128) NOT NULL,
      job_id varchar(64) NOT NULL REFERENCES governance.code_analysis_jobs(job_id) ON DELETE CASCADE,
      kind varchar(32) NOT NULL, object_key text NOT NULL, sha256 varchar(64) NOT NULL, size_bytes bigint NOT NULL,
      metadata jsonb NOT NULL DEFAULT '{}'::jsonb, created_at timestamptz NOT NULL DEFAULT now()
    )""")
    op.execute("""CREATE TABLE governance.code_analysis_audits (
      audit_id varchar(64) PRIMARY KEY, tenant_id varchar(128) NOT NULL, user_id varchar(128) NOT NULL,
      job_id varchar(64) NOT NULL REFERENCES governance.code_analysis_jobs(job_id) ON DELETE CASCADE,
      status varchar(32) NOT NULL, provider varchar(64) NOT NULL, plan_hash varchar(64) NOT NULL,
      data_contract_hash varchar(64) NOT NULL, candidate_code_hash varchar(64), runner_image_digest varchar(255) NOT NULL,
      result_hash varchar(64), trace_id varchar(128), duration_ms integer NOT NULL, error_code varchar(80),
      created_at timestamptz NOT NULL DEFAULT now()
    )""")
    op.execute("CREATE INDEX idx_code_analysis_files_expiry ON governance.code_analysis_files(expires_at)")
    op.execute("CREATE INDEX idx_code_analysis_jobs_owner ON governance.code_analysis_jobs(tenant_id,user_id,updated_at DESC)")
    op.execute("CREATE INDEX idx_code_analysis_jobs_lease ON governance.code_analysis_jobs(status,lease_expires_at,created_at)")
    op.execute("CREATE INDEX idx_code_analysis_artifacts_job ON governance.code_analysis_artifacts(job_id,created_at)")
    op.execute("CREATE INDEX idx_code_analysis_audits_job ON governance.code_analysis_audits(job_id,created_at DESC)")
    for table in ("code_analysis_files", "code_analysis_jobs", "code_analysis_artifacts", "code_analysis_audits"):
        _rls(table)


def downgrade() -> None:
    for table in ("code_analysis_audits", "code_analysis_artifacts", "code_analysis_jobs", "code_analysis_files"):
        op.execute(f"DROP TABLE IF EXISTS governance.{table}")

"""Separate raw-file retention from results and snapshot execution quotas."""

from alembic import op


revision = "20260805_17"
down_revision = "20260804_16"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """ALTER TABLE governance.code_analysis_files
           ADD COLUMN purged_at timestamptz"""
    )
    op.execute(
        """ALTER TABLE governance.code_analysis_jobs
           ADD COLUMN resource_limits jsonb NOT NULL DEFAULT '{}'::jsonb,
           ALTER COLUMN file_id DROP NOT NULL"""
    )
    op.execute(
        """ALTER TABLE governance.code_analysis_jobs
           DROP CONSTRAINT code_analysis_jobs_file_id_fkey,
           ADD CONSTRAINT code_analysis_jobs_file_id_fkey
             FOREIGN KEY (file_id)
             REFERENCES governance.code_analysis_files(file_id)
             ON DELETE SET NULL"""
    )
    # Execution audits are retained when a user deletes a task/chat.  They no
    # longer expose task contents, but keep hashes, image identity and Trace.
    op.execute(
        """ALTER TABLE governance.code_analysis_audits
           ALTER COLUMN job_id DROP NOT NULL,
           DROP CONSTRAINT code_analysis_audits_job_id_fkey,
           ADD CONSTRAINT code_analysis_audits_job_id_fkey
             FOREIGN KEY (job_id)
             REFERENCES governance.code_analysis_jobs(job_id)
             ON DELETE SET NULL"""
    )
    op.execute(
        """CREATE INDEX idx_code_analysis_files_pending_purge
           ON governance.code_analysis_files(expires_at)
           WHERE purged_at IS NULL"""
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS governance.idx_code_analysis_files_pending_purge")
    op.execute(
        """ALTER TABLE governance.code_analysis_audits
           DROP CONSTRAINT code_analysis_audits_job_id_fkey,
           ADD CONSTRAINT code_analysis_audits_job_id_fkey
             FOREIGN KEY (job_id)
             REFERENCES governance.code_analysis_jobs(job_id)
             ON DELETE CASCADE"""
    )
    op.execute(
        """ALTER TABLE governance.code_analysis_jobs
           DROP CONSTRAINT code_analysis_jobs_file_id_fkey,
           ADD CONSTRAINT code_analysis_jobs_file_id_fkey
             FOREIGN KEY (file_id)
             REFERENCES governance.code_analysis_files(file_id)
             ON DELETE CASCADE,
           DROP COLUMN resource_limits"""
    )
    op.execute(
        "ALTER TABLE governance.code_analysis_files DROP COLUMN purged_at"
    )

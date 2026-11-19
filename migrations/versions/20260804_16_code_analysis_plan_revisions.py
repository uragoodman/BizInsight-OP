"""Add immutable revision lineage to code-analysis plans."""

from alembic import op

revision = "20260804_16"
down_revision = "20260803_15"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""ALTER TABLE governance.code_analysis_jobs
      ADD COLUMN parent_job_id varchar(64)
        REFERENCES governance.code_analysis_jobs(job_id) ON DELETE SET NULL,
      ADD COLUMN source_turn_id varchar(128)
        REFERENCES governance.chat_turns(turn_id) ON DELETE SET NULL,
      ADD COLUMN revision integer NOT NULL DEFAULT 1""")
    op.execute("""ALTER TABLE governance.code_analysis_jobs
      ADD CONSTRAINT code_analysis_jobs_status_check CHECK (
        status IN (
          'awaiting_confirmation','queued','generating','validating','running',
          'completed','failed','cancelled','expired','superseded'
        )
      ),
      ADD CONSTRAINT code_analysis_jobs_revision_check CHECK (revision >= 1)""")
    op.execute(
        "CREATE INDEX idx_code_analysis_jobs_parent "
        "ON governance.code_analysis_jobs(parent_job_id,revision)"
    )
    op.execute(
        "CREATE INDEX idx_code_analysis_jobs_source_turn "
        "ON governance.code_analysis_jobs(source_turn_id)"
    )
    op.execute("GRANT SELECT ON governance.chat_turns TO bizinsight_code_api")


def downgrade() -> None:
    op.execute("REVOKE SELECT ON governance.chat_turns FROM bizinsight_code_api")
    op.execute("DROP INDEX IF EXISTS governance.idx_code_analysis_jobs_source_turn")
    op.execute("DROP INDEX IF EXISTS governance.idx_code_analysis_jobs_parent")
    op.execute("""ALTER TABLE governance.code_analysis_jobs
      DROP CONSTRAINT code_analysis_jobs_revision_check,
      DROP CONSTRAINT code_analysis_jobs_status_check,
      DROP COLUMN revision,
      DROP COLUMN source_turn_id,
      DROP COLUMN parent_job_id""")

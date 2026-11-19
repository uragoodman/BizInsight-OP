"""Unified evaluation evidence envelopes without duplicating domain metrics."""
from alembic import op

revision = "20260824_36"
down_revision = "20260823_35"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""CREATE TABLE governance.evaluation_runs (
      run_id varchar(80) PRIMARY KEY, subject_type varchar(32) NOT NULL, subject_id varchar(160) NOT NULL,
      baseline_ref jsonb NOT NULL DEFAULT '{}'::jsonb, candidate_ref jsonb NOT NULL DEFAULT '{}'::jsonb,
      dataset_ref varchar(512), dataset_hash char(64), commit_sha varchar(64) NOT NULL,
      config_snapshot jsonb NOT NULL, runner_version varchar(64) NOT NULL,
      status varchar(16) NOT NULL, gate_status varchar(20) NOT NULL, gate_reasons jsonb NOT NULL DEFAULT '[]'::jsonb,
      created_by varchar(128) NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), completed_at timestamptz,
      CHECK(subject_type IN ('rag_release','memory_policy','context_packer','data_versioning','benchmark')),
      CHECK(status IN ('running','passed','warned','blocked','unavailable','failed')),
      CHECK(gate_status IN ('pass','warn','block','not_applicable'))
    )""")
    op.execute("CREATE INDEX idx_evaluation_runs_subject ON governance.evaluation_runs(subject_type,subject_id,created_at DESC)")
    op.execute("""CREATE TABLE governance.evaluation_run_evidence (
      evidence_id varchar(80) PRIMARY KEY, run_id varchar(80) NOT NULL REFERENCES governance.evaluation_runs(run_id) ON DELETE CASCADE,
      source_type varchar(32) NOT NULL, source_id varchar(160) NOT NULL, role varchar(16) NOT NULL,
      summary jsonb NOT NULL DEFAULT '{}'::jsonb, content_hash char(64), created_at timestamptz NOT NULL DEFAULT now(),
      CHECK(source_type IN ('rag_release_evaluation','rag_shadow_summary','data_replay_audit','memory_decision','benchmark_file','golden_artifact')),
      CHECK(role IN ('baseline','candidate','gate','online','benchmark')),
      UNIQUE(run_id,source_type,source_id,role)
    )""")
    op.execute("""CREATE TABLE governance.evaluation_comparisons (
      comparison_id varchar(80) PRIMARY KEY, run_id varchar(80) NOT NULL REFERENCES governance.evaluation_runs(run_id) ON DELETE CASCADE,
      metric_deltas jsonb NOT NULL, thresholds jsonb NOT NULL, verdict varchar(16) NOT NULL, reasons jsonb NOT NULL DEFAULT '[]'::jsonb,
      created_at timestamptz NOT NULL DEFAULT now(), CHECK(verdict IN ('pass','warn','block')),
      UNIQUE(run_id)
    )""")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS governance.evaluation_comparisons")
    op.execute("DROP TABLE IF EXISTS governance.evaluation_run_evidence")
    op.execute("DROP TABLE IF EXISTS governance.evaluation_runs")

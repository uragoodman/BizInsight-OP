"""Store Agent evaluation runs and case-level evidence in one envelope."""

from alembic import op


revision = "20260915_41"
down_revision = "20260901_40"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The original envelope was designed before the Agent-vs-Workflow gate
    # existed.  Keep all previous subjects and add an explicit Agent subject.
    op.execute("ALTER TABLE governance.evaluation_runs DROP CONSTRAINT IF EXISTS evaluation_runs_subject_type_check")
    op.execute("""
      ALTER TABLE governance.evaluation_runs ADD CONSTRAINT evaluation_runs_subject_type_check
      CHECK(subject_type IN ('rag_release','rag_retrieval','memory_policy','context_packer',
                             'data_versioning','benchmark','agent_evaluation'))
    """)
    op.execute("""
      CREATE TABLE IF NOT EXISTS governance.evaluation_run_cases (
        run_id varchar(80) NOT NULL REFERENCES governance.evaluation_runs(run_id) ON DELETE CASCADE,
        arm varchar(32) NOT NULL,
        case_id varchar(160) NOT NULL,
        category varchar(64) NOT NULL,
        provider_called boolean NOT NULL DEFAULT false,
        trace_id varchar(128),
        result jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY(run_id, arm, case_id),
        CHECK(arm IN ('agent','fallback','agent_fallback','workflow','baseline'))
      )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_run_cases_case ON governance.evaluation_run_cases(case_id, created_at DESC)")
    op.execute("ALTER TABLE governance.tenant_traces ADD COLUMN IF NOT EXISTS evaluation_run_id varchar(80)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_tenant_traces_evaluation_run ON governance.tenant_traces(evaluation_run_id)")
    op.execute("ALTER TABLE governance.llm_usage_events ADD COLUMN IF NOT EXISTS evaluation_run_id varchar(80)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_llm_usage_events_evaluation_run ON governance.llm_usage_events(evaluation_run_id)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS governance.idx_llm_usage_events_evaluation_run")
    op.execute("ALTER TABLE governance.llm_usage_events DROP COLUMN IF EXISTS evaluation_run_id")
    op.execute("DROP INDEX IF EXISTS governance.idx_tenant_traces_evaluation_run")
    op.execute("ALTER TABLE governance.tenant_traces DROP COLUMN IF EXISTS evaluation_run_id")
    op.execute("DROP TABLE IF EXISTS governance.evaluation_run_cases")
    op.execute("ALTER TABLE governance.evaluation_runs DROP CONSTRAINT IF EXISTS evaluation_runs_subject_type_check")
    op.execute("""
      ALTER TABLE governance.evaluation_runs ADD CONSTRAINT evaluation_runs_subject_type_check
      CHECK(subject_type IN ('rag_release','rag_retrieval','memory_policy','context_packer','data_versioning','benchmark'))
    """)

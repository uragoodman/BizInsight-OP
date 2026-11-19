"""RAG release control-plane, Shadow evidence and rollout audit."""
from alembic import op

revision = "20260823_35"
down_revision = "20260822_34"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""CREATE TABLE governance.rag_releases (
      release_id varchar(80) PRIMARY KEY, parent_release_id varchar(80), status varchar(20) NOT NULL,
      spec jsonb NOT NULL, spec_hash char(64) NOT NULL UNIQUE, corpus_hash char(64) NOT NULL,
      created_by varchar(128) NOT NULL, approved_by varchar(128), created_at timestamptz NOT NULL DEFAULT now(), approved_at timestamptz,
      CHECK (status IN ('draft','building','evaluated','shadowing','canary_5','canary_25','canary_100','promotable','published','paused','rolled_back','expired','failed'))
    )""")
    op.execute("""CREATE UNIQUE INDEX uq_rag_one_published ON governance.rag_releases ((status)) WHERE status='published'""")
    op.execute("""CREATE TABLE governance.rag_release_artifacts (
      artifact_id varchar(80) PRIMARY KEY, release_id varchar(80) NOT NULL REFERENCES governance.rag_releases(release_id),
      backend varchar(32) NOT NULL, local_snapshot_ref text, qdrant_collection varchar(160), artifact_hash char(64) NOT NULL,
      chunk_count integer NOT NULL DEFAULT 0, vector_count integer NOT NULL DEFAULT 0, sparse_vector_count integer NOT NULL DEFAULT 0,
      build_status varchar(16) NOT NULL, validation jsonb NOT NULL DEFAULT '{}'::jsonb, created_at timestamptz NOT NULL DEFAULT now(),
      CHECK(build_status IN ('building','ready','failed','expired'))
    )""")
    op.execute("""CREATE TABLE governance.rag_release_evaluations (
      evaluation_id varchar(80) PRIMARY KEY, release_id varchar(80) NOT NULL REFERENCES governance.rag_releases(release_id),
      baseline_release_id varchar(80), kind varchar(24) NOT NULL, dataset_version varchar(128) NOT NULL,
      metrics jsonb NOT NULL, thresholds jsonb NOT NULL, status varchar(16) NOT NULL, evaluator jsonb NOT NULL DEFAULT '{}'::jsonb,
      failure_reason text, created_at timestamptz NOT NULL DEFAULT now(),
      CHECK(kind IN ('deterministic','ragas')), CHECK(status IN ('passed','failed','unavailable','running'))
    )""")
    op.execute("""CREATE TABLE governance.rag_rollout_policies (
      policy_id varchar(80) PRIMARY KEY, release_id varchar(80) NOT NULL REFERENCES governance.rag_releases(release_id),
      stage varchar(20) NOT NULL, shadow_sample_rate numeric NOT NULL DEFAULT 0, canary_percent integer NOT NULL DEFAULT 0,
      tenant_allowlist jsonb NOT NULL DEFAULT '[]'::jsonb, min_samples integer NOT NULL DEFAULT 100,
      thresholds jsonb NOT NULL, auto_rollback boolean NOT NULL DEFAULT true, active boolean NOT NULL DEFAULT true,
      created_by varchar(128) NOT NULL, updated_at timestamptz NOT NULL DEFAULT now(),
      CHECK(stage IN ('off','shadowing','canary_5','canary_25','canary_100')),
      CHECK(shadow_sample_rate >= 0 AND shadow_sample_rate <= 1), CHECK(canary_percent IN (0,5,25,100))
    )""")
    op.execute("CREATE UNIQUE INDEX uq_rag_active_rollout ON governance.rag_rollout_policies(release_id) WHERE active")
    op.execute("""CREATE TABLE governance.rag_shadow_jobs (
      job_id varchar(80) PRIMARY KEY, tenant_id varchar(128) NOT NULL, user_id varchar(128) NOT NULL,
      stable_release_id varchar(80) NOT NULL, candidate_release_id varchar(80) NOT NULL, mode varchar(16) NOT NULL,
      query_payload text NOT NULL, query_hash char(64) NOT NULL, request_context jsonb NOT NULL DEFAULT '{}'::jsonb,
      status varchar(16) NOT NULL DEFAULT 'queued', attempts integer NOT NULL DEFAULT 0, lease_token varchar(80), lease_expires_at timestamptz,
      expires_at timestamptz NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), completed_at timestamptz,
      CHECK(mode IN ('shadow','canary_compare')), CHECK(status IN ('queued','running','completed','failed','expired'))
    )""")
    op.execute("""CREATE TABLE governance.rag_shadow_comparisons (
      comparison_id varchar(80) PRIMARY KEY, job_id varchar(80) UNIQUE REFERENCES governance.rag_shadow_jobs(job_id),
      tenant_id varchar(128) NOT NULL, user_id varchar(128) NOT NULL, stable_release_id varchar(80) NOT NULL,
      candidate_release_id varchar(80) NOT NULL, mode varchar(16) NOT NULL, stable_result jsonb NOT NULL,
      candidate_result jsonb NOT NULL, comparison jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now()
    )""")
    op.execute("""CREATE TABLE governance.rag_release_decisions (
      decision_id varchar(80) PRIMARY KEY, release_id varchar(80) NOT NULL REFERENCES governance.rag_releases(release_id),
      action varchar(24) NOT NULL, actor varchar(128) NOT NULL, reason text NOT NULL DEFAULT '', evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
      created_at timestamptz NOT NULL DEFAULT now(), CHECK(action IN ('created','built','evaluated','shadow_started','canary_started','paused','promoted','rolled_back','expired'))
    )""")
    op.execute("""CREATE TABLE governance.rag_release_incidents (
      incident_id varchar(80) PRIMARY KEY, release_id varchar(80) NOT NULL REFERENCES governance.rag_releases(release_id),
      tenant_id varchar(128), severity varchar(12) NOT NULL, code varchar(64) NOT NULL, details jsonb NOT NULL DEFAULT '{}'::jsonb,
      resolved_at timestamptz, created_at timestamptz NOT NULL DEFAULT now(), CHECK(severity IN ('warning','critical'))
    )""")
    op.execute("""CREATE TABLE governance.rag_retrieval_feedback (
      feedback_id varchar(80) PRIMARY KEY, tenant_id varchar(128) NOT NULL, user_id varchar(128) NOT NULL,
      trace_id varchar(128) NOT NULL, release_id varchar(80) NOT NULL, rating smallint NOT NULL, category varchar(64), created_at timestamptz NOT NULL DEFAULT now(),
      CHECK(rating IN (-1,1)), UNIQUE(tenant_id,user_id,trace_id)
    )""")
    for table in ('rag_shadow_jobs','rag_shadow_comparisons','rag_retrieval_feedback'):
        op.execute(f"ALTER TABLE governance.{table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE governance.{table} FORCE ROW LEVEL SECURITY")
        op.execute(f"CREATE POLICY {table}_isolation ON governance.{table} USING (tenant_id=current_setting('app.tenant_id',true) AND user_id IS NOT DISTINCT FROM current_setting('app.user_id',true)) WITH CHECK (tenant_id=current_setting('app.tenant_id',true) AND user_id=current_setting('app.user_id',true))")


def downgrade() -> None:
    for table in ('rag_retrieval_feedback','rag_release_incidents','rag_release_decisions','rag_shadow_comparisons','rag_shadow_jobs','rag_rollout_policies','rag_release_evaluations','rag_release_artifacts','rag_releases'):
        op.execute(f"DROP TABLE IF EXISTS governance.{table}")

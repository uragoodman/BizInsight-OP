"""Report / insight data-version manifest lineage (GMV pilot)."""
from alembic import op

revision = "20260822_34"
down_revision = "20260821_33"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Reports and AI insights record the sealed data-version manifest their facts
    # came from.  Legacy unversioned artifacts keep NULL and replay honestly
    # reports `unavailable` instead of pretending latest data matches.
    op.execute("ALTER TABLE governance.tenant_reports ADD COLUMN IF NOT EXISTS data_version_manifest_id varchar(80)")
    op.execute("ALTER TABLE governance.tenant_reports ADD COLUMN IF NOT EXISTS data_version_manifest_hash char(64)")
    op.execute("ALTER TABLE governance.tenant_report_insights ADD COLUMN IF NOT EXISTS data_version_manifest_id varchar(80)")
    op.execute("ALTER TABLE governance.tenant_report_insights ADD COLUMN IF NOT EXISTS data_version_manifest_hash char(64)")
    op.execute("ALTER TABLE governance.data_replay_audits DROP CONSTRAINT IF EXISTS data_replay_audits_artifact_type_check")
    op.execute("""ALTER TABLE governance.data_replay_audits ADD CONSTRAINT data_replay_audits_artifact_type_check
        CHECK (artifact_type IN ('analysis_memory','complex_execution','report','insight'))""")


def downgrade() -> None:
    op.execute("ALTER TABLE governance.data_replay_audits DROP CONSTRAINT IF EXISTS data_replay_audits_artifact_type_check")
    op.execute("""ALTER TABLE governance.data_replay_audits ADD CONSTRAINT data_replay_audits_artifact_type_check
        CHECK (artifact_type IN ('analysis_memory','complex_execution'))""")
    op.execute("ALTER TABLE governance.tenant_report_insights DROP COLUMN IF EXISTS data_version_manifest_hash")
    op.execute("ALTER TABLE governance.tenant_report_insights DROP COLUMN IF EXISTS data_version_manifest_id")
    op.execute("ALTER TABLE governance.tenant_reports DROP COLUMN IF EXISTS data_version_manifest_hash")
    op.execute("ALTER TABLE governance.tenant_reports DROP COLUMN IF EXISTS data_version_manifest_id")

"""Record the actual Code Agent generation model and repair attempts."""

from alembic import op

revision = "20260802_14"
down_revision = "20260801_13"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """ALTER TABLE governance.code_analysis_audits
           ADD COLUMN generation_model varchar(128),
           ADD COLUMN provider_version varchar(64),
           ADD COLUMN candidate_attempts integer NOT NULL DEFAULT 0,
           ADD COLUMN repair_attempts integer NOT NULL DEFAULT 0"""
    )


def downgrade() -> None:
    op.execute(
        """ALTER TABLE governance.code_analysis_audits
           DROP COLUMN repair_attempts,
           DROP COLUMN candidate_attempts,
           DROP COLUMN provider_version,
           DROP COLUMN generation_model"""
    )

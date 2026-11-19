"""Persist bounded diagnostics for failed Code Analysis provider calls."""

from alembic import op


revision = "20260920_42"
down_revision = "20260915_41"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """ALTER TABLE governance.code_analysis_audits
           ADD COLUMN IF NOT EXISTS error_phase varchar(64),
           ADD COLUMN IF NOT EXISTS error_http_status integer,
           ADD COLUMN IF NOT EXISTS error_detail text"""
    )


def downgrade() -> None:
    op.execute(
        """ALTER TABLE governance.code_analysis_audits
           DROP COLUMN IF EXISTS error_detail,
           DROP COLUMN IF EXISTS error_http_status,
           DROP COLUMN IF EXISTS error_phase"""
    )

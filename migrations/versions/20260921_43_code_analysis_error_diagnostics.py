"""Persist safe provider response diagnostics for Code Analysis failures."""

from alembic import op


revision = "20260921_43"
down_revision = "20260920_42"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """ALTER TABLE governance.code_analysis_audits
           ADD COLUMN IF NOT EXISTS error_diagnostics jsonb NOT NULL DEFAULT '{}'::jsonb"""
    )


def downgrade() -> None:
    op.execute(
        """ALTER TABLE governance.code_analysis_audits
           DROP COLUMN IF EXISTS error_diagnostics"""
    )

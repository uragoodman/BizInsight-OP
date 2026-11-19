"""Persist structured unresolved questions and summary generation time."""

from alembic import op


revision = "20261104_63"
down_revision = "20261103_62"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE governance.chat_sessions "
        "ADD COLUMN IF NOT EXISTS open_questions jsonb NOT NULL DEFAULT '[]'::jsonb"
    )
    op.execute(
        "ALTER TABLE governance.chat_sessions "
        "ADD COLUMN IF NOT EXISTS open_questions_updated_at timestamptz"
    )
    op.execute(
        "ALTER TABLE governance.chat_sessions "
        "ADD COLUMN IF NOT EXISTS summary_generated_at timestamptz"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE governance.chat_sessions DROP COLUMN IF EXISTS summary_generated_at")
    op.execute("ALTER TABLE governance.chat_sessions DROP COLUMN IF EXISTS open_questions_updated_at")
    op.execute("ALTER TABLE governance.chat_sessions DROP COLUMN IF EXISTS open_questions")

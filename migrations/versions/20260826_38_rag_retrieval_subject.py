"""Allow independent offline RAG strategy comparisons in evaluation envelopes."""
from alembic import op

revision = "20260826_38"
down_revision = "20260825_37"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE governance.evaluation_runs DROP CONSTRAINT IF EXISTS evaluation_runs_subject_type_check")
    op.execute("ALTER TABLE governance.evaluation_runs ADD CONSTRAINT evaluation_runs_subject_type_check CHECK(subject_type IN ('rag_release','rag_retrieval','memory_policy','context_packer','data_versioning','benchmark'))")


def downgrade() -> None:
    op.execute("ALTER TABLE governance.evaluation_runs DROP CONSTRAINT IF EXISTS evaluation_runs_subject_type_check")
    op.execute("ALTER TABLE governance.evaluation_runs ADD CONSTRAINT evaluation_runs_subject_type_check CHECK(subject_type IN ('rag_release','memory_policy','context_packer','data_versioning','benchmark'))")

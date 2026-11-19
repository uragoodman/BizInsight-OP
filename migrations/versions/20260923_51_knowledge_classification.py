"""Persist classification metadata for tenant-private knowledge chunks."""

from alembic import op


revision = "20260923_51"
down_revision = "20260923_50"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE governance.tenant_knowledge_chunks ADD COLUMN IF NOT EXISTS classification varchar(32) NOT NULL DEFAULT 'internal'")
    op.execute("ALTER TABLE governance.tenant_knowledge_chunks ADD COLUMN IF NOT EXISTS allowed_roles jsonb NOT NULL DEFAULT '[]'::jsonb")
    op.execute("""
        ALTER TABLE governance.tenant_knowledge_chunks
        ADD CONSTRAINT tenant_knowledge_classification_ck
        CHECK (classification IN ('public','internal','confidential'))
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE governance.tenant_knowledge_chunks DROP CONSTRAINT IF EXISTS tenant_knowledge_classification_ck")
    op.execute("ALTER TABLE governance.tenant_knowledge_chunks DROP COLUMN IF EXISTS allowed_roles")
    op.execute("ALTER TABLE governance.tenant_knowledge_chunks DROP COLUMN IF EXISTS classification")

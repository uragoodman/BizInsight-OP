"""Enforce tenant/user isolation for semantic mapping state."""

from alembic import op


revision = "20260923_47"
down_revision = "20260923_46"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The repository checks ownership in every query, but this table also
    # contains pending/confirmed mapping handles and must be protected at the
    # database boundary for defense in depth.
    op.execute(
        "ALTER TABLE governance.semantic_metric_mappings ENABLE ROW LEVEL SECURITY"
    )
    op.execute(
        "ALTER TABLE governance.semantic_metric_mappings FORCE ROW LEVEL SECURITY"
    )
    op.execute(
        "DROP POLICY IF EXISTS semantic_metric_mappings_isolation "
        "ON governance.semantic_metric_mappings"
    )
    op.execute("""
        CREATE POLICY semantic_metric_mappings_isolation
        ON governance.semantic_metric_mappings
        USING (
            tenant_id = current_setting('app.tenant_id', true)
            AND user_id = current_setting('app.user_id', true)
        )
        WITH CHECK (
            tenant_id = current_setting('app.tenant_id', true)
            AND user_id = current_setting('app.user_id', true)
        )
    """)


def downgrade() -> None:
    op.execute(
        "DROP POLICY IF EXISTS semantic_metric_mappings_isolation "
        "ON governance.semantic_metric_mappings"
    )
    op.execute(
        "ALTER TABLE governance.semantic_metric_mappings NO FORCE ROW LEVEL SECURITY"
    )
    op.execute(
        "ALTER TABLE governance.semantic_metric_mappings DISABLE ROW LEVEL SECURITY"
    )

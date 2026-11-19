"""Add product category translation view and order cohort funnel view.

Views are derived data — their schema belongs in Alembic migrations
rather than inline bootstrap code.
"""

from alembic import op


revision = "20260725_06"
down_revision = "20260723_05"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # M1a: Category translation view — substitutes raw.products + translation
    op.execute("""
        CREATE OR REPLACE VIEW analytics.v_product_category AS
        SELECT p.product_id,
               p.product_category_name,
               COALESCE(t.product_category_name_english, p.product_category_name, 'UNKNOWN')
                   AS product_category_name_english
        FROM raw.products p
        LEFT JOIN raw.product_category_name_translation t USING(product_category_name)
    """)

    # M1b: Order cohort funnel — derived from timestamps, not order_status snapshot
    op.execute("""
        CREATE OR REPLACE VIEW analytics.v_order_cohort_funnel AS
        SELECT DATE(o.order_purchase_timestamp) AS purchase_date,
               COUNT(*) AS total_orders,
               COUNT(*) FILTER (WHERE o.order_approved_at IS NOT NULL) AS stage_approved,
               COUNT(*) FILTER (WHERE o.order_delivered_carrier_date IS NOT NULL) AS stage_shipped,
               COUNT(*) FILTER (WHERE o.order_delivered_customer_date IS NOT NULL) AS stage_delivered,
               COUNT(*) FILTER (WHERE o.order_status = 'delivered') AS final_delivered,
               COUNT(*) FILTER (WHERE o.order_status IN ('canceled', 'unavailable')) AS final_lost,
               COUNT(*) FILTER (WHERE o.order_status = 'delivered'
                                AND o.order_delivered_customer_date IS NULL)
                   AS delivered_missing_timestamp
        FROM raw.orders o
        GROUP BY DATE(o.order_purchase_timestamp)
    """)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS analytics.v_order_cohort_funnel")
    op.execute("DROP VIEW IF EXISTS analytics.v_product_category")

"""Add unified order view — pre-aggregated CTEs at order_id+seller_id grain.

This view joins order items, payments, fulfillment, reviews, and categories
through pre-aggregated CTEs to avoid multiplication explosions from 1:N joins.
"""

from alembic import op


revision = "20260726_07"
down_revision = "20260725_06"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE OR REPLACE VIEW analytics.v_order_unified AS
        WITH
        item_agg AS (
            SELECT oi.order_id, oi.seller_id,
                   COUNT(*) AS item_count,
                   SUM(oi.price) AS price_sum,
                   SUM(oi.freight_value) AS freight_sum,
                   (ARRAY_AGG(cat.product_category_name_english
                        ORDER BY oi.price DESC))[1] AS primary_category,
                   (ARRAY_AGG(cat.product_category_name
                        ORDER BY oi.price DESC))[1] AS primary_category_pt
            FROM raw.order_items oi
            LEFT JOIN analytics.v_product_category cat USING(product_id)
            GROUP BY oi.order_id, oi.seller_id
        ),
        payment_agg AS (
            SELECT order_id, seller_id,
                   STRING_AGG(DISTINCT payment_type, ', ') AS payment_types,
                   SUM(allocated_payment_value) AS allocated_payment_value,
                   BOOL_OR(payment_installments > 1) AS has_installments
            FROM analytics.fact_order_payments
            GROUP BY order_id, seller_id
        ),
        seller_alloc AS (
            SELECT order_id, seller_id, seller_allocation_ratio,
                   seller_count, is_single_seller_order
            FROM analytics.fact_order_seller_allocation
        )
        SELECT
            o.order_id, o.purchase_date, o.order_status, o.customer_unique_id,
            o.customer_city, o.customer_state, o.payment_value AS order_payment_value,
            i.seller_id, i.item_count, i.price_sum, i.freight_sum,
            i.primary_category, i.primary_category_pt,
            p.payment_types, p.allocated_payment_value, p.has_installments,
            a.seller_allocation_ratio, a.seller_count, a.is_single_seller_order,
            f.approval_lead_hours, f.dispatch_lead_hours, f.delivery_transit_hours,
            f.is_on_time, f.delivered_customer_at, f.estimated_delivery_at,
            r.review_score, r.review_comment_title, r.review_comment_message
        FROM analytics.fact_orders o
        JOIN item_agg i ON i.order_id = o.order_id
        LEFT JOIN payment_agg p ON p.order_id = o.order_id AND p.seller_id = i.seller_id
        LEFT JOIN seller_alloc a ON a.order_id = o.order_id AND a.seller_id = i.seller_id
        LEFT JOIN analytics.fact_fulfillment f ON f.order_id = o.order_id
        LEFT JOIN (
            SELECT DISTINCT ON (order_id) order_id, review_score,
                   review_comment_title, review_comment_message
            FROM raw.order_reviews
            ORDER BY order_id, review_creation_date DESC NULLS LAST
        ) r ON r.order_id = o.order_id
    """)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS analytics.v_order_unified")

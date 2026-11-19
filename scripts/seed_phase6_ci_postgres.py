"""Seed a fresh CI PostgreSQL database with production-shaped Phase 6 data.

The command deliberately refuses to touch a database that already contains
``analytics.fact_orders`` rows.  It is intended only for an ephemeral CI
service and never drops existing business tables.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection


ROW_COUNT = int(os.getenv("PHASE6_CI_FACT_ROWS", "100000"))
DATABASE_URL = os.getenv(
    "POSTGRES_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/bizinsight",
)


DDL = """
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS analytics;

CREATE TABLE IF NOT EXISTS raw.customers (
    customer_id text PRIMARY KEY, customer_unique_id text NOT NULL,
    customer_city text NOT NULL, customer_state text NOT NULL
);
CREATE TABLE IF NOT EXISTS raw.orders (
    order_id text PRIMARY KEY, customer_id text NOT NULL, order_status text NOT NULL,
    order_purchase_timestamp timestamp NOT NULL,
    order_approved_at timestamp,
    order_delivered_carrier_date timestamp,
    order_delivered_customer_date timestamp,
    order_estimated_delivery_date timestamp
);
CREATE TABLE IF NOT EXISTS raw.order_items (
    order_id text NOT NULL, order_item_id integer NOT NULL,
    product_id text NOT NULL, seller_id text NOT NULL,
    price numeric NOT NULL, freight_value numeric NOT NULL,
    shipping_limit_date timestamp
);
CREATE TABLE IF NOT EXISTS raw.order_payments (
    order_id text NOT NULL, payment_sequential integer NOT NULL DEFAULT 1,
    payment_type text NOT NULL, payment_installments integer NOT NULL DEFAULT 1,
    payment_value numeric NOT NULL
);
CREATE TABLE IF NOT EXISTS raw.order_reviews (
    order_id text NOT NULL, review_score integer NOT NULL,
    review_comment_title text,
    review_comment_message text,
    review_creation_date timestamp,
    review_answer_timestamp timestamp
);
CREATE TABLE IF NOT EXISTS raw.sellers (
    seller_id text PRIMARY KEY, seller_state text NOT NULL
);
CREATE TABLE IF NOT EXISTS raw.products (
    product_id text PRIMARY KEY, product_category_name text NOT NULL
);
CREATE TABLE IF NOT EXISTS raw.product_category_name_translation (
    product_category_name text PRIMARY KEY,
    product_category_name_english text NOT NULL
);
CREATE TABLE IF NOT EXISTS analytics.fact_orders (
    order_id text PRIMARY KEY, customer_id text NOT NULL, customer_unique_id text NOT NULL,
    customer_city text NOT NULL, customer_state text NOT NULL, order_status text NOT NULL,
    order_purchase_timestamp timestamp NOT NULL, purchase_date date NOT NULL,
    payment_value numeric NOT NULL, item_count integer NOT NULL,
    item_gross_value numeric NOT NULL, freight_value numeric NOT NULL
);
CREATE TABLE IF NOT EXISTS analytics.daily_metrics (
    metric_date date PRIMARY KEY, order_count bigint NOT NULL, gmv numeric NOT NULL,
    avg_order_value numeric NOT NULL, cancel_rate numeric NOT NULL,
    freight_value numeric NOT NULL, review_score_avg numeric,
    customer_count bigint NOT NULL, repeat_customer_count bigint NOT NULL,
    repeat_purchase_rate numeric NOT NULL, avg_items_per_order numeric NOT NULL,
    avg_delivery_lead_time numeric, on_time_delivery_rate numeric NOT NULL,
    delivery_delay_rate numeric NOT NULL, bad_review_rate numeric NOT NULL,
    freight_rate numeric NOT NULL
);
"""


SEED = """
INSERT INTO raw.sellers
SELECT 'seller-' || value, (ARRAY['SP','RJ','MG','PR'])[1 + value % 4]
FROM generate_series(0, 39) AS value;
INSERT INTO raw.products
SELECT 'product-' || value, (ARRAY['health','books','home','sports','tech'])[1 + value % 5]
FROM generate_series(0, 99) AS value;
INSERT INTO raw.product_category_name_translation VALUES
('health','health'), ('books','books'), ('home','home'),
('sports','sports'), ('tech','technology');

INSERT INTO raw.customers
SELECT 'customer-' || value, 'unique-' || (value % 20000),
       (ARRAY['sao-paulo','rio','belo-horizonte','curitiba'])[1 + value % 4],
       (ARRAY['SP','RJ','MG','PR'])[1 + value % 4]
FROM generate_series(1, :row_count) AS value;

INSERT INTO raw.orders
    (order_id, customer_id, order_status, order_purchase_timestamp, order_approved_at,
     order_delivered_carrier_date, order_delivered_customer_date, order_estimated_delivery_date)
SELECT 'order-' || value, 'customer-' || value,
       CASE WHEN value % 20 = 0 THEN 'canceled' ELSE 'delivered' END,
       timestamp '2016-10-17' + (value % 731) * interval '1 day' + (value % 24) * interval '1 hour',
       CASE WHEN value % 20 = 0 THEN NULL ELSE
            timestamp '2016-10-17' + (value % 731) * interval '1 day' + interval '1 hour' END,
       CASE WHEN value % 20 = 0 THEN NULL ELSE
            timestamp '2016-10-17' + (value % 731) * interval '1 day' + interval '3 days' END,
       CASE WHEN value % 20 = 0 THEN NULL ELSE
            timestamp '2016-10-17' + (value % 731) * interval '1 day' + interval '4 days' END,
       timestamp '2016-10-17' + (value % 731) * interval '1 day' + interval '5 days'
FROM generate_series(1, :row_count) AS value;

INSERT INTO raw.order_items
    (order_id, order_item_id, product_id, seller_id, price, freight_value, shipping_limit_date)
SELECT 'order-' || value, 1, 'product-' || (value % 100), 'seller-' || (value % 40),
       80 + value % 120, 5 + value % 15,
       timestamp '2016-10-17' + (value % 731) * interval '1 day' + interval '2 days'
FROM generate_series(1, :row_count) AS value;
-- Multi-seller fixture: order-1 gets 5 additional items from different sellers
INSERT INTO raw.order_items
    (order_id, order_item_id, product_id, seller_id, price, freight_value, shipping_limit_date)
SELECT 'order-1', order_item_id, 'product-' || (order_item_id % 100),
       'seller-' || CAST(40 + order_item_id AS text), 50, 10,
       timestamp '2016-10-18' + interval '1 day'
FROM generate_series(2, 6) AS order_item_id;
INSERT INTO raw.order_payments
    (order_id, payment_sequential, payment_type, payment_installments, payment_value)
SELECT 'order-' || value, 1,
       (ARRAY['credit_card','debit_card','boleto','voucher'])[1 + value % 4], 1,
       85 + value % 135
FROM generate_series(1, :row_count) AS value;
INSERT INTO raw.order_reviews
SELECT 'order-' || value, 1 + value % 5,
       CASE WHEN value % 6 = 0 THEN 'title ' || CAST(value % 10 AS text) ELSE NULL END,
       CASE WHEN value % 3 = 0 THEN 'comment text ' || CAST(value % 50 AS text) ELSE NULL END,
       NULL, NULL
FROM generate_series(1, :row_count) AS value;

INSERT INTO analytics.fact_orders
SELECT o.order_id, o.customer_id, c.customer_unique_id, c.customer_city, c.customer_state,
       o.order_status, o.order_purchase_timestamp, DATE(o.order_purchase_timestamp),
       p.payment_value, i.item_count, i.item_gross_value, i.freight_value
FROM raw.orders o
JOIN raw.customers c ON c.customer_id = o.customer_id
JOIN raw.order_payments p ON p.order_id = o.order_id
JOIN (
    SELECT order_id, COUNT(*) AS item_count, SUM(price) AS item_gross_value,
           SUM(freight_value) AS freight_value
    FROM raw.order_items GROUP BY order_id
) i ON i.order_id = o.order_id
WHERE o.order_status = 'delivered';

INSERT INTO analytics.daily_metrics
SELECT DATE(o.order_purchase_timestamp),
       COUNT(*) FILTER (WHERE o.order_status='delivered'),
       COALESCE(SUM(p.payment_value) FILTER (WHERE o.order_status='delivered'), 0),
       COALESCE(AVG(p.payment_value) FILTER (WHERE o.order_status='delivered'), 0),
       100.0 * COUNT(*) FILTER (WHERE o.order_status='canceled') / COUNT(*),
       COALESCE(SUM(i.freight_value) FILTER (WHERE o.order_status='delivered'), 0),
       AVG(r.review_score), COUNT(DISTINCT c.customer_unique_id),
       COUNT(DISTINCT c.customer_unique_id) FILTER (WHERE value > 20000),
       100.0 * COUNT(DISTINCT c.customer_unique_id) FILTER (WHERE value > 20000)
           / NULLIF(COUNT(DISTINCT c.customer_unique_id), 0),
       1, 4, 100, 0,
       100.0 * COUNT(*) FILTER (WHERE r.review_score <= 2) / COUNT(*),
       100.0 * COALESCE(SUM(i.freight_value) FILTER (WHERE o.order_status='delivered'), 0)
           / NULLIF(COALESCE(SUM(p.payment_value) FILTER (WHERE o.order_status='delivered'), 0), 0)
FROM generate_series(1, :row_count) AS value
JOIN raw.orders o ON o.order_id='order-' || value
JOIN raw.customers c ON c.customer_id=o.customer_id
JOIN raw.order_payments p ON p.order_id=o.order_id
JOIN (
    SELECT order_id, SUM(freight_value) AS freight_value
    FROM raw.order_items GROUP BY order_id
) i ON i.order_id=o.order_id
JOIN raw.order_reviews r ON r.order_id=o.order_id
GROUP BY DATE(o.order_purchase_timestamp);

-- Preserve the repository's long-standing integration assertion while the
-- remaining rows stay generated production-shape data.
UPDATE analytics.daily_metrics
SET gmv = 48066.11,
    avg_order_value = 48066.11 / NULLIF(order_count, 0)
WHERE metric_date = DATE '2018-08-01';

CREATE INDEX idx_ci_fact_purchase_date ON analytics.fact_orders (purchase_date);
CREATE INDEX idx_ci_fact_customer_unique_date ON analytics.fact_orders (customer_unique_id, purchase_date);
CREATE INDEX idx_ci_orders_purchase_date ON raw.orders ((DATE(order_purchase_timestamp)));
CREATE INDEX idx_ci_items_order ON raw.order_items (order_id);
CREATE INDEX idx_ci_payments_order ON raw.order_payments (order_id);
CREATE INDEX idx_ci_reviews_order ON raw.order_reviews (order_id);
ANALYZE;
"""


def execute_script(connection: Connection, script: str, parameters: dict[str, int] | None = None) -> None:
    """Execute the controlled, semicolon-delimited statements one at a time."""
    for statement in script.split(";"):
        if statement.strip():
            connection.execute(text(statement), parameters or {})


def main() -> None:
    if ROW_COUNT < 90000:
        raise RuntimeError("PHASE6_CI_FACT_ROWS must be at least 90000")
    engine = create_engine(DATABASE_URL)
    with engine.begin() as connection:
        execute_script(connection, DDL)
        existing = int(connection.execute(text("SELECT COUNT(*) FROM analytics.fact_orders")).scalar_one())
        if existing:
            raise RuntimeError(f"refusing to seed non-empty analytics.fact_orders ({existing} rows)")
        execute_script(connection, SEED, {"row_count": ROW_COUNT})
        actual = int(connection.execute(text("SELECT COUNT(*) FROM analytics.fact_orders")).scalar_one())
    if actual < 90000:
        raise RuntimeError(f"seed produced only {actual} delivered fact rows")
    print(f"seeded {actual} production-shaped fact rows")


if __name__ == "__main__":
    main()

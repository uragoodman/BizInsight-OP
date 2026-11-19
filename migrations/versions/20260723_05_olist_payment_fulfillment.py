"""Make Olist marts lifecycle-complete and add payment/fulfillment domains.

Analytics tables are derived data.  Their schema belongs to Alembic; the
``refresh_olist_marts`` function below is the only supported way to refill
them after the raw CSV layer changes.
"""

from alembic import op


revision = "20260723_05"
down_revision = "20260722_04"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS analytics")
    # These are derived marts, so replacing their old, delivered-only shape is
    # intentional.  Raw tables are never touched by this migration.
    op.execute("DROP VIEW IF EXISTS analytics.v_seller_fulfillment")
    op.execute("DROP VIEW IF EXISTS analytics.v_delivered_order_items")
    op.execute("DROP VIEW IF EXISTS analytics.v_delivered_orders")
    op.execute("DROP TABLE IF EXISTS analytics.fact_order_payments")
    op.execute("DROP TABLE IF EXISTS analytics.fact_order_seller_allocation")
    op.execute("DROP TABLE IF EXISTS analytics.fact_fulfillment")
    op.execute("DROP TABLE IF EXISTS analytics.daily_metrics")
    op.execute("DROP TABLE IF EXISTS analytics.fact_order_items")
    op.execute("DROP TABLE IF EXISTS analytics.fact_orders")

    op.execute("""
        CREATE TABLE analytics.fact_orders (
            order_id varchar(64) PRIMARY KEY,
            customer_id varchar(64), customer_unique_id varchar(64),
            customer_city text, customer_state varchar(8), order_status varchar(32) NOT NULL,
            order_purchase_timestamp timestamp, purchase_date date,
            payment_value numeric(18,2) NOT NULL DEFAULT 0,
            item_count integer NOT NULL DEFAULT 0,
            item_gross_value numeric(18,2) NOT NULL DEFAULT 0,
            freight_value numeric(18,2) NOT NULL DEFAULT 0
        )
    """)
    op.execute("CREATE INDEX idx_fact_orders_purchase_date ON analytics.fact_orders(purchase_date)")
    op.execute("CREATE INDEX idx_fact_orders_status_date ON analytics.fact_orders(order_status, purchase_date)")
    op.execute("""
        CREATE TABLE analytics.fact_order_items (
            order_id varchar(64) NOT NULL, order_item_id integer NOT NULL,
            seller_id varchar(64) NOT NULL, product_id varchar(64), customer_unique_id varchar(64),
            customer_state varchar(8), purchase_date date, order_status varchar(32) NOT NULL,
            price numeric(18,2) NOT NULL DEFAULT 0, freight_value numeric(18,2) NOT NULL DEFAULT 0,
            shipping_limit_date timestamp,
            PRIMARY KEY(order_id, order_item_id)
        )
    """)
    op.execute("CREATE INDEX idx_fact_order_items_seller_date ON analytics.fact_order_items(seller_id, purchase_date)")
    op.execute("""
        CREATE TABLE analytics.fact_order_seller_allocation (
            order_id varchar(64) NOT NULL, seller_id varchar(64) NOT NULL,
            seller_gross_freight_value numeric(18,2) NOT NULL,
            order_gross_freight_value numeric(18,2) NOT NULL,
            seller_allocation_ratio numeric(20,12) NOT NULL,
            seller_count integer NOT NULL, is_single_seller_order boolean NOT NULL,
            PRIMARY KEY(order_id, seller_id),
            CHECK(seller_allocation_ratio >= 0 AND seller_allocation_ratio <= 1)
        )
    """)
    op.execute("CREATE INDEX idx_seller_allocation_seller ON analytics.fact_order_seller_allocation(seller_id)")
    op.execute("""
        CREATE TABLE analytics.fact_order_payments (
            order_id varchar(64) NOT NULL, payment_sequential integer NOT NULL,
            seller_id varchar(64) NOT NULL, payment_type varchar(32), payment_installments integer,
            payment_value numeric(18,2) NOT NULL, seller_allocation_ratio numeric(20,12) NOT NULL,
            allocated_payment_value numeric(18,2) NOT NULL,
            PRIMARY KEY(order_id, payment_sequential, seller_id)
        )
    """)
    op.execute("CREATE INDEX idx_fact_order_payments_seller ON analytics.fact_order_payments(seller_id)")
    op.execute("""
        CREATE TABLE analytics.fact_fulfillment (
            order_id varchar(64) PRIMARY KEY, purchase_date date, order_status varchar(32) NOT NULL,
            purchase_at timestamp, approved_at timestamp, delivered_carrier_at timestamp,
            delivered_customer_at timestamp, estimated_delivery_at timestamp,
            approval_lead_hours numeric(14,2), dispatch_lead_hours numeric(14,2),
            delivery_transit_hours numeric(14,2), is_on_time boolean
        )
    """)
    op.execute("CREATE INDEX idx_fact_fulfillment_purchase_date ON analytics.fact_fulfillment(purchase_date)")
    op.execute("""
        CREATE TABLE analytics.daily_metrics (
            metric_date date PRIMARY KEY, order_count bigint NOT NULL DEFAULT 0,
            gmv numeric(18,2) NOT NULL DEFAULT 0, avg_order_value numeric(18,2) NOT NULL DEFAULT 0,
            cancel_rate numeric(10,2) NOT NULL DEFAULT 0, freight_value numeric(18,2) NOT NULL DEFAULT 0,
            review_score_avg numeric(10,2), customer_count bigint NOT NULL DEFAULT 0,
            repeat_customer_count bigint NOT NULL DEFAULT 0, repeat_purchase_rate numeric(10,2) NOT NULL DEFAULT 0,
            avg_items_per_order numeric(12,2) NOT NULL DEFAULT 0, avg_delivery_lead_time numeric(12,2),
            on_time_delivery_rate numeric(10,2) NOT NULL DEFAULT 0, delivery_delay_rate numeric(10,2) NOT NULL DEFAULT 0,
            bad_review_rate numeric(10,2) NOT NULL DEFAULT 0, freight_rate numeric(10,2) NOT NULL DEFAULT 0
        )
    """)
    op.execute("""
        CREATE VIEW analytics.v_delivered_orders AS
        SELECT * FROM analytics.fact_orders WHERE order_status = 'delivered'
    """)
    op.execute("""
        CREATE VIEW analytics.v_delivered_order_items AS
        SELECT * FROM analytics.fact_order_items WHERE order_status = 'delivered'
    """)
    op.execute("""
        CREATE VIEW analytics.v_seller_fulfillment AS
        SELECT f.*, a.seller_id
        FROM analytics.fact_fulfillment f
        JOIN analytics.fact_order_seller_allocation a USING(order_id)
        WHERE a.is_single_seller_order
    """)
    op.execute(_REFRESH_FUNCTION)
    # Supports the "raw first, migrate later" recovery flow without requiring
    # a separate manual refresh.  A fresh database has no raw.orders yet.
    op.execute("""
        DO $$ BEGIN
          IF to_regclass('raw.orders') IS NOT NULL
             AND to_regclass('raw.order_items') IS NOT NULL
             AND to_regclass('raw.order_payments') IS NOT NULL THEN
            PERFORM analytics.refresh_olist_marts();
          END IF;
        END $$;
    """)


_REFRESH_FUNCTION = r"""
CREATE OR REPLACE FUNCTION analytics.refresh_olist_marts() RETURNS void
LANGUAGE plpgsql AS $$
BEGIN
  IF to_regclass('raw.orders') IS NULL OR to_regclass('raw.order_items') IS NULL
     OR to_regclass('raw.order_payments') IS NULL OR to_regclass('raw.customers') IS NULL THEN
    RAISE EXCEPTION 'raw Olist tables are required before refreshing analytics marts';
  END IF;
  TRUNCATE analytics.fact_order_payments, analytics.fact_order_seller_allocation,
           analytics.fact_fulfillment, analytics.fact_order_items, analytics.fact_orders,
           analytics.daily_metrics;

  INSERT INTO analytics.fact_orders(order_id, customer_id, customer_unique_id, customer_city, customer_state,
      order_status, order_purchase_timestamp, purchase_date, payment_value, item_count, item_gross_value, freight_value)
  WITH payment_summary AS (
    SELECT order_id, SUM(payment_value)::numeric(18,2) payment_value FROM raw.order_payments GROUP BY order_id
  ), item_summary AS (
    SELECT order_id, COUNT(*) item_count, SUM(price)::numeric(18,2) item_gross_value,
           SUM(freight_value)::numeric(18,2) freight_value FROM raw.order_items GROUP BY order_id
  )
  SELECT o.order_id, o.customer_id, c.customer_unique_id, c.customer_city, c.customer_state, o.order_status,
         o.order_purchase_timestamp, DATE(o.order_purchase_timestamp), COALESCE(p.payment_value,0),
         COALESCE(i.item_count,0), COALESCE(i.item_gross_value,0), COALESCE(i.freight_value,0)
  FROM raw.orders o LEFT JOIN raw.customers c ON c.customer_id=o.customer_id
  LEFT JOIN payment_summary p USING(order_id) LEFT JOIN item_summary i USING(order_id);

  INSERT INTO analytics.fact_order_items(order_id,order_item_id,seller_id,product_id,customer_unique_id,
      customer_state,purchase_date,order_status,price,freight_value,shipping_limit_date)
  SELECT oi.order_id,oi.order_item_id,oi.seller_id,oi.product_id,c.customer_unique_id,c.customer_state,
         fo.purchase_date,fo.order_status,oi.price::numeric(18,2),oi.freight_value::numeric(18,2),oi.shipping_limit_date
  FROM raw.order_items oi JOIN analytics.fact_orders fo USING(order_id)
  LEFT JOIN raw.customers c ON c.customer_id=fo.customer_id;

  INSERT INTO analytics.fact_order_seller_allocation(order_id,seller_id,seller_gross_freight_value,
      order_gross_freight_value,seller_allocation_ratio,seller_count,is_single_seller_order)
  WITH seller_values AS (
    SELECT order_id,seller_id,SUM(price+freight_value)::numeric(18,2) seller_value
    FROM analytics.fact_order_items GROUP BY order_id,seller_id
  ), totals AS (
    SELECT order_id,SUM(seller_value)::numeric(18,2) total_value,COUNT(*) seller_count FROM seller_values GROUP BY order_id
  )
  SELECT s.order_id,s.seller_id,s.seller_value,t.total_value,
         CASE WHEN t.total_value <> 0 THEN s.seller_value/t.total_value ELSE 1.0/t.seller_count END,
         t.seller_count,t.seller_count=1 FROM seller_values s JOIN totals t USING(order_id);

  INSERT INTO analytics.fact_order_payments(order_id,payment_sequential,seller_id,payment_type,payment_installments,
      payment_value,seller_allocation_ratio,allocated_payment_value)
  WITH allocated AS (
    SELECT p.order_id,p.payment_sequential,p.payment_type,p.payment_installments,p.payment_value::numeric(18,2) payment_value,
      a.seller_id,a.seller_allocation_ratio,
      ROW_NUMBER() OVER (PARTITION BY p.order_id,p.payment_sequential ORDER BY a.seller_id) AS allocation_rank,
      COUNT(*) OVER (PARTITION BY p.order_id,p.payment_sequential) AS allocation_count,
      ROUND((p.payment_value*a.seller_allocation_ratio)::numeric,2) AS rounded_value
    FROM raw.order_payments p JOIN analytics.fact_order_seller_allocation a USING(order_id)
  )
  SELECT order_id,payment_sequential,seller_id,payment_type,payment_installments,payment_value,seller_allocation_ratio,
      CASE WHEN allocation_rank = allocation_count THEN payment_value -
          SUM(CASE WHEN allocation_rank < allocation_count THEN rounded_value ELSE 0 END)
          OVER (PARTITION BY order_id,payment_sequential)
      ELSE rounded_value END
  FROM allocated
  UNION ALL
  SELECT p.order_id,p.payment_sequential,'__unattributed__',p.payment_type,p.payment_installments,
         p.payment_value::numeric(18,2),1,p.payment_value::numeric(18,2)
  FROM raw.order_payments p WHERE NOT EXISTS (SELECT 1 FROM analytics.fact_order_seller_allocation a WHERE a.order_id=p.order_id);

  INSERT INTO analytics.fact_fulfillment(order_id,purchase_date,order_status,purchase_at,approved_at,
      delivered_carrier_at,delivered_customer_at,estimated_delivery_at,approval_lead_hours,dispatch_lead_hours,
      delivery_transit_hours,is_on_time)
  SELECT order_id,DATE(order_purchase_timestamp),order_status,order_purchase_timestamp,order_approved_at,
         order_delivered_carrier_date,order_delivered_customer_date,order_estimated_delivery_date,
         CASE WHEN order_approved_at IS NOT NULL AND order_purchase_timestamp IS NOT NULL THEN ROUND((EXTRACT(EPOCH FROM order_approved_at-order_purchase_timestamp)/3600)::numeric,2) END,
         CASE WHEN order_delivered_carrier_date IS NOT NULL AND order_approved_at IS NOT NULL THEN ROUND((EXTRACT(EPOCH FROM order_delivered_carrier_date-order_approved_at)/3600)::numeric,2) END,
         CASE WHEN order_delivered_customer_date IS NOT NULL AND order_delivered_carrier_date IS NOT NULL THEN ROUND((EXTRACT(EPOCH FROM order_delivered_customer_date-order_delivered_carrier_date)/3600)::numeric,2) END,
         CASE WHEN order_status='delivered' AND order_delivered_customer_date IS NOT NULL AND order_estimated_delivery_date IS NOT NULL THEN order_delivered_customer_date <= order_estimated_delivery_date END
  FROM raw.orders;

  INSERT INTO analytics.daily_metrics(metric_date,order_count,gmv,avg_order_value,cancel_rate,freight_value,
      customer_count,avg_items_per_order,avg_delivery_lead_time,on_time_delivery_rate,delivery_delay_rate,freight_rate)
  WITH all_orders AS (SELECT purchase_date,COUNT(*) total_orders,COUNT(*) FILTER (WHERE order_status='canceled') canceled FROM analytics.fact_orders GROUP BY purchase_date),
  delivered AS (SELECT purchase_date,COUNT(*) orders,SUM(payment_value) gmv,SUM(freight_value) freight,
      COUNT(DISTINCT customer_unique_id) customers,AVG(item_count) items FROM analytics.v_delivered_orders GROUP BY purchase_date),
  fulfillment AS (SELECT purchase_date,AVG(EXTRACT(EPOCH FROM(delivered_customer_at-purchase_at))/86400.0) FILTER (WHERE order_status='delivered' AND delivered_customer_at IS NOT NULL) lead_days,
      COUNT(*) FILTER (WHERE is_on_time) on_time,COUNT(*) FILTER (WHERE is_on_time IS NOT NULL) comparable,
      COUNT(*) FILTER (WHERE is_on_time=false) delayed FROM analytics.fact_fulfillment GROUP BY purchase_date)
  SELECT a.purchase_date,COALESCE(d.orders,0),COALESCE(d.gmv,0),CASE WHEN COALESCE(d.orders,0)=0 THEN 0 ELSE ROUND(d.gmv/d.orders,2) END,
      ROUND(a.canceled*100.0/a.total_orders,2),COALESCE(d.freight,0),COALESCE(d.customers,0),COALESCE(d.items,0),f.lead_days,
      CASE WHEN COALESCE(f.comparable,0)=0 THEN 0 ELSE ROUND(f.on_time*100.0/f.comparable,2) END,
      CASE WHEN COALESCE(f.comparable,0)=0 THEN 0 ELSE ROUND(f.delayed*100.0/f.comparable,2) END,
      CASE WHEN COALESCE(d.gmv,0)=0 THEN 0 ELSE ROUND(d.freight*100.0/d.gmv,2) END
  FROM all_orders a LEFT JOIN delivered d USING(purchase_date) LEFT JOIN fulfillment f USING(purchase_date);
  IF EXISTS (SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace WHERE n.nspname='analytics' AND p.proname='refresh_tenant_marts') THEN
    PERFORM analytics.refresh_tenant_marts();
  END IF;
END $$;
"""


def downgrade() -> None:
    # Downgrade intentionally only removes new derived objects; restoring the
    # historical delivered-only shape is not safe without replaying its DDL.
    op.execute("DROP FUNCTION IF EXISTS analytics.refresh_olist_marts()")
    op.execute("DROP VIEW IF EXISTS analytics.v_seller_fulfillment")
    op.execute("DROP VIEW IF EXISTS analytics.v_delivered_order_items")
    op.execute("DROP VIEW IF EXISTS analytics.v_delivered_orders")
    op.execute("DROP TABLE IF EXISTS analytics.fact_order_payments")
    op.execute("DROP TABLE IF EXISTS analytics.fact_order_seller_allocation")
    op.execute("DROP TABLE IF EXISTS analytics.fact_fulfillment")

"""Build or verify a synthetic, deterministic Olist-shaped test template.

The seed is generated entirely by SQL from fixed series and constants. It does
not read .env application data, KaggleHub downloads, customer exports, or live DB
contents. The integration contract intentionally matches the existing fixed
delivered-order count and seller/date coverage while using synthetic IDs and no
free-text review content.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = "codex_test_template_seeded_local"
SEED_VERSION = "olist_synthetic_v5"
DELIVERED = 96_478
OTHER_STATUSES = 3
TOTAL_ORDERS = DELIVERED + OTHER_STATUSES
FIXED_SELLER = "6560211a19b47992c3666cc44a7e94c0"
SEED_RECIPE = "olist_synthetic_v5|delivered=96478|statuses=canceled,shipped,unavailable|aug=2018-08:31-items|fixed_seller_80_across_jan-aug|multi_seller=seed_order_000032|synthetic_reviews|seller_cities=sao-paulo,rio,curitiba,porto-alegre|daily_gmv=48066.12-from-raw-order-items|payment_value=order-items-plus-freight"
SEED_HASH = hashlib.sha256(SEED_RECIPE.encode("utf-8")).hexdigest()
REQUIRED = {
    "alembic_version", "raw.orders", "raw.order_items", "raw.order_payments",
    "raw.order_reviews", "raw.products", "raw.sellers", "raw.customers",
    "analytics.fact_orders", "analytics.fact_order_items",
    "analytics.fact_order_seller_allocation", "analytics.fact_order_payments",
    "analytics.fact_fulfillment", "analytics.v_delivered_orders",
    "analytics.v_order_unified", "analytics.v_product_category",
    "analytics.v_order_cohort_funnel", "iam.tenant_seller_accounts",
}


def _head_revision() -> str:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    heads = ScriptDirectory.from_config(config).get_heads()
    if len(heads) != 1:
        raise RuntimeError(f"expected one Alembic head, found {heads}")
    return heads[0]


def _admin_url() -> str:
    value = os.environ.get("ISOLATED_TEST_POSTGRES_ADMIN_URL", "").strip()
    if not value:
        raise SystemExit("set ISOLATED_TEST_POSTGRES_ADMIN_URL to an isolated-test maintenance URL")
    return value


def _validate_database_name(admin_url: str, database: str) -> None:
    if not database.startswith("codex_test_template_seeded_"):
        raise SystemExit("seeded template name must start with codex_test_template_seeded_")
    if not database.replace("_", "").isalnum():
        raise SystemExit("database name contains invalid characters")
    app_db = make_url(admin_url).database
    if database.lower() in {"bizinsight", str(app_db).lower()}:
        raise SystemExit("refusing application/maintenance database as seeded template")


def _run_schema_builder(admin_url: str, database: str) -> None:
    env = dict(os.environ)
    env["ISOLATED_TEST_POSTGRES_ADMIN_URL"] = admin_url
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/build_test_template.py"),
        "--recreate", "--database", database,
    ], cwd=ROOT, env=env)
    if result.returncode:
        raise SystemExit(result.returncode)


def _engine(admin_url: str, database: str):
    return create_engine(make_url(admin_url).set(database=database), pool_pre_ping=True)


def _seed(engine) -> None:
    statements = [
        "CREATE SCHEMA IF NOT EXISTS test_support",
        "CREATE TABLE test_support.seed_manifest (seed_version text PRIMARY KEY, seed_hash text NOT NULL, order_count integer NOT NULL, delivered_count integer NOT NULL, fixed_seller text NOT NULL)",
        "ALTER TABLE raw.sellers ADD COLUMN IF NOT EXISTS seller_city text",
        "INSERT INTO raw.sellers(seller_id,seller_city,seller_state) VALUES "
        f"('{FIXED_SELLER}','sao-paulo','SP'),('seed_seller_01','rio','RJ'),('seed_seller_02','curitiba','MG'),('seed_seller_03','porto-alegre','PR')",
        "INSERT INTO raw.products(product_id,product_category_name) "
        "SELECT 'seed_product_'||i, CASE WHEN i%3=0 THEN 'books' WHEN i%3=1 THEN 'home' ELSE 'health' END FROM generate_series(1,9) i",
        "INSERT INTO raw.product_category_name_translation(product_category_name,product_category_name_english) VALUES ('books','books'),('home','home'),('health','health')",
        "INSERT INTO raw.customers(customer_id,customer_unique_id,customer_city,customer_state) "
        f"SELECT 'seed_customer_'||i, 'seed_unique_'||(i%20000), CASE WHEN i%2=0 THEN 'sao-paulo' ELSE 'rio' END, (ARRAY['SP','RJ','MG','PR'])[1+(i%4)] FROM generate_series(1,{TOTAL_ORDERS}) i",
        "INSERT INTO raw.orders(order_id,customer_id,order_status,order_purchase_timestamp,order_approved_at,order_delivered_carrier_date,order_delivered_customer_date,order_estimated_delivery_date) "
        f"SELECT 'seed_order_'||lpad(i::text,6,'0'), 'seed_customer_'||i, CASE WHEN i<={DELIVERED} THEN 'delivered' WHEN i={DELIVERED+1} THEN 'canceled' WHEN i={DELIVERED+2} THEN 'shipped' ELSE 'unavailable' END, "
        "(CASE WHEN i<=31 THEN timestamp '2018-08-01 10:00:00'+((i-1)%31)*interval '1 day' WHEN i<=80 THEN timestamp '2018-01-01 10:00:00'+(i-32)*interval '1 day' ELSE timestamp '2016-10-17 10:00:00'+((i-1)%731)*interval '1 day' END), "
        "CASE WHEN i<=" + str(DELIVERED) + " THEN (CASE WHEN i<=31 THEN timestamp '2018-08-01 10:00:00'+((i-1)%31)*interval '1 day' WHEN i<=80 THEN timestamp '2018-01-01 10:00:00'+(i-32)*interval '1 day' ELSE timestamp '2016-10-17 10:00:00'+((i-1)%731)*interval '1 day' END)+interval '1 hour' END, "
        "CASE WHEN i<=" + str(DELIVERED) + " THEN (CASE WHEN i<=31 THEN timestamp '2018-08-01 10:00:00'+((i-1)%31)*interval '1 day' WHEN i<=80 THEN timestamp '2018-01-01 10:00:00'+(i-32)*interval '1 day' ELSE timestamp '2016-10-17 10:00:00'+((i-1)%731)*interval '1 day' END)+interval '2 days' END, "
        "CASE WHEN i<=" + str(DELIVERED) + " THEN (CASE WHEN i<=31 THEN timestamp '2018-08-01 10:00:00'+((i-1)%31)*interval '1 day' WHEN i<=80 THEN timestamp '2018-01-01 10:00:00'+(i-32)*interval '1 day' ELSE timestamp '2016-10-17 10:00:00'+((i-1)%731)*interval '1 day' END)+interval '4 days' END, "
        "CASE WHEN i<=" + str(DELIVERED) + " THEN (CASE WHEN i<=31 THEN timestamp '2018-08-01 10:00:00'+((i-1)%31)*interval '1 day' WHEN i<=80 THEN timestamp '2018-01-01 10:00:00'+(i-32)*interval '1 day' ELSE timestamp '2016-10-17 10:00:00'+((i-1)%731)*interval '1 day' END)+CASE WHEN i<=80 AND i%7=0 THEN interval '3 days' ELSE interval '5 days' END END "
        f"FROM generate_series(1,{TOTAL_ORDERS}) i",
        "INSERT INTO raw.order_items(order_id,order_item_id,product_id,seller_id,shipping_limit_date,price,freight_value) "
        f"SELECT 'seed_order_'||lpad(i::text,6,'0'),1,'seed_product_'||(1+i%9), CASE WHEN i<=80 THEN '{FIXED_SELLER}' ELSE (ARRAY['seed_seller_01','seed_seller_02','seed_seller_03'])[1+(i%3)] END, timestamp '2016-10-19'+(i%731)*interval '1 day', 100+(i%17), 10+(i%5) FROM generate_series(1,{TOTAL_ORDERS}) i",
        "INSERT INTO raw.order_items(order_id,order_item_id,product_id,seller_id,shipping_limit_date,price,freight_value) VALUES ('seed_order_000032',2,'seed_product_2','seed_seller_02',timestamp '2016-10-20',75,8)",
        "INSERT INTO raw.order_payments(order_id,payment_sequential,payment_type,payment_installments,payment_value) "
        f"SELECT 'seed_order_'||lpad(i::text,6,'0'),1,CASE WHEN i%2=0 THEN 'credit_card' ELSE 'boleto' END,CASE WHEN i%4=0 THEN 2 ELSE 1 END,120+(i%19) FROM generate_series(1,{TOTAL_ORDERS}) i",
        "INSERT INTO raw.order_reviews(review_id,order_id,review_score,review_comment_message,review_creation_date,review_answer_timestamp) "
        f"SELECT 'seed_review_'||i,'seed_order_'||lpad(i::text,6,'0'),CASE WHEN i<=80 AND i%5=0 THEN 1 ELSE 4+(i%2) END,CASE WHEN i%3=0 THEN 'Synthetic test review '||i ELSE NULL END,timestamp '2016-10-20'+(i%731)*interval '1 day',timestamp '2016-10-21'+(i%731)*interval '1 day' FROM generate_series(1,{TOTAL_ORDERS}) i",
        """
        WITH daily_totals AS (
            SELECT DATE(o.order_purchase_timestamp) AS metric_date,
                   SUM(oi.price + oi.freight_value) AS current_gmv
            FROM raw.orders o
            JOIN raw.order_items oi USING (order_id)
            WHERE o.order_status = 'delivered'
              AND DATE(o.order_purchase_timestamp) BETWEEN DATE '2018-08-01' AND DATE '2018-08-03'
            GROUP BY DATE(o.order_purchase_timestamp)
        ), anchor_items AS (
            SELECT oi.order_id, oi.order_item_id,
                   48066.12::numeric - totals.current_gmv AS adjustment
            FROM daily_totals totals
            JOIN raw.orders o
              ON DATE(o.order_purchase_timestamp) = totals.metric_date
             AND o.order_id = 'seed_order_' || LPAD(EXTRACT(DAY FROM totals.metric_date)::integer::text, 6, '0')
            JOIN raw.order_items oi USING (order_id)
        )
        UPDATE raw.order_items oi
           SET price = oi.price + anchor_items.adjustment
          FROM anchor_items
         WHERE oi.order_id = anchor_items.order_id
           AND oi.order_item_id = anchor_items.order_item_id
        """,
        """
        UPDATE raw.order_payments payments
           SET payment_value = item_totals.gmv
          FROM (
            SELECT order_id, SUM(price + freight_value) AS gmv
            FROM raw.order_items
            GROUP BY order_id
          ) item_totals
         WHERE payments.order_id = item_totals.order_id
        """,
        "SELECT analytics.refresh_olist_marts()",
    ]
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))
        conn.execute(text("INSERT INTO test_support.seed_manifest VALUES (:version,:hash,:orders,:delivered,:seller)"), {
            "version": SEED_VERSION, "hash": SEED_HASH, "orders": TOTAL_ORDERS,
            "delivered": DELIVERED, "seller": FIXED_SELLER,
        })


def _verify(admin_url: str, database: str) -> dict[str, object]:
    engine = _engine(admin_url, database)
    try:
        with engine.connect() as conn:
            present = {str(row[0]) for row in conn.execute(text("""
              SELECT n.nspname||'.'||c.relname FROM pg_class c
              JOIN pg_namespace n ON n.oid=c.relnamespace
              WHERE c.relkind IN ('r','v','m','p')
            """))}
            present.add("alembic_version") if "public.alembic_version" in present else None
            missing = sorted(REQUIRED - present)
            version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one_or_none()
            expected_head = _head_revision()
            if version != expected_head:
                missing.append(f"Alembic version {version!r} != expected head {expected_head!r}")
            seller_columns = set(conn.execute(text("""
              SELECT column_name FROM information_schema.columns
              WHERE table_schema='raw' AND table_name='sellers'
            """)).scalars())
            if "seller_city" not in seller_columns:
                missing.append("raw.sellers.seller_city column missing")
            fixed_seller_city = conn.execute(text(
                "SELECT seller_city FROM raw.sellers WHERE seller_id=:seller"
            ), {"seller": FIXED_SELLER}).scalar_one_or_none()
            if not fixed_seller_city:
                missing.append("fixed seller has no seller_city")
            catalog_row = conn.execute(text("""
              SELECT oi.seller_id, s.seller_city, s.seller_state,
                     CASE
                       WHEN COALESCE(NULLIF(s.seller_city, ''), NULLIF(s.seller_state, '')) IS NOT NULL
                         THEN concat_ws(' · ', NULLIF(s.seller_city, ''), NULLIF(s.seller_state, ''))
                              || '｜' || LEFT(oi.seller_id, 8) || '…'
                       ELSE oi.seller_id
                     END AS label,
                     COUNT(DISTINCT oi.order_id) AS order_count
              FROM analytics.v_delivered_order_items oi
              JOIN analytics.v_delivered_orders fo ON oi.order_id=fo.order_id
              LEFT JOIN raw.sellers s ON s.seller_id=oi.seller_id
              WHERE oi.seller_id=:seller
              GROUP BY oi.seller_id, s.seller_city, s.seller_state
              ORDER BY order_count DESC, oi.seller_id ASC
              LIMIT 1
            """), {"seller": FIXED_SELLER}).mappings().one_or_none()
            if not catalog_row or not catalog_row["seller_city"] or not catalog_row["seller_state"]:
                missing.append("seller lookup cannot return city/state for fixed seller")
            elif (
                not catalog_row["seller_id"]
                or not catalog_row["label"]
                or catalog_row["seller_city"] not in catalog_row["label"]
                or FIXED_SELLER[:8] not in catalog_row["label"]
                or int(catalog_row["order_count"] or 0) <= 0
            ):
                missing.append("seller lookup cannot return a labeled seller with order_count")
            seed = conn.execute(text("SELECT seed_version,seed_hash,order_count,delivered_count,fixed_seller FROM test_support.seed_manifest")).mappings().one_or_none()
            if not seed:
                missing.append("seed manifest row missing")
            elif (seed["seed_version"], seed["seed_hash"], seed["order_count"], seed["delivered_count"], seed["fixed_seller"]) != (SEED_VERSION, SEED_HASH, TOTAL_ORDERS, DELIVERED, FIXED_SELLER):
                missing.append("seed manifest differs from deterministic recipe")
            counts = {
                "orders": conn.execute(text("SELECT COUNT(*) FROM raw.orders")).scalar_one(),
                "items": conn.execute(text("SELECT COUNT(*) FROM raw.order_items")).scalar_one(),
                "payments": conn.execute(text("SELECT COUNT(*) FROM raw.order_payments")).scalar_one(),
                "reviews": conn.execute(text("SELECT COUNT(*) FROM raw.order_reviews")).scalar_one(),
                "delivered": conn.execute(text("SELECT COUNT(*) FROM analytics.fact_orders WHERE order_status='delivered'")).scalar_one(),
                "aug_seller_items": conn.execute(text("SELECT COUNT(*) FROM analytics.fact_order_items WHERE seller_id=:seller AND purchase_date BETWEEN DATE '2018-08-01' AND DATE '2018-08-31'"), {"seller": FIXED_SELLER}).scalar_one(),
                "multi_seller_orders": conn.execute(text("SELECT COUNT(DISTINCT order_id) FROM analytics.fact_order_seller_allocation WHERE seller_count>1")).scalar_one(),
                "order_statuses": conn.execute(text("SELECT COUNT(DISTINCT order_status) FROM raw.orders")).scalar_one(),
                "synthetic_review_rows": conn.execute(text("SELECT COUNT(*) FROM raw.order_reviews WHERE review_comment_message LIKE 'Synthetic test review %' AND review_comment_title IS NULL")).scalar_one(),
                "unexpected_review_text": conn.execute(text("SELECT COUNT(*) FROM raw.order_reviews WHERE (review_comment_title IS NOT NULL OR review_comment_message IS NOT NULL) AND (review_comment_title IS NOT NULL OR review_comment_message NOT LIKE 'Synthetic test review %')")).scalar_one(),
                "seller_payment_rows": conn.execute(text("SELECT COUNT(*) FROM analytics.fact_order_payments WHERE seller_id=:seller"), {"seller": FIXED_SELLER}).scalar_one(),
            }
            gmv_consistency = conn.execute(text("""
              WITH raw_daily AS (
                SELECT DATE(o.order_purchase_timestamp) AS metric_date,
                       COUNT(DISTINCT o.order_id) FILTER (WHERE o.order_status='delivered') AS order_count,
                       COALESCE(SUM(oi.price + oi.freight_value) FILTER (WHERE o.order_status='delivered'), 0) AS gmv
                FROM raw.orders o
                LEFT JOIN raw.order_items oi USING (order_id)
                GROUP BY DATE(o.order_purchase_timestamp)
              )
              SELECT COUNT(*) FILTER (
                       WHERE ABS(dm.gmv - COALESCE(raw_daily.gmv, 0)) > 0.01
                          OR dm.order_count <> COALESCE(raw_daily.order_count, 0)
                          OR ABS(dm.avg_order_value - CASE
                                WHEN COALESCE(raw_daily.order_count, 0) = 0 THEN 0
                                ELSE ROUND(raw_daily.gmv / raw_daily.order_count, 2)
                             END) > 0.01
                     ) AS inconsistent_days,
                     COUNT(*) FILTER (
                       WHERE dm.metric_date BETWEEN DATE '2018-08-01' AND DATE '2018-08-03'
                         AND ABS(dm.gmv - 48066.12) > 0.01
                     ) AS incorrect_fixture_days,
                     COUNT(*) FILTER (
                       WHERE dm.metric_date BETWEEN DATE '2018-08-01' AND DATE '2018-08-03'
                     ) AS fixture_days,
                     COUNT(*) FILTER (
                       WHERE ABS(dm.gmv - COALESCE(category_gmv.gmv, 0)) > 0.01
                     ) AS category_inconsistent_days,
                     MAX(dm.gmv) FILTER (WHERE dm.metric_date = DATE '2018-08-04') AS aug_04_gmv,
                     SUM(dm.gmv) FILTER (WHERE dm.metric_date = DATE '2018-08-01') AS aug_01_trend_gmv,
                     SUM(category_gmv.gmv) FILTER (WHERE category_gmv.metric_date = DATE '2018-08-01') AS aug_01_category_gmv
              FROM analytics.daily_metrics dm
              LEFT JOIN raw_daily USING (metric_date)
              LEFT JOIN LATERAL (
                SELECT category_totals.metric_date, SUM(category_totals.gmv) AS gmv
                FROM (
                  SELECT DATE(o.order_purchase_timestamp) AS metric_date,
                         COALESCE(p.product_category_name, 'UNKNOWN') AS category,
                         SUM(oi.price + oi.freight_value) AS gmv
                  FROM raw.orders o
                  JOIN raw.order_items oi USING (order_id)
                  LEFT JOIN raw.products p USING (product_id)
                  WHERE o.order_status='delivered'
                  GROUP BY DATE(o.order_purchase_timestamp), COALESCE(p.product_category_name, 'UNKNOWN')
                ) category_totals
                GROUP BY category_totals.metric_date
              ) category_gmv ON category_gmv.metric_date = dm.metric_date
            """)).mappings().one()
            payment_mismatch_orders = int(conn.execute(text("""
              WITH item_totals AS (
                SELECT order_id, SUM(price + freight_value) AS gmv
                FROM raw.order_items GROUP BY order_id
              ), payment_totals AS (
                SELECT order_id, SUM(payment_value) AS payment_value
                FROM raw.order_payments GROUP BY order_id
              )
              SELECT COUNT(*)
              FROM item_totals
              FULL OUTER JOIN payment_totals USING (order_id)
              WHERE item_totals.order_id IS NULL
                 OR payment_totals.order_id IS NULL
                 OR ABS(item_totals.gmv - payment_totals.payment_value) > 0.01
            """)).scalar_one())
            if int(gmv_consistency["inconsistent_days"] or 0) != 0:
                missing.append(f"daily GMV/order-count/AOV mismatch on {gmv_consistency['inconsistent_days']} date(s)")
            if int(gmv_consistency["category_inconsistent_days"] or 0) != 0:
                missing.append(f"daily GMV/category breakdown mismatch on {gmv_consistency['category_inconsistent_days']} date(s)")
            if payment_mismatch_orders != 0:
                missing.append(f"payment amount differs from order item gross plus freight on {payment_mismatch_orders} order(s)")
            if int(gmv_consistency["fixture_days"] or 0) != 3 or int(gmv_consistency["incorrect_fixture_days"] or 0) != 0:
                missing.append("Aug 1-3 fixture GMV does not match 48066.12 in the derived daily aggregate")
            if gmv_consistency["aug_01_category_gmv"] is None or abs(float(gmv_consistency["aug_01_trend_gmv"] or 0) - float(gmv_consistency["aug_01_category_gmv"] or 0)) > 0.01:
                missing.append("Aug 1 trend GMV does not reconcile with the source-item/category GMV")
            if gmv_consistency["aug_04_gmv"] is None or float(gmv_consistency["aug_04_gmv"]) >= 48066.12:
                missing.append("Aug 4 fixture must remain below the Aug 1-3 GMV baseline")
            expectations = {"orders": TOTAL_ORDERS, "delivered": DELIVERED, "aug_seller_items": 31, "multi_seller_orders": 1, "order_statuses": 4}
            for name, expected in expectations.items():
                if counts[name] != expected:
                    missing.append(f"{name}={counts[name]!r}, expected {expected!r}")
            if counts["items"] <= counts["orders"] or counts["payments"] != counts["orders"] or counts["reviews"] != counts["orders"]:
                missing.append("raw item/payment/review cardinality checks failed")
            if counts["seller_payment_rows"] <= 0:
                missing.append("fixed seller has no allocated payment rows")
            if counts["synthetic_review_rows"] <= 0 or counts["unexpected_review_text"] != 0:
                missing.append("review text is absent or not the deterministic synthetic fixture")
            if missing:
                raise RuntimeError("seeded template verification failed: " + "; ".join(missing))
            return {
                "alembic_version": version, "expected_head": expected_head, **counts,
                "daily_gmv_inconsistent_days": int(gmv_consistency["inconsistent_days"] or 0),
                "category_gmv_inconsistent_days": int(gmv_consistency["category_inconsistent_days"] or 0),
                "payment_gmv_mismatch_orders": payment_mismatch_orders,
                "aug_01_gmv": float(gmv_consistency["aug_01_trend_gmv"] or 0),
                "aug_01_category_gmv": float(gmv_consistency["aug_01_category_gmv"] or 0),
                "aug_04_gmv": float(gmv_consistency["aug_04_gmv"] or 0),
                "seed_hash": SEED_HASH,
            }
    finally:
        engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=os.environ.get("BIZINSIGHT_TEST_SEEDED_TEMPLATE_DB", DEFAULT_DATABASE))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--recreate", action="store_true", help="build schema-only template and deterministic synthetic seed")
    mode.add_argument("--verify", action="store_true", help="verify without modifying the database")
    args = parser.parse_args()
    admin_url = _admin_url()
    _validate_database_name(admin_url, args.database)
    if args.recreate:
        _run_schema_builder(admin_url, args.database)
        engine = _engine(admin_url, args.database)
        try:
            _seed(engine)
        finally:
            engine.dispose()
    try:
        result = _verify(admin_url, args.database)
    except Exception as exc:
        print(f"SEEDED_TEMPLATE_VERIFY_FAILED database={args.database}: {exc}", file=sys.stderr)
        return 1
    print("SEEDED_TEMPLATE_OK " + " ".join(f"{key}={value}" for key, value in result.items()))
    print(f"DATA_PROVENANCE=deterministic synthetic test data; RECIPE={SEED_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

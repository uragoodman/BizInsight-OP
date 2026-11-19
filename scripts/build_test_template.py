"""Create/verify a schema-only PostgreSQL template for isolated test shards.

Requires ISOLATED_TEST_POSTGRES_ADMIN_URL (a maintenance URL such as /postgres).
The application .env URL is never used as a target; the target is always a
separate codex_test_template_* database with empty raw tables.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
RAW_BOOTSTRAP = """
CREATE SCHEMA IF NOT EXISTS raw;
CREATE TABLE IF NOT EXISTS raw.customers (customer_id text PRIMARY KEY, customer_unique_id text, customer_city text, customer_state text);
CREATE TABLE IF NOT EXISTS raw.orders (
  order_id text PRIMARY KEY, customer_id text, order_status text,
  order_purchase_timestamp timestamp, order_approved_at timestamp,
  order_delivered_carrier_date timestamp, order_delivered_customer_date timestamp,
  order_estimated_delivery_date timestamp
);
CREATE TABLE IF NOT EXISTS raw.order_items (
  order_id text, order_item_id integer, product_id text, seller_id text,
  shipping_limit_date timestamp, price numeric, freight_value numeric
);
CREATE TABLE IF NOT EXISTS raw.order_payments (
  order_id text, payment_sequential integer, payment_type text,
  payment_installments integer, payment_value numeric
);
CREATE TABLE IF NOT EXISTS raw.order_reviews (
  review_id text, order_id text, review_score integer, review_comment_title text,
  review_comment_message text, review_creation_date timestamp, review_answer_timestamp timestamp
);
CREATE TABLE IF NOT EXISTS raw.sellers (
  seller_id text PRIMARY KEY, seller_city text, seller_state text
);
CREATE TABLE IF NOT EXISTS raw.products (product_id text PRIMARY KEY, product_category_name text);
CREATE TABLE IF NOT EXISTS raw.product_category_name_translation (
  product_category_name text PRIMARY KEY, product_category_name_english text
);
"""
REQUIRED_RELATIONS = {
    "alembic_version", "analytics.fact_orders", "analytics.fact_order_items",
    "analytics.v_product_category", "analytics.v_order_cohort_funnel",
    "analytics.v_order_unified", "iam.tenant_seller_accounts",
}


def _safe_target(admin_url: str, database: str) -> None:
    if not database.startswith("codex_test_template_"):
        raise SystemExit("Template target must start with codex_test_template_")
    if not database.replace("_", "").isalnum():
        raise SystemExit("Template database name contains invalid characters")
    app_url = dotenv_values(ROOT / ".env").get("POSTGRES_URL")
    protected = {"bizinsight"}
    if app_url:
        protected.add(make_url(str(app_url)).database.lower())
    if database.lower() in protected:
        raise SystemExit("Refusing .env/bizinsight as a test template target")
    admin_db = make_url(admin_url).database
    if database.lower() == str(admin_db).lower():
        raise SystemExit("Template target must differ from maintenance database")


def _admin_engine(admin_url: str):
    return create_engine(make_url(admin_url).set(database="postgres"), isolation_level="AUTOCOMMIT", pool_pre_ping=True)


def _exists(engine, database: str) -> bool:
    with engine.connect() as conn:
        return bool(conn.execute(text("SELECT 1 FROM pg_database WHERE datname=:db"), {"db": database}).scalar())


def _drop(engine, database: str) -> None:
    with engine.connect() as conn:
        conn.execute(text("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=:db AND pid <> pg_backend_pid()"), {"db": database})
        conn.exec_driver_sql(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)')


def _verify(admin_url: str, database: str) -> list[str]:
    target_url = make_url(admin_url).set(database=database)
    engine = create_engine(target_url, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            present = {row[0] for row in conn.execute(text("""
              SELECT n.nspname || '.' || c.relname
              FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
              WHERE c.relkind IN ('r','v','m','p')
            """))}
            if "public.alembic_version" in present:
                present.add("alembic_version")
            missing = sorted(REQUIRED_RELATIONS - present)
            raw_tables = conn.execute(text("""
              SELECT quote_ident(n.nspname)||'.'||quote_ident(c.relname)
              FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
              WHERE n.nspname='raw' AND c.relkind IN ('r','p')
            """)).scalars().all()
            for relation in raw_tables:
                row_count = conn.exec_driver_sql(f"SELECT COUNT(*) FROM {relation}").scalar_one()
                if row_count:
                    missing.append(f"{relation} must be empty (found business rows)")
            version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one_or_none()
            if not version:
                missing.append("alembic_version is empty")
            return missing
    finally:
        engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=os.environ.get("BIZINSIGHT_TEST_TEMPLATE_DB", "codex_test_template_local"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--recreate", action="store_true", help="drop and rebuild this isolated schema-only database")
    mode.add_argument("--verify", action="store_true", help="verify without modifying the database")
    args = parser.parse_args()
    admin_url = os.environ.get("ISOLATED_TEST_POSTGRES_ADMIN_URL", "").strip()
    if not admin_url:
        parser.error("set ISOLATED_TEST_POSTGRES_ADMIN_URL to an isolated-test PostgreSQL maintenance URL")
    _safe_target(admin_url, args.database)
    admin = _admin_engine(admin_url)
    try:
        exists = _exists(admin, args.database)
        if args.verify:
            if not exists:
                raise SystemExit(f"template database {args.database!r} does not exist")
            missing = _verify(admin_url, args.database)
            if missing:
                raise SystemExit("Template verification failed: " + "; ".join(missing))
            print(f"TEMPLATE_VERIFIED={args.database} BUSINESS_ROWS=0")
            return 0
        if exists:
            _drop(admin, args.database)
        with admin.connect() as conn:
            conn.exec_driver_sql(f'CREATE DATABASE "{args.database}"')
    finally:
        admin.dispose()

    target_url = make_url(admin_url).set(database=args.database).render_as_string(hide_password=False)
    target = create_engine(target_url)
    try:
        with target.begin() as conn:
            for statement in RAW_BOOTSTRAP.strip().split(";"):
                if statement.strip():
                    conn.exec_driver_sql(statement)
    finally:
        target.dispose()

    env = dict(os.environ)
    env["POSTGRES_URL"] = target_url
    env["BIZINSIGHT_TEST_RUNNER"] = "1"
    env["BIZINSIGHT_TEST_SHARD"] = "postgres"
    completed = subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, env=env)
    if completed.returncode:
        print(f"TEMPLATE_BUILD_FAILED database={args.database}; database retained for inspection", file=sys.stderr)
        return completed.returncode
    missing = _verify(admin_url, args.database)
    if missing:
        print("TEMPLATE_SCHEMA_INVALID=" + "; ".join(missing), file=sys.stderr)
        return 1
    print(f"TEMPLATE_BUILT={args.database} BUSINESS_ROWS=0; treat as immutable and use only as CREATE DATABASE TEMPLATE source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

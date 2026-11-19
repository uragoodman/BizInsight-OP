"""Run one pytest shard against an explicit disposable database.

Examples:
  python scripts/run_test_shard.py --shard no-db --name p2 -- tests/test_semantic_compiler_p2.py
  python scripts/run_test_shard.py --shard postgres --name lifecycle --template codex_test_template -- tests/test_seller_lifecycle_postgres.py
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
DATABASE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


def _protected_databases() -> set[str]:
    raw = dotenv_values(ROOT / ".env").get("POSTGRES_URL")
    return {"bizinsight"} | ({make_url(str(raw)).database.lower()} if raw else set())


def _check_database_name(database: str | None, *, allow_template: bool = False) -> None:
    if not database or not DATABASE_NAME_RE.fullmatch(database):
        raise SystemExit("Database name must be a simple PostgreSQL identifier")
    if database.lower() in _protected_databases():
        raise SystemExit("Refusing the developer .env/bizinsight database")
    prefixes = ("codex_test_template",) if allow_template else ("codex_test_",)
    if not database.startswith(prefixes):
        raise SystemExit(f"Database name must start with {prefixes}")


def _admin_engine(admin_url: str):
    return create_engine(make_url(admin_url).set(database="postgres"), isolation_level="AUTOCOMMIT", pool_pre_ping=True)


def _template_activity(engine, template: str) -> list[dict[str, object]]:
    with engine.connect() as conn:
        return [dict(row) for row in conn.execute(text("""
            SELECT pid, usename, application_name, state
            FROM pg_stat_activity WHERE datname=:db AND pid <> pg_backend_pid()
            ORDER BY pid
        """), {"db": template}).mappings()]


def _create_database(admin_url: str, clone: str, template: str | None) -> None:
    engine = _admin_engine(admin_url)
    try:
        if template:
            _check_database_name(template, allow_template=True)
            activity = _template_activity(engine, template)
            if activity:
                raise RuntimeError(f"Template {template!r} has active connections: {activity}")
            with engine.connect() as conn:
                exists = conn.execute(text("SELECT 1 FROM pg_database WHERE datname=:db"), {"db": template}).scalar()
                if not exists:
                    raise RuntimeError(f"Template database {template!r} does not exist")
                conn.exec_driver_sql(f'CREATE DATABASE "{clone}" TEMPLATE "{template}"')
        else:
            with engine.connect() as conn:
                conn.exec_driver_sql(f'CREATE DATABASE "{clone}"')
    finally:
        engine.dispose()


def _drop_clone(admin_url: str, clone: str) -> None:
    _check_database_name(clone)
    engine = _admin_engine(admin_url)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=:db AND pid <> pg_backend_pid()"), {"db": clone})
            conn.exec_driver_sql(f'DROP DATABASE IF EXISTS "{clone}" WITH (FORCE)')
    finally:
        engine.dispose()


def _database_exists(admin_url: str, database: str) -> bool:
    engine = _admin_engine(admin_url)
    try:
        with engine.connect() as conn:
            return bool(conn.execute(text("SELECT 1 FROM pg_database WHERE datname=:db"), {"db": database}).scalar())
    finally:
        engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", choices=("no-db", "postgres"), required=True)
    parser.add_argument("--name", required=True, help="short shard label")
    parser.add_argument("--template", default=os.environ.get("BIZINSIGHT_TEST_TEMPLATE_DB"))
    parser.add_argument("--keep", action="store_true", help="keep the disposable clone for debugging")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    pytest_args = list(args.pytest_args)
    if pytest_args and pytest_args[0] == "--":
        pytest_args.pop(0)
    if not pytest_args:
        parser.error("provide pytest paths/options after --")

    env = dict(os.environ)
    env["BIZINSIGHT_TEST_RUNNER"] = "1"
    env["BIZINSIGHT_TEST_SHARD"] = args.shard
    # Test shards must never inherit active defaults or provider raw-text
    # permission from the developer's shell or .env file.
    env["COMPLEX_ROUTE_MODE"] = "off"
    env["SEMANTIC_INTERPRETER_MODE"] = "off"
    env["COMPLEX_ROUTE_CANARY_ENABLED"] = "false"
    env["SEMANTIC_INTERPRETER_SEND_RAW_QUESTION"] = "false"
    label = re.sub(r"[^a-z0-9_]+", "_", args.name.lower()).strip("_") or "shard"
    label = label[:32]
    admin_url = env.get("ISOLATED_TEST_POSTGRES_ADMIN_URL", "")
    nonce = uuid.uuid4().hex[:12]
    clone = f"codex_test_{label}_{nonce}" if args.shard == "postgres" else f"codex_test_no_db_{label}_{nonce}"
    template_database = str(args.template or "").strip() or None
    kept = False
    started = time.perf_counter()
    try:
        if args.shard == "no-db":
            env["POSTGRES_URL"] = f"postgresql+psycopg://codex_test:codex_test@127.0.0.1:9/{clone}"
            # no-db also blocks enrollment and provider use at every gate.
            env["COMPLEX_ROUTE_CANARY_TENANTS"] = "[]"
            env["COMPLEX_ROUTE_CANARY_TEMPLATES"] = "[]"
            env["COMPLEX_ROUTE_CANARY_SAMPLE_RATE"] = "0"
            for key in (
                "ISOLATED_TEST_POSTGRES_ADMIN_URL", "WP4_TEST_POSTGRES_URL",
                "WP3_REGISTRY_TEST_DATABASE_URL", "GLOBAL_ACCOUNT_ERASURE_TEST_URL",
                "WP2_TEST_POSTGRES_TEMPLATE",
            ):
                env.pop(key, None)
        else:
            if not admin_url:
                parser.error("PostgreSQL shard requires ISOLATED_TEST_POSTGRES_ADMIN_URL")
            _check_database_name(clone)
            _create_database(admin_url, clone, template_database)
            env["POSTGRES_URL"] = make_url(admin_url).set(database=clone).render_as_string(hide_password=False)
            # Legacy integration tests used bespoke env names. Bind those names
            # to the same runner-owned disposable database, never a stale .env URL.
            env["WP4_TEST_POSTGRES_URL"] = env["POSTGRES_URL"]
            env["WP3_REGISTRY_TEST_DATABASE_URL"] = env["POSTGRES_URL"]
            env["GLOBAL_ACCOUNT_ERASURE_TEST_URL"] = env["POSTGRES_URL"]
            env["BIZINSIGHT_TEST_SHARD_DATABASE"] = clone
            if template_database:
                env["WP2_TEST_POSTGRES_TEMPLATE"] = template_database
            else:
                env.pop("WP2_TEST_POSTGRES_TEMPLATE", None)
        return_code = subprocess.call([sys.executable, "-m", "pytest", *pytest_args], cwd=ROOT, env=env)
        return return_code
    except Exception as exc:
        print(f"SHARD_SETUP_OR_CLEANUP_ERROR={type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        if args.shard == "postgres" and admin_url:
            try:
                if args.keep:
                    kept = _database_exists(admin_url, clone)
                else:
                    _drop_clone(admin_url, clone)
            except Exception as exc:
                print(f"CLONE_CLEANUP_ERROR database={clone!r}: {exc}", file=sys.stderr)
                try:
                    kept = _database_exists(admin_url, clone)
                except Exception:
                    kept = True
        elapsed = time.perf_counter() - started
        print(f"SHARD={args.name} ELAPSED_SECONDS={elapsed:.2f}")
        if args.shard == "postgres":
            residual = kept
            if admin_url and not args.keep:
                try:
                    residual = _database_exists(admin_url, clone)
                except Exception:
                    residual = True
            print(f"CLONE_DATABASE={clone} TEMPLATE_DATABASE={template_database or 'none'} RESIDUAL={str(residual).lower()} KEPT={str(args.keep).lower()}")


if __name__ == "__main__":
    raise SystemExit(main())

"""Run read-only EXPLAIN ANALYZE on the isolated seller acceptance fixture.

The script deliberately reuses the fixture from
``tests/test_seller_metric_value_acceptance.py`` so it never targets an
application database. It requires the caller to provide the isolated test
maintenance URL in ``ISOLATED_TEST_POSTGRES_ADMIN_URL`` and only accepts the
loopback test endpoint on port 15432.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import make_url


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests" / "test_seller_metric_value_acceptance.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_fixture_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_seller_metric_value_acceptance_fixture", FIXTURE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load isolated fixture at {FIXTURE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _assert_isolated_url() -> None:
    value = os.getenv("ISOLATED_TEST_POSTGRES_ADMIN_URL")
    if not value:
        raise SystemExit("Set ISOLATED_TEST_POSTGRES_ADMIN_URL to the isolated test maintenance URL")
    url = make_url(value)
    if url.host not in {"127.0.0.1", "localhost", "::1"} or url.port != 15432:
        raise SystemExit("Refusing EXPLAIN: only the loopback isolated test PostgreSQL on port 15432 is allowed")
    if url.database not in {"postgres", "template1"}:
        raise SystemExit("Refusing EXPLAIN: the isolated maintenance URL must target postgres or template1")


def _plan_summary(raw_plan: Any) -> dict[str, object]:
    if isinstance(raw_plan, list):
        raw_plan = raw_plan[0]
    root = raw_plan.get("Plan", {})
    nodes: list[dict[str, object]] = []

    def visit(node: dict[str, Any]) -> None:
        nodes.append({
            "node": node.get("Node Type"),
            "relation": node.get("Relation Name"),
            "alias": node.get("Alias"),
            "index": node.get("Index Name"),
            "plan_rows": node.get("Plan Rows"),
            "actual_rows": node.get("Actual Rows"),
            "rows_removed_by_filter": node.get("Rows Removed by Filter"),
            "filter": node.get("Filter"),
            "index_condition": node.get("Index Cond"),
            "loops": node.get("Actual Loops"),
            "shared_hit_blocks": node.get("Shared Hit Blocks"),
            "shared_read_blocks": node.get("Shared Read Blocks"),
            "actual_total_time_ms": node.get("Actual Total Time"),
        })
        for child in node.get("Plans", []):
            visit(child)

    visit(root)
    return {
        "planning_time_ms": raw_plan.get("Planning Time"),
        "execution_time_ms": raw_plan.get("Execution Time"),
        "root_shared_hit_blocks": root.get("Shared Hit Blocks"),
        "root_shared_read_blocks": root.get("Shared Read Blocks"),
        "nodes": nodes,
    }


def main() -> int:
    _assert_isolated_url()
    fixture = _load_fixture_module()
    fixture_gen = fixture.isolated_seller_db.__wrapped__()
    engine = None
    try:
        engine, captured = next(fixture_gen)

        import app.core.config as app_config
        from app.auth.context import build_authorization_context
        from app.data.repositories import explorer
        from app.scope.seller_scope import SellerScopeIntent, resolve_seller_scope
        from app.tenancy import SellerGrant, TenantContext, activate_tenant_context

        # The fixture's database is isolated and empty until its exact seed
        # function runs. Settings only open the seller query feature path.
        from app.core.config import Settings
        original_get_settings = app_config.get_settings
        app_config.get_settings = lambda: Settings(
            seller_dimension_enabled=True, seller_multi_scope_enabled=True,
        )
        original_engine = explorer.engine
        explorer.engine = engine

        tenant = TenantContext(
            tenant_id="tenant-platform-breakdown",
            user_id="user-platform-breakdown",
            role="admin", scope_type="platform",
            seller_grants=tuple(
                SellerGrant(seller, "2024-01-01", "2024-01-31")
                for seller in ("A", "B", "C")
            ),
        )
        authz = build_authorization_context(tenant, metric_scope_type="platform")

        def run_breakdown(scope_type: str, seller_ids: tuple[str, ...]) -> None:
            scope = resolve_seller_scope(
                SellerScopeIntent(
                    seller_scope_type=scope_type,
                    seller_ids=seller_ids,
                    start_date=fixture.BREAKDOWN_DATE,
                    end_date=fixture.BREAKDOWN_DATE,
                    metric_ids=("gmv", "order_count"),
                    dimensions=("seller",),
                ),
                tenant=tenant,
                authz=authz,
            )
            with activate_tenant_context(tenant):
                explorer.breakdown_by_seller(scope, "gmv", limit=10)

        run_breakdown("selected", ("A", "B"))
        run_breakdown("all", ())

        matching: dict[str, tuple[str, object]] = {}
        for statement, parameters in captured:
            sql = str(statement)
            if "GROUP BY oi.seller_id" in sql:
                matching.setdefault("seller_items_top_n", (sql, parameters))
            elif "AS row_count" in sql and "oi.seller_id IS NULL" in sql:
                matching.setdefault("unattributed", (sql, parameters))
            elif "AS gmv" in sql and "COUNT(DISTINCT oi.order_id)" in sql:
                if "oi.seller_id = ANY" in sql:
                    matching.setdefault("selected_total", (sql, parameters))
                elif "oi.seller_id IS NOT NULL" in sql:
                    matching.setdefault("all_known_total", (sql, parameters))
        required = {"seller_items_top_n", "selected_total", "all_known_total", "unattributed"}
        missing = required - set(matching)
        if missing:
            raise RuntimeError(f"Fixture did not capture expected repository SQL: {sorted(missing)}")

        plans: dict[str, object] = {}
        bindings: dict[str, dict[str, object]] = {}
        with engine.connect() as connection:
            for label in sorted(matching):
                sql, parameters = matching[label]
                bindings[label] = {
                    str(key): (str(value) if key in {"start_date", "end_date"} else value)
                    for key, value in (parameters.items() if isinstance(parameters, dict) else [])
                }
                result = connection.exec_driver_sql(
                    "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql,
                    parameters,
                ).scalar_one()
                plans[label] = _plan_summary(result)

        print(json.dumps({
            "fixture": "tests/test_seller_metric_value_acceptance.py::isolated_seller_db",
            "database_target": "loopback isolated PostgreSQL, unique temporary database",
            "seed_rows": 15,
            "query_bindings": bindings,
            "query_plans": plans,
        }, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    finally:
        if engine is not None:
            try:
                next(fixture_gen)
            except StopIteration:
                pass
        else:
            try:
                next(fixture_gen)
            except StopIteration:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

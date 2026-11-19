"""Reproducible WP2/WP1 integration verification against an isolated PostgreSQL.

Required environment:
  ISOLATED_TEST_POSTGRES_ADMIN_URL  maintenance URL on a disposable PostgreSQL
  WP2_TEST_POSTGRES_TEMPLATE       template database with raw/analytics data
                                   (default: bizinsight)

The script clones the template, migrates the clone to Alembic head, seeds only
the identity/data-version fixtures needed for the WP2 path, then exercises:

  real task table -> tasks.service.relation -> context compiler
  -> real TriageContext/ContextBundle -> real workflow persistence

Only the LLM/data-producing leaf nodes are deterministic fakes
(metric_definition, generate_answer, critic, memory nodes); the task table,
relation resolver, explorer context, compiler, orchestrator, TurnResolution,
trace and chat_turns persistence are real.

The clone and trace directory are removed on exit.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    admin_text = os.getenv("ISOLATED_TEST_POSTGRES_ADMIN_URL", "").strip()
    template = os.getenv("WP2_TEST_POSTGRES_TEMPLATE", "bizinsight").strip()
    if not admin_text:
        print("WP2_REAL_CHAIN_SKIPPED: ISOLATED_TEST_POSTGRES_ADMIN_URL is not set")
        return 2
    admin_url = make_url(admin_text)
    admin = create_engine(
        admin_url.set(database="postgres"),
        isolation_level="AUTOCOMMIT",
        pool_pre_ping=True,
    )
    database = f"wp2_real_chain_{uuid.uuid4().hex[:10]}"
    trace_dir = Path(tempfile.mkdtemp(prefix="wp2_real_chain_traces_"))
    try:
        with admin.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname=:name"), {"name": template},
            ).scalar()
            if not exists:
                print(f"WP2_REAL_CHAIN_SKIPPED: template database {template!r} does not exist")
                return 2
            conn.execute(text(f'CREATE DATABASE "{database}" TEMPLATE "{template}"'))

        db_url = admin_url.set(database=database)
        env = {
            **os.environ,
            "POSTGRES_URL": db_url.render_as_string(hide_password=False),
            "DATA_VERSIONING_ENABLED": "true",
            "TENANT_AUTH_REQUIRED": "false",
            "ABAC_DATABASE_ENABLED": "false",
            "RELATION_RESOLVER_LLM_ENABLED": "false",
            "TRACE_BACKEND": "file",
            "TRACE_DIR": str(trace_dir),
            "CHECKPOINT_BACKEND": "sqlite",
            "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        }
        upgrade = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=300,
        )
        if upgrade.returncode != 0:
            print(upgrade.stdout[-4000:])
            print(upgrade.stderr[-4000:])
            raise RuntimeError("alembic upgrade failed on the isolated clone")
        return _run_verification(db_url, env, trace_dir)
    finally:
        try:
            with admin.connect() as conn:
                conn.execute(text(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)'))
        except Exception:
            traceback.print_exc()
        shutil.rmtree(trace_dir, ignore_errors=True)


def _seed_fixtures(db_url) -> None:
    engine = create_engine(db_url, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO iam.tenants(tenant_id, tenant_name, scope_type, status)
            VALUES ('platform', 'wp2 real chain', 'platform', 'active')
            ON CONFLICT (tenant_id) DO NOTHING
        """))
        conn.execute(text("""
            INSERT INTO iam.users(user_id, display_name, email, status)
            VALUES ('wp2-real-chain', 'wp2 real chain', 'wp2-real-chain@example.invalid', 'active')
            ON CONFLICT (user_id) DO NOTHING
        """))
        conn.execute(text("""
            INSERT INTO iam.tenant_memberships(tenant_id, user_id, role)
            VALUES ('platform', 'wp2-real-chain', 'admin')
            ON CONFLICT (tenant_id, user_id) DO NOTHING
        """))
        source = conn.execute(text("""
            SELECT tenant_id, min(seller_id)
              FROM analytics.tenant_fact_order_items
             GROUP BY tenant_id
             ORDER BY count(*) DESC, tenant_id
             LIMIT 1
        """)).first()
        has_any_version = conn.execute(text("""
            SELECT 1 FROM governance.data_versions
            WHERE dataset_id='tenant_gmv_items' AND status='sealed'
            LIMIT 1
        """)).scalar()

    # CI's Phase 6/7 seeds create fact rows, but deliberately do not seal a
    # version.  Create the smallest real, tenant-scoped source snapshot needed
    # by resolve_gmv_manifest before copying its manifest metadata to platform.
    if not has_any_version:
        if source is None:
            engine.dispose()
            raise RuntimeError("WP2 fixture requires seeded tenant_fact_order_items")
        os.environ["POSTGRES_URL"] = db_url.render_as_string(hide_password=False)
        from app.data_versions.service import seal_gmv_version
        from app.tenancy import SELLER_SCOPE, TenantContext

        source_tenant, source_seller = str(source[0]), str(source[1])
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO iam.tenant_memberships(tenant_id, user_id, role)
                VALUES (:tenant, 'wp2-real-chain', 'admin')
                ON CONFLICT (tenant_id, user_id) DO NOTHING
            """), {"tenant": source_tenant})
        seal_gmv_version(TenantContext(
            tenant_id=source_tenant, user_id="wp2-real-chain", role="admin",
            scope_type=SELLER_SCOPE, seller_id=source_seller,
        ))

    with engine.begin() as conn:
        has_platform_version = conn.execute(text("""
            SELECT 1 FROM governance.data_versions
            WHERE tenant_id='platform' AND dataset_id='tenant_gmv_items' AND status='sealed'
            LIMIT 1
        """)).scalar()
        if not has_platform_version:
            conn.execute(text("""
                INSERT INTO governance.data_versions
                    (version_id, tenant_id, dataset_id, status, content_hash, schema_hash,
                     row_count, min_business_date, max_business_date, affected_sellers,
                     storage_ref, parent_version_id, sealed_at, created_at)
                SELECT 'wp2_chain_gmv', 'platform', dataset_id, 'sealed', content_hash,
                       schema_hash, row_count, min_business_date, max_business_date,
                       affected_sellers, storage_ref, NULL, now(), now()
                FROM governance.data_versions
                WHERE dataset_id='tenant_gmv_items' AND status='sealed'
                ORDER BY sealed_at DESC LIMIT 1
            """))
    engine.dispose()


def _run_verification(db_url, env, trace_dir: Path) -> int:
    _seed_fixtures(db_url)
    os.environ.update({
        "POSTGRES_URL": env["POSTGRES_URL"],
        "DATA_VERSIONING_ENABLED": "true",
        "TENANT_AUTH_REQUIRED": "false",
        "ABAC_DATABASE_ENABLED": "false",
        "RELATION_RESOLVER_LLM_ENABLED": "false",
        "TRACE_BACKEND": "file",
        "TRACE_DIR": str(trace_dir),
        "CHECKPOINT_BACKEND": "sqlite",
    })

    from sqlalchemy import create_engine as create
    from app.tenancy import TenantContext
    from app.schemas.explorer import ExplorerQuery
    from app.data.repositories.explorer import preview_explorer, controlled_scope
    from app.metrics.authorization import scope_fingerprint
    from app.data_versions.service import resolve_gmv_manifest
    from app.auth.context import build_authorization_context
    from app.scope.canonical import CanonicalScope
    from app.tasks.service import create_task, relation
    from app.graph import workflow as wf
    from app.core.config import get_settings

    # Imports above can populate the cached Settings object through app-level
    # module initialization.  Re-read the explicit isolated-test overrides so
    # trace output cannot silently fall back to the developer's .env path.
    get_settings.cache_clear()
    assert get_settings().trace_backend == "file", get_settings().trace_backend
    assert Path(get_settings().trace_dir) == trace_dir, get_settings().trace_dir

    engine = create(db_url)
    with engine.connect() as conn:
        seller, min_date, max_date = conn.execute(text("""
            SELECT seller_id, MIN(purchase_date), MAX(purchase_date)
            FROM analytics.v_delivered_order_items
            GROUP BY seller_id ORDER BY COUNT(*) DESC LIMIT 1
        """)).one()

    context = TenantContext(
        tenant_id="platform", user_id="wp2-real-chain",
        role="admin", scope_type="platform",
    )
    payload = {
        "seller_id": seller, "metric_name": "gmv",
        "start_date": str(min_date), "end_date": str(max_date), "granularity": "day",
    }
    query = ExplorerQuery.model_validate(payload)
    snapshot = preview_explorer(context, query)
    scope = controlled_scope(snapshot)
    scope_data = scope.model_dump(mode="json")
    manifest = resolve_gmv_manifest(context, scope_data, scope_fingerprint(scope_data))
    authz = build_authorization_context(
        context, allowed_metric_ids=scope.allowed_metrics,
        allowed_dimensions=scope.allowed_dimensions, metric_scope_type="seller",
    )
    canonical = CanonicalScope.resolve(
        authz, seller_ids=[scope.seller_id],
        start_date=scope.default_start_date, end_date=scope.default_end_date,
        metric_ids=scope.allowed_metrics, dimensions=scope.allowed_dimensions,
        granularity=scope.granularity,
    )
    boundary = {
        "permission_version": authz.permission_version,
        "policy_set_version": authz.policy_set_version,
        "authorization_fingerprint": authz.authz_digest,
        "query_scope_fingerprint": canonical.query_scope_fingerprint,
        "authz_ref": authz.authz_ref,
    }
    suffix = uuid.uuid4().hex[:8]
    session = f"wp2-real-chain-session-{suffix}"
    task_a = create_task(context, session, "GMV 分析", payload)["task_id"]
    task_b = create_task(context, session, "退款分析", payload)["task_id"]
    task_state = {
        "scope_fingerprint": canonical.query_scope_fingerprint,
        "data_version_manifest_hash": manifest["manifest_hash"],
        "latest_turn_id": "wp2-seed-turn",
        "context_slots": {
            "metric": {
                "value": "gmv", "source": "current_turn",
                "source_turn_id": "wp2-seed-turn",
            },
            "date": {
                "value": f"{scope.default_start_date} 至 {scope.default_end_date}",
                "source": "current_turn", "source_turn_id": "wp2-seed-turn",
            },
        },
    }
    with engine.begin() as conn:
        for task_id in (task_a, task_b):
            conn.execute(text("""
                UPDATE governance.conversation_tasks
                   SET task_state=CAST(:state AS jsonb), task_state_version=3, status='running'
                 WHERE task_id=:task
            """), {"state": json.dumps(task_state, ensure_ascii=False), "task": task_id})

    multiple = relation(
        context, session, "继续", payload, None,
        scope_fingerprint=canonical.query_scope_fingerprint,
        expected_data_version=manifest["manifest_hash"],
    )
    assert multiple["action"] == "clarify", multiple
    assert multiple["reason"] == "multiple_active_tasks", multiple
    selected = relation(
        context, session, "继续看一下这个指标的趋势", payload, task_a,
        scope_fingerprint=canonical.query_scope_fingerprint,
        expected_data_version=manifest["manifest_hash"],
    )
    assert selected["action"] == "continue", selected
    selected_date = selected["context_resolution"]["known_slots"]["date"]
    assert selected_date["source"] == "task_state", selected_date
    assert selected_date["source_turn_id"] == "wp2-seed-turn", selected_date
    assert selected_date["source_task_state_version"] == 3, selected_date

    # Keep the real resolver/compiler/orchestrator/triage/persistence path.
    def fake_metric(state):
        state["metric_context"] = "GMV definition"
        state.setdefault("node_history", []).append({"node": "metric_definition", "summary": "e2e fake"})
        return state

    def fake_answer(state):
        state["answer"] = "GMV is a metric."
        state.setdefault("node_history", []).append({"node": "generate_answer", "summary": "e2e fake"})
        return state

    def fake_critic(state, **kwargs):
        state["critic_verdict"] = "approved"
        state.setdefault("critic_decisions", []).append({"llm_issues": [], "revision_target": None})
        state.setdefault("node_history", []).append({"node": "critic", "summary": "e2e fake"})
        return state

    def fake_memory(state, **kwargs):
        state["memory_session_loaded"] = True
        return state

    def fake_retrieve(state):
        return state

    def fake_persist_memory(state):
        state["memory_write_status"] = "skipped_wp2_real_chain"
        return state

    persistence_turn_id = f"wp2-real-turn-{suffix}"
    trace_id = f"wp2-real-trace-{suffix}"
    with patch_triage_classifier(), \
         patch_target(wf, "metric_definition", fake_metric), \
         patch_target(wf, "generate_answer", fake_answer), \
         patch_target(wf, "critic", fake_critic), \
         patch_target(wf, "load_session_memory", fake_memory), \
         patch_target(wf, "retrieve_analysis_memories", fake_retrieve), \
         patch_target(wf, "persist_memory_p0", fake_persist_memory):
        result = wf.run_chat_workflow(
            "继续看一下这个指标的趋势", f"wp2-real-thread-{suffix}", context, payload,
            True, None, False, None, session, f"client-{suffix}", task_a,
            persistence_turn_id=persistence_turn_id,
            authorization_boundary=boundary, trace_id=trace_id,
            turn_relation_decision=selected,
        )
    assert result.get("answer") == "GMV is a metric.", result

    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT task_id, status, context_snapshot
            FROM governance.chat_turns
            WHERE turn_id=:turn
        """), {"turn": persistence_turn_id}).mappings().one_or_none()
    assert row is not None, "chat_turns row was not persisted"
    assert row["status"] == "completed", row
    assert row["task_id"] == task_a, row
    resolution = (row["context_snapshot"] or {}).get("turn_resolution") or {}
    assert resolution["turn_id"] == persistence_turn_id, resolution
    assert resolution["client_turn_id"] == f"client-{suffix}", resolution
    assert resolution["session_id"] == session, resolution
    assert resolution["task_id"] == task_a, resolution
    assert resolution["known_slots"]["date"]["source"] == "task_state", resolution
    assert any(
        item.get("field") == "date" and item.get("source_turn_id") == "wp2-seed-turn"
        for item in resolution["inherited_fields"]
    ), resolution

    trace_files = list(trace_dir.glob("*.json"))
    assert trace_files, "trace file was not written"
    trace = json.loads(trace_files[0].read_text(encoding="utf-8"))
    assert "context_compiler" in trace["graph_path"], trace["graph_path"]
    assert "triage" in trace["graph_path"], trace["graph_path"]
    triage_bundle = trace["state_snapshot"]["triage_context"]["bundle"]
    assert triage_bundle["agent"] == "triage"
    assert triage_bundle["turn_id"] == persistence_turn_id, triage_bundle
    assert triage_bundle["resolved_request"]["turn_id"] == persistence_turn_id, triage_bundle
    assert triage_bundle["resolved_request"]["client_turn_id"] == f"client-{suffix}", triage_bundle
    assert triage_bundle["task_id"] == task_a, triage_bundle
    assert trace["state_snapshot"]["turn_resolution"]["task_id"] == task_a
    print("WP2_REAL_CHAIN_OK")
    print(json.dumps({
        "task_a": task_a,
        "task_b": task_b,
        "session": session,
        "persistence_turn_id": persistence_turn_id,
        "bundle_turn_id": triage_bundle["turn_id"],
        "bundle_client_turn_id": triage_bundle["resolved_request"]["client_turn_id"],
        "trace_id": trace_id,
        "relation_multiple": multiple["action"],
        "relation_selected": selected["action"],
        "inherited_date": selected_date,
        "graph_path": trace["graph_path"],
    }, ensure_ascii=False))
    return 0


def patch_target(module, name, replacement):
    from unittest.mock import patch

    return patch.object(module, name, side_effect=replacement)


def patch_triage_classifier():
    from unittest.mock import patch

    return patch("app.graph.nodes.triage.llm_classify", lambda *a, **k: "metric_definition")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

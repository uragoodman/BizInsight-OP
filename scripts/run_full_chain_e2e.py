"""Run browser-to-API-to-PostgreSQL workflows on an isolated seeded clone.

This runner deliberately does not reuse web/tests/e2e/global-setup.ts: that
legacy setup seeds whichever POSTGRES_URL it receives. Here the only app URL is
a unique codex_test_* clone, and the clone is dropped in a finally block.
Real provider usage is required and verified from governance.llm_usage_events;
no LLM adapter or workflow node is mocked.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from dotenv import dotenv_values
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
TEMPLATE_DEFAULT = "codex_test_template_seeded_local"
FIXED_SELLER = "6560211a19b47992c3666cc44a7e94c0"
ALLOWED_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}

sys.path.insert(0, str(ROOT))
from scripts import run_test_shard as shard_runner  # noqa: E402


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _configure_ascii_console_output() -> None:
    """Keep runner output safe when conda run re-encodes it using a legacy code page."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="ascii", errors="backslashreplace")


def _ascii_console_text(value: str) -> str:
    return value.encode("ascii", errors="backslashreplace").decode("ascii")


def _run_browser_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    output_log: Path,
) -> int:
    """Stream browser output to a UTF-8 log and an ASCII-safe console."""

    with output_log.open("w", encoding="utf-8", newline="") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("Browser process stdout was not captured")
        for line in process.stdout:
            log.write(line)
            log.flush()
            sys.stdout.write(_ascii_console_text(line))
            sys.stdout.flush()
        return process.wait()


def _read_database_urls() -> str:
    config = dotenv_values(ROOT / ".env")
    raw_admin = os.environ.get("ISOLATED_TEST_POSTGRES_ADMIN_URL", "").strip()
    raw_app = str(config.get("POSTGRES_URL") or os.environ.get("POSTGRES_URL") or "").strip()
    if raw_admin:
        parsed = make_url(raw_admin)
        if (parsed.host or "").lower() not in ALLOWED_LOCAL_HOSTS:
            raise RuntimeError("ISOLATED_TEST_POSTGRES_ADMIN_URL must point to a local PostgreSQL host")
        admin = parsed.set(database="postgres").render_as_string(hide_password=False)
        app_source = parsed
    else:
        if not raw_app:
            raise RuntimeError("POSTGRES_URL is missing; cannot derive a local maintenance connection")
        app_source = make_url(raw_app)
        if (app_source.host or "").lower() not in ALLOWED_LOCAL_HOSTS:
            raise RuntimeError("Refusing a non-local .env POSTGRES_URL for this E2E clone")
        # The app URL is read only to reuse local connection credentials. Every
        # builder/clone connection below is explicitly redirected to postgres.
        admin = app_source.set(database="postgres").render_as_string(hide_password=False)
    if (app_source.database or "").lower() == "postgres":
        raise RuntimeError("POSTGRES_URL must identify the local application DSN, not a maintenance DSN")
    host = (make_url(admin).host or "").lower()
    if host not in ALLOWED_LOCAL_HOSTS:
        raise RuntimeError("Maintenance URL escaped the local-host allowlist")
    return admin


def _verify_seeded_template(admin_url: str, template: str) -> None:
    environment = dict(os.environ)
    environment["ISOLATED_TEST_POSTGRES_ADMIN_URL"] = admin_url
    environment["BIZINSIGHT_TEST_SEEDED_TEMPLATE_DB"] = template
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_seeded_test_template.py"), "--verify"],
        cwd=ROOT,
        env=environment,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError("seeded synthetic template verification failed")


def _seed_browser_accounts(database_url: str, run_suffix: str) -> dict[str, Any]:
    # The same password hashing service as the application is used. Import only
    # after POSTGRES_URL has been replaced with the unique disposable clone.
    os.environ["POSTGRES_URL"] = database_url
    os.environ["ENVIRONMENT"] = "development"
    from app.iam.service import hash_password

    platform_tenant = "platform"
    seller_tenant = f"e2e_seller_{run_suffix}"
    platform_user = f"e2e_admin_{run_suffix}"
    seller_user = f"e2e_owner_{run_suffix}"
    analyst_user = f"e2e_analyst_{run_suffix}"
    platform_email = f"e2e-platform-{run_suffix}@example.test"
    seller_email = f"e2e-seller-{run_suffix}@example.test"
    analyst_email = f"e2e-seller-analyst-{run_suffix}@example.test"
    platform_password = secrets.token_urlsafe(30) + "!aA7"
    seller_password = secrets.token_urlsafe(30) + "!bB8"
    analyst_password = secrets.token_urlsafe(30) + "!cC9"
    platform_hash = hash_password(platform_password)
    seller_hash = hash_password(seller_password)
    analyst_hash = hash_password(analyst_password)
    engine = create_engine(database_url, pool_pre_ping=True, pool_size=1, max_overflow=0)
    try:
        with engine.begin() as conn:
            existing_tenants = int(conn.execute(text("SELECT count(*) FROM iam.tenants")).scalar_one())
            existing_mappings = int(conn.execute(text("SELECT count(*) FROM iam.tenant_seller_accounts")).scalar_one())
            if existing_tenants or existing_mappings:
                raise RuntimeError("seeded template clone unexpectedly contains IAM tenants or seller mappings")
            seller_items = int(conn.execute(text(
                "SELECT count(*) FROM analytics.fact_order_items WHERE seller_id=:seller"
            ), {"seller": FIXED_SELLER}).scalar_one())
            if seller_items <= 0:
                raise RuntimeError("synthetic fixed seller has no fact rows")

            conn.execute(text("""INSERT INTO iam.tenants(tenant_id,tenant_name,scope_type)
                VALUES(:tenant,'Full-chain synthetic platform','platform')"""), {"tenant": platform_tenant})
            conn.execute(text("""INSERT INTO iam.tenants(tenant_id,tenant_name,scope_type)
                VALUES(:tenant,'Full-chain synthetic seller','seller')"""), {"tenant": seller_tenant})
            conn.execute(text("INSERT INTO iam.users(user_id,display_name,email,password_hash) VALUES(:id,:name,:email,:hash)"), {
                "id": platform_user, "name": "Full-chain E2E Platform", "email": platform_email, "hash": platform_hash,
            })
            conn.execute(text("INSERT INTO iam.users(user_id,display_name,email,password_hash) VALUES(:id,:name,:email,:hash)"), {
                "id": seller_user, "name": "Full-chain E2E Seller", "email": seller_email, "hash": seller_hash,
            })
            conn.execute(text("INSERT INTO iam.users(user_id,display_name,email,password_hash) VALUES(:id,:name,:email,:hash)"), {
                "id": analyst_user, "name": "Full-chain E2E Seller Analyst", "email": analyst_email, "hash": analyst_hash,
            })
            conn.execute(text("""INSERT INTO iam.tenant_memberships(tenant_id,user_id,role,granted_by)
                VALUES(:tenant,:user,'admin',:actor)"""), {
                "tenant": platform_tenant, "user": platform_user, "actor": platform_user,
            })
            conn.execute(text("""INSERT INTO iam.tenant_memberships(tenant_id,user_id,role,granted_by)
                VALUES(:tenant,:user,'owner',:actor)"""), {
                "tenant": seller_tenant, "user": seller_user, "actor": seller_user,
            })
            conn.execute(text("""INSERT INTO iam.tenant_memberships(tenant_id,user_id,role,granted_by)
                VALUES(:tenant,:user,'analyst',:actor)"""), {
                "tenant": seller_tenant, "user": analyst_user, "actor": seller_user,
            })
            conn.execute(text("""INSERT INTO iam.membership_seller_grants
                (tenant_id,user_id,seller_id,access_start_date,access_end_date,granted_by)
                VALUES(:tenant,:user,:seller,DATE '2018-01-01',DATE '2018-12-31',:actor)"""), {
                "tenant": seller_tenant, "user": seller_user, "seller": FIXED_SELLER, "actor": seller_user,
            })
            conn.execute(text("""INSERT INTO iam.membership_seller_grants
                (tenant_id,user_id,seller_id,access_start_date,access_end_date,granted_by)
                VALUES(:tenant,:user,:seller,DATE '2018-01-01',DATE '2018-12-31',:actor)"""), {
                "tenant": seller_tenant, "user": analyst_user, "seller": FIXED_SELLER, "actor": seller_user,
            })
            conn.execute(text("""INSERT INTO iam.tenant_seller_accounts(tenant_id,seller_id,source_name)
                VALUES(:tenant,:seller,'synthetic_olist_e2e')"""), {"tenant": seller_tenant, "seller": FIXED_SELLER})

            # The disposable seeded clone contains no other tenant mappings.
            # Set the seller RLS context before the existing materializer runs.
            conn.execute(text("SELECT set_config('app.tenant_id',:tenant,true)"), {"tenant": seller_tenant})
            conn.execute(text("SELECT set_config('app.user_id',:user,true)"), {"user": seller_user})
            conn.execute(text("SELECT set_config('app.scope_type','seller',true)"))
            conn.execute(text("SELECT analytics.refresh_tenant_marts()"))
            tenant_items = int(conn.execute(text(
                "SELECT count(*) FROM analytics.tenant_fact_order_items WHERE tenant_id=:tenant"
            ), {"tenant": seller_tenant}).scalar_one())
            tenant_days = int(conn.execute(text(
                "SELECT count(*) FROM analytics.tenant_daily_metrics WHERE tenant_id=:tenant"
            ), {"tenant": seller_tenant}).scalar_one())
            if tenant_items <= 0 or tenant_days <= 0:
                raise RuntimeError("tenant marts did not materialize the synthetic seller fixture")
    finally:
        engine.dispose()

    return {
        "platform": {
            "email": platform_email,
            "password": platform_password,
            "display_name": "Full-chain E2E Platform",
            "tenant_id": platform_tenant,
            "user_id": platform_user,
        },
        "seller": {
            "email": seller_email,
            "password": seller_password,
            "display_name": "Full-chain E2E Seller",
            "tenant_id": seller_tenant,
            "user_id": seller_user,
            "seller_id": FIXED_SELLER,
        },
        "seller_analyst": {
            "email": analyst_email,
            "password": analyst_password,
            "display_name": "Full-chain E2E Seller Analyst",
            "tenant_id": seller_tenant,
            "user_id": analyst_user,
            "seller_id": FIXED_SELLER,
        },
        "fixture": {"synthetic_seller_items": tenant_items, "synthetic_seller_days": tenant_days},
    }


def _seed_action_plan_fixtures(database_url: str, identities: dict[str, Any]) -> dict[str, int]:
    """Seed isolated, synthetic parent plans and drafts used by the browser matrix."""
    engine = create_engine(database_url, pool_pre_ping=True, pool_size=1, max_overflow=0)
    plan_specs: dict[str, list[tuple[str, str]]] = {
        "platform": [
            ("e2e_ap_platform_evidence", "E2E 证据链检查草案"),
            *[(f"e2e_ap_platform_page_{index:02d}", f"E2E 分页草案 {index:02d}") for index in range(1, 21)],
        ],
        "seller": [
            ("e2e_ap_seller_title", "E2E 标题编辑草案"),
            ("e2e_ap_seller_content", "E2E 内容编辑草案"),
            ("e2e_ap_seller_invalid", "E2E JSON 校验草案"),
            ("e2e_ap_seller_accept", "E2E 采纳状态草案"),
            ("e2e_ap_seller_archive", "E2E 归档状态草案"),
            ("e2e_ap_seller_delete_cancel", "E2E 取消删除草案"),
            ("e2e_ap_seller_delete_confirm", "E2E 确认删除草案"),
            ("e2e_ap_seller_foreign", "E2E 卖家所有者私有草案"),
        ],
        "seller_analyst": [("e2e_ap_analyst_own", "E2E 分析员自有草案")],
    }
    counts: dict[str, int] = {}
    try:
        with engine.begin() as conn:
            for actor_key, specs in plan_specs.items():
                actor = identities[actor_key]
                tenant_id = str(actor["tenant_id"])
                user_id = str(actor["user_id"])
                scope_type = "platform" if actor_key == "platform" else "seller"
                conn.execute(text("SELECT set_config('app.tenant_id',:tenant,true)"), {"tenant": tenant_id})
                conn.execute(text("SELECT set_config('app.user_id',:user,true)"), {"user": user_id})
                conn.execute(text("SELECT set_config('app.scope_type',:scope,true)"), {"scope": scope_type})

                parent_id = f"e2e_parent_{actor_key}_{user_id[-12:]}"
                trace_id = f"e2e_trace_{actor_key}_{user_id[-12:]}"
                conn.execute(text("""INSERT INTO governance.complex_analysis_plans
                    (plan_id,tenant_id,user_id,chat_session_id,chat_turn_id,question,template_name,
                     scope_data,params_data,plan_data,plan_hash,status,expires_at,completed_at,trace_id,result_data)
                    VALUES(:plan,:tenant,:user,NULL,NULL,:question,'rfm_segmentation',
                     CAST(:scope AS jsonb),CAST(:params AS jsonb),CAST(:data AS jsonb),:hash,'completed',
                     now()+interval '30 days',now(),:trace,CAST(:result AS jsonb))"""), {
                    "plan": parent_id,
                    "tenant": tenant_id,
                    "user": user_id,
                    "question": "Synthetic action-plan E2E fixture; no user data.",
                    "scope": json.dumps({"scope_type": scope_type, "seller_id": actor.get("seller_id")}),
                    "params": json.dumps({"start_date": "2018-01-01", "end_date": "2018-08-31"}),
                    "data": json.dumps({"template_name": "rfm_segmentation", "fixture": True}),
                    "hash": f"e2e-seed-{parent_id}",
                    "trace": trace_id,
                    "result": json.dumps({"status": "ok", "evidence_ids": [f"e2e_evidence_{actor_key}"]}),
                })
                for action_plan_id, title in specs:
                    conn.execute(text("""INSERT INTO governance.action_plans
                        (action_plan_id,tenant_id,user_id,complex_plan_id,title,content,status,version)
                        VALUES(:id,:tenant,:user,:parent,:title,CAST(:content AS jsonb),'draft',1)"""), {
                        "id": action_plan_id,
                        "tenant": tenant_id,
                        "user": user_id,
                        "parent": parent_id,
                        "title": title,
                        "content": json.dumps({
                            "trace_id": trace_id,
                            "evidence_ids": [f"e2e_evidence_{actor_key}"],
                            "summary": "Synthetic evidence for isolated browser testing.",
                        }, ensure_ascii=False),
                    })
                counts[actor_key] = len(specs)
    finally:
        engine.dispose()
    return counts


def _seed_quality_trace_fixtures(trace_dir: Path, identities: dict[str, Any]) -> dict[str, int]:
    """Write deterministic trace-v2 and legacy traces into this run's temp directory."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    platform = identities["platform"]

    def write_trace(trace_id: str, *, legacy: bool) -> None:
        payload: dict[str, Any] = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "graph_path": ["finalize"],
            "state_snapshot": {
                "tenant_id": platform["tenant_id"],
                "user_id": platform["user_id"],
                "critic_verdict": "approved",
                "node_timings": [{"node": "finalize", "duration_ms": 1.0}],
            },
            "events": [{"node": "finalize", "event": "completed"}],
        }
        if not legacy:
            payload["trace_version"] = {
                "trace_kind": "agent_chat",
                "generation": "trace-v2",
                "schema_version": "2.0",
                "workflow_version": "governed-agent-v1",
                "prompt_version": "e2e-fixture-v1",
                "rag_release_version": "e2e-fixture-v1",
                "permission_policy_version": "e2e-fixture-v1",
            }
        (trace_dir / f"{trace_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    write_trace("e2e_quality_trace_v2", legacy=False)
    write_trace("e2e_quality_legacy_01", legacy=True)
    write_trace("e2e_quality_legacy_02", legacy=True)
    return {"trace_v2": 1, "legacy": 2}


def _wait_ready(url: str, process: subprocess.Popen[Any], timeout_seconds: int = 120) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "not checked"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"backend exited before readiness (exit={process.returncode})")
        try:
            with urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
                last_error = f"HTTP {response.status}"
        except (OSError, URLError) as exc:
            last_error = type(exc).__name__
        time.sleep(1)
    raise RuntimeError(f"backend readiness timed out: {last_error}")


def _stop_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=12)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=8)


def _playwright_results(path: Path) -> tuple[dict[str, int], list[dict[str, str]]]:
    counts = Counter({"passed": 0, "failed": 0, "skipped": 0, "timedOut": 0, "interrupted": 0})
    failures: list[dict[str, str]] = []
    if not path.exists():
        return dict(counts), [{"test_id": "<runner>", "reason": "Playwright JSON report was not produced"}]
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return dict(counts), [{"test_id": "<runner>", "reason": f"Playwright JSON report invalid: {type(exc).__name__}"}]

    def visit(suite: dict[str, Any], parent: str = "") -> None:
        title = str(suite.get("title") or "")
        prefix = " / ".join(value for value in (parent, title) if value)
        file_name = str(suite.get("file") or "")
        for spec in suite.get("specs", []):
            spec_title = str(spec.get("title") or "")
            test_id = "::".join(value for value in (file_name.replace("\\", "/"), prefix, spec_title) if value)
            for item in spec.get("tests", []):
                results = item.get("results") or []
                result = results[-1] if results else {}
                status = str(result.get("status") or "interrupted")
                counts[status] += 1
                if status not in {"passed", "skipped"}:
                    errors = result.get("errors") or []
                    message = str((errors[0] or {}).get("message") or status) if errors else status
                    message = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._~-]+", "Bearer [REDACTED]", message)
                    message = re.sub(r"(?i)(?:postgres(?:ql)?\+?\w*://)\S+", "[REDACTED_DSN]", message)
                    failures.append({"test_id": test_id, "reason": message[:1200]})
        for child in suite.get("suites", []):
            visit(child, prefix)

    for suite in document.get("suites", []):
        visit(suite)
    return dict(counts), failures


def _playwright_audit(path: Path) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "http_errors": [], "trace_ids_by_test": {}, "plan_ids_by_test": {},
        "client_turn_ids_by_test": {}, "session_ids_by_test": {},
        "expected_terminal_statuses_by_test": {},
        "expected_browser_terminal_events_by_test": {},
        "observed_browser_terminal_events_by_test": {},
        "history_resume_by_test": {},
        "history_metric_by_test": {},
        "favicon_requests_by_test": {},
        "test_results": [], "missing_trace_tests": [], "parse_errors": [],
    }
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        audit["parse_errors"].append(f"{type(exc).__name__}: Playwright report could not be parsed")
        return audit

    def collect_plan_ids(value: Any) -> set[str]:
        found: set[str] = set()
        if isinstance(value, dict):
            for key, nested in value.items():
                if key == "plan_id" and nested:
                    found.add(str(nested))
                else:
                    found.update(collect_plan_ids(nested))
        elif isinstance(value, list):
            for nested in value:
                found.update(collect_plan_ids(nested))
        return found

    def visit(suite: dict[str, Any]) -> None:
        for spec in suite.get("specs", []):
            spec_title = str(spec.get("title") or "")
            file_name = str(spec.get("file") or "")
            for test_case in spec.get("tests", []):
                for result in test_case.get("results", []):
                    test_title = spec_title
                    result_plan_ids: set[str] = set()
                    for attachment in result.get("attachments", []):
                        body = attachment.get("body")
                        if not isinstance(body, str):
                            continue
                        try:
                            decoded = base64.b64decode(body).decode("utf-8")
                            payload = json.loads(decoded)
                        except Exception:
                            continue
                        attachment_name = str(attachment.get("name") or "")
                        if attachment_name == "browser-diagnostics.json" and isinstance(payload, dict):
                            test_title = str(payload.get("test") or test_title)
                            traces = sorted({str(value) for value in payload.get("traceIds", []) if value})
                            if traces:
                                audit["trace_ids_by_test"].setdefault(test_title, [])
                                audit["trace_ids_by_test"][test_title] = sorted(
                                    set(audit["trace_ids_by_test"][test_title]) | set(traces)
                                )
                            diagnostic_plan_ids = sorted({str(value) for value in payload.get("planIds", []) if value})
                            if diagnostic_plan_ids:
                                audit["plan_ids_by_test"].setdefault(test_title, [])
                                audit["plan_ids_by_test"][test_title] = sorted(
                                    set(audit["plan_ids_by_test"][test_title]) | set(diagnostic_plan_ids)
                                )
                            client_turn_ids = sorted({str(value) for value in payload.get("clientTurnIds", []) if value})
                            if client_turn_ids:
                                audit["client_turn_ids_by_test"][test_title] = client_turn_ids
                            session_ids = sorted({str(value) for value in payload.get("sessionIds", []) if value})
                            if session_ids:
                                audit["session_ids_by_test"][test_title] = session_ids
                            for item in payload.get("httpErrors", []):
                                if isinstance(item, dict):
                                    audit["http_errors"].append({**item, "test_title": test_title})
                            favicon_requests = [
                                item for item in payload.get("faviconRequests", []) if isinstance(item, dict)
                            ]
                            if favicon_requests:
                                audit["favicon_requests_by_test"][test_title] = favicon_requests
                            recorded_urls = {str(item.get("url") or "") for item in payload.get("httpErrors", []) if isinstance(item, dict)}
                            for message in payload.get("consoleErrors", []):
                                message_text = str(message)
                                if "404" not in message_text or "favicon.ico" not in message_text:
                                    continue
                                match = re.search(r"https?://[^\s\]]+favicon\.ico", message_text)
                                if not match:
                                    continue
                                favicon_url = match.group(0)
                                if favicon_url in recorded_urls:
                                    continue
                                request_meta = next(
                                    (item for item in favicon_requests if str(item.get("url")) == favicon_url),
                                    {},
                                )
                                audit["http_errors"].append({
                                    "method": request_meta.get("method") or "unobserved",
                                    "url": favicon_url,
                                    "status": 404,
                                    "resource_type": request_meta.get("resource_type") or "unobserved",
                                    "test_title": test_title,
                                    "classification": "benign_favicon",
                                    "evidence_source": "browser_console_404",
                                })
                        elif attachment_name == "expanded-case-evidence.json" and isinstance(payload, dict):
                            test_title = str(payload.get("test") or test_title)
                            traces = {
                                str(value) for value in payload.get("trace_ids", []) if value
                            }
                            if not traces:
                                traces = {
                                    str(item.get("trace_id"))
                                    for item in payload.get("turns", [])
                                    if isinstance(item, dict) and item.get("trace_id")
                                }
                            if traces:
                                audit["trace_ids_by_test"].setdefault(test_title, [])
                                audit["trace_ids_by_test"][test_title] = sorted(
                                    set(audit["trace_ids_by_test"][test_title]) | traces
                                )
                            for field, report_field in (
                                ("plan_ids", "plan_ids_by_test"),
                                ("client_turn_ids", "client_turn_ids_by_test"),
                                ("session_ids", "session_ids_by_test"),
                            ):
                                raw_values = [str(value) for value in payload.get(field, []) if value]
                                values = list(dict.fromkeys(raw_values))
                                if values:
                                    if field == "client_turn_ids":
                                        # The expanded case evidence preserves first-request order;
                                        # browser diagnostics may sort these IDs and lose status alignment.
                                        audit[report_field][test_title] = values
                                    else:
                                        audit[report_field].setdefault(test_title, [])
                                        audit[report_field][test_title] = list(dict.fromkeys(
                                            [*audit[report_field][test_title], *values]
                                        ))
                            expected_statuses = payload.get("expected_terminal_statuses")
                            if isinstance(expected_statuses, list):
                                # Keep the declared contract shape: each request can
                                # require one status, accept several statuses, or
                                # expect no durable turn (None).
                                audit["expected_terminal_statuses_by_test"][test_title] = list(expected_statuses)
                            for payload_field, audit_field in (
                                ("expected_browser_terminal_events", "expected_browser_terminal_events_by_test"),
                                ("observed_browser_terminal_events", "observed_browser_terminal_events_by_test"),
                            ):
                                events = payload.get(payload_field)
                                if isinstance(events, list):
                                    audit[audit_field][test_title] = [
                                        str(value) if value is not None else None for value in events
                                    ]
                        elif attachment_name == "history-resume-evidence.json" and isinstance(payload, dict):
                            test_title = str(payload.get("test") or test_title)
                            audit["history_resume_by_test"][test_title] = {
                                key: payload.get(key) for key in (
                                    "session_id", "turn_id", "client_turn_id", "thread_id",
                                    "initial_trace_id", "resumed_trace_id",
                                )
                            }
                            traces = sorted({
                                str(payload.get(key)) for key in ("initial_trace_id", "resumed_trace_id")
                                if payload.get(key)
                            })
                            if traces:
                                audit["trace_ids_by_test"].setdefault(test_title, [])
                                audit["trace_ids_by_test"][test_title] = sorted(
                                    set(audit["trace_ids_by_test"][test_title]) | set(traces)
                                )
                            if payload.get("client_turn_id"):
                                audit["client_turn_ids_by_test"][test_title] = [str(payload["client_turn_id"])]
                            if payload.get("session_id"):
                                audit["session_ids_by_test"][test_title] = [str(payload["session_id"])]
                        elif attachment_name == "history-metric-evidence.json" and isinstance(payload, dict):
                            test_title = str(payload.get("test") or test_title)
                            audit["history_metric_by_test"][test_title] = {
                                key: payload.get(key) for key in (
                                    "session_id", "old_turn_id", "new_turn_id",
                                    "old_client_turn_id", "new_client_turn_id",
                                    "old_trace_id", "new_trace_id", "confirmed_metric_id",
                                    "old_thread_id", "new_thread_id",
                                )
                            }
                            traces = sorted({
                                str(payload.get(key)) for key in ("old_trace_id", "new_trace_id")
                                if payload.get(key)
                            })
                            if traces:
                                audit["trace_ids_by_test"].setdefault(test_title, [])
                                audit["trace_ids_by_test"][test_title] = sorted(
                                    set(audit["trace_ids_by_test"][test_title]) | set(traces)
                                )
                            if payload.get("session_id"):
                                audit["session_ids_by_test"][test_title] = [str(payload["session_id"])]
                        elif str(attachment.get("contentType") or "").startswith("application/json"):
                            plan_ids = sorted(collect_plan_ids(payload))
                            if plan_ids:
                                result_plan_ids.update(plan_ids)
                                audit["plan_ids_by_test"].setdefault(test_title, [])
                                audit["plan_ids_by_test"][test_title] = sorted(
                                    set(audit["plan_ids_by_test"][test_title]) | set(plan_ids)
                                )
                    audit["test_results"].append({
                        "test_id": f"{file_name}::{test_title}" if file_name else test_title,
                        "test_title": test_title,
                        "status": str(result.get("status") or "unknown"),
                        "duration_ms": int(result.get("duration") or 0),
                        "trace_ids": audit["trace_ids_by_test"].get(test_title, []),
                        "evidence_status": (
                            "ok" if audit["trace_ids_by_test"].get(test_title) else "missing_trace"
                        ),
                        "plan_ids": sorted(result_plan_ids),
                        "client_turn_ids": audit["client_turn_ids_by_test"].get(test_title, []),
                        "session_ids": audit["session_ids_by_test"].get(test_title, []),
                        "expected_terminal_statuses": audit["expected_terminal_statuses_by_test"].get(test_title),
                        "expected_browser_terminal_events": audit["expected_browser_terminal_events_by_test"].get(test_title),
                        "observed_browser_terminal_events": audit["observed_browser_terminal_events_by_test"].get(test_title),
                        "errors": [str((error or {}).get("message") or "")[:800] for error in result.get("errors", [])],
                    })
                    if not audit["trace_ids_by_test"].get(test_title):
                        audit["missing_trace_tests"].append(test_title)
        for child in suite.get("suites", []):
            visit(child)

    for suite in document.get("suites", []):
        visit(suite)
    return audit


def _chat_turn_disposition(
    *,
    status: str,
    interaction_state: str,
    task_type: str,
    execution_status: str | None,
    error_code: str | None,
) -> str:
    """Separate a deliberate suspended checkpoint from an orphaned active turn."""
    if (
        status == "interrupted"
        and interaction_state == "awaiting_human_input"
        and execution_status in {None, "suspended"}
        and not error_code
    ):
        if task_type == "anomaly_analysis":
            return "expected_suspended_premise_confirmation"
        return "expected_suspended_human_input"
    if (
        status == "interrupted"
        and interaction_state == "awaiting_scope"
        and execution_status in {None, "suspended"}
        and not error_code
    ):
        return "expected_suspended_scope_input"
    return "unresolved_nonterminal_turn"


def _turn_status_by_test(
    playwright_audit: dict[str, Any],
    turn_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Associate durable turn outcomes with browser cases by client turn ID."""
    rows_by_client_id: dict[str, list[dict[str, Any]]] = {}
    for raw_row in turn_rows:
        row = dict(raw_row)
        client_turn_id = str(row.get("client_turn_id") or "")
        if client_turn_id:
            rows_by_client_id.setdefault(client_turn_id, []).append(row)

    client_ids_by_test = playwright_audit.get("client_turn_ids_by_test") or {}
    expected_by_test = playwright_audit.get("expected_terminal_statuses_by_test") or {}
    output: dict[str, dict[str, Any]] = {}
    test_titles = dict.fromkeys([*client_ids_by_test, *expected_by_test])
    for test_title in test_titles:
        raw_client_ids = client_ids_by_test.get(test_title) or []
        client_turn_ids = list(dict.fromkeys(str(value) for value in raw_client_ids if value))
        expected = expected_by_test.get(test_title)
        expected_statuses = list(expected) if isinstance(expected, list) else None
        checks: list[dict[str, Any]] = []
        for index, client_turn_id in enumerate(client_turn_ids):
            matches = rows_by_client_id.get(client_turn_id, [])
            observed_statuses = sorted({str(row.get("status") or "") for row in matches})
            expected_status = (
                expected_statuses[index]
                if expected_statuses is not None and index < len(expected_statuses)
                else None
            )
            if expected_status is None:
                matches_expected: bool | None = len(matches) == 0
            else:
                accepted_statuses = (
                    [expected_status]
                    if isinstance(expected_status, str)
                    else expected_status
                    if isinstance(expected_status, list)
                    and all(isinstance(status, str) for status in expected_status)
                    else []
                )
                matches_expected = (
                    len(matches) == 1
                    and len(observed_statuses) == 1
                    and observed_statuses[0] in accepted_statuses
                )
            checks.append({
                "client_turn_id": client_turn_id,
                "expected_status": expected_status,
                "observed_statuses": observed_statuses,
                "turn_ids": sorted({str(row.get("turn_id") or "") for row in matches if row.get("turn_id")}),
                "trace_ids": sorted({
                    str(row.get("final_trace_id") or row.get("trace_id") or "")
                    for row in matches
                    if row.get("final_trace_id") or row.get("trace_id")
                }),
                "matches_expected": matches_expected,
            })
        expectation_is_complete = (
            expected_statuses is not None
            and bool(client_turn_ids)
            and len(expected_statuses) == len(client_turn_ids)
        )
        checks_match = all(item["matches_expected"] is True for item in checks)
        output[test_title] = {
            "client_turn_ids": client_turn_ids,
            "expected_terminal_statuses": expected_statuses,
            "observed_turns": [
                {
                    "turn_id": str(row.get("turn_id") or "") or None,
                    "client_turn_id": str(row.get("client_turn_id") or "") or None,
                    "status": str(row.get("status") or "") or None,
                    "interaction_state": str(row.get("interaction_state") or "none"),
                    "task_type": str(row.get("task_type") or ""),
                    "trace_id": str(row.get("final_trace_id") or row.get("trace_id") or "") or None,
                    "execution_status": str(row.get("execution_status") or "") or None,
                    "error_code": str(row.get("error_code") or "") or None,
                }
                for client_turn_id in client_turn_ids
                for row in rows_by_client_id.get(client_turn_id, [])
            ],
            "status_checks": checks,
            "unmatched_client_turn_ids": [
                item["client_turn_id"] for item in checks if not item["observed_statuses"]
            ],
            "status_match": checks_match if expectation_is_complete else None,
            "expectation_error": (
                "expected_status_count_must_match_unique_client_turn_ids"
                if expected_statuses is not None and not expectation_is_complete
                else None
            ),
        }
    return output


def _history_resume_evidence(database_url: str, identity: dict[str, str], audit: dict[str, Any]) -> dict[str, Any]:
    """Verify browser premise recovery reused one durable Turn and Execution."""
    engine = create_engine(database_url, pool_pre_ping=True)
    evidence: dict[str, Any] = {"cases": [], "passed": True}
    try:
        with engine.connect() as conn:
            for test_title, observed in (audit.get("history_resume_by_test") or {}).items():
                required = ("session_id", "turn_id", "client_turn_id", "thread_id")
                if any(not observed.get(key) for key in required):
                    case = {"test_title": test_title, "passed": False, "reason": "browser evidence missing identifiers"}
                    evidence["cases"].append(case)
                    evidence["passed"] = False
                    continue
                rows = conn.execute(text("""
                    SELECT t.turn_id, t.client_turn_id, t.status AS turn_status,
                           t.trace_id, t.final_trace_id,
                           e.execution_id, e.status AS execution_status,
                           e.langgraph_thread_id
                    FROM governance.chat_turns t
                    LEFT JOIN governance.conversation_executions e
                      ON e.turn_id=t.turn_id AND e.tenant_id=t.tenant_id AND e.user_id=t.user_id
                    WHERE t.tenant_id=:tenant AND t.user_id=:user
                      AND t.chat_session_id=:session AND t.turn_id=:turn
                    ORDER BY e.attempt_no
                """), {
                    "tenant": identity["tenant_id"],
                    "user": identity["user_id"],
                    "session": observed["session_id"],
                    "turn": observed["turn_id"],
                }).mappings().all()
                turn_count = int(conn.execute(text("""
                    SELECT count(*) FROM governance.chat_turns
                    WHERE tenant_id=:tenant AND user_id=:user AND chat_session_id=:session
                """), {"tenant": identity["tenant_id"], "user": identity["user_id"],
                      "session": observed["session_id"]}).scalar_one())
                execution_ids = sorted({str(row["execution_id"]) for row in rows if row["execution_id"]})
                linked_threads = sorted({str(row["langgraph_thread_id"]) for row in rows if row["langgraph_thread_id"]})
                latest = rows[-1] if rows else None
                final_turn_status = str(latest["turn_status"] or "") if latest else ""
                final_execution_status = str(latest["execution_status"] or "") if latest else ""
                checks = {
                    "one_turn_in_session": turn_count == 1 and len({str(row["turn_id"]) for row in rows}) == 1,
                    "same_client_turn": bool(latest and latest["client_turn_id"] == observed["client_turn_id"]),
                    "one_execution": len(execution_ids) == 1,
                    "same_thread": observed["thread_id"] in linked_threads,
                    "turn_completed": final_turn_status == "completed",
                    "execution_completed": final_execution_status == "completed",
                    "turn_execution_terminal_status_matches": final_turn_status == final_execution_status,
                    "terminal_result_present": bool(
                        latest
                        and (latest["final_trace_id"] or latest["trace_id"])
                        == observed.get("resumed_trace_id")
                    ),
                }
                case = {
                    "test_title": test_title,
                    "session_id": observed["session_id"],
                    "turn_id": observed["turn_id"],
                    "client_turn_id": observed["client_turn_id"],
                    "thread_id": observed["thread_id"],
                    "initial_trace_id": observed.get("initial_trace_id"),
                    "resumed_trace_id": observed.get("resumed_trace_id"),
                    "terminal_status": final_turn_status or None,
                    "execution_terminal_status": final_execution_status or None,
                    "execution_ids": execution_ids,
                    "final_trace_id": str((latest["final_trace_id"] or latest["trace_id"]) if latest else "") or None,
                    "turn_count_in_session": turn_count,
                    "checks": checks,
                    "passed": all(checks.values()),
                }
                evidence["cases"].append(case)
                evidence["passed"] = evidence["passed"] and case["passed"]
    finally:
        engine.dispose()
    if not evidence["cases"]:
        evidence["passed"] = False
        evidence["reason"] = "history browser suite produced no premise-resume evidence"
    return evidence


def _history_metric_evidence(database_url: str, identity: dict[str, str], audit: dict[str, Any]) -> dict[str, Any]:
    """Verify metric confirmation created a distinct authorized Turn/Execution pair."""
    engine = create_engine(database_url, pool_pre_ping=True)
    evidence: dict[str, Any] = {"cases": [], "passed": True}
    try:
        with engine.connect() as conn:
            for test_title, observed in (audit.get("history_metric_by_test") or {}).items():
                required = (
                    "session_id", "old_turn_id", "new_turn_id", "old_client_turn_id",
                    "new_client_turn_id", "confirmed_metric_id", "old_thread_id", "new_thread_id",
                )
                missing = [key for key in required if not observed.get(key)]
                if missing:
                    case = {"test_title": test_title, "passed": False,
                            "reason": "browser metric evidence missing identifiers", "missing": missing}
                    evidence["cases"].append(case)
                    evidence["passed"] = False
                    continue
                rows = conn.execute(text("""
                    SELECT t.turn_id, t.client_turn_id, t.status AS turn_status,
                           t.trace_id, t.final_trace_id,
                           e.execution_id, e.status AS execution_status,
                           e.langgraph_thread_id
                    FROM governance.chat_turns t
                    LEFT JOIN governance.conversation_executions e
                      ON e.turn_id=t.turn_id AND e.tenant_id=t.tenant_id AND e.user_id=t.user_id
                    WHERE t.tenant_id=:tenant AND t.user_id=:user
                      AND t.chat_session_id=:session
                      AND t.turn_id IN (:old_turn, :new_turn)
                    ORDER BY t.created_at, e.attempt_no
                """), {
                    "tenant": identity["tenant_id"], "user": identity["user_id"],
                    "session": observed["session_id"],
                    "old_turn": observed["old_turn_id"],
                    "new_turn": observed["new_turn_id"],
                }).mappings().all()
                turn_ids = {str(row["turn_id"]) for row in rows}
                session_count = int(conn.execute(text("""
                    SELECT count(*) FROM governance.chat_turns
                    WHERE tenant_id=:tenant AND user_id=:user AND chat_session_id=:session
                """), {"tenant": identity["tenant_id"], "user": identity["user_id"],
                      "session": observed["session_id"]}).scalar_one())
                by_turn = {str(row["turn_id"]): row for row in rows}
                old = by_turn.get(str(observed["old_turn_id"]))
                new = by_turn.get(str(observed["new_turn_id"]))
                old_execs = {str(row["execution_id"]) for row in rows
                             if row["client_turn_id"] == observed["old_client_turn_id"] and row["execution_id"]}
                new_execs = {str(row["execution_id"]) for row in rows
                             if row["client_turn_id"] == observed["new_client_turn_id"] and row["execution_id"]}
                checks = {
                    "exactly_two_turns_in_session": session_count == 2 and len(turn_ids) == 2,
                    "old_turn_preserved": bool(old and str(old["turn_id"]) == str(observed["old_turn_id"])),
                    "new_turn_created": bool(new and str(new["turn_id"]) == str(observed["new_turn_id"])),
                    "distinct_turn_and_client_ids": bool(
                        old and new and old["turn_id"] != new["turn_id"]
                        and old["client_turn_id"] == observed["old_client_turn_id"]
                        and new["client_turn_id"] == observed["new_client_turn_id"]
                        and observed["old_client_turn_id"] != observed["new_client_turn_id"]
                    ),
                    "execution_cardinality_is_valid": (
                        (len(old_execs) == 1 and len(new_execs) == 1)
                        or (not old_execs and not new_execs)
                    ),
                    "distinct_threads": bool(
                        observed["old_thread_id"] != observed["new_thread_id"]
                        and (
                            (old and new and old["langgraph_thread_id"] == observed["old_thread_id"]
                             and new["langgraph_thread_id"] == observed["new_thread_id"])
                            or (not old_execs and not new_execs)
                        )
                    ),
                    "clarification_turn_abstained_and_confirmation_completed": bool(
                        old and new and old["turn_status"] == "abstained" and new["turn_status"] == "completed"
                        and (
                            (old_execs and new_execs
                             and old["execution_status"] == "completed"
                             and new["execution_status"] == "completed")
                            or (not old_execs and not new_execs)
                        )
                    ),
                    "trace_ids_match_database": bool(
                        old and new
                        and (old["final_trace_id"] or old["trace_id"]) == observed.get("old_trace_id")
                        and (new["final_trace_id"] or new["trace_id"]) == observed.get("new_trace_id")
                    ),
                    "both_traces_attached": bool(observed.get("old_trace_id") and observed.get("new_trace_id")),
                }
                case = {
                    "test_title": test_title, "session_id": observed["session_id"],
                    "old_turn_id": str(old["turn_id"]) if old else None,
                    "new_turn_id": str(new["turn_id"]) if new else None,
                    "old_client_turn_id": observed["old_client_turn_id"],
                    "new_client_turn_id": observed["new_client_turn_id"],
                    "old_execution_ids": sorted(old_execs), "new_execution_ids": sorted(new_execs),
                    "requested_thread_ids": {
                        "old": observed["old_thread_id"], "new": observed["new_thread_id"],
                    },
                    "persistence_mode": (
                        "durable_lifecycle" if old_execs and new_execs
                        else "legacy_chat_history_without_execution_rows"
                    ),
                    "old_thread_id": str(old["langgraph_thread_id"])
                    if old and old["langgraph_thread_id"] else None,
                    "new_thread_id": str(new["langgraph_thread_id"])
                    if new and new["langgraph_thread_id"] else None,
                    "old_turn_status": str(old["turn_status"]) if old else None,
                    "new_turn_status": str(new["turn_status"]) if new else None,
                    "trace_ids": [observed.get("old_trace_id"), observed.get("new_trace_id")],
                    "confirmed_metric_id": observed["confirmed_metric_id"],
                    "session_turn_count": session_count, "checks": checks,
                    "passed": all(checks.values()),
                }
                evidence["cases"].append(case)
                evidence["passed"] = evidence["passed"] and case["passed"]
    finally:
        engine.dispose()
    if not evidence["cases"]:
        evidence["passed"] = False
        evidence["reason"] = "history browser suite produced no metric-candidate evidence"
    return evidence


def _database_evidence(
    database_url: str,
    identities: dict[str, Any],
    playwright_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    engine = create_engine(database_url, pool_pre_ping=True)
    playwright_audit = playwright_audit or {}
    trace_to_tests: dict[str, list[str]] = {}
    for test_title, traces in (playwright_audit.get("trace_ids_by_test") or {}).items():
        for trace in traces:
            trace_to_tests.setdefault(str(trace), []).append(str(test_title))
    client_turn_to_tests: dict[str, list[str]] = {}
    for test_title, client_turn_ids in (playwright_audit.get("client_turn_ids_by_test") or {}).items():
        for client_turn_id in client_turn_ids:
            client_turn_to_tests.setdefault(str(client_turn_id), []).append(str(test_title))
    plan_to_tests: dict[str, list[str]] = {}
    for test_title, plan_ids in (playwright_audit.get("plan_ids_by_test") or {}).items():
        for plan_id in plan_ids:
            plan_to_tests.setdefault(str(plan_id), []).append(str(test_title))
    tenants = [identities["platform"]["tenant_id"], identities["seller"]["tenant_id"]]
    users = [identities["platform"]["user_id"], identities["seller"]["user_id"]]
    result: dict[str, Any] = {
        "provider_usage": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "models": [], "providers": [], "by_test": []},
        "chat_turns": None,
        "turn_status_by_test": {},
        "chat_turns_nonterminal": [],
        "chat_turns_awaiting_human_input": [],
        "chat_turns_awaiting_scope": [],
        "complex_plans": None,
        "complex_plans_nonterminal": [],
        "complex_executions": None,
    }
    try:
        inspector = inspect(engine)
        with engine.connect() as conn:
            if "llm_usage_events" in inspector.get_table_names(schema="governance"):
                rows = conn.execute(text("""SELECT provider_id,model,operation,usage_source,
                    count(*) AS calls,coalesce(sum(input_tokens),0) AS input_tokens,
                    coalesce(sum(output_tokens),0) AS output_tokens
                    FROM governance.llm_usage_events
                    WHERE tenant_id IN (:platform_tenant,:seller_tenant)
                      AND user_id IN (:platform_user,:seller_user)
                      AND usage_source='provider_reported'
                      AND (coalesce(input_tokens,0)+coalesce(output_tokens,0))>0
                    GROUP BY provider_id,model,operation,usage_source
                    ORDER BY provider_id,model,operation"""), {
                    "platform_tenant": tenants[0], "seller_tenant": tenants[1],
                    "platform_user": users[0], "seller_user": users[1],
                }).mappings().all()
                by_test: list[dict[str, Any]] = []
                for test_title, trace_ids in (playwright_audit.get("trace_ids_by_test") or {}).items():
                    totals: dict[str, Any] = {
                        "test_title": test_title, "calls": 0, "input_tokens": 0,
                        "output_tokens": 0, "models": [], "providers": [],
                    }
                    trace_rows: list[dict[str, Any]] = []
                    for trace_id in sorted(set(str(item) for item in trace_ids if item)):
                        trace_rows.extend(conn.execute(text("""SELECT provider_id,model,operation,
                            count(*) AS calls,coalesce(sum(input_tokens),0) AS input_tokens,
                            coalesce(sum(output_tokens),0) AS output_tokens
                            FROM governance.llm_usage_events
                            WHERE tenant_id IN (:platform_tenant,:seller_tenant)
                              AND user_id IN (:platform_user,:seller_user)
                              AND trace_id=:trace_id AND usage_source='provider_reported'
                              AND (coalesce(input_tokens,0)+coalesce(output_tokens,0))>0
                            GROUP BY provider_id,model,operation"""), {
                            "platform_tenant": tenants[0], "seller_tenant": tenants[1],
                            "platform_user": users[0], "seller_user": users[1], "trace_id": trace_id,
                        }).mappings().all())
                    totals["calls"] = sum(int(row["calls"]) for row in trace_rows)
                    totals["input_tokens"] = sum(int(row["input_tokens"]) for row in trace_rows)
                    totals["output_tokens"] = sum(int(row["output_tokens"]) for row in trace_rows)
                    totals["models"] = sorted({str(row["model"]) for row in trace_rows})
                    totals["providers"] = sorted({str(row["provider_id"]) for row in trace_rows})
                    by_test.append(totals)
                result["provider_usage"] = {
                    "calls": sum(int(row["calls"]) for row in rows),
                    "input_tokens": sum(int(row["input_tokens"]) for row in rows),
                    "output_tokens": sum(int(row["output_tokens"]) for row in rows),
                    "models": sorted({str(row["model"]) for row in rows}),
                    "providers": sorted({str(row["provider_id"]) for row in rows}),
                    "by_operation": [dict(row) for row in rows],
                    "by_test": by_test,
                    "evidence": "governance.llm_usage_events.usage_source=provider_reported",
                }
            for table, result_key in (
                ("chat_turns", "chat_turns"),
                ("complex_analysis_plans", "complex_plans"),
                ("analysis_execution_audits", "complex_executions"),
            ):
                if table not in inspector.get_table_names(schema="governance"):
                    continue
                columns = {column["name"] for column in inspector.get_columns(table, schema="governance")}
                if "tenant_id" not in columns:
                    continue
                if "status" in columns:
                    counts = conn.execute(text(
                        f"SELECT status,count(*) AS count FROM governance.{table} "
                        "WHERE tenant_id IN (:platform_tenant,:seller_tenant) GROUP BY status ORDER BY status"
                    ), {"platform_tenant": tenants[0], "seller_tenant": tenants[1]}).mappings().all()
                    result[result_key] = {str(row["status"]): int(row["count"]) for row in counts}
                    if table == "chat_turns":
                        required = {
                            "turn_id", "tenant_id", "user_id", "chat_session_id", "client_turn_id",
                            "status", "interaction_state", "task_type", "trace_id", "final_trace_id",
                            "final_execution_id", "created_at", "error_code",
                        }
                        if required <= columns:
                            status_rows = conn.execute(text("""SELECT t.turn_id,t.client_turn_id,t.status,
                                t.interaction_state,t.task_type,t.trace_id,t.final_trace_id,t.error_code,
                                e.status AS execution_status
                                FROM governance.chat_turns t
                                LEFT JOIN LATERAL (
                                  SELECT status FROM governance.conversation_executions
                                  WHERE turn_id=t.turn_id AND tenant_id=t.tenant_id AND user_id=t.user_id
                                  ORDER BY attempt_no DESC LIMIT 1
                                ) e ON true
                                WHERE t.tenant_id IN (:platform_tenant,:seller_tenant)
                                  AND t.user_id IN (:platform_user,:seller_user)
                                ORDER BY t.created_at,t.turn_id"""), {
                                "platform_tenant": tenants[0], "seller_tenant": tenants[1],
                                "platform_user": users[0], "seller_user": users[1],
                            }).mappings().all()
                            result["turn_status_by_test"] = _turn_status_by_test(
                                playwright_audit,
                                [dict(row) for row in status_rows],
                            )
                            rows = conn.execute(text("""SELECT t.turn_id,t.tenant_id,t.user_id,t.chat_session_id,
                                t.client_turn_id,t.status,t.interaction_state,t.task_type,t.trace_id,
                                t.final_trace_id,t.final_execution_id,t.created_at,t.error_code,
                                e.execution_id AS latest_execution_id,e.status AS latest_execution_status,
                                e.trace_id AS execution_trace_id
                                FROM governance.chat_turns t
                                LEFT JOIN LATERAL (
                                  SELECT execution_id,status,trace_id FROM governance.conversation_executions
                                  WHERE turn_id=t.turn_id AND tenant_id=t.tenant_id AND user_id=t.user_id
                                  ORDER BY attempt_no DESC LIMIT 1
                                ) e ON true
                                WHERE t.tenant_id IN (:platform_tenant,:seller_tenant)
                                  AND t.status IN ('running','pending','interrupted')
                                ORDER BY t.created_at,t.turn_id"""), {
                                "platform_tenant": tenants[0], "seller_tenant": tenants[1],
                            }).mappings().all()
                            for row in rows:
                                trace_id = str(row["final_trace_id"] or row["trace_id"] or row["execution_trace_id"] or "")
                                client_turn_id = str(row["client_turn_id"] or "")
                                candidates = set(trace_to_tests.get(trace_id, [])) if trace_id else set()
                                if client_turn_id:
                                    candidates.update(client_turn_to_tests.get(client_turn_id, []))
                                record = {
                                    "turn_id": str(row["turn_id"]),
                                    "tenant_id": str(row["tenant_id"]),
                                    "user_id": str(row["user_id"]),
                                    "chat_session_id": str(row["chat_session_id"]),
                                    "client_turn_id": client_turn_id or None,
                                    "status": str(row["status"]),
                                    "interaction_state": str(row["interaction_state"] or "none"),
                                    "task_type": str(row["task_type"] or ""),
                                    "trace_id": trace_id or None,
                                    "final_execution_id": str(row["final_execution_id"] or row["latest_execution_id"] or "") or None,
                                    "execution_status": str(row["latest_execution_status"] or "") or None,
                                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                                    "error_code": row["error_code"],
                                    "playwright_test_candidates": sorted(candidates),
                                }
                                record["disposition"] = _chat_turn_disposition(
                                    status=record["status"],
                                    interaction_state=record["interaction_state"],
                                    task_type=record["task_type"],
                                    execution_status=record["execution_status"],
                                    error_code=record["error_code"],
                                )
                                if record["disposition"] == "unresolved_nonterminal_turn":
                                    result["chat_turns_nonterminal"].append(record)
                                elif record["disposition"] == "expected_suspended_scope_input":
                                    result["chat_turns_awaiting_scope"].append(record)
                                else:
                                    result["chat_turns_awaiting_human_input"].append(record)
                    if table == "complex_analysis_plans":
                        required = {
                            "plan_id", "tenant_id", "user_id", "template_name", "status", "chat_session_id",
                            "chat_turn_id", "created_at", "expires_at", "trace_id", "scope_data",
                        }
                        if required <= columns:
                            rows = conn.execute(text("""SELECT p.plan_id,p.tenant_id,p.user_id,p.template_name,p.status,
                                p.chat_session_id,p.chat_turn_id,p.created_at,p.expires_at,p.trace_id,p.scope_data,
                                t.turn_id AS linked_turn_id,t.client_turn_id AS linked_client_turn_id,
                                t.status AS linked_turn_status,t.interaction_state AS linked_interaction_state,
                                e.status AS linked_execution_status
                                FROM governance.complex_analysis_plans p
                                LEFT JOIN governance.chat_turns t
                                  ON t.tenant_id=p.tenant_id AND t.user_id=p.user_id
                                 AND (t.turn_id=p.chat_turn_id OR t.client_turn_id=p.chat_turn_id)
                                LEFT JOIN LATERAL (
                                  SELECT status FROM governance.conversation_executions
                                  WHERE turn_id=t.turn_id AND tenant_id=t.tenant_id AND user_id=t.user_id
                                  ORDER BY attempt_no DESC LIMIT 1
                                ) e ON true
                                WHERE p.tenant_id IN (:platform_tenant,:seller_tenant)
                                  AND p.status IN ('pending_confirmation','running')
                                ORDER BY p.created_at,p.plan_id"""), {
                                "platform_tenant": tenants[0], "seller_tenant": tenants[1],
                            }).mappings().all()
                            for row in rows:
                                trace_id = str(row["trace_id"] or "")
                                plan_id = str(row["plan_id"])
                                raw_scope = row["scope_data"] if isinstance(row["scope_data"], dict) else {}
                                seller_ids = raw_scope.get("seller_ids")
                                if not isinstance(seller_ids, (list, tuple)):
                                    seller_ids = [raw_scope["seller_id"]] if raw_scope.get("seller_id") else []
                                scope_summary = {
                                    "scope_type": raw_scope.get("scope_type") or raw_scope.get("type"),
                                    "seller_count": len(seller_ids),
                                    "start_date": raw_scope.get("start_date") or raw_scope.get("date_from"),
                                    "end_date": raw_scope.get("end_date") or raw_scope.get("date_to"),
                                }
                                candidates = set(plan_to_tests.get(plan_id, []))
                                if trace_id:
                                    candidates.update(trace_to_tests.get(trace_id, []))
                                linked_client_turn_id = str(row["linked_client_turn_id"] or "")
                                if linked_client_turn_id:
                                    candidates.update(client_turn_to_tests.get(linked_client_turn_id, []))
                                result["complex_plans_nonterminal"].append({
                                    "plan_id": plan_id,
                                    "tenant_id": str(row["tenant_id"]),
                                    "user_id": str(row["user_id"]),
                                    "template_name": str(row["template_name"]),
                                    "status": str(row["status"]),
                                    "chat_session_id": str(row["chat_session_id"]),
                                    "chat_turn_id": str(row["chat_turn_id"] or ""),
                                    "linked_turn_id": str(row["linked_turn_id"] or "") or None,
                                    "linked_turn_status": str(row["linked_turn_status"] or "") or None,
                                    "linked_interaction_state": str(row["linked_interaction_state"] or "") or None,
                                    "linked_execution_status": str(row["linked_execution_status"] or "") or None,
                                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                                    "expires_at": row["expires_at"].isoformat() if row["expires_at"] else None,
                                    "trace_id": trace_id or None,
                                    "scope_summary": scope_summary,
                                    "playwright_test_candidates": sorted(candidates),
                                    "disposition": (
                                        "pending_plan_from_attributed_browser_test"
                                        if str(row["status"]) == "pending_confirmation" and candidates
                                        else "unresolved_nonterminal_plan"
                                    ),
                                })
                else:
                    result[result_key] = int(conn.execute(text(
                        f"SELECT count(*) FROM governance.{table} WHERE tenant_id IN (:platform_tenant,:seller_tenant)"
                    ), {"platform_tenant": tenants[0], "seller_tenant": tenants[1]}).scalar_one())
        return result
    finally:
        engine.dispose()


def _safe_log(source: Path, target: Path, secret_values: list[str]) -> None:
    if not source.exists():
        return
    try:
        value = source.read_text(encoding="utf-8", errors="replace")
        for secret in sorted((item for item in secret_values if item), key=len, reverse=True):
            value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._~-]+", "Bearer [REDACTED]", value)
        value = re.sub(r"(?i)(?:postgres(?:ql)?\+?\w*://)\S+", "[REDACTED_DSN]", value)
        target.write_text(value, encoding="utf-8")
    except Exception:
        target.write_text("Backend log could not be sanitized; omitted.\n", encoding="utf-8")


def _load_expanded_cases() -> list[dict[str, Any]]:
    path = WEB / "tests" / "e2e" / "cases" / "expanded-cases.jsonl"
    if not path.is_file():
        raise ValueError(f"Expanded case file is missing: {path}")
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            case = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid expanded JSONL at line {line_number}: {exc.msg}") from exc
        case_id = str(case.get("case_id") or "").strip()
        if not case_id or case_id in seen:
            raise ValueError(f"Expanded case_id is empty or duplicated at line {line_number}")
        turns = case.get("turns")
        entry = str(case.get("entry") or "").strip()
        if not str(case.get("feature") or "").strip() or not isinstance(turns, list):
            raise ValueError(f"Expanded case {case_id} must declare feature and a turns list")
        page_workflows = {
            "reports": "report_workflow",
            "metrics": "metric_target_workflow",
            "preferences": "preference_workflow",
            "action_plans": "action_plan_workflow",
            "quality": "quality_workflow",
        }
        if entry in page_workflows:
            workflow_key = page_workflows[entry]
            if turns or not isinstance(case.get(workflow_key), dict):
                raise ValueError(
                    f"Expanded page case {case_id} must declare {workflow_key} and no chat turns"
                )
        elif entry == "knowledge":
            if not isinstance(case.get("knowledge_workflow"), dict):
                raise ValueError(f"Expanded knowledge case {case_id} must declare knowledge_workflow")
        elif entry in {"chat", "seller_scoped_chat"}:
            if not turns:
                raise ValueError(f"Expanded chat case {case_id} must declare at least one turn")
        else:
            raise ValueError(f"Expanded case {case_id} has unsupported entry {entry!r}")
        if case.get("persona") not in {"platform_admin", "seller_owner", "seller_analyst"}:
            raise ValueError(f"Expanded case {case_id} has an unsupported persona")
        seen.add(case_id)
        cases.append(case)
    return cases


def _split_csv_values(values: list[str]) -> set[str]:
    return {
        item.strip()
        for value in values
        for item in value.split(",")
        if item.strip()
    }


def main() -> int:
    _configure_ascii_console_output()
    parser = argparse.ArgumentParser(description="Run browser-to-API-to-PostgreSQL full-chain E2E on an isolated clone")
    parser.add_argument("--suite", choices=("full", "expanded", "history"), default="full", help="Select the full-chain, expanded, or history-resume browser suite")
    parser.add_argument("--grep", help="Run only Playwright tests matching this title/pattern")
    parser.add_argument("--feature", action="append", default=[], help="Expanded feature filter; may be repeated")
    parser.add_argument("--case", action="append", default=[], help="Expanded case_id filter; may be repeated or comma-separated")
    parser.add_argument("--skip-cases", action="append", default=[], help="Expanded case_ids to exclude; may be repeated or comma-separated")
    parser.add_argument("--max-cases", type=int, help="Run at most N selected expanded cases in JSONL order")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed Playwright test")
    parser.add_argument("--max-failures", type=int, help="Stop Playwright after N failed tests; default is unlimited")
    parser.add_argument(
        "--allow-no-llm",
        action="store_true",
        help="Allow a filtered browser scenario that does not call the configured LLM provider",
    )
    args = parser.parse_args()
    if args.suite == "history" and (args.grep or args.feature or args.case or args.skip_cases or args.max_cases is not None or args.allow_no_llm):
        parser.error("--suite history runs its complete browser contract and does not accept filters or --allow-no-llm")
    if args.fail_fast and args.max_failures is not None:
        parser.error("--fail-fast cannot be combined with --max-failures")
    if args.max_failures is not None and args.max_failures <= 0:
        parser.error("--max-failures must be greater than zero")
    selected_expanded_cases: list[dict[str, Any]] = []
    excluded_expanded_case_ids: list[str] = []
    if args.suite == "expanded":
        if args.grep:
            parser.error("--grep is only valid for --suite full; use --feature/--case for expanded cases")
        if args.max_cases is not None and args.max_cases <= 0:
            parser.error("--max-cases must be greater than zero")
        all_cases = _load_expanded_cases()
        requested_ids = _split_csv_values(args.case)
        skipped_ids = _split_csv_values(args.skip_cases)
        known_ids = {str(case["case_id"]) for case in all_cases}
        unknown_ids = (requested_ids | skipped_ids) - known_ids
        if unknown_ids:
            parser.error("Unknown expanded case_id(s): " + ", ".join(sorted(unknown_ids)))
        feature_filter = set(args.feature)
        eligible_cases = [
            case for case in all_cases
            if (not feature_filter or case["feature"] in feature_filter)
            and (not requested_ids or case["case_id"] in requested_ids)
        ]
        selected_expanded_cases = [case for case in eligible_cases if case["case_id"] not in skipped_ids]
        if args.max_cases is not None:
            selected_expanded_cases = selected_expanded_cases[:args.max_cases]
        if not selected_expanded_cases:
            parser.error("Expanded filters selected zero cases")
        selected_ids = {str(case["case_id"]) for case in selected_expanded_cases}
        excluded_expanded_case_ids = [str(case["case_id"]) for case in eligible_cases if case["case_id"] not in selected_ids]
    elif args.feature or args.case or args.skip_cases or args.max_cases is not None:
        parser.error("--feature/--case/--skip-cases/--max-cases require --suite expanded")
    expanded_filter_used = bool(args.feature or args.case or args.skip_cases or args.max_cases is not None)
    if args.allow_no_llm and not (args.grep or (args.suite == "expanded" and expanded_filter_used)):
        parser.error("--allow-no-llm is only valid for a filtered suite")

    if not (WEB / "node_modules" / "playwright" / "cli.js").exists():
        raise SystemExit("Playwright CLI is missing; install the existing web package dependencies first")
    node = shutil.which("node")
    if not node:
        raise SystemExit("Node.js was not found on PATH")

    config = dotenv_values(ROOT / ".env")
    if not config.get("DEEPSEEK_API_KEY") or not config.get("DEEPSEEK_BASE_URL"):
        raise SystemExit("Real LLM provider credentials are not configured in .env; no mock fallback is allowed")
    admin_url = _read_database_urls()
    template = os.environ.get("BIZINSIGHT_TEST_SEEDED_TEMPLATE_DB", TEMPLATE_DEFAULT)
    if not template.startswith("codex_test_template_seeded_"):
        raise SystemExit("E2E requires a codex_test_template_seeded_* template")
    shard_runner._check_database_name(template, allow_template=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    suffix = uuid.uuid4().hex[:12]
    clone_prefix = (
        "codex_test_expanded_" if args.suite == "expanded"
        else "codex_test_history_" if args.suite == "history"
        else "codex_test_fullchain_"
    )
    clone = f"{clone_prefix}{suffix}"
    shard_runner._check_database_name(clone)
    output_dir = ROOT / "data" / "personal" / "full_chain_e2e" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    artifact_dir = output_dir / "playwright-artifacts"
    artifact_dir.mkdir()
    backend_log_raw = output_dir / "backend.raw.log"
    backend_log = output_dir / "backend.log"
    playwright_console_log = output_dir / "playwright-console.log"
    playwright_json = output_dir / "playwright-report.json"
    report_path = output_dir / "report.json"
    frontend_port, backend_port = _free_port(), _free_port()
    if frontend_port == backend_port:
        backend_port = _free_port()

    state: dict[str, Any] = {
        "run_id": run_id,
        "head": "unknown",
        "suite": args.suite,
        "selected_case_ids": [str(case["case_id"]) for case in selected_expanded_cases],
        "excluded_case_ids": excluded_expanded_case_ids,
        "template": template,
        "clone_database": clone,
        "database_isolation": "fresh PostgreSQL clone of deterministic synthetic seeded template",
        "llm": {
            "required": not args.allow_no_llm,
            "adapter": "application configured provider; no mock/stub",
            "configured": True,
            "playwright_grep": args.grep,
        },
        "failure_policy": {
            "fail_fast": bool(args.fail_fast),
            "max_failures": 1 if args.fail_fast else args.max_failures,
        },
        "ports": {"backend": backend_port, "frontend": frontend_port},
        "tests": {},
        "database_evidence": {},
        "cleanup": {"clone_dropped": False, "residual": None},
        "artifacts": {
            "directory": str(output_dir),
            "playwright_json": str(playwright_json),
            "playwright_console_log": str(playwright_console_log),
            "backend_log": str(backend_log),
        },
    }
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        state["head"] = head
        _verify_seeded_template(admin_url, template)
        with urlopen("http://127.0.0.1:16333/collections", timeout=5) as qdrant_response:
            if qdrant_response.status != 200:
                raise RuntimeError(f"Local read-only Qdrant preflight failed with HTTP {qdrant_response.status}")
        shard_runner._create_database(admin_url, clone, template)
        database_url = make_url(admin_url).set(database=clone).render_as_string(hide_password=False)
        identities = _seed_browser_accounts(database_url, suffix)
        identities["fixture"]["action_plan_counts"] = _seed_action_plan_fixtures(database_url, identities)
        state["synthetic_fixture"] = identities["fixture"]

        temp_runtime = output_dir / "runtime"
        for directory in ("traces", "charts", "rag-cache", "quality"):
            (temp_runtime / directory).mkdir(parents=True, exist_ok=True)
        identities["fixture"]["quality_trace_counts"] = _seed_quality_trace_fixtures(temp_runtime / "traces", identities)
        config_env = {key: str(value) for key, value in config.items() if value is not None}
        server_env = dict(os.environ)
        server_env.update(config_env)
        server_env.update({
            "POSTGRES_URL": database_url,
            "TENANT_AUTH_REQUIRED": "true",
            "TENANT_AUTH_SECRET": str(config.get("TENANT_AUTH_SECRET") or secrets.token_urlsafe(64)),
            "ENVIRONMENT": "development",
            "TRACE_BACKEND": "file",
            "TRACE_DIR": str(temp_runtime / "traces"),
            "QUALITY_DATA_DIR": str(temp_runtime / "quality"),
            "CHECKPOINT_BACKEND": "sqlite",
            "CHECKPOINT_PATH": str(temp_runtime / "checkpoints.sqlite"),
            "PHASE6_STORE_PATH": str(temp_runtime / "phase6.sqlite"),
            "REPORT_CHART_DIR": str(temp_runtime / "charts"),
            "RAG_CACHE_DIR": str(temp_runtime / "rag-cache"),
            "QDRANT_URL": "http://127.0.0.1:16333",
            "SEMANTIC_VERIFIER_MODE": "off",
            "SEMANTIC_VERIFIER_SEND_RAW_QUESTION": "false",
            "SEMANTIC_SHADOW_ENABLED": "false",
            "SEMANTIC_SHADOW_SAMPLE_RATE": "0",
            "SELLER_DIMENSION_ENABLED": "false",
            "SELLER_MULTI_SCOPE_ENABLED": "false",
            "MEM0_ENABLED": "false",
            "ABAC_DATABASE_ENABLED": "false",
            "MEMORY_PREFERENCE_PAYLOAD_KEY": secrets.token_hex(32),
            "OTEL_EXPORTER_OTLP_ENDPOINT": "",
            "MAX_LATENCY_MS": "600000",
        })
        # Qdrant is read-only during this suite (indexing buttons are not used).
        server_log_handle = backend_log_raw.open("w", encoding="utf-8")
        backend_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(backend_port)],
            cwd=ROOT, env=server_env, stdout=server_log_handle, stderr=subprocess.STDOUT,
        )
        try:
            _wait_ready(f"http://127.0.0.1:{backend_port}/health/ready", backend_process)
        finally:
            server_log_handle.close()

        frontend_env = dict(server_env)
        frontend_env.update({
            "E2E_SUITE": args.suite,
            "E2E_EXPANDED_CASE_IDS": ",".join(str(case["case_id"]) for case in selected_expanded_cases),
            "E2E_FRONTEND_PORT": str(frontend_port),
            "E2E_BASE_URL": f"http://127.0.0.1:{frontend_port}",
            "E2E_API_TARGET": f"http://127.0.0.1:{backend_port}",
            "E2E_ARTIFACT_DIR": str(artifact_dir),
            "E2E_JSON_REPORT": str(playwright_json),
            "E2E_PLATFORM_EMAIL": identities["platform"]["email"],
            "E2E_PLATFORM_PASSWORD": identities["platform"]["password"],
            "E2E_PLATFORM_DISPLAY_NAME": identities["platform"]["display_name"],
            "E2E_PLATFORM_USER_ID": identities["platform"]["user_id"],
            "E2E_SELLER_EMAIL": identities["seller"]["email"],
            "E2E_SELLER_PASSWORD": identities["seller"]["password"],
            "E2E_SELLER_DISPLAY_NAME": identities["seller"]["display_name"],
            "E2E_SELLER_ID": identities["seller"]["seller_id"],
            "E2E_SELLER_ANALYST_EMAIL": identities["seller_analyst"]["email"],
            "E2E_SELLER_ANALYST_PASSWORD": identities["seller_analyst"]["password"],
            "E2E_SELLER_ANALYST_DISPLAY_NAME": identities["seller_analyst"]["display_name"],
        })
        browser_command = [node, str(WEB / "node_modules" / "playwright" / "cli.js"), "test", "--config", "playwright.full-chain.config.ts"]
        max_failures = 1 if args.fail_fast else args.max_failures
        if max_failures is not None:
            browser_command.extend(["--max-failures", str(max_failures)])
        if args.grep and args.suite == "full":
            browser_command.extend(["--grep", args.grep])
        browser_return = _run_browser_process(
            browser_command,
            cwd=WEB,
            env=frontend_env,
            output_log=playwright_console_log,
        )

        _stop_process(backend_process)
        server_log_handle = None
        counts, failures = _playwright_results(playwright_json)
        playwright_audit = _playwright_audit(playwright_json)
        state["tests"] = {
            "counts": counts,
            "failures": failures,
            "playwright_exit_code": browser_return,
            "cases": playwright_audit.get("test_results", []),
            "http_errors": playwright_audit.get("http_errors", []),
            "http_error_classifications": dict(Counter(
                str(item.get("classification") or "unknown") for item in playwright_audit.get("http_errors", [])
            )),
            "trace_ids_by_test": playwright_audit.get("trace_ids_by_test", {}),
            "client_turn_ids_by_test": playwright_audit.get("client_turn_ids_by_test", {}),
            "missing_trace_tests": playwright_audit.get("missing_trace_tests", []),
            "plan_ids_by_test": playwright_audit.get("plan_ids_by_test", {}),
            "favicon_requests_by_test": playwright_audit.get("favicon_requests_by_test", {}),
            "audit_parse_errors": playwright_audit.get("parse_errors", []),
        }
        try:
            evidence = _database_evidence(database_url, identities, playwright_audit)
        except Exception as exc:
            evidence = {
                "query_error": f"{type(exc).__name__}: {str(exc)[:600]}",
                "provider_usage": {"calls": 0},
                "chat_turns_nonterminal": [],
                "chat_turns_awaiting_human_input": [],
                "chat_turns_awaiting_scope": [],
                "complex_plans_nonterminal": [],
            }
        state["database_evidence"] = evidence
        turn_status_map = evidence.get("turn_status_by_test") or {}
        status_expectation_failures = [
            {
                "test_title": test_title,
                "client_turn_ids": details.get("client_turn_ids", []),
                "expected_terminal_statuses": details.get("expected_terminal_statuses"),
                "status_checks": details.get("status_checks", []),
                "expectation_error": details.get("expectation_error"),
            }
            for test_title, details in turn_status_map.items()
            if details.get("status_match") is False
        ]
        unclassified_terminal_statuses = [
            test_title for test_title, details in turn_status_map.items()
            if details.get("expected_terminal_statuses") is None
        ]
        state["tests"]["terminal_status_expectation_failures"] = status_expectation_failures
        state["tests"]["unclassified_terminal_status_tests"] = unclassified_terminal_statuses
        if args.suite == "history":
            history_evidence = _history_resume_evidence(
                database_url, identities["seller"], playwright_audit,
            )
            metric_evidence = _history_metric_evidence(
                database_url, identities["platform"], playwright_audit,
            )
            state["history_resume_evidence"] = history_evidence
            state["history_metric_evidence"] = metric_evidence
            if playwright_audit.get("missing_trace_tests"):
                state["tests"]["failures"].append({
                    "test_id": "<history-trace-evidence>",
                    "reason": "Every history browser case must attach at least one trace ID",
                })
            if not history_evidence["passed"]:
                state["tests"]["failures"].append({
                    "test_id": "<history-resume-database-audit>",
                    "reason": "Browser premise resume did not reuse exactly one durable Turn and Execution",
                })
            if not metric_evidence["passed"]:
                state["tests"]["failures"].append({
                    "test_id": "<history-metric-database-audit>",
                    "reason": "Historical metric confirmation did not create exactly one distinct authorized Turn and Execution",
                })
        usage_count = int((evidence.get("provider_usage") or {}).get("calls", 0))
        state["llm"].update({
            "actual_provider_reported_usage_calls": usage_count,
            "provider_evidence": "not_required" if args.allow_no_llm and usage_count == 0 else ("verified" if usage_count > 0 else "missing"),
            "models": (evidence.get("provider_usage") or {}).get("models", []),
            "providers": (evidence.get("provider_usage") or {}).get("providers", []),
            "input_tokens": (evidence.get("provider_usage") or {}).get("input_tokens", 0),
            "output_tokens": (evidence.get("provider_usage") or {}).get("output_tokens", 0),
        })
        provider_requirement_met = usage_count > 0 or args.allow_no_llm
        history_requirement_met = args.suite != "history" or bool(
            state.get("history_resume_evidence", {}).get("passed")
            and state.get("history_metric_evidence", {}).get("passed")
            and not state.get("tests", {}).get("missing_trace_tests")
        )
        expected_status_cases = [
            details for details in turn_status_map.values()
            if details.get("expected_terminal_statuses") is not None
        ]
        turn_status_requirement_met = all(
            details.get("status_match") is True for details in expected_status_cases
        )
        state["status"] = "passed" if (
            browser_return == 0
            and provider_requirement_met
            and history_requirement_met
            and turn_status_requirement_met
        ) else "failed"
        if browser_return == 0 and usage_count == 0 and not args.allow_no_llm:
            state["tests"]["failures"].append({
                "test_id": "<real-provider-verification>",
                "reason": "No non-zero provider_reported token usage was recorded for either synthetic E2E identity",
            })
    except Exception as exc:
        state["status"] = "setup_failed"
        state["setup_error"] = f"{type(exc).__name__}: {str(exc)[:1200]}"
    finally:
        process_obj = locals().get("backend_process")
        _stop_process(process_obj)
        secrets_to_redact = [
            str(config.get("DEEPSEEK_API_KEY") or ""),
            str(config.get("TENANT_AUTH_SECRET") or ""),
            str(locals().get("identities", {}).get("platform", {}).get("password", "")),
            str(locals().get("identities", {}).get("seller", {}).get("password", "")),
            str(locals().get("identities", {}).get("seller_analyst", {}).get("password", "")),
        ]
        _safe_log(backend_log_raw, backend_log, secrets_to_redact)
        try:
            backend_log_raw.unlink(missing_ok=True)
        except OSError:
            pass
        if "server_log_handle" in locals() and server_log_handle and not server_log_handle.closed:
            server_log_handle.close()
        try:
            if shard_runner._database_exists(admin_url, clone):
                shard_runner._drop_clone(admin_url, clone)
            residual = shard_runner._database_exists(admin_url, clone)
            state["cleanup"] = {"clone_dropped": not residual, "residual": residual}
        except Exception as exc:
            state["cleanup"] = {"clone_dropped": False, "residual": True, "error": f"{type(exc).__name__}: {str(exc)[:600]}"}
        if state.get("cleanup", {}).get("residual"):
            state["status"] = "cleanup_failed"
        report_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"FULL_CHAIN_STATUS={state.get('status')}")
    print(f"HEAD={state.get('head')}")
    print(f"SUITE={args.suite}")
    if selected_expanded_cases:
        print("SELECTED_CASE_IDS=" + ",".join(str(case["case_id"]) for case in selected_expanded_cases))
    print(f"TEMPLATE={template}")
    print(f"DATABASE_CLONE={clone} RESIDUAL={str(state.get('cleanup', {}).get('residual')).lower()}")
    counts = state.get("tests", {}).get("counts", {})
    print("TEST_COUNTS=" + json.dumps(counts, sort_keys=True))
    print(f"REAL_LLM_PROVIDER_USAGE_CALLS={state.get('llm', {}).get('actual_provider_reported_usage_calls', 0)}")
    if args.suite == "history":
        history_cases = state.get("history_resume_evidence", {}).get("cases", [])
        print("HISTORY_RESUME_EVIDENCE=" + json.dumps(history_cases, ensure_ascii=False, sort_keys=True))
        metric_cases = state.get("history_metric_evidence", {}).get("cases", [])
        print("HISTORY_METRIC_EVIDENCE=" + json.dumps(metric_cases, ensure_ascii=False, sort_keys=True))
    evidence = state.get("database_evidence", {})
    print(f"NONTERMINAL_CHAT_TURNS={len(evidence.get('chat_turns_nonterminal', []))}")
    print(f"AWAITING_HUMAN_INPUT_CHAT_TURNS={len(evidence.get('chat_turns_awaiting_human_input', []))}")
    print(f"AWAITING_SCOPE_CHAT_TURNS={len(evidence.get('chat_turns_awaiting_scope', []))}")
    print(f"NONTERMINAL_COMPLEX_PLANS={len(evidence.get('complex_plans_nonterminal', []))}")
    print(f"TERMINAL_STATUS_CASES={len(evidence.get('turn_status_by_test', {}))}")
    print(f"UNCLASSIFIED_TERMINAL_STATUS_TESTS={len(state.get('tests', {}).get('unclassified_terminal_status_tests', []))}")
    print(f"TERMINAL_STATUS_EXPECTATION_FAILURES={len(state.get('tests', {}).get('terminal_status_expectation_failures', []))}")
    for failure in state.get("tests", {}).get("terminal_status_expectation_failures", []):
        print(f"FAILED_TURN_STATUS={failure.get('test_title')} EXPECTED={failure.get('expected_terminal_statuses')}")
    print("HTTP_ERROR_CLASSIFICATIONS=" + json.dumps(state.get("tests", {}).get("http_error_classifications", {}), sort_keys=True))
    for failure in state.get("tests", {}).get("failures", []):
        print(f"FAILED_TEST={failure.get('test_id')} REASON={str(failure.get('reason', '')).splitlines()[0][:500]}")
    if state.get("setup_error"):
        print(f"SETUP_ERROR={state['setup_error']}")
    print(f"REPORT={report_path}")
    return 0 if state.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

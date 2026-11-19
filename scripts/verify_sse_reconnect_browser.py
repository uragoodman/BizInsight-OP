# -*- coding: utf-8 -*-
"""Browser-level SSE disconnect/reconnect acceptance for durable chat.

Starts a disposable PostgreSQL database and a real FastAPI process, runs two
Chromium scenarios (active reconnect while the original stream stays alive;
true browser abort then reconnect), and verifies the committed Turn in
PostgreSQL.

This script does not change the application, migrations or API behaviour.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
ADMIN_URL = os.getenv(
    "ISOLATED_TEST_POSTGRES_ADMIN_URL",
    "postgresql+psycopg://postgres:abc123@localhost:5432/postgres",
)
PYTHON = sys.executable
PORT = int(os.getenv("SSE_VERIFY_PORT", "28080"))


def _run(command: list[str], *, env: dict[str, str], cwd: Path = ROOT) -> None:
    completed = subprocess.run(
        command, cwd=cwd, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({' '.join(command)}):\n{completed.stdout[-3000:]}\n{completed.stderr[-3000:]}"
        )


def _wait_health(url: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health/live", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.5)
    raise RuntimeError(f"API did not become healthy at {url}")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - best effort on consoles without reconfigure
            pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default="")
    parser.add_argument("--keep-database", action="store_true")
    parser.add_argument("--keep-artifacts", action="store_true")
    args = parser.parse_args()

    suffix = uuid.uuid4().hex[:10]
    database = f"bizinsight_sse_{suffix}"
    admin = create_engine(ADMIN_URL, isolation_level="AUTOCOMMIT")
    clone_url = ADMIN_URL.rsplit("/", 1)[0] + f"/{database}"
    temp_dir = Path(tempfile.mkdtemp(prefix="sse_reconnect_"))
    api_log = temp_dir / "api.log"
    output_a = temp_dir / "sse-reconnect-a.json"
    output_b = temp_dir / "sse-reconnect-b.json"
    api_process: subprocess.Popen | None = None
    exit_code = 1
    try:
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)'))
            connection.execute(text(f'CREATE DATABASE "{database}"'))

        env = {
            **os.environ,
            "POSTGRES_URL": clone_url,
            "TENANT_AUTH_REQUIRED": "false",
            "USER_ID": "sse-eval-user",
            "DB_USE_RLS_ROLE": "false",
            "MEM0_ENABLED": "false",
            "RELATION_RESOLVER_LLM_ENABLED": "false",
            "TRACE_BACKEND": "file",
            "TRACE_DIR": str(temp_dir / "traces"),
            "CHECKPOINT_BACKEND": "sqlite",
            "PYTHONPATH": str(ROOT),
        }
        _run([PYTHON, "scripts/seed_phase6_ci_postgres.py"], env=env)
        _run([PYTHON, "-m", "alembic", "upgrade", "head"], env=env)
        _run([PYTHON, "-m", "scripts.seed_phase7_ci_postgres"], env=env)
        seed_engine = create_engine(clone_url)
        try:
            with seed_engine.begin() as connection:
                connection.execute(text("""
                    INSERT INTO iam.tenants(tenant_id, tenant_name, scope_type, status)
                    VALUES ('platform', 'SSE reconnect evaluation', 'platform', 'active')
                    ON CONFLICT (tenant_id) DO NOTHING
                """))
                connection.execute(text("""
                    INSERT INTO iam.users(user_id, display_name, email, status)
                    VALUES ('sse-eval-user', 'SSE reconnect evaluation',
                            'sse-eval-user@example.invalid', 'active')
                    ON CONFLICT (user_id) DO NOTHING
                """))
                connection.execute(text("""
                    INSERT INTO iam.tenant_memberships(tenant_id, user_id, role)
                    VALUES ('platform', 'sse-eval-user', 'admin')
                    ON CONFLICT (tenant_id, user_id) DO NOTHING
                """))
        finally:
            seed_engine.dispose()

        api_process = subprocess.Popen(
            [PYTHON, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(PORT)],
            cwd=ROOT, env=env,
            stdout=api_log.open("w", encoding="utf-8"), stderr=subprocess.STDOUT,
        )
        base_url = f"http://127.0.0.1:{PORT}"
        _wait_health(base_url)

        playwright_env = {
            **os.environ,
            "SSE_BACKEND_URL": base_url,
            "SSE_OUTPUT_A": str(output_a),
            "SSE_OUTPUT_B": str(output_b),
        }
        playwright_cli = ROOT / "web" / "node_modules" / ".bin" / (
            "playwright.cmd" if os.name == "nt" else "playwright"
        )
        completed = subprocess.run(
            [str(playwright_cli), "test", "tests/e2e/sse-reconnect.spec.ts",
             "-c", "playwright.sse.config.ts", "--reporter=list"],
            cwd=ROOT / "web", env=playwright_env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=420,
        )
        if completed.returncode != 0:
            stream_output = ""
            for path in (output_a, output_b):
                if path.exists():
                    stream_output += path.read_text(encoding="utf-8", errors="replace")[-3000:]
            raise RuntimeError(
                "Playwright reconnect test failed:\n"
                f"stream output:\n{stream_output}\n"
                f"{completed.stdout[-6000:]}\n{completed.stderr[-4000:]}"
            )

        playwright_summary = completed.stdout[-2000:]
        result_a = json.loads(output_a.read_text(encoding="utf-8"))
        result_b = json.loads(output_b.read_text(encoding="utf-8"))
        reconnect_a = result_a["second"]
        completed_a = next(event for event in result_a["first"] if event.get("event") == "completed")
        completed_b = next(event for event in reconnect_a if event.get("event") == "completed")
        heartbeats_a = [event for event in reconnect_a if event.get("event") == "heartbeat"]
        intermediate_a = [
            event.get("event") for event in reconnect_a
            if event.get("event") in {"node_started", "node_completed", "plan"}
        ]
        reconnect_b_events = result_b["reconnect"]["events"]
        completed_disconnect = next(
            (event for event in reconnect_b_events if event.get("event") == "completed"), None,
        )
        heartbeats_b = [event for event in reconnect_b_events if event.get("event") == "heartbeat"]

        engine = create_engine(clone_url)
        try:
            with engine.connect() as connection:
                def turn_row(client_turn_id: str):
                    return connection.execute(text("""
                        SELECT turn_id, status, final_execution_id, commit_version
                          FROM governance.chat_turns
                         WHERE tenant_id='platform' AND client_turn_id=:client
                    """), {"client": client_turn_id}).mappings().one_or_none()

                row_a = turn_row(result_a["body"]["chat_turn_id"])
                row_b = turn_row(result_b["body"]["chat_turn_id"])
                if row_a is None:
                    raise RuntimeError("scenario A durable Turn was not persisted")
                if row_b is None:
                    raise RuntimeError("scenario B durable Turn was not persisted")
                exec_a = connection.execute(text("""
                    SELECT count(*) FROM governance.conversation_executions WHERE turn_id=:turn
                """), {"turn": row_a["turn_id"]}).scalar_one()
                exec_b = connection.execute(text("""
                    SELECT count(*) FROM governance.conversation_executions WHERE turn_id=:turn
                """), {"turn": row_b["turn_id"]}).scalar_one()
        finally:
            engine.dispose()

        mismatches = []
        if row_a["status"] != "completed":
            mismatches.append(f"scenario A status={row_a['status']}")
        if row_a["turn_id"] != completed_b.get("durable_turn_id"):
            mismatches.append("scenario A durable_turn_id mismatch")
        if int(row_a["commit_version"] or 0) != int(completed_b.get("commit_version") or -1):
            mismatches.append("scenario A commit_version mismatch")
        if str(row_a["final_execution_id"] or "") != str(completed_b.get("execution_id") or ""):
            mismatches.append("scenario A execution_id mismatch")
        if exec_a != 1:
            mismatches.append(f"scenario A execution_count={exec_a}")
        if not heartbeats_a:
            mismatches.append("scenario A no heartbeat observed")
        if intermediate_a:
            mismatches.append(f"scenario A intermediate events replayed: {intermediate_a}")
        if exec_b != 1:
            mismatches.append(f"scenario B execution_count={exec_b}")
        if row_b["status"] != "completed":
            mismatches.append(f"scenario B status={row_b['status']}")
        if completed_disconnect is None:
            mismatches.append("scenario B produced no completed event")
        else:
            if row_b["turn_id"] != completed_disconnect.get("durable_turn_id"):
                mismatches.append("scenario B durable_turn_id mismatch")
            if int(row_b["commit_version"] or 0) != int(completed_disconnect.get("commit_version") or -1):
                mismatches.append("scenario B commit_version mismatch")
            if str(row_b["final_execution_id"] or "") != str(completed_disconnect.get("execution_id") or ""):
                mismatches.append("scenario B execution_id mismatch")
        if bool(result_b["reconnect"].get("timedOut")):
            mismatches.append("scenario B reconnect timed out instead of observing the commit")
        if mismatches:
            raise RuntimeError("SSE reconnect contract mismatch: " + "; ".join(mismatches))

        evidence = {
            "schema_version": "sse-reconnect-acceptance-v1",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "backend": base_url,
            "playwright_summary": playwright_summary,
            "scenario_a_active_reconnect": {
                "chat_session_id": result_a["body"]["chat_session_id"],
                "client_turn_id": result_a["body"]["chat_turn_id"],
                "durable_turn_id": row_a["turn_id"],
                "execution_id": row_a["final_execution_id"],
                "commit_version": row_a["commit_version"],
                "execution_count": exec_a,
                "first_stream_event_names": [event.get("event") for event in result_a["first"]],
                "reconnect_event_names": [event.get("event") for event in reconnect_a],
                "heartbeat_count": len(heartbeats_a),
                "intermediate_events_replayed": intermediate_a,
                "result": "PASS",
                "contract": (
                    "heartbeat while running, one completed after Turn Commit; "
                    "intermediate node events are not replayed"
                ),
            },
            "scenario_b_true_disconnect": {
                "chat_session_id": result_b["body"]["chat_session_id"],
                "client_turn_id": result_b["body"]["chat_turn_id"],
                "durable_turn_id": row_b["turn_id"],
                "turn_status": row_b["status"],
                "execution_count": exec_b,
                "reconnect_event_names": [event.get("event") for event in reconnect_b_events],
                "heartbeat_count": len(heartbeats_b),
                "execution_id": row_b["final_execution_id"],
                "commit_version": row_b["commit_version"],
                "completed_event": completed_disconnect,
                "reconnect_timed_out": bool(result_b["reconnect"].get("timedOut")),
                "result": "PASS",
                "contract": (
                    "the graph worker continues after the browser aborts and commits the Turn; "
                    "reconnect observes the committed result without a second execution"
                ),
            },
            "overall": "PASS: active reconnect and true browser disconnect recovery both observed",
        }
        evidence_text = json.dumps(evidence, ensure_ascii=False, indent=2)
        print("SSE_RECONNECT_OK " + evidence_text)
        if args.evidence:
            Path(args.evidence).parent.mkdir(parents=True, exist_ok=True)
            Path(args.evidence).write_text(evidence_text + "\n", encoding="utf-8")
        exit_code = 0
    finally:
        if api_process is not None:
            api_process.terminate()
            try:
                api_process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                api_process.kill()
        if not args.keep_database:
            with admin.connect() as connection:
                connection.execute(text(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)'))
        admin.dispose()
        if exit_code != 0:
            print(f"API log: {api_log}", file=sys.stderr)
            if api_log.exists():
                print(api_log.read_text(encoding="utf-8", errors="replace")[-4000:], file=sys.stderr)
        if exit_code != 0 or args.keep_artifacts:
            print(f"artifacts kept: {temp_dir}", file=sys.stderr)
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

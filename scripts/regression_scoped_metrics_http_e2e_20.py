"""Run the 20 scoped-metric regressions through the real HTTP/SSE API.

Run inside the API container (so application settings and the DB connection are
identical to the server):

    python scripts/regression_scoped_metrics_http_e2e_20.py

Each case gets an independent session/turn. The trusted expected series is
fetched through the authenticated data-explorer HTTP endpoint; the question
itself is submitted only through POST /chat/stream. The script then checks the
SSE terminal event, persisted trace, chat-turn status, and execution status.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import time
from datetime import datetime, timezone
from uuid import uuid4

import requests
from sqlalchemy import text

from app.core.config import get_settings
from app.data.database import engine
from app.data.rls import set_local_context
from app.tenancy import TenantContext
from scripts.regression_scoped_metrics_10 import CASES as CASES_ONE, SELLERS as SELLERS_ONE
from scripts.regression_scoped_metrics_10_v2 import CASES as CASES_TWO, SELLERS as SELLERS_TWO


START_DATE = "2017-09-01"
END_DATE = "2018-08-31"
BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = (15, 600)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _jwt(user_id: str) -> str:
    settings = get_settings()
    if not settings.tenant_auth_secret:
        raise RuntimeError("TENANT_AUTH_SECRET is not configured in the API container")
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    claims = _b64url(json.dumps({
        "tenant_id": "platform",
        "sub": user_id,
        "scope_type": "platform",
        "role": "admin",
        "exp": int(time.time()) + 6 * 60 * 60,
    }, separators=(",", ":")).encode())
    signing_input = f"{header}.{claims}"
    signature = hmac.new(
        settings.tenant_auth_secret.encode(), signing_input.encode("ascii"), hashlib.sha256
    ).digest()
    return f"{signing_input}.{_b64url(signature)}"


def _json(response: requests.Response, label: str) -> dict:
    if response.status_code != 200:
        raise RuntimeError(f"{label}: HTTP {response.status_code}: {response.text[:600]}")
    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError(f"{label}: invalid JSON: {response.text[:300]}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"{label}: expected JSON object")
    return result


def _series(payload: dict) -> list[dict]:
    return list(payload.get("series") or [])


def _tool_points(trace: dict) -> list[dict]:
    state = trace.get("state_snapshot") or {}
    result = state.get("tool_result") or {}
    nested = result.get("result") or {}
    trend = nested.get("trend_result") or {}
    return list(trend.get("points") or [])


def _points_equal(expected: list[dict], actual: list[dict]) -> bool:
    if len(expected) != len(actual):
        return False
    for left, right in zip(expected, actual):
        left_bucket = str(left.get("bucket") or left.get("metric_date") or "")
        right_bucket = str(right.get("bucket") or right.get("metric_date") or "")
        try:
            values_equal = math.isclose(float(left["value"]), float(right["value"]), rel_tol=1e-9, abs_tol=1e-8)
        except (KeyError, TypeError, ValueError):
            return False
        if left_bucket != right_bucket or not values_equal:
            return False
    return True


def _cases() -> list[dict]:
    all_cases: list[dict] = []
    for index, ((metric, grain, question), seller) in enumerate(zip(CASES_ONE, SELLERS_ONE), start=1):
        all_cases.append({
            "case": index, "group": 1, "metric": metric, "grain": grain,
            "question": question, "seller_id": seller,
        })
    for index, ((metric, question), seller) in enumerate(zip(CASES_TWO, SELLERS_TWO), start=11):
        all_cases.append({
            "case": index, "group": 2, "metric": metric, "grain": "month",
            "question": question, "seller_id": seller,
        })
    if len(all_cases) != 20:
        raise AssertionError(f"Expected 20 cases, got {len(all_cases)}")
    return all_cases


def _read_sse(response: requests.Response, started_at: float) -> tuple[list[dict], dict | None]:
    response.encoding = "utf-8"
    events: list[dict] = []
    terminal = None
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        try:
            event = json.loads(line[5:].strip())
        except (ValueError, TypeError):
            continue
        events.append({"elapsed_ms": round((time.perf_counter() - started_at) * 1000), "event": event})
        if event.get("event") in {"completed", "failed"}:
            terminal = event
            break
    return events, terminal


def _verify_execution_rows(context: TenantContext, trace_ids: list[str]) -> dict[str, str]:
    if not trace_ids:
        return {}
    with engine.begin() as connection:
        set_local_context(connection, context)
        rows = connection.execute(text("""
            SELECT e.trace_id, e.status AS execution_status, t.status AS turn_status
            FROM governance.conversation_executions e
            JOIN governance.chat_turns t ON t.turn_id=e.turn_id
            WHERE e.trace_id = ANY(:trace_ids)
        """), {"trace_ids": trace_ids}).mappings().all()
    return {
        str(row["trace_id"]): f"{row['execution_status']}/{row['turn_status']}"
        for row in rows
    }


def main() -> int:
    user_id = f"http-e2e-20-{uuid4().hex[:20]}"
    headers = {
        "Authorization": f"Bearer {_jwt(user_id)}",
        "Accept": "application/json, text/event-stream",
    }
    cases = _cases()
    results: list[dict] = []
    session = requests.Session()
    session.headers.update(headers)

    health = _json(session.get(f"{BASE_URL}/health/ready", timeout=20), "readiness")
    print(f"HTTP readiness: {health.get('status', 'ok')}", flush=True)

    for case in cases:
        case_started = time.perf_counter()
        session_id = f"http-e2e-{uuid4().hex}"
        client_turn_id = f"http-e2e-turn-{uuid4().hex}"
        explorer = {
            "seller_id": case["seller_id"],
            "metric_name": case["metric"],
            "start_date": START_DATE,
            "end_date": END_DATE,
            "granularity": case["grain"],
        }
        record: dict = {**case, "session_id": session_id, "client_turn_id": client_turn_id}
        try:
            preview = _json(
                session.post(f"{BASE_URL}/data-explorer/preview", json=explorer, timeout=60),
                f"case {case['case']} trusted preview",
            )
            expected = _series(preview)
            if not expected:
                raise RuntimeError("trusted HTTP preview returned no points")
            if (preview.get("context") or {}).get("metric_name") != case["metric"]:
                raise RuntimeError("trusted preview resolved a different metric")
            record["expected_points"] = len(expected)

            body = {
                "message": case["question"],
                "use_current_explorer_context": True,
                "explorer_context": explorer,
                "memory_enabled": False,
                "chat_session_id": session_id,
                "chat_turn_id": client_turn_id,
            }
            started = time.perf_counter()
            with session.post(
                f"{BASE_URL}/chat/stream", json=body, stream=True, timeout=REQUEST_TIMEOUT,
            ) as response:
                record["http_status"] = response.status_code
                if response.status_code != 200:
                    raise RuntimeError(f"chat stream HTTP {response.status_code}: {response.text[:600]}")
                events, terminal = _read_sse(response, started)
            record["events"] = events
            record["event_count"] = len(events)
            record["terminal"] = terminal
            if not terminal or terminal.get("event") != "completed":
                raise RuntimeError(f"stream has no completed event: {terminal}")
            trace_id = str(terminal.get("trace_id") or "")
            record["trace_id"] = trace_id
            record["answer"] = terminal.get("answer") or ""
            if not trace_id:
                raise RuntimeError("terminal SSE event has no trace_id")
            if terminal.get("error") or terminal.get("clarification"):
                raise RuntimeError(f"terminal SSE is error/clarification: {terminal}")

            trace = _json(session.get(f"{BASE_URL}/traces/{trace_id}", timeout=30), f"case {case['case']} trace")
            state = trace.get("state_snapshot") or {}
            decision = state.get("metric_planning_decision") or {}
            actual = _tool_points(trace)
            record["decision_status"] = decision.get("status")
            record["actual_points"] = len(actual)
            record["graph_path"] = trace.get("graph_path") or []
            record["points_match_preview"] = _points_equal(expected, actual)
            record["answer_contains_answer"] = bool(record["answer"].strip())
            if decision.get("status") != "confirmed":
                raise RuntimeError(f"metric decision not confirmed: {decision}")
            if not record["points_match_preview"]:
                raise RuntimeError("chat query points differ from trusted HTTP preview")
            if not record["answer_contains_answer"]:
                raise RuntimeError("completed SSE event contains an empty answer")

            turns = _json(
                session.get(f"{BASE_URL}/chat/sessions/{session_id}/turns", timeout=30),
                f"case {case['case']} persisted turn",
            )
            turn_items = turns.get("items") or []
            matching = [item for item in turn_items if item.get("trace_id") == trace_id]
            record["turn_status"] = matching[0].get("status") if matching else None
            if record["turn_status"] != "completed":
                raise RuntimeError(f"chat turn did not persist as completed: {record['turn_status']!r}")
            record["status"] = "passed"
        except Exception as exc:  # noqa: BLE001 - continue suite and preserve per-case evidence
            record["status"] = "failed"
            record["failure"] = f"{type(exc).__name__}: {exc}"
        record["elapsed_ms"] = round((time.perf_counter() - case_started) * 1000)
        results.append(record)
        print(
            f"[{record['status'].upper()}] {case['case']:02d}/20 "
            f"{case['metric']} / {case['grain']} "
            f"HTTP={record.get('http_status')} "
            f"points={record.get('actual_points', 0)}/{record.get('expected_points', 0)} "
            f"turn={record.get('turn_status')} trace={record.get('trace_id', '-') } "
            f"{record.get('failure', '')}",
            flush=True,
        )

    trace_ids = [str(item["trace_id"]) for item in results if item.get("trace_id")]
    try:
        statuses = _verify_execution_rows(TenantContext(
            tenant_id="platform", user_id=user_id, role="admin", scope_type="platform",
        ), trace_ids)
    except Exception as exc:  # keep HTTP results, but make persistence verification explicit
        statuses = {}
        db_verification_error = f"{type(exc).__name__}: {exc}"
    else:
        db_verification_error = None
    for item in results:
        item["execution_turn_db_status"] = statuses.get(str(item.get("trace_id") or ""))
        if item.get("status") == "passed" and item["execution_turn_db_status"] != "completed/completed":
            item["status"] = "failed"
            item["failure"] = f"DB execution/turn state is {item['execution_turn_db_status']!r}"

    artifact = {
        "schema_version": "1.0",
        "suite": "scoped_metric_http_sse_e2e_20",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "api_base_url": BASE_URL,
        "case_count": len(results),
        "passed": sum(item.get("status") == "passed" for item in results),
        "failed": sum(item.get("status") != "passed" for item in results),
        "db_verification_error": db_verification_error,
        "results": results,
    }
    print(json.dumps({
        "suite": artifact["suite"], "passed": artifact["passed"], "failed": artifact["failed"],
        "db_verification_error": db_verification_error,
        "cases": [{key: item.get(key) for key in (
            "case", "metric", "grain", "status", "http_status", "decision_status",
            "expected_points", "actual_points", "points_match_preview", "turn_status",
            "execution_turn_db_status", "trace_id", "elapsed_ms", "failure",
        )} for item in results],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0 if artifact["passed"] == 20 else 1


if __name__ == "__main__":
    raise SystemExit(main())

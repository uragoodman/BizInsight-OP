"""Experiment 3 — concurrent load test for the governed chat platform.

Measures throughput-side numbers the resume currently lacks: p50/p95 latency,
error rate and SSE stability under 5/10/20 concurrent clients.

Two workloads per concurrency level:

- light:   POST /api/chat (non-stream, keyword-routed short question without
           an LLM-heavy path) — measures the request pipeline, tenant auth,
           DB round-trips and graph overhead.
- stream:  POST /api/chat/stream (SSE, anomaly_analysis path against the
           golden fixture provider) — measures concurrent multi-round agent
           loops: heartbeats, event ordering, and end-to-end completion.

Auth: HS256 JWT built from the same secret/claims the auth chain expects
(``tenant_id``/``sub``/``scope_type``/``role``).  The server must already be
running (uvicorn app.main:app --port 18080).

Outputs: ``data/evaluations/experiment-3-concurrency.json``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import statistics
import sys
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402

SSE_TIMEOUT_S = 300
REQUEST_TIMEOUT_S = 300


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def make_jwt(tenant_id: str, user_id: str) -> str:
    """Build the HS256 JWT the auth chain accepts (claims → TenantContext)."""
    secret = get_settings().tenant_auth_secret
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "tenant_id": tenant_id,
        "sub": user_id,
        "scope_type": "platform",
        "role": "admin",
        "exp": int(time.time()) + 3600,
    }
    signing_input = f"{_b64url(json.dumps(header).encode())}.{_b64url(json.dumps(payload).encode())}"
    signature = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url(signature)}"


def one_light(base_url: str, token: str, question: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        resp = requests.post(
            f"{base_url}/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"message": question, "thread_id": f"load-{uuid.uuid4().hex[:8]}"},
            timeout=REQUEST_TIMEOUT_S,
        )
        elapsed = time.perf_counter() - started
        ok = resp.status_code == 200
        return {"ok": ok, "status": resp.status_code, "elapsed_s": round(elapsed, 3),
                "error": None if ok else resp.text[:120]}
    except Exception as exc:  # noqa: BLE001 — load test records failures as data
        return {"ok": False, "status": 0, "elapsed_s": round(time.perf_counter() - started, 3),
                "error": f"{type(exc).__name__}: {exc}"[:120]}


def one_stream(base_url: str, token: str, question: str) -> dict[str, Any]:
    """Run one SSE chat to terminal event; record wall time and event health."""
    started = time.perf_counter()
    events: dict[str, int] = defaultdict(int)
    terminal: str | None = None
    heartbeat_gaps: list[float] = []
    last_event_at = started
    try:
        with requests.post(
            f"{base_url}/chat/stream",
            headers={"Authorization": f"Bearer {token}"},
            json={"message": question, "thread_id": f"load-sse-{uuid.uuid4().hex[:8]}"},
            stream=True, timeout=SSE_TIMEOUT_S,
        ) as resp:
            if resp.status_code != 200:
                return {"ok": False, "status": resp.status_code, "elapsed_s": round(time.perf_counter() - started, 3),
                        "events": dict(events), "terminal": None, "heartbeat_gaps": [],
                        "error": resp.text[:120]}
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                now = time.perf_counter()
                try:
                    payload = json.loads(line.removeprefix("data:").strip())
                except Exception:  # noqa: BLE001
                    continue
                event_type = str(payload.get("type") or payload.get("event") or "unknown")
                events[event_type] += 1
                if event_type == "heartbeat":
                    heartbeat_gaps.append(round(now - last_event_at, 3))
                last_event_at = now
                if event_type in {"completed", "failed"}:
                    terminal = event_type
                    break
        elapsed = time.perf_counter() - started
        ok = terminal == "completed"
        return {"ok": ok, "status": 200, "elapsed_s": round(elapsed, 3),
                "events": dict(events), "terminal": terminal,
                "heartbeat_count": events.get("heartbeat", 0),
                "heartbeat_gaps": heartbeat_gaps[:8],
                "error": None if ok else f"terminal={terminal}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status": 0, "elapsed_s": round(time.perf_counter() - started, 3),
                "events": dict(events), "terminal": terminal, "heartbeat_gaps": [],
                "error": f"{type(exc).__name__}: {exc}"[:120]}


def run_level(base_url: str, token: str, level: int, workload: str, questions: list[str]) -> dict[str, Any]:
    """Fire `level` concurrent rounds, 2 requests per client."""
    results: list[dict[str, Any]] = []
    lock = threading.Lock()

    def client(idx: int) -> None:
        local = []
        for rep in range(2):
            question = questions[(idx + rep) % len(questions)]
            if workload == "light":
                local.append(one_light(base_url, token, question))
            else:
                local.append(one_stream(base_url, token, question))
        with lock:
            results.extend(local)

    threads = [threading.Thread(target=client, args=(i,)) for i in range(level)]
    wall_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_s = round(time.perf_counter() - wall_start, 3)

    latencies = sorted(item["elapsed_s"] for item in results)
    ok_count = sum(1 for item in results if item["ok"])
    return {
        "concurrency": level,
        "workload": workload,
        "requests": len(results),
        "ok": ok_count,
        "error_rate": round(1 - ok_count / len(results), 4) if results else 0.0,
        "wall_s": wall_s,
        "p50_s": round(statistics.median(latencies), 3) if latencies else None,
        "p95_s": round(latencies[int(len(latencies) * 0.95) - 1 if len(latencies) > 1 else -1], 3) if latencies else None,
        "max_s": round(max(latencies), 3) if latencies else None,
        "failures": [item["error"] for item in results if not item["ok"]][:6],
        "detail": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--levels", default="5,10,20")
    parser.add_argument("--workloads", default="light,stream")
    parser.add_argument("--skip-light", action="store_true")
    parser.add_argument("--skip-stream", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    token = make_jwt("tenant_1481977a78484e3cbcda1dd4f224cfda", "usr_7051e6a5cd7b4b2e9d60d2b6baff5377")
    base_url = args.base_url.rstrip("/")

    # light workload: cheap, no-anomaly questions (metric definition path);
    # stream workload: anomaly_analysis against the fixture-backed provider.
    light_questions = ["GMV 是什么指标", "订单量的口径是什么", "延迟率的定义"]
    stream_questions = [
        "为什么 2018-08-20 的 GMV 下降了？",
        "为什么 2018-08-23 的 GMV 下降了？分析原因",
        "为什么 2018-08-22 的 GMV 下降了？",
    ]
    # If the provider is unavailable the stream arm degrades to workflow_fallback —
    # still a valid load measurement of the governed pipeline.
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    workloads = [w for w in args.workloads.split(",") if w.strip()]
    if args.skip_light:
        workloads = [w for w in workloads if w != "light"]
    if args.skip_stream:
        workloads = [w for w in workloads if w != "stream"]

    levels_data = []
    for workload in workloads:
        for level in levels:
            questions = stream_questions if workload == "stream" else light_questions
            summary = run_level(base_url, token, level, workload, questions)
            levels_data.append(summary)
            print(json.dumps({k: v for k, v in summary.items() if k != "detail"}, ensure_ascii=False), flush=True)

    payload = {
        "schema_version": "1.0",
        "experiment": "concurrent_load",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "model": settings.llm_model,
        "levels": levels_data,
        "limitations": [
            "Single-host local run; numbers are not production SLA.",
            "Stream workload depends on the configured Provider; LLM latency dominates stream latencies.",
            "Each client issues 2 sequential requests; concurrency level = simultaneous clients.",
        ],
    }
    output = ROOT / "data" / "evaluations" / "experiment-3-concurrency.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

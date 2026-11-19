"""Run Postman-like black-box checks against a running BizInsight API.

Usage (PowerShell):
    $env:RUN_HTTP_SMOKE = "1"
    python scripts/api_smoke.py --chat

The script intentionally uses HTTP only.  It does not import application
internals, so a passing run verifies the same boundary used by the browser.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, timedelta
from typing import Any
from uuid import uuid4

import httpx


DEFAULT_SELLER_ID = "1f50f920176fa81dab994f9023523100"
DEFAULT_METRIC = "seller_avg_freight_per_order"


class SmokeFailure(RuntimeError):
    pass


def _headers() -> dict[str, str]:
    headers: dict[str, str] = {"Accept": "application/json"}
    token = os.getenv("BIZINSIGHT_API_TOKEN", "").strip()
    if token:
        headers["Authorization"] = token if token.lower().startswith("bearer ") else f"Bearer {token}"
    api_key = os.getenv("BIZINSIGHT_API_KEY", "").strip()
    if api_key:
        headers["x-api-key"] = api_key
    analysis_token = os.getenv("BIZINSIGHT_ANALYSIS_TOKEN", "").strip()
    actor_id = os.getenv("BIZINSIGHT_ACTOR_ID", "").strip()
    if analysis_token:
        headers["x-analysis-token"] = analysis_token
    if actor_id:
        headers["x-actor-id"] = actor_id
    return headers


def _json(response: httpx.Response, label: str, expected: set[int] = {200}) -> dict[str, Any]:
    if response.status_code not in expected:
        raise SmokeFailure(f"{label}: HTTP {response.status_code}: {response.text[:500]}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise SmokeFailure(f"{label}: response is not JSON: {response.text[:300]}") from exc
    if not isinstance(payload, dict):
        raise SmokeFailure(f"{label}: expected JSON object, got {type(payload).__name__}")
    return payload


def _call(client: httpx.Client, method: str, path: str, label: str, **kwargs: Any) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.request(method, path, **kwargs)
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    payload = _json(response, label)
    print(f"[PASS] {label:<28} HTTP {response.status_code}  {elapsed_ms} ms")
    return payload


def run(*, base_url: str, seller_id: str, run_chat: bool) -> int:
    base_url = base_url.rstrip("/")
    timeout = float(os.getenv("BIZINSIGHT_HTTP_SMOKE_TIMEOUT", "180"))
    headers = _headers()
    with httpx.Client(base_url=base_url, headers=headers, timeout=timeout) as client:
        health = _call(client, "GET", "/health/live", "health")
        if health.get("status") != "ok":
            raise SmokeFailure(f"health: unexpected payload {health}")

        frontend = client.get("/explorer")
        if frontend.status_code != 200 or "<html" not in frontend.text.lower() or "/assets/" not in frontend.text:
            raise SmokeFailure(f"built frontend delivery: HTTP {frontend.status_code}; expected SPA HTML with assets")
        print(f"[PASS] {'built frontend delivery':<28} HTTP {frontend.status_code}")

        bounds = _call(client, "GET", "/data-explorer/time-bounds", "data time bounds")
        start_date = os.getenv("BIZINSIGHT_SMOKE_START_DATE", "2017-08-31")
        end_date = os.getenv("BIZINSIGHT_SMOKE_END_DATE", "2018-08-31")
        if bounds.get("min_date") and start_date < str(bounds["min_date"]):
            start_date = str(bounds["min_date"])
        if bounds.get("max_date") and end_date > str(bounds["max_date"]):
            end_date = str(bounds["max_date"])
        if start_date > end_date:
            raise SmokeFailure(f"scope dates are outside data bounds: {start_date} > {end_date}")

        sellers = _call(client, "GET", "/data-explorer/sellers?limit=1", "seller lookup")
        if not sellers.get("items"):
            raise SmokeFailure("seller lookup: no seller is available for the smoke test")

        preview_body = {
            "seller_id": seller_id,
            "metric_name": DEFAULT_METRIC,
            "start_date": start_date,
            "end_date": end_date,
            "granularity": "day",
        }
        preview = _call(client, "POST", "/data-explorer/preview", "seller metric preview", json=preview_body)
        context = preview.get("context") or {}
        if context.get("metric_name") != DEFAULT_METRIC:
            raise SmokeFailure(f"seller metric preview: unexpected metric {context.get('metric_name')!r}")

        experiment_scope = {key: preview_body[key] for key in ("seller_id", "start_date", "end_date")}
        eligible = _call(
            client, "POST", "/causal-experiments/eligible", "causal experiment discovery",
            json=experiment_scope,
        )
        if not isinstance(eligible.get("items"), list):
            raise SmokeFailure("causal experiment discovery: missing items list")
        print(f"[INFO] eligible registered experiments={len(eligible['items'])}")
        selected_id = os.getenv("BIZINSIGHT_SMOKE_EXPERIMENT_ID", "").strip()
        if selected_id:
            if selected_id not in {str(item.get("experiment_id")) for item in eligible["items"]}:
                raise SmokeFailure("configured experiment ID is not eligible in the current authorized scope")
            estimate = _call(
                client, "POST", "/causal-experiments/estimate", "causal ITT estimate",
                json={**experiment_scope, "experiment_id": selected_id},
            )
            if estimate.get("status") not in {"ok", "not_available"}:
                raise SmokeFailure(f"causal ITT estimate: unexpected status {estimate.get('status')!r}")

        if run_chat:
            chat_body = {
                "message": "帮我计算平均每单运费的均值、最大值和最小值，并说明差异。",
                "use_current_explorer_context": True,
                "explorer_context": preview_body,
                "memory_enabled": False,
                "chat_session_id": f"http-smoke-{uuid4().hex}",
            }
            chat = _call(client, "POST", "/chat", "scoped statistics chat", json=chat_body)
            if not chat.get("trace_id"):
                raise SmokeFailure("scoped statistics chat: missing trace_id")
            answer = str(chat.get("answer") or "")
            if "内部异常" in answer or "KeyError" in answer:
                raise SmokeFailure(f"scoped statistics chat: backend failure in answer: {answer[:500]}")
            print(f"[INFO] trace_id={chat['trace_id']}")

    print("[PASS] HTTP smoke suite completed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("BIZINSIGHT_API_BASE_URL", "http://127.0.0.1:28080"))
    parser.add_argument("--seller-id", default=os.getenv("BIZINSIGHT_SMOKE_SELLER_ID", DEFAULT_SELLER_ID))
    parser.add_argument("--chat", action="store_true", help="also call the real /chat workflow and LLM")
    args = parser.parse_args()
    try:
        return run(base_url=args.base_url, seller_id=args.seller_id, run_chat=args.chat)
    except (httpx.HTTPError, SmokeFailure) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

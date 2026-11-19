"""Record one real HTTP/SSE governed-agent request and its persisted trace.

The script deliberately does not write the JWT to disk.  It captures transport
events, terminal business status, trace evidence, health checks, and Qdrant
point counts so a full-chain run is reviewable after the request finishes.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from app.core.config import get_settings


TENANT_ID = "tenant_1481977a78484e3cbcda1dd4f224cfda"
USER_ID = "usr_7051e6a5cd7b4b2e9d60d2b6baff5377"


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def make_jwt() -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "tenant_id": TENANT_ID,
        "sub": USER_ID,
        "scope_type": "platform",
        "role": "admin",
        "exp": int(time.time()) + 3600,
    }
    encoded_header = _b64url(json.dumps(header, separators=(",", ":")).encode())
    encoded_payload = _b64url(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{encoded_header}.{encoded_payload}"
    signature = hmac.new(
        get_settings().tenant_auth_secret.encode(), signing_input.encode(), hashlib.sha256
    ).digest()
    return f"{signing_input}.{_b64url(signature)}"


def get_json(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        response = requests.get(url, headers=headers or {}, timeout=10)
        try:
            body = response.json()
        except ValueError:
            body = {"text": response.text[:500]}
        return {"status": response.status_code, "body": body}
    except Exception as exc:  # noqa: BLE001 - evidence should include probe failures
        return {"status": 0, "error": f"{type(exc).__name__}: {exc}"}


def qdrant_info(qdrant_url: str, collection: str) -> dict[str, Any]:
    return get_json(f"{qdrant_url.rstrip('/')}/collections/{collection}")


def record(base_url: str, question: str, qdrant_url: str, collection: str) -> dict[str, Any]:
    base_url = base_url.rstrip("/")
    token = make_jwt()
    auth_headers = {"Authorization": f"Bearer {token}"}
    thread_id = f"real-full-chain-{uuid.uuid4().hex[:12]}"
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    before = {
        "health_live": get_json(f"{base_url}/health/live"),
        "health_ready": get_json(f"{base_url}/health/ready"),
        "qdrant": qdrant_info(qdrant_url, collection),
    }
    events: list[dict[str, Any]] = []
    stream_error: str | None = None
    terminal: dict[str, Any] | None = None
    trace_id: str | None = None
    try:
        with requests.post(
            f"{base_url}/chat/stream",
            headers=auth_headers,
            json={"message": question, "thread_id": thread_id},
            stream=True,
            timeout=300,
        ) as response:
            http_status = response.status_code
            if response.status_code == 200:
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    try:
                        payload = json.loads(line.removeprefix("data:").strip())
                    except Exception as exc:  # noqa: BLE001
                        stream_error = f"invalid_sse_json: {exc}"
                        continue
                    events.append({
                        "at_ms": round((time.perf_counter() - started) * 1000, 3),
                        "payload": payload,
                    })
                    if payload.get("trace_id"):
                        trace_id = payload["trace_id"]
                    event_name = payload.get("event") or payload.get("type")
                    if event_name in {"completed", "failed"}:
                        terminal = payload
                        break
            else:
                stream_error = response.text[:500]
    except Exception as exc:  # noqa: BLE001
        http_status = 0
        stream_error = f"{type(exc).__name__}: {exc}"

    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    trace_fetch: dict[str, Any] | None = None
    if trace_id:
        trace_response = get_json(f"{base_url}/traces/{trace_id}", auth_headers)
        trace_fetch = {"trace_id": trace_id, **trace_response}
    after = {
        "health_live": get_json(f"{base_url}/health/live"),
        "health_ready": get_json(f"{base_url}/health/ready"),
        "qdrant": qdrant_info(qdrant_url, collection),
    }
    return {
        "schema_version": "1.0",
        "experiment": "real_full_chain_single_request",
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "model": get_settings().llm_model,
        "rag_backend_configured": get_settings().rag_backend,
        "question": question,
        "tenant_id": TENANT_ID,
        "user_id": USER_ID,
        "scope_type": "platform",
        "role": "admin",
        "thread_id": thread_id,
        "http_status": http_status,
        "terminal_event": terminal.get("event") if terminal else None,
        "terminal": terminal,
        "elapsed_ms": elapsed_ms,
        "event_count": len(events),
        "events": events,
        "trace_ids": [trace_id] if trace_id else [],
        "trace_fetches": [trace_fetch] if trace_fetch else [],
        "health_before": before,
        "health_after": after,
        "qdrant_before": before["qdrant"],
        "qdrant_after": after["qdrant"],
        "stream_error": stream_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:28080")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--collection", default="bizinsight_knowledge")
    parser.add_argument(
        "--question",
        default="为什么 2018-08-20 的 GMV 下降了？请查询实际数据并分析原因。",
    )
    args = parser.parse_args()
    artifact = record(args.base_url, args.question, args.qdrant_url, args.collection)
    output = Path(__file__).resolve().parents[1] / "data" / "evaluations"
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output / f"real-full-chain-{stamp}.json"
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    terminal = artifact.get("terminal") or {}
    print(json.dumps({
        "artifact": str(path),
        "http_status": artifact["http_status"],
        "terminal_event": artifact["terminal_event"],
        "terminal_error": terminal.get("error"),
        "elapsed_ms": artifact["elapsed_ms"],
        "event_count": artifact["event_count"],
        "trace_id": artifact["trace_ids"][0] if artifact["trace_ids"] else None,
        "rag_backend": artifact["rag_backend_configured"],
        "stream_error": artifact["stream_error"],
    }, ensure_ascii=False))
    return 0 if artifact["http_status"] == 200 and artifact["terminal_event"] == "completed" and not terminal.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())

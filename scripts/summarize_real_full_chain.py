"""Build a compact index from real-full-chain request artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "data" / "evaluations"


def summarize(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    trace = ((payload.get("trace_fetches") or [{}])[0].get("body") or {})
    snapshot = trace.get("state_snapshot") or {}
    rag = snapshot.get("rag_result") or {}
    tool_calls = snapshot.get("tool_calls") or []
    node_events = trace.get("events") or []
    return {
        "artifact": str(path.relative_to(ROOT)).replace("\\", "/"),
        "question": payload.get("question"),
        "started_at": payload.get("started_at"),
        "http_status": payload.get("http_status"),
        "terminal_event": payload.get("terminal_event"),
        "terminal_error": (payload.get("terminal") or {}).get("error"),
        "elapsed_ms": payload.get("elapsed_ms"),
        "event_count": payload.get("event_count"),
        "trace_id": (payload.get("trace_ids") or [None])[0],
        "model": payload.get("model"),
        "rag_backend_configured": payload.get("rag_backend_configured"),
        "rag_trace_backend": rag.get("backend"),
        "rag_hit_count": len(rag.get("hits") or []),
        "qdrant_points_before": ((payload.get("qdrant_before") or {}).get("body") or {}).get("result", {}).get("points_count"),
        "qdrant_points_after": ((payload.get("qdrant_after") or {}).get("body") or {}).get("result", {}).get("points_count"),
        "tool_call_count": len(tool_calls),
        "tool_call_statuses": {str(item.get("status")): sum(1 for call in tool_calls if call.get("status") == item.get("status")) for item in tool_calls},
        "tools": [f"{item.get('server')}:{item.get('tool_name')}" for item in tool_calls],
        "trace_node_count": len(node_events),
        "trace_nodes": [item.get("node") for item in node_events],
        "errors": snapshot.get("errors") or [],
        "stream_error": payload.get("stream_error"),
    }


def main() -> int:
    rows = [summarize(path) for path in sorted(EVIDENCE_DIR.glob("real-full-chain-*.json"))]
    output = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "real_full_chain_single_request",
        "runs": rows,
    }
    target = EVIDENCE_DIR / "real-full-chain-summary.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"artifact": str(target), "run_count": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

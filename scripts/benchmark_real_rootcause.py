"""Real-provider root-cause A/B benchmark.

Runs the same real business questions against the legacy serial route and the
current governed route.  PostgreSQL and the configured LLM provider are not
patched; only the route mode is varied.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.graph.workflow import build_chat_graph  # noqa: E402


CASES = [
    ("rootcause-aov-factor", "为什么 2018-08-20 的 GMV 下降了？", {"must_contain": ("avg_order_value", "客单价")}),
    ("rootcause-abstain", "为什么 2018-08-21 的 GMV 下降了？", {"must_contain": ("abstain", "弃答", "证据不足", "无法确认")}),
    ("rootcause-linked", "为什么 2018-08-22 的 GMV 下降了？", {"must_contain": ("delivery_delay_rate", "配送", "履约")}),
]


def evaluate_answer(case_id: str, result: dict) -> tuple[bool, str]:
    text = " ".join(str(result.get(key) or "") for key in ("answer", "analysis_result"))
    expected = next(item[2] for item in CASES if item[0] == case_id)
    matched = next((term for term in expected["must_contain"] if term.lower() in text.lower()), None)
    return bool(matched), matched or "no_expected_term_found"


def run(mode: str) -> dict:
    os.environ["BIZINSIGHT_ROOT_CAUSE_MODE"] = mode
    graph = build_chat_graph(checkpointer=InMemorySaver())
    rows = []
    for case_id, question, _expected in CASES:
        trace_id = f"real-ab-{mode}-{case_id}-{uuid4().hex[:8]}"
        started = perf_counter()
        result = graph.invoke(
            {"user_message": question, "trace_id": trace_id, "thread_id": trace_id, "audit_actor": "real-ab"},
            config={"configurable": {"thread_id": trace_id}},
        )
        elapsed_ms = round((perf_counter() - started) * 1000, 3)
        calls = result.get("tool_calls", []) or []
        keys = [(str(item.get("tool_name") or ""), repr(item.get("arguments") or {})) for item in calls]
        timings = result.get("node_timings", []) or []
        rows.append({
            "id": case_id,
            "question": question,
            "elapsed_ms": elapsed_ms,
            "tool_call_count": len(calls),
            "duplicate_call_count": len(keys) - len(set(keys)),
            "node_count": len(result.get("node_history", []) or []),
            "llm_usage": result.get("llm_usage", {}),
            "decision_driver": result.get("decision_driver"),
            "critic_verdict": result.get("critic_verdict"),
            "error_count": len(result.get("errors", []) or []),
            "errors": result.get("errors", []) or [],
            "answer": result.get("answer", "") or "",
            "analysis_result": result.get("analysis_result", {}) or {},
            "correct": evaluate_answer(case_id, result)[0],
            "matched_expected_term": evaluate_answer(case_id, result)[1],
            "node_timings": timings,
        })
    elapsed = [row["elapsed_ms"] for row in rows]
    return {
        "mode": mode,
        "provider": "real",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases": rows,
        "summary": {
            "case_count": len(rows),
            "avg_elapsed_ms": round(sum(elapsed) / len(elapsed), 3) if elapsed else 0,
            "max_elapsed_ms": max(elapsed) if elapsed else 0,
            "avg_tool_calls": round(sum(row["tool_call_count"] for row in rows) / len(rows), 3) if rows else 0,
            "duplicate_call_count": sum(row["duplicate_call_count"] for row in rows),
            "error_count": sum(row["error_count"] for row in rows),
            "correct_count": sum(1 for row in rows if row["correct"]),
            "correctness_rate": round(sum(1 for row in rows if row["correct"]) / len(rows), 4) if rows else 0,
        },
    }


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "governed"
    if mode not in {"governed", "legacy_serial"}:
        raise SystemExit("mode must be governed or legacy_serial")
    payload = run(mode)
    output_path = os.path.join(ROOT, "data", "evaluations", f"real-rootcause-{mode}.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"mode": mode, "summary": payload["summary"], "output": output_path}, ensure_ascii=False))

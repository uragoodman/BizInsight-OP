"""Evaluate the independent ordinary-analysis/report dataset.

The first version is an offline contract gate. It intentionally scores task
routing, evidence/tool usage, and report structure separately from root-cause
judgment. Provider mode is added only after the offline dataset is stable.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.graph.workflow import build_chat_graph
from evaluation.fixtures import offline_dependencies
from evaluation.scorers import score_phase6_report, score_trajectory


DATASET = ROOT / "evaluation" / "datasets" / "analysis_report_v1.json"


def _tool_names(state: dict[str, Any]) -> list[str]:
    return [str(item.get("tool_name") or "") for item in state.get("tool_calls", []) if isinstance(item, dict)]


def _score(state: dict[str, Any], expect: dict[str, Any]) -> dict[str, bool]:
    trajectory = score_trajectory(state)
    history = {str(item.get("node")) for item in state.get("node_history", []) if isinstance(item, dict)}
    tool_names = _tool_names(state)
    answer = str(state.get("final_answer") or state.get("answer") or "").strip()
    checks = {
        "task_type": state.get("task_type") == expect["task_type"],
        "answer_present": bool(answer),
        "evidence_bound": bool(trajectory.get("evidence_bound")),
    }
    if expect.get("require_rag"):
        checks["rag_evidence"] = bool(state.get("rag_citations") or state.get("rag_result"))
    if expect.get("require_tool"):
        checks["tool_called"] = bool(tool_names)
    if expect.get("require_trend"):
        checks["trend_tool"] = any("trend" in name for name in tool_names)
    if expect.get("require_comparison"):
        checks["comparison_tool"] = any("compare" in name or "trend" in name for name in tool_names)
    if expect.get("require_contribution"):
        checks["contribution_tool"] = any("contribution" in name or "breakdown" in name for name in tool_names)
    if expect.get("require_report"):
        report = state.get("periodic_report") or {}
        checks["report_present"] = bool(report)
        checks["report_contract"] = all(score_phase6_report(report).values()) if report else False
    if expect.get("require_report_draft"):
        draft = state.get("analysis_result") or {}
        checks["report_draft_present"] = bool(str(draft.get("summary") or "").strip()) and bool(answer)
    if expect.get("require_lineage"):
        report = state.get("periodic_report") or {}
        checks["lineage"] = bool(report.get("lineage_refs"))
    checks["critic_present"] = "critic" in history
    return checks


def run(*, case_ids: set[str] | None = None) -> dict[str, Any]:
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    cases = [item for item in dataset["cases"] if case_ids is None or item["id"] in case_ids]
    details: list[dict[str, Any]] = []
    with offline_dependencies():
        graph = build_chat_graph(checkpointer=InMemorySaver())
        for case in cases:
            result = graph.invoke(
                {"user_message": case["query"], "trace_id": f"analysis-{case['id']}", "thread_id": f"analysis-{case['id']}", "audit_actor": "analysis-evaluator"},
                config={"configurable": {"thread_id": f"analysis-{case['id']}"}},
            )
            checks = _score(result, case["expect"])
            details.append({
                "id": case["id"], "category": case["category"], "query": case["query"],
                "passed": all(checks.values()), "expected": case["expect"], "checks": checks,
                "task_type": result.get("task_type"),
                "tool_names": _tool_names(result),
                "node_history": result.get("node_history", []),
            })
    passed = sum(bool(item["passed"]) for item in details)
    by_category: dict[str, dict[str, int]] = {}
    for item in details:
        bucket = by_category.setdefault(item["category"], {"total": 0, "passed": 0})
        bucket["total"] += 1
        bucket["passed"] += int(item["passed"])
    report = {
        "dataset_version": dataset["dataset_version"],
        "evaluation_mode": "offline_contract",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": len(details), "passed": passed,
        "pass_rate": round(passed / len(details), 4) if details else 0.0,
        "by_category": by_category, "details": details,
    }
    output = ROOT / "data" / "evaluations" / "analysis-report-offline.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "total": len(details), "passed": passed, "pass_rate": report["pass_rate"], "by_category": by_category}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    selected = next((item for item in sys.argv if item.startswith("--case-ids=")), "")
    ids = {item for item in selected.split("=", 1)[1].split(",") if item} if selected else None
    report = run(case_ids=ids)
    raise SystemExit(0 if report["passed"] == report["total"] else 1)

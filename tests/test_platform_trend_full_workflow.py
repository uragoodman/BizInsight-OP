"""Daily trend acceptance through chat; reuse independent PostgreSQL facts."""
import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from test_platform_metric_trend_acceptance import (
    DATES, END, EXPECTED, EXPECTED_SQL_RELATIONS, START,
    isolated_trend_db,  # noqa: F401 -- expose the existing pytest fixture
)
from app.data.repositories import daily_metrics
from app.metrics.registry import get_registry


def _fmt(value):
    return f"{float(value):.2f}"


@pytest.mark.parametrize("metric_id", sorted(EXPECTED))
def test_platform_trend_full_workflow(monkeypatch, isolated_trend_db, tmp_path, metric_id):
    from app.graph import workflow
    from app.core import time_expression
    from app.graph.nodes import triage
    from app.llm import client as llm
    from app.mcp import client as mcp
    from app.tenancy import TenantContext
    from app.graph.nodes.answer import _display_unit

    engine, statements, sql_events, _ = isolated_trend_db
    definition = get_registry().get(metric_id)
    monkeypatch.setattr(daily_metrics, "engine", engine)
    monkeypatch.setattr(time_expression, "engine", engine)
    monkeypatch.setattr(triage, "llm_classify", lambda *a, **k: None)
    monkeypatch.setattr(llm, "llm_extract", lambda *a, **k: None)
    monkeypatch.setattr(llm, "llm_chat", lambda *a, **k: None)
    def block_fallback(*args, **kwargs):
        raise RuntimeError("Isolated acceptance blocks external MCP fallback")
    monkeypatch.setattr(mcp, "call_local_mcp_tool", block_fallback)
    captured = {}
    monkeypatch.setattr(workflow, "write_trace", lambda tid, state: captured.update(state))
    monkeypatch.setattr(workflow, "_persist_chat_turn", lambda *a, **k: None)
    question = f"请查询 {START} 至 {END} 的“{definition.display_name}”每天的变化趋势。"
    offset = len(statements)
    result = {}
    error = None
    try:
        result = workflow.run_chat_workflow(
            question, tenant_context=TenantContext(
                tenant_id="platform", user_id="trend-acceptance", role="admin", scope_type="platform",
            ), memory_enabled=False, trace_id=f"platform-trend-{uuid4().hex}",
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    plans = (captured.get("metric_query_batch_plan") or {}).get("plans") or []
    plan = plans[0] if len(plans) == 1 else {}
    items = captured.get("query_results") or []
    item = items[0] if len(items) == 1 else {}
    inner = item.get("result") or {}
    trend = inner.get("trend_result") or {}
    points = trend.get("points") or []
    values = EXPECTED[metric_id][0]
    sql = statements[offset:]
    bundles = captured.get("evidence_bundles") or []
    bundle = next((b for b in bundles if b.get("evidence_id") == item.get("evidence_id")), {})
    content = bundle.get("content") or {}
    calls = captured.get("tool_calls") or []
    matching = [c for c in calls if c.get("request_id") == item.get("request_id")]
    presentation = result.get("presentation") or {}
    facts = presentation.get("facts") or []
    answer = result.get("answer") or ""
    checks = {
        "planning": plan.get("metric_name") == metric_id and plan.get("query_type") == "metric_trend"
            and plan.get("start_date") == str(START) and plan.get("end_date") == str(END)
            and plan.get("granularity") == "day",
        "query": inner.get("status") == "ok" and trend.get("metric_name") == metric_id
            and trend.get("start_date") == str(START) and trend.get("end_date") == str(END)
            and trend.get("granularity") == "day"
            and [p.get("bucket") for p in points] == [str(d) for d in DATES]
            and len(points) == len(values)
            and all(p.get("value") is not None and abs(p["value"] - v) <= .01 for p, v in zip(points, values)),
        "sql_sources": bool(sql) and all(r in " ".join(sql).lower() for r in EXPECTED_SQL_RELATIONS[metric_id]),
        "binding": bool(item.get("request_id")) and len(matching) == 1 and bool(bundle)
            and content.get("request_id") == item.get("request_id")
            and content.get("metric_id") == metric_id and content.get("query_type") == "metric_trend"
            and content.get("result") == inner,
        "critic": (captured.get("critic_result") or {}).get("data_review", {}).get("status") == "approved",
        "tool_unit": trend.get("unit") == definition.unit,
        "fact_unit_and_evidence": bool(facts) and facts[0].get("unit") == definition.unit
            and item.get("evidence_id") in (facts[0].get("evidence_ids") or []),
        "answer": all(str(d) in answer and f"{d}={_fmt(v)}" in answer for d, v in zip(DATES, values))
            and _display_unit(definition.unit) in answer,
        "completed": presentation.get("status") == "completed",
        "clean_completion": not captured.get("errors") and captured.get("critic_verdict") == "approved",
        "no_repeated_tool_calls": len(calls) == 1,
    }
    # Independent first/last, extrema and latest-step expectations; nonmonotonic
    # series in the fixture prevent simply calling every trend an increase.
    first, last = values[0], values[-1]
    direction = "上升" if last > first else "下降" if last < first else "持平"
    latest_direction = "上升" if last > values[-2] else "下降" if last < values[-2] else "持平"
    checks["answer_summary"] = (
        f"整体{direction} {_fmt(abs(last-first))}" in answer
        and f"最近一期较上一期{latest_direction} {_fmt(abs(last-values[-2]))}" in answer
        and f"最高点为 {DATES[values.index(max(values))]}" in answer
        and f"最低点为 {DATES[values.index(min(values))]}" in answer
    )
    report = dict(metric_id=metric_id, question=question, expected_values=values,
                  expected_unit=definition.unit, checks=checks, workflow_error=error,
                  full_task_pass=all(checks.values()), tool_calls=calls,
                  sql_events=sql_events[offset:], workflow_state=captured, workflow_result=result)
    trace_dir = tmp_path / "platform_trend_workflow_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / f"{metric_id}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    assert all(checks.values()), {"metric": metric_id, "error": error, "checks": checks}

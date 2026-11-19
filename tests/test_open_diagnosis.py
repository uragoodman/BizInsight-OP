"""AGENT_PLAN Phase 6: open business-diagnosis agent tests."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from app.agents.open_diagnosis import run_open_diagnosis
from app.agents.open_diagnosis_contracts import OpenDiagnosisReport, ProblemDiagnosis
from app.analysis.contracts import MetricPoint, MetricSeries, PeriodSpec


def _series(metric_name: str, end_value: float, base: float = 100.0) -> MetricSeries:
    end = date(2025, 3, 31)
    start = end - timedelta(days=13)
    period = PeriodSpec.model_validate({"start_date": start.isoformat(), "end_date": end.isoformat(), "granularity": "day"})
    points = [
        MetricPoint(bucket=start + timedelta(days=index), value=base if index < 13 else end_value)
        for index in range(14)
    ]
    return MetricSeries(
        metric_name=metric_name, unit="BRL", period=period, source_refs=["test.fixture"],
        definition_version="1.0", freshness_at=datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc),
        points=points,
    )


def _fetcher(metric_name: str, period, **_: object) -> MetricSeries:
    values = {"gmv": 50.0, "delivery_delay_rate": 200.0, "order_count": 100.0}
    return _series(metric_name, values.get(metric_name, 100.0))


def _scope(metrics: list[str] | None = None) -> dict:
    return {
        "seller_id": "seller-1",
        "allowed_metrics": metrics or ["gmv", "delivery_delay_rate", "order_count"],
        "default_start_date": "2025-03-18", "default_end_date": "2025-03-31",
    }


def _fake_agent_action(system_prompt: str, user_message: str, **_: object):
    """Scripted triage + drill-down policy mirroring evaluation/agent_policy.py."""
    if "分诊器" in system_prompt:
        return {"plan": {
            "goal": "优先诊断影响最大的异常",
            "problems": [
                {"problem_id": "p1", "metric_name": "gmv", "metric_date": "2025-03-31", "direction": "down",
                 "priority": 1, "hypothesis": "GMV 单日腰斩，需归因", "scan_evidence_ids": [],
                 "budget": {"max_rounds": 4, "max_tool_calls": 4, "max_tokens": 4000, "max_cost": 1.0}},
                {"problem_id": "p2", "metric_name": "delivery_delay_rate", "metric_date": "2025-03-31", "direction": "up",
                 "priority": 2, "hypothesis": "延迟率翻倍，需确认", "scan_evidence_ids": [],
                 "budget": {"max_rounds": 4, "max_tool_calls": 4, "max_tokens": 4000, "max_cost": 1.0}},
                {"problem_id": "p3", "metric_name": "fabricated_metric", "metric_date": "2025-03-31", "direction": "up",
                 "priority": 3, "hypothesis": "不在扫描中的指标必须被丢弃", "scan_evidence_ids": [],
                 "budget": {"max_rounds": 2, "max_tool_calls": 2, "max_tokens": 100, "max_cost": 1.0}},
            ],
        }}
    if "计划器" in system_prompt:
        return {"plan": {
            "goal": "解释指标异常", "hypotheses": ["存在可验证的维度或关联指标原因"],
            "steps": [{"step_id": "s1", "objective": "获取趋势证据", "allowed_actions": ["CALL_TOOL", "FINISH", "ABSTAIN"],
                       "completion_condition": "已有趋势证据"}],
            "stop_conditions": ["证据足够", "无显著原因"],
            "budget": {"max_rounds": 4, "max_tool_calls": 4, "max_tokens": 4000, "max_cost": 1.0},
        }}
    # 决策器：先查本指标趋势，再引用证据 FINISH。
    calls = _fake_agent_action.counts
    key = user_message[:32]
    calls[key] = calls.get(key, 0) + 1
    if calls[key] == 1:
        metric = "gmv" if "gmv" in user_message else "delivery_delay_rate"
        return {"action": {"action_type": "CALL_TOOL", "tool_name": "query_metric_trend", "metric_name": metric,
                           "period": {"start_date": "2025-03-25", "end_date": "2025-03-31", "granularity": "day"}}}
    import re
    ids = re.findall(r'"(ev_[0-9a-f]+)"', system_prompt) or ["ev_missing"]
    return {"action": {"action_type": "FINISH", "conclusion": "趋势证据支持该异常需要关注。",
                       "evidence_ids": ids[:1], "residual_uncertainty": "单日证据，持续性待观察"}}


_fake_agent_action.counts = {}


def _enabled_settings():
    from app.core.config import get_settings
    settings = get_settings()
    return patch.object(settings, "open_diagnosis_enabled", True)


def test_refuses_auto_run_without_permission_switch():
    with patch("app.agents.open_diagnosis._scan_candidates") as scanner:
        report = run_open_diagnosis(_scope())
    assert report.status == "refused_no_permission"
    assert report.problems == []
    assert report.published is False
    scanner.assert_not_called()


def test_refuses_metrics_outside_whitelist():
    with _enabled_settings(), patch("app.agents.open_diagnosis._scan_candidates") as scanner:
        report = run_open_diagnosis(_scope(metrics=["gmv", "profit_margin"]))
    assert report.status == "refused_scope"
    assert "profit_margin" in report.refusal_reason
    scanner.assert_not_called()


def test_scan_triage_drilldown_abstains_when_range_has_no_complete_comparison_periods():
    _fake_agent_action.counts = {}
    with _enabled_settings(), \
         patch("app.analysis.periods.query_metric_series", _fetcher), \
         patch("app.llm.client.llm_agent_action", _fake_agent_action), \
         patch("app.graph.nodes.diagnostic.query_business_data", _trend_fixture):
        report = run_open_diagnosis(_scope(), trace_id="open-diag-test")
    assert isinstance(report, OpenDiagnosisReport)
    assert report.status == "completed"
    assert report.decision_driver == "llm"
    assert report.published is False and report.review_status == "pending_review"
    # The fabricated metric never enters the problem list; scanned metrics rank first.
    assert [p.priority for p in report.problems] == sorted(p.priority for p in report.problems)
    assert all(p.metric_name in {"gmv", "delivery_delay_rate"} for p in report.problems)
    abstained = [p for p in report.problems if p.status == "abstained"]
    assert len(abstained) == 2
    assert all(p.abstain_reason == "insufficient_evidence" for p in abstained)
    assert report.scan_summary["anomalies_found"] == 2
    assert report.scan_summary["problems_selected"] == 2
    assert report.agent_run is not None
    assert report.agent_run.domain == "open_diagnosis"
    assert all(problem.agent_run is not None for problem in abstained)


def _trend_fixture(task_type: str, question: str, **_: object) -> dict:
    metric = "gmv" if "gmv" in question else "delivery_delay_rate"
    series = _fetcher(metric, None)
    return {
        "server": "mock-mcp", "tool_name": "query_metric_trend", "arguments": {},
        "request_id": "open-diag", "idempotency_key": "open-diag",
        "result": {"status": "ok", "parsed": {"query_type": "metric_trend"},
                   "trend_result": {
                       "metric_name": metric, "start_date": "2025-03-25", "end_date": "2025-03-31",
                       "granularity": "day", "source_table": "test.fixture",
                       "points": [{"bucket": point.bucket.isoformat(), "value": point.value} for point in series.points],
                   }},
    }


def test_cooldown_skips_recently_diagnosed_metric():
    _fake_agent_action.counts = {}
    now = datetime.now(timezone.utc).isoformat()
    with _enabled_settings(), \
         patch("app.analysis.periods.query_metric_series", _fetcher), \
         patch("app.llm.client.llm_agent_action", _fake_agent_action), \
         patch("app.graph.nodes.diagnostic.query_business_data", _trend_fixture):
        report = run_open_diagnosis(_scope(metrics=["gmv"]), trace_id="cooldown-test", cooldown_index={"gmv": now})
    assert report.problems[0].status == "skipped_cooldown"


def test_llm_unavailable_degrades_to_deterministic_fallback_without_claims():
    with _enabled_settings(), \
         patch("app.analysis.periods.query_metric_series", _fetcher), \
         patch("app.llm.client.llm_agent_action", lambda *args, **kwargs: None), \
         patch("app.graph.nodes.diagnostic.query_business_data", _trend_fixture):
        report = run_open_diagnosis(_scope(), trace_id="fallback-test")
    assert report.decision_driver == "workflow_fallback"
    # Deterministic ranking still selects the scanned anomalies, and the drill
    # down degrades to an abstention instead of manufacturing a conclusion.
    assert {p.metric_name for p in report.problems} <= {"gmv", "delivery_delay_rate"}
    assert all(p.status == "abstained" and p.abstain_reason == "llm_unavailable_workflow_fallback" for p in report.problems)


def test_diagnosed_findings_emit_pending_review_event_only():
    _fake_agent_action.counts = {}
    from app.analysis.events import EventOutbox

    with _enabled_settings(), \
         patch("app.analysis.periods.query_metric_series", _fetcher), \
         patch("app.llm.client.llm_agent_action", _fake_agent_action), \
         patch("app.graph.nodes.diagnostic.query_business_data", _trend_fixture):
        report = run_open_diagnosis(_scope(metrics=["gmv"]), trace_id="outbox-test")
    emitted = [event for event in EventOutbox("data/open-diagnosis-test.sqlite").pending()]
    assert not emitted  # nothing was auto-published by the runner itself


def test_contract_diagnosed_problem_requires_evidence():
    with pytest.raises(Exception):
        ProblemDiagnosis(problem_id="p", metric_name="gmv", metric_date="2025-03-31",
                         priority=1, status="diagnosed", conclusion="c", evidence_ids=[])

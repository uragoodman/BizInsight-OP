"""AGENT_PLAN Phase 9: alert triage decision layer tests."""

from __future__ import annotations

import re
from datetime import date
from unittest.mock import patch

import pytest

from app.agents.alert_triage import review_diagnosis_proposal, triage_anomaly
from app.analysis.contracts import AnomalyFinding
from app.analysis.events import EventOutbox
from app.core.config import get_settings


def _finding(metric_name: str = "gmv", direction: str = "down") -> AnomalyFinding:
    return AnomalyFinding(
        metric_name=metric_name, metric_date=date(2025, 3, 31),
        value=50.0, baseline=100.0, score=-0.5, threshold=0.3,
        direction=direction, method="change_threshold", sufficient_data=True,
        duration_days=1, evidence_ids=["ev_scan_1"],
    )


def _enabled(tenant: str = "tenant-a", metrics: list[str] | None = None):
    settings = get_settings()
    whitelist = ",".join(f'"{m}"' for m in (metrics or ["gmv"]))
    return (
        patch.object(settings, "alert_auto_diagnosis_enabled", True),
        patch.object(settings, "alert_auto_diagnosis_tenants_json", f'["{tenant}"]'),
        patch.object(settings, "alert_auto_diagnosis_metrics_json", f"[{whitelist}]"),
    )


def _trend_fixture(task_type: str, question: str, **_: object) -> dict:
    points = [{"bucket": f"2025-03-{day:02d}", "value": 100.0 if day < 31 else 50.0} for day in range(25, 32)]
    return {
        "server": "mock-mcp", "tool_name": "query_metric_trend", "arguments": {},
        "request_id": "alert-triage", "idempotency_key": "alert-triage",
        "result": {"status": "ok", "parsed": {"query_type": "metric_trend"},
                   "trend_result": {"metric_name": "gmv", "start_date": "2025-03-25", "end_date": "2025-03-31",
                                     "granularity": "day", "source_table": "test.fixture", "points": points}},
    }


def _scripted_action(system_prompt: str, user_message: str, **_: object):
    if "计划器" in system_prompt:
        return {"plan": {
            "goal": "解释指标异常", "hypotheses": ["存在可验证的关联原因"],
            "steps": [{"step_id": "s1", "objective": "获取趋势证据", "allowed_actions": ["CALL_TOOL", "FINISH", "ABSTAIN"],
                       "completion_condition": "已有趋势证据"}],
            "stop_conditions": ["证据足够", "无显著原因"],
            "budget": {"max_rounds": 4, "max_tool_calls": 4, "max_tokens": 4000, "max_cost": 1.0},
        }}
    calls = _scripted_action.counts
    key = user_message[:32]
    calls[key] = calls.get(key, 0) + 1
    # 新流程：关键证据由服务端骨架保证采集，LLM 只做一次终局判断。
    ids = re.findall(r'"(ev_[0-9a-f]+)"', system_prompt)
    return {"action": {"action_type": "FINISH", "conclusion": "趋势证据确认异常需关注。", "evidence_ids": ids[:1]}}


_scripted_action.counts = {}


def test_switch_off_keeps_pure_deterministic_alerting(tmp_path):
    outbox = EventOutbox(str(tmp_path / "alert-triage.sqlite"))
    with patch("app.agents.open_diagnosis.diagnose_problem") as drill:
        decision = triage_anomaly(_finding(), tenant_id="tenant-a", outbox=outbox)
    assert decision["allowed"] is False and decision["diagnosis_started"] is False
    assert any("switch" in reason for reason in decision["reasons"])
    drill.assert_not_called()
    assert not [e for e in outbox.pending() if e.event_type == "DiagnosisProposed"]


def test_tenant_without_switch_is_refused():
    enabled, tenants, metrics = _enabled(tenant="tenant-b")
    with enabled, tenants, metrics, patch("app.agents.open_diagnosis.diagnose_problem") as drill:
        decision = triage_anomaly(_finding(), tenant_id="tenant-a")
    assert decision["allowed"] is False
    assert any("tenant" in reason for reason in decision["reasons"])
    drill.assert_not_called()


def test_metric_outside_whitelist_is_refused():
    enabled, tenants, metrics = _enabled(metrics=["order_count"])
    with enabled, tenants, metrics, patch("app.agents.open_diagnosis.diagnose_problem") as drill:
        decision = triage_anomaly(_finding(metric_name="delivery_delay_rate"), tenant_id="tenant-a")
    assert decision["allowed"] is False
    assert any("whitelisted" in reason for reason in decision["reasons"])
    drill.assert_not_called()


def test_enabled_tenant_abstains_when_alert_scope_cannot_support_comparison(tmp_path):
    _scripted_action.counts = {}
    outbox = EventOutbox(str(tmp_path / "alert-triage.sqlite"))
    enabled, tenants, metrics = _enabled(tenant="tenant-a")
    with enabled, tenants, metrics, \
         patch("app.llm.client.llm_agent_action", _scripted_action), \
         patch("app.graph.nodes.diagnostic.query_business_data", _trend_fixture), \
         patch("app.graph.nodes.diagnostic.query_scoped_business_data", _trend_fixture):
        decision = triage_anomaly(_finding(), tenant_id="tenant-a", outbox=outbox, trace_id="alert-triage-e2e")
    assert decision["allowed"] is True and decision["diagnosis_started"] is True
    assert decision["agent_run"]["domain"] == "alert_triage"
    assert decision["agent_run"]["decision_driver"] == "llm"
    diagnosis = decision["diagnosis"]
    assert diagnosis["status"] == "abstained"
    assert diagnosis["abstain_reason"] == "insufficient_evidence"
    assert diagnosis["evidence_ids"] == ["ev_scan_1"]
    assert diagnosis["scan_evidence_ids"] == ["ev_scan_1"]
    proposals = [e for e in outbox.pending() if e.event_type == "DiagnosisProposed"]
    assert proposals == []


def test_cooldown_blocks_repeated_auto_diagnosis(tmp_path):
    _scripted_action.counts = {}
    outbox = EventOutbox(str(tmp_path / "alert-triage.sqlite"))
    enabled, tenants, metrics = _enabled(tenant="tenant-a")
    with enabled, tenants, metrics, \
         patch("app.llm.client.llm_agent_action", _scripted_action), \
         patch("app.graph.nodes.diagnostic.query_business_data", _trend_fixture), \
         patch("app.graph.nodes.diagnostic.query_scoped_business_data", _trend_fixture):
        first = triage_anomaly(_finding(), tenant_id="tenant-a", outbox=outbox, trace_id="cd-1")
        second = triage_anomaly(_finding(), tenant_id="tenant-a", outbox=outbox, trace_id="cd-2")
    assert first["diagnosis_started"] is True
    assert second["allowed"] is False
    assert any("cooldown" in reason for reason in second["reasons"])


def test_diagnosis_attempt_cooldown_is_atomic_and_separate_from_notifications(tmp_path):
    outbox = EventOutbox(str(tmp_path / "alert-triage.sqlite"))
    assert outbox.claim_diagnosis_attempt("tenant-a", "gmv", 24) is True
    assert outbox.claim_diagnosis_attempt("tenant-a", "gmv", 24) is False
    assert outbox.claim_diagnosis_attempt("tenant-b", "gmv", 24) is True
    assert outbox.pending() == []


def test_review_rejects_unknown_action(tmp_path):
    with pytest.raises(ValueError):
        review_diagnosis_proposal(EventOutbox(str(tmp_path / "alert-triage.sqlite")), "k", "publish", "actor")

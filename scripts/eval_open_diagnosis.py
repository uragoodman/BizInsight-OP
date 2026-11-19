"""Phase 6 acceptance evidence: open diagnosis on a multi-metric anomaly case.

Runs the governed open-diagnosis pipeline offline (deterministic scan fixtures,
scripted triage/drill-down policy) and records the ranked, evidence-bound
problem list plus the no-permission refusal into
``data/evaluations/open-diagnosis-eval.json``.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.contracts import CONTRACT_SCHEMA_VERSION
from app.agents.open_diagnosis import run_open_diagnosis
from app.analysis.contracts import MetricPoint, MetricSeries, PeriodSpec
from evaluation.fixtures import offline_dependencies


def _series(metric_name: str, end_value: float, base: float = 100.0) -> MetricSeries:
    end = date(2025, 3, 31)
    start = end - timedelta(days=13)
    period = PeriodSpec.model_validate({"start_date": start.isoformat(), "end_date": end.isoformat(), "granularity": "day"})
    return MetricSeries(
        metric_name=metric_name, unit="BRL", period=period, source_refs=["eval.fixture"],
        definition_version="1.0",
        freshness_at=datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc),
        points=[
            MetricPoint(bucket=start + timedelta(days=index), value=base if index < 13 else end_value)
            for index in range(14)
        ],
    )


_SCAN_VALUES = {"gmv": 50.0, "delivery_delay_rate": 200.0, "order_count": 100.0, "avg_order_value": 10.0}


def _scan_fetcher(metric_name: str, _period: Any, **_: object) -> MetricSeries:
    return _series(metric_name, _SCAN_VALUES.get(metric_name, 100.0))


def _trend_fixture(task_type: str, question: str, **_: object) -> dict:
    metric = "gmv" if "gmv" in question else "delivery_delay_rate"
    series = _scan_fetcher(metric, None)
    return {
        "server": "mock-mcp", "tool_name": "query_metric_trend", "arguments": {},
        "request_id": "open-diag-eval", "idempotency_key": "open-diag-eval",
        "result": {"status": "ok", "parsed": {"query_type": "metric_trend"},
                   "trend_result": {
                       "metric_name": metric, "start_date": "2025-03-25", "end_date": "2025-03-31",
                       "granularity": "day", "source_table": "eval.fixture",
                       "points": [{"bucket": point.bucket.isoformat(), "value": point.value} for point in series.points],
                   }},
    }


def _scripted_action(system_prompt: str, user_message: str, **_: object) -> dict[str, Any]:
    if "分诊器" in system_prompt:
        return {"plan": {
            "goal": "优先诊断强度最高且相互印证的问题",
            "problems": [
                {"problem_id": "p1", "metric_name": "gmv", "metric_date": "2025-03-31", "direction": "down",
                 "priority": 1, "hypothesis": "GMV 单日腰斩且延迟率翻倍，可能同源", "scan_evidence_ids": [],
                 "budget": {"max_rounds": 4, "max_tool_calls": 4, "max_tokens": 4000, "max_cost": 1.0}},
                {"problem_id": "p2", "metric_name": "delivery_delay_rate", "metric_date": "2025-03-31", "direction": "up",
                 "priority": 2, "hypothesis": "延迟率异常需独立确认", "scan_evidence_ids": [],
                 "budget": {"max_rounds": 4, "max_tool_calls": 4, "max_tokens": 4000, "max_cost": 1.0}},
            ],
        }}
    if "计划器" in system_prompt:
        return {"plan": {
            "goal": "解释指标异常", "hypotheses": ["存在可验证的关联指标或维度原因"],
            "steps": [{"step_id": "s1", "objective": "获取趋势证据", "allowed_actions": ["CALL_TOOL", "FINISH", "ABSTAIN"],
                       "completion_condition": "已有趋势证据"}],
            "stop_conditions": ["证据足够", "无显著原因"],
            "budget": {"max_rounds": 4, "max_tool_calls": 4, "max_tokens": 4000, "max_cost": 1.0},
        }}
    calls = _scripted_action.counts
    key = user_message[:32]
    calls[key] = calls.get(key, 0) + 1
    if calls[key] == 1:
        metric = "gmv" if "gmv" in user_message else "delivery_delay_rate"
        return {"action": {"action_type": "CALL_TOOL", "tool_name": "query_metric_trend", "metric_name": metric,
                           "period": {"start_date": "2025-03-25", "end_date": "2025-03-31", "granularity": "day"}}}
    import re
    ids = re.findall(r'"(ev_[0-9a-f]+)"', system_prompt)
    return {"action": {"action_type": "FINISH", "conclusion": "趋势证据确认该异常，需结合关联指标持续观察。",
                       "evidence_ids": ids[:1], "residual_uncertainty": "单日证据，持续性待观察"}}


_scripted_action.counts = {}


def _run() -> dict[str, Any]:
    from app.core.config import get_settings

    settings = get_settings()
    scope = {
        "seller_id": "seller-1",
        "allowed_metrics": ["gmv", "delivery_delay_rate", "order_count"],
        "default_start_date": "2025-03-18", "default_end_date": "2025-03-31",
    }
    with offline_dependencies(), patch.object(settings, "open_diagnosis_enabled", True), \
         patch("app.analysis.periods.query_metric_series", _scan_fetcher):
        with (
            patch("app.llm.client.llm_agent_action", _scripted_action),
            patch("app.graph.nodes.diagnostic.query_business_data", _trend_fixture),
        ):
            report = run_open_diagnosis(scope, trace_id="open-diag-eval").model_dump(mode="json")
        with patch("app.llm.client.llm_agent_action", lambda *a, **k: None), \
             patch("app.graph.nodes.diagnostic.query_business_data", _trend_fixture):
            fallback_report = run_open_diagnosis(scope, trace_id="open-diag-fallback").model_dump(mode="json")
        with patch.object(settings, "open_diagnosis_enabled", False):
            refused = run_open_diagnosis(scope, trace_id="open-diag-refused").model_dump(mode="json")

    diagnosed = [item for item in report["problems"] if item["status"] == "diagnosed"]
    checks = {
        "multi_metric_scan": report["scan_summary"].get("anomalies_found", 0) >= 2,
        "ranked_problem_list": [item["priority"] for item in report["problems"]] == sorted(item["priority"] for item in report["problems"]),
        "top_problem_diagnosed": bool(diagnosed) and report["problems"][0]["status"] in {"diagnosed", "abstained"},
        "conclusions_evidence_bound": bool(diagnosed) and all(item["evidence_ids"] for item in diagnosed),
        "never_auto_published": report["published"] is False and report["review_status"] == "pending_review",
        "no_permission_refused": refused["status"] == "refused_no_permission" and not refused["problems"],
        "llm_unavailable_degrades_safely": fallback_report["decision_driver"] == "workflow_fallback",
    }
    return {
        "evaluation_mode": "offline_scripted_protocol",
        "provider_enabled": False,
        "dataset_version": "open-diagnosis-v1",
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report": report,
        "fallback_report": fallback_report,
        "refused_report": refused,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    result = _run()
    output = ROOT / "data" / "evaluations" / "open-diagnosis-eval.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "checks": result["checks"], "output": str(output)}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

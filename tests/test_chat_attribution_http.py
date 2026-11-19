"""HTTP-level integration coverage for the governed chat investigation loop.

External provider and business-source boundaries are deterministic fixtures;
the FastAPI route, compiled LangGraph, candidate selection, authorization,
attribution, evidence transitions, finalization and response assembly are real.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver
import pytest

from app.agents.contracts import ToolInvocation
from app.api.auth import require_tenant_context
from app.investigation.evidence import build_attribution_evidence
from app.main import app
from app.metrics.authorization import authorize_metric
from app.tenancy import TenantContext


def test_chat_http_runs_governed_attribution_and_dimension_drilldown(monkeypatch):
    from app.graph import workflow
    from app.graph.nodes import diagnostic as diagnostic_node
    from app.investigation.contracts import InvestigationState
    context = TenantContext(
        tenant_id="http-attribution-tenant",
        user_id="http-attribution-user",
        role="admin",
        scope_type="platform",
    )
    previous_override = app.dependency_overrides.get(require_tenant_context)
    app.dependency_overrides[require_tenant_context] = lambda: context
    monkeypatch.setattr(workflow, "get_checkpointer", lambda: MemorySaver())

    baseline_start, baseline_end = date(2025, 1, 1), date(2025, 1, 31)
    current_start, current_end = date(2025, 2, 1), date(2025, 2, 28)
    first, last = baseline_start, current_end
    monkeypatch.setattr(
        diagnostic_node,
        "_dates",
        lambda _question: (first.isoformat(), last.isoformat()),
    )
    monkeypatch.setattr(
        diagnostic_node,
        "_in_scope_comparison_windows",
        lambda *_args, **_kwargs: (
            baseline_start, baseline_end, current_start, current_end, "month",
        ),
    )

    authorized_metrics = [
        authorize_metric(
            metric_id,
            provenance_type="exact",
            provenance_id=f"http-test:{metric_id}",
            scope="platform",
            scope_data={"tenant_id": context.tenant_id, "user_id": context.user_id},
        ).model_dump(mode="json")
        for metric_id in ("gmv", "order_count", "avg_order_value")
    ]

    def fake_triage(state):
        state["task_type"] = "anomaly_analysis"
        state["next_agent"] = "data_query"
        state.setdefault("handoffs", []).append({
            "from": "triage", "to": "data_query", "reason": "HTTP integration fixture",
        })
        state.setdefault("node_history", []).append({"node": "triage", "summary": "fixture route"})
        return state

    def fake_data_query(state):
        state["diagnostic_target_metric"] = "gmv"
        state["authorized_metrics"] = deepcopy(authorized_metrics)
        invocation = ToolInvocation(
            server="test", tool_name="query_gmv", arguments={}, status="ok",
            request_id="http-attribution-initial", idempotency_key="http-attribution-initial",
        )
        state.setdefault("tool_calls", []).append(invocation.model_dump(mode="json"))
        state["tool_result"] = {
            "server": "test", "tool_name": "query_gmv", "arguments": {},
            "request_id": "http-attribution-initial",
            "result": {"status": "ok", "metric_result": {"metric_name": "gmv", "value": 1600.0}},
        }
        state.setdefault("root_cause_evidence", [])
        state.setdefault("evidence_bundles", [])
        series_by_metric = {"gmv": {}, "order_count": {}, "avg_order_value": {}}
        day = first
        while day <= last:
            if day <= baseline_end:
                series_by_metric["gmv"][day] = 1000.0 / 31
                series_by_metric["order_count"][day] = 100.0 / 31
                series_by_metric["avg_order_value"][day] = 10.0
            else:
                series_by_metric["gmv"][day] = 1600.0 / 28
                series_by_metric["order_count"][day] = 200.0 / 28
                series_by_metric["avg_order_value"][day] = 8.0
            day += timedelta(days=1)
        for metric_id, series in series_by_metric.items():
            points = [
                {"bucket": bucket.isoformat(), "value": value}
                for bucket, value in series.items()
            ]
            typed = build_attribution_evidence(
                evidence_id=f"http-trend-{metric_id}",
                evidence_type="trend",
                metric_id=metric_id,
                trace_id=str(state.get("trace_id") or "http-test-trace"),
                claim="authorized complete daily test series",
                values={"result": {"status": "ok", "points": points}},
                source_action_id=f"http-query-{metric_id}",
                source_ref="query_metric_trend",
                scope={"tenant_id": context.tenant_id, "user_id": context.user_id},
                method_id="observed_trend_v1",
                method_version="1.0",
                authorized_metric_provenance="tool_gateway_policy_validated",
                comparison={"period": {
                    "start_date": first.isoformat(), "end_date": last.isoformat(),
                    "granularity": "day",
                }},
            )
            state["root_cause_evidence"].append({
                "purpose": "trend_baseline" if metric_id == "gmv" else f"cross_metric:{metric_id}",
                "metric_id": metric_id,
                "query_id": typed.evidence_id,
                "result": {"status": "ok", "result": {"points": points}},
                "investigation_evidence": typed.model_dump(mode="json"),
            })
            state["evidence_bundles"].append({
                "evidence_id": typed.evidence_id,
                "source_type": "test_fixture",
                "source_ref": "query_metric_trend",
                "content": {"result": {"points": points}},
                "trusted": True,
            })
        state.setdefault("node_history", []).append({"node": "data_query", "summary": "fixture data"})
        return state

    def fake_premise_check(state):
        state["premise_check"] = {"status": "aligned"}
        state.setdefault("node_history", []).append({"node": "premise_check", "summary": "aligned"})
        return state

    def fake_diagnostic_plan(state):
        from app.agents.diagnostic_contracts import DiagnosticPlan

        state["diagnostic_target_metric"] = "gmv"
        state["diagnostic_plan"] = DiagnosticPlan.model_validate({
            "goal": "调查 GMV 变化",
            "hypotheses": ["变化可能与订单量及其他结构因素相关"],
            "steps": [{
                "step_id": "adaptive-investigation",
                "objective": "基于证据自适应调查",
                "allowed_actions": ["CALL_TOOL", "FINISH", "ABSTAIN"],
                "completion_condition": "获得有效证据或达到停止条件",
            }],
            "stop_conditions": ["证据不足时停止"],
            "budget": {"max_rounds": 6, "max_tool_calls": 6},
        }).model_dump(mode="json")
        state["diagnostic_step_state"] = {"adaptive-investigation": "completed"}
        diagnostic_node._ensure_graph_investigation_state(state)
        state.setdefault("node_history", []).append({"node": "diagnostic_plan", "summary": "fixture plan"})
        return state

    selected: dict[str, object] = {"round": 0}
    original_catalog = diagnostic_node._candidate_tool_catalog

    def capture_candidates(state):
        catalog = original_catalog(state)
        investigation = InvestigationState.model_validate(state["investigation_state"])
        selected["candidate_set_id"] = (
            investigation.candidate_set.candidate_set_id
            if investigation.candidate_set else None
        )
        selected["evidence_ids"] = [item.evidence_id for item in investigation.evidence]
        selected["completed_actions"] = [
            item.action_type for item in investigation.action_results if item.status == "completed"
        ]
        selected["catalog"] = catalog
        return catalog

    def fake_agent_action(_system_prompt, _user_prompt):
        selected["round"] = int(selected["round"]) + 1
        catalog = selected.get("catalog", {})
        if not catalog:
            return {
                "action_type": "ABSTAIN",
                "reason": "insufficient_evidence",
                "supporting_evidence_ids": selected.get("evidence_ids", []),
            }
        if selected["round"] <= 2:
            # Exercise the HTTP path's bounded schema repair and server-side
            # candidate fallback: raw model-generated queries are rejected.
            raw_query = next(
                action for action in catalog.values()
                if getattr(action, "action_type", None) == "CALL_TOOL"
            )
            return raw_query.model_dump(mode="json")
        completed = set(selected.get("completed_actions", []))
        if "DECOMPOSE_METRIC" not in completed:
            candidate_id = next(
                candidate_id for candidate_id, action in catalog.items()
                if getattr(action, "action_type", None) == "DECOMPOSE_METRIC"
            )
            return {
                "action_type": "CALL_CANDIDATE",
                "candidate_set_id": selected["candidate_set_id"],
                "candidate_id": candidate_id,
            }
        if not any(
            action_type in {"BREAKDOWN_DIMENSION", "DRILL_DOWN"}
            for action_type in completed
        ):
            candidate_id = next(
                candidate_id for candidate_id, action in catalog.items()
                if getattr(action, "metric_name", None) == "order_count"
                and getattr(action, "dimension", None) == "product_category"
            )
            return {
                "action_type": "CALL_CANDIDATE",
                "candidate_set_id": selected["candidate_set_id"],
                "candidate_id": candidate_id,
            }
        return {
            "action_type": "ABSTAIN",
            "reason": "insufficient_evidence",
            "supporting_evidence_ids": selected.get("evidence_ids", []),
        }

    def fake_breakdown_query(query, _state):
        if query.period is not None:
            period_start = date.fromisoformat(query.period.start_date)
            period_end = date.fromisoformat(query.period.end_date)
            points = []
            cursor = period_start
            while cursor <= period_end:
                points.append({"bucket": cursor.isoformat(), "value": 1.0})
                cursor += timedelta(days=1)
            return {
                "tool_name": query.tool_name,
                "arguments": {},
                "result": {
                    "status": "ok",
                    "trend_result": {
                        "status": "ok",
                        "points": points,
                        "count": len(points),
                        "statistics": {
                            "count": len(points), "mean": 1.0, "std": 0.0,
                            "min": 1.0, "q1": 1.0, "median": 1.0,
                            "q3": 1.0, "max": 1.0,
                        },
                        "min_bucket": points[0]["bucket"],
                        "max_bucket": points[0]["bucket"],
                    },
                },
            }
        if query.metric_name == "avg_order_value" and query.dimension == "month_position":
            comparison = query.comparison
            baseline_groups = {"regular": {"numerator": 1000.0, "denominator": 100.0}}
            current_groups = {"regular": {"numerator": 1600.0, "denominator": 200.0}}
            return {
                "tool_name": query.tool_name,
                "arguments": {},
                "result": {
                    "status": "ok",
                    "contribution_result": {
                        "status": "ok",
                        "complete": True,
                        "metric_name": "avg_order_value",
                        "dimension": "month_position",
                        "baseline_period": {
                            "start_date": comparison.baseline_start,
                            "end_date": comparison.baseline_end,
                        },
                        "current_period": {
                            "start_date": comparison.current_start,
                            "end_date": comparison.current_end,
                        },
                        "baseline_groups": baseline_groups,
                        "current_groups": current_groups,
                        "baseline_numerator_total": 1000.0,
                        "current_numerator_total": 1600.0,
                        "baseline_denominator_total": 100.0,
                        "current_denominator_total": 200.0,
                    },
                },
            }
        return {
            "tool_name": query.tool_name,
            "arguments": {},
            "result": {
                "status": "ok",
                "contribution_result": {
                    "baseline_total": 100.0,
                    "current_total": 200.0,
                    "total_delta": 100.0,
                    "baseline_items": [{"dimension_value": "Furniture", "value": 100.0}],
                    "current_items": [{"dimension_value": "Furniture", "value": 200.0}],
                },
            },
        }

    traces: dict[str, dict] = {}

    def capture_trace(trace_id, payload):
        traces[str(trace_id)] = deepcopy(payload)

    def fake_critic(state, **_review_options):
        # The workflow wrapper passes the configured review mode to the critic.
        # This HTTP fixture replaces the critic behavior, but must retain its call contract.
        state["critic_verdict"] = "approved"
        state["critic_revision_target"] = None
        state.setdefault("critic_decisions", []).append({"llm_issues": [], "revision_target": None})
        state.setdefault("node_history", []).append({"node": "critic", "summary": "fixture approved"})
        return state

    def fake_persist_memory(state):
        state.setdefault("node_history", []).append({"node": "persist_memory_p0", "summary": "no-op fixture"})
        return state

    monkeypatch.setattr(workflow, "triage", fake_triage)
    monkeypatch.setattr(workflow, "data_query", fake_data_query)
    monkeypatch.setattr(workflow, "premise_check", fake_premise_check)
    monkeypatch.setattr(workflow, "diagnostic_plan", fake_diagnostic_plan)
    monkeypatch.setattr(workflow, "critic", fake_critic)
    monkeypatch.setattr(workflow, "persist_memory_p0", fake_persist_memory)
    monkeypatch.setattr(workflow, "write_trace", capture_trace)
    monkeypatch.setattr(diagnostic_node, "_candidate_tool_catalog", capture_candidates)
    monkeypatch.setattr("app.llm.client.llm_agent_action", fake_agent_action)
    monkeypatch.setattr(diagnostic_node, "_execute_query", fake_breakdown_query)

    try:
        response = TestClient(app).post("/chat", json={
            "message": "为什么 2025 年 2 月 GMV 相比 1 月变化？",
            "memory_enabled": False,
        })
    finally:
        if previous_override is None:
            app.dependency_overrides.pop(require_tenant_context, None)
        else:
            app.dependency_overrides[require_tenant_context] = previous_override

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["task_type"] == "anomaly_analysis"
    assert body["trace_id"] in traces
    assert body["answer"]
    if body["error"]:
        error_state = InvestigationState.model_validate(
            traces[body["trace_id"]].get("investigation_state") or {}
        )
        raise AssertionError({
            "errors": [item.get("code") for item in body.get("errors", [])],
            "actions": [
                (item.action_type, item.metric_id, item.dimension, item.status,
                 item.failure_code, item.failure_detail)
                for item in error_state.action_results
            ],
        })
    assert body["errors"] == []

    trace = traces[body["trace_id"]]
    assert not trace.get("errors"), [item.get("code") for item in trace.get("errors", [])]
    history = [item["node"] for item in trace["node_history"]]
    executed_nodes = [item["node"] for item in trace["node_timings"]]
    for required_node in (
        "data_query", "premise_check", "diagnostic_plan", "diagnostic_decide",
        "policy_execute", "observe", "diagnostic_finalize", "generate_answer",
        "critic",
    ):
        assert required_node in executed_nodes, {
            "missing_node": required_node,
            "executed_nodes": executed_nodes,
            "errors": trace.get("errors"),
            "action": trace.get("diagnostic_action"),
            "decision_round": selected.get("round"),
        }
    assert "persist_trace" in history

    investigation = InvestigationState.model_validate(trace["investigation_state"])
    assert investigation.evidence
    assert {item.trace_id for item in investigation.evidence} == {body["trace_id"]}
    assert len(investigation.action_results) == 6
    assert investigation.remaining_budget == 0
    assert all(item.status == "completed" for item in investigation.action_results)
    product_evidence = next(
        item for item in investigation.evidence if item.method_id == "product_shapley_v1"
    )
    assert product_evidence.values["total_delta"] == pytest.approx(600.0)
    dimension_evidence = next(
        item for item in investigation.evidence
        if item.evidence_type == "dimension_contribution" and item.metric_id == "order_count"
    )
    assert dimension_evidence.values["total_delta"] == pytest.approx(100.0)
    assert dimension_evidence.values["query"]["dimension"] == "product_category"
    decision_sources = [item.decision_source for item in investigation.action_results]
    assert "workflow_fallback" in decision_sources
    assert "agent" in decision_sources
    fallback_event = next(
        item for item in investigation.observations
        if item.get("event") == "workflow_fallback_selected_candidate"
    )
    assert fallback_event["candidate_set_id"]
    assert fallback_event["reason"]
    assert trace["diagnostic_llm_failures"][-1]["recovered"] is False
    assert not trace.get("errors")

    from app.investigation.evaluation import (
        ActionMatcher,
        EvidenceMatcher,
        InvestigationEvaluationCase,
        evaluate_investigation_path,
    )

    path_case = InvestigationEvaluationCase(
        case_id="http-gmv-product-driver-path",
        target_metric_id="gmv",
        max_actions=6,
        required_action_groups=[
            [ActionMatcher(
                action_type="DECOMPOSE_METRIC",
                metric_id="gmv",
                method_id="product_shapley_v1",
            )],
            [
                ActionMatcher(
                    action_type="BREAKDOWN_DIMENSION",
                    metric_id="order_count",
                    dimension="product_category",
                ),
                ActionMatcher(
                    action_type="DRILL_DOWN",
                    metric_id="order_count",
                    dimension="product_category",
                ),
            ],
        ],
        allowed_action_matchers=[
            ActionMatcher(
                action_type="DECOMPOSE_METRIC",
                metric_id="gmv",
                method_id="product_shapley_v1",
            ),
            ActionMatcher(
                action_type="BREAKDOWN_DIMENSION",
                metric_id="order_count",
                dimension="product_category",
            ),
            ActionMatcher(
                action_type="DRILL_DOWN",
                metric_id="order_count",
                dimension="product_category",
            ),
            ActionMatcher(
                action_type="BREAKDOWN_DIMENSION",
                metric_id="avg_order_value",
                dimension="month_position",
            ),
            ActionMatcher(action_type="ANALYZE_TREND", metric_id="order_count"),
            ActionMatcher(action_type="ANALYZE_DISTRIBUTION", metric_id="order_count"),
            ActionMatcher(action_type="CHECK_ANOMALY", metric_id="order_count"),
        ],
        required_evidence_groups=[
            [EvidenceMatcher(
                evidence_type="metric_decomposition",
                metric_id="gmv",
                method_id="product_shapley_v1",
            )],
            [EvidenceMatcher(
                evidence_type="dimension_contribution",
                metric_id="order_count",
            )],
        ],
        allowed_stop_reasons=[
            "ENOUGH_EVIDENCE",
            "NO_HIGH_VALUE_ACTION",
            "CAPABILITY_GAP",
            "DATA_INSUFFICIENT",
            "USER_SCOPE_LIMIT",
            "BUDGET_EXHAUSTED",
        ],
    )
    path_quality = evaluate_investigation_path(path_case, investigation)
    assert path_quality.passed, (
        f"issues={path_quality.issues}; stop={investigation.stop_reason}; "
        f"remaining_budget={investigation.remaining_budget}; "
        f"diagnostic_round={trace.get('diagnostic_round')}; "
        f"max_rounds={trace.get('diagnostic_plan', {}).get('budget', {}).get('max_rounds')}; "
        f"decision_history={[item.get('summary') for item in trace.get('node_history', []) if item.get('node') == 'diagnostic_decide']}; "
        f"actions={[(item.action_type, item.status, item.failure_code) for item in investigation.action_results]}; "
        f"llm_failures={[(item.get('attempt'), item.get('class'), item.get('recovered')) for item in trace.get('diagnostic_llm_failures', [])]}"
    )
    assert path_quality.action_obligation_coverage == 1.0
    assert path_quality.evidence_obligation_coverage == 1.0
    assert path_quality.evidence_grounding_rate == 1.0
    assert path_quality.agent_action_count == 1
    assert path_quality.action_relevance_rate == 1.0
    assert path_quality.fallback_action_count == 1


def test_chat_http_provider_unavailable_closes_with_verified_point_decomposition(monkeypatch):
    """The authorized route gives a bounded numeric terminal without a Provider."""
    import contextlib
    import io

    from app.graph import workflow
    from app.graph.nodes import diagnostic as diagnostic_node
    from app.investigation.contracts import InvestigationState
    from evaluation.fixtures import offline_dependencies

    context = TenantContext(
        tenant_id="terminal-gmv-tenant", user_id="terminal-gmv-user",
        role="admin", scope_type="platform",
    )
    previous_override = app.dependency_overrides.get(require_tenant_context)
    app.dependency_overrides[require_tenant_context] = lambda: context
    monkeypatch.setattr(workflow, "get_checkpointer", lambda: MemorySaver())
    traces: dict[str, dict] = {}

    def targeted_triage(state):
        state["task_type"] = "anomaly_analysis"
        state["next_agent"] = "data_query"
        state.setdefault("handoffs", []).append({
            "from": "triage", "to": "data_query", "reason": "authorized terminal regression",
        })
        return state

    def capture_trace(trace_id, payload):
        traces[str(trace_id)] = deepcopy(payload)

    def execute_query(query, _state):
        if query.period is not None:
            start, end = date.fromisoformat(query.period.start_date), date.fromisoformat(query.period.end_date)
            metric = query.metric_name
            baseline, current = {
                "gmv": (1000.0, 600.0), "order_count": (100.0, 60.0),
                "avg_order_value": (10.0, 10.0), "delivery_delay_rate": (10.0, 10.0),
            }.get(metric, (1.0, 1.0))
            points, day = [], start
            while day <= end:
                points.append({
                    "bucket": day.isoformat(),
                    "value": current if day == date(2018, 4, 18) else baseline,
                })
                day += timedelta(days=1)
            if query.tool_name == "query_metric_statistics":
                numbers = [item["value"] for item in points]
                return {"tool_name": query.tool_name, "arguments": {}, "result": {
                    "status": "ok", "statistics_result": {
                        "metric_name": metric, "start_date": start.isoformat(),
                        "end_date": end.isoformat(), "granularity": "day",
                        "distribution_basis": "time_bucket_values", "count": len(numbers),
                        "statistics": {
                            "count": len(numbers), "mean": sum(numbers) / len(numbers),
                            "std": 0.0, "min": min(numbers), "q1": min(numbers),
                            "median": sum(numbers) / len(numbers), "q3": max(numbers),
                            "max": max(numbers),
                        },
                    },
                }}
            return {"tool_name": query.tool_name, "arguments": {}, "result": {
                "status": "ok", "trend_result": {
                    "metric_name": metric, "start_date": start.isoformat(),
                    "end_date": end.isoformat(), "granularity": "day", "points": points,
                },
            }}

        baseline_total, current_total = 1000.0, 600.0
        return {"tool_name": query.tool_name, "arguments": {}, "result": {
            "status": "ok", "contribution_result": {
                "status": "ok", "metric_name": query.metric_name,
                "dimension": query.dimension,
                "current_date": query.comparison.current_date,
                "baseline_date": query.comparison.baseline_date,
                "baseline_total": baseline_total, "current_total": current_total,
                "total_delta": current_total - baseline_total,
                "baseline_items": [
                    {"dimension_value": f"K{i}", "value": baseline_total / 200}
                    for i in range(200)
                ],
                "current_items": [
                    {"dimension_value": f"K{i}", "value": current_total / 200}
                    for i in range(200)
                ],
            },
        }}

    monkeypatch.setattr(workflow, "triage", targeted_triage)
    monkeypatch.setattr(workflow, "write_trace", capture_trace)
    monkeypatch.setattr(diagnostic_node, "_execute_query", execute_query)
    try:
        with offline_dependencies(), contextlib.redirect_stdout(io.StringIO()):
            # The old Golden query stub predates item-wise Critic contracts.
            # Preserve its facts, but emit the requested identity and executed
            # metric/operation as the real gateway does.
            from evaluation.fixtures import fixture_mcp
            import importlib
            data_query_node = importlib.import_module("app.graph.nodes.data_query")

            def bound_fixture_query(**kwargs):
                result = fixture_mcp(**kwargs)
                plan = kwargs.get("plan")
                result["request_id"] = kwargs["request_id"]
                result["idempotency_key"] = kwargs["request_id"]
                if plan is not None:
                    result["result"]["parsed"] = plan.model_dump(mode="json")
                return result

            monkeypatch.setattr(data_query_node, "query_business_data", bound_fixture_query)
            response = TestClient(app).post("/chat", json={
                "message": "为什么 2018-04-18 的 GMV 下降了？", "memory_enabled": False,
            })
    finally:
        if previous_override is None:
            app.dependency_overrides.pop(require_tenant_context, None)
        else:
            app.dependency_overrides[require_tenant_context] = previous_override

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["error"] is False, body.get("errors")
    trace = traces[body["trace_id"]]
    investigation = InvestigationState.model_validate(trace["investigation_state"])
    decomposition = next(item for item in investigation.evidence if item.evidence_type == "metric_decomposition")
    assert decomposition.method_id == "product_shapley_v1"
    assert decomposition.comparison["baseline_period"] == {
        "start_date": "2018-04-17", "end_date": "2018-04-17",
    }
    assert decomposition.comparison["current_period"] == {
        "start_date": "2018-04-18", "end_date": "2018-04-18",
    }
    assert decomposition.values["baseline_factor_values"] == {
        "order_count": 100.0, "avg_order_value": 10.0,
    }
    assert decomposition.values["current_factor_values"] == {
        "order_count": 60.0, "avg_order_value": 10.0,
    }
    assert decomposition.values["total_delta"] == pytest.approx(-400.0)
    assert investigation.action_results[-1].action_type == "DECOMPOSE_METRIC"
    assert investigation.action_results[-1].validation_result.status == "valid"
    assert investigation.stop_reason.code == "ENOUGH_EVIDENCE"
    assert not trace.get("errors")
    assert trace["finish_decision"]["found_factor"] == "order_count"
    assert trace["critic_verdict"] == "approved"
    assert "订单量变化解释了成交总额的数值下降" in trace["analysis_result"]["summary"]
    assert "业务原因" in trace["analysis_result"]["summary"]

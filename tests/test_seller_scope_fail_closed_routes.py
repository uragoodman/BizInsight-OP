from __future__ import annotations

import pytest

from app.graph.nodes.branch_evidence import branch_evidence
from app.graph.nodes.data_query import data_query
from app.graph.nodes.diagnostic import diagnostic_plan
from app.graph.nodes.report import report
from app.graph.nodes.seller_scope_guard import SELLER_SCOPE_UNSUPPORTED_REASON
from app.graph.workflow import _route_after_data_query, _route_after_triage


@pytest.mark.parametrize("task_type", ["anomaly_analysis", "report_generation"])
def test_triage_routes_non_metric_seller_operations_to_terminal_guard(task_type: str) -> None:
    state = {
        "task_type": task_type,
        "next_agent": "data_query",
        "seller_scope_payload": {"signed": True},
    }

    assert _route_after_triage(state) == "seller_scope_unsupported"


def test_seller_metric_query_skips_generic_analysis_memory_retrieval() -> None:
    assert _route_after_triage({
        "task_type": "metric_query",
        "seller_scope_payload": {"signed": True},
    }) == "data_query"


@pytest.mark.parametrize("reason", ["ambiguous_dimension", "ambiguous_metric"])
def test_seller_scope_ambiguity_routes_to_clarification_before_data_query(reason) -> None:
    assert _route_after_triage({
        "task_type": "metric_query",
        "seller_scope_payload": {"signed": True},
        "context_compilation": {
            "needs_clarification": True,
            "clarification_reason": reason,
        },
    }) == "context_clarification"


@pytest.mark.parametrize("task_type", ["anomaly_analysis", "report_generation"])
def test_data_query_direct_entry_refuses_before_any_mcp_call(monkeypatch, task_type: str) -> None:
    def unexpected_call(*_args, **_kwargs):
        pytest.fail("seller-scoped diagnostic/report must not call a generic MCP tool")

    monkeypatch.setattr("app.graph.nodes.data_query.query_business_data", unexpected_call)
    monkeypatch.setattr("app.graph.nodes.data_query.query_scoped_business_data", unexpected_call)
    state = {
        "task_type": task_type,
        "user_message": "分析异常并生成报告",
        "seller_scope_payload": {"signed": True},
        "trace_id": "trace-seller-guard",
    }

    result = data_query(state)

    assert result["seller_scope_terminal"] is True
    assert result["tool_result"]["result"]["status"] == "unsupported"
    assert result["tool_result"]["result"]["reason_code"] == "seller_scope_operation_unsupported"
    assert result["answer"] == SELLER_SCOPE_UNSUPPORTED_REASON
    assert result["query_results"] == []


def test_data_query_router_cannot_enter_diagnostic_or_report_with_seller_scope() -> None:
    for task_type in ("anomaly_analysis", "report_generation"):
        assert _route_after_data_query({
            "task_type": task_type,
            "seller_scope_payload": {"signed": True},
            "tool_result": {"result": {"status": "ok"}},
        }) == "seller_scope_unsupported"

    assert _route_after_data_query({
        "task_type": "metric_query",
        "seller_scope_payload": {"signed": True},
        "registered_decomposition_request": {"status": "ready"},
    }) == "seller_scope_unsupported"


def test_report_builder_is_not_called_for_seller_scope(monkeypatch) -> None:
    def unexpected_report(*_args, **_kwargs):
        pytest.fail("platform periodic report builder must not run for seller scope")

    monkeypatch.setattr("app.graph.nodes.report.build_periodic_report_artifact", unexpected_report)
    state = {
        "task_type": "report_generation",
        "user_message": "生成本月经营报告",
        "seller_scope_payload": {"signed": True},
    }

    result = report(state)

    assert result["seller_scope_terminal"] is True
    assert result["tool_result"]["result"]["status"] == "unsupported"
    assert result["answer"] == SELLER_SCOPE_UNSUPPORTED_REASON


def test_diagnostic_nodes_do_not_query_or_plan_for_seller_scope(monkeypatch) -> None:
    def unexpected_llm(*_args, **_kwargs):
        pytest.fail("diagnostic planning must not run for seller scope")

    def unexpected_query(*_args, **_kwargs):
        pytest.fail("branch evidence must not fall back to platform query")

    monkeypatch.setattr("app.llm.client.llm_agent_action", unexpected_llm)
    monkeypatch.setattr("app.graph.nodes.branch_evidence.query_business_data", unexpected_query)
    state = {
        "task_type": "anomaly_analysis",
        "user_message": "为什么 GMV 下降",
        "seller_scope_payload": {"signed": True},
    }

    planned = diagnostic_plan(dict(state))
    branched = branch_evidence(dict(state))

    assert planned["seller_scope_terminal"] is True
    assert branched["seller_scope_terminal"] is True
    assert planned["tool_result"]["result"]["status"] == "unsupported"
    assert branched["tool_result"]["result"]["status"] == "unsupported"


@pytest.mark.parametrize(
    "scope_type,scope_ids,tenant_scope,expected_auth_scope",
    [
        ("single", ("A",), "seller", "seller"),
        ("selected", ("A", "B"), "platform", "platform"),
        ("all", (), "platform", "platform"),
    ],
)
def test_planner_uses_server_verified_seller_constraints_without_ids(
    monkeypatch, scope_type, scope_ids, tenant_scope, expected_auth_scope,
) -> None:
    from datetime import date

    from app.core.config import Settings
    from app.graph.nodes import data_query as data_query_node
    from app.scope.seller_scope import SellerScopeIntent, issue_seller_scope_payload
    from app.tenancy import SellerGrant, TenantContext

    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: Settings(
            tenant_auth_secret="scope-test-secret",
            seller_dimension_enabled=True,
            seller_multi_scope_enabled=True,
        ),
    )
    grants = tuple(
        SellerGrant(seller_id, "2024-01-01", "2024-01-31")
        for seller_id in ("A", "B")
    )
    tenant = TenantContext(
        tenant_id="tenant-planner", user_id="user-planner", role="admin",
        scope_type=tenant_scope, seller_id="A" if tenant_scope == "seller" else None,
        seller_grants=grants,
    )
    intent = SellerScopeIntent(
        seller_scope_type=scope_type, seller_ids=scope_ids,
        start_date=date(2024, 1, 1), end_date=date(2024, 1, 31),
        metric_ids=("gmv", "order_count"), dimensions=("seller",),
    )
    payload = issue_seller_scope_payload(intent, tenant=tenant)
    monkeypatch.setattr(data_query_node, "current_tenant_context", lambda: tenant)

    scope, planner_data, resolved_tenant = data_query_node._seller_scope_planning_context(payload)

    assert scope == expected_auth_scope
    assert resolved_tenant is tenant
    assert planner_data["scope_type"] == expected_auth_scope
    assert planner_data["seller_scope_type"] == scope_type
    assert planner_data["allowed_metrics"] == ["gmv", "order_count"]
    assert planner_data["allowed_dimensions"] == ["seller"]
    assert planner_data["start_date"] == "2024-01-01"
    assert planner_data["end_date"] == "2024-01-31"
    assert isinstance(planner_data["query_scope_fingerprint"], str)
    assert len(planner_data["query_scope_fingerprint"]) == 64
    assert "seller_ids" not in planner_data
    assert "seller_scope_payload" not in planner_data
    assert payload["provenance"]["token"] not in str(planner_data)

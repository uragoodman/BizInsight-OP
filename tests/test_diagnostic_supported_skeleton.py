from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from app.graph.nodes import diagnostic as diagnostic_node
from app.metrics.authorization import authorize_metric
from app.schemas.explorer import ControlledAnalysisScope
from app.tenancy import SellerGrant, TenantContext, activate_tenant_context


def test_locked_scope_without_trusted_tenant_does_not_force_legacy_evidence_floor(monkeypatch):
    monkeypatch.setattr("app.tenancy.current_tenant_context", lambda: None)
    state = {
        "controlled_analysis_scope": {
            "seller_id": "unavailable-context-seller",
            "metric_name": "seller_order_bad_review_rate",
            "default_start_date": "2024-01-01",
            "default_end_date": "2024-01-31",
            "granularity": "month",
            "allowed_metrics": ["seller_order_bad_review_rate"],
        },
        "diagnostic_target_metric": "seller_order_bad_review_rate",
        "user_message": "分析本月订单差评率与配送状态的统计关联",
    }

    assert diagnostic_node._required_evidence_kinds_for(state) == set()


def test_supported_association_is_not_blocked_by_unavailable_required_steps(monkeypatch):
    seller_id = "assoc-skeleton-seller"
    metric_id = "seller_order_bad_review_rate"
    start, end = date(2024, 1, 1), date(2024, 1, 31)
    context = TenantContext(
        tenant_id="assoc-skeleton-tenant",
        user_id="assoc-skeleton-user",
        role="admin",
        scope_type="seller",
        seller_id=seller_id,
        seller_grants=(SellerGrant(
            seller_id=seller_id,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        ),),
    )
    scope = ControlledAnalysisScope.model_validate({
        "seller_id": seller_id,
        "allowed_metrics": [metric_id],
        "default_start_date": start,
        "default_end_date": end,
        "metric_name": metric_id,
        "granularity": "day",
    })
    authorized_metric = authorize_metric(
        metric_id,
        provenance_type="explorer_selection",
        provenance_id="test-explorer-selection",
        scope="seller",
        scope_data={
            **scope.model_dump(mode="json"),
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
        },
    )
    state = {
        "trace_id": "association-supported-skeleton-test",
        "task_type": "anomaly_analysis",
        "user_message": "分析 2024-01-01 至 2024-01-31 店铺订单差评率的关联因素",
        "controlled_analysis_scope": scope.model_dump(mode="json"),
        "diagnostic_target_metric": metric_id,
        "authorized_metrics": [authorized_metric.model_dump(mode="json")],
        "root_cause_evidence": [{
            "purpose": "trend_baseline",
            "metric_id": metric_id,
        }],
        "tool_calls": [],
        "runtime_budget": {"max_tool_calls": 8, "max_graph_steps": 32},
        "diagnostic_step_state": {},
        "diagnostic_capability_gaps": [],
        "node_history": [],
    }
    plan = diagnostic_node.DiagnosticPlan.model_validate({
        "goal": "检查低评分订单率的关联因素",
        "hypotheses": ["准时状态与低评分可能存在统计关联"],
        "steps": [
            {
                "step_id": "target-trend",
                "objective": "确认目标指标趋势",
                "kind": "trend",
                "required": True,
                "allowed_actions": ["CALL_TOOL", "FINISH", "ABSTAIN"],
                "completion_condition": "已取得目标趋势",
            },
            {
                "step_id": "dimension",
                "objective": "按维度拆解目标指标",
                "kind": "dimension_contribution",
                "required": True,
                "allowed_actions": ["CALL_TOOL", "FINISH", "ABSTAIN"],
                "completion_condition": "已取得维度拆解",
            },
            {
                "step_id": "related",
                "objective": "检查相关经营指标",
                "kind": "related_metric",
                "required": True,
                "allowed_actions": ["CALL_TOOL", "FINISH", "ABSTAIN"],
                "completion_condition": "已取得相关指标",
            },
        ],
        "stop_conditions": ["证据不足"],
        "budget": {"max_rounds": 6, "max_tool_calls": 8},
    })
    selected: dict[str, object] = {}

    original_catalog = diagnostic_node._candidate_tool_catalog

    # Seller-scoped tools are default-deny for unknown tenant IDs. Grant the
    # specific test tenant's governed data-query tools so this test exercises
    # candidate selection rather than the expected policy denial path.
    monkeypatch.setattr(
        "app.security.governance.ToolPermissionPolicy.allows",
        lambda _self, _tenant, _role, agent, tool, **_kwargs: (
            agent == "data_query"
            and tool in {
                "query_metric_trend",
                "query_scoped_registered_association",
            }
        ),
    )

    def capture_candidates(current_state):
        catalog = original_catalog(current_state)
        investigation = diagnostic_node._ensure_graph_investigation_state(current_state)
        selected["candidate_set_id"] = (
            investigation.candidate_set.candidate_set_id
            if investigation.candidate_set else None
        )
        selected["catalog"] = catalog
        selected["investigation"] = investigation
        return catalog

    def select_registered_association(_prompt, _question):
        catalog = selected["catalog"]
        assert selected["candidate_set_id"] is not None
        candidate_id = next(
            key for key, action in catalog.items()
            if action.tool_name == "query_scoped_registered_association"
            and action.association_pair_id == "delivery_timing_bad_review_v1"
        )
        return {
            "action_type": "CALL_CANDIDATE",
            "candidate_set_id": selected["candidate_set_id"],
            "candidate_id": candidate_id,
        }

    monkeypatch.setattr(diagnostic_node, "_candidate_tool_catalog", capture_candidates)
    monkeypatch.setattr("app.llm.client.llm_agent_action", select_registered_association)

    with activate_tenant_context(context):
        supported_kinds = diagnostic_node._required_evidence_kinds_for(state)
        plan, _ = diagnostic_node._ensure_required_skeleton(
            plan, supported_kinds=supported_kinds,
        )
        state["diagnostic_plan"] = plan.model_dump(mode="json")
        state["diagnostic_step_state"] = {
            step.step_id: "pending" for step in plan.steps
        }
        state["investigation_state"] = diagnostic_node._ensure_graph_investigation_state(state).model_dump(mode="json")
        diagnostic_node.diagnostic_decide(state)

    assert supported_kinds == {"trend"}
    required_kinds = {
        step.kind for step in plan.steps if step.required
    }
    assert required_kinds == {"trend"}
    assert selected["candidate_set_id"] is not None
    chosen = diagnostic_node._ACTION_ADAPTER.validate_python(state["diagnostic_action"])
    assert chosen.tool_name == "query_scoped_registered_association"
    assert chosen.association_pair_id == "delivery_timing_bad_review_v1"
    assert state["diagnostic_selected_candidate_id"] in selected["catalog"]


def test_diagnostic_plan_prompt_and_budget_follow_supported_evidence_kinds(monkeypatch):
    seller_id = "assoc-plan-seller"
    metric_id = "seller_order_bad_review_rate"
    start, end = date(2024, 1, 1), date(2024, 1, 31)
    context = TenantContext(
        tenant_id="assoc-plan-tenant",
        user_id="assoc-plan-user",
        role="admin",
        scope_type="seller",
        seller_id=seller_id,
        seller_grants=(SellerGrant(
            seller_id=seller_id,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        ),),
    )
    scope = ControlledAnalysisScope.model_validate({
        "seller_id": seller_id,
        "allowed_metrics": [metric_id],
        "default_start_date": start,
        "default_end_date": end,
        "metric_name": metric_id,
        "granularity": "day",
    })
    response = {
        "goal": "分析低评分订单率",
        "hypotheses": ["可用证据可能解释指标变化"],
        "steps": [
            {
                "step_id": "trend", "objective": "查询趋势", "kind": "trend",
                "required": True, "allowed_actions": ["CALL_TOOL", "FINISH"],
                "completion_condition": "取得趋势",
            },
            {
                "step_id": "dimension", "objective": "维度拆解",
                "kind": "dimension_contribution", "required": True,
                "allowed_actions": ["CALL_TOOL", "FINISH"],
                "completion_condition": "取得拆解",
            },
            {
                "step_id": "related", "objective": "相关指标",
                "kind": "related_metric", "required": True,
                "allowed_actions": ["CALL_TOOL", "FINISH"],
                "completion_condition": "取得相关指标",
            },
        ],
        "stop_conditions": ["证据不足"],
        "budget": {"max_rounds": 1, "max_tool_calls": 1},
    }
    prompts: list[str] = []
    monkeypatch.setattr(
        "app.security.governance.ToolPermissionPolicy.allows",
        lambda _self, _tenant, _role, agent, tool, **_kwargs: (
            agent == "data_query" and tool == "query_metric_trend"
        ),
    )
    monkeypatch.setattr(
        "app.llm.client.llm_agent_action",
        lambda prompt, _question: prompts.append(prompt) or response,
    )
    state = {
        "trace_id": "association-plan-contract-test",
        "task_type": "anomaly_analysis",
        "user_message": "分析 2024-01 店铺订单差评率的变化",
        "controlled_analysis_scope": scope.model_dump(mode="json"),
        "diagnostic_target_metric": metric_id,
        "tool_calls": [],
        "runtime_budget": {"max_tool_calls": 4, "max_graph_steps": 12},
        "diagnostic_step_state": {},
        "node_history": [],
        "errors": [],
    }

    with activate_tenant_context(context):
        result = diagnostic_node.diagnostic_plan(state)

    plan = diagnostic_node.DiagnosticPlan.model_validate(result["diagnostic_plan"])
    assert len(prompts) == 1
    assert "hypotheses 和 stop_conditions 必须是字符串数组" in prompts[0]
    assert "hypotheses 每项是简短字符串，禁止输出 {text,status} 对象" in prompts[0]
    assert '"hypotheses":["订单量变化可能与目标指标变化有关"]' in prompts[0]
    enforcement_text = prompts[0].split("注意：服务端只会强制采集", 1)[1].split("budget 必须", 1)[0]
    assert "目标指标趋势" in enforcement_text
    assert "维度贡献拆解" not in enforcement_text
    assert "已授权相关指标趋势" not in enforcement_text
    assert {step.kind for step in plan.steps if step.required} == {"trend"}
    assert plan.budget.max_tool_calls == 2


def test_required_related_metric_query_needs_authorized_metric_provenance(monkeypatch):
    from app.agents.diagnostic_contracts import DiagnosticStep

    seller_id = "assoc-related-seller"
    target_metric = "seller_order_bad_review_rate"
    related_metric = "seller_avg_freight_per_order"
    start, end = date(2024, 1, 1), date(2024, 1, 31)
    context = TenantContext(
        tenant_id="assoc-related-tenant",
        user_id="assoc-related-user",
        role="admin",
        scope_type="seller",
        seller_id=seller_id,
        seller_grants=(SellerGrant(seller_id=seller_id),),
    )
    scope = ControlledAnalysisScope.model_validate({
        "seller_id": seller_id,
        "allowed_metrics": [target_metric, related_metric],
        "default_start_date": start,
        "default_end_date": end,
        "metric_name": target_metric,
        "granularity": "day",
    })
    state = {
        "user_message": "分析店铺订单差评率关联因素",
        "diagnostic_target_metric": target_metric,
        "controlled_analysis_scope": scope.model_dump(mode="json"),
        "authorized_metrics": [],
        "root_cause_evidence": [],
        "diagnostic_denied_metrics": [],
        "runtime_budget": {"max_tool_calls": 4},
    }
    step = DiagnosticStep(
        step_id="related", objective="检查相关指标", kind="related_metric",
        allowed_actions=["PARALLEL_FACTS"], completion_condition="已取得相关指标",
        required=True,
    )
    monkeypatch.setattr(
        diagnostic_node, "_related_metrics", lambda *_args, **_kwargs: [related_metric],
    )
    monkeypatch.setattr(
        "app.security.governance.ToolPermissionPolicy.allows",
        lambda _self, _tenant, _role, agent, tool, **_kwargs: (
            agent == "data_query" and tool == "query_metric_trend"
        ),
    )

    with activate_tenant_context(context):
        assert diagnostic_node._authorized_related_metrics(
            state, scope_name="seller",
        ) == []
        assert diagnostic_node._required_action_for(state, step) is None

        state["authorized_metrics"] = [{
            "metric_id": related_metric,
            "provenance_type": "explorer_selection",
            "provenance_id": "server-issued-test-selection",
        }]
        assert diagnostic_node._authorized_related_metrics(
            state, scope_name="seller",
        ) == [related_metric]
        action = diagnostic_node._required_action_for(state, step)

    assert action is not None
    assert action.action_type == "PARALLEL_FACTS"
    assert [fact.metric_name for fact in action.facts] == [related_metric]


def test_platform_registered_related_metrics_receive_server_credentials():
    from app.metrics.authorization import authorize_metric

    context = TenantContext(
        tenant_id="agent-eval-tenant", user_id="agent-eval-user",
        role="admin", scope_type="platform",
    )
    scope_data = {
        "tenant_id": context.tenant_id, "user_id": context.user_id,
        "permissions": context.role, "scope_type": "platform",
    }
    target = authorize_metric(
        "gmv", provenance_type="exact", provenance_id="mention:gmv",
        scope="platform", scope_data=scope_data,
    )
    state = {
        "user_message": "为什么 GMV 下降？",
        "diagnostic_target_metric": "gmv",
        "authorized_metrics": [target.model_dump(mode="json")],
    }

    with activate_tenant_context(context):
        related = diagnostic_node._authorized_related_metrics(state, scope_name="platform")

    assert related
    credentials = {item["metric_id"]: item for item in state["authorized_metrics"]}
    assert all(credentials[item]["provenance_type"] == "registered_relation" for item in related)


def test_platform_dimension_step_uses_supported_point_comparison_when_week_windows_do_not_fit():
    from app.agents.diagnostic_contracts import DiagnosticStep

    context = TenantContext(
        tenant_id="diagnostic-window-test", user_id="diagnostic-window-admin",
        role="admin", scope_type="platform",
    )
    state = {
        "user_message": "为什么 2017-06-25 的 GMV 下降？",
        "diagnostic_target_metric": "gmv",
        "diagnostic_denied_metrics": [],
        "root_cause_evidence": [],
        "controlled_analysis_scope": None,
        "runtime_budget": {"max_tool_calls": 6},
        "tool_calls": [],
    }
    step = DiagnosticStep(
        step_id="dimension", objective="拆解贡献", kind="dimension_contribution",
        allowed_actions=["CALL_TOOL"], completion_condition="有贡献证据", required=True,
    )

    with activate_tenant_context(context):
        action = diagnostic_node._required_action_for(state, step)

    assert action is not None
    assert action.tool_name == "compare_breakdown_by_dimension"
    assert action.comparison.current_date == "2017-06-25"
    assert action.comparison.baseline_date == "2017-06-24"


def test_server_required_trend_uses_sealed_request_contract_for_live_authorization(monkeypatch):
    from app.context.compiler import compile_turn_context
    from app.graph.nodes.diagnostic import _plan_for_query

    tenant = TenantContext(
        tenant_id="platform", user_id="required-contract-authz-test",
        role="admin", scope_type="platform",
    )
    compiled = compile_turn_context({
        "user_message": (
            "请比较平台订单量在基线日2017-04-07与当前日2017-04-08，"
            "先查询趋势，再按product_category展示统计贡献，不推断业务因果。"
        ),
    })
    context_compilation = compiled["context_compilation"]
    contract = context_compilation["request_contract"]
    checks = []

    class AllowedAuthz:
        def require_action(self, value):
            checks.append(("action", value))

        def require_metrics(self, values):
            checks.append(("metrics", tuple(values)))

        def require_tool(self, value):
            checks.append(("tool", value))

        def require_dimensions(self, values):
            checks.append(("dimensions", tuple(values)))

    monkeypatch.setattr("app.tenancy.current_tenant_context", lambda: tenant)
    monkeypatch.setattr("app.auth.context.build_authorization_context", lambda *_a, **_k: AllowedAuthz())
    state = {
        **compiled,
        "context_compilation": context_compilation,
        "request_contract_hash": contract["contract_hash"],
        "tenant_id": tenant.tenant_id,
        "user_id": tenant.user_id,
        "user_role": tenant.role,
        "scope_type": "platform",
        "authorized_metrics": [],
        "diagnostic_required_action": True,
    }
    query = SimpleNamespace(
        metric_name="order_count", tool_name="query_metric_trend", dimension=None,
        period=SimpleNamespace(start_date="2017-04-07", end_date="2017-04-08", granularity="day"),
        comparison=None, statistics=[], association_pair_id=None,
    )

    plan = diagnostic_node._plan_for_query(query, state)

    assert plan.authorized_metric is not None
    assert plan.authorized_metric.metric_id == "order_count"
    assert plan.authorized_metric.provenance_type == "exact"
    assert plan.authorized_metric.provenance_id == f"request-contract:{contract['contract_hash']}"
    assert state["authorized_metrics"][0]["metric_id"] == "order_count"
    assert checks == [
        ("action", "analyze"), ("metrics", ("order_count",)), ("tool", "query_metric_trend"),
    ]


def test_required_contract_authorization_fails_closed_on_metric_drift_or_missing_identity(monkeypatch):
    from app.context.compiler import compile_turn_context
    from app.graph.nodes.diagnostic import _plan_for_query

    tenant = TenantContext(
        tenant_id="platform", user_id="required-contract-authz-negative-test",
        role="admin", scope_type="platform",
    )
    compiled = compile_turn_context({
        "user_message": (
            "请比较平台订单量在基线日2017-04-07与当前日2017-04-08，"
            "先查询趋势，再按product_category展示统计贡献。"
        ),
    })
    compilation = compiled["context_compilation"]
    contract = compilation["request_contract"]
    state = {
        **compiled,
        "context_compilation": compilation,
        "request_contract_hash": contract["contract_hash"],
        "tenant_id": tenant.tenant_id, "user_id": tenant.user_id,
        "user_role": tenant.role, "scope_type": "platform",
        "authorized_metrics": [], "diagnostic_required_action": True,
    }
    query = SimpleNamespace(
        metric_name="review_score_avg", tool_name="query_metric_trend", dimension=None,
        period=SimpleNamespace(start_date="2017-04-07", end_date="2017-04-08", granularity="day"),
        comparison=None, statistics=[], association_pair_id=None,
    )
    monkeypatch.setattr("app.tenancy.current_tenant_context", lambda: tenant)

    drifted = diagnostic_node._plan_for_query(query, state)
    assert drifted.authorized_metric is None
    assert state["authorized_metrics"] == []

    monkeypatch.setattr("app.tenancy.current_tenant_context", lambda: None)
    query.metric_name = "order_count"
    missing_identity = diagnostic_node._plan_for_query(query, state)
    assert missing_identity.authorized_metric is None
    assert state["authorized_metrics"] == []

from app.graph.nodes.data_query import data_query


def test_single_explicit_month_plans_registered_period_query(monkeypatch) -> None:
    from app.agents.data_query_agent import plan_confirmed_metric_queries
    from app.context.compiler import compile_turn_context

    question = "查询 2018 年 8 月的 GMV。"
    compilation = compile_turn_context({"user_message": question})["context_compilation"]
    monkeypatch.setattr(
        "app.llm.client.llm_extract",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("exact metric must not call LLM")),
    )

    planned = plan_confirmed_metric_queries(
        question,
        operation_intent=compilation["operation_intent"],
        request_contract=compilation["request_contract"],
    )

    assert planned.decision.status == "confirmed"
    assert len(planned.plans) == 1
    plan = planned.plans[0]
    assert plan.query_type == "metric_period"
    assert plan.metric_date is None
    assert (plan.start_date, plan.end_date) == ("2018-08-01", "2018-08-31")
    assert plan.tool_name == "query_metric_period"


def test_period_comparison_followup_plans_registered_month_pair(monkeypatch) -> None:
    from app.agents.data_query_agent import plan_confirmed_metric_queries
    from app.context.compiler import compile_turn_context

    question = "和 2018 年 7 月相比呢？请比较 GMV 并说明变化方向。"
    prior = {
        "source_turn_id": "prior-turn",
        "scope_fingerprint": "scope-a",
        "context_slots": {
            "metric": {"value": "gmv", "source": "current_turn"},
            "period": {
                "value": {"start_date": "2018-08-01", "end_date": "2018-08-31"},
                "source": "current_turn",
            },
        },
    }
    compilation = compile_turn_context({
        "user_message": question,
        "current_query_scope_fingerprint": "scope-a",
        "previous_turn_context": prior,
    })["context_compilation"]
    monkeypatch.setattr(
        "app.llm.client.llm_extract",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("exact metric must not call LLM")),
    )

    planned = plan_confirmed_metric_queries(
        question,
        operation_intent=compilation["operation_intent"],
        request_contract=compilation["request_contract"],
        comparison_periods=compilation["operation_intent"]["comparison_periods"],
    )

    assert compilation["needs_clarification"] is False
    assert planned.decision.status == "confirmed"
    assert len(planned.plans) == 1
    plan = planned.plans[0]
    assert plan.query_type == "metric_compare"
    assert plan.metric_date is None
    assert (plan.baseline_start, plan.baseline_end) == ("2018-07-01", "2018-07-31")
    assert (plan.current_start, plan.current_end) == ("2018-08-01", "2018-08-31")
    assert plan.granularity == "month"
    assert plan.tool_name == "compare_metric_period"


def test_data_query_turns_invalid_gateway_tuple_into_controlled_result(monkeypatch) -> None:
    calls: list[dict] = []

    def invalid_gateway(**kwargs):
        calls.append(kwargs)
        return ("bad", "result")

    monkeypatch.setattr("app.graph.nodes.data_query.query_business_data", invalid_gateway)
    state = data_query({"user_message": "查询 2018-08-10 的 GMV", "trace_id": "tuple-contract"})
    assert state["tool_result"]["result"]["status"] == "unsupported"
    assert "tuple" in state["tool_result"]["result"]["reason"]
    assert len(calls) == 1
    expected_request_id = calls[0]["request_id"]
    assert state["tool_result"]["request_id"] == expected_request_id
    assert state["query_results"] == [state["tool_result"]["batch_results"][0]]
    item = state["query_results"][0]
    assert item["request_id"] == expected_request_id
    assert item["idempotency_key"] == expected_request_id
    assert item["result"]["status"] == "unsupported"
    assert item["result"]["reason_code"] == "invalid_tool_result_shape"
    assert item["evidence_id"] == state["evidence_bundles"][0]["evidence_id"]


def test_compiled_generic_continuation_reuses_only_the_same_scope_authorized_metric() -> None:
    from app.graph.nodes.data_query import _followup_authorized_metrics
    from app.metrics.authorization import AuthorizedMetric

    metric_id = "seller_avg_late_delivery_days"
    authorized = AuthorizedMetric(
        metric_id=metric_id,
        provenance_type="exact",
        provenance_id="prior-exact-mention",
        query_scope_fingerprint="scope-fingerprint-a",
        scope_fingerprint="scope-fingerprint-a",
        catalog_digest="catalog-v1",
    )
    state = {
        "current_query_scope_fingerprint": "scope-fingerprint-a",
        "controlled_analysis_scope": {"allowed_metrics": [metric_id]},
        "previous_turn_context": {
            "source_turn_id": "prior-turn",
            "scope_fingerprint": "scope-fingerprint-a",
            "authorized_metrics": [authorized.model_dump(mode="json")],
        },
        "context_compilation": {
            "relation": "continue",
            "needs_clarification": False,
            "source_turn_id": "prior-turn",
            "known_slots": {
                "metric": {"value": metric_id, "source": "previous_turn_context"},
            },
        },
    }

    reused = _followup_authorized_metrics(state, "店铺平均延迟送达天数再按月看看趋势")
    assert [item.metric_id for item in reused] == [metric_id]


def test_compiled_explicit_metric_switch_does_not_reuse_previous_authorization() -> None:
    from app.graph.nodes.data_query import _followup_authorized_metrics
    from app.metrics.authorization import AuthorizedMetric

    previous_metric = "seller_avg_late_delivery_days"
    authorized = AuthorizedMetric(
        metric_id=previous_metric,
        provenance_type="exact",
        provenance_id="prior-exact-mention",
        query_scope_fingerprint="scope-fingerprint-a",
        scope_fingerprint="scope-fingerprint-a",
        catalog_digest="catalog-v1",
    )
    state = {
        "current_query_scope_fingerprint": "scope-fingerprint-a",
        "controlled_analysis_scope": {"allowed_metrics": [previous_metric, "gmv"]},
        "previous_turn_context": {
            "source_turn_id": "prior-turn",
            "scope_fingerprint": "scope-fingerprint-a",
            "authorized_metrics": [authorized.model_dump(mode="json")],
        },
        "context_compilation": {
            "relation": "independent",
            "needs_clarification": False,
            "source_turn_id": "prior-turn",
            "known_slots": {
                "metric": {"value": "gmv", "source": "current_turn"},
            },
        },
    }

    assert not _followup_authorized_metrics(state, "这个 GMV 呢？")


def test_compiled_followup_with_changed_scope_cannot_reuse_previous_authorization() -> None:
    from app.graph.nodes.data_query import _followup_authorized_metrics
    from app.metrics.authorization import AuthorizedMetric

    metric_id = "seller_avg_late_delivery_days"
    authorized = AuthorizedMetric(
        metric_id=metric_id,
        provenance_type="exact",
        provenance_id="prior-exact-mention",
        query_scope_fingerprint="scope-fingerprint-a",
        scope_fingerprint="scope-fingerprint-a",
        catalog_digest="catalog-v1",
    )
    state = {
        "current_query_scope_fingerprint": "scope-fingerprint-b",
        "controlled_analysis_scope": {"allowed_metrics": [metric_id]},
        "previous_turn_context": {
            "source_turn_id": "prior-turn",
            "scope_fingerprint": "scope-fingerprint-a",
            "authorized_metrics": [authorized.model_dump(mode="json")],
        },
        "context_compilation": {
            "relation": "continue",
            "needs_clarification": False,
            "source_turn_id": "prior-turn",
            "known_slots": {
                "metric": {"value": metric_id, "source": "previous_turn_context"},
            },
        },
    }

    assert not _followup_authorized_metrics(state, "再按月看看")


def test_month_position_flows_through_to_query_planner(monkeypatch) -> None:
    """月初/月中/月末 queries are no longer hard-blocked; they reach the gateway."""
    called_with: list = []

    def fake_query(**kwargs):
        called_with.append(kwargs)
        return {"server": "local-mcp", "tool_name": "breakdown_by_dimension", "arguments": kwargs, "request_id": "test", "result": {"status": "ok", "parsed": {}}}

    monkeypatch.setattr("app.graph.nodes.data_query.query_business_data", fake_query)
    state = data_query({"user_message": "统计本月订单量在月初/月中/月末的分布", "trace_id": "month-position"})
    assert len(called_with) >= 1, "month_position query should reach the gateway, not be blocked"
    assert state["tool_result"]["tool_name"] == "breakdown_by_dimension"


def test_template_request_without_confirmed_plan_does_not_produce_execution_result(monkeypatch) -> None:
    from app.graph.nodes import data_query as data_query_module
    from app.mcp.schemas import MetricPlanningDecision, MetricQueryBatchPlan

    calls = []
    decision = MetricPlanningDecision(
        status="clarification_required",
        reason="模板分析需要先创建并确认计划。",
        reason_code="semantic_confirmation_required",
    )
    batch = MetricQueryBatchPlan(decision=decision, plans=[])
    monkeypatch.setattr(data_query_module, "plan_confirmed_metric_queries", lambda *_args, **_kwargs: batch)
    monkeypatch.setattr(
        data_query_module,
        "query_business_data",
        lambda **kwargs: calls.append(("platform", kwargs)) or {"result": {"status": "ok"}},
    )
    monkeypatch.setattr(
        data_query_module,
        "query_scoped_business_data",
        lambda **kwargs: calls.append(("scoped", kwargs)) or {"result": {"status": "ok"}},
    )

    state = data_query({
        "user_message": "Please run RFM segmentation",
        "trace_id": "template-needs-confirmation",
        "task_type": "metric_query",
    })

    assert state["metric_planning_decision"]["status"] == "clarification_required"
    assert state["tool_result"]["result"]["status"] == "clarification_required"
    assert "template_only" not in state["tool_result"]["result"]
    assert state["query_results"] == []
    assert calls == []
    assert all(item.get("tool_name") != "whitelist_analysis_template" for item in state.get("tool_calls", []))


def test_inherited_metric_period_followup_uses_contract_compare_plan(monkeypatch) -> None:
    from app.agents.data_query_agent import plan_confirmed_metric_queries
    from app.context.compiler import compile_turn_context
    from app.metrics.authorization import AuthorizedMetric

    question = "和 2018 年 7 月相比呢？"
    prior = {
        "source_turn_id": "prior-turn",
        "scope_fingerprint": "scope-a",
        "context_slots": {
            "metric": {"value": "order_count", "source": "current_turn"},
            "period": {
                "value": {"start_date": "2018-08-01", "end_date": "2018-08-31"},
                "source": "current_turn",
            },
        },
    }
    compilation = compile_turn_context({
        "user_message": question,
        "current_query_scope_fingerprint": "scope-a",
        "previous_turn_context": prior,
    })["context_compilation"]
    monkeypatch.setattr(
        "app.llm.client.llm_extract",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("inherited metric must not call LLM")),
    )
    authorized = AuthorizedMetric(
        metric_id="order_count",
        provenance_type="exact",
        provenance_id="prior-exact",
        query_scope_fingerprint="scope-a",
        scope_fingerprint="scope-a",
        catalog_digest="catalog-v1",
    )

    planned = plan_confirmed_metric_queries(
        question,
        operation_intent=compilation["operation_intent"],
        request_contract=compilation["request_contract"],
        comparison_periods=compilation["operation_intent"]["comparison_periods"],
        followup_authorized_metrics=[authorized],
    )

    assert planned.decision.status == "confirmed"
    assert planned.decision.query_type == "metric_compare"
    plan = planned.plans[0]
    assert plan.metric_name == "order_count"
    assert plan.query_type == "metric_compare"
    assert (plan.baseline_start, plan.baseline_end) == ("2018-07-01", "2018-07-31")
    assert (plan.current_start, plan.current_end) == ("2018-08-01", "2018-08-31")
    assert plan.granularity == "month"
    assert plan.tool_name == "compare_metric_period"


def test_inherited_gmv_period_contribution_plans_registered_period_query(monkeypatch) -> None:
    from app.agents.data_query_agent import plan_confirmed_metric_queries
    from app.context.compiler import compile_turn_context
    from app.metrics.authorization import AuthorizedMetric

    question = "再按商品品类解释这个变化的统计贡献，只说查询支持的内容。"
    previous = {
        "source_turn_id": "turn-prior-comparison",
        "scope_fingerprint": "scope-a",
        "result_context": {
            "version": "result-context-v1",
            "source_turn_id": "turn-prior-comparison",
            "source_trace_id": "trace-prior-comparison",
            "scope_fingerprint": "scope-a",
            "data_version_manifest_hash": "manifest-a",
            "metric_id": "gmv",
            "query_type": "metric_compare",
            "result_status": "completed",
            "comparison": {
                "metric_name": "gmv", "baseline_value": 100.0,
                "current_value": 120.0, "delta_value": 20.0,
                "baseline_period": "2018-07-01 至 2018-07-31",
                "current_period": "2018-08-01 至 2018-08-31",
                "evidence_ids": ["ev-prior-comparison"],
            },
            "query_period": {
                "baseline_start": "2018-07-01", "baseline_end": "2018-07-31",
                "current_start": "2018-08-01", "current_end": "2018-08-31",
            },
            "evidence_ids": ["ev-prior-comparison"],
        },
    }
    compilation = compile_turn_context({
        "user_message": question,
        "previous_turn_context": previous,
        "chat_session_id": "session-a",
        "current_query_scope_fingerprint": "scope-a",
        "data_version_manifest_hash": "manifest-a",
    })["context_compilation"]
    assert compilation["request_contract"]["parse_status"] == "resolved"
    assert compilation["request_contract"]["required_tool"] == "compare_breakdown_by_dimension_period"
    assert compilation["operation_intent"]["comparison_periods"] == {
        "baseline": {"start_date": "2018-07-01", "end_date": "2018-07-31"},
        "current": {"start_date": "2018-08-01", "end_date": "2018-08-31"},
        "granularity": "month",
    }

    monkeypatch.setattr(
        "app.llm.client.llm_extract",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("trusted follow-up must not call LLM")),
    )
    authorized = AuthorizedMetric(
        metric_id="gmv",
        provenance_type="exact",
        provenance_id="prior-gmv-mention",
        query_scope_fingerprint="scope-a",
        scope_fingerprint="scope-a",
        catalog_digest="catalog-v1",
    )

    planned = plan_confirmed_metric_queries(
        question,
        operation_intent=compilation["operation_intent"],
        request_contract=compilation["request_contract"],
        comparison_periods=compilation["operation_intent"]["comparison_periods"],
        followup_authorized_metrics=[authorized],
    )

    assert planned.decision.status == "confirmed"
    assert planned.decision.operation_intent == "dimensional_statistical_contribution"
    assert planned.decision.query_type == "metric_contribution"
    assert planned.decision.dimension == "product_category"
    assert planned.decision.reason_code == "exact_match"
    assert planned.decision.confirmed_metrics == ["gmv"]
    assert len(planned.plans) == 1
    plan = planned.plans[0]
    assert plan.metric_name == "gmv"
    assert plan.query_type == "metric_contribution"
    assert plan.dimension == "product_category"
    assert (plan.baseline_start, plan.baseline_end) == ("2018-07-01", "2018-07-31")
    assert (plan.current_start, plan.current_end) == ("2018-08-01", "2018-08-31")
    assert plan.granularity == "month"
    assert plan.tool_name == "compare_breakdown_by_dimension_period"


def test_runtime_accepts_only_correlated_preexecution_contract_refusal() -> None:
    import pytest
    from app.graph.runtime import _validate_boundary

    state = {
        "context_compilation": {"request_contract": {"contract_hash": "contract-a"}},
        "request_contract_failures": [{
            "layer": "planner", "reason": "contract_violation_planner_operation",
            "contract_hash": "contract-a",
        }],
        "tool_calls": [],
        "query_results": [],
        "tool_result": {
            "server": "request-contract-guard", "tool_name": None,
            "request_id": "trace-a:1",
            "result": {
                "status": "unsupported",
                "reason_code": "contract_violation_planner_operation",
                "parsed": {},
            },
        },
    }

    _validate_boundary("data_query", state)

    state["request_contract_failures"] = []
    with pytest.raises(ValueError, match="no correlated tool invocation"):
        _validate_boundary("data_query", state)


def test_contract_compare_plan_mismatch_fails_closed() -> None:
    from app.agents.contracts import MetricQueryPlan
    from app.agents.data_query_agent import _contract_plan_mismatch

    contract = {
        "operation_intent": "ordinary_metric_query",
        "comparison": {"kind": "period_pair"},
    }

    assert _contract_plan_mismatch(
        contract,
        [MetricQueryPlan(query_type="metric_value", metric_name="order_count")],
    ) == "request_contract_comparison_plan_mismatch"
    assert _contract_plan_mismatch(
        contract,
        [MetricQueryPlan(query_type="metric_compare", metric_name="order_count")],
    ) is None


def test_day_pair_does_not_override_an_explicit_trend_plan() -> None:
    from app.agents.data_query_agent import _contract_query_type

    contract = {
        "operation_intent": "ordinary_metric_query",
        "comparison": {"kind": "day_pair"},
    }

    assert _contract_query_type(contract, "metric_trend") == "metric_trend"
    assert _contract_query_type(contract, "metric_compare") == "metric_compare"


def test_business_causal_contract_uses_trend_before_requested_dimension_contribution(monkeypatch) -> None:
    from app.agents.data_query_agent import plan_confirmed_metric_queries
    from app.context.compiler import compile_turn_context

    user_question = (
        "2018-08-04 的 GMV 为什么低于 2018-08-01 至 2018-08-03？"
        "请按商品品类分析可验证的贡献因素。"
    )
    compilation = compile_turn_context({"user_message": user_question})["context_compilation"]
    assert compilation["request_contract"]["operation_intent"] == "business_causal_investigation"
    assert compilation["request_contract"]["dimension"] == "product_category"
    monkeypatch.setattr(
        "app.llm.client.llm_extract",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("exact metric must not call LLM")),
    )

    planned = plan_confirmed_metric_queries(
        "查询 2018-08-01 到 2018-08-04 的 GMV 趋势",
        operation_intent=compilation["operation_intent"],
        request_contract=compilation["request_contract"],
    )

    assert planned.decision.status == "confirmed"
    assert planned.decision.operation_intent == "business_causal_investigation"
    assert planned.decision.query_type == "metric_trend"
    assert planned.decision.dimension == "product_category"
    assert len(planned.plans) == 1
    assert planned.plans[0].query_type == "metric_trend"


def test_business_causal_period_pair_is_mirrored_and_passes_compiled_contract() -> None:
    from app.context.compiler import compile_turn_context
    from app.context.request_contract import validate_compiled_request_contract

    compilation = compile_turn_context({
        "user_message": (
            "2018 年 8 月 GMV 为什么比 2018 年 7 月下降？"
            "只按 product_category 维度做根因诊断，并区分统计贡献与因果。"
        ),
    })["context_compilation"]

    comparison = compilation["request_contract"]["comparison"]
    assert comparison["kind"] == "period_pair"
    assert compilation["operation_intent"]["kind"] == "business_causal_investigation"
    assert compilation["operation_intent"]["comparison_periods"] == {
        key: value for key, value in comparison.items() if key != "kind"
    }
    assert validate_compiled_request_contract(compilation) == (True, None)


def test_registered_decomposition_period_pair_is_mirrored_and_validated() -> None:
    from app.context.compiler import compile_turn_context
    from app.context.request_contract import validate_compiled_request_contract

    compilation = compile_turn_context({
        "user_message": "请用注册数值分解方法解释 2018 年 8 月 GMV 相比 2018 年 7 月的变化。",
    })["context_compilation"]

    comparison = compilation["request_contract"]["comparison"]
    assert comparison["kind"] == "period_pair"
    assert compilation["operation_intent"]["kind"] == "registered_numeric_decomposition"
    assert compilation["operation_intent"]["comparison_periods"] == {
        key: value for key, value in comparison.items() if key != "kind"
    }
    assert validate_compiled_request_contract(compilation) == (True, None)


def test_semantic_hint_inside_exact_metric_alias_does_not_force_planner_llm() -> None:
    from app.agents.data_query_agent import _has_unresolved_semantic_hint
    from app.metrics.registry import get_registry

    catalog = get_registry().planning_catalog(scope="platform", surface="chat")

    # "订单" is configured as a semantic term, but here it is fully covered by
    # the exact registered metric alias "订单量" and is not a second intent.
    assert _has_unresolved_semantic_hint("订单量", catalog) is False


def test_ordinary_contract_still_uses_registered_dimension_breakdown() -> None:
    from app.agents.data_query_agent import _contract_query_type

    contract = {
        "operation_intent": "ordinary_metric_query",
        "dimension": "product_category",
        "comparison": {"kind": "unresolved"},
    }

    assert _contract_query_type(contract, "metric_breakdown") == "metric_breakdown"


def test_contract_plan_mismatch_is_scoped_to_period_pairs() -> None:
    from app.agents.contracts import MetricQueryPlan
    from app.agents.data_query_agent import _contract_plan_mismatch

    day_pair = {
        "operation_intent": "ordinary_metric_query",
        "comparison": {"kind": "day_pair"},
    }
    assert _contract_plan_mismatch(
        day_pair,
        [MetricQueryPlan(query_type="metric_trend", metric_name="gmv")],
    ) is None

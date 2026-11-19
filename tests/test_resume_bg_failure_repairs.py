"""Offline regressions for the four interrupted development B/G run outcomes."""
import copy
import json
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/testing/resume_bg_ab_dev_20261016"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("real network forbidden")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    from app.resilience.policy import _BREAKERS
    _BREAKERS.clear()


def records():
    return [json.loads(p.read_text(encoding="utf-8")) for p in (DATA / "results_v1/traces").glob("*.run_record.json")]


def case(case_id):
    return next(c for c in json.loads((DATA / "cases_6.json").read_text(encoding="utf-8"))["cases"] if c["case_id"] == case_id)


def test_dimensional_statistical_contribution_intent_routes_to_diagnostic_without_classifier(monkeypatch):
    from app.context.compiler import compile_turn_context
    from app.graph.nodes import triage as triage_node

    state = {"user_message": (
        "请比较平台运费率在2017年3月与2017年4月的变化，按 month_position "
        "说明各位置的统计贡献；只陈述数值，不推断业务因果。"
    )}
    compile_turn_context(state)
    monkeypatch.setattr(triage_node, "llm_classify", lambda *_a, **_k: pytest.fail("structured contribution intent must route deterministically"))

    triage_node.triage(state)

    assert state["task_type"] == "anomaly_analysis"
    assert state["next_agent"] == "data_query"


def test_business_causal_intent_routes_to_diagnostic_without_classifier(monkeypatch):
    from app.context.compiler import compile_turn_context
    from app.context.request_contract import validate_compiled_request_contract
    from app.graph.nodes import triage as triage_node
    from app.graph.workflow import _route_after_data_query

    state = {"user_message": (
        "2018 年 8 月 GMV 为什么比 2018 年 7 月下降？"
        "只按 product_category 维度做根因诊断，并区分统计贡献与因果。"
    )}
    compile_turn_context(state)
    assert validate_compiled_request_contract(state["context_compilation"]) == (True, None)
    monkeypatch.setattr(
        triage_node,
        "llm_classify",
        lambda *_a, **_k: pytest.fail("business causal intent must route deterministically"),
    )

    triage_node.triage(state)

    assert state["task_type"] == "anomaly_analysis"
    assert state["next_agent"] == "data_query"
    assert _route_after_data_query(state) == "premise_check"


def test_diagnostic_candidate_branches_respect_registered_and_authorized_dimensions():
    from app.graph.nodes.diagnostic import _branch_is_trusted_for_action_metric

    def branch(dimension, *, metric_id="freight_value", parent_evidence_id=None, depth=0):
        return SimpleNamespace(
            metric_id=metric_id,
            dimension=dimension,
            parent_evidence_id=parent_evidence_id,
            depth=depth,
        )

    # An unscoped root is trusted, but a named root cannot bypass the same
    # registered/authorized dimension allow-list used for child branches.
    assert _branch_is_trusted_for_action_metric(branch(None), "freight_value", {"product_category"})
    assert _branch_is_trusted_for_action_metric(
        branch("product_category"), "freight_value", {"product_category"}
    )
    assert not _branch_is_trusted_for_action_metric(
        branch("customer_state"), "freight_value", {"product_category"}
    )
    assert not _branch_is_trusted_for_action_metric(
        branch("product_category", metric_id="order_count"), "freight_value", {"product_category"}
    )


def test_default_diagnostic_dimension_uses_current_identity_grants():
    from app.graph.nodes.diagnostic import _first_dimension
    from app.tenancy.context import TenantContext, activate_tenant_context

    tenant = TenantContext(
        tenant_id="platform", user_id="dimension-default-test", role="admin", scope_type="platform",
    )
    with activate_tenant_context(tenant):
        assert _first_dimension("freight_value", "platform") == "product_category"
        # A scope cannot add a dimension that the identity policy does not grant.
        assert _first_dimension(
            "freight_value", "platform",
            state={"controlled_analysis_scope": {"allowed_dimensions": ["customer_state"]}},
        ) is None


def test_statistical_contribution_query_requests_complete_registered_group_limit():
    from app.agents.data_query_agent import build_metric_query_plan

    plan = build_metric_query_plan(
        "请比较平台 freight_value 2017-11-20 与 2017-11-21 按品类拆解统计贡献"
    )

    assert plan.query_type == "metric_contribution"
    assert plan.tool_name == "compare_breakdown_by_dimension"
    assert plan.limit == 100


def test_dc05_registry_planner_uses_full_group_limit_for_exact_dimension_request():
    from app.agents.data_query_agent import plan_confirmed_metric_queries
    from app.context.compiler import compile_turn_context

    question = (
        "请解释平台运费总额从 2017-11-20 到 2017-11-21 的变化，按已注册的商品品类维度 "
        "展示可核验的统计贡献，并说明这不等于业务因果。"
    )
    state = {"user_message": question}
    compile_turn_context(state)
    intent = state["context_compilation"]["operation_intent"]
    planned = plan_confirmed_metric_queries(
        question,
        operation_intent=intent,
        comparison_dates={"baseline_date": "2017-11-20", "current_date": "2017-11-21"},
        scope="platform",
        scope_data={
            "tenant_id": "platform", "user_id": "resume-bg-experiment",
            "permissions": "admin", "scope_type": "platform",
        },
    )

    assert intent["kind"] == "dimensional_statistical_contribution"
    assert planned.decision.query_type == "metric_contribution"
    assert planned.plans[0].metric_name == "freight_value"
    assert planned.plans[0].dimension == "product_category"
    assert planned.plans[0].baseline_date == "2017-11-20"
    assert planned.plans[0].current_date == "2017-11-21"
    assert planned.plans[0].limit == 100


def test_month_position_contribution_intent_reaches_registered_period_tool(monkeypatch):
    from app.agents.data_query_agent import plan_confirmed_metric_queries
    from app.context.compiler import compile_turn_context
    from app.mcp import client as mcp_client
    from app.tenancy import TenantContext, activate_tenant_context

    question = (
        "请比较平台运费率在2017年3月与2017年4月的变化，按 month_position "
        "说明各位置的统计贡献；只陈述数值，不推断业务因果。"
    )
    state = {"user_message": question}
    compile_turn_context(state)
    operation_intent = state["context_compilation"]["operation_intent"]
    plan_batch = plan_confirmed_metric_queries(
        question,
        operation_intent=operation_intent,
        comparison_periods=operation_intent["comparison_periods"],
        scope="platform",
        scope_data={
            "tenant_id": "platform", "user_id": "resume-bg-experiment",
            "permissions": "admin", "scope_type": "platform",
        },
    )

    assert plan_batch.decision.status == "confirmed"
    assert plan_batch.decision.query_type == "metric_contribution"
    assert plan_batch.decision.operation_intent == "dimensional_statistical_contribution"
    plan = plan_batch.plans[0]
    assert plan.metric_name == "freight_rate"
    assert plan.dimension == "month_position"
    assert (plan.baseline_start, plan.baseline_end) == ("2017-03-01", "2017-03-31")
    assert (plan.current_start, plan.current_end) == ("2017-04-01", "2017-04-30")
    assert plan.granularity == "month" and plan.limit == 100

    captured = {}

    def local_tool(tool_name, arguments, **kwargs):
        captured.update(tool_name=tool_name, arguments=arguments, kwargs=kwargs)
        return {"status": "ok", "items": [], "contributions": [], "parsed": {}}

    monkeypatch.setattr(mcp_client, "call_local_mcp_tool", local_tool)
    context = TenantContext(
        tenant_id="platform", user_id="resume-bg-experiment", role="admin", scope_type="platform",
    )
    with activate_tenant_context(context):
        result = mcp_client.query_business_data(
            task_type="metric_query", question=question, plan=plan, request_id="dc07-period-route",
        )

    assert result.get("tool_name") is not None, result.get("result", {}).get("reason")
    assert captured["tool_name"] == "compare_freight_rate_ratio_facts_by_dimension_period"
    assert captured["arguments"] == {
        "metric_name": "freight_rate",
        "current_start": "2017-04-01", "current_end": "2017-04-30",
        "baseline_start": "2017-03-01", "baseline_end": "2017-03-31",
        "granularity": "month", "dimension": "month_position",
    }
    assert result["request_id"] == "dc07-period-route"


def test_li03_capability_refusal_completes_through_data_query_answer_and_critic(monkeypatch):
    from app.context.compiler import compile_turn_context
    from app.graph.nodes import data_query as data_query_node
    from app.graph.nodes.answer import generate_answer
    from app.graph.nodes.critic import critic
    from app.llm import client as llm_client
    from app.tenancy import TenantContext, activate_tenant_context

    for name in ("llm_chat", "llm_extract", "llm_classify"):
        monkeypatch.setattr(llm_client, name, lambda *_a, **_k: pytest.fail("capability refusal must be deterministic"))

    state = {
        "trace_id": "li03-current-path",
        "user_message": "平台订单量拆分不支持 payment_type；请说明限制。",
        "tenant_id": "platform", "user_id": "resume-bg-experiment", "user_role": "admin",
        "task_type": "metric_query", "tool_calls": [], "errors": [], "evidence_bundles": [],
    }
    compile_turn_context(state)
    context = TenantContext(
        tenant_id="platform", user_id="resume-bg-experiment", role="admin", scope_type="platform",
    )
    with activate_tenant_context(context):
        data_query_node.data_query(state)
        generate_answer(state)
        critic(state)

    assert state["terminal_code"] == "CAPABILITY_GAP"
    assert state["decision_records"][-1]["sql_executed"] is False
    assert state["decision_records"][-1]["dimension"] == "payment_type"
    assert state["tool_calls"] == []
    control = state["control_events"][-1]
    assert control["event_type"] == "capability_preflight_stop"
    assert control["sql_executed"] is False
    assert state["critic_verdict"] == "approved"
    assert not state["errors"]


@pytest.mark.parametrize("arm", ["basic_tool_loop", "governed_workflow"])
def test_nc01_real_outputs_normalize_without_changing_numbers_or_auto_approving_text(arm):
    from evaluation.resume_bg_protocol import score_bg_run
    record = next(r for r in records() if r["case_id"] == "NC01" and r["arm"] == arm)
    original = copy.deepcopy(record)
    score = score_bg_run(case("NC01"), arm=arm, trace=record["trace"], lifecycle_status="completed")
    assert all(score["numeric_checks"].values())
    assert score["scope_matches"] and score["period_matches"] and score["evidence_valid"]
    assert score["manual_review_required"] and not score["task_acceptance_pass"]
    assert score["task_acceptance_status"] == "pending_manual_review"
    assert record == original
    # A genuine explicit human result can close the separate text gate; Critic
    # passed alone must never serve as that result.
    reviewed = score_bg_run(case("NC01"), arm=arm, trace=record["trace"], lifecycle_status="completed", human_review={"decision": "pass"})
    assert reviewed["task_acceptance_pass"]
    state = record["trace"]["state_snapshot"]
    if arm == "governed_workflow":
        state["analysis_presentation"]["metric_decomposition"]["delta_value"] = 999
    else:
        state["final"]["presentation"]["delta"] = 999
        record["trace"]["workflow_result"]["presentation"]["delta"] = 999
    bad = score_bg_run(case("NC01"), arm=arm, trace=record["trace"], lifecycle_status="completed", human_review={"decision": "pass"})
    assert not bad["numeric_checks"]["delta"] and not bad["task_acceptance_pass"]


def test_nested_b_source_authorization_cannot_be_forged():
    from evaluation.resume_bg_protocol import score_bg_run
    r = next(r for r in records() if r["case_id"] == "NC01" and r["arm"] == "basic_tool_loop")
    r["trace"]["state_snapshot"]["evidence_bundles"][0]["content"]["source_queries"][0]["envelope"]["result"]["parsed"]["authorized_metric"] = None
    score = score_bg_run(case("NC01"), arm=r["arm"], trace=r["trace"], lifecycle_status="completed", human_review={"decision": "pass"})
    assert not score["evidence_valid"] and not score["task_acceptance_pass"]


def test_li03_registry_refusal_becomes_structured_limitation_without_numeric_claim(monkeypatch):
    from app.graph.nodes.data_query import _record_planning_capability_limit
    from app.graph.nodes.answer import generate_answer
    from app.graph.nodes.critic import critic
    from app.mcp.schemas import MetricPlanningDecision
    from app.tenancy import TenantContext, activate_tenant_context
    from app.llm import client
    from evaluation.resume_bg_protocol import score_bg_run
    for name in ("llm_chat", "llm_extract", "llm_classify"):
        monkeypatch.setattr(client, name, lambda *a, **k: pytest.fail("no model is needed for this refusal"))
    r = next(r for r in records() if r["case_id"] == "LI03" and r["arm"] == "governed_workflow")
    state = r["trace"]["state_snapshot"]
    state["errors"] = []  # replay the first refusal, not its later failed revisions
    state["_revision_counts"] = {}
    decision = MetricPlanningDecision.model_validate(state["metric_planning_decision"])
    context = TenantContext(tenant_id="platform", user_id="resume-bg-experiment", role="admin", scope_type="platform")
    with activate_tenant_context(context):
        _record_planning_capability_limit(state, decision, request_id="li03-refusal", scope="platform")
        generate_answer(state)
        critic(state)
    assert state["terminal_code"] == "CAPABILITY_GAP"
    assert state["decision_records"][-1]["sql_executed"] is False
    assert state["claims"] == []
    assert not state["errors"] and state["critic_verdict"] == "approved"
    score = score_bg_run(case("LI03"), arm="governed_workflow", trace={"state_snapshot": state}, lifecycle_status="completed")
    assert score["correct_safe_stop"]
    assert not score["effective_diagnostic_completed"]
    assert not score["task_acceptance_pass"]  # external text review still pending


def test_timeout_terminal_event_is_joined_before_unknown_usage_snapshot(monkeypatch, tmp_path, capsys):
    import httpx
    from openai import APITimeoutError
    from test_provider_candidate_context_boundary import _install_offline_client, _state
    from app.core.execution_context import activate_state, reset_state
    from evaluation.provider_candidate_batch import token_usage_snapshot
    def delayed_sdk_timeout(_kwargs):
        time.sleep(0.025)  # later than the competing old 0.005s outer timer
        raise APITimeoutError(request=httpx.Request("POST", "https://offline.invalid"))
    llm, config, ledger, sends = _install_offline_client(monkeypatch, tmp_path, timeout=0.005, raw_create=delayed_sdk_timeout)
    state = _state("timeout-repair", config)
    token = activate_state(state)
    try:
        assert llm.llm_extract("system", "offline") is None
    finally:
        reset_state(token)
    assert sends == {"timeout-repair": 1}
    assert [r["event"] for r in ledger.events] == ["sdk_send_started", "sdk_send_failed"]
    usage = token_usage_snapshot({"state_snapshot": state}, ledger.events, run_id="timeout-repair")
    assert usage["metering_errors"] == []
    assert usage["unknown_or_failed_usage_requests"] == 1 and usage["total_tokens"] is None
    output = capsys.readouterr().out
    assert "trying backup" not in output and "no fallback available" in output


def test_late_sdk_event_cannot_reopen_failed_lifecycle_and_selection_is_read_only(tmp_path):
    from evaluation.provider_candidate_batch import RunLedger
    ledger = RunLedger(tmp_path / "events.jsonl")
    ledger.append("r", "started")
    ledger.append("r", "sdk_send_started", attempt_id="a")
    ledger.append("r", "failed")
    ledger.append("r", "sdk_send_failed", attempt_id="a")
    original = ledger.path.read_bytes()
    assert ledger.latest()["r"]["event"] == "failed"
    assert ledger.resume_candidates([{"run_id": "r"}, {"run_id": "new"}]) == [{"run_id": "new"}]
    assert ledger.reconcile_interrupted() == {"interrupted_before_send": 0, "interrupted_result_unknown": 0}
    assert ledger.path.read_bytes() == original


def test_executor_unknown_usage_stops_even_with_complete_send_terminal_pair(monkeypatch, tmp_path):
    import httpx
    from openai import APITimeoutError
    from test_provider_candidate_context_boundary import _install_offline_client
    from evaluation.resume_bg_executor import run_bg_unit
    from evaluation.provider_candidate_batch import CostPolicy, EstimatedCostLedger, ExperimentConfig
    from app.graph import workflow
    from app.tenancy import TenantContext

    def timeout(_kwargs):
        raise APITimeoutError(request=httpx.Request("POST", "https://offline.invalid"))
    _llm, _config, ledger, sends = _install_offline_client(monkeypatch, tmp_path, raw_create=timeout)
    monkeypatch.setattr(workflow, "get_settings", lambda: SimpleNamespace(llm_timeout_seconds=0.01))
    costs = EstimatedCostLedger(CostPolicy(cumulative_estimate_limit_cny=30), tmp_path / "costs.jsonl")
    entry = {"run_id": "unknown-unit", "case_id": "LI03", "arm": "basic_tool_loop"}
    with pytest.raises(RuntimeError):
        run_bg_unit(entry, case("LI03"), tenant_context=TenantContext(tenant_id="platform", user_id="resume-bg-experiment", role="admin", scope_type="platform"),
                    trace_dir=tmp_path / "traces", ledger=ledger, cost_ledger=costs, experiment_config=ExperimentConfig())
    r = json.loads((tmp_path / "traces/unknown-unit.run_record.json").read_text(encoding="utf-8"))
    assert r["result_lifecycle_status"] == "failed"
    assert r["measurements"]["token_usage"]["total_tokens"] is None
    assert r["measurements"]["token_usage"]["metering_errors"] == []
    assert ledger.latest()["unknown-unit"]["error_type"] == "provider_usage_unknown"
    assert sends == {"unknown-unit": 1}

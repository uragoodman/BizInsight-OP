"""LangGraph 工作流编排测试 — Target 完整协作版。

验证目标：
1. 四种 task_type 分别走各自的协作边路径。
2. Specialist 节点在正确的依赖就绪后才执行。
3. Critic 审核 final answer（非中间产物）。
4. Critic 定向返工：revision_target → 特定 specialist。
5. revision_counts 防无限循环。
6. /chat API 兼容性。
"""

from fastapi.testclient import TestClient

from app.graph.workflow import _route_after_data_query, _route_after_premise_check, build_chat_graph, run_chat_workflow
from app.main import app

client = TestClient(app)


def _patch_graph_entry(monkeypatch, task_type: str) -> None:
    """Keep graph routing tests off provider, memory-store, and database services."""
    from app.graph import workflow
    from app.graph.nodes import triage as triage_node

    monkeypatch.setattr(triage_node, "llm_classify", lambda *_args, **_kwargs: task_type)
    monkeypatch.setattr(workflow, "resolve_explorer_context", lambda state: state)
    monkeypatch.setattr(workflow, "load_session_memory", lambda state: state)
    monkeypatch.setattr(workflow, "retrieve_analysis_memories", lambda state: state)
    monkeypatch.setattr(workflow, "persist_memory_p0", lambda state: state)


def _patch_metric_definition_path(monkeypatch, *, keep_trace: bool = False) -> None:
    from app.graph import workflow

    _patch_graph_entry(monkeypatch, "metric_definition")

    def fake_metric_definition(state):
        state["metric_context"] = "测试指标口径"
        state["rag_citations"] = []
        state.setdefault("_revision_counts", {})
        state["_revision_counts"]["metric_definition"] = state["_revision_counts"].get("metric_definition", 0) + 1
        state.setdefault("node_history", []).append({"node": "metric_definition", "summary": "fixture"})
        return state

    def fake_answer(state):
        state["answer"] = "测试定义答案"
        state["evidence_bundles"] = [{"evidence_id": "fixture-evidence"}]
        state["claims"] = [{
            "claim_type": "definition",
            "value": "测试定义答案",
            "source_evidence_ids": ["fixture-evidence"],
        }]
        state.setdefault("node_history", []).append({"node": "generate_answer", "summary": "fixture"})
        return state

    def fake_critic(state, **_review_options):
        state["critic_verdict"] = "approved"
        state["critic_revision_target"] = None
        state.setdefault("critic_decisions", []).append({"llm_issues": [], "revision_target": None})
        state.setdefault("node_history", []).append({"node": "critic", "summary": "fixture approved"})
        return state

    monkeypatch.setattr(workflow, "metric_definition", fake_metric_definition)
    monkeypatch.setattr(workflow, "generate_answer", fake_answer)
    monkeypatch.setattr(workflow, "critic", fake_critic)
    if not keep_trace:
        monkeypatch.setattr(workflow, "persist_trace", lambda state: state)


# ============================================================
# Graph 构建
# ============================================================


def test_build_chat_graph_returns_compiled_graph() -> None:
    """验证 graph 能正常编译。"""
    graph = build_chat_graph()
    node_names = set(graph.get_graph().nodes)
    assert "root_cause" not in node_names
    assert "branch_evidence" not in node_names
    assert graph is not None
    # This is a structural test only. Invoking the graph exercises LLM, DB,
    # and persistence integrations; those belong in separately mocked tests.
    assert {"init_state", "triage", "data_query", "diagnostic_plan", "critic"} <= node_names


# ============================================================
# 四种路径测试
# ============================================================


def test_metric_definition_path(monkeypatch) -> None:
    """metric_definition：triage → metric_definition → generate_answer → critic"""
    from app.graph.workflow import build_chat_graph
    _patch_metric_definition_path(monkeypatch)

    graph = build_chat_graph()
    result = graph.invoke({"user_message": "GMV 是什么？", "trace_id": "t1"})
    assert result.get("task_type") == "metric_definition"

    history = [e["node"] for e in result.get("node_history", [])]
    assert "triage" in history
    assert "metric_definition" in history
    assert "generate_answer" in history
    assert "critic" in history
    # 不应经过不需要的节点
    assert "data_query" not in history
    assert "root_cause" not in history
    assert "report" not in history
    assert result.get("critic_verdict") in ("approved", "needs_revision")


def test_metric_query_path(monkeypatch) -> None:
    """metric_query：triage → data_query → generate_answer → critic"""
    from app.agents.contracts import ToolInvocation
    from app.graph import workflow

    _patch_graph_entry(monkeypatch, "metric_query")

    def fake_data_query(state):
        invocation = ToolInvocation(
            server="test", tool_name="query_gmv", arguments={}, status="ok",
            request_id="test-request", idempotency_key="test-request",
        )
        state.setdefault("tool_calls", []).append(invocation.model_dump(mode="json"))
        state["tool_result"] = {
            "server": "test", "tool_name": "query_gmv", "arguments": {},
            "request_id": "test-request", "result": {"status": "ok", "parsed": {}},
        }
        state["query_results"] = []
        state.setdefault("node_history", []).append({"node": "data_query", "summary": "fixture"})
        return state

    def fake_answer(state):
        state["answer"] = "测试指标答案"
        state["claims"] = []
        state.setdefault("node_history", []).append({"node": "generate_answer", "summary": "fixture"})
        return state

    def fake_critic(state, **_review_options):
        state["critic_verdict"] = "approved"
        state["critic_revision_target"] = None
        state.setdefault("critic_decisions", []).append({"llm_issues": [], "revision_target": None})
        state.setdefault("node_history", []).append({"node": "critic", "summary": "fixture approved"})
        return state

    monkeypatch.setattr(workflow, "data_query", fake_data_query)
    monkeypatch.setattr(workflow, "generate_answer", fake_answer)
    monkeypatch.setattr(workflow, "critic", fake_critic)
    monkeypatch.setattr(workflow, "persist_trace", lambda state: state)
    graph = build_chat_graph()
    result = graph.invoke({"user_message": "查询 2018-08-10 的 GMV", "trace_id": "t2"})
    assert result.get("task_type") == "metric_query"

    history = [e["node"] for e in result.get("node_history", [])]
    assert "triage" in history
    assert "data_query" in history, {
        "task_type": result.get("task_type"),
        "next_agent": result.get("next_agent"),
        "error_codes": [item.get("code") for item in result.get("errors", []) if isinstance(item, dict)],
        "triage_unresolved": result.get("triage_unresolved"),
        "triage_reason": result.get("triage_unresolved_reason"),
    }
    assert "generate_answer" in history
    assert "critic" in history
    # 不应经过不需要的节点
    assert "metric_definition" not in history
    assert "root_cause" not in history
    assert "report" not in history


def test_anomaly_metric_clarification_bypasses_root_cause() -> None:
    """Generic anomaly SOP questions must not enter Root Cause without a metric result."""
    state = {
        "task_type": "anomaly_analysis",
        "tool_result": {"result": {"status": "clarification_required"}},
        "errors": [],
    }

    assert _route_after_data_query(state) == "critic"


def test_legacy_root_cause_flag_cannot_bypass_governed_diagnostic(monkeypatch) -> None:
    monkeypatch.setenv("BIZINSIGHT_ROOT_CAUSE_MODE", "legacy_serial")
    assert _route_after_premise_check({"premise_check": {"status": "aligned"}}) == "diagnostic_plan"


def test_legacy_root_cause_revision_target_is_normalized_to_governed_path() -> None:
    from app.graph.workflow import _route_from_critic

    for legacy_target in ("root_cause", "branch_evidence"):
        assert _route_from_critic({
            "critic_verdict": "needs_revision",
            "critic_revision_target": legacy_target,
            "errors": [],
        }) == "diagnostic_plan"


def test_anomaly_analysis_path(monkeypatch) -> None:
    """异常分析即使计划生成失败，也留在受控诊断循环，不切换到 legacy 路径。"""
    from app.agents.contracts import ToolInvocation
    from app.graph import workflow

    _patch_graph_entry(monkeypatch, "anomaly_analysis")

    def fake_query(task_type: str, question: str, **_: object) -> dict:
        if "趋势" in question:
            payload = {
                "status": "ok", "parsed": {"query_type": "metric_trend"},
                "trend_result": {"metric_name": "gmv", "source_table": "analytics.daily_metrics", "points": [
                    {"bucket": "2018-08-09", "value": 120.0},
                    {"bucket": "2018-08-10", "value": 100.0},
                ]},
            }
            return {"tool_name": "query_metric_trend", "arguments": {}, "result": payload}
        if "拆解" in question:
            payload = {
                "status": "ok", "parsed": {"query_type": "metric_breakdown"},
                "breakdown_result": {"status": "ok", "dimension": "product_category", "source_table": "analytics.daily_metrics", "items": [
                    {"dimension_value": "books", "value": 60.0, "share": 60.0},
                ]},
            }
            return {"tool_name": "breakdown_by_dimension", "arguments": {}, "result": payload}
        payload = {
            "status": "ok", "parsed": {"query_type": "metric_value"},
            "metric_result": {"metric_name": "gmv", "metric_date": "2018-08-10", "value": 100.0, "source_table": "analytics.daily_metrics"},
        }
        return {"tool_name": "query_gmv", "arguments": {}, "result": payload}

    # 初始数据查询走桩；诊断循环应由服务端计划和 policy 决定是否继续调用工具。
    monkeypatch.setattr("app.graph.nodes.data_query.query_business_data", fake_query)
    monkeypatch.setattr("app.graph.nodes.diagnostic.query_business_data", fake_query)

    def fake_graph_data_query(state):
        invocation = ToolInvocation(
            server="test", tool_name="query_metric_trend", arguments={}, status="ok",
            request_id="test-trend", idempotency_key="test-trend",
        )
        payload = {
            "status": "ok",
            "parsed": {"query_type": "metric_trend"},
            "trend_result": {"metric_name": "gmv", "source_table": "analytics.daily_metrics", "points": [
                {"bucket": "2018-08-09", "value": 120.0},
                {"bucket": "2018-08-10", "value": 100.0},
            ]},
        }
        state.setdefault("tool_calls", []).append(invocation.model_dump(mode="json"))
        state["tool_result"] = {
            "server": "test", "tool_name": "query_metric_trend", "arguments": {},
            "request_id": "test-trend", "evidence_id": "fixture-trend", "result": payload,
        }
        state["query_results"] = [state["tool_result"]]
        state.setdefault("_revision_counts", {})
        state["_revision_counts"]["data_query"] = state["_revision_counts"].get("data_query", 0) + 1
        state.setdefault("node_history", []).append({"node": "data_query", "summary": "fixture trend"})
        return state

    monkeypatch.setattr(workflow, "data_query", fake_graph_data_query)
    # 显式让 triage 选择此测试的异常分析任务；诊断规划器随后走确定性
    # workflow_fallback，不依赖真实 provider，也不覆盖分类失败的 fail-closed 路径。
    monkeypatch.setattr("app.llm.client.llm_agent_action", lambda *_a, **_k: None)
    monkeypatch.setattr("app.llm.client.llm_classify", lambda *_a, **_k: "")
    monkeypatch.setattr("app.llm.client.llm_extract", lambda *_a, **_k: None)
    monkeypatch.setattr("app.llm.client.llm_chat", lambda *_a, **_k: "")
    monkeypatch.setattr("app.agents.critic_agent.llm_extract", lambda *_a, **_k: None)
    monkeypatch.setattr(workflow, "persist_trace", lambda state: state)
    graph = build_chat_graph()
    result = graph.invoke({"user_message": "为什么 2018-08-10 的 GMV 下降了？", "trace_id": "t3"})
    assert result.get("task_type") == "anomaly_analysis"

    history = [e["node"] for e in result.get("node_history", [])]
    assert "triage" in history
    assert "data_query" in history, {
        "task_type": result.get("task_type"),
        "next_agent": result.get("next_agent"),
        "error_codes": [item.get("code") for item in result.get("errors", []) if isinstance(item, dict)],
        "triage_unresolved": result.get("triage_unresolved"),
        "triage_reason": result.get("triage_unresolved_reason"),
    }
    assert "diagnostic_plan" in history
    assert "diagnostic_decide" in history
    assert "branch_evidence" not in history
    assert "root_cause" not in history
    assert "generate_answer" in history
    # If the standalone graph has no authorized scope, the governed policy
    # safely finalizes with an error instead of invoking tools or Critic.
    if not result.get("errors"):
        assert "critic" in history
    assert result.get("diagnostic_fallback") is True


def test_report_generation_path(monkeypatch) -> None:
    """report_generation：triage → metric_definition → data_query → report → generate_answer → critic"""
    from app.agents.contracts import ToolInvocation

    # This test verifies compiled-graph routing, not the PostgreSQL, RAG, report
    # generation, answer-generation, or LLM integrations. Keep those boundaries
    # deterministic so a developer can run the routing regression without the
    # Docker services or provider credentials.
    def fake_triage(state):
        state["task_type"] = "report_generation"
        state["next_agent"] = "metric_definition"
        state.setdefault("handoffs", []).append({"from": "triage", "to": "metric_definition", "reason": "test route"})
        state.setdefault("node_history", []).append({"node": "triage", "summary": "report route fixture"})
        return state

    def fake_metric_definition(state):
        state["metric_context"] = "测试指标口径"
        state["rag_citations"] = []
        state.setdefault("node_history", []).append({"node": "metric_definition", "summary": "fixture"})
        return state

    def fake_data_query(state):
        invocation = ToolInvocation(
            server="test", tool_name="query_gmv", arguments={}, status="ok",
            request_id="test-request", idempotency_key="test-request",
        )
        state.setdefault("tool_calls", []).append(invocation.model_dump(mode="json"))
        state["tool_result"] = {
            "server": "test", "tool_name": "query_gmv", "arguments": {},
            "request_id": "test-request", "result": {"status": "ok", "parsed": {}},
        }
        state["query_results"] = []
        state.setdefault("node_history", []).append({"node": "data_query", "summary": "fixture"})
        return state

    def fake_report(state):
        state["analysis_result"] = {"summary": "测试报告", "task_type": "report_generation"}
        state.setdefault("node_history", []).append({"node": "report", "summary": "fixture"})
        return state

    def fake_answer(state):
        state["answer"] = "测试报告已生成"
        state["claims"] = []
        state.setdefault("node_history", []).append({"node": "generate_answer", "summary": "fixture"})
        return state

    def fake_critic(state, **_review_options):
        state["critic_verdict"] = "approved"
        state["critic_revision_target"] = None
        state.setdefault("critic_decisions", []).append({"llm_issues": [], "revision_target": None})
        state.setdefault("node_history", []).append({"node": "critic", "summary": "fixture approved"})
        return state

    monkeypatch.setattr("app.graph.workflow.triage", fake_triage)
    monkeypatch.setattr("app.graph.workflow.metric_definition", fake_metric_definition)
    monkeypatch.setattr("app.graph.workflow.data_query", fake_data_query)
    monkeypatch.setattr("app.graph.workflow.report", fake_report)
    monkeypatch.setattr("app.graph.workflow.generate_answer", fake_answer)
    monkeypatch.setattr("app.graph.workflow.critic", fake_critic)

    graph = build_chat_graph()
    result = graph.invoke({"user_message": "生成本周经营分析报告", "trace_id": "t4"})
    assert result.get("task_type") == "report_generation"

    history = [e["node"] for e in result.get("node_history", [])]
    assert "triage" in history
    assert "metric_definition" in history
    assert "data_query" in history      # 先查数
    assert "report" in history          # 再整合出报告
    assert "generate_answer" in history
    assert "critic" in history
    # 不应经过不需要的节点
    assert "root_cause" not in history


# ============================================================
# 运行时兼容性
# ============================================================


def test_run_chat_workflow_returns_compatible_response() -> None:
    """验证 run_chat_workflow 返回值与 ChatResponse schema 兼容。"""
    result = run_chat_workflow("GMV 是什么？")
    assert "task_type" in result
    assert "answer" in result
    assert "trace_id" in result


def test_chat_api_returns_compatible_schema() -> None:
    """验证 /chat 接口返回值兼容。"""
    response = client.post("/chat", json={"message": "GMV 是什么？"})
    body = response.json()
    assert response.status_code == 200
    assert "task_type" in body
    assert "answer" in body
    assert "trace_id" in body


def test_trace_records_graph_path(monkeypatch) -> None:
    """验证 trace 包含 graph_path + state_snapshot + events。"""
    _patch_metric_definition_path(monkeypatch, keep_trace=True)
    graph = build_chat_graph()
    result = graph.invoke({"user_message": "GMV 是什么？", "trace_id": "t5"})
    from app.harness.storage import read_trace

    trace = read_trace(result["trace_id"])
    assert trace is not None
    assert "graph_path" in trace
    assert "state_snapshot" in trace
    assert "events" in trace

    from app.harness.evaluator import evaluate_trace_contract

    contract = evaluate_trace_contract(trace)
    assert contract["passed"], contract


def test_critic_runs_after_answer(monkeypatch) -> None:
    """验证 critic 在 generate_answer 之后运行（审核最终 answer）。"""
    _patch_metric_definition_path(monkeypatch)
    graph = build_chat_graph()
    result = graph.invoke({"user_message": "GMV 是什么？", "trace_id": "t6"})
    history = [e["node"] for e in result.get("node_history", [])]
    # generate_answer 应在 critic 之前
    gen_idx = history.index("generate_answer")
    crit_idx = history.index("critic")
    assert gen_idx < crit_idx, "generate_answer must run before critic"


def test_revision_counts_recorded(monkeypatch) -> None:
    """验证每个 specialist 在 _revision_counts 中留下记录。"""
    _patch_metric_definition_path(monkeypatch)
    graph = build_chat_graph()
    result = graph.invoke({"user_message": "GMV 是什么？", "trace_id": "t7"})
    counts = result.get("_revision_counts", {})
    assert counts.get("metric_definition", 0) >= 1


def test_root_cause_handoffs_pending_queries_to_data_query() -> None:
    """Root Cause 首次执行只提交计划，由图回调 Data Query Agent 执行。"""
    from app.graph.nodes.root_cause import root_cause

    result = root_cause({"user_message": "为什么 2018-08-10 的 GMV 下降了？", "tool_result": {}})
    assert result["pending_query"]["purpose"] == "trend_baseline"
    assert len(result["root_cause_queue"]) == 5
    assert {item["purpose"].split(":", 1)[1] for item in result["root_cause_queue"]} == {
        "customer_state", "customer_city", "seller_state", "product_category", "payment_type",
    }
    assert result["handoffs"][-1]["to"] == "data_query"


def test_critic_returns_report_and_governed_diagnostic_as_revision_targets(monkeypatch) -> None:
    """Draft + Critic 返工必须回到生成草案的 specialist。"""
    from app.graph.nodes.critic import critic

    issue = {"verdict": "failed", "issues": ["missing evidence"]}
    monkeypatch.setattr("app.graph.nodes.critic.llm_review_groundedness", lambda *_, **__: issue)
    monkeypatch.setattr("app.graph.nodes.critic.llm_review_evidence_completeness", lambda *_, **__: {"verdict": "passed", "issues": []})
    monkeypatch.setattr("app.graph.nodes.critic.llm_review_logic_consistency", lambda *_, **__: {"verdict": "passed", "issues": []})
    # RAG is not an optional Critic review input. The current Critic contract
    # reviews groundedness, evidence completeness and logic consistency.
    monkeypatch.setattr("app.graph.nodes.critic.review_data_query_result", lambda *_: {"status": "approved"})

    report_result = critic({"task_type": "report_generation", "answer": "draft", "rag_result": {}, "tool_result": {}})
    anomaly_result = critic({"task_type": "anomaly_analysis", "answer": "draft", "rag_result": {}, "tool_result": {}})
    assert report_result["critic_revision_target"] == "report", {
        "verdict": report_result.get("critic_verdict"),
        "decision": report_result.get("critic_decisions"),
        "llm_review": (report_result.get("critic_result") or {}).get("llm_review"),
        "errors": [item.get("code") for item in report_result.get("errors", []) if isinstance(item, dict)],
    }
    assert anomaly_result["critic_revision_target"] == "generate_answer"


def test_report_consumes_critic_feedback(monkeypatch) -> None:
    """被打回的报告草案必须保留并呈现 Critic 的修订上下文。"""
    from app.graph.nodes.report import report

    monkeypatch.setattr("app.graph.nodes.report.generate_report", lambda *_: "# 经营分析报告（草案）")
    result = report({
        "user_message": "生成本周经营分析报告",
        "_revision_counts": {"report": 1},
        "critic_decisions": [{"revision_target": "report", "llm_issues": ["缺少数据来源"]}],
    })
    assert "Critic 修订记录" in result["analysis_result"]["summary"]
    assert result["analysis_result"]["revision_feedback"] == ["缺少数据来源"]

"""评测入口 — Target 多 Agent 评测。

本文件是项目自研 Harness 的扩展版，评测范围包括：
1. RAG 命中率（原有）
2. Triage 路由正确率（新增）
3. 多 Agent 路径合规性（新增）
4. Critic 审核完整性（新增）
5. 单条 trace 的路由、证据、返工与放行契约（Target）
"""

from app.graph.workflow import build_chat_graph
from app.rag.retriever import retrieve_knowledge


def evaluate_trace_contract(trace: dict) -> dict:
    """检查一条持久化 trace 是否满足 Target 多 Agent 可审计契约。

    该函数不重新运行 workflow，只消费 trace，因此可被离线 Harness、CI 和
    事后回放共同使用。它覆盖 Target 完成标准中最关键的可观测性：handoff、
    工具证据、Critic 裁决、最终回答；异常路径还必须呈现 Root Cause 回查证据。
    """

    snapshot = trace.get("state_snapshot", {})
    graph_path = trace.get("graph_path", [])
    task_type = snapshot.get("task_type")
    checks = {
        "has_graph_path": bool(graph_path),
        "has_handoffs": bool(snapshot.get("handoffs")),
        "has_critic_decision": bool(snapshot.get("critic_decisions")),
        "has_final_answer": bool(snapshot.get("final_answer") or snapshot.get("answer")),
        "claims_evidence_bound": bool(snapshot.get("claims")) and all(
            claim.get("source_evidence_ids") for claim in snapshot.get("claims", [])
        ),
        "has_tool_evidence": task_type not in {"metric_query", "anomaly_analysis", "report_generation"} or bool(snapshot.get("tool_calls")),
        "anomaly_loop_recorded": task_type != "anomaly_analysis" or (
            graph_path.count("root_cause") >= 2
            and graph_path.count("data_query") >= 2
            and bool(snapshot.get("root_cause_evidence"))
        ),
        "critic_terminal": snapshot.get("critic_verdict") == "approved",
    }
    return {
        "task_type": task_type,
        "passed": all(checks.values()),
        "checks": checks,
        "graph_path": graph_path,
    }


def _evaluate_rag() -> dict:
    """执行 RAG 小样本命中率评测。"""
    cases = [
        {"query": "GMV 是什么？", "doc_types": {"metric"}, "expected_source": "docs/knowledge/metrics/gmv.md"},
        {"query": "异常分析应该怎么做？", "doc_types": {"sop"}, "expected_source": "docs/knowledge/sop/anomaly_analysis.md"},
        {"query": "生成周报需要哪些模块？", "doc_types": {"template"}, "expected_source": "docs/knowledge/templates/weekly_report.md"},
    ]
    passed = 0
    details = []
    for case in cases:
        result = retrieve_knowledge(case["query"], doc_types=case["doc_types"], top_k=3)
        sources = [hit.citation.source_path for hit in result.hits]
        ok = any(source.endswith(case["expected_source"]) for source in sources)
        passed += int(ok)
        details.append({"query": case["query"], "strategy": result.strategy, "expected_source": case["expected_source"], "sources": sources, "passed": ok})
    return {"cases": len(cases), "passed": passed, "hit_rate": passed / len(cases), "details": details}


def _evaluate_triage() -> dict:
    """执行 Triage Agent 路由正确率评测。

    验证不同问题类型是否被路由到正确的 task_type。
    LLM 不可用时降级到关键词匹配，仍应覆盖基本路由。
    """
    cases = [
        {"query": "GMV 是什么？", "expected": "metric_definition"},
        {"query": "什么是客单价？", "expected": "metric_definition"},
        {"query": "查询 2018-08-10 的 GMV", "expected": "metric_query"},
        {"query": "最近 7 天订单量趋势", "expected": "metric_query"},
        {"query": "为什么 GMV 下降了？", "expected": "anomaly_analysis"},
        {"query": "订单量异常的原因", "expected": "anomaly_analysis"},
        {"query": "生成本周经营分析报告", "expected": "report_generation"},
        {"query": "写一份 GMV 分析日报", "expected": "report_generation"},
    ]
    graph = build_chat_graph()
    passed = 0
    details = []
    for case in cases:
        result = graph.invoke({"user_message": case["query"], "trace_id": f"eval-{cases.index(case)}"})
        ok = result.get("task_type") == case["expected"]
        passed += int(ok)
        details.append({"query": case["query"], "expected": case["expected"], "actual": result.get("task_type"), "passed": ok})
    return {"cases": len(cases), "passed": passed, "accuracy": passed / len(cases), "details": details}


def _evaluate_critic() -> dict:
    """执行 Critic 审核完整性评测。

    验证 graph 执行后 critic 是否正常输出审核结果。
    """
    graph = build_chat_graph()
    paths = [
        {"query": "GMV 是什么？", "expected_type": "metric_definition"},
        {"query": "查询 2018-08-10 的 GMV", "expected_type": "metric_query"},
    ]
    passed = 0
    details = []
    for path in paths:
        result = graph.invoke({"user_message": path["query"], "trace_id": f"eval-critic-{paths.index(path)}"})
        has_critic_verdict = result.get("critic_verdict") is not None
        has_critic_result = bool(result.get("critic_result"))
        has_node_history = bool(result.get("node_history"))
        task_type_ok = result.get("task_type") == path["expected_type"]
        ok = all([has_critic_verdict, has_critic_result, task_type_ok])
        passed += int(ok)
        details.append({
            "query": path["query"],
            "task_type_ok": task_type_ok,
            "has_critic_verdict": has_critic_verdict,
            "has_critic_result": has_critic_result,
            "has_node_history": has_node_history,
            "verdict": result.get("critic_verdict"),
            "passed": ok,
        })
    return {"cases": len(paths), "passed": passed, "completeness": passed / len(paths), "details": details}


def run_evaluation() -> dict:
    """执行项目多维度评测。"""
    rag = _evaluate_rag()
    triage = _evaluate_triage()
    critic_review = _evaluate_critic()
    all_passed = rag["passed"] + triage["passed"] + critic_review["passed"]
    all_cases = rag["cases"] + triage["cases"] + critic_review["cases"]
    return {
        "status": "ok",
        "message": "Target multi-agent evaluation harness",
        "summary": {"total_cases": all_cases, "total_passed": all_passed, "overall_rate": round(all_passed / all_cases, 4) if all_cases else 0},
        "rag": rag,
        "triage": triage,
        "critic": critic_review,
    }

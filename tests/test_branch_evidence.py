"""P1.1 受控并行分支测试：per-branch 结果、失败隔离、证据汇聚。"""

from __future__ import annotations

from app.graph.nodes.branch_evidence import branch_evidence


def _queue(monkeypatch):
    items = [
        {"purpose": "trend_baseline", "question": "查询最近 7 天 gmv 趋势", "query_id": "q1"},
        {"purpose": "dimension_contribution:product_category", "question": "拆解品类贡献度", "query_id": "q2"},
        {"purpose": "dimension_contribution:customer_state", "question": "拆解客户州贡献度", "query_id": "q3"},
    ]

    def fake_build(question):
        return items
    monkeypatch.setattr("app.graph.nodes.branch_evidence._build_follow_up_queue", fake_build)
    return items


def test_branch_evidence_writes_per_branch_results(monkeypatch) -> None:
    _queue(monkeypatch)

    def fake_query(task_type, question, **kwargs):
        return {"tool_name": "query_trend", "result": {"status": "ok", "points": [{"bucket": "2018-08-01", "value": 100.0}]}, "request_id": kwargs.get("request_id", "")}
    monkeypatch.setattr("app.graph.nodes.branch_evidence.query_business_data", fake_query)

    state = {"task_type": "anomaly_analysis", "user_message": "为什么 gmv 下降", "trace_id": "t1", "tenant_id": "p"}
    result = branch_evidence(state)

    # 每个分支只写自己的 key（不共享 dict 原地覆盖）
    assert set(result["branch_results"].keys()) == {
        "trend_baseline", "dimension_contribution:product_category", "dimension_contribution:customer_state",
    }
    assert all(v["status"] == "ok" for v in result["branch_results"].values())
    # 证据汇聚给 root_cause 统一评估，队列置空（不再串行循环）
    assert len(result["root_cause_evidence"]) == 3
    assert result["root_cause_queue"] == []
    assert result["next_agent"] == "root_cause"
    assert "branch_timings" in result


def test_branch_failure_is_isolated_as_unavailable(monkeypatch) -> None:
    _queue(monkeypatch)

    def fake_query(task_type, question, **kwargs):
        if "品类" in question:
            raise RuntimeError("mcp down")
        return {"tool_name": "query_trend", "result": {"status": "ok", "points": [{"bucket": "2018-08-01", "value": 100.0}]}, "request_id": kwargs.get("request_id", "")}
    monkeypatch.setattr("app.graph.nodes.branch_evidence.query_business_data", fake_query)

    state = {"task_type": "anomaly_analysis", "user_message": "为什么 gmv 下降", "trace_id": "t1"}
    result = branch_evidence(state)

    failed = result["branch_results"]["dimension_contribution:product_category"]
    assert failed["status"] == "unavailable"
    assert "branch_error" in failed["result"]["result"]["reason"]
    # 失败只隔离该分支；其余分支证据仍在，主链不被阻断
    assert result["branch_results"]["trend_baseline"]["status"] == "ok"
    assert len(result["root_cause_evidence"]) == 3


def test_branch_evidence_skips_non_anomaly() -> None:
    state = {"task_type": "metric_query", "user_message": "查 GMV"}
    result = branch_evidence(state)
    assert "branch_results" not in result  # 非异常任务透传，不改 state


def test_branch_true_concurrency_uses_per_thread_context(monkeypatch) -> None:
    """真实并发重叠：共享同一个 Context 会抛 'cannot enter context' 重入错误。

    用 barrier 强制所有分支同时到达查询函数——若共享 context，第二个线程的
    ctx.run 必然失败；修复后每个线程各自 copy_context，所有分支应全部成功。
    """
    import threading

    from app.graph.nodes.branch_evidence import branch_evidence

    _queue(monkeypatch)
    barrier = threading.Barrier(3)
    active = {"n": 0, "max": 0}
    lock = threading.Lock()

    def fake_query(task_type, question, **kwargs):
        with lock:
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
        barrier.wait(timeout=10)  # 3 个分支同时到达 → 强制并发重叠
        with lock:
            active["n"] -= 1
        return {"tool_name": "q", "result": {"status": "ok", "points": []}, "request_id": kwargs.get("request_id", "")}
    monkeypatch.setattr("app.graph.nodes.branch_evidence.query_business_data", fake_query)

    result = branch_evidence({"task_type": "anomaly_analysis", "user_message": "why", "trace_id": "t"})
    assert active["max"] == 3  # 3 个分支真实并发
    # 全部成功（无 'already entered' 崩溃），失败隔离未误伤
    assert len(result["branch_results"]) == 3
    assert all(v["status"] == "ok" for v in result["branch_results"].values())

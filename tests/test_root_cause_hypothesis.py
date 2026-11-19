"""Tests for root cause hypothesis chain evaluation (single + batch paths).

覆盖场景：
  1. 单指标最后一条证据在队列为空时仍被评价
  2. 单指标正常耗尽队列仍包含完整链
  3. 批量每个 metric 的最后一条证据会被评价
  4. 批量 metric 之间不串链
  5. 批量 supported + 交叉检查提前停止
  6. 批量连续 refuted + 高优维度全覆盖后提前停止
  7. _evaluate_step 按 query_id 精确绑定 MCP EvidenceBundle
"""

from app.graph.nodes import root_cause as rc_module
from app.graph.nodes.root_cause import (
    _evaluate_pending_track_evidence,
    _evaluate_step,
    _find_evidence_id_by_query_id,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_evidence(metric_id: str, query_id: str, purpose_suffix: str = "customer_state") -> dict:
    """构建一条最小根因证据，status 不为 ok 以跳过 calculate_change_contribution。"""
    return {
        "metric_id": metric_id,
        "query_id": query_id,
        "purpose": f"dimension_contribution:{purpose_suffix}",
        "hypothesis": f"test hypothesis for {metric_id}/{purpose_suffix}",
        "result": {"result": {"contribution_result": {"status": "error", "reason": "mock"}}},
    }


def _make_hs_step(step: int, query_id: str, metric_id: str,
                  verdict: str = "inconclusive",
                  purpose: str = "dimension_contribution:test") -> dict:
    return {
        "step": step, "query_id": query_id, "metric_id": metric_id,
        "verdict": verdict, "purpose": purpose,
        "hypothesis": "test", "rationale": "mock",
        "next_action": "continue", "evidence_ids": [],
    }


# ---------------------------------------------------------------------------
# 1) 单指标最后一条证据
# ---------------------------------------------------------------------------

def test_single_last_evidence_evaluated_when_queue_empty(monkeypatch) -> None:
    """单指标：队列为空时 root_cause_evidence 中最后一条证据仍被放入 hypothesis_chain。"""
    evaluated_qids = []
    monkeypatch.setattr(
        rc_module, "_evaluate_step",
        lambda entry, step, bundles: (
            evaluated_qids.append(entry.get("query_id"))
            or _make_hs_step(step, entry.get("query_id", ""), "gmv")
        ),
    )

    state = {
        "user_message": "为什么 GMV 下降",
        "tool_result": {"result": {"metric_result": {"metric_name": "gmv"}}},
        "root_cause_queue": [],
        "root_cause_evidence": [
            _make_evidence("gmv", "rcq_last", "customer_state"),
        ],
        "_evaluated_query_ids": [],
        "hypothesis_chain": [],
        "evidence_bundles": [],
        "trace_id": "test",
    }
    result = rc_module.root_cause(state)

    assert len(result["hypothesis_chain"]) == 1, (
        f"期望 hypothesis_chain 有 1 个条目，实际 {len(result['hypothesis_chain'])}"
    )
    assert result["hypothesis_chain"][0]["query_id"] == "rcq_last"
    assert "rcq_last" in evaluated_qids


# ---------------------------------------------------------------------------
# 2) 单指标正常耗尽
# ---------------------------------------------------------------------------

def test_single_queue_exhaustion_produces_complete_chain(monkeypatch) -> None:
    """单指标正常耗尽队列：所有证据都被评价。"""
    step_counter = iter(range(1, 10))
    monkeypatch.setattr(
        rc_module, "_evaluate_step",
        lambda entry, step, bundles: _make_hs_step(
            next(step_counter), entry.get("query_id", ""), "gmv"
        ),
    )

    state = {
        "user_message": "为什么 GMV 下降",
        "tool_result": {"result": {"metric_result": {"metric_name": "gmv"}}},
        "root_cause_queue": [],
        "root_cause_evidence": [
            _make_evidence("gmv", "rcq_a", "customer_state"),
            _make_evidence("gmv", "rcq_b", "customer_city"),
        ],
        "_evaluated_query_ids": [],
        "hypothesis_chain": [],
        "evidence_bundles": [],
        "trace_id": "test",
    }
    result = rc_module.root_cause(state)
    assert len(result["hypothesis_chain"]) == 2


# ---------------------------------------------------------------------------
# 3) 批量最后一条证据
# ---------------------------------------------------------------------------

def _make_track(metric_id: str, queue: list | None = None) -> dict:
    return {
        "metric_id": metric_id,
        "primary_result": {"result": {"parsed": {"metric_name": metric_id}}},
        "queue": queue or [],
        "_evaluated_query_ids": [],
        "_stop_reason": "",
    }


def test_batch_last_evidence_evaluated_when_queue_empty(monkeypatch) -> None:
    """批量：每个 metric 最后一条证据在 track 队列为空时仍被评价。"""
    evaluated_qids = []
    monkeypatch.setattr(
        rc_module, "_evaluate_step",
        lambda entry, step, bundles: (
            evaluated_qids.append(entry.get("query_id"))
            or _make_hs_step(step, entry.get("query_id", ""), entry.get("metric_id", ""))
        ),
    )

    state = {
        "root_cause_tracks": [
            _make_track("gmv"),
            _make_track("avg_order_value"),
        ],
        "root_cause_evidence": [
            _make_evidence("gmv", "rcq_gmv_last", "customer_state"),
            _make_evidence("avg_order_value", "rcq_aov_last", "payment_type"),
        ],
        "evidence_bundles": [],
        "trace_id": "test",
        "node_history": [],
    }
    result = rc_module._batch_root_cause(state)

    chains = result["hypothesis_chains_by_metric"]
    assert len(chains.get("gmv", [])) == 1, "GMV hypothesis 缺失"
    assert len(chains.get("avg_order_value", [])) == 1, "AOV hypothesis 缺失"
    assert chains["gmv"][0]["query_id"] == "rcq_gmv_last"
    assert chains["avg_order_value"][0]["query_id"] == "rcq_aov_last"
    assert "rcq_gmv_last" in evaluated_qids
    assert "rcq_aov_last" in evaluated_qids


# ---------------------------------------------------------------------------
# 4) 批量不串链
# ---------------------------------------------------------------------------

def test_batch_metric_chains_do_not_cross_contaminate(monkeypatch) -> None:
    """批量 metric 的 hypothesis_chain 各自独立。"""
    monkeypatch.setattr(
        rc_module, "_evaluate_step",
        lambda entry, step, bundles: _make_hs_step(
            step, entry.get("query_id", ""), entry.get("metric_id", "")
        ),
    )

    state = {
        "root_cause_tracks": [
            _make_track("gmv"),
            _make_track("order_count"),
        ],
        "root_cause_evidence": [
            _make_evidence("gmv", "rcq_g1", "customer_state"),
            _make_evidence("order_count", "rcq_o1", "payment_type"),
            _make_evidence("gmv", "rcq_g2", "customer_city"),
        ],
        "evidence_bundles": [],
        "trace_id": "test",
        "node_history": [],
    }
    result = rc_module._batch_root_cause(state)

    chains = result["hypothesis_chains_by_metric"]
    assert len(chains["gmv"]) == 2
    assert len(chains["order_count"]) == 1
    # GMV 的 query_id 不混入 order_count
    gmv_qids = {h["query_id"] for h in chains["gmv"]}
    order_qids = {h["query_id"] for h in chains["order_count"]}
    assert gmv_qids == {"rcq_g1", "rcq_g2"}
    assert order_qids == {"rcq_o1"}


# ---------------------------------------------------------------------------
# 5) 批量提前停止 — supported + cross-check
# ---------------------------------------------------------------------------

def test_batch_stop_supported_after_cross_check() -> None:
    """_evaluate_pending_track_evidence: supported + 至少 1 交叉检查 → 停止。"""
    track = {
        "metric_id": "gmv",
        "hypothesis_chain": [
            _make_hs_step(1, "rcq_1", "gmv", "inconclusive",
                          "dimension_contribution:customer_state"),
        ],
        "_evaluated_query_ids": ["rcq_1"],
    }
    # 模拟一条新证据需要评价
    state = {
        "root_cause_evidence": [
            _make_evidence("gmv", "rcq_2", "customer_city"),
        ],
        "evidence_bundles": [],
    }

    # 手动 mock _evaluate_step 返回 supported
    monkeypatch = __import__("pytest").MonkeyPatch()
    monkeypatch.setattr(
        rc_module, "_evaluate_step",
        lambda entry, step, bundles: _make_hs_step(
            2, "rcq_2", "gmv", "supported", "dimension_contribution:customer_city"
        ),
    )
    try:
        result = _evaluate_pending_track_evidence(track, state)
        assert result == "dominant_contributor_supported_after_cross_check"
        # 最后一步的 next_action 应为 stop
        assert track["hypothesis_chain"][-1]["next_action"] == "stop"
    finally:
        monkeypatch.undo()


# ---------------------------------------------------------------------------
# 6) 批量提前停止 — 连续 refuted + 高优维度
# ---------------------------------------------------------------------------

def test_batch_stop_consecutive_refuted_priority_covered() -> None:
    """_evaluate_pending_track_evidence: 连续 2 refuted + 高优维度全覆盖 → 停止。"""
    track = {
        "metric_id": "freight_value",
        "hypothesis_chain": [
            _make_hs_step(1, "rcq_1", "freight_value", "refuted",
                          "dimension_contribution:customer_state"),
        ],
        "_evaluated_query_ids": ["rcq_1"],
    }
    state = {
        "root_cause_evidence": [
            _make_evidence("freight_value", "rcq_2", "product_category"),
        ],
        "evidence_bundles": [],
    }

    monkeypatch = __import__("pytest").MonkeyPatch()
    monkeypatch.setattr(
        rc_module, "_evaluate_step",
        lambda entry, step, bundles: _make_hs_step(
            2, "rcq_2", "freight_value", "refuted",
            "dimension_contribution:product_category"
        ),
    )
    # freight_value 的高优维度是 product_category, customer_state
    # 两个维度都已覆盖，应停止
    try:
        result = _evaluate_pending_track_evidence(track, state)
        assert result == "no_material_contributor_in_examined_dimensions"
        # 最后一步的 next_action 应为 stop
        assert track["hypothesis_chain"][-1]["next_action"] == "stop"
    finally:
        monkeypatch.undo()


# ---------------------------------------------------------------------------
# 7) query_id → MCP EvidenceBundle 精确绑定
# ---------------------------------------------------------------------------

def test_find_evidence_id_by_query_id_matches() -> None:
    """_find_evidence_id_by_query_id 按 query_id 精确匹配 MCP bundle。"""
    bundles = [
        {"source_type": "mcp", "evidence_id": "mcp_a", "content": {"query_id": "rcq_111"}},
        {"source_type": "mcp", "evidence_id": "mcp_b", "content": {"query_id": "rcq_222"}},
        {"source_type": "root_cause", "evidence_id": "rc_1", "content": {}},
    ]
    assert _find_evidence_id_by_query_id(bundles, "rcq_111") == "mcp_a"
    assert _find_evidence_id_by_query_id(bundles, "rcq_222") == "mcp_b"
    assert _find_evidence_id_by_query_id(bundles, "rcq_999") == ""


def test_evaluate_step_embeds_mcp_evidence_id(monkeypatch) -> None:
    """_evaluate_step 结果中的 evidence_ids 包含对应 MCP bundle 的 evidence_id。"""
    from datetime import date
    from_app = __import__("app.analysis.contracts", fromlist=["ContributionFinding", "ContributionItem"])
    monkeypatch.setattr(
        "app.graph.nodes.root_cause.calculate_change_contribution",
        lambda **_kwargs: from_app.ContributionFinding(
            items=[from_app.ContributionItem(
                dimension_value="SP", current_value=200.0, baseline_value=100.0,
                delta_value=100.0, contribution_ratio=1.0,
            )],
            total_delta=100.0, dimension="customer_state",
            evidence_ids=["mcp_xyz"], metric_name="gmv",
            current_date=date(2018, 8, 10), baseline_date=date(2018, 8, 9),
        ),
    )

    entry = {
        "metric_id": "gmv",
        "query_id": "rcq_xyz",
        "purpose": "dimension_contribution:customer_state",
        "hypothesis": "test",
        "result": {"result": {"contribution_result": {
            "status": "ok", "metric_name": "gmv",
            "current_date": "2018-08-10", "baseline_date": "2018-08-09",
            "current_items": [], "baseline_items": [], "total_delta": 100.0,
        }}},
    }
    bundles = [
        {"source_type": "mcp", "evidence_id": "mcp_xyz", "content": {"query_id": "rcq_xyz"}},
    ]
    hs = _evaluate_step(entry, 1, bundles)
    assert "mcp_xyz" in hs.get("evidence_ids", []), (
        f"期望 evidence_ids 包含 mcp_xyz，实际 {hs.get('evidence_ids')}"
    )

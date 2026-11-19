"""Root Cause Agent — 通过 Data Query Agent 多轮回查并完成贡献度归因。

执行模型是一个显式的 LangGraph 协作循环：
1. Root Cause 根据异常问题建立趋势和维度拆解的证据队列。
2. 图将 `pending_query` handoff 给 Data Query Agent 执行 MCP 工具。
3. Data Query Agent 把结果写入 `root_cause_evidence`，再回到本节点。
4. 队列耗尽后，本节点才产出最终归因结论交给 Critic。

MECE 写入字段：analysis_result, pending_query, root_cause_queue,
                current_agent, node_history, handoffs, messages
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any
from uuid import uuid4

from app.agents.data_query_agent import build_metric_query_plan
from app.agents.contracts import EvidenceBundle, stable_evidence_id
from app.analysis.contracts import ContributionFinding, HypothesisStep, HypothesisVerdict
from app.analysis.semantic import get_metric_definition
from app.analysis.contracts import MetricPoint, MetricSeries, PeriodSpec
from app.analysis.contribution import calculate_change_contribution
from app.analysis.statistical_anomaly import detect_series_anomalies
from app.graph.state import AgentState


DIMENSION_LABELS = {
    "customer_state": "客户州",
    "customer_city": "客户城市",
    "seller_state": "卖家州",
    "product_category": "品类",
    "payment_type": "支付方式",
}
DIMENSION_KEYWORDS: dict[str, tuple[str, ...]] = {
    # Long phrases come first in matching so "客户城市" is not reduced to
    # the broader customer-state intent.
    "product_category": ("商品类别", "商品分类", "商品", "品类", "sku", "product", "item"),
    "customer_city": ("客户城市", "城市", "customer city"),
    "customer_state": ("客户州", "客户地区", "客户区域", "customer state", "customer region"),
    "seller_state": ("卖家州", "卖家地区", "seller state"),
    "payment_type": ("支付方式", "付款方式", "payment type", "payment method"),
}
MAX_ROOT_CAUSE_DIMENSIONS = 5
MIN_CONTRIBUTION_RATIO = 0.01
CONFIRMATION_THRESHOLD = 0.5
MAX_ROOT_CAUSE_ITERATIONS = 1 + MAX_ROOT_CAUSE_DIMENSIONS

HIGH_PRIORITY_DIMENSIONS: dict[str, list[str]] = {
    "freight_value": ["product_category", "customer_state"],
    "gmv": ["product_category", "customer_state"],
    "order_count": ["product_category", "payment_type"],
}


def infer_requested_dimensions(question: str) -> list[str]:
    """Extract explicit dimension intent without allowing the model to widen it.

    This is deliberately deterministic.  A question asking for a product must
    not silently become a geography analysis merely because a model fallback
    happens to prefer geographic dimensions.
    """

    normalized = str(question or "").casefold()
    hits: list[str] = []
    for dimension, keywords in DIMENSION_KEYWORDS.items():
        if any(keyword.casefold() in normalized for keyword in keywords):
            hits.append(dimension)
    return hits


def _dimension_candidates(
    metric_name: str,
    question: str,
    *,
    scope_allowed: set[str] | None = None,
    max_dimensions: int = MAX_ROOT_CAUSE_DIMENSIONS,
) -> tuple[list[str], list[str]]:
    """Return selected dimensions and explicit-but-unavailable dimensions."""

    try:
        metric_allowed = set(get_metric_definition(metric_name).allowed_dimensions)
    except Exception:
        metric_allowed = set()
    requested = infer_requested_dimensions(question)
    allowed = metric_allowed.copy()
    if scope_allowed is not None:
        allowed &= scope_allowed

    unavailable = [dimension for dimension in requested if dimension not in allowed]
    if requested:
        # An explicit request wins over generic priorities.  If it is not
        # available, return no substitute dimension; the result must explain
        # the limitation instead of answering a different question.
        return [dimension for dimension in requested if dimension in allowed][:max_dimensions], unavailable

    if scope_allowed is not None:
        priority = HIGH_PRIORITY_DIMENSIONS.get(metric_name, [])
        if not priority:
            priority = ["product_category", "payment_type", "month_position"]
        return [dimension for dimension in priority if dimension in allowed][:max_dimensions], unavailable

    # Preserve the unrestricted legacy behavior for callers that do not have
    # a locked explorer scope.  Scoped execution uses the safer branch above.
    return [dimension for dimension in DIMENSION_LABELS if dimension in allowed][:max_dimensions], unavailable


def _mapping(value: Any) -> dict[str, Any]:
    """把外部/MCP 返回值安全地收敛为 mapping，避免旧链路空值崩溃。"""
    return value if isinstance(value, dict) else {}


def _safe_date(value: Any, fallback: date | None = None) -> date:
    """解析外部日期；异常时返回明确的兜底日期。"""
    fallback = fallback or date.today()
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return fallback


def _nested_result(value: Any) -> dict[str, Any]:
    """读取统一工具结果的内层 result，兼容 None/错误结构。"""
    return _mapping(_mapping(value).get("result"))


def _build_follow_up_queue(
    question: str,
    historical_hints: str = "",
    *,
    scope: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """把异常问题转换成由 Data Query Agent 执行的补充证据队列。"""

    if historical_hints and len(question) < 15:
        question = f"{question}\n[历史上下文: {historical_hints}]"
    plan = build_metric_query_plan(question)
    if not plan.metric_name:
        return []

    metric_name = plan.metric_name
    metric_date = plan.metric_date
    # 当 LLM 未解析出日期时，回退到数据集最新日期或今天
    if not metric_date:
        from app.data.repositories.daily_metrics import get_latest_metric_date
        fallback = get_latest_metric_date()
        metric_date = fallback if fallback else str(date.today())
    # `最近 7 天` 由 Data Query Agent 解析为相对最新数据日期，适合没有
    # 当前系统日期与业务数据日期一致这一假设的离线数据集。
    queue = [{
        "purpose": "trend_baseline",
        "question": f"查询最近 7 天 {metric_name} 趋势",
        "query_id": f"rcq_{uuid4().hex[:8]}",
    }]
    scope_allowed = set(scope.get("allowed_dimensions") or []) if scope else None
    selected_dimensions, _ = _dimension_candidates(
        metric_name,
        question,
        scope_allowed=scope_allowed,
    )
    current_date = _safe_date(metric_date)
    baseline_date = current_date - timedelta(days=1)
    for dimension in selected_dimensions:
        label = DIMENSION_LABELS.get(dimension, dimension)
        queue.append({
            "purpose": f"dimension_contribution:{dimension}",
            "question": (
                f"拆解 {current_date.isoformat()} 与 {baseline_date.isoformat()} 的 "
                f"{metric_name} 按 {label}贡献度"
            ),
            "hypothesis": f"{label}结构变化能解释{metric_name}观察到的变化",
            "query_id": f"rcq_{uuid4().hex[:8]}",
        })
    return queue


def _build_follow_up_queue_for_metric(
    metric_name: str,
    metric_date: str,
    max_followups: int,
    *,
    question: str = "",
    scope: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Build a capability-aware bounded queue for one batch metric."""
    queue = [{"purpose": "trend_baseline", "question": f"查询最近 7 天 {metric_name} 趋势",
              "query_id": f"rcq_{uuid4().hex[:8]}"}]
    scope_allowed = set(scope.get("allowed_dimensions") or []) if scope else None
    selected_dimensions, _ = _dimension_candidates(
        metric_name,
        question,
        scope_allowed=scope_allowed,
        max_dimensions=max(0, max_followups - 1),
    )
    current_date = _safe_date(metric_date)
    baseline_date = current_date - timedelta(days=1)
    for dimension in selected_dimensions:
        label = DIMENSION_LABELS.get(dimension, dimension)
        queue.append({
                "purpose": f"dimension_contribution:{dimension}",
                "question": f"拆解 {current_date.isoformat()} 与 {baseline_date.isoformat()} 的 {metric_name} 按 {label} 贡献度",
                "hypothesis": f"{label}结构变化能解释{metric_name}观察到的变化",
                "query_id": f"rcq_{uuid4().hex[:8]}",
        })
    return queue


def _evaluate_pending_track_evidence(track: dict, state: AgentState) -> str | None:
    """评价一个 track 中未处理的新证据，并检查停止条件。返回 stop_reason 或 None。"""
    metric_evidence = [
        e for e in state.get("root_cause_evidence", [])
        if e.get("metric_id") == track["metric_id"]
        and e.get("purpose", "").startswith("dimension_contribution:")
        and e.get("query_id", "") not in track.get("_evaluated_query_ids", [])
    ]
    if not metric_evidence:
        return None

    for entry in metric_evidence:
        step = len(track.get("hypothesis_chain", [])) + 1
        hs = _evaluate_step(entry, step, state.get("evidence_bundles", []))
        track.setdefault("hypothesis_chain", []).append(hs)
        track.setdefault("_evaluated_query_ids", []).append(entry.get("query_id", ""))

    # === 停止条件：与单路径 root_cause() 保持一致 ===
    chain = track["hypothesis_chain"]
    metric_name = track["metric_id"]

    # 1) supported + 至少 1 个额外维度交叉检查
    supported_found = any(h["verdict"] == "supported" for h in chain)
    contrib_count = sum(1 for h in chain if h["purpose"].startswith("dimension_contribution:"))
    if supported_found and contrib_count >= 2:
        chain[-1]["next_action"] = "stop"
        return "dominant_contributor_supported_after_cross_check"

    # 2) 连续 2 个 refuted + 高优先级维度全覆盖
    recent = chain[-2:]
    if len(recent) == 2 and all(h["verdict"] == "refuted" for h in recent):
        required_dims = _get_priority_dimensions(metric_name)
        covered_dims = {
            h["purpose"].split(":", 1)[1]
            for h in chain if h["purpose"].startswith("dimension_contribution:")
        }
        if required_dims and required_dims <= covered_dims:
            chain[-1]["next_action"] = "stop"
            return "no_material_contributor_in_examined_dimensions"

    return None


def _batch_root_cause(state: AgentState) -> AgentState:
    """Run independent, serial root-cause tracks for every planned metric."""
    tracks = state.get("root_cause_tracks")
    if tracks is None:
        raw_results = state.get("query_results") or [state.get("tool_result", {})]
        metric_ids = [
            _nested_result(item).get("parsed", {}).get("metric_name")
            for item in raw_results if isinstance(item, dict)
        ]
        metric_ids = [metric for metric in metric_ids if metric]
        if not metric_ids:
            return _finalize_root_cause(state, limitations=["未获得可用于根因分析的已确认指标。"])
        per_metric_limit = 5 if len(metric_ids) <= 3 else 4
        scope = state.get("controlled_analysis_scope", {})
        requested_dimensions = list(state.get("requested_dimensions") or infer_requested_dimensions(str(state.get("user_message") or "")))
        state["requested_dimensions"] = requested_dimensions
        metric_allowed_union: set[str] = set()
        for metric_id in metric_ids:
            try:
                metric_allowed_union.update(get_metric_definition(metric_id).allowed_dimensions)
            except Exception:
                continue
        scope_allowed = set(scope.get("allowed_dimensions") or []) if scope else None
        if scope_allowed is not None:
            metric_allowed_union &= scope_allowed
        state["unavailable_requested_dimensions"] = [
            dimension for dimension in requested_dimensions if dimension not in metric_allowed_union
        ]
        default_date = str(scope.get("default_end_date") or "2018-10-17")
        tracks = []
        for metric_id, primary in zip(metric_ids, raw_results):
            parsed = _mapping(_nested_result(primary).get("parsed"))
            metric_date = parsed.get("metric_date") or parsed.get("end_date") or default_date
            tracks.append({
                "metric_id": metric_id,
                "primary_result": primary,
                "queue": _build_follow_up_queue_for_metric(
                    metric_id,
                    metric_date,
                    per_metric_limit,
                    question=str(state.get("user_message") or ""),
                    scope=scope,
                ),
                "max_followups": per_metric_limit,
                "_evaluated_query_ids": [],
                "_stop_reason": "",
            })
        state["root_cause_tracks"] = tracks

    for track in tracks:
        # 1) 始终先评价新证据（放在空队列检查之前，确保最后一条证据不遗漏）
        stop_reason = _evaluate_pending_track_evidence(track, state)
        if stop_reason:
            track["_stop_reason"] = stop_reason

        # 已停止的 track 跳过，不再出队列
        if track.get("_stop_reason"):
            continue

        queue = track.get("queue", [])
        if not queue:
            continue

        pending = queue.pop(0)
        pending["metric_id"] = track["metric_id"]
        state["pending_query"] = pending
        state["next_agent"] = "root_cause"
        state["current_agent"] = "root_cause"
        state.setdefault("node_history", []).append({
            "node": "root_cause",
            "summary": f"Root cause {track['metric_id']}: requested {pending['purpose']} ({len(queue)}/{track['max_followups']} follow-ups remain).",
        })
        return state

    # All tracks exhausted or stopped → aggregate
    all_evidence = state.get("root_cause_evidence", [])
    findings: list[dict[str, Any]] = []
    summaries: list[str] = []
    for track in tracks:
        metric_id = track["metric_id"]
        evidence = [entry for entry in all_evidence if entry.get("metric_id") == metric_id]
        summary = _build_grounded_summary(track["primary_result"], evidence)
        stop_reason = track.get("_stop_reason", "batch_queue_exhausted")
        findings.append({
            "metric_id": metric_id, "summary": summary, "evidence_count": len(evidence),
            "stop_reason": stop_reason,
        })
        summaries.append(summary)
        content = {"metric_id": metric_id, "summary": summary, "evidence_count": len(evidence)}
        state.setdefault("evidence_bundles", []).append(EvidenceBundle(
            evidence_id=stable_evidence_id("root_cause", f"batch:{metric_id}", content),
            source_type="root_cause", source_ref=f"root_cause:{metric_id}", content=content,
        ).model_dump(mode="json"))
    # 聚合所有 track 的 hypothesis_chain
    state["hypothesis_chains_by_metric"] = {
        t["metric_id"]: t.get("hypothesis_chain", [])
        for t in tracks if t.get("hypothesis_chain")
    }
    state["analysis_result"] = {
        "summary": "\n\n".join(summaries), "per_metric_findings": findings,
        "hypothesis_chains_by_metric": state["hypothesis_chains_by_metric"],
        "task_type": "anomaly_analysis", "stop_reason": "batch_completed",
        "limitations": [
            "贡献分析不等同于因果证明。维度贡献表明相关关系，不能独立确立因果关系。",
        ],
    }
    state["pending_query"] = None
    state["next_agent"] = "generate_answer"
    state["current_agent"] = "root_cause"
    state.setdefault("node_history", []).append({
        "node": "root_cause", "summary": f"Completed {len(tracks)} independent root-cause tracks.",
    })
    return state


def _contribution_from_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair current/baseline breakdowns and calculate change contribution."""

    contributions: list[dict[str, Any]] = []
    for entry in evidence:
        purpose = entry.get("purpose", "")
        if not purpose.startswith("dimension_contribution:"):
            continue
        dimension = purpose.split(":", 1)[1]
        payload = _mapping(_nested_result(entry).get("result")).get("contribution_result") or {}
        if payload.get("status") != "ok":
            continue
        finding = calculate_change_contribution(
            str(payload.get("metric_name")), dimension,
            _safe_date(payload.get("current_date")),
            _safe_date(payload.get("baseline_date")),
            payload.get("current_items", []), payload.get("baseline_items", []),
            payload.get("total_delta"),
        )
        if not finding.items:
            continue
        top = finding.items[0]
        if top.contribution_ratio is not None and abs(top.contribution_ratio) < MIN_CONTRIBUTION_RATIO:
            continue
        from app.metrics.registry import get_registry

        metric_definition = get_registry().get(finding.metric_name)
        contributions.append({
            "metric_name": finding.metric_name,
            "metric_definition": metric_definition.documentation,
            "definition_version": metric_definition.version,
            "unit": metric_definition.unit,
            "aggregation": metric_definition.aggregation,
            "dimension": dimension,
            "items": [item.model_dump(mode="json") for item in finding.items],
            "current_date": finding.current_date.isoformat(),
            "baseline_date": finding.baseline_date.isoformat(),
            "top_contributor": top.dimension_value,
            "current_value": top.current_value,
            "baseline_value": top.baseline_value,
            "delta_value": top.delta_value,
            "contribution_ratio": top.contribution_ratio,
            "total_delta": finding.total_delta,
            "residual_value": finding.residual_value,
            "residual_type": finding.residual_type,
            "residual_explainable": finding.residual_explainable,
            "residual_tolerance": finding.residual_tolerance,
            "residual_note": finding.residual_note,
            "current_period": finding.current_period.model_dump(mode="json") if finding.current_period else None,
            "baseline_period": finding.baseline_period.model_dump(mode="json") if finding.baseline_period else None,
            "evidence_ids": finding.evidence_ids,
            "source_table": payload.get("source_table"),
        })
    return contributions


def _anomalies_from_evidence(metric_name: str, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trend_entry = next((item for item in evidence if item.get("purpose") == "trend_baseline"), {})
    trend = _mapping(_nested_result(trend_entry).get("result")).get("trend_result") or {}
    raw_points = trend.get("points", [])
    if not raw_points:
        return []
    points = [MetricPoint(bucket=_safe_date(point.get("bucket")), value=point.get("value")) for point in raw_points if isinstance(point, dict)]
    period = PeriodSpec(start_date=points[0].bucket, end_date=points[-1].bucket)
    series = MetricSeries(
        metric_name=metric_name, unit="value", period=period, points=points,
        source_refs=[trend.get("source_table", "mcp")], definition_version=get_metric_definition(metric_name).version,
    )
    return [item.model_dump(mode="json") for item in detect_series_anomalies(series, min_samples=min(7, max(3, len(points) - 1)))]


def _build_grounded_summary(primary: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
    """仅根据 Data Query Agent 回传的 MCP 证据生成异常结论。

    Root Cause 不再绕过 MCP 直连 repository，避免出现 trace 看不到的第二套
    数据来源，也避免重复查询同一段趋势数据。
    """

    primary_payload = _nested_result(primary)
    metric = _mapping(primary_payload.get("metric_result"))
    lines = [f"## 异常分析：{metric.get('metric_name', '未知指标')}"]
    if metric.get("value") is not None:
        lines.append(
            f"- 当前值（{metric.get('metric_date', '未知日期')}）：**{metric['value']:.2f}**"
        )

    trend_entry = next((item for item in evidence if item.get("purpose") == "trend_baseline"), {})
    trend = _nested_result(_nested_result(trend_entry).get("result")).get("trend_result") or {}
    points = [point for point in trend.get("points", []) if isinstance(point, dict) and point.get("value") is not None]
    if len(points) >= 2:
        previous, current = points[-2], points[-1]
        delta = current["value"] - previous["value"]
        ratio = delta / previous["value"] * 100 if previous["value"] else 0
        direction = "上升" if delta >= 0 else "下降"
        lines.append(
            f"- 趋势基线：相较 {previous.get('bucket')} {direction} "
            f"**{abs(delta):.2f}**（{ratio:+.2f}%）。"
        )
    else:
        lines.append("- 趋势基线：未获得足够的连续数据点，结论仅基于当前值和维度拆解。")

    lines.append(
        f"- 证据来源：{metric.get('source_table', trend.get('source_table', 'MCP query results'))}。"
    )
    lines.append("- 原因假设：维度变化仅表示相关贡献，仍需结合业务事件验证，不能直接视为因果结论。")
    return "\n".join(lines)


LOGGER = logging.getLogger(__name__)


def _find_evidence_id_by_query_id(bundles: list[dict], query_id: str) -> str:
    """按 query_id 精确匹配 MCP EvidenceBundle。"""
    for b in bundles:
        if (b.get("source_type") == "mcp"
            and b.get("content", {}).get("query_id") == query_id):
            return b["evidence_id"]
    return ""


def _get_priority_dimensions(
    metric_name: str,
    question: str = "",
    scope: dict[str, Any] | None = None,
) -> set[str]:
    """Return metric- and scope-supported priority dimensions."""
    if not question and scope is None:
        required = set(HIGH_PRIORITY_DIMENSIONS.get(metric_name, []))
        if not required:
            required = set(list(DIMENSION_LABELS.keys())[:2])
        try:
            return required & set(get_metric_definition(metric_name).allowed_dimensions)
        except Exception:
            return set()
    """获取某指标的高优先级维度，已剔除不在 allowed_dimensions 中的维度。"""
    scope_allowed = set(scope.get("allowed_dimensions") or []) if scope else None
    selected, _ = _dimension_candidates(
        metric_name,
        question,
        scope_allowed=scope_allowed,
    )
    return set(selected)


def _evaluate_step(entry: dict, step: int, bundles: list[dict]) -> dict:
    """评价一条维度贡献证据 — 复用 calculate_change_contribution 统一计算。

    按 query_id 精确关联 MCP EvidenceBundle，不依赖 tool_name 或 question 模糊匹配。
    confidence = abs(top.contribution_ratio) 可能大于 1.0（正负项抵消时，单维度
    delta 可能超过 total_delta），此时仍属 supported。
    """
    purpose = entry.get("purpose", "")
    hypothesis = entry.get("hypothesis", "")
    query_id = entry.get("query_id", "")
    metric_id = entry.get("metric_id", "") or ""

    mcp_eid = _find_evidence_id_by_query_id(bundles, query_id)

    result = _mapping(entry.get("result"))
    payload = _nested_result(result).get("contribution_result") or {}
    if payload.get("status") != "ok":
        return HypothesisStep(
            step=step, hypothesis=hypothesis, purpose=purpose,
            query_id=query_id, metric_id=metric_id,
            evidence_ids=[mcp_eid] if mcp_eid else [],
            verdict=HypothesisVerdict.INCONCLUSIVE,
            rationale=f"{purpose} 查询未能返回有效贡献数据",
            next_action="continue",
        ).model_dump(mode="json")

    metric_name = payload.get("metric_name", "")
    dim = purpose.split(":", 1)[1] if ":" in purpose else ""
    try:
        current_d = _safe_date(payload.get("current_date"))
        baseline_d = _safe_date(payload.get("baseline_date"))
    except (TypeError, ValueError):
        current_d = baseline_d = date.today()

    finding = calculate_change_contribution(
        metric_name=metric_name, dimension=dim,
        current_date=current_d, baseline_date=baseline_d,
        current_items=payload.get("current_items", []),
        baseline_items=payload.get("baseline_items", []),
        overall_total_delta=payload.get("total_delta"),
    )

    top = finding.items[0] if finding.items else None
    ratio = abs(top.contribution_ratio) if top and top.contribution_ratio is not None else 0

    if finding.total_delta == 0:
        verdict, action, rationale = HypothesisVerdict.INCONCLUSIVE, "continue", "变化量为零，无法判断贡献方向"
    elif ratio >= CONFIRMATION_THRESHOLD:
        top_name = top.dimension_value if top else "unknown"
        verdict, action, rationale = HypothesisVerdict.SUPPORTED, "cross_check", \
            f"最大贡献项{top_name}的贡献解释度为{ratio:.1%}"
    elif ratio < MIN_CONTRIBUTION_RATIO:
        verdict, action, rationale = HypothesisVerdict.REFUTED, "continue", \
            f"无显著贡献维度项（最大贡献率 < {MIN_CONTRIBUTION_RATIO:.0%}）"
    else:
        verdict, action, rationale = HypothesisVerdict.INCONCLUSIVE, "continue", \
            f"贡献解释度{ratio:.1%}，不足以确认或拒绝假设"

    return HypothesisStep(
        step=step, hypothesis=hypothesis, purpose=purpose,
        query_id=query_id, metric_id=metric_id,
        evidence_ids=[mcp_eid] if mcp_eid else [],
        derived_evidence_ids=finding.evidence_ids,
        verdict=verdict, confidence=ratio, rationale=rationale,
        next_action=action,
    ).model_dump(mode="json")


def _finalize_root_cause(state: AgentState, stop_reason: str = "dimension_queue_exhausted_or_below_minimum_impact", limitations: list[str] | None = None) -> AgentState:
    """聚合所有收集的证据到 analysis_result，清理循环状态。"""
    primary_result = state.get("primary_tool_result", state.get("tool_result", {}))
    contribution = _contribution_from_evidence(state.get("root_cause_evidence", []))
    evidence = state.get("root_cause_evidence", [])
    summary = _build_grounded_summary(primary_result, evidence)
    if contribution:
        summary += "\n\n### 维度贡献度\n" + "\n".join(
            f"- {item['dimension']}：{item['top_contributor']}（变化 {item['delta_value']:+.2f}，"
            f"贡献率 {item['contribution_ratio'] if item['contribution_ratio'] is not None else 'N/A'}）"
            for item in contribution
        )

    metric_name = _mapping(_nested_result(primary_result).get("metric_result")).get("metric_name", "gmv")
    anomalies = _anomalies_from_evidence(metric_name, evidence)

    state["analysis_result"] = {
        "summary": summary,
        "contribution": contribution,
        "anomalies": anomalies,
        "follow_up_queries": evidence,
        "hypothesis_chain": state.get("hypothesis_chain", []),
        "stop_reason": stop_reason,
        "task_type": "anomaly_analysis",
    }
    # 附加因果限制说明
    causal_disclaimer = "贡献分析不等同于因果证明。维度贡献表明相关关系，不能独立确立因果关系。"
    if limitations is None:
        limitations = []
    if causal_disclaimer not in str(limitations):
        limitations = list(limitations) + [causal_disclaimer]
    requested_dimensions = list(state.get("requested_dimensions") or [])
    contribution_dimensions = {str(item.get("dimension")) for item in contribution}
    missing_dimensions = [
        dimension for dimension in requested_dimensions
        if dimension not in contribution_dimensions
    ]
    unavailable_dimensions = list(state.get("unavailable_requested_dimensions") or [])
    if unavailable_dimensions:
        limitations = list(limitations or []) + [
            "当前锁定范围或指标定义不支持请求的维度："
            + ", ".join(unavailable_dimensions)
            + "；因此不能回答该维度下的具体对象。"
        ]
    elif missing_dimensions and requested_dimensions:
        limitations = list(limitations or []) + [
            "当前数据没有返回请求维度的可验证贡献证据；因此不能可靠定位具体对象。"
        ]
    state["result_completeness"] = {
        "status": "complete" if not missing_dimensions and not unavailable_dimensions else "inconclusive",
        "requested_dimensions": requested_dimensions,
        "satisfied_dimensions": sorted(contribution_dimensions & set(requested_dimensions)),
        "missing_dimensions": missing_dimensions,
        "required_fields": ["dimension", "top_contributor", "current_value", "baseline_value", "delta_value", "evidence_ids"],
    }
    state["analysis_result"]["limitations"] = limitations
    content = {"summary": summary, "contribution": contribution}
    state.setdefault("evidence_bundles", []).append(EvidenceBundle(
        evidence_id=stable_evidence_id("root_cause", "root_cause_analysis", content),
        source_type="root_cause", source_ref="root_cause_analysis", content=content,
    ).model_dump(mode="json"))
    state["pending_query"] = None
    state["next_agent"] = "generate_answer"
    state.setdefault("messages", []).append({
        "from": "root_cause", "to": "critic", "type": "evidence_bundle",
        "content": {
            "follow_up_query_count": len(state.get("root_cause_evidence", [])),
            "contribution": contribution,
        },
    })
    state["current_agent"] = "root_cause"
    state.setdefault("_revision_counts", {})
    state["_revision_counts"]["root_cause"] = state["_revision_counts"].get("root_cause", 0) + 1
    state.setdefault("node_history", []).append({
        "node": "root_cause",
        "summary": (
            f"Completed root-cause analysis with {len(state.get('root_cause_evidence', []))} "
            f"follow-up queries and {len(contribution)} contribution dimensions."
        ),
    })
    return state


def root_cause(state: AgentState) -> AgentState:
    """创建或消费补充证据队列，并在数据齐备后生成根因结论。"""

    question = state.get("user_message", "")
    state.setdefault("requested_dimensions", infer_requested_dimensions(question))

    if len(state.get("query_results", [])) > 1 or state.get("root_cause_tracks") is not None:
        return _batch_root_cause(state)

    queue = state.get("root_cause_queue")

    if queue is None:
        # 首次进入：保留上游的初始指标查询，之后由 data_query 按队列补数。
        state["primary_tool_result"] = state.get("tool_result", {})

        # Phase 7: 短追问时注入历史假设和指标上下文
        historical_hints = []
        hypo_ctx = state.get("historical_hypothesis_context", {})
        if hypo_ctx:
            mids = list(hypo_ctx.keys())[:2]
            if mids:
                historical_hints.append("上轮指标: " + ", ".join(mids))
        claims_ctx = state.get("historical_claims_context", [])
        if claims_ctx:
            vals = [c.get("value", "")[:60] for c in claims_ctx[:2] if c.get("value")]
            if vals:
                historical_hints.append("上轮结论: " + "; ".join(vals))
        historical_hint_str = " | ".join(historical_hints) if historical_hints else ""

        scope = state.get("controlled_analysis_scope") or None
        requested_dimensions = infer_requested_dimensions(question)
        state["requested_dimensions"] = requested_dimensions
        primary_metric = _mapping(_nested_result(state.get("tool_result", {})).get("metric_result")).get("metric_name")
        try:
            metric_allowed = set(get_metric_definition(str(primary_metric)).allowed_dimensions) if primary_metric else set()
        except Exception:
            metric_allowed = set()
        scope_allowed = set(scope.get("allowed_dimensions") or []) if scope else None
        effective_allowed = metric_allowed if scope_allowed is None else metric_allowed & scope_allowed
        state["unavailable_requested_dimensions"] = [
            dimension for dimension in requested_dimensions if dimension not in effective_allowed
        ]
        queue = _build_follow_up_queue(question, historical_hint_str, scope=scope)
        state["root_cause_queue"] = queue
        state.setdefault("root_cause_evidence", [])
        state["_root_cause_iterations"] = 0
        state["_evaluated_query_ids"] = list()
        state["hypothesis_chain"] = []

    # === 评价所有未处理的新证据（无论队列是否为空）===
    # 必须放在 if queue 外部，确保最后一条 data_query 回写后仍能被评价。
    # 当弹出最后一项时 queue 变空，下轮进入时 if queue 为 False，
    # 但此时 root_cause_evidence 中的最后一条证据尚未进入 hypothesis_chain。
    evaluated_ids = set(state.get("_evaluated_query_ids", []))
    new_entries = [
        e for e in state.get("root_cause_evidence", [])
        if e.get("purpose", "").startswith("dimension_contribution:")
        and e.get("query_id", "") not in evaluated_ids
    ]
    for entry in new_entries:
        step = len(state.get("hypothesis_chain", [])) + 1
        hs = _evaluate_step(entry, step, state.get("evidence_bundles", []))
        state.setdefault("hypothesis_chain", []).append(hs)
        evaluated_ids.add(entry.get("query_id", ""))
    state["_evaluated_query_ids"] = list(evaluated_ids)

    # === 检查停止条件 ===
    chain = state["hypothesis_chain"]
    if chain:
        stop_reason, stop_limitations = None, None
        # metric_result / result 在 not_found、unsupported 或仅趋势路径上是
        # 显式 None（DataQueryAgentResult 的 Optional 字段），不能靠 .get 默认值
        # 兜底——键存在且值为 None 时链式取值会抛 AttributeError。逐层判空。
        tool_inner = (state.get("tool_result") or {}).get("result") or {}
        metric_result = tool_inner.get("metric_result") or {}
        metric_name = (
            metric_result.get("metric_name") or ""
        ) or (state.get("analysis_result") or {}).get("metric_name") or ""

        # 1) supported + 至少1个额外维度交叉检查
        supported_found = any(h["verdict"] == "supported" for h in chain)
        contrib_count = sum(1 for h in chain if h["purpose"].startswith("dimension_contribution:"))
        if supported_found and contrib_count >= 2:
            chain[-1]["next_action"] = "stop"
            stop_reason = "dominant_contributor_supported_after_cross_check"
            stop_limitations = ["假设得到数据支持但未穷尽全部维度；贡献分析不等同于因果证明。"]

        # 2) 连续2个 refuted + 已覆盖全部高优先级维度
        if not stop_reason:
            recent = chain[-2:]
            if len(recent) == 2 and all(h["verdict"] == "refuted" for h in recent):
                required_dims = _get_priority_dimensions(metric_name)
                covered_dims = {
                    h["purpose"].split(":", 1)[1]
                    for h in chain if h["purpose"].startswith("dimension_contribution:")
                }
                if required_dims and required_dims <= covered_dims:
                    chain[-1]["next_action"] = "stop"
                    stop_reason = "no_material_contributor_in_examined_dimensions"
                    stop_limitations = ["已覆盖的高优先级维度未发现显著贡献；贡献分析不等同于因果证明。"]

        if stop_reason:
            return _finalize_root_cause(state, stop_reason=stop_reason, limitations=stop_limitations)

    if queue:
        # 迭代次数上限检查，防止 root_cause → data_query 子循环失控
        loop_count = state.setdefault("_root_cause_iterations", 0)
        if loop_count >= MAX_ROOT_CAUSE_ITERATIONS:
            LOGGER.warning("Root cause iteration limit reached (%d); forcing finalization", MAX_ROOT_CAUSE_ITERATIONS)
            return _finalize_root_cause(state,
                stop_reason="root_cause_iteration_limit",
                limitations=["分析维度验证已覆盖主要维度；为控制查询成本，未继续展开更多维度。"],
            )
        state["_root_cause_iterations"] = loop_count + 1

        pending = queue.pop(0)
        # LangGraph 以节点返回值合并 state；列表原地修改并不构成可靠更新，
        # 因而必须重新写入队列，保证下一轮能看到已出队的计划。
        state["root_cause_queue"] = queue
        state["pending_query"] = pending
        state["next_agent"] = "data_query"
        state.setdefault("handoffs", []).append({
            "from": "root_cause",
            "to": "data_query",
            "reason": f"Need {pending['purpose']} evidence before final attribution.",
        })
        state["current_agent"] = "root_cause"
        state.setdefault("node_history", []).append({
            "node": "root_cause",
            "summary": f"Requested {pending['purpose']}; {len(queue)} follow-up queries remain.",
        })
        return state

    # 末次进入：队列已空，委托 _finalize_root_cause 统一处理
    return _finalize_root_cause(state)

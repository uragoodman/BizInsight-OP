"""Generate Answer 节点 — 根据当前 state 生成最终回答与结构化展示结论。

Target 阶段：不同 Agent 路径把答案放在不同地方：
- `root_cause` / `report` → `analysis_result["summary"]`
- `metric_definition` → `metric_context`
- `data_query`（metric_query）→ `tool_result`（需拼装）

本节点统一从 state 中按优先级提取答案源，**同时**构建
``AnalysisPresentation`` 结构化展示结论供前端渲染卡片。
"""

import hashlib
import json
import math
from datetime import date
from typing import Any

from app.graph.state import AgentState
from app.graph.presentation_status import diagnostic_terminal_presentation_status
from app.agents.contracts import Claim, stable_evidence_id
from app.security import ensure_safe_untrusted_content
from app.metrics.comparison_units import (
    comparison_unit_contract,
    display_unit,
    ensure_comparison_unit_note,
)


def _metric_display_name(metric_name: str) -> str:
    """Resolve user-facing metric text through the Registry only."""

    try:
        from app.metrics.registry import get_registry

        registry = get_registry()
        return registry.get(registry.resolve(metric_name)).display_name
    except Exception:
        # Unknown legacy/storage IDs remain printable, but no second display
        # name table is maintained in this module.
        return metric_name


def _display_unit(unit: str | None) -> str:
    """Render known registered units in Chinese; retain unknown canonical units."""
    return display_unit(unit)


def _decomposition_unit_text(unit: str | None) -> str:
    """Render canonical decomposition units in user-facing Chinese prose."""
    canonical = str(unit or "").strip()
    return "个百分点" if canonical == "percentage_points" else canonical


def _metric_decomposition_text(decomposition: dict | None) -> str:
    if not isinstance(decomposition, dict):
        return ""
    factors = decomposition.get("factors") or []
    if not factors or not decomposition.get("evidence_ids"):
        return ""
    baseline_period = decomposition.get("baseline_period") or {}
    current_period = decomposition.get("current_period") or {}
    baseline_label = f"{baseline_period.get('start_date', '')} 至 {baseline_period.get('end_date', '')}"
    current_label = f"{current_period.get('start_date', '')} 至 {current_period.get('end_date', '')}"
    unit = _decomposition_unit_text(decomposition.get("unit"))
    change_unit = _decomposition_unit_text(decomposition.get("change_unit"))
    factor_text = "；".join(
        f"{item.get('display_name') or item.get('metric_id')}贡献 {float(item['contribution']):+,.2f} {_decomposition_unit_text(item.get('unit'))}"
        for item in factors
    )
    evidence = "、".join(str(item) for item in decomposition["evidence_ids"])
    return (
        f"已验证的注册指标数值分解（{decomposition.get('display_name') or decomposition.get('metric_id')}，"
        f"基线期 {baseline_label}，当前期 {current_label}）：基线值 {float(decomposition['baseline_value']):,.2f} {unit}，"
        f"当前值 {float(decomposition['current_value']):,.2f} {unit}，变化 {float(decomposition['delta_value']):+,.2f} {change_unit}；"
        f"{factor_text}。这是指标数值分解，不证明业务因果，业务原因仍未确认。证据：{evidence}。"
    )


def _registered_decomposition_answer_contract(decomposition: dict) -> dict[str, object]:
    """Build the auditable, deterministic answer contract for adaptive mode.

    This is intentionally derived from the validated presentation card, not
    from an LLM summary.  Adaptive Critic may skip semantic review only when
    the final answer remains byte-for-byte equal to this rendered result.
    """
    ledger = {
        "schema_version": "registered_decomposition_answer_contract_v1",
        "metric_id": decomposition.get("metric_id"),
        "method_id": decomposition.get("method_id"),
        "baseline_period": decomposition.get("baseline_period"),
        "current_period": decomposition.get("current_period"),
        "baseline_value": decomposition.get("baseline_value"),
        "current_value": decomposition.get("current_value"),
        "delta_value": decomposition.get("delta_value"),
        "unit": decomposition.get("unit"),
        "change_unit": decomposition.get("change_unit"),
        "factors": decomposition.get("factors"),
        "evidence_ids": list(decomposition.get("evidence_ids") or []),
        "business_cause_confirmed": False,
    }
    rendered_answer = _metric_decomposition_text(decomposition)
    ledger_json = json.dumps(ledger, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return {
        "schema_version": ledger["schema_version"],
        "ledger": ledger,
        "ledger_sha256": hashlib.sha256(ledger_json.encode("utf-8")).hexdigest(),
        "rendered_answer": rendered_answer,
        "rendered_answer_sha256": hashlib.sha256(rendered_answer.encode("utf-8")).hexdigest(),
    }


def _source_display_name(source: object) -> str:
    """Convert an internal SQL/table expression to a user-facing source label."""
    value = str(source or "").strip()
    if not value:
        return "服务端受控指标查询"
    if value.startswith("(") or value.lower().startswith(("select ", "with ")):
        alias = value.rsplit(")", 1)[-1].strip()
        return f"服务端受控指标查询（{alias}）" if alias and " " not in alias else "服务端受控指标查询"
    return value


# ================================================================
# 结构化展示结论构建
# ================================================================


def _make_scope(
    state: AgentState, metric_decomposition: dict | None = None,
    contribution_cards: list[dict[str, Any]] | None = None,
) -> dict | None:
    scope = state.get("controlled_analysis_scope")
    if scope:
        return {
            "seller_id": scope.get("seller_id"),
            "start_date": str(scope.get("default_start_date", "")),
            "end_date": str(scope.get("default_end_date", "")),
            "granularity": str(scope.get("granularity", "")),
        }

    # A governed platform decomposition may not have explorer context. Its
    # authorized scope and period live on the same validated evidence used to
    # build the card; never infer them from candidate truth or the user text.
    if not isinstance(metric_decomposition, dict):
        cards = contribution_cards or []
        if cards:
            investigation = state.get("investigation_state") or {}
            target = investigation.get("target") or {}
            canonical_scope = target.get("canonical_scope") or {}
            baseline = cards[0].get("baseline_period") or {}
            current = cards[0].get("current_period") or {}
            if canonical_scope and baseline.get("start_date") and current.get("end_date"):
                seller_id = canonical_scope.get("seller_id")
                return {
                    "seller_id": str(seller_id) if seller_id else None,
                    "start_date": str(baseline["start_date"]),
                    "end_date": str(current["end_date"]),
                    "granularity": "day" if (
                        baseline.get("start_date") == baseline.get("end_date")
                        and current.get("start_date") == current.get("end_date")
                    ) else "period",
                }
        return None
    evidence_ids = metric_decomposition.get("evidence_ids") or []
    investigation = state.get("investigation_state") or {}
    target = investigation.get("target") or {}
    canonical_scope = target.get("canonical_scope") or {}
    fingerprint = str(target.get("scope_fingerprint") or "")
    evidence = next((
        item for item in investigation.get("evidence", [])
        if isinstance(item, dict)
        and item.get("evidence_id") in evidence_ids
        and item.get("evidence_type") == "metric_decomposition"
        and item.get("metric_id") == metric_decomposition.get("metric_id")
        and item.get("scope_fingerprint") == fingerprint
    ), None)
    comparison = evidence.get("comparison") or {} if isinstance(evidence, dict) else {}
    baseline = comparison.get("baseline_period") or {}
    current = comparison.get("current_period") or {}
    start_date = str(baseline.get("start_date") or "")
    end_date = str(current.get("end_date") or "")
    if not canonical_scope or not start_date or not end_date:
        return None
    seller_id = canonical_scope.get("seller_id")
    if not seller_id and str(canonical_scope.get("scope_type") or canonical_scope.get("tenant_id") or "").casefold() not in {"platform", ""}:
        return None
    return {
        "seller_id": str(seller_id) if seller_id else None,
        "start_date": start_date,
        "end_date": end_date,
        "granularity": "day" if baseline.get("start_date") == baseline.get("end_date") and current.get("start_date") == current.get("end_date") else "period",
    }


def _find_evidence_id(
    bundles: list[dict], source_type: str, source_ref: str,
) -> str:
    """Find the first evidence_id in existing bundles matching source_type + source_ref.
    Returns empty string if not found — callers should fall back gracefully."""
    for b in bundles:
        if b.get("source_type") == source_type and b.get("source_ref") == source_ref:
            return b["evidence_id"]
    return ""


def _find_evidence_ids(
    bundles: list[dict], source_type: str, source_ref_prefix: str,
) -> list[str]:
    """Find all evidence_ids whose source_ref starts with the given prefix."""
    eids: list[str] = []
    for b in bundles:
        if b.get("source_type") == source_type and str(b.get("source_ref", "")).startswith(source_ref_prefix):
            eids.append(b["evidence_id"])
    return eids


def _find_batch_item_evidence_id(bundles: list[dict], item: dict) -> str:
    """按批量项的真实结果匹配其 MCP 证据，确保每张指标卡可追溯。"""
    item_result = item.get("result")
    item_tool = item.get("tool_name")
    for bundle in bundles:
        if bundle.get("source_type") != "mcp":
            continue
        if item_tool and bundle.get("source_ref") != item_tool:
            continue
        if bundle.get("content", {}).get("result") == item_result:
            return bundle["evidence_id"]
    return ""


def _comparison_unit_contract(metric_name: str) -> tuple[str | None, str | None, str]:
    """Resolve comparison units from registry plus the compare operation contract.

    ``delta_ratio`` is the relative change in percent (the query contract
    computes delta / baseline * 100). Absolute delta follows the metric unit,
    except that a percent-valued metric's absolute delta is in percentage
    points. Unknown/unregistered metrics keep level and delta units null.
    """
    metric_unit, change_unit, relative_unit = comparison_unit_contract(metric_name)
    # Keep the existing compare-card contract for the relative delta when the
    # metric itself is unknown; its formula is defined by the compare API.
    return metric_unit, change_unit, relative_unit or "%"


def _make_facts_from_tool(
    tool_result: dict, bundles: list[dict],
) -> tuple[list[dict], list[str]]:
    """Extract fact cards and their evidence_ids from tool_result."""
    facts: list[dict] = []
    all_eids: list[str] = []
    inner = tool_result.get("result", {})

    # 批量结果包含第一项；不能同时渲染兼容用的顶层结果，否则第一项会重复。
    batch = tool_result.get("batch_results", [])
    metric_items = batch if len(batch) > 1 else [tool_result]
    for item in metric_items:
        item_inner = item.get("result", {}) if isinstance(item, dict) else {}
        metric = item_inner.get("metric_result")
        if not metric:
            continue
        eid = _find_batch_item_evidence_id(bundles, item) if len(batch) > 1 else (
            _find_evidence_id(bundles, "mcp", "query_scoped_metric")
            or _find_evidence_id(bundles, "mcp", "registry_query")
            # Legacy MCP tools keep their concrete names (for example
            # ``query_gmv``).  Their bundle is just as traceable as the newer
            # registry gateway and must not produce an unbound fact card.
            or _find_evidence_id(bundles, "mcp", str(tool_result.get("tool_name") or ""))
        )
        # A typed missing-data response is evidence of unavailability, not a
        # numeric fact with the value "None". Keep its evidence reference for
        # the limitation card, but do not create a metric fact or numeric claim.
        if metric.get("value") is None or metric.get("status") in {"not_found", "insufficient_data"}:
            if eid:
                all_eids.append(eid)
            continue
        facts.append({
            "metric_name": metric.get("metric_name", ""),
            "display_name": _metric_display_name(metric.get("metric_name", "")),
            "value": metric.get("value"),
            "unit": metric.get("unit", ""),
            "date_or_range": (
                f"{metric.get('period_start_date')} 至 {metric.get('period_end_date')}"
                if metric.get("period_start_date") and metric.get("period_end_date")
                else metric.get("metric_date", "")
            ),
            "evidence_ids": [eid] if eid else [],
        })
        if eid:
            all_eids.append(eid)

    # Explorer result
    explorer = inner.get("explorer_result")
    if explorer:
        summary = explorer.get("summary", {})
        context = explorer.get("context", {})
        # source_ref 固定记录数据来源；seller_id 位于稳定 ID 的哈希输入中，
        # 不能作为 source_ref 查询条件。
        eid = _find_evidence_id(bundles, "explorer", "data-explorer")
        facts.append({
            "metric_name": summary.get("metric_name", ""),
            "display_name": _metric_display_name(summary.get("metric_name", "")),
            "value": summary.get("value"),
            "unit": summary.get("unit", ""),
            "date_or_range": f"{context.get('start_date')} 至 {context.get('end_date')}",
            "evidence_ids": [eid] if eid else [],
        })
        if eid:
            all_eids.append(eid)

    # Trend
    trend = inner.get("trend_result")
    if trend and not facts:
        points = trend.get("points", [])
        last_point = points[-1] if points and isinstance(points[-1], dict) else {}
        value = last_point.get("value")
        eid = _find_batch_item_evidence_id(bundles, tool_result)
        if not eid:
            eid = _find_evidence_id(bundles, "mcp", "query_scoped_trend") or _find_evidence_id(bundles, "mcp", "query_metric_trend")
        facts.append({
            "metric_name": trend.get("metric_name", ""),
            "display_name": _metric_display_name(trend.get("metric_name", "")),
            "value": value,
            "unit": trend.get("unit", ""),
            # A trend fact is the last returned bucket, not an aggregate over
            # the entire selected range. Keep the label anchored to that bucket.
            "date_or_range": str(last_point.get("bucket") or ""),
            "evidence_ids": [eid] if eid else [],
        })
        if eid:
            all_eids.append(eid)

    return facts, all_eids


def _daily_trend_sparsity_limitation(trend_result: object) -> str | None:
    """Describe unreturned daily buckets without treating them as zero values."""
    if not isinstance(trend_result, dict) or trend_result.get("granularity") != "day":
        return None
    try:
        start = date.fromisoformat(str(trend_result.get("start_date") or ""))
        end = date.fromisoformat(str(trend_result.get("end_date") or ""))
    except ValueError:
        return None
    total_days = (end - start).days + 1
    if total_days <= 0:
        return None
    points = trend_result.get("points")
    if not isinstance(points, list):
        return None
    observed_days = {
        bucket
        for point in points
        if isinstance(point, dict)
        and isinstance(point.get("value"), (int, float))
        and not isinstance(point.get("value"), bool)
        and math.isfinite(float(point["value"]))
        and isinstance((bucket := point.get("bucket")), str)
        and bucket[:10] >= start.isoformat()
        and bucket[:10] <= end.isoformat()
    }
    if len(observed_days) >= total_days:
        return None
    return (
        f"该日趋势在所选 {total_days} 个自然日中仅返回 {len(observed_days)} 个有数值的日期；"
        "未返回的日期不能当作 0，因此这些点不足以说明连续的每日变化或业务因果。"
    )


def _scope_sidecars_enabled(state: AgentState, scope_active: bool) -> bool:
    """Keep automatically queried Explorer cards off narrow, contract-bound requests."""
    if not scope_active:
        return False
    if state.get("task_type") == "anomaly_analysis":
        return False
    compilation = state.get("context_compilation") or {}
    request_contract = compilation.get("request_contract") or {}
    operation_intent = str(
        request_contract.get("operation_intent")
        or (compilation.get("operation_intent") or {}).get("kind")
        or ""
    )
    return operation_intent not in {
        "ordinary_metric_query",
        "business_causal_investigation",
        "dimensional_statistical_contribution",
        "registered_numeric_decomposition",
    }


def _has_incomplete_comparison(tool_result: dict) -> bool:
    """Whether any returned compare payload lacks either operand."""
    batch = tool_result.get("batch_results", [])
    items = batch if len(batch) > 1 else [tool_result]
    for item in items:
        inner = item.get("result", {}) if isinstance(item, dict) else {}
        compare = inner.get("compare_result") if isinstance(inner, dict) else None
        if compare and (
            compare.get("status") != "ok"
            or compare.get("left_value") is None
            or compare.get("right_value") is None
        ):
            return True
    return False


def _make_changes_from_tool(
    tool_result: dict, bundles: list[dict],
) -> tuple[list[dict], list[str]]:
    """Extract change cards from compare_result."""
    changes: list[dict] = []
    all_eids: list[str] = []
    batch = tool_result.get("batch_results", [])
    compare_items = batch if len(batch) > 1 else [tool_result]
    for item in compare_items:
        item_inner = item.get("result", {}) if isinstance(item, dict) else {}
        cr = item_inner.get("compare_result")
        if (
            not cr
            or cr.get("status") != "ok"
            or cr.get("current_value", cr.get("left_value")) is None
            or cr.get("baseline_value", cr.get("right_value")) is None
        ):
            continue
        eid = _find_batch_item_evidence_id(bundles, item) if len(batch) > 1 else (
            _find_evidence_id(bundles, "mcp", "query_scoped_metric_compare")
            or _find_evidence_id(bundles, "mcp", "compare")
            or _find_evidence_id(bundles, "mcp", str(tool_result.get("tool_name") or ""))
        )
        current_date = cr.get("current_date") or cr.get("left_date", "")
        baseline_date = cr.get("baseline_date") or cr.get("right_date", "")
        current_label = current_date
        if cr.get("left_end_date") and cr.get("left_end_date") != current_date:
            current_label = f"{current_date} 至 {cr.get('left_end_date')}"
        baseline_label = baseline_date
        if cr.get("right_end_date") and cr.get("right_end_date") != baseline_date:
            baseline_label = f"{baseline_date} 至 {cr.get('right_end_date')}"
        metric_name = str(cr.get("metric_name") or "")
        metric_unit, change_unit, delta_ratio_unit = _comparison_unit_contract(metric_name)
        changes.append({
            "metric_name": metric_name,
            "display_name": _metric_display_name(metric_name),
            "current_value": cr.get("current_value", cr.get("left_value")),
            "baseline_value": cr.get("baseline_value", cr.get("right_value")),
            "delta_value": cr.get("delta_value"),
            "unit": metric_unit,
            "change_unit": change_unit,
            "delta_ratio": cr.get("delta_ratio"),
            "delta_ratio_unit": delta_ratio_unit,
            "current_period": current_label,
            "baseline_period": baseline_label,
            "evidence_ids": [eid] if eid else [],
        })
        if eid:
            all_eids.append(eid)
    return changes, all_eids


def _contribution_field_evidence_ids(
    contribution: dict[str, Any], item: dict[str, Any] | None,
    bundles: list[dict], tool_calls: list[dict], evidence_ids: list[str],
) -> dict[str, list[str]]:
    """Bind displayed numeric atoms to exact request-correlated contribution evidence."""
    metric_id = str(contribution.get("metric_name") or "")
    dimension = str(contribution.get("dimension") or contribution.get("dim_name") or "")
    calls = {
        str(call.get("request_id")): call for call in tool_calls
        if isinstance(call, dict) and call.get("request_id")
    }
    sources: list[tuple[str, dict[str, Any], int]] = []
    for evidence_id in evidence_ids:
        bundle = next((row for row in bundles if isinstance(row, dict) and str(row.get("evidence_id") or "") == str(evidence_id)), None)
        if not bundle or bundle.get("trusted", True) is not True:
            continue
        content = bundle.get("content") if isinstance(bundle.get("content"), dict) else {}
        request_id = str(content.get("request_id") or "")
        call = calls.get(request_id)
        if not request_id or call is None or str(call.get("evidence_id") or "") != str(evidence_id):
            continue
        args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        call_metric = str(call.get("planned_metric_id") or args.get("metric_name") or "")
        if call.get("status") != "ok" or call_metric != metric_id or str(args.get("dimension") or "") != dimension:
            continue
        result = content.get("result") if isinstance(content.get("result"), dict) else {}
        source = result.get("contribution_result") if isinstance(result.get("contribution_result"), dict) else {}
        if result.get("status") != "ok" or source.get("metric_name") != metric_id or source.get("dimension") != dimension:
            continue
        # Use the server-parsed plan's effective limit when the normalized
        # diagnostic invocation does not repeat this optional tool parameter.
        # This is needed to verify an "Other" residual; row counts alone are
        # not sufficient evidence that the contribution tool returned a full
        # top-N result.
        parsed_plan = result.get("parsed") if isinstance(result.get("parsed"), dict) else {}
        raw_limit = args.get("limit", parsed_plan.get("limit"))
        try:
            effective_limit = int(raw_limit)
        except (TypeError, ValueError):
            effective_limit = 0
        sources.append((str(evidence_id), source, effective_limit))

    def same_number(actual: Any, expected: Any) -> bool:
        if isinstance(actual, bool) or isinstance(expected, bool) or not isinstance(actual, (int, float)) or not isinstance(expected, (int, float)):
            return False
        return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-6)

    bindings: dict[str, list[str]] = {}
    for field, source_field in (("baseline_value", "baseline_total"), ("current_value", "current_total"), ("total_delta", "total_delta")):
        ids = [eid for eid, source, _limit in sources if same_number(contribution.get(field), source.get(source_field))]
        if ids:
            bindings[field] = ids
    if item is not None:
        key = str(item.get("dimension_value") or "")
        rows: list[tuple[str, Any, Any, Any]] = []
        for eid, source, effective_limit in sources:
            current_items = [row for row in source.get("current_items", []) if isinstance(row, dict)]
            baseline_items = [row for row in source.get("baseline_items", []) if isinstance(row, dict)]
            current = next((row.get("value") for row in current_items if str(row.get("dimension_value")) == key), None)
            baseline = next((row.get("value") for row in baseline_items if str(row.get("dimension_value")) == key), None)
            # Contribution projection groups omitted dimensions into a visible
            # "Other" row. Bind it only when both source totals and complete
            # returned group lists reconcile to that exact residual.
            if current is None and baseline is None and key.casefold() == "other":
                source_current_total = source.get("current_total")
                source_baseline_total = source.get("baseline_total")
                current_limit = len(current_items)
                baseline_limit = len(baseline_items)
                if (
                    effective_limit > 0
                    and current_limit == effective_limit
                    and baseline_limit == effective_limit
                    and isinstance(source_current_total, (int, float))
                    and isinstance(source_baseline_total, (int, float))
                ):
                    current = float(source_current_total) - sum(float(row["value"]) for row in current_items if isinstance(row.get("value"), (int, float)))
                    baseline = float(source_baseline_total) - sum(float(row["value"]) for row in baseline_items if isinstance(row.get("value"), (int, float)))
            if current is not None and baseline is not None:
                rows.append((eid, baseline, current, source.get("total_delta")))
        for field, value_index in (("baseline_value", 0), ("current_value", 1)):
            ids = [eid for eid, baseline, current, _ in rows if same_number(item.get(field), (baseline, current)[value_index])]
            if ids:
                bindings[field] = ids
        ids = [eid for eid, baseline, current, _ in rows if same_number(item.get("delta_value"), float(current) - float(baseline))]
        if ids:
            bindings["delta_value"] = ids
        ids = [
            eid for eid, baseline, current, total_delta in rows
            if isinstance(total_delta, (int, float)) and float(total_delta) != 0
            and same_number(item.get("contribution_ratio"), (float(current) - float(baseline)) / float(total_delta))
        ]
        if ids:
            bindings["contribution_ratio"] = ids
    return bindings


def _make_contributions(
    analysis_result: dict, bundles: list[dict], tool_calls: list[dict] | None = None,
) -> tuple[list[dict], list[str]]:
    """Extract contribution cards from analysis_result.

    Root cause _contribution_from_evidence outputs all paired groups, period
    dates, metric definition metadata, and the additive reconciliation result.
    """
    contributions: list[dict] = []
    all_eids: list[str] = []
    raw = analysis_result.get("contribution", [])
    if not raw:
        return contributions, all_eids
    for contrib in raw:
        # Use actual root_cause evidence IDs from state.evidence_bundles
        rc_eids = _find_evidence_ids(bundles, "root_cause", "root_cause:")
        if not rc_eids:
            rc_eids = _find_evidence_ids(bundles, "root_cause", "root_cause_analysis")
        contribution_eids = contrib.get("evidence_ids", [])
        # Drop IDs not in evidence_bundles; fall back to root_cause bundle IDs
        valid_eids = [eid for eid in contribution_eids if any(b["evidence_id"] == eid for b in bundles)]
        final_eids = valid_eids or rc_eids
        tool_calls = tool_calls or []
        card_identity = {
            **contrib,
            "dimension": contrib.get("dimension", contrib.get("dim_name", "")),
        }
        items = []
        for item in contrib.get("items", []):
            normalized = {
                "dimension_value": item.get("dimension_value", ""),
                "current_value": item.get("current_value"),
                "baseline_value": item.get("baseline_value"),
                "delta_value": item.get("delta_value", 0),
                "contribution_ratio": item.get("contribution_ratio"),
                "components": item.get("components") or [],
            }
            derived_bindings = _contribution_field_evidence_ids(
                card_identity, normalized, bundles, tool_calls, final_eids,
            )
            supplied_bindings = item.get("field_evidence_ids") or {}
            if isinstance(supplied_bindings, dict):
                for field, ids in supplied_bindings.items():
                    trusted_ids = [str(eid) for eid in ids or [] if str(eid) in final_eids]
                    if trusted_ids:
                        derived_bindings[field] = sorted(set(derived_bindings.get(field, []) + trusted_ids))
            normalized["field_evidence_ids"] = derived_bindings
            items.append(normalized)
        dim_val = contrib.get("top_contributor", "")
        delta = contrib.get("delta_value")
        ratio = contrib.get("contribution_ratio")
        if not items and dim_val and delta is not None:
            normalized = {
                "dimension_value": dim_val,
                "current_value": contrib.get("current_value"),
                "baseline_value": contrib.get("baseline_value"),
                "delta_value": delta,
                "contribution_ratio": ratio,
                "components": [],
            }
            normalized["field_evidence_ids"] = _contribution_field_evidence_ids(
                card_identity, normalized, bundles, tool_calls, final_eids,
            )
            items.append(normalized)
        card = {
            "dimension": contrib.get("dimension", contrib.get("dim_name", "")),
            "metric_name": contrib.get("metric_name", ""),
            "metric_definition": contrib.get("metric_definition", ""),
            "definition_version": contrib.get("definition_version", ""),
            "unit": contrib.get("unit", ""),
            "contribution_unit": contrib.get("contribution_unit") or contrib.get("unit", ""),
            "aggregation": contrib.get("aggregation", ""),
            "baseline_value": contrib.get("baseline_value"),
            "current_value": contrib.get("current_value"),
            "current_date": contrib.get("current_date", ""),
            "baseline_date": contrib.get("baseline_date", ""),
            "total_delta": contrib.get("total_delta"),
            "current_period": contrib.get("current_period"),
            "baseline_period": contrib.get("baseline_period"),
            "residual_value": contrib.get("residual_value"),
            "residual_type": contrib.get("residual_type", "unknown"),
            "residual_explainable": contrib.get("residual_explainable", False),
            "residual_tolerance": contrib.get("residual_tolerance"),
            "residual_note": contrib.get("residual_note", ""),
            "analysis_nature": "descriptive_contribution",
            "method": contrib.get("method") or "additive_change_v1",
            "items": items,
            "evidence_ids": final_eids,
        }
        card_bindings = _contribution_field_evidence_ids(
            card_identity, None, bundles, tool_calls, final_eids,
        )
        supplied_card_bindings = contrib.get("field_evidence_ids") or {}
        if isinstance(supplied_card_bindings, dict):
            for field, ids in supplied_card_bindings.items():
                trusted_ids = [str(eid) for eid in ids or [] if str(eid) in final_eids]
                if trusted_ids:
                    card_bindings[field] = sorted(set(card_bindings.get(field, []) + trusted_ids))
        card["field_evidence_ids"] = card_bindings
        contributions.append(card)
        all_eids.extend(final_eids)
    return contributions, all_eids


def _validated_contribution_cards(state: AgentState) -> list[dict[str, Any]]:
    """Project only registered contribution evidence validated for this request."""
    contract = (((state.get("context_compilation") or {}).get("request_contract")) or {})
    if (
        contract.get("operation_intent") == "dimensional_statistical_contribution"
        and not state.get("diagnostic_plan")
    ):
        return _validated_request_contribution_cards(state)
    try:
        from app.graph.nodes.diagnostic import validated_dimension_contribution_cards
        cards = validated_dimension_contribution_cards(state)
        # Additive diagnostic contributions inherit the metric unit. Keep the
        # card and its answer projection on the same explicit unit contract.
        for card in cards:
            if isinstance(card, dict):
                card.setdefault("contribution_unit", card.get("unit"))
                for item in card.get("items") or []:
                    if isinstance(item, dict):
                        item.setdefault("components", [])
        return cards
    except Exception:  # noqa: BLE001 - failure to validate must not create a card
        return []


def _validated_request_contribution_cards(state: AgentState) -> list[dict[str, Any]]:
    """Build answer cards from the exact authorized result bound to the request contract.

    The Critic owns the shared contract/tool/evidence validation. Asking for its
    pre-output-validation view avoids a second, weaker date/metric parser here;
    the final Critic pass still requires this card's evidence to be cited.
    """
    try:
        from app.graph.nodes.critic import _review_request_contract
        review = _review_request_contract(
            state, require_output_binding=False, record_observation=False,
        )
        if review.get("status") != "passed":
            return []
        evidence_ids = set(map(str, review.get("contribution_evidence_ids") or []))
        if not evidence_ids:
            return []

        from app.analysis.attribution_methods import (
            AttributionDataError,
            symmetric_mix_within_attribution,
        )
        from app.metrics.registry import get_registry

        contract = (((state.get("context_compilation") or {}).get("request_contract")) or {})
        comparison = contract.get("comparison") or {}
        metric_id = str(contract.get("metric_id") or "")
        dimension = str(contract.get("dimension") or "")
        registry = get_registry()
        definition = registry.get(metric_id)
        scope = "seller" if (state.get("controlled_analysis_scope") or {}).get("seller_id") else "platform"
        capability = definition.attribution.for_scope(scope)
        period_aggregation = definition.period_aggregation
        period_method = None
        if comparison.get("kind") == "period_pair":
            from app.analysis.contribution_capabilities import registered_period_contribution_method

            period_method = registered_period_contribution_method(
                definition,
                metric_id=metric_id,
                dimension=dimension,
                scope=scope,
                granularity=str(comparison.get("granularity") or "") or None,
            )
            if period_method is None:
                _record_contribution_projection_failure(
                    state,
                    "CAPABILITY_GAP",
                    "该指标、维度和比较期间组合未注册可验证的统计贡献方法。",
                )
                return []
        proportional_overlap = (
            "proportional_overlap_v1" in capability.method_ids
            and dimension in capability.dimensions
            and metric_id in {"order_count", "customer_count", "repeat_customer_count"}
            and dimension in {"product_category", "seller", "payment_type"}
            and comparison.get("kind") in {"day_pair", "period_pair"}
        )
        if proportional_overlap:
            return _validated_proportional_request_cards(
                state, contract, evidence_ids, definition,
            )
        # Period additive projection follows the same centralized capability
        # matrix as the planner; a registry method declaration alone is not
        # enough to open a new period path.
        period_additive_enabled = (
            comparison.get("kind") == "period_pair"
            and period_method == "additive_change_v1"
            and "additive_change_v1" in capability.method_ids
            and dimension in capability.dimensions
            and scope in definition.implementations
        )
        if (
            (
                comparison.get("kind") == "day_pair"
                or period_additive_enabled
            )
            and "additive_change_v1" in capability.method_ids
            and dimension in capability.dimensions
            and scope in definition.implementations
        ):
            return _validated_additive_request_cards(state, contract, evidence_ids, definition)
        if (
            "symmetric_mix_within_v1" not in capability.method_ids
            or period_aggregation is None
            or dimension not in capability.dimensions
            or scope not in definition.implementations
        ):
            _record_contribution_projection_failure(
                state, "CAPABILITY_GAP", "注册表没有声明该范围、维度和 symmetric_mix_within_v1 期间分解能力。",
            )
            return []

        cards: list[dict[str, Any]] = []
        bundles = state.get("evidence_bundles") or []
        for bundle in bundles:
            if not isinstance(bundle, dict) or bundle.get("trusted") is not True:
                continue
            evidence_id = str(bundle.get("evidence_id") or "")
            if evidence_id not in evidence_ids:
                continue
            content = bundle.get("content") or {}
            result = content.get("result") or {}
            contribution = result.get("contribution_result") or {}
            parsed = result.get("parsed") or {}
            if result.get("status") != "ok" or not contribution:
                continue
            if (
                contribution.get("metric_name") != metric_id
                or contribution.get("dimension") != dimension
                or parsed.get("query_type") != "metric_contribution"
                or parsed.get("metric_name") != metric_id
                or parsed.get("dimension") != dimension
            ):
                continue
            try:
                if comparison.get("kind") == "day_pair":
                    baseline_period = {
                        "start_date": comparison["baseline_date"],
                        "end_date": comparison["baseline_date"],
                    }
                    current_period = {
                        "start_date": comparison["current_date"],
                        "end_date": comparison["current_date"],
                    }
                elif comparison.get("kind") == "period_pair":
                    baseline_period = dict(comparison["baseline"])
                    current_period = dict(comparison["current"])
                else:
                    _record_contribution_projection_failure(
                        state, "DATA_INSUFFICIENT", "请求契约没有可用于贡献计算的完整基线／当前比较期。",
                    )
                    continue

                # The registered period aggregation is the sole source for the
                # sufficient-statistic roles and scale. Query-returned labels
                # are checked against it; they do not define another formula.
                numerator_key = period_aggregation.numerator_metric
                denominator_key = period_aggregation.denominator_metric
                if (
                    contribution.get("numerator_metric") != numerator_key
                    or contribution.get("denominator_metric") != denominator_key
                    or contribution.get("current_period") != current_period
                    or contribution.get("baseline_period") != baseline_period
                ):
                    _record_contribution_projection_failure(
                        state, "INVALID_DATA", "贡献结果的期间或分子／分母指标与注册契约不一致。",
                    )
                    continue
                if contribution.get("complete") is not True:
                    _record_contribution_projection_failure(
                        state, "DATA_INSUFFICIENT", "工具结果未声明完整分组覆盖，不能生成整体贡献卡。",
                    )
                    continue

                def checked_groups(value: object, label: str) -> dict[str, dict[str, float]]:
                    if not isinstance(value, dict) or not value:
                        raise KeyError(f"{label}_groups missing or empty")
                    checked: dict[str, dict[str, float]] = {}
                    for group, facts in value.items():
                        if not isinstance(group, str) or not group or not isinstance(facts, dict):
                            raise ValueError(f"{label}_groups malformed")
                        pair: dict[str, float] = {}
                        for field in ("numerator", "denominator"):
                            number = facts.get(field)
                            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number)):
                                raise KeyError(f"{label}_groups.{group}.{field} missing or invalid")
                            pair[field] = float(number)
                        checked[group] = pair
                    return checked

                current_groups = checked_groups(contribution.get("current_groups"), "current")
                baseline_groups = checked_groups(contribution.get("baseline_groups"), "baseline")

                def checked_total(field: str) -> float:
                    value = contribution.get(field)
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                        raise KeyError(f"{field} missing or invalid")
                    return float(value)

                current_numerator = checked_total("current_numerator_total")
                current_denominator = checked_total("current_denominator_total")
                baseline_numerator = checked_total("baseline_numerator_total")
                baseline_denominator = checked_total("baseline_denominator_total")

                def totals_match(groups: dict[str, dict[str, float]], numerator: float, denominator: float) -> bool:
                    sum_numerator = sum(item["numerator"] for item in groups.values())
                    sum_denominator = sum(item["denominator"] for item in groups.values())
                    return (
                        math.isclose(sum_numerator, numerator, rel_tol=1e-9, abs_tol=1e-6)
                        and math.isclose(sum_denominator, denominator, rel_tol=1e-9, abs_tol=1e-6)
                    )

                if not totals_match(current_groups, current_numerator, current_denominator) or not totals_match(
                    baseline_groups, baseline_numerator, baseline_denominator,
                ):
                    _record_contribution_projection_failure(
                        state, "INVALID_DATA", "分组分子／分母合计与同周期总体合计不一致。",
                    )
                    continue

                metric_unit = str(definition.unit or "")
                contribution_unit = "percentage_points" if metric_unit == "%" else metric_unit
                finding = symmetric_mix_within_attribution(
                    metric_id=metric_id,
                    baseline_groups=baseline_groups,
                    current_groups=current_groups,
                    numerator_key="numerator",
                    denominator_key="denominator",
                    multiplier=float(period_aggregation.multiplier),
                    unit=contribution_unit,
                    evidence_ids=[evidence_id],
                )

                components_by_group: dict[str, list[dict[str, Any]]] = {}
                for component in finding.display_components:
                    group, _, _effect = component.component_id.rpartition(":")
                    if not group:
                        raise ValueError("registered attribution component has no group key")
                    components_by_group.setdefault(group, []).append({
                        "component_type": component.component_type,
                        "value": float(component.value),
                        "unit": contribution_unit,
                        "evidence_ids": [evidence_id],
                    })

                rows: list[dict[str, Any]] = []
                for group in sorted(set(baseline_groups) | set(current_groups)):
                    base = baseline_groups.get(group)
                    current = current_groups.get(group)
                    base_value = (
                        float(period_aggregation.multiplier) * base["numerator"] / base["denominator"]
                        if base and base["denominator"] > 0 else None
                    )
                    current_value = (
                        float(period_aggregation.multiplier) * current["numerator"] / current["denominator"]
                        if current and current["denominator"] > 0 else None
                    )
                    effects = components_by_group.get(group, [])
                    group_delta = sum(float(item["value"]) for item in effects)
                    rows.append({
                        "dimension_value": group,
                        "baseline_value": base_value,
                        "current_value": current_value,
                        "delta_value": group_delta,
                        "contribution_ratio": group_delta / finding.total_delta if finding.total_delta else None,
                        "components": effects,
                        "field_evidence_ids": {
                            field: [evidence_id] for field in (
                                "baseline_value", "current_value", "delta_value", "contribution_ratio",
                            )
                        },
                    })
                shown_delta = sum(float(item["delta_value"]) for item in rows)
                if not math.isclose(
                    shown_delta, finding.total_delta, rel_tol=1e-9,
                    abs_tol=max(1e-8, float(finding.residual_tolerance)),
                ):
                    _record_contribution_projection_failure(
                        state, "INVALID_DATA", "展示分组贡献与注册方法计算的整体变化未能对账。",
                    )
                    continue

                if (
                    not math.isclose(
                        current_numerator / current_denominator * float(period_aggregation.multiplier),
                        finding.current_value, rel_tol=1e-9, abs_tol=1e-8,
                    )
                    or not math.isclose(
                        baseline_numerator / baseline_denominator * float(period_aggregation.multiplier),
                        finding.baseline_value, rel_tol=1e-9, abs_tol=1e-8,
                    )
                ):
                    raise ValueError("registered aggregate values do not reconcile")
                cards.append({
                    "dimension": dimension,
                    "metric_name": metric_id,
                    "unit": metric_unit,
                    "contribution_unit": contribution_unit,
                    "aggregation": "registered_symmetric_mix_within_contribution",
                    "baseline_value": finding.baseline_value,
                    "current_value": finding.current_value,
                    "baseline_date": baseline_period["end_date"],
                    "current_date": current_period["end_date"],
                    "baseline_period": dict(baseline_period),
                    "current_period": dict(current_period),
                    "total_delta": finding.total_delta,
                    "residual_value": finding.residual_value,
                    "residual_type": finding.residual_type,
                    "residual_explainable": finding.residual_explainable,
                    "residual_tolerance": finding.residual_tolerance,
                    "residual_note": finding.residual_note,
                    "analysis_nature": "descriptive_contribution",
                    "method": finding.method_id,
                    "items": rows,
                    "evidence_ids": [evidence_id],
                    "field_evidence_ids": {
                        field: [evidence_id] for field in (
                            "baseline_value", "current_value", "total_delta",
                        )
                    },
                })
            except AttributionDataError as exc:
                reason = str(exc)
                code = "DATA_INSUFFICIENT" if "zero effective denominator" in reason else "INVALID_DATA"
                _record_contribution_projection_failure(state, code, reason)
                continue
            except KeyError as exc:
                _record_contribution_projection_failure(state, "DATA_INSUFFICIENT", str(exc))
                continue
            except (TypeError, ValueError, OverflowError) as exc:
                _record_contribution_projection_failure(state, "INVALID_DATA", str(exc))
                continue
        if not cards and not state.get("contribution_projection"):
            _record_contribution_projection_failure(
                state, "DATA_INSUFFICIENT", "没有找到与已审核请求契约绑定的完整贡献事实。",
            )
        return cards
    except Exception as exc:  # noqa: BLE001 - unexpected projection errors stop the answer path
        _record_contribution_projection_failure(
            state, "EXECUTION_FAILED", f"贡献投影执行失败（{type(exc).__name__}）。",
        )
        return []


def _validated_additive_request_cards(state, contract, evidence_ids, definition):
    """Project registered additive facts only after the shared request gate passed."""
    from datetime import date
    from app.analysis.contribution import calculate_change_contribution

    metric, dimension = contract.get("metric_id"), contract.get("dimension")
    comparison = contract.get("comparison") or {}
    comparison_kind = comparison.get("kind")
    period_mode = comparison_kind == "period_pair"
    if period_mode:
        from app.analysis.contribution_capabilities import registered_period_contribution_method

        scope = "seller" if (state.get("controlled_analysis_scope") or {}).get("seller_id") else "platform"
        period_method = registered_period_contribution_method(
            definition,
            metric_id=str(metric or ""),
            dimension=str(dimension or ""),
            scope=scope,
            granularity=str(comparison.get("granularity") or "") or None,
        )
        if period_method != "additive_change_v1":
            _record_contribution_projection_failure(
                state, "CAPABILITY_GAP", "该指标和范围没有注册 period additive projection 能力。",
            )
            return []
        period_roles = {}
        for role in ("baseline", "current"):
            raw_period = comparison.get(role)
            if not isinstance(raw_period, dict):
                _record_contribution_projection_failure(
                    state, "INVALID_DATA", "期间贡献 contract 缺少完整期间。",
                )
                return []
            try:
                start = date.fromisoformat(str(raw_period.get("start_date") or ""))
                end = date.fromisoformat(str(raw_period.get("end_date") or ""))
            except ValueError:
                _record_contribution_projection_failure(
                    state, "INVALID_DATA", "期间贡献 contract 包含无效日期。",
                )
                return []
            from calendar import monthrange

            if (
                start.day != 1
                or (start.year, start.month) != (end.year, end.month)
                or end.day != monthrange(start.year, start.month)[1]
            ):
                _record_contribution_projection_failure(
                    state, "INVALID_DATA", "当前 additive period projection 只接受完整自然月。",
                )
                return []
            period_roles[role] = {
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "granularity": "month",
            }
    elif comparison_kind != "day_pair":
        _record_contribution_projection_failure(
            state, "CAPABILITY_GAP", "加法贡献只接受已注册的日期对或自然月期间对。",
        )
        return []

    candidates = []
    for bundle in state.get("evidence_bundles") or []:
        if not isinstance(bundle, dict) or bundle.get("trusted") is not True or str(bundle.get("evidence_id") or "") not in evidence_ids:
            continue
        result = ((bundle.get("content") or {}).get("result") or {})
        payload, parsed = result.get("contribution_result") or {}, result.get("parsed") or {}
        if result.get("status") != "ok" or payload.get("status") != "ok":
            continue
        if (
            payload.get("metric_name") != metric
            or payload.get("dimension") != dimension
            or parsed.get("metric_name") != metric
            or parsed.get("dimension") != dimension
            or parsed.get("query_type") != "metric_contribution"
        ):
            continue
        if period_mode:
            parsed_periods = {
                "baseline": {
                    "start_date": parsed.get("baseline_start"),
                    "end_date": parsed.get("baseline_end"),
                },
                "current": {
                    "start_date": parsed.get("current_start"),
                    "end_date": parsed.get("current_end"),
                },
            }
            payload_periods = {
                "baseline": payload.get("baseline_period"),
                "current": payload.get("current_period"),
            }
            if any(
                parsed_periods[role] != {
                    "start_date": comparison[role].get("start_date"),
                    "end_date": comparison[role].get("end_date"),
                }
                for role in ("baseline", "current")
            ):
                continue
            if any(
                not isinstance(payload_periods[role], dict)
                or payload_periods[role].get("start_date") != period_roles[role]["start_date"]
                or payload_periods[role].get("end_date") != period_roles[role]["end_date"]
                or (
                    payload_periods[role].get("granularity") is not None
                    and payload_periods[role].get("granularity") != "month"
                )
                for role in ("baseline", "current")
            ):
                continue
            if payload.get("method") not in (None, "additive_change_v1"):
                continue
        elif (
            payload.get("baseline_date") != comparison.get("baseline_date")
            or payload.get("current_date") != comparison.get("current_date")
            or parsed.get("baseline_date") != comparison.get("baseline_date")
            or parsed.get("current_date") != comparison.get("current_date")
        ):
            continue
        candidates.append((str(bundle["evidence_id"]), payload))
    if len(candidates) != 1:
        _record_contribution_projection_failure(state, "DATA_INSUFFICIENT", "缺少唯一且匹配请求的已授权加总贡献证据。")
        return []
    evidence_id, payload = candidates[0]
    try:
        values = [payload.get(key) for key in ("baseline_total", "current_total", "total_delta")]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
            raise ValueError("invalid totals")
        for key in ("baseline_items", "current_items"):
            items = payload.get(key)
            if not isinstance(items, list) or not items:
                raise ValueError("missing groups")
            for item in items:
                if not isinstance(item, dict) or isinstance(item.get("value"), bool) or not isinstance(item.get("value"), (int, float)) or not math.isfinite(item["value"]):
                    raise ValueError("invalid group value")
        if period_mode:
            baseline_period = period_roles["baseline"]
            current_period = period_roles["current"]
            baseline_date = date.fromisoformat(baseline_period["end_date"])
            current_date = date.fromisoformat(current_period["end_date"])
        else:
            baseline_date = date.fromisoformat(comparison["baseline_date"])
            current_date = date.fromisoformat(comparison["current_date"])
        finding = calculate_change_contribution(metric, dimension, current_date, baseline_date,
            payload["current_items"], payload["baseline_items"], overall_total_delta=float(values[2]),
            current_total=float(values[1]), baseline_total=float(values[0]),
            current_period=current_period if period_mode else None,
            baseline_period=baseline_period if period_mode else None)
        if not finding.residual_explainable:
            raise ValueError("unreconciled residual")
        rows = [item.model_dump(mode="json") for item in finding.items]
        for row in rows:
            row["components"] = []
            row["field_evidence_ids"] = {field: [evidence_id] for field in ("baseline_value", "current_value", "delta_value", "contribution_ratio")}
        unit = str(definition.unit or "")
        if not unit:
            raise ValueError("unknown unit")
        return [{"dimension": dimension, "metric_name": metric, "unit": unit,
            "contribution_unit": unit, "aggregation": "registered_additive_contribution",
            "baseline_value": float(values[0]), "current_value": float(values[1]), "total_delta": float(values[2]),
            "baseline_date": baseline_date.isoformat(), "current_date": current_date.isoformat(),
            "baseline_period": (
                baseline_period if period_mode else
                {"start_date": baseline_date.isoformat(), "end_date": baseline_date.isoformat()}
            ),
            "current_period": (
                current_period if period_mode else
                {"start_date": current_date.isoformat(), "end_date": current_date.isoformat()}
            ),
            "residual_value": finding.residual_value, "residual_type": finding.residual_type,
            "residual_explainable": finding.residual_explainable, "residual_tolerance": finding.residual_tolerance,
            "residual_note": finding.residual_note, "analysis_nature": "descriptive_contribution",
            "method": "additive_change_v1", "items": rows, "evidence_ids": [evidence_id],
            "field_evidence_ids": {field: [evidence_id] for field in ("baseline_value", "current_value", "total_delta")}}]
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        _record_contribution_projection_failure(state, "INVALID_DATA", f"加总贡献证据无法对账（{type(exc).__name__}）。")
        return []


def _validated_proportional_request_cards(state, contract, evidence_ids, definition):
    """Project allocated order-count facts without pretending they are raw counts.

    Group values are order-equivalents: an order that touches ``k`` categories
    contributes ``1/k`` to each. Both sides must reconcile with their own
    distinct-order total before any card is produced.
    """
    metric, dimension = contract.get("metric_id"), contract.get("dimension")
    comparison = contract.get("comparison") or {}
    kind = comparison.get("kind")
    candidates = []
    for bundle in state.get("evidence_bundles") or []:
        if not isinstance(bundle, dict) or bundle.get("trusted") is not True:
            continue
        evidence_id = str(bundle.get("evidence_id") or "")
        if evidence_id not in evidence_ids:
            continue
        result = ((bundle.get("content") or {}).get("result") or {})
        payload = result.get("contribution_result") or {}
        parsed = result.get("parsed") or {}
        if result.get("status") != "ok" or payload.get("status") != "ok":
            continue
        if (
            payload.get("metric_name") != metric
            or payload.get("dimension") != dimension
            or parsed.get("metric_name") != metric
            or parsed.get("dimension") != dimension
            or parsed.get("query_type") != "metric_contribution"
        ):
            continue
        if payload.get("method") != "proportional_overlap_v1" or payload.get("allocation_method") != "proportional_overlap_v1":
            continue
        if kind == "day_pair":
            if (
                payload.get("current_date") != comparison.get("current_date")
                or payload.get("baseline_date") != comparison.get("baseline_date")
            ):
                continue
            current_period = {
                "start_date": comparison["current_date"], "end_date": comparison["current_date"],
            }
            baseline_period = {
                "start_date": comparison["baseline_date"], "end_date": comparison["baseline_date"],
            }
        elif kind == "period_pair":
            current_period = dict(comparison.get("current") or {})
            baseline_period = dict(comparison.get("baseline") or {})
            payload_current = payload.get("current_period") or {}
            payload_baseline = payload.get("baseline_period") or {}
            if any((
                payload_current.get("start_date") != current_period.get("start_date"),
                payload_current.get("end_date") != current_period.get("end_date"),
                payload_baseline.get("start_date") != baseline_period.get("start_date"),
                payload_baseline.get("end_date") != baseline_period.get("end_date"),
            )):
                continue
        else:
            continue
        candidates.append((evidence_id, payload))
    if len(candidates) != 1:
        _record_contribution_projection_failure(
            state, "DATA_INSUFFICIENT", "缺少唯一且匹配请求的按品类分摊订单贡献证据。",
        )
        return []
    evidence_id, payload = candidates[0]
    try:
        totals = {}
        for key in ("baseline_total", "current_total", "total_delta"):
            value = payload.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"invalid {key}")
            totals[key] = float(value)
        if not math.isclose(
            totals["total_delta"],
            totals["current_total"] - totals["baseline_total"],
            rel_tol=1e-9, abs_tol=1e-6,
        ):
            raise ValueError("reported change does not reconcile with allocated totals")

        def checked_groups(key: str) -> dict[str, float]:
            items = payload.get(key)
            if not isinstance(items, list) or not items:
                raise ValueError(f"missing {key}")
            groups: dict[str, float] = {}
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError(f"invalid {key} row")
                name = item.get("dimension_value")
                value = item.get("value")
                if not isinstance(name, str) or not name:
                    raise ValueError(f"{key} row has no dimension value")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(f"{key} row has an invalid value")
                if name in groups:
                    raise ValueError(f"{key} contains duplicate group {name!r}")
                groups[name] = float(value)
            return groups

        current_groups = checked_groups("current_items")
        baseline_groups = checked_groups("baseline_items")
        if not math.isclose(
            sum(current_groups.values()), totals["current_total"], rel_tol=1e-9, abs_tol=1e-6,
        ) or not math.isclose(
            sum(baseline_groups.values()), totals["baseline_total"], rel_tol=1e-9, abs_tol=1e-6,
        ):
            raise ValueError("allocated groups do not reconcile with distinct-order totals")

        names = sorted(
            current_groups.keys() | baseline_groups.keys(),
            key=lambda name: (
                -abs(current_groups.get(name, 0.0) - baseline_groups.get(name, 0.0)),
                name,
            ),
        )
        rows = []
        for name in names:
            current_value = current_groups.get(name, 0.0)
            baseline_value = baseline_groups.get(name, 0.0)
            delta = current_value - baseline_value
            rows.append({
                "dimension_value": name,
                "baseline_value": round(baseline_value, 6),
                "current_value": round(current_value, 6),
                "delta_value": round(delta, 6),
                "contribution_ratio": (
                    round(delta / totals["total_delta"], 6)
                    if totals["total_delta"] else None
                ),
                "components": [],
                "field_evidence_ids": {
                    field: [evidence_id] for field in (
                        "baseline_value", "current_value", "delta_value", "contribution_ratio",
                    )
                },
            })
        unit = str(definition.unit or "")
        if not unit:
            raise ValueError("unknown unit")
        allocation_note = str(payload.get("allocation_note") or "").strip()
        return [{
            "dimension": dimension, "metric_name": metric,
            "unit": unit, "contribution_unit": unit,
            "aggregation": "registered_proportional_overlap_contribution",
            "baseline_value": totals["baseline_total"], "current_value": totals["current_total"],
            "total_delta": totals["total_delta"],
            "baseline_date": baseline_period["end_date"], "current_date": current_period["end_date"],
            "baseline_period": baseline_period, "current_period": current_period,
            "residual_value": 0.0, "residual_type": "unknown",
            "residual_explainable": True, "residual_tolerance": 0.0,
            "residual_note": allocation_note,
            "analysis_nature": "descriptive_contribution",
            "method": "proportional_overlap_v1",
            "items": rows, "evidence_ids": [evidence_id],
            "field_evidence_ids": {
                field: [evidence_id] for field in (
                    "baseline_value", "current_value", "total_delta",
                )
            },
        }]
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        _record_contribution_projection_failure(
            state, "INVALID_DATA", f"按品类分摊订单贡献证据无法对账（{type(exc).__name__}）。",
        )
        return []


def _record_contribution_projection_failure(state: AgentState, code: str, reason: str) -> None:
    """Keep invalid evidence as an error and registered capability gaps typed."""
    is_capability_boundary = code == "CAPABILITY_GAP"
    state["contribution_projection"] = {
        "status": "unsupported" if is_capability_boundary else "failed",
        "terminal_code": code,
        "reason": str(reason),
        "analysis_nature": "descriptive_contribution_not_business_causality",
    }
    state["terminal_code"] = code
    if is_capability_boundary:
        # A missing registered method is a product boundary, not an internal
        # execution error. Preserve the typed reason for presentation without
        # making a controlled unsupported result look like a workflow failure.
        return
    errors = state.setdefault("errors", [])
    if not any(
        isinstance(error, dict) and error.get("code") == "contribution_projection_failed"
        for error in errors
    ):
        errors.append({
            "code": "contribution_projection_failed",
            "terminal_code": code,
            "message": str(reason),
            "retryable": False,
            "agent_name": "generate_answer",
        })


def _dimension_contribution_text(card: dict[str, Any], *, feedback_revision: bool = False) -> str:
    metric = _metric_display_name(str(card.get("metric_name") or "指标"))
    unit = _display_unit(card.get("unit"))
    canonical_contribution_unit = card.get("contribution_unit") or card.get("unit")
    contribution_unit = _display_unit(canonical_contribution_unit)
    contribution_precision = 4 if canonical_contribution_unit == "percentage_points" else 2
    baseline_period = card.get("baseline_period") or {}
    current_period = card.get("current_period") or {}
    rows = []
    for item in card.get("items") or []:
        ratio = item.get("contribution_ratio")
        share = f"（占总变化 {float(ratio) * 100:.1f}%）" if isinstance(ratio, (int, float)) else ""
        baseline_value = item.get("baseline_value")
        current_value = item.get("current_value")
        baseline_text = f"{float(baseline_value):,.2f} {unit}" if baseline_value is not None else "该期无此分组"
        current_text = f"{float(current_value):,.2f} {unit}" if current_value is not None else "该期无此分组"
        delta_text = f"{float(item['delta_value']):+,.{contribution_precision}f} {contribution_unit}"
        rows.append(
            f"{item.get('dimension_value')}：基线 {baseline_text}，"
            f"当前 {current_text}，统计贡献 {delta_text}{share}"
            + (
                "（" + "、".join(
                    f"{effect.get('component_type')} {float(effect['value']):+,.{contribution_precision}f} {_display_unit(effect.get('unit') or contribution_unit)}"
                    for effect in item.get("components") or []
                ) + "）"
                if item.get("components") else ""
            )
        )
    evidence = "、".join(map(str, card.get("evidence_ids") or []))
    if card.get("method") == "proportional_overlap_v1":
        prefix = (
            "已验证的按品类分摊订单贡献：每个订单按其在范围内涉及的品类数均分，"
            "各品类分摊值之和等于同口径总去重订单数；分摊值为订单等价量，"
            "不等于各品类独立去重订单数。"
        )
    else:
        prefix = "按审核反馈补充已验证的维度比较：" if feedback_revision else "已验证的维度统计贡献："
    return (
        f"{prefix}{metric}按 {card.get('dimension')} 维度，基线期 {baseline_period.get('start_date')} 至 "
        f"{baseline_period.get('end_date')} 总值 {float(card['baseline_value']):,.2f} {unit}；当前期 "
        f"{current_period.get('start_date')} 至 {current_period.get('end_date')} 总值 "
        f"{float(card['current_value']):,.2f} {unit}；总变化 {float(card['total_delta']):+,.{contribution_precision}f} {contribution_unit}。"
        f"分组统计贡献：{'；'.join(rows)}。这些是数值比较与统计贡献，不证明业务因果，业务原因仍未确认。"
        + ("该两期比较也不证明期间内持续趋势或异常。" if feedback_revision else "")
        + f"证据：{evidence}。"
    )


def _is_validated_dimension_contribution_card(
    card: dict[str, Any], validated_cards: list[dict[str, Any]],
) -> bool:
    """Only render detailed dimension prose for an exact validated card."""
    if card.get("method") not in {"additive_change_v1", "symmetric_mix_within_v1", "proportional_overlap_v1"} or card.get("analysis_nature") != "descriptive_contribution":
        return False
    required_fields = (
        "dimension", "metric_name", "unit", "contribution_unit", "baseline_value", "current_value",
        "total_delta", "baseline_period", "current_period", "items", "evidence_ids",
    )
    if any(card.get(field) is None for field in required_fields):
        return False
    try:
        aggregate_values = [card["baseline_value"], card["current_value"], card["total_delta"]]
        if not all(math.isfinite(float(value)) for value in aggregate_values):
            return False

        def valid_group_value(item: dict[str, Any], field: str) -> bool:
            """Allow a missing side only for an evidenced entry/exit group.

            A symmetric period comparison can legitimately have a group only in
            the current period (entry) or only in the baseline period (exit).
            Treating that absent value as a numeric-data failure made a valid
            deterministic card contradict its own answer text.
            """
            value = item[field]
            if value is not None:
                return math.isfinite(float(value))
            other_field = "current_value" if field == "baseline_value" else "baseline_value"
            if not math.isfinite(float(item[other_field])):
                return False
            component_types = {
                str(component.get("component_type") or "")
                for component in (item.get("components") or [])
                if isinstance(component, dict)
            }
            expected = "entry_structural_effect" if field == "baseline_value" else "exit_structural_effect"
            return component_types == {expected}

        for item in card["items"]:
            if not isinstance(item, dict):
                return False
            if not math.isfinite(float(item["delta_value"])):
                return False
            if not valid_group_value(item, "baseline_value"):
                return False
            if not valid_group_value(item, "current_value"):
                return False
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if not card.get("dimension") or not card.get("metric_name") or not card.get("unit") or not card.get("evidence_ids"):
        return False
    fields = (
        "dimension", "metric_name", "unit", "contribution_unit", "baseline_value", "current_value",
        "total_delta", "baseline_period", "current_period", "items", "evidence_ids",
    )

    def comparable_value(field: str, value: Any) -> Any:
        if field != "items" or not isinstance(value, list):
            return value
        # Field-level source bindings are provenance metadata added during
        # presentation projection. They must not make an otherwise identical
        # validated numeric result compare unequal to its source card.
        return [
            {key: item_value for key, item_value in item.items() if key != "field_evidence_ids"}
            if isinstance(item, dict) else item
            for item in value
        ]

    return any(
        all(
            comparable_value(field, card.get(field)) == comparable_value(field, validated.get(field))
            for field in fields
        )
        for validated in validated_cards
    )


def _make_evidence_refs(all_eids: list[str], state: AgentState) -> list[dict]:
    """Build evidence refs for all referenced evidence IDs."""
    bundles = {b["evidence_id"]: b for b in state.get("evidence_bundles", [])}
    refs: list[dict] = []
    seen: set[str] = set()
    for eid in all_eids:
        if eid in seen:
            continue
        seen.add(eid)
        b = bundles.get(eid)
        if b:
            content = b.get("content", {})
            # Build a short summary from content without dumping full content
            summary = str(content.get("question") or content.get("summary") or content.get("metric_name") or "")
            if len(summary) > 80:
                summary = summary[:80] + "…"
            refs.append({
                "evidence_id": eid,
                "source_type": b.get("source_type", ""),
                "summary": summary,
            })
    return refs


_LLM_RECOMMENDATION_PROMPT = (
    "你是一个经营分析助手的建议模块。基于以下事实、变化和贡献数据，"
    "给出 1-2 条'下一步核查或行动优先级'建议。\n"
    "规则：\n"
    "1. 只能引用已提供的 evidence_id\n"
    "2. 不新增数值、事实、原因或因果断言\n"
    "3. 有贡献数据时优先建议核查贡献最大的维度\n"
    "4. 无贡献数据时建议按允许维度进一步拆解\n\n"
    "返回 JSON：{{ \"recommendations\": [{{\"text\": \"…\", \"evidence_ids\": [\"ev_xxx\"]}}] }}\n\n"
)


def _build_recommendations(analysis_result: dict, all_eids: list[str], state: AgentState | None = None) -> list[dict]:
    """受证据约束的建议生成：仅在有贡献数据时调用 LLM。"""
    contribution = analysis_result.get("contribution", [])
    if not contribution:
        return []

    from app.llm.client import llm_chat
    from app.context.orchestrator import (
        ContextOrchestrator, OrchestratedItem, Source, TrustTier,
    )
    from app.llm.routing import model_for

    context_parts = []
    for contrib in contribution[:2]:
        dim = contrib.get("dimension", "")
        top_val = contrib.get("top_contributor", "")
        delta = contrib.get("delta_value")
        ratio = contrib.get("contribution_ratio")
        context_parts.append(
            f"维度={dim}, 主要贡献={top_val}, "
            f"delta={delta}, 贡献率={ratio}"
        )
    context = "; ".join(context_parts)
    refreshed_ids = set(str(item) for item in ((state or {}).get("evidence_refresh") or {}).get("evidence_ids") or [])
    evidence_ids = tuple(dict.fromkeys(str(item) for item in all_eids if str(item) in refreshed_ids))
    if not evidence_ids:
        return []
    contribution_item = OrchestratedItem(
        item_id="answer.recommendation_contribution",
        source=Source.TOOL_EVIDENCE,
        trust_tier=TrustTier.CURRENT_EVIDENCE,
        content=context,
        source_refs=evidence_ids,
        scope_fingerprint=str((state or {}).get("current_query_scope_fingerprint") or ""),
        version=str((state or {}).get("data_version_manifest_hash") or "current"),
        required=True,
        priority=100,
        evidence_ids=evidence_ids,
    )
    from app.context.packer import ContextBudgetExceeded

    try:
        bundle = ContextOrchestrator().build_specialist_bundle(
            state or {}, "answer", system_prompt=_LLM_RECOMMENDATION_PROMPT,
            model=model_for("chat")[0], additional_items=[contribution_item],
        )
    except ContextBudgetExceeded:
        # Recommendations are optional. An over-budget context must not turn
        # an otherwise grounded answer into a failed workflow turn.
        return []
    if not bundle.text:
        return []
    try:
        raw = llm_chat(
            system_prompt=_LLM_RECOMMENDATION_PROMPT,
            user_message=bundle.text,
        )
    except Exception:
        return []

    # llm_chat returns str; parse JSON
    if isinstance(raw, str):
        import json
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []

    if not isinstance(raw, dict):
        return []
    recs = raw.get("recommendations", [])
    if not isinstance(recs, list):
        return []
    validated: list[dict] = []
    known_eids = set(all_eids)
    for rec in recs:
        text = rec.get("text", "") if isinstance(rec, dict) else ""
        if not text:
            continue
        ids = rec.get("evidence_ids", []) if isinstance(rec, dict) else []
        # Discard recommendation if it cannot cite real evidence
        valid_ids = [eid for eid in ids if eid in known_eids]
        if valid_ids:
            validated.append({"text": text, "evidence_ids": valid_ids})
    return validated


# ================================================================
# 文本答案生成（保留原有逻辑）
# ================================================================


def _summarize_metric_query(tool_result: dict) -> str:
    """从 metric_query 的 tool_result 中提取人类可读的答案。"""

    def _fmt(v: float | None) -> str:
        return f"{v:.2f}" if v is not None else "暂无数据"

    result = tool_result.get("result", {})

    if tool_result.get("_batch_item"):
        parsed = result.get("parsed") if isinstance(result, dict) else {}
        parsed = parsed if isinstance(parsed, dict) else {}
        metric = _metric_display_name(str(parsed.get("metric_name") or "该指标"))
        operation = str(parsed.get("query_type") or "查询")
        period = str(parsed.get("metric_date") or parsed.get("start_date") or "")
        if period and parsed.get("end_date") and parsed.get("end_date") != period:
            period = f"{period} 至 {parsed.get('end_date')}"
        period_text = f"，期间 {period}" if period else ""
        status = str(result.get("status") or "failed") if isinstance(result, dict) else "failed"
        reason = str(result.get("reason") or "") if isinstance(result, dict) else ""
        if status == "not_found":
            return f"{metric}（{operation}{period_text}）：当前范围无数据。" + (f"原因：{reason}" if reason else "")
        if status == "unsupported":
            return f"{metric}（{operation}{period_text}）：当前能力不支持，未完成该项。" + (f"原因：{reason}" if reason else "")
        if status not in {"ok"}:
            return f"{metric}（{operation}{period_text}）：执行失败，未完成该项。" + (f"原因：{reason}" if reason else "")

    capability_boundary = result.get("capability_boundary") if isinstance(result, dict) else None
    if isinstance(capability_boundary, dict):
        reason = str(capability_boundary.get("reason") or result.get("reason") or "当前请求超出已注册的统计贡献能力。")
        return reason

    clarification = result.get("clarification") if isinstance(result, dict) else None
    if isinstance(clarification, dict):
        candidates = clarification.get("ambiguous_candidates", [])
        names = "、".join(str(item.get("display_name") or item.get("metric_id")) for item in candidates[:3])
        reason = clarification.get("reason") or "请确认要分析的指标。"
        return f"未执行数据查询。{reason}" + (f"\n\n可确认的指标：{names}" if names else "")

    if result.get("template_only"):
        return "模板分析尚未执行；需要先创建并确认分析计划。"

    batch_results = tool_result.get("batch_results")
    if isinstance(batch_results, list) and len(batch_results) > 1:
        summaries = []
        for item in batch_results:
            if not isinstance(item, dict):
                continue
            summaries.append(_summarize_metric_query({key: value for key, value in item.items() if key != "batch_results"}))
        if summaries:
            answer = "\n\n".join(f"{index}. {summary}" for index, summary in enumerate(summaries, start=1))
            batch_status = str(tool_result.get("batch_status") or "")
            if batch_status == "partial_unavailable":
                answer += "\n\n本批次部分指标不可用；以上有效结果可参考，但整批任务未全部完成。"
            elif batch_status == "execution_failed":
                answer += "\n\n本批次有指标执行失败；以上已取得的结果仍保留，整批任务状态为失败。"
            return answer

    explorer_result = result.get("explorer_result")
    if explorer_result:
        context = explorer_result.get("context", {})
        summary = explorer_result.get("summary", {})
        return (
            f"当前浏览范围已锁定：店铺 {context.get('seller_id')}，"
            f"{context.get('start_date')} 至 {context.get('end_date')}，"
            f"{summary.get('metric_name')} 为 {_fmt(summary.get('value'))} {summary.get('unit', '')}。"
            f" 数据来源：{_source_display_name(explorer_result.get('lineage', {}).get('source_table'))}。"
        )

    metric_result = result.get("metric_result")
    if metric_result:
        period_label = (
            f"{metric_result.get('period_start_date')} 至 {metric_result.get('period_end_date')}"
            if metric_result.get("period_start_date") and metric_result.get("period_end_date")
            else str(metric_result.get("metric_date") or "")
        )
        if metric_result.get("status") == "not_found" or metric_result.get("value") is None:
            return (
                f"{period_label} 的 {_metric_display_name(metric_result.get('metric_name'))} 暂无数据。"
                f" 数据来源：{_source_display_name(metric_result.get('source_table'))}。"
            )
        return (
            f"{period_label} 的 {_metric_display_name(metric_result.get('metric_name'))} 为 "
            f"{_fmt(metric_result.get('value'))} {_display_unit(metric_result.get('unit'))}。"
            f" 数据来源：{_source_display_name(metric_result.get('source_table'))}。"
        )

    statistics_result = result.get("statistics_result")
    if statistics_result:
        stats = statistics_result.get("statistics") or {}
        labels = {
            "count": "样本数", "mean": "平均值", "average": "平均值",
            "std": "标准差", "min": "最小值", "q1": "P25",
            "median": "中位数", "q3": "P75", "max": "最大值",
        }
        summary = "，".join(
            f"{labels.get(name, name)}={_fmt(value)}"
            for name, value in stats.items()
            if value is not None
        )
        return (
            f"{_metric_display_name(statistics_result.get('metric_name'))}在 "
            f"{statistics_result.get('start_date')} 至 {statistics_result.get('end_date')} "
            f"按{statistics_result.get('granularity')}粒度统计，共{statistics_result.get('count', 0)}个时间点："
            f"{summary}。最小值时间点：{statistics_result.get('min_bucket')}，"
            f"最大值时间点：{statistics_result.get('max_bucket')}。"
            f"数据来源：{_source_display_name(statistics_result.get('source_table'))}。"
        )

    trend_result = result.get("trend_result")
    if trend_result:
        points = [point for point in trend_result.get("points", []) if isinstance(point, dict) and isinstance(point.get("value"), (int, float))]
        metric = _metric_display_name(trend_result.get("metric_name"))
        unit = _display_unit(trend_result.get("unit"))
        if not points:
            return f"{metric} 在 {trend_result.get('start_date')} 至 {trend_result.get('end_date')} 未返回可用趋势数据。"
        first, last = points[0], points[-1]
        first_value, last_value = float(first["value"]), float(last["value"])
        delta = last_value - first_value
        ratio = (delta / abs(first_value) * 100) if first_value else None
        direction = "上升" if delta > 0 else "下降" if delta < 0 else "持平"
        peak = max(points, key=lambda point: float(point["value"]))
        trough = min(points, key=lambda point: float(point["value"]))
        latest_change = ""
        if len(points) >= 2:
            previous = float(points[-2]["value"])
            latest_delta = last_value - previous
            latest_ratio = (latest_delta / abs(previous) * 100) if previous else None
            latest_direction = "上升" if latest_delta > 0 else "下降" if latest_delta < 0 else "持平"
            latest_change = f"最近一期较上一期{latest_direction} {_fmt(abs(latest_delta))}" + (f"（{latest_ratio:+.1f}%）" if latest_ratio is not None else "") + "。"
        preview_points = points if len(points) <= 6 else [*points[:3], *points[-3:]]
        preview = ", ".join(f"{point.get('bucket')}={_fmt(point.get('value'))}" for point in preview_points)
        return (
            f"{metric} 在 {trend_result.get('start_date')} 至 {trend_result.get('end_date')} 共 {len(points)} 个时间点，"
            f"从 {first.get('bucket')} 的 {_fmt(first_value)} 到 {last.get('bucket')} 的 {_fmt(last_value)}，整体{direction} {_fmt(abs(delta))}"
            + (f"（{ratio:+.1f}%）。" if ratio is not None else "。")
            + f"最高点为 {peak.get('bucket')} 的 {_fmt(peak.get('value'))}，最低点为 {trough.get('bucket')} 的 {_fmt(trough.get('value'))}。"
            + latest_change
            + f" 数据点：{preview}。"
            + (f"数值单位：{unit}。" if unit else "")
            + f"数据来源：{_source_display_name(trend_result.get('source_table'))}。"
        )

    compare_result = result.get("compare_result")
    if compare_result:
        compare_result = dict(compare_result)
        current_date = compare_result.get("current_date") or compare_result.get("left_date")
        baseline_date = compare_result.get("baseline_date") or compare_result.get("right_date")
        if compare_result.get("left_end_date") and compare_result.get("left_end_date") != current_date:
            current_date = f"{current_date} 至 {compare_result.get('left_end_date')}"
        if compare_result.get("right_end_date") and compare_result.get("right_end_date") != baseline_date:
            baseline_date = f"{baseline_date} 至 {compare_result.get('right_end_date')}"
        if (
            compare_result.get("status") != "ok"
            or compare_result.get("current_value", compare_result.get("left_value")) is None
            or compare_result.get("baseline_value", compare_result.get("right_value")) is None
        ):
            return (
                f"{_metric_display_name(compare_result.get('metric_name'))} 无法完成比较："
                f"当前期 {current_date}={_fmt(compare_result.get('current_value', compare_result.get('left_value')))}，"
                f"基线期 {baseline_date}={_fmt(compare_result.get('baseline_value', compare_result.get('right_value')))}。"
                "至少一侧缺少有效数据，未计算或解释变化。"
            )
        metric_id = str(compare_result.get("metric_name") or "")
        metric_unit, change_unit, ratio_unit = _comparison_unit_contract(metric_id)
        current_value = _fmt(compare_result.get("current_value", compare_result.get("left_value")))
        baseline_value = _fmt(compare_result.get("baseline_value", compare_result.get("right_value")))
        delta_value = _fmt(compare_result.get("delta_value"))
        delta_ratio = _fmt(compare_result.get("delta_ratio"))
        return (
            f"{_metric_display_name(metric_id)} 对比结果："
            f"当前期 {current_date}={current_value}{' ' + _display_unit(metric_unit) if metric_unit else ''}，"
            f"基线期 {baseline_date}={baseline_value}{' ' + _display_unit(metric_unit) if metric_unit else ''}，"
            f"差值={delta_value}{' ' + _display_unit(change_unit) if change_unit else ''}，"
            f"变化率={delta_ratio}{' ' + _display_unit(ratio_unit) if ratio_unit else ''}。"
        )

    multi_breakdown = result.get("multi_dimension_breakdown_result")
    if isinstance(multi_breakdown, dict):
        status = str(multi_breakdown.get("status") or "failed")
        metric = _metric_display_name(str(multi_breakdown.get("metric_name") or "该指标"))
        reason = str(multi_breakdown.get("message") or result.get("reason") or "")
        if status == "unsupported":
            return f"{metric} 的多维交叉查询当前不支持。" + (f"原因：{reason}" if reason else "")
        if status != "ok":
            return f"{metric} 的多维交叉查询未取得结果。" + (f"原因：{reason}" if reason else "")
        dimensions = [str(item) for item in multi_breakdown.get("dimensions") or []]
        items = multi_breakdown.get("items") or []
        preview = []
        for item in items[:10]:
            if not isinstance(item, dict):
                continue
            values = item.get("dimensions") if isinstance(item.get("dimensions"), dict) else {}
            group = "，".join(f"{name}={value}" for name, value in values.items())
            preview.append(f"{group}：{_fmt(item.get('value'))}")
        period = f"{multi_breakdown.get('start_date')} 至 {multi_breakdown.get('end_date')}"
        axes = " × ".join(dimensions)
        answer = (
            f"{period} 的 {metric} 按 {axes} 精确交叉汇总："
            f"{'；'.join(preview) if preview else ('所有分组均因小样本保护未展示' if multi_breakdown.get('suppressed_cell_count') else '当前范围无数据')}。"
        )
        if multi_breakdown.get("suppressed_cell_count"):
            answer += f"另有 {multi_breakdown['suppressed_cell_count']} 个小样本分组未展示。"
        if multi_breakdown.get("cell_counts_may_overlap"):
            answer += "各交叉格中的去重订单数或客户数可能重叠，不能把各格直接相加作为总体。"
        if multi_breakdown.get("truncated"):
            answer += "结果受返回条数上限限制，当前展示的交叉格不代表完整列表。"
        return answer

    breakdown_result = result.get("breakdown_result")
    if breakdown_result:
        items = breakdown_result.get("items", [])[:10]
        preview = ", ".join(
            f"{i.get('dimension_value')}={_fmt(i.get('value'))}" for i in items
        )
        return (
            f"{breakdown_result.get('metric_date')} 的 {_metric_display_name(breakdown_result.get('metric_name'))} 按 "
            f"{breakdown_result.get('dimension')} 拆解：{preview}。"
        )

    reason = result.get("reason", "当前问题暂不支持。")
    return f"当前未能完成查询。原因：{reason}"


def _trend_missing_status(tool_result: dict) -> str:
    """State missingness only when the returned day buckets prove it."""
    from datetime import date, timedelta

    result = tool_result.get("result") or {}
    trend = result.get("trend_result") or {}
    if result.get("status") != "ok" or trend.get("status") != "ok" or trend.get("granularity") != "day":
        return "缺失状态：当前结果无法确认完整性。"
    try:
        start, end = date.fromisoformat(trend["start_date"]), date.fromisoformat(trend["end_date"])
        if end < start or (end - start).days > 3660:
            raise ValueError("invalid range")
        parsed = result.get("parsed") or {}
        if parsed.get("start_date") != start.isoformat() or parsed.get("end_date") != end.isoformat():
            raise ValueError("returned range differs from request")
        points = trend.get("points")
        if not isinstance(points, list):
            raise ValueError("missing points")
        observed = {}
        for point in points:
            bucket = date.fromisoformat(point["bucket"])
            if bucket in observed or bucket < start or bucket > end:
                raise ValueError("ambiguous bucket")
            value = point.get("value")
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError("invalid value")
            observed[bucket] = value
        missing = [start + timedelta(days=i) for i in range((end - start).days + 1) if observed.get(start + timedelta(days=i)) is None]
        if missing:
            return "缺失状态：以下日期没有可用数值：" + "、".join(d.isoformat() for d in missing) + "。"
        return "缺失状态：请求日期范围内每日均有可用数值。"
    except (KeyError, TypeError, ValueError, OverflowError):
        return "缺失状态：当前结果无法确认完整性。"


def _semantic_interpretation_text(state: AgentState) -> str:
    """Render the server-validated semantic mapping before the answer."""

    decision = state.get("metric_planning_decision") or {}
    semantic = decision.get("semantic_interpretation") if isinstance(decision, dict) else None
    if not isinstance(semantic, dict) or not semantic.get("metric_id"):
        return ""
    source = semantic.get("source_text") or "用户表达"
    display = semantic.get("display_name") or semantic.get("metric_id")
    dimension = semantic.get("dimension_label") or semantic.get("dimension") or "当前范围"
    interpretation = semantic.get("interpretation") or f"按{dimension}分析{display}"
    guardrail = "；“原因”仅按统计关联或贡献度分析，不代表因果关系" if "原因" in str(state.get("user_message", "")) or "导致" in str(state.get("user_message", "")) else ""
    prefix = "语义映射（待确认）" if semantic.get("requires_confirmation") else "本次分析口径"
    return f"{prefix}：将“{source}”理解为“{display}”，按“{dimension}”分析。{interpretation}{guardrail}。"


def _confirmed_premise_direction_text(state: AgentState) -> str:
    """Render a deterministic correction after the user resumes a premise pause."""
    premise = state.get("premise_check") or {}
    if not isinstance(premise, dict) or premise.get("status") != "confirmed_by_user":
        return ""

    claimed = str(premise.get("claimed_direction") or "")
    actual = str(premise.get("actual_direction") or "")
    if claimed == actual:
        return ""

    direction_labels = {"up": "上升", "down": "下降", "flat": "持平"}
    claimed_label = direction_labels.get(claimed)
    actual_label = direction_labels.get(actual)
    if not claimed_label or not actual_label:
        return ""

    metric = str(premise.get("metric_display_name") or premise.get("metric_name") or "该指标")
    baseline_date = premise.get("baseline_date")
    current_date = premise.get("current_date")
    if not isinstance(baseline_date, str) or not isinstance(current_date, str):
        return ""
    return (
        f"方向核验：{metric}从 {baseline_date} 到 {current_date} 实际{actual_label}，"
        f"并非{claimed_label}。"
    )


def _grounded_narrative(state: AgentState, evidence: object, fallback: str) -> tuple[str, str]:
    """Generate a concise narrative from current-turn evidence, with a safe fallback."""
    import re
    from app.llm.client import llm_chat
    from app.llm.routing import model_for
    from app.context.orchestrator import ContextOrchestrator
    from app.context.packer import (
        ContextBudgetExceeded, ContextItem, ContextSource, TrustLevel, pack_for_llm,
    )

    ensure_safe_untrusted_content(evidence, "answer_evidence")
    resolution = state.get("turn_resolution") or {}
    compilation = state.get("context_compilation") or {}
    question = str(
        (resolution.get("resolved_text") if isinstance(resolution, dict) else "")
        or compilation.get("standalone_question")
        or state.get("user_message")
        or ""
    )
    system = (
        "你是 BizInsight 的经营分析助手。只根据本轮给出的证据回答用户问题。"
        "不得补造数字、日期、店铺、筛选条件、因果关系或证据中没有的定义。"
        "数据分析回答需明确指标、时间范围和维度（证据未提供时说明缺失）；"
        "知识问题需区分注册表权威定义和知识库补充材料。简洁作答，不输出 JSON。"
    )
    model = model_for("chat")[0]
    # WP1: the answer Bundle carries validated current evidence itself. The
    # final packer check below covers this exact rendered Bundle plus request.
    try:
        answer_bundle = ContextOrchestrator().build_specialist_bundle(
            state, "answer", instruction=question, system_prompt=system, model=model,
        )
    except ContextBudgetExceeded:
        return fallback, "context_budget_exceeded"
    included_ids = set(answer_bundle.included_item_ids)
    grounding_items = [
        item for item in answer_bundle.items
        if item.item_id in included_ids
        and (
            item.trust_tier.value == "controlled_definition"
            or (item.trust_tier.value == "current_evidence" and item.evidence_ids)
        )
    ]
    evidence_text = "\n".join(item.content for item in grounding_items)
    if not evidence_text.strip():
        return fallback, "context_evidence_unavailable"
    context_block = (
        f"\n\n[受控上下文参考，仅用于解释和承接；不得改写本轮数值、授权或证据]\n{answer_bundle.text}"
        if answer_bundle.text else ""
    )
    user_message = (
        f"用户问题：{question}{context_block}\n\n"
        "仅可使用 Bundle 中明确标记为 current_evidence 或 controlled_definition 的内容作答。"
    )
    # Final guard: pack the complete prompt (system + instruction + context +
    # evidence) before the provider call.  Required overflow is an explicit
    # fallback, not a silent truncation.
    try:
        pack_for_llm(
            "answer",
            [ContextItem(
                item_id="final_prompt", source=ContextSource.CURRENT_QUESTION,
                content=user_message, priority=100, required=True,
                trust_level=TrustLevel.USER_INPUT,
            )],
            system_prompt=system, model=model, state=state,
            overflow_policy="reject",
        )
    except ContextBudgetExceeded:
        return fallback, "context_budget_exceeded"
    try:
        generated = str(llm_chat(
            system_prompt=system,
            user_message=user_message,
            temperature=0.1,
            max_tokens=900,
        ) or "").strip()
    except Exception:
        generated = ""
    if not generated:
        return fallback, "template_fallback"

    # Guard against numeric claims absent from the serialized evidence. Formatting
    # commas are normalized so 1,234 and 1234 compare as the same value.
    from decimal import Decimal, InvalidOperation
    def normalize(value: str) -> str:
        value = re.sub(r"[,，]", "", value)
        suffix = "%" if value.endswith("%") else ""
        try:
            return format(Decimal(value.rstrip("%" )).normalize(), "f") + suffix
        except InvalidOperation:
            return value
    evidence_numbers = {normalize(item) for item in re.findall(r"(?<!\w)[+-]?\d+(?:[.,]\d+)?%?", evidence_text)}
    answer_numbers = {normalize(item) for item in re.findall(r"(?<!\w)[+-]?\d+(?:[.,]\d+)?%?", generated)}
    if answer_numbers - evidence_numbers:
        return fallback, "template_fallback_numeric_guard"
    query_result = evidence.get("query_result") if isinstance(evidence, dict) else None
    breakdown = ((query_result or {}).get("result") or {}).get("breakdown_result") if isinstance(query_result, dict) else None
    required_dimensions = [
        str(item.get("dimension_value") or "")
        for item in (breakdown or {}).get("items", [])
        if item.get("dimension_value")
    ]
    if required_dimensions and any(value not in generated for value in required_dimensions):
        return fallback, "template_fallback_incomplete_breakdown"
    return generated, "llm_grounded"


# ================================================================
# 主函数
# ================================================================


def _critic_answer_revision_feedback(state: AgentState) -> dict[str, Any] | None:
    """Return latest answer-target Critic feedback without treating it as evidence."""
    planning_decision = state.get("metric_planning_decision") or {}
    context_intent = (state.get("context_compilation") or {}).get("operation_intent")
    context_intent = context_intent if isinstance(context_intent, dict) else {}
    registered_numeric_decomposition = (
        state.get("task_type") == "metric_query"
        and (
            planning_decision.get("operation_intent") == "registered_numeric_decomposition"
            or context_intent.get("kind") == "registered_numeric_decomposition"
        )
    )
    if (
        (state.get("task_type") != "anomaly_analysis" and not registered_numeric_decomposition)
        or state.get("critic_verdict") != "needs_revision"
        or state.get("critic_revision_target") != "generate_answer"
    ):
        return None

    reviews = ((state.get("critic_result") or {}).get("llm_review") or {})
    if not isinstance(reviews, dict):
        return None
    categories: list[str] = []
    issues: list[str] = []
    for category in ("groundedness", "completeness", "consistency"):
        review = reviews.get(category)
        if not isinstance(review, dict):
            continue
        values = review.get("issues") or []
        if review.get("verdict") not in {"failed", "rejected", "needs_revision"} or not values:
            continue
        categories.append(category)
        for item in values:
            if isinstance(item, str):
                issues.append(item)
            elif isinstance(item, dict):
                text = item.get("issue") or item.get("message") or item.get("description")
                if isinstance(text, str):
                    issues.append(text)
    if not issues:
        return None

    normalized = "\n".join(issues).casefold()
    return {
        "categories": list(dict.fromkeys(categories)),
        "trend_requested": any(token in normalized for token in (
            "trend", "comparison", "compare", "period change", "movement",
            "趋势", "走势", "比较", "对比", "变化",
        )),
        # A causal request can only trigger a boundary statement. Its wording
        # and factual content are never copied into the answer.
        "causal_requested": any(token in normalized for token in (
            "cause", "causal", "root cause", "reason", "因果", "原因", "导致",
        )),
    }


def _verified_decomposition_for_revision(
    state: AgentState, decomposition: dict | None,
) -> dict | None:
    """Require current validated evidence, scope and period before revising facts."""
    if not isinstance(decomposition, dict):
        return None
    evidence_ids = decomposition.get("evidence_ids") or []
    if not evidence_ids:
        return None
    trusted_ids = {
        str(bundle.get("evidence_id"))
        for bundle in (state.get("evidence_bundles") or [])
        if isinstance(bundle, dict) and bundle.get("trusted") is True and bundle.get("evidence_id")
    }
    normalized_ids = set(map(str, evidence_ids))
    if not normalized_ids <= trusted_ids:
        return None
    try:
        from app.graph.nodes.diagnostic import _completed_valid_evidence_ids

        if not normalized_ids <= set(_completed_valid_evidence_ids(state)):
            return None
    except Exception:
        return None
    try:
        baseline = float(decomposition["baseline_value"])
        current = float(decomposition["current_value"])
        delta = float(decomposition["delta_value"])
        import math

        if not all(math.isfinite(value) for value in (baseline, current, delta)):
            return None
        if abs((current - baseline) - delta) > max(0.01, abs(delta) * 1e-6):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    if not decomposition.get("unit") or not decomposition.get("change_unit"):
        return None
    for key in ("baseline_period", "current_period"):
        period = decomposition.get(key)
        if not isinstance(period, dict) or not period.get("start_date") or not period.get("end_date"):
            return None
    return decomposition


def _bounded_period_comparison_text(decomposition: dict) -> str:
    """Describe only the verified baseline/current comparison, not a trajectory."""
    metric = decomposition.get("display_name") or _metric_display_name(str(decomposition.get("metric_id") or "指标"))
    baseline_period = decomposition["baseline_period"]
    current_period = decomposition["current_period"]
    baseline = float(decomposition["baseline_value"])
    current = float(decomposition["current_value"])
    delta = float(decomposition["delta_value"])
    unit = _display_unit(decomposition.get("unit"))
    change_unit = _display_unit(decomposition.get("change_unit"))
    if delta > 0:
        verb, direction = "升至", "增加"
    elif delta < 0:
        verb, direction = "降至", "减少"
    else:
        verb, direction = "保持", "变化"
    return (
        f"按已验证的基线期（{baseline_period['start_date']} 至 {baseline_period['end_date']}）"
        f"与当前期（{current_period['start_date']} 至 {current_period['end_date']}）比较，"
        f"{metric}从 {baseline:,.2f} {unit}{verb} {current:,.2f} {unit}，"
        f"{direction} {abs(delta):,.2f} {change_unit}。"
        "这只说明两个比较期之间的数值差异，不证明期间内持续变化，也不构成异常判断。"
    )


def generate_answer(state: AgentState) -> AgentState:
    """从不同 Agent 路径的结果中提取最终答案 + 构建结构化展示结论。

    优先级（与 triage 分配的 task_type 对应）：
    1. `analysis_result`（来自 root_cause / report 节点）
    2. `metric_context`（来自 metric_definition 节点）
    3. `tool_result`（来自 data_query 节点）
    4. 兜底
    """

    # ================================================================
    # 第一步：按优先级选取答案源
    # ================================================================

    from app.context.request_contract import (
        record_contract_observation, record_contract_violation,
        validate_compiled_request_contract,
    )
    compilation = state.get("context_compilation") or {}
    if compilation.get("request_contract"):
        valid_contract, contract_reason = validate_compiled_request_contract(
            compilation,
            state_contract_hash=state.get("request_contract_hash"),
            require_state_anchor=True,
        )
        record_contract_observation(state, "answer_entry", {
            "contract_valid": valid_contract,
            "validation_reason": contract_reason,
        })
        expected_clarification = bool(
            compilation.get("needs_clarification")
            and (compilation.get("request_contract") or {}).get("parse_status") == "clarification_required"
            and contract_reason == (compilation.get("request_contract") or {}).get("uncertainty_reason")
        )
        if not valid_contract and not expected_clarification:
            record_contract_violation(state, "answer", contract_reason or "request_contract_invalid")

    if isinstance(state.get("result_followup_context"), dict) and state.get("result_followup_context"):
        from app.context.result_context import build_result_followup_answer

        try:
            return build_result_followup_answer(state)
        except ValueError as exc:
            state.setdefault("errors", []).append({
                "code": "result_followup_invalid",
                "message": f"Unable to answer from result context: {exc}",
                "agent_name": "generate_answer",
                "retryable": False,
            })
            return state

    analysis_result = state.get("analysis_result", {})
    metric_context = state.get("metric_context", "")
    tool_result = state.get("tool_result", {})
    token_budget_limited = any(
        isinstance(error, dict) and error.get("code") == "token_budget_exceeded"
        for error in (state.get("errors") or [])
    )

    if token_budget_limited and not (analysis_result and analysis_result.get("summary")):
        answer_text = "本次分析因 Token 预算已用尽而提前停止，尚未完成必要的查询或诊断；当前不提供诊断结论。"
        answer_source = "budget_limit"
    elif analysis_result and analysis_result.get("summary"):
        answer_text = analysis_result["summary"]
        answer_source = "analysis"
    elif metric_context:
        answer_text = f"指标知识检索结果：{metric_context or '暂无知识库内容。'}"
        answer_source = "metric_context"
    elif tool_result:
        answer_text = _summarize_metric_query(tool_result)
        answer_source = "tool_result"
    else:
        answer_text = "暂无结果。"
        answer_source = "none"

    raw_decomposition = analysis_result.get("metric_decomposition") if isinstance(analysis_result, dict) else None
    request_contract = (compilation.get("request_contract") or {}) if isinstance(compilation, dict) else {}
    contribution_requested = request_contract.get("operation_intent") == "dimensional_statistical_contribution"
    validated_contribution_cards = (
        _validated_contribution_cards(state)
        if state.get("task_type") == "anomaly_analysis" or contribution_requested
        else []
    )
    projection_failure = state.get("contribution_projection") or {}
    if projection_failure.get("status") in {"failed", "unsupported"}:
        reason = str(projection_failure.get("reason") or "必要的分组充分统计量无法验证。")
        if projection_failure.get("status") == "unsupported":
            answer_text = (
                "当前指标、维度和比较期间组合尚未注册可验证的统计贡献方法，"
                "因此不展示分组贡献数值，也不推断业务因果。"
            )
        else:
            answer_text = (
                "已执行维度贡献查询，但无法完成已验证的统计贡献计算。"
                f"限制：{reason}。未生成贡献数值或业务因果结论。"
            )
        answer_source = "analysis"
    raw_contributions = analysis_result.get("contribution", [])
    raw_additive_contributions = [
        item for item in raw_contributions
        if isinstance(item, dict) and item.get("method") == "additive_change_v1"
    ]
    valid_additive_evidence_ids = {
        evidence_id
        for card in validated_contribution_cards
        for evidence_id in card.get("evidence_ids", [])
    }
    invalid_additive_contribution_present = any(
        not (item_ids := set(map(str, item.get("evidence_ids") or [])))
        or not item_ids <= valid_additive_evidence_ids
        for item in raw_additive_contributions
    )
    unvalidated_contribution_summary = (
        state.get("task_type") == "anomaly_analysis"
        and invalid_additive_contribution_present
    )
    display_limitations = list(analysis_result.get("limitations", []))
    trend_result = (
        (tool_result.get("result") or {}).get("trend_result")
        if isinstance(tool_result, dict) else None
    )
    sparsity_limitation = _daily_trend_sparsity_limitation(trend_result)
    if sparsity_limitation and sparsity_limitation not in display_limitations:
        display_limitations.append(sparsity_limitation)
    if projection_failure.get("status") == "unsupported":
        unsupported_detail = str(projection_failure.get("reason") or "")
        if unsupported_detail:
            display_limitations = [
                item for item in display_limitations
                if not isinstance(item, str) or unsupported_detail not in item
            ]
        limitation = "当前指标、维度和比较期间组合未注册可验证的统计贡献方法；未展示贡献数值。"
        if limitation not in display_limitations:
            display_limitations.append(limitation)
    elif projection_failure.get("status") == "failed":
        limitation = f"维度贡献未完成（{projection_failure.get('terminal_code')}）：{projection_failure.get('reason')}"
        if limitation not in display_limitations:
            display_limitations.append(limitation)
    if unvalidated_contribution_summary:
        # Finalizer-generated investigation directions may contain percentages
        # derived from the same invalid contribution. Keep unrelated limitations,
        # but do not project those stale values as if the rejected evidence held.
        display_limitations = [
            item for item in display_limitations
            if not (isinstance(item, str) and "调查方向" in item)
        ]
    trusted_bundle_ids = {
        str(bundle.get("evidence_id"))
        for bundle in (state.get("evidence_bundles") or [])
        if isinstance(bundle, dict) and bundle.get("trusted") is True and bundle.get("evidence_id")
    }
    metric_decomposition = (
        raw_decomposition
        if isinstance(raw_decomposition, dict)
        and raw_decomposition.get("evidence_ids")
        and set(map(str, raw_decomposition.get("evidence_ids", []))) <= trusted_bundle_ids
        else None
    )
    planning_decision = state.get("metric_planning_decision") or {}
    compiled_intent = (state.get("context_compilation") or {}).get("operation_intent") or {}
    registered_numeric_decomposition = (
        state.get("task_type") == "metric_query"
        and (
            planning_decision.get("operation_intent") == "registered_numeric_decomposition"
            or (isinstance(compiled_intent, dict) and compiled_intent.get("kind") == "registered_numeric_decomposition")
        )
    )
    decomposition_text = _metric_decomposition_text(metric_decomposition)
    if decomposition_text:
        # For this explicit operation, the registered deterministic result is
        # the user-facing numeric answer. A model summary can retain malformed
        # or conflicting numeric/unit wording, so do not concatenate it as a
        # second source of truth. Limitations are appended below as before.
        answer_text = (
            decomposition_text
            if registered_numeric_decomposition
            else f"{answer_text}\n\n{decomposition_text}" if answer_text else decomposition_text
        )
        answer_source = "analysis"

    if validated_contribution_cards:
        # The validated investigation evidence is the source for both prose and
        # cards. A summary can add context, but never substitutes for the values.
        contribution_text = "\n\n".join(
            _dimension_contribution_text(card) for card in validated_contribution_cards
        )
        if (
            request_contract.get("operation_intent") == "dimensional_statistical_contribution"
            and (
                not state.get("diagnostic_plan")
                or any(card.get("method") == "symmetric_mix_within_v1" for card in validated_contribution_cards)
            )
        ):
            answer_text = contribution_text
        else:
            answer_text = f"{answer_text}\n\n{contribution_text}" if answer_text else contribution_text
        answer_source = "analysis"
    elif unvalidated_contribution_summary:
        # Do not repeat an unverified top-contributor claim from a summary when
        # its source evidence is stale, conflicting, out of scope or unitless.
        answer_text = (
            "维度贡献结果未通过当前范围、周期、数值和单位校验，未将其作为结论展示。"
        )
        answer_source = "analysis"

    revision_feedback = _critic_answer_revision_feedback(state)
    valid_revision_decomposition = None
    if revision_feedback and metric_decomposition:
        valid_revision_decomposition = _verified_decomposition_for_revision(state, metric_decomposition)
        # Do not repeat a stale or out-of-scope decomposition while responding
        # to a request for revised answer text.
        if valid_revision_decomposition is None:
            metric_decomposition = None
            decomposition_text = ""
            answer_text = analysis_result.get("summary", answer_text)

    revision_text = ""
    revision_evidence_ids: list[str] = []
    revision_status = "not_requested"
    revision_kind: list[str] = []
    if revision_feedback:
        revision_status = "bounded_or_limited"
        if revision_feedback["trend_requested"]:
            if valid_revision_decomposition:
                revision_text = _bounded_period_comparison_text(valid_revision_decomposition)
                revision_evidence_ids = list(map(str, valid_revision_decomposition["evidence_ids"]))
                revision_kind.append("verified_period_comparison")
                revision_status = "revised"
            elif validated_contribution_cards:
                revision_text = _dimension_contribution_text(
                    validated_contribution_cards[0], feedback_revision=True,
                )
                revision_evidence_ids = list(map(str, validated_contribution_cards[0]["evidence_ids"]))
                revision_kind.append("verified_dimension_contribution_comparison")
                revision_status = "revised"
            else:
                revision_text = (
                    "本轮已通过校验的证据不足以支持补充趋势变化描述，因此未扩展该结论。"
                )
                revision_kind.append("unsupported_trend_request_bounded")

        causal_readiness = (
            state.get("causal_readiness")
            or (analysis_result.get("causal_readiness") if isinstance(analysis_result, dict) else None)
            or {}
        )
        causal_is_verified = (
            isinstance(causal_readiness, dict)
            and causal_readiness.get("causal_effect_estimated") is True
        )
        if revision_feedback["causal_requested"] and not causal_is_verified:
            boundary = "现有已验证证据不能确认业务原因或因果关系，本次不补充因果结论。"
            if boundary not in answer_text and boundary not in revision_text:
                revision_text = f"{revision_text}\n{boundary}" if revision_text else boundary
            revision_kind.append("causal_request_bounded")

        state["answer_revision"] = {
            "status": revision_status,
            "feedback_categories": revision_feedback["categories"],
            "applied_changes": revision_kind,
            "evidence_ids": revision_evidence_ids,
            "feedback_used_as_evidence": False,
        }
        if revision_text:
            answer_text = f"{answer_text}\n\n{revision_text}" if answer_text else revision_text
    else:
        state["answer_revision"] = {
            "status": "not_requested",
            "feedback_categories": [],
            "applied_changes": [],
            "evidence_ids": [],
            "feedback_used_as_evidence": False,
        }

    grounding_status = "deterministic"
    metric_result = (tool_result.get("result") or {}) if isinstance(tool_result, dict) else {}
    can_ground_data = answer_source == "tool_result" and metric_result.get("status") == "ok"
    # WP0's Answer whitelist excludes ordinary RAG.  Knowledge-only answers
    # therefore keep the MetricDefinition node's retrieved text/citations as
    # its output; only a controlled Registry definition may enter the Answer
    # grounding LLM call.
    can_ground_knowledge = (
        answer_source == "metric_context"
        and bool(state.get("metric_authoritative_definition"))
    )
    if can_ground_data or can_ground_knowledge:
        if can_ground_data:
            evidence_payload = {
                "query_result": tool_result,
                "evidence_bundles": [
                    bundle for bundle in state.get("evidence_bundles", [])
                    if bundle.get("source_type") in {"mcp", "explorer"}
                ],
            }
        else:
            evidence_payload = {
                "authoritative_registry_definition": state.get("metric_authoritative_definition"),
            }
        answer_text, grounding_status = _grounded_narrative(state, evidence_payload, answer_text)
        try:
            from app.observability import record_event
            record_event("answer.grounding", {
                "trace_id": state.get("trace_id"),
                "answer_source": answer_source,
                "grounding_status": grounding_status,
                "evidence_count": (
                    len(state.get("evidence_bundles", [])) if can_ground_data
                    else int(bool(state.get("metric_authoritative_definition")))
                ),
            })
        except Exception:
            pass
    state["answer_grounding"] = {"status": grounding_status, "source": answer_source}

    # A model narrative can omit or misstate an explicitly requested missing
    # status. Render this bounded trend from the actual tool result instead.
    trend_payload = ((tool_result.get("result") or {}).get("trend_result") or {}) if isinstance(tool_result, dict) else {}
    if answer_source == "tool_result" and trend_payload and "缺失" in str(state.get("user_message") or ""):
        answer_text = _summarize_metric_query(tool_result) + "\n\n" + _trend_missing_status(tool_result)
        state["answer_grounding"] = {"status": "deterministic_trend_missing_status", "source": answer_source}

    semantic_note = _semantic_interpretation_text(state)
    if semantic_note:
        answer_text = semantic_note + "\n\n" + answer_text

    premise_direction_note = _confirmed_premise_direction_text(state)
    if premise_direction_note and "方向核验：" not in answer_text:
        answer_text = f"{premise_direction_note}\n\n{answer_text}" if answer_text else premise_direction_note

    skipped_metrics = (state.get("metric_planning_decision") or {}).get("skipped_metrics", [])
    if skipped_metrics:
        answer_text += "\n\n本次分析已根据你的选择跳过：" + "、".join(str(item) for item in skipped_metrics) + "。"

    # 追加 analysis_result 中的限制说明
    if display_limitations:
        answer_text += "\n\n" + "\n".join(f"⚠️ {lim}" for lim in display_limitations)
    if token_budget_limited:
        answer_text += "\n\nToken 预算耗尽后已停止后续模型请求和数据查询。"

    # 追加聊天文字与面板默认范围的差异说明
    analysis_context_note = tool_result.get("analysis_context_note", "")
    if analysis_context_note:
        answer_text = analysis_context_note + "\n\n" + answer_text

    # Phase 7: 追加历史上下文（非信任，仅参考）
    historical_claims = state.get("historical_claims_context", [])
    if historical_claims:
        claim_texts = [c.get("value", "")[:80] for c in historical_claims[:2] if c.get("value")]
        if claim_texts:
            answer_text += "\n\n📋 上轮分析结论参考:\n" + "\n".join(f"· {t}" for t in claim_texts)
            answer_text += "\n（以上为本轮分析前已知结论，本轮结论以上方数值为准。）"

    # ================================================================
    # 第二步：写入主答案字段
    # ================================================================

    # Adaptive mode is eligible to save the optional LLM Critic only for a
    # narrow class of answers: a registered decomposition whose user-facing
    # answer is rendered entirely from its validated card.  Do not retain
    # model/free-form summaries, presentation add-ons, or a prior Critic
    # revision on this path; those require semantic review instead.
    adaptive_rendered_decomposition = (
        state.get("critic_llm_review_mode") == "adaptive"
        and registered_numeric_decomposition
        and isinstance(metric_decomposition, dict)
        and bool(decomposition_text)
        and not revision_feedback
        and not state.get("errors")
    )
    if adaptive_rendered_decomposition:
        answer_text = decomposition_text
        state["adaptive_answer_contract"] = _registered_decomposition_answer_contract(metric_decomposition)
    else:
        # A missing contract is fail-closed in the adaptive policy.  Clear a
        # stale one before a generate-answer revision can reuse it.
        state.pop("adaptive_answer_contract", None)

    state["answer"] = answer_text
    state["final_answer"] = answer_text

    # ================================================================
    # 第三步：构建结构化展示结论（AnalysisPresentation）
    # ================================================================

    bundles = state.get("evidence_bundles", [])

    facts, fact_eids = _make_facts_from_tool(tool_result, bundles)
    changes, change_eids = _make_changes_from_tool(tool_result, bundles)
    query_results = state.get("query_results") or []
    typed_no_data = (
        str(((query_results[0].get("result") or {}).get("status") or ""))
        if len(query_results) == 1 and isinstance(query_results[0], dict)
        else ""
    )
    if typed_no_data in {"not_found", "insufficient_data"} and not facts and not changes:
        no_data_limitation = (
            "受控指标查询已执行并通过数据绑定检查，但该指标和期间没有可用数值；未生成数值事实。"
        )
        if no_data_limitation not in display_limitations:
            display_limitations.append(no_data_limitation)
    # The answer model may omit units even when the bound ChangeCard has them.
    # Add a deterministic note from the same Registry contract so the prose
    # cannot leave a percentage-point delta ambiguous. Unknown units stay
    # unknown; no label is inferred from the values.
    for change in changes:
        metric_id = str(change.get("metric_name") or "")
        if not metric_id or not change.get("evidence_ids"):
            continue
        answer_text, unit_note_added = ensure_comparison_unit_note(answer_text, metric_id)
        if unit_note_added:
            state["answer"] = answer_text
            state["final_answer"] = answer_text
    if state.get("task_type") == "anomaly_analysis" and (
        raw_additive_contributions or validated_contribution_cards
    ):
        retained_non_additive = [
            item for item in raw_contributions
            if not (isinstance(item, dict) and item.get("method") == "additive_change_v1")
        ]
        contribution_analysis = {
            **analysis_result,
            "contribution": [*validated_contribution_cards, *retained_non_additive],
        }
    else:
        contribution_analysis = analysis_result
    contributions, contrib_eids = _make_contributions(
        contribution_analysis, bundles, state.get("tool_calls") or [],
    )
    unrenderable_contribution = any(
        not _is_validated_dimension_contribution_card(item, validated_contribution_cards)
        for item in contributions
    )
    if unrenderable_contribution:
        safe_limitation = "部分贡献卡不满足已验证的维度贡献契约，未将其展开为数值明细。"
        if safe_limitation not in display_limitations:
            display_limitations.append(safe_limitation)
    limitations = display_limitations

    # 确定 status
    clarification_required = (
        isinstance(tool_result.get("result"), dict)
        and tool_result["result"].get("status") == "clarification_required"
    )
    is_unsupported = (
        isinstance(tool_result.get("result"), dict)
        and tool_result["result"].get("status") == "unsupported"
    )
    projection_unsupported = projection_failure.get("status") == "unsupported"
    has_non_budget_errors = any(
        not (isinstance(error, dict) and error.get("code") == "token_budget_exceeded")
        for error in (state.get("errors") or [])
    )
    projection_data_gap_only = (
        state.get("terminal_code") == "DATA_INSUFFICIENT"
        and all(
            not isinstance(error, dict)
            or error.get("code") in {"token_budget_exceeded", "contribution_projection_failed"}
            for error in (state.get("errors") or [])
        )
    )
    diagnostic_terminal_status = (
        diagnostic_terminal_presentation_status(analysis_result)
        if state.get("task_type") == "anomaly_analysis"
        or (state.get("registered_decomposition_request") or {}).get("status") == "ready"
        else None
    )
    if clarification_required:
        presentation_status = "clarification_required"
    elif projection_unsupported:
        presentation_status = "partial" if facts or changes else "unsupported"
    elif state.get("unavailable_requested_dimensions"):
        # Valid trend facts may still be shown, but the explicitly requested
        # attribution was not completed and must never look like a full answer.
        presentation_status = "partial"
    elif (state.get("metric_batch_execution") or {}).get("status") == "partial_unavailable":
        presentation_status = "partial"
    elif is_unsupported and not facts and not changes:
        presentation_status = "unsupported"
    elif projection_data_gap_only:
        presentation_status = "partial"
    elif has_non_budget_errors or _has_incomplete_comparison(tool_result) or diagnostic_terminal_status == "failed" or (
        (state.get("metric_batch_execution") or {}).get("status") == "execution_failed"
    ):
        presentation_status = "failed"
    elif state.get("terminal_code") == "DATA_INSUFFICIENT":
        # A typed missing-data result may come from an ordinary registered
        # comparison, outside the diagnostic subgraph. Preserve its terminal
        # limitation in the user-facing status instead of rendering it as a
        # completed numeric analysis.
        presentation_status = "partial"
    elif token_budget_limited or diagnostic_terminal_status == "partial":
        presentation_status = "partial"
    else:
        presentation_status = "completed"

    # ================================================================
    # 辅助展示卡片：漏斗 / 联合分析 / 评论文本
    # 这些查询在受控范围内自动执行，结果写入展示但不断主流程。
    # 必须在 all_eids / evidence_refs 之前执行，确保其 evidence
    # 能进入 presentation.evidence。
    # ================================================================

    scope = state.get("controlled_analysis_scope")
    scope_active = bool(scope and state.get("use_current_explorer_context"))
    scope_sidecars_enabled = _scope_sidecars_enabled(state, scope_active)

    # 漏斗数据
    funnels: list[dict] = []
    funnel_eids: list[str] = []
    if scope_sidecars_enabled:
        try:
            from app.data.repositories.explorer import query_scoped_order_funnel
            from app.schemas.explorer import ControlledAnalysisScope
            s = ControlledAnalysisScope.model_validate(scope)
            funnel_result = query_scoped_order_funnel(s)
            raw_funnel = funnel_result.get("funnel_data", [])
            explorer_eid = _find_evidence_id(bundles, "explorer", "data-explorer")
            eids = [explorer_eid] if explorer_eid else []
            funnel_eids = list(eids)
            funnels = [
                {"date": day["date"], "stages": day["stages"], "evidence_ids": list(eids)}
                for day in raw_funnel
            ]
        except Exception:
            pass

    # 联合分析
    joint_clusters: list[dict] = []
    jc_eids: list[str] = []
    if scope_sidecars_enabled:
        try:
            from app.data.repositories.joint_analysis import query_joint_cluster
            s = ControlledAnalysisScope.model_validate(scope)
            cluster = query_joint_cluster(s)
            if cluster.get("status") == "ok":
                jc_evidence_id = stable_evidence_id(
                    "explorer", "joint_cluster",
                    {"order_count": cluster.get("order_count"),
                     "avg_score": cluster.get("avg_score"),
                     "bad_review_rate": cluster.get("bad_review_rate"),
                     "on_time_rate": cluster.get("on_time_rate"),
                     "gross_value": cluster.get("gross_value"),
                     "freight_rate": cluster.get("freight_rate"),
                     "sample_size": cluster.get("sample_size"),
                     "seller_id": scope.get("seller_id"),
                     "start_date": scope.get("default_start_date"),
                     "end_date": scope.get("default_end_date"),
                     "query_version": "1.0"},
                )
                jc_bundle = {
                    "evidence_id": jc_evidence_id,
                    "source_type": "explorer",
                    "source_ref": "joint_cluster",
                    "content": {"seller_id": scope.get("seller_id"),
                                "start_date": scope.get("default_start_date"),
                                "end_date": scope.get("default_end_date"),
                                "order_count": cluster.get("order_count"),
                                "avg_score": cluster.get("avg_score"),
                                "bad_review_rate": cluster.get("bad_review_rate"),
                                "on_time_rate": cluster.get("on_time_rate"),
                                "gross_value": cluster.get("gross_value"),
                                "freight_rate": cluster.get("freight_rate"),
                                "sample_size": cluster.get("sample_size"),
                                "query_version": "1.0"},
                    "trusted": True,
                }
                bundles.append(jc_bundle)
                jc_eids = [jc_evidence_id]
                metrics = [
                    {"metric_name": "review_score_avg", "display_name": "平均评分",
                     "value": cluster.get("avg_score"), "unit": "score"},
                    {"metric_name": "bad_review_rate", "display_name": "差评率",
                     "value": cluster.get("bad_review_rate"), "unit": "%"},
                    {"metric_name": "on_time_delivery_rate", "display_name": "准时送达率",
                     "value": cluster.get("on_time_rate"), "unit": "%"},
                    {"metric_name": "gmv", "display_name": "成交总额",
                     "value": cluster.get("gross_value"), "unit": "BRL"},
                    {"metric_name": "freight_rate", "display_name": "运费率",
                     "value": cluster.get("freight_rate"), "unit": "%"},
                ]
                period_label = f"{scope.get('default_start_date')} 至 {scope.get('default_end_date')}"
                joint_clusters.append({
                    "question": "履约·评价·销量 联合分析",
                    "seller_id": scope.get("seller_id", ""),
                    "period": period_label,
                    "metrics": metrics,
                    "sample_size": cluster.get("sample_size", 0),
                    "evidence_ids": list(jc_eids),
                })
        except Exception:
            pass

    # 评论文本抽样
    review_samples_card: dict | None = None
    rt_eids: list[str] = []
    if scope_sidecars_enabled:
        try:
            from app.data.repositories.review_text import (
                query_review_comment_stats, query_review_samples,
            )
            s = ControlledAnalysisScope.model_validate(scope)
            stats = query_review_comment_stats(s.seller_id, s.default_start_date, s.default_end_date)
            samples = query_review_samples(
                s.seller_id, s.default_start_date, s.default_end_date,
                max_score=2, limit=5,
            )
            if stats.get("status") == "ok":
                rt_evidence_id = stable_evidence_id(
                    "explorer", "review_text",
                    {"total_orders": stats["total_orders"],
                     "comment_rate": stats["comment_rate"],
                     "avg_comment_length": stats["avg_comment_length"],
                     "bad_with_comment_count": stats["bad_with_comment_count"],
                     "sample_count": len(samples),
                     "seller_id": scope.get("seller_id"),
                     "start_date": scope.get("default_start_date"),
                     "end_date": scope.get("default_end_date"),
                     "query_version": "1.0"},
                )
                rt_bundle = {
                    "evidence_id": rt_evidence_id,
                    "source_type": "explorer",
                    "source_ref": "review_text",
                    "content": {"seller_id": scope.get("seller_id"),
                                "start_date": scope.get("default_start_date"),
                                "end_date": scope.get("default_end_date"),
                                "total_orders": stats["total_orders"],
                                "comment_rate": stats["comment_rate"],
                                "avg_comment_length": stats["avg_comment_length"],
                                "bad_with_comment_count": stats["bad_with_comment_count"],
                                "sample_count": len(samples)},
                    "trusted": True,
                }
                bundles.append(rt_bundle)
                rt_eids = [rt_evidence_id]
                review_samples_card = {
                    "summary": (
                        f"找到 {len(samples)} 条差评评论"
                        f"（共 {stats['total_orders']} 单，含文本率 {stats['comment_rate']}%）"
                    ),
                    "comment_rate": stats["comment_rate"],
                    "avg_length": stats["avg_comment_length"],
                    "samples": samples,
                    "evidence_ids": list(rt_eids),
                }
        except Exception:
            pass

    # 风险评分
    risk_card: dict | None = None
    risk_eids: list[str] = []
    if scope_sidecars_enabled:
        try:
            from app.data.repositories.risk_indicator import query_risk_score
            s = ControlledAnalysisScope.model_validate(scope)
            rr = query_risk_score(s.seller_id, s.default_start_date, s.default_end_date)
            if rr.get("status") == "ok":
                risk_eid = stable_evidence_id(
                    "explorer", "risk_score",
                    {"score": rr["score"], "level": rr["level"],
                     "dimensions": rr["dimensions"], "sample_size": rr["sample_size"],
                     "seller_id": scope.get("seller_id"),
                     "start_date": scope.get("default_start_date"),
                     "end_date": scope.get("default_end_date"),
                     "query_version": "1.0"},
                )
                bundles.append({
                    "evidence_id": risk_eid, "source_type": "explorer",
                    "source_ref": "risk_score",
                    "content": {"seller_id": scope.get("seller_id"),
                                "start_date": scope.get("default_start_date"),
                                "end_date": scope.get("default_end_date"),
                                "score": rr["score"], "level": rr["level"],
                                "dimensions": rr["dimensions"],
                                "sample_size": rr["sample_size"]},
                    "trusted": True,
                })
                risk_eids = [risk_eid]
                dims = rr.get("dimensions", {})
                risk_card = {
                    "score": rr["score"],
                    "level": rr["level"],
                    "dimensions": [
                        {"name": "差评率", "value": dims.get("bad_review_rate", 0),
                         "weight": 0.4, "contribution": round(dims.get("bad_review_rate", 0) * 0.4, 1)},
                        {"name": "延迟率", "value": dims.get("delay_rate", 0),
                         "weight": 0.3, "contribution": round(dims.get("delay_rate", 0) * 0.3, 1)},
                        {"name": "运费率", "value": dims.get("freight_rate", 0),
                         "weight": 0.3, "contribution": round(dims.get("freight_rate", 0) * 0.3, 1)},
                    ],
                    "sample_size": rr["sample_size"],
                    "evidence_ids": list(risk_eids),
                }
        except Exception:
            pass

    # Phase 8: 白名单复杂分析模板
    template_analyses: list[dict] = []

    # 收集所有被引用的 evidence_id（含辅助展示卡新增的）
    all_eids = list(dict.fromkeys(
        fact_eids + change_eids + contrib_eids
        + (metric_decomposition.get("evidence_ids", []) if metric_decomposition else [])
        + revision_evidence_ids
        + funnel_eids + jc_eids + rt_eids + risk_eids
    ))
    evidence_refs = _make_evidence_refs(all_eids, state)

    # 建议生成（仅在有贡献数据时触发）
    recommendations = _build_recommendations(analysis_result, all_eids, state)

    # 假设验证链映射
    hypothesis_chains = state.get("hypothesis_chains_by_metric") or {}
    if not hypothesis_chains:
        chain = analysis_result.get("hypothesis_chain", [])
        if chain:
            metric_name = (tool_result.get("result", {}) or {}).get("metric_result", {}) or {}
            mid = metric_name.get("metric_name", "")
            hypothesis_chains = {mid: chain} if mid else {}

    semantic_interpretation = (state.get("metric_planning_decision") or {}).get("semantic_interpretation")
    query_result = tool_result.get("result") if isinstance(tool_result, dict) else None
    multi_dimension_result = (
        query_result.get("multi_dimension_breakdown_result")
        if isinstance(query_result, dict) else None
    )
    if (
        isinstance(semantic_interpretation, dict)
        and isinstance(query_result, dict)
        and query_result.get("status") == "ok"
        and isinstance(multi_dimension_result, dict)
        and multi_dimension_result.get("status") == "ok"
    ):
        dimensions = [
            str(dimension)
            for dimension in multi_dimension_result.get("dimensions") or []
            if isinstance(dimension, str) and dimension
        ]
        if dimensions:
            semantic_interpretation = {**semantic_interpretation, "dimensions": dimensions}

    presentation = {
        "status": presentation_status,
        "scope": _make_scope(state, metric_decomposition, validated_contribution_cards),
        "semantic_interpretation": semantic_interpretation,
        "facts": facts,
        "changes": changes,
        "contributions": contributions,
        "metric_decomposition": metric_decomposition,
        "funnel": funnels,
        "joint_clusters": joint_clusters,
        "review_samples": review_samples_card,
        "risk": risk_card,
        "template_analyses": template_analyses,
        "limitations": limitations,
        "recommendations": recommendations,
        "hypothesis_chains_by_metric": hypothesis_chains,
        "evidence": evidence_refs,
        "trace_id": state.get("trace_id", ""),
    }
    state["analysis_presentation"] = presentation

    # ================================================================
    # 第四步：生成 Claims 契约供 critic 节点校验
    # ================================================================

    if clarification_required:
        state["claims"] = []
    else:
        claims: list[dict] = []
        for fact in facts:
            claims.append(Claim(
                claim_type="metric", value=str(fact.get("value", "")),
                source_evidence_ids=fact.get("evidence_ids", []),
            ).model_dump())
        for chg in changes:
            claims.append(Claim(
                claim_type="analysis",
                value=f"{chg.get('metric_name')} 变化 {chg.get('delta_value')} ({chg.get('delta_ratio')}%)",
                source_evidence_ids=chg.get("evidence_ids", []),
            ).model_dump())
        for contrib in contributions:
            if _is_validated_dimension_contribution_card(contrib, validated_contribution_cards):
                claims.append(Claim(
                    claim_type="analysis",
                    value=_dimension_contribution_text(contrib),
                    source_evidence_ids=contrib.get("evidence_ids", []),
                ).model_dump())
        if metric_decomposition:
            claims.append(Claim(
                claim_type="analysis",
                value=decomposition_text,
                source_evidence_ids=metric_decomposition.get("evidence_ids", []),
            ).model_dump())
        if revision_text and revision_evidence_ids:
            claims.append(Claim(
                claim_type="analysis",
                value=revision_text,
                source_evidence_ids=revision_evidence_ids,
            ).model_dump())
        for rec in recommendations:
            claims.append(Claim(
                claim_type="recommendation", value=rec.get("text", ""),
                source_evidence_ids=rec.get("evidence_ids", []),
            ).model_dump())
        # 辅助展示卡的 claims
        for jc in joint_clusters:
            claims.append(Claim(
                claim_type="analysis",
                value=f"联合分析: {jc.get('question')} ({jc.get('sample_size')} 单)",
                source_evidence_ids=jc.get("evidence_ids", []),
            ).model_dump())
        if review_samples_card:
            claims.append(Claim(
                claim_type="analysis",
                value=f"评论抽样: {review_samples_card.get('summary', '')}",
                source_evidence_ids=review_samples_card.get("evidence_ids", []),
            ).model_dump())
        if risk_card:
            claims.append(Claim(
                claim_type="analysis",
                value=f"风险评分: score={risk_card['score']}, level={risk_card['level']}",
                source_evidence_ids=risk_card.get("evidence_ids", []),
            ).model_dump())
        # 兜底
        if not claims and not (
            state.get("terminal_code") == "CAPABILITY_GAP"
            and any(item.get("registry_capability_checked") and item.get("sql_executed") is False
                    for item in state.get("decision_records", []))
        ):
            evidence_ids = [item["evidence_id"] for item in state.get("evidence_bundles", [])]
            claim_type = "analysis" if analysis_result else "definition" if metric_context else "metric"
            claims.append(Claim(
                claim_type=claim_type, value=answer_text, source_evidence_ids=evidence_ids,
            ).model_dump())
        state["claims"] = claims

    # ================================================================
    # 第五步：P0 Memory
    # ================================================================

    state_memories = state.get("analysis_memories", [])
    if state_memories:
        memory_lines = []
        for m in state_memories[:3]:
            mem_id = m.get("memory_id", "").split("_")[-1][:6]
            mem_text = m.get("insight_text", "")[:120]
            mem_trace = m.get("source_trace_id", "")
            memory_lines.append(f"  · {mem_text}（记忆 ID: mem_{mem_id}，来源 Trace: {mem_trace}）")
        state["answer"] += "\n\n参考的历史审核结论：\n" + "\n".join(memory_lines) + "\n（注：历史结论仅供上下文参考，不计入本轮数值依据）"
        state["final_answer"] = state["answer"]

    # ================================================================
    # 第六步：节点审计元信息
    # ================================================================

    state["current_agent"] = "generate_answer"
    state.setdefault("_revision_counts", {})
    state["_revision_counts"]["generate_answer"] = state["_revision_counts"].get("generate_answer", 0) + 1

    state.setdefault("node_history", []).append({
        "node": "generate_answer",
        "summary": f"Answer generated ({len(answer_text)} chars, source={answer_source})",
    })
    contract = (compilation.get("request_contract") or {}) if isinstance(compilation, dict) else {}
    record_contract_observation(state, "answer_output", {
        "answer_present": bool(state.get("answer")),
        "presentation_status": (state.get("analysis_presentation") or {}).get("status"),
        "presentation_evidence_ids": [
            str(item.get("evidence_id"))
            for item in ((state.get("analysis_presentation") or {}).get("evidence") or [])
            if isinstance(item, dict) and item.get("evidence_id")
        ],
        "contract_hash": contract.get("contract_hash"),
    })

    return state

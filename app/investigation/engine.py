"""Bounded shared investigation loop; adapters own selection and execution."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import date, timedelta
import hashlib
import json
import math
from typing import Literal, TypeVar

from app.investigation.contracts import (
    ActionResult,
    AttributionEvidence,
    AgentDecision,
    CalculationProgress,
    CandidateAction,
    CandidateSet,
    DataValidationResult,
    InvestigationProgress,
    InvestigationState,
    StopReason,
    UnresolvedBranch,
    WorkflowSelection,
    branch_share_for_priority,
)

from app.investigation.fingerprint import action_fingerprint
from app.investigation.stop_policy import (
    hard_stop_reason,
    progress_snapshot,
    soft_stop_blockers,
    terminal_progress_status,
)

MIN_ANOMALY_VALID_BUCKETS = 8


# Match the deterministic finding label used by the diagnostic summary. A
# root-metric dimension branch below this share is not an individually
# material candidate for another drill-down; keep it visible as low priority.
DIMENSION_FOLLOWUP_SHARE_THRESHOLD = 0.30

EvidenceT = TypeVar("EvidenceT", bound=AttributionEvidence)
_HIGH_VALUE_ACTION_SCORE_THRESHOLD = 0.55


def ensure_root_branch(state: InvestigationState) -> UnresolvedBranch:
    """Ensure a new investigation has one stable root node for its target."""
    existing = next((branch for branch in state.branches if branch.branch_type == "root_target"), None)
    if existing is not None:
        return existing
    digest = hashlib.sha256(
        f"{state.target.metric_id}:{state.target.scope_fingerprint}".encode("utf-8")
    ).hexdigest()[:16]
    root = UnresolvedBranch(
        branch_id=f"root_{digest}",
        branch_type="root_target",
        metric_id=state.target.metric_id,
        depth=0,
        status="open",
        label=f"{state.target.metric_id} 根分析目标",
    )
    state.branches.insert(0, root)
    if state.current_focus_branch_id is None:
        state.current_focus_branch_id = root.branch_id
    return root


def _reconciliation_ratio(total_delta: float, residual: float) -> float:
    if abs(total_delta) <= 1e-12:
        return 1.0 if abs(residual) <= 1e-12 else 0.0
    return max(0.0, min(1.0, 1.0 - abs(residual) / abs(total_delta)))


def _breakdown_reconciliation(values: dict) -> tuple[bool, bool]:
    """Return (numerically_valid, fully_explained) for additive group evidence.

    A top-K breakdown can be useful evidence while still having a coverage
    residual. Such a result must not close its parent investigation branch.
    """
    if values.get("attribution_status", "ok") != "ok":
        return False, False
    components = values.get("components")
    total_delta = values.get("total_delta")
    residual = values.get("residual_value")
    residual_type = values.get("residual_type")
    residual_explainable = values.get("residual_explainable")
    if (
        not isinstance(components, list)
        or not isinstance(total_delta, (int, float))
        or isinstance(total_delta, bool)
        or not isinstance(residual, (int, float))
        or isinstance(residual, bool)
        or residual_type not in {"coverage", "rounding", "model", "unknown"}
        or not isinstance(residual_explainable, bool)
    ):
        return False, False

    component_values: list[float] = []
    for component in components:
        if not isinstance(component, dict):
            return False, False
        value = component.get("delta_value", component.get("value"))
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False, False
        component_values.append(float(value))

    tolerance = max(0.03, len(component_values) * 0.005 + abs(float(total_delta)) * 1e-12)
    arithmetic_valid = abs(sum(component_values) + float(residual) - float(total_delta)) <= tolerance
    if not arithmetic_valid:
        return False, False

    has_unexpanded_other = any(
        str(component.get("dimension_value", component.get("component_id", ""))).strip().casefold()
        in {"other", "__other__"}
        and abs(float(component.get("delta_value", component.get("value", 0.0)))) > tolerance
        for component in components
    )
    fully_explained = (
        residual_type != "coverage"
        and residual_explainable
        and abs(float(residual)) <= tolerance
        and not has_unexpanded_other
    )
    return True, fully_explained


def _period_date(value: object) -> str | None:
    if isinstance(value, dict):
        value = value.get("end_date", value.get("end"))
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10]).isoformat()
    except ValueError:
        return None


def _cross_evidence_reconciliation_failure(
    state: InvestigationState,
    evidence: list[AttributionEvidence],
) -> tuple[str, list[str]] | None:
    """Compare compatible period aggregates before branching.

    The metric registry decides whether totals are additive. Daily additive
    trends are summed over complete comparison windows; a point at a period end
    is never treated as a multi-day total. Weighted ratios use registered
    aggregate evidence and never sum daily or grouped ratios.
    Evidence with a different scope, filters, definition, or comparison window
    is outside this check and cannot create a cross-evidence contradiction.
    """
    candidates = [*state.evidence, *evidence]
    if len(candidates) < 2:
        return None
    try:
        from app.metrics.registry import get_registry
        definition = get_registry().get(state.target.metric_id)
    except (KeyError, ValueError):
        return None
    aggregation = getattr(definition, "aggregation", None)
    scope_name = "seller" if any(
        key in state.target.canonical_scope for key in ("seller_id", "seller_ids")
    ) else "platform"
    registered_methods = set(definition.attribution.for_scope(scope_name).method_ids)
    if aggregation == "weighted_ratio":
        if (
            getattr(definition, "period_aggregation", None) is None
            or not registered_methods
        ):
            return None
    elif aggregation != "sum":
        return None

    tolerance = 0.03
    conflicts: set[str] = set()

    def matching_comparison(item: AttributionEvidence) -> tuple[dict, dict] | None:
        metadata = item.comparison.get("comparison")
        if not isinstance(metadata, dict):
            query = item.values.get("query")
            metadata = query.get("comparison") if isinstance(query, dict) else None
        if not isinstance(metadata, dict):
            return None
        filters = metadata.get("filters") or item.values.get("query_filters") or {}
        if not isinstance(filters, dict):
            return None
        current_point = _period_date(metadata.get("current_date"))
        baseline_point = _period_date(metadata.get("baseline_date"))
        period = (
            _period_date(metadata.get("current_start")) or current_point,
            _period_date(metadata.get("current_end") or metadata.get("current_date")),
            _period_date(metadata.get("baseline_start")) or baseline_point,
            _period_date(metadata.get("baseline_end") or metadata.get("baseline_date")),
        )
        expected = (
            state.target.current_start.isoformat(), state.target.current_end.isoformat(),
            state.target.baseline_start.isoformat() if state.target.baseline_start else None,
            state.target.baseline_end.isoformat() if state.target.baseline_end else None,
        )
        return (metadata, filters) if period == expected else None

    def comparison_snapshot(
        item: AttributionEvidence, *, require_registered_ratio: bool = False,
    ) -> tuple[float, float, float] | None:
        if require_registered_ratio:
            period_aggregation = definition.period_aggregation
            if item.evidence_type == "grouped_ratio_attribution":
                baseline_groups = item.values.get("baseline_groups")
                current_groups = item.values.get("current_groups")
                multiplier = item.values.get("multiplier")
                if (
                    item.method_id not in registered_methods
                    or not isinstance(baseline_groups, dict)
                    or not isinstance(current_groups, dict)
                    or isinstance(multiplier, bool)
                    or not isinstance(multiplier, (int, float))
                    or abs(float(multiplier) - period_aggregation.multiplier) > 1e-12
                ):
                    return None

                def registered_group_ratio(groups: dict) -> float | None:
                    numerator = denominator = 0.0
                    for row in groups.values():
                        if not isinstance(row, dict):
                            return None
                        n, d = row.get("numerator"), row.get("denominator")
                        if any(
                            isinstance(value, bool) or not isinstance(value, (int, float))
                            or not math.isfinite(float(value)) for value in (n, d)
                        ):
                            return None
                        numerator += float(n)
                        denominator += float(d)
                    if denominator <= 0:
                        return None
                    return numerator / denominator * period_aggregation.multiplier

                expected_baseline = registered_group_ratio(baseline_groups)
                expected_current = registered_group_ratio(current_groups)
                if expected_baseline is None or expected_current is None:
                    return None
                declared = (
                    item.values.get("baseline_value"), item.values.get("current_value"),
                    item.values.get("total_delta"),
                )
                if any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(float(value)) for value in declared
                ):
                    return None
                if (
                    abs(float(declared[0]) - expected_baseline) > tolerance
                    or abs(float(declared[1]) - expected_current) > tolerance
                    or abs(float(declared[2]) - (expected_current - expected_baseline)) > tolerance
                ):
                    return None
                return expected_baseline, expected_current, expected_current - expected_baseline

            registered = item.values.get("period_aggregation")
            if (
                not isinstance(registered, dict)
                or registered.get("kind") != period_aggregation.kind
                or registered.get("numerator_metric") != period_aggregation.numerator_metric
                or registered.get("denominator_metric") != period_aggregation.denominator_metric
                or not isinstance(registered.get("multiplier"), (int, float))
                or isinstance(registered.get("multiplier"), bool)
                or abs(float(registered["multiplier"]) - period_aggregation.multiplier) > 1e-12
            ):
                return None
            value_fields = (
                "baseline_numerator_total", "current_numerator_total",
                "baseline_denominator_total", "current_denominator_total",
            )
            totals = [item.values.get(key) for key in value_fields]
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) for value in totals
            ):
                return None
            baseline_num, current_num, baseline_den, current_den = map(float, totals)
            if baseline_den <= 0 or current_den <= 0:
                return None
            expected_baseline = baseline_num / baseline_den * period_aggregation.multiplier
            expected_current = current_num / current_den * period_aggregation.multiplier
            declared = item.values
            baseline = declared.get("baseline_value")
            current = declared.get("current_value")
            delta = declared.get("total_delta")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) for value in (baseline, current, delta)
            ):
                return None
            if (
                abs(float(baseline) - expected_baseline) > tolerance
                or abs(float(current) - expected_current) > tolerance
                or abs(float(delta) - (expected_current - expected_baseline)) > tolerance
            ):
                return None
            return expected_baseline, expected_current, expected_current - expected_baseline

        sources = [item.values]
        payload = item.values.get("result")
        if isinstance(payload, dict):
            sources.append(payload)
            for key in ("result", "comparison_result", "metric_comparison"):
                nested = payload.get(key)
                if isinstance(nested, dict):
                    sources.append(nested)
        for source in sources:
            baseline = source.get("baseline_value", source.get("baseline_total"))
            current = source.get("current_value", source.get("current_total"))
            delta = source.get("total_delta", source.get("delta_value", source.get("delta")))
            if all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value)) for value in (baseline, current)
            ):
                baseline, current = float(baseline), float(current)
                if not isinstance(delta, (int, float)) or isinstance(delta, bool) or not math.isfinite(float(delta)):
                    delta = current - baseline
                return baseline, current, float(delta)
        return None

    for contribution in candidates:
        if contribution.evidence_type not in {"dimension_contribution", "grouped_ratio_attribution"}:
            continue
        if aggregation == "weighted_ratio" and (
            contribution.evidence_type != "grouped_ratio_attribution"
            or contribution.method_id not in registered_methods
        ):
            continue
        if contribution.metric_id != state.target.metric_id:
            continue
        matched = matching_comparison(contribution)
        if matched is None:
            continue
        comparison, filters = matched
        current_end = _period_date(comparison.get("current_end") or comparison.get("current_date"))
        baseline_end = _period_date(comparison.get("baseline_end") or comparison.get("baseline_date"))
        if filters:
            # A filtered breakdown is a subset of the root trend. It is only
            # comparable when the trend explicitly declares the same filters.
            required_filters = filters
        else:
            required_filters = {}

        aggregate = comparison_snapshot(contribution)
        if aggregate is None:
            continue
        baseline_value, current_value, total_delta = aggregate

        for trend in candidates:
            if aggregation != "sum":
                # A ratio trend point is not sufficient to reconstruct a
                # multi-day weighted ratio. Use registered component-period
                # totals below, or skip if no trusted comparison exists.
                continue
            if trend.evidence_type != "trend" or trend.metric_id != contribution.metric_id:
                continue
            if trend.scope_fingerprint != contribution.scope_fingerprint:
                continue
            period = trend.comparison.get("period")
            if not isinstance(period, dict):
                continue
            period_start = _period_date(period.get("start_date") or period.get("start"))
            period_end = _period_date(period.get("end_date") or period.get("end"))
            baseline_start = state.target.baseline_start.isoformat() if state.target.baseline_start else None
            current_start = state.target.current_start.isoformat()
            if (
                not period_start or not period_end or not baseline_start
                or period_start > min(baseline_start, current_start)
                or period_end < max(baseline_end, current_end)
                or period.get("granularity") != "day"
            ):
                continue
            trend_filters = trend.values.get("query", {}).get("filters", {}) if isinstance(trend.values.get("query"), dict) else {}
            if trend_filters != required_filters:
                continue
            result = trend.values.get("result")
            points = result.get("points") if isinstance(result, dict) else None
            if not isinstance(points, list):
                continue
            if result.get("status") != "ok":
                continue
            by_date = {
                str(point.get("bucket", point.get("metric_date", "")))[:10]: point.get("value")
                for point in points if isinstance(point, dict)
            }

            def sum_complete_days(start: str, end: str) -> float | None:
                total = 0.0
                cursor, stop = date.fromisoformat(start), date.fromisoformat(end)
                while cursor <= stop:
                    value = by_date.get(cursor.isoformat())
                    if (
                        isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                    ):
                        return None
                    total += float(value)
                    cursor += timedelta(days=1)
                return total

            trend_baseline = sum_complete_days(baseline_start, baseline_end)
            trend_current = sum_complete_days(current_start, current_end)
            if trend_baseline is None or trend_current is None:
                continue
            trend_delta = trend_current - trend_baseline
            if (
                abs(float(trend_baseline) - baseline_value) > tolerance
                or abs(float(trend_current) - current_value) > tolerance
                or abs(trend_delta - total_delta) > tolerance
            ):
                conflicts.update((contribution.evidence_id, trend.evidence_id))

        # Some adapters provide a period-comparison result directly instead of
        # (or in addition to) a trend series. Reconcile only when it declares
        # the exact same comparison window, filters and authorized scope.
        for comparison_evidence in candidates:
            if comparison_evidence.evidence_id == contribution.evidence_id:
                continue
            if comparison_evidence.evidence_type not in {
                "period_comparison", "metric_comparison", "metric_observation",
                "grouped_ratio_attribution",
            }:
                continue
            if (
                comparison_evidence.metric_id != contribution.metric_id
                or comparison_evidence.scope_fingerprint != contribution.scope_fingerprint
            ):
                continue
            comparison_match = matching_comparison(comparison_evidence)
            if comparison_match is None or comparison_match[1] != filters:
                continue
            observed = comparison_snapshot(
                comparison_evidence,
                require_registered_ratio=aggregation == "weighted_ratio",
            )
            if observed is None:
                continue
            if any(abs(actual - expected) > tolerance for actual, expected in zip(observed, aggregate)):
                conflicts.update((contribution.evidence_id, comparison_evidence.evidence_id))
    if conflicts:
        return (
            "趋势与维度贡献在相同指标、范围、筛选和比较周期下无法对账；证据已保留，停止归因。",
            sorted(conflicts),
        )
    return None


def _distribution_failure(analysis: dict) -> tuple[str, str] | None:
    """Validate descriptive statistics as a time-bucket summary, not raw rows."""
    status = analysis.get("status")
    if status in {"not_found", "insufficient_data"}:
        return "DATA_INSUFFICIENT", "当前范围没有足够的有效时间桶用于分布统计。"
    if status != "ok":
        return "INVALID_DATA", "分布统计状态不是成功状态。"
    if analysis.get("distribution_basis") != "time_bucket_values":
        return "INVALID_DATA", "分布统计未声明其统计对象为时间桶指标值。"

    count = analysis.get("count")
    stats = analysis.get("statistics")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1 or not isinstance(stats, dict):
        return "INVALID_DATA", "分布统计缺少有效时间桶数量或 statistics 对象。"
    required = ("count", "mean", "std", "min", "q1", "median", "q3", "max")
    if any(name not in stats for name in required):
        return "INVALID_DATA", "分布统计缺少请求的描述统计量。"
    stat_count = stats.get("count")
    if not isinstance(stat_count, int) or isinstance(stat_count, bool) or stat_count != count:
        return "INVALID_DATA", "分布统计中的样本数与有效时间桶数量不一致。"

    numeric: dict[str, float] = {}
    for name in required[1:]:
        value = stats.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            return "INVALID_DATA", f"分布统计量 {name} 不是有限数值。"
        numeric[name] = float(value)
    if numeric["std"] < 0:
        return "INVALID_DATA", "分布标准差不得为负数。"
    tolerance = max(1e-9, max(abs(value) for value in numeric.values()) * 1e-9)
    if (numeric["min"] > numeric["q1"] + tolerance
            or numeric["q1"] > numeric["median"] + tolerance
            or numeric["median"] > numeric["q3"] + tolerance
            or numeric["q3"] > numeric["max"] + tolerance
            or numeric["mean"] < numeric["min"] - tolerance
            or numeric["mean"] > numeric["max"] + tolerance):
        return "INVALID_DATA", "分布统计量不满足最小值、分位数、中位数、均值和最大值的顺序约束。"
    for name in ("min_bucket", "max_bucket"):
        bucket = analysis.get(name)
        try:
            date.fromisoformat(bucket)
        except (TypeError, ValueError):
            return "INVALID_DATA", f"分布统计的 {name} 不是有效 ISO 日期。"
    return None


def _evidence_failure(action: object, evidence: AttributionEvidence) -> tuple[str, str] | None:
    """Validate action-specific evidence shape before it can enter investigation state."""
    action_type = getattr(action, "action_type", "")
    values = evidence.values
    if "residual_tolerance" in values:
        tolerance = values.get("residual_tolerance")
        if (not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool)
                or not math.isfinite(float(tolerance)) or float(tolerance) < 0):
            return "INVALID_DATA", "残差容差必须是有限且非负的数值。"
        residual_value = values.get("residual_value")
        if values.get("residual_explainable") is True and isinstance(residual_value, (int, float)):
            if isinstance(residual_value, bool) or not math.isfinite(float(residual_value)):
                return "INVALID_DATA", "可解释残差必须是有限数值。"
            if abs(float(residual_value)) > float(tolerance) + 1e-12:
                return "INVALID_DATA", "残差超出声明的解释容差。"
    if "residual_note" in values and not isinstance(values.get("residual_note"), str):
        return "INVALID_DATA", "残差说明必须是字符串。"

    if action_type in {"BREAKDOWN_DIMENSION", "DRILL_DOWN"}:
        method_id = getattr(action, "method_id", "")
        if method_id == "symmetric_mix_within_v1":
            if evidence.evidence_type != "grouped_ratio_attribution":
                return "INVALID_DATA", "比率分组动作返回了错误类型的 Evidence。"
            if values.get("attribution_status") != "ok" or values.get("complete") is not True:
                return "DATA_INSUFFICIENT", "比率分组事实不完整或未能完成确定性分解。"
            if values.get("method_id") != method_id or values.get("method_version") != "1.0":
                return "INVALID_DATA", "比率分组 Evidence 的方法版本与已选候选不一致。"
            if values.get("dimension") != getattr(action, "dimension", None):
                return "INVALID_DATA", "比率分组 Evidence 的维度与已选候选不一致。"
            baseline_groups, current_groups = values.get("baseline_groups"), values.get("current_groups")
            raw_components, groups = values.get("raw_components"), values.get("components")
            if not all(isinstance(item, dict) for item in (baseline_groups, current_groups)) or not isinstance(raw_components, list) or not isinstance(groups, list):
                return "INVALID_DATA", "比率分组 Evidence 缺少充分统计量或贡献明细。"
            try:
                from app.analysis.attribution_methods import symmetric_mix_within_attribution

                recomputed = symmetric_mix_within_attribution(
                    metric_id=evidence.metric_id,
                    baseline_groups=baseline_groups,
                    current_groups=current_groups,
                    multiplier=float(values.get("multiplier")),
                    unit=evidence.unit,
                )
                tolerance = 0.03
                total_delta = values.get("total_delta")
                residual = values.get("residual_value")
                baseline_value = values.get("baseline_value")
                current_value = values.get("current_value")
                if (
                    not isinstance(total_delta, (int, float)) or isinstance(total_delta, bool)
                    or not isinstance(residual, (int, float)) or isinstance(residual, bool)
                    or not isinstance(baseline_value, (int, float)) or isinstance(baseline_value, bool)
                    or not isinstance(current_value, (int, float)) or isinstance(current_value, bool)
                    or not all(math.isfinite(float(value)) for value in (total_delta, residual, baseline_value, current_value))
                    or abs(float(total_delta) - recomputed.total_delta) > tolerance
                    or abs(float(residual) - recomputed.residual_value) > tolerance
                    or abs(float(baseline_value) - recomputed.baseline_value) > tolerance
                    or abs(float(current_value) - recomputed.current_value) > tolerance
                    or values.get("residual_explainable") is not True
                ):
                    return "INVALID_DATA", "比率分组结果无法与后端重新计算的总变化及残差对账。"
                expected_effects: dict[str, float] = {}
                for component in recomputed.raw_components:
                    group = component.component_id.rsplit(":", 1)[0]
                    expected_effects[group] = expected_effects.get(group, 0.0) + component.value
                actual_effects: dict[str, float] = {}
                for item in groups:
                    key, value = item.get("dimension_value"), item.get("delta_value")
                    if not isinstance(key, str) or key in actual_effects or not isinstance(value, (int, float)) or isinstance(value, bool):
                        return "INVALID_DATA", "比率分组贡献包含重复维度或无效数值。"
                    actual_effects[key] = float(value)
                if set(actual_effects) != set(expected_effects) or any(
                    abs(actual_effects[key] - value) > tolerance for key, value in expected_effects.items()
                ):
                    return "INVALID_DATA", "比率分组贡献明细与确定性方法输出不一致。"
                expected_raw = {item.component_id: item.value for item in recomputed.raw_components}
                if any(not isinstance(item, dict) for item in raw_components):
                    return "INVALID_DATA", "比率分组原始组件格式错误。"
                actual_raw = {item.get("component_id"): item.get("value") for item in raw_components}
                if (
                    len(actual_raw) != len(raw_components)
                    or set(actual_raw) != set(expected_raw)
                    or any(
                    not isinstance(actual_raw[key], (int, float))
                    or isinstance(actual_raw[key], bool)
                    or not math.isfinite(float(actual_raw[key]))
                    or abs(float(actual_raw[key]) - value) > tolerance
                    for key, value in expected_raw.items()
                    )
                ):
                    return "INVALID_DATA", "比率分组原始组内/构成/进入退出项与后端重算不一致。"
            except (TypeError, ValueError, OverflowError):
                return "INVALID_DATA", "比率分组事实无法由注册的确定性方法复算。"
            return None
        valid, _ = _breakdown_reconciliation(values)
        if evidence.evidence_type != "dimension_contribution" or not valid:
            return "INVALID_DATA", "分组归因证据缺少有效对账字段，或分组贡献与残差无法对齐总变化。"
        return None

    if action_type == "DECOMPOSE_METRIC":
        raw = values.get("raw_components")
        display = values.get("display_components")
        factor_metrics = values.get("factor_metrics")
        delta = values.get("total_delta")
        residual = values.get("residual_value")
        if evidence.evidence_type != "metric_decomposition":
            return "INVALID_DATA", "分解动作返回了错误类型的 Evidence。"
        if values.get("residual_explainable") is not True:
            return "DATA_INSUFFICIENT", "分解残差尚未被方法口径解释。"
        if (
            not isinstance(raw, list) or not isinstance(display, list)
            or not isinstance(factor_metrics, list)
            or any(not isinstance(item, str) or not item for item in factor_metrics)
            or factor_metrics != getattr(action, "factor_metric_ids", None)
            or len(set(factor_metrics)) != len(factor_metrics)
            or len(display) != len(factor_metrics)
            or not isinstance(delta, (int, float)) or isinstance(delta, bool)
            or not isinstance(residual, (int, float)) or isinstance(residual, bool)
        ):
            return "INVALID_DATA", "分解证据缺少 raw/display components 或数值对账字段。"

        def sum_components(items: list[dict]) -> float | None:
            total = 0.0
            component_ids: set[str] = set()
            for item in items:
                if not isinstance(item, dict):
                    return None
                component_id = item.get("component_id")
                if not isinstance(component_id, str) or not component_id or component_id in component_ids:
                    return None
                component_ids.add(component_id)
                value = item.get("value")
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                    return None
                total += float(value)
            return total

        raw_sum, display_sum = sum_components(raw), sum_components(display)
        tolerance = max(0.03, abs(float(delta)) * 1e-9)
        if (
            raw_sum is None or display_sum is None
            or not math.isfinite(float(delta)) or not math.isfinite(float(residual))
            or abs(raw_sum + float(residual) - float(delta)) > tolerance
            or abs(display_sum + float(residual) - float(delta)) > tolerance
        ):
            return "INVALID_DATA", "raw/display 分解项未分别与总变化对账。"
        method_id = getattr(action, "method_id", "")
        raw_types = [item.get("component_type") for item in raw]
        display_types = [item.get("component_type") for item in display]
        if method_id == "product_shapley_v1":
            interactions = [
                item for item in raw if item.get("component_type") == "interaction_effect"
            ]
            interaction_delta = values.get("interaction_delta")
            if (
                values.get("display_method") != "shapley"
                or raw_types.count("factor_effect") != 2
                or len(interactions) != 1
                or len(raw) != 3
                or len(display) != 2
                or any(kind != "shapley_factor_effect" for kind in display_types)
                or not isinstance(interaction_delta, (int, float))
                or isinstance(interaction_delta, bool)
                or not math.isfinite(float(interaction_delta))
                or abs(float(interaction_delta) - float(interactions[0]["value"])) > tolerance
            ):
                return "INVALID_DATA", "product Shapley evidence lacks a valid interaction/display allocation."
        elif values.get("display_method") not in {"raw", None}:
            return "INVALID_DATA", "non-Shapley decomposition declares an unregistered display allocation."
        return None

    if action_type in {"ANALYZE_TREND", "CHECK_ANOMALY", "CHECK_DATA_QUALITY"}:
        expected_evidence_type = {
            "ANALYZE_TREND": "trend",
            "CHECK_ANOMALY": "time_series_anomaly",
            "CHECK_DATA_QUALITY": "time_series_quality",
        }[action_type]
        if evidence.evidence_type != expected_evidence_type:
            return "INVALID_DATA", "时间序列动作返回了错误类型的 Evidence。"
        if action_type == "ANALYZE_TREND":
            payload = values.get("result")
            if not isinstance(payload, dict) or not isinstance(payload.get("points"), list):
                return "INVALID_DATA", "趋势结果缺少结构化 points。"
            if payload.get("status") not in {"ok", "partial"}:
                return "DATA_INSUFFICIENT", f"趋势查询状态为 {payload.get('status') or 'unknown'}。"
            seen_buckets: set[str] = set()
            numeric_count = 0
            for point in payload["points"]:
                if not isinstance(point, dict):
                    return "INVALID_DATA", "趋势点格式无效。"
                bucket = str(point.get("bucket", point.get("metric_date", "")))[:10]
                try:
                    date.fromisoformat(bucket)
                except ValueError:
                    return "INVALID_DATA", "趋势点日期格式无效。"
                if bucket in seen_buckets:
                    return "INVALID_DATA", "趋势结果包含重复时间桶。"
                seen_buckets.add(bucket)
                value = point.get("value")
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                    numeric_count += 1
            if numeric_count == 0:
                return "DATA_INSUFFICIENT", "趋势结果没有可用的有限数值点。"
            return None

        analysis = values.get("series_analysis")
        if not isinstance(analysis, dict):
            return "INVALID_DATA", "时间序列检查缺少结构化分析结果。"
        status = analysis.get("status")
        if status == "insufficient_data":
            return "DATA_INSUFFICIENT", "时间序列点数不足，无法完成本项检查。"
        if status == "invalid_data":
            return "INVALID_DATA", "时间序列质量检查发现无效或重复的输入数据。"
        if status != "ok":
            return "CAPABILITY_GAP", f"时间序列检查状态为 {status or 'unknown'}。"
        if action_type == "CHECK_ANOMALY" and int(analysis.get("numeric_bucket_count") or 0) < MIN_ANOMALY_VALID_BUCKETS:
            return "DATA_INSUFFICIENT", f"异常检测需要至少 {MIN_ANOMALY_VALID_BUCKETS} 个有效时间桶。"
        if action_type == "CHECK_DATA_QUALITY":
            check_id = getattr(action, "check_id", None)
            if analysis.get("check_id") != check_id:
                return "INVALID_DATA", "数据质量 Evidence 与服务端候选指定的检查类型不一致。"
            raw_result = values.get("result")
            if not isinstance(raw_result, dict):
                return "INVALID_DATA", "数据质量 Evidence 缺少原始时间桶结果，无法复算。"
            try:
                from app.investigation.series_analysis import check_series_quality

                recomputed = check_series_quality(raw_result, str(check_id))
            except (TypeError, ValueError):
                return "INVALID_DATA", "数据质量 Evidence 无法由确定性检查器复算。"
            if recomputed != analysis:
                return "INVALID_DATA", "数据质量统计与原始时间桶结果复算不一致。"
        return None

    if action_type == "ANALYZE_DISTRIBUTION":
        if evidence.evidence_type != "distribution_summary":
            return "INVALID_DATA", "分布分析动作返回了错误类型的 Evidence。"
        analysis = values.get("series_analysis")
        if not isinstance(analysis, dict):
            return "INVALID_DATA", "分布分析缺少结构化统计结果。"
        return _distribution_failure(analysis)

    if action_type == "ANALYZE_ENTITY_DISTRIBUTION":
        if evidence.evidence_type != "entity_distribution_summary":
            return "INVALID_DATA", "实体分布动作返回了错误类型的 Evidence。"
        analysis = values.get("series_analysis")
        if not isinstance(analysis, dict):
            return "INVALID_DATA", "实体分布缺少结构化统计结果。"
        if analysis.get("distribution_basis") != "order_entity" or analysis.get("entity") != "delivered_seller_order":
            return "INVALID_DATA", "实体分布 Evidence 的统计基准不匹配。"
        if analysis.get("status") == "not_found" or int(analysis.get("sample_size") or 0) == 0:
            return "DATA_INSUFFICIENT", "范围内没有可用的订单级运费样本。"
        if analysis.get("status") != "ok":
            return "CAPABILITY_GAP", f"实体分布状态为 {analysis.get('status') or 'unknown'}。"
        stats = analysis.get("statistics")
        required_stats = {"mean", "std", "min", "q1", "median", "q3", "max"}
        if not isinstance(stats, dict) or not required_stats <= set(stats):
            return "INVALID_DATA", "订单分布统计字段不完整。"
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in stats.values()):
            return "INVALID_DATA", "订单分布包含非有限数值。"
        return None

    if action_type == "ANALYZE_ASSOCIATION":
        from app.analysis.association_catalog import get_association_pair

        if evidence.evidence_type != "registered_association":
            return "INVALID_DATA", "关联分析动作返回了错误类型的 Evidence。"
        analysis = values.get("series_analysis")
        if not isinstance(analysis, dict):
            return "INVALID_DATA", "关联 Evidence 缺少结构化统计结果。"
        if analysis.get("pair_id") != getattr(action, "pair_id", None):
            return "INVALID_DATA", "关联 Evidence 与服务端注册的变量对不匹配。"
        if analysis.get("target_metric") != getattr(action, "metric_id", None):
            return "INVALID_DATA", "关联 Evidence 的目标指标与已授权指标不匹配。"
        pair = get_association_pair(str(getattr(action, "pair_id", "")))
        if pair is None or pair.target_metric_id != getattr(action, "metric_id", None):
            return "DENIED", "关联 Evidence 不属于当前服务器注册目录。"
        if evidence.method_id != pair.method_id:
            return "INVALID_DATA", "关联 Evidence 的方法版本与注册目录不一致。"
        result_method = analysis.get("method_id")
        if result_method is not None and result_method != pair.method_id:
            return "INVALID_DATA", "关联结果声明的方法与注册目录不一致。"
        if analysis.get("analysis_contract") != pair.evidence_contract():
            return "INVALID_DATA", "关联 Evidence 缺少或篡改了服务端注册的统计口径。"
        population = analysis.get("population_orders")
        eligible = analysis.get("eligible_orders")
        excluded = analysis.get("excluded_missing_pair_fields")
        if (
            any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (population, eligible, excluded))
            or eligible + excluded != population
            or not analysis.get("missing_policy")
            or not analysis.get("limitations")
        ):
            return "INVALID_DATA", "关联 Evidence 的样本、缺失值口径或限制说明不完整。"
        if analysis.get("status") == "insufficient_data":
            return "DATA_INSUFFICIENT", "关联分析的有效样本量不足。"
        if analysis.get("status") == "over_limit":
            return "CAPABILITY_GAP", "关联分析样本超过受控计算上限，未对截断样本估计。"
        if analysis.get("status") == "undefined":
            return "DATA_INSUFFICIENT", "关联分析至少一个变量在有效样本中没有变化。"
        if analysis.get("status") != "ok":
            return "CAPABILITY_GAP", f"关联分析状态为 {analysis.get('status') or 'unknown'}。"
        if int(analysis.get("sample_size") or 0) < pair.minimum_sample_size:
            return "DATA_INSUFFICIENT", "关联分析有效样本少于注册的最低要求。"
        if pair.kind == "binary":
            interval = analysis.get("risk_difference_ci_95")
            effect = analysis.get("risk_difference")
        else:
            interval = analysis.get("spearman_rho_ci_95_approx")
            effect = analysis.get("spearman_rho")
        if (
            not isinstance(interval, list) or len(interval) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in interval)
            or isinstance(effect, bool) or not isinstance(effect, (int, float)) or not math.isfinite(float(effect))
            or not interval[0] <= effect <= interval[1]
        ):
            return "INVALID_DATA", "关联分析缺少有效效应量或 95% 不确定区间。"
        return None

    return None


def _base_validation_status(evidence: list[AttributionEvidence]) -> str:
    """Classify accepted Evidence completeness independently from execution success."""
    for item in evidence:
        statuses = [item.data_quality.get("status")]
        payload = item.values.get("result")
        if isinstance(payload, dict):
            statuses.append(payload.get("status"))
            nested = payload.get("result")
            if isinstance(nested, dict):
                statuses.append(nested.get("status"))
        if "partial" in statuses:
            return "incomplete_result"
    return "valid"


def _set_selected_branch_status(
    state: InvestigationState,
    candidate: CandidateAction,
    status: str,
) -> None:
    branch_id = candidate.branch_id
    if not branch_id:
        return
    state.branches = [
        branch.model_copy(update={"status": status}) if branch.branch_id == branch_id else branch
        for branch in state.branches
    ]


def _record_unresolved_branches(state: InvestigationState, candidate: CandidateAction, observed: list[AttributionEvidence]) -> None:
    """Promote large, evidence-backed group deltas to bounded child branches.

    Only typed server evidence can create branches. Free-form LLM observations
    never influence a future dimension filter.
    """
    action = candidate.action
    if getattr(action, "action_type", "") == "DECOMPOSE_METRIC":
        for evidence in observed:
            if evidence.metric_id != action.metric_id or evidence.scope_fingerprint != state.target.scope_fingerprint:
                continue
            factor_metrics = evidence.values.get("factor_metrics")
            components = evidence.values.get("display_components")
            total_delta = evidence.values.get("total_delta")
            if (
                not isinstance(factor_metrics, list)
                or factor_metrics != getattr(action, "factor_metric_ids", None)
                or not isinstance(components, list)
                or len(components) != len(factor_metrics)
                or not isinstance(total_delta, (int, float))
            ):
                continue
            component_values: dict[str, float] = {}
            component_units: dict[str, str] = {}
            positional_values: list[float] = []
            positional_units: list[str | None] = []
            for component in components:
                if not isinstance(component, dict):
                    component_values = {}
                    break
                component_id = component.get("component_id")
                value = component.get("value")
                if (
                    not isinstance(component_id, str)
                    or not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                ):
                    component_values = {}
                    break
                positional_values.append(float(value))
                raw_unit = component.get("unit")
                positional_units.append(str(raw_unit) if raw_unit else None)
                # Product/Shapley display components use canonical factor IDs.
                if component_id in factor_metrics:
                    component_values[component_id] = float(value)
                    if raw_unit:
                        component_units[component_id] = str(raw_unit)
            if state.root_effect_value is None:
                state.root_effect_value = float(total_delta)
            denominator = abs(float(state.root_effect_value))
            for index, factor_metric in enumerate(factor_metrics):
                effect = component_values.get(factor_metric)
                effect_unit = component_units.get(factor_metric)
                if effect is None and index < len(positional_values):
                    # Ratio methods use numerator_effect/denominator_effect IDs;
                    # their registered action order is the authoritative map.
                    effect = positional_values[index]
                    effect_unit = positional_units[index]
                if effect is None:
                    continue
                branch_id = f"branch_{evidence.evidence_id}_factor_{factor_metric}"
                if any(branch.branch_id == branch_id for branch in state.branches):
                    continue
                root_share = abs(effect) / denominator if denominator > 1e-12 else None
                state.branches.append(UnresolvedBranch(
                    branch_id=branch_id,
                    parent_branch_id=(
                        candidate.branch_id if any(
                            branch.branch_id == candidate.branch_id for branch in state.branches
                        ) else None
                    ),
                    parent_evidence_id=evidence.evidence_id,
                    branch_type="metric_effect",
                    metric_id=factor_metric,
                    dimension="__metric_factor__",
                    dimension_value=factor_metric,
                    effect_value=effect,
                    effect_unit=effect_unit or evidence.unit or None,
                    root_attributed_effect_value=effect,
                    root_effect_share=root_share,
                    effect_share_of_root=root_share,
                    depth=0,
                    inherited_filters={},
                    status="open",
                    label=f"{factor_metric} 对 {action.metric_id} 变化的注册分解贡献",
                ))
            if component_values and candidate.branch_id:
                # The validated factor decomposition resolves the root-level
                # metric split; the newly opened factor branches represent
                # downstream questions and remain independently investigable.
                _set_selected_branch_status(state, candidate, "explained")
        return

    dimension = getattr(action, "dimension", None)
    if not dimension:
        return
    inherited = dict(getattr(action, "inherited_filters", {}) or {})
    parent_evidence_id = getattr(action, "parent_evidence_id", None)
    # The candidate carries the server-selected branch identity. Prefer it so
    # two unquantified root branches with the same metric/filters cannot cause
    # evidence from one investigation focus to close the other branch.
    parent_branch = next((
        branch for branch in state.branches
        if branch.branch_id == candidate.branch_id
        and branch.status in {"open", "investigating"}
        and branch.metric_id == action.metric_id
    ), None)
    if parent_branch is None:
        parent_branch = next((
            branch for branch in state.branches
            if branch.status in {"open", "investigating"}
            and branch.metric_id == action.metric_id
            and branch.inherited_filters == inherited
            and (parent_evidence_id is None or branch.parent_evidence_id == parent_evidence_id)
        ), None)
    depth = (parent_branch.depth + 1) if parent_branch else 1
    if depth > state.drill_depth_limit:
        return
    parent_updated = False
    for evidence in observed:
        if evidence.metric_id != action.metric_id or evidence.scope_fingerprint != state.target.scope_fingerprint:
            continue
        components = evidence.values.get("components")
        total_delta = evidence.values.get("total_delta")
        if not isinstance(components, list) or not components or not isinstance(total_delta, (int, float)):
            continue
        numerically_valid, fully_explained = _breakdown_reconciliation(evidence.values)
        if not numerically_valid:
            continue
        if parent_branch is not None and fully_explained and not parent_updated:
            state.branches = [
                branch.model_copy(update={"status": "explained"}) if branch.branch_id == parent_branch.branch_id else branch
                for branch in state.branches
            ]
            parent_updated = True
        if state.root_effect_value is None:
            state.root_effect_value = float(total_delta)
        denominator = abs(float(state.root_effect_value))
        root_metric_effect = action.metric_id == state.target.metric_id
        coverage_tolerance = max(
            0.03,
            len(components) * 0.005 + abs(float(total_delta)) * 1e-12,
        )
        for component in components:
            if not isinstance(component, dict):
                continue
            value = component.get("delta_value", component.get("value"))
            group = component.get("dimension_value", component.get("component_id"))
            if not isinstance(value, (int, float)) or group is None or str(group).strip().casefold() == "":
                continue
            is_other = str(group).strip().casefold() in {"other", "__other__"}
            if is_other:
                if abs(float(value)) > coverage_tolerance:
                    branch_id = f"branch_{evidence.evidence_id}_{dimension}_other"
                    if not any(branch.branch_id == branch_id for branch in state.branches):
                        root_share = (
                            abs(float(value)) / denominator
                            if root_metric_effect and denominator > 1e-12 else None
                        )
                        state.branches.append(UnresolvedBranch(
                            branch_id=branch_id,
                            parent_branch_id=(
                                candidate.branch_id if any(
                                    branch.branch_id == candidate.branch_id for branch in state.branches
                                ) else None
                            ),
                            parent_evidence_id=evidence.evidence_id,
                            branch_type="dimension_contribution",
                            metric_id=action.metric_id,
                            dimension=str(dimension),
                            dimension_value="Other",
                            effect_value=float(value),
                            effect_unit=evidence.unit or None,
                            root_attributed_effect_value=(float(value) if root_metric_effect else None),
                            root_effect_share=root_share,
                            effect_share_of_root=root_share,
                            depth=depth,
                            inherited_filters=inherited,
                            status="blocked",
                            label=f"{dimension}=Other（聚合覆盖，无法安全下钻）",
                        ))
                continue
            branch_id = f"branch_{evidence.evidence_id}_{dimension}_{str(group)[:32]}"
            if any(branch.branch_id == branch_id for branch in state.branches):
                continue
            root_share = (
                abs(float(value)) / denominator
                if root_metric_effect and denominator > 1e-12 else None
            )
            state.branches.append(UnresolvedBranch(
                branch_id=branch_id,
                parent_branch_id=(
                    candidate.branch_id if any(
                        branch.branch_id == candidate.branch_id for branch in state.branches
                    ) else None
                ),
                parent_evidence_id=evidence.evidence_id,
                branch_type="dimension_contribution",
                metric_id=action.metric_id,
                dimension=str(dimension),
                dimension_value=str(group),
                effect_value=float(value),
                effect_unit=evidence.unit or None,
                root_attributed_effect_value=(float(value) if root_metric_effect else None),
                root_effect_share=root_share,
                effect_share_of_root=root_share,
                depth=depth,
                inherited_filters=dict(getattr(action, "inherited_filters", {})) | {str(dimension): str(group)},
                status=(
                    "open"
                    if root_metric_effect
                    and root_share is not None
                    and root_share >= DIMENSION_FOLLOWUP_SHARE_THRESHOLD
                    else "low_priority"
                    if root_metric_effect
                    and root_share is not None
                    else "open"
                ),
                explanation_level=("statistical_contribution" if root_metric_effect else None),
                label=f"{dimension}={group}",
            ))
        residual = evidence.values.get("residual_value")
        if (
            evidence.values.get("residual_type") == "coverage"
            and isinstance(residual, (int, float))
            and abs(float(residual)) > coverage_tolerance
        ):
            branch_id = f"branch_{evidence.evidence_id}_{dimension}_coverage_residual"
            if not any(branch.branch_id == branch_id for branch in state.branches):
                root_share = (
                    abs(float(residual)) / denominator
                    if root_metric_effect and denominator > 1e-12 else None
                )
                state.branches.append(UnresolvedBranch(
                    branch_id=branch_id,
                    parent_branch_id=(
                        candidate.branch_id if any(
                            branch.branch_id == candidate.branch_id for branch in state.branches
                        ) else None
                    ),
                    parent_evidence_id=evidence.evidence_id,
                    branch_type="dimension_contribution",
                    metric_id=action.metric_id,
                    dimension=str(dimension),
                    dimension_value="__coverage_residual__",
                    effect_value=float(residual),
                    effect_unit=evidence.unit or None,
                    root_attributed_effect_value=(float(residual) if root_metric_effect else None),
                    root_effect_share=root_share,
                    effect_share_of_root=root_share,
                    depth=depth,
                    inherited_filters=inherited,
                    status="blocked",
                    label=f"{dimension} 分组覆盖残差（数据不完整，无法归属具体分组）",
                ))


def prepare_candidates(
    state: InvestigationState,
    candidates: Iterable[CandidateAction],
    *,
    current_period: dict,
    baseline_period: dict | None = None,
    authorization_blocked: Callable[[CandidateAction], bool | str] | None = None,
    authorization_fingerprint: str | None = None,
) -> list[CandidateAction]:
    """Prepare one bounded server-owned candidate set for either surface.

    Chat invokes this once per LangGraph decision turn; the report adapter uses
    the same transition in its loop. The helper owns fingerprint de-duplication,
    authorization pruning, top-K enforcement and hard-stop state.
    """
    if state.stop_reason is not None:
        state.candidate_actions = []
        state.candidate_set = None
        return []
    state.candidate_actions = []
    state.candidate_set = None
    if state.remaining_budget <= 0:
        state.stop_reason = StopReason(
            code="BUDGET_EXHAUSTED", detail="调查动作预算已用尽。", initiator="budget"
        )
        state.candidate_actions = []
        return []
    executed = set(state.executed_fingerprints)
    unique: list[CandidateAction] = []
    seen_fingerprints: set[str] = set()
    for candidate in candidates:
        fingerprint = action_fingerprint(
            candidate.action,
            current_period=current_period,
            baseline_period=baseline_period,
            canonical_scope=state.target.canonical_scope,
            metric_definition_version=state.target.metric_definition_version,
            method_version=state.target.method_version,
        )
        if fingerprint in executed or fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        branch = next((
            item for item in state.branches
            if getattr(candidate.action, "parent_evidence_id", None)
            and item.parent_evidence_id == candidate.action.parent_evidence_id
        ), None)
        unique.append(candidate.model_copy(update={
            "action_fingerprint": fingerprint,
            "branch_id": candidate.branch_id or (branch.branch_id if branch else None),
        }))
    association_already_attempted = any(
        result.action_type == "ANALYZE_ASSOCIATION"
        for result in state.action_results
    )
    if association_already_attempted:
        before_count = len(unique)
        unique = [
            candidate for candidate in unique
            if candidate.action.action_type != "ANALYZE_ASSOCIATION"
        ]
        if len(unique) != before_count:
            state.observations.append({
                "event": "association_pair_search_limited",
                "detail": (
                    "一次调查最多分析一个注册关联变量对；如需比较其他变量关系，"
                    "请另建分析，并将结果视为探索性发现。"
                ),
            })
            if not unique:
                state.stop_reason = StopReason(
                    code="NO_HIGH_VALUE_ACTION",
                    detail=(
                        "本次调查已分析一个注册关联变量对；未执行第二个关系，"
                        "以避免未校正的多重比较。"
                    ),
                    initiator="backend",
                )
                state.candidate_actions = []
                return []
    reason = hard_stop_reason(
        remaining_budget=state.remaining_budget,
        candidates_available=len(unique),
    )
    if reason:
        state.stop_reason = reason
        return []
    permitted: list[CandidateAction] = []
    for candidate in unique:
        rejection = authorization_blocked(candidate) if authorization_blocked else False
        if rejection:
            reason = rejection if isinstance(rejection, str) else "candidate blocked by authorization policy"
            item = {"candidate_id": candidate.candidate_id, "reason": reason}
            if item not in state.policy_rejections:
                state.policy_rejections.append(item)
            continue
        permitted.append(candidate)
    if not permitted:
        state.stop_reason = StopReason(
            code="AUTHORIZATION_BLOCKED", detail="所有候选动作均未通过授权。", initiator="policy"
        )
        state.progress.authorization_blocked = True
        state.candidate_actions = []
        return []
    # Authorization filtering must happen before Top-K pruning. Otherwise
    # unauthorized high-ranked actions can hide eligible candidates below K.
    permitted.sort(key=lambda item: (
        -item.features.rule_score,
        item.features.estimated_query_cost,
        item.action_fingerprint,
    ))
    permitted = permitted[:6]
    state.candidate_actions = permitted
    candidate_set_payload = {
        "investigation_state_version": state.state_version,
        "scope_fingerprint": state.target.scope_fingerprint,
        "authorization_fingerprint": authorization_fingerprint,
        "candidate_ids": [item.candidate_id for item in permitted],
        "action_fingerprints": [item.action_fingerprint for item in permitted],
    }
    candidate_set_serialized = json.dumps(
        candidate_set_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    candidate_set_id = "set_" + hashlib.sha256(candidate_set_serialized.encode("utf-8")).hexdigest()[:32]
    state.candidate_set = CandidateSet(
        candidate_set_id=candidate_set_id,
        investigation_state_version=state.state_version,
        scope_fingerprint=state.target.scope_fingerprint,
        authorization_fingerprint=authorization_fingerprint,
        candidate_ids=[item.candidate_id for item in permitted],
    )
    state.progress = progress_snapshot(
        open_branch_count=sum(branch.status in {"open", "investigating"} for branch in state.branches),
        highest_open_branch_share=max(
            (max(0.0, branch_share_for_priority(branch)) for branch in state.branches if branch.status in {"open", "investigating"}),
            default=0.0,
        ),
        max_remaining_action_score=max(item.features.rule_score for item in permitted),
        remaining_budget=state.remaining_budget,
        capability_gap=state.progress.capability_gap,
        authorization_blocked=False,
    )
    return permitted


def select_candidate(
    state: InvestigationState,
    candidate_id: str | AgentDecision | None,
    *,
    candidate_set_id: str | None = None,
) -> CandidateAction | None:
    """Resolve the selector's opaque ID against this round's server catalog."""
    decision: AgentDecision | None = candidate_id if isinstance(candidate_id, AgentDecision) else None
    if decision is not None:
        candidate_set_id = decision.candidate_set_id
        candidate_id = decision.candidate_id
        if decision.decision == "soft_stop":
            if (
                state.candidate_set is None
                or candidate_set_id != state.candidate_set.candidate_set_id
                or state.candidate_set.investigation_state_version != state.state_version
                or state.candidate_set.scope_fingerprint != state.target.scope_fingerprint
            ):
                state.stop_reason = StopReason(
                    code="NO_HIGH_VALUE_ACTION", detail="Agent 决策引用了过期候选集。", initiator="backend"
                )
                return None
            reason = decision.soft_stop_reason or "NO_HIGH_VALUE_ACTION"
            blockers = soft_stop_blockers(state) if reason == "ENOUGH_EVIDENCE" else []
            highest = max(state.candidate_actions, key=lambda item: item.features.rule_score, default=None)
            if blockers:
                state.observations.append({
                    "event": "soft_stop_rejected", "requested_stop": reason, "blockers": blockers,
                })
                state.stop_reason = None
                open_branch_ids = {
                    branch.branch_id for branch in state.branches
                    if branch.status in {"open", "investigating"}
                }
                fallback = next(
                    (item for item in state.candidate_actions if item.branch_id in open_branch_ids),
                    highest,
                )
                candidate_id = fallback.candidate_id if fallback else None
            elif (
                reason == "NO_HIGH_VALUE_ACTION" and highest
                and highest.features.rule_score >= _HIGH_VALUE_ACTION_SCORE_THRESHOLD
            ):
                state.observations.append({
                    "event": "soft_stop_rejected", "requested_stop": reason,
                    "blockers": ["high_value_candidate_remains"],
                })
                state.stop_reason = None
                candidate_id = highest.candidate_id
            else:
                state.stop_reason = StopReason(
                    code=reason, detail=decision.rationale or "Agent 请求停止继续调查。", initiator="agent"
                )
                return None
    if state.candidate_set is not None:
        expected_set_id = state.candidate_set.candidate_set_id
        if candidate_set_id is None or candidate_set_id != expected_set_id:
            state.stop_reason = StopReason(
                code="NO_HIGH_VALUE_ACTION", detail="Agent 决策引用了过期候选集。", initiator="backend"
            )
            return None
        if (
            state.candidate_set.investigation_state_version != state.state_version
            or state.candidate_set.scope_fingerprint != state.target.scope_fingerprint
        ):
            state.stop_reason = StopReason(
                code="NO_HIGH_VALUE_ACTION", detail="调查状态已变化，旧候选集已失效。", initiator="backend"
            )
            return None
    if candidate_id is None:
        if state.stop_reason is None:
            state.stop_reason = StopReason(
                code="ENOUGH_EVIDENCE", detail="调查 Agent 选择停止继续取证。", initiator="agent"
            )
        return None
    selected = next((item for item in state.candidate_actions if item.candidate_id == candidate_id), None)
    if selected is None:
        state.stop_reason = StopReason(
            code="NO_HIGH_VALUE_ACTION",
            detail="Agent 未选择本轮服务端提供的合法候选动作。",
            initiator="backend",
        )
    elif selected.branch_id:
        state.current_focus_branch_id = selected.branch_id
        state.branches = [
            branch.model_copy(update={"status": "investigating"})
            if branch.branch_id == selected.branch_id and branch.status == "open"
            else branch
            for branch in state.branches
        ]
    if selected is not None:
        state.state_version += 1
    return selected


def record_action_result(
    state: InvestigationState,
    candidate: CandidateAction,
    observed: Iterable[EvidenceT],
    *,
    current_period: dict,
    baseline_period: dict | None = None,
    decision_source: Literal["agent", "workflow_fallback", "backend", "unknown"] = "unknown",
    failure_detail: str = "合法动作没有产生可用证据。",
    failure_code: Literal[
        "DATA_INSUFFICIENT", "CAPABILITY_GAP", "EXECUTION_FAILED",
        "INVALID_DATA", "AUTHORIZATION_BLOCKED",
    ] | None = "DATA_INSUFFICIENT",
    failure_context: dict | None = None,
) -> bool:
    """Validate and record one action's typed evidence and spend one budget unit."""
    if not state.branches and state.stop_reason is None:
        ensure_root_branch(state)
    if candidate.branch_id is None:
        focus = next((
            branch for branch in state.branches
            if branch.branch_id == state.current_focus_branch_id
            and branch.status in {"open", "investigating"}
        ), None)
        if focus is None:
            focus = next((branch for branch in state.branches if branch.branch_type == "root_target"), None)
        if focus is not None:
            candidate = candidate.model_copy(update={"branch_id": focus.branch_id})
    evidence = list(observed)
    fingerprint = action_fingerprint(
        candidate.action,
        current_period=current_period,
        baseline_period=baseline_period,
        canonical_scope=state.target.canonical_scope,
        metric_definition_version=state.target.metric_definition_version,
        method_version=state.target.method_version,
    )

    def append_outcome(
        status: Literal["completed", "capability_gap", "denied", "failed"],
        *, code: str | None = None, detail: str = "",
        evidence_ids: list[str] | None = None,
        validation_checks: list[dict] | None = None,
        failure_context: dict | None = None,
    ) -> None:
        action = candidate.action
        action_id = f"action_{len(state.action_results) + 1}"
        validation_status = {
            "INVALID_DATA": "invalid_data",
            "DATA_INSUFFICIENT": "incomplete_result",
        }.get(code or "", "unknown")
        calculation_progress = CalculationProgress(
            action_id=action_id,
            data_quality_status=validation_status,
        )
        state.action_results.append(ActionResult(
            action_id=action_id,
            candidate_id=candidate.candidate_id,
            candidate_set_id=(state.candidate_set.candidate_set_id if state.candidate_set else None),
            decision_source=decision_source,
            action_type=action.action_type,
            metric_id=action.metric_id,
            dimension=getattr(action, "dimension", None),
            method_id=getattr(action, "method_id", None),
            parent_evidence_id=getattr(action, "parent_evidence_id", None),
            branch_id=candidate.branch_id,
            action_fingerprint=fingerprint,
            status=status,
            failure_code=code,
            failure_detail=detail,
            failure_context=failure_context,
            calculation_progress=calculation_progress,
            validation_result=DataValidationResult(
                status=validation_status,
                 checks=(validation_checks if validation_checks is not None else
                         ([{"name": "evidence_contract", "status": "failed" if code else "not_run", "detail": detail}] if detail else [])),
                failed_fields_or_groups=[],
            ),
            evidence_ids=evidence_ids or [],
        ))
        state.calculation_progress = calculation_progress
        state.state_version += 1

    if candidate.action_fingerprint and candidate.action_fingerprint != fingerprint:
        _set_selected_branch_status(state, candidate, "blocked")
        append_outcome(
            "denied", code="AUTHORIZATION_BLOCKED",
            detail="候选动作绑定的目标或范围已变化。",
        )
        state.stop_reason = StopReason(
            code="AUTHORIZATION_BLOCKED", detail="候选动作绑定的目标或范围已变化。", initiator="policy"
        )
        state.progress.authorization_blocked = True
        return False
    if fingerprint in state.executed_fingerprints:
        _set_selected_branch_status(state, candidate, "open")
        append_outcome(
            "denied", code="DUPLICATE_ACTION",
            detail="查询语义已执行，拒绝重复入账。",
        )
        state.stop_reason = StopReason(
            code="NO_HIGH_VALUE_ACTION", detail="查询语义已执行，拒绝重复入账。", initiator="backend"
        )
        return False
    if state.remaining_budget <= 0:
        _set_selected_branch_status(state, candidate, "open")
        state.stop_reason = StopReason(
            code="BUDGET_EXHAUSTED", detail="调查动作预算已用尽。", initiator="budget"
        )
        return False

    if not evidence:
        _set_selected_branch_status(state, candidate, "open")
        state.executed_fingerprints.append(fingerprint)
        # Empty observations are a normal, typed terminal outcome for queries
        # that complete without usable rows. Older adapters may omit the
        # failure code; never let that leak through as an invalid StopReason.
        effective_code = failure_code or "DATA_INSUFFICIENT"
        effective_detail = failure_detail or "No validated evidence was produced; investigation cannot continue."
        result_status = (
            "capability_gap" if effective_code == "CAPABILITY_GAP"
            else "denied" if effective_code == "AUTHORIZATION_BLOCKED"
            else "failed"
        )
        append_outcome(
            result_status, code=effective_code, detail=effective_detail,
            failure_context=failure_context,
        )
        state.remaining_budget = max(0, state.remaining_budget - 1)
        state.stop_reason = StopReason(
            code=effective_code,
            detail=effective_detail,
            initiator="backend",
            failure_context=failure_context,
        )
        return False
    for item in evidence:
        if item.scope_fingerprint != state.target.scope_fingerprint:
            _set_selected_branch_status(state, candidate, "blocked")
            state.executed_fingerprints.append(fingerprint)
            append_outcome(
                "denied", code="AUTHORIZATION_BLOCKED",
                detail="Evidence 范围与调查目标不一致。",
            )
            state.remaining_budget = max(0, state.remaining_budget - 1)
            state.stop_reason = StopReason(
                code="AUTHORIZATION_BLOCKED", detail="Evidence 范围与调查目标不一致。", initiator="policy"
            )
            state.progress.authorization_blocked = True
            return False

    existing_evidence_ids = {item.evidence_id for item in state.evidence}
    incoming_evidence_ids: set[str] = set()
    expected_method = getattr(candidate.action, "method_id", None)
    invalid_detail = None
    invalid_code = "INVALID_DATA"
    for item in evidence:
        if item.source_action_id != candidate.candidate_id:
            invalid_detail = "Evidence 来源动作与服务端选中的候选动作不一致。"
            break
        if item.metric_id != candidate.action.metric_id:
            invalid_detail = "Evidence 指标与服务端候选动作不一致。"
            break
        if expected_method and item.method_id != expected_method:
            invalid_detail = "Evidence 方法与服务端候选动作声明不一致。"
            break
        failure = _evidence_failure(candidate.action, item)
        if failure:
            invalid_code, invalid_detail = failure
            break
        if item.evidence_id in existing_evidence_ids or item.evidence_id in incoming_evidence_ids:
            invalid_detail = "Evidence ID 重复，拒绝重复入账。"
            break
        incoming_evidence_ids.add(item.evidence_id)
    if invalid_detail:
        _set_selected_branch_status(state, candidate, "open")
        state.executed_fingerprints.append(fingerprint)
        append_outcome("failed", code=invalid_code, detail=invalid_detail)
        state.remaining_budget = max(0, state.remaining_budget - 1)
        state.stop_reason = StopReason(
            code=invalid_code, detail=invalid_detail, initiator="backend"
        )
        return False

    state.executed_fingerprints.append(fingerprint)
    state.evidence.extend(evidence)
    cross_failure = _cross_evidence_reconciliation_failure(state, evidence)
    if cross_failure is not None:
        detail, conflicting_ids = cross_failure
        # Keep the source evidence for audit, but block the investigation tree
        # so no earlier branch label survives as a conclusion after conflict.
        state.branches = [branch.model_copy(update={"status": "blocked"}) for branch in state.branches]
        append_outcome(
            "failed", code="INVALID_DATA", detail=detail,
            evidence_ids=[item.evidence_id for item in evidence],
            validation_checks=[{
                "name": "cross_evidence_reconciliation",
                "status": "failed",
                "detail": detail,
                "conflicting_evidence_ids": conflicting_ids,
                "aggregation_rule": "metric_registry",
            }],
        )
        state.remaining_budget = max(0, state.remaining_budget - 1)
        state.stop_reason = StopReason(code="INVALID_DATA", detail=detail, initiator="backend")
        return False
    _record_unresolved_branches(state, candidate, evidence)
    # An action can return valid evidence without resolving its branch (for
    # example, a trend check). Do not leave the branch looking actively in-flight.
    if candidate.branch_id:
        state.branches = [
            branch.model_copy(update={"status": "open"})
            if branch.branch_id == candidate.branch_id and branch.status == "investigating"
            else branch
            for branch in state.branches
        ]
    active_branch = max(
        (branch for branch in state.branches if branch.status in {"open", "investigating"}),
        key=lambda item: (branch_share_for_priority(item), -item.depth),
        default=None,
    )
    state.current_focus_branch_id = active_branch.branch_id if active_branch else None
    value_records = [item.values for item in evidence]
    component_values = [
        component for values in value_records
        for component in values.get("components", [])
        if isinstance(component, dict)
    ]
    total_abs = sum(abs(float(item.get("delta_value", item.get("value", 0)))) for item in component_values)
    concentration = (
        max((abs(float(item.get("delta_value", item.get("value", 0)))) for item in component_values), default=0.0) / total_abs
        if total_abs > 1e-12 else None
    )
    residual = next((
        values.get("residual_value") for values in value_records
        if isinstance(values.get("residual_value"), (int, float))
        and not isinstance(values.get("residual_value"), bool)
    ), None)
    residual_type = next((values.get("residual_type") for values in value_records if values.get("residual_type") in {"coverage", "rounding", "model", "unknown"}), "unknown")
    residual_explainable = next((values.get("residual_explainable") for values in value_records if isinstance(values.get("residual_explainable"), bool)), False)
    residual_tolerance = next((
        float(values["residual_tolerance"])
        for values in value_records
        if isinstance(values.get("residual_tolerance"), (int, float))
        and not isinstance(values.get("residual_tolerance"), bool)
        and math.isfinite(float(values["residual_tolerance"]))
        and float(values["residual_tolerance"]) >= 0
    ), None)
    residual_note = next((
        values["residual_note"] for values in value_records
        if isinstance(values.get("residual_note"), str) and values["residual_note"]
    ), "")
    impacts = [item.values["total_delta"] for item in evidence if isinstance(item.values.get("total_delta"), (int, float))]
    computed_ratio = next((
        float(values["reconciliation_ratio"])
        for values in value_records
        if isinstance(values.get("reconciliation_ratio"), (int, float))
        and not isinstance(values.get("reconciliation_ratio"), bool)
        and 0.0 <= float(values["reconciliation_ratio"]) <= 1.0
    ), None)
    if computed_ratio is None:
        for values in value_records:
            total_delta = values.get("total_delta")
            value_residual = values.get("residual_value")
            if (
                isinstance(total_delta, (int, float)) and not isinstance(total_delta, bool)
                and isinstance(value_residual, (int, float)) and not isinstance(value_residual, bool)
            ):
                computed_ratio = _reconciliation_ratio(float(total_delta), float(value_residual))
                break
    action = candidate.action
    action_id = f"action_{len(state.action_results) + 1}"
    calculation_progress = CalculationProgress(
        action_id=action_id,
        reconciliation_ratio=computed_ratio,
        residual_value=next((
            float(values["residual_value"])
            for values in value_records
            if isinstance(values.get("residual_value"), (int, float))
            and not isinstance(values.get("residual_value"), bool)
        ), None),
        residual_type=residual_type,
        residual_explainable=residual_explainable,
        residual_tolerance=residual_tolerance,
        residual_note=residual_note,
        data_quality_status=_base_validation_status(evidence),
    )
    validation_status = _base_validation_status(evidence)
    validation_result = DataValidationResult(
        status=validation_status,
        checks=[{
            "name": "evidence_contract",
            "status": "passed" if validation_status == "valid" else "incomplete",
            "detail": (
                "Evidence 结构和口径校验通过。"
                if validation_status == "valid"
                else "查询结果为 partial；保留为部分证据，但不可用于证明调查已充分。"
            ),
        }],
    )
    state.action_results.append(ActionResult(
        action_id=action_id,
        candidate_id=candidate.candidate_id,
        candidate_set_id=(state.candidate_set.candidate_set_id if state.candidate_set else None),
        decision_source=decision_source,
        action_type=action.action_type,
        metric_id=action.metric_id,
        dimension=getattr(action, "dimension", None),
        method_id=getattr(action, "method_id", None),
        parent_evidence_id=getattr(action, "parent_evidence_id", None),
        branch_id=candidate.branch_id,
        action_fingerprint=fingerprint,
        status="completed",
        observed_impact=sum(float(item) for item in impacts) if impacts else None,
        concentration=concentration,
        explained_share=next((float(values["explained_share"]) for values in value_records if isinstance(values.get("explained_share"), (int, float))), None),
        residual_value=float(residual) if residual is not None else None,
        residual_type=residual_type,
        residual_explainable=residual_explainable,
        residual_tolerance=residual_tolerance,
        residual_note=residual_note,
        data_quality=next((values.get("data_quality") for values in value_records if isinstance(values.get("data_quality"), dict)), {}),
        calculation_progress=calculation_progress,
        validation_result=validation_result,
        evidence_ids=[item.evidence_id for item in evidence],
    ))
    state.calculation_progress = calculation_progress
    state.state_version += 1
    state.observations.append({
        "action_id": state.action_results[-1].action_id,
        "candidate_id": candidate.candidate_id,
        "evidence_ids": [item.evidence_id for item in evidence],
        "observed_impact": state.action_results[-1].observed_impact,
    })
    state.remaining_budget = max(0, state.remaining_budget - 1)
    state.progress = progress_snapshot(
        open_branch_count=sum(branch.status in {"open", "investigating"} for branch in state.branches),
        highest_open_branch_share=max((max(0.0, branch_share_for_priority(branch)) for branch in state.branches if branch.status in {"open", "investigating"}), default=0.0),
        max_remaining_action_score=0.0,
        remaining_budget=state.remaining_budget,
        capability_gap=state.progress.capability_gap,
        authorization_blocked=state.progress.authorization_blocked,
    )
    return True


def continue_after_candidate_data_gap(
    state: InvestigationState,
    action_result: ActionResult | None = None,
) -> bool:
    """Release only a candidate-scoped data-gap stop when budget remains.

    Both the report loop and the LangGraph chat adapter use this transition so
    an empty result is consistently recorded as an unsuccessful action without
    being mistaken for a global investigation failure.
    """
    result = action_result or (state.action_results[-1] if state.action_results else None)
    if (
        result is None
        or result.failure_code != "DATA_INSUFFICIENT"
        or state.remaining_budget <= 0
    ):
        return False
    state.observations.append({
        "event": "candidate_data_insufficient",
        "candidate_id": result.candidate_id,
        "action_id": result.action_id,
        "detail": result.failure_detail,
        "continuing_with_alternatives": True,
    })
    state.stop_reason = None
    return True


def run_investigation(
    state: InvestigationState,
    *,
    candidate_factory: Callable[[InvestigationState], list[CandidateAction]],
    selector: Callable[[list[CandidateAction], InvestigationState], WorkflowSelection | AgentDecision | None],
    executor: Callable[[CandidateAction, InvestigationState], Iterable[EvidenceT]],
    current_period: dict,
    baseline_period: dict | None = None,
    authorization_blocked: Callable[[CandidateAction], bool] | None = None,
    progress_callback: Callable[[InvestigationProgress, CandidateAction | None], None] | None = None,
    max_actions: int = 6,
    max_drill_depth: int = 3,
) -> InvestigationState:
    """Execute one selected, server-issued action per round under hard bounds.

    LLM selectors must return a candidate-set-bound ``AgentDecision``.
    Trusted deterministic callbacks use ``WorkflowSelection``. Raw strings are
    rejected so a model response cannot be silently rebound to a newer set.
    """
    if not 0 <= max_actions <= 6 or not 0 <= max_drill_depth <= 3:
        raise ValueError("investigation limits exceed the v1 hard ceiling")
    if not state.branches and not state.action_results and state.stop_reason is None:
        ensure_root_branch(state)
    hard_action_limit = min(max_actions, state.remaining_budget)

    def publish_progress(max_score: float, current: CandidateAction | None = None) -> None:
        state.progress = progress_snapshot(
            open_branch_count=sum(branch.status in {"open", "investigating"} for branch in state.branches),
            highest_open_branch_share=max(
                (max(0.0, branch_share_for_priority(branch)) for branch in state.branches if branch.status in {"open", "investigating"}),
                default=0.0,
            ),
            max_remaining_action_score=max_score,
            remaining_budget=state.remaining_budget,
            capability_gap=state.progress.capability_gap,
            authorization_blocked=state.progress.authorization_blocked,
        )
        if progress_callback:
            progress_callback(state.progress, current)

    for _ in range(hard_action_limit):
        permitted = prepare_candidates(
            state, candidate_factory(state), current_period=current_period,
            baseline_period=baseline_period, authorization_blocked=authorization_blocked,
        )
        if not permitted:
            break
        publish_progress(max(item.features.rule_score for item in permitted))
        selection = selector(permitted, state)
        decision_source: Literal["agent", "workflow_fallback", "backend", "unknown"] = "unknown"
        if isinstance(selection, AgentDecision):
            decision_source = "agent"
        elif isinstance(selection, WorkflowSelection):
            decision_source = "workflow_fallback"
        elif selection is None:
            if (
                state.stop_reason is not None
                and state.stop_reason.code == "ENOUGH_EVIDENCE"
                and state.stop_reason.initiator == "agent"
                and state.candidate_set is not None
            ):
                selection = AgentDecision(
                    candidate_set_id=state.candidate_set.candidate_set_id,
                    decision="soft_stop", soft_stop_reason="ENOUGH_EVIDENCE",
                    rationale=state.stop_reason.detail,
                )
                decision_source = "agent"
            else:
                decision_source = "backend"
                if state.stop_reason is None:
                    state.stop_reason = StopReason(
                        code="NO_HIGH_VALUE_ACTION",
                        detail="Selector 未提供动作；调查由后端安全停止。",
                        initiator="backend",
                    )
                break
        else:
            state.observations.append({
                "event": "selector_output_rejected",
                "reason": "raw_candidate_id_is_not_a_typed_agent_decision_or_workflow_selection",
                "candidate_set_id": state.candidate_set.candidate_set_id if state.candidate_set else None,
            })
            state.stop_reason = StopReason(
                code="EXECUTION_FAILED",
                detail="Selector 必须返回绑定当前候选集的 AgentDecision 或服务端 WorkflowSelection。",
                initiator="policy",
            )
            break
        selected = select_candidate(
            state,
            selection.candidate_id if isinstance(selection, WorkflowSelection) else selection,
            candidate_set_id=(
                state.candidate_set.candidate_set_id if state.candidate_set else None
            ),
        )
        if selected is None and state.stop_reason and state.stop_reason.code == "ENOUGH_EVIDENCE":
            blockers = soft_stop_blockers(state)
            if blockers:
                state.observations.append({
                    "event": "soft_stop_rejected",
                    "requested_stop": "ENOUGH_EVIDENCE",
                    "blockers": blockers,
                })
                state.stop_reason = None
                open_branch_ids = {
                    branch.branch_id for branch in state.branches
                    if branch.status in {"open", "investigating"}
                }
                forced_candidate = next(
                    (item for item in permitted if item.branch_id in open_branch_ids),
                    permitted[0],
                )
                selected = select_candidate(
                    state, forced_candidate.candidate_id,
                    candidate_set_id=(state.candidate_set.candidate_set_id if state.candidate_set else None),
                )
                decision_source = "backend"
        if selected is None:
            break
        if isinstance(selection, AgentDecision) and selection.decision == "soft_stop" and any(
            event.get("event") == "soft_stop_rejected"
            and event.get("requested_stop") == selection.soft_stop_reason
            for event in state.observations[-2:]
        ):
            decision_source = "backend"
        state.observations.append({
            "event": "candidate_selected",
            "candidate_id": selected.candidate_id,
            "candidate_set_id": state.candidate_set.candidate_set_id if state.candidate_set else None,
            "decision_source": decision_source,
        })
        action = selected.action
        if getattr(action, "action_type", "") == "DRILL_DOWN":
            branch = next((b for b in state.branches if b.parent_evidence_id == getattr(action, "parent_evidence_id", None)), None)
            if branch and branch.depth >= max_drill_depth:
                state.stop_reason = StopReason(
                    code="USER_SCOPE_LIMIT", detail="已达到最大维度下钻深度。", initiator="budget"
                )
                break
        try:
            observed = list(executor(selected, state))
        except Exception as exc:  # fail closed; adapters retain detailed trace
            record_action_result(
                state, selected, [], current_period=current_period,
                baseline_period=baseline_period,
                decision_source=decision_source,
                failure_detail=f"候选动作执行失败：{type(exc).__name__}",
                failure_code="EXECUTION_FAILED",
            )
            break
        if not record_action_result(
            state, selected, observed, current_period=current_period,
            baseline_period=baseline_period, decision_source=decision_source,
        ):
            if continue_after_candidate_data_gap(state):
                # Rebuild the candidate set against the updated state. The
                # failed attempt remains in the ledger and consumes its
                # ordinary budget unit.
                continue
            break
        publish_progress(0.0, selected)

    # Preserve the useful terminal explanation when a data-gap candidate was
    # the last available action. If another candidate produced valid evidence,
    # retain the ordinary no-more-actions stop reason instead.
    if (
        state.stop_reason is not None
        and state.stop_reason.code == "NO_HIGH_VALUE_ACTION"
        and not state.evidence
        and state.action_results
        and state.action_results[-1].failure_code == "DATA_INSUFFICIENT"
    ):
        last_failure = state.action_results[-1]
        state.stop_reason = StopReason(
            code="DATA_INSUFFICIENT",
            detail=last_failure.failure_detail,
            initiator="backend",
        )

    if state.stop_reason is None:
        state.stop_reason = StopReason(
            code="BUDGET_EXHAUSTED" if state.remaining_budget <= 0 else "COMPLETED",
            detail="调查动作预算已用尽。" if state.remaining_budget <= 0 else "候选动作已处理完毕。",
            initiator="budget" if state.remaining_budget <= 0 else "backend",
        )
    state.progress = progress_snapshot(
        open_branch_count=sum(branch.status in {"open", "investigating"} for branch in state.branches),
        highest_open_branch_share=max((max(0.0, branch_share_for_priority(branch)) for branch in state.branches if branch.status in {"open", "investigating"}), default=0.0),
        max_remaining_action_score=0.0,
        remaining_budget=state.remaining_budget,
        capability_gap=state.stop_reason.code == "CAPABILITY_GAP",
        authorization_blocked=state.stop_reason.code == "AUTHORIZATION_BLOCKED",
    )
    terminal_status = terminal_progress_status(
        state.stop_reason.code, state.progress.open_branch_count,
    )
    if terminal_status:
        state.progress = state.progress.model_copy(update={"investigation_progress": terminal_status})
    if progress_callback:
        progress_callback(state.progress, None)
    return state

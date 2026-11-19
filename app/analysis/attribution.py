"""受控归因补查（attribution query planner）。

"变化归因"角度需要比报告事实快照更细的证据：维度下钻贡献 + 乘法指标分解。
本模块负责：
- 继承报告 v1 的 scope（platform/seller）、period、租户权限，不扩大范围。
- 只读查询，绝不修改已锁定的 report_json。
- 新证据以稳定 evidence_id 写入洞察的输入证据快照（attribution_evidence）。
- 不支持或数据不完整时记录明确的能力/数据缺口；已支持查询发生执行异常时必须失败，不能伪装成缺口。
"""

from __future__ import annotations

import json
import math
from hashlib import sha256
from typing import Any

from app.analysis.contracts import (
    ContributionFinding,
    FactorContribution,
    InsightAngle,
    InsightBrief,
    MetricDecomposition,
    PeriodicReport,
)
from app.analysis.contribution import calculate_change_contribution
from app.analysis.attribution_methods import (
    AttributionDataError,
    bind_registered_ratio_factor_identities,
    product_shapley_attribution,
    resolve_product_factor_values,
    symmetric_ratio_attribution,
    symmetric_mix_within_attribution,
)
from app.analysis.periods import previous_period
from app.analysis.semantic import get_metric_definition
from app.metrics.registry import get_registry
from app.investigation.evidence import build_attribution_evidence
from app.data.repositories.daily_metrics import (
    SUPPORTED_BREAKDOWN_DIMENSIONS,
    SUPPORTED_RATIO_FACT_DIMENSIONS,
    compare_breakdown_by_dimension_period,
)
from app.data.repositories.explorer import breakdown_scoped_month_position, breakdown_scoped_product_category
from app.schemas.explorer import ControlledAnalysisScope

# seller 范围可用下钻维度（有专属 scoped 查询函数）。
_SELLER_DIMENSIONS = ("product_category", "month_position")

# 单次补查上限，控制延迟与 token 成本。
MAX_METRICS = 2
MAX_DIMENSIONS = 3


class AttributionExecutionError(RuntimeError):
    """A registered attribution query failed; callers must mark execution failed."""

    failure_code = "EXECUTION_FAILED"

    def __init__(self, metric_id: str, dimension: str) -> None:
        self.metric_id = metric_id
        self.dimension = dimension
        # Keep the persisted/API-facing message free of raw driver details.
        super().__init__(
            f"归因查询执行失败（metric_id={metric_id}, dimension={dimension}）。"
        )


class AttributionDataInsufficient(RuntimeError):
    """A supported attribution query completed, but its report facts do not reconcile."""

    def __init__(self, period_statuses: dict[str, str]) -> None:
        self.period_statuses = period_statuses
        super().__init__("报告事实与归因补查无法对账，已停止生成该归因结果。")


def _stable_id(prefix: str, payload: Any) -> str:
    """由内容哈希生成稳定证据 ID：同一输入永远得到同一 ID，保证幂等与可审计。"""
    raw = json.dumps(payload, sort_keys=True, default=str)
    return f"{prefix}_{sha256(raw.encode()).hexdigest()[:16]}"


def _metric(report: PeriodicReport, name: str):
    """从报告指标列表中按名查找 ReportMetric；未找到返回 None（调用方跳过）。"""
    for metric in report.metrics:
        if metric.metric_name == name:
            return metric
    return None


def _same_period(left, right) -> bool:
    """Compare the full period contract, including granularity and timezone."""
    if left is None or right is None:
        return False
    return left.model_dump(mode="json") == right.model_dump(mode="json")


def build_decomposition(
    report: PeriodicReport, brief: InsightBrief, *, trace_id: str | None = None,
) -> tuple[list[MetricDecomposition], dict[str, dict[str, object]]]:
    """对已注册且事实快照具备充分数据的乘积/比率指标作确定性分解。

    参数：
    - `report`：已锁定的报告事实快照（PeriodicReport）。
    - `brief`：洞察生成约束，focus_metrics 决定分解哪些指标。

    返回值：
    - `list[MetricDecomposition]`：各乘法指标的分解证据。
    - `dict`：evidence_id → 内容快照（供 Critic 与审计引用）。

    分解基于已锁定的报告事实快照计算，不发起数据库查询，不违反事实层不可变性。
    不完整、不对账或分母无效时只记录能力缺口，不生成业务分解。
    """
    decompositions: list[MetricDecomposition] = []
    evidence: dict[str, dict[str, object]] = {}
    focused = brief.focus_metrics or [metric.metric_name for metric in report.metrics]
    for primary in report.metrics:
        metric_name = primary.metric_name
        if metric_name not in focused:
            continue
        try:
            definition = get_registry().get(metric_name)
            scope_capability = definition.attribution.for_scope(report.scope.scope_type)
        except Exception:  # noqa: BLE001 - missing registry metadata fails closed
            continue
        factor_specs = scope_capability.factor_specs.get("product_shapley_v1", ())
        comparison = primary.period_over_period
        if (
            "product_shapley_v1" not in scope_capability.method_ids
            or len(factor_specs) != 2
            or primary.value is None
            or comparison is None
            or comparison.baseline_value is None
            or not _same_period(comparison.current_period, report.period)
            or comparison.baseline_period is None
        ):
            continue

        factors = [item.factor_id for item in factor_specs]
        source_metric_ids = {
            source_id for spec in factor_specs for source_id in spec.source_metric_ids
        }
        period_totals: dict[str, tuple[float, float]] = {}
        invalid_sources = False
        for source_metric_id in source_metric_ids:
            source = _metric(report, source_metric_id)
            source_comparison = source.period_over_period if source is not None else None
            if (
                source is None or source.value is None or source_comparison is None
                or source_comparison.baseline_value is None
                or not _same_period(source_comparison.current_period, report.period)
                or not _same_period(source_comparison.baseline_period, comparison.baseline_period)
            ):
                invalid_sources = True
                break
            period_totals[source_metric_id] = (
                float(source_comparison.baseline_value), float(source.value),
            )
        if invalid_sources:
            continue
        try:
            baseline_factors, current_factors = resolve_product_factor_values(
                factor_specs, period_totals,
            )
        except AttributionDataError:
            continue
        current_value = float(primary.value)
        baseline_value = float(comparison.baseline_value)
        decomposition = product_shapley_attribution(
            metric_id=metric_name,
            baseline_factors=baseline_factors,
            current_factors=current_factors,
            unit=definition.unit,
            evidence_ids=[],
        )
        if not (
            math.isclose(decomposition.baseline_value, baseline_value, rel_tol=1e-8, abs_tol=0.01)
            and math.isclose(decomposition.current_value, current_value, rel_tol=1e-8, abs_tol=0.01)
        ):
            # The configured factors must reproduce the locked report metric;
            # mismatched period or factor contracts cannot produce attribution.
            continue
        # Keep the established raw-factor contract and interaction for audit;
        # separately expose Shapley-allocated factor effects for explanation.
        raw_by_factor = {item.component_id.split(":", 1)[0]: item.value for item in decomposition.raw_components if item.component_type == "factor_effect"}
        display_by_factor = {item.component_id: item.value for item in decomposition.display_components}
        factor_deltas = [raw_by_factor[name] for name in factors]
        interaction = next(item.value for item in decomposition.raw_components if item.component_type == "interaction_effect")
        total_delta = round(decomposition.total_delta, 2)
        eid = _stable_id("ev_decomp", {
            "metric": metric_name, "factors": list(factors),
            "current": round(current_value, 2), "baseline": round(baseline_value, 2),
            "factor_deltas": [round(value, 2) for value in factor_deltas], "interaction": round(interaction, 2),
            "display_method": decomposition.display_method,
            "display_factor_deltas": [round(display_by_factor[name], 2) for name in factors],
        })
        decompositions.append(MetricDecomposition(
            metric_name=metric_name, factor_metrics=list(factors),
            current_value=round(current_value, 2), baseline_value=round(baseline_value, 2),
            total_delta=total_delta,
            factors=[
                FactorContribution(factor_metric=factor, delta_value=round(delta, 2),
                                   contribution_ratio=None if total_delta == 0 else round(delta / total_delta, 4))
                for factor, delta in zip(factors, factor_deltas, strict=True)
            ],
            interaction_delta=round(interaction, 2),
            display_factors=[
                FactorContribution(factor_metric=factor, delta_value=round(display_by_factor[factor], 2),
                                   contribution_ratio=None if total_delta == 0 else round(display_by_factor[factor] / total_delta, 4))
                for factor in factors
            ],
            display_method="shapley",
            evidence_ids=[eid],
        ))
        scope = {
            "scope_type": report.scope.scope_type,
            "seller_id": report.scope.seller_id,
            "current_period": report.period.model_dump(mode="json"),
            "baseline_period": primary.period_over_period.baseline_period.model_dump(mode="json")
            if primary.period_over_period.baseline_period else None,
        }
        typed = build_attribution_evidence(
            evidence_id=eid, evidence_type="metric_decomposition", metric_id=metric_name,
            trace_id=trace_id,
            claim=f"{definition.display_name} 变化按注册的二因子 Shapley 方法作统计分解。",
            values={"total_delta": total_delta,
                    "factor_metrics": list(factors),
                    # Preserve the complete method output for audit. The report
                    # presentation contract separates factor effects from the
                    # interaction, but raw attribution must retain all terms.
                    "raw_components": [item.model_dump(mode="json") for item in decomposition.raw_components],
                    "interaction_delta": round(interaction, 2),
                    "display_method": "shapley",
                    "reconciliation_ratio": decomposition.reconciliation_ratio,
                    "residual_value": decomposition.residual_value,
                    "residual_type": decomposition.residual_type,
                    "residual_explainable": decomposition.residual_explainable,
                    "residual_tolerance": decomposition.residual_tolerance,
                    "residual_note": decomposition.residual_note,
                    "display_components": [item.model_dump(mode="json") for item in decomposition.display_components]},
            source_action_id="locked_report_snapshot_decomposition", source_ref="report_metric_snapshot",
            scope=scope, method_id="product_shapley_v1", method_version="1.0",
            authorized_metric_provenance="locked_report_metric_snapshot",
            comparison={"current_period": report.period.model_dump(mode="json"),
                        "baseline_period": scope["baseline_period"]},
            limitations=["交互项在展示层按 Shapley 方法均分；统计分解不构成因果结论。"],
        )
        evidence[eid] = {
            "kind": "metric_decomposition", "metric_name": metric_name,
            "factor_metrics": list(factors),
            "current_value": round(current_value, 2), "baseline_value": round(baseline_value, 2),
            "total_delta": total_delta,
            "factors": [
                {"factor_metric": factor, "delta_value": round(delta, 2)}
                for factor, delta in zip(factors, factor_deltas, strict=True)
            ],
            "interaction_delta": round(interaction, 2),
            "display_method": "shapley",
            "display_factors": [
                {"factor_metric": factor, "delta_value": round(display_by_factor[factor], 2)}
                for factor in factors
            ],
            "investigation_evidence": typed.model_dump(mode="json"),
        }

    # Registered weighted ratios can be decomposed without a new query when
    # both ratio components are already present in the immutable report.
    for primary in report.metrics:
        metric_name = primary.metric_name
        if metric_name not in focused:
            continue
        try:
            definition = get_registry().get(metric_name)
            capability = definition.attribution.for_scope(report.scope.scope_type)
        except Exception:  # noqa: BLE001 - missing registry metadata fails closed
            continue
        if "symmetric_ratio_v1" not in capability.method_ids:
            continue
        period_aggregation = definition.period_aggregation
        if period_aggregation is None or period_aggregation.kind != "weighted_ratio":
            continue
        factors = [
            period_aggregation.numerator_metric,
            period_aggregation.denominator_metric,
        ]
        comparison = primary.period_over_period
        reason = ""
        numerator_metric = _metric(report, factors[0])
        denominator_metric = _metric(report, factors[1])
        if comparison is None or comparison.baseline_value is None or primary.value is None:
            reason = "目标比率的当前值或基期值缺失。"
        elif numerator_metric is None or denominator_metric is None:
            reason = "锁定报告快照缺少已注册的分子或分母指标。"
        elif (
            numerator_metric.value is None or denominator_metric.value is None
            or numerator_metric.period_over_period is None
            or denominator_metric.period_over_period is None
            or numerator_metric.period_over_period.baseline_value is None
            or denominator_metric.period_over_period.baseline_value is None
        ):
            reason = "分子或分母缺少当前值/基期值。"
        elif (
            comparison.baseline_period is None
            or not _same_period(comparison.current_period, report.period)
            or numerator_metric.period_over_period is None
            or denominator_metric.period_over_period is None
            or not _same_period(numerator_metric.period_over_period.current_period, comparison.current_period)
            or not _same_period(denominator_metric.period_over_period.current_period, comparison.current_period)
            or not _same_period(numerator_metric.period_over_period.baseline_period, comparison.baseline_period)
            or not _same_period(denominator_metric.period_over_period.baseline_period, comparison.baseline_period)
        ):
            reason = "目标指标、分子和分母的当前期/基期不一致，不能进行比率分解。"

        decomposition = None
        if not reason:
            n0 = float(numerator_metric.period_over_period.baseline_value)
            n1 = float(numerator_metric.value)
            d0 = float(denominator_metric.period_over_period.baseline_value)
            d1 = float(denominator_metric.value)
            try:
                decomposition = symmetric_ratio_attribution(
                    metric_id=metric_name,
                    baseline_numerator=n0,
                    current_numerator=n1,
                    baseline_denominator=d0,
                    current_denominator=d1,
                    multiplier=period_aggregation.multiplier,
                    unit=definition.unit,
                )
                decomposition = bind_registered_ratio_factor_identities(
                    decomposition, scope=report.scope.scope_type,
                )
                observed_baseline = float(comparison.baseline_value)
                observed_current = float(primary.value)
                # Report values are rounded for presentation. A wider-than-
                # display discrepancy indicates incompatible periods/semantics.
                if (
                    abs(decomposition.baseline_value - observed_baseline) > 0.03
                    or abs(decomposition.current_value - observed_current) > 0.03
                ):
                    reason = "注册分子/分母计算值与报告中的比率值无法对账。"
                    decomposition = None
            except (AttributionDataError, TypeError, ValueError, OverflowError):
                reason = "分子/分母无效或不能满足对称分解条件。"
                decomposition = None

        baseline_period = comparison.baseline_period if comparison else None
        scope = {
            "scope_type": report.scope.scope_type,
            "seller_id": report.scope.seller_id,
            "current_period": report.period.model_dump(mode="json"),
            "baseline_period": baseline_period.model_dump(mode="json") if baseline_period else None,
        }
        if decomposition is None:
            gap_id = _stable_id("ev_decomp_gap", {
                "metric": metric_name, "factors": factors, "scope": scope,
                "reason": reason or "ratio decomposition is unavailable",
            })
            typed_gap = build_attribution_evidence(
                evidence_id=gap_id, evidence_type="attribution_capability_gap",
                metric_id=metric_name,
                trace_id=trace_id,
                claim="当前报告快照不具备可对账的分子/分母充分数据，未执行比率分解。",
                values={
                    "status": "capability_gap", "reason": reason or "分解所需数据缺失。",
                    "factor_metrics": factors,
                },
                source_action_id="locked_report_snapshot_decomposition",
                source_ref="report_metric_snapshot",
                scope=scope, method_id="symmetric_ratio_v1", method_version="1.0",
                authorized_metric_provenance="locked_report_metric_snapshot",
                comparison={"current_period": report.period.model_dump(mode="json"),
                            "baseline_period": scope["baseline_period"]},
                limitations=["没有生成分子/分母贡献结论；统计分解不构成因果结论。"],
            )
            evidence[gap_id] = {
                "kind": "capability_gap", "metric_name": metric_name,
                "factor_metrics": factors, "reason": reason,
                "investigation_evidence": typed_gap.model_dump(mode="json"),
            }
            continue

        total_delta = round(decomposition.total_delta, 2)
        effects = {item.component_id: item.value for item in decomposition.raw_components}
        n_effect = effects["numerator_effect"]
        d_effect = effects["denominator_effect"]
        eid = _stable_id("ev_ratio_decomp", {
            "metric": metric_name, "factors": factors,
            "baseline": [n0, d0], "current": [n1, d1],
            "multiplier": period_aggregation.multiplier,
            "effects": [round(n_effect, 8), round(d_effect, 8)],
        })
        factors_out = [
            FactorContribution(
                factor_metric=factors[0], delta_value=round(n_effect, 2),
                contribution_ratio=None if abs(total_delta) <= 1e-12 else round(n_effect / total_delta, 4),
            ),
            FactorContribution(
                factor_metric=factors[1], delta_value=round(d_effect, 2),
                contribution_ratio=None if abs(total_delta) <= 1e-12 else round(d_effect / total_delta, 4),
            ),
        ]
        item = MetricDecomposition(
            metric_name=metric_name, method_id="symmetric_ratio_v1",
            method_version="1.0", factor_metrics=factors,
            current_value=round(decomposition.current_value, 2),
            baseline_value=round(decomposition.baseline_value, 2),
            total_delta=total_delta, factors=factors_out, interaction_delta=0.0,
            display_factors=factors_out, display_method="symmetric_ratio",
            evidence_ids=[eid],
        )
        decompositions.append(item)
        typed = build_attribution_evidence(
            evidence_id=eid, evidence_type="metric_decomposition", metric_id=metric_name,
            trace_id=trace_id,
            claim=f"{definition.display_name} 变化按注册分子与分母作对称分解。",
            values={
                "total_delta": total_delta, "method_id": "symmetric_ratio_v1",
                "factor_metrics": factors,
                "factor_identities": [
                    item.model_dump(mode="json") for item in decomposition.factor_identities
                ],
                "display_method": "raw",
                "baseline_numerator": n0, "current_numerator": n1,
                "baseline_denominator": d0, "current_denominator": d1,
                "multiplier": period_aggregation.multiplier,
                "raw_components": [part.model_dump(mode="json") for part in decomposition.raw_components],
                "display_components": [part.model_dump(mode="json") for part in decomposition.display_components],
                "interaction_delta": 0.0,
                "reconciliation_ratio": decomposition.reconciliation_ratio,
                "residual_value": decomposition.residual_value,
                "residual_type": decomposition.residual_type,
                "residual_explainable": decomposition.residual_explainable,
                "residual_tolerance": decomposition.residual_tolerance,
                "residual_note": decomposition.residual_note,
            },
            source_action_id="locked_report_snapshot_decomposition", source_ref="report_metric_snapshot",
            scope=scope, method_id="symmetric_ratio_v1", method_version="1.0",
            authorized_metric_provenance="locked_report_metric_snapshot",
            comparison={"current_period": report.period.model_dump(mode="json"),
                        "baseline_period": scope["baseline_period"]},
            limitations=["分子/分母对称分解只描述数值变化，不代表因果效应。"],
        )
        evidence[eid] = {
            "kind": "metric_decomposition", "metric_name": metric_name,
            "method_id": "symmetric_ratio_v1", "factor_metrics": factors,
            "current_value": item.current_value, "baseline_value": item.baseline_value,
            "total_delta": total_delta,
            "factors": [part.model_dump(mode="json") for part in factors_out],
            "interaction_delta": 0.0, "display_method": "symmetric_ratio",
            "investigation_evidence": typed.model_dump(mode="json"),
        }
    return decompositions, evidence


def build_grouped_ratio_attributions(
    report: PeriodicReport,
    brief: InsightBrief,
    *, trace_id: str | None = None,
) -> dict[str, dict[str, object]]:
    """Build governed mix/within evidence for ratios with complete group facts.

    Only Registry-declared platform ratio metrics and dimensions are eligible.
    The dedicated source query groups the one-row-per-order fact table, so
    every order belongs to exactly one group and numerator/denominator totals
    can be reconciled before Evidence is emitted. This method-specific
    capability is intentionally separate from ordinary breakdown dimensions:
    it does not implicitly authorize fan-out dimensions such as item category.
    """
    if report.scope.scope_type != "platform":
        return {}

    evidence: dict[str, dict[str, object]] = {}
    for primary in report.metrics:
        metric_name = primary.metric_name
        if brief.focus_metrics and metric_name not in brief.focus_metrics:
            continue
        try:
            definition = get_registry().get(metric_name)
            capability = definition.attribution.for_scope("platform")
        except Exception:  # noqa: BLE001 - missing declarations fail closed
            continue
        if "symmetric_mix_within_v1" not in capability.method_ids:
            continue
        aggregation = definition.period_aggregation
        comparison = primary.period_over_period
        if aggregation is None or aggregation.kind != "weighted_ratio":
            continue
        factor_names = [aggregation.numerator_metric, aggregation.denominator_metric]
        numerator_metric, denominator_metric = (_metric(report, name) for name in factor_names)
        if brief.dimension:
            dimensions = {brief.dimension}
        else:
            dimensions = set(capability.dimensions)
        dimensions &= set(capability.dimensions) & set(SUPPORTED_RATIO_FACT_DIMENSIONS)

        for dimension in sorted(dimensions):
            if comparison is None or comparison.baseline_period is None or comparison.baseline_value is None or primary.value is None:
                reason = "目标比率缺少当前期或基期数值。"
                facts = None
            elif numerator_metric is None or denominator_metric is None:
                reason = "锁定报告缺少注册的分子或分母指标。"
                facts = None
            elif (
                numerator_metric.value is None or denominator_metric.value is None
                or numerator_metric.period_over_period is None
                or denominator_metric.period_over_period is None
                or numerator_metric.period_over_period.baseline_value is None
                or denominator_metric.period_over_period.baseline_value is None
            ):
                reason = "锁定报告缺少分子或分母的当前值/基期值。"
                facts = None
            elif (
                not _same_period(comparison.current_period, report.period)
                or not _same_period(numerator_metric.period_over_period.current_period, comparison.current_period)
                or not _same_period(denominator_metric.period_over_period.current_period, comparison.current_period)
                or not _same_period(numerator_metric.period_over_period.baseline_period, comparison.baseline_period)
                or not _same_period(denominator_metric.period_over_period.baseline_period, comparison.baseline_period)
            ):
                reason = "目标指标、分子和分母的比较区间不一致。"
                facts = None
            else:
                try:
                    facts = _locked_ratio_snapshot(report, primary, dimension)
                    reason = ""
                except AttributionDataInsufficient as exc:
                    facts = None
                    reason = "; ".join(f"{key}={value}" for key, value in sorted(exc.period_statuses.items()))

            attribution = None
            if facts is not None and facts.get("status") == "ok" and facts.get("complete") is True:
                base_cmp = numerator_metric.period_over_period
                den_cmp = denominator_metric.period_over_period
                expected_totals = (
                    (facts.get("baseline_numerator_total"), base_cmp.baseline_value),
                    (facts.get("current_numerator_total"), numerator_metric.value),
                    (facts.get("baseline_denominator_total"), den_cmp.baseline_value),
                    (facts.get("current_denominator_total"), denominator_metric.value),
                )
                if any(value is None or expected is None or abs(float(value) - float(expected)) > 0.03
                       for value, expected in expected_totals):
                    reason = "分组分子/分母总量与锁定报告快照无法对账。"
                else:
                    try:
                        attribution = symmetric_mix_within_attribution(
                            metric_id=metric_name,
                            baseline_groups=facts["baseline_groups"],
                            current_groups=facts["current_groups"],
                            multiplier=aggregation.multiplier,
                            unit=definition.unit,
                        )
                        if (
                            abs(attribution.baseline_value - float(comparison.baseline_value)) > 0.03
                            or abs(attribution.current_value - float(primary.value)) > 0.03
                        ):
                            attribution = None
                            reason = "分组比率与锁定报告中的目标指标无法对账。"
                    except (AttributionDataError, TypeError, ValueError, OverflowError):
                        attribution = None
                        reason = "分组数据的分母无效或不足以进行 mix/within 分解。"
            elif facts is not None:
                reason = "比较期中至少一期没有完整分组事实。"

            scope = {
                "scope_type": "platform",
                "current_period": report.period.model_dump(mode="json"),
                "baseline_period": comparison.baseline_period.model_dump(mode="json") if comparison else None,
            }
            if attribution is None:
                eid = _stable_id("ev_group_ratio_gap", {
                    "metric": metric_name, "dimension": dimension, "scope": scope, "reason": reason,
                })
                typed = build_attribution_evidence(
                    evidence_id=eid, evidence_type="attribution_capability_gap",
                    metric_id=metric_name,
                    trace_id=trace_id,
                    claim="当前范围未能取得可对账的完整分组分子/分母数据，未执行分组比率归因。",
                    values={"status": "capability_gap", "reason": reason, "dimension": dimension,
                            "factor_metrics": factor_names},
                    source_action_id=f"grouped_ratio:{metric_name}:{dimension}",
                    source_ref="platform_grouped_ratio_facts", scope=scope,
                    method_id="symmetric_mix_within_v1", method_version="1.0",
                    authorized_metric_provenance="locked_report_scope",
                    comparison={"current_period": scope["current_period"], "baseline_period": scope["baseline_period"]},
                    limitations=["未生成组内或构成贡献结论；统计分解不构成因果效应。"],
                )
                evidence[eid] = {
                    "kind": "capability_gap", "metric_name": metric_name,
                    "dimension": dimension, "reason": reason,
                    "investigation_evidence": typed.model_dump(mode="json"),
                }
                continue

            group_ids = sorted(set(facts["baseline_groups"]) | set(facts["current_groups"]))
            group_effects = []
            for group in group_ids:
                base = facts["baseline_groups"].get(group)
                current = facts["current_groups"].get(group)
                component_items = [item for item in attribution.raw_components
                                   if item.component_id.startswith(f"{group}:")]
                group_effects.append({
                    "dimension_value": group,
                    "status": "continuing" if base is not None and current is not None else "entry" if current is not None else "exit",
                    "baseline_numerator": base["numerator"] if base is not None else None,
                    "baseline_denominator": base["denominator"] if base is not None else None,
                    "current_numerator": current["numerator"] if current is not None else None,
                    "current_denominator": current["denominator"] if current is not None else None,
                    "components": [item.model_dump(mode="json") for item in component_items],
                    "total_effect": sum(item.value for item in component_items),
                })
            eid = _stable_id("ev_group_ratio", {
                "metric": metric_name, "dimension": dimension, "scope": scope,
                "groups": group_effects, "method": attribution.method_id,
            })
            typed = build_attribution_evidence(
                evidence_id=eid, evidence_type="grouped_ratio_attribution", metric_id=metric_name,
                trace_id=trace_id,
                claim=f"{definition.display_name} 的变化按 {dimension} 维度拆为组内变化与组别构成变化。",
                values={
                    "status": "ok", "dimension": dimension,
                    "factor_metrics": factor_names,
                    "baseline_value": attribution.baseline_value,
                    "current_value": attribution.current_value,
                    "total_delta": attribution.total_delta,
                    "components": [item.model_dump(mode="json") for item in attribution.raw_components],
                    "groups": group_effects,
                    "residual_value": attribution.residual_value,
                    "residual_type": attribution.residual_type,
                    "residual_explainable": attribution.residual_explainable,
                    "residual_tolerance": attribution.residual_tolerance,
                    "residual_note": attribution.residual_note,
                },
                source_action_id=f"grouped_ratio:{metric_name}:{dimension}",
                source_ref=facts["source_table"], scope=scope,
                method_id=attribution.method_id, method_version=attribution.method_version,
                authorized_metric_provenance="locked_report_scope",
                unit=definition.unit,
                comparison={"current_period": facts["current_period"], "baseline_period": facts["baseline_period"]},
                data_quality={"complete": True, "group_count": len(group_ids),
                              "reconciled": abs(attribution.residual_value) <= 1e-8},
                limitations=["组内/构成分解描述统计变化，不证明维度造成了比率变化。"],
            )
            evidence[eid] = {
                "kind": "grouped_ratio_attribution", "metric_name": metric_name,
                "dimension": dimension, "method_id": attribution.method_id,
                "method_version": attribution.method_version,
                "baseline_value": attribution.baseline_value, "current_value": attribution.current_value,
                "total_delta": attribution.total_delta,
                "components": [item.model_dump(mode="json") for item in attribution.raw_components],
                "groups": group_effects, "evidence_id": eid,
                "investigation_evidence": typed.model_dump(mode="json"),
            }
    return evidence


def _seller_scope(report: PeriodicReport, period, metrics: list[str]) -> ControlledAnalysisScope:
    """构造 seller 范围的受控查询 scope：继承报告 seller_id 与期间，指标白名单锁定。"""
    return ControlledAnalysisScope(
        seller_id=report.scope.seller_id,
        default_start_date=period.start_date, default_end_date=period.end_date,
        metric_name=metrics[0], granularity="day", allowed_metrics=tuple(metrics),
    )


def _locked_dimension_snapshot(report: PeriodicReport, metric, dimension: str, baseline_period) -> dict[str, Any]:
    """Load only report-pinned breakdown facts; never fall back to live tables."""
    from app.analysis.report_alignment import (
        attribution_snapshot_key,
        validate_report_breakdown_totals,
    )

    snapshot = report.attribution_breakdowns.get(
        attribution_snapshot_key(metric.metric_name, dimension),
    )
    if not isinstance(snapshot, dict):
        raise AttributionDataInsufficient({"snapshot": "report_snapshot_missing"})
    if snapshot.get("status") != "ok":
        codes = snapshot.get("reason_codes")
        if not isinstance(codes, dict):
            codes = {"snapshot": str(snapshot.get("reason_code") or snapshot.get("status"))}
        raise AttributionDataInsufficient(codes)
    comparison = metric.period_over_period
    if comparison is None or comparison.baseline_period is None:
        raise AttributionDataInsufficient({"comparison": "report_period_missing"})
    if (
        snapshot.get("current_period") != report.period.model_dump(mode="json")
        or snapshot.get("baseline_period") != baseline_period.model_dump(mode="json")
        or comparison.baseline_period.model_dump(mode="json") != baseline_period.model_dump(mode="json")
    ):
        raise AttributionDataInsufficient({"snapshot": "report_period_mismatch"})
    if any(snapshot.get(name) is None for name in (
        "current_total", "baseline_total", "total_delta", "current_items", "baseline_items",
    )):
        raise AttributionDataInsufficient({"snapshot": "report_snapshot_incomplete"})
    digest = snapshot.get("snapshot_digest")
    body = {key: value for key, value in snapshot.items() if key != "snapshot_digest"}
    expected_digest = sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()
    if not isinstance(digest, str) or digest != expected_digest:
        raise AttributionDataInsufficient({"snapshot": "report_snapshot_digest_invalid"})
    validate_report_breakdown_totals(
        report, metric, baseline_period,
        float(snapshot.get("current_total")), float(snapshot.get("baseline_total")),
    )
    return snapshot


def _locked_ratio_snapshot(report: PeriodicReport, metric, dimension: str) -> dict[str, Any]:
    """Load report-pinned ratio sufficient statistics and reconcile to report facts."""
    from app.analysis.report_alignment import ratio_attribution_snapshot_key

    snapshot = report.attribution_breakdowns.get(
        ratio_attribution_snapshot_key(metric.metric_name, dimension),
    )
    if not isinstance(snapshot, dict):
        raise AttributionDataInsufficient({"snapshot": "report_ratio_snapshot_missing"})
    if snapshot.get("status") != "ok":
        codes = snapshot.get("reason_codes")
        if not isinstance(codes, dict):
            codes = {"snapshot": str(snapshot.get("reason_code") or snapshot.get("status"))}
        raise AttributionDataInsufficient(codes)

    comparison = metric.period_over_period
    baseline_period = comparison.baseline_period if comparison is not None else None
    if (
        comparison is None or baseline_period is None
        or not _same_period(comparison.current_period, report.period)
        or snapshot.get("current_period") != report.period.model_dump(mode="json")
        or snapshot.get("baseline_period") != baseline_period.model_dump(mode="json")
        or snapshot.get("method_id") != "symmetric_mix_within_v1"
        or snapshot.get("metric_id") != metric.metric_name
        or snapshot.get("dimension") != dimension
        or snapshot.get("scope_type") != report.scope.scope_type
        or snapshot.get("complete") is not True
    ):
        raise AttributionDataInsufficient({"snapshot": "report_ratio_snapshot_period_or_contract_mismatch"})

    digest = snapshot.get("snapshot_digest")
    body = {key: value for key, value in snapshot.items() if key != "snapshot_digest"}
    expected_digest = sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()
    if not isinstance(digest, str) or digest != expected_digest:
        raise AttributionDataInsufficient({"snapshot": "report_ratio_snapshot_digest_invalid"})

    from app.metrics.registry import get_registry

    aggregation = get_registry().get(metric.metric_name).period_aggregation
    if aggregation is None:
        raise AttributionDataInsufficient({"snapshot": "registered_ratio_contract_missing"})
    numerator_metric = _metric(report, aggregation.numerator_metric)
    denominator_metric = _metric(report, aggregation.denominator_metric)
    if (
        numerator_metric is None or denominator_metric is None
        or numerator_metric.value is None or denominator_metric.value is None
        or numerator_metric.period_over_period is None
        or denominator_metric.period_over_period is None
        or numerator_metric.period_over_period.baseline_value is None
        or denominator_metric.period_over_period.baseline_value is None
    ):
        raise AttributionDataInsufficient({"report": "ratio_factor_totals_missing"})
    if (
        snapshot.get("numerator_metric") != aggregation.numerator_metric
        or snapshot.get("denominator_metric") != aggregation.denominator_metric
    ):
        raise AttributionDataInsufficient({"snapshot": "report_ratio_snapshot_metric_mismatch"})

    expected = (
        (snapshot.get("baseline_numerator_total"), numerator_metric.period_over_period.baseline_value),
        (snapshot.get("current_numerator_total"), numerator_metric.value),
        (snapshot.get("baseline_denominator_total"), denominator_metric.period_over_period.baseline_value),
        (snapshot.get("current_denominator_total"), denominator_metric.value),
    )
    if any(
        actual is None or target is None or not math.isclose(float(actual), float(target), rel_tol=1e-8, abs_tol=0.03)
        for actual, target in expected
    ):
        raise AttributionDataInsufficient({"snapshot": "report_ratio_snapshot_total_mismatch"})
    baseline_groups = snapshot.get("baseline_groups")
    current_groups = snapshot.get("current_groups")
    if not isinstance(baseline_groups, dict) or not isinstance(current_groups, dict):
        raise AttributionDataInsufficient({"snapshot": "report_ratio_snapshot_groups_invalid"})
    try:
        group_totals = (
            (sum(float(group["numerator"]) for group in baseline_groups.values()), snapshot["baseline_numerator_total"]),
            (sum(float(group["numerator"]) for group in current_groups.values()), snapshot["current_numerator_total"]),
            (sum(float(group["denominator"]) for group in baseline_groups.values()), snapshot["baseline_denominator_total"]),
            (sum(float(group["denominator"]) for group in current_groups.values()), snapshot["current_denominator_total"]),
        )
        if any(
            not math.isfinite(float(actual)) or not math.isclose(float(actual), float(expected_total), rel_tol=1e-8, abs_tol=0.03)
            for actual, expected_total in group_totals
        ):
            raise ValueError("group totals do not reconcile")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise AttributionDataInsufficient({"snapshot": "report_ratio_snapshot_group_totals_invalid"}) from exc
    return snapshot


def _seller_dimension_finding(report: PeriodicReport, metric, dimension: str, baseline_period) -> ContributionFinding:
    """seller 范围：对单个维度做当前/基期双期间拆解，合成贡献度证据。

    参数：
    - `report` / `metric` / `dimension`：报告、目标指标、下钻维度。
    - `baseline_period`：对比基期（通常是上一周期）。

    返回值：`ContributionFinding`。无可用分组数据时抛出 `AttributionDataInsufficient`。
    """
    snapshot = _locked_dimension_snapshot(report, metric, dimension, baseline_period)
    comparison = metric.period_over_period
    return calculate_change_contribution(
        metric.metric_name, dimension, report.period.end_date, baseline_period.end_date,
        snapshot["current_items"], snapshot["baseline_items"],
        comparison.delta_value if comparison else None,
        snapshot["current_total"], snapshot["baseline_total"],
        report.period, baseline_period,
    )


def _platform_dimension_finding(report: PeriodicReport, metric, dimension: str, baseline_period) -> ContributionFinding:
    """platform 范围：用全量拆解接口做双期间对比，合成贡献度证据。

    参数与返回值同 `_seller_dimension_finding`。
    """
    snapshot = _locked_dimension_snapshot(report, metric, dimension, baseline_period)
    return calculate_change_contribution(
        metric.metric_name, dimension, report.period.end_date, baseline_period.end_date,
        snapshot["current_items"], snapshot["baseline_items"],
        snapshot["total_delta"], snapshot["current_total"], snapshot["baseline_total"],
        report.period, baseline_period,
    )


def build_dimension_data_gap_evidence(
    report: PeriodicReport, *, metric_id: str, metric_unit: str, dimension: str,
    baseline_period, period_statuses: dict[str, str], trace_id: str | None = None,
    source_action_id: str = "locked_report_dimension_breakdown",
) -> tuple[str, dict[str, object]]:
    """Build the same traceable DATA_INSUFFICIENT evidence for both report adapters."""
    from app.investigation.evidence import build_attribution_evidence

    current_period = report.period.model_dump(mode="json")
    baseline_payload = baseline_period.model_dump(mode="json")
    scope = {
        "scope_type": report.scope.scope_type,
        "seller_id": report.scope.seller_id,
        "current_period": current_period,
        "baseline_period": baseline_payload,
    }
    evidence_id = _stable_id("ev_attribution_data_gap", {
        "report_id": report.report_id,
        "metric": metric_id,
        "dimension": dimension,
        "current_period": current_period,
        "baseline_period": baseline_payload,
        "statuses": period_statuses,
    })
    reason = "报告未保存可用且与报告事实对齐的维度快照，因此本次不生成该维度归因。"
    typed = build_attribution_evidence(
        evidence_id=evidence_id,
        evidence_type="attribution_data_gap",
        metric_id=metric_id,
        trace_id=trace_id,
        claim=reason,
        values={
            "status": "data_insufficient",
            "gap_code": "DATA_INSUFFICIENT",
            "dimension": dimension,
            "period_statuses": period_statuses,
        },
        source_action_id=source_action_id,
        source_ref="report.attribution_breakdowns",
        scope=scope,
        method_id="additive_change_v1",
        method_version="1.0",
        authorized_metric_provenance="locked_report_scope",
        unit=metric_unit,
        comparison={"current_period": current_period, "baseline_period": baseline_payload},
        data_quality={"status": "snapshot_unavailable", "period_statuses": period_statuses},
        limitations=["缺少可用于该维度双期归因的分组数据，不据此推断业务原因。"],
    )
    return evidence_id, {
        "kind": "data_gap",
        "gap_code": "DATA_INSUFFICIENT",
        "metric_name": metric_id,
        "dimension": dimension,
        "current_period": current_period,
        "baseline_period": baseline_payload,
        "period_statuses": period_statuses,
        "reason": reason,
        "evidence_id": evidence_id,
        "investigation_evidence": typed.model_dump(mode="json"),
    }


def build_dimension_breakdowns(
    report: PeriodicReport, brief: InsightBrief, *, trace_id: str | None = None,
) -> tuple[list[ContributionFinding], dict[str, dict[str, object]]]:
    """对变化最大的求和型指标做维度下钻补查。

    参数：
    - `report` / `brief`：报告事实快照与洞察约束。

    返回值：
    - `list[ContributionFinding]`：各维度下钻的贡献证据。
    - `dict`：evidence_id → 内容快照。

    只对在当前 scope 显式声明 ``additive_change_v1`` 的指标下钻，避免
    把比率/均值指标误当作可加总贡献。
    """
    contributions: list[ContributionFinding] = []
    evidence: dict[str, dict[str, object]] = {}
    from app.metrics.registry import get_registry

    baseline_period = previous_period(report.period)
    for metric in report.metrics[:MAX_METRICS]:
        if brief.focus_metrics and metric.metric_name not in brief.focus_metrics:
            continue
        # 只对求和型且环比有有效 delta 的指标下钻。
        if metric.period_over_period is None:
            continue
        if metric.period_over_period.delta_value is None or metric.period_over_period.baseline_value is None:
            continue
        try:
            definition = get_metric_definition(metric.metric_name, report.scope.scope_type)
            capability = get_registry().get(metric.metric_name).attribution.for_scope(report.scope.scope_type)
        except Exception:
            continue
        if "additive_change_v1" not in capability.method_ids:
            continue
        if report.scope.scope_type == "seller":
            dimensions = sorted(set(capability.dimensions) & set(_SELLER_DIMENSIONS))
        else:
            dimensions = sorted(set(capability.dimensions) & set(definition.allowed_dimensions) & SUPPORTED_BREAKDOWN_DIMENSIONS)
        for dimension in list(dimensions)[:MAX_DIMENSIONS]:
            try:
                finding = _seller_dimension_finding(report, metric, dimension, baseline_period) if report.scope.scope_type == "seller" else _platform_dimension_finding(report, metric, dimension, baseline_period)
            except AttributionDataInsufficient as exc:
                gap_id, record = build_dimension_data_gap_evidence(
                    report,
                    metric_id=metric.metric_name,
                    metric_unit=metric.unit,
                    dimension=dimension,
                    baseline_period=baseline_period,
                    period_statuses=exc.period_statuses,
                    trace_id=trace_id,
                )
                evidence[gap_id] = record
                continue
            except Exception as exc:  # noqa: BLE001 - do not hide source/query failures
                raise AttributionExecutionError(metric.metric_name, dimension) from exc
            if finding is not None:
                contributions.append(finding)
                for eid in finding.evidence_ids:
                    evidence[eid] = {"kind": "contribution", **finding.model_dump(mode="json")}
    return contributions, evidence


def query_attribution_evidence(
    report: PeriodicReport, brief: InsightBrief, *, trace_id: str | None = None,
) -> tuple[list[ContributionFinding], list[MetricDecomposition], dict[str, dict[str, object]]]:
    """变化归因角度的受控补查。仅当 brief.angle == CONTRIBUTION 时执行。

    参数：
    - `report`：已锁定的报告事实快照。
    - `brief`：洞察约束（角度、重点指标、下钻维度）。

    返回值：
    - `list[ContributionFinding]`：维度下钻贡献证据。
    - `list[MetricDecomposition]`：乘法指标分解证据。
    - `dict`：合并后的 evidence_id → 内容快照。
    """
    if brief.angle != InsightAngle.CONTRIBUTION:
        return [], [], {}
    contributions, contrib_evidence = build_dimension_breakdowns(report, brief, trace_id=trace_id)
    decompositions, decomp_evidence = build_decomposition(report, brief, trace_id=trace_id)
    grouped_ratio_evidence = build_grouped_ratio_attributions(report, brief, trace_id=trace_id)
    return contributions, decompositions, {**contrib_evidence, **decomp_evidence, **grouped_ratio_evidence}

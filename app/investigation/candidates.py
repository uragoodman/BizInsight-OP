"""Server-side legal-candidate generation and deterministic Top-K pruning."""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Mapping

from app.investigation.contracts import (
    AnalysisAction,
    AnalysisTarget,
    AnalyzeDistributionAction,
    AnalyzeEntityDistributionAction,
    AnalyzeAssociationAction,
    AnalyzeTrendAction,
    BreakdownDimensionAction,
    CandidateAction,
    CandidateFeatures,
    CheckAnomalyAction,
    CheckDataQualityAction,
    ComparePeriodAction,
    DecomposeMetricAction,
    UnresolvedBranch,
    branch_share_for_priority,
)


def candidate_id_for(action: AnalysisAction) -> str:
    payload = action.model_dump(mode="json")
    if action.action_type in {"DRILL_DOWN", "BREAKDOWN_DIMENSION"}:
        payload["action_type"] = "DIMENSION_BREAKDOWN"
        payload.pop("parent_evidence_id", None)
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "cand_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def generate_candidate_actions(
    target: AnalysisTarget,
    *,
    allowed_methods: Iterable[str],
    allowed_dimensions: Iterable[str],
    branches: Iterable[UnresolvedBranch] = (),
    focus_branch_id: str | None = None,
    executed_candidate_ids: Iterable[str] = (),
    data_available_methods: Iterable[str] | None = None,
    supported_action_types: Iterable[str] = ("BREAKDOWN_DIMENSION",),
    factor_metrics_by_method: Mapping[str, tuple[str, ...]] | None = None,
    quality_check_ids: Iterable[str] = (),
    entity_distribution_metric_ids: Iterable[str] = (),
    association_pair_ids_by_metric: Mapping[str, tuple[str, ...]] | None = None,
    top_k: int | None = 6,
) -> list[CandidateAction]:
    """Generate authorized, capability-backed candidates, never LLM invented.

    ``supported_action_types``, methods, factor metrics, dimensions, and quality
    checks must be derived by a trusted backend capability resolver. Default
    action support is intentionally narrow for backwards-safe adoption.
    """
    if top_k is not None and (top_k < 1 or top_k > 6):
        raise ValueError("top_k must be None or between 1 and 6")
    methods = set(allowed_methods)
    available = set(data_available_methods) if data_available_methods is not None else methods
    dimensions = sorted(set(allowed_dimensions))
    action_types = set(supported_action_types)
    executed = set(executed_candidate_ids)
    open_branches = [branch for branch in branches if branch.status in {"open", "investigating"}]
    active_branch = next(
        (item for item in open_branches if item.branch_id == focus_branch_id),
        None,
    ) or max(open_branches, key=lambda item: (branch_share_for_priority(item), -item.depth), default=None)
    if active_branch and active_branch.depth >= 3:
        active_branch = None
    if active_branch:
        dimensions = [dimension for dimension in dimensions if dimension != active_branch.dimension]
    branch_metric = active_branch.metric_id if active_branch else target.metric_id
    branch_share = (
        max(0.0, branch_share_for_priority(active_branch)) if active_branch else 1.0
    )
    candidates: list[CandidateAction] = []

    def add(
        action: AnalysisAction, *, score: float, gap: str, cost: float = 1.0,
        data_availability: str = "unknown",
    ) -> None:
        candidate_id = candidate_id_for(action)
        if candidate_id in executed:
            return
        candidates.append(CandidateAction(
            candidate_id=candidate_id,
            action=action,
            branch_id=active_branch.branch_id if active_branch else None,
            features=CandidateFeatures(
                relevance_score=min(1.0, score),
                unresolved_target_share=branch_share,
                estimated_query_cost=cost,
                data_availability=data_availability,
                evidence_gap=gap,
                rule_score=score,
            ),
        ))

    has_baseline = target.baseline_start is not None and target.baseline_end is not None
    if has_baseline and "COMPARE_PERIOD" in action_types and "compare_period_v1" in methods & available:
        add(ComparePeriodAction(action_type="COMPARE_PERIOD", metric_id=branch_metric), score=0.95,
            gap=f"period_comparison:{branch_metric}", data_availability="available")
    if has_baseline and "DECOMPOSE_METRIC" in action_types:
        for method_id, factor_ids in sorted((factor_metrics_by_method or {}).items()):
            if method_id in methods & available and len(factor_ids) >= 2:
                add(DecomposeMetricAction(action_type="DECOMPOSE_METRIC", metric_id=branch_metric,
                    method_id=method_id, factor_metric_ids=list(factor_ids)), score=0.9,
                    gap=f"metric_decomposition:{branch_metric}:{method_id}", data_availability="available")
    if active_branch is None:
        dimension_action_type = "BREAKDOWN_DIMENSION"
    elif active_branch.branch_type == "root_target":
        # The root target is the starting point of an investigation, so the
        # agent-facing action is a drill-down when that semantic action is
        # explicitly supported. A backend that only supports the flat
        # breakdown action may expose that instead.
        dimension_action_type = (
            "DRILL_DOWN" if "DRILL_DOWN" in action_types
            else "BREAKDOWN_DIMENSION"
        )
    elif active_branch.branch_type == "metric_effect":
        # A factor branch (for example order_count under GMV) has not yet been
        # split by a dimension. Prefer the agent-facing DRILL_DOWN action for
        # an evidence-backed factor branch; adapters may explicitly expose the
        # same first-level query as BREAKDOWN_DIMENSION instead.
        dimension_action_type = (
            "DRILL_DOWN" if "DRILL_DOWN" in action_types
            else "BREAKDOWN_DIMENSION"
        )
    else:
        # Dimension-contribution descendants carry inherited group filters;
        # further exploration is a true drill-down and cannot be relabeled as
        # an unfiltered breakdown if the backend does not support it.
        dimension_action_type = "DRILL_DOWN"
    if has_baseline and dimension_action_type in action_types:
        for dimension in dimensions:
            # The registry selects the mathematical contract. A ratio metric
            # must never silently fall back to additive contributions.
            for method_id in sorted(methods & available & {
                "additive_change_v1", "symmetric_mix_within_v1",
            }):
                action = BreakdownDimensionAction(
                    action_type=dimension_action_type,
                    metric_id=branch_metric,
                    dimension=dimension,
                    parent_evidence_id=active_branch.parent_evidence_id if active_branch else None,
                    inherited_filters=(dict(active_branch.inherited_filters) if active_branch else {}),
                    method_id=method_id,
                )
                depth_cost = 0.1 * active_branch.depth if active_branch else 0.0
                score = 0.8 if active_branch else 0.55
                if method_id == "symmetric_mix_within_v1":
                    score += 0.02
                add(action, score=score,
                    gap=f"breakdown:{branch_metric}:{dimension}:{method_id}", cost=1.0 + depth_cost,
                    data_availability="available")
    if "ANALYZE_TREND" in action_types:
        add(AnalyzeTrendAction(action_type="ANALYZE_TREND", metric_id=branch_metric), score=0.65,
            gap=f"trend:{branch_metric}")
    if "ANALYZE_DISTRIBUTION" in action_types:
        add(AnalyzeDistributionAction(action_type="ANALYZE_DISTRIBUTION", metric_id=branch_metric,
            statistic="distribution_summary"), score=0.6, gap=f"distribution:{branch_metric}")
    if (
        "ANALYZE_ENTITY_DISTRIBUTION" in action_types
        and branch_metric in set(entity_distribution_metric_ids)
    ):
        add(AnalyzeEntityDistributionAction(
            action_type="ANALYZE_ENTITY_DISTRIBUTION", metric_id=branch_metric,
        ), score=0.62, gap=f"entity_distribution:{branch_metric}")
    if "ANALYZE_ASSOCIATION" in action_types:
        from app.analysis.association_catalog import is_registered_pair_for_metric

        for pair_id in sorted((association_pair_ids_by_metric or {}).get(branch_metric, ())):
            if is_registered_pair_for_metric(pair_id, branch_metric):
                add(AnalyzeAssociationAction(
                    action_type="ANALYZE_ASSOCIATION", metric_id=branch_metric,
                    pair_id=pair_id,
                ), score=0.7, gap=f"association:{pair_id}", data_availability="available")
    if "CHECK_ANOMALY" in action_types:
        add(CheckAnomalyAction(action_type="CHECK_ANOMALY", metric_id=branch_metric), score=0.5,
            gap=f"anomaly:{branch_metric}")
    if "CHECK_DATA_QUALITY" in action_types:
        for check_id in sorted(set(quality_check_ids)):
            add(CheckDataQualityAction(action_type="CHECK_DATA_QUALITY", metric_id=branch_metric,
                check_id=check_id), score=0.45, gap=f"data_quality:{branch_metric}:{check_id}")

    candidates.sort(key=lambda item: (
        -item.features.rule_score,
        item.features.estimated_query_cost,
        item.action_fingerprint or item.candidate_id,
    ))
    return candidates if top_k is None else candidates[:top_k]

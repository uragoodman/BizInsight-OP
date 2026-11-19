from datetime import date, timedelta

import pytest

from app.analysis.attribution_methods import (
    AttributionDataError,
    additive_change_attribution,
    product_shapley_attribution,
    symmetric_mix_within_attribution,
    symmetric_ratio_attribution,
)
from app.investigation.candidates import generate_candidate_actions
from app.investigation.contracts import (
    ActionResult, AgentDecision, AnalysisTarget, UnresolvedBranch, WorkflowSelection,
)
from app.investigation.contracts import (
    AttributionEvidence, CandidateAction, CandidateFeatures, DataValidationResult,
    InvestigationState, StopReason,
)
from app.investigation.engine import prepare_candidates, record_action_result, run_investigation, select_candidate
from app.investigation.evidence import scope_fingerprint
from app.investigation.fingerprint import action_fingerprint
from app.investigation.stop_policy import (
    hard_stop_reason, progress_snapshot, soft_stop_blockers, terminal_progress_status,
)


def _target() -> AnalysisTarget:
    return AnalysisTarget(
        metric_id="gmv", current_start=date(2026, 2, 1), current_end=date(2026, 2, 28),
        baseline_start=date(2026, 1, 1), baseline_end=date(2026, 1, 31),
        canonical_scope={"tenant_id": "t1", "seller_ids": ["s1"]},
        scope_fingerprint="scope-v1", requested_analysis_type="diagnostic",
    )


def _workflow_selection(candidate_id: str) -> WorkflowSelection:
    return WorkflowSelection(candidate_id=candidate_id)


def test_analysis_target_scope_is_deeply_immutable_and_remains_json_serializable():
    source_scope = {"tenant_id": "t1", "filters": {"region": ["north"]}}
    target = AnalysisTarget(
        metric_id="gmv", current_start=date(2026, 2, 1), current_end=date(2026, 2, 28),
        baseline_start=date(2026, 1, 1), baseline_end=date(2026, 1, 31),
        canonical_scope=source_scope, scope_fingerprint="scope-v1",
        requested_analysis_type="diagnostic",
    )

    source_scope["filters"]["region"].append("south")
    assert target.canonical_scope["filters"]["region"] == ("north",)
    with pytest.raises(TypeError, match="immutable"):
        target.canonical_scope["filters"]["region"] += ("south",)
    with pytest.raises(TypeError, match="immutable"):
        target.canonical_scope["filters"]["new"] = True

    dumped = target.model_dump(mode="json")
    assert dumped["canonical_scope"] == {
        "tenant_id": "t1", "filters": {"region": ["north"]},
    }


def test_target_without_baseline_only_generates_non_comparative_actions():
    from app.investigation.contracts import AnalyzeTrendAction

    target = AnalysisTarget(
        metric_id="gmv", current_start=date(2026, 2, 1), current_end=date(2026, 2, 28),
        canonical_scope={"tenant_id": "t1", "seller_ids": ["s1"]},
        scope_fingerprint="scope-trend-only", requested_analysis_type="trend",
    )
    candidates = generate_candidate_actions(
        target,
        allowed_methods={"additive_change_v1", "compare_period_v1"},
        allowed_dimensions={"product_category"},
        supported_action_types={"COMPARE_PERIOD", "DECOMPOSE_METRIC", "BREAKDOWN_DIMENSION", "ANALYZE_TREND"},
        factor_metrics_by_method={"product_shapley_v1": ("order_count", "avg_order_value")},
    )

    assert len(candidates) == 1
    assert isinstance(candidates[0].action, AnalyzeTrendAction)


def test_agent_investigation_path_decomposes_then_drills_the_largest_supported_factor():
    from app.analysis.attribution_methods import product_shapley_attribution
    from app.investigation.evaluation import (
        ActionMatcher, BranchMatcher, EvidenceMatcher, InvestigationEvaluationCase,
        evaluate_investigation_path,
    )
    from app.investigation.contracts import DecomposeMetricAction

    target = _target()
    state = InvestigationState(target=target, remaining_budget=4)
    factor_ids = ("order_count", "avg_order_value")

    def candidate_factory(current_state):
        if not current_state.action_results:
            return generate_candidate_actions(
                target, allowed_methods={"product_shapley_v1"}, allowed_dimensions=(),
                supported_action_types={"DECOMPOSE_METRIC"},
                factor_metrics_by_method={"product_shapley_v1": factor_ids},
            )
        if len(current_state.action_results) == 1:
            return generate_candidate_actions(
                target, allowed_methods={"additive_change_v1"},
                allowed_dimensions={"product_category"}, branches=current_state.branches,
                focus_branch_id=current_state.current_focus_branch_id,
                supported_action_types={"BREAKDOWN_DIMENSION", "DRILL_DOWN"},
            )
        current_state.stop_reason = StopReason(
            code="CAPABILITY_GAP",
            detail="Fixture has no deeper category-level query source.",
            initiator="backend",
        )
        return []

    def selector(candidates, current_state):
        selected = next(
            item for item in candidates
            if (
                isinstance(item.action, DecomposeMetricAction)
                if not current_state.action_results
                else item.action.metric_id == "order_count"
            )
        )
        return AgentDecision(
            candidate_set_id=current_state.candidate_set.candidate_set_id,
            decision="select_candidate", candidate_id=selected.candidate_id,
            rationale="Select the largest unresolved factor supported by the current evidence.",
        )

    def executor(candidate, _current_state):
        if candidate.action.action_type == "DECOMPOSE_METRIC":
            result = product_shapley_attribution(
                metric_id="gmv",
                baseline_factors={"order_count": 100, "avg_order_value": 10},
                current_factors={"order_count": 80, "avg_order_value": 10.5},
                evidence_ids=["ev-gmv-factor-decomposition"],
            )
            values = {
                "baseline_value": result.baseline_value,
                "current_value": result.current_value,
                "total_delta": result.total_delta,
                "factor_metrics": list(factor_ids),
                "raw_components": [item.model_dump(mode="json") for item in result.raw_components],
                "display_components": [item.model_dump(mode="json") for item in result.display_components],
                "display_method": result.display_method,
                "interaction_delta": next(
                    item.value for item in result.raw_components
                    if item.component_type == "interaction_effect"
                ),
                "residual_value": result.residual_value,
                "residual_explainable": result.residual_explainable,
            }
            return [AttributionEvidence(
                evidence_id="ev-gmv-factor-decomposition",
                evidence_type="metric_decomposition", metric_id="gmv",
                values=values, source_action_id=candidate.candidate_id,
                source_ref="deterministic_product_shapley_fixture",
                scope_fingerprint=target.scope_fingerprint,
                method_id="product_shapley_v1", method_version="1.0",
                authorized_metric_provenance="evaluation_fixture",
            )]

        assert candidate.action.metric_id == "order_count"
        return [AttributionEvidence(
            evidence_id="ev-order-category-breakdown",
            evidence_type="dimension_contribution", metric_id="order_count",
            values={
                "attribution_status": "ok", "total_delta": -205.0,
                "components": [{"dimension_value": "Furniture", "delta_value": -205.0}],
                "residual_value": 0.0, "residual_type": "unknown",
                "residual_explainable": True,
            },
            source_action_id=candidate.candidate_id,
            source_ref="deterministic_additive_fixture",
            scope_fingerprint=target.scope_fingerprint,
            method_id="additive_change_v1", method_version="1.0",
            authorized_metric_provenance="evaluation_fixture",
        )]

    result = run_investigation(
        state, candidate_factory=candidate_factory, selector=selector, executor=executor,
        current_period={"start": "2026-02-01", "end": "2026-02-28"},
        baseline_period={"start": "2026-01-01", "end": "2026-01-31"},
        max_actions=4,
    )

    case = InvestigationEvaluationCase(
        case_id="gmv-factor-then-largest-branch", target_metric_id="gmv", max_actions=4,
        required_action_groups=[
            [ActionMatcher(action_type="DECOMPOSE_METRIC", metric_id="gmv", method_id="product_shapley_v1")],
            [ActionMatcher(action_type="DRILL_DOWN", metric_id="order_count", dimension="product_category")],
        ],
        allowed_action_matchers=[
            ActionMatcher(action_type="DECOMPOSE_METRIC", metric_id="gmv"),
            ActionMatcher(action_type="DRILL_DOWN", metric_id="order_count", dimension="product_category"),
        ],
        required_evidence_groups=[
            [EvidenceMatcher(evidence_type="metric_decomposition", metric_id="gmv")],
            [EvidenceMatcher(evidence_type="dimension_contribution", metric_id="order_count")],
        ],
        required_branch_groups=[[
            BranchMatcher(
                metric_id="order_count", dimension="__metric_factor__",
                dimension_value="order_count", acceptable_statuses=["explained"],
            ),
        ]],
        allowed_stop_reasons=["CAPABILITY_GAP"], expected_capability_gap=True,
        require_conclusion_evidence=True,
    )
    evaluation = evaluate_investigation_path(
        case, result,
        conclusion_evidence_ids=["ev-gmv-factor-decomposition", "ev-order-category-breakdown"],
    )

    assert evaluation.passed, evaluation.issues
    assert [item.action_type for item in result.action_results] == ["DECOMPOSE_METRIC", "DRILL_DOWN"]
    assert result.action_results[1].metric_id == "order_count"
    assert next(branch for branch in result.branches if branch.dimension == "__metric_factor__"
                and branch.dimension_value == "order_count").status == "explained"
    assert result.stop_reason.code == "CAPABILITY_GAP"


def test_target_rejects_partial_baseline_period():
    with pytest.raises(ValueError, match="both be set or both be omitted"):
        AnalysisTarget(
            metric_id="gmv", current_start=date(2026, 2, 1), current_end=date(2026, 2, 28),
            baseline_start=date(2026, 1, 1),
            canonical_scope={"tenant_id": "t1"}, scope_fingerprint="scope-partial-baseline",
            requested_analysis_type="trend",
        )


def test_trend_investigation_runs_without_a_baseline_period():
    target = AnalysisTarget(
        metric_id="gmv", current_start=date(2026, 2, 1), current_end=date(2026, 2, 28),
        canonical_scope={"tenant_id": "t1", "seller_ids": ["s1"]},
        scope_fingerprint="scope-trend-only", requested_analysis_type="trend",
    )
    state = InvestigationState(target=target, remaining_budget=1)
    candidates = generate_candidate_actions(
        target, allowed_methods=set(), allowed_dimensions=(),
        supported_action_types={"ANALYZE_TREND"},
    )

    result = run_investigation(
        state,
        candidate_factory=lambda _state: candidates,
        selector=lambda legal, _state: WorkflowSelection(candidate_id=legal[0].candidate_id),
        executor=lambda candidate, _state: [AttributionEvidence(
            evidence_id="trend-without-baseline", evidence_type="trend", metric_id="gmv",
            values={"result": {"status": "ok", "points": [{"bucket": "2026-02-01", "value": 8.0}]}},
            source_action_id=candidate.candidate_id, source_ref="trend-query",
            scope_fingerprint=target.scope_fingerprint, method_id="observed_trend_v1",
            method_version="1.0", authorized_metric_provenance="evaluation_fixture",
        )],
        current_period={"start": "2026-02-01", "end": "2026-02-28"},
        max_actions=1,
    )

    assert result.action_results[0].status == "completed"
    assert result.evidence[0].evidence_id == "trend-without-baseline"


def test_distribution_evidence_requires_finite_ordered_statistics_over_time_buckets():
    from app.investigation.contracts import AnalyzeDistributionAction

    action = AnalyzeDistributionAction(
        action_type="ANALYZE_DISTRIBUTION", metric_id="gmv", statistic="distribution_summary",
    )
    candidate = CandidateAction(candidate_id="distribution-candidate", action=action)
    valid_analysis = {
        "status": "ok", "distribution_basis": "time_bucket_values", "count": 3,
        "statistics": {
            "count": 3, "mean": 2.0, "std": 0.8, "min": 1.0,
            "q1": 1.5, "median": 2.0, "q3": 2.5, "max": 3.0,
        },
        "min_bucket": "2026-02-01", "max_bucket": "2026-02-03",
    }

    def evidence(evidence_id: str, analysis: dict) -> AttributionEvidence:
        return AttributionEvidence(
            evidence_id=evidence_id, evidence_type="distribution_summary", metric_id="gmv",
            claim="时间桶指标值分布统计", values={"series_analysis": analysis},
            source_action_id=candidate.candidate_id, source_ref="query_metric_statistics",
            scope_fingerprint="scope-v1", method_id="descriptive_distribution_v1",
            method_version="1.0", authorized_metric_provenance="exact",
        )

    valid_state = InvestigationState(target=_target(), remaining_budget=1)
    assert record_action_result(
        valid_state, candidate, [evidence("valid-distribution", valid_analysis)],
        current_period={},
    )
    assert valid_state.action_results[0].status == "completed"

    malformed = []
    wrong_mean_type = {**valid_analysis, "statistics": {**valid_analysis["statistics"], "mean": True}}
    malformed.append(wrong_mean_type)
    wrong_count = {**valid_analysis, "statistics": {**valid_analysis["statistics"], "count": 4}}
    malformed.append(wrong_count)
    wrong_quantile_order = {**valid_analysis, "statistics": {**valid_analysis["statistics"], "q1": 2.5}}
    malformed.append(wrong_quantile_order)
    malformed.append({**valid_analysis, "min_bucket": "not-a-date"})
    malformed.append({key: value for key, value in valid_analysis.items() if key != "distribution_basis"})

    for index, analysis in enumerate(malformed):
        state = InvestigationState(target=_target(), remaining_budget=1)
        accepted = record_action_result(
            state, candidate, [evidence(f"invalid-distribution-{index}", analysis)],
            current_period={},
        )
        assert not accepted
        assert state.action_results[0].failure_code == "INVALID_DATA"

    bad_residual_evidence = evidence("invalid-residual-tolerance", valid_analysis).model_copy(update={
        "values": {
            "series_analysis": valid_analysis, "residual_value": 0.1,
            "residual_tolerance": 0.01, "residual_explainable": True,
        },
    })
    state = InvestigationState(target=_target(), remaining_budget=1)
    assert not record_action_result(
        state, candidate, [bad_residual_evidence], current_period={},
    )
    assert state.action_results[0].failure_code == "INVALID_DATA"


def test_unquantified_root_branch_is_valid_but_descendant_requires_parent_evidence():
    root = UnresolvedBranch(
        branch_id="root-gmv", metric_id="gmv", depth=0,
        parent_evidence_id=None, effect_value=None, effect_share_of_root=None,
        status="open",
    )
    assert root.parent_evidence_id is None
    assert root.effect_value is None
    assert root.effect_share_of_root is None

    with pytest.raises(ValueError, match="must reference parent evidence"):
        UnresolvedBranch(
            branch_id="child-without-parent", metric_id="gmv", depth=1,
            parent_evidence_id=None, effect_value=-5.0, effect_share_of_root=0.5,
        )


def test_investigation_rejects_orphaned_or_cyclic_branch_lineage():
    target = _target()
    evidence_items = [
        AttributionEvidence(
            evidence_id=evidence_id, evidence_type="fixture", metric_id="gmv",
            source_action_id=action_id, source_ref="fixture",
            scope_fingerprint=target.scope_fingerprint, method_id="compare_period_v1",
            method_version="1.0", authorized_metric_provenance="exact",
        )
        for evidence_id, action_id in (("ev-parent", "action-parent"), ("ev-child", "action-child"))
    ]
    action_results = [
        ActionResult(
            action_id=action_id, candidate_id=action_id, status="completed",
            branch_id=("parent" if evidence_id == "ev-parent" else "child"),
            metric_id="gmv", evidence_ids=[evidence_id],
            validation_result=DataValidationResult(status="valid"),
        )
        for evidence_id, action_id in (("ev-parent", "action-parent"), ("ev-child", "action-child"))
    ]
    parent = UnresolvedBranch(
        branch_id="parent", branch_type="root_target", metric_id="gmv", depth=0,
    )
    child = UnresolvedBranch(
        branch_id="child", parent_branch_id="parent", parent_evidence_id="ev-parent",
        branch_type="dimension_contribution", metric_id="gmv", depth=1,
    )
    state = InvestigationState(
        target=target, branches=[parent, child], current_focus_branch_id="child",
        evidence=evidence_items, action_results=action_results,
    )
    assert state.current_focus_branch_id == "child"

    with pytest.raises(ValueError, match="missing parent branch"):
        InvestigationState(target=target, branches=[child])

    cyclic_parent = parent.model_copy(update={
        "parent_branch_id": "child", "parent_evidence_id": "ev-child",
    })
    with pytest.raises(ValueError, match="cycles"):
        InvestigationState(
            target=target, branches=[cyclic_parent, child],
            evidence=evidence_items, action_results=action_results,
        )


def test_child_metric_dimension_effect_does_not_claim_root_metric_share():
    from app.investigation.contracts import BreakdownDimensionAction
    from app.investigation.engine import ensure_root_branch

    state = InvestigationState(target=_target(), remaining_budget=1, root_effect_value=-100)
    root = ensure_root_branch(state)
    parent_evidence = AttributionEvidence(
        evidence_id="ev-product", evidence_type="metric_decomposition", metric_id="gmv",
        source_action_id="product-candidate", source_ref="fixture",
        scope_fingerprint=state.target.scope_fingerprint, method_id="product_shapley_v1",
        method_version="1.0", authorized_metric_provenance="exact",
    )
    state.evidence.append(parent_evidence)
    state.action_results.append(ActionResult(
        action_id="product-action", candidate_id="product-candidate", status="completed",
        branch_id=root.branch_id, metric_id="gmv", evidence_ids=[parent_evidence.evidence_id],
        validation_result=DataValidationResult(status="valid"),
    ))
    factor = UnresolvedBranch(
        branch_id="orders-effect", parent_branch_id=root.branch_id,
        parent_evidence_id="ev-product", branch_type="metric_effect",
        metric_id="order_count", dimension="__metric_factor__",
        dimension_value="order_count", effect_value=-70, effect_unit="GBP",
        root_attributed_effect_value=-70, root_effect_share=0.7,
        effect_share_of_root=0.7, depth=0,
    )
    state.branches.append(factor)
    candidate = CandidateAction(
        candidate_id="orders-by-category",
        action={
            "action_type": "DRILL_DOWN", "metric_id": "order_count",
            "dimension": "product_category", "parent_evidence_id": "ev-product",
            "inherited_filters": {}, "method_id": "additive_change_v1",
        },
        branch_id=factor.branch_id,
    )
    evidence = AttributionEvidence(
        evidence_id="ev-orders-by-category", evidence_type="dimension_contribution",
        metric_id="order_count", values={
            "total_delta": -10,
            "components": [{"dimension_value": "furniture", "delta_value": -10}],
            "residual_value": 0, "residual_type": "unknown",
            "residual_explainable": True, "attribution_status": "ok",
        }, unit="orders", source_action_id=candidate.candidate_id,
        source_ref="fixture", scope_fingerprint=state.target.scope_fingerprint,
        method_id="additive_change_v1", method_version="1.0",
        authorized_metric_provenance="exact",
    )

    assert record_action_result(
        state, candidate, [evidence], current_period={}, baseline_period={},
    )
    child = next(branch for branch in state.branches if branch.dimension_value == "furniture")
    assert child.parent_branch_id == factor.branch_id
    assert child.effect_unit == "orders"
    assert child.effect_value == -10
    assert child.root_attributed_effect_value is None
    assert child.root_effect_share is None
    assert child.effect_share_of_root is None
    InvestigationState.model_validate(state.model_dump(mode="json"))


def test_unquantified_root_branch_can_generate_candidates_without_faking_impact():
    root = UnresolvedBranch(
        branch_id="root-gmv", metric_id="gmv", depth=0,
        parent_evidence_id=None, effect_value=None, effect_share_of_root=None,
        status="open",
    )
    candidates = generate_candidate_actions(
        _target(), allowed_methods={"additive_change_v1"},
        allowed_dimensions={"product_category"}, branches=[root],
        focus_branch_id="root-gmv", supported_action_types={"DRILL_DOWN"},
    )

    assert len(candidates) == 1
    assert candidates[0].branch_id == "root-gmv"
    assert candidates[0].features.unresolved_target_share == 0.0
    assert candidates[0].action.parent_evidence_id is None


def test_root_branch_result_updates_the_exact_selected_branch():
    from app.investigation.contracts import BreakdownDimensionAction

    untouched = UnresolvedBranch(
        branch_id="root-gmv-secondary", metric_id="gmv", depth=0,
        parent_evidence_id=None, effect_value=None, effect_share_of_root=None,
        status="open",
    )
    focused = UnresolvedBranch(
        branch_id="root-gmv-focused", metric_id="gmv", depth=0,
        parent_evidence_id=None, effect_value=None, effect_share_of_root=None,
        status="investigating",
    )
    state = InvestigationState(
        target=_target(), branches=[untouched, focused], remaining_budget=1,
        current_focus_branch_id=focused.branch_id,
    )
    candidate = CandidateAction(
        candidate_id="root-branch-breakdown", branch_id=focused.branch_id,
        action=BreakdownDimensionAction(
            action_type="DRILL_DOWN", metric_id="gmv", dimension="product_category",
            parent_evidence_id=None, method_id="additive_change_v1",
        ),
    )
    evidence = AttributionEvidence(
        evidence_id="ev-root-branch", evidence_type="dimension_contribution", metric_id="gmv",
        values={
            "total_delta": -20,
            "components": [{"dimension_value": "furniture", "delta_value": -20}],
            "residual_value": 0,
            "residual_type": "rounding",
            "residual_explainable": True,
            "attribution_status": "ok",
        },
        source_action_id=candidate.candidate_id, source_ref="mcp-result",
        scope_fingerprint="scope-v1", method_id="additive_change_v1",
        method_version="1.0", authorized_metric_provenance="exact",
    )

    assert record_action_result(
        state, candidate, [evidence], current_period={}, baseline_period={},
    )
    branches = {branch.branch_id: branch for branch in state.branches}
    assert branches[untouched.branch_id].status == "open"
    assert branches[focused.branch_id].status == "explained"
    assert branches["branch_ev-root-branch_product_category_furniture"].depth == 1


def test_product_attribution_retains_interaction_and_shapley_display():
    result = product_shapley_attribution(
        metric_id="gmv", baseline_factors={"orders": 10, "aov": 5},
        current_factors={"orders": 12, "aov": 6},
    )
    assert result.total_delta == 22
    assert [round(item.value, 6) for item in result.raw_components] == [10, 10, 2]
    assert sum(item.value for item in result.display_components) == pytest.approx(22)
    assert result.display_method == "shapley"
    assert result.causal_claim is False


def test_product_attribution_rejects_delta_below_endpoint_float_precision():
    scale = 1e8
    with pytest.raises(AttributionDataError, match="below endpoint float precision"):
        product_shapley_attribution(
            metric_id="gmv", baseline_factors={"orders": scale, "aov": scale},
            current_factors={"orders": scale + 1, "aov": scale - 1},
        )


def test_product_attribution_records_resolvable_rounding_residual():
    scale = 1e8
    result = product_shapley_attribution(
        metric_id="gmv", baseline_factors={"orders": scale, "aov": scale},
        current_factors={"orders": scale + 1, "aov": scale + 1},
    )
    raw_sum = sum(item.value for item in result.raw_components)
    display_sum = sum(item.value for item in result.display_components)

    assert result.residual_type == "rounding"
    assert result.residual_value != 0.0
    assert abs(result.residual_value) <= result.residual_tolerance
    assert raw_sum + result.residual_value == pytest.approx(result.total_delta)
    assert display_sum + result.residual_value == pytest.approx(result.total_delta)


def test_product_decomposition_evidence_requires_method_consistent_interaction_and_display():
    from app.analysis.attribution_methods import product_shapley_attribution
    from app.investigation.contracts import DecomposeMetricAction

    target = _target()
    method_result = product_shapley_attribution(
        metric_id="gmv", baseline_factors={"orders": 10, "aov": 5},
        current_factors={"orders": 12, "aov": 6},
    )
    action = DecomposeMetricAction(
        action_type="DECOMPOSE_METRIC", metric_id="gmv",
        method_id="product_shapley_v1", factor_metric_ids=["orders", "aov"],
    )
    candidate = CandidateAction(candidate_id="product-shapley", action=action)
    evidence = AttributionEvidence(
        evidence_id="product-shapley-evidence", evidence_type="metric_decomposition",
        metric_id="gmv", values={
            "total_delta": method_result.total_delta,
            "factor_metrics": ["orders", "aov"],
            "raw_components": [item.model_dump(mode="json") for item in method_result.raw_components],
            "display_method": method_result.display_method,
            "display_components": [item.model_dump(mode="json") for item in method_result.display_components],
            "interaction_delta": next(
                item.value for item in method_result.raw_components
                if item.component_type == "interaction_effect"
            ),
            "residual_value": method_result.residual_value,
            "residual_tolerance": method_result.residual_tolerance,
            "residual_note": method_result.residual_note,
            "residual_explainable": method_result.residual_explainable,
        }, source_action_id="product-shapley", source_ref="test",
        scope_fingerprint=target.scope_fingerprint, method_id="product_shapley_v1",
        method_version="1.0", authorized_metric_provenance="test",
    )

    accepted_state = InvestigationState(target=target, remaining_budget=1)
    assert record_action_result(
        accepted_state, candidate, [evidence], current_period={}, baseline_period={},
    )
    assert accepted_state.action_results[0].status == "completed"
    assert accepted_state.action_results[0].calculation_progress is not None
    assert accepted_state.calculation_progress.reconciliation_ratio == pytest.approx(1.0)
    assert accepted_state.action_results[0].residual_tolerance == method_result.residual_tolerance
    assert accepted_state.action_results[0].residual_note == method_result.residual_note
    assert accepted_state.action_results[0].validation_result.status == "valid"
    assert accepted_state.progress.investigation_progress == "stalled"
    factor_branches = [item for item in accepted_state.branches if item.branch_type == "metric_effect"]
    root_branch = next(item for item in accepted_state.branches if item.branch_type == "root_target")
    assert {item.metric_id for item in factor_branches} == {"orders", "aov"}
    assert root_branch.status == "explained"
    assert all(item.parent_branch_id == root_branch.branch_id for item in factor_branches)
    assert all(item.parent_evidence_id == evidence.evidence_id for item in factor_branches)
    factor_candidates = generate_candidate_actions(
        target, allowed_methods={"additive_change_v1"},
        allowed_dimensions={"product_category"}, branches=factor_branches,
        focus_branch_id=next(item.branch_id for item in factor_branches if item.metric_id == "orders"),
        supported_action_types={"DRILL_DOWN"},
    )
    assert factor_candidates
    assert factor_candidates[0].action.action_type == "DRILL_DOWN"
    assert factor_candidates[0].action.metric_id == "orders"
    assert factor_candidates[0].action.parent_evidence_id == evidence.evidence_id

    malformed_values = dict(evidence.values)
    malformed_values["raw_components"] = [dict(item) for item in malformed_values["raw_components"]]
    malformed_values["raw_components"][-1]["component_type"] = "factor_effect"
    rejected_state = InvestigationState(target=target, remaining_budget=1)
    assert not record_action_result(
        rejected_state, candidate, [evidence.model_copy(update={"values": malformed_values})],
        current_period={}, baseline_period={},
    )
    assert rejected_state.action_results[0].failure_code == "INVALID_DATA"


def test_ratio_decomposition_branches_keep_component_change_unit_not_metric_level_unit():
    from app.investigation.contracts import DecomposeMetricAction
    from app.investigation.engine import _record_unresolved_branches

    target = AnalysisTarget(
        metric_id="freight_rate", current_start=date(2024, 5, 4), current_end=date(2024, 5, 4),
        baseline_start=date(2024, 5, 3), baseline_end=date(2024, 5, 3),
        canonical_scope={"tenant_id": "platform"}, scope_fingerprint="freight-rate-scope",
        requested_analysis_type="diagnostic",
    )
    state = InvestigationState(target=target)
    factor_ids = ["freight_value", "gmv"]
    action = DecomposeMetricAction(
        action_type="DECOMPOSE_METRIC", metric_id="freight_rate",
        method_id="symmetric_ratio_v1", factor_metric_ids=factor_ids,
    )
    candidate = CandidateAction(candidate_id="freight-rate-decomposition", action=action)
    evidence = AttributionEvidence(
        evidence_id="freight-rate-decomposition-evidence", evidence_type="metric_decomposition",
        metric_id="freight_rate", values={
            "total_delta": 0.5,
            "factor_metrics": factor_ids,
            "display_components": [
                {"component_id": "numerator_effect", "value": 0.8, "unit": "percentage_points"},
                {"component_id": "denominator_effect", "value": -0.3, "unit": "percentage_points"},
            ],
        }, unit="%", source_action_id="freight-rate-decomposition", source_ref="offline-test",
        scope_fingerprint=target.scope_fingerprint, method_id="symmetric_ratio_v1",
        method_version="1.0", authorized_metric_provenance="test",
    )

    _record_unresolved_branches(state, candidate, [evidence])

    factors = {item.metric_id: item for item in state.branches if item.branch_type == "metric_effect"}
    assert factors["freight_value"].effect_unit == "percentage_points"
    assert factors["gmv"].effect_unit == "percentage_points"


def test_symmetric_mix_within_reconciles_entry_and_exit_as_structural():
    result = symmetric_mix_within_attribution(
        metric_id="avg_freight",
        baseline_groups={"A": {"numerator": 100, "denominator": 10},
                         "B": {"numerator": 200, "denominator": 20}},
        current_groups={"A": {"numerator": 240, "denominator": 20},
                        "C": {"numerator": 120, "denominator": 10}},
    )
    assert sum(item.value for item in result.raw_components) == pytest.approx(result.total_delta)
    assert any(item.component_type == "entry_structural_effect" for item in result.raw_components)
    assert any(item.component_type == "exit_structural_effect" for item in result.raw_components)
    assert not any(item.component_id == "B:within" for item in result.raw_components)
    assert result.causal_claim is False


def test_symmetric_mix_within_rejects_zero_period_denominator():
    with pytest.raises(AttributionDataError, match="zero effective denominator"):
        symmetric_mix_within_attribution(
            metric_id="rate", baseline_groups={"A": {"numerator": 0, "denominator": 0}},
            current_groups={"A": {"numerator": 1, "denominator": 1}},
        )


def test_additive_attribution_rejects_top_k_coverage_gap():
    with pytest.raises(AttributionDataError, match="coverage"):
        additive_change_attribution(
            metric_id="gmv", baseline_value=100, current_value=120,
            baseline_groups={"A": 90}, current_groups={"A": 100},
        )


@pytest.mark.parametrize("bad_groups", [None, [], "not-a-mapping"])
def test_additive_attribution_rejects_malformed_group_containers(bad_groups):
    with pytest.raises(AttributionDataError, match="mapping"):
        additive_change_attribution(
            metric_id="gmv", baseline_value=10, current_value=12,
            baseline_groups=bad_groups, current_groups={"A": 12},
        )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf"), True])
def test_additive_attribution_rejects_non_finite_and_boolean_inputs(bad_value):
    with pytest.raises(AttributionDataError):
        additive_change_attribution(
            metric_id="gmv", baseline_value=bad_value, current_value=10,
            baseline_groups={"A": 1}, current_groups={"A": 10},
        )
    with pytest.raises(AttributionDataError):
        additive_change_attribution(
            metric_id="gmv", baseline_value=1, current_value=10,
            baseline_groups={"A": 1}, current_groups={"A": bad_value},
        )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf"), False])
def test_product_shapley_rejects_non_finite_and_boolean_factors(bad_value):
    with pytest.raises(AttributionDataError):
        product_shapley_attribution(
            metric_id="gmv", baseline_factors={"orders": 1, "aov": 2},
            current_factors={"orders": bad_value, "aov": 3},
        )


@pytest.mark.parametrize("bad_metric_id", [None, "", 7])
def test_attribution_methods_normalize_invalid_metric_id_errors(bad_metric_id):
    methods = [
        lambda: additive_change_attribution(
            metric_id=bad_metric_id, baseline_value=1, current_value=2,
            baseline_groups={"A": 1}, current_groups={"A": 2},
        ),
        lambda: symmetric_mix_within_attribution(
            metric_id=bad_metric_id,
            baseline_groups={"A": {"numerator": 1, "denominator": 2}},
            current_groups={"A": {"numerator": 2, "denominator": 2}},
        ),
        lambda: symmetric_ratio_attribution(
            metric_id=bad_metric_id, baseline_numerator=1, current_numerator=2,
            baseline_denominator=2, current_denominator=2,
        ),
        lambda: product_shapley_attribution(
            metric_id=bad_metric_id, baseline_factors={"a": 1, "b": 2},
            current_factors={"a": 2, "b": 2},
        ),
    ]
    for call in methods:
        with pytest.raises(AttributionDataError, match="metric_id"):
            call()


@pytest.mark.parametrize("bad_evidence_ids", ["evidence", ["", "valid"], [None]])
def test_attribution_methods_normalize_invalid_evidence_reference_errors(bad_evidence_ids):
    methods = [
        lambda: additive_change_attribution(
            metric_id="gmv", baseline_value=1, current_value=2,
            baseline_groups={"A": 1}, current_groups={"A": 2}, evidence_ids=bad_evidence_ids,
        ),
        lambda: symmetric_mix_within_attribution(
            metric_id="rate",
            baseline_groups={"A": {"numerator": 1, "denominator": 2}},
            current_groups={"A": {"numerator": 2, "denominator": 2}}, evidence_ids=bad_evidence_ids,
        ),
        lambda: symmetric_ratio_attribution(
            metric_id="rate", baseline_numerator=1, current_numerator=2,
            baseline_denominator=2, current_denominator=2, evidence_ids=bad_evidence_ids,
        ),
        lambda: product_shapley_attribution(
            metric_id="gmv", baseline_factors={"a": 1, "b": 2},
            current_factors={"a": 2, "b": 2}, evidence_ids=bad_evidence_ids,
        ),
    ]
    for call in methods:
        with pytest.raises(AttributionDataError, match="evidence_ids"):
            call()


def test_candidates_are_capability_driven_bounded_and_deduplicated():
    target = _target()
    candidates = generate_candidate_actions(
        target, allowed_methods={"additive_change_v1", "product_shapley_v1", "compare_period_v1"},
        allowed_dimensions={"category", "seller"},
        supported_action_types={"COMPARE_PERIOD", "DECOMPOSE_METRIC", "BREAKDOWN_DIMENSION"},
        factor_metrics_by_method={"product_shapley_v1": ("order_count", "avg_order_value")},
        top_k=6,
    )
    assert len(candidates) == 4
    assert {item.action.action_type for item in candidates} == {"COMPARE_PERIOD", "DECOMPOSE_METRIC", "BREAKDOWN_DIMENSION"}
    again = generate_candidate_actions(
        target, allowed_methods={"additive_change_v1", "product_shapley_v1", "compare_period_v1"},
        allowed_dimensions={"category", "seller"},
        supported_action_types={"COMPARE_PERIOD", "DECOMPOSE_METRIC", "BREAKDOWN_DIMENSION"},
        factor_metrics_by_method={"product_shapley_v1": ("order_count", "avg_order_value")},
        executed_candidate_ids={item.candidate_id for item in candidates},
    )
    assert again == []


def test_ratio_breakdown_candidate_keeps_registered_non_additive_method():
    candidates = generate_candidate_actions(
        _target(),
        allowed_methods={"symmetric_mix_within_v1"},
        allowed_dimensions={"month_position"},
        supported_action_types={"BREAKDOWN_DIMENSION"},
    )
    assert len(candidates) == 1
    assert candidates[0].action.method_id == "symmetric_mix_within_v1"
    assert candidates[0].action.dimension == "month_position"


def test_root_branch_breakdown_respects_supported_action_whitelist():
    target = _target()
    root = UnresolvedBranch(
        branch_id="root", branch_type="root_target", metric_id="gmv", depth=0,
    )

    candidates = generate_candidate_actions(
        target,
        allowed_methods={"additive_change_v1"},
        allowed_dimensions={"product_category"},
        branches=[root],
        focus_branch_id="root",
        supported_action_types={"BREAKDOWN_DIMENSION"},
    )

    assert len(candidates) == 1
    assert candidates[0].action.action_type == "BREAKDOWN_DIMENSION"


def test_metric_effect_branch_uses_first_dimension_breakdown():
    target = _target()
    child = UnresolvedBranch(
        branch_id="child", parent_evidence_id="evidence-parent",
        branch_type="metric_effect", metric_id="order_count", depth=1,
        dimension="__metric_factor__", inherited_filters={"product_category": "Furniture"},
    )
    common = dict(
        target=target,
        allowed_methods={"additive_change_v1"},
        allowed_dimensions={"seller_state"},
        branches=[child],
        focus_branch_id="child",
    )

    candidates = generate_candidate_actions(
        **common, supported_action_types={"BREAKDOWN_DIMENSION"},
    )

    assert len(candidates) == 1
    assert candidates[0].action.action_type == "BREAKDOWN_DIMENSION"
    assert candidates[0].action.inherited_filters == {"product_category": "Furniture"}


def test_dimension_descendant_requires_an_authorized_drill_down_action():
    target = _target()
    child = UnresolvedBranch(
        branch_id="dimension-child", parent_evidence_id="evidence-parent",
        branch_type="dimension_contribution", metric_id="order_count", depth=1,
        dimension="product_category", dimension_value="Furniture",
        inherited_filters={"product_category": "Furniture"},
    )
    common = dict(
        target=target,
        allowed_methods={"additive_change_v1"},
        allowed_dimensions={"seller_state"},
        branches=[child],
        focus_branch_id="dimension-child",
    )

    not_supported = generate_candidate_actions(
        **common, supported_action_types={"BREAKDOWN_DIMENSION"},
    )
    supported = generate_candidate_actions(
        **common, supported_action_types={"DRILL_DOWN"},
    )

    assert not_supported == []
    assert len(supported) == 1
    assert supported[0].action.action_type == "DRILL_DOWN"
    assert supported[0].action.inherited_filters == {"product_category": "Furniture"}


def test_drill_down_fingerprint_ignores_parent_evidence_but_keeps_inherited_filters():
    from app.investigation.contracts import BreakdownDimensionAction

    common = dict(metric_id="order_count", dimension="category", method_id="additive_change_v1")
    first = BreakdownDimensionAction(action_type="DRILL_DOWN", parent_evidence_id="ev1",
        inherited_filters={"category": "furniture"}, **common)
    second = BreakdownDimensionAction(action_type="BREAKDOWN_DIMENSION", parent_evidence_id="ev2",
        inherited_filters={"category": "furniture"}, **common)
    args = dict(current_period={"month": "2026-02"}, baseline_period={"month": "2026-01"},
        canonical_scope={"seller_ids": ["s1"]}, method_version="1.0")
    assert action_fingerprint(first, **args) == action_fingerprint(second, **args)
    other = BreakdownDimensionAction(action_type="DRILL_DOWN", parent_evidence_id="ev3",
        inherited_filters={"category": "electronics"}, **common)
    assert action_fingerprint(first, **args) != action_fingerprint(other, **args)
    assert action_fingerprint(first, **args) != action_fingerprint(
        first, **{**args, "metric_definition_version": "2.0"},
    )


def test_scope_fingerprint_canonicalizes_unordered_grants_and_filters():
    left = {"seller_ids": ["seller-b", "seller-a"], "filters": {"region": ["south", "north"]}}
    right = {"filters": {"region": ["north", "south"]}, "seller_ids": ["seller-a", "seller-b"]}
    assert scope_fingerprint(left) == scope_fingerprint(right)


def test_candidate_generator_turns_trusted_branch_into_filtered_child_action():
    branch = UnresolvedBranch(
        branch_id="b1", parent_evidence_id="ev-parent", metric_id="gmv",
        dimension="product_category", dimension_value="furniture",
        effect_value=-50, effect_share_of_root=0.7, depth=1,
        inherited_filters={"product_category": "furniture"},
    )
    candidates = generate_candidate_actions(
        _target(), allowed_methods={"additive_change_v1"},
        allowed_dimensions={"product_category", "seller_state"}, branches=[branch],
        supported_action_types={"DRILL_DOWN"},
    )
    assert len(candidates) == 1
    action = candidates[0].action
    assert action.action_type == "DRILL_DOWN"
    assert action.dimension == "seller_state"
    assert action.parent_evidence_id == "ev-parent"
    assert action.inherited_filters == {"product_category": "furniture"}


def test_candidate_features_allow_effect_share_over_one_for_offsetting_effects():
    # A branch can exceed 100% of the net delta when other contributions
    # offset it; this share is explanatory, not a probability.
    branch = UnresolvedBranch(
        branch_id="b-offset", parent_evidence_id="ev-parent", metric_id="gmv",
        dimension="product_category", dimension_value="furniture",
        effect_value=-140, effect_share_of_root=1.4, depth=1,
        inherited_filters={"product_category": "furniture"},
    )
    candidates = generate_candidate_actions(
        _target(), allowed_methods={"additive_change_v1"},
        allowed_dimensions={"seller_state"}, branches=[branch],
        supported_action_types={"DRILL_DOWN"},
    )
    assert candidates
    assert candidates[0].features.unresolved_target_share == pytest.approx(1.4)


def test_candidate_generator_respects_explicit_investigation_focus_branch():
    branches = [
        UnresolvedBranch(
            branch_id="largest", parent_evidence_id="ev-1", metric_id="gmv",
            dimension="category", dimension_value="furniture", effect_value=-80,
            effect_share_of_root=0.8, depth=1, inherited_filters={"category": "furniture"},
        ),
        UnresolvedBranch(
            branch_id="focused", parent_evidence_id="ev-2", metric_id="gmv",
            dimension="category", dimension_value="electronics", effect_value=-20,
            effect_share_of_root=0.2, depth=1, inherited_filters={"category": "electronics"},
        ),
    ]
    candidates = generate_candidate_actions(
        _target(), allowed_methods={"additive_change_v1"},
        allowed_dimensions={"seller_state"}, branches=branches,
        focus_branch_id="focused", supported_action_types={"DRILL_DOWN"},
    )
    assert len(candidates) == 1
    assert candidates[0].branch_id == "focused"
    assert candidates[0].action.inherited_filters == {"category": "electronics"}


def test_select_candidate_marks_selected_branch_as_current_focus():
    branch = UnresolvedBranch(
        branch_id="branch-focus", parent_evidence_id="ev-parent", metric_id="gmv",
        dimension="category", dimension_value="furniture", effect_value=-25,
        effect_share_of_root=0.7, depth=1, inherited_filters={"category": "furniture"},
    )
    state = InvestigationState(target=_target(), branches=[branch])
    candidate = CandidateAction(
        candidate_id="candidate-focus",
        action={
            "action_type": "DRILL_DOWN", "metric_id": "gmv", "dimension": "seller_state",
            "parent_evidence_id": "ev-parent", "inherited_filters": {"category": "furniture"},
            "method_id": "additive_change_v1",
        },
        branch_id="branch-focus",
    )
    state.candidate_actions = [candidate]
    selected = select_candidate(state, "candidate-focus")
    assert selected is candidate
    assert state.current_focus_branch_id == "branch-focus"
    assert state.branches[0].status == "investigating"


def test_hard_stop_priority_and_progress_are_separate_from_reconciliation():
    reason = hard_stop_reason(remaining_budget=0, candidates_available=2, authorization_blocked=True)
    assert reason and reason.code == "AUTHORIZATION_BLOCKED"
    progress = progress_snapshot(open_branch_count=1,
        highest_open_branch_share=0.3, max_remaining_action_score=0.5, remaining_budget=2)
    assert progress.investigation_progress == "in_progress"
    assert not hasattr(progress, "reconciliation_ratio")


def test_legacy_reconciliation_ratio_migrates_to_calculation_progress():
    state = InvestigationState.model_validate({
        "target": _target().model_dump(mode="json"),
        "progress": {"reconciliation_ratio": 0.75, "open_branch_count": 2},
    })
    assert state.calculation_progress.reconciliation_ratio == 0.75
    assert state.calculation_progress.data_quality_status == "unknown"
    assert state.progress.open_branch_count == 2
    assert not hasattr(state.progress, "reconciliation_ratio")

    from app.investigation.contracts import InvestigationSummary

    summary = InvestigationSummary.model_validate({
        "metric_id": "gmv", "scope_fingerprint": "scope-v1",
        "stop_reason": {"code": "COMPLETED", "initiator": "backend"},
        "progress": {"reconciliation_ratio": 0.8, "open_branch_count": 0},
        "actions_used": 0, "remaining_budget": 6,
    })
    assert summary.calculation_progress.reconciliation_ratio == 0.8
    assert not hasattr(summary.progress, "reconciliation_ratio")


def test_soft_stop_requires_completed_evidence_and_no_unresolved_branches():
    state = InvestigationState(target=_target())
    assert soft_stop_blockers(state) == ["no_completed_target_scoped_evidence"]

    state.action_results.append(ActionResult(
        action_id="action-1", candidate_id="candidate-1", status="completed",
        evidence_ids=["evidence-1"],
        validation_result=DataValidationResult(status="valid"),
    ))
    state.evidence.append(AttributionEvidence(
        evidence_id="evidence-1", evidence_type="comparison", metric_id="gmv",
        source_action_id="candidate-1", source_ref="fixture", scope_fingerprint="scope-v1",
        method_id="compare_period_v1", method_version="1.0",
        authorized_metric_provenance="exact",
    ))
    state.branches.append(UnresolvedBranch(
        branch_id="branch-open", parent_evidence_id="evidence-1", metric_id="gmv",
        dimension="product_category", dimension_value="furniture", effect_value=-10,
        effect_share_of_root=0.6, depth=1,
    ))
    assert soft_stop_blockers(state) == ["unresolved_branches:branch-open"]

    state.branches[0] = state.branches[0].model_copy(update={"status": "low_priority"})
    assert soft_stop_blockers(state) == []


def test_soft_stop_requires_base_validation_for_legacy_completed_evidence():
    state = InvestigationState(target=_target())
    state.action_results.append(ActionResult(
        action_id="legacy-action", candidate_id="legacy-candidate", status="completed",
        evidence_ids=["legacy-evidence"],
    ))
    state.evidence.append(AttributionEvidence(
        evidence_id="legacy-evidence", evidence_type="comparison", metric_id="gmv",
        source_action_id="legacy-candidate", source_ref="legacy-fixture",
        scope_fingerprint="scope-v1", method_id="compare_period_v1",
        method_version="1.0", authorized_metric_provenance="exact",
    ))

    assert soft_stop_blockers(state) == ["target_evidence_not_validated"]
    state.action_results[0] = state.action_results[0].model_copy(update={
        "validation_result": DataValidationResult(status="valid"),
    })
    assert soft_stop_blockers(state) == []


def test_partial_trend_evidence_is_retained_but_cannot_validate_enough_evidence_stop():
    state = InvestigationState(target=_target())
    candidate = CandidateAction(
        candidate_id="partial-trend-candidate",
        action={"action_type": "ANALYZE_TREND", "metric_id": "gmv"},
    )
    evidence = AttributionEvidence(
        evidence_id="partial-trend-evidence", evidence_type="trend", metric_id="gmv",
        source_action_id=candidate.candidate_id, source_ref="fixture",
        scope_fingerprint="scope-v1", method_id="observed_trend_v1",
        method_version="1.0", authorized_metric_provenance="exact",
        values={"result": {"status": "partial", "points": [
            {"bucket": "2026-02-01", "value": 10.0},
        ]}},
        data_quality={"status": "partial"},
    )

    assert record_action_result(
        state, candidate, [evidence], current_period={}, baseline_period={},
    )
    result = state.action_results[0]
    assert result.status == "completed"
    assert result.validation_result.status == "incomplete_result"
    assert result.calculation_progress.data_quality_status == "incomplete_result"
    assert soft_stop_blockers(state) == [
        "target_evidence_not_validated",
        f"unresolved_branches:{state.branches[0].branch_id}",
    ]


def test_engine_rejects_premature_enough_evidence_and_executes_open_branch_candidate():
    target = _target()
    branch = UnresolvedBranch(
        branch_id="branch-open", parent_evidence_id="prior-evidence", metric_id="gmv",
        dimension="product_category", dimension_value="furniture", effect_value=-10,
        effect_share_of_root=0.6, depth=1,
    )
    prior_evidence = AttributionEvidence(
        evidence_id="prior-evidence", evidence_type="comparison", metric_id="gmv",
        source_action_id="prior-candidate", source_ref="fixture", scope_fingerprint="scope-v1",
        method_id="compare_period_v1", method_version="1.0",
        authorized_metric_provenance="exact",
    )
    state = InvestigationState(
        target=target, remaining_budget=2, branches=[branch], evidence=[prior_evidence],
        action_results=[ActionResult(
            action_id="prior-action", candidate_id="prior-candidate", status="completed",
            evidence_ids=["prior-evidence"],
            validation_result=DataValidationResult(status="valid"),
        )],
    )
    candidate = CandidateAction(
        candidate_id="candidate-for-open-branch",
        action={"action_type": "COMPARE_PERIOD", "metric_id": "gmv"},
        branch_id=branch.branch_id,
        features=CandidateFeatures(rule_score=0.8),
    )
    executed = []

    def selector(_legal, current_state):
        current_state.stop_reason = StopReason(
            code="ENOUGH_EVIDENCE", detail="model proposed a soft stop", initiator="agent",
        )
        return None

    result = run_investigation(
        state, candidate_factory=lambda _state: [candidate], selector=selector,
        executor=lambda selected, _state: executed.append(selected.candidate_id) or [
            AttributionEvidence(
                evidence_id="follow-up-evidence", evidence_type="comparison", metric_id="gmv",
                source_action_id=selected.candidate_id, source_ref="fixture", scope_fingerprint="scope-v1",
                method_id="compare_period_v1", method_version="1.0",
                authorized_metric_provenance="exact",
            )
        ],
        current_period={}, baseline_period={}, max_actions=1,
    )
    assert executed == [candidate.candidate_id]
    assert result.stop_reason.code != "ENOUGH_EVIDENCE"
    assert result.observations[0] == {
        "event": "soft_stop_rejected",
        "requested_stop": "ENOUGH_EVIDENCE",
        "blockers": ["unresolved_branches:branch-open"],
    }


@pytest.mark.parametrize(
    "stop_code, expected",
    [
        ("ENOUGH_EVIDENCE", "sufficiently_investigated"),
        ("CAPABILITY_GAP", "capability_limited"),
        ("AUTHORIZATION_BLOCKED", "authorization_blocked"),
        ("BUDGET_EXHAUSTED", "budget_exhausted"),
        ("NO_HIGH_VALUE_ACTION", "no_high_value_action"),
        ("DATA_INSUFFICIENT", "data_insufficient"),
        ("INVALID_DATA", "invalid_data"),
        ("EXECUTION_FAILED", "execution_failed"),
        ("USER_SCOPE_LIMIT", "user_scope_limit"),
    ],
)
def test_terminal_progress_exposes_distinct_stop_reasons(stop_code, expected):
    assert terminal_progress_status(stop_code, open_branch_count=1) == expected


def test_engine_executes_only_server_candidate_ids_with_hard_action_budget():
    target = _target()
    state = InvestigationState(target=target, remaining_budget=2)
    candidate = CandidateAction(
        candidate_id="server-candidate-1",
        action={"action_type": "COMPARE_PERIOD", "metric_id": "gmv"},
        features=CandidateFeatures(rule_score=0.9),
    )
    calls = []

    def candidates(_state):
        return [candidate]

    def selector(legal, _state):
        return _workflow_selection(legal[0].candidate_id)

    def executor(selected, _state):
        calls.append(selected.candidate_id)
        return [AttributionEvidence(
            evidence_id=f"e{len(calls)}", evidence_type="comparison", metric_id="gmv",
            source_action_id="server-candidate-1", source_ref="query-ref", scope_fingerprint="scope-v1",
            method_id="compare_period_v1", method_version="1.0",
            authorized_metric_provenance="exact",
        )]

    result = run_investigation(
        state, candidate_factory=candidates, selector=selector, executor=executor,
        current_period={"month": "2026-02"}, baseline_period={"month": "2026-01"},
        max_actions=2,
    )
    assert len(calls) == 1
    assert len(result.evidence) == 1
    assert result.remaining_budget == 1
    assert result.stop_reason.code == "NO_HIGH_VALUE_ACTION"
    action_result = result.action_results[0]
    assert action_result.action_type == "COMPARE_PERIOD"
    assert action_result.metric_id == "gmv"
    assert action_result.action_fingerprint
    assert action_result.residual_value is None
    assert action_result.residual_type == "unknown"
    assert action_result.calculation_progress is not None
    assert action_result.calculation_progress.residual_value is None
    assert action_result.calculation_progress.reconciliation_ratio is None
    legacy_action_result = ActionResult.model_validate({
        "action_id": "legacy-action", "candidate_id": "legacy-candidate",
    })
    assert legacy_action_result.residual_value is None


def test_engine_rejects_selector_output_not_in_candidate_set():
    state = InvestigationState(target=_target())
    candidate = CandidateAction(candidate_id="legal", action={"action_type": "COMPARE_PERIOD", "metric_id": "gmv"})
    executed = []
    result = run_investigation(
        state, candidate_factory=lambda _: [candidate],
        selector=lambda *_: _workflow_selection("invented"),
        executor=lambda *_: executed.append(True), current_period={}, baseline_period={},
    )
    assert not executed
    assert result.stop_reason.code == "NO_HIGH_VALUE_ACTION"


def test_engine_rejects_raw_selector_strings_instead_of_binding_them_to_current_set():
    state = InvestigationState(target=_target())
    candidate = CandidateAction(candidate_id="legal", action={"action_type": "COMPARE_PERIOD", "metric_id": "gmv"})
    executed = []
    result = run_investigation(
        state, candidate_factory=lambda _: [candidate], selector=lambda *_: "legal",
        executor=lambda *_: executed.append(True), current_period={}, baseline_period={},
    )
    assert not executed
    assert result.stop_reason.code == "EXECUTION_FAILED"
    assert result.observations[-1]["event"] == "selector_output_rejected"


def test_shared_candidate_preparation_retains_policy_rejection_reason():
    state = InvestigationState(target=_target(), remaining_budget=2)
    candidate = CandidateAction(
        candidate_id="restricted-candidate",
        action={"action_type": "COMPARE_PERIOD", "metric_id": "gmv"},
    )
    permitted = prepare_candidates(
        state, [candidate], current_period={}, baseline_period={},
        authorization_blocked=lambda _: "metric is not authorized for this scope",
    )
    assert permitted == []
    assert state.stop_reason.code == "AUTHORIZATION_BLOCKED"
    assert state.policy_rejections == [{
        "candidate_id": "restricted-candidate",
        "reason": "metric is not authorized for this scope",
    }]


def test_investigation_limits_registered_association_search_to_one_pair():
    from app.investigation.contracts import AnalyzeAssociationAction, ComparePeriodAction

    state = InvestigationState(target=_target(), remaining_budget=3)
    state.action_results.append(ActionResult(
        action_id="association-1", candidate_id="candidate-1",
        action_type="ANALYZE_ASSOCIATION", status="completed",
    ))
    candidates = [
        CandidateAction(
            candidate_id="association-2",
            action=AnalyzeAssociationAction(
                action_type="ANALYZE_ASSOCIATION", metric_id="gmv", pair_id="pair-2",
            ),
        ),
        CandidateAction(
            candidate_id="period-compare",
            action=ComparePeriodAction(action_type="COMPARE_PERIOD", metric_id="gmv"),
        ),
    ]

    prepared = prepare_candidates(state, candidates, current_period={}, baseline_period={})

    assert [item.candidate_id for item in prepared] == ["period-compare"]
    assert any(item.get("event") == "association_pair_search_limited" for item in state.observations)


def test_authorization_filter_happens_before_top_k_pruning():
    state = InvestigationState(target=_target(), remaining_budget=2)
    candidates = [
        CandidateAction(
            candidate_id=f"candidate-{index}",
            action={"action_type": "COMPARE_PERIOD", "metric_id": f"metric-{index}"},
        )
        for index in range(8)
    ]
    permitted = prepare_candidates(
        state, candidates, current_period={}, baseline_period={},
        authorization_blocked=lambda candidate: (
            False if candidate.action.metric_id in {"metric-6", "metric-7"}
            else "metric is not authorized"
        ),
    )
    assert all(item.action_fingerprint for item in permitted)
    assert {item.candidate_id for item in permitted} == {"candidate-6", "candidate-7"}
    assert len(state.policy_rejections) == 6


def test_shared_candidate_top_k_uses_rank_cost_and_bound_fingerprint_order():
    state = InvestigationState(target=_target(), remaining_budget=2)
    candidates = [
        CandidateAction(
            candidate_id=f"cost-{cost}",
            action={"action_type": "COMPARE_PERIOD", "metric_id": f"metric-{cost}"},
            features=CandidateFeatures(rule_score=0.7, estimated_query_cost=float(cost)),
        )
        for cost in (5, 3, 1, 2, 7, 6, 4)
    ]
    permitted = prepare_candidates(
        state, candidates, current_period={}, baseline_period={},
    )
    expected = sorted(
        [item.model_copy(update={
            "action_fingerprint": action_fingerprint(
                item.action, current_period={}, baseline_period={},
                canonical_scope=state.target.canonical_scope,
                metric_definition_version=state.target.metric_definition_version,
                method_version=state.target.method_version,
            ),
        }) for item in candidates],
        key=lambda item: (
            -item.features.rule_score,
            item.features.estimated_query_cost,
            item.action_fingerprint,
        ),
    )[:6]
    assert [item.candidate_id for item in permitted] == [item.candidate_id for item in expected]


def test_candidate_generator_can_return_ranked_set_before_shared_top_k():
    candidates = generate_candidate_actions(
        _target(), allowed_methods={"additive_change_v1"},
        allowed_dimensions={f"dimension-{index}" for index in range(9)},
        supported_action_types={"BREAKDOWN_DIMENSION"}, top_k=None,
    )
    assert len(candidates) == 9


def test_candidate_set_binds_selector_to_current_state_scope_and_authorization():
    state = InvestigationState(target=_target(), remaining_budget=2)
    candidates = generate_candidate_actions(
        state.target, allowed_methods={"additive_change_v1"},
        allowed_dimensions={"product_category"}, supported_action_types={"BREAKDOWN_DIMENSION"},
    )
    prepared = prepare_candidates(
        state, candidates,
        current_period={"start": "2026-02-01", "end": "2026-02-28"},
        baseline_period={"start": "2026-01-01", "end": "2026-01-31"},
        authorization_fingerprint="authz-v3",
    )
    assert prepared and state.candidate_set is not None
    candidate_set_id = state.candidate_set.candidate_set_id
    assert state.candidate_set.authorization_fingerprint == "authz-v3"
    decision = AgentDecision(
        candidate_set_id=candidate_set_id, decision="select_candidate",
        candidate_id=prepared[0].candidate_id, rationale="继续检查主要维度贡献",
    )
    assert select_candidate(state, decision) == prepared[0]
    assert state.state_version == 1


def test_stale_candidate_set_decision_is_rejected_fail_closed():
    state = InvestigationState(target=_target(), remaining_budget=2)
    candidates = generate_candidate_actions(
        state.target, allowed_methods={"additive_change_v1"},
        allowed_dimensions={"product_category"}, supported_action_types={"BREAKDOWN_DIMENSION"},
    )
    prepared = prepare_candidates(
        state, candidates, current_period={}, baseline_period={},
        authorization_fingerprint="authz-v3",
    )
    assert prepared and state.candidate_set is not None
    decision = AgentDecision(
        candidate_set_id="set-from-previous-round", decision="select_candidate",
        candidate_id=prepared[0].candidate_id,
    )
    assert select_candidate(state, decision) is None
    assert state.stop_reason.code == "NO_HIGH_VALUE_ACTION"


def test_candidate_set_is_invalidated_when_investigation_state_changes():
    state = InvestigationState(target=_target(), remaining_budget=2)
    candidates = generate_candidate_actions(
        state.target, allowed_methods={"additive_change_v1"},
        allowed_dimensions={"product_category"}, supported_action_types={"BREAKDOWN_DIMENSION"},
    )
    prepared = prepare_candidates(
        state, candidates, current_period={}, baseline_period={},
        authorization_fingerprint="authz-v3",
    )
    assert prepared and state.candidate_set is not None
    decision = AgentDecision(
        candidate_set_id=state.candidate_set.candidate_set_id,
        decision="select_candidate", candidate_id=prepared[0].candidate_id,
    )
    state.state_version += 1
    assert select_candidate(state, decision) is None
    assert state.stop_reason.code == "NO_HIGH_VALUE_ACTION"


def test_enough_evidence_soft_stop_cannot_skip_open_investigation_branch():
    state = InvestigationState(target=_target(), remaining_budget=2)
    branch = UnresolvedBranch(
        branch_id="open-branch", parent_evidence_id="ev-parent", metric_id="gmv",
        effect_value=-50, effect_share_of_root=0.8, depth=1,
    )
    state.branches = [branch]
    candidates = generate_candidate_actions(
        state.target, allowed_methods={"additive_change_v1"}, branches=state.branches,
        focus_branch_id=branch.branch_id, allowed_dimensions={"product_category"},
        supported_action_types={"DRILL_DOWN"},
    )
    prepared = prepare_candidates(
        state, candidates, current_period={}, baseline_period={},
        authorization_fingerprint="authz-v3",
    )
    assert prepared and state.candidate_set is not None
    decision = AgentDecision(
        candidate_set_id=state.candidate_set.candidate_set_id,
        decision="soft_stop", soft_stop_reason="ENOUGH_EVIDENCE",
        rationale="看起来已经足够",
    )
    selected = select_candidate(state, decision)
    assert selected is not None
    assert selected.branch_id == branch.branch_id
    assert state.stop_reason is None
    assert state.observations[-1]["event"] == "soft_stop_rejected"


def test_no_high_value_soft_stop_is_allowed_when_only_low_ranked_action_remains():
    state = InvestigationState(target=_target(), remaining_budget=2)
    candidate = CandidateAction(
        candidate_id="low-value", action={"action_type": "ANALYZE_TREND", "metric_id": "gmv"},
        features=CandidateFeatures(rule_score=0.2),
    )
    prepared = prepare_candidates(
        state, [candidate], current_period={}, baseline_period={},
        authorization_fingerprint="authz-v3",
    )
    assert prepared and state.candidate_set is not None
    decision = AgentDecision(
        candidate_set_id=state.candidate_set.candidate_set_id,
        decision="soft_stop", soft_stop_reason="NO_HIGH_VALUE_ACTION",
    )
    assert select_candidate(state, decision) is None
    assert state.stop_reason.code == "NO_HIGH_VALUE_ACTION"


def test_agent_soft_stop_contract_cannot_mix_candidate_and_stop_fields():
    with pytest.raises(ValueError, match="soft stop requires a reason"):
        AgentDecision(
            candidate_set_id="set-1", decision="soft_stop", candidate_id="candidate-1",
        )


def test_empty_action_result_spends_budget_and_hard_stops():
    from app.investigation.contracts import UnresolvedBranch

    state = InvestigationState(target=_target(), remaining_budget=2)
    state.branches = [UnresolvedBranch(
        branch_id="branch-1", parent_evidence_id="ev-parent", metric_id="gmv",
        dimension="product_category", dimension_value="furniture", effect_value=-4,
        effect_share_of_root=0.4, depth=1,
    )]
    candidate = CandidateAction(
        candidate_id="no-evidence",
        action={"action_type": "COMPARE_PERIOD", "metric_id": "gmv"},
        branch_id="branch-1",
    )
    result = run_investigation(
        state, candidate_factory=lambda _: [candidate],
        selector=lambda legal, _: _workflow_selection(legal[0].candidate_id),
        executor=lambda *_: [], current_period={}, baseline_period={},
    )
    assert result.remaining_budget == 1
    assert result.stop_reason.code == "DATA_INSUFFICIENT"
    assert result.action_results[0].status == "failed"
    assert result.action_results[0].failure_code == "DATA_INSUFFICIENT"
    assert result.action_results[0].evidence_ids == []
    assert result.branches[0].status == "open"


def test_data_insufficient_candidate_does_not_block_another_legal_candidate():
    from app.investigation.contracts import AnalyzeTrendAction, ComparePeriodAction

    target = _target()
    empty_candidate = CandidateAction(
        candidate_id="period-without-data",
        action=ComparePeriodAction(action_type="COMPARE_PERIOD", metric_id="gmv"),
    )
    trend_candidate = CandidateAction(
        candidate_id="trend-with-data",
        action=AnalyzeTrendAction(action_type="ANALYZE_TREND", metric_id="gmv"),
    )
    state = InvestigationState(target=target, remaining_budget=3)

    def select_first_unexecuted(candidates, _state):
        candidate = next(
            item for item in candidates
            if item.candidate_id == "period-without-data"
        ) if not any(
            result.candidate_id == "period-without-data"
            for result in _state.action_results
        ) else next(item for item in candidates if item.candidate_id == "trend-with-data")
        return _workflow_selection(candidate.candidate_id)

    def execute(candidate, _state):
        if candidate.candidate_id == "period-without-data":
            return []
        return [AttributionEvidence(
            evidence_id="trend-evidence-after-data-gap",
            evidence_type="trend", metric_id="gmv",
            values={"result": {"status": "ok", "points": [
                {"bucket": "2026-02-01", "value": 12.5},
            ]}},
            source_action_id=candidate.candidate_id,
            source_ref="query_metric_trend",
            scope_fingerprint=target.scope_fingerprint,
            method_id="observed_trend_v1", method_version="1.0",
            authorized_metric_provenance="exact",
            data_quality={"status": "ok"},
        )]

    result = run_investigation(
        state,
        candidate_factory=lambda _state: [empty_candidate, trend_candidate],
        selector=select_first_unexecuted,
        executor=execute,
        current_period={}, baseline_period={},
    )

    assert [item.candidate_id for item in result.action_results] == [
        "period-without-data", "trend-with-data",
    ]
    assert [item.failure_code for item in result.action_results] == [
        "DATA_INSUFFICIENT", None,
    ]
    assert [item.evidence_id for item in result.evidence] == [
        "trend-evidence-after-data-gap",
    ]
    assert result.remaining_budget == 1
    assert any(
        item.get("event") == "candidate_data_insufficient"
        and item.get("continuing_with_alternatives") is True
        for item in result.observations
    )
    assert result.stop_reason.code == "NO_HIGH_VALUE_ACTION"


def test_empty_action_with_missing_failure_code_defaults_to_data_insufficient():
    state = InvestigationState(target=_target(), remaining_budget=1)
    candidate = CandidateAction(
        candidate_id="empty-without-code",
        action={"action_type": "COMPARE_PERIOD", "metric_id": "gmv"},
    )

    accepted = record_action_result(
        state, candidate, [], current_period={}, baseline_period={},
        failure_code=None, failure_detail="",
    )

    assert not accepted
    assert state.stop_reason is not None
    assert state.stop_reason.code == "DATA_INSUFFICIENT"
    assert state.action_results[0].failure_code == "DATA_INSUFFICIENT"
    assert state.action_results[0].failure_detail


def test_executor_exception_is_recorded_as_execution_failure():
    state = InvestigationState(target=_target(), remaining_budget=2)
    candidate = CandidateAction(
        candidate_id="exploding-query",
        action={"action_type": "COMPARE_PERIOD", "metric_id": "gmv"},
    )

    def fail_execution(*_args):
        raise RuntimeError("backend unavailable")

    result = run_investigation(
        state, candidate_factory=lambda _: [candidate],
        selector=lambda legal, _: _workflow_selection(legal[0].candidate_id),
        executor=fail_execution, current_period={}, baseline_period={},
    )
    assert result.stop_reason.code == "EXECUTION_FAILED"
    assert result.action_results[0].status == "failed"
    assert result.action_results[0].failure_code == "EXECUTION_FAILED"
    assert result.remaining_budget == 1


def test_engine_rejects_evidence_for_another_metric():
    state = InvestigationState(target=_target(), remaining_budget=2)
    candidate = CandidateAction(
        candidate_id="gmv-comparison",
        action={"action_type": "COMPARE_PERIOD", "metric_id": "gmv"},
    )
    evidence = AttributionEvidence(
        evidence_id="wrong-metric", evidence_type="comparison", metric_id="order_count",
        source_action_id="gmv-comparison", source_ref="query-ref", scope_fingerprint="scope-v1",
        method_id="compare_period_v1", method_version="1.0",
        authorized_metric_provenance="exact",
    )
    result = run_investigation(
        state, candidate_factory=lambda _: [candidate],
        selector=lambda legal, _: _workflow_selection(legal[0].candidate_id),
        executor=lambda *_: [evidence], current_period={}, baseline_period={}, max_actions=1,
    )
    assert result.evidence == []
    assert result.remaining_budget == 1
    assert result.stop_reason.code == "INVALID_DATA"
    assert result.action_results[0].status == "failed"
    assert result.action_results[0].failure_code == "INVALID_DATA"


def test_engine_rejects_evidence_with_wrong_source_action_id():
    state = InvestigationState(target=_target(), remaining_budget=1)
    candidate = CandidateAction(
        candidate_id="authoritative-candidate",
        action={"action_type": "COMPARE_PERIOD", "metric_id": "gmv"},
    )
    evidence = AttributionEvidence(
        evidence_id="wrong-source", evidence_type="comparison", metric_id="gmv",
        source_action_id="unrelated-action", source_ref="query-ref", scope_fingerprint="scope-v1",
        method_id="compare_period_v1", method_version="1.0",
        authorized_metric_provenance="exact",
    )
    result = run_investigation(
        state, candidate_factory=lambda _: [candidate],
        selector=lambda legal, _: _workflow_selection(legal[0].candidate_id),
        executor=lambda *_: [evidence], current_period={}, baseline_period={}, max_actions=1,
    )
    assert result.evidence == []
    assert result.stop_reason.code == "INVALID_DATA"
    assert "来源动作" in result.action_results[0].failure_detail


def test_engine_derives_child_branches_and_result_features_only_from_typed_evidence():
    from app.investigation.contracts import BreakdownDimensionAction

    state = InvestigationState(target=_target(), remaining_budget=1)
    candidate = CandidateAction(
        candidate_id="category-breakdown",
        action=BreakdownDimensionAction(
            action_type="BREAKDOWN_DIMENSION", metric_id="gmv",
            dimension="product_category", method_id="additive_change_v1",
        ),
    )
    evidence = AttributionEvidence(
        evidence_id="ev-category", evidence_type="dimension_contribution", metric_id="gmv",
        values={
            "total_delta": -100,
            "components": [
                {"dimension_value": "furniture", "delta_value": -70},
                {"dimension_value": "other", "delta_value": -30},
            ],
            "residual_value": 0,
            "residual_type": "unknown",
            "residual_explainable": True,
            "data_quality": {"status": "ok"},
        },
        source_action_id="category-breakdown", source_ref="mcp-result",
        scope_fingerprint="scope-v1", method_id="additive_change_v1",
        method_version="1.0", authorized_metric_provenance="exact",
    )
    result = run_investigation(
        state, candidate_factory=lambda _: [candidate],
        selector=lambda legal, _: _workflow_selection(legal[0].candidate_id),
        executor=lambda *_: [evidence], current_period={}, baseline_period={},
        max_actions=1,
    )
    furniture = next(branch for branch in result.branches if branch.dimension_value == "furniture")
    other = next(branch for branch in result.branches if branch.dimension_value == "Other")
    root = next(branch for branch in result.branches if branch.branch_type == "root_target")
    assert len(result.branches) == 3
    assert furniture.parent_branch_id == root.branch_id
    assert furniture.parent_evidence_id == "ev-category"
    assert furniture.root_attributed_effect_value == pytest.approx(-70)
    assert furniture.root_effect_share == pytest.approx(0.7)
    assert furniture.effect_value == -70
    assert furniture.status == "open"
    assert furniture.explanation_level == "statistical_contribution"
    assert other.status == "blocked"
    assert other.parent_branch_id == root.branch_id
    assert other.effect_value == -30
    assert result.action_results[0].concentration == pytest.approx(0.7)
    assert result.action_results[0].residual_type == "unknown"
    assert result.action_results[0].residual_explainable is True
    assert result.action_results[0].data_quality == {"status": "ok"}
    assert any(item.startswith("unresolved_branches:") for item in soft_stop_blockers(result))


def test_branch_update_marks_only_matching_parent_explained_and_uses_its_depth():
    from app.investigation.contracts import BreakdownDimensionAction

    furniture = UnresolvedBranch(
        branch_id="furniture", parent_evidence_id="ev-category", metric_id="gmv",
        dimension="product_category", dimension_value="furniture", effect_value=-70,
        effect_share_of_root=0.7, depth=1, inherited_filters={"product_category": "furniture"},
    )
    electronics = UnresolvedBranch(
        branch_id="electronics", parent_evidence_id="ev-category", metric_id="gmv",
        dimension="product_category", dimension_value="electronics", effect_value=-20,
        effect_share_of_root=0.2, depth=1, inherited_filters={"product_category": "electronics"},
    )
    state = InvestigationState(target=_target(), branches=[furniture, electronics], remaining_budget=2)
    candidate = CandidateAction(
        candidate_id="furniture-drill",
        action=BreakdownDimensionAction(
            action_type="DRILL_DOWN", metric_id="gmv", dimension="seller_state",
            parent_evidence_id="ev-category", inherited_filters={"product_category": "furniture"},
            method_id="additive_change_v1",
        ),
    )
    evidence = AttributionEvidence(
        evidence_id="ev-furniture", evidence_type="dimension_contribution", metric_id="gmv",
        values={
            "total_delta": -70,
            "components": [{"dimension_value": "SP", "delta_value": -70}],
            "residual_value": 0,
            "residual_type": "unknown",
            "residual_explainable": True,
            "attribution_status": "ok",
        },
        source_action_id="furniture-drill", source_ref="mcp-result", scope_fingerprint="scope-v1",
        method_id="additive_change_v1", method_version="1.0", authorized_metric_provenance="exact",
    )
    accepted = record_action_result(
        state, candidate, [evidence], current_period={}, baseline_period={},
    )
    assert accepted
    branches = {branch.branch_id: branch for branch in state.branches}
    assert branches["furniture"].status == "explained"
    assert branches["electronics"].status == "open"
    child = next(branch for branch in state.branches if branch.branch_id.startswith("branch_ev-furniture"))
    assert child.depth == 2
    assert child.inherited_filters == {"product_category": "furniture", "seller_state": "SP"}


def test_coverage_residual_keeps_parent_branch_open():
    from app.investigation.contracts import BreakdownDimensionAction

    parent = UnresolvedBranch(
        branch_id="furniture", parent_evidence_id="ev-category", metric_id="gmv",
        dimension="product_category", dimension_value="furniture", effect_value=-70,
        effect_share_of_root=0.7, depth=1,
        inherited_filters={"product_category": "furniture"},
    )
    state = InvestigationState(target=_target(), branches=[parent], remaining_budget=2)
    candidate = CandidateAction(
        candidate_id="furniture-partial-drill",
        action=BreakdownDimensionAction(
            action_type="DRILL_DOWN", metric_id="gmv", dimension="seller_state",
            parent_evidence_id="ev-category", inherited_filters={"product_category": "furniture"},
            method_id="additive_change_v1",
        ),
    )
    evidence = AttributionEvidence(
        evidence_id="ev-furniture-partial", evidence_type="dimension_contribution", metric_id="gmv",
        values={
            "total_delta": -70,
            "components": [{"dimension_value": "SP", "delta_value": -50}],
            "residual_value": -20,
            "residual_type": "coverage",
            "residual_explainable": False,
            "attribution_status": "ok",
        },
        source_action_id="furniture-partial-drill", source_ref="mcp-result",
        scope_fingerprint="scope-v1", method_id="additive_change_v1",
        method_version="1.0", authorized_metric_provenance="exact",
    )

    accepted = record_action_result(
        state, candidate, [evidence], current_period={}, baseline_period={},
    )

    assert accepted  # partial, reconciled evidence remains useful
    branches = {branch.branch_id: branch for branch in state.branches}
    assert branches["furniture"].status == "open"
    assert branches["furniture"].effect_value == -70
    assert branches["branch_ev-furniture-partial_seller_state_SP"].effect_value == -50
    residual = branches["branch_ev-furniture-partial_seller_state_coverage_residual"]
    assert residual.status == "blocked"
    assert residual.effect_value == -20
    assert state.action_results[0].residual_type == "coverage"


def test_breakdown_evidence_rejects_unreconciled_components():
    from app.investigation.contracts import BreakdownDimensionAction

    state = InvestigationState(target=_target(), remaining_budget=1)
    candidate = CandidateAction(
        candidate_id="bad-breakdown",
        action=BreakdownDimensionAction(
            action_type="BREAKDOWN_DIMENSION", metric_id="gmv", dimension="product_category",
            method_id="additive_change_v1",
        ),
    )
    evidence = AttributionEvidence(
        evidence_id="ev-bad-breakdown", evidence_type="dimension_contribution", metric_id="gmv",
        values={
            "total_delta": -100,
            "components": [{"dimension_value": "furniture", "delta_value": -70}],
            "residual_value": 0,
            "residual_type": "unknown",
            "residual_explainable": True,
            "attribution_status": "ok",
        },
        source_action_id="bad-breakdown", source_ref="mcp-result",
        scope_fingerprint="scope-v1", method_id="additive_change_v1",
        method_version="1.0", authorized_metric_provenance="exact",
    )

    accepted = record_action_result(
        state, candidate, [evidence], current_period={}, baseline_period={},
    )

    assert not accepted
    assert state.evidence == []
    assert state.stop_reason is not None
    assert state.stop_reason.code == "INVALID_DATA"


def test_trend_action_rejects_empty_series_and_accepts_finite_typed_points():
    from app.investigation.contracts import AnalyzeTrendAction

    def evidence(evidence_id: str, points: list[dict]):
        return AttributionEvidence(
            evidence_id=evidence_id, evidence_type="trend", metric_id="gmv",
            values={"result": {"status": "ok", "points": points}},
            source_action_id="trend-candidate", source_ref="query_metric_trend",
            scope_fingerprint="scope-v1", method_id="observed_trend_v1",
            method_version="1.0", authorized_metric_provenance="exact",
            data_quality={"status": "ok"},
        )

    action = AnalyzeTrendAction(action_type="ANALYZE_TREND", metric_id="gmv")
    candidate = CandidateAction(candidate_id="trend-candidate", action=action)
    empty_state = InvestigationState(target=_target(), remaining_budget=1)
    accepted = record_action_result(
        empty_state, candidate, [evidence("empty-trend", [])],
        current_period={}, baseline_period={},
    )
    assert not accepted
    assert empty_state.stop_reason.code == "DATA_INSUFFICIENT"
    assert empty_state.action_results[0].status == "failed"

    valid_state = InvestigationState(target=_target(), remaining_budget=1)
    accepted = record_action_result(
        valid_state, candidate,
        [evidence("valid-trend", [{"bucket": "2026-02-01", "value": 12.5}])],
        current_period={}, baseline_period={},
    )
    assert accepted
    assert valid_state.action_results[0].status == "completed"


def test_trend_action_rejects_duplicate_time_buckets():
    from app.investigation.contracts import AnalyzeTrendAction

    state = InvestigationState(target=_target(), remaining_budget=1)
    candidate = CandidateAction(
        candidate_id="duplicate-trend",
        action=AnalyzeTrendAction(action_type="ANALYZE_TREND", metric_id="gmv"),
    )
    evidence = AttributionEvidence(
        evidence_id="duplicate-buckets", evidence_type="trend", metric_id="gmv",
        values={"result": {"status": "ok", "points": [
            {"bucket": "2026-02-01", "value": 1},
            {"bucket": "2026-02-01", "value": 2},
        ]}},
        source_action_id="duplicate-trend", source_ref="query_metric_trend",
        scope_fingerprint="scope-v1", method_id="observed_trend_v1",
        method_version="1.0", authorized_metric_provenance="exact",
    )
    accepted = record_action_result(
        state, candidate, [evidence], current_period={}, baseline_period={},
    )
    assert not accepted
    assert state.stop_reason.code == "INVALID_DATA"


def test_anomaly_action_rejects_insufficient_sample_as_business_evidence():
    from app.investigation.contracts import CheckAnomalyAction

    state = InvestigationState(target=_target(), remaining_budget=1)
    candidate = CandidateAction(
        candidate_id="anomaly-check",
        action=CheckAnomalyAction(action_type="CHECK_ANOMALY", metric_id="gmv"),
    )
    evidence = AttributionEvidence(
        evidence_id="insufficient-anomaly", evidence_type="time_series_anomaly", metric_id="gmv",
        values={"series_analysis": {
            "status": "insufficient_data", "numeric_bucket_count": 4, "minimum_required": 8,
        }},
        source_action_id="anomaly-check", source_ref="query_metric_trend",
        scope_fingerprint="scope-v1", method_id="rolling_series_anomaly_v1",
        method_version="1.0", authorized_metric_provenance="exact",
    )
    accepted = record_action_result(
        state, candidate, [evidence], current_period={}, baseline_period={},
    )
    assert not accepted
    assert state.stop_reason.code == "DATA_INSUFFICIENT"
    assert state.evidence == []


def test_data_quality_evidence_is_recomputed_from_raw_series_before_acceptance():
    from app.investigation.contracts import CheckDataQualityAction
    from app.investigation.series_analysis import check_series_quality

    action = CheckDataQualityAction(
        action_type="CHECK_DATA_QUALITY", metric_id="gmv", check_id="zero_value_buckets",
    )
    candidate = CandidateAction(candidate_id="quality-check", action=action)
    raw_result = {"status": "ok", "points": [
        {"bucket": "2026-02-01", "value": 0},
        {"bucket": "2026-02-02", "value": 4},
    ]}
    analysis = check_series_quality(raw_result, action.check_id)
    evidence = AttributionEvidence(
        evidence_id="quality-evidence", evidence_type="time_series_quality", metric_id="gmv",
        values={"result": raw_result, "series_analysis": analysis},
        source_action_id=candidate.candidate_id, source_ref="query_metric_trend",
        scope_fingerprint="scope-v1", method_id="time_series_quality_check_v1",
        method_version="1.0", authorized_metric_provenance="exact",
    )
    state = InvestigationState(target=_target(), remaining_budget=1)

    assert record_action_result(state, candidate, [evidence], current_period={}, baseline_period={})
    assert state.action_results[0].status == "completed"


def test_anomaly_evidence_rejects_duplicate_bucket_input():
    from app.investigation.contracts import CheckAnomalyAction
    from app.investigation.series_analysis import detect_series_anomalies_for_payload

    action = CheckAnomalyAction(action_type="CHECK_ANOMALY", metric_id="gmv")
    candidate = CandidateAction(candidate_id="anomaly-duplicate", action=action)
    payload = {"status": "ok", "points": [
        {"bucket": "2026-02-01", "value": 1},
        {"bucket": "2026-02-01", "value": 2},
        *[
            {"bucket": (date(2026, 2, 2) + timedelta(days=index)).isoformat(), "value": 3}
            for index in range(7)
        ],
    ]}
    analysis = detect_series_anomalies_for_payload(
        payload, metric_id="gmv", unit="BRL", start_date="2026-02-01",
        end_date="2026-02-09", granularity="day", definition_version="1.0",
        source_ref="fixture",
    )
    evidence = AttributionEvidence(
        evidence_id="anomaly-duplicate-evidence", evidence_type="time_series_anomaly", metric_id="gmv",
        values={"series_analysis": analysis}, source_action_id=candidate.candidate_id,
        source_ref="query_metric_trend", scope_fingerprint="scope-v1",
        method_id="rolling_series_anomaly_v1", method_version="1.0",
        authorized_metric_provenance="exact",
    )
    state = InvestigationState(target=_target(), remaining_budget=1)

    assert not record_action_result(state, candidate, [evidence], current_period={}, baseline_period={})
    assert state.stop_reason.code == "INVALID_DATA"
    assert state.evidence == []


@pytest.mark.parametrize("tamper", ["count", "rate", "check_id"])
def test_data_quality_evidence_rejects_fabricated_or_mismatched_summary(tamper):
    from app.investigation.contracts import CheckDataQualityAction
    from app.investigation.series_analysis import check_series_quality

    action = CheckDataQualityAction(
        action_type="CHECK_DATA_QUALITY", metric_id="gmv", check_id="zero_value_buckets",
    )
    candidate = CandidateAction(candidate_id="quality-check-tamper", action=action)
    raw_result = {"status": "ok", "points": [
        {"bucket": "2026-02-01", "value": 0},
        {"bucket": "2026-02-02", "value": 4},
    ]}
    analysis = check_series_quality(raw_result, action.check_id)
    if tamper == "count":
        analysis["bucket_count"] = 99
    elif tamper == "rate":
        analysis["affected_bucket_rate"] = 0.9
    else:
        analysis["check_id"] = "negative_value_buckets"
    evidence = AttributionEvidence(
        evidence_id=f"quality-evidence-{tamper}", evidence_type="time_series_quality", metric_id="gmv",
        values={"result": raw_result, "series_analysis": analysis},
        source_action_id=candidate.candidate_id, source_ref="query_metric_trend",
        scope_fingerprint="scope-v1", method_id="time_series_quality_check_v1",
        method_version="1.0", authorized_metric_provenance="exact",
    )
    state = InvestigationState(target=_target(), remaining_budget=1)

    assert not record_action_result(state, candidate, [evidence], current_period={}, baseline_period={})
    assert state.stop_reason.code == "INVALID_DATA"
    assert state.evidence == []


def test_nested_branch_share_remains_relative_to_root_delta():
    from app.investigation.contracts import BreakdownDimensionAction

    parent = UnresolvedBranch(
        branch_id="parent", parent_evidence_id="ev-root", metric_id="gmv",
        dimension="product_category", dimension_value="furniture", effect_value=-70,
        effect_share_of_root=0.7, depth=1,
        inherited_filters={"product_category": "furniture"},
    )
    state = InvestigationState(
        target=_target(), branches=[parent], root_effect_value=-100, remaining_budget=1,
    )
    candidate = CandidateAction(
        candidate_id="nested-breakdown",
        action=BreakdownDimensionAction(
            action_type="DRILL_DOWN", metric_id="gmv", dimension="seller_state",
            parent_evidence_id="ev-root", inherited_filters={"product_category": "furniture"},
            method_id="additive_change_v1",
        ),
        branch_id="parent",
    )
    evidence = AttributionEvidence(
        evidence_id="ev-nested", evidence_type="dimension_contribution", metric_id="gmv",
        values={
            "total_delta": -70,
            "components": [
                {"dimension_value": "SP", "delta_value": -56},
                {"dimension_value": "RJ", "delta_value": -14},
            ],
            "residual_value": 0,
            "residual_type": "unknown",
            "residual_explainable": True,
            "attribution_status": "ok",
        },
        source_action_id="nested-breakdown", source_ref="mcp-result",
        scope_fingerprint="scope-v1", method_id="additive_change_v1",
        method_version="1.0", authorized_metric_provenance="exact",
    )
    assert record_action_result(
        state, candidate, [evidence], current_period={}, baseline_period={},
    )
    child = next(branch for branch in state.branches if branch.dimension_value == "SP")
    assert child.effect_share_of_root == pytest.approx(0.56)


def test_registry_attribution_catalog_explicitly_declares_support_or_limitation():
    from app.metrics.registry import MetricRegistry

    registry = MetricRegistry()
    registry.load()
    definitions = registry.list()
    assert len(definitions) >= 40
    assert all(item.attribution.limitation for item in definitions)
    assert registry.get("gmv").attribution.for_scope("platform").method_ids == (
        "additive_change_v1", "product_shapley_v1",
    )
    product_factors = registry.get("gmv").attribution.for_scope("platform").factor_specs[
        "product_shapley_v1"
    ]
    assert [(item.factor_id, item.kind, item.source_metric_ids) for item in product_factors] == [
        ("order_count", "sum", ("order_count",)),
        ("avg_order_value", "ratio_of_sums", ("gmv", "order_count")),
    ]
    assert not registry.get("customer_count").attribution.for_scope("platform").method_ids


def _cross_check_evidence_pair(*, trend_current=600.0, contribution_current=600.0,
                               contribution_delta=-400.0, trend_scope="scope-v1",
                               components=None, residual=0.0, residual_type="rounding"):
    from app.investigation.contracts import AnalyzeTrendAction, BreakdownDimensionAction

    trend_action = AnalyzeTrendAction(action_type="ANALYZE_TREND", metric_id="gmv")
    trend_points = []
    for start, end, total, last_value in (
        (date(2026, 1, 1), date(2026, 1, 31), 1000.0, 10.0),
        (date(2026, 2, 1), date(2026, 2, 28), trend_current, trend_current / 10),
    ):
        day_count = (end - start).days + 1
        regular_value = (total - last_value) / (day_count - 1)
        for day_offset in range(day_count):
            bucket = start + timedelta(days=day_offset)
            trend_points.append({
                "bucket": bucket.isoformat(),
                "value": last_value if bucket == end else regular_value,
            })
    trend = AttributionEvidence(
        evidence_id="ev-trend-reconcile", evidence_type="trend", metric_id="gmv",
        values={"result": {"status": "ok", "points": trend_points}},
        source_action_id="trend-reconcile", source_ref="query_metric_trend",
        scope_fingerprint=trend_scope, method_id="observed_trend_v1", method_version="1.0",
        authorized_metric_provenance="exact",
        comparison={"period": {
            "start_date": "2026-01-01", "end_date": "2026-02-28", "granularity": "day",
        }},
    )
    contribution_action = BreakdownDimensionAction(
        action_type="BREAKDOWN_DIMENSION", metric_id="gmv", dimension="product_category",
        method_id="additive_change_v1",
    )
    contribution = AttributionEvidence(
        evidence_id="ev-dimension-reconcile", evidence_type="dimension_contribution", metric_id="gmv",
        values={
            "baseline_value": 1000.0, "current_value": contribution_current,
            "total_delta": contribution_delta,
            "components": components if components is not None else [
                {"dimension_value": "electronics", "delta_value": contribution_delta}
            ],
            "residual_value": residual, "residual_type": residual_type,
            "residual_explainable": residual_type != "coverage",
            "attribution_status": "ok",
        },
        source_action_id="breakdown-reconcile", source_ref="compare_breakdown_by_dimension",
        scope_fingerprint="scope-v1", method_id="additive_change_v1", method_version="1.0",
        authorized_metric_provenance="exact",
        comparison={"comparison": {
            "current_start": "2026-02-01", "current_end": "2026-02-28",
            "baseline_start": "2026-01-01", "baseline_end": "2026-01-31",
            "filters": {},
        }},
    )
    return trend_action, trend, contribution_action, contribution


def test_cross_evidence_reconciliation_accepts_matching_trend_and_full_breakdown():
    trend_action, trend, contribution_action, contribution = _cross_check_evidence_pair()
    trend_by_date = {point["bucket"]: point["value"] for point in trend.values["result"]["points"]}
    assert sum(point["value"] for point in trend.values["result"]["points"] if point["bucket"].startswith("2026-01")) == pytest.approx(1000.0)
    assert sum(point["value"] for point in trend.values["result"]["points"] if point["bucket"].startswith("2026-02")) == pytest.approx(600.0)
    assert trend_by_date["2026-01-31"] != 1000.0
    assert trend_by_date["2026-02-28"] != 600.0
    state = InvestigationState(target=_target(), remaining_budget=2)
    assert record_action_result(
        state, CandidateAction(candidate_id="trend-reconcile", action=trend_action), [trend],
        current_period={}, baseline_period={},
    )
    assert record_action_result(
        state, CandidateAction(candidate_id="breakdown-reconcile", action=contribution_action), [contribution],
        current_period={}, baseline_period={},
    )
    assert state.stop_reason is None
    assert len(state.evidence) == 2
    assert state.action_results[-1].validation_result.status == "valid"


def test_cross_evidence_period_aggregation_conflict_preserves_evidence_and_stops():
    trend_action, trend, contribution_action, contribution = _cross_check_evidence_pair(
        trend_current=600.0, contribution_current=650.0, contribution_delta=-350.0,
    )
    state = InvestigationState(target=_target(), remaining_budget=2)
    assert record_action_result(
        state, CandidateAction(candidate_id="trend-reconcile", action=trend_action), [trend],
        current_period={}, baseline_period={},
    )

    accepted = record_action_result(
        state, CandidateAction(candidate_id="breakdown-reconcile", action=contribution_action), [contribution],
        current_period={}, baseline_period={},
    )

    assert not accepted
    assert {item.evidence_id for item in state.evidence} == {trend.evidence_id, contribution.evidence_id}
    assert state.stop_reason.code == "INVALID_DATA"
    result = state.action_results[-1]
    assert result.failure_code == "INVALID_DATA"
    check = next(item for item in result.validation_result.checks if item["name"] == "cross_evidence_reconciliation")
    assert check["status"] == "failed"
    assert set(check["conflicting_evidence_ids"]) == {trend.evidence_id, contribution.evidence_id}
    assert not any(branch.parent_evidence_id == contribution.evidence_id for branch in state.branches)


def test_cross_evidence_reconciliation_keeps_single_day_point_comparison():
    from app.investigation.contracts import AnalyzeTrendAction

    target = _target().model_copy(update={
        "baseline_start": date(2026, 1, 31), "baseline_end": date(2026, 1, 31),
        "current_start": date(2026, 2, 28), "current_end": date(2026, 2, 28),
    })
    trend_action = AnalyzeTrendAction(action_type="ANALYZE_TREND", metric_id="gmv")
    trend = AttributionEvidence(
        evidence_id="ev-single-day-trend", evidence_type="trend", metric_id="gmv",
        values={"result": {"status": "ok", "points": [
            {"bucket": "2026-01-31", "value": 1000.0},
            {"bucket": "2026-02-28", "value": 600.0},
        ]}},
        source_action_id="single-day-trend", source_ref="query_metric_trend",
        scope_fingerprint="scope-v1", method_id="observed_trend_v1", method_version="1.0",
        authorized_metric_provenance="exact",
        comparison={"period": {
            "start_date": "2026-01-31", "end_date": "2026-02-28", "granularity": "day",
        }},
    )
    _old_trend_action, _old_trend, breakdown_action, contribution = _cross_check_evidence_pair()
    contribution = contribution.model_copy(update={
        "evidence_id": "ev-single-day-breakdown",
        "source_action_id": "single-day-breakdown",
        "comparison": {"comparison": {
            "current_date": "2026-02-28", "baseline_date": "2026-01-31", "filters": {},
        }},
    })
    state = InvestigationState(target=target, remaining_budget=2)
    assert record_action_result(
        state, CandidateAction(candidate_id="single-day-trend", action=trend_action), [trend],
        current_period={}, baseline_period={},
    )
    assert record_action_result(
        state, CandidateAction(candidate_id="single-day-breakdown", action=breakdown_action), [contribution],
        current_period={}, baseline_period={},
    )


def test_cross_evidence_ratio_trend_endpoints_are_not_treated_as_period_totals():
    from app.investigation.engine import _cross_evidence_reconciliation_failure

    _trend_action, trend, _breakdown_action, contribution = _cross_check_evidence_pair(
        trend_current=600.0, contribution_current=600.0, contribution_delta=-400.0,
    )
    trend = trend.model_copy(update={
        "metric_id": "freight_rate",
        "values": {"result": {"status": "ok", "points": [
            {"bucket": "2026-01-31", "value": 8.0},
            {"bucket": "2026-02-28", "value": 6.0},
        ]}},
    })
    contribution = contribution.model_copy(update={
        "metric_id": "freight_rate", "evidence_type": "grouped_ratio_attribution",
        "method_id": "symmetric_mix_within_v1",
        "values": {**contribution.values, "baseline_value": 10.0, "current_value": 8.0, "total_delta": -2.0},
    })
    state = InvestigationState(target=_target().model_copy(update={
        "metric_id": "freight_rate", "canonical_scope": {"tenant_id": "t1"},
    }), evidence=[trend])
    # Daily ratios cannot establish the monthly weighted ratio without the
    # registered period numerator/denominator aggregation.
    assert _cross_evidence_reconciliation_failure(state, [contribution]) is None


def test_cross_evidence_weighted_ratio_uses_registered_period_numerator_denominator():
    from app.investigation.engine import _cross_evidence_reconciliation_failure
    from app.metrics.registry import get_registry

    aggregation = get_registry().get("freight_rate").period_aggregation
    assert aggregation is not None

    def grouped_ratio(evidence_id: str, current_ratio: float) -> AttributionEvidence:
        baseline_ratio = 10.0
        multiplier = aggregation.multiplier
        baseline_groups = {"all": {"numerator": baseline_ratio * 10 / multiplier, "denominator": 10.0}}
        current_groups = {"all": {"numerator": current_ratio * 10 / multiplier, "denominator": 10.0}}
        _trend_action, _trend, _breakdown_action, evidence = _cross_check_evidence_pair()
        return evidence.model_copy(update={
            "evidence_id": evidence_id,
            "evidence_type": "grouped_ratio_attribution",
            "metric_id": "freight_rate",
            "method_id": "symmetric_mix_within_v1",
            "values": {
                "attribution_status": "ok", "baseline_value": baseline_ratio,
                "current_value": current_ratio, "total_delta": current_ratio - baseline_ratio,
                "baseline_groups": baseline_groups, "current_groups": current_groups,
                "multiplier": multiplier,
            },
        })

    existing = grouped_ratio("ev-ratio-a", 11.0)
    incoming = grouped_ratio("ev-ratio-b", 12.0)
    state = InvestigationState(
        target=_target().model_copy(update={
            "metric_id": "freight_rate", "canonical_scope": {"tenant_id": "t1"},
        }),
        evidence=[existing],
    )
    failure = _cross_evidence_reconciliation_failure(state, [incoming])
    assert failure is not None
    assert set(failure[1]) == {existing.evidence_id, incoming.evidence_id}


def test_cross_evidence_conflict_found_after_breakdown_blocks_prior_conclusion():
    trend_action, trend, contribution_action, contribution = _cross_check_evidence_pair(
        trend_current=600.0, contribution_current=650.0, contribution_delta=-350.0,
    )
    state = InvestigationState(target=_target(), remaining_budget=2)
    assert record_action_result(
        state, CandidateAction(candidate_id="breakdown-reconcile", action=contribution_action), [contribution],
        current_period={}, baseline_period={},
    )
    assert any(branch.status == "explained" for branch in state.branches)

    accepted = record_action_result(
        state, CandidateAction(candidate_id="trend-reconcile", action=trend_action), [trend],
        current_period={}, baseline_period={},
    )

    assert not accepted
    assert all(branch.status == "blocked" for branch in state.branches)


def test_cross_evidence_reconciliation_keeps_partial_coverage_residual_explicit():
    trend_action, trend, contribution_action, contribution = _cross_check_evidence_pair(
        components=[{"dimension_value": "electronics", "delta_value": -100.0}],
        residual=-300.0, residual_type="coverage",
    )
    state = InvestigationState(target=_target(), remaining_budget=2)
    assert record_action_result(
        state, CandidateAction(candidate_id="trend-reconcile", action=trend_action), [trend],
        current_period={}, baseline_period={},
    )
    assert record_action_result(
        state, CandidateAction(candidate_id="breakdown-reconcile", action=contribution_action), [contribution],
        current_period={}, baseline_period={},
    )
    assert state.stop_reason is None
    residual_branch = next(branch for branch in state.branches if branch.dimension_value == "__coverage_residual__")
    assert residual_branch.effect_value == -300.0
    assert residual_branch.status == "blocked"


def test_cross_evidence_reconciliation_does_not_compare_different_scopes():
    from app.investigation.engine import _cross_evidence_reconciliation_failure

    _trend_action, _trend, _contribution_action, contribution = _cross_check_evidence_pair()
    other_scope_trend = _cross_check_evidence_pair(trend_scope="scope-other")[1]
    state = InvestigationState(target=_target(), evidence=[contribution])
    assert _cross_evidence_reconciliation_failure(state, [other_scope_trend]) is None


def test_cross_evidence_reconciliation_does_not_compare_different_filters():
    from app.investigation.engine import _cross_evidence_reconciliation_failure

    _trend_action, trend, _breakdown_action, contribution = _cross_check_evidence_pair()
    contribution = contribution.model_copy(update={
        "comparison": {"comparison": {
            "current_start": "2026-02-01", "current_end": "2026-02-28",
            "baseline_start": "2026-01-01", "baseline_end": "2026-01-31",
            "filters": {"seller_state": "SP"},
        }},
    })
    state = InvestigationState(target=_target(), evidence=[contribution])
    assert _cross_evidence_reconciliation_failure(state, [trend]) is None


def test_cross_evidence_reconciliation_rejects_mismatched_period_comparison_result():
    from app.investigation.contracts import ComparePeriodAction

    _trend_action, _trend, _breakdown_action, contribution = _cross_check_evidence_pair()
    comparison = contribution.model_copy(update={
        "evidence_id": "ev-period-comparison",
        "evidence_type": "period_comparison",
        "values": {"baseline_value": 1000.0, "current_value": 700.0, "total_delta": -300.0},
        "source_action_id": "period-comparison-action",
        "source_ref": "compare_metric_periods",
        "method_id": "compare_period_v1",
    })
    state = InvestigationState(target=_target(), remaining_budget=2)
    assert record_action_result(
        state,
        CandidateAction(candidate_id="period-comparison-action", action=ComparePeriodAction(
            action_type="COMPARE_PERIOD", metric_id="gmv",
        )),
        [comparison], current_period={}, baseline_period={},
    )
    accepted = record_action_result(
        state,
        CandidateAction(candidate_id="breakdown-reconcile", action=_breakdown_action),
        [contribution], current_period={}, baseline_period={},
    )
    assert not accepted
    assert {item.evidence_id for item in state.evidence} == {
        comparison.evidence_id, contribution.evidence_id,
    }
    assert state.stop_reason.code == "INVALID_DATA"


def test_cross_evidence_ratio_uses_registered_aggregate_and_never_sums_group_ratios():
    from app.investigation.engine import _cross_evidence_reconciliation_failure
    from app.metrics.registry import get_registry

    definition = get_registry().get("freight_rate")
    assert definition.aggregation == "weighted_ratio"
    assert definition.period_aggregation is not None
    trend_action, trend, _breakdown_action, contribution = _cross_check_evidence_pair(
        trend_current=120.0, contribution_current=120.0, contribution_delta=20.0,
        components=[
            {"dimension_value": "group-a", "delta_value": 7.0},
            {"dimension_value": "group-b", "delta_value": 13.0},
        ],
    )
    trend = trend.model_copy(update={
        "metric_id": "freight_rate",
        "values": {"result": {"status": "ok", "points": [
            {"bucket": "2026-01-31", "value": 100.0},
            {"bucket": "2026-02-28", "value": 120.0},
        ]}},
    })
    contribution = contribution.model_copy(update={
        "metric_id": "freight_rate", "evidence_type": "grouped_ratio_attribution",
        "method_id": "symmetric_mix_within_v1",
        "values": {
            **contribution.values,
            "baseline_value": 100.0,
            # Grouped ratio values are not additive. Aggregate values and the
            # registered period rule are the only cross-evidence comparison.
            "components": [
                {"dimension_value": "group-a", "delta_value": 35.0},
                {"dimension_value": "group-b", "delta_value": -15.0},
            ],
        },
    })
    target = _target().model_copy(update={"metric_id": "freight_rate"})
    state = InvestigationState(target=target, evidence=[trend])
    assert _cross_evidence_reconciliation_failure(state, [contribution]) is None

"""Deterministic protocol evaluation cases (not a substitute for live LLM evals)."""

from datetime import date

import pytest

from app.investigation.candidates import generate_candidate_actions
from app.investigation.contracts import (
    ActionResult, AgentDecision, AnalysisTarget, AttributionEvidence, DataValidationResult,
    InvestigationState, StopReason, UnresolvedBranch, WorkflowSelection,
)
from app.investigation.engine import run_investigation
from app.investigation.evaluation import (
    ActionMatcher, BranchMatcher, EvidenceMatcher, InvestigationEvaluationCase,
    evaluate_investigation_path, summarize_investigation_evaluations,
)
from app.metrics.registry import get_registry


_REGISTRY = get_registry()
_PLATFORM_METRICS = _REGISTRY.list("platform")
SUPPORTED_BREAKDOWNS = [
    (metric.metric_id, dimension)
    for metric in _PLATFORM_METRICS
    if "additive_change_v1" in metric.attribution.for_scope("platform").method_ids
    for dimension in metric.attribution.for_scope("platform").dimensions
]
PRODUCT_CASES = [
    ("gmv", ("order_count", "avg_order_value")),
]
RATIO_CASES = [
    (metric.metric_id, (
        metric.period_aggregation.numerator_metric,
        metric.period_aggregation.denominator_metric,
    ))
    for metric in _PLATFORM_METRICS
    if "symmetric_ratio_v1" in metric.attribution.for_scope("platform").method_ids
    and metric.period_aggregation is not None
]
UNSUPPORTED_CASES = [
    metric.metric_id for metric in _PLATFORM_METRICS
    if not metric.attribution.for_scope("platform").method_ids
][:14]


def _target(metric_id: str) -> AnalysisTarget:
    return AnalysisTarget(
        metric_id=metric_id, current_start=date(2026, 2, 1), current_end=date(2026, 2, 28),
        baseline_start=date(2026, 1, 1), baseline_end=date(2026, 1, 31),
        canonical_scope={"scope_type": "platform", "tenant_id": "eval-tenant"},
        scope_fingerprint="eval-scope", requested_analysis_type="diagnostic",
    )


@pytest.mark.parametrize("metric_id,dimension", SUPPORTED_BREAKDOWNS)
def test_supported_additive_scenarios_only_emit_the_requested_metric_and_dimension(metric_id, dimension):
    candidates = generate_candidate_actions(
        _target(metric_id), allowed_methods={"additive_change_v1"},
        allowed_dimensions={dimension}, supported_action_types={"BREAKDOWN_DIMENSION"},
    )
    assert len(candidates) == 1
    assert candidates[0].action.metric_id == metric_id
    assert candidates[0].action.dimension == dimension
    assert candidates[0].features.data_availability == "available"


@pytest.mark.parametrize("metric_id,dimension", SUPPORTED_BREAKDOWNS)
def test_supported_additive_investigation_executes_relevant_action_once(metric_id, dimension):
    """Exercise the end-to-end deterministic investigation protocol, not a fixed LLM path."""
    target = _target(metric_id)
    state = InvestigationState(target=target, remaining_budget=3)
    generated = generate_candidate_actions(
        target, allowed_methods={"additive_change_v1"},
        allowed_dimensions={dimension}, supported_action_types={"BREAKDOWN_DIMENSION"},
    )
    calls = []

    def execute(candidate, _state):
        calls.append(candidate.action)
        return [AttributionEvidence(
            evidence_id=f"eval-{metric_id}-{dimension}",
            evidence_type="dimension_contribution", metric_id=metric_id,
            claim="synthetic evaluation evidence", values={
                "total_delta": -10.0,
                "components": [{"component_id": "group-a", "value": -10.0}],
                "residual_value": 0.0,
                "residual_type": "unknown",
                "residual_explainable": True,
                "attribution_status": "ok",
            }, source_action_id=candidate.candidate_id, source_ref="eval-fixture",
            scope_fingerprint=target.scope_fingerprint, method_id="additive_change_v1",
            method_version="1.0", authorized_metric_provenance="evaluation_fixture",
        )]

    result = run_investigation(
        state,
        candidate_factory=lambda _state: generated,
        selector=lambda legal, _state: WorkflowSelection(candidate_id=next(
            item.candidate_id for item in legal
            if item.action.metric_id == metric_id and item.action.dimension == dimension
        )),
        executor=execute,
        current_period={"start": target.current_start.isoformat(), "end": target.current_end.isoformat()},
        baseline_period={"start": target.baseline_start.isoformat(), "end": target.baseline_end.isoformat()},
        max_actions=3,
    )
    assert len(calls) == 1
    assert calls[0].metric_id == metric_id
    assert calls[0].dimension == dimension
    assert len(result.action_results) == 1
    assert result.action_results[0].evidence_ids == [f"eval-{metric_id}-{dimension}"]
    assert result.stop_reason.code == "NO_HIGH_VALUE_ACTION"


@pytest.mark.parametrize("metric_id,factors", PRODUCT_CASES)
def test_product_scenarios_expose_only_registered_factor_decomposition(metric_id, factors):
    assert "product_shapley_v1" in _REGISTRY.get(metric_id).attribution.for_scope("platform").method_ids
    candidates = generate_candidate_actions(
        _target(metric_id), allowed_methods={"product_shapley_v1"},
        allowed_dimensions=(), supported_action_types={"DECOMPOSE_METRIC"},
        factor_metrics_by_method={"product_shapley_v1": factors},
    )
    assert len(candidates) == 1
    action = candidates[0].action
    assert action.action_type == "DECOMPOSE_METRIC"
    assert tuple(action.factor_metric_ids) == factors


@pytest.mark.parametrize("metric_id,factors", RATIO_CASES)
def test_ratio_scenarios_expose_registered_numerator_denominator_decomposition(metric_id, factors):
    assert "symmetric_ratio_v1" in _REGISTRY.get(metric_id).attribution.for_scope("platform").method_ids
    candidates = generate_candidate_actions(
        _target(metric_id), allowed_methods={"symmetric_ratio_v1"},
        allowed_dimensions=(), supported_action_types={"DECOMPOSE_METRIC"},
        factor_metrics_by_method={"symmetric_ratio_v1": factors},
    )
    assert len(candidates) == 1
    assert candidates[0].action.metric_id == metric_id
    assert tuple(candidates[0].action.factor_metric_ids) == factors


@pytest.mark.parametrize("metric_id", UNSUPPORTED_CASES)
def test_unsupported_metric_scenarios_fail_closed_without_attribution_candidates(metric_id):
    candidates = generate_candidate_actions(
        _target(metric_id), allowed_methods=(), allowed_dimensions={"product_category", "seller_state"},
        supported_action_types={"DECOMPOSE_METRIC", "BREAKDOWN_DIMENSION", "DRILL_DOWN"},
        factor_metrics_by_method={"product_shapley_v1": ("factor_a", "factor_b")},
    )
    assert candidates == []


def _path_state(
    action_type: str,
    evidence_type: str,
    action_id: str = "selected-action",
    *,
    metric_id: str = "gmv",
    dimension: str | None = None,
    method_id: str = "additive_change_v1",
) -> InvestigationState:
    target = _target(metric_id)
    evidence = AttributionEvidence(
        evidence_id=f"evidence-{action_type}-{metric_id}", evidence_type=evidence_type, metric_id=metric_id,
        source_action_id=action_id, source_ref="evaluation-fixture",
        scope_fingerprint=target.scope_fingerprint, method_id=method_id,
        method_version="1.0", authorized_metric_provenance="evaluation-fixture",
    )
    state = InvestigationState(target=target, remaining_budget=2, evidence=[evidence])
    state.action_results.append(ActionResult(
        action_id="action-1", candidate_id=action_id, action_type=action_type,
        metric_id=metric_id, dimension=dimension,
        method_id=method_id, action_fingerprint=f"fp-{action_type}-{metric_id}-{dimension}",
        status="completed", evidence_ids=[evidence.evidence_id], decision_source="agent",
        validation_result=DataValidationResult(status="valid"),
    ))
    state.stop_reason = StopReason(
        code="NO_HIGH_VALUE_ACTION", detail="fixture complete", initiator="backend",
    )
    return state


def _runtime_agent_state(
    metric_id: str,
    *,
    action_type: str,
    method_id: str,
    dimension: str | None = None,
    factors: tuple[str, ...] = (),
) -> InvestigationState:
    """Build evaluation evidence by executing the real bounded investigation loop."""
    target = _target(metric_id)
    methods = {method_id}
    dimensions = {dimension} if dimension else set()
    generated = generate_candidate_actions(
        target,
        allowed_methods=methods,
        allowed_dimensions=dimensions,
        supported_action_types={action_type},
        factor_metrics_by_method={method_id: factors} if factors else {},
    )
    assert generated, (metric_id, action_type, method_id, dimension)
    state = InvestigationState(target=target, remaining_budget=3)

    def select(legal, current_state):
        chosen = next((
            item for item in legal
            if item.action.action_type == action_type
            and item.action.metric_id == metric_id
            and getattr(item.action, "dimension", None) == dimension
            and getattr(item.action, "method_id", None) == method_id
        ), None)
        assert chosen is not None
        assert current_state.candidate_set is not None
        return AgentDecision(
            candidate_set_id=current_state.candidate_set.candidate_set_id,
            decision="select_candidate",
            candidate_id=chosen.candidate_id,
            rationale="evaluation harness selected a relevant registered candidate",
        )

    def execute(candidate, _current_state):
        if action_type == "DECOMPOSE_METRIC":
            if method_id == "product_shapley_v1":
                raw_components = [
                    {"component_id": factors[0], "value": 2.0, "component_type": "factor_effect"},
                    {"component_id": factors[1], "value": 3.0, "component_type": "factor_effect"},
                    {"component_id": "interaction", "value": 1.0, "component_type": "interaction_effect"},
                ]
                display_components = [
                    {"component_id": factors[0], "value": 2.5, "component_type": "shapley_factor_effect"},
                    {"component_id": factors[1], "value": 3.5, "component_type": "shapley_factor_effect"},
                ]
                display_method = "shapley"
                interaction_delta = 1.0
            else:
                raw_components = [
                    {"component_id": "numerator_effect", "value": 2.0, "component_type": "factor_effect"},
                    {"component_id": "denominator_effect", "value": 1.0, "component_type": "factor_effect"},
                ]
                display_components = raw_components
                display_method = "raw"
                interaction_delta = 0.0
            evidence_type = "metric_decomposition"
            values = {
                "total_delta": 3.0 if method_id == "symmetric_ratio_v1" else 6.0,
                "factor_metrics": list(factors),
                "raw_components": raw_components,
                "display_components": display_components,
                "display_method": display_method,
                "interaction_delta": interaction_delta,
                "residual_value": 0.0,
                "residual_type": "unknown",
                "residual_explainable": True,
                "residual_tolerance": 0.01,
            }
        else:
            evidence_type = "dimension_contribution"
            values = {
                "attribution_status": "ok",
                "total_delta": -10.0,
                "components": [{"dimension_value": "group-a", "delta_value": -10.0}],
                "residual_value": 0.0,
                "residual_type": "unknown",
                "residual_explainable": True,
                "residual_tolerance": 0.01,
            }
        return [AttributionEvidence(
            evidence_id=f"runtime-eval-{metric_id}-{method_id}-{action_type}-{dimension or 'none'}",
            evidence_type=evidence_type,
            metric_id=metric_id,
            claim="synthetic evidence passed through the runtime investigation ledger",
            values=values,
            source_action_id=candidate.candidate_id,
            source_ref="runtime-evaluation-harness",
            scope_fingerprint=target.scope_fingerprint,
            method_id=method_id,
            method_version="1.0",
            authorized_metric_provenance="evaluation_fixture",
        )]

    return run_investigation(
        state,
        candidate_factory=lambda _state: generated,
        selector=select,
        executor=execute,
        current_period={"start": target.current_start.isoformat(), "end": target.current_end.isoformat()},
        baseline_period={"start": target.baseline_start.isoformat(), "end": target.baseline_end.isoformat()},
        max_actions=3,
    )


def _runtime_capability_gap_state(metric_id: str) -> InvestigationState:
    """Exercise the engine's terminal handling for a server-known capability gap."""
    target = _target(metric_id)
    state = InvestigationState(target=target, remaining_budget=3)

    def candidate_factory(current_state):
        current_state.stop_reason = StopReason(
            code="CAPABILITY_GAP",
            detail="registry declares no attribution method for this metric and scope",
            initiator="backend",
        )
        return []

    return run_investigation(
        state,
        candidate_factory=candidate_factory,
        selector=lambda _legal, _current_state: pytest.fail(
            "a capability gap must stop before asking the Agent to select an action"
        ),
        executor=lambda _candidate, _current_state: pytest.fail(
            "a capability gap must not execute a query"
        ),
        current_period={"start": target.current_start.isoformat(), "end": target.current_end.isoformat()},
        baseline_period={"start": target.baseline_start.isoformat(), "end": target.baseline_end.isoformat()},
        max_actions=3,
    )


def test_path_evaluation_accepts_alternative_actions_without_fixed_sequence():
    case = InvestigationEvaluationCase(
        case_id="gmv-investigation-alternatives", target_metric_id="gmv", max_actions=2,
        required_action_groups=[[
            ActionMatcher(action_type="BREAKDOWN_DIMENSION", metric_id="gmv", dimension="product_category"),
            ActionMatcher(action_type="ANALYZE_TREND", metric_id="gmv"),
        ]],
        required_evidence_groups=[[
            EvidenceMatcher(evidence_type="dimension_contribution", metric_id="gmv"),
            EvidenceMatcher(evidence_type="trend", metric_id="gmv"),
        ]],
        allowed_stop_reasons=["NO_HIGH_VALUE_ACTION", "ENOUGH_EVIDENCE"],
    )

    dimension_path = evaluate_investigation_path(
        case, _path_state("BREAKDOWN_DIMENSION", "dimension_contribution", dimension="product_category"),
    )
    trend_path = evaluate_investigation_path(
        case, _path_state("ANALYZE_TREND", "trend"),
    )

    assert dimension_path.passed
    assert trend_path.passed
    assert dimension_path.action_obligation_coverage == trend_path.action_obligation_coverage == 1.0
    assert dimension_path.evidence_grounding_rate == trend_path.evidence_grounding_rate == 1.0


def test_path_evaluation_rejects_ungrounded_evidence_duplicate_actions_and_budget_overrun():
    case = InvestigationEvaluationCase(
        case_id="gmv-invalid-path", target_metric_id="gmv", max_actions=1,
        required_action_groups=[[
            ActionMatcher(action_type="BREAKDOWN_DIMENSION", metric_id="gmv", dimension="product_category"),
        ]],
        required_evidence_groups=[[EvidenceMatcher(evidence_type="dimension_contribution")]],
        allowed_stop_reasons=["ENOUGH_EVIDENCE"],
        allowed_action_matchers=[
            ActionMatcher(action_type="BREAKDOWN_DIMENSION", metric_id="gmv", dimension="product_category"),
        ],
    )
    state = _path_state("BREAKDOWN_DIMENSION", "dimension_contribution", dimension="product_category")
    state.evidence[0] = state.evidence[0].model_copy(update={"scope_fingerprint": "other-scope"})
    state.action_results.append(state.action_results[0].model_copy(update={"action_id": "action-2"}))

    result = evaluate_investigation_path(case, state)

    assert not result.passed
    assert "unlinked_or_out_of_scope_evidence" in result.issues
    assert "duplicate_action_fingerprint" in result.issues
    assert "action_budget_exceeded" in result.issues
    assert "stop_reason_not_allowed" in result.issues
    assert result.action_relevance_rate == 1.0


def test_path_evaluation_reports_irrelevant_and_denied_action_rates():
    case = InvestigationEvaluationCase(
        case_id="gmv-action-quality", target_metric_id="gmv", max_actions=3,
        allowed_stop_reasons=["NO_HIGH_VALUE_ACTION"],
        allowed_action_matchers=[
            ActionMatcher(action_type="BREAKDOWN_DIMENSION", metric_id="gmv", dimension="product_category"),
        ],
    )
    state = _path_state("BREAKDOWN_DIMENSION", "dimension_contribution", dimension="product_category")
    state.action_results.append(state.action_results[0].model_copy(update={
        "action_id": "action-irrelevant", "candidate_id": "candidate-irrelevant",
        "action_type": "CHECK_ANOMALY", "dimension": None,
        "action_fingerprint": "fingerprint-irrelevant", "evidence_ids": [],
    }))
    state.action_results.append(state.action_results[0].model_copy(update={
        "action_id": "action-denied", "candidate_id": "candidate-denied",
        "status": "denied", "action_fingerprint": "fingerprint-denied", "evidence_ids": [],
    }))

    result = evaluate_investigation_path(case, state)

    assert result.action_relevance_rate == pytest.approx(2 / 3)
    assert result.relevant_agent_action_count == 2
    assert result.agent_action_count == 3
    assert result.attempted_actions == 3
    assert result.invalid_action_count == 1
    assert result.invalid_action_rate == pytest.approx(1 / 3)
    assert result.fallback_action_count == 0
    assert result.fallback_action_rate == 0.0
    assert "irrelevant_agent_action" in result.issues
    assert "invalid_or_denied_action" in result.issues


def test_workflow_fallback_actions_are_reported_but_excluded_from_agent_relevance():
    case = InvestigationEvaluationCase(
        case_id="fallback-not-agent-quality", target_metric_id="gmv", max_actions=2,
        allowed_stop_reasons=["NO_HIGH_VALUE_ACTION"],
        allowed_action_matchers=[
            ActionMatcher(action_type="BREAKDOWN_DIMENSION", metric_id="gmv", dimension="product_category"),
        ],
    )
    state = _path_state("BREAKDOWN_DIMENSION", "dimension_contribution", dimension="product_category")
    state.action_results[0] = state.action_results[0].model_copy(update={"decision_source": "workflow_fallback"})

    result = evaluate_investigation_path(case, state)
    summary = summarize_investigation_evaluations([result])

    assert result.action_relevance_rate is None
    assert result.agent_action_count == 0
    assert result.fallback_action_count == 1
    assert result.fallback_action_rate == 1.0
    assert summary.mean_action_relevance_rate is None
    assert summary.fallback_action_rate == 1.0


def test_path_evaluation_checks_branch_disposition_and_grounded_conclusion_references():
    case = InvestigationEvaluationCase(
        case_id="gmv-branch-completion", target_metric_id="gmv", max_actions=2,
        required_action_groups=[], required_evidence_groups=[],
        required_branch_groups=[[
            BranchMatcher(
                metric_id="order_count", dimension="product_category",
                dimension_value="furniture", acceptable_statuses=["explained", "low_priority"],
            ),
        ]],
        allowed_stop_reasons=["ENOUGH_EVIDENCE"], require_conclusion_evidence=True,
    )
    state = _path_state("BREAKDOWN_DIMENSION", "dimension_contribution")
    state.branches.append(UnresolvedBranch(
        branch_id="branch-1", parent_evidence_id=state.evidence[0].evidence_id,
        metric_id="order_count", dimension="product_category", dimension_value="furniture",
        effect_value=-12, effect_share_of_root=0.6, depth=1, status="low_priority",
    ))
    state.stop_reason = StopReason(code="ENOUGH_EVIDENCE", detail="done", initiator="backend")

    result = evaluate_investigation_path(
        case, state, conclusion_evidence_ids=[state.evidence[0].evidence_id],
    )
    assert result.passed
    assert result.branch_obligation_coverage == 1.0

    state.branches[0] = state.branches[0].model_copy(update={"status": "open"})
    rejected = evaluate_investigation_path(case, state)
    assert not rejected.passed
    assert "required_branch_not_resolved_or_explicitly_deferred" in rejected.issues
    assert "conclusion_evidence_reference_missing_or_ungrounded" in rejected.issues


@pytest.mark.parametrize("validation_status", ["incomplete_result", "unknown"])
def test_conclusion_evidence_must_pass_base_data_validation(validation_status):
    case = InvestigationEvaluationCase(
        case_id=f"unvalidated-conclusion-{validation_status}",
        target_metric_id="gmv", max_actions=1,
        required_action_groups=[], required_evidence_groups=[],
        allowed_stop_reasons=["NO_HIGH_VALUE_ACTION"],
        require_conclusion_evidence=True,
    )
    state = _path_state("ANALYZE_TREND", "trend")
    state.action_results[0] = state.action_results[0].model_copy(update={
        "validation_result": DataValidationResult(status=validation_status),
    })

    result = evaluate_investigation_path(
        case, state, conclusion_evidence_ids=[state.evidence[0].evidence_id],
    )

    assert not result.passed
    assert "conclusion_evidence_missing_or_not_validated" in result.issues


def _evaluation_corpus():
    corpus = []
    for index, (metric_id, dimension) in enumerate(SUPPORTED_BREAKDOWNS):
        corpus.append((
            InvestigationEvaluationCase(
                case_id=f"additive-{index:02d}", target_metric_id=metric_id, max_actions=3,
                required_action_groups=[[
                    ActionMatcher(action_type="BREAKDOWN_DIMENSION", metric_id=metric_id, dimension=dimension),
                    ActionMatcher(action_type="ANALYZE_TREND", metric_id=metric_id),
                ]],
                allowed_action_matchers=[
                    ActionMatcher(action_type="BREAKDOWN_DIMENSION", metric_id=metric_id, dimension=dimension),
                    ActionMatcher(action_type="ANALYZE_TREND", metric_id=metric_id),
                ],
                required_evidence_groups=[[
                    EvidenceMatcher(evidence_type="dimension_contribution", metric_id=metric_id),
                    EvidenceMatcher(evidence_type="trend", metric_id=metric_id),
                ]],
                allowed_stop_reasons=["NO_HIGH_VALUE_ACTION", "ENOUGH_EVIDENCE"],
            ),
            _runtime_agent_state(
                metric_id, action_type="BREAKDOWN_DIMENSION",
                method_id="additive_change_v1", dimension=dimension,
            ),
        ))
    for index, (metric_id, factors) in enumerate(PRODUCT_CASES):
        corpus.append((
            InvestigationEvaluationCase(
                case_id=f"product-{index:02d}", target_metric_id=metric_id, max_actions=3,
                required_action_groups=[[
                    ActionMatcher(action_type="DECOMPOSE_METRIC", metric_id=metric_id,
                                  method_id="product_shapley_v1"),
                ]],
                allowed_action_matchers=[
                    ActionMatcher(action_type="DECOMPOSE_METRIC", metric_id=metric_id,
                                  method_id="product_shapley_v1"),
                ],
                required_evidence_groups=[[
                    EvidenceMatcher(evidence_type="metric_decomposition", metric_id=metric_id,
                                    method_id="product_shapley_v1"),
                ]],
                allowed_stop_reasons=["NO_HIGH_VALUE_ACTION", "ENOUGH_EVIDENCE"],
            ),
            _runtime_agent_state(
                metric_id, action_type="DECOMPOSE_METRIC",
                method_id="product_shapley_v1", factors=factors,
            ),
        ))
    for index, (metric_id, factors) in enumerate(RATIO_CASES):
        corpus.append((
            InvestigationEvaluationCase(
                case_id=f"ratio-{index:02d}", target_metric_id=metric_id, max_actions=3,
                required_action_groups=[[
                    ActionMatcher(action_type="DECOMPOSE_METRIC", metric_id=metric_id,
                                  method_id="symmetric_ratio_v1"),
                ]],
                allowed_action_matchers=[
                    ActionMatcher(action_type="DECOMPOSE_METRIC", metric_id=metric_id,
                                  method_id="symmetric_ratio_v1"),
                ],
                required_evidence_groups=[[
                    EvidenceMatcher(evidence_type="metric_decomposition", metric_id=metric_id,
                                    method_id="symmetric_ratio_v1"),
                ]],
                allowed_stop_reasons=["NO_HIGH_VALUE_ACTION", "ENOUGH_EVIDENCE"],
            ),
            _runtime_agent_state(
                metric_id, action_type="DECOMPOSE_METRIC",
                method_id="symmetric_ratio_v1", factors=factors,
            ),
        ))
    for index, metric_id in enumerate(UNSUPPORTED_CASES):
        state = _runtime_capability_gap_state(metric_id)
        corpus.append((
            InvestigationEvaluationCase(
                case_id=f"unsupported-{index:02d}", target_metric_id=metric_id, max_actions=0,
                allowed_stop_reasons=["CAPABILITY_GAP"], expected_capability_gap=True,
            ),
            state,
        ))
    return corpus


@pytest.mark.parametrize("case,state", _evaluation_corpus(), ids=lambda item: item.case_id if hasattr(item, "case_id") else None)
def test_path_quality_evaluation_corpus_has_30_layered_cases(case, state):
    result = evaluate_investigation_path(case, state)
    assert result.passed, (case.case_id, result.issues)


def test_30_case_evaluation_summary_reports_budget_relevance_and_invalid_rates():
    results = [evaluate_investigation_path(case, state) for case, state in _evaluation_corpus()]
    summary = summarize_investigation_evaluations(results)

    assert summary.case_count == 30
    assert summary.passed_cases == 30
    assert summary.pass_rate == 1.0
    assert summary.invalid_action_rate == 0.0
    assert summary.mean_action_relevance_rate == 1.0
    assert summary.fallback_action_rate == 0.0
    assert summary.duplicate_action_rate == 0.0
    assert summary.budget_exhaustion_rate == 0.0
    assert summary.useful_stop_rate == 1.0
    assert summary.capability_gap_recognition_rate == 1.0
    assert summary.evidence_grounding_rate == 1.0


def test_evaluation_summary_weights_action_relevance_by_agent_action_count():
    from app.investigation.evaluation import InvestigationEvaluationResult

    def result(case_id, relevant, total):
        return InvestigationEvaluationResult(
            case_id=case_id, passed=True, completed_actions=total,
            attempted_actions=total, invalid_action_count=0,
            invalid_action_rate=0.0, action_relevance_rate=relevant / total,
            relevant_agent_action_count=relevant, agent_action_count=total,
            action_obligation_coverage=1.0, evidence_obligation_coverage=1.0,
            branch_obligation_coverage=1.0, evidence_grounding_rate=1.0,
            stop_reason_allowed=True, stop_reason_code="NO_HIGH_VALUE_ACTION",
            useful_stop=True, within_budget=True, duplicate_action_count=0,
        )

    summary = summarize_investigation_evaluations([
        result("one-action-case", 1, 1),
        result("nine-action-case", 1, 9),
    ])

    assert summary.mean_action_relevance_rate == pytest.approx(0.2)
    assert summary.relevant_agent_action_count == 2
    assert summary.agent_action_count == 10


def test_evaluation_summary_weights_evidence_grounding_by_evidence_count():
    from app.investigation.evaluation import InvestigationEvaluationResult

    def result(case_id, grounded, total):
        return InvestigationEvaluationResult(
            case_id=case_id, passed=True, completed_actions=1,
            attempted_actions=1, invalid_action_count=0,
            invalid_action_rate=0.0, action_obligation_coverage=1.0,
            evidence_obligation_coverage=1.0, branch_obligation_coverage=1.0,
            evidence_grounding_rate=(grounded / total if total else 1.0),
            grounded_evidence_count=grounded, evidence_count=total,
            stop_reason_allowed=True, stop_reason_code="NO_HIGH_VALUE_ACTION",
            useful_stop=True, within_budget=True, duplicate_action_count=0,
        )

    summary = summarize_investigation_evaluations([
        result("one-evidence-case", 0, 1),
        result("nine-evidence-case", 9, 9),
    ])

    assert summary.evidence_grounding_rate == pytest.approx(0.9)
    assert summary.grounded_evidence_count == 9
    assert summary.evidence_count == 10


def test_evaluation_summary_surfaces_repeated_actions_and_unhelpful_budget_stops():
    from app.investigation.evaluation import InvestigationEvaluationResult

    result = InvestigationEvaluationResult(
        case_id="budget-and-repeat", passed=False,
        completed_actions=1, attempted_actions=2, invalid_action_count=0,
        invalid_action_rate=0.0, fallback_action_count=0,
        fallback_action_rate=0.0, action_relevance_rate=1.0,
        relevant_agent_action_count=1, agent_action_count=1,
        action_obligation_coverage=1.0, evidence_obligation_coverage=1.0,
        branch_obligation_coverage=1.0, evidence_grounding_rate=1.0,
        stop_reason_allowed=True, stop_reason_code="BUDGET_EXHAUSTED",
        useful_stop=False, within_budget=True, duplicate_action_count=1,
        issues=["duplicate_action_fingerprint"],
    )
    summary = summarize_investigation_evaluations([result])

    assert summary.duplicate_action_rate == 0.5
    assert summary.budget_exhaustion_rate == 1.0
    assert summary.useful_stop_rate == 0.0

"""Insight evidence agent (AGENT_PLAN Phase 8 / A3).

Replaces the fixed ``MAX_METRICS``/``MAX_DIMENSIONS`` selection inside the
insight attribution pipeline with a governed plan-execute loop:

  InsightPlan (LLM, validated) → decide (one typed action per round)
    → deterministic policy validation (metric ∈ report, dimension allowlist,
      no repeats) → deterministic breakdown query → observation
    → QUERY_EVIDENCE / REPLAN / FINISH / ABSTAIN

Hard boundaries:
- ``report_json`` is only ever read; extra evidence is computed beside it.
- Mathematical attribution stays deterministic: multiplicative decomposition
  is always ``attribution.build_decomposition``, never an LLM action.
- Any LLM/validation failure degrades deterministically: the remaining
  candidates drain through the same policy layer and the run is reported as
  ``decision_driver="workflow_fallback"`` unless an LLM action already ran.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Callable

from pydantic import TypeAdapter, ValidationError

from app.agents.insight_contracts import (
    AbstainInsightAction,
    FinishInsightAction,
    InsightAction,
    InsightPlan,
    InsightQuery,
    QueryEvidenceAction,
    ReplanInsightAction,
)
from app.analysis.attribution import (
    MAX_DIMENSIONS,
    MAX_METRICS,
    _metric,
    _platform_dimension_finding,
    _seller_dimension_finding,
    _SELLER_DIMENSIONS,
    AttributionDataInsufficient,
    build_dimension_data_gap_evidence,
    build_decomposition,
    build_grouped_ratio_attributions,
)
from app.analysis.contracts import ContributionFinding, InsightAngle, InsightBrief, MetricDecomposition, PeriodicReport
from app.analysis.periods import previous_period
from app.core.config import get_settings
from app.data.repositories.daily_metrics import SUPPORTED_BREAKDOWN_DIMENSIONS
from app.analysis.semantic import get_metric_definition
from app.metrics.registry import get_registry
from app.investigation.candidates import candidate_id_for
from app.investigation.contracts import BreakdownDimensionAction
from app.investigation.evidence import build_attribution_evidence, scope_fingerprint
from app.investigation.engine import record_action_result, run_investigation
from app.investigation.contracts import (
    AgentDecision, AnalysisTarget, AttributionEvidence, BreakdownDimensionAction, CandidateAction,
    CandidateFeatures, DecomposeMetricAction, InvestigationState,
    InvestigationSummary, StopReason, WorkflowSelection,
)

_ACTION_ADAPTER = TypeAdapter(InsightAction)


def legal_query_surface(report: PeriodicReport, brief: InsightBrief) -> list[InsightQuery]:
    """Every (metric, dimension) pair the policy layer would ever allow."""
    scope = report.scope.scope_type
    surface: list[InsightQuery] = []
    for metric in report.metrics:
        if metric.period_over_period is None:
            continue
        if metric.period_over_period.delta_value is None or metric.period_over_period.baseline_value is None:
            continue
        if brief.focus_metrics and metric.metric_name not in brief.focus_metrics:
            continue
        try:
            definition = get_metric_definition(metric.metric_name, scope)
            capability = get_registry().get(metric.metric_name).attribution.for_scope(scope)
        except Exception:  # noqa: BLE001 - absent declarations fail closed
            continue
        if "additive_change_v1" not in capability.method_ids:
            continue
        dimensions = set(capability.dimensions)
        if scope == "seller":
            dimensions &= set(_SELLER_DIMENSIONS)
        else:
            dimensions &= set(definition.allowed_dimensions) & SUPPORTED_BREAKDOWN_DIMENSIONS
        for dimension in dimensions:
            action = BreakdownDimensionAction(
                action_type="BREAKDOWN_DIMENSION", metric_id=metric.metric_name,
                dimension=dimension, method_id="additive_change_v1",
            )
            surface.append(InsightQuery(
                candidate_id=candidate_id_for(action),
                metric_name=metric.metric_name, dimension=dimension,
            ))
    surface.sort(key=lambda item: (
        -abs(next((metric.period_over_period.delta_value or 0.0 for metric in report.metrics
                   if metric.metric_name == item.metric_name and metric.period_over_period), 0.0)),
        item.candidate_id,
    ))
    return surface[:6]


def _llm_plan(surface: list[InsightQuery], report: PeriodicReport) -> InsightPlan | None:
    from app.llm.client import llm_agent_action

    settings = get_settings()
    prompt = (
        "你是报告洞察的证据选材器。报告已锁定，以下是允许补查的 (metric, dimension) 候选白名单 JSON。"
        "请输出 JSON 对象 plan：goal、evidence_gaps（描述缺口）、candidate_ids（从服务端候选中选择、"
        f"最多 {settings.insight_agent_max_queries} 项）、stop_conditions、budget（对象 {{max_rounds, "
        f"max_queries, max_tokens, max_cost}}；max_tokens 不得超过 {settings.max_tokens}，"
        f"max_cost 不得超过 {settings.max_cost} 美元）。只能返回候选 ID，不得输出或构造指标、维度、日期或查询参数。候选：\n"
        + json.dumps([{"candidate_id": item.candidate_id, "metric": item.metric_name,
                       "dimension": item.dimension} for item in surface], ensure_ascii=False)
    )
    try:
        raw = llm_agent_action(prompt, json.dumps({
            "report_type": report.report_type, "period": report.period.model_dump(mode="json"),
            "metrics": [{"metric_name": m.metric_name, "delta": m.period_over_period.delta_value if m.period_over_period else None} for m in report.metrics],
        }, ensure_ascii=False))
        if isinstance(raw, dict) and isinstance(raw.get("plan"), dict):
            raw = raw["plan"]
        if not raw:
            return None
        return InsightPlan.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - degrade deterministically
        print(f"[DEBUG/insight-agent] plan validation failed: {type(exc).__name__}: {exc}")
        return None


def _fallback_plan(surface: list[InsightQuery]) -> InsightPlan:
    """The historical fixed selection expressed as a plan (fallback parity)."""
    metrics_in_order: list[str] = []
    for item in surface:
        if item.metric_name not in metrics_in_order:
            metrics_in_order.append(item.metric_name)
    selected: list[InsightQuery] = []
    for metric_name in metrics_in_order[:MAX_METRICS]:
        selected.extend(q for q in surface if q.metric_name == metric_name)
        selected = selected[:MAX_METRICS * MAX_DIMENSIONS]
    selected = selected or [surface[0]]
    return InsightPlan.model_validate({
        "goal": "固定选材回退：对变化最大的求和型指标补查固定数量的维度拆解。",
        "evidence_gaps": ["固定选材（workflow_fallback）"],
        "candidate_ids": [item.candidate_id for item in selected],
        "stop_conditions": ["已补查全部候选"],
        # The shared investigation engine executes one selected query per
        # decision round, so the fallback must budget enough rounds for its
        # intended fixed-selection parity.
        "budget": {"max_rounds": max(1, min(6, len(selected))), "max_queries": len(selected), "max_tokens": 0, "max_cost": 0.0},
    })


def _effective_query_budget(plan: InsightPlan, server_max_queries: int) -> int:
    """Clamp data actions by every declared/server ceiling.

    The engine takes one candidate per round; therefore ``max_rounds`` is a
    real execution ceiling, not merely an LLM planning hint.
    """
    return max(0, min(
        6,
        int(plan.budget.max_rounds),
        int(plan.budget.max_queries),
        int(server_max_queries),
    ))


@contextmanager
def _insight_llm_budget_context():
    """Ensure report attribution LLM calls use the shared token/cost meter."""
    from app.core.execution_context import active_state, activate_state, reset_state

    settings = get_settings()
    state = active_state()
    token = None
    if state is None:
        state = {"llm_usage": {"input_tokens": 0, "output_tokens": 0, "estimated_cost": 0.0}}
        token = activate_state(state)
    usage = state.setdefault("llm_usage", {
        "input_tokens": 0, "output_tokens": 0, "estimated_cost": 0.0,
    })
    budget = state.setdefault("runtime_budget", {})
    # Keep a stricter caller budget, while ensuring a missing/zero caller value
    # cannot disable the server's configured hard ceiling.
    configured_tokens = int(settings.max_tokens)
    configured_cost = float(settings.max_cost)
    current_tokens = int(budget.get("max_tokens") or configured_tokens)
    current_cost = float(budget.get("max_cost") or configured_cost)
    budget["max_tokens"] = min(current_tokens, configured_tokens)
    budget["max_cost"] = min(current_cost, configured_cost)
    state["llm_usage"] = usage
    try:
        yield state
    finally:
        if token is not None:
            reset_state(token)


def _apply_plan_llm_budget(plan: InsightPlan) -> None:
    """Narrow subsequent selector calls to the validated plan budget."""
    from app.core.execution_context import active_state

    state = active_state()
    if state is None:
        return
    settings = get_settings()
    budget = state.setdefault("runtime_budget", {})
    budget["max_tokens"] = min(int(budget.get("max_tokens") or settings.max_tokens), plan.budget.max_tokens)
    budget["max_cost"] = min(float(budget.get("max_cost") or settings.max_cost), plan.budget.max_cost)


def _plan_llm_budget_exhausted(plan: InsightPlan) -> bool:
    """Check usage before starting another selector call (no avoidable spend)."""
    from app.core.execution_context import active_state

    state = active_state() or {}
    usage = state.get("llm_usage") or {}
    settings = get_settings()
    token_limit = min(int(settings.max_tokens), int(plan.budget.max_tokens))
    cost_limit = min(float(settings.max_cost), float(plan.budget.max_cost))
    used_tokens = int(usage.get("input_tokens", 0) or 0) + int(usage.get("output_tokens", 0) or 0)
    used_cost = float(usage.get("estimated_cost", 0.0) or 0.0)
    return used_tokens >= token_limit or used_cost >= cost_limit


def _llm_decide(
    plan: InsightPlan,
    executed: set[tuple[str, str]],
    observations: list[dict[str, Any]],
    candidates: list[InsightQuery],
    *,
    candidate_set_id: str,
) -> dict[str, Any] | None:
    from app.llm.client import llm_agent_action

    prompt = (
        "你是报告洞察的证据选材决策器。每轮只输出一个 JSON action，action_type ∈ "
        "QUERY_EVIDENCE / REPLAN / FINISH / ABSTAIN。"
        f"本轮 candidate_set_id={candidate_set_id}。所有 action 必须原样携带此 ID。"
        "QUERY_EVIDENCE 示例：{\"action_type\":\"QUERY_EVIDENCE\",\"candidate_set_id\":\"<本轮ID>\",\"candidate_ids\":[\"cand_...\"]}；"
        "REPLAN 用 remaining_candidate_ids 更新待查清单，也必须携带 candidate_set_id；"
        "证据足够时输出带 candidate_set_id 的 FINISH。不得选择白名单之外或已查过的组合。"
        "每个候选 ID 对应唯一服务端生成的查询，禁止自行提供指标名或维度。计划、候选与已观察证据如下：\n"
        + json.dumps({"candidate_set_id": candidate_set_id,
                      "plan": plan.model_dump(mode="json"),
                      "candidate_ids": [q.candidate_id for q in candidates],
                      "executed": sorted(executed), "observations": observations}, ensure_ascii=False)
    )
    try:
        raw = llm_agent_action(prompt, "选择下一个证据补查动作。")
        if isinstance(raw, dict) and isinstance(raw.get("action"), dict):
            raw = raw["action"]
        if not raw:
            return None
        action = _ACTION_ADAPTER.validate_python(raw)
        if action.candidate_set_id != candidate_set_id:
            return None
        available_ids = {item.candidate_id for item in candidates}
        if isinstance(action, QueryEvidenceAction) and not set(action.candidate_ids) <= available_ids:
            return None
        if isinstance(action, ReplanInsightAction) and not set(action.remaining_candidate_ids) <= available_ids:
            return None
        return action
    except Exception as exc:  # noqa: BLE001 - degrade deterministically
        print(f"[DEBUG/insight-agent] decide validation failed: {type(exc).__name__}: {exc}")
        return None


def _execute_query(report: PeriodicReport, query: InsightQuery, baseline_period) -> ContributionFinding | None:
    metric = _metric(report, query.metric_name)
    if metric is None:
        return None
    # Let the shared Investigation Engine classify and audit query failures as
    # EXECUTION_FAILED. Swallowing the exception here made infrastructure/query
    # errors indistinguishable from a legitimate empty result (DATA_INSUFFICIENT).
    if report.scope.scope_type == "seller":
        return _seller_dimension_finding(report, metric, query.dimension, baseline_period)
    return _platform_dimension_finding(report, metric, query.dimension, baseline_period)


def query_attribution_evidence_agent(
    report: PeriodicReport, brief: InsightBrief,
    progress: Callable[[str, str], None] | None = None,
    trace_id: str | None = None,
) -> tuple[list[ContributionFinding], list[MetricDecomposition], dict[str, dict[str, object]], str, InvestigationSummary | None]:
    with _insight_llm_budget_context():
        return _query_attribution_evidence_agent(report, brief, progress, trace_id)


def _audit_snapshot_decompositions(
    report: PeriodicReport,
    brief: InsightBrief,
    decompositions: list[MetricDecomposition],
    evidence: dict[str, dict[str, object]],
) -> InvestigationSummary | None:
    """Record snapshot-only deterministic decompositions in the shared ledger.

    These calculations need no MCP query, but they are still investigation
    actions and their evidence must be traceable to a server-issued candidate.
    """
    if not report.metrics:
        return None
    target_metric_id = (
        decompositions[0].metric_name if decompositions
        else (brief.focus_metrics[0] if brief.focus_metrics else report.metrics[0].metric_name)
    )
    target_metric = _metric(report, target_metric_id)
    if target_metric is None or target_metric.period_over_period is None:
        return None
    comparison = target_metric.period_over_period
    baseline = comparison.baseline_period or previous_period(report.period)
    scope_snapshot = {
        "scope_type": report.scope.scope_type,
        "seller_id": report.scope.seller_id,
        "current_period": report.period.model_dump(mode="json"),
        "baseline_period": baseline.model_dump(mode="json"),
    }
    try:
        definition_version = get_registry().get(target_metric_id).version
    except Exception:  # noqa: BLE001 - a missing registry definition cannot be audited as executable
        return None
    target = AnalysisTarget(
        metric_id=target_metric_id,
        current_start=report.period.start_date,
        current_end=report.period.end_date,
        baseline_start=baseline.start_date,
        baseline_end=baseline.end_date,
        canonical_scope=scope_snapshot,
        scope_fingerprint=scope_fingerprint(scope_snapshot),
        requested_analysis_type="contribution",
        method_version="1.0",
        metric_definition_version=definition_version,
    )
    investigation = InvestigationState(target=target, remaining_budget=6)
    current_period = report.period.model_dump(mode="json")
    baseline_period = baseline.model_dump(mode="json")

    _record_snapshot_decompositions(
        decompositions, evidence, investigation,
        current_period=current_period, baseline_period=baseline_period,
    )

    # Let the shared engine decide and record the terminal state after all
    # snapshot actions have been accounted for.
    run_investigation(
        investigation,
        candidate_factory=lambda _state: [],
        selector=lambda _candidates, _state: None,
        executor=lambda _candidate, _state: [],
        current_period=current_period,
        baseline_period=baseline_period,
        max_actions=1,
    )
    return _investigation_summary(investigation)


def _record_snapshot_decompositions(
    decompositions: list[MetricDecomposition],
    evidence: dict[str, dict[str, object]],
    investigation: InvestigationState,
    *,
    current_period: dict,
    baseline_period: dict,
) -> int:
    """Attach deterministic snapshot calculations to a live investigation."""
    recorded_before = len(investigation.action_results)

    for decomposition in decompositions:
        if decomposition.metric_name != investigation.target.metric_id:
            # A single InvestigationState has one immutable root target.
            # Do not attach another metric's result to it; that result remains
            # in the report evidence bundle under its own metric identity.
            continue
        evidence_id = next(iter(decomposition.evidence_ids), "")
        row = evidence.get(evidence_id, {})
        raw_typed = row.get("investigation_evidence")
        if not isinstance(raw_typed, dict):
            continue
        action = DecomposeMetricAction(
            action_type="DECOMPOSE_METRIC",
            metric_id=decomposition.metric_name,
            method_id=decomposition.method_id,
            factor_metric_ids=list(decomposition.factor_metrics),
        )
        candidate_id = candidate_id_for(action)
        candidate = CandidateAction(
            candidate_id=candidate_id,
            action=action,
            features=CandidateFeatures(
                relevance_score=1.0,
                data_availability="available",
                evidence_gap=f"metric_decomposition:{decomposition.metric_name}",
                rule_score=1.0,
            ),
        )
        typed_source = AttributionEvidence.model_validate(raw_typed)
        typed = typed_source.model_copy(update={
            "source_action_id": candidate_id,
            "values": {
                **typed_source.values,
                "factor_metrics": list(decomposition.factor_metrics),
            },
        })
        record_action_result(
            investigation, candidate, [typed],
            current_period=current_period, baseline_period=baseline_period,
            decision_source="backend",
        )
    return len(investigation.action_results) - recorded_before


def _investigation_summary(investigation: InvestigationState) -> InvestigationSummary:
    return InvestigationSummary(
        metric_id=investigation.target.metric_id,
        scope_fingerprint=investigation.target.scope_fingerprint,
        stop_reason=investigation.stop_reason,
        progress=investigation.progress,
        calculation_progress=investigation.calculation_progress,
        actions_used=len(investigation.action_results),
        remaining_budget=investigation.remaining_budget,
        branches=investigation.branches,
        action_results=investigation.action_results,
        policy_rejections=investigation.policy_rejections,
        decision_events=investigation.observations,
    )


def _query_attribution_evidence_agent(
    report: PeriodicReport, brief: InsightBrief,
    progress: Callable[[str, str], None] | None = None,
    trace_id: str | None = None,
) -> tuple[list[ContributionFinding], list[MetricDecomposition], dict[str, dict[str, object]], str, InvestigationSummary | None]:
    """Agent-selected attribution evidence.

    Returns ``(contributions, decompositions, evidence, decision_driver)`` —
    the first three match ``attribution.query_attribution_evidence`` so the
    insight pipeline can swap the selector without touching its contracts.
    """
    if brief.angle != InsightAngle.CONTRIBUTION:
        return [], [], {}, "workflow_fallback", None
    surface = legal_query_surface(report, brief)
    decompositions, decomp_evidence = build_decomposition(report, brief, trace_id=trace_id)
    grouped_ratio_evidence = build_grouped_ratio_attributions(report, brief, trace_id=trace_id)
    decomp_evidence.update(grouped_ratio_evidence)
    if not surface:
        # A missing dimension-breakdown surface does not mean there is no
        # deterministic attribution to perform. Ratio/product decompositions
        # consume the immutable report snapshot and are independent of the
        # additive breakdown action space.
        summary = _audit_snapshot_decompositions(report, brief, decompositions, decomp_evidence)
        return [], decompositions, decomp_evidence, "workflow_fallback", summary
    query_by_id = {query.candidate_id: query for query in surface}
    plan = _llm_plan(surface, report)
    if plan is not None and not set(plan.candidate_ids) <= set(query_by_id):
        plan = None
    decision_driver = "llm"
    if plan is None:
        plan = _fallback_plan(surface)
        decision_driver = "workflow_fallback"
    _apply_plan_llm_budget(plan)

    contributions: list[ContributionFinding] = []
    evidence: dict[str, dict[str, object]] = {}
    executed: set[tuple[str, str]] = set()
    remaining_ids = [candidate_id for candidate_id in plan.candidate_ids if candidate_id in query_by_id]
    max_queries = _effective_query_budget(plan, get_settings().insight_agent_max_queries)
    root_metric = surface[0].metric_name
    root_report_metric = _metric(report, root_metric)
    comparison_baseline = (
        root_report_metric.period_over_period.baseline_period
        if root_report_metric is not None and root_report_metric.period_over_period is not None
        else None
    )
    baseline_period = comparison_baseline or previous_period(report.period)
    scope_snapshot = {
        "scope_type": report.scope.scope_type,
        "seller_id": report.scope.seller_id,
        "current_period": report.period.model_dump(mode="json"),
        "baseline_period": baseline_period.model_dump(mode="json"),
    }
    target = AnalysisTarget(
        metric_id=root_metric,
        current_start=report.period.start_date, current_end=report.period.end_date,
        baseline_start=baseline_period.start_date, baseline_end=baseline_period.end_date,
        canonical_scope=scope_snapshot,
        scope_fingerprint=scope_fingerprint(scope_snapshot),
        requested_analysis_type="contribution", method_version="1.0",
        metric_definition_version=get_registry().get(root_metric).version,
    )
    investigation = InvestigationState(target=target, remaining_budget=max_queries)
    decomposition_actions = _record_snapshot_decompositions(
        decompositions, decomp_evidence, investigation,
        current_period=report.period.model_dump(mode="json"),
        baseline_period=baseline_period.model_dump(mode="json"),
    )
    observations: list[dict[str, Any]] = [
        {"action_type": "DECOMPOSE_METRIC", **item}
        for item in investigation.observations
    ]

    def candidate_factory(_state: InvestigationState) -> list[CandidateAction]:
        output: list[CandidateAction] = []
        for candidate_id in remaining_ids:
            query = query_by_id.get(candidate_id)
            if query is None or (query.metric_name, query.dimension) in executed:
                continue
            action = BreakdownDimensionAction(
                action_type="BREAKDOWN_DIMENSION", metric_id=query.metric_name,
                dimension=query.dimension, method_id="additive_change_v1",
            )
            output.append(CandidateAction(
                candidate_id=candidate_id, action=action,
                features=CandidateFeatures(
                    relevance_score=0.5, unresolved_target_share=1.0,
                    estimated_query_cost=1.0,
                    evidence_gap=f"breakdown:{query.metric_name}:{query.dimension}",
                    rule_score=0.5,
                ),
            ))
        return output[:6]

    def selector(candidates: list[CandidateAction], _state: InvestigationState) -> WorkflowSelection | AgentDecision | None:
        nonlocal decision_driver, remaining_ids
        if not candidates:
            return None
        if decision_driver != "llm":
            return WorkflowSelection(candidate_id=candidates[0].candidate_id)
        if _plan_llm_budget_exhausted(plan):
            decision_driver = "workflow_fallback"
            return WorkflowSelection(candidate_id=candidates[0].candidate_id)
        available = [query_by_id[item.candidate_id] for item in candidates]
        active_set = _state.candidate_set
        if active_set is None:
            decision_driver = "workflow_fallback"
            return WorkflowSelection(candidate_id=candidates[0].candidate_id)
        action = _llm_decide(
            plan, executed, observations, available,
            candidate_set_id=active_set.candidate_set_id,
        )
        if action is None:
            # An invalid, missing, or stale-set response did not select the
            # action. Keep the deterministic fallback auditable as such.
            decision_driver = "workflow_fallback"
            _state.observations.append({
                "event": "selector_fallback",
                "reason": "missing_invalid_or_stale_agent_decision",
                "candidate_set_id": active_set.candidate_set_id,
            })
            return WorkflowSelection(candidate_id=candidates[0].candidate_id)
        if isinstance(action, FinishInsightAction):
            return AgentDecision(
                candidate_set_id=active_set.candidate_set_id,
                decision="soft_stop", soft_stop_reason="ENOUGH_EVIDENCE",
                rationale=action.conclusion,
            )
        if isinstance(action, AbstainInsightAction):
            _state.stop_reason = StopReason(code="DATA_INSUFFICIENT", detail=action.reason, initiator="agent")
            return None
        if isinstance(action, ReplanInsightAction):
            remaining_ids = [item for item in action.remaining_candidate_ids if item in query_by_id]
            selected_id = next((item for item in remaining_ids if item in {candidate.candidate_id for candidate in candidates}), None)
        else:
            selected_id = next((item for item in action.candidate_ids if item in {candidate.candidate_id for candidate in candidates}), None)
        if selected_id is None:
            decision_driver = "workflow_fallback"
            _state.observations.append({
                "event": "selector_fallback",
                "reason": "agent_selected_no_current_candidate",
                "candidate_set_id": active_set.candidate_set_id,
            })
            return WorkflowSelection(candidate_id=candidates[0].candidate_id)
        return AgentDecision(
            candidate_set_id=active_set.candidate_set_id,
            decision="select_candidate", candidate_id=selected_id,
            rationale="报告洞察 Agent 从当前候选集中选择补充证据。",
        )

    def executor(candidate: CandidateAction, state: InvestigationState):
        query = query_by_id[candidate.candidate_id]
        key = (query.metric_name, query.dimension)
        executed.add(key)
        try:
            finding = _execute_query(report, query, baseline_period)
        except AttributionDataInsufficient as exc:
            metric = _metric(report, query.metric_name)
            gap_id, gap = build_dimension_data_gap_evidence(
                report,
                metric_id=query.metric_name,
                metric_unit=metric.unit if metric is not None else "",
                dimension=query.dimension,
                baseline_period=baseline_period,
                period_statuses=exc.period_statuses,
                trace_id=trace_id,
                source_action_id=candidate.candidate_id,
            )
            evidence[gap_id] = gap
            observations.append({
                "action_type": "QUERY_EVIDENCE",
                "candidate_id": candidate.candidate_id,
                "outcome": "DATA_INSUFFICIENT",
                "evidence_ids": [gap_id],
            })
            # The gap is audited as a typed limitation, not as successful
            # business evidence; the shared engine records DATA_INSUFFICIENT.
            return []
        if finding is None:
            return []
        contributions.append(finding)
        eid = finding.evidence_ids[0]
        typed = build_attribution_evidence(
            evidence_id=eid, evidence_type="dimension_contribution", metric_id=query.metric_name,
            trace_id=trace_id,
            claim=f"{query.dimension} 维度的分组变化贡献到指标变动。",
            values={"total_delta": finding.total_delta,
                    "components": [item.model_dump(mode="json") for item in finding.items],
                    "residual_value": finding.residual_value,
                    "residual_type": finding.residual_type,
                    "residual_explainable": finding.residual_explainable},
            source_action_id=candidate.candidate_id, source_ref="report_attribution_query",
            scope=scope_snapshot, method_id="additive_change_v1", method_version="1.0",
            authorized_metric_provenance="locked_report_metric_snapshot",
            comparison={"current_period": report.period.model_dump(mode="json"),
                        "baseline_period": baseline_period.model_dump(mode="json")},
            limitations=["统计贡献不代表因果关系。"],
        )
        evidence[eid] = {"kind": "contribution", **finding.model_dump(mode="json"),
                         "investigation_evidence": typed.model_dump(mode="json")}
        observations.append({"action_type": "QUERY_EVIDENCE", "candidate_id": candidate.candidate_id,
                             "evidence_ids": [eid]})
        return [typed]

    def emit_investigation_progress(progress_state, current_candidate) -> None:
        if progress is None:
            return
        completed = len(investigation.action_results)
        remaining = progress_state.remaining_budget
        action_label = (
            current_candidate.action.action_type
            if current_candidate is not None else (investigation.stop_reason.code if investigation.stop_reason else "调查中")
        )
        progress(
            "attribution",
            f"归因调查：已完成 {completed} 项，剩余预算 {remaining} 项；当前状态：{action_label}。",
        )

    run_investigation(
        investigation, candidate_factory=candidate_factory, selector=selector,
        executor=executor, current_period=report.period.model_dump(mode="json"),
        baseline_period=baseline_period.model_dump(mode="json"),
        progress_callback=emit_investigation_progress,
        max_actions=max(0, max_queries - decomposition_actions),
    )

    summary = _investigation_summary(investigation)
    return contributions, decompositions, {**evidence, **decomp_evidence}, decision_driver, summary

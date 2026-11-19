"""AGENT_PLAN Phase 8: insight evidence agent tests.

The report is seller-scoped so the legal surface needs no metric-definition
lookup; the deterministic breakdown query is stubbed at the agent module
boundary, keeping these tests hermetic.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import re
from unittest.mock import patch

from app.agents.insight_agent import (
    _effective_query_budget,
    _llm_decide,
    legal_query_surface,
    query_attribution_evidence_agent,
)
from app.agents.insight_contracts import InsightPlan, InsightQuery, QueryEvidenceAction
from app.analysis.attribution import query_attribution_evidence
from app.analysis.contracts import (
    ContributionFinding,
    ContributionItem,
    InsightAngle,
    InsightBrief,
    MetricPoint,
    MetricSeries,
    PeriodComparison,
    PeriodSpec,
    PeriodicReport,
    ReportScope,
    ReportMetric,
)
from app.analysis.periods import previous_period
from app.investigation.candidates import candidate_id_for
from app.investigation.contracts import BreakdownDimensionAction


def _metric(name: str, value: float, baseline: float) -> ReportMetric:
    period = PeriodSpec(start_date=date(2018, 7, 30), end_date=date(2018, 8, 5))
    return ReportMetric(
        metric_name=name, value=value, unit="元",
        period_over_period=PeriodComparison(
            comparison_type="period_over_period", current_period=period,
            baseline_period=period, current_value=value, baseline_value=baseline,
            delta_value=round(value - baseline, 2), delta_ratio=-16.67, status="ok",
        ),
        evidence_ids=[f"ev_{name}"],
    )


def _report() -> PeriodicReport:
    period = PeriodSpec(start_date=date(2018, 7, 30), end_date=date(2018, 8, 5))
    # Three additive metrics: the fixed selection caps at MAX_METRICS=2, so a
    # plan that reaches the third metric strictly expands evidence coverage.
    metrics = [_metric("gmv", 100.0, 120.0), _metric("order_count", 90.0, 100.0), _metric("freight_value", 20.0, 25.0)]
    return PeriodicReport(
        report_id="report_agent", report_type="weekly", period=period,
        title="经营分析周报", summary="覆盖 3 个核心指标。", metrics=metrics,
        evidence_records={f"ev_{m.metric_name}": {"kind": "metric_series", "metric_name": m.metric_name} for m in metrics},
        lineage_refs=["fact_order_items -> gmv"],
        data_freshness_at=datetime(2018, 8, 5, 23, 59, 59, tzinfo=timezone.utc),
        scope={"scope_type": "seller", "seller_id": "seller-1"},
    )


_BRIEF = InsightBrief(angle=InsightAngle.CONTRIBUTION, min_words=5, max_words=200)


def _finding(metric_name: str, dimension: str) -> ContributionFinding:
    return ContributionFinding(
        metric_name=metric_name, dimension=dimension,
        current_date=date(2018, 8, 5), baseline_date=date(2018, 7, 29),
        total_delta=20.0,
        items=[ContributionItem(dimension_value="A", current_value=80.0, baseline_value=60.0, delta_value=20.0, contribution_ratio=1.0)],
        evidence_ids=[f"ev_{metric_name}_{dimension}"],
    )


def _fake_finding(report, metric, dimension: str, baseline_period):
    return _finding(metric.metric_name, dimension)


def _plan_for(pairs: list[tuple[str, str]]) -> dict:
    return {"plan": {
        "goal": "补齐维度贡献证据缺口",
        "evidence_gaps": ["缺少第三指标的下钻证据"],
        "candidate_ids": [candidate_id_for(BreakdownDimensionAction(
            action_type="BREAKDOWN_DIMENSION", metric_id=m, dimension=d,
            method_id="additive_change_v1",
        )) for m, d in pairs],
        "stop_conditions": ["已补查全部候选"],
        "budget": {"max_rounds": 6, "max_queries": 6, "max_tokens": 12000, "max_cost": 1.0},
    }}


def _candidate_set_id_from_prompt(system_prompt: str) -> str:
    matches = re.findall(r'"candidate_set_id"\s*:\s*"([^"]+)"', system_prompt)
    assert matches
    return matches[-1]


def test_report_selector_rejects_decision_from_stale_candidate_set():
    def stale_response(system_prompt, _user_message, **_kwargs):
        return {"action": {
            "action_type": "QUERY_EVIDENCE",
            "candidate_set_id": "set-old",
            "candidate_ids": ["candidate-current"],
        }}

    candidate = InsightQuery(
        candidate_id="candidate-current", metric_name="gmv", dimension="product_category",
    )
    with patch("app.llm.client.llm_agent_action", stale_response):
        decision = _llm_decide(
            InsightPlan.model_validate(_plan_for([("gmv", "product_category")])["plan"]),
            set(), [], [candidate], candidate_set_id="set-current",
        )
    assert decision is None


def test_report_selector_accepts_candidate_only_with_current_candidate_set():
    candidate = InsightQuery(
        candidate_id="candidate-current", metric_name="gmv", dimension="product_category",
    )

    def current_response(system_prompt, _user_message, **_kwargs):
        return {"action": {
            "action_type": "QUERY_EVIDENCE",
            "candidate_set_id": _candidate_set_id_from_prompt(system_prompt),
            "candidate_ids": [candidate.candidate_id],
        }}

    with patch("app.llm.client.llm_agent_action", current_response):
        decision = _llm_decide(
            InsightPlan.model_validate(_plan_for([("gmv", "product_category")])["plan"]),
            set(), [], [candidate], candidate_set_id="set-current",
        )
    assert isinstance(decision, QueryEvidenceAction)
    assert decision.candidate_set_id == "set-current"


def test_legal_surface_covers_all_additive_metrics_and_seller_dimensions():
    surface = legal_query_surface(_report(), _BRIEF)
    metrics = {item.metric_name for item in surface}
    dimensions = {item.dimension for item in surface}
    assert metrics == {"gmv", "order_count", "freight_value"}
    assert dimensions == {"product_category", "month_position"}


def test_agent_selection_expands_coverage_beyond_fixed_selection():
    pairs = [("gmv", "product_category"), ("gmv", "month_position"),
             ("order_count", "product_category"), ("order_count", "month_position"),
             ("freight_value", "product_category"), ("freight_value", "month_position")]
    with patch("app.llm.client.llm_agent_action", lambda *a, **k: _plan_for(pairs)), \
         patch("app.agents.insight_agent._seller_dimension_finding", _fake_finding):
        contributions, decompositions, evidence, driver, summary = query_attribution_evidence_agent(_report(), _BRIEF)
    # The stub returns an LLM plan but no valid per-round decision, so the
    # actual query traversal is workflow fallback rather than an LLM choice.
    assert driver == "workflow_fallback"
    assert summary is not None
    assert summary.actions_used == len(summary.action_results)
    covered = {(item.metric_name, item.dimension) for item in contributions}
    assert ("freight_value", "product_category") in covered  # the third metric is reachable
    assert all(eid.startswith("ev_") for eid in evidence)
    assert all("investigation_evidence" in row for row in evidence.values())
    assert all(row["investigation_evidence"]["authorized_metric_provenance"] == "locked_report_metric_snapshot"
               for row in evidence.values())

    # The fixed selection caps at MAX_METRICS=2 metrics; with the same stubbed
    # finder it can never reach freight_value.
    with patch("app.analysis.attribution._seller_dimension_finding", _fake_finding):
        fixed_contributions, _, _ = query_attribution_evidence(_report(), _BRIEF)
    fixed_covered = {(item.metric_name, item.dimension) for item in fixed_contributions}
    assert "freight_value" not in {metric for metric, _ in fixed_covered}
    assert len(covered) > len(fixed_covered)


def test_report_agent_honors_max_rounds_even_when_query_budget_is_larger():
    pairs = [
        ("gmv", "product_category"), ("gmv", "month_position"),
        ("order_count", "product_category"), ("order_count", "month_position"),
        ("freight_value", "product_category"), ("freight_value", "month_position"),
    ]
    plan_payload = _plan_for(pairs)["plan"]
    plan_payload["budget"] = {
        "max_rounds": 1, "max_queries": 6, "max_tokens": 12000, "max_cost": 1.0,
    }
    plan = InsightPlan.model_validate(plan_payload)

    with patch("app.agents.insight_agent._llm_plan", lambda *_: plan), \
         patch("app.agents.insight_agent._llm_decide", lambda _plan, _done, _obs, candidates, **kwargs: QueryEvidenceAction(
             action_type="QUERY_EVIDENCE", candidate_set_id=kwargs["candidate_set_id"],
             candidate_ids=[candidates[0].candidate_id],
         )), \
         patch("app.agents.insight_agent._seller_dimension_finding", _fake_finding):
        contributions, _, _, driver, summary = query_attribution_evidence_agent(_report(), _BRIEF)

    assert driver == "llm"
    assert len(contributions) == 1
    assert summary is not None and summary.actions_used == 1
    assert any(item.decision_source == "agent" for item in summary.action_results), (
        [(item.action_type, item.decision_source) for item in summary.action_results]
        + [summary.decision_events]
    )


def test_effective_report_query_budget_uses_all_server_and_agent_limits():
    plan_payload = _plan_for([("gmv", "product_category")])["plan"]
    plan_payload["budget"] = {
        "max_rounds": 2, "max_queries": 5, "max_tokens": 12000, "max_cost": 1.0,
    }
    plan = InsightPlan.model_validate(plan_payload)
    assert _effective_query_budget(plan, 6) == 2
    assert _effective_query_budget(plan, 1) == 1


def test_zero_plan_token_budget_disables_further_llm_decisions():
    pairs = [("gmv", "product_category")]
    plan_payload = _plan_for(pairs)["plan"]
    plan_payload["budget"] = {
        "max_rounds": 1, "max_queries": 1, "max_tokens": 0, "max_cost": 1.0,
    }
    plan = InsightPlan.model_validate(plan_payload)
    with patch("app.agents.insight_agent._llm_plan", lambda *_: plan), \
         patch("app.agents.insight_agent._llm_decide") as decide, \
         patch("app.agents.insight_agent._seller_dimension_finding", _fake_finding):
        contributions, _, _, driver, summary = query_attribution_evidence_agent(_report(), _BRIEF)
    decide.assert_not_called()
    assert len(contributions) == 1
    assert driver == "workflow_fallback"
    assert summary is not None and summary.actions_used == 1


def test_invalid_or_stale_report_decision_is_recorded_as_workflow_fallback():
    plan = _plan_for([("gmv", "product_category")])
    with patch("app.agents.insight_agent._llm_plan", lambda *_: InsightPlan.model_validate(plan["plan"])), \
         patch("app.agents.insight_agent._llm_decide", lambda *_args, **_kwargs: None), \
         patch("app.agents.insight_agent._seller_dimension_finding", _fake_finding):
        contributions, _, _, driver, summary = query_attribution_evidence_agent(_report(), _BRIEF)

    assert contributions
    assert driver == "workflow_fallback"
    assert summary is not None
    assert summary.actions_used == 1
    assert any(
        event.get("event") == "selector_fallback"
        and event.get("reason") == "missing_invalid_or_stale_agent_decision"
        for event in summary.decision_events
    )


def test_report_query_exception_is_audited_as_execution_failure_not_empty_data():
    pairs = [("gmv", "product_category")]
    plan = InsightPlan.model_validate(_plan_for(pairs)["plan"])

    def _raise_query_error(*_args, **_kwargs):
        raise RuntimeError("warehouse connection failed")

    with patch("app.agents.insight_agent._llm_plan", lambda *_: plan), \
         patch("app.agents.insight_agent._llm_decide", lambda _plan, _done, _obs, candidates, **kwargs: QueryEvidenceAction(
             action_type="QUERY_EVIDENCE", candidate_set_id=kwargs["candidate_set_id"],
             candidate_ids=[candidates[0].candidate_id],
         )), \
         patch("app.agents.insight_agent._seller_dimension_finding", _raise_query_error):
        contributions, _, _, _driver, summary = query_attribution_evidence_agent(_report(), _BRIEF)

    assert contributions == []
    assert summary is not None
    assert len(summary.action_results) == 1
    result = summary.action_results[0]
    assert result.status == "failed"
    assert result.failure_code == "EXECUTION_FAILED"
    assert "RuntimeError" in (result.failure_detail or "")
    assert "warehouse connection failed" not in (result.failure_detail or "")


def test_report_json_is_never_mutated():
    report = _report()
    before = report.model_dump(mode="json")
    pairs = [("gmv", "product_category"), ("order_count", "month_position")]
    with patch("app.llm.client.llm_agent_action", lambda *a, **k: _plan_for(pairs)), \
         patch("app.agents.insight_agent._seller_dimension_finding", _fake_finding):
        query_attribution_evidence_agent(report, _BRIEF)
    assert report.model_dump(mode="json") == before


def test_llm_unavailable_falls_back_to_fixed_selection_parity():
    with patch("app.llm.client.llm_agent_action", lambda *a, **k: None), \
         patch("app.agents.insight_agent._seller_dimension_finding", _fake_finding):
        contributions, _, evidence, driver, summary = query_attribution_evidence_agent(_report(), _BRIEF)
    assert driver == "workflow_fallback"
    assert summary is not None
    covered = {(item.metric_name, item.dimension) for item in contributions}
    # Fallback mirrors the fixed selection: first MAX_METRICS metrics × first
    # MAX_DIMENSIONS seller dimensions.
    assert covered == {("gmv", "product_category"), ("gmv", "month_position"),
                       ("order_count", "product_category"), ("order_count", "month_position")}


def test_policy_denies_queries_outside_allowlist():
    pairs = [("gmv", "product_category")]
    plan = _plan_for(pairs)
    def _decide(system_prompt, user_message, **_):
        if "选材器" in system_prompt and "候选" in system_prompt:
            return plan
        valid_id = _plan_for([("gmv", "product_category")])["plan"]["candidate_ids"][0]
        return {"action": {"action_type": "QUERY_EVIDENCE",
                            "candidate_set_id": _candidate_set_id_from_prompt(system_prompt),
                            "candidate_ids": ["invented", valid_id]}}
    with patch("app.llm.client.llm_agent_action", _decide), \
         patch("app.agents.insight_agent._seller_dimension_finding", _fake_finding):
        contributions, _, _, driver, _summary = query_attribution_evidence_agent(_report(), _BRIEF)
    # The out-of-candidate response is rejected; only the legal server-side
    # fallback candidate is executed.
    assert driver == "workflow_fallback"
    covered = {(item.metric_name, item.dimension) for item in contributions}
    assert covered == {("gmv", "product_category")}  # the illegal query never executed


def test_abstain_returns_partial_evidence_without_claims():
    plan = _plan_for([("gmv", "product_category")])
    calls = {"n": 0}
    def _decide(system_prompt, user_message, **_):
        if "选材器" in system_prompt and "候选" in system_prompt:
            return plan
        calls["n"] += 1
        if calls["n"] == 1:
            candidate_id = _plan_for([("gmv", "product_category")])["plan"]["candidate_ids"][0]
            return {"action": {"action_type": "QUERY_EVIDENCE",
                                "candidate_set_id": _candidate_set_id_from_prompt(system_prompt),
                                "candidate_ids": [candidate_id]}}
        return {"action": {"action_type": "ABSTAIN",
                            "candidate_set_id": _candidate_set_id_from_prompt(system_prompt),
                            "reason": "insufficient_gaps"}}
    with patch("app.llm.client.llm_agent_action", _decide), \
         patch("app.agents.insight_agent._seller_dimension_finding", _fake_finding):
        contributions, _, _, driver, _summary = query_attribution_evidence_agent(_report(), _BRIEF)
    assert driver == "llm"
    assert len(contributions) == 1  # evidence gathered before the abstention is kept


def test_report_agent_requires_current_candidate_set_for_finish_or_abstain():
    candidate = InsightQuery(
        candidate_id="candidate-current", metric_name="gmv", dimension="product_category",
    )
    plan = InsightPlan.model_validate(_plan_for([("gmv", "product_category")])["plan"])
    for decision in (
        {"action_type": "FINISH", "candidate_set_id": "old-set"},
        {"action_type": "ABSTAIN", "candidate_set_id": "old-set", "reason": "insufficient_gaps"},
    ):
        with patch("app.llm.client.llm_agent_action", lambda *_a, **_k: {"action": decision}):
            result = _llm_decide(
                plan, set(), [], [candidate], candidate_set_id="current-set",
            )
        assert result is None


def test_multiplicative_decomposition_stays_deterministic():
    report = _report()
    pairs = [("gmv", "product_category")]
    with patch("app.llm.client.llm_agent_action", lambda *a, **k: _plan_for(pairs)), \
         patch("app.agents.insight_agent._seller_dimension_finding", _fake_finding):
        _, decompositions, _, driver, _summary = query_attribution_evidence_agent(report, _BRIEF)
    # This report has no multiplicative factor metrics beyond gmv; decomposition
    # is computed from the locked snapshot only — never from an agent action.
    assert isinstance(decompositions, list)
    assert driver in {"llm", "workflow_fallback"}


def test_gmv_decomposition_retains_raw_interaction_and_shapley_display():
    from app.analysis.attribution import build_decomposition

    report = _report().model_copy(update={
        "metrics": [
            _metric("gmv", 72.0, 50.0),
            _metric("order_count", 12.0, 10.0),
            _metric("avg_order_value", 6.0, 5.0),
        ],
        # product_shapley_v1 is deliberately declared for platform GMV only;
        # seller attribution remains limited to verified additive queries.
        "scope": ReportScope(scope_type="platform"),
    })
    decompositions, evidence = build_decomposition(
        report,
        _BRIEF.model_copy(update={"focus_metrics": ["gmv"]}),
    )
    assert len(decompositions) == 1
    item = decompositions[0]
    assert item.total_delta == 22
    assert item.interaction_delta == 2
    assert item.display_method == "shapley"
    assert sum(component.delta_value for component in item.display_factors) == 22
    stored = evidence[item.evidence_ids[0]]["investigation_evidence"]
    assert stored["method_id"] == "product_shapley_v1"
    raw_values = [part["value"] for part in stored["values"]["raw_components"]]
    display_values = [part["value"] for part in stored["values"]["display_components"]]
    assert any(part["component_type"] == "interaction_effect"
               for part in stored["values"]["raw_components"])
    total = stored["values"]["total_delta"]
    residual = stored["values"]["residual_value"]
    assert abs(sum(raw_values) + residual - total) <= 0.03
    assert abs(sum(display_values) + residual - total) <= 0.03


def test_deterministic_ratio_decomposition_runs_without_breakdown_candidates(monkeypatch):
    report = _report().model_copy(update={
        "metrics": [
            _metric("gmv", 72.0, 50.0),
            _metric("order_count", 12.0, 10.0),
            _metric("avg_order_value", 6.0, 5.0),
        ],
        "scope": ReportScope(scope_type="platform"),
    })
    brief = _BRIEF.model_copy(update={"focus_metrics": ["avg_order_value"]})

    # avg_order_value intentionally has no additive dimension breakdowns.
    assert legal_query_surface(report, brief) == []
    # No report-pinned grouped facts are present: attribution must record a
    # data gap rather than read mutable facts during insight generation.
    contributions, decompositions, evidence, driver, summary = query_attribution_evidence_agent(report, brief)

    assert contributions == []
    assert driver == "workflow_fallback"
    assert summary is not None
    assert summary.stop_reason.code == "NO_HIGH_VALUE_ACTION"
    assert summary.actions_used == 1
    assert summary.action_results[0].action_type == "DECOMPOSE_METRIC"
    assert summary.action_results[0].status == "completed"
    assert summary.action_results[0].evidence_ids == decompositions[0].evidence_ids
    assert len(decompositions) == 1
    assert decompositions[0].metric_name == "avg_order_value"
    assert decompositions[0].total_delta == 1.0
    assert evidence[decompositions[0].evidence_ids[0]]["method_id"] == "symmetric_ratio_v1"


def test_report_investigation_ledger_includes_snapshot_decomposition_and_query_actions():
    report = _report().model_copy(update={
        "metrics": [
            _metric("gmv", 72.0, 50.0),
            _metric("order_count", 12.0, 10.0),
            _metric("avg_order_value", 6.0, 5.0),
        ],
        "scope": ReportScope(scope_type="platform"),
    })
    brief = _BRIEF.model_copy(update={"focus_metrics": ["gmv"]})
    expected_trace_id = "insight_trace_test_001"

    with patch("app.agents.insight_agent._llm_plan", lambda *_: None), \
         patch("app.agents.insight_agent._platform_dimension_finding", _fake_finding):
        contributions, decompositions, evidence, driver, summary = query_attribution_evidence_agent(
            report, brief, trace_id=expected_trace_id,
        )

    assert driver == "workflow_fallback"
    assert contributions
    assert len(decompositions) == 1
    assert summary is not None
    actions = summary.action_results
    assert any(item.action_type == "DECOMPOSE_METRIC" and item.status == "completed" for item in actions)
    assert any(item.action_type == "BREAKDOWN_DIMENSION" and item.status == "completed" for item in actions)
    decomposition_result = next(item for item in actions if item.action_type == "DECOMPOSE_METRIC")
    assert decomposition_result.evidence_ids == decompositions[0].evidence_ids
    assert all(evidence_id in evidence for item in actions for evidence_id in item.evidence_ids)
    typed_evidence = [
        record["investigation_evidence"]
        for record in evidence.values()
        if isinstance(record.get("investigation_evidence"), dict)
    ]
    assert typed_evidence
    assert {item["trace_id"] for item in typed_evidence} == {expected_trace_id}
    assert summary.actions_used <= 6


def test_insight_artifact_entrypoint_records_agent_run():
    from app.analysis.insights import generate_insight_artifact

    report = _report()
    captured = {}

    def capture_generation(_brief, _selected_ids, _kind_map, context):
        captured.update(context)
        return []

    with patch("app.llm.client.llm_agent_action", lambda *a, **k: _plan_for([("gmv", "product_category")])), \
         patch("app.agents.insight_agent._seller_dimension_finding", _fake_finding), \
         patch("app.analysis.insights._llm_generate", capture_generation):
        artifact = generate_insight_artifact(report, 1, _BRIEF, "tester")
    assert artifact.agent_run is not None
    assert artifact.agent_run.domain == "insight"
    assert artifact.investigation_summary is not None
    assert artifact.investigation_summary["actions_used"] >= 1
    assert artifact.investigation_summary["stop_reason"]["code"]
    assert artifact.agent_run.plan["investigation_summary"] == artifact.investigation_summary
    assert "stop=" in artifact.agent_run.trace[0].summary
    assert captured["investigation_summary"] == artifact.investigation_summary

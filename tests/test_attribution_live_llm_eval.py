"""Opt-in live-provider smoke test for the governed attribution selector.

This test sends synthetic-only prompts to the configured model. It is skipped
by default; enable explicitly with RUN_ATTRIBUTION_LLM_EVAL=1.
"""

from __future__ import annotations

import os
from datetime import date

import pytest

from app.agents.diagnostic_contracts import (
    CallToolAction,
    DiagnosticComparison,
    DiagnosticPeriod,
    DiagnosticPlan,
    DiagnosticStep,
)
from app.investigation.contracts import (
    AnalyzeTrendAction,
    BreakdownDimensionAction,
    CandidateAction,
    CandidateFeatures,
    CandidateSet,
    InvestigationState,
)
from app.investigation.engine import ensure_root_branch


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_ATTRIBUTION_LLM_EVAL") != "1",
    reason="set RUN_ATTRIBUTION_LLM_EVAL=1 to call the configured LLM with synthetic evaluation data",
)


def test_live_llm_selects_supported_candidate_for_open_investigation(monkeypatch) -> None:
    """A significant first-level factor must not silently end open drill-down."""
    from app.graph.nodes import diagnostic

    start, end = date(2026, 2, 1), date(2026, 2, 28)
    baseline_start, baseline_end = date(2026, 1, 1), date(2026, 1, 31)
    plan = DiagnosticPlan(
        goal="用受支持的数据动作调查 GMV 变化",
        hypotheses=["GMV 下降可能集中于某些商品品类"],
        steps=[DiagnosticStep(
            step_id="evaluate-open-branch",
            objective="继续检查尚未解释的 GMV 变化",
            allowed_actions=["CALL_TOOL", "FINISH", "ABSTAIN"],
            completion_condition="已有证据充分解释变化或没有有价值的合法动作",
            kind="dimension_contribution",
            required=True,
        )],
        stop_conditions=["预算耗尽或不存在高价值合法动作时停止"],
        budget={"max_rounds": 6, "max_tool_calls": 6},
    )
    state_dict = {
        "user_message": "合成评测：2026-02 GMV 相比 2026-01 下降，继续调查未解释的部分。",
        "diagnostic_target_metric": "gmv",
        "diagnostic_round": 1,
        "diagnostic_plan": plan.model_dump(mode="json"),
        "root_cause_evidence": [
            {
                "purpose": "trend_baseline", "metric_id": "gmv",
                "result": {"trend_result": {"points": [
                    {"metric_date": "2026-01-31", "value": 100.0},
                    {"metric_date": "2026-02-28", "value": 80.0},
                ]}},
            },
            {
                "purpose": "cross_metric:order_count", "metric_id": "order_count",
                "result": {"trend_result": {"points": [
                    {"metric_date": "2026-01-31", "value": 100.0},
                    {"metric_date": "2026-02-28", "value": 80.0},
                ]}},
            },
        ],
    }
    state = diagnostic._ensure_graph_investigation_state(state_dict)
    assert isinstance(state, InvestigationState)
    state.remaining_budget = 3
    target = state.target
    start, end = target.current_start, target.current_end
    baseline_start, baseline_end = target.baseline_start, target.baseline_end
    assert baseline_start is not None and baseline_end is not None
    root = ensure_root_branch(state)
    state.branches = [root.model_copy(update={
        "root_attributed_effect_value": -20.0,
        "root_effect_share": 1.0,
        "effect_share_of_root": 1.0,
        "effect_value": -20.0,
        "status": "investigating",
    })]
    state.current_focus_branch_id = root.branch_id

    selected_candidate_id = "eval-candidate-gmv-category"
    unrelated_candidate_id = "eval-candidate-gmv-trend"
    candidates = [
        CandidateAction(
            candidate_id=selected_candidate_id,
            action=BreakdownDimensionAction(
                action_type="BREAKDOWN_DIMENSION",
                metric_id="gmv",
                dimension="product_category",
                method_id="additive_change_v1",
            ),
            branch_id=root.branch_id,
            features=CandidateFeatures(
                relevance_score=0.95, unresolved_target_share=1.0,
                data_availability="available", evidence_gap="explain gmv change by category",
                rule_score=0.95,
            ),
        ),
        CandidateAction(
            candidate_id=unrelated_candidate_id,
            action=AnalyzeTrendAction(action_type="ANALYZE_TREND", metric_id="gmv"),
            features=CandidateFeatures(
                relevance_score=0.2, data_availability="available",
                evidence_gap="repeat already collected target trend", already_explored=True,
                rule_score=0.2,
            ),
        ),
    ]
    state.candidate_actions = candidates
    state.candidate_set = CandidateSet(
        candidate_set_id="eval-candidate-set-live-01",
        investigation_state_version=state.state_version,
        scope_fingerprint=target.scope_fingerprint,
        candidate_ids=[item.candidate_id for item in candidates],
    )
    state_dict["investigation_state"] = state.model_dump(mode="json")

    candidate_catalog = {
        selected_candidate_id: CallToolAction(
            action_type="CALL_TOOL",
            tool_name="compare_breakdown_by_dimension",
            metric_name="gmv",
            dimension="product_category",
            comparison=DiagnosticComparison(
                current_start=start.isoformat(), current_end=end.isoformat(),
                baseline_start=baseline_start.isoformat(), baseline_end=baseline_end.isoformat(),
                granularity="month",
            ),
        ),
        unrelated_candidate_id: CallToolAction(
            action_type="CALL_TOOL",
            tool_name="query_metric_trend",
            metric_name="gmv",
            period=DiagnosticPeriod(
                start_date=start.isoformat(), end_date=end.isoformat(), granularity="month",
            ),
        ),
    }
    monkeypatch.setattr(diagnostic, "_candidate_tool_catalog", lambda _state: candidate_catalog)

    raw_model_actions: list[dict | None] = []
    from app.llm import client as llm_client

    live_call = llm_client.llm_agent_action

    def capture_live_call(*args, **kwargs):
        raw = live_call(*args, **kwargs)
        raw_model_actions.append(raw)
        return raw

    monkeypatch.setattr(llm_client, "llm_agent_action", capture_live_call)
    selected = diagnostic._llm_decide_action(state_dict)

    assert raw_model_actions, "the configured provider did not return a model response"
    assert getattr(selected, "action_type", None) in {"CALL_TOOL", "CALL_CANDIDATE"}
    assert state_dict.get("diagnostic_selected_candidate_id") == selected_candidate_id
    assert state_dict.get("diagnostic_decision_source") == "agent"
    assert not state_dict.get("diagnostic_llm_failures"), state_dict.get("diagnostic_llm_failures")

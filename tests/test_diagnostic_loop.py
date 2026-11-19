from unittest.mock import patch
from datetime import date
from types import SimpleNamespace
import pytest

from app.agents.diagnostic_contracts import (
    CallCandidateAction, DiagnosticFactQuery, DiagnosticPeriod, DiagnosticPlan,
    ParallelFactsAction,
)
from app.graph.nodes.diagnostic import (
    _audit_soft_finish, _candidate_tool_catalog, _findings_block, _in_scope_comparison_windows, _plan_for_query,
    _related_metrics,
    _required_action_for, diagnostic_finalize, diagnostic_plan, observe, policy_execute,
)
from app.agents.diagnostic_contracts import DiagnosticStep
from app.investigation.contracts import (
    ActionResult, AnalysisTarget, AttributionEvidence, CalculationProgress,
    CandidateAction, CandidateFeatures, CandidateSet, DataValidationResult,
    InvestigationState, UnresolvedBranch,
)
from app.graph.nodes.critic import _review_diagnostic_trajectory
from app.tenancy import TenantContext, activate_tenant_context


def test_related_metrics_uses_explicit_scope_for_shared_metric_ids(monkeypatch):
    monkeypatch.setattr(
        "app.graph.nodes.diagnostic.get_settings",
        lambda: SimpleNamespace(diagnostic_related_metrics_json="{}"),
    )

    assert _related_metrics("gmv", scope_name="platform") == ["order_count", "avg_order_value"]
    assert _related_metrics("gmv", scope_name="seller") == []


@pytest.fixture
def authorized_tenant_context():
    context = TenantContext(
        tenant_id="test-tenant", user_id="test-user", role="admin", scope_type="platform",
    )
    with activate_tenant_context(context):
        yield context


def test_provider_failure_builds_only_registry_supported_seller_freight_steps(
    monkeypatch, authorized_tenant_context,
):
    monkeypatch.setattr("app.llm.client.llm_agent_action", lambda *_a, **_k: None)
    state = {
        "user_message": "本月平均每单运费有什么变化？",
        "diagnostic_target_metric": "seller_avg_freight_per_order",
        "tenant_id": authorized_tenant_context.tenant_id,
        "user_id": authorized_tenant_context.user_id,
        "user_role": authorized_tenant_context.role,
        "controlled_analysis_scope": {
            "seller_id": "seller-1",
            "default_start_date": "2025-02-01",
            "default_end_date": "2025-02-28",
            "granularity": "day",
            "allowed_metrics": ["seller_avg_freight_per_order"],
            "allowed_dimensions": [],
        },
        "runtime_budget": {"max_graph_steps": 32, "max_tool_calls": 8},
        "node_history": [],
    }

    result = diagnostic_plan(state)
    plan = DiagnosticPlan.model_validate(result["diagnostic_plan"])
    assert result["diagnostic_fallback"] is True
    assert result["decision_driver"] == "workflow_fallback"
    assert "seller_avg_freight_per_order" in plan.goal
    assert all("GMV" not in str(step.objective) for step in plan.steps)
    assert {step.kind for step in plan.steps} == {"trend"}
    assert all(step.required for step in plan.steps)


def _state() -> dict:
    plan = DiagnosticPlan.model_validate({
        "goal": "test", "hypotheses": ["h"],
        "steps": [{"step_id": "s1", "objective": "observe", "allowed_actions": ["CALL_TOOL", "FINISH"], "completion_condition": "evidence"}],
        "stop_conditions": ["done"], "budget": {"max_rounds": 3, "max_tool_calls": 2},
    })
    return {
        "user_message": "为什么 2018-08-20 的 GMV 下降了？",
        "diagnostic_plan": plan.model_dump(mode="json"),
        "diagnostic_step_state": {"s1": "pending"},
        "tool_calls": [], "evidence_bundles": [], "root_cause_evidence": [],
        "observations": [], "node_history": [], "errors": [],
        "runtime_budget": {"max_tool_calls": 2},
    }


def _typed_evidence_snapshot(
    evidence_id: str, *, source_action_id: str, scope_fingerprint: str,
    metric_id: str = "gmv",
) -> dict:
    return AttributionEvidence(
        evidence_id=evidence_id, evidence_type="trend", metric_id=metric_id,
        source_action_id=source_action_id, source_ref="test-query",
        scope_fingerprint=scope_fingerprint, method_id="observed_trend_v1",
        method_version="1.0", authorized_metric_provenance="test-fixture",
    ).model_dump(mode="json")


@pytest.mark.parametrize("requested_stop", ["FINISH", "ABSTAIN"])
def test_chat_terminal_action_rechecks_evidence_and_continues_with_server_candidate(requested_stop):
    target = AnalysisTarget(
        metric_id="gmv", current_start=date(2018, 8, 1), current_end=date(2018, 8, 31),
        baseline_start=date(2018, 7, 1), baseline_end=date(2018, 7, 31),
        canonical_scope={"tenant_id": "tenant", "seller_ids": ["seller"]},
        scope_fingerprint="scope-fingerprint", requested_analysis_type="diagnostic",
    )
    evidence = AttributionEvidence(
        evidence_id="prior-evidence", evidence_type="comparison", metric_id="gmv",
        source_action_id="prior-candidate", source_ref="fixture", scope_fingerprint="scope-fingerprint",
        method_id="compare_period_v1", method_version="1.0", authorized_metric_provenance="exact",
    )
    branch = UnresolvedBranch(
        branch_id="open-branch", parent_evidence_id=evidence.evidence_id, metric_id="gmv",
        dimension="product_category", dimension_value="furniture", effect_value=-10,
        effect_share_of_root=0.7, depth=1,
    )
    candidate = CandidateAction(
        candidate_id="legal-next-step",
        action={"action_type": "COMPARE_PERIOD", "metric_id": "gmv"},
        branch_id=branch.branch_id, features=CandidateFeatures(rule_score=0.8),
    )
    investigation = InvestigationState(
        target=target, remaining_budget=2, evidence=[evidence], branches=[branch],
        action_results=[ActionResult(
            action_id="prior-action", candidate_id="prior-candidate", status="completed",
            evidence_ids=[evidence.evidence_id],
        )], candidate_actions=[candidate],
        candidate_set=CandidateSet(
            candidate_set_id="set_current", investigation_state_version=0,
            scope_fingerprint=target.scope_fingerprint, candidate_ids=[candidate.candidate_id],
        ),
    )

    replacement = _audit_soft_finish(
        investigation, {candidate.candidate_id: object()}, requested_stop=requested_stop,
    )

    assert isinstance(replacement, CallCandidateAction)
    assert replacement.candidate_id == candidate.candidate_id
    assert replacement.candidate_set_id == "set_current"
    assert investigation.current_focus_branch_id == branch.branch_id
    assert investigation.observations[-1]["event"] == "soft_stop_rejected"
    assert investigation.observations[-1]["requested_stop"] == requested_stop


def test_chat_soft_finish_abstains_when_audit_fails_without_legal_candidate():
    target = AnalysisTarget(
        metric_id="gmv", current_start=date(2018, 8, 1), current_end=date(2018, 8, 31),
        baseline_start=date(2018, 7, 1), baseline_end=date(2018, 7, 31),
        canonical_scope={"tenant_id": "tenant", "seller_ids": ["seller"]},
        scope_fingerprint="scope-fingerprint", requested_analysis_type="diagnostic",
    )
    investigation = InvestigationState(target=target)

    replacement = _audit_soft_finish(investigation, {})

    assert replacement.action_type == "ABSTAIN"
    assert investigation.stop_reason.code == "DATA_INSUFFICIENT"


def test_policy_denies_unregistered_typed_tool_without_calling_mcp():
    state = _state()
    state["diagnostic_action"] = {
        "action_type": "CALL_TOOL", "tool_name": "not_registered", "metric_name": "gmv",
        "period": {"start_date": "2018-08-15", "end_date": "2018-08-20", "granularity": "day"},
    }
    with patch("app.graph.nodes.diagnostic.query_business_data") as gateway:
        result = policy_execute(state)
    assert result["policy_decision"]["decision"] == "deny"
    gateway.assert_not_called()


def test_required_diagnostic_queries_share_the_six_action_hard_budget():
    state = _state()
    state["diagnostic_required_action"] = True
    state["diagnostic_investigation_action_count"] = 6
    state["diagnostic_action"] = {
        "action_type": "CALL_TOOL", "tool_name": "query_metric_trend", "metric_name": "gmv",
        "period": {"start_date": "2018-08-14", "end_date": "2018-08-20", "granularity": "day"},
    }
    with patch("app.graph.nodes.diagnostic.query_business_data") as gateway:
        result = policy_execute(state)
    assert result["policy_decision"]["decision"] == "deny"
    assert "budget" in result["policy_decision"]["reason"]
    gateway.assert_not_called()


def test_observe_completes_step_and_records_dimension_evidence(authorized_tenant_context):
    state = _state()
    state["diagnostic_target_metric"] = "gmv"
    state["controlled_analysis_scope"] = {
        "seller_id": "seller-1", "allowed_metrics": ["gmv"],
        "allowed_dimensions": ["product_category"],
        "default_start_date": "2018-02-01", "default_end_date": "2018-02-28",
        "granularity": "day",
    }
    state["diagnostic_action"] = {
        "action_type": "CALL_TOOL", "tool_name": "compare_breakdown_by_dimension", "metric_name": "gmv",
        "dimension": "product_category",
        "comparison": {
            "current_start": "2018-02-01", "current_end": "2018-02-28",
            "baseline_start": "2018-01-01", "baseline_end": "2018-01-31",
            "granularity": "month",
        },
    }
    state["diagnostic_tool_results"] = [{
        "server": "mock", "tool_name": "compare_breakdown_by_dimension", "arguments": {},
        "result": {"status": "ok", "contribution_result": {
            "status": "ok", "current_total": 120.0, "baseline_total": 100.0,
            "total_delta": 20.0,
            "current_items": [{"dimension_value": "A", "value": 70.0}, {"dimension_value": "B", "value": 50.0}],
            "baseline_items": [{"dimension_value": "A", "value": 60.0}, {"dimension_value": "B", "value": 40.0}],
        }},
    }]
    result = observe(state)
    assert result["diagnostic_step_state"] == {"s1": "completed"}
    assert result["root_cause_evidence"][0]["purpose"] == "dimension_contribution:product_category"
    assert result["root_cause_evidence"][0]["investigation_evidence"]["authorized_metric_provenance"] == "tool_gateway_policy_validated"
    assert result["evidence_bundles"][0]["trusted"] is True
    assert result["evidence_bundles"][0]["validation_status"] == "accepted_by_investigation_engine"
    assert result["investigation_state"]["action_results"][0]["status"] == "completed"


def test_chat_data_gap_is_candidate_scoped_and_allows_next_agent_round(authorized_tenant_context, monkeypatch):
    from app.agents.diagnostic_contracts import CallToolAction, DiagnosticComparison
    from app.graph.nodes.diagnostic import _record_graph_investigation_results, diagnostic_decide

    state = _state()
    plan = DiagnosticPlan.model_validate({
        "goal": "test dimension investigation",
        "hypotheses": ["one available dimension can explain the change"],
        "steps": [{
            "step_id": "dimension-required", "objective": "dimension contribution",
            "allowed_actions": ["CALL_TOOL", "ABSTAIN", "FINISH"],
            "completion_condition": "dimension evidence", "kind": "dimension_contribution",
            "required": True,
        }],
        "stop_conditions": ["budget exhausted"],
        "budget": {"max_rounds": 3, "max_tool_calls": 6},
    })
    state.update({
        "diagnostic_plan": plan.model_dump(mode="json"),
        "diagnostic_step_state": {"dimension-required": "pending"},
        "diagnostic_target_metric": "gmv",
        "diagnostic_required_action": True,
        "runtime_budget": {"max_tool_calls": 8},
        "diagnostic_investigation_action_count": 1,
        "controlled_analysis_scope": {
            "seller_id": "seller-1", "allowed_metrics": ["gmv"],
            "allowed_dimensions": ["product_category", "month_position"],
            "default_start_date": "2018-01-01", "default_end_date": "2018-02-28",
            "granularity": "month",
        },
    })
    query = CallToolAction(
        action_type="CALL_TOOL", tool_name="compare_breakdown_by_dimension",
        metric_name="gmv", dimension="product_category",
        comparison=DiagnosticComparison(
            current_start="2018-02-01", current_end="2018-02-28",
            baseline_start="2018-01-01", baseline_end="2018-01-31",
            granularity="month",
        ),
    )
    typed = AttributionEvidence(
        evidence_id="empty-dimension-evidence", evidence_type="dimension_contribution",
        metric_id="gmv", values={"attribution_status": "ok"}, source_action_id="raw-tool-result",
        source_ref=query.tool_name, scope_fingerprint="pre-validation",
        method_id="additive_change_v1", method_version="1.0",
        authorized_metric_provenance="tool_gateway_policy_validated",
    )

    _record_graph_investigation_results(
        state, query, [(query, {"result": {"status": "empty"}}, typed)],
    )

    assert state["errors"] == []
    assert state["diagnostic_step_state"]["dimension-required"] == "data_insufficient"
    assert state["diagnostic_candidate_data_gaps"][0]["dimension"] == "product_category"
    assert state["investigation_state"]["stop_reason"] is None
    assert state["investigation_state"]["action_results"][0]["failure_code"] == "DATA_INSUFFICIENT"

    alternative = CallToolAction(
        action_type="CALL_TOOL", tool_name="query_metric_trend", metric_name="gmv",
        period={"start_date": "2018-01-01", "end_date": "2018-02-28", "granularity": "day"},
    )
    monkeypatch.setattr("app.graph.nodes.diagnostic._llm_decide_action", lambda _state: alternative)
    next_state = diagnostic_decide(state)
    assert next_state["diagnostic_action"]["tool_name"] == "query_metric_trend"
    assert next_state["investigation_state"]["stop_reason"] is None


def test_chat_exhausted_alternatives_report_data_insufficient(monkeypatch):
    from app.agents.diagnostic_contracts import AbstainAction
    from app.graph.nodes.diagnostic import _ensure_graph_investigation_state, diagnostic_decide
    from app.investigation.contracts import StopReason

    state = _state()
    state["diagnostic_target_metric"] = "gmv"
    state["diagnostic_plan"] = DiagnosticPlan.model_validate({
        "goal": "test", "hypotheses": ["h"],
        "steps": [{
            "step_id": "dimension-required", "objective": "dimension contribution",
            "allowed_actions": ["CALL_TOOL", "ABSTAIN"],
            "completion_condition": "dimension evidence", "kind": "dimension_contribution",
            "required": True,
        }],
        "stop_conditions": ["no alternatives"],
        "budget": {"max_rounds": 3, "max_tool_calls": 6},
    }).model_dump(mode="json")
    state["diagnostic_step_state"] = {"dimension-required": "data_insufficient"}
    investigation = _ensure_graph_investigation_state(state)
    assert investigation is not None
    investigation.stop_reason = StopReason(
        code="NO_HIGH_VALUE_ACTION", detail="No more candidates.", initiator="backend",
    )
    state["investigation_state"] = investigation.model_dump(mode="json")
    state["diagnostic_candidate_data_gaps"] = [{"detail": "No rows for product category."}]
    monkeypatch.setattr(
        "app.graph.nodes.diagnostic._llm_decide_action",
        lambda _state: AbstainAction(action_type="ABSTAIN", reason="insufficient_evidence"),
    )

    result = diagnostic_decide(state)

    assert result["diagnostic_action"]["action_type"] == "ABSTAIN"
    assert result["investigation_state"]["stop_reason"]["code"] == "DATA_INSUFFICIENT"


def test_observe_does_not_publish_unreconciled_breakdown_as_trusted_evidence(authorized_tenant_context):
    state = _state()
    state["diagnostic_target_metric"] = "gmv"
    state["controlled_analysis_scope"] = {
        "seller_id": "seller-1", "allowed_metrics": ["gmv"],
        "allowed_dimensions": ["product_category"],
        "default_start_date": "2018-02-01", "default_end_date": "2018-02-28",
        "granularity": "day",
    }
    state["diagnostic_action"] = {
        "action_type": "CALL_TOOL", "tool_name": "compare_breakdown_by_dimension", "metric_name": "gmv",
        "dimension": "product_category",
        "comparison": {
            "current_start": "2018-02-01", "current_end": "2018-02-28",
            "baseline_start": "2018-01-01", "baseline_end": "2018-01-31",
            "granularity": "month",
        },
    }
    state["diagnostic_tool_results"] = [{
        "server": "mock", "tool_name": "compare_breakdown_by_dimension", "arguments": {},
        "result": {"status": "ok", "contribution_result": {
            "status": "ok", "current_total": 120.0, "baseline_total": 100.0,
            "total_delta": 25.0,
            # The returned groups sum to only 10; the missing 10 is not
            # represented as an explicit, classified coverage residual.
            "current_items": [{"dimension_value": "A", "value": 70.0}],
            "baseline_items": [{"dimension_value": "A", "value": 60.0}],
        }},
    }]

    result = observe(state)

    assert result["root_cause_evidence"] == []
    assert result["evidence_bundles"][0]["trusted"] is False
    assert result["evidence_bundles"][0]["validation_status"] == "pending_typed_evidence_validation"
    assert result["diagnostic_step_state"] == {"s1": "pending"}
    assert result["investigation_state"]["evidence"] == []
    assert result["investigation_state"]["action_results"][0]["status"] == "failed"
    assert result["investigation_state"]["action_results"][0]["failure_code"] == "INVALID_DATA"
    assert result["investigation_state"]["stop_reason"]["code"] == "INVALID_DATA"
    assert result["observations"][-1]["evidence_ids"] == []
    assert result["observations"][-1]["payload"]["rejected_evidence_count"] == 1


def test_replan_cannot_modify_observed_step():
    state = _state()
    state["diagnostic_step_state"] = {"s1": "completed"}
    state["diagnostic_action"] = {
        "action_type": "REPLAN", "stop_conditions": ["done"],
        "steps": [{"step_id": "s1", "objective": "changed", "allowed_actions": ["CALL_TOOL"], "completion_condition": "evidence"}],
    }
    result = policy_execute(state)
    assert result["policy_decision"]["decision"] == "deny"
    assert "modify" in result["policy_decision"]["reason"]


def test_finish_must_reference_existing_evidence():
    state = _state()
    state["diagnostic_action"] = {"action_type": "FINISH", "conclusion": "supported", "evidence_ids": ["missing"]}
    result = diagnostic_finalize(state)
    assert result["analysis_result"]["abstain"] is True
    assert result["termination_initiator"] == "policy"


def test_invalid_terminal_evidence_overrides_soft_no_action_stop_reason():
    state = _state()
    state["evidence_bundles"] = [{"evidence_id": "ev-unverified", "trusted": True}]
    state["investigation_state"] = {
        "target": {"metric_id": "gmv", "scope_fingerprint": "scope-v1"},
        "stop_reason": {
            "code": "NO_HIGH_VALUE_ACTION", "detail": "No remaining candidate.",
            "initiator": "backend",
        },
        "evidence": [],
        "action_results": [],
    }
    state["diagnostic_action"] = {
        "action_type": "FINISH", "conclusion": "Unverified conclusion.",
        "evidence_ids": ["ev-unverified"],
    }

    result = diagnostic_finalize(state)

    assert result["analysis_result"]["abstain"] is True
    assert result["investigation_state"]["stop_reason"]["code"] == "INVALID_DATA"


@pytest.mark.parametrize(
    ("reason", "required_step_status", "expected_code"),
    [
        ("budget_exhausted", None, "BUDGET_EXHAUSTED"),
        ("insufficient_evidence", "data_insufficient", "DATA_INSUFFICIENT"),
        ("ambiguous_question", None, "NO_HIGH_VALUE_ACTION"),
    ],
)
def test_terminal_abstention_is_written_to_investigation_ledger(
    monkeypatch, reason, required_step_status, expected_code,
):
    emitted_events = []
    monkeypatch.setattr("app.core.execution_context.enqueue_sse_event", emitted_events.append)
    state = _state()
    state["diagnostic_investigation_action_count"] = 1
    state["diagnostic_action"] = {
        "action_type": "ABSTAIN", "reason": reason, "supporting_evidence_ids": [],
    }
    state["investigation_state"] = {
        "stop_reason": None,
        "progress": {},
        "calculation_progress": {},
        "remaining_budget": 4,
        "action_results": [],
        "branches": [],
    }
    if required_step_status:
        state["diagnostic_plan"]["steps"][0]["required"] = True
        state["diagnostic_step_state"]["s1"] = required_step_status

    result = diagnostic_finalize(state)

    stop_reason = result["investigation_state"]["stop_reason"]
    assert stop_reason["code"] == expected_code
    assert result["analysis_result"]["investigation_summary"]["stop_reason"] == stop_reason
    assert emitted_events[-1].stop_reason == expected_code


def test_successful_finish_populates_missing_investigation_terminal_reason(monkeypatch):
    emitted_events = []
    monkeypatch.setattr("app.core.execution_context.enqueue_sse_event", emitted_events.append)
    state = _state()
    state["diagnostic_investigation_action_count"] = 1
    state["evidence_bundles"] = [{"evidence_id": "ev-supported", "trusted": True}]
    state["investigation_state"] = {
        "target": {"metric_id": "gmv", "scope_fingerprint": "scope-v1"},
        "evidence": [_typed_evidence_snapshot(
            "ev-supported", source_action_id="candidate-1", scope_fingerprint="scope-v1",
        )],
        "action_results": [{
            "action_id": "action-1", "candidate_id": "candidate-1", "metric_id": "gmv",
            "status": "completed", "evidence_ids": ["ev-supported"],
            "validation_result": {"status": "valid"},
        }],
    }
    state["diagnostic_action"] = {
        "action_type": "FINISH", "conclusion": "Evidence-backed conclusion.",
        "evidence_ids": ["ev-supported"],
    }

    result = diagnostic_finalize(state)

    assert result["analysis_result"]["abstain"] is False
    assert result["investigation_state"]["stop_reason"]["code"] == "ENOUGH_EVIDENCE"
    assert result["analysis_result"]["investigation_summary"]["stop_reason"]["code"] == "ENOUGH_EVIDENCE"
    assert emitted_events[-1].stop_reason == "ENOUGH_EVIDENCE"


def test_finish_cannot_reference_untrusted_raw_tool_bundle():
    state = _state()
    state["evidence_bundles"] = [{
        "evidence_id": "ev-raw", "trusted": False,
        "validation_status": "pending_typed_evidence_validation",
    }]
    state["diagnostic_action"] = {
        "action_type": "FINISH", "conclusion": "unsupported conclusion",
        "evidence_ids": ["ev-raw"],
    }

    result = diagnostic_finalize(state)

    assert result["analysis_result"]["abstain"] is True
    assert result["termination_initiator"] == "policy"
    assert result["errors"][-1]["code"] == "diagnostic_evidence_reference_invalid"


def test_finish_cannot_rely_on_incomplete_investigation_evidence():
    state = _state()
    state["evidence_bundles"] = [{"evidence_id": "ev-partial", "trusted": True}]
    state["investigation_state"] = {
        "target": {"scope_fingerprint": "scope-v1"},
        "evidence": [_typed_evidence_snapshot(
            "ev-partial", source_action_id="candidate-partial", scope_fingerprint="scope-v1",
        )],
        "action_results": [{
            "action_id": "action-partial",
            "candidate_id": "candidate-partial",
            "metric_id": "gmv",
            "status": "completed",
            "evidence_ids": ["ev-partial"],
            "validation_result": {"status": "incomplete_result"},
        }],
    }
    state["diagnostic_action"] = {
        "action_type": "FINISH",
        "conclusion": "The metric definitely declined because of this factor.",
        "evidence_ids": ["ev-partial"],
    }

    result = diagnostic_finalize(state)

    assert result["analysis_result"]["abstain"] is True
    assert result["finish_decision"]["action_type"] == "ABSTAIN"
    assert result["diagnostic_guardrails"][-1]["code"] == "finish_cites_untrusted_or_unvalidated_evidence"
    assert any("完整性校验" in item
               for item in result["analysis_result"]["limitations"])


def test_final_answer_discloses_partial_evidence_even_when_not_cited():
    state = _state()
    state["evidence_bundles"] = [
        {"evidence_id": "ev-partial", "trusted": True},
        {"evidence_id": "ev-valid", "trusted": True},
    ]
    state["investigation_state"] = {
        "target": {"scope_fingerprint": "scope-v1"},
        "evidence": [
            _typed_evidence_snapshot(
                "ev-partial", source_action_id="candidate-partial", scope_fingerprint="scope-v1",
            ),
            _typed_evidence_snapshot(
                "ev-valid", source_action_id="candidate-valid", scope_fingerprint="scope-v1",
            ),
        ],
        "action_results": [
            {
                "action_id": "action-partial",
                "candidate_id": "candidate-partial",
                "metric_id": "gmv",
                "status": "completed",
                "evidence_ids": ["ev-partial"],
                "validation_result": {"status": "incomplete_result"},
            },
            {
                "action_id": "action-valid",
                "candidate_id": "candidate-valid",
                "metric_id": "gmv",
                "status": "completed",
                "evidence_ids": ["ev-valid"],
                "validation_result": {"status": "valid"},
            },
        ],
    }
    state["diagnostic_action"] = {
        "action_type": "FINISH",
        "conclusion": "This conclusion uses the validated evidence only.",
        "evidence_ids": ["ev-valid"],
    }

    result = diagnostic_finalize(state)

    assert result["analysis_result"]["abstain"] is False
    assert any("完整性校验" in item
               for item in result["analysis_result"]["limitations"])


@pytest.mark.parametrize("mismatch", ["action", "scope", "missing_evidence"])
def test_finish_rejects_evidence_without_valid_action_scope_provenance(mismatch):
    state = _state()
    state["evidence_bundles"] = [{"evidence_id": "ev-cited", "trusted": True}]
    evidence_rows = []
    if mismatch != "missing_evidence":
        evidence_rows.append(_typed_evidence_snapshot(
            "ev-cited",
            source_action_id="other-candidate" if mismatch == "action" else "candidate-1",
            scope_fingerprint="other-scope" if mismatch == "scope" else "scope-v1",
        ))
    state["investigation_state"] = {
        "target": {"scope_fingerprint": "scope-v1"},
        "evidence": evidence_rows,
        "action_results": [{
            "action_id": "action-1", "candidate_id": "candidate-1", "metric_id": "gmv",
            "status": "completed", "evidence_ids": ["ev-cited"],
            "validation_result": {"status": "valid"},
        }],
    }
    state["diagnostic_action"] = {
        "action_type": "FINISH", "conclusion": "Supported claim.",
        "evidence_ids": ["ev-cited"],
    }

    result = diagnostic_finalize(state)

    assert result["analysis_result"]["abstain"] is True
    assert result["finish_decision"]["action_type"] == "ABSTAIN"
    assert result["diagnostic_guardrails"][-1]["code"] == "finish_cites_untrusted_or_unvalidated_evidence"


def test_finish_rejects_investigation_whose_root_metric_differs_from_request():
    state = _state()
    state["diagnostic_target_metric"] = "order_count"
    state["evidence_bundles"] = [{"evidence_id": "ev-gmv", "trusted": True}]
    state["investigation_state"] = {
        "target": {"metric_id": "gmv", "scope_fingerprint": "scope-v1"},
        "evidence": [_typed_evidence_snapshot(
            "ev-gmv", source_action_id="candidate-gmv", scope_fingerprint="scope-v1",
        )],
        "action_results": [{
            "action_id": "action-gmv", "candidate_id": "candidate-gmv", "metric_id": "gmv",
            "status": "completed", "evidence_ids": ["ev-gmv"],
            "validation_result": {"status": "valid"},
        }],
    }
    state["diagnostic_action"] = {
        "action_type": "FINISH", "conclusion": "Orders changed due to GMV.",
        "evidence_ids": ["ev-gmv"],
    }

    result = diagnostic_finalize(state)

    assert result["analysis_result"]["abstain"] is True
    assert any(
        item.get("code") == "investigation_root_target_mismatch"
        for item in result["diagnostic_guardrails"]
    )
    assert any("根指标与本轮请求不一致" in item
               for item in result["analysis_result"]["limitations"])


def test_final_answer_discloses_unresolved_branches_and_terminal_reason():
    from app.investigation.contracts import AnalysisTarget, InvestigationState, StopReason, UnresolvedBranch
    from app.investigation.evidence import scope_fingerprint

    state = _state()
    scope = {
        "tenant_id": "t1",
        "default_start_date": "2026-01-01",
        "default_end_date": "2026-02-28",
    }
    state.update({"diagnostic_target_metric": "gmv", "controlled_analysis_scope": scope})
    fingerprint = scope_fingerprint(scope)
    state["diagnostic_action"] = {
        "action_type": "FINISH", "conclusion": "GMV 下降主要由订单量变化贡献。",
        "evidence_ids": ["ev-supported"],
    }
    state["evidence_bundles"] = [{"evidence_id": "ev-supported", "trusted": True}]
    investigation = InvestigationState(
        target=AnalysisTarget(
            metric_id="gmv", current_start="2026-02-01", current_end="2026-02-28",
            baseline_start="2026-01-01", baseline_end="2026-01-31",
            canonical_scope=scope, scope_fingerprint=fingerprint,
            requested_analysis_type="diagnostic",
        ),
        evidence=[AttributionEvidence(
            evidence_id="ev-supported", evidence_type="metric_decomposition", metric_id="gmv",
            source_action_id="candidate-1", source_ref="fixture", scope_fingerprint=fingerprint,
            method_id="product_shapley_v1", method_version="1.0",
            authorized_metric_provenance="test-fixture",
        )],
        branches=[UnresolvedBranch(
            branch_id="branch-other", parent_evidence_id="ev-supported", metric_id="gmv",
            dimension="product_category", dimension_value="Other", effect_value=-20,
            effect_share_of_root=0.2, depth=1, status="blocked",
            label="商品类别=Other（聚合覆盖，无法安全下钻）",
        )],
        action_results=[ActionResult(
            action_id="action-1", candidate_id="candidate-1", action_type="DECOMPOSE_METRIC",
            metric_id="gmv", status="completed", evidence_ids=["ev-supported"],
            calculation_progress=CalculationProgress(
                action_id="action-1", reconciliation_ratio=0.98,
                residual_value=2.0, residual_type="rounding", residual_explainable=True,
                data_quality_status="valid",
            ),
            validation_result=DataValidationResult(status="valid"),
        )],
        calculation_progress=CalculationProgress(
            action_id="action-1", reconciliation_ratio=0.98,
            residual_value=2.0, residual_type="rounding", residual_explainable=True,
            data_quality_status="valid",
        ),
        stop_reason=StopReason(
            code="NO_HIGH_VALUE_ACTION", detail="覆盖残差没有可安全执行的下钻候选。",
            initiator="backend",
        ),
    )
    state["investigation_state"] = investigation.model_dump(mode="json")

    result = diagnostic_finalize(state)

    summary = result["analysis_result"]["investigation_summary"]
    assert summary["stop_reason"]["code"] == "NO_HIGH_VALUE_ACTION"
    assert summary["unresolved_branches"][0]["status"] == "blocked"
    assert summary["calculation_progress"]["reconciliation_ratio"] == 0.98
    assert summary["action_results"][0]["validation_result"]["status"] == "valid"
    limitations = "\n".join(result["analysis_result"]["limitations"])
    assert "没有剩余的高价值合法调查动作" in limitations
    assert "商品类别=Other" in limitations
    assert "尚未完成核查" not in limitations  # blocked coverage is labeled as a data boundary


def test_finish_cannot_use_target_metric_as_root_cause():
    state = _state()
    state["diagnostic_target_metric"] = "gmv"
    state["evidence_bundles"] = [{"evidence_id": "ev1"}]
    state["diagnostic_action"] = {
        "action_type": "FINISH",
        "conclusion": "GMV is the cause of the GMV decline.",
        "evidence_ids": ["ev1"],
        "found_factor": "gmv",
    }
    result = diagnostic_finalize(state)
    assert result["analysis_result"]["abstain"] is True
    assert result["analysis_result"]["stop_reason"] == "insufficient_evidence"
    assert result["diagnostic_guardrails"][0]["code"] == "target_metric_cannot_be_root_cause"


def test_capability_gap_detail_is_visible_in_evidence_summary_and_final_limitations():
    state = _state()
    state["diagnostic_capability_gaps"] = [{
        "kind": "period_attribution",
        "detail": "当前范围内没有两个完整的比较周期。",
    }]
    summary = _findings_block(state)
    assert "当前范围内没有两个完整的比较周期" in summary

    state["evidence_bundles"] = [{"evidence_id": "ev-supported"}]
    state["diagnostic_action"] = {
        "action_type": "FINISH", "conclusion": "仅能确认趋势。",
        "evidence_ids": ["ev-supported"],
    }
    result = diagnostic_finalize(state)
    assert any("period_attribution" in item and "两个完整的比较周期" in item
               for item in result["analysis_result"]["limitations"])


def test_metric_decomposition_evidence_is_available_to_follow_up_planner():
    state = _state()
    state["root_cause_evidence"] = [{
        "purpose": "metric_decomposition:gmv",
        "metric_id": "gmv",
        "investigation_evidence": {
            "method_id": "product_shapley_v1",
            "values": {
                "baseline_value": 100,
                "current_value": 150,
                "total_delta": 50,
                "display_method": "shapley",
                "factor_metrics": ["order_count", "avg_order_value"],
                "display_components": [
                    {"component_id": "order_count", "value": 35},
                    {"component_id": "avg_order_value", "value": 15},
                ],
                "raw_components": [
                    {"component_id": "interaction", "component_type": "interaction_effect", "value": 10},
                ],
            },
        },
    }]

    summary = _findings_block(state)

    assert "gmv/product_shapley_v1" in summary
    assert "100" in summary and "150" in summary and "+50" in summary
    assert "order_count=+35" in summary and "+70.0%" in summary
    assert "avg_order_value=+15" in summary and "+30.0%" in summary
    assert "+10" in summary
    assert "shapley" in summary
    assert "不代表因果关系" in summary


def test_metric_decomposition_summary_rejects_unrepresentable_numeric_values():
    state = _state()
    state["root_cause_evidence"] = [{
        "purpose": "metric_decomposition:gmv",
        "metric_id": "gmv",
        "investigation_evidence": {
            "method_id": "product_shapley_v1",
            "values": {
                "baseline_value": 10**10000,
                "current_value": 20,
                "total_delta": 10,
                "display_components": [{"component_id": "order_count", "value": 10}],
            },
        },
    }]

    summary = _findings_block(state)

    assert "gmv" in summary
    assert "Evidence" in summary
    assert "10000" not in summary


def test_scoped_diagnostic_query_uses_scoped_gateway(authorized_tenant_context):
    state = _state()
    state["controlled_analysis_scope"] = {
        "allowed_metrics": ["gmv"], "seller_id": "seller-1",
        "default_start_date": "2018-08-14", "default_end_date": "2018-08-20",
        "granularity": "day",
    }
    state["diagnostic_action"] = {
        "action_type": "CALL_TOOL", "tool_name": "query_metric_trend", "metric_name": "gmv",
        "period": {"start_date": "2018-08-14", "end_date": "2018-08-20", "granularity": "day"},
    }
    with patch("app.graph.nodes.diagnostic.query_scoped_business_data", return_value={"server": "scoped", "tool_name": "query_scoped_trend", "arguments": {}, "result": {"status": "ok"}}) as scoped, patch("app.graph.nodes.diagnostic.query_business_data") as normal:
        policy_execute(state)
    scoped.assert_called_once()
    normal.assert_not_called()


def test_required_dimension_attribution_uses_latest_complete_periods_not_arbitrary_halves(authorized_tenant_context):
    state = _state()
    state["user_message"] = "解释 GMV 变化"
    state["diagnostic_target_metric"] = "gmv"
    state["controlled_analysis_scope"] = {
        "allowed_metrics": ["gmv"], "seller_id": "seller-1",
        "allowed_dimensions": ["product_category", "month_position"],
        "default_start_date": "2026-01-01", "default_end_date": "2026-02-28",
    }
    step = DiagnosticStep(
        step_id="dimension", objective="拆解维度贡献", kind="dimension_contribution",
        allowed_actions=["CALL_TOOL"], completion_condition="有贡献证据", required=True,
    )
    action = _required_action_for(state, step)
    assert action is not None
    # The two full natural months are selected; the system must not split
    # arbitrary date halves and label them as monthly periods.
    assert action.comparison.current_start == "2026-02-01"
    assert action.comparison.current_end == "2026-02-28"
    assert action.comparison.baseline_start == "2026-01-01"
    assert action.comparison.baseline_end == "2026-01-31"
    assert action.comparison.granularity == "month"


def test_diagnostic_plan_reuses_only_authorized_metric_and_carries_safe_drill_filter(authorized_tenant_context):
    from app.agents.diagnostic_contracts import CallToolAction, DiagnosticComparison
    from app.investigation.evidence import build_attribution_evidence, scope_fingerprint

    scope = {
        "seller_id": "seller-1", "default_start_date": "2018-01-01",
        "default_end_date": "2018-02-28", "allowed_metrics": ["gmv"],
        "allowed_dimensions": ["product_category", "month_position"],
    }
    query = CallToolAction(
        action_type="CALL_TOOL", tool_name="compare_breakdown_by_dimension",
        metric_name="gmv", dimension="month_position",
        comparison=DiagnosticComparison(
            current_start="2018-01-30", current_end="2018-02-28",
            baseline_start="2018-01-01", baseline_end="2018-01-29",
            granularity="month", filters={"product_category": "furniture"},
            parent_evidence_id="ev-parent",
        ),
    )
    authorized = {
        "metric_id": "gmv", "provenance_type": "exact", "provenance_id": "mention-1",
        "scope_fingerprint": "scope-fp", "query_scope_fingerprint": "scope-fp",
        "catalog_digest": "catalog-v1",
    }
    plan = _plan_for_query(query, {"authorized_metrics": [authorized]})
    assert plan.authorized_metric.metric_id == "gmv"
    assert plan.filters == {"product_category": "furniture"}

    typed = build_attribution_evidence(
        evidence_id="ev-parent", evidence_type="dimension_contribution", metric_id="gmv",
        claim="category attribution", values={
            "total_delta": -100,
            "components": [{"dimension_value": "furniture", "delta_value": -70}],
        }, source_action_id="root-action", source_ref="mcp",
        scope=scope, method_id="additive_change_v1", method_version="1.0",
        authorized_metric_provenance="exact",
        comparison={"comparison": {"filters": {}}},
    )
    state = _state()
    state.update({
        "diagnostic_target_metric": "gmv", "controlled_analysis_scope": scope,
        "root_cause_evidence": [{"purpose": "dimension_contribution:product_category",
                                 "investigation_evidence": typed.model_dump(mode="json")}],
    })
    catalog = _candidate_tool_catalog(state)
    drill = [
        item for item in catalog.values()
        if item.dimension == "month_position" and item.comparison
        and item.comparison.filters == {"product_category": "furniture"}
    ]
    assert len(drill) == 1
    assert drill[0].comparison.parent_evidence_id == "ev-parent"


def test_chat_observation_uses_shared_investigation_transition_and_preserves_branch(authorized_tenant_context):
    from app.agents.diagnostic_contracts import CallToolAction, DiagnosticComparison
    from app.graph.nodes.diagnostic import (
        _candidate_tool_catalog, _ensure_graph_investigation_state,
        _record_graph_investigation_results,
    )
    from app.investigation.evidence import build_attribution_evidence

    scope = {
        "seller_id": "seller-1", "default_start_date": "2018-01-01",
        "default_end_date": "2018-02-28", "allowed_metrics": ["gmv"],
        "allowed_dimensions": ["product_category", "month_position"],
    }
    state = {
        "user_message": "分析 GMV 变化",
        "diagnostic_target_metric": "gmv",
        "controlled_analysis_scope": scope,
        "diagnostic_investigation_action_count": 0,
        "tool_calls": [], "node_history": [],
    }
    investigation = _ensure_graph_investigation_state(state)
    assert investigation is not None
    query = CallToolAction(
        action_type="CALL_TOOL", tool_name="compare_breakdown_by_dimension",
        metric_name="gmv", dimension="product_category",
        comparison=DiagnosticComparison(
            current_start="2018-01-30", current_end="2018-02-28",
            baseline_start="2018-01-01", baseline_end="2018-01-29",
            granularity="month",
        ),
    )
    result = {"result": {"status": "ok", "contribution_result": {}}}
    typed = build_attribution_evidence(
        evidence_id="ev-chat-shared", evidence_type="dimension_contribution", metric_id="gmv",
        claim="category breakdown", values={
            "total_delta": -100,
            "components": [
                {"dimension_value": "furniture", "delta_value": -70},
                {"dimension_value": "Other", "delta_value": -30},
            ],
            "residual_value": 0,
            "residual_type": "unknown",
            "residual_explainable": True,
            "attribution_status": "ok",
            "query": {"dimension": "product_category"},
        }, source_action_id="action-1", source_ref="mcp", scope=scope,
        method_id="additive_change_v1", method_version="1.0",
        authorized_metric_provenance="verified_scope",
    )
    state["diagnostic_investigation_action_count"] = 1
    _record_graph_investigation_results(state, query, [(query, result, typed)])

    persisted = state["investigation_state"]
    assert persisted["remaining_budget"] == 5
    assert [item["evidence_id"] for item in persisted["evidence"]] == ["ev-chat-shared"]
    assert len(persisted["action_results"]) == 1
    assert persisted["evidence"][0]["source_action_id"] == persisted["action_results"][0]["candidate_id"]
    root = next(item for item in persisted["branches"] if item["branch_type"] == "root_target")
    furniture = next(item for item in persisted["branches"] if item["dimension_value"] == "furniture")
    assert furniture["parent_branch_id"] == root["branch_id"]
    assert furniture["parent_evidence_id"] == "ev-chat-shared"
    child = [
        candidate for candidate in _candidate_tool_catalog(state).values()
        if candidate.dimension == "month_position"
    ]
    assert child
    assert child[0].comparison.filters == {"product_category": "furniture"}


def test_registry_timeout_keeps_structured_cause_through_diagnostic_action_and_stop(authorized_tenant_context):
    from app.agents.diagnostic_contracts import CallToolAction, DiagnosticPeriod
    from app.graph.nodes.diagnostic import (
        _ensure_graph_investigation_state, _record_graph_investigation_results,
        _record_result,
    )

    state = _state()
    state.update({
        "trace_id": "diag-timeout-run",
        "tenant_id": authorized_tenant_context.tenant_id,
        "user_id": authorized_tenant_context.user_id,
        "user_role": authorized_tenant_context.role,
        "diagnostic_target_metric": "gmv",
        "diagnostic_investigation_action_count": 0,
    })
    query = CallToolAction(
        action_type="CALL_TOOL", tool_name="query_metric_trend", metric_name="gmv",
        period=DiagnosticPeriod(
            start_date="2018-08-01", end_date="2018-08-07", granularity="day",
        ),
    )
    investigation = _ensure_graph_investigation_state(state)
    assert investigation is not None
    failure_context = {
        "category": "tool_timeout", "service": "metric_registry",
        "code": "tool_timeout", "cause_type": "TimeoutError",
        "request_id": "diag-timeout-run:1", "retryable": None,
    }
    result = {
        "server": "metric-registry", "tool_name": "registry_trend",
        "arguments": {"metric_name": "gmv"}, "request_id": "diag-timeout-run:1",
        "result": {
            "status": "execution_failed", "reason": "数据工具查询超时，未取得该项结果。",
            "failure_context": failure_context,
        },
    }
    evidence = AttributionEvidence.model_validate(_typed_evidence_snapshot(
        "unusable-timeout-evidence",
        source_action_id="selected-candidate-will-be-bound-by-engine",
        scope_fingerprint=investigation.target.scope_fingerprint,
    )
    )

    _record_result(state, query, result)
    _record_graph_investigation_results(state, query, [(query, result, evidence)])

    assert state["tool_calls"][0]["status"] == "failed"
    assert state["tool_calls"][0]["request_id"] == "diag-timeout-run:1"
    persisted = state["investigation_state"]
    assert persisted["stop_reason"]["code"] == "EXECUTION_FAILED"
    assert persisted["stop_reason"]["failure_context"] == failure_context
    action_result = persisted["action_results"][-1]
    assert action_result["failure_code"] == "EXECUTION_FAILED"
    assert action_result["failure_context"] == failure_context
    assert action_result["evidence_ids"] == []
    assert state["errors"][-1]["request_id"] == "diag-timeout-run:1"
    assert state["errors"][-1]["failure_context"] == failure_context


@pytest.mark.parametrize("failure_kind,category", [
    ("timeout", "tool_timeout"), ("database", "database_error"),
    ("permission", "authorization_denied"),
])
def test_diagnostic_gateway_exception_preserves_typed_cause_and_request(
    monkeypatch, authorized_tenant_context, failure_kind, category,
):
    from app.graph.nodes import diagnostic as node
    from app.resilience import ExternalServiceError
    from sqlalchemy.exc import OperationalError

    failures = {
        "timeout": ExternalServiceError("mcp", "timeout", "private_query_secret"),
        "database": OperationalError("SELECT private_query_secret", {}, RuntimeError("secret")),
        "permission": PermissionError("private_query_secret"),
    }
    calls = []

    def failed_gateway(*args, **kwargs):
        calls.append(kwargs)
        raise failures[failure_kind]

    monkeypatch.setattr(node, "query_business_data", failed_gateway)
    state = _state()
    state.update({
        "trace_id": "direct-diagnostic-fault",
        "tenant_id": authorized_tenant_context.tenant_id,
        "user_id": authorized_tenant_context.user_id,
        "user_role": authorized_tenant_context.role,
        "diagnostic_target_metric": "gmv",
    })
    from app.agents.diagnostic_contracts import CallToolAction
    candidate_id, action = next(
        (candidate_id, action)
        for candidate_id, action in node._candidate_tool_catalog(state).items()
        if isinstance(action, CallToolAction)
    )
    state["diagnostic_selected_candidate_id"] = candidate_id
    state["diagnostic_action"] = action.model_dump(mode="json")
    executed = node.policy_execute(state)
    assert len(calls) == 1, (executed.get("policy_decision"), executed.get("diagnostic_tool_results"))
    result = executed["diagnostic_tool_results"][0]
    context = result["result"]["failure_context"]
    assert context["category"] == category
    assert context["request_id"] == calls[0]["request_id"] == result["request_id"]
    assert context["request_id"].startswith("direct-diagnostic-fault:")
    assert "private_query_secret" not in str(result)
    assert result["arguments"] == node._query_arguments(action)
    observed = node.observe(executed)
    assert observed["investigation_state"]["stop_reason"]["failure_context"] == context
    assert observed["investigation_state"]["action_results"][-1]["failure_context"] == context
    assert observed["tool_calls"][-1]["status"] == ("unsupported" if failure_kind == "permission" else "failed")
    assert observed["tool_calls"][-1]["request_id"] == context["request_id"]


def test_parallel_diagnostic_execution_stops_after_first_terminal_db_failure(
    monkeypatch, authorized_tenant_context,
):
    from sqlalchemy.exc import OperationalError
    calls = []

    def failed_gateway(*args, **kwargs):
        calls.append(kwargs)
        raise OperationalError("SELECT protected_data", {}, RuntimeError("driver detail"))

    monkeypatch.setattr("app.graph.nodes.diagnostic.query_business_data", failed_gateway)
    state = _state()
    state.update({
        "trace_id": "parallel-db-stop",
        "tenant_id": authorized_tenant_context.tenant_id,
        "user_id": authorized_tenant_context.user_id,
        "user_role": authorized_tenant_context.role,
        "diagnostic_target_metric": "gmv",
        "diagnostic_required_action": True,
        "runtime_budget": {"max_tool_calls": 8},
        "diagnostic_action": ParallelFactsAction(
            action_type="PARALLEL_FACTS",
            facts=[
                DiagnosticFactQuery(
                    tool_name="query_metric_trend", metric_name=metric,
                    period=DiagnosticPeriod(
                        start_date="2018-08-19", end_date="2018-08-20", granularity="day",
                    ),
                )
                for metric in ("gmv", "order_count", "avg_order_value")
            ],
        ).model_dump(mode="json"),
    })
    executed = policy_execute(state)

    assert len(calls) == 1, executed.get("policy_decision")
    assert len(executed["diagnostic_tool_results"]) == 1
    observed = observe(executed)
    investigation = observed["investigation_state"]
    assert investigation["stop_reason"]["code"] == "EXECUTION_FAILED"
    assert investigation["action_results"][-1]["failure_code"] == "EXECUTION_FAILED"

    finalized = diagnostic_finalize(observed)
    assert finalized["diagnostic_action"]["action_type"] == "ABSTAIN"
    assert finalized["analysis_result"]["abstain"] is True
    assert finalized["analysis_result"]["investigation_summary"]["stop_reason"]["code"] == "EXECUTION_FAILED"
    assert not any(
        error.get("code") == "diagnostic_evidence_reference_invalid"
        for error in finalized["errors"]
    )


def test_parallel_diagnostic_execution_stops_after_unsupported_capability_result(
    monkeypatch, authorized_tenant_context,
):
    from app.agents.diagnostic_contracts import DiagnosticFactQuery, DiagnosticPeriod, ParallelFactsAction
    from app.graph.nodes import diagnostic as node

    calls = []

    def unsupported_query(query, _state):
        calls.append(query)
        return {
            "status": "unsupported",
            "tool_name": query.tool_name,
            "arguments": node._query_arguments(query),
            "result": {"status": "unsupported", "reason_code": "metric_not_registered"},
        }

    monkeypatch.setattr(node, "_execute_query", unsupported_query)
    state = _state()
    state.update({
        "trace_id": "parallel-unsupported-stop",
        "tenant_id": authorized_tenant_context.tenant_id,
        "user_id": authorized_tenant_context.user_id,
        "user_role": authorized_tenant_context.role,
        "diagnostic_target_metric": "gmv",
        "diagnostic_required_action": True,
        "runtime_budget": {"max_tool_calls": 8},
        "diagnostic_action": ParallelFactsAction(
            action_type="PARALLEL_FACTS",
            facts=[
                DiagnosticFactQuery(
                    tool_name="query_metric_trend", metric_name=metric,
                    period=DiagnosticPeriod(
                        start_date="2018-08-19", end_date="2018-08-20", granularity="day",
                    ),
                )
                for metric in ("gmv", "order_count", "avg_order_value")
            ],
        ).model_dump(mode="json"),
    })

    executed = node.policy_execute(state)

    assert len(calls) == 1
    assert len(executed["diagnostic_tool_results"]) == 1
    assert executed["policy_decision"]["decision"] == "allow"
    assert "unsupported result" in executed["policy_decision"]["reason"]
    observed = node.observe(executed)
    assert observed["investigation_state"]["stop_reason"]["code"] == "CAPABILITY_GAP"


def test_chat_contribution_observation_uses_shared_additive_method_and_explicit_other():
    from app.agents.diagnostic_contracts import CallToolAction, DiagnosticComparison
    from app.graph.nodes.diagnostic import _typed_contribution_values

    query = CallToolAction(
        action_type="CALL_TOOL", tool_name="compare_breakdown_by_dimension",
        metric_name="gmv", dimension="product_category",
        comparison=DiagnosticComparison(
            current_start="2026-02-01", current_end="2026-02-28",
            baseline_start="2026-01-01", baseline_end="2026-01-31",
            granularity="month",
        ),
    )
    payload = {
        "status": "ok", "current_total": 100.0, "baseline_total": 80.0, "total_delta": 20.0,
        "current_items": [{"dimension_value": "A", "value": 60.0}],
        "baseline_items": [{"dimension_value": "A", "value": 50.0}],
    }
    result = {"result": {"status": "ok", "contribution_result": payload}}
    values = _typed_contribution_values(result, query)
    assert values["attribution_status"] == "ok"
    assert values["method_id"] == "additive_change_v1"
    assert values["baseline_value"] == 80.0
    assert values["current_value"] == 100.0
    assert {item["dimension_value"]: item["delta_value"] for item in values["components"]} == {
        "A": 10.0, "Other": 10.0,
    }
    assert values["explained_share"] == 1.0


def test_diagnostic_agent_candidate_catalog_is_registry_and_scope_bounded(authorized_tenant_context):
    state = _state()
    state["controlled_analysis_scope"] = {
        "allowed_metrics": ["gmv"], "allowed_dimensions": ["product_category"],
        "seller_id": "seller-1", "default_start_date": "2018-08-15",
        "default_end_date": "2018-08-20", "granularity": "day",
    }
    candidates = _candidate_tool_catalog(state)
    assert candidates
    assert all(query.metric_name == "gmv" for query in candidates.values())
    assert all(query.dimension in {None, "product_category"} for query in candidates.values())
    assert all(query.period is None or query.period.start_date == "2018-08-15" for query in candidates.values())
    from app.investigation.contracts import InvestigationState

    investigation = InvestigationState.model_validate(state["investigation_state"])
    assert investigation.candidate_set is not None
    assert investigation.candidate_set.scope_fingerprint == investigation.target.scope_fingerprint
    assert investigation.candidate_set.authorization_fingerprint
    assert set(investigation.candidate_set.candidate_ids) == set(candidates)


def test_stale_agent_candidate_decision_uses_current_server_fallback_candidate(
    authorized_tenant_context, monkeypatch,
):
    from app.graph.nodes.diagnostic import _llm_decide_action

    state = _state()
    state["diagnostic_target_metric"] = "gmv"
    state["controlled_analysis_scope"] = {
        "allowed_metrics": ["gmv"], "allowed_dimensions": ["product_category"],
        "seller_id": "seller-1", "default_start_date": "2018-08-15",
        "default_end_date": "2018-08-20", "granularity": "day",
    }
    catalog = _candidate_tool_catalog(state)
    assert catalog
    investigation = InvestigationState.model_validate(state["investigation_state"])
    candidate_id = next(iter(catalog))
    assert investigation.candidate_set is not None

    monkeypatch.setattr(
        "app.llm.client.llm_agent_action",
        lambda *_args, **_kwargs: {
            "action_type": "CALL_CANDIDATE",
            "candidate_set_id": "set-from-previous-turn",
            "candidate_id": candidate_id,
        },
    )
    action = _llm_decide_action(state)
    assert action == catalog[state["diagnostic_selected_candidate_id"]]
    assert state["diagnostic_decision_source"] == "workflow_fallback"
    assert state["diagnostic_fallback_reason"]
    refreshed = InvestigationState.model_validate(state["investigation_state"])
    assert refreshed.state_version == investigation.state_version + 1
    assert refreshed.observations[-1]["event"] == "workflow_fallback_selected_candidate"
    assert refreshed.observations[-1]["candidate_set_id"] == refreshed.candidate_set.candidate_set_id
    assert refreshed.stop_reason is None
    assert state["diagnostic_llm_failures"][-1]["recovered"] is False

    state["diagnostic_action"] = action.model_dump(mode="json")
    monkeypatch.setattr(
        "app.graph.nodes.diagnostic._execute_query",
        lambda *_args, **_kwargs: {"status": "ok", "result": {"status": "ok", "points": []}},
    )
    executed = policy_execute(state)
    assert executed["policy_decision"]["decision"] == "allow"
    assert executed["diagnostic_tool_results"]


def test_diagnostic_agent_rejects_raw_tool_query_even_when_it_matches_a_candidate(
    authorized_tenant_context, monkeypatch,
):
    from app.graph.nodes.diagnostic import _llm_decide_action

    state = _state()
    state["diagnostic_target_metric"] = "gmv"
    state["controlled_analysis_scope"] = {
        "allowed_metrics": ["gmv"], "allowed_dimensions": ["product_category"],
        "seller_id": "seller-1", "default_start_date": "2018-08-15",
        "default_end_date": "2018-08-20", "granularity": "day",
    }
    catalog = _candidate_tool_catalog(state)
    assert catalog
    raw_query = next(iter(catalog.values())).model_dump(mode="json")
    calls = []

    def return_raw_query(*_args, **_kwargs):
        calls.append(True)
        return raw_query

    monkeypatch.setattr("app.llm.client.llm_agent_action", return_raw_query)

    action = _llm_decide_action(state)

    assert len(calls) == 2  # one bounded schema-repair attempt, then server-side selection
    assert state["diagnostic_decision_source"] == "workflow_fallback"
    assert state["diagnostic_llm_failures"][-1]["recovered"] is False
    assert "server candidate by ID" in state["diagnostic_llm_failures"][-1]["message"]
    assert state["diagnostic_fallback_reason"]
    assert action == catalog[state["diagnostic_selected_candidate_id"]]


def test_diagnostic_agent_receives_server_candidate_features_and_open_branch_context(
    authorized_tenant_context, monkeypatch,
):
    from app.graph.nodes.diagnostic import _llm_decide_action

    state = _state()
    state["diagnostic_target_metric"] = "gmv"
    state["controlled_analysis_scope"] = {
        "allowed_metrics": ["gmv"], "allowed_dimensions": ["product_category"],
        "seller_id": "seller-1", "default_start_date": "2018-08-15",
        "default_end_date": "2018-08-20", "granularity": "day",
    }
    catalog = _candidate_tool_catalog(state)
    assert catalog
    investigation = InvestigationState.model_validate(state["investigation_state"])
    assert investigation.candidate_set is not None
    investigation.branches = [
        UnresolvedBranch(
            branch_id=f"open-product-category-branch-{index}", parent_evidence_id="ev-parent",
            metric_id="gmv", dimension="product_category", dimension_value=f"Category-{index}",
            effect_value=-35 + index, effect_share_of_root=1.0 - index * 0.05, depth=1,
            inherited_filters={"product_category": f"Category-{index}"}, status="open",
            label=f"Category-{index} has an unresolved change",
        )
        for index in range(10)
    ]
    investigation.current_focus_branch_id = "open-product-category-branch-0"
    state["investigation_state"] = investigation.model_dump(mode="json")
    catalog = _candidate_tool_catalog(state)
    assert catalog
    investigation = InvestigationState.model_validate(state["investigation_state"])
    candidate_id = next(iter(catalog))
    prompts = []

    def select_current_candidate(prompt, *_args, **_kwargs):
        prompts.append(prompt)
        return {
            "action_type": "CALL_CANDIDATE",
            "candidate_set_id": investigation.candidate_set.candidate_set_id,
            "candidate_id": candidate_id,
        }

    monkeypatch.setattr("app.llm.client.llm_agent_action", select_current_candidate)
    action = _llm_decide_action(state)

    assert action.action_type == "CALL_TOOL"
    assert prompts
    prompt = prompts[0]
    assert "selection_context" in prompt
    assert "target_branch_share" in prompt
    assert "backend_priority_score" in prompt
    assert "open_branches" in prompt
    assert "open-product-category-branch-0" in prompt
    assert "open-product-category-branch-7" in prompt
    assert "open-product-category-branch-8" not in prompt
    assert "open-product-category-branch-9" not in prompt
    assert "effect_share_of_root" in prompt
    assert "omitted_open_branch_count" in prompt
    assert "remaining_action_budget" in prompt
    assert "not measured effects or predicted results" in prompt


def test_ratio_attribution_candidate_uses_registered_facts_tool_and_dimension(
    authorized_tenant_context, monkeypatch,
):
    monkeypatch.setattr(
        "app.graph.nodes.diagnostic._dates",
        lambda _question: ("2018-07-01", "2018-08-31"),
    )
    state = {
        "user_message": "分析 freight_rate 变化",
        "diagnostic_target_metric": "freight_rate",
        "diagnostic_investigation_action_count": 0,
        "tool_calls": [],
    }
    candidates = _candidate_tool_catalog(state)
    ratio = [
        item for item in candidates.values()
        if item.tool_name == "compare_freight_rate_ratio_facts_by_dimension_period"
    ]
    assert len(ratio) == 1
    assert ratio[0].metric_name == "freight_rate"
    assert ratio[0].dimension == "month_position"
    assert ratio[0].comparison.granularity == "month"


def test_average_order_value_candidate_uses_only_registered_facts_tool_and_dimension(
    authorized_tenant_context, monkeypatch,
):
    monkeypatch.setattr(
        "app.graph.nodes.diagnostic._dates",
        lambda _question: ("2018-07-01", "2018-08-31"),
    )
    state = {
        "user_message": "分析 avg_order_value 变化",
        "diagnostic_target_metric": "avg_order_value",
        "diagnostic_investigation_action_count": 0,
        "tool_calls": [],
    }

    candidates = _candidate_tool_catalog(state)
    ratio = [
        item for item in candidates.values()
        if item.tool_name == "compare_avg_order_value_ratio_facts_by_dimension_period"
    ]

    assert len(ratio) == 1
    assert ratio[0].metric_name == "avg_order_value"
    assert ratio[0].dimension == "month_position"
    assert ratio[0].comparison.granularity == "month"
    assert all(
        item.dimension != "product_category"
        for item in candidates.values()
        if item.metric_name == "avg_order_value"
    )


def test_factor_branch_without_its_own_authorized_metric_fails_closed(authorized_tenant_context):
    from app.investigation.contracts import InvestigationState, UnresolvedBranch
    from app.graph.nodes.diagnostic import _ensure_graph_investigation_state

    state = _state()
    state["diagnostic_target_metric"] = "gmv"
    state["controlled_analysis_scope"] = {
        "seller_id": "seller-1", "allowed_metrics": ["gmv", "order_count", "avg_order_value"],
        "allowed_dimensions": ["product_category"],
        "default_start_date": "2018-07-01", "default_end_date": "2018-08-31",
        "granularity": "month",
    }
    state["user_message"] = "分析 GMV 变化"
    investigation = _ensure_graph_investigation_state(state)
    assert investigation is not None
    branch = UnresolvedBranch(
        branch_id="orders-factor", parent_evidence_id="ev-gmv-decomposition",
        metric_id="order_count", dimension="__metric_factor__",
        dimension_value="order_count", effect_value=-80, effect_share_of_root=0.8,
        depth=0, inherited_filters={}, label="订单量分解贡献",
    )
    investigation.branches = [branch]
    investigation.current_focus_branch_id = branch.branch_id
    state["investigation_state"] = investigation.model_dump(mode="json")

    candidates = _candidate_tool_catalog(state)
    assert all(query.metric_name != "order_count" for query in candidates.values())
    persisted = state["investigation_state"]
    assert persisted["branches"][0]["status"] == "blocked"
    assert any(
        item["candidate_id"] == branch.branch_id
        and "AuthorizedMetric" in item["reason"]
        for item in persisted["policy_rejections"]
    )

    # Once the metric resolver supplies a real child-metric credential, the
    # corresponding query can be exposed and its QueryPlan keeps that proof.
    from app.metrics.authorization import authorize_metric, validate_authorized_metric

    state["authorized_metrics"] = [authorize_metric(
        "order_count", provenance_type="exact", provenance_id="mention-orders",
        scope="seller", scope_data=state["controlled_analysis_scope"],
    ).model_dump(mode="json")]
    investigation = _ensure_graph_investigation_state(state)
    investigation.branches = [branch.model_copy(update={"status": "open"})]
    investigation.current_focus_branch_id = branch.branch_id
    state["investigation_state"] = investigation.model_dump(mode="json")
    candidates = _candidate_tool_catalog(state)
    child_query = next(
        query for query in candidates.values()
        if query.metric_name == "order_count" and query.dimension == "product_category"
    )
    from app.graph.nodes.diagnostic import _plan_for_query
    plan = _plan_for_query(child_query, state)
    assert plan.authorized_metric is not None
    assert plan.authorized_metric.metric_id == "order_count"
    validate_authorized_metric(
        plan.authorized_metric, scope="seller", scope_data=state["controlled_analysis_scope"],
    )


def test_distribution_candidate_maps_to_authorized_statistics_query(authorized_tenant_context):
    state = _state()
    state["controlled_analysis_scope"] = {
        "allowed_metrics": ["gmv"], "allowed_dimensions": ["product_category"],
        "seller_id": "seller-1", "default_start_date": "2018-08-01",
        "default_end_date": "2018-08-31", "granularity": "month",
    }
    candidates = _candidate_tool_catalog(state)
    query = next(item for item in candidates.values() if item.tool_name == "query_metric_statistics")
    assert query.metric_name == "gmv"
    assert query.period.granularity == "month"
    assert query.statistics == ["count", "mean", "std", "min", "q1", "median", "q3", "max"]
    plan = _plan_for_query(query)
    assert plan.query_type == "metric_statistics"
    assert plan.statistics == query.statistics


def test_series_actions_are_candidates_only_when_registry_query_capability_allows_them(authorized_tenant_context):
    state = _state()
    state["controlled_analysis_scope"] = {
        "allowed_metrics": ["gmv"], "allowed_dimensions": ["product_category"],
        "seller_id": "seller-1", "default_start_date": "2018-08-01",
        "default_end_date": "2018-08-31", "granularity": "day",
    }
    candidates = _candidate_tool_catalog(state)
    catalog = state["investigation_state"]["candidate_actions"]
    action_by_id = {item["candidate_id"]: item["action"] for item in catalog}
    actions = {
        action_by_id[next(key for key, value in candidates.items() if value is query)]["action_type"]
        for query in candidates.values()
    }
    assert {"ANALYZE_DISTRIBUTION", "CHECK_ANOMALY", "CHECK_DATA_QUALITY"} <= actions
    assert all(
        query.tool_name in {"query_metric_trend", "query_metric_statistics", "compare_breakdown_by_dimension"}
        for query in candidates.values()
    )


def test_chat_candidate_catalog_keeps_unquantified_root_branch_investigable(authorized_tenant_context):
    from app.graph.nodes.diagnostic import _ensure_graph_investigation_state

    state = _state()
    state.update({
        "diagnostic_target_metric": "gmv",
        "controlled_analysis_scope": {
            "seller_id": "seller-1", "default_start_date": "2018-01-01",
            "default_end_date": "2018-02-28", "granularity": "month",
            "allowed_metrics": ["gmv"],
            "allowed_dimensions": ["product_category", "month_position"],
        },
    })
    investigation = _ensure_graph_investigation_state(state)
    assert investigation is not None
    root = UnresolvedBranch(
        branch_id="root-gmv", metric_id="gmv", depth=0,
        parent_evidence_id=None, effect_value=None, effect_share_of_root=None,
    )
    investigation.branches = [root]
    investigation.current_focus_branch_id = root.branch_id
    state["investigation_state"] = investigation.model_dump(mode="json")

    catalog = _candidate_tool_catalog(state)
    persisted = InvestigationState.model_validate(state["investigation_state"])
    root_candidates = [
        item for item in persisted.candidate_actions
        if item.branch_id == root.branch_id
    ]

    assert catalog
    assert root_candidates
    assert any(item.action.action_type == "DRILL_DOWN" for item in root_candidates)
    assert all(item.features.unresolved_target_share == 0.0 for item in root_candidates)


def test_chat_decomposition_is_not_offered_without_two_disjoint_periods(monkeypatch):
    from datetime import date

    from app.graph.nodes.diagnostic import _available_chat_decompositions

    state = _state()
    state["diagnostic_target_metric"] = "gmv"
    # The default seven-day request cannot form a supported before/after
    # comparison. Complete series alone must not make attribution executable.
    monkeypatch.setattr(
        "app.graph.nodes.diagnostic._trusted_daily_series",
        lambda _state, _metric: {date(2025, 3, day): float(day) for day in range(1, 32)},
    )
    assert _available_chat_decompositions(state, "gmv", "platform") == {}


def test_chat_product_decomposition_uses_authorized_series_and_shared_evidence(
    authorized_tenant_context, monkeypatch,
):
    from datetime import date, timedelta

    from app.agents.diagnostic_contracts import ExecuteDecompositionAction
    from app.graph.nodes.diagnostic import _available_chat_decompositions, _candidate_tool_catalog
    from app.investigation.evidence import build_attribution_evidence

    baseline_start, baseline_end = date(2025, 1, 1), date(2025, 1, 31)
    current_start, current_end = date(2025, 2, 1), date(2025, 2, 28)
    first, last = baseline_start, current_end
    monkeypatch.setattr(
        "app.graph.nodes.diagnostic._in_scope_comparison_windows",
        lambda *_args, **_kwargs: (
            baseline_start, baseline_end, current_start, current_end, "month",
        ),
    )
    monkeypatch.setattr(
        "app.graph.nodes.diagnostic._dates",
        lambda _question: (first.isoformat(), last.isoformat()),
    )

    state = _state()
    state["diagnostic_target_metric"] = "gmv"
    values: dict[str, dict[date, float]] = {"gmv": {}, "order_count": {}, "avg_order_value": {}}
    day = baseline_start
    while day <= current_end:
        if day <= baseline_end:
            values["gmv"][day] = 1000.0 / 31
            values["order_count"][day] = 100.0 / 31
            values["avg_order_value"][day] = 10.0
        else:
            values["gmv"][day] = 1600.0 / 28
            values["order_count"][day] = 200.0 / 28
            values["avg_order_value"][day] = 8.0
        day += timedelta(days=1)
    scope_snapshot = {"tenant_id": state.get("tenant_id"), "user_id": state.get("user_id")}
    for metric_id, series in values.items():
        typed = build_attribution_evidence(
            evidence_id=f"ev-trend-{metric_id}", evidence_type="trend", metric_id=metric_id,
            claim="authorized complete daily trend", values={
                "result": {
                    "status": "ok",
                    "points": [{"bucket": bucket.isoformat(), "value": value} for bucket, value in series.items()],
                },
            }, source_action_id=f"action-{metric_id}", source_ref="query_metric_trend",
            scope=scope_snapshot, method_id="observed_trend_v1", method_version="1.0",
            authorized_metric_provenance="tool_gateway_policy_validated",
            comparison={"period": {
                "start_date": first.isoformat(), "end_date": last.isoformat(), "granularity": "day",
            }},
        )
        state.setdefault("root_cause_evidence", []).append({
            "metric_id": metric_id, "investigation_evidence": typed.model_dump(mode="json"),
        })

    avg_order_value_evidence = next(
        item for item in state["root_cause_evidence"]
        if item.get("metric_id") == "avg_order_value"
    )
    state["root_cause_evidence"].remove(avg_order_value_evidence)
    assert _available_chat_decompositions(state, "gmv", "platform") == {}
    state["root_cause_evidence"].append(avg_order_value_evidence)

    catalog = _candidate_tool_catalog(state)
    candidate_id, action = next(
        (candidate_id, action)
        for candidate_id, action in catalog.items()
        if isinstance(action, ExecuteDecompositionAction)
    )
    state["diagnostic_selected_candidate_id"] = candidate_id
    state["diagnostic_action"] = action.model_dump(mode="json")

    executed = policy_execute(state)
    assert executed["policy_decision"]["decision"] == "allow"
    assert executed["diagnostic_local_attribution_evidence"].method_id == "product_shapley_v1"
    assert executed.get("diagnostic_tool_results", []) == []

    observed = observe(executed)
    typed = observed["root_cause_evidence"][-1]["investigation_evidence"]
    assert typed["method_id"] == "product_shapley_v1"
    assert typed["authorized_metric_provenance"] == "tool_gateway_policy_validated"
    assert typed["values"]["total_delta"] == pytest.approx(600.0)
    assert len(observed["investigation_state"]["action_results"]) == 1
    assert observed["investigation_state"]["remaining_budget"] == 5

    # Exercise the next chat diagnostic adapter turn, not just the deterministic
    # method: the accepted decomposition must expose its largest factor branch
    # as a server candidate, the Agent may select that handle, and observe must
    # feed the resulting typed evidence back into the same investigation.
    from app.graph.nodes import diagnostic as diagnostic_node
    from app.metrics.authorization import authorize_metric

    observed["authorized_metrics"] = [authorize_metric(
        "order_count", provenance_type="exact", provenance_id="mention-order-count",
        scope="platform", scope_data={},
    ).model_dump(mode="json")]
    observed["diagnostic_step_state"] = {"s1": "completed"}
    selected: dict[str, str] = {}
    original_catalog = diagnostic_node._candidate_tool_catalog

    def capture_candidates(current_state):
        catalog = original_catalog(current_state)
        investigation = InvestigationState.model_validate(current_state["investigation_state"])
        selected["candidate_set_id"] = investigation.candidate_set.candidate_set_id
        selected["candidate_id"] = next(
            candidate_id for candidate_id, action in catalog.items()
            if getattr(action, "metric_name", None) == "order_count"
            and getattr(action, "dimension", None) == "product_category"
        )
        return catalog

    monkeypatch.setattr(diagnostic_node, "_candidate_tool_catalog", capture_candidates)
    monkeypatch.setattr(
        "app.llm.client.llm_agent_action",
        lambda *_args, **_kwargs: {
            "action_type": "CALL_CANDIDATE",
            "candidate_set_id": selected["candidate_set_id"],
            "candidate_id": selected["candidate_id"],
        },
    )
    decided = diagnostic_node.diagnostic_decide(observed)
    assert decided["diagnostic_selected_candidate_id"] == selected["candidate_id"]
    assert decided["diagnostic_decision_source"] == "agent"
    assert decided["diagnostic_action"]["metric_name"] == "order_count"

    monkeypatch.setattr(
        "app.graph.nodes.diagnostic._execute_query",
        lambda query, _state: {
            "tool_name": query.tool_name,
            "arguments": {},
            "result": {
                "status": "ok",
                "contribution_result": {
                    "baseline_total": 100.0,
                    "current_total": 80.0,
                    "total_delta": -20.0,
                    "baseline_items": [{"dimension_value": "Furniture", "value": 100.0}],
                    "current_items": [{"dimension_value": "Furniture", "value": 80.0}],
                },
            },
        },
    )
    executed = diagnostic_node.policy_execute(decided)
    assert executed["policy_decision"]["decision"] == "allow"
    followed_up = diagnostic_node.observe(executed)
    assert len(followed_up["investigation_state"]["action_results"]) == 2
    assert followed_up["investigation_state"]["remaining_budget"] == 4
    assert any(
        item.get("purpose") == "dimension_contribution:product_category"
        and item.get("metric_id") == "order_count"
        for item in followed_up["root_cause_evidence"]
    )


def test_chat_decomposition_rejects_untrusted_or_incomplete_daily_trend_evidence(monkeypatch):
    from app.graph.nodes.diagnostic import _trusted_daily_series
    from app.investigation.evidence import build_attribution_evidence

    scope = {"tenant_id": None, "user_id": None}
    period = {"start_date": "2025-01-01", "end_date": "2025-01-03", "granularity": "day"}
    state = _state()
    state["user_message"] = "GMV trend through 2025-01-03"
    monkeypatch.setattr(
        "app.graph.nodes.diagnostic._dates",
        lambda _question: ("2025-01-01", "2025-01-03"),
    )
    for provenance, points in (
        ("unverified", [
            {"bucket": "2025-01-01", "value": 1},
            {"bucket": "2025-01-02", "value": 2},
            {"bucket": "2025-01-03", "value": 3},
        ]),
        ("tool_gateway_policy_validated", [
            {"bucket": "2025-01-01", "value": 1},
            {"bucket": "2025-01-03", "value": 3},
        ]),
    ):
        typed = build_attribution_evidence(
            evidence_id=f"ev-{provenance}", evidence_type="trend", metric_id="gmv",
            claim="test trend", values={"result": {"status": "ok", "points": points}},
            source_action_id="action", source_ref="query_metric_trend", scope=scope,
            method_id="observed_trend_v1", method_version="1.0",
            authorized_metric_provenance=provenance,
            comparison={"period": period},
        )
        state["root_cause_evidence"] = [{
            "metric_id": "gmv", "investigation_evidence": typed.model_dump(mode="json"),
        }]
        assert _trusted_daily_series(state, "gmv") is None


def test_chat_ratio_decomposition_uses_registered_additive_factors(
    authorized_tenant_context, monkeypatch,
):
    from datetime import date, timedelta

    from app.agents.diagnostic_contracts import ExecuteDecompositionAction
    from app.graph.nodes.diagnostic import _candidate_tool_catalog
    from app.investigation.evidence import build_attribution_evidence

    baseline_start, baseline_end = date(2025, 1, 1), date(2025, 1, 31)
    current_start, current_end = date(2025, 2, 1), date(2025, 2, 28)
    monkeypatch.setattr(
        "app.graph.nodes.diagnostic._in_scope_comparison_windows",
        lambda *_args, **_kwargs: (
            baseline_start, baseline_end, current_start, current_end, "month",
        ),
    )
    monkeypatch.setattr(
        "app.graph.nodes.diagnostic._dates",
        lambda _question: (baseline_start.isoformat(), current_end.isoformat()),
    )
    state = _state()
    state["diagnostic_target_metric"] = "freight_rate"
    scope = {"tenant_id": state.get("tenant_id"), "user_id": state.get("user_id")}
    series_values: dict[str, dict[date, float]] = {"freight_value": {}, "gmv": {}, "freight_rate": {}}
    day = baseline_start
    while day <= current_end:
        if day <= baseline_end:
            series_values["freight_value"][day] = 100.0 / 31
            series_values["gmv"][day] = 1000.0 / 31
        else:
            series_values["freight_value"][day] = 120.0 / 28
            series_values["gmv"][day] = 800.0 / 28
        series_values["freight_rate"][day] = series_values["freight_value"][day] / series_values["gmv"][day] * 100
        day += timedelta(days=1)
    for metric_id, values in series_values.items():
        typed = build_attribution_evidence(
            evidence_id=f"ev-trend-{metric_id}", evidence_type="trend", metric_id=metric_id,
            claim="authorized complete daily trend", values={
                "result": {
                    "status": "ok",
                    "points": [{"bucket": bucket.isoformat(), "value": value} for bucket, value in values.items()],
                },
            }, source_action_id=f"action-{metric_id}", source_ref="query_metric_trend",
            scope=scope, method_id="observed_trend_v1", method_version="1.0",
            authorized_metric_provenance="tool_gateway_policy_validated",
            comparison={"period": {
                "start_date": baseline_start.isoformat(),
                "end_date": current_end.isoformat(), "granularity": "day",
            }},
        )
        state.setdefault("root_cause_evidence", []).append({
            "metric_id": metric_id, "investigation_evidence": typed.model_dump(mode="json"),
        })

    catalog = _candidate_tool_catalog(state)
    candidate_id, action = next(
        (candidate_id, action)
        for candidate_id, action in catalog.items()
        if isinstance(action, ExecuteDecompositionAction)
    )
    state["diagnostic_selected_candidate_id"] = candidate_id
    state["diagnostic_action"] = action.model_dump(mode="json")
    executed = policy_execute(state)
    assert executed["policy_decision"]["decision"] == "allow"
    assert executed["diagnostic_local_attribution_evidence"].method_id == "symmetric_ratio_v1"

    observed = observe(executed)
    evidence = observed["root_cause_evidence"][-1]["investigation_evidence"]
    assert evidence["metric_id"] == "freight_rate"
    assert evidence["values"]["total_delta"] == pytest.approx(5.0)
    assert evidence["values"]["raw_components"]


@pytest.mark.parametrize("tool_name", ["query_metric_trend", "registry_trend"])
def test_anomaly_action_reuses_only_exact_scope_matched_trend_evidence(authorized_tenant_context, tool_name):
    from app.agents.diagnostic_contracts import CallToolAction, DiagnosticPeriod
    from app.graph.nodes.diagnostic import (
        _cached_trend_result, _ensure_graph_investigation_state, _query_arguments,
    )
    from app.investigation.evidence import build_attribution_evidence

    state = _state()
    scope = {
        "allowed_metrics": ["gmv"], "allowed_dimensions": ["product_category"],
        "seller_id": "seller-1", "default_start_date": "2018-08-01",
        "default_end_date": "2018-08-31", "granularity": "day",
    }
    state["controlled_analysis_scope"] = scope
    investigation = _ensure_graph_investigation_state(state)
    query = CallToolAction(
        action_type="CALL_TOOL", tool_name="query_metric_trend", metric_name="gmv",
        period=DiagnosticPeriod(
            start_date="2018-08-01", end_date="2018-08-31", granularity="day",
        ),
    )
    source_id = "source-trend-1"
    source_result = {
        "status": "ok",
        "trend_result": {
            "metric_name": "gmv",
            "points": [{"bucket": "2018-08-01", "value": 10.0}],
        },
    }
    typed = build_attribution_evidence(
        evidence_id=source_id, evidence_type="trend", metric_id="gmv",
        claim="authorized trend", values={"result": source_result},
        source_action_id=source_id, source_ref="query_metric_trend",
        scope=scope, method_id="observed_trend_v1", method_version="1.0",
        authorized_metric_provenance="tool_gateway_policy_validated",
    )
    state["tool_calls"] = [{
        "tool_name": tool_name, "arguments": _query_arguments(query),
        "evidence_id": source_id,
    }]
    state["evidence_bundles"] = [{
        "evidence_id": source_id, "trusted": True,
        "content": {"result": source_result},
    }]
    state["root_cause_evidence"] = [{
        "query_id": source_id, "investigation_evidence": typed.model_dump(mode="json"),
    }]
    cached = _cached_trend_result(state, query)
    assert cached is not None
    assert cached["cached_source_evidence_id"] == source_id
    assert cached["result"] == source_result

    state["root_cause_evidence"][0]["investigation_evidence"]["scope_fingerprint"] = "stale-scope"
    assert _cached_trend_result(state, query) is None


def test_cached_anomaly_action_does_not_repeat_mcp_query(authorized_tenant_context, monkeypatch):
    from app.agents.diagnostic_contracts import CallToolAction, DiagnosticPeriod
    from app.graph.nodes.diagnostic import _ensure_graph_investigation_state, _query_arguments
    from app.investigation.candidates import candidate_id_for
    from app.investigation.contracts import (
        CandidateAction, CandidateFeatures, CheckAnomalyAction,
    )

    state = _state()
    state["controlled_analysis_scope"] = {
        "allowed_metrics": ["gmv"], "allowed_dimensions": ["product_category"],
        "seller_id": "seller-1", "default_start_date": "2018-08-01",
        "default_end_date": "2018-08-31", "granularity": "day",
    }
    investigation = _ensure_graph_investigation_state(state)
    anomaly_action = CheckAnomalyAction(action_type="CHECK_ANOMALY", metric_id="gmv")
    candidate_id = candidate_id_for(anomaly_action)
    investigation.candidate_actions = [CandidateAction(
        candidate_id=candidate_id, action=anomaly_action,
        features=CandidateFeatures(evidence_gap="anomaly:gmv"),
    )]
    state["investigation_state"] = investigation.model_dump(mode="json")
    state["diagnostic_selected_candidate_id"] = candidate_id
    query = CallToolAction(
        action_type="CALL_TOOL", tool_name="query_metric_trend", metric_name="gmv",
        period=DiagnosticPeriod(
            start_date="2018-08-01", end_date="2018-08-31", granularity="day",
        ),
    )
    state["diagnostic_action"] = query.model_dump(mode="json")
    state["tool_calls"] = [{
        "tool_name": query.tool_name, "arguments": _query_arguments(query),
    }]
    cached_payload = {"status": "ok", "trend_result": {"points": []}}
    monkeypatch.setattr(
        "app.graph.nodes.diagnostic._cached_trend_result",
        lambda *_args: {
            "result": cached_payload,
            "tool_name": query.tool_name,
            "cached_source_evidence_id": "ev-trend",
        },
    )
    with patch("app.graph.nodes.diagnostic._execute_query") as gateway:
        result = policy_execute(state)
    gateway.assert_not_called()
    assert result["policy_decision"]["decision"] == "allow"
    assert result["diagnostic_tool_results"][0]["cached_source_evidence_id"] == "ev-trend"
    assert result["diagnostic_investigation_action_count"] == 1


def test_candidate_catalog_fails_closed_without_trusted_tenant_context(monkeypatch):
    state = _state()
    state["controlled_analysis_scope"] = {
        "allowed_metrics": ["gmv"], "allowed_dimensions": ["product_category"],
        "seller_id": "seller-1", "default_start_date": "2018-08-15",
        "default_end_date": "2018-08-20", "granularity": "day",
    }
    monkeypatch.setattr("app.tenancy.current_tenant_context", lambda: None)
    assert _candidate_tool_catalog(state) == {}
    assert state["investigation_state"]["stop_reason"]["code"] == "AUTHORIZATION_BLOCKED"


def test_resume_with_changed_scope_stops_instead_of_resetting_investigation(authorized_tenant_context):
    from app.graph.nodes.diagnostic import _candidate_tool_catalog, _ensure_graph_investigation_state

    state = _state()
    state["diagnostic_target_metric"] = "gmv"
    state["controlled_analysis_scope"] = {
        "allowed_metrics": ["gmv"], "allowed_dimensions": ["product_category"],
        "seller_id": "seller-1", "default_start_date": "2018-08-01",
        "default_end_date": "2018-08-31", "granularity": "month",
    }
    investigation = _ensure_graph_investigation_state(state)
    assert investigation is not None
    investigation.remaining_budget = 2
    investigation.executed_fingerprints = ["already-executed"]
    original_scope = investigation.target.model_dump(mode="json")["canonical_scope"]
    state["investigation_state"] = investigation.model_dump(mode="json")

    state["controlled_analysis_scope"]["seller_id"] = "seller-2"
    assert _candidate_tool_catalog(state) == {}
    invalidated = state["investigation_state"]
    assert invalidated["target"]["canonical_scope"] == original_scope
    assert invalidated["remaining_budget"] == 2
    assert invalidated["executed_fingerprints"] == ["already-executed"]
    assert invalidated["stop_reason"]["code"] == "AUTHORIZATION_BLOCKED"
    assert state["diagnostic_investigation_invalidated"] == "target_changed"


def test_malformed_saved_investigation_state_fails_closed_without_fresh_budget():
    from app.graph.nodes.diagnostic import _ensure_graph_investigation_state

    state = _state()
    state["diagnostic_target_metric"] = "gmv"
    state["diagnostic_investigation_action_count"] = 6
    state["investigation_state"] = {"remaining_budget": 0, "executed_fingerprints": ["spent"]}

    assert _ensure_graph_investigation_state(state) is None
    assert state["investigation_state"]["remaining_budget"] == 0
    assert state["investigation_state"]["executed_fingerprints"] == ["spent"]
    assert state["investigation_state"]["stop_reason"]["code"] == "CAPABILITY_GAP"
    assert state["errors"][0]["code"] == "diagnostic_investigation_state_invalid"


def test_rejected_decomposition_evidence_is_not_published_to_chat_state():
    from app.agents.diagnostic_contracts import ExecuteDecompositionAction
    from app.graph.nodes.diagnostic import _ensure_graph_investigation_state
    from app.investigation.contracts import (
        CandidateAction, CandidateFeatures, DecomposeMetricAction,
    )
    from app.investigation.evidence import build_attribution_evidence

    state = _state()
    state["diagnostic_target_metric"] = "gmv"
    investigation = _ensure_graph_investigation_state(state)
    assert investigation is not None
    candidate_id = "cand-test-decomposition"
    candidate = CandidateAction(
        candidate_id=candidate_id,
        action=DecomposeMetricAction(
            action_type="DECOMPOSE_METRIC", metric_id="gmv",
            method_id="product_shapley_v1", factor_metric_ids=["order_count", "avg_order_value"],
        ),
        features=CandidateFeatures(),
    )
    investigation.candidate_actions = [candidate]
    state["investigation_state"] = investigation.model_dump(mode="json")
    state["diagnostic_selected_candidate_id"] = candidate_id
    state["diagnostic_action"] = ExecuteDecompositionAction(
        action_type="DECOMPOSE_METRIC", candidate_id=candidate_id,
    ).model_dump(mode="json")
    state["diagnostic_local_attribution_evidence"] = build_attribution_evidence(
        evidence_id="ev-wrong-metric", evidence_type="metric_decomposition",
        metric_id="order_count", claim="mismatched metric evidence",
        values={"total_delta": 1.0}, source_action_id=candidate_id,
        source_ref="test", scope=investigation.target.canonical_scope,
        method_id="product_shapley_v1", method_version="1.0",
        authorized_metric_provenance="tool_gateway_policy_validated",
    )

    result = observe(state)
    assert result["root_cause_evidence"] == []
    assert result["evidence_bundles"] == []
    assert result["investigation_state"]["stop_reason"]["code"] == "INVALID_DATA"
    assert result["investigation_state"]["action_results"][0]["status"] == "failed"
    assert result["errors"][-1]["code"] == "diagnostic_decomposition_evidence_rejected"


def test_empty_dimension_grant_does_not_expose_dimension_candidates(authorized_tenant_context):
    state = _state()
    state["controlled_analysis_scope"] = {
        "allowed_metrics": ["gmv"], "allowed_dimensions": [],
        "seller_id": "seller-1", "default_start_date": "2018-08-15",
        "default_end_date": "2018-08-20", "granularity": "day",
    }
    candidates = _candidate_tool_catalog(state)
    assert candidates
    assert all(item.dimension is None for item in candidates.values())


def test_seller_candidate_uses_two_periods_inside_locked_range_not_last_two_days(authorized_tenant_context):
    state = _state()
    state["user_message"] = "分析范围内 GMV 变化"
    state["controlled_analysis_scope"] = {
        "allowed_metrics": ["gmv"], "seller_id": "seller-1",
        "allowed_dimensions": ["product_category", "month_position"],
        "default_start_date": "2017-09-01", "default_end_date": "2018-08-31",
        "granularity": "month",
    }
    candidates = _candidate_tool_catalog(state)
    breakdown = next(q for q in candidates.values() if q.dimension == "product_category")
    comparison = breakdown.comparison
    assert comparison.baseline_start == "2018-07-01"
    assert comparison.baseline_end == "2018-07-31"
    assert comparison.current_start == "2018-08-01"
    assert comparison.current_end == "2018-08-31"
    assert comparison.granularity == "month"
    plan = _plan_for_query(breakdown)
    assert plan.current_start == comparison.current_start
    assert plan.baseline_end == comparison.baseline_end


def test_platform_candidate_requires_two_complete_natural_periods_within_scope():
    windows = _in_scope_comparison_windows(
        date(2017, 9, 1), date(2018, 8, 31), seller_scoped=False,
    )
    assert windows == (
        date(2018, 7, 1), date(2018, 7, 31),
        date(2018, 8, 1), date(2018, 8, 31), "month",
    )
    assert _in_scope_comparison_windows(
        date(2018, 8, 1), date(2018, 8, 10), seller_scoped=False,
    ) is None


def test_selected_month_granularity_uses_latest_complete_months_for_seller_scope():
    windows = _in_scope_comparison_windows(
        date(2017, 9, 1), date(2018, 8, 31), seller_scoped=True, granularity="month",
    )
    assert windows == (
        date(2018, 7, 1), date(2018, 7, 31),
        date(2018, 8, 1), date(2018, 8, 31), "month",
    )
    assert _in_scope_comparison_windows(
        date(2018, 8, 1), date(2018, 8, 10), seller_scoped=True, granularity="month",
    ) is None


def test_seller_day_granularity_does_not_fabricate_monthly_comparison_windows():
    assert _in_scope_comparison_windows(
        date(2018, 7, 1), date(2018, 8, 31), seller_scoped=True, granularity="day",
    ) is None


def test_seller_unspecified_granularity_uses_only_complete_natural_periods():
    assert _in_scope_comparison_windows(
        date(2025, 3, 18), date(2025, 3, 31), seller_scoped=True,
    ) is None


def test_model_query_outside_candidate_catalog_is_denied_before_mcp():
    state = _state()
    state["controlled_analysis_scope"] = {
        "allowed_metrics": ["gmv"], "seller_id": "seller-1",
        "default_start_date": "2018-08-14", "default_end_date": "2018-08-20",
    }
    state["diagnostic_action"] = {
        "action_type": "CALL_TOOL", "tool_name": "query_metric_trend", "metric_name": "gmv",
        "period": {"start_date": "2018-08-13", "end_date": "2018-08-20", "granularity": "day"},
    }
    with patch("app.graph.nodes.diagnostic.query_scoped_business_data") as scoped:
        result = policy_execute(state)
    assert result["policy_decision"]["decision"] == "deny"
    scoped.assert_not_called()


def test_step_allowlist_rejects_terminal_action_outside_plan():
    state = _state()
    state["diagnostic_plan"]["steps"][0]["allowed_actions"] = ["CALL_TOOL"]
    state["diagnostic_action"] = {"action_type": "FINISH", "conclusion": "premature", "evidence_ids": ["x"]}
    result = policy_execute(state)
    assert result["errors"][-1]["code"] == "diagnostic_action_not_allowed"
    assert result["diagnostic_action"]["action_type"] == "ABSTAIN"


def test_critic_rejects_finish_with_pending_steps():
    state = {"diagnostic_plan": {"steps": [{"step_id": "s1"}]}, "diagnostic_step_state": {"s1": "pending"}, "observations": [], "evidence_bundles": [{"evidence_id": "ev1"}], "finish_decision": {"evidence_ids": ["ev1"]}, "tool_calls": []}
    review = _review_diagnostic_trajectory(state)
    assert review["status"] == "rejected"
    assert any("incomplete" in issue for issue in review["issues"])


@pytest.mark.parametrize("evidence_mode", ["missing", "association_only"])
def test_candidate_exhaustion_with_insufficient_evidence_is_structured_abstain(monkeypatch, evidence_mode):
    from app.agents.diagnostic_contracts import AbstainAction
    from app.graph.nodes.diagnostic import _llm_decide_action

    state = _state()
    state["diagnostic_target_metric"] = "gmv"
    state["diagnostic_step_state"] = {"s1": "completed"}
    if evidence_mode == "association_only":
        state["root_cause_evidence"] = [{
            "metric_id": "gmv", "purpose": "association:registered_pair",
            "investigation_evidence": {
                "evidence_id": "association-only", "evidence_type": "registered_association",
                "metric_id": "gmv", "values": {"series_analysis": {"status": "ok"}},
            },
        }]
    monkeypatch.setattr("app.graph.nodes.diagnostic._candidate_tool_catalog", lambda _state: {})
    monkeypatch.setattr("app.llm.client.llm_agent_action", lambda *_args, **_kwargs: None)

    action = _llm_decide_action(state)

    assert isinstance(action, AbstainAction)
    assert action.reason == "insufficient_evidence"
    assert not any(item.get("code") == "diagnostic_action_invalid" for item in state.get("errors", []))
    assert state["diagnostic_llm_failures"][-1]["class"] == "empty_response"


def test_registered_factor_mismatch_stops_as_invalid_data(authorized_tenant_context, monkeypatch):
    from datetime import date, timedelta

    from app.agents.diagnostic_contracts import ExecuteDecompositionAction
    from app.graph.nodes.diagnostic import _candidate_tool_catalog

    baseline_start, baseline_end = date(2025, 1, 1), date(2025, 1, 31)
    current_start, current_end = date(2025, 2, 1), date(2025, 2, 28)
    monkeypatch.setattr(
        "app.graph.nodes.diagnostic._in_scope_comparison_windows",
        lambda *_args, **_kwargs: (
            baseline_start, baseline_end, current_start, current_end, "month",
        ),
    )
    monkeypatch.setattr(
        "app.graph.nodes.diagnostic._dates",
        lambda _question: (baseline_start.isoformat(), current_end.isoformat()),
    )

    series = {}
    day = baseline_start
    while day <= current_end:
        if day <= baseline_end:
            series[day] = 1000.0 / 31
        else:
            series[day] = 500.0 / 28
        day += timedelta(days=1)

    def trusted(_state, metric_id):
        if metric_id == "gmv":
            return series
        values = {}
        day = baseline_start
        while day <= current_end:
            if day <= baseline_end:
                values[day] = 100.0 / 31
            else:
                values[day] = 60.0 / 28
            day += timedelta(days=1)
        return values

    monkeypatch.setattr("app.graph.nodes.diagnostic._trusted_daily_series", trusted)
    state = _state()
    state["diagnostic_target_metric"] = "gmv"
    state["diagnostic_step_state"] = {"s1": "completed"}
    catalog = _candidate_tool_catalog(state)
    candidate_id, action = next(
        (candidate_id, action) for candidate_id, action in catalog.items()
        if isinstance(action, ExecuteDecompositionAction)
    )
    state["diagnostic_selected_candidate_id"] = candidate_id
    state["diagnostic_action"] = action.model_dump(mode="json")

    result = policy_execute(state)
    observed = observe(result)

    assert result["policy_decision"]["decision"] == "deny"
    assert observed["investigation_state"]["stop_reason"]["code"] == "INVALID_DATA"
    assert not observed.get("errors")
    assert not result.get("diagnostic_local_attribution_evidence")

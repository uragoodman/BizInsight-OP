"""Critic 节点 — 结构化质检 + LLM 审核双路径。

审核流程：
1. 结构化质检（必选）：检查数据是否完整、RAG 是否有命中。
2. LLM 审核（可选）：groundedness + evidence completeness + logic consistency。
   LLM 不可用时自动跳过，不影响主链路。

MECE 写入字段：critic_result, critic_verdict, critic_decisions,
                critic_revision_target, current_agent, node_history

路径感知：
- metric_definition 只查 RAG 证据。
- metric_query 只查数据查询结果。
- anomaly_analysis / report_generation 查两者。
"""

from math import ceil
from typing import Any

from app.agents.critic_agent import (
    llm_review_evidence_completeness,
    llm_review_groundedness,
    llm_review_logic_consistency,
    llm_review_combined_experiment,
    review_data_query_batch,
    review_single_data_query,
    review_data_query_result,
)
from app.graph.state import AgentState
from app.agents.contracts import CriticRevisionTarget
from app.graph.critic_review_mode import CriticLLMReviewMode
from app.graph.critic_review_policy import (
    decide_optional_llm_review,
    decide_optional_review_cascade,
)
from app.graph.critic_llm_thinking_mode import CriticLLMThinkingMode
from app.graph.pre_critic_snapshot import capture_pre_critic_answer_snapshot
from app.graph.presentation_status import diagnostic_terminal_presentation_status
from app.core.config import get_settings
from app.core.execution_context import turn_soft_limit_reached
from app.analysis.contracts import PeriodicReport
from app.analysis.periods import previous_period, validate_natural_period, year_ago_period
from app.analysis.language_guardrails import unsupported_causal_assertions
from app.analysis.numeric_output_review import review_numeric_output
from hashlib import sha256
from pathlib import Path
from json import dumps


def _same_number(left: float | None, right: float | None, tolerance: float = 0.01) -> bool:
    if left is None or right is None:
        return left is right
    return abs(left - right) <= tolerance


def _review_registered_numeric_decomposition(
    state: AgentState, request_override: dict[str, object] | None = None,
) -> dict[str, object]:
    """Review the registered decomposition result through its validated evidence binding.

    This operation executes in the governed diagnostic graph, so it has no
    ordinary `tool_result` query envelope. The result is accepted only when
    the final structured analysis points to the exact validated decomposition
    and the registered input series remain available in the same scope.
    """
    request = request_override or state.get("registered_decomposition_request") or {}
    status = request.get("status")
    if status != "ready":
        if status in {"unsupported", "authorization_blocked", "dates_required"}:
            return {"status": "incomplete", "operation": "registered_numeric_decomposition",
                    "limitation_status": status, "message": "The registered decomposition was not executable."}
        return {"status": "rejected", "message": "Registered decomposition request is malformed."}

    investigation = state.get("investigation_state") or {}
    stop = investigation.get("stop_reason") or {}
    if stop.get("code") in {
        "DATA_INSUFFICIENT", "CAPABILITY_GAP", "AUTHORIZATION_BLOCKED",
        "USER_SCOPE_LIMIT", "BUDGET_EXHAUSTED", "EXECUTION_FAILED", "INVALID_DATA",
    }:
        return {
            "status": "incomplete", "operation": "registered_numeric_decomposition",
            "stop_code": stop.get("code"), "message": stop.get("detail"),
        }

    from app.graph.nodes.diagnostic import _completed_valid_evidence_ids, _trusted_daily_series

    target_metric = str(request.get("target_metric_id") or "")
    method_id = str(request.get("method_id") or "")
    analysis = state.get("analysis_result") or {}
    card = analysis.get("metric_decomposition") or {}
    evidence_ids = [str(item) for item in card.get("evidence_ids", [])]
    valid_ids = set(_completed_valid_evidence_ids(state))
    candidates = [
        evidence for evidence in investigation.get("evidence", [])
        if isinstance(evidence, dict)
        and evidence.get("evidence_id") in valid_ids
        and evidence.get("evidence_id") in evidence_ids
        and evidence.get("evidence_type") == "metric_decomposition"
        and evidence.get("metric_id") == target_metric
        and evidence.get("method_id") == method_id
        and evidence.get("authorized_metric_provenance") == "tool_gateway_policy_validated"
        and (evidence.get("data_quality") or {}).get("status") == "valid"
        and (evidence.get("data_quality") or {}).get("input_series_complete") is True
    ]
    dates = (request.get("baseline_date"), request.get("current_date"))
    if len(candidates) != 1 or len(evidence_ids) != 1 or analysis.get("abstain") is True:
        return {"status": "rejected", "message": "The structured answer is not bound to one valid registered decomposition evidence."}
    evidence = candidates[0]
    values = evidence.get("values") or {}
    comparison = evidence.get("comparison") or {}
    baseline = comparison.get("baseline_period") or {}
    current = comparison.get("current_period") or {}
    if (
        not target_metric or not method_id
        or card.get("metric_id") != target_metric
        or card.get("method_id") != method_id
        or dates != (baseline.get("start_date"), current.get("start_date"))
        or baseline.get("start_date") != baseline.get("end_date")
        or current.get("start_date") != current.get("end_date")
        or values.get("baseline_value") is None
        or values.get("current_value") is None
        or values.get("total_delta") is None
    ):
        return {"status": "rejected", "message": "The decomposition target, method, dates, or numeric result do not match the request."}
    input_ids = list(dict.fromkeys(str(item) for item in request.get("input_metric_ids", [])))
    if not input_ids or any(_trusted_daily_series(state, metric_id) is None for metric_id in input_ids):
        return {"status": "rejected", "message": "A registered input series is not complete and valid in the requested scope."}
    if set(str(item) for item in values.get("factor_metrics", [])) != set(
        str(item) for item in request.get("factor_metric_ids", [])
    ):
        return {"status": "rejected", "message": "The decomposition factors do not match the registered method contract."}
    return {
        "status": "passed", "operation": "registered_numeric_decomposition",
        "target_metric_id": target_metric, "method_id": method_id,
        "evidence_ids": evidence_ids,
    }


def _registered_numeric_decomposition_request(state: AgentState) -> dict[str, object] | None:
    """Return the request only when the server-side operation intent is explicit."""
    decision = state.get("metric_planning_decision") or {}
    compilation = state.get("context_compilation") or {}
    compiled_intent = compilation.get("operation_intent")
    operation_intent = (
        decision.get("operation_intent")
        or (compiled_intent.get("kind") if isinstance(compiled_intent, dict) else None)
    )
    if operation_intent != "registered_numeric_decomposition":
        return None
    request = state.get("registered_decomposition_request")
    if not request:
        request = decision.get("registered_decomposition")
    return request if isinstance(request, dict) else {}


def _registered_decomposition_review_context(
    state: AgentState, request: dict[str, object], data_review: dict[str, object],
) -> dict[str, object]:
    """Build reviewer context from the current plan and validated evidence only."""
    from app.graph.nodes.diagnostic import _completed_valid_evidence_ids

    analysis = state.get("analysis_result") or {}
    card = analysis.get("metric_decomposition") or {}
    investigation = state.get("investigation_state") or {}
    investigation_target = investigation.get("target") or {}
    valid_ids = set(_completed_valid_evidence_ids(state))
    card_ids = {str(item) for item in card.get("evidence_ids", []) if item}
    target_id = str(request.get("target_metric_id") or "")
    method_id = str(request.get("method_id") or "")
    evidence = [
        item for item in investigation.get("evidence", [])
        if isinstance(item, dict)
        and str(item.get("evidence_id") or "") in valid_ids & card_ids
        and item.get("evidence_type") == "metric_decomposition"
        and item.get("metric_id") == target_id
        and item.get("method_id") == method_id
    ] if data_review.get("status") == "passed" else []
    stop_reason = analysis.get("stop_reason") or (investigation.get("stop_reason") or {})
    if isinstance(stop_reason, dict):
        stop_reason = dict(stop_reason)
        if stop_reason.get("failure_context") is None:
            stop_reason.pop("failure_context", None)
    return {
        "operation_intent": "registered_numeric_decomposition",
        "task_type": "metric_query",
        "target_metric_id": target_id,
        "method_id": method_id,
        "scope": (
            request.get("scope")
            or state.get("controlled_analysis_scope")
            or (investigation_target.get("canonical_scope") if isinstance(investigation_target, dict) else None)
        ),
        "baseline_date": request.get("baseline_date"),
        "current_date": request.get("current_date"),
        "decomposition_review_status": data_review.get("status"),
        "verified_numeric_evidence": evidence,
        "limitations": list(analysis.get("limitations") or []),
        "stop_reason": stop_reason,
        "business_causality": "not_established_by_numeric_decomposition",
    }


def _statistical_contribution_review_context(
    state: AgentState, request_contract_review: dict[str, object],
) -> dict[str, object]:
    """Build Critic context from validated contribution evidence or a verified limit."""
    compilation = state.get("context_compilation") or {}
    contract = compilation.get("request_contract") or {}
    accepted_ids = {
        str(item) for item in (request_contract_review.get("contribution_evidence_ids") or []) if item
    }
    evidence: list[dict[str, object]] = []
    if request_contract_review.get("status") == "passed":
        for bundle in state.get("evidence_bundles") or []:
            if not isinstance(bundle, dict) or bundle.get("trusted") is not True:
                continue
            evidence_id = str(bundle.get("evidence_id") or "")
            content = bundle.get("content") or {}
            result = content.get("result") or {} if isinstance(content, dict) else {}
            if (
                evidence_id in accepted_ids
                and isinstance(content, dict)
                and content.get("metric_id") == contract.get("metric_id")
                and content.get("query_type") == "metric_contribution"
                and isinstance(result, dict)
                and result.get("status") == "ok"
            ):
                evidence.append({"evidence_id": evidence_id, "content": content})
    # Recompute the ledger from the authorized tool evidence instead of trusting
    # answer text or a caller-supplied presentation. Projection also checks the
    # registered method, period totals and numerical reconciliation.
    verified_ledger: list[dict[str, object]] = []
    if request_contract_review.get("status") == "passed" and evidence:
        from copy import deepcopy
        from app.graph.nodes.answer import _validated_request_contribution_cards

        accepted_evidence = {item["evidence_id"] for item in evidence}
        for card in _validated_request_contribution_cards(deepcopy(state)):
            card_ids = set(map(str, card.get("evidence_ids") or []))
            if card.get("method") == "symmetric_mix_within_v1" and card_ids and card_ids <= accepted_evidence:
                verified_ledger.append(card)
    capability_stop = _validated_contribution_capability_stop(
        state, contract, request_contract_review,
    )
    return {
        "operation_intent": "dimensional_statistical_contribution",
        "metric_id": contract.get("metric_id"),
        "dimension": contract.get("dimension"),
        "comparison": contract.get("comparison"),
        "contract_hash": contract.get("contract_hash"),
        "contract_review_status": request_contract_review.get("status"),
        "review_outcome": "capability_limited" if capability_stop else "contribution_result",
        # This is a validated operational limitation, not numerical evidence.
        # Keep it separate from evidence IDs and contribution values.
        "capability_limitation": capability_stop,
        "verified_contribution_evidence": evidence,
        "verified_contribution_ledger": verified_ledger,
        "business_causality": "not_established_by_statistical_contribution",
    }


def _review_symmetric_contribution_answer(state: AgentState, context: dict[str, object]) -> dict[str, object]:
    """Verify the server-rendered ratio-contribution block, not arbitrary prose."""
    presentation = state.get("analysis_presentation") or {}
    cards = presentation.get("contributions") or [] if isinstance(presentation, dict) else []
    ratio_cards = [card for card in cards if isinstance(card, dict) and card.get("method") == "symmetric_mix_within_v1"]
    ledger = context.get("verified_contribution_ledger") or []
    if not ratio_cards and not ledger:
        return {"status": "not_applicable", "issues": [], "coverage": "ratio contribution block only"}
    if len(ratio_cards) != 1 or len(ledger) != 1:
        return {"status": "rejected", "issues": ["Ratio contribution card or verified ledger is missing or ambiguous."], "coverage": "ratio contribution block only"}
    from app.graph.nodes.answer import _dimension_contribution_text, _is_validated_dimension_contribution_card

    if not _is_validated_dimension_contribution_card(ratio_cards[0], ledger):
        return {"status": "rejected", "issues": ["Ratio contribution card differs from verified ledger."], "coverage": "ratio contribution block only"}
    answer = str(state.get("answer") or state.get("final_answer") or "")
    expected = _dimension_contribution_text(ledger[0])
    # Revision prose may be appended; it does not replace the canonical block.
    if answer.count("已验证的维度统计贡献：") != 1 or answer.count(expected) != 1:
        return {"status": "rejected", "issues": ["Canonical ratio contribution answer block is missing, duplicated or differs from the verified ledger."], "coverage": "ratio contribution block only"}
    return {"status": "approved", "issues": [], "coverage": "ratio contribution block only; other prose is not checked"}

PATHS_REQUIRING_RAG = {"metric_definition", "anomaly_analysis", "report_generation"}
PATHS_REQUIRING_DATA = {"metric_query", "anomaly_analysis", "report_generation"}
_BATCH_STATUS_SEVERITY = {
    "all_success": 0,
    "no_data": 0,
    "partial_unavailable": 1,
    "execution_failed": 2,
}


def _build_diagnostic_investigation_context(state: AgentState) -> dict[str, object] | None:
    """Build a role-separated Critic view from validated investigation state."""
    if state.get("task_type") != "anomaly_analysis" or not state.get("diagnostic_plan"):
        return None

    from app.agents.diagnostic_contracts import DiagnosticPlan
    from app.graph.nodes.diagnostic import (
        _completed_valid_evidence_ids,
        _conflicting_evidence_ids,
        _optional_data_gap_is_scoped_to_valid_decomposition,
        validated_dimension_contribution_evidence,
    )
    from app.investigation.contracts import ActionResult
    from app.investigation.engine import MIN_ANOMALY_VALID_BUCKETS

    try:
        plan_steps = list(DiagnosticPlan.model_validate(state["diagnostic_plan"]).steps)
    except Exception:  # invalid plans cannot relax the Critic's assessment
        plan_steps = []
    investigation = state.get("investigation_state") or {}
    investigation = investigation if isinstance(investigation, dict) else {}
    target = investigation.get("target") or {}
    target = target if isinstance(target, dict) else {}
    step_states = state.get("diagnostic_step_state") or {}
    step_states = step_states if isinstance(step_states, dict) else {}
    valid_ids = set(_completed_valid_evidence_ids(state))
    validated_contributions = {
        item["evidence_id"]: item
        for item in validated_dimension_contribution_evidence(state)
    }
    conflict_ids = _conflicting_evidence_ids(investigation)
    evidence_by_id = {
        str(item["evidence_id"]): item for item in investigation.get("evidence", [])
        if isinstance(item, dict) and item.get("evidence_id")
    }

    decompositions = [
        evidence_by_id[eid] for eid in sorted(valid_ids)
        if eid in evidence_by_id
        and evidence_by_id[eid].get("evidence_type") == "metric_decomposition"
        and evidence_by_id[eid].get("metric_id") == target.get("metric_id")
    ]
    factor_metrics = {str(target.get("metric_id") or "")}
    for evidence in decompositions:
        factor_metrics.update(str(item) for item in (evidence.get("values") or {}).get("factor_metrics", []) if item)

    verified_evidence: list[dict[str, object]] = []
    for eid in sorted(valid_ids):
        evidence = evidence_by_id.get(eid)
        if not evidence:
            continue
        kind, metric_id = evidence.get("evidence_type"), str(evidence.get("metric_id") or "")
        if kind == "metric_decomposition":
            values = evidence.get("values") or {}
        elif kind == "dimension_contribution" and eid in validated_contributions:
            contribution = validated_contributions[eid]
            values = {
                "dimension": contribution["dimension"],
                "baseline_value": contribution["baseline_value"],
                "current_value": contribution["current_value"],
                "total_delta": contribution["total_delta"],
                "unit": contribution["unit"],
                "method_id": contribution["method_id"],
                "components": contribution["items"],
                "residual_value": contribution.get("residual_value"),
                "attribution_status": "ok",
            }
        elif kind == "trend" and metric_id in factor_metrics:
            raw_values = evidence.get("values") or {}
            query, result = raw_values.get("query") or {}, raw_values.get("result") or {}
            points = result.get("points", []) if isinstance(result, dict) else []
            values = {
                "query": query,
                "result": {key: result.get(key) for key in (
                    "metric_name", "start_date", "end_date", "granularity", "unit", "status",
                )},
                "numeric_point_count": sum(
                    1 for point in points if isinstance(point, dict)
                    and isinstance(point.get("value"), (int, float))
                    and not isinstance(point.get("value"), bool)
                ),
            }
        else:
            continue
        verified_evidence.append({
            "evidence_id": eid, "evidence_type": kind, "metric_id": metric_id,
            "values": values, "comparison": evidence.get("comparison") or {},
            "unit": values.get("unit") or evidence.get("unit") or None,
            "source_action_id": evidence.get("source_action_id"),
            "scope_fingerprint": evidence.get("scope_fingerprint"),
            "role": "verified_numeric_evidence",
        })

    verified_contribution_evidence = [
        item for item in verified_evidence
        if item.get("evidence_type") == "dimension_contribution"
    ]

    steps = [
        {
            "step_id": step.step_id, "kind": step.kind, "required": bool(step.required),
            "requirement_basis": "effective_required_property_in_validated_server_plan",
            "state": step_states.get(step.step_id, "pending"),
        }
        for step in plan_steps
    ]
    step_by_kind = {step.kind: step for step in plan_steps if step.kind}
    gaps_by_action: dict[str, dict] = {}
    for gap in state.get("diagnostic_candidate_data_gaps") or []:
        if isinstance(gap, dict) and gap.get("action_id"):
            gaps_by_action[str(gap["action_id"])] = gap

    actions: list[dict[str, object]] = []
    limitations: list[dict[str, object]] = []
    for action in investigation.get("action_results", []):
        if not isinstance(action, dict):
            continue
        action_id = str(action.get("action_id") or "")
        failure_code = action.get("failure_code")
        gap = gaps_by_action.get(action_id, {})
        optional = False
        decomposition_id = None
        if failure_code == "DATA_INSUFFICIENT":
            try:
                optional, decomposition_id = _optional_data_gap_is_scoped_to_valid_decomposition(
                    state, ActionResult.model_validate(action),
                    allow_budget_stop_context=True,
                )
            except Exception:  # malformed records cannot be classified as ignorable
                pass

        action_kind = None
        if action.get("action_type") == "ANALYZE_TREND":
            action_kind = "trend" if action.get("metric_id") == target.get("metric_id") else "related_metric"
        elif action.get("action_type") in {"DECOMPOSE_METRIC", "BREAKDOWN_DIMENSION", "DRILL_DOWN"}:
            action_kind = "dimension_contribution"
        matched_step = next((step for step in plan_steps if step.step_id == gap.get("required_step_id")), None)
        matched_step = matched_step or step_by_kind.get(action_kind)
        validation = action.get("validation_result") or {}
        checks = validation.get("checks", []) if isinstance(validation, dict) else []
        action_valid_ids = [str(eid) for eid in action.get("evidence_ids", []) if str(eid) in valid_ids]
        bucket_count = bucket_evidence_id = None
        bucket_period = bucket_scope = None
        if optional:
            decomposition = evidence_by_id.get(str(decomposition_id or ""), {})
            decomposition_inputs = set(
                (decomposition.get("values") or {}).get("input_evidence_ids", [])
            )
            series = next((
                (eid, evidence_by_id[eid]) for eid in sorted(valid_ids)
                if eid in evidence_by_id
                and evidence_by_id[eid].get("evidence_type") == "trend"
                and evidence_by_id[eid].get("metric_id") == action.get("metric_id")
                and evidence_by_id[eid].get("scope_fingerprint") == target.get("scope_fingerprint")
                and (
                    eid in decomposition_inputs
                    or str(decomposition.get("evidence_type") or "") == "dimension_contribution"
                    and str(evidence_by_id[eid].get("metric_id") or "") == str(target.get("metric_id") or "")
                )
            ), None)
            if series:
                bucket_evidence_id, evidence = series
                source_values = evidence.get("values") or {}
                source_query = source_values.get("query") or {}
                result = source_values.get("result") or {}
                points = result.get("points", []) if isinstance(result, dict) else []
                bucket_count = sum(
                    1 for point in points if isinstance(point, dict)
                    and isinstance(point.get("value"), (int, float))
                    and not isinstance(point.get("value"), bool)
                )
                bucket_period = source_query.get("period") or source_query.get("comparison")
                bucket_scope = evidence.get("scope_fingerprint")
        threshold = (
            {"rule": "minimum_valid_time_buckets", "minimum": MIN_ANOMALY_VALID_BUCKETS,
             "source": "app.investigation.engine.MIN_ANOMALY_VALID_BUCKETS"}
            if action.get("action_type") == "CHECK_ANOMALY" else None
        )
        required = False if optional else (bool(matched_step.required) if matched_step else None)
        requirement_basis = (
            "server_optional_gap_validation_linked_to_validated_numeric_evidence" if optional else
            "effective_required_property_in_validated_server_plan" if matched_step else
            "not_server_classified_as_optional_deep_check"
        )
        record = {
            "action_id": action_id, "candidate_id": action.get("candidate_id"),
            "branch_id": action.get("branch_id"), "action_type": action.get("action_type"),
            "target_metric": action.get("metric_id"), "dimension": action.get("dimension"),
            "status": action.get("status"), "failure_code": failure_code,
            "failure_reason": action.get("failure_detail") or gap.get("detail"),
            "matched_plan_step": ({
                "step_id": matched_step.step_id, "kind": matched_step.kind,
                "required": bool(matched_step.required),
                "requirement_basis": "effective_required_property_in_validated_server_plan",
                "state": step_states.get(matched_step.step_id, "pending"),
            } if matched_step else None),
            "requirement": {"required": required, "basis": requirement_basis},
            "continuing_with_alternatives": gap.get("continuing_with_alternatives"),
            "validation_status": validation.get("status") if isinstance(validation, dict) else None,
            "validation_checks": [check for check in checks if isinstance(check, dict)],
            "verified_evidence_ids": action_valid_ids,
            "conflicting_evidence_ids": sorted(conflict_ids),
            "limitation_relation_evidence_id": decomposition_id,
            "observed_valid_bucket_count": bucket_count,
            "bucket_count_source_evidence_id": bucket_evidence_id,
            "bucket_count_source_period": bucket_period,
            "bucket_count_source_scope_fingerprint": bucket_scope,
            "bucket_count_basis": "linked_valid_trend_in_same_scope_and_period" if bucket_evidence_id else None,
            "public_validation_rule": threshold,
        }
        actions.append(record)
        if failure_code or action.get("status") in {"failed", "denied", "capability_gap"}:
            conflict = failure_code == "INVALID_DATA" or bool(conflict_ids)
            blocking = not optional
            classification = (
                "optional_deep_check_incomplete" if optional else
                "evidence_conflict_or_invalid_data" if conflict else
                "authorization_or_execution_failure"
                if failure_code in {"AUTHORIZATION_BLOCKED", "EXECUTION_FAILED"} or action.get("status") == "denied" else
                "required_or_unresolved_data_gap"
            )
            limitations.append({
                "action_id": action_id, "action_type": record["action_type"],
                "target_metric": record["target_metric"], "failure_code": failure_code,
                "failure_reason": record["failure_reason"],
                "classification": classification,
                "severity": "local_limitation" if optional else "blocking" if conflict else "blocking_or_unresolved",
                "required": required, "requirement_basis": requirement_basis,
                "validation_status": record["validation_status"],
                "validation_checks": record["validation_checks"],
                "limitation_relation_evidence_id": decomposition_id,
                "observed_valid_bucket_count": bucket_count,
                "bucket_count_source_evidence_id": bucket_evidence_id,
                "bucket_count_source_period": bucket_period,
                "bucket_count_source_scope_fingerprint": bucket_scope,
                "bucket_count_basis": record["bucket_count_basis"],
                "public_validation_rule": threshold,
                "action_failure_is_numerical_evidence": False,
                # The local shortfall must be disclosed, even when it does not
                # invalidate a separately validated decomposition.
                "may_be_ignored": False,
                "invalidates_linked_decomposition": bool(blocking),
                "global_stop_may_be_ignored": False,
            })

    stop = investigation.get("stop_reason") or {}
    causal = state.get("causal_readiness") or (state.get("analysis_result") or {}).get("causal_readiness") or {}
    causal = causal if isinstance(causal, dict) else {}
    causal_estimated = causal.get("causal_effect_estimated") is True
    return {
        "schema_version": "1.0",
        "target": {
            "metric_id": target.get("metric_id"),
            "current_period": {"start_date": target.get("current_start"), "end_date": target.get("current_end")},
            "baseline_period": {"start_date": target.get("baseline_start"), "end_date": target.get("baseline_end")},
            "scope_fingerprint": target.get("scope_fingerprint"),
        },
        "plan_steps": steps,
        "validated_numeric_decompositions": [
            {**item, "role": "validated_numeric_decomposition_not_business_causality"}
            for item in verified_evidence if item["evidence_type"] == "metric_decomposition"
        ],
        "verified_numeric_evidence": verified_evidence,
        # Keep a named contribution view for reviewers that distinguish
        # contribution claims from the broader numeric evidence collection.
        # It is a projection of the same validated records, not a second trust
        # source or a separate approval signal.
        "verified_contribution_evidence": verified_contribution_evidence,
        "contribution_ratio_semantics": {
            "method_id": "additive_change_v1",
            "formula": "component_delta / total_delta when total_delta is nonzero",
            "interpretation": "signed share of net change, not absolute share",
            "opposing_effects": (
                "positive and negative absolute subtotals can each exceed 100%; "
                "check signed contributions and the validated residual against total_delta"
            ),
        } if verified_contribution_evidence else None,
        "verified_evidence_ids": sorted(valid_ids),
        "action_results": actions,
        "investigation_limitations": limitations,
        "terminal_reason": stop,
        "global_stop": {
            **stop,
            "is_terminal": bool(stop.get("code")),
            "may_be_ignored": False,
            "independent_of_local_action_gap": stop.get("code") == "BUDGET_EXHAUSTED",
        },
        "business_causality": {
            "status": "estimated" if causal_estimated else "unconfirmed",
            "readiness_status": causal.get("status"),
            "causal_effect_estimated": causal_estimated,
            "decomposition_alone_establishes_business_causality": False,
            "basis": causal.get("reason") or "No causal claim is inferred from numerical decomposition.",
        },
        "conflicting_evidence_ids": sorted(conflict_ids),
    }


def _review_diagnostic_trajectory(state: AgentState) -> dict[str, object]:
    """Audit the diagnostic loop without treating budget as work to consume."""
    plan = state.get("diagnostic_plan") or {}
    if not plan:
        return {"status": "skipped", "issues": []}
    issues: list[str] = []
    step_state = state.get("diagnostic_step_state", {})
    observations = state.get("observations", [])
    observed_ids = {eid for item in observations for eid in item.get("evidence_ids", [])}
    if any(status == "completed" for status in step_state.values()) and not observed_ids:
        issues.append("Completed diagnostic steps have no observation evidence.")
    finish = state.get("finish_decision") or {}
    intent = (state.get("context_compilation") or {}).get("operation_intent") or {}
    statistical_task = isinstance(intent, dict) and intent.get("kind") == "dimensional_statistical_contribution"
    if finish and finish.get("action_type") != "ABSTAIN" and statistical_task:
        # Optional exploration can remain pending after a descriptive task is
        # complete. Revalidate the same server evidence floor used by decide;
        # a marker or an LLM's FINISH alone cannot waive plan obligations.
        from app.graph.nodes.diagnostic import _validated_statistical_contribution_terminal

        statistical_finish = _validated_statistical_contribution_terminal(
            state, allow_completed_stop=True,
        )
        if statistical_finish is None or not set(statistical_finish.evidence_ids) <= set(finish.get("evidence_ids") or []):
            issues.append("Terminal statistical task does not satisfy its validated required evidence contract.")
    elif finish and finish.get("action_type") != "ABSTAIN" and any(status == "pending" for status in step_state.values()):
        issues.append("Terminal diagnostic decision was made while plan steps remain incomplete.")
    if finish and any(eid not in {b.get("evidence_id") for b in state.get("evidence_bundles", [])} for eid in finish.get("evidence_ids", [])):
        issues.append("Terminal diagnostic decision references unknown evidence.")
    signatures = [
        (call.get("tool_name"), repr(call.get("arguments") or {}))
        for call in state.get("tool_calls", [])
    ]
    if len(signatures) != len(set(signatures)):
        issues.append("Diagnostic trajectory contains a repeated tool call.")
    requested = list(state.get("requested_dimensions") or [])
    unavailable = set(state.get("unavailable_requested_dimensions") or [])
    if unavailable:
        gaps = [
            item for item in state.get("diagnostic_capability_gaps", [])
            if isinstance(item, dict) and item.get("kind") == "requested_dimension"
        ]
        recorded = {str(item.get("requested_dimension")) for item in gaps}
        analysis = state.get("analysis_result") or {}
        investigation = state.get("investigation_state") or {}
        stop = investigation.get("stop_reason") or {} if isinstance(investigation, dict) else {}
        contributions = analysis.get("contribution") or []
        mismatched_contributions = [
            item for item in contributions
            if isinstance(item, dict) and item.get("dimension") not in requested
        ]
        action_results = investigation.get("action_results", []) if isinstance(investigation, dict) else []
        mismatched_actions = [
            item for item in action_results
            if isinstance(item, dict)
            and item.get("action_type") in {"BREAKDOWN_DIMENSION", "DRILL_DOWN"}
            and item.get("dimension") not in requested
        ]
        if mismatched_contributions or mismatched_actions:
            issues.append("A dimension contribution was executed outside the user's explicit dimension request.")
        if not unavailable <= recorded:
            issues.append("The requested unavailable dimension is missing its structured capability record.")
        if stop.get("code") not in {"CAPABILITY_GAP", "AUTHORIZATION_BLOCKED"}:
            issues.append("The unavailable requested dimension did not terminate with a structured limitation.")
        if not issues:
            return {
                # Critic approves the truthful, bounded terminal answer.  It
                # does not approve a completed dimension analysis; the
                # explicit outcome keeps that distinction machine-readable.
                "status": "approved",
                "outcome": "capability_limited",
                "issues": [],
                "requested_dimensions": requested,
                "unavailable_requested_dimensions": sorted(unavailable),
                "stop_code": stop.get("code"),
            }
    return {"status": "approved" if not issues else "rejected", "issues": issues}


def _review_claim_bindings(state: AgentState) -> dict[str, object]:
    """Verify every emitted claim points to evidence recorded in the same trace."""
    known_ids = {item.get("evidence_id") for item in state.get("evidence_bundles", [])}
    issues: list[str] = []
    for index, claim in enumerate(state.get("claims", [])):
        references = claim.get("source_evidence_ids", [])
        if not references:
            issues.append(f"Claim {index} has no evidence reference.")
        elif any(item not in known_ids for item in references):
            issues.append(f"Claim {index} references unknown evidence.")
    return {"status": "approved" if not issues else "rejected", "issues": issues}


def _is_bound_single_no_data_result(
    state: AgentState,
    data_review: dict[str, object],
    request_contract_review: dict[str, object],
) -> bool:
    """Recognize only a single, fully bound typed no-data metric result.

    This is a successful terminal answer to the user's query (the system
    accurately reports that no value exists), not approval of a numeric fact.
    Multi-item partial batches, unsupported operations, execution errors,
    unbound results, and contract/auth failures remain fail-closed.
    """

    if any((
        state.get("task_type") != "metric_query",
        state.get("terminal_code") != "DATA_INSUFFICIENT",
        request_contract_review.get("status") != "passed",
        bool(state.get("errors")),
        state.get("unavailable_requested_dimensions"),
        data_review.get("status") != "partial",
        data_review.get("batch_status") != "partial_unavailable",
    )):
        return False

    query_results = state.get("query_results") or []
    plans = (state.get("metric_query_batch_plan") or {}).get("plans") or []
    calls = [item for item in (state.get("tool_calls") or []) if isinstance(item, dict)]
    items = data_review.get("items") or []
    if len(query_results) != 1 or len(plans) != 1 or len(calls) != 1 or len(items) != 1:
        return False
    wrapper = query_results[0] if isinstance(query_results[0], dict) else {}
    result = wrapper.get("result") if isinstance(wrapper.get("result"), dict) else {}
    item = items[0] if isinstance(items[0], dict) else {}
    status = str(result.get("status") or "")
    if status not in {"not_found", "insufficient_data"}:
        return False
    if any((
        item.get("status") != "unavailable",
        item.get("execution_status") != status,
        calls[0].get("status") != status,
        item.get("request_id") != wrapper.get("request_id"),
        item.get("evidence_id") != wrapper.get("evidence_id"),
    )):
        return False
    required_bindings = {
        "has_request_id", "plan_matches_result", "exact_tool_invocation_matches",
        "tool_status_matches_result", "evidence_matches_request_plan_result",
    }
    bindings = item.get("binding_checks") if isinstance(item.get("binding_checks"), dict) else {}
    if not required_bindings <= set(bindings) or any(bindings.get(key) is not True for key in required_bindings):
        return False

    planning = state.get("metric_planning_decision") or {}
    presentation = state.get("analysis_presentation") or {}
    answer = str(state.get("answer") or state.get("final_answer") or "")
    if not isinstance(planning, dict) or planning.get("status") != "confirmed":
        return False
    if not isinstance(presentation, dict) or presentation.get("status") not in {"partial", "completed"}:
        return False
    # No numeric fact/card may be approved through this branch. The answer
    # must explicitly state that the requested period has no available value.
    facts = presentation.get("facts") or []
    if any(isinstance(fact, dict) and fact.get("value") is not None for fact in facts):
        return False
    if presentation.get("changes") or presentation.get("contributions") or presentation.get("metric_decomposition"):
        return False
    if not any(marker in answer.casefold() for marker in ("无数据", "暂无数据", "未找到数据", "no data")):
        return False
    return True


def _validated_contribution_capability_stop(
    state: AgentState,
    request_contract: dict[str, object],
    request_contract_review: dict[str, object],
) -> dict[str, object] | None:
    """Return a verified, non-executed capability refusal for this contract.

    A refusal is not contribution evidence and does not complete the analysis.
    It can, however, satisfy the user's task when the requested registered
    capability is explicitly unavailable and the system preserves that limit.
    Every relevant execution boundary is checked here so a generic
    ``fail_closed`` result cannot be mistaken for a valid capability stop.
    """
    metric_id = request_contract.get("metric_id")
    dimension = request_contract.get("dimension")
    if (
        request_contract.get("operation_intent") == "dimensional_statistical_contribution"
        and request_contract_review.get("status") == "fail_closed"
        and request_contract_review.get("planning_status") == "unavailable_requested_dimension"
        and metric_id and dimension
    ):
        limitation = _validated_requested_dimension_stop(state, request_contract)
        if limitation is not None:
            return limitation
    if (
        request_contract.get("operation_intent") != "dimensional_statistical_contribution"
        or request_contract_review.get("status") != "fail_closed"
        or request_contract_review.get("planning_status") != "unsupported"
        or not metric_id
        or not dimension
        or state.get("terminal_code") != "CAPABILITY_GAP"
    ):
        return None
    execution_error_codes = {
        "database_error", "execution_failed", "token_budget_exceeded",
        "investigation_execution_failed", "tool_timeout", "provider_error",
    }
    if any(
        isinstance(error, dict)
        and (
            str(error.get("code") or "") in execution_error_codes
            or str(error.get("code") or "").startswith("contract_violation_")
        )
        for error in (state.get("errors") or [])
    ):
        return None

    decision = state.get("metric_planning_decision") or {}
    if not isinstance(decision, dict) or any((
        decision.get("status") != "unsupported",
        decision.get("reason_code") not in {"unsupported_dimension", "period_contribution_not_registered"},
        decision.get("operation_intent") != "dimensional_statistical_contribution",
        decision.get("query_type") != "metric_contribution",
        decision.get("dimension") != dimension,
        list(decision.get("confirmed_metrics") or []) != [metric_id],
    )):
        return None

    batch = state.get("metric_query_batch_plan") or {}
    plans = batch.get("plans") if isinstance(batch, dict) else None
    if plans or state.get("request_contract_query_reservations") or state.get("query_results"):
        return None

    result_envelope = state.get("tool_result") or {}
    if not isinstance(result_envelope, dict) or result_envelope.get("request_id") is None:
        return None
    result = result_envelope.get("result") or {}
    clarification = (
        result.get("capability_boundary") or result.get("clarification")
        if isinstance(result, dict) else None
    )
    if (
        not isinstance(result, dict)
        or result.get("status") != "unsupported"
        or not isinstance(clarification, dict)
        or clarification.get("status") != "unsupported"
        or clarification.get("reason_code") != decision.get("reason_code")
        or clarification.get("operation_intent") != request_contract.get("operation_intent")
        or clarification.get("query_type") != "metric_contribution"
        or clarification.get("dimension") != dimension
        or list(clarification.get("confirmed_metrics") or []) != [metric_id]
    ):
        return None

    request_id = str(result_envelope.get("request_id"))
    calls = [item for item in (state.get("tool_calls") or []) if isinstance(item, dict)]
    control_events = [
        item for item in (state.get("control_events") or [])
        if isinstance(item, dict)
        and item.get("event_type") == "capability_preflight_stop"
        and item.get("request_id") == request_id
    ]
    if control_events:
        if len(control_events) != 1 or calls:
            return None
        control_event = control_events[0]
    else:
        # Backward compatibility for traces generated before capability stops
        # were recorded as separate control events.
        if len(calls) != 1 or any((
            calls[0].get("server") != "metric-registry-planner",
            calls[0].get("tool_name") != "metric_clarification",
            calls[0].get("request_id") != request_id,
            calls[0].get("status") != "unsupported",
        )):
            return None
        control_event = None

    question_hash = sha256(str(state.get("user_message") or "").encode()).hexdigest()
    records = [
        item for item in (state.get("decision_records") or [])
        if isinstance(item, dict)
        and item.get("request_id") == request_id
        and item.get("limitation_code") == "CAPABILITY_GAP"
    ]
    if len(records) != 1:
        return None
    record = records[0]
    if any((
        record.get("metric_id") != metric_id,
        record.get("dimension") != dimension,
        record.get("operation") != "metric_contribution",
        record.get("authorization_checked") is not True,
        record.get("registry_capability_checked") is not True,
        record.get("sql_executed") is not False,
        record.get("question_sha256") != question_hash,
        control_event is not None and any((
            control_event.get("metric_id") != metric_id,
            control_event.get("dimension") != dimension,
            control_event.get("operation") != "metric_contribution",
            control_event.get("scope") != record.get("scope"),
            control_event.get("authorization_checked") is not True,
            control_event.get("registry_capability_checked") is not True,
            control_event.get("sql_executed") is not False,
            control_event.get("decision_record_id") != record.get("decision_id"),
        )),
        control_event is None and (
            not isinstance(calls[0].get("arguments"), dict)
            or record.get("scope") != calls[0]["arguments"].get("scope")
        ),
    )):
        return None

    presentation = state.get("analysis_presentation") or {}
    semantic = presentation.get("semantic_interpretation") if isinstance(presentation, dict) else None
    if (
        not isinstance(presentation, dict)
        or presentation.get("status") != "unsupported"
        or not isinstance(semantic, dict)
        or semantic.get("metric_id") != metric_id
        or semantic.get("dimension") != dimension
        or presentation.get("facts")
        or presentation.get("contributions")
        or presentation.get("changes")
        or presentation.get("metric_decomposition")
    ):
        return None

    answer = str(state.get("answer") or state.get("final_answer") or "").strip()
    reason = str(decision.get("reason") or "").strip()
    if not answer or not reason or reason not in answer:
        return None

    return {
        "outcome": "capability_limited",
        "terminal_code": "CAPABILITY_GAP",
        "metric_id": metric_id,
        "dimension": dimension,
        "operation": "metric_contribution",
        "comparison": request_contract.get("comparison"),
        "reason_code": str(decision.get("reason_code") or "unsupported_dimension"),
        "reason": reason,
        "request_id": request_id,
        "authorization_checked": True,
        "registry_capability_checked": True,
        "sql_executed": False,
        "analysis_completed": False,
    }


def _validated_requested_dimension_stop(
    state: AgentState,
    request_contract: dict[str, object],
) -> dict[str, object] | None:
    """Validate a policy-backed early stop for an explicitly requested dimension.

    The request has been parsed and checked against server-owned capability and
    authorization state, but no dimension query ran. This is a truthful,
    bounded terminal outcome; it must not be misclassified as an incomplete
    evidence sequence or as successful contribution analysis.
    """
    metric_id = str(request_contract.get("metric_id") or "")
    dimension = str(request_contract.get("dimension") or "")
    if (
        request_contract.get("operation_intent") != "dimensional_statistical_contribution"
        or not metric_id or not dimension
        or dimension not in set(map(str, state.get("unavailable_requested_dimensions") or []))
    ):
        return None

    investigation = state.get("investigation_state") or {}
    stop = investigation.get("stop_reason") if isinstance(investigation, dict) else None
    stop_code = str((stop or {}).get("code") or "") if isinstance(stop, dict) else ""
    if stop_code not in {"CAPABILITY_GAP", "AUTHORIZATION_BLOCKED"}:
        return None
    if state.get("terminal_code") not in {None, stop_code}:
        return None
    if any(
        isinstance(error, dict)
        and (
            str(error.get("code") or "").startswith("contract_violation_")
            or str(error.get("code") or "") in {
                "database_error", "execution_failed", "token_budget_exceeded",
                "investigation_execution_failed", "tool_timeout", "provider_error",
            }
        )
        for error in (state.get("errors") or [])
    ):
        return None

    gaps = [
        item for item in (state.get("diagnostic_capability_gaps") or [])
        if isinstance(item, dict)
        and item.get("kind") == "requested_dimension"
        and item.get("metric_id") == metric_id
        and item.get("requested_dimension") == dimension
        and item.get("code") == stop_code
    ]
    if len(gaps) != 1:
        return None

    # A policy refusal cannot stand in for evidence. Reject this exception if
    # the run executed a dimension contribution or emitted any contribution.
    action_results = investigation.get("action_results") or [] if isinstance(investigation, dict) else []
    if any(
        isinstance(item, dict)
        and item.get("action_type") in {"BREAKDOWN_DIMENSION", "DRILL_DOWN"}
        for item in action_results
    ):
        return None
    evidence = investigation.get("evidence") or [] if isinstance(investigation, dict) else []
    if any(
        isinstance(item, dict) and item.get("evidence_type") == "dimension_contribution"
        for item in evidence
    ):
        return None
    presentation = state.get("analysis_presentation") or {}
    if (
        not isinstance(presentation, dict)
        or presentation.get("status") != "partial"
        or presentation.get("facts")
        or presentation.get("contributions")
    ):
        return None
    if (state.get("analysis_result") or {}).get("abstain") is not True:
        return None

    return {
        "outcome": "capability_limited",
        "terminal_code": stop_code,
        "metric_id": metric_id,
        "dimension": dimension,
        "operation": "metric_contribution",
        "comparison": request_contract.get("comparison"),
        "reason_code": "requested_dimension_unavailable",
        "reason": str(gaps[0].get("reason") or ""),
        "authorization_checked": stop_code == "AUTHORIZATION_BLOCKED",
        "registry_capability_checked": stop_code == "CAPABILITY_GAP",
        "sql_executed": False,
        "analysis_completed": False,
    }


def _review_result_completeness(
    state: AgentState,
    *,
    request_contract_review: dict[str, object] | None = None,
) -> dict[str, object]:
    """Check requested dimensions without turning unavailable data into a loop."""

    compilation = state.get("context_compilation") or {}
    request_contract = compilation.get("request_contract") or {}
    if request_contract.get("operation_intent") == "dimensional_statistical_contribution":
        # A completed contribution requires contract-bound evidence. A
        # separately verified CAPABILITY_GAP can satisfy the request only as a
        # bounded refusal; it never stands in for a completed analysis.
        contract_review = request_contract_review or _review_request_contract(state)
        if contract_review.get("status") != "passed":
            capability_stop = _validated_contribution_capability_stop(
                state, request_contract, contract_review,
            )
            if capability_stop is not None:
                return {
                    "status": "approved",
                    "outcome": "capability_limited",
                    "analysis_completed": False,
                    "issues": [],
                    "capability_limitation": capability_stop,
                }
            return {
                "status": "rejected",
                "issues": ["The requested statistical-contribution evidence is not completely validated and bound."],
                "request_contract_status": contract_review.get("status"),
            }
        return {
            "status": "approved", "issues": [],
            "dimension": request_contract.get("dimension"),
            "contract_hash": request_contract.get("contract_hash"),
        }

    if state.get("task_type") != "anomaly_analysis":
        return {"status": "skipped", "issues": []}

    requested = list(state.get("requested_dimensions") or [])
    if not requested:
        try:
            from app.graph.nodes.root_cause import infer_requested_dimensions
            requested = infer_requested_dimensions(str(state.get("user_message") or ""))
        except Exception:
            requested = []
    if not requested:
        return {"status": "skipped", "issues": []}

    unavailable = set(state.get("unavailable_requested_dimensions") or [])
    if unavailable:
        gaps = [
            item for item in state.get("diagnostic_capability_gaps", [])
            if isinstance(item, dict) and item.get("kind") == "requested_dimension"
        ]
        recorded = {str(item.get("requested_dimension")) for item in gaps}
        if unavailable <= recorded and (state.get("analysis_result") or {}).get("abstain") is True:
            return {
                "status": "inconclusive",
                "issues": [],
                "requested_dimensions": requested,
                "unavailable_requested_dimensions": sorted(unavailable),
            }

    completeness = state.get("result_completeness") or {}
    if completeness.get("status") == "inconclusive":
        # The truthful result is a bounded limitation, not another query.
        return {"status": "inconclusive", "issues": [], "requested_dimensions": requested}

    contributions = (state.get("analysis_result") or {}).get("contribution") or []
    missing: list[str] = []
    for dimension in requested:
        matches = [item for item in contributions if item.get("dimension") == dimension]
        if not matches:
            missing.append(dimension)
            continue
        # Diagnostic finalize emits evidence_ids at the dimension-card level
        # and one row per dimension value. Requiring a flattened
        # top_contributor shape rejected valid explicit-dimension results and
        # sent them back through repeated planning.
        invalid_card = False
        for card in matches:
            rows = card.get("items") or []
            if (
                card.get("total_delta") is None
                or not card.get("evidence_ids")
                or not rows
                or any(
                    not isinstance(row, dict)
                    or any(row.get(field) in (None, "") for field in (
                        "dimension_value", "current_value", "baseline_value", "delta_value",
                    ))
                    for row in rows
                )
            ):
                invalid_card = True
                break
        if invalid_card:
            missing.append(dimension)

    if missing:
        return {
            "status": "rejected",
            "issues": [f"Missing verifiable result fields for dimensions: {', '.join(missing)}"],
            "missing_dimensions": missing,
        }
    return {"status": "approved", "issues": [], "requested_dimensions": requested}


def _review_causal_language(state: AgentState) -> dict[str, object]:
    """Reject unsupported causal assertions while allowing correlation language."""

    registered_decomposition = (
        state.get("task_type") == "metric_query"
        and _registered_numeric_decomposition_request(state) is not None
    )
    if state.get("task_type") != "anomaly_analysis" and not registered_decomposition:
        return {"status": "skipped", "issues": []}
    answer = str(state.get("answer") or "")
    # This path currently produces observational attribution only. A lexical
    # guard catches positive causal claims but cannot authorize causal claims.
    if unsupported_causal_assertions(answer):
        return {"status": "rejected", "issues": ["答案包含没有证据支持的因果断言。"]}
    return {"status": "approved", "issues": []}


def _review_periodic_report(state: AgentState) -> dict[str, object]:
    payload = state.get("periodic_report")
    if not payload:
        return {"status": "skipped", "issues": []}
    issues: list[str] = []
    try:
        report = PeriodicReport.model_validate(payload)
        validate_natural_period(report.period)
    except Exception as exc:
        return {"status": "rejected", "issues": [f"Invalid periodic report contract: {exc}"]}
    metric_evidence = {item for metric in report.metrics for item in metric.evidence_ids}
    known_evidence = set(report.evidence_records)
    if any(metric.period_over_period is None or metric.year_over_year is None for metric in report.metrics):
        issues.append("Every report metric requires period-over-period and year-over-year comparisons.")
    for metric in report.metrics:
        if any(item not in known_evidence for item in metric.evidence_ids):
            issues.append(f"Metric {metric.metric_name} references missing evidence.")
        pop, yoy = metric.period_over_period, metric.year_over_year
        if pop:
            if pop.current_period != report.period or pop.baseline_period != previous_period(report.period):
                issues.append(f"Metric {metric.metric_name} has an invalid period-over-period interval.")
            if not _same_number(metric.value, pop.current_value):
                issues.append(f"Metric {metric.metric_name} value disagrees with period-over-period evidence.")
            if pop.current_value is not None and pop.baseline_value is not None:
                expected_delta = round(pop.current_value - pop.baseline_value, 2)
                if not _same_number(pop.delta_value, expected_delta):
                    issues.append(f"Metric {metric.metric_name} period-over-period delta is inconsistent.")
        if yoy:
            if yoy.current_period != report.period or yoy.baseline_period != year_ago_period(report.period):
                issues.append(f"Metric {metric.metric_name} has an invalid year-over-year interval.")
            if not _same_number(metric.value, yoy.current_value):
                issues.append(f"Metric {metric.metric_name} value disagrees with year-over-year evidence.")
        if metric.target_comparison and not _same_number(metric.value, metric.target_comparison.current_value):
            issues.append(f"Metric {metric.metric_name} value disagrees with target evidence.")
    for chart in report.charts:
        if any(item not in known_evidence for item in chart.evidence_ids):
            issues.append(f"Chart {chart.chart_type} references missing evidence.")
        if chart.status == "ok" and chart.path and chart.content_hash:
            path = Path(chart.path)
            if not path.is_file():
                issues.append(f"Chart {chart.chart_type} file does not exist.")
            elif sha256(path.read_bytes()).hexdigest() != chart.content_hash:
                issues.append(f"Chart {chart.chart_type} content hash is invalid.")
        if chart.chart_type == "trend" and not set(chart.evidence_ids).intersection(metric_evidence):
            issues.append("Trend chart is not bound to the report metric evidence.")
        if chart.chart_type == "contribution" and not any(
            set(chart.evidence_ids).intersection(item.evidence_ids) for item in report.contributions
        ):
            issues.append("Contribution chart is not bound to contribution evidence.")
    for anomaly in report.anomalies:
        if any(item not in known_evidence for item in anomaly.evidence_ids):
            issues.append(f"Anomaly {anomaly.metric_name}/{anomaly.metric_date} references missing evidence.")
    for contribution in report.contributions:
        if any(item not in known_evidence for item in contribution.evidence_ids):
            issues.append(f"Contribution {contribution.metric_name}/{contribution.dimension} references missing evidence.")
    if report.data_freshness_at is None:
        issues.append("Report data freshness is missing.")
    return {"status": "approved" if not issues else "rejected", "issues": issues}


def _validate_server_diagnostic_contract_sequence(
    state: AgentState, contract: dict[str, Any],
) -> tuple[str | None, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Validate a compiler-required diagnostic sequence against its real evidence.

    This is a distinct planner shape from metric_query's MetricQueryBatchPlan:
    the server generates required trend/contribution steps in diagnostic_plan.
    It must still prove the same contract, tool calls, authorized plans and
    evidence before the Critic can accept the contribution.
    """
    sequence = contract.get("required_evidence_sequence")
    if not (
        state.get("task_type") == "anomaly_analysis"
        and sequence == ["trend", "dimension_contribution"]
        and isinstance(state.get("diagnostic_plan"), dict)
    ):
        return None

    def invalid(code: str):
        return code, {}, [], []

    plan = state.get("diagnostic_plan") or {}
    steps = [item for item in (plan.get("steps") or []) if isinstance(item, dict) and item.get("required") is True]
    kinds = [str(item.get("kind") or "") for item in steps]
    if kinds.count("trend") != 1 or kinds.count("dimension_contribution") != 1:
        return invalid("contract_violation_diagnostic_required_steps")
    step_state = state.get("diagnostic_step_state") or {}
    if any(step_state.get(step.get("step_id")) != "completed" for step in steps):
        return invalid("contract_violation_diagnostic_required_step_incomplete")

    expected_metric = str(contract.get("metric_id") or "")
    comparison = contract.get("comparison") or {}
    if not expected_metric or comparison.get("kind") not in {"day_pair", "period_pair"}:
        return invalid("contract_violation_diagnostic_contract_incomplete")

    if comparison.get("kind") == "day_pair":
        expected_trend = {
            "start_date": comparison.get("baseline_date"),
            "end_date": comparison.get("current_date"),
        }
        expected_contribution = {
            "baseline_date": comparison.get("baseline_date"),
            "current_date": comparison.get("current_date"),
        }
    else:
        baseline = comparison.get("baseline") or {}
        current = comparison.get("current") or {}
        expected_trend = {
            "start_date": baseline.get("start_date"),
            "end_date": current.get("end_date"),
        }
        expected_contribution = {
            "baseline_start": baseline.get("start_date"),
            "baseline_end": baseline.get("end_date"),
            "current_start": current.get("start_date"),
            "current_end": current.get("end_date"),
            "granularity": comparison.get("granularity"),
        }

    evidence_bundles = [
        item for item in (state.get("evidence_bundles") or [])
        if isinstance(item, dict) and item.get("trusted") is True
    ]
    calls = [item for item in (state.get("tool_calls") or []) if isinstance(item, dict)]
    trend_matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    contribution_matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for bundle in evidence_bundles:
        content = bundle.get("content") if isinstance(bundle.get("content"), dict) else {}
        result = content.get("result") if isinstance(content.get("result"), dict) else {}
        parsed = result.get("parsed") if isinstance(result.get("parsed"), dict) else {}
        authorized = parsed.get("authorized_metric") if isinstance(parsed.get("authorized_metric"), dict) else {}
        if not (
            result.get("status") == "ok"
            and parsed.get("metric_name") == expected_metric
            and authorized.get("metric_id") == expected_metric
        ):
            continue
        source_ref = str(bundle.get("source_ref") or "")
        if parsed.get("query_type") == "metric_trend":
            trend_result = result.get("trend_result") if isinstance(result.get("trend_result"), dict) else {}
            if (
                source_ref not in {"registry_trend", "query_metric_trend"}
                or trend_result.get("status") != "ok"
                or trend_result.get("start_date") != expected_trend["start_date"]
                or trend_result.get("end_date") != expected_trend["end_date"]
            ):
                continue
            matched_calls = [
                call for call in calls
                if call.get("evidence_id") == bundle.get("evidence_id")
                and call.get("tool_name") == source_ref
                and call.get("status") == "ok"
                and (call.get("arguments") or {}).get("metric_name") == expected_metric
            ]
            if len(matched_calls) == 1:
                trend_matches.append((bundle, matched_calls[0]))
        elif parsed.get("query_type") == "metric_contribution":
            contribution = result.get("contribution_result") if isinstance(result.get("contribution_result"), dict) else {}
            if (
                source_ref != contract.get("required_tool")
                or parsed.get("dimension") != contract.get("dimension")
                or contribution.get("status") != "ok"
            ):
                continue
            parsed_period = {key: parsed.get(key) for key in expected_contribution}
            if parsed_period != expected_contribution:
                continue
            matched_calls = []
            for call in calls:
                args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
                call_comparison = args.get("comparison") if isinstance(args.get("comparison"), dict) else args
                call_identity = {
                    key: call_comparison.get(key) for key in expected_contribution
                }
                if (
                    call.get("evidence_id") == bundle.get("evidence_id")
                    and call.get("tool_name") == source_ref
                    and call.get("status") == "ok"
                    and args.get("metric_name") == expected_metric
                    and args.get("dimension") == contract.get("dimension")
                    and call_identity == expected_contribution
                ):
                    matched_calls.append(call)
            if len(matched_calls) == 1:
                contribution_matches.append((bundle, matched_calls[0]))

    if len(trend_matches) != 1:
        return invalid("contract_violation_diagnostic_trend_evidence")
    if len(contribution_matches) != 1:
        return invalid("contract_violation_diagnostic_contribution_evidence")
    parsed_contribution = (
        contribution_matches[0][0].get("content", {}).get("result", {}).get("parsed") or {}
    )
    decision = {
        "operation_intent": "dimensional_statistical_contribution",
        "query_type": "metric_contribution",
        "dimension": contract.get("dimension"),
        "status": "confirmed",
        "confirmed_metrics": [expected_metric],
    }
    return None, decision, [parsed_contribution], [contribution_matches[0][1]]


def _review_request_contract(
    state: AgentState, *, require_output_binding: bool = True, record_observation: bool = True,
) -> dict[str, object]:
    """Review the compiler snapshot against planner, tool and final citations."""
    compilation = state.get("context_compilation") or {}
    contract = compilation.get("request_contract") if isinstance(compilation, dict) else None
    if not isinstance(contract, dict):
        return {"status": "skipped", "issues": [], "reason": "no_compiled_request_contract"}
    from app.context.request_contract import (
        _contribution_execution_tool_matches, contribution_plan_mismatch, record_contract_observation,
        validate_compiled_request_contract,
    )

    def observe(payload: dict[str, object]) -> None:
        if record_observation:
            record_contract_observation(state, "critic", payload)

    valid, reason = validate_compiled_request_contract(
        compilation,
        state_contract_hash=state.get("request_contract_hash"),
        require_state_anchor=True,
    )
    expected_clarification = bool(
        compilation.get("needs_clarification")
        and contract.get("parse_status") == "clarification_required"
        and reason == contract.get("uncertainty_reason")
    )
    if not valid and expected_clarification:
        result = {
            "status": "clarification_required", "issues": [],
            "reason": reason, "contract_hash": contract.get("contract_hash"),
        }
        observe({"status": result["status"], "reason": reason})
        return result
    if not valid:
        result = {
            "status": "rejected", "issues": [reason or "request_contract_invalid"],
            "reason": reason, "contract_hash": contract.get("contract_hash"),
        }
        observe({"status": result["status"], "reason": reason})
        return result

    if contract.get("operation_intent") != "dimensional_statistical_contribution":
        result = {"status": "passed", "issues": [], "contract_hash": contract.get("contract_hash")}
        observe({"status": "passed"})
        return result

    requested_dimension_stop = _validated_requested_dimension_stop(state, contract)
    if requested_dimension_stop is not None:
        result = {
            "status": "fail_closed",
            "issues": [],
            "planning_status": "unavailable_requested_dimension",
            "contract_hash": contract.get("contract_hash"),
            "capability_limitation": requested_dimension_stop,
        }
        observe({"status": "fail_closed", "reason": "unavailable_requested_dimension"})
        return result

    diagnostic_validation = _validate_server_diagnostic_contract_sequence(state, contract)
    if diagnostic_validation is None:
        decision = state.get("metric_planning_decision") or {}
        plans = (state.get("metric_query_batch_plan") or {}).get("plans") or []
        matched_calls: list[dict[str, object]] | None = None
    else:
        mismatch, decision, plans, matched_calls = diagnostic_validation
        if mismatch:
            result = {"status": "rejected", "issues": [mismatch], "contract_hash": contract.get("contract_hash")}
            observe({"status": "rejected", "reason": mismatch})
            return result

    mismatch = contribution_plan_mismatch(contract, decision, plans)
    if mismatch:
        result = {"status": "rejected", "issues": [mismatch], "contract_hash": contract.get("contract_hash")}
        observe({"status": "rejected", "reason": mismatch})
        return result
    if decision.get("status") != "confirmed":
        result = {
            "status": "fail_closed", "issues": [],
            "planning_status": decision.get("status"),
            "contract_hash": contract.get("contract_hash"),
        }
        observe({"status": "fail_closed"})
        return result

    expected_metric = contract.get("metric_id")
    calls = [item for item in (state.get("tool_calls") or []) if isinstance(item, dict)]
    if matched_calls is None:
        matched_calls = [
            item for item in calls
            if item.get("planned_metric_id") == expected_metric
            and item.get("planned_operation") == "metric_contribution"
            and (item.get("arguments") or {}).get("dimension") == contract.get("dimension")
        ]
    if not matched_calls:
        result = {"status": "rejected", "issues": ["contract_violation_tool_call_missing_or_mismatched"], "contract_hash": contract.get("contract_hash")}
        observe({"status": "rejected", "reason": result["issues"][0]})
        return result

    matching_plans = [
        plan for plan in plans
        if isinstance(plan, dict)
        and plan.get("metric_name") == expected_metric
        and plan.get("query_type") == "metric_contribution"
        and plan.get("dimension") == contract.get("dimension")
    ]
    if len(matching_plans) != 1:
        result = {"status": "rejected", "issues": ["contract_violation_plan_binding_missing_or_ambiguous"], "contract_hash": contract.get("contract_hash")}
        observe({"status": "rejected", "reason": result["issues"][0]})
        return result
    matching_plan = matching_plans[0]

    evidence_ids: set[str] = set()
    comparison = contract.get("comparison") or {}
    for bundle in (state.get("evidence_bundles") or []):
        if not isinstance(bundle, dict) or bundle.get("trusted") is not True:
            continue
        evidence_id = str(bundle.get("evidence_id") or "")
        content = bundle.get("content") or {}
        if not evidence_id or not isinstance(content, dict):
            continue
        result = content.get("result") or {}
        parsed = result.get("parsed") or {}
        contribution = result.get("contribution_result") or {}
        metric_id = content.get("metric_id") or parsed.get("metric_name") or contribution.get("metric_name")
        query_type = content.get("query_type") or parsed.get("query_type")
        dimension = parsed.get("dimension") or contribution.get("dimension")
        if (
            result.get("status") != "ok"
            or metric_id != expected_metric
            or query_type != "metric_contribution"
            or dimension != contract.get("dimension")
        ):
            continue
        if comparison.get("kind") == "day_pair":
            if (
                parsed.get("baseline_date") != comparison.get("baseline_date")
                or parsed.get("current_date") != comparison.get("current_date")
            ):
                continue
        elif comparison.get("kind") == "period_pair":
            baseline = comparison.get("baseline") or {}
            current = comparison.get("current") or {}
            if any((
                parsed.get("baseline_start") != baseline.get("start_date"),
                parsed.get("baseline_end") != baseline.get("end_date"),
                parsed.get("current_start") != current.get("start_date"),
                parsed.get("current_end") != current.get("end_date"),
                parsed.get("granularity") != comparison.get("granularity"),
            )):
                continue
        else:
            continue

        # A trusted evidence bundle must also be backed by the exact executed
        # tool call. Initial query results bind through request_id; diagnostic
        # follow-ups bind through the evidence_id recorded on ToolInvocation.
        source_request_id = str(content.get("request_id") or "")
        source_ref = str(bundle.get("source_ref") or "")
        def call_matches_contract(call: object) -> bool:
            if not isinstance(call, dict):
                return False
            call_args = call.get("arguments") or {}
            if not (
                (
                    str(call.get("evidence_id") or "") == evidence_id
                    or (source_request_id and str(call.get("request_id") or "") == source_request_id)
                )
                and call.get("tool_name") == source_ref
                and _contribution_execution_tool_matches(contract, matching_plan, call)
                and call_args.get("metric_name") == expected_metric
                and call_args.get("dimension") == contract.get("dimension")
            ):
                return False
            if comparison.get("kind") == "day_pair":
                date_args = call_args.get("comparison") or call_args
                return (
                    date_args.get("baseline_date") == comparison.get("baseline_date")
                    and date_args.get("current_date") == comparison.get("current_date")
                )
            baseline = comparison.get("baseline") or {}
            current = comparison.get("current") or {}
            comparison_args = call_args.get("comparison") or call_args
            return all((
                comparison_args.get("baseline_start") == baseline.get("start_date"),
                comparison_args.get("baseline_end") == baseline.get("end_date"),
                comparison_args.get("current_start") == current.get("start_date"),
                comparison_args.get("current_end") == current.get("end_date"),
                comparison_args.get("granularity") == comparison.get("granularity"),
            ))

        call_is_bound = any(call_matches_contract(call) for call in calls)
        if not call_is_bound:
            continue
        evidence_ids.add(evidence_id)
    cited_ids = {
        str(evidence_id)
        for claim in (state.get("claims") or []) if isinstance(claim, dict)
        for evidence_id in (claim.get("source_evidence_ids") or [])
    }
    cited_ids.update(
        str(item.get("evidence_id"))
        for item in ((state.get("analysis_presentation") or {}).get("evidence") or [])
        if isinstance(item, dict) and item.get("evidence_id")
    )
    if not evidence_ids:
        result = {"status": "rejected", "issues": ["contract_violation_contribution_evidence_missing"], "contract_hash": contract.get("contract_hash")}
    elif require_output_binding and not (evidence_ids & cited_ids):
        result = {"status": "rejected", "issues": ["contract_violation_contribution_evidence_not_bound_to_output"], "contract_hash": contract.get("contract_hash")}
    else:
        result = {
            "status": "passed", "issues": [], "contract_hash": contract.get("contract_hash"),
            "contribution_evidence_ids": sorted(evidence_ids),
        }
    observe({"status": result["status"], "issues": result["issues"]})
    return result


def _run_optional_llm_review(review_fn, *args, state: AgentState, **kwargs) -> dict[str, Any]:
    """Capture one review result without treating a missing verdict as a pass."""
    # Test doubles and older review callables may not accept WP1 context
    # keywords; omit a keyword only when that callable cannot receive it.
    context_keywords = {"context_text", "request_text", "budget_optional"} & set(kwargs)
    if context_keywords:
        import inspect

        params = inspect.signature(review_fn).parameters
        accepts_kwargs = any(
            param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()
        )
        if not accepts_kwargs:
            for name in context_keywords:
                if name not in params:
                    kwargs.pop(name)
    calls = state.get("llm_calls") or []
    start = len(calls)
    try:
        raw = review_fn(*args, **kwargs)
    except Exception as exc:  # preserve the failure as review state, not approval
        raw = None
        raised = type(exc).__name__
    else:
        raised = None
    calls = (state.get("llm_calls") or [])[start:]
    call_ids = [str(item.get("call_id")) for item in calls if item.get("call_id")]
    attempt_ids = [
        str(attempt.get("attempt_id"))
        for item in calls for attempt in (item.get("attempts") or [])
        if attempt.get("attempt_id")
    ]
    if isinstance(raw, dict) and raw.get("verdict") in {"passed", "failed"} and isinstance(raw.get("issues", []), list):
        issues = raw.get("issues", [])
        if not all(isinstance(item, (str, dict)) for item in issues):
            return {
                **raw, "verdict": "not_completed", "completion_status": "not_completed",
                "failure_kind": "invalid_output", "call_ids": call_ids,
                "sdk_attempt_ids": attempt_ids,
            }
        return {
            **raw, "completion_status": "completed", "failure_kind": None,
            "call_ids": call_ids, "sdk_attempt_ids": attempt_ids,
        }

    latest = calls[-1] if calls else {}
    outcome = str(latest.get("outcome") or "")
    attempts = latest.get("attempts") or []
    exceptions = [str(item.get("exception_type") or "") for item in attempts]
    if raised:
        failure_kind = "review_execution_error"
    elif outcome == "not_sent" and latest.get("not_sent_reason") == "soft_budget_limit":
        failure_kind = "not_run_due_soft_limit"
    elif outcome == "not_sent":
        failure_kind = "not_sent"
    elif outcome in {"parse_failure", "empty_response", "processing_error"}:
        failure_kind = "invalid_output"
    elif outcome == "truncated":
        failure_kind = "truncated"
    elif outcome == "reasoning_only":
        failure_kind = "reasoning_only"
    elif outcome == "provider_error":
        failure_kind = "timeout" if any("timeout" in item.casefold() for item in exceptions) else "provider_error"
    elif outcome == "configuration_error":
        failure_kind = "configuration_error"
    else:
        failure_kind = "missing_or_invalid_verdict"
    reason = (
        latest.get("not_sent_reason")
        or (attempts[-1].get("exception_type") if attempts else None)
        or raised
    )
    return {
        "verdict": "not_completed", "completion_status": "not_completed",
        "failure_kind": failure_kind, "reason": reason, "issues": [],
        "call_ids": call_ids, "sdk_attempt_ids": attempt_ids,
    }


def _required_llm_review_names(task_type: str, registered_decomposition_review: bool) -> set[str]:
    """Return the server-owned minimum LLM review set after deterministic checks.

    Numeric, evidence-binding, and contract checks remain deterministic. The
    three LLM reviews are not a universal mandatory bundle: groundedness is
    the minimum language review, while completeness/consistency are advisory
    and may be skipped when the task soft budget is reached.
    """
    task_required = {
        "metric_query": {"groundedness"},
        "anomaly_analysis": {"groundedness"},
        "report_generation": {"groundedness"},
    }
    required = set(task_required.get(task_type, {"groundedness"}))
    if registered_decomposition_review:
        required.add("groundedness")
    return required


def _review_not_run(reason: str) -> dict[str, Any]:
    return {
        "verdict": "not_run", "completion_status": "not_run",
        "failure_kind": reason, "issues": [], "call_ids": [], "sdk_attempt_ids": [],
    }


def _review_not_completed(reason: str) -> dict[str, Any]:
    return {
        "verdict": "not_completed", "completion_status": "not_completed",
        "failure_kind": reason, "issues": [], "call_ids": [], "sdk_attempt_ids": [],
    }


def _critic_context_limit_supports_human_review(
    task_type: str, state: AgentState,
) -> bool:
    """Only route budget failures to the durable report approval workflow."""
    return bool(
        task_type == "report_generation"
        and state.get("report_id")
        and state.get("report_version")
    )


def _completed_review_budget_reserve(state: AgentState, review: dict[str, Any]) -> int:
    """Return the conservative preflight cost of a completed review request."""
    call_ids = {str(item) for item in (review.get("call_ids") or []) if item}
    for call in reversed(state.get("llm_calls") or []):
        if str(call.get("call_id") or "") not in call_ids:
            continue
        preflight = call.get("token_budget_preflight") or {}
        if preflight.get("allowed") is True:
            try:
                return max(0, int(preflight.get("required_tokens", 0) or 0))
            except (TypeError, ValueError, OverflowError):
                return 0
    return 0


def _optional_review_fits_with_required_followup_reserve(
    state: AgentState, *, optional_call_reserve: int, required_call_reserve: int,
) -> bool:
    """Keep optional reviews from consuming the budget for one required retry."""
    budget = state.get("runtime_budget") or {}
    usage = state.get("llm_usage") or {}
    try:
        limit = int(budget.get("max_tokens", 0) or 0)
        consumed = int(usage.get("input_tokens", 0) or 0) + int(
            usage.get("output_tokens", 0) or 0
        ) + int(usage.get("preflight_reserved_tokens", 0) or 0)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return True
    if limit <= 0 or required_call_reserve <= 0:
        return True
    return limit - consumed >= optional_call_reserve + required_call_reserve


def critic(
    state: AgentState,
    *,
    llm_review_mode: CriticLLMReviewMode | str = CriticLLMReviewMode.FULL,
    llm_thinking_mode: CriticLLMThinkingMode | str = CriticLLMThinkingMode.PROVIDER_DEFAULT,
) -> AgentState:
    """路径感知审核：结构化质检 + 可选的 LLM 审核。

    参数：
    - `state`：当前工作流状态。

    返回值：
    - `AgentState`：写入审核结果后的状态。
    """

    # Trace normally persists only the terminal state.  Preserve the answer
    # that Critic sees before this node can issue a revision decision.
    capture_pre_critic_answer_snapshot(state)

    # 打点入口：critic 不负责重新生成答案，而是对当前 state 做“可追溯性闸门”。
    # 第一层是确定性的结构化校验，第二层是可选 LLM 审核，第三步由策略表决定
    # approved / revision / human_review / rejected；metric_query 的 RAG 型 LLM 审核
    # 会按路径跳过，但数据结果和 claim-evidence 绑定仍必须检查。
    review_mode = CriticLLMReviewMode.parse(llm_review_mode)
    thinking_mode = CriticLLMThinkingMode.parse(llm_thinking_mode)
    thinking_summary = thinking_mode.trace_summary()
    thinking_summary["default_non_thinking"] = bool(get_settings().critic_default_non_thinking)
    thinking_summary["effective_policy"] = (
        "disabled_if_supported"
        if get_settings().critic_default_non_thinking
        or thinking_mode is CriticLLMThinkingMode.NON_THINKING_EXPERIMENT
        else "provider_default"
    )
    state["critic_llm_thinking_config"] = thinking_summary
    task_type = state.get("task_type", "")
    request_contract = (((state.get("context_compilation") or {}).get("request_contract")) or {})
    statistical_contribution_intent = (
        request_contract.get("operation_intent") == "dimensional_statistical_contribution"
    )

    # === 第一层：结构化质检 ===
    # Governed diagnostics use authorized typed query evidence; their graph
    # does not run the metric-definition RAG node. Keep its independent data,
    # trajectory, claim and causal checks without requesting absent RAG output.
    governed_diagnostic = False
    diagnostic_has_valid_evidence = False
    if task_type == "anomaly_analysis" and state.get("diagnostic_plan"):
        from app.graph.nodes.diagnostic import _completed_valid_evidence_ids
        diagnostic_has_valid_evidence = bool(_completed_valid_evidence_ids(state))
        governed_diagnostic = diagnostic_has_valid_evidence or (
            _validated_requested_dimension_stop(state, request_contract) is not None
        )
    requires_rag = (
        task_type in PATHS_REQUIRING_RAG
        and not governed_diagnostic
        and not statistical_contribution_intent
    )
    rag_review = (
        state.get("rag_review") or {"status": "skipped"}
        if requires_rag
        else {"status": "skipped"}
    )
    raw_tool_result = state.get("tool_result", {})
    raw_query_result = raw_tool_result.get("result", {}) if isinstance(raw_tool_result, dict) else {}
    batch_results = raw_tool_result.get("batch_results", []) if isinstance(raw_tool_result, dict) else []
    batch_plans = (state.get("metric_query_batch_plan") or {}).get("plans", [])
    planning_decision = state.get("metric_planning_decision") or {}
    registered_decomposition_request = _registered_numeric_decomposition_request(state)
    registered_decomposition_intent = (
        task_type == "metric_query" and registered_decomposition_request is not None
    )
    registered_decomposition_request = registered_decomposition_request or {}
    # clarification_required 表示规划阶段安全停住：实际数据工具尚未调用，
    # 因此不能把它误判成“数据查询失败”。
    data_review = (
        {
            "status": "approved",
            "source": "result_context",
            "evidence_ids": [
                str(bundle.get("evidence_id"))
                for bundle in (state.get("evidence_bundles") or [])
                if isinstance(bundle, dict) and bundle.get("evidence_id")
            ],
            "message": "Same-scope result follow-up; no new business query was expected.",
        }
        if state.get("result_followup_context")
        else _review_registered_numeric_decomposition(state, registered_decomposition_request)
        if task_type == "metric_query" and registered_decomposition_intent
        else {
            "status": "skipped",
            "source": "validated_investigation_engine",
            "evidence_ids": sorted(_completed_valid_evidence_ids(state)),
            "message": "No flat query result is expected for governed diagnostics; the action/evidence ledger is checked by diagnostic trajectory validation.",
        }
        if governed_diagnostic
        else
        {"status": "skipped", "message": "metric clarification required"}
        if isinstance(raw_query_result, dict) and raw_query_result.get("status") == "clarification_required"
        else {"status": "skipped", "message": "whitelist analysis template"}
        if isinstance(raw_query_result, dict) and raw_query_result.get("template_only")
        else review_data_query_batch(
            batch_results, batch_plans, state.get("evidence_bundles", []), state.get("tool_calls", [])
        )
        if task_type in PATHS_REQUIRING_DATA and isinstance(batch_results, list) and len(batch_results) > 1
        else review_single_data_query(
            (state.get("query_results") or [{}])[0],
            batch_plans[0],
            state.get("evidence_bundles", []),
            state.get("tool_calls", []),
        )
        if task_type in PATHS_REQUIRING_DATA
        and len(state.get("query_results") or []) == 1
        and len(batch_plans) == 1
        else review_data_query_result(raw_query_result)
        if task_type in PATHS_REQUIRING_DATA
        else {"status": "skipped"}
    )
    request_contract_review = _review_request_contract(state)
    if request_contract_review.get("status") == "rejected":
        from app.context.request_contract import record_contract_violation
        issue = str((request_contract_review.get("issues") or ["request_contract_invalid"])[0])
        record_contract_violation(state, "critic", issue)

    verified_no_data_completion = _is_bound_single_no_data_result(
        state, data_review, request_contract_review,
    )
    if verified_no_data_completion:
        # A completely bound, single-item not-found response is a truthful
        # completed answer to the query, not an incomplete batch. It carries
        # no numeric fact and does not relax any plan/evidence/contract check.
        data_review = {
            **data_review,
            "status": "approved",
            "outcome": "no_data",
            "batch_status": "no_data",
            "message": "The authorized metric query completed with an explicit no-data result.",
        }
        state["metric_batch_execution"] = {"status": "no_data"}

    # Keep the execution state aligned with the item-wise review. Critic may
    # discover a broken plan/evidence binding after the query node classified
    # the raw tool outcomes; that can raise batch severity, never lower it.
    reviewed_batch_status = data_review.get("batch_status")
    current_batch_status = (state.get("metric_batch_execution") or {}).get("status")
    known_batch_statuses = [
        status
        for status in (current_batch_status, reviewed_batch_status)
        if status in _BATCH_STATUS_SEVERITY
    ]
    if known_batch_statuses:
        final_batch_status = max(
            known_batch_statuses,
            key=_BATCH_STATUS_SEVERITY.__getitem__,
        )
        state["metric_batch_execution"] = {"status": final_batch_status}

    capability_limited_without_evidence = (
        _validated_requested_dimension_stop(state, request_contract) is not None
        and not diagnostic_has_valid_evidence
    )
    claim_review = (
        {"status": "approved", "issues": [], "coverage": "server-validated capability/authorization limitation; no factual evidence claims"}
        if capability_limited_without_evidence
        else _review_claim_bindings(state)
    )
    numeric_review = review_numeric_output(state)
    report_review = _review_periodic_report(state) if task_type == "report_generation" else {"status": "skipped", "issues": []}
    diagnostic_review = _review_diagnostic_trajectory(state) if task_type == "anomaly_analysis" else {"status": "skipped", "issues": []}
    completeness_review = _review_result_completeness(
        state, request_contract_review=request_contract_review,
    )
    causal_review = _review_causal_language(state)

    # === 第二层：LLM 审核（可选，失败时跳过） ===
    # 审核最终 answer，而非中间产物 analysis_result
    answer = state.get("answer", "") or state.get("final_answer", "") or state.get("metric_context", "") or ""
    investigation_context = _build_diagnostic_investigation_context(state)
    contribution_contract = (((state.get("context_compilation") or {}).get("request_contract")) or {})
    if contribution_contract.get("operation_intent") == "dimensional_statistical_contribution":
        investigation_context = _statistical_contribution_review_context(
            state, request_contract_review,
        )
        if investigation_context.get("review_outcome") == "capability_limited":
            # The structured limitation can ground a refusal, but must never be
            # presented as support for a metric value or contribution claim.
            pass
    ratio_answer_review = (
        _review_symmetric_contribution_answer(state, investigation_context)
        if contribution_contract.get("operation_intent") == "dimensional_statistical_contribution"
        and isinstance(investigation_context, dict)
        else {"status": "not_applicable", "issues": [], "coverage": "ratio contribution block only"}
    )
    registered_decomposition_review = registered_decomposition_intent
    if registered_decomposition_review:
        investigation_context = _registered_decomposition_review_context(
            state, registered_decomposition_request, data_review,
        )

    # 普通 metric_query 的事实来源是强类型 MCP 结果，已由上面的
    # review_data_query_result 检查；它仍跳过面向 RAG 的可选审核。显式注册数值分解
    # 使用独立的已验证分解证据上下文，因此按 full/off 模式进入原三项审核。
    optional_llm_review_applicable = task_type not in {"metric_query", "metric_definition"} or registered_decomposition_review
    optional_llm_review_off = (
        review_mode is CriticLLMReviewMode.CRITIC_LLM_REVIEW_OFF
        and (task_type == "anomaly_analysis" or registered_decomposition_review)
    )
    terminal_execution_failure = any(
        isinstance(error, dict)
        and (
            str(error.get("code") or "") in {
                "database_error", "execution_failed", "token_budget_exceeded",
                "investigation_execution_failed",
            }
            or str(error.get("code") or "").startswith("contract_violation_")
            or (
                isinstance(error.get("failure_context"), dict)
                and str(error["failure_context"].get("category") or "")
                in {"database_error", "tool_timeout", "execution_error"}
            )
        )
        for error in (state.get("errors") or [])
    )
    adaptive_policy = None
    adaptive_skip = False
    adaptive_repair = False
    adaptive_integrity_reject = False
    combined_review = None
    required_review_names = _required_llm_review_names(task_type, registered_decomposition_review)
    optional_review_stop_reason = None
    if review_mode is CriticLLMReviewMode.ADAPTIVE and optional_llm_review_applicable:
        adaptive_policy = decide_optional_llm_review(
            task_type=task_type,
            registered_decomposition_review=registered_decomposition_review,
            terminal_execution_failure=terminal_execution_failure,
            state=state,
            deterministic_reviews={
                "request_contract_review": request_contract_review,
                "data_review": data_review,
                "claim_review": claim_review,
                "numeric_review": numeric_review,
                "completeness_review": completeness_review,
                "causal_review": causal_review,
            },
        )
        adaptive_skip = adaptive_policy["action"] == "skip"
        adaptive_repair = adaptive_policy["action"] == "repair"
        adaptive_integrity_reject = adaptive_policy["action"] == "reject"
    cascade_mode = str(getattr(get_settings(), "critic_optional_review_cascade_mode", "off"))
    cascade_decision = None
    cascade_active = False
    cascade_skip_names: set[str] = set()
    if (
        optional_llm_review_applicable
        and review_mode is CriticLLMReviewMode.FULL
        and cascade_mode in {"shadow", "active"}
    ):
        cascade_decision = decide_optional_review_cascade(
            task_type=task_type,
            registered_decomposition_review=registered_decomposition_review,
            terminal_execution_failure=terminal_execution_failure,
            state=state,
            deterministic_reviews={
                "request_contract_review": request_contract_review,
                "data_review": data_review,
                "claim_review": claim_review,
                "numeric_review": numeric_review,
                "completeness_review": completeness_review,
                "causal_review": causal_review,
            },
        )
        cascade_active = (
            cascade_mode == "active"
            and cascade_decision.get("action") == "groundedness_only"
        )
        if cascade_active:
            cascade_skip_names = set(cascade_decision.get("skip_review_names") or [])
    planned_reviewers = (
        ["combined"]
        if review_mode is CriticLLMReviewMode.COMBINED_REVIEW_EXPERIMENT
        else ["groundedness"]
        if cascade_active
        else ["groundedness", "completeness", "consistency"]
    )
    # WP1: the critic receives only current evidence and the controlled
    # definition.  Historical conversation/memory is deliberately excluded;
    # the answer under review is already passed separately to each reviewer.
    from app.context.critic_evidence import build_critic_evidence_items
    from app.context.orchestrator import ContextOrchestrator, Source
    from app.context.packer import ContextBudgetExceeded
    from app.llm.routing import model_for

    # WP1 Critic source governance: verified investigation context is carried
    # as a tool_evidence/current_evidence Bundle item, not as a separate
    # INVESTIGATION_CONTEXT prompt source.
    state["critic_investigation_context"] = investigation_context
    critic_bundle = None
    critic_context_text = ""
    critic_request_text = ""
    critic_context_metadata: dict[str, Any] | None = None
    critic_context_failure: dict[str, Any] | None = None

    def prepare_critic_bundle() -> bool:
        """Pack the Critic context once, immediately before its first LLM call."""
        nonlocal critic_bundle, critic_context_text, critic_request_text
        nonlocal critic_context_metadata, critic_context_failure
        if critic_bundle is not None:
            return True
        if critic_context_failure is not None:
            return False

        summary_items, summary_metadata = build_critic_evidence_items(
            state, investigation_context if isinstance(investigation_context, dict) else None,
        )
        evidence_item_ids = {
            f"evidence.{evidence_id}"
            for evidence_id in (
                summary_metadata.get("required_evidence_ids", [])
                + summary_metadata.get("optional_evidence_ids", [])
            )
        }
        try:
            critic_bundle = ContextOrchestrator().build_specialist_bundle(
                state,
                "critic",
                instruction=str(state.get("user_message") or ""),
                model=model_for("extract")[0],
                additional_items=summary_items,
                replace_sources={Source.TOOL_EVIDENCE},
            )
        except ContextBudgetExceeded as exc:
            pack_diagnostics = dict(getattr(exc, "diagnostics", {}) or {})
            candidate_evidence_ids = sorted(
                item_id.removeprefix("evidence.") for item_id in evidence_item_ids
            )
            packer_omitted_evidence_ids = sorted(
                str(evidence_id)
                for evidence_id in (pack_diagnostics.get("omitted_evidence_ids") or [])
            )
            optional_ids = set(summary_metadata.get("optional_evidence_ids") or [])
            optional_omitted = sorted(set(packer_omitted_evidence_ids) & optional_ids)
            critic_context_metadata = {
                **summary_metadata,
                **pack_diagnostics,
                "failure_kind": "context_budget_exceeded",
                "summary_version": summary_metadata.get("schema_version"),
                "request_sent": False,
                "candidate_packed_item_ids": list(pack_diagnostics.get("included_item_ids") or []),
                "candidate_packed_evidence_ids": list(pack_diagnostics.get("included_evidence_ids") or []),
                "included_item_ids": [],
                "included_evidence_ids": [],
                "candidate_evidence_ids": candidate_evidence_ids,
                "retained_evidence_ids": [],
                "omitted_evidence_ids": candidate_evidence_ids,
                "packer_omitted_evidence_ids": packer_omitted_evidence_ids,
                "optional_omitted_evidence_ids": optional_omitted,
                "planned_reviewers": list(planned_reviewers),
                "required_reviewers": sorted(
                    {"combined"}
                    if review_mode is CriticLLMReviewMode.COMBINED_REVIEW_EXPERIMENT
                    else required_review_names
                ),
            }
            critic_context_failure = {
                "failure_kind": "context_budget_exceeded",
                **critic_context_metadata,
            }
            return False

        critic_context_text = critic_bundle.text
        critic_request_text = str(critic_bundle.resolved_request.get("text") or "")
        included_items = set(critic_bundle.included_item_ids)
        omitted_items = {
            str(item.get("item_id") or "") for item in critic_bundle.omitted
            if item.get("item_id")
        }
        optional_omitted = sorted({
            evidence_id
            for item in summary_items
            if not item.required and item.item_id not in included_items
            for evidence_id in item.evidence_ids
        })
        omitted_evidence_ids = sorted(
            item_id.removeprefix("evidence.")
            for item_id in evidence_item_ids & omitted_items
        )
        retained_evidence_ids = sorted(
            item_id.removeprefix("evidence.")
            for item_id in evidence_item_ids & included_items
        )
        optional_ids = set(summary_metadata.get("optional_evidence_ids") or [])
        optional_omitted = sorted(set(omitted_evidence_ids) & optional_ids)
        critic_context_metadata = {
            **summary_metadata,
            "summary_version": summary_metadata.get("schema_version"),
            "estimated_tokens": critic_bundle.budget.get("estimated_tokens"),
            "available_tokens": critic_bundle.budget.get("available_context_tokens"),
            "required_item_count": sum(1 for item in summary_items if item.required),
            "optional_item_count": sum(1 for item in summary_items if not item.required),
            "retained_evidence_ids": retained_evidence_ids,
            "omitted_evidence_ids": omitted_evidence_ids,
            "optional_omitted_evidence_ids": optional_omitted,
            "planned_reviewers": list(planned_reviewers),
            "required_reviewers": sorted(
                {"combined"}
                if review_mode is CriticLLMReviewMode.COMBINED_REVIEW_EXPERIMENT
                else required_review_names
            ),
        }
        return True

    if request_contract_review.get("status") == "rejected":
        not_run = {
            "verdict": "not_completed", "completion_status": "not_completed",
            "failure_kind": "request_contract_violation", "issues": [],
            "call_ids": [], "sdk_attempt_ids": [],
        }
        groundedness = dict(not_run)
        completeness = dict(not_run)
        consistency = dict(not_run)
        llm_review_status = "incomplete"
    elif not optional_llm_review_applicable:
        groundedness = {"verdict": "skipped", "issues": []}
        completeness = {"verdict": "skipped", "issues": []}
        consistency = {"verdict": "skipped", "issues": []}
    elif optional_llm_review_off:
        # This is an explicit evaluation ablation, never a pass or Provider failure.
        groundedness = {"verdict": "evaluation_ablation", "issues": [], "reason": "llm_review_disabled_for_evaluation"}
        completeness = {"verdict": "evaluation_ablation", "issues": [], "reason": "llm_review_disabled_for_evaluation"}
        consistency = {"verdict": "evaluation_ablation", "issues": [], "reason": "llm_review_disabled_for_evaluation"}
    elif adaptive_skip:
        # This is a production policy decision, not an evaluation ablation.
        # The policy is recorded below with its individual deterministic gates.
        groundedness = {"verdict": "skipped_by_adaptive_policy", "issues": [], "reason": adaptive_policy["reason_code"]}
        completeness = {"verdict": "skipped_by_adaptive_policy", "issues": [], "reason": adaptive_policy["reason_code"]}
        consistency = {"verdict": "skipped_by_adaptive_policy", "issues": [], "reason": adaptive_policy["reason_code"]}
    elif adaptive_repair or adaptive_integrity_reject:
        # The registered deterministic answer contract is authoritative here.
        # Do not ask an LLM to judge an answer that is already known to differ
        # from it.  A text mismatch is repairable by deterministic rendering;
        # a malformed/broken contract is a terminal integrity failure.
        status = "not_run_due_answer_integrity_repair" if adaptive_repair else "not_run_due_answer_integrity_failure"
        not_run = {
            "verdict": "not_run", "completion_status": "not_run",
            "failure_kind": status, "issues": [], "call_ids": [], "sdk_attempt_ids": [],
        }
        groundedness = dict(not_run)
        completeness = dict(not_run)
        consistency = dict(not_run)
    elif terminal_execution_failure:
        # A terminal execution/data failure is already authoritative. Keep
        # deterministic checks, but do not spend more model calls asking the
        # optional reviewers to repair an answer that cannot be completed.
        # This is neither a pass nor a Provider failure.
        not_run = {
            "verdict": "not_completed",
            "completion_status": "not_completed",
            "failure_kind": "upstream_terminal_execution_failure",
            "issues": [],
            "call_ids": [],
            "sdk_attempt_ids": [],
        }
        groundedness = dict(not_run)
        completeness = dict(not_run)
        consistency = dict(not_run)
    elif review_mode is CriticLLMReviewMode.COMBINED_REVIEW_EXPERIMENT:
        thinking_kwargs = (
            {"enable_thinking": False}
            if get_settings().critic_default_non_thinking
            or thinking_mode is CriticLLMThinkingMode.NON_THINKING_EXPERIMENT else {}
        )
        if turn_soft_limit_reached(state):
            optional_review_stop_reason = "not_run_due_soft_limit"
            combined_review = _review_not_run(optional_review_stop_reason)
        elif not prepare_critic_bundle():
            combined_review = _review_not_completed("context_budget_exceeded")
        else:
            # WP0: ordinary RAG is not a Critic source.  The controlled context
            # bundle carries current evidence and registry definitions only.
            combined_review = _run_optional_llm_review(
                llm_review_combined_experiment, answer, "", task_type,
                critic_request_text, investigation_context,
                state=state, context_text=critic_context_text,
                request_text=critic_request_text, **thinking_kwargs,
            )
        not_separate = {"verdict": "not_run_combined_experiment", "issues": []}
        groundedness = dict(not_separate)
        completeness = dict(not_separate)
        consistency = dict(not_separate)
    else:
        review_context = (
            investigation_context
            if (task_type == "anomaly_analysis" or registered_decomposition_review
                or contribution_contract.get("operation_intent") == "dimensional_statistical_contribution")
            else None
        )
        thinking_kwargs = (
            {"enable_thinking": False}
            if get_settings().critic_default_non_thinking
            or thinking_mode is CriticLLMThinkingMode.NON_THINKING_EXPERIMENT
            else {}
        )
        review_functions = (
            ("groundedness", llm_review_groundedness, (answer, "", review_context)),
            ("completeness", llm_review_evidence_completeness, (answer, task_type, review_context)),
            ("consistency", llm_review_logic_consistency, (answer, task_type, review_context)),
        )
        review_results: dict[str, dict[str, Any]] = {}
        context_budget_failed = False
        required_call_reserve = 0
        for name, review_fn, args in review_functions:
            if optional_review_stop_reason:
                review_results[name] = _review_not_run(optional_review_stop_reason)
                continue
            if cascade_active and name in cascade_skip_names:
                review_results[name] = _review_not_run("risk_cascade_skip")
                continue
            if name not in required_review_names and turn_soft_limit_reached(state):
                review_results[name] = _review_not_run("not_run_due_soft_limit")
                optional_review_stop_reason = "not_run_due_soft_limit"
                continue
            if name not in required_review_names and required_call_reserve > 0:
                optional_call_reserve = ceil(required_call_reserve * 1.15)
                if not _optional_review_fits_with_required_followup_reserve(
                    state,
                    optional_call_reserve=optional_call_reserve,
                    required_call_reserve=required_call_reserve,
                ):
                    review_results[name] = _review_not_run(
                        "not_run_due_required_review_reserve",
                    )
                    optional_review_stop_reason = "not_run_due_required_review_reserve"
                    continue
            if not prepare_critic_bundle():
                context_budget_failed = True
                break
            review_results[name] = _run_optional_llm_review(
                review_fn, *args, state=state, context_text=critic_context_text,
                request_text=critic_request_text,
                budget_optional=name not in required_review_names,
                **thinking_kwargs,
            )
            if name in required_review_names and review_results[name].get("completion_status") == "completed":
                required_call_reserve = _completed_review_budget_reserve(
                    state, review_results[name],
                )
            review = review_results[name]
            failure_kind = str(review.get("failure_kind") or "")
            if get_settings().critic_review_stop_on_failure_enabled and review.get("completion_status") != "completed":
                optional_review_stop_reason = {
                    "not_sent": "not_run_due_budget",
                    "not_run_due_soft_limit": "not_run_due_soft_limit",
                    "timeout": "not_run_due_provider_error",
                    "provider_error": "not_run_due_provider_error",
                }.get(failure_kind, "not_run_due_review_failure")
        if context_budget_failed:
            # A single shared context failure prevented every reviewer from
            # seeing its required evidence. Preserve that distinction on all
            # reviewer records; none of them passed or returned a verdict.
            review_results = {
                name: _review_not_completed("context_budget_exceeded")
                for name, _review_fn, _args in review_functions
            }
            optional_review_stop_reason = "context_budget_exceeded"
        groundedness = review_results["groundedness"]
        completeness = review_results["completeness"]
        consistency = review_results["consistency"]

    if optional_llm_review_off:
        llm_review_status = "evaluation_ablation"
    elif adaptive_skip:
        llm_review_status = "skipped_by_adaptive_policy"
    elif adaptive_repair:
        llm_review_status = "not_run_due_answer_integrity_repair"
    elif adaptive_integrity_reject:
        llm_review_status = "not_run_due_answer_integrity_failure"
    elif not optional_llm_review_applicable:
        llm_review_status = "not_applicable"
    elif terminal_execution_failure:
        llm_review_status = "not_run_due_terminal_execution_failure"
    else:
        reviews_by_name = (
            {"combined": combined_review}
            if combined_review is not None
            else {"groundedness": groundedness, "completeness": completeness, "consistency": consistency}
        )
        required_names = {"combined"} if combined_review is not None else required_review_names
        required_reviews = [reviews_by_name[name] for name in required_names if name in reviews_by_name]
        optional_reviews = [item for name, item in reviews_by_name.items() if name not in required_names]
        cascade_optional_skipped = bool(
            cascade_active
            and cascade_skip_names
            and all(
                reviews_by_name.get(name, {}).get("failure_kind") == "risk_cascade_skip"
                for name in cascade_skip_names
            )
        )
        if any(item.get("completion_status") != "completed" for item in required_reviews):
            llm_review_status = "incomplete"
        elif any(item.get("verdict") == "failed" for item in required_reviews + optional_reviews):
            llm_review_status = "failed"
        elif cascade_optional_skipped:
            llm_review_status = "required_passed_optional_skipped_by_risk_cascade"
        elif any(item.get("completion_status") != "completed" for item in optional_reviews):
            llm_review_status = "required_passed_optional_incomplete"
        else:
            llm_review_status = "passed"

    if (
        llm_review_status == "passed"
        and critic_context_metadata
        and critic_context_metadata.get("optional_omitted_evidence_ids")
    ):
        # Required evidence remains complete, but the context itself was not
        # exhaustive. Keep that visible in the aggregate review status.
        llm_review_status = "required_passed_optional_incomplete"

    all_review_results = (
        {"combined": combined_review}
        if combined_review is not None
        else {
            "groundedness": groundedness,
            "completeness": completeness,
            "consistency": consistency,
        }
    )
    llm_issues = [
        issue for result in all_review_results.values()
        for issue in result.get("issues", [])
    ]
    optional_review_findings = [
        {"review": name, "issues": item.get("issues", [])}
        for name, item in {
            "groundedness": groundedness,
            "completeness": completeness,
            "consistency": consistency,
        }.items()
        if name not in required_review_names and item.get("verdict") == "failed"
    ]
    if completeness_review.get("status") == "inconclusive" and combined_review is None:
        # Missing data is an answer limitation, not a reason to keep querying.
        llm_issues = [
            issue for name, result in all_review_results.items()
            if name != "completeness"
            for issue in result.get("issues", [])
        ]
    if llm_review_status == "failed" and not llm_issues:
        llm_issues.append("One or more LLM reviews returned failed without an actionable issue.")
    if llm_review_status == "incomplete":
        llm_issues.extend(
            f"{name} review not completed ({review.get('failure_kind') or 'unknown'})."
            for name, review in ((("combined", combined_review),) if combined_review is not None else (
                ("groundedness", groundedness),
                ("completeness", completeness),
                ("consistency", consistency),
            ))
            if review.get("completion_status") != "completed"
        )

    # === 裁决逻辑 ===
    # 默认先假设通过；后面的结构化审核、LLM 审核和人工门禁可以把它改成其他结果。
    verdict = "approved"
    # revision_target 记录返工目标节点，避免返工时重新 triage 丢失已有上下文。
    revision_target = None

    if adaptive_repair:
        verdict = "needs_revision"
        revision_target = CriticRevisionTarget.GENERATE_ANSWER
        llm_issues.append("Adaptive answer integrity mismatch: discarded the non-deterministic answer and requested deterministic regeneration.")
    if adaptive_integrity_reject:
        verdict = "rejected"
        revision_target = None
        state.setdefault("errors", []).append({
            "code": "adaptive_answer_contract_integrity_failure",
            "message": "Adaptive answer contract or evidence binding failed verification; the run stopped without reusing the current answer.",
            "reason_code": adaptive_policy.get("reason_code") if adaptive_policy else "unknown",
            "retryable": False,
            "agent_name": "critic",
        })

    # RAG 结构化检查失败，只回到指标定义节点重新取上下文。
    if requires_rag and rag_review.get("status") == "rejected":
        verdict = "needs_revision"
        revision_target = CriticRevisionTarget.METRIC_DEFINITION

    # 数据结构化检查失败，只回到数据查询节点重新获取结果。
    if task_type in PATHS_REQUIRING_DATA and data_review.get("status") == "rejected":
        verdict = "needs_revision"
        revision_target = CriticRevisionTarget.DATA_QUERY

    # claim 没有绑定有效 evidence 时，答案生成节点必须重新组装引用。
    if claim_review["status"] == "rejected":
        verdict = "needs_revision"
        revision_target = CriticRevisionTarget.GENERATE_ANSWER
    if numeric_review["status"] == "rejected":
        verdict = "needs_revision"
        revision_target = CriticRevisionTarget.GENERATE_ANSWER
    if ratio_answer_review["status"] == "rejected":
        verdict = "needs_revision"
        revision_target = CriticRevisionTarget.GENERATE_ANSWER
    if completeness_review.get("status") == "rejected":
        verdict = "needs_revision"
        request_contract = (((state.get("context_compilation") or {}).get("request_contract")) or {})
        revision_target = (
            CriticRevisionTarget.GENERATE_ANSWER
            if request_contract.get("operation_intent") == "dimensional_statistical_contribution"
            else CriticRevisionTarget.DIAGNOSTIC_PLAN
        )
    if causal_review.get("status") == "rejected":
        verdict = "needs_revision"
        revision_target = CriticRevisionTarget.GENERATE_ANSWER
    if report_review["status"] == "rejected":
        verdict = "needs_revision"
        revision_target = CriticRevisionTarget.REPORT
        llm_issues.extend(report_review["issues"])

    context_limit_review_candidate = bool(
        critic_context_failure
        and str(get_settings().critic_limit_policy or "").strip().casefold() == "human_review"
        and _critic_context_limit_supports_human_review(task_type, state)
        and verdict == "approved"
        and not state.get("errors")
    )

    # LLM 审核结果可加重裁决（但不主动降低）。报告和归因是草案型
    # specialist，应回到原作者修订，而不是重新 triage 丢失上下文。
    # 只有前面的确定性检查都通过时，LLM 审核问题才会把 verdict 从 approved 改为返工。
    if verdict == "approved" and llm_issues and not context_limit_review_candidate:
        verdict = "needs_revision"
        revision_target = {
            "report_generation": CriticRevisionTarget.REPORT,
            # The optional LLM reviewers audit the answer text. Evidence and
            # required-step failures have already been classified by the
            # deterministic checks above and retain their specialist target.
            # A text-only revision must not restart the investigation plan.
            "anomaly_analysis": CriticRevisionTarget.GENERATE_ANSWER,
            "metric_query": (
                CriticRevisionTarget.GENERATE_ANSWER
                if registered_decomposition_review
                else CriticRevisionTarget.TRIAGE
            ),
        }.get(task_type, CriticRevisionTarget.TRIAGE)

    # 每个结构化报告草稿都必须经过持久化人工审核门禁；报告风险会随 interrupt payload
    # 一起交给审核者，不能仅凭 Critic 自动通过后直接发布。
    if task_type == "report_generation" and state.get("report_id") and verdict == "approved":
        verdict = "human_review"
        revision_target = None

    # Incomplete comparisons are terminal abstentions. Do not let unrelated
    # claim checks or optional reviewers turn them into a retry of the same
    # dates/query, and never approve them as completed comparisons.
    if task_type in PATHS_REQUIRING_DATA and data_review.get("status") == "incomplete":
        verdict = "rejected"
        revision_target = None

    # Batch limitations and execution failures are terminal: keep successful
    # evidence, but never retry the already executed items or approve the whole
    # batch as complete.
    if task_type in PATHS_REQUIRING_DATA and data_review.get("batch_status") in {
        "partial_unavailable", "execution_failed",
    }:
        verdict = "rejected"
        revision_target = None
    if diagnostic_review.get("status") == "rejected":
        verdict = "rejected"
        revision_target = None
    if terminal_execution_failure:
        # A model or tool execution failure is terminal for this run. It cannot
        # be approved merely because the remaining deterministic review fields
        # happened to pass on a partial state.
        verdict = "rejected"
        revision_target = None
    if request_contract_review.get("status") == "rejected":
        verdict = "rejected"
        revision_target = None

    # 按任务类型定义返工上限策略。普通问答最多一次返工后直接
    # 写错误落盘，不进入 human_review（普通问答没有审核界面）。
    # 报告在结构化审核通过后才进入 human_review，返工不合格直接失败。
    revision_counts = state.get("_revision_counts", {})
    revision_target_value = getattr(revision_target, "value", revision_target)
    revision_target_key = str(revision_target_value) if revision_target_value is not None else None
    revision_count = revision_counts.get(revision_target_key, revision_counts.get(revision_target, 0))
    if verdict == "needs_revision" and revision_target_key and revision_count > get_settings().max_revisions:
        _policy_table = {
            "metric_query": "rejected",
            "metric_definition": "rejected",
            "anomaly_analysis": "rejected",
            "report_generation": "rejected",  # 不合格报告直接失败，不进入人工审核
        }
        policy = _policy_table.get(task_type, "approved")
        if policy == "rejected":
            target_description = {
                "generate_answer": "答案生成",
                "data_query": "数据查询",
                "diagnostic_plan": "诊断计划",
                "planning": "分析计划",
            }.get(revision_target_key, revision_target_key)
            state.setdefault("errors", []).append({
                "code": "critic_revision_limit_exceeded",
                "message": (
                    f"{target_description}经有限返工后仍未通过 Critic，已达到该节点的返工上限并停止。"
                    + ("此错误表示答案返工上限，不表示数据查询重复执行。" if revision_target_key == "generate_answer" else "")
                ),
                "revision_target": revision_target_key,
                "revision_count": revision_count,
                "max_revisions": get_settings().max_revisions,
                "retryable": False,
                "agent_name": "critic",
            })
            verdict = "rejected"
            revision_target = None
        else:
            verdict = "approved"
            revision_target = None
            llm_issues.append("Revision limit reached; approved with recorded residual issues.")

    # In FULL mode an absent/invalid verdict is not a successful review. Stop
    # this Critic pass without routing another model cycle against an exhausted
    # send budget. The three per-review records retain the exact failure kind.
    context_limit_human_review = bool(
        context_limit_review_candidate
        and verdict in {"approved", "human_review"}
        and not state.get("errors")
    )
    if optional_llm_review_applicable and not optional_llm_review_off and not adaptive_skip and not adaptive_repair and not adaptive_integrity_reject and llm_review_status == "incomplete":
        verdict = "human_review" if context_limit_human_review else "rejected"
        revision_target = None
        if not context_limit_human_review and not any(
            isinstance(error, dict) and error.get("code") == "critic_review_incomplete"
            for error in (state.get("errors") or [])
        ):
            incomplete_error = {
                "code": "critic_review_incomplete",
                "message": "A required Critic review did not complete; the result is not approved.",
                "retryable": False, "agent_name": "critic",
            }
            if critic_context_failure:
                incomplete_error["failure_kind"] = "context_budget_exceeded"
                incomplete_error["context"] = critic_context_metadata or {}
            state.setdefault("errors", []).append(incomplete_error)
    if verdict == "rejected" and not any(
        isinstance(error, dict) and error.get("code") in {"critic_review_incomplete", "critic_rejected"}
        for error in (state.get("errors") or [])
    ):
        state.setdefault("errors", []).append({
            "code": "critic_rejected",
            "message": "Critic rejected the result after deterministic or required review checks.",
            "retryable": False, "agent_name": "critic",
        })
    approval_basis = (
        "critic_context_budget_exceeded_requires_human_review" if context_limit_human_review else
        "optional_llm_review_evaluation_ablation" if optional_llm_review_off else
        "optional_llm_review_skipped_by_adaptive_policy" if adaptive_skip else
        "adaptive_answer_integrity_repair" if adaptive_repair else
        "adaptive_answer_contract_integrity_failure" if adaptive_integrity_reject else
        "optional_llm_review_not_applicable" if not optional_llm_review_applicable else
        "deterministic_and_combined_experimental_review_passed" if llm_review_status == "passed" and combined_review is not None else
        "deterministic_and_all_optional_llm_reviews_passed" if llm_review_status == "passed" else
        "deterministic_and_required_llm_reviews_passed_optional_skipped_by_risk_cascade" if llm_review_status == "required_passed_optional_skipped_by_risk_cascade" else
        "deterministic_and_required_llm_reviews_passed_optional_incomplete" if llm_review_status == "required_passed_optional_incomplete" else
        "optional_llm_review_failed" if llm_review_status == "failed" else
        "full_optional_llm_review_incomplete"
    )

    result = {
        "agent": "CriticAgent",
        "status": verdict,
        "request_contract_review": request_contract_review,
        "data_review": data_review,
        "rag_review": rag_review,
        "claim_review": claim_review,
        "numeric_review": numeric_review,
        "ratio_answer_review": ratio_answer_review,
        "report_review": report_review,
        "diagnostic_review": diagnostic_review,
        "completeness_review": completeness_review,
        "causal_review": causal_review,
        "llm_review": {
            "mode": review_mode.value,
            "status": llm_review_status,
            "required_reviews": sorted(required_review_names),
            "optional_stop_reason": optional_review_stop_reason,
            "optional_findings": optional_review_findings,
            "approval_basis": approval_basis,
            "groundedness": groundedness,
            "completeness": completeness,
            "consistency": consistency,
            "combined_experiment": combined_review,
            "context": critic_context_metadata,
            "context_limit_action": (
                "human_review" if context_limit_human_review else
                "persist_partial" if critic_context_failure else
                "none"
            ),
        },
        "adaptive_llm_review_policy": adaptive_policy,
        "optional_review_cascade": {
            "mode": cascade_mode,
            "decision": cascade_decision,
            "applied": cascade_active,
        },
    }

    # 将完整审核明细和最终裁决写回 state，后续路由函数只读取这几个字段，不重新推理。
    state["critic_result"] = result
    state["critic_verdict"] = verdict
    state["critic_revision_target"] = revision_target
    # This graph currently builds the answer presentation before Critic. Apply
    # Critic's final batch decision to that structured presentation so an
    # earlier draft cannot leave a rejected/partial batch marked completed.
    presentation = state.get("analysis_presentation")
    batch_presentation_status = {
        "all_success": "completed",
        "no_data": "completed",
        "partial_unavailable": "partial",
        "execution_failed": "failed",
    }.get((state.get("metric_batch_execution") or {}).get("status"))
    diagnostic_presentation_status = (
        diagnostic_terminal_presentation_status(state.get("analysis_result"))
        if task_type == "anomaly_analysis"
        else None
    )
    # A diagnostic stop is the more specific outcome for the final answer. It
    # must not be overwritten by a successful preliminary query batch.
    final_presentation_status = diagnostic_presentation_status or batch_presentation_status
    if numeric_review["status"] == "rejected":
        final_presentation_status = "partial" if verdict == "needs_revision" else "failed"
    # The answer node runs before Critic and may already have marked a valid
    # result as completed. A terminal Critic rejection must never leave that
    # earlier success status in the final response. Keep stronger pre-existing
    # statuses; distinguish an incomplete review (partial result) from a
    # completed negative review or deterministic rejection (failed result).
    critic_rejected = verdict == "rejected"
    critic_review_incomplete = (
        optional_llm_review_applicable
        and not optional_llm_review_off
        and not adaptive_skip
        and llm_review_status == "incomplete"
    )
    prior_presentation_status = (
        presentation.get("status") if isinstance(presentation, dict) else None
    )
    if (critic_rejected or (context_limit_human_review and critic_review_incomplete)) and final_presentation_status in {None, "completed"}:
        if prior_presentation_status not in {None, "completed"}:
            final_presentation_status = prior_presentation_status
        else:
            final_presentation_status = "partial" if critic_review_incomplete else "failed"
    if isinstance(presentation, dict) and final_presentation_status:
        presentation = dict(presentation)
        presentation["status"] = final_presentation_status
        if critic_review_incomplete:
            limitations = list(presentation.get("limitations") or [])
            review_limitation = "可选 LLM 审核未完整完成；当前结果未获完整审核。"
            if review_limitation not in limitations:
                limitations.append(review_limitation)
            presentation["limitations"] = limitations
        state["analysis_presentation"] = presentation

    state.setdefault("critic_decisions", []).append(
        {
            "node": "critic",
            "verdict": verdict,
            "revision_target": revision_target,
            "rag_status": rag_review.get("status", "unknown"),
            "data_status": data_review.get("status", "unknown"),
            "claim_status": claim_review["status"],
            "llm_issues": llm_issues,
            "llm_review_mode": review_mode.value,
            "adaptive_llm_review_policy": adaptive_policy,
            "optional_review_cascade": {
                "mode": cascade_mode,
                "decision": cascade_decision,
                "applied": cascade_active,
            },
        }
    )
    state["current_agent"] = "critic"
    state.setdefault("node_history", []).append(
        {
            "node": "critic",
            "summary": f"Path={task_type} Verdict={verdict} LLM={groundedness.get('verdict','skipped')}",
        }
    )
    return state

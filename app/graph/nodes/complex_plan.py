"""Create a pending complex-analysis plan after a sealed route decision."""

from __future__ import annotations

import json
from collections.abc import Mapping

from app.complex_analysis.canary import (
    evaluate_canary_enrollment,
    issue_sealed_route_decision,
    template_is_canary_allowlisted,
)
from app.complex_analysis.capability_router import route_semantic_contract
from app.complex_analysis.service import create_plan_v2
from app.core.config import get_settings
from app.context.semantic_catalog_v2 import DEFAULT_SEMANTIC_CATALOG
from app.context.semantic_contract_v2 import SealedRouteDecision, SemanticContractV21
from app.graph.state import AgentState
from app.observability.tracing import record_event, record_measurement
from app.tenancy import current_tenant_context


def _require_scope(state: AgentState) -> dict | None:
    if not state.get("use_current_explorer_context"):
        return None
    scope = state.get("controlled_analysis_scope") or {}
    if not isinstance(scope, Mapping):
        return None
    seller_id = scope.get("seller_id")
    start_date = scope.get("default_start_date")
    end_date = scope.get("default_end_date")
    if not seller_id or not start_date or not end_date:
        return None
    return {
        "seller_id": str(seller_id),
        "metric_name": str(scope.get("metric_name") or "gmv"),
        "start_date": start_date,
        "end_date": end_date,
        "granularity": str(scope.get("granularity") or "day"),
    }


def _record_canary_preflight_outcome(
    state: AgentState,
    *,
    outcome: str,
    reason_code: str,
) -> None:
    """Record the terminal preflight result for an enrolled Phase 3 route.

    The intent router records that a catalog route was selected before this
    node checks trusted scope and creates the pending plan. Keep this separate
    low-cardinality event so a later scope/plan rejection is visible without
    counting the initial route selection as a second route action.
    """

    canary = state.get("complex_route_canary_observation")
    if (
        not isinstance(canary, dict)
        or canary.get("eligible") is not True
        or state.get("complex_route_canary_selected") is not True
    ):
        return

    final_action = "clarify" if outcome == "clarify" else "complex_plan"
    final_route_status = "NEEDS_CONTEXT" if outcome == "clarify" else "SELECTED"
    canary.update({
        "action": final_action,
        "reason_code": reason_code,
        "route_status": final_route_status,
        "preflight_outcome": outcome,
        "preflight_reason_code": reason_code,
    })
    record_measurement(
        "semantic_route_canary_preflight_outcomes",
        labels={"outcome": outcome},
    )
    guardrail_reason = {
        "canary_not_authorized": "canary_authorization_failure",
        "semantic_interpreter_failed": "interpreter_or_validator_failure",
        "semantic_router_not_ready": "router_validation_failure",
        "scope_required": "scope_preflight_failure",
        "scope_rejected": "scope_preflight_failure",
        "plan_creation_rejected": "plan_creation_failure",
    }.get(reason_code)
    if outcome == "clarify" and guardrail_reason:
        record_measurement(
            "semantic_route_canary_guardrails",
            labels={"reason": guardrail_reason},
        )

    capability_id = canary.get("capability_id")
    capability = next(
        (item for item in DEFAULT_SEMANTIC_CATALOG.capabilities
         if item.capability_id == capability_id),
        None,
    )
    interpreter = state.get("semantic_interpreter_observation")
    interpreter = interpreter if isinstance(interpreter, Mapping) else {}
    record_event(
        "semantic_router.canary.plan_preflight",
        {
            "action": final_action,
            "reason_code": reason_code,
            "route_status": final_route_status,
            "route_id": canary.get("route_id"),
            "capability_id": capability_id,
            "template_id": capability.template_id if capability is not None else None,
            "semantic_status": (
                "CONFIRMED" if interpreter.get("status") == "ok" else "invalid"
            ),
            "elapsed_ms": interpreter.get("elapsed_ms"),
            "duration_ms": interpreter.get("elapsed_ms"),
            "first_token_ms": interpreter.get("first_token_ms"),
            "input_tokens": interpreter.get("input_tokens"),
            "output_tokens": interpreter.get("output_tokens"),
            "estimated_cost": interpreter.get("estimated_cost"),
            "llm.estimated_cost": interpreter.get("estimated_cost"),
            "usage_status": interpreter.get("usage_status"),
            "timeout_kind": interpreter.get("timeout_kind"),
        },
    )


def _mark_scope_required(state: AgentState, decision: dict) -> AgentState:
    decision["kind"] = "clarify"
    decision["matched_by"] = "scope_preflight"
    evidence = dict(decision.get("evidence") or {})
    evidence["clarification_reason"] = "scope_required"
    decision["evidence"] = evidence
    decision["route_decision_status"] = "NEEDS_CONTEXT"
    state["complex_route_decision"] = decision
    state["route_decision_status"] = "NEEDS_CONTEXT"
    state["complex_clarification"] = {"reason_code": "scope_required"}
    _record_canary_preflight_outcome(
        state,
        outcome="clarify",
        reason_code="scope_required",
    )
    return state


def _mark_plan_rejected(
    state: AgentState,
    decision: dict,
    *,
    reason_code: str,
    answer: str,
) -> AgentState:
    decision["kind"] = "clarify"
    decision["matched_by"] = "plan_preflight_rejected"
    evidence = dict(decision.get("evidence") or {})
    evidence["clarification_reason"] = reason_code
    decision["evidence"] = evidence
    decision["route_decision_status"] = "NEEDS_CONTEXT"
    state["complex_route_decision"] = decision
    state["route_decision_status"] = "NEEDS_CONTEXT"
    state["complex_clarification"] = {"reason_code": reason_code}
    state["task_type"] = "complex_analysis"
    state["answer"] = answer
    state.setdefault("node_history", []).append(
        {"node": "complex_plan", "summary": f"Plan creation rejected reason={reason_code}."}
    )
    _record_canary_preflight_outcome(
        state,
        outcome="clarify",
        reason_code=reason_code,
    )
    return state


def complex_plan(state: AgentState) -> AgentState:
    raw_decision = state.get("complex_route_decision")
    settings = get_settings()
    if settings.complex_route_mode != "active":
        return state

    if not isinstance(raw_decision, SealedRouteDecision):
        return _mark_plan_rejected(
            state,
            {},
            reason_code="semantic_router_not_ready",
            answer="复杂分析功能当前尚未启用；本轮未创建或执行复杂分析计划。",
        )

    enrollment = evaluate_canary_enrollment(settings, current_tenant_context())
    if (
        state.get("complex_route_canary_selected") is not True
        or not enrollment.eligible
        or not template_is_canary_allowlisted(raw_decision.template_id, settings)
    ):
        return _mark_plan_rejected(
            state,
            raw_decision.model_dump(mode="json"),
            reason_code="canary_not_authorized",
            answer="当前请求不在复杂分析灰度范围内，未创建计划。",
        )

    interpreter = state.get("semantic_interpreter_observation") or {}
    if not isinstance(interpreter, Mapping) or interpreter.get("status") != "ok":
        return _mark_plan_rejected(
            state,
            raw_decision.model_dump(mode="json"),
            reason_code="semantic_interpreter_failed",
            answer="语义校验未通过，本轮未创建或执行复杂分析计划。",
        )

    try:
        contract_json = json.dumps(
            state.get("semantic_contract") or {},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        contract = SemanticContractV21.model_validate_json(contract_json)
        reproduced = issue_sealed_route_decision(
            contract,
            route_semantic_contract(contract),
            settings,
        )
    except Exception:  # noqa: BLE001 - recomputation is the trust boundary
        return _mark_plan_rejected(
            state,
            raw_decision.model_dump(mode="json"),
            reason_code="semantic_router_not_ready",
            answer="路由依据校验未通过，本轮未创建或执行复杂分析计划。",
        )
    if reproduced.model_dump(mode="json") != raw_decision.model_dump(mode="json"):
        return _mark_plan_rejected(
            state,
            raw_decision.model_dump(mode="json"),
            reason_code="semantic_router_not_ready",
            answer="路由依据与当前语义契约不一致，本轮未创建或执行复杂分析计划。",
        )
    decision = raw_decision.model_dump(mode="json")

    if not raw_decision.verify_seal() or raw_decision.route_decision_status != "SELECTED":
        decision["kind"] = "clarify"
        decision["matched_by"] = "invalid_sealed_decision"
        decision["evidence"] = {
            **dict(decision.get("evidence") or {}),
            "clarification_reason": "semantic_router_not_ready",
        }
        decision["route_decision_status"] = "NEEDS_CONTEXT"
        state["complex_route_decision"] = decision
        state["route_decision_status"] = "NEEDS_CONTEXT"
        state["complex_clarification"] = {"reason_code": "semantic_router_not_ready"}
        return _mark_plan_rejected(
            state,
            decision,
            reason_code="semantic_router_not_ready",
            answer="复杂分析路由校验未通过；本轮未创建或执行复杂分析计划。",
        )

    # The raw contract contains source spans/surfaces and is no longer needed
    # after the typed decision has passed the active canary guard.
    state["semantic_contract"] = {}

    scope_payload = _require_scope(state)
    if scope_payload is None:
        return _mark_scope_required(state, decision)

    context = current_tenant_context()
    template_name = raw_decision.template_id
    if context is None or not template_name:
        return _mark_scope_required(state, decision)

    try:
        plan = create_plan_v2(
            context,
            str(state.get("user_message") or ""),
            scope_payload,
            state.get("chat_session_id"),
            state.get("chat_turn_id"),
            template_name,
            route_decision=raw_decision,
        )
    except PermissionError:
        return _mark_plan_rejected(
            state,
            decision,
            reason_code="scope_rejected",
            answer="当前账号无法访问所选店铺或日期范围，未创建计划。",
        )
    except ValueError:
        return _mark_plan_rejected(
            state,
            decision,
            reason_code="plan_creation_rejected",
            answer="计划校验未通过，未创建计划。请检查分析范围或重新发起请求。",
        )
    state["complex_plan"] = plan
    state["complex_clarification"] = {
        "status": "confirmation_required",
        "reason_code": "plan_confirmation_required",
    }
    state["task_type"] = "complex_analysis"
    state["answer"] = "已生成复杂分析计划，请确认后执行。"
    state.setdefault("node_history", []).append(
        {"node": "complex_plan", "summary": f"Created pending plan for template={template_name}."}
    )
    _record_canary_preflight_outcome(
        state,
        outcome="plan_created",
        reason_code="plan_confirmation_required",
    )
    return state

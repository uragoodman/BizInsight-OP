"""Phase 2 shadow route observation with an inactive, fail-closed boundary."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping

from app.complex_analysis.canary import issue_sealed_route_decision, template_is_canary_allowlisted
from app.complex_analysis.capability_router import route_semantic_contract
from app.context.request_contract import (
    validate_compiled_request_contract,
    validate_request_contract,
)
from app.context.route_shadow_audit import build_route_shadow_record
from app.context.semantic_catalog_v2 import DEFAULT_SEMANTIC_CATALOG, RouteOutcome
from app.context.semantic_contract_v2 import SemanticContractV21
from app.core.config import get_settings
from app.graph.state import AgentState
from app.observability.tracing import record_event, record_measurement

LOGGER = logging.getLogger(__name__)

_GENERIC_ROUTER_CLARIFICATION_REASONS = frozenset({"semantic_status_missing"})


def _opaque_route_reference(value: str, *, kind: str) -> str:
    return hashlib.sha256(f"capability-router-observation-v1:{kind}:{value}".encode("utf-8")).hexdigest()


def _shadow_route_observation(contract_payload: dict, state: AgentState, settings) -> dict:
    """Route a validated contract for observation only; never change the outcome."""

    try:
        contract = SemanticContractV21.model_validate_json(
            json.dumps(contract_payload, ensure_ascii=False, separators=(",", ":"))
        )
        outcome = route_semantic_contract(contract)
    except Exception as exc:  # noqa: BLE001 - shadow must not affect routing
        return {"status": "failed", "error_type": type(exc).__name__}
    safe_outcome = outcome.model_dump(mode="json")
    basis = safe_outcome.get("decision_basis")
    if isinstance(basis, dict):
        for key, kind in (("operation_ids", "operation"), ("clause_ids", "clause")):
            basis[key] = [
                _opaque_route_reference(str(value), kind=kind)
                for value in basis.get(key, [])
            ]
    observation = {"status": "routed", "outcome": safe_outcome}
    try:
        audit_record = build_route_shadow_record(
            state,
            contract,
            outcome,
            settings=settings,
        )
    except Exception as exc:  # noqa: BLE001 - audit failure must not affect routing
        LOGGER.warning("capability route shadow audit rejected error_type=%s", type(exc).__name__)
        observation["audit_status"] = "rejected"
    else:
        if audit_record is not None:
            observation["audit"] = audit_record
    return observation


def _record_active_canary_event(
    state: AgentState,
    *,
    action: str,
    reason_code: str,
    route_status: str,
    route_id: str | None = None,
    capability_id: str | None = None,
    template_id: str | None = None,
) -> None:
    """Emit bounded route/provider metadata, never the prompt or principal IDs."""

    observation = state.get("semantic_interpreter_observation")
    observation = observation if isinstance(observation, Mapping) else {}
    safe_task_types = {
        "metric_definition", "metric_query", "anomaly_analysis",
        "report_generation", "seller_catalog", "clarification",
    }
    safe_next_agents = safe_task_types | {"data_query", "context_clarification"}
    task_type = state.get("task_type")
    next_agent = state.get("next_agent")
    elapsed_ms = observation.get("elapsed_ms")
    input_tokens = observation.get("input_tokens")
    output_tokens = observation.get("output_tokens")
    estimated_cost = observation.get("estimated_cost")
    if action in {"complex_plan", "clarify", "legacy_triage"}:
        record_measurement("semantic_route_canary_actions", labels={"action": action})
    if isinstance(elapsed_ms, (int, float)):
        record_measurement("semantic_route_canary_latency_ms", float(elapsed_ms))
    if isinstance(input_tokens, (int, float)):
        record_measurement("semantic_route_canary_input_tokens", float(input_tokens))
    if isinstance(output_tokens, (int, float)):
        record_measurement("semantic_route_canary_output_tokens", float(output_tokens))
    if isinstance(estimated_cost, (int, float)):
        record_measurement("semantic_route_canary_estimated_cost_usd", float(estimated_cost))
    guardrail_reason = {
        "semantic_interpreter_failed": "interpreter_or_validator_failure",
        "router_validation_failed": "router_validation_failure",
        "sealed_decision_rejected": "sealed_decision_rejected",
    }.get(reason_code)
    if action == "clarify" and guardrail_reason:
        record_measurement(
            "semantic_route_canary_guardrails",
            labels={"reason": guardrail_reason},
        )
    record_event(
        "semantic_router.canary",
        {
            "action": action,
            "reason_code": reason_code,
            "route_status": route_status,
            "route_id": route_id,
            "capability_id": capability_id,
            "template_id": template_id,
            "legacy_task_type": task_type if task_type in safe_task_types else "unknown",
            "legacy_next_agent": next_agent if next_agent in safe_next_agents else "unknown",
            "legacy_unresolved": bool(state.get("triage_unresolved")),
            # Capability IDs and legacy triage task types have no frozen
            # equivalence map yet; report that explicitly instead of inventing
            # an agreement/difference score.
            "route_diff_status": "unmapped",
            "semantic_status": "CONFIRMED" if observation.get("status") == "ok" else "invalid",
            "elapsed_ms": elapsed_ms,
            "duration_ms": elapsed_ms,
            "first_token_ms": observation.get("first_token_ms"),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost": estimated_cost,
            "llm.estimated_cost": estimated_cost,
            "usage_status": observation.get("usage_status"),
            "timeout_kind": observation.get("timeout_kind"),
        },
    )


def _active_canary_clarify(
    state: AgentState,
    *,
    reason_code: str,
    route_status: str = "NEEDS_CONTEXT",
    route_id: str | None = None,
    capability_id: str | None = None,
    template_id: str | None = None,
    error_type: str | None = None,
    candidate_templates: list[dict[str, str]] | None = None,
) -> None:
    state["complex_route_decision"] = {
        "kind": "clarify",
        "route_decision_status": route_status,
        "evidence": {"clarification_reason": reason_code},
    }
    state["route_decision_status"] = route_status
    clarification = {"reason_code": reason_code}
    if candidate_templates:
        clarification["candidate_templates"] = candidate_templates
    state["complex_clarification"] = clarification
    state["semantic_contract"] = {}
    canary = state.setdefault("complex_route_canary_observation", {})
    canary.update({
        "action": "clarify",
        "reason_code": reason_code,
        "route_status": route_status,
    })
    if error_type:
        canary["error_type"] = error_type
    _record_active_canary_event(
        state,
        action="clarify",
        reason_code=reason_code,
        route_status=route_status,
        route_id=route_id,
        capability_id=capability_id,
        template_id=template_id,
    )


def _safe_template_candidates(outcome: RouteOutcome, settings) -> list[dict[str, str]]:
    """Resolve router candidates to registered, visible, allowlisted choices."""

    if outcome.status != "AMBIGUOUS" or len(outcome.candidate_capability_ids) < 2:
        return []
    from app.analysis.templates.registry import list_chat_visible_templates

    visible = {
        str(item.get("name")): str(item.get("display_name") or item.get("name"))
        for item in list_chat_visible_templates()
        if item.get("name")
    }
    capabilities = {
        item.capability_id: item for item in DEFAULT_SEMANTIC_CATALOG.capabilities
    }
    candidates: list[dict[str, str]] = []
    for reference in outcome.candidate_capability_ids:
        capability = capabilities.get(reference.capability_id)
        template_id = capability.template_id if capability is not None else None
        if (
            not template_id
            or not template_is_canary_allowlisted(template_id, settings)
            or template_id not in visible
        ):
            # Never turn an incomplete allowlist into an implicit preference.
            return []
        if not any(item["template_id"] == template_id for item in candidates):
            candidates.append({"template_id": template_id, "display_name": visible[template_id]})
    return candidates if len(candidates) >= 2 else []


def _compiled_ordinary_metric_query_bypass_reason(state: AgentState) -> str | None:
    """Return why a compiler-owned ordinary query should stay on legacy triage.

    This bypass only avoids complex-capability routing. It does not execute a
    query or skip the normal metric planner, registry, authorization, or scope
    checks. An unresolved metric is sent to that planner for clarification;
    only an explicit, compiler-resolved metric is considered resolved here.
    """
    if state.get("task_type") != "metric_query" or state.get("next_agent") != "data_query":
        return None
    if state.get("triage_unresolved") is True:
        return None
    compilation = state.get("context_compilation")
    if not isinstance(compilation, Mapping):
        return None
    if compilation.get("needs_clarification") or compilation.get("missing_slots"):
        return None
    valid, _reason = validate_compiled_request_contract(
        compilation,
        state_contract_hash=state.get("request_contract_hash"),
        require_state_anchor=True,
    )
    if not valid:
        return None
    contract = compilation.get("request_contract")
    if not isinstance(contract, Mapping):
        return None
    if (
        contract.get("operation_intent") != "ordinary_metric_query"
        or contract.get("parse_status") != "resolved"
        or contract.get("uncertainty_reason")
    ):
        return None
    candidate_metrics = contract.get("candidate_metrics")
    if candidate_metrics not in (None, []):
        return None
    metric_id = contract.get("metric_id")
    if metric_id is None or (isinstance(metric_id, str) and not metric_id.strip()):
        # Keep the ordinary request on its existing planner path so it can
        # request a metric confirmation. This is not permission to execute.
        return "ordinary_metric_query_metric_unresolved"
    if not isinstance(metric_id, str):
        return None
    return "resolved_ordinary_metric_query"


def _is_resolved_multi_dimension_metric_query(state: AgentState) -> bool:
    """Keep a trusted multi-axis query on the ordinary metric planner path."""
    if _compiled_ordinary_metric_query_bypass_reason(state) != "resolved_ordinary_metric_query":
        return False
    compilation = state.get("context_compilation")
    if not isinstance(compilation, Mapping):
        return False
    contract = compilation.get("request_contract")
    if not isinstance(contract, Mapping):
        return False
    dimensions = contract.get("dimensions")
    return bool(
        contract.get("required_capability") == "breakdown"
        and contract.get("required_tool") == "query_metric_by_dimensions"
        and isinstance(dimensions, list)
        and 2 <= len(dimensions) <= 4
    )


def _has_compiled_result_followup(state: AgentState) -> bool:
    """Return whether the compiler recognized a result follow-up.

    The downstream data_query node still validates same-scope result context
    before answering. This predicate only keeps the capability router from
    overriding that deterministic follow-up path.
    """

    compilation = state.get("context_compilation")
    if not isinstance(compilation, Mapping):
        return False
    followup = compilation.get("result_followup")
    return isinstance(followup, Mapping) and isinstance(followup.get("kind"), str)


def _compiled_clarification_reason(state: AgentState) -> str | None:
    """Preserve deterministic compiler reasons when the semantic router is vague."""

    compilation = state.get("context_compilation")
    if not isinstance(compilation, Mapping) or compilation.get("needs_clarification") is not True:
        return None
    reason = compilation.get("clarification_reason")
    if not isinstance(reason, str) or not reason or len(reason) > 80:
        return None
    if not (reason[0].islower() and all(char.islower() or char.isdigit() or char == "_" for char in reason)):
        return None
    return reason


def _trusted_dimension_required_reason(state: AgentState) -> str | None:
    """Return the compiler's current, sealed dimension clarification reason.

    This narrow bridge only supplies user-facing guidance when the capability
    router can say no more than ``semantic_status_missing``. It never changes
    the router outcome or makes an unresolved request executable.
    """

    compilation = state.get("context_compilation")
    if not isinstance(compilation, Mapping):
        return None
    reason = compilation.get("clarification_reason")
    if reason != "dimension_required" or compilation.get("needs_clarification") is not True:
        return None

    contract = compilation.get("request_contract")
    if not isinstance(contract, Mapping):
        return None
    contract_hash = contract.get("contract_hash")
    if (
        not isinstance(contract_hash, str)
        or not contract_hash
        or compilation.get("request_contract_hash") != contract_hash
        or state.get("request_contract_hash") != contract_hash
    ):
        return None
    if (
        contract.get("source") != "current_user_message"
        or contract.get("parse_status") != "clarification_required"
        or contract.get("uncertainty_reason") != reason
        or contract.get("operation_intent") != "dimensional_statistical_contribution"
        or contract.get("dimension") is not None
    ):
        return None

    mirrored_intent = compilation.get("operation_intent")
    if (
        not isinstance(mirrored_intent, Mapping)
        or mirrored_intent.get("kind") != contract.get("operation_intent")
    ):
        return None

    # The compiler creates this resolution from the current turn and
    # contract. Its identity checks prevent a stale clarification from a
    # previous turn from being reused here.
    resolution = state.get("turn_resolution")
    from app.context.turn_resolver import durable_turn_id

    current_turn_id = durable_turn_id(state)
    if (
        not isinstance(resolution, Mapping)
        or not current_turn_id
        or resolution.get("turn_id") != current_turn_id
        or resolution.get("relation") != "clarify"
        or resolution.get("needs_clarification") is not True
        or resolution.get("clarification_reason") != reason
        or resolution.get("request_contract_hash") != contract_hash
    ):
        return None

    # Unresolved contribution contracts intentionally validate as
    # ``(False, uncertainty_reason)``. The request validator checks the
    # contract's own digest and shape before returning that expected reason;
    # do not mistake this result for an executable/fully resolved contract.
    valid, validation_reason = validate_request_contract(dict(contract))
    if valid or validation_reason != reason:
        return None
    return reason


def _active_canary_route(state: AgentState, settings) -> None:
    canary = state.get("complex_route_canary_observation")
    if not isinstance(canary, dict) or canary.get("eligible") is not True:
        state["complex_route_canary_observation"] = {
            "eligible": False,
            "action": "clarify",
            "reason_code": "semantic_router_not_ready",
        }
        state["complex_route_decision"] = {
            "kind": "clarify",
            "route_decision_status": "NEEDS_CONTEXT",
            "evidence": {"clarification_reason": "semantic_router_not_ready"},
        }
        state["route_decision_status"] = "NEEDS_CONTEXT"
        state["complex_clarification"] = {
            "status": "complex_analysis_unavailable",
            "reason_code": "semantic_router_not_ready",
        }
        return

    if _is_resolved_multi_dimension_metric_query(state):
        # The ordinary query planner still applies metric registry, tool
        # capability, tenant authorization, and date-range checks. The
        # template router is not the executor for this compiled operation.
        reason_code = "resolved_multi_dimension_metric_query"
        canary.update({
            "action": "legacy_triage",
            "reason_code": reason_code,
            "route_status": "BYPASSED",
        })
        state["semantic_contract"] = {}
        _record_active_canary_event(
            state,
            action="legacy_triage",
            reason_code=reason_code,
            route_status="BYPASSED",
        )
        return

    if _has_compiled_result_followup(state):
        # Result follow-ups are handled by the existing compiler/data_query/
        # answer path. Do not let a probabilistic semantic candidate replace
        # that deterministic same-scope result contract.
        reason_code = "compiled_result_followup"
        canary.update({
            "action": "legacy_triage",
            "reason_code": reason_code,
            "route_status": "BYPASSED",
        })
        state["semantic_contract"] = {}
        _record_active_canary_event(
            state,
            action="legacy_triage",
            reason_code=reason_code,
            route_status="BYPASSED",
        )
        return

    ordinary_query_bypass_reason = _compiled_ordinary_metric_query_bypass_reason(state)
    interpreter = state.get("semantic_interpreter_observation")
    has_valid_interpreter_candidate = (
        isinstance(interpreter, Mapping)
        and interpreter.get("status") == "ok"
        and isinstance(state.get("semantic_contract"), dict)
        and bool(state.get("semantic_contract"))
    )
    if ordinary_query_bypass_reason and not has_valid_interpreter_candidate:
        # Keep the legacy ordinary-query path when interpretation did not
        # produce a validated candidate. A valid candidate is still checked
        # by the deterministic capability router below, so compiler misses do
        # not hide registered complex-analysis capabilities.
        canary.update({
            "action": "legacy_triage",
            "reason_code": ordinary_query_bypass_reason,
            "route_status": "BYPASSED",
        })
        state["semantic_contract"] = {}
        _record_active_canary_event(
            state,
            action="legacy_triage",
            reason_code=ordinary_query_bypass_reason,
            route_status="BYPASSED",
        )
        return

    if (
        not isinstance(interpreter, Mapping)
        or interpreter.get("status") != "ok"
        or not isinstance(state.get("semantic_contract"), dict)
        or not state.get("semantic_contract")
    ):
        trusted_compiler_reason = _trusted_dimension_required_reason(state)
        if trusted_compiler_reason:
            _active_canary_clarify(state, reason_code=trusted_compiler_reason)
            clarification_observation = state.get("complex_route_canary_observation")
            if isinstance(clarification_observation, dict):
                clarification_observation.update({
                    "route_reason_code": "semantic_interpreter_failed",
                    "router_reason_code": None,
                    "clarification_reason_code": trusted_compiler_reason,
                    "clarification_reason_source": "context_compiler",
                })
            return
        _active_canary_clarify(state, reason_code="semantic_interpreter_failed")
        return

    try:
        contract = SemanticContractV21.model_validate_json(
            json.dumps(state["semantic_contract"], ensure_ascii=False, separators=(",", ":"))
        )
        outcome = route_semantic_contract(contract)
    except Exception as exc:  # noqa: BLE001 - canary failures must fail closed
        _active_canary_clarify(
            state,
            reason_code="router_validation_failed",
            error_type=type(exc).__name__,
        )
        return

    safe_observation = _shadow_route_observation(state["semantic_contract"], state, settings)
    routing_context = state.setdefault("routing_context", {})
    if not isinstance(routing_context, dict):
        routing_context = {}
        state["routing_context"] = routing_context
    routing_context["capability_router"] = safe_observation
    capability_id = outcome.capability_ids[0].capability_id if len(outcome.capability_ids) == 1 else None
    capability = next(
        (item for item in DEFAULT_SEMANTIC_CATALOG.capabilities
         if item.capability_id == capability_id),
        None,
    )
    template_id = capability.template_id if capability is not None else None
    canary.update({"route_status": outcome.status, "route_id": outcome.route_id})

    if outcome.status == "SELECTED":
        if capability is not None and not template_is_canary_allowlisted(
            capability.template_id, settings,
        ):
            canary.update({
                "action": "legacy_triage",
                "reason_code": "template_not_allowlisted",
            })
            state["semantic_contract"] = {}
            _record_active_canary_event(
                state,
                action="legacy_triage",
                reason_code="template_not_allowlisted",
                route_status=outcome.status,
                route_id=outcome.route_id,
                capability_id=capability_id,
                template_id=template_id,
            )
            return
        try:
            sealed = issue_sealed_route_decision(contract, outcome, settings)
        except Exception as exc:  # noqa: BLE001 - a bad seal never falls back to a plan
            _active_canary_clarify(
                state,
                reason_code="sealed_decision_rejected",
                route_id=outcome.route_id,
                capability_id=capability_id,
                template_id=template_id,
                error_type=type(exc).__name__,
            )
            return
        state["complex_route_decision"] = sealed
        state["route_decision_status"] = "SELECTED"
        state["task_type"] = "complex_analysis"
        canary.update({
            "action": "complex_plan",
            "reason_code": "selected",
            "capability_id": capability_id,
        })
        _record_active_canary_event(
            state,
            action="complex_plan",
            reason_code="selected",
            route_status=outcome.status,
            route_id=outcome.route_id,
            capability_id=capability_id,
            template_id=template_id,
        )
        return

    if outcome.status in {"NEEDS_CONTEXT", "AMBIGUOUS", "CONFLICT"}:
        rejection_reasons = outcome.decision_basis.rejection_reasons
        router_reason = rejection_reasons[0] if rejection_reasons else None
        compiled_reason = _compiled_clarification_reason(state)
        reason = router_reason or compiled_reason or "needs_context"
        if router_reason:
            reason_source = "router"
        elif compiled_reason:
            reason_source = "context_compiler"
        else:
            reason_source = "fallback"
        if (
            outcome.status == "NEEDS_CONTEXT"
            and router_reason in _GENERIC_ROUTER_CLARIFICATION_REASONS
        ):
            trusted_compiler_reason = _trusted_dimension_required_reason(state)
            if trusted_compiler_reason:
                reason = trusted_compiler_reason
                reason_source = "context_compiler"
        _active_canary_clarify(
            state,
            reason_code=reason,
            route_status=outcome.status,
            route_id=outcome.route_id,
            capability_id=capability_id,
            template_id=template_id,
            candidate_templates=_safe_template_candidates(outcome, settings),
        )
        canary["route_reason_code"] = router_reason or "needs_context"
        canary["router_reason_code"] = router_reason
        canary["clarification_reason_code"] = reason
        canary["clarification_reason_source"] = reason_source
        return

    # Phase 3 owns only explicitly selected complex-analysis capabilities.
    # General conversation and non-matching business requests retain triage.
    canary.update({"action": "legacy_triage", "reason_code": "no_canary_capability"})
    state["semantic_contract"] = {}
    _record_active_canary_event(
        state,
        action="legacy_triage",
        reason_code="no_canary_capability",
        route_status=outcome.status,
        route_id=outcome.route_id,
    )


def intent_router(state: AgentState) -> AgentState:
    """Observe shadow routes or apply the configured active route rollout."""

    settings = get_settings()
    route_mode = settings.complex_route_mode
    interpreter_mode = settings.semantic_interpreter_mode

    state["complex_route_decision"] = {}
    state["route_decision_status"] = route_mode
    shadow_observation_present = (
        interpreter_mode == "shadow"
        and isinstance(state.get("semantic_interpreter_observation"), dict)
    )
    active_canary_observation_present = (
        route_mode == "active"
        and interpreter_mode == "active"
        and state.get("complex_route_canary_selected") is True
        and isinstance(state.get("semantic_interpreter_observation"), dict)
    )
    if not (shadow_observation_present or active_canary_observation_present):
        state["semantic_interpreter_observation"] = {
            "mode": interpreter_mode,
            "status": "reserved_phase0_noop",
        }
    if route_mode == "shadow":
        contract_payload = state.get("semantic_contract")
        if isinstance(contract_payload, dict) and contract_payload:
            observation = _shadow_route_observation(contract_payload, state, settings)
        else:
            observation = {
                "status": "skipped",
                "reason_code": "no_validated_contract",
            }
        legacy_task_type = state.get("task_type")
        legacy_next_agent = state.get("next_agent")
        legacy_unresolved = state.get("triage_unresolved")
        if (
            isinstance(legacy_task_type, str)
            and isinstance(legacy_next_agent, str)
            and isinstance(legacy_unresolved, bool)
        ):
            observation["legacy_decision"] = {
                "task_type": legacy_task_type,
                "next_agent": legacy_next_agent,
                "unresolved": legacy_unresolved,
            }
        else:
            observation["legacy_status"] = "triage_baseline_missing"
        state.setdefault("routing_context", {})
        routing_context = state["routing_context"]
        if not isinstance(routing_context, dict):
            routing_context = {}
            state["routing_context"] = routing_context
        routing_context["capability_router"] = observation

    if route_mode == "active":
        _active_canary_route(state, settings)

    state.setdefault("node_history", []).append(
        {
            "node": "intent_router",
            "summary": (
                f"phase3 route_mode={route_mode}; semantic_interpreter_mode="
                f"{interpreter_mode}"
            ),
        }
    )
    LOGGER.info(
        "semantic routing status route_mode=%s interpreter_mode=%s",
        route_mode,
        interpreter_mode,
    )
    return state

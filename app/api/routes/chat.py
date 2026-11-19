import json
import logging
import queue
import threading
import time
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.graph.workflow import _premise_confirmation, chat_context_scope_fingerprint, resume_human_review, run_chat_workflow, stream_chat_workflow
from app.graph.terminal_status import result_terminal_status
from app.schemas.chat import ChatRequest, ChatResponse, HumanReviewRequest, SemanticMappingConfirmationRequest, SemanticMappingSkipRequest
from app.api.auth import require_actor_match, require_analysis_write, require_tenant_context
from app.tenancy import TenantContext
from app.observability import record_event
from app.core.time_expression import ambiguous_year_list
from app.iam.service import require_permission
from app.tasks.service import create_task, record_task_progress, relation, update_task
from app.data.repositories.explorer import controlled_scope, preview_explorer
from app.metrics.registry import get_registry
from app.metrics.semantic_mapping_repository import confirm_mapping, skip_mapping
from app.auth.context import build_authorization_context
from app.scope.canonical import CanonicalScope
from app.chat_lifecycle.integration import (
    ChatLifecycleBinding,
    begin_chat_lifecycle,
    finalize_chat_lifecycle,
    replay_response,
    resolve_turn_task_type,
)

router = APIRouter(prefix="/chat", tags=["chat"])
LOGGER = logging.getLogger(__name__)


def _memory_session_for_request(payload: ChatRequest) -> str | None:
    """Keep the old memory-session identity outside the durable chat path.

    Legacy memory-only requests can still provide ``memory_session_id``. For
    durable scoped chat, conversation state already comes from chat_sessions
    and chat_turns. Do not copy that identity into the deprecated
    memory_sessions path.
    """
    if not (payload.chat_session_id and payload.use_current_explorer_context and payload.explorer_context):
        return payload.memory_session_id
    if payload.memory_session_id and payload.memory_session_id != payload.chat_session_id:
        raise HTTPException(status_code=400, detail="memory_session_id_must_match_chat_session_id")
    return None


def _record_progress_best_effort(
    context: TenantContext,
    task_id: str | None,
    *,
    status: str,
    trace_id: str,
    turn_id: str | None,
    scope_fingerprint: str,
    relation_decision: dict | None = None,
) -> None:
    from app.core.config import get_settings

    if not task_id or not getattr(get_settings(), "task_continuity_enabled", True):
        return
    try:
        record_task_progress(
            context, task_id, status=status, trace_id=trace_id,
            turn_id=turn_id or trace_id, scope_fingerprint=scope_fingerprint,
            relation_decision=relation_decision,
        )
    except Exception:  # noqa: BLE001 - preserve the completed user response on snapshot failure
        LOGGER.exception("Unable to persist task progress snapshot")


def _mapping_scope_data(
    context: TenantContext,
    explorer_payload,
) -> dict[str, object]:
    """Rebuild the exact server scope used when semantic mappings are created.

    Confirmation requests contain only a mapping handle.  The scope and
    authorization inputs are reconstructed from the trusted explorer snapshot
    so a client cannot widen the mapping scope or accidentally invalidate a
    valid mapping by omitting server-derived fields.
    """
    if explorer_payload is None:
        return {
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
            "scope_type": context.scope_type,
            "seller_ids": list(context.allowed_seller_ids),
            "permissions": context.role,
        }
    snapshot = preview_explorer(context, explorer_payload)
    controlled = controlled_scope(snapshot)
    authz = build_authorization_context(
        context,
        allowed_metric_ids=controlled.allowed_metrics,
        allowed_dimensions=controlled.allowed_dimensions,
        metric_scope_type="seller",
    )
    canonical = CanonicalScope.resolve(
        authz,
        seller_ids=[controlled.seller_id],
        start_date=controlled.default_start_date,
        end_date=controlled.default_end_date,
        metric_ids=controlled.allowed_metrics,
        dimensions=controlled.allowed_dimensions,
        granularity=controlled.granularity,
    )
    return {
        **controlled.model_dump(mode="json"),
        "tenant_id": context.tenant_id,
        "user_id": context.user_id,
        "permissions": context.role,
        "scope_type": "seller",
        "authorization_fingerprint": authz.authz_digest,
        "permission_version": authz.permission_version,
        "policy_set_version": authz.policy_set_version,
        "authz_ref": authz.authz_ref,
        "query_scope_fingerprint": canonical.query_scope_fingerprint,
    }


def _resolve_chat_seller_scope_fingerprint(context: TenantContext, seller_scope_payload) -> str | None:
    """Revalidate signed Explorer provenance at chat ingress for turn binding."""
    if seller_scope_payload is None:
        return None
    from app.auth.context import build_authorization_context
    from app.scope.seller_scope import resolve_seller_scope, verify_seller_scope_payload

    intent = verify_seller_scope_payload(seller_scope_payload, tenant=context)
    auth_scope = "seller" if context.scope_type == "seller" else "platform"
    authz = build_authorization_context(context, metric_scope_type=auth_scope)
    scope = resolve_seller_scope(intent, tenant=context, authz=authz)
    return scope.query_scope_fingerprint


def _task_status_from_result(result: dict) -> str:
    """Map a graph result to the durable task state without leaving running."""
    clarification = result.get("clarification")
    # A registered-capability refusal is a completed result, not an open
    # clarification interaction.  Keep unknown/pre-graph clarification codes
    # fail-closed as user interactions.
    if clarification and not (
        isinstance(clarification, dict) and clarification.get("status") == "unsupported"
    ):
        if clarification.get("status") == "complex_scope_required":
            return "awaiting_scope"
        return "awaiting_confirmation"
    terminal_status = result.get("terminal_status")
    if terminal_status == "completed":
        # A transport-level completed event can still represent a safe
        # capability refusal; persist the semantic outcome as abstained.
        return result_terminal_status(result, default_error_status="failed")
    if terminal_status in {"abstained", "failed"}:
        return terminal_status
    if result.get("error") or result.get("errors") or result.get("error_code"):
        return result_terminal_status(result)
    return result_terminal_status(result)


def _expected_data_manifest_hash(context: TenantContext, payload) -> str | None:
    """Resolve the current version gate for relation-time evidence suggestions."""
    from app.core.config import get_settings

    if (
        not getattr(get_settings(), "task_continuity_enabled", True)
        or not getattr(get_settings(), "data_versioning_enabled", False)
        or payload is None
    ):
        return None
    try:
        raw = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload
        from app.schemas.explorer import ExplorerQuery

        snapshot = preview_explorer(context, ExplorerQuery.model_validate(raw))
        if str((snapshot.get("context") or {}).get("metric_name") or "") != "gmv":
            return None
        controlled = controlled_scope(snapshot).model_dump(mode="json")
        from app.data_versions.service import current_gmv_manifest_hash
        from app.metrics.authorization import scope_fingerprint as metric_scope_fingerprint

        return current_gmv_manifest_hash(
            context, controlled, metric_scope_fingerprint(controlled),
        )
    except Exception:  # noqa: BLE001 - unavailable version metadata disables reuse
        LOGGER.exception("Unable to resolve the current data version for evidence reuse")
        return None


def _clarification_interaction_state(clarification: dict | None) -> str:
    """Map a pre-graph clarification to the durable interaction state."""

    status = str((clarification or {}).get("status") or "")
    if status in {
        "complex_scope_required",
        "seller_scope_change_requires_new_scope",
        "awaiting_scope",
    }:
        return "awaiting_scope"
    if status == "task_relationship_required":
        return "awaiting_human_input"
    return "awaiting_metric_confirmation"


_SELLER_SCOPE_CHANGE_MARKERS = (
    "换成另一个店铺", "换成另一个店", "换成其他店铺", "换成其他店",
    "换成别的店铺", "换成别的店", "换一个店铺", "换一个店",
    "换个店铺", "换个店", "换店铺", "切换店铺",
    "另一家店铺", "另一家店", "其他店铺", "其他店", "别的店铺", "别的店",
)


def _seller_scope_change_clarification(payload: ChatRequest) -> dict | None:
    """Refuse a seller switch requested only in chat while a scope is locked.

    The chat boundary never derives a new seller from prose. An explicit
    ``seller_scope_payload`` is the only supported carrier; without it the user
    must re-select the scope in Explorer instead of silently querying the old
    seller.
    """

    if not payload.use_current_explorer_context or payload.explorer_context is None:
        return None
    if payload.seller_scope_payload is not None:
        return None
    if not any(marker in payload.message for marker in _SELLER_SCOPE_CHANGE_MARKERS):
        return None
    return {
        "status": "seller_scope_change_requires_new_scope",
        "reason_code": "seller_scope_change_requires_new_scope",
        "reason": "当前会话的店铺范围已锁定，不能通过聊天切换到其他店铺。请先在 Explorer 重新选择店铺和日期范围。",
    }


def _lifecycle_result_status(result: dict) -> tuple[str, str, str | None, dict | None]:
    """Map graph output to the single durable execution state machine."""

    errors = result.get("errors") or []
    if (result.get("error") or result.get("error_code")) and not errors:
        errors = [{"code": str(result.get("error_code") or "EXECUTION_ABSTAINED"), "message": str(result.get("error_detail") or "Execution did not produce a grounded answer.")}]
    clarification = result.get("clarification")
    if result.get("event") == "suspended" or result.get("_status") in {"suspended", "interrupted"} or result.get("__interrupt__"):
        return "suspended", "awaiting_human_input", None, None
    # `unsupported` communicates a terminal capability boundary.  Treating it
    # as pending input changes a completed refusal into a suspended turn.
    if clarification and not (
        isinstance(clarification, dict) and clarification.get("status") == "unsupported"
    ):
        return "suspended", _clarification_interaction_state(clarification), None, None
    terminal_status = result.get("terminal_status")
    if terminal_status == "completed":
        # `completed` closes the SSE stream; classify the durable semantic
        # outcome independently (for example, an unsupported capability is
        # an abstention, not a successful task).
        terminal_status = result_terminal_status(result, default_error_status="failed")
    if terminal_status in {"completed", "abstained", "failed"}:
        error_code = str(result.get("error_code") or "") or None
        detail = (
            {"code": error_code, "message": str(result.get("error_detail") or "")}
            if error_code else None
        )
        return terminal_status, "none", error_code, detail
    if errors:
        last = errors[-1] if isinstance(errors[-1], dict) else {}
        return result_terminal_status(result), "none", str(last.get("code") or "INTERNAL_ERROR"), last
    inferred = result_terminal_status(result, default_error_status="failed")
    return inferred, "none", None, None


def _persist_pre_graph_trace(
    tenant_context: TenantContext,
    *,
    trace_id: str,
    task_type: str,
    answer: str,
    clarification: dict | None,
    scope_fingerprint: str | None,
    task_id: str | None = None,
    relation_decision: dict | None = None,
    status: str = "suspended",
    interaction_state: str | None = None,
    node_name: str = "pre_graph_clarification",
    summary: str | None = None,
) -> None:
    """Persist a minimal audit trace when a response stops before the graph.

    Task-relation and scope clarifications are returned by the HTTP layer, so
    they never reach ``persist_trace``. Without this snapshot the trace id shown
    to the user would resolve to 404 and the audit could not prove that no
    business query ran. The payload is intentionally minimal: no raw question,
    SQL, or model text, and empty tool/query collections.
    """

    try:
        from app.harness.storage import write_trace

        safe_clarification = {
            key: value for key, value in (clarification or {}).items()
            if key not in {"original_message", "task_candidates", "context_resolution"}
        }
        safe_relation = {
            key: (relation_decision or {}).get(key)
            for key in ("action", "reason", "task_id", "decision", "driver")
            if isinstance(relation_decision, dict) and relation_decision.get(key) is not None
        }
        write_trace(trace_id, {
            "trace_id": trace_id,
            "tenant_id": tenant_context.tenant_id,
            "user_id": tenant_context.user_id,
            "scope_type": tenant_context.scope_type,
            "seller_id": tenant_context.seller_id,
            "task_id": task_id,
            "task_type": task_type,
            "status": status,
            "interaction_state": (
                interaction_state
                if interaction_state is not None
                else _clarification_interaction_state(clarification)
            ),
            "answer": answer,
            "clarification": safe_clarification,
            "current_query_scope_fingerprint": scope_fingerprint,
            "turn_relation_decision": safe_relation,
            "query_results": [],
            "tool_calls": [],
            "node_history": [{
                "node": node_name,
                "summary": summary or "Stopped before workflow execution: clarification required.",
                "reason_code": (
                    safe_clarification.get("status")
                    or safe_clarification.get("reason_code")
                ),
            }],
        })
    except Exception:  # noqa: BLE001 - trace persistence cannot change the response
        LOGGER.exception("Unable to persist pre-graph clarification trace")


def _is_suspended_premise_resume(binding, payload: ChatRequest) -> bool:
    """Recognize only the exact, server-bound suspended premise checkpoint."""
    if binding is None or binding.execution is None or binding.active_reconnect:
        return False
    turn = binding.turn
    snapshot = turn.result_snapshot if isinstance(turn.result_snapshot, dict) else {}
    if (
        turn.status not in {"running", "suspended"}
        or snapshot.get("interaction_reason_code") != "premise_direction_mismatch"
        or not _premise_confirmation(payload.message)
    ):
        return False
    execution_thread_id = binding.execution.thread_id
    if not payload.thread_id or payload.thread_id != execution_thread_id:
        raise HTTPException(status_code=409, detail="suspended_premise_thread_mismatch")
    if (
        binding.execution.query_scope_fingerprint != binding.query_scope_fingerprint
        or turn.scope_fingerprint != binding.query_scope_fingerprint
    ):
        raise HTTPException(status_code=409, detail="suspended_premise_scope_mismatch")
    return True


def _finalize_binding(
    binding: ChatLifecycleBinding | None, result: dict | None, *,
    failed: Exception | None = None, task_id: str | None = None,
    relation_decision: dict | None = None,
    turn_resolution: dict | None = None,
) -> bool:
    embedded_resolution = result.pop("turn_resolution", None) if isinstance(result, dict) else None
    if turn_resolution is None and isinstance(embedded_resolution, dict):
        turn_resolution = embedded_resolution
    if failed is not None:
        if binding is None or binding.execution is None:
            return True
        return finalize_chat_lifecycle(
            binding, result={"answer": "", "errors": [{"code": "INTERNAL_ERROR", "message": str(failed)}]},
            status="failed", error_code="INTERNAL_ERROR", error_detail={"exception_type": type(failed).__name__},
            task_id=task_id, task_status="failed" if task_id else None,
            relation_decision=relation_decision,
        )
    status, interaction, error_code, detail = _lifecycle_result_status(result or {})
    if status == "suspended" and isinstance(result, dict):
        if result.get("event") == "completed":
            result["event"] = "suspended"
        # terminal_status belongs to SseCompletedEvent only. A graph's
        # completed event can be converted into a suspended interaction here.
        result.pop("terminal_status", None)
        result["interaction_state"] = interaction
    elif isinstance(result, dict) and result.get("event") == "completed":
        # Keep the SSE event name as transport completion while exposing the
        # separately classified durable outcome to clients and test reports.
        result["terminal_status"] = status
    if binding is None or binding.execution is None:
        return True
    commit_result = dict(result or {})
    # Persist the normalized outcome, not the inconsistent transport value
    # that may have arrived as ``completed`` alongside a terminal error.
    commit_result["terminal_status"] = status
    if error_code:
        commit_result["error"] = True
        commit_result["error_code"] = error_code
    if (
        isinstance(commit_result.get("result_context"), dict)
        and binding is not None and binding.turn is not None
    ):
        commit_result["result_context"]["source_turn_id"] = binding.turn.turn_id
        commit_result["result_context"]["source_trace_id"] = (
            str(commit_result.get("trace_id") or "")
            or commit_result["result_context"].get("source_trace_id")
        )
    if isinstance(turn_resolution, dict) and turn_resolution:
        commit_result["turn_resolution"] = turn_resolution
        resolution_scope = str(turn_resolution.get("scope_fingerprint") or "")
        binding_scope = str(getattr(binding, "query_scope_fingerprint", "") or "")
        scope_matches = not binding_scope or resolution_scope == binding_scope
        if (
            status == "completed" and scope_matches
            and not isinstance(commit_result.get("turn_context"), dict)
        ):
            # The SSE terminal event intentionally carries no compiler state.
            # Persist only its server-built, structured slots so a later turn
            # in the same authorized scope can inherit an exact date/period.
            known_slots = turn_resolution.get("known_slots")
            if isinstance(known_slots, dict) and known_slots:
                commit_result["turn_context"] = {
                    "authorized_metrics": [],
                    "metric_planning_decision": {},
                    "context_slots": known_slots,
                }
    committed = finalize_chat_lifecycle(
        binding, result=commit_result, status=status, interaction_state=interaction,
        error_code=error_code, error_detail=detail,
        task_id=task_id,
        task_status=_task_status_from_result(commit_result) if task_id else None,
        relation_decision=relation_decision,
    )
    if not committed:
        raise RuntimeError("chat_turn_commit_not_persisted")
    if isinstance(result, dict):
        metadata = binding.service.commit_metadata(binding.turn.turn_id)
        result.update({
            "durable_turn_id": metadata.get("turn_id"),
            "execution_id": metadata.get("final_execution_id") or (binding.execution.execution_id if binding.execution else None),
            "commit_version": metadata.get("commit_version"),
        })
    return committed


@router.post("/semantic-mappings/{mapping_id}/confirm")
def confirm_semantic_mapping(
    mapping_id: str,
    payload: SemanticMappingConfirmationRequest,
    tenant_context: TenantContext = Depends(require_tenant_context),
) -> dict:
    """Confirm a server-created semantic candidate idempotently."""
    require_permission(tenant_context, "analyze")
    scope_data = _mapping_scope_data(tenant_context, payload.explorer_context)
    row = confirm_mapping(
        tenant_context,
        mapping_id,
        scope=scope_data,
        catalog_digest=get_registry().catalog_digest,
        session_id=payload.chat_session_id,
    )
    if row is None:
        raise HTTPException(status_code=409, detail="语义确认已过期、范围已变化或不属于当前会话")
    return {"status": row["status"], "mapping_id": row["mapping_id"], "execution_request_id": row.get("execution_request_id")}


@router.post("/semantic-mappings/{mapping_id}/skip")
def skip_semantic_mapping(
    mapping_id: str,
    payload: SemanticMappingSkipRequest,
    tenant_context: TenantContext = Depends(require_tenant_context),
) -> dict:
    """Persist the user's explicit choice not to use a candidate metric."""
    require_permission(tenant_context, "analyze")
    scope_data = _mapping_scope_data(tenant_context, payload.explorer_context)
    row = skip_mapping(
        tenant_context, mapping_id, scope=scope_data,
        catalog_digest=get_registry().catalog_digest,
        session_id=payload.chat_session_id,
    )
    if row is None:
        raise HTTPException(status_code=409, detail="候选已处理、已过期、范围已变化或不属于当前会话")
    return {"status": "skipped_by_user", "mapping_id": row["mapping_id"], "mention_id": row.get("mention_id")}


def _time_intent_clarification(message: str) -> dict | None:
    years = ambiguous_year_list(message)
    if not years:
        return None
    return {
        "status": "time_intent_required",
        "years": list(years),
        "original_message": message,
        "reason": f"检测到多个年份（{', '.join(map(str, years))}），但未说明要看年度趋势、两期对比还是分别查询。",
    }


@router.post("", response_model=ChatResponse)
def chat(payload: ChatRequest, tenant_context: TenantContext = Depends(require_tenant_context)) -> ChatResponse:
    """同步聊天接口：完整执行工作流后一次性返回结果。

    适用于批处理、脚本调用、不需要实时进度的场景。
    内部调用 run_chat_workflow()，等待图执行完毕后才返回。
    """
    # 关键打点说明：这里是同步链路的 HTTP 入口。JWT 依赖已经先把租户、用户和
    # scope_type 放入 tenant_context；随后 analyze 权限校验、任务关联/时间澄清、
    # 复杂分析预路由和 LangGraph 工作流依次执行。排查请求时应从本函数开始记录。
    try:
        require_permission(tenant_context, "analyze")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    try:
        seller_query_scope_fingerprint = _resolve_chat_seller_scope_fingerprint(
            tenant_context, payload.seller_scope_payload,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    memory_session_id = _memory_session_for_request(payload)
    lifecycle_binding: ChatLifecycleBinding | None = None
    requested_trace_id = str(uuid4())
    requested_thread_id = payload.thread_id or requested_trace_id
    if payload.use_current_explorer_context and payload.explorer_context is not None:
        try:
            lifecycle_binding = begin_chat_lifecycle(
                tenant_context,
                session_id=payload.chat_session_id,
                client_turn_id=payload.chat_turn_id,
                inherited_from_turn_id=payload.inherited_from_turn_id,
                question=payload.message,
                explorer_payload=payload.explorer_context.model_dump(mode="json"),
                thread_id=requested_thread_id,
                trace_id=requested_trace_id,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if lifecycle_binding and lifecycle_binding.is_replay:
            return ChatResponse(**replay_response(lifecycle_binding))
        if lifecycle_binding and lifecycle_binding.active_reconnect:
            raise HTTPException(status_code=409, detail="chat_turn_already_running; reconnect through /chat/stream")
        if lifecycle_binding and lifecycle_binding.execution is not None:
            requested_thread_id = lifecycle_binding.execution.thread_id or requested_thread_id
            requested_trace_id = lifecycle_binding.execution.trace_id or requested_trace_id
    elif payload.seller_scope_payload is not None and payload.chat_session_id and payload.chat_turn_id:
        # C extends begin_chat_lifecycle with this explicit seller carrier.
        # It is already verified above and stays separate from the legacy
        # Explorer lifecycle payload and its single-seller boundary.
        try:
            lifecycle_binding = begin_chat_lifecycle(
                tenant_context,
                session_id=payload.chat_session_id,
                client_turn_id=payload.chat_turn_id,
                inherited_from_turn_id=payload.inherited_from_turn_id,
                question=payload.message,
                explorer_payload=None,
                seller_scope_payload=payload.seller_scope_payload.model_dump(mode="json"),
                thread_id=requested_thread_id,
                trace_id=requested_trace_id,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if lifecycle_binding and lifecycle_binding.is_replay:
            return ChatResponse(**replay_response(lifecycle_binding))
        if lifecycle_binding and lifecycle_binding.active_reconnect:
            raise HTTPException(status_code=409, detail="chat_turn_already_running; reconnect through /chat/stream")
        if lifecycle_binding and lifecycle_binding.execution is not None:
            requested_thread_id = lifecycle_binding.execution.thread_id or requested_thread_id
            requested_trace_id = lifecycle_binding.execution.trace_id or requested_trace_id
    premise_resume = _is_suspended_premise_resume(lifecycle_binding, payload)
    task_id = payload.task_id
    relation_decision = {"action": "continue", "reason": "confirmed_suspended_premise"} if premise_resume else None
    request_scope_fingerprint = (
        lifecycle_binding.query_scope_fingerprint if lifecycle_binding
        else seller_query_scope_fingerprint or chat_context_scope_fingerprint(tenant_context)
    )
    expected_data_version = (
        _expected_data_manifest_hash(tenant_context, payload.explorer_context)
        if payload.use_current_explorer_context else None
    )
    scope_change_clarification = (
        _seller_scope_change_clarification(payload) if not premise_resume else None
    )
    if scope_change_clarification:
        response_data = {
            "task_type": "clarification",
            "answer": scope_change_clarification["reason"],
            "trace_id": requested_trace_id,
        }
        if task_id and lifecycle_binding is None:
            update_task(
                tenant_context, task_id, status="completed",
                summary={"objective": payload.message, "reason_code": scope_change_clarification["reason_code"]},
            )
        _persist_pre_graph_trace(
            tenant_context,
            trace_id=requested_trace_id,
            task_type="clarification",
            answer=response_data["answer"],
            clarification=scope_change_clarification,
            scope_fingerprint=request_scope_fingerprint,
            task_id=task_id,
            relation_decision=relation_decision,
            status="completed",
            interaction_state="none",
            node_name="pre_graph_terminal_refusal",
            summary="Stopped before workflow execution: seller scope change refused.",
        )
        _finalize_binding(lifecycle_binding, response_data, task_id=task_id, relation_decision=relation_decision)
        return ChatResponse(**response_data)
    if payload.chat_session_id and not premise_resume:
        decision = relation(
            tenant_context, payload.chat_session_id, payload.message,
            payload.explorer_context.model_dump(mode="json") if payload.explorer_context else None,
            task_id, scope_fingerprint=request_scope_fingerprint,
            expected_data_version=expected_data_version,
        )
        relation_decision = decision
        if decision["action"] == "clarify":
            response_data = {
                "task_type": "clarification",
                "answer": "请确认这是继续当前分析任务，还是开始新的分析任务。",
                "trace_id": requested_trace_id,
                "clarification": {"status":"task_relationship_required", "original_message": payload.message, **decision},
            }
            if task_id and lifecycle_binding is None:
                update_task(tenant_context, task_id, status="awaiting_confirmation", summary={"objective": payload.message, "reason": decision.get("reason")})
            _persist_pre_graph_trace(
                tenant_context,
                trace_id=requested_trace_id,
                task_type="clarification",
                answer=response_data["answer"],
                clarification=response_data["clarification"],
                scope_fingerprint=request_scope_fingerprint,
                task_id=task_id,
                relation_decision=relation_decision,
            )
            _finalize_binding(lifecycle_binding, response_data, task_id=task_id, relation_decision=relation_decision)
            return ChatResponse(**response_data)
        if decision["action"] == "new":
            task_id = create_task(tenant_context,payload.chat_session_id,decision.get("resolved_message") or payload.message,payload.explorer_context.model_dump(mode="json") if payload.explorer_context else None)["task_id"]
        else:
            task_id = decision.get("task_id")
        if task_id:
            update_task(tenant_context,task_id,status="running",reopen=decision["action"] == "continue")
    if lifecycle_binding and lifecycle_binding.execution and task_id:
        if not lifecycle_binding.service.bind_active_task(lifecycle_binding.execution, task_id):
            _finalize_binding(
                lifecycle_binding, None,
                failed=RuntimeError("session_focus_task_binding_failed"),
                task_id=task_id, relation_decision=relation_decision,
            )
            raise HTTPException(status_code=409, detail="Unable to bind task to the current session focus")
    time_clarification = _time_intent_clarification(payload.message)
    if time_clarification:
        response_data = {
            "task_type": "metric_query",
            "answer": "请先确认这些年份的分析方式。",
            "trace_id": requested_trace_id,
            "clarification": time_clarification,
        }
        if task_id and lifecycle_binding is None:
            update_task(tenant_context, task_id, status="awaiting_confirmation", summary={"objective": payload.message, "clarification": time_clarification})
        _persist_pre_graph_trace(
            tenant_context,
            trace_id=requested_trace_id,
            task_type="metric_query",
            answer=response_data["answer"],
            clarification=time_clarification,
            scope_fingerprint=request_scope_fingerprint,
            task_id=task_id,
            relation_decision=relation_decision,
        )
        _finalize_binding(lifecycle_binding, response_data, task_id=task_id, relation_decision=relation_decision)
        return ChatResponse(**response_data)
    # 进入核心图执行：thread_id 决定 LangGraph checkpoint 的线程；memory_session_id
    # 和 chat_session_id 是跨轮会话标识，不能把三者混为同一个生命周期概念。
    result_context_holder: dict = {}
    try:
        result = run_chat_workflow(
            payload.message, requested_thread_id, tenant_context,
            payload.explorer_context.model_dump(mode="json") if payload.explorer_context else None,
            payload.use_current_explorer_context, memory_session_id, payload.memory_enabled,
            payload.metric_confirmation, payload.chat_session_id, payload.chat_turn_id, task_id,
            payload.semantic_mapping_ids, payload.skipped_semantic_mapping_ids,
            persistence_turn_id=lifecycle_binding.persistence_turn_id if lifecycle_binding else None,
            authorization_boundary=lifecycle_binding.authorization_boundary if lifecycle_binding else None,
            trace_id=requested_trace_id,
            inherited_from_turn_id=payload.inherited_from_turn_id,
            turn_relation_decision=relation_decision,
            seller_scope_payload=(
                payload.seller_scope_payload.model_dump(mode="json")
                if payload.seller_scope_payload else None
            ),

            on_result_context=result_context_holder.update,
        )
    except PermissionError as exc:
        _finalize_binding(
            lifecycle_binding, None, failed=exc, task_id=task_id,
            relation_decision=relation_decision,
        )
        raise HTTPException(status_code=403, detail="Pending workflow checkpoint is outside the current session or scope.") from exc
    except Exception as exc:
        _finalize_binding(lifecycle_binding, None, failed=exc, task_id=task_id, relation_decision=relation_decision)
        if task_id and lifecycle_binding is None:
            update_task(tenant_context, task_id, status="failed", trace_id="", summary={"objective": payload.message, "error": str(exc)})
        raise HTTPException(status_code=500, detail="Chat execution failed; the trace was retained.") from exc
    if result_context_holder:
        result["result_context"] = dict(result_context_holder)
    _finalize_binding(lifecycle_binding, result, task_id=task_id, relation_decision=relation_decision)
    response_result = dict(result)
    response_result.pop("result_context", None)
    if task_id:
        final_status = _task_status_from_result(result)
        final_trace_id = str(result.get("trace_id") or "")
        if lifecycle_binding is None:
            update_task(
                tenant_context, task_id, status=final_status,
                trace_id=final_trace_id,
                summary={"objective": payload.message, "task_type": result.get("task_type")},
            )
        if lifecycle_binding is None:
            _record_progress_best_effort(
                tenant_context, task_id, status=final_status, trace_id=final_trace_id,
                turn_id=payload.chat_turn_id,
                scope_fingerprint=request_scope_fingerprint,
                relation_decision=relation_decision,
            )
    return ChatResponse(**response_result)


@router.post("/stream")
def chat_stream(
    payload: ChatRequest,
    tenant_context: TenantContext = Depends(require_tenant_context),
) -> StreamingResponse:
    """SSE 流式聊天接口：将工作流节点事件实时推送给前端。"""
    memory_session_id = _memory_session_for_request(payload)
    def events():
        # 在可能调用慢速 LLM 的任务关系解析之前，先把 started 事件刷给前端。
        # 这样浏览器或反向代理能区分“请求仍在工作”和“连接已经失活”。
        # 流式接口先生成一个 SSE 请求级 trace_id，让前端在真正进入图之前就能
        # 看到 started 事件；后续节点事件和 completed 事件应沿用这条可观测链路。
        stream_trace_id = str(uuid4())
        lifecycle_binding: ChatLifecycleBinding | None = None
        relation_decision = None
        stream_turn_resolution: dict = {}
        stream_result_context: dict = {}
        yield f"data: {json.dumps({'event':'started','trace_id':stream_trace_id,'sequence_number':0}, ensure_ascii=False)}\n\n"
        try:
            require_permission(tenant_context, "analyze")
            seller_query_scope_fingerprint = _resolve_chat_seller_scope_fingerprint(
                tenant_context, payload.seller_scope_payload,
            )
            task_id = payload.task_id
            requested_thread_id = payload.thread_id or stream_trace_id
            requested_trace_id = stream_trace_id
            if payload.use_current_explorer_context and payload.explorer_context is not None:
                lifecycle_binding = begin_chat_lifecycle(
                    tenant_context,
                    session_id=payload.chat_session_id,
                    client_turn_id=payload.chat_turn_id,
                    inherited_from_turn_id=payload.inherited_from_turn_id,
                    question=payload.message,
                    explorer_payload=payload.explorer_context.model_dump(mode="json"),
                    thread_id=requested_thread_id,
                    trace_id=requested_trace_id,
                )
                if lifecycle_binding and lifecycle_binding.is_replay:
                    replay = replay_response(lifecycle_binding)
                    yield f"data: {json.dumps({'event': 'completed', 'sequence_number': 0, **replay}, ensure_ascii=False, default=str)}\n\n"
                    return
                if lifecycle_binding and lifecycle_binding.active_reconnect:
                    sequence = 1
                    while True:
                        snapshot = lifecycle_binding.service.current_turn_snapshot(
                            lifecycle_binding.persistence_turn_id,
                        )
                        if snapshot is None:
                            yield f"data: {json.dumps({'event':'failed','sequence_number':sequence,'detail':'durable_turn_missing'}, ensure_ascii=False)}\n\n"
                            return
                        status = str(snapshot.get("status") or "")
                        if status in {"completed", "abstained", "failed", "cancelled", "interrupted"}:
                            result_snapshot = dict(snapshot.get("result_snapshot") or {})
                            stored_task_type = snapshot.get("task_type")
                            snapshot_task_type = result_snapshot.get("task_type")
                            metadata = lifecycle_binding.service.commit_metadata(
                                lifecycle_binding.persistence_turn_id,
                            )
                            replay = {
                                "event": "completed",
                                "sequence_number": sequence,
                                "task_type": resolve_turn_task_type(stored_task_type, snapshot_task_type),
                                "answer": snapshot.get("answer") or result_snapshot.get("answer") or "",
                                "trace_id": snapshot.get("trace_id") or result_snapshot.get("trace_id") or "",
                                "error": status in {"abstained", "failed", "cancelled", "interrupted"},
                                "errors": result_snapshot.get("errors") or [],
                                "clarification": result_snapshot.get("clarification"),
                                "presentation": result_snapshot.get("presentation"),
                                "durable_turn_id": metadata.get("turn_id") or lifecycle_binding.persistence_turn_id,
                                "execution_id": metadata.get("final_execution_id"),
                                "commit_version": metadata.get("commit_version"),
                            }
                            record_event("chat.sse_completed_after_turn_commit", {
                                "trace_id": replay["trace_id"],
                                "durable_turn_id": replay["durable_turn_id"],
                                "reconnected": True,
                            })
                            yield f"data: {json.dumps(replay, ensure_ascii=False, default=str)}\n\n"
                            return
                        heartbeat = {"event": "heartbeat", "sequence_number": sequence, "trace_id": stream_trace_id}
                        yield f"data: {json.dumps(heartbeat, ensure_ascii=False)}\n\n"
                        sequence += 1
                        time.sleep(1)
                if lifecycle_binding and lifecycle_binding.execution is not None:
                    requested_thread_id = lifecycle_binding.execution.thread_id or requested_thread_id
                    requested_trace_id = lifecycle_binding.execution.trace_id or requested_trace_id
            elif payload.seller_scope_payload is not None and payload.chat_session_id and payload.chat_turn_id:
                # C extends begin_chat_lifecycle with this explicit seller
                # carrier. It remains disjoint from Explorer lifecycle scope.
                lifecycle_binding = begin_chat_lifecycle(
                    tenant_context,
                    session_id=payload.chat_session_id,
                    client_turn_id=payload.chat_turn_id,
                    inherited_from_turn_id=payload.inherited_from_turn_id,
                    question=payload.message,
                    explorer_payload=None,
                    seller_scope_payload=payload.seller_scope_payload.model_dump(mode="json"),
                    thread_id=requested_thread_id,
                    trace_id=requested_trace_id,
                )
                if lifecycle_binding and lifecycle_binding.is_replay:
                    replay = replay_response(lifecycle_binding)
                    yield f"data: {json.dumps({'event': 'completed', 'sequence_number': 0, **replay}, ensure_ascii=False, default=str)}\n\n"
                    return
                if lifecycle_binding and lifecycle_binding.active_reconnect:
                    raise HTTPException(status_code=409, detail="chat_turn_already_running; reconnect through /chat/stream")
                if lifecycle_binding and lifecycle_binding.execution is not None:
                    requested_thread_id = lifecycle_binding.execution.thread_id or requested_thread_id
                    requested_trace_id = lifecycle_binding.execution.trace_id or requested_trace_id
            premise_resume = _is_suspended_premise_resume(lifecycle_binding, payload)
            if premise_resume:
                relation_decision = {"action": "continue", "reason": "confirmed_suspended_premise"}
            request_scope_fingerprint = (
                lifecycle_binding.query_scope_fingerprint if lifecycle_binding
                else seller_query_scope_fingerprint or chat_context_scope_fingerprint(tenant_context)
            )
            expected_data_version = (
                _expected_data_manifest_hash(tenant_context, payload.explorer_context)
                if payload.use_current_explorer_context else None
            )
            scope_change_clarification = (
                _seller_scope_change_clarification(payload) if not premise_resume else None
            )
            if scope_change_clarification:
                event = {
                    "event": "completed",
                    "sequence_number": 0,
                    "task_type": "clarification",
                    "answer": scope_change_clarification["reason"],
                    "trace_id": requested_trace_id,
                }
                if task_id and lifecycle_binding is None:
                    update_task(
                        tenant_context, task_id, status="completed",
                        summary={"objective": payload.message, "reason_code": scope_change_clarification["reason_code"]},
                    )
                _persist_pre_graph_trace(
                    tenant_context,
                    trace_id=requested_trace_id,
                    task_type="clarification",
                    answer=event["answer"],
                    clarification=scope_change_clarification,
                    scope_fingerprint=request_scope_fingerprint,
                    task_id=task_id,
                    relation_decision=relation_decision,
                    status="completed",
                    interaction_state="none",
                    node_name="pre_graph_terminal_refusal",
                    summary="Stopped before workflow execution: seller scope change refused.",
                )
                _finalize_binding(lifecycle_binding, event, task_id=task_id, relation_decision=relation_decision)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                return
            if payload.chat_session_id and not premise_resume:
                decision = relation(
                    tenant_context, payload.chat_session_id, payload.message,
                    payload.explorer_context.model_dump(mode="json") if payload.explorer_context else None,
                    task_id, scope_fingerprint=request_scope_fingerprint,
                    expected_data_version=expected_data_version,
                )
                relation_decision = decision
                if decision["action"] == "clarify":
                    event = {
                        "event": "completed",
                        "sequence_number": 0,
                        "task_type": "clarification",
                        "answer": "请确认这是继续当前分析任务，还是开始新的分析任务。",
                        "trace_id": requested_trace_id,
                        "clarification": {"status":"task_relationship_required", "original_message":payload.message, **decision},
                    }
                    if task_id and lifecycle_binding is None:
                        update_task(
                            tenant_context, task_id, status="awaiting_confirmation",
                            summary={"objective": payload.message, "reason": decision.get("reason")},
                        )
                    _persist_pre_graph_trace(
                        tenant_context,
                        trace_id=requested_trace_id,
                        task_type="clarification",
                        answer=event["answer"],
                        clarification=event["clarification"],
                        scope_fingerprint=request_scope_fingerprint,
                        task_id=task_id,
                        relation_decision=relation_decision,
                    )
                    _finalize_binding(lifecycle_binding, event, task_id=task_id, relation_decision=relation_decision)
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    return
                if decision["action"] == "new":
                    task_id=create_task(tenant_context,payload.chat_session_id,decision.get("resolved_message") or payload.message,payload.explorer_context.model_dump(mode="json") if payload.explorer_context else None)["task_id"]
                else:
                    task_id=decision.get("task_id")
                if task_id:
                    update_task(tenant_context,task_id,status="running",reopen=decision["action"] == "continue")
            if lifecycle_binding and lifecycle_binding.execution and task_id:
                if not lifecycle_binding.service.bind_active_task(lifecycle_binding.execution, task_id):
                    raise RuntimeError("session_focus_task_binding_failed")
            time_clarification = _time_intent_clarification(payload.message)
            if time_clarification:
                event={"event":"completed","sequence_number":0,"task_type":"metric_query","answer":"请先确认这些年份的分析方式。","trace_id":requested_trace_id,"clarification":time_clarification}
                if task_id and lifecycle_binding is None:
                    update_task(
                        tenant_context, task_id, status="awaiting_confirmation",
                        summary={"objective": payload.message, "clarification": time_clarification},
                    )
                _persist_pre_graph_trace(
                    tenant_context,
                    trace_id=requested_trace_id,
                    task_type="metric_query",
                    answer=event["answer"],
                    clarification=time_clarification,
                    scope_fingerprint=request_scope_fingerprint,
                    task_id=task_id,
                    relation_decision=relation_decision,
                )
                _finalize_binding(lifecycle_binding, event, task_id=task_id, relation_decision=relation_decision)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                return
            # The graph runs in a dedicated thread so a client disconnect cannot
            # cancel the durable execution before Turn Commit. The request
            # generator only forwards events; the worker finalizes the Turn even
            # when no browser is listening, and a reconnect observes the
            # committed result through the durable lifecycle state.
            event_queue: queue.Queue = queue.Queue()

            def _consume_and_commit() -> None:
                try:
                    for event in stream_chat_workflow(
                        payload.message, requested_thread_id, tenant_context,
                        payload.explorer_context.model_dump(mode="json") if payload.explorer_context else None,
                        payload.use_current_explorer_context, memory_session_id, payload.memory_enabled,
                        payload.metric_confirmation, payload.chat_session_id, payload.chat_turn_id, task_id,
                        requested_trace_id, False, payload.semantic_mapping_ids, payload.skipped_semantic_mapping_ids,
                        persistence_turn_id=lifecycle_binding.persistence_turn_id if lifecycle_binding else None,
                        authorization_boundary=lifecycle_binding.authorization_boundary if lifecycle_binding else None,
                        inherited_from_turn_id=payload.inherited_from_turn_id,
                        turn_relation_decision=relation_decision,
                        seller_scope_payload=(
                            payload.seller_scope_payload.model_dump(mode="json")
                            if payload.seller_scope_payload else None
                        ),

                        on_turn_resolution=lambda value: stream_turn_resolution.update(value),
                        on_result_context=stream_result_context.update,
                    ):
                        if event.get("event") == "completed" and task_id:
                            final_status = _task_status_from_result(event)
                            final_trace_id = str(event.get("trace_id") or "")
                            if lifecycle_binding is None:
                                update_task(
                                    tenant_context, task_id, status=final_status,
                                    trace_id=final_trace_id,
                                    summary={"objective": payload.message, "task_type": event.get("task_type")},
                                )
                        if event.get("event") == "completed":
                            if stream_result_context:
                                event["result_context"] = dict(stream_result_context)
                            try:
                                _finalize_binding(
                                    lifecycle_binding, event, task_id=task_id,
                                    relation_decision=relation_decision,
                                    turn_resolution=stream_turn_resolution,
                                )
                            finally:
                                event.pop("result_context", None)
                            if lifecycle_binding and lifecycle_binding.execution:
                                record_event("chat.sse_completed_after_turn_commit", {
                                    "trace_id": event.get("trace_id"),
                                    "durable_turn_id": lifecycle_binding.persistence_turn_id,
                                })
                            if task_id and lifecycle_binding is None:
                                _record_progress_best_effort(
                                    tenant_context, task_id, status=final_status, trace_id=final_trace_id,
                                    turn_id=payload.chat_turn_id,
                                    scope_fingerprint=request_scope_fingerprint,
                                    relation_decision=relation_decision,
                                )
                        elif event.get("event") == "suspended":
                            _finalize_binding(lifecycle_binding, event, task_id=task_id, relation_decision=relation_decision)
                        elif event.get("event") == "failed":
                            if task_id and lifecycle_binding is None:
                                update_task(
                                    tenant_context, task_id, status="failed",
                                    trace_id=stream_trace_id,
                                    summary={"objective": payload.message, "error": event.get("detail")},
                                )
                            _finalize_binding(
                                lifecycle_binding, None,
                                failed=RuntimeError(str(event.get("detail") or "workflow failed")),
                                task_id=task_id,
                            )
                        event_queue.put(event)
                except BaseException as exc:  # noqa: BLE001 - persist a terminal state for the worker
                    try:
                        _finalize_binding(
                            lifecycle_binding, None, failed=exc,
                            task_id=task_id, relation_decision=relation_decision,
                        )
                    except Exception:  # noqa: BLE001 - keep the worker alive for diagnostics
                        LOGGER.exception("chat stream worker failed to finalize after error")
                    if task_id and lifecycle_binding is None:
                        try:
                            update_task(
                                tenant_context, task_id, status="failed", trace_id=stream_trace_id,
                                summary={"objective": payload.message, "error": str(exc)},
                            )
                        except Exception:  # noqa: BLE001
                            LOGGER.exception("chat stream worker failed to mark task failed")
                    event_queue.put({"event": "failed", "detail": str(exc)})
                finally:
                    event_queue.put(None)

            threading.Thread(
                target=_consume_and_commit,
                daemon=True,
                name=f"chat-stream-{stream_trace_id}",
            ).start()
            while True:
                queued_event = event_queue.get()
                if queued_event is None:
                    break
                yield f"data: {json.dumps(queued_event, ensure_ascii=False, default=str)}\n\n"
        except Exception as exc:
            _finalize_binding(lifecycle_binding, None, failed=exc, task_id=task_id, relation_decision=relation_decision)
            if task_id and lifecycle_binding is None:
                update_task(tenant_context, task_id, status="failed", trace_id=stream_trace_id, summary={"objective": payload.message, "error": str(exc)})
            yield f"data: {json.dumps({'event': 'failed', 'detail': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache, no-store, no-transform", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


@router.post("/{thread_id}/review", response_model=ChatResponse)
def review(
    thread_id: str, payload: HumanReviewRequest,
    authenticated_actor: str = Depends(require_analysis_write),
    tenant_context: TenantContext = Depends(require_tenant_context),
) -> ChatResponse:
    """人工审核恢复接口：提交审核决定，从中断点继续图执行。

    当 critic 裁决为 human_review 时，图在 human_review 节点中断，
    等待外部审核。本接口接收审核决定后，LangGraph 从检查点恢复执行。

    参数：
    - thread_id：工作流线程 ID（对应检查点的 thread_id）
    - payload：审核决定（action, actor, reason, answer）

    权限：
    - 需要 analysis_write_token（写报告权限）
    - 审核人必须与 payload.actor 一致（防止越权审批）
    - 检查点所属租户必须与当前 tenant_context 一致（防止跨租户恢复）
    """
    require_actor_match(authenticated_actor, payload.actor)
    try:
        return ChatResponse(**resume_human_review(thread_id, payload.model_dump(exclude_none=True), tenant_context))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="Cross-tenant review resume denied") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/test-llm")
def test_llm() -> dict:
    """测试 LLM API 连接和响应是否正常。"""
    import time

    from app.llm.client import llm_chat

    start = time.time()
    reply = llm_chat(
        system_prompt="你是一个连接测试工具，只回复一个词。",
        user_message='回复"ok"即可，不要多说其他内容。',
        # 某些支持推理的兼容模型会先消耗 reasoning token，之后才输出可见内容。
        # 这里使用固定且较小但足够完成连通性检查的输出预算，避免测试结果受模型随机性影响。
        temperature=0.0,
        max_tokens=128,
    )
    elapsed_ms = round((time.time() - start) * 1000)

    if reply:
        return {"status": "ok", "reply": reply, "latency_ms": elapsed_ms}
    return {"status": "error", "detail": "LLM 无响应，请检查 API Key 和 Base URL 配置是否正确。"}

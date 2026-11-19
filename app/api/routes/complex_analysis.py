"""HTTP API for confirmation-gated whitelist analyses and action drafts."""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text

from app.api.auth import require_tenant_context
from app.complex_analysis.service import cancel_plan, execute_plan, get_plan_v2, revise_plan
from app.data.database import engine
from app.iam.service import require_permission
from app.tenancy import TenantContext
from app.core.exceptions import sanitize_exception

router = APIRouter(prefix="/complex-analysis", tags=["complex-analysis"])
LOGGER = logging.getLogger(__name__)


class PlanCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000)
    explorer_context: dict
    chat_session_id: str | None = Field(default=None, max_length=128)
    chat_turn_id: str | None = Field(default=None, max_length=128)


class PlanRevisionRequest(BaseModel):
    explorer_context: dict | None = None
    focus_key: str | None = Field(default=None, max_length=80)
    note: str | None = Field(default=None, max_length=280)


class ActionPlanUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    content: dict | None = None
    status: Literal["draft", "accepted", "archived"] | None = None


def _scope(conn, context: TenantContext) -> None:
    conn.execute(text("SELECT set_config('app.tenant_id', :v, true)"), {"v": context.tenant_id})
    conn.execute(text("SELECT set_config('app.user_id', :v, true)"), {"v": context.user_id})


def _require_analyze(context: TenantContext) -> None:
    try:
        require_permission(context, "analyze")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def _resume_plan_lifecycle(context: TenantContext, plan: dict) -> tuple[object | None, str | None]:
    """Reconnect the suspended chat turn created by the plan request, if any."""

    session_id = str(plan.get("chat_session_id") or "")
    client_turn_id = str(plan.get("chat_turn_id") or "")
    scope = plan.get("scope") if isinstance(plan.get("scope"), dict) else {}
    params = plan.get("params") if isinstance(plan.get("params"), dict) else {}
    if not session_id or not client_turn_id:
        return None, None

    with engine.begin() as conn:
        _scope(conn, context)
        linked = conn.execute(text("""
            SELECT t.task_id
            FROM governance.chat_turns t
            JOIN governance.conversation_executions e
              ON e.turn_id=t.turn_id AND e.tenant_id=t.tenant_id AND e.user_id=t.user_id
            WHERE t.tenant_id=:tenant AND t.user_id=:user
              AND t.chat_session_id=:session AND t.client_turn_id=:client_turn
              AND t.status='running' AND e.status='suspended'
            ORDER BY e.attempt_no DESC LIMIT 1
        """), {
            "tenant": context.tenant_id,
            "user": context.user_id,
            "session": session_id,
            "client_turn": client_turn_id,
        }).mappings().one_or_none()
    if linked is None:
        return None, None

    # Re-run the same authorization and canonical-scope checks used by the
    # original chat request before resuming its suspended lifecycle execution.
    from app.chat_lifecycle.integration import begin_chat_lifecycle

    binding = begin_chat_lifecycle(
        context,
        session_id=session_id,
        client_turn_id=client_turn_id,
        question=str(plan.get("question") or "复杂分析计划"),
        explorer_payload={
            "seller_id": scope.get("seller_id"),
            "metric_name": scope.get("metric_name") or "gmv",
            "start_date": params.get("start_date") or scope.get("default_start_date"),
            "end_date": params.get("end_date") or scope.get("default_end_date"),
            "granularity": scope.get("granularity") or "day",
        },
        thread_id="",
        trace_id=str(plan.get("plan_id") or ""),
    )
    if binding is None or binding.execution is None:
        raise RuntimeError("complex_plan_chat_lifecycle_not_resumable")
    return binding, str(linked.get("task_id") or "") or None


def _finalize_plan_lifecycle(binding, *, result: dict, status: str, task_id: str | None) -> None:
    from app.chat_lifecycle.integration import finalize_chat_lifecycle

    error_code = str(result.get("error_code") or "") or None
    finalize_chat_lifecycle(
        binding,
        result=result,
        status=status,
        interaction_state="none",
        error_code=error_code,
        error_detail={"code": error_code} if error_code else None,
        task_id=task_id,
        task_status=status if task_id else None,
    )


def _finalize_expired_revision(context: TenantContext, plan_id: str) -> None:
    """Close a suspended chat turn when revise discovers and commits expiry."""

    plan = get_plan_v2(context, plan_id)
    if not plan or plan.get("status") != "expired":
        return
    binding, task_id = _resume_plan_lifecycle(context, plan)
    if binding is None:
        return
    _finalize_plan_lifecycle(
        binding,
        result={
            "task_type": "complex_analysis",
            "answer": "复杂分析计划已过期，请重新发起分析。",
            "trace_id": str(plan.get("trace_id") or ""),
            "error": True,
            "error_code": "COMPLEX_PLAN_EXPIRED",
        },
        status="failed",
        task_id=task_id,
    )


@router.post("/plans")
def make_plan(payload: PlanCreateRequest, context: TenantContext = Depends(require_tenant_context)) -> dict:
    _require_analyze(context)
    # Phase 0 has no online semantic router. A public natural-language request
    # cannot manufacture the server-sealed route decision required by the plan
    # service; template confirmation uses its separate protocol in a later phase.
    raise HTTPException(status_code=409, detail="sealed_route_decision_required")


@router.get("/plans/{plan_id}")
def read_plan(plan_id: str, context: TenantContext = Depends(require_tenant_context)) -> dict:
    plan = get_plan_v2(context, plan_id)
    if not plan: raise HTTPException(status_code=404, detail="Complex analysis plan not found")
    return plan


@router.post("/plans/{plan_id}/revise")
def revise(plan_id: str, payload: PlanRevisionRequest, context: TenantContext = Depends(require_tenant_context)) -> dict:
    _require_analyze(context)
    try:
        return revise_plan(context, plan_id, payload.explorer_context, {"focus_key": payload.focus_key, "note": payload.note, "source": "user"})
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail="Complex analysis plan not found") from exc
    except ValueError as exc:
        if "expired" in str(exc).lower():
            try:
                _finalize_expired_revision(context, plan_id)
            except Exception:  # noqa: BLE001 - preserve the 409 while recording lifecycle failure
                LOGGER.exception("Unable to finalize expired complex-plan revision lifecycle")
        raise HTTPException(status_code=409 if "pending" in str(exc) or "expired" in str(exc) else 400, detail=str(exc)) from exc


@router.post("/plans/{plan_id}/cancel")
def cancel(plan_id: str, context: TenantContext = Depends(require_tenant_context)) -> dict:
    _require_analyze(context)
    if not cancel_plan(context, plan_id): raise HTTPException(status_code=409, detail="Plan cannot be cancelled")
    return {"cancelled": True}


@router.post("/plans/{plan_id}/confirm")
def confirm(plan_id: str, context: TenantContext = Depends(require_tenant_context)) -> StreamingResponse:
    _require_analyze(context)
    def events():
        yield f"data: {json.dumps({'event':'started','trace_id':plan_id,'sequence_number':0}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'event':'node_started','node':'execute_complex_analysis','sequence_number':1}, ensure_ascii=False)}\n\n"
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="complex-analysis-sse")
        lifecycle_binding = None
        task_id = None
        try:
            plan = get_plan_v2(context, plan_id)
            if plan is None:
                raise PermissionError("Complex analysis plan not found")
            lifecycle_binding, task_id = _resume_plan_lifecycle(context, plan)
            future = executor.submit(execute_plan, context, plan_id)
            sequence = 2
            while True:
                try:
                    result = future.result(timeout=2)
                    break
                except FutureTimeout:
                    # Keep slow template/LLM execution visible to browsers and
                    # reverse proxies. The work continues in the worker.
                    yield f"data: {json.dumps({'event':'heartbeat','trace_id':plan_id,'sequence_number':sequence}, ensure_ascii=False)}\n\n"
                    sequence += 1
            final_status = "cancelled" if result.get("cancelled") else "completed"
            if lifecycle_binding is not None:
                _finalize_plan_lifecycle(
                    lifecycle_binding, result=result, status=final_status, task_id=task_id,
                )
            yield f"data: {json.dumps({'event':'node_completed','node':'execute_complex_analysis','task_type':'complex_analysis','sequence_number':sequence}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'event':'completed','sequence_number':sequence + 1,'error':False, **result}, ensure_ascii=False, default=str)}\n\n"
        except Exception as exc:
            try:
                current_plan = get_plan_v2(context, plan_id)
                current_status = str((current_plan or {}).get("status") or "")
                if lifecycle_binding is not None and current_status in {"failed", "expired", "cancelled"}:
                    failure_result = {
                        "task_type": "complex_analysis",
                        "answer": "复杂分析未能完成。",
                        "trace_id": str((current_plan or {}).get("trace_id") or ""),
                        "error": True,
                        "error_code": "COMPLEX_PLAN_EXECUTION_FAILED",
                    }
                    _finalize_plan_lifecycle(
                        lifecycle_binding,
                        result=failure_result,
                        status="cancelled" if current_status == "cancelled" else "failed",
                        task_id=task_id,
                    )
            except Exception:  # noqa: BLE001 - keep the SSE failure sanitized
                pass
            public_message, _ = sanitize_exception(exc)
            # Validation errors are already safe and more helpful than the
            # generic error mapping; unexpected exceptions stay sanitized.
            detail = str(exc) if isinstance(exc, (ValueError, PermissionError)) else public_message
            yield f"data: {json.dumps({'event':'failed','sequence_number':sequence if 'sequence' in locals() else 2,'detail':detail}, ensure_ascii=False)}\n\n"
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control":"no-cache, no-store, no-transform", "Connection":"keep-alive", "X-Accel-Buffering":"no"})


@router.get("/executions")
def list_executions(plan_id: str | None = Query(default=None, max_length=64), offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=50), context: TenantContext = Depends(require_tenant_context)) -> dict:
    with engine.begin() as conn:
        _scope(conn, context)
        rows = conn.execute(text("""SELECT execution_id,complex_plan_id,plan_revision,template_name,status,plan_hash,input_fingerprint,
          template_code_hash,result_hash,trace_id,error_code,started_at,finished_at,duration_ms
          FROM governance.analysis_execution_audits
          WHERE tenant_id=:tenant AND user_id=:user
            AND (CAST(:plan_id AS varchar) IS NULL OR complex_plan_id=CAST(:plan_id AS varchar))
          ORDER BY started_at DESC OFFSET :offset LIMIT :limit"""), {
              "tenant": context.tenant_id, "user": context.user_id,
              "plan_id": plan_id, "offset": offset, "limit": limit + 1,
          }).mappings().all()
    return {"items": [dict(row) for row in rows[:limit]], "next_offset": offset + limit if len(rows) > limit else None}


@router.get("/action-plans")
def list_action_plans(offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=50), context: TenantContext = Depends(require_tenant_context)) -> dict:
    with engine.begin() as conn:
        _scope(conn, context)
        rows = conn.execute(text("""SELECT action_plan_id,title,status,version,complex_plan_id,content,created_at,updated_at
          FROM governance.action_plans WHERE tenant_id=:tenant AND user_id=:user
          ORDER BY updated_at DESC OFFSET :offset LIMIT :limit"""), {
              "tenant": context.tenant_id, "user": context.user_id,
              "offset": offset, "limit": limit + 1,
          }).mappings().all()
    return {"items":[dict(r) for r in rows[:limit]], "next_offset": offset+limit if len(rows)>limit else None}


@router.get("/action-plans/{action_plan_id}")
def get_action_plan(action_plan_id: str, context: TenantContext = Depends(require_tenant_context)) -> dict:
    with engine.begin() as conn:
        _scope(conn, context)
        row=conn.execute(text("SELECT * FROM governance.action_plans WHERE action_plan_id=:id AND tenant_id=:tenant AND user_id=:user"), {
            "id": action_plan_id, "tenant": context.tenant_id, "user": context.user_id,
        }).mappings().one_or_none()
    if not row: raise HTTPException(status_code=404, detail="Action plan not found")
    return dict(row)


@router.put("/action-plans/{action_plan_id}")
def update_action_plan(action_plan_id: str, payload: ActionPlanUpdate, context: TenantContext = Depends(require_tenant_context)) -> dict:
    _require_analyze(context)
    if not payload.model_fields_set: raise HTTPException(status_code=400, detail="No changes supplied")
    with engine.begin() as conn:
        _scope(conn, context)
        row=conn.execute(text("SELECT title,content,status FROM governance.action_plans WHERE action_plan_id=:id AND tenant_id=:tenant AND user_id=:user FOR UPDATE"), {
            "id": action_plan_id, "tenant": context.tenant_id, "user": context.user_id,
        }).mappings().one_or_none()
        if not row: raise HTTPException(status_code=404, detail="Action plan not found")
        updated=conn.execute(text("""UPDATE governance.action_plans SET title=:title,content=CAST(:content AS jsonb),status=:status,version=version+1,updated_at=now()
          WHERE action_plan_id=:id AND tenant_id=:tenant AND user_id=:user
          RETURNING action_plan_id,title,content,status,version,updated_at"""), {
              "id": action_plan_id, "tenant": context.tenant_id, "user": context.user_id,
              "title": payload.title or row["title"],
              "content": json.dumps(payload.content if payload.content is not None else row["content"], ensure_ascii=False),
              "status": payload.status or row["status"],
          }).mappings().one()
    return dict(updated)


@router.delete("/action-plans/{action_plan_id}")
def delete_action_plan(action_plan_id: str, context: TenantContext = Depends(require_tenant_context)) -> dict:
    _require_analyze(context)
    with engine.begin() as conn:
        _scope(conn, context)
        row=conn.execute(text("DELETE FROM governance.action_plans WHERE action_plan_id=:id AND tenant_id=:tenant AND user_id=:user RETURNING action_plan_id"), {
            "id": action_plan_id, "tenant": context.tenant_id, "user": context.user_id,
        }).one_or_none()
    if not row: raise HTTPException(status_code=404, detail="Action plan not found")
    return {"deleted":True}

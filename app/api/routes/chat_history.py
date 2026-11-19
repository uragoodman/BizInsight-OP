"""Lazy-loading chat-history HTTP API."""

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.auth import require_tenant_context
from app.chat_history.repository import ChatHistoryRepository
from app.auth.context import build_authorization_context
from app.data.repositories.explorer import controlled_scope, preview_explorer
from app.iam.service import require_permission
from app.schemas.explorer import ExplorerQuery
from app.scope.canonical import CanonicalScope
from app.tenancy import TenantContext
from app.tasks.service import create_task, task_context, update_task
from pydantic import BaseModel, Field

router = APIRouter(prefix="/chat/sessions", tags=["chat-history"])

class TaskCreate(BaseModel):
    objective: str = Field(min_length=1, max_length=4000)
    scope: dict | None = None
class SessionCreate(BaseModel):
    title: str = Field(default="新会话", max_length=160)
class SessionScopeUpdate(BaseModel):
    scope: ExplorerQuery | None
class TaskDecision(BaseModel):
    action: str = Field(pattern="^(continue|new)$")
    note: str = Field(default="", max_length=280)


@router.get("")
def list_sessions(offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=50),
                  context: TenantContext = Depends(require_tenant_context)) -> dict:
    items, has_more = ChatHistoryRepository(context).list_sessions(offset, limit)
    return {"items": items, "next_offset": offset + len(items) if has_more else None}


@router.post("", status_code=201)
def create_session(payload: SessionCreate, context: TenantContext = Depends(require_tenant_context)) -> dict:
    return ChatHistoryRepository(context).create_session(payload.title)


@router.put("/{session_id}/scope")
def update_session_scope(
    session_id: str,
    payload: SessionScopeUpdate,
    context: TenantContext = Depends(require_tenant_context),
) -> dict:
    scope = None
    scope_fingerprint = None
    if payload.scope is not None:
        try:
            require_permission(context, "analyze")
            preview = preview_explorer(context, payload.scope)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if preview.get("summary", {}).get("status") != "ok":
            raise HTTPException(status_code=409, detail="当前范围没有可用数据")
        validated = preview["context"]
        scope = {
            "seller_id": validated["seller_id"],
            "metric_name": validated["metric_name"],
            "start_date": validated["start_date"],
            "end_date": validated["end_date"],
            "granularity": validated["granularity"],
        }
        controlled = controlled_scope(preview)
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
        scope_fingerprint = canonical.query_scope_fingerprint
    return ChatHistoryRepository(context).set_session_scope(
        session_id, scope, scope_fingerprint=scope_fingerprint,
    )


@router.get("/{session_id}/turns")
def list_turns(session_id: str, offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100),
               context: TenantContext = Depends(require_tenant_context)) -> dict:
    items, has_more = ChatHistoryRepository(context).list_turns(session_id, offset, limit)
    return {"items": items, "next_offset": offset + len(items) if has_more else None}

@router.get("/{session_id}/task-context")
def get_task_context(session_id: str, context: TenantContext = Depends(require_tenant_context)) -> dict:
    return task_context(context, session_id)

@router.post("/{session_id}/tasks")
def add_task(session_id: str, payload: TaskCreate, context: TenantContext = Depends(require_tenant_context)) -> dict:
    return create_task(context, session_id, payload.objective, payload.scope)

@router.post("/tasks/{task_id}/decision")
def decide_task(task_id: str, payload: TaskDecision, context: TenantContext = Depends(require_tenant_context)) -> dict:
    return update_task(context, task_id, status="running" if payload.action == "continue" else "superseded", pending={"decision":payload.action,"note":payload.note}, reopen=payload.action == "continue") or {}


@router.delete("/{session_id}")
def delete_session(session_id: str, context: TenantContext = Depends(require_tenant_context)) -> dict:
    if not ChatHistoryRepository(context).delete_session(session_id):
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"deleted": True}


@router.delete("/{session_id}/turns/{turn_id}")
def delete_turn(session_id: str, turn_id: str, context: TenantContext = Depends(require_tenant_context)) -> dict:
    if not ChatHistoryRepository(context).delete_turn(session_id, turn_id):
        raise HTTPException(status_code=404, detail="Chat turn not found")
    return {"deleted": True}

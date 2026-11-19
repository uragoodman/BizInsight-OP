"""P0 Memory 管理接口：只暴露当前可信租户上下文可见的记忆。"""

from typing import Any
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.api.auth import require_actor_match, require_analysis_write, require_tenant_context
from app.memory.consolidation import ConsolidationUnavailable
from app.memory.consolidation_jobs import enqueue_consolidation_job
from app.memory.contracts import MemoryConsolidationRequest, MemoryReviewRequest, MemoryStaleRequest
from app.memory.repository import MemoryRepository
from app.memory.policy import MemoryPolicy
from app.memory.preference_management import (
    PreferenceManagementError,
    get_personalization_enabled,
    list_preferences,
    preference_history,
    retry_failed_preference,
    set_personalization_enabled,
)
from app.memory.preference_writer import (
    PreferenceWriteConflict,
    PreferenceWriteError,
    PreferenceWriter,
)
from app.data.database import engine
from app.data.rls import set_local_context
from app.tenancy import TenantContext
from app.observability import record_event


router = APIRouter(tags=["memory"])


class PreferenceUpdateRequest(BaseModel):
    value: dict[str, Any]
    chat_session_id: str = Field(min_length=1, max_length=128)
    source_turn_id: str | None = Field(default=None, max_length=128)
    source_trace_id: str | None = Field(default=None, max_length=128)
    expected_version: int = Field(ge=1)
    idempotency_key: str = Field(min_length=1, max_length=256)
    confirmed: bool


class PreferenceCreateRequest(BaseModel):
    memory_key: str = Field(min_length=1, max_length=128)
    memory_type: str = Field(min_length=1, max_length=32)
    value: dict[str, Any]
    chat_session_id: str = Field(min_length=1, max_length=128)
    source_turn_id: str | None = Field(default=None, max_length=128)
    source_trace_id: str | None = Field(default=None, max_length=128)
    expected_version: int = Field(default=0, ge=0, le=0)
    idempotency_key: str = Field(min_length=1, max_length=256)
    confirmed: bool


class PreferenceDeleteRequest(BaseModel):
    expected_version: int = Field(ge=1)
    idempotency_key: str = Field(min_length=1, max_length=256)
    confirmed: bool


class PreferenceReconcileRequest(BaseModel):
    expected_version: int = Field(ge=1)
    idempotency_key: str = Field(min_length=1, max_length=256)


class DeleteAllPreferencesRequest(BaseModel):
    confirmation_phrase: str
    idempotency_key: str = Field(min_length=1, max_length=128)


class PersonalizationSettingRequest(BaseModel):
    enabled: bool


def _preference_error(exc: Exception) -> HTTPException:
    code = getattr(exc, "code", "preference_operation_failed")
    if isinstance(exc, PreferenceWriteConflict):
        status_code = 409
    elif isinstance(exc, PreferenceWriteError):
        status_code = 422
    elif isinstance(exc, PreferenceManagementError):
        status_code = 409
    else:
        status_code = 500
    return HTTPException(status_code=status_code, detail=code)


def _owned_preference(context: TenantContext, registry_id: str) -> dict[str, Any] | None:
    with engine.begin() as connection:
        set_local_context(connection, context, scope_type=context.scope_type)
        row = connection.execute(text("""
            SELECT registry_id, memory_key, memory_type, version, status, seller_id
              FROM governance.memory_registry
             WHERE registry_id=:registry_id AND tenant_id=:tenant_id
               AND user_id=:user_id
        """), {
            "registry_id": registry_id, "tenant_id": context.tenant_id,
            "user_id": context.user_id,
        }).mappings().first()
    return dict(row) if row else None


def _audit_policy_rejection(context: TenantContext, reason: str) -> None:
    """Persist only identity, status and bounded reason; never the submitted value."""
    try:
        with engine.begin() as connection:
            set_local_context(connection, context, scope_type=context.scope_type)
            connection.execute(text("""
                INSERT INTO governance.memory_preference_audit
                    (audit_id, tenant_id, user_id, seller_id, registry_id,
                     operation_id, event_type, actor_user_id, status, reason_code)
                VALUES (:audit_id, :tenant_id, :user_id, :seller_id, NULL,
                        :operation_id, 'policy_rejected', :actor, 'rejected', :reason)
            """), {
                "audit_id": f"maud_{uuid.uuid4().hex}",
                "tenant_id": context.tenant_id, "user_id": context.user_id,
                "seller_id": context.seller_id, "operation_id": f"op_{uuid.uuid4().hex}",
                "actor": context.user_id, "reason": reason[:128],
            })
    except Exception as exc:  # noqa: BLE001 - do not claim an audited rejection if persistence failed
        raise HTTPException(status_code=503, detail="preference_audit_unavailable") from exc
    record_event("memory.preference_policy_rejected", {"memory.reason": reason[:128]})


@router.get("/memory/preferences")
def get_preferences(
    chat_session_id: str | None = None,
    context: TenantContext = Depends(require_tenant_context),
) -> dict[str, Any]:
    try:
        return {
            "items": list_preferences(context, chat_session_id=chat_session_id),
            "personalization_enabled": get_personalization_enabled(context),
        }
    except Exception as exc:  # noqa: BLE001 - do not expose DB/provider exception text
        raise HTTPException(status_code=503, detail="preference_store_unavailable") from exc


@router.get("/memory/preferences/settings")
def get_preference_settings(
    context: TenantContext = Depends(require_tenant_context),
) -> dict[str, bool]:
    return {"personalization_enabled": get_personalization_enabled(context)}


@router.put("/memory/preferences/settings")
def put_preference_settings(
    payload: PersonalizationSettingRequest,
    context: TenantContext = Depends(require_tenant_context),
) -> dict[str, bool]:
    return {"personalization_enabled": set_personalization_enabled(context, payload.enabled)}


@router.post("/memory/preferences", status_code=202)
def create_preference(
    payload: PreferenceCreateRequest,
    context: TenantContext = Depends(require_tenant_context),
) -> dict[str, Any]:
    decision = MemoryPolicy.evaluate_upsert(
        context, memory_key=payload.memory_key, memory_type=payload.memory_type,
        value=payload.value, explicit_user_request=True,
        user_confirmed=payload.confirmed,
    )
    if not decision.accepted:
        _audit_policy_rejection(context, decision.reason)
        raise HTTPException(status_code=422, detail=decision.reason)
    try:
        queued = PreferenceWriter(context).queue_upsert(
            memory_key=payload.memory_key, memory_type=payload.memory_type,
            value=payload.value, chat_session_id=payload.chat_session_id,
            source_turn_id=payload.source_turn_id, source_trace_id=payload.source_trace_id,
            expected_version=0, idempotency_key=payload.idempotency_key,
            policy_decision=decision,
        )
    except PreferenceWriteError as exc:
        raise _preference_error(exc) from exc
    return {"registry_id": queued.registry_id, "operation_id": queued.operation_id,
            "status": queued.status, "changed": queued.changed,
            "message": (
                "preference_pending_persistence"
                if queued.status.startswith("pending_")
                else "preference_already_persisted"
                if queued.status == "active"
                else "preference_operation_not_completed"
            )}


@router.put("/memory/preferences/{registry_id}")
def update_preference(
    registry_id: str,
    payload: PreferenceUpdateRequest,
    context: TenantContext = Depends(require_tenant_context),
) -> dict[str, Any]:
    row = _owned_preference(context, registry_id)
    if row is None:
        raise HTTPException(status_code=404, detail="preference_not_found")
    decision = MemoryPolicy.evaluate_upsert(
        context, memory_key=str(row["memory_key"]), memory_type=str(row["memory_type"]),
        value=payload.value, explicit_user_request=True,
        user_confirmed=payload.confirmed,
    )
    if not decision.accepted:
        _audit_policy_rejection(context, decision.reason)
        raise HTTPException(status_code=422, detail=decision.reason)
    try:
        queued = PreferenceWriter(context).queue_upsert(
            memory_key=str(row["memory_key"]), memory_type=str(row["memory_type"]),
            value=payload.value, chat_session_id=payload.chat_session_id,
            source_turn_id=payload.source_turn_id, source_trace_id=payload.source_trace_id,
            expected_version=payload.expected_version,
            idempotency_key=payload.idempotency_key, policy_decision=decision,
            seller_id=row["seller_id"],
        )
    except PreferenceWriteError as exc:
        raise _preference_error(exc) from exc
    return {"registry_id": queued.registry_id, "operation_id": queued.operation_id,
            "status": queued.status, "changed": queued.changed}


@router.delete("/memory/preferences/{registry_id}")
def delete_preference(
    registry_id: str,
    payload: PreferenceDeleteRequest,
    context: TenantContext = Depends(require_tenant_context),
) -> dict[str, Any]:
    if not payload.confirmed:
        raise HTTPException(status_code=422, detail="explicit_user_confirmation_required")
    if _owned_preference(context, registry_id) is None:
        raise HTTPException(status_code=404, detail="preference_not_found")
    decision = MemoryPolicy.evaluate_delete(
        context, registry_id=registry_id, user_confirmed=True,
    )
    try:
        queued = PreferenceWriter(context).queue_delete(
            registry_id=registry_id, expected_version=payload.expected_version,
            idempotency_key=payload.idempotency_key, policy_decision=decision,
        )
    except PreferenceWriteError as exc:
        raise _preference_error(exc) from exc
    return {"registry_id": queued.registry_id, "operation_id": queued.operation_id,
            "status": queued.status, "changed": queued.changed}


@router.delete("/memory/preferences")
def delete_all_preferences(
    payload: DeleteAllPreferencesRequest,
    context: TenantContext = Depends(require_tenant_context),
) -> dict[str, Any]:
    if payload.confirmation_phrase != "DELETE_ALL_PREFERENCES":
        raise HTTPException(status_code=422, detail="second_confirmation_required")
    with engine.begin() as connection:
        set_local_context(connection, context, scope_type=context.scope_type)
        rows = connection.execute(text("""
            SELECT r.registry_id, r.version, r.status FROM governance.memory_registry r
             WHERE r.tenant_id=:tenant_id AND r.user_id=:user_id
               AND (
                   r.status IN ('active','pending_create','pending_update','pending_delete','failed')
                   OR (
                       r.memory_type='scope_default' AND r.status='superseded'
                       AND r.expires_at <= clock_timestamp()
                       AND NOT EXISTS (
                           SELECT 1 FROM governance.memory_registry newer
                            WHERE newer.tenant_id=r.tenant_id
                              AND newer.user_id=r.user_id
                              AND newer.seller_id IS NOT DISTINCT FROM r.seller_id
                              AND newer.memory_key=r.memory_key
                              AND newer.version > r.version
                       )
                   )
               )
             ORDER BY r.registry_id
        """), {
            "tenant_id": context.tenant_id, "user_id": context.user_id,
        }).mappings().all()
    queued: list[str] = []
    unfinished: list[dict[str, str]] = []
    for row in rows:
        registry_id = str(row["registry_id"])
        # A pending create can be cancelled before delivery; failed entries
        # can be revoked by reconciling their dead-letter operation ID.
        if row["status"] not in {"active", "pending_create", "superseded", "failed"}:
            unfinished.append({"registry_id": registry_id, "reason": f"already_{row['status']}"})
            continue
        try:
            result = PreferenceWriter(context).queue_delete(
                registry_id=registry_id, expected_version=int(row["version"]),
                idempotency_key=f"{payload.idempotency_key}:{registry_id}",
                policy_decision=MemoryPolicy.evaluate_delete(
                    context, registry_id=registry_id, user_confirmed=True,
                ),
            )
            queued.append(result.registry_id)
        except PreferenceWriteError as exc:
            unfinished.append({"registry_id": registry_id, "reason": exc.code})
    return {"queued_count": len(queued), "deleted_count": 0,
            "pending_registry_ids": queued,
            "unfinished": unfinished, "complete": not unfinished and not queued}


@router.post("/memory/preferences/{registry_id}/retry")
def retry_preference(
    registry_id: str,
    context: TenantContext = Depends(require_tenant_context),
) -> dict[str, Any]:
    try:
        status = retry_failed_preference(context, registry_id)
    except PreferenceManagementError as exc:
        raise _preference_error(exc) from exc
    return {"registry_id": registry_id, "status": status}


@router.post("/memory/preferences/{registry_id}/reconcile", status_code=202)
def reconcile_preference(
    registry_id: str,
    payload: PreferenceReconcileRequest,
    context: TenantContext = Depends(require_tenant_context),
) -> dict[str, Any]:
    row = _owned_preference(context, registry_id)
    if row is None:
        raise HTTPException(status_code=404, detail="preference_not_found")
    if row["status"] != "active":
        raise HTTPException(status_code=409, detail="preference_not_reconcilable")
    try:
        queued = PreferenceWriter(context).queue_reconciliation(
            registry_id=registry_id,
            expected_version=payload.expected_version,
            idempotency_key=payload.idempotency_key,
        )
    except PreferenceWriteError as exc:
        raise _preference_error(exc) from exc
    return {
        "registry_id": queued.registry_id,
        "operation_id": queued.operation_id,
        "status": queued.status,
        "changed": queued.changed,
    }


@router.get("/memory/preferences/{registry_id}/history")
def get_preference_history(
    registry_id: str,
    context: TenantContext = Depends(require_tenant_context),
) -> dict[str, Any]:
    if _owned_preference(context, registry_id) is None:
        raise HTTPException(status_code=404, detail="preference_not_found")
    return {"items": preference_history(context, registry_id)}




@router.get("/memory/sessions/{memory_session_id}")
def get_session_memory(memory_session_id: str, context: TenantContext = Depends(require_tenant_context)) -> dict:
    row = MemoryRepository(context).get_session(memory_session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Memory session not found")
    return row


@router.delete("/memory/sessions/{memory_session_id}")
def clear_session_memory(memory_session_id: str, context: TenantContext = Depends(require_tenant_context)) -> dict:
    return {"cleared": MemoryRepository(context).clear_session(memory_session_id)}


@router.get("/memories")
def list_memories(limit: int = 20, status: str | None = None, context: TenantContext = Depends(require_tenant_context)) -> dict:
    # 列表接口也受 RLS 约束；不接收前端传入的 tenant_id 或 user_id。
    if status and status not in {"active", "pending_review", "superseded", "stale", "revoked", "rejected"}:
        raise HTTPException(status_code=422, detail="Unsupported memory status")
    repository = MemoryRepository(context)
    if status and status != "active":
        return {"items": repository.list_memories_by_status(status, limit=limit)}
    return {"items": repository.list_analysis_memories(limit=limit)}


@router.get("/memories/{memory_id}/supersede-candidates")
def supersede_candidates(memory_id: str, context: TenantContext = Depends(require_tenant_context)) -> dict:
    repository = MemoryRepository(context)
    pending = repository.get_analysis_memory(memory_id)
    if not pending:
        raise HTTPException(status_code=404, detail="Memory not found")
    if pending.get("status") != "pending_review":
        raise HTTPException(status_code=409, detail="Memory is not pending review")
    return {"items": repository.list_supersede_candidates(
        str(pending["scope_fingerprint"]), metric_tags=list(pending.get("metric_tags") or []),
        task_type=str(pending["task_type"]), limit=20,
    )}


@router.get("/memories/{memory_id}/decisions")
def list_memory_decisions(memory_id: str, context: TenantContext = Depends(require_tenant_context)) -> dict:
    repository = MemoryRepository(context)
    if not repository.get_analysis_memory(memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"items": repository.list_memory_decisions(memory_id)}


@router.post("/memories/consolidation-jobs")
@router.post("/memories/consolidation-proposals", deprecated=True)
def create_consolidation_job(
    payload: MemoryConsolidationRequest,
    actor: str = Depends(require_analysis_write),
    context: TenantContext = Depends(require_tenant_context),
) -> dict:
    try:
        require_actor_match(actor, payload.actor)
        item = enqueue_consolidation_job(MemoryRepository(context), payload.scope_fingerprint, payload.actor)
        record_event("memory.consolidation_queued", {"memory.scope_fingerprint": payload.scope_fingerprint})
        return item
    except ConsolidationUnavailable as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/memories/consolidation-jobs")
def list_consolidation_jobs(limit: int = 20, context: TenantContext = Depends(require_tenant_context)) -> dict:
    return {"items": MemoryRepository(context).list_consolidation_jobs(limit)}


@router.get("/memories/consolidation-jobs/{job_id}")
def get_consolidation_job(job_id: str, context: TenantContext = Depends(require_tenant_context)) -> dict:
    item = MemoryRepository(context).get_consolidation_job(job_id)
    if not item:
        raise HTTPException(status_code=404, detail="Consolidation job not found")
    return item


@router.get("/memory/source-snapshots/{source_data_version}")
def get_memory_source_snapshot(source_data_version: str, context: TenantContext = Depends(require_tenant_context)) -> dict:
    item = MemoryRepository(context).get_source_snapshot(source_data_version)
    if not item:
        raise HTTPException(status_code=404, detail="Source snapshot not found")
    return item


@router.post("/memories/stale")
def stale_memories(
    payload: MemoryStaleRequest,
    actor: str = Depends(require_analysis_write),
    context: TenantContext = Depends(require_tenant_context),
) -> dict:
    require_actor_match(actor, payload.actor)
    ids = MemoryRepository(context).stale_memories_for_data_version(
        payload.scope_fingerprint, payload.task_type, payload.source_data_version, payload.actor, payload.reason,
    )
    record_event("memory.staled", {"memory.count": len(ids), "memory.source_data_version": payload.source_data_version})
    return {"staled_memory_ids": ids}


@router.post("/memories/{memory_id}/review")
def review_memory(memory_id: str, payload: MemoryReviewRequest, actor: str = Depends(require_analysis_write), context: TenantContext = Depends(require_tenant_context)) -> dict:
    try:
        require_actor_match(actor, payload.actor)
        item = MemoryRepository(context).review_pending_memory(
            memory_id, payload.action, payload.actor, payload.reason, payload.old_memory_id,
        )
        record_event("memory.review_approved" if payload.action != "reject" else "memory.review_rejected", {"memory.action": payload.action})
        return item
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/memories/{memory_id}/revoke")
def revoke_memory(memory_id: str, context: TenantContext = Depends(require_tenant_context)) -> dict:
    if not MemoryRepository(context).revoke_analysis_memory(memory_id):
        raise HTTPException(status_code=404, detail="Active memory not found")
    return {"revoked": True}

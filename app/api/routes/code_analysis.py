"""HTTP boundary for governed uploaded-file Code Agent analysis."""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from app.api.auth import require_tenant_context
from app.code_analysis.database import engine
from app.code_analysis.service import (
    cancel_job,
    confirm_job,
    create_job,
    delete_job,
    grant_access,
    get_file,
    get_job,
    get_audit,
    list_access,
    list_jobs,
    record_access_event,
    revoke_access,
    revise_job,
    upload_file,
)
from app.code_analysis.storage import get_artifact_store
from app.core.config import get_settings
from app.tenancy import TenantContext
from app.iam.service import require_permission

router = APIRouter(prefix="/code-analysis", tags=["code-analysis"])


class JobCreateRequest(BaseModel):
    file_id: str = Field(min_length=8, max_length=64)
    question: str = Field(min_length=1, max_length=4000)
    chat_session_id: str | None = Field(default=None, max_length=128)
    source_turn_id: str | None = Field(default=None, max_length=128)
    parent_job_id: str | None = Field(default=None, max_length=64)


class JobConfirmRequest(BaseModel):
    plan_hash: str = Field(min_length=64, max_length=64)


class JobReviseRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class ShareRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    permission: str = Field(pattern="^(read|execute|audit)$")
    expires_at: str | None = None


class DatasetPrefetchRequest(BaseModel):
    """受控业务数据集预取请求：只接受白名单指标 + seller/日期范围（P2）。"""
    seller_id: str = Field(min_length=1, max_length=64)
    start_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    metric_names: list[str] = Field(min_length=1, max_length=10)


@router.post("/files")
async def upload(
    file: UploadFile = File(...),
    context: TenantContext = Depends(require_tenant_context),
) -> dict:
    try:
        require_permission(context, "code_analysis")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    raw = await file.read(get_settings().code_analysis_max_upload_bytes + 1)
    try:
        return upload_file(context, file.filename or "upload", file.content_type, raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/files/{file_id}")
def file_contract(
    file_id: str, context: TenantContext = Depends(require_tenant_context)
) -> dict:
    try:
        require_permission(context, "code_analysis_read")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    record = get_file(context, file_id)
    if not record:
        raise HTTPException(status_code=404, detail="Uploaded file not found")
    return {
        "file_id": record["file_id"],
        "filename": record["original_name"],
        "file_type": record["file_type"],
        "size_bytes": record["size_bytes"],
        "sha256": record["sha256"],
        "expires_at": record["expires_at"],
        "data_contract": record["contract_data"],
    }


@router.post("/datasets")
def prefetch_dataset(
    payload: DatasetPrefetchRequest, context: TenantContext = Depends(require_tenant_context)
) -> dict:
    """把受控范围内的业务事实预取为 Parquet 工件，返回可直接 create_job 的 file_id。

    治理边界：指标白名单来自 metric registry；seller 必须在授权范围内；
    runner 只能读取这份工件，LLM 永不获得数据库凭据或网络数据库权限。
    """
    from datetime import date

    from app.code_analysis.dataset import prefetch_business_dataset

    try:
        require_permission(context, "code_analysis")
        return prefetch_business_dataset(
            context,
            seller_id=payload.seller_id,
            start_date=date.fromisoformat(payload.start_date),
            end_date=date.fromisoformat(payload.end_date),
            metric_names=payload.metric_names,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/jobs")
def create(
    payload: JobCreateRequest, context: TenantContext = Depends(require_tenant_context)
) -> dict:
    try:
        require_permission(context, "code_analysis")
        return create_job(
            context,
            payload.file_id,
            payload.question,
            payload.chat_session_id,
            payload.source_turn_id,
            payload.parent_job_id,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail="Uploaded file not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/jobs")
def jobs(
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=50),
    chat_session_id: str | None = Query(default=None, max_length=128),
    created_after: datetime | None = Query(default=None),
    file_id: str | None = Query(default=None, max_length=64),
    context: TenantContext = Depends(require_tenant_context),
) -> dict:
    try:
        require_permission(context, "code_analysis_read")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return list_jobs(context, offset, limit, chat_session_id, created_after, file_id)


@router.get("/jobs/{job_id}")
def job(job_id: str, context: TenantContext = Depends(require_tenant_context)) -> dict:
    try:
        require_permission(context, "code_analysis_read")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    record = get_job(context, job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    return record


@router.get("/jobs/{job_id}/audit")
def audit(
    job_id: str, context: TenantContext = Depends(require_tenant_context)
) -> dict:
    try:
        require_permission(context, "code_analysis_audit")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    record = get_audit(context, job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis audit not found")
    return record


@router.post("/jobs/{job_id}/confirm")
def confirm(
    job_id: str,
    payload: JobConfirmRequest,
    context: TenantContext = Depends(require_tenant_context),
) -> dict:
    try:
        require_permission(context, "code_analysis")
        return confirm_job(context, job_id, payload.plan_hash)
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail="Analysis job not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/revise")
def revise(
    job_id: str,
    payload: JobReviseRequest,
    context: TenantContext = Depends(require_tenant_context),
) -> dict:
    try:
        require_permission(context, "code_analysis")
        return revise_job(context, job_id, payload.question)
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail="Analysis job not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/cancel")
def cancel(
    job_id: str, context: TenantContext = Depends(require_tenant_context)
) -> dict:
    try:
        require_permission(context, "code_analysis")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if not cancel_job(context, job_id):
        raise HTTPException(status_code=409, detail="Analysis job cannot be cancelled")
    return {"cancelled": True}


@router.delete("/jobs/{job_id}")
def delete(
    job_id: str, context: TenantContext = Depends(require_tenant_context)
) -> dict:
    try:
        require_permission(context, "code_analysis")
        return {"deleted": delete_job(context, job_id)}
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail="Analysis job not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/jobs/{job_id}/events")
def events(
    job_id: str, context: TenantContext = Depends(require_tenant_context)
) -> StreamingResponse:
    try:
        require_permission(context, "code_analysis_read")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if not get_job(context, job_id):
        raise HTTPException(status_code=404, detail="Analysis job not found")

    def stream():
        previous = None
        sequence = 0
        for _ in range(120):  # 10 minutes at a five-second polling cadence
            record = get_job(context, job_id)
            if not record:
                sequence += 1
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "event": "failed",
                            "detail": "Analysis job not found",
                            "sequence_number": sequence,
                        }
                    )
                    + "\n\n"
                )
                return
            status = record["status"]
            if status != previous:
                sequence += 1
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "event": "status",
                            "job_id": job_id,
                            "status": status,
                            "error_code": record.get("error_code"),
                            "sequence_number": sequence,
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n\n"
                )
                previous = status
            if status in {"completed", "failed", "cancelled", "expired", "superseded"}:
                sequence += 1
                yield (
                    "data: "
                    + json.dumps(
                        {"event": status, "job": record, "sequence_number": sequence},
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n\n"
                )
                return
            sequence += 1
            yield (
                "data: "
                + json.dumps(
                    {
                        "event": "heartbeat",
                        "job_id": job_id,
                        "status": status,
                        "sequence_number": sequence,
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )
            time.sleep(5)
        sequence += 1
        yield (
            "data: "
            + json.dumps(
                {
                    "event": "timeout",
                    "job_id": job_id,
                    "detail": "Status stream reached its time limit",
                    "sequence_number": sequence,
                }
            )
            + "\n\n"
        )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/jobs/{job_id}/artifacts/{artifact_id}")
def artifact(
    job_id: str,
    artifact_id: str,
    context: TenantContext = Depends(require_tenant_context),
) -> Response:
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(
            text("SELECT set_config('app.tenant_id', :v, true)"),
            {"v": context.tenant_id},
        )
        conn.execute(
            text("SELECT set_config('app.user_id', :v, true)"), {"v": context.user_id}
        )
        conn.execute(
            text("SELECT set_config('app.user_role', :v, true)"), {"v": context.role}
        )
        row = (
            conn.execute(
                text(
                    "SELECT kind,object_key,sha256,size_bytes,metadata "
                    "FROM governance.code_analysis_artifacts "
                    "WHERE job_id=:job AND artifact_id=:artifact "
                    "AND tenant_id=:tenant"
                ),
                {
                    "job": job_id,
                    "artifact": artifact_id,
                    "tenant": context.tenant_id,
                    "user": context.user_id,
                },
            )
            .mappings()
            .one_or_none()
        )
    if not row:
        raise HTTPException(status_code=404, detail="Artifact not found")
    try:
        require_permission(context, "code_analysis_read")
        if str(row["kind"]) in {"candidate_code", "runner_log", "validation"}:
            require_permission(context, "code_analysis_audit")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    try:
        data = get_artifact_store().get(
            row["object_key"], max_bytes=int(row["size_bytes"])
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=410, detail="Artifact has expired") from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=409, detail="Artifact failed its size integrity check"
        ) from exc
    if (
        len(data) != int(row["size_bytes"])
        or hashlib.sha256(data).hexdigest() != row["sha256"]
    ):
        raise HTTPException(
            status_code=409, detail="Artifact failed its integrity check"
        )
    try:
        record_access_event(
            context,
            "artifact",
            artifact_id,
            "download",
            metadata={"job_id": job_id, "kind": str(row["kind"])},
        )
    except Exception:
        # Access telemetry must not make a valid artifact unavailable.
        pass
    kind = str(row["kind"])
    metadata = dict(row["metadata"] or {})
    fallback_names = {
        "candidate_code": "analysis.py",
        "validation": "validation.json",
        "result": "result.json",
        "runner_log": "runner.log",
        "chart": "chart",
    }
    media = {
        "candidate_code": "text/x-python; charset=utf-8",
        "validation": "application/json",
        "result": "application/json",
        "runner_log": "text/plain; charset=utf-8",
    }.get(kind, "application/octet-stream")
    display_name = str(
        metadata.get("name") or fallback_names.get(kind, kind or "artifact")
    )
    if kind == "chart":
        candidate_media = str(metadata.get("content_type") or "")
        if candidate_media == "image/png":
            media = candidate_media
        elif display_name.lower().endswith(".png"):
            media = "image/png"
        else:
            raise HTTPException(
                status_code=409, detail="Unsupported chart artifact type"
            )
    display_name = re.split(r"[\\/]", display_name.replace("\r", "").replace("\n", ""))[
        -1
    ]
    if not display_name:
        display_name = fallback_names.get(kind, "artifact")
    ascii_name = re.sub(r"[^A-Za-z0-9_.-]", "_", display_name).strip(
        "._"
    ) or fallback_names.get(kind, "artifact")
    disposition = f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(display_name, safe='')}"
    return Response(
        data,
        media_type=media,
        headers={
            "Cache-Control": "private, no-store",
            "Content-Disposition": disposition,
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post("/files/{file_id}/shares")
def share_file(
    file_id: str,
    payload: ShareRequest,
    context: TenantContext = Depends(require_tenant_context),
) -> dict:
    try:
        require_permission(context, "code_analysis_share")
        from datetime import datetime
        expires = datetime.fromisoformat(payload.expires_at) if payload.expires_at else None
        return grant_access(context, "file", file_id, payload.user_id, payload.permission, expires)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/shares")
def share_job(
    job_id: str,
    payload: ShareRequest,
    context: TenantContext = Depends(require_tenant_context),
) -> dict:
    try:
        require_permission(context, "code_analysis_share")
        from datetime import datetime
        expires = datetime.fromisoformat(payload.expires_at) if payload.expires_at else None
        return grant_access(context, "job", job_id, payload.user_id, payload.permission, expires)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/jobs/{job_id}/shares")
def job_shares(
    job_id: str, context: TenantContext = Depends(require_tenant_context)
) -> dict:
    try:
        require_permission(context, "code_analysis_share")
        return {"items": list_access(context, "job", job_id)}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/shares/{grant_id}")
def unshare(
    grant_id: str, context: TenantContext = Depends(require_tenant_context)
) -> dict:
    try:
        require_permission(context, "code_analysis_share")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if not revoke_access(context, grant_id):
        raise HTTPException(status_code=404, detail="Access grant not found")
    return {"revoked": True}

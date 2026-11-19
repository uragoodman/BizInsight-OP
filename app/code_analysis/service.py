"""Persistence, upload validation and lease-queue operations for Code Agent jobs."""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import text

from app.code_analysis.storage import get_artifact_store, owner_artifact_prefix
from app.code_analysis.database import engine
from app.code_analysis.integrity import plan_fingerprint
from app.code_analysis.profiling import binary_contract, is_sensitive_field
from app.core.config import get_settings
from app.tenancy import TenantContext

logger = logging.getLogger(__name__)

ALLOWED_TYPES = {
    "csv": {"text/csv", "application/csv", "text/plain"},
    "xlsx": {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
    "parquet": {"application/vnd.apache.parquet", "application/octet-stream"},
}
PUBLIC_JOB_FIELDS = {
    "job_id",
    "chat_session_id",
    "source_turn_id",
    "file_id",
    "parent_job_id",
    "revision",
    "question",
    "plan_data",
    "plan_hash",
    "resource_limits",
    "status",
    "expires_at",
    "confirmed_at",
    "started_at",
    "completed_at",
    "result_data",
    "error_code",
    "created_at",
    "updated_at",
}


def _id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(12)}"


def _scope(conn, context: TenantContext) -> None:
    conn.execute(
        text("SELECT set_config('app.tenant_id', :v, true)"), {"v": context.tenant_id}
    )
    conn.execute(
        text("SELECT set_config('app.user_id', :v, true)"), {"v": context.user_id}
    )
    conn.execute(
        text("SELECT set_config('app.user_role', :v, true)"), {"v": context.role}
    )


def _lock_owner(conn, context: TenantContext) -> None:
    conn.execute(
        text("SELECT pg_advisory_xact_lock(hashtextextended(:owner,0))"),
        {"owner": f"code-analysis:{context.tenant_id}:{context.user_id}"},
    )


def _resource_limits(settings) -> dict[str, Any]:
    """Freeze the execution envelope into the plan that the user confirms."""
    return {
        "timeout_seconds": max(
            1, int(settings.code_analysis_execution_timeout_seconds)
        ),
        "memory_mb": max(128, int(settings.code_analysis_runner_memory_mb)),
        "cpu_cores": max(0.1, float(settings.code_analysis_runner_cpu_cores)),
        "pids": max(16, int(settings.code_analysis_runner_pids)),
        "max_output_bytes": max(1, int(settings.code_analysis_max_output_bytes)),
        "max_result_bytes": max(1, int(settings.code_analysis_max_result_bytes)),
        "runner_image": str(settings.code_analysis_runner_image),
    }


def _follow_up_context(parent: dict[str, Any]) -> dict[str, Any]:
    """Keep a bounded, result-only context for a same-file follow-up."""
    result = parent.get("result_data")
    result = result if isinstance(result, dict) else {}

    def bounded_text(value: Any, limit: int = 4000) -> str:
        return str(value or "").strip()[:limit]

    def bounded_items(value: Any, limit: int = 10) -> list[str]:
        if not isinstance(value, list):
            return []
        return [bounded_text(item, 1200) for item in value[:limit]]

    def bounded_json(value: Any, limit: int = 8000) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))[:limit]
        except (TypeError, ValueError):
            return ""

    return {
        "previous_question": bounded_text(parent.get("question")),
        "previous_status": bounded_text(parent.get("status"), 80),
        "previous_conclusion": bounded_text(result.get("conclusion")),
        "previous_method": bounded_text(result.get("method")),
        "previous_summary": bounded_json(result.get("summary")),
        "previous_evidence": bounded_items(result.get("evidence")),
        "previous_limitations": bounded_items(result.get("limitations")),
        "previous_recommendations": bounded_items(result.get("recommendations")),
    }


def _enforce_job_quotas(
    conn, context: TenantContext, settings, *, check_active: bool = True
) -> None:
    if check_active:
        active = int(
            conn.execute(
                text("""SELECT count(*) FROM governance.code_analysis_jobs
                  WHERE tenant_id=:tenant AND user_id=:user
                    AND status IN ('awaiting_confirmation','queued','generating','validating','running')"""),
                {"tenant": context.tenant_id, "user": context.user_id},
            ).scalar_one()
        )
        if active >= settings.code_analysis_max_active_jobs_per_user:
            raise ValueError(
                f"Active analysis job limit reached ({settings.code_analysis_max_active_jobs_per_user})"
            )
    daily = int(
        conn.execute(
            text("""SELECT count(*) FROM governance.code_analysis_jobs
              WHERE tenant_id=:tenant AND user_id=:user
                AND created_at >= (
                date_trunc('day',now() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'
              )"""),
            {"tenant": context.tenant_id, "user": context.user_id},
        ).scalar_one()
    )
    if daily >= settings.code_analysis_max_jobs_per_user_per_day:
        raise ValueError(
            f"Daily analysis job limit reached ({settings.code_analysis_max_jobs_per_user_per_day})"
        )


def _hash(value: Any) -> str:
    return plan_fingerprint(value)


def _file_type(filename: str, content_type: str | None, raw: bytes) -> str:
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in ALLOWED_TYPES:
        raise ValueError("Only CSV, XLSX and Parquet files are supported")
    # Magic bytes stop simple extension spoofing. CSV has no stable magic, so
    # reject binary control bytes instead.
    if suffix == "xlsx" and not raw.startswith(b"PK"):
        raise ValueError("The uploaded XLSX content is invalid")
    if suffix == "parquet" and not (raw.startswith(b"PAR1") and raw[-4:] == b"PAR1"):
        raise ValueError("The uploaded Parquet content is invalid")
    if suffix == "csv" and b"\x00" in raw[:8192]:
        raise ValueError("The uploaded CSV content is invalid")
    return suffix


def _csv_contract(raw: bytes, settings) -> dict:
    decoded = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            decoded = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        raise ValueError("CSV encoding is not supported")
    reader = csv.DictReader(io.StringIO(decoded, newline=""))
    fields = reader.fieldnames or []
    if not fields or len(fields) > settings.code_analysis_max_columns:
        raise ValueError("CSV has no columns or exceeds the column limit")
    samples: list[dict] = []
    nulls = {field: 0 for field in fields}
    kinds = {
        field: {"number": 0, "boolean": 0, "datetime": 0, "string": 0}
        for field in fields
    }
    numeric = {
        field: {"count": 0, "sum": 0.0, "min": None, "max": None} for field in fields
    }
    rows = 0
    for row in reader:
        rows += 1
        if rows > settings.code_analysis_max_rows:
            raise ValueError("CSV exceeds the row limit")
        for field in fields:
            value = (row.get(field) or "").strip()
            if not value:
                nulls[field] += 1
                continue
            lowered = value.lower()
            if lowered in {"true", "false"}:
                kinds[field]["boolean"] += 1
                continue
            try:
                number = float(value)
            except ValueError:
                number = None
            if number is not None and math.isfinite(number):
                kinds[field]["number"] += 1
                stats = numeric[field]
                stats["count"] += 1
                stats["sum"] += number
                stats["min"] = (
                    number if stats["min"] is None else min(stats["min"], number)
                )
                stats["max"] = (
                    number if stats["max"] is None else max(stats["max"], number)
                )
                continue
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                kinds[field]["string"] += 1
            else:
                kinds[field]["datetime"] += 1
        if len(samples) < 5:
            samples.append(
                {
                    field: "[masked]"
                    if is_sensitive_field(field)
                    else str(row.get(field, ""))[:120]
                    for field in fields
                }
            )
    columns: list[dict[str, Any]] = []
    for field in fields:
        observed = [kind for kind, count in kinds[field].items() if count]
        inferred = (
            observed[0] if len(observed) == 1 else "string" if observed else "unknown"
        )
        column: dict[str, Any] = {
            "name": field,
            "type": inferred,
            "null_rate": round(nulls[field] / rows, 4) if rows else 0,
            "sensitive": is_sensitive_field(field),
        }
        stats = numeric[field]
        if inferred == "number" and stats["count"]:
            column.update(
                {
                    "min": stats["min"],
                    "max": stats["max"],
                    "mean": round(stats["sum"] / stats["count"], 6),
                }
            )
        columns.append(column)
    return {
        "file_type": "csv",
        "rows": rows,
        "columns": columns,
        "samples": samples,
        "warnings": ["Formula cells are never executed; files are read as data only."],
    }


def upload_file(
    context: TenantContext, filename: str, content_type: str | None, raw: bytes
) -> dict:
    settings = get_settings()
    if not filename or len(raw) == 0:
        raise ValueError("An empty file cannot be analyzed")
    if len(raw) > settings.code_analysis_max_upload_bytes:
        raise ValueError("File exceeds the 50 MB upload limit")
    file_type = _file_type(filename, content_type, raw)
    contract = (
        _csv_contract(raw, settings)
        if file_type == "csv"
        else binary_contract(file_type, raw, settings)
    )
    file_id = _id("file")
    owner_prefix = owner_artifact_prefix(context.tenant_id, context.user_id)
    key = f"{owner_prefix}/uploads/{file_id}/source.{file_type}"
    store = get_artifact_store()
    object_key: str | None = None
    digest = ""
    size = 0
    expires_at = datetime.now(timezone.utc) + timedelta(
        hours=settings.code_analysis_retention_hours
    )
    try:
        with engine.begin() as conn:
            _scope(conn, context)
            _lock_owner(conn, context)
            stored_bytes = int(
                conn.execute(
                    text(
                        "SELECT COALESCE(sum(size_bytes),0) FROM governance.code_analysis_files"
                        " WHERE tenant_id=:tenant AND user_id=:user AND purged_at IS NULL"
                    ),
                    {"tenant": context.tenant_id, "user": context.user_id},
                ).scalar_one()
            )
            if (
                stored_bytes + len(raw)
                > settings.code_analysis_max_stored_upload_bytes_per_user
            ):
                raise ValueError(
                    "Stored upload quota exceeded "
                    f"({settings.code_analysis_max_stored_upload_bytes_per_user} bytes per user)"
                )
            object_key, digest, size = store.put(key, raw)
            if (
                stored_bytes + size
                > settings.code_analysis_max_stored_upload_bytes_per_user
            ):
                raise ValueError(
                    "Stored upload quota exceeded "
                    f"({settings.code_analysis_max_stored_upload_bytes_per_user} bytes per user)"
                )
            conn.execute(
                text("""INSERT INTO governance.code_analysis_files
              (file_id,tenant_id,user_id,original_name,file_type,content_type,object_key,sha256,size_bytes,contract_data,expires_at)
              VALUES(:id,:tenant,:user,:name,:type,:content_type,:key,:sha,:size,CAST(:contract AS jsonb),:expires)"""),
                {
                    "id": file_id,
                    "tenant": context.tenant_id,
                    "user": context.user_id,
                    "name": filename[:255],
                    "type": file_type,
                    "content_type": content_type or "",
                    "key": object_key,
                    "sha": digest,
                    "size": size,
                    "contract": json.dumps(contract, ensure_ascii=False),
                    "expires": expires_at,
                },
            )
    except Exception:
        if object_key is not None:
            try:
                store.delete(object_key)
            except Exception as cleanup_exc:
                logger.warning(
                    "Failed to compensate an uploaded object after database insertion failed",
                    extra={
                        "file_id": file_id,
                        "error_type": type(cleanup_exc).__name__,
                    },
                )
        raise
    return {
        "file_id": file_id,
        "filename": filename,
        "file_type": file_type,
        "sha256": digest,
        "size_bytes": size,
        "expires_at": expires_at,
        "data_contract": contract,
    }


def get_file(context: TenantContext, file_id: str) -> dict | None:
    with engine.begin() as conn:
        _scope(conn, context)
        row = (
            conn.execute(
                text("""SELECT * FROM governance.code_analysis_files
                  WHERE file_id=:id AND tenant_id=:tenant
                    AND purged_at IS NULL AND expires_at > now()"""),
                {
                    "id": file_id,
                    "tenant": context.tenant_id,
                    "user": context.user_id,
                },
            )
            .mappings()
            .one_or_none()
        )
    return dict(row) if row else None


def create_job(
    context: TenantContext,
    file_id: str,
    question: str,
    chat_session_id: str | None = None,
    source_turn_id: str | None = None,
    parent_job_id: str | None = None,
) -> dict:
    settings = get_settings()
    if source_turn_id and not chat_session_id:
        raise ValueError("A source chat turn requires a chat session")
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("Analysis question is required")
    if len(normalized_question) > 4000:
        raise ValueError("Analysis question exceeds the 4000 character limit")
    job_id = _id("caj")
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        _scope(conn, context)
        _lock_owner(conn, context)
        source = (
            conn.execute(
                text(
                    """SELECT file_id,sha256,contract_data,expires_at,purged_at
                       FROM governance.code_analysis_files
                       WHERE file_id=:id
                         AND tenant_id=:tenant
                         AND (user_id=:user OR EXISTS (
                           SELECT 1 FROM governance.code_analysis_access_grants g
                            WHERE g.tenant_id=code_analysis_files.tenant_id
                              AND g.resource_type='file'
                              AND g.resource_id=code_analysis_files.file_id
                              AND g.grantee_user_id=:user
                              AND g.permission IN ('execute','audit')
                              AND g.revoked_at IS NULL
                              AND (g.expires_at IS NULL OR g.expires_at > now())
                         ))
                         FOR UPDATE"""
                ),
                {
                    "id": file_id,
                    "tenant": context.tenant_id,
                    "user": context.user_id,
                },
            )
            .mappings()
            .one_or_none()
        )
        if not source:
            raise PermissionError("Uploaded file not found")
        if source["purged_at"] is not None or source["expires_at"] <= now:
            raise ValueError("Uploaded file has expired")
        follow_up_context: dict[str, Any] | None = None
        if parent_job_id:
            parent = (
                conn.execute(
                    text("""SELECT job_id,file_id,question,status,result_data
                           FROM governance.code_analysis_jobs
                           WHERE job_id=:parent
                             AND tenant_id=:tenant AND user_id=:user"""),
                    {
                        "parent": parent_job_id,
                        "tenant": context.tenant_id,
                        "user": context.user_id,
                    },
                )
                .mappings()
                .one_or_none()
            )
            if not parent:
                raise ValueError("Parent analysis job not found")
            if parent["file_id"] != file_id:
                raise ValueError("A continuation must use the same uploaded file")
            if parent["status"] != "completed" or not isinstance(parent["result_data"], dict):
                raise ValueError("A continuation requires a completed parent analysis")
            follow_up_context = _follow_up_context(dict(parent))
        _enforce_job_quotas(conn, context, settings)
        if source_turn_id:
            turn = conn.execute(
                text("""SELECT turn_id FROM governance.chat_turns
                  WHERE turn_id=:turn AND chat_session_id=:session
                    AND tenant_id=:tenant AND user_id=:user"""),
                {
                    "turn": source_turn_id,
                    "session": chat_session_id,
                    "tenant": context.tenant_id,
                    "user": context.user_id,
                },
            ).scalar_one_or_none()
            if not turn:
                raise ValueError("Source chat turn not found in this session")
        resource_limits = _resource_limits(settings)
        plan = {
            "question": normalized_question,
            "file_id": file_id,
            "file_sha256": source["sha256"],
            "data_contract": source["contract_data"],
            "retention_hours": settings.code_analysis_retention_hours,
            "resource_limits": resource_limits,
            "parent_job_id": parent_job_id,
            "expected_output": [
                "conclusion",
                "method",
                "evidence",
                "charts",
                "limitations",
                "recommendations",
                "reproducibility",
            ],
        }
        if follow_up_context:
            plan["conversation_context"] = follow_up_context
        plan_hash = _hash(plan)
        plan_expires_at = min(
            now + timedelta(minutes=30),
            source["expires_at"],
        )
        conn.execute(
            text("""INSERT INTO governance.code_analysis_jobs
          (job_id,tenant_id,user_id,chat_session_id,source_turn_id,file_id,parent_job_id,question,
           plan_data,plan_hash,resource_limits,status,expires_at)
          VALUES(:id,:tenant,:user,:session,:turn,:file,:parent,:question,CAST(:plan AS jsonb),
                 :hash,CAST(:limits AS jsonb),'awaiting_confirmation',:expires)"""),
            {
                "id": job_id,
                "tenant": context.tenant_id,
                "user": context.user_id,
                "session": chat_session_id,
                "turn": source_turn_id,
                "file": file_id,
                "parent": parent_job_id,
                "question": normalized_question,
                "plan": json.dumps(plan, ensure_ascii=False),
                "hash": plan_hash,
                "limits": json.dumps(resource_limits, ensure_ascii=False),
                "expires": plan_expires_at,
            },
        )
    return get_job(context, job_id) or {}


def get_job(context: TenantContext, job_id: str) -> dict | None:
    with engine.begin() as conn:
        _scope(conn, context)
        row = (
            conn.execute(
                text("""SELECT * FROM governance.code_analysis_jobs
                  WHERE job_id=:id AND tenant_id=:tenant"""),
                {
                    "id": job_id,
                    "tenant": context.tenant_id,
                    "user": context.user_id,
                },
            )
            .mappings()
            .one_or_none()
        )
        if not row:
            return None
        artifacts = (
            conn.execute(
                text(
                    "SELECT artifact_id,kind,sha256,size_bytes,metadata,created_at "
                    "FROM governance.code_analysis_artifacts "
                    "WHERE job_id=:id AND tenant_id=:tenant "
                    "ORDER BY created_at"
                ),
                {
                    "id": job_id,
                    "tenant": context.tenant_id,
                    "user": context.user_id,
                },
            )
            .mappings()
            .all()
        )
        audit = (
            conn.execute(
                text("""SELECT status,provider,generation_model,provider_version,
          candidate_attempts,repair_attempts,plan_hash,data_contract_hash,candidate_code_hash,
          runner_image_digest,result_hash,trace_id,duration_ms,error_code,created_at,
          actor_id,initiator_type,permission_decision,provider_request_id,input_sha256,
          input_tokens,output_tokens,estimated_cost,error_category,
           error_phase,error_http_status,error_detail,error_diagnostics
          FROM governance.code_analysis_audits
          WHERE job_id=:id AND tenant_id=:tenant
          ORDER BY created_at DESC LIMIT 1"""),
                {
                    "id": job_id,
                    "tenant": context.tenant_id,
                    "user": context.user_id,
                },
            )
            .mappings()
            .one_or_none()
        )
    result = {key: row[key] for key in PUBLIC_JOB_FIELDS if key in row}
    result["artifacts"] = [dict(x) for x in artifacts]
    result["latest_audit"] = dict(audit) if audit else None
    return result


def list_jobs(
    context: TenantContext,
    offset: int,
    limit: int,
    chat_session_id: str | None = None,
    created_after: datetime | None = None,
    file_id: str | None = None,
) -> dict:
    where = " WHERE tenant_id=:tenant"
    params: dict[str, Any] = {
        "tenant": context.tenant_id,
        "offset": max(offset, 0),
        "limit": min(limit, 50) + 1,
    }
    if chat_session_id:
        params["session"] = chat_session_id
        where += " AND chat_session_id=:session"
    if created_after:
        params["created_after"] = created_after
        where += " AND created_at >= :created_after"
    if file_id:
        params["file_id"] = file_id
        where += " AND file_id=:file_id"
    with engine.begin() as conn:
        _scope(conn, context)
        total = conn.execute(
            text(f"SELECT count(*) FROM governance.code_analysis_jobs{where}"),
            params,
        ).scalar_one()
        rows = (
            conn.execute(
                text(
                    "SELECT job_id,question,status,file_id,created_at,updated_at,completed_at,error_code "
                    f"FROM governance.code_analysis_jobs{where} "
                    "ORDER BY updated_at DESC OFFSET :offset LIMIT :limit"
                ),
                params,
            )
            .mappings()
            .all()
        )
    return {
        "items": [dict(x) for x in rows[:limit]],
        "next_offset": offset + limit if offset + limit < total else None,
        "total": int(total),
    }


def confirm_job(context: TenantContext, job_id: str, plan_hash: str) -> dict:
    expired = False
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        _scope(conn, context)
        row = (
            conn.execute(
                text("""SELECT j.status,j.plan_hash,j.plan_data,j.resource_limits,
                              j.expires_at,f.expires_at AS file_expires_at,
                              f.purged_at
                       FROM governance.code_analysis_jobs j
                       LEFT JOIN governance.code_analysis_files f
                         ON f.file_id=j.file_id
                       WHERE j.job_id=:id
                         AND j.tenant_id=:tenant AND j.user_id=:user
                       FOR UPDATE OF j"""),
                {
                    "id": job_id,
                    "tenant": context.tenant_id,
                    "user": context.user_id,
                },
            )
            .mappings()
            .one_or_none()
        )
        if not row:
            raise PermissionError("Analysis job not found")
        if row["status"] != "awaiting_confirmation":
            raise ValueError("Only a pending job can be confirmed")
        if (
            row["expires_at"] <= now
            or row["file_expires_at"] is None
            or row["file_expires_at"] <= now
            or row["purged_at"] is not None
        ):
            conn.execute(
                text(
                    "UPDATE governance.code_analysis_jobs SET status='expired',updated_at=now() WHERE job_id=:id"
                    " AND tenant_id=:tenant AND user_id=:user"
                ),
                {
                    "id": job_id,
                    "tenant": context.tenant_id,
                    "user": context.user_id,
                },
            )
            expired = True
        else:
            calculated_hash = _hash(row["plan_data"])
            if not secrets.compare_digest(calculated_hash, row["plan_hash"]):
                raise ValueError("Stored analysis plan failed its integrity check")
            if not secrets.compare_digest(row["plan_hash"], plan_hash):
                raise ValueError(
                    "Analysis plan changed; refresh and confirm the latest version"
                )
            planned_limits = row["plan_data"].get("resource_limits")
            stored_limits = row["resource_limits"] or {}
            if planned_limits is not None and planned_limits != stored_limits:
                raise ValueError("Stored resource limits failed their integrity check")
            conn.execute(
                text(
                    "UPDATE governance.code_analysis_jobs SET status='queued',confirmed_at=now(),updated_at=now() WHERE job_id=:id"
                    " AND tenant_id=:tenant AND user_id=:user"
                ),
                {
                    "id": job_id,
                    "tenant": context.tenant_id,
                    "user": context.user_id,
                },
            )
    # Raise only after the transaction commits so the terminal state is durable
    # and subsequent reads/SSE connections do not keep reporting a confirmable job.
    if expired:
        raise ValueError("Analysis plan has expired")
    return get_job(context, job_id) or {}


def revise_job(context: TenantContext, job_id: str, question: str) -> dict:
    settings = get_settings()
    revised_question = question.strip()
    if not revised_question:
        raise ValueError("Analysis question is required")
    if len(revised_question) > 4000:
        raise ValueError("Analysis question exceeds the 4000 character limit")
    new_job_id = _id("caj")
    expired = False
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        _scope(conn, context)
        row = (
            conn.execute(
                text("""SELECT j.status,j.expires_at,j.file_id,j.chat_session_id,
                              j.source_turn_id,j.question,j.plan_data,j.revision,
                              j.resource_limits,f.expires_at AS file_expires_at,
                              f.purged_at
                       FROM governance.code_analysis_jobs j
                       LEFT JOIN governance.code_analysis_files f
                         ON f.file_id=j.file_id
                       WHERE j.job_id=:id
                         AND j.tenant_id=:tenant AND j.user_id=:user
                       FOR UPDATE OF j"""),
                {
                    "id": job_id,
                    "tenant": context.tenant_id,
                    "user": context.user_id,
                },
            )
            .mappings()
            .one_or_none()
        )
        if not row:
            raise PermissionError("Analysis job not found")
        if row["status"] != "awaiting_confirmation":
            raise ValueError("Only a pending job can be revised")
        if (
            row["expires_at"] <= now
            or row["file_expires_at"] is None
            or row["file_expires_at"] <= now
            or row["purged_at"] is not None
        ):
            conn.execute(
                text(
                    "UPDATE governance.code_analysis_jobs SET status='expired',updated_at=now() WHERE job_id=:id"
                    " AND tenant_id=:tenant AND user_id=:user"
                ),
                {
                    "id": job_id,
                    "tenant": context.tenant_id,
                    "user": context.user_id,
                },
            )
            expired = True
        else:
            if revised_question == row["question"]:
                raise ValueError("Revised analysis question must be different")
            _lock_owner(conn, context)
            _enforce_job_quotas(conn, context, settings, check_active=False)
            plan = dict(row["plan_data"])
            plan["question"] = revised_question
            resource_limits = dict(row["resource_limits"] or {})
            if not resource_limits:
                resource_limits = _resource_limits(settings)
            plan["resource_limits"] = resource_limits
            plan_hash = _hash(plan)
            revision = int(row["revision"] or 1) + 1
            plan_expires_at = min(
                now + timedelta(minutes=30),
                row["file_expires_at"],
            )
            conn.execute(
                text(
                    "UPDATE governance.code_analysis_jobs SET status='superseded',updated_at=now() WHERE job_id=:id"
                    " AND tenant_id=:tenant AND user_id=:user"
                ),
                {
                    "id": job_id,
                    "tenant": context.tenant_id,
                    "user": context.user_id,
                },
            )
            conn.execute(
                text("""INSERT INTO governance.code_analysis_jobs
                  (job_id,tenant_id,user_id,chat_session_id,source_turn_id,file_id,parent_job_id,revision,
                   question,plan_data,plan_hash,resource_limits,status,expires_at)
                  VALUES(:id,:tenant,:user,:session,:turn,:file,:parent,:revision,:question,
                         CAST(:plan AS jsonb),:hash,CAST(:limits AS jsonb),
                         'awaiting_confirmation',:expires)"""),
                {
                    "id": new_job_id,
                    "tenant": context.tenant_id,
                    "user": context.user_id,
                    "session": row["chat_session_id"],
                    "turn": row["source_turn_id"],
                    "file": row["file_id"],
                    "parent": job_id,
                    "revision": revision,
                    "question": revised_question,
                    "plan": json.dumps(plan, ensure_ascii=False),
                    "hash": plan_hash,
                    "limits": json.dumps(resource_limits, ensure_ascii=False),
                    "expires": plan_expires_at,
                },
            )
    if expired:
        raise ValueError("Analysis plan has expired")
    return get_job(context, new_job_id) or {}


def cancel_job(context: TenantContext, job_id: str) -> bool:
    with engine.begin() as conn:
        _scope(conn, context)
        row = conn.execute(
            text(
                "UPDATE governance.code_analysis_jobs SET cancel_requested=true, "
                "status=CASE WHEN status IN ('awaiting_confirmation','queued') "
                "THEN 'cancelled' ELSE status END,updated_at=now() "
                "WHERE job_id=:id AND tenant_id=:tenant AND user_id=:user "
                "AND status NOT IN ('completed','failed','cancelled','expired','superseded') "
                "RETURNING job_id"
            ),
            {
                "id": job_id,
                "tenant": context.tenant_id,
                "user": context.user_id,
            },
        ).one_or_none()
    return bool(row)


def _resource_owner(conn, context: TenantContext, resource_type: str, resource_id: str) -> str | None:
    table = {
        "file": "code_analysis_files",
        "job": "code_analysis_jobs",
    }.get(resource_type)
    if not table:
        raise ValueError("Unsupported Code Analysis resource type")
    resource_column = "file_id" if resource_type == "file" else "job_id"
    return conn.execute(
        text(f"SELECT user_id FROM governance.{table} WHERE tenant_id=:tenant AND {resource_column}=:resource"),
        {"tenant": context.tenant_id, "resource": resource_id},
    ).scalar_one_or_none()


def grant_access(
    context: TenantContext,
    resource_type: str,
    resource_id: str,
    grantee_user_id: str,
    permission: str,
    expires_at: datetime | None = None,
) -> dict:
    if resource_type not in {"file", "job"}:
        raise ValueError("Only files and jobs can be shared")
    if permission not in {"read", "execute", "audit"}:
        raise ValueError("Unsupported resource permission")
    if grantee_user_id == context.user_id:
        raise ValueError("A resource cannot be shared with its owner")
    grant_id = _id("cag")
    with engine.begin() as conn:
        _scope(conn, context)
        owner = _resource_owner(conn, context, resource_type, resource_id)
        if owner != context.user_id and context.role not in {"owner", "admin"}:
            raise PermissionError("Only the resource owner or tenant admin can share it")
        member = conn.execute(
            text("""SELECT 1 FROM iam.tenant_memberships
                   WHERE tenant_id=:tenant AND user_id=:user AND revoked_at IS NULL"""),
            {"tenant": context.tenant_id, "user": grantee_user_id},
        ).scalar_one_or_none()
        if not member:
            raise ValueError("Grantee is not an active member of this tenant")
        conn.execute(
            text("""INSERT INTO governance.code_analysis_access_grants
              (grant_id,tenant_id,resource_type,resource_id,grantee_user_id,
               permission,granted_by,expires_at)
              VALUES(:id,:tenant,:type,:resource,:grantee,:permission,:actor,:expires)
              ON CONFLICT (tenant_id,resource_type,resource_id,grantee_user_id,permission)
              WHERE revoked_at IS NULL
              DO UPDATE SET expires_at=EXCLUDED.expires_at,revoked_at=NULL
              RETURNING grant_id"""),
            {
                "id": grant_id, "tenant": context.tenant_id, "type": resource_type,
                "resource": resource_id, "grantee": grantee_user_id,
                "permission": permission, "actor": context.user_id, "expires": expires_at,
            },
        ).scalar_one()
        grant_id = str(grant_id)
    return {"grant_id": grant_id, "resource_type": resource_type, "resource_id": resource_id,
            "grantee_user_id": grantee_user_id, "permission": permission, "expires_at": expires_at}


def revoke_access(context: TenantContext, grant_id: str) -> bool:
    with engine.begin() as conn:
        _scope(conn, context)
        result = conn.execute(
            text("""UPDATE governance.code_analysis_access_grants g
                   SET revoked_at=now()
                   WHERE grant_id=:id AND tenant_id=:tenant
                     AND (granted_by=:user OR EXISTS (
                       SELECT 1 FROM governance.code_analysis_files f
                        WHERE g.resource_type='file' AND f.file_id=g.resource_id
                          AND f.tenant_id=g.tenant_id AND f.user_id=:user
                     ) OR EXISTS (
                       SELECT 1 FROM governance.code_analysis_jobs j
                        WHERE g.resource_type='job' AND j.job_id=g.resource_id
                          AND j.tenant_id=g.tenant_id AND j.user_id=:user
                     )) AND revoked_at IS NULL
                   RETURNING grant_id"""),
            {"id": grant_id, "tenant": context.tenant_id, "user": context.user_id},
        ).scalar_one_or_none()
    return result is not None


def list_access(context: TenantContext, resource_type: str, resource_id: str) -> list[dict]:
    with engine.begin() as conn:
        _scope(conn, context)
        owner = _resource_owner(conn, context, resource_type, resource_id)
        if owner != context.user_id and context.role not in {"owner", "admin"}:
            raise PermissionError("Only the resource owner or tenant admin can inspect shares")
        rows = conn.execute(
            text("""SELECT grant_id,resource_type,resource_id,grantee_user_id,
                          permission,granted_by,expires_at,revoked_at,created_at
                   FROM governance.code_analysis_access_grants
                   WHERE tenant_id=:tenant AND resource_type=:type AND resource_id=:resource
                   ORDER BY created_at"""),
            {"tenant": context.tenant_id, "type": resource_type, "resource": resource_id},
        ).mappings().all()
    return [dict(row) for row in rows]


def record_access_event(
    context: TenantContext,
    resource_type: str,
    resource_id: str,
    action: str,
    *,
    allowed: bool = True,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist a non-sensitive access decision for later audit review."""
    with engine.begin() as conn:
        _scope(conn, context)
        conn.execute(
            text("""INSERT INTO governance.code_analysis_access_events
              (event_id,tenant_id,user_id,resource_type,resource_id,action,allowed,metadata)
              VALUES(:id,:tenant,:user,:type,:resource,:action,:allowed,CAST(:metadata AS jsonb))"""),
            {
                "id": _id("cae"), "tenant": context.tenant_id, "user": context.user_id,
                "type": resource_type, "resource": resource_id, "action": action,
                "allowed": allowed, "metadata": json.dumps(metadata or {}, ensure_ascii=False),
            },
        )


def get_audit(context: TenantContext, job_id: str) -> dict | None:
    with engine.begin() as conn:
        _scope(conn, context)
        row = conn.execute(
            text("""SELECT a.* FROM governance.code_analysis_audits a
                   WHERE a.job_id=:job AND a.tenant_id=:tenant
                   ORDER BY a.created_at DESC LIMIT 1"""),
            {"job": job_id, "tenant": context.tenant_id},
        ).mappings().one_or_none()
    return dict(row) if row else None


def delete_job(context: TenantContext, job_id: str) -> bool:
    """Delete a terminal standalone job after removing all stored artifacts."""
    store = get_artifact_store()
    with engine.begin() as conn:
        _scope(conn, context)
        row = (
            conn.execute(
                text("""SELECT status,chat_session_id
                  FROM governance.code_analysis_jobs
                  WHERE job_id=:id AND tenant_id=:tenant AND user_id=:user
                  FOR UPDATE"""),
                {
                    "id": job_id,
                    "tenant": context.tenant_id,
                    "user": context.user_id,
                },
            )
            .mappings()
            .one_or_none()
        )
        if not row:
            raise PermissionError("Analysis job not found")
        if row["chat_session_id"]:
            raise ValueError(
                "Chat-linked analysis jobs are deleted with their chat history"
            )
        if row["status"] not in {
            "completed",
            "failed",
            "cancelled",
            "expired",
            "superseded",
        }:
            raise ValueError("Only a terminal analysis job can be deleted")
        artifact_keys = conn.execute(
            text("""SELECT object_key
              FROM governance.code_analysis_artifacts
              WHERE job_id=:id AND tenant_id=:tenant AND user_id=:user
              ORDER BY created_at"""),
            {
                "id": job_id,
                "tenant": context.tenant_id,
                "user": context.user_id,
            },
        ).scalars()
        for object_key in artifact_keys:
            try:
                store.delete(object_key)
            except Exception as exc:
                logger.warning(
                    "Code-analysis job deletion failed; database references were retained",
                    extra={
                        "job_id": job_id,
                        "error_type": type(exc).__name__,
                    },
                )
                raise ValueError(
                    "Stored analysis artifacts could not be deleted; retry later"
                ) from exc
        deleted = conn.execute(
            text("""DELETE FROM governance.code_analysis_jobs
              WHERE job_id=:id AND tenant_id=:tenant AND user_id=:user"""),
            {
                "id": job_id,
                "tenant": context.tenant_id,
                "user": context.user_id,
            },
        )
    return bool(deleted.rowcount)


def cleanup_expired(limit: int = 100) -> int:
    """Purge expired source objects while preserving jobs, results and audits."""
    removed = 0
    store = get_artifact_store()
    with engine.begin() as conn:
        rows = (
            conn.execute(
                text(
                    "SELECT file_id,object_key FROM governance.code_analysis_files "
                    "WHERE expires_at <= now() AND purged_at IS NULL "
                    "ORDER BY expires_at LIMIT :limit FOR UPDATE SKIP LOCKED"
                ),
                {"limit": limit},
            )
            .mappings()
            .all()
        )
        for row in rows:
            try:
                store.delete(row["object_key"])
            except Exception as exc:
                logger.warning(
                    "Code-analysis source retention cleanup failed; purge will be retried",
                    extra={
                        "file_id": row["file_id"],
                        "error_type": type(exc).__name__,
                    },
                )
                continue
            conn.execute(
                text("""UPDATE governance.code_analysis_files
                  SET purged_at=now()
                  WHERE file_id=:id AND purged_at IS NULL"""),
                {"id": row["file_id"]},
            )
            # A queued job cannot safely start once its raw source has been
            # purged.  A running worker receives a cancellation signal and
            # converges through its normal lease-fenced terminal transition.
            conn.execute(
                text("""UPDATE governance.code_analysis_jobs
                   SET status=CASE
                         WHEN status IN ('awaiting_confirmation','queued')
                           THEN 'expired'
                         ELSE status
                       END,
                       cancel_requested=CASE
                         WHEN status IN ('generating','validating','running')
                           THEN true
                         ELSE cancel_requested
                       END,
                       error_code=CASE
                         WHEN status IN ('awaiting_confirmation','queued')
                           THEN 'source_expired'
                         ELSE error_code
                       END,
                       updated_at=now()
                   WHERE file_id=:id
                     AND status IN ('awaiting_confirmation','queued',
                                    'generating','validating','running')"""),
                {"id": row["file_id"]},
            )
            removed += 1
    return removed

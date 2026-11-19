"""Lease-based Code Agent worker.

Run with ``python -m app.code_analysis.worker`` in the dedicated worker image.
It is the only component allowed to invoke Docker for the disposable runner.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import re
import subprocess
import tempfile
import threading
import time

from sqlalchemy import text
from sqlalchemy.exc import OperationalError, ProgrammingError, SQLAlchemyError

from app.code_analysis.integrity import compact_fingerprint, plan_fingerprint
from app.code_analysis.completeness import validate_result_completeness
from app.code_analysis.policy import validate_candidate_code, validate_candidate_requirements
from app.code_analysis.database import engine
from app.code_analysis.service import cleanup_expired
from app.code_analysis.providers import ProviderContext, get_provider
from app.code_analysis.storage import get_artifact_store, owner_artifact_prefix
from app.agents.runtime import make_agent_run
from app.core.config import get_settings
from app.harness.storage import write_trace

RESULT_PREFIX = "BIZINSIGHT_RESULT:"
logger = logging.getLogger(__name__)


def _digest(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def _digest_json(value: object) -> str:
    return compact_fingerprint(value)


def _load_verified_source(job: dict, store) -> bytes:
    """Bound the object read and verify database size/hash before execution."""
    expected_size = int(job["file_size_bytes"])
    if (
        expected_size < 0
        or expected_size > get_settings().code_analysis_max_upload_bytes
    ):
        raise InputIntegrityError(
            "Stored input size is outside the configured upload bound"
        )
    try:
        source = store.get(
            job["file_object_key"],
            max_bytes=expected_size,
        )
    except (OSError, ValueError) as exc:
        raise InputIntegrityError(
            "Stored input could not be read within its declared size"
        ) from exc
    if len(source) != expected_size:
        raise InputIntegrityError("Stored input size did not match the database")
    if not hmac.compare_digest(_digest(source), str(job["file_sha256"])):
        raise InputIntegrityError("Stored input failed its SHA-256 check")
    return source


def _verify_plan_integrity(job: dict) -> None:
    plan = job["plan_data"]
    contract = job["file_contract_data"]
    stored_limits = job.get("resource_limits") or {}
    planned_limits = plan.get("resource_limits", stored_limits) if isinstance(plan, dict) else {}
    if (
        not isinstance(plan, dict)
        or not hmac.compare_digest(
            plan_fingerprint(plan),
            str(job["plan_hash"]),
        )
        or not hmac.compare_digest(
            str(plan.get("file_sha256") or ""),
            str(job["file_sha256"]),
        )
        or not hmac.compare_digest(
            compact_fingerprint(plan.get("data_contract")),
            compact_fingerprint(contract),
        )
        or not isinstance(contract, dict)
        or contract.get("file_type") != job.get("file_type")
        or job.get("file_type") not in {"csv", "xlsx", "parquet"}
        or not hmac.compare_digest(
            compact_fingerprint(planned_limits),
            compact_fingerprint(stored_limits),
        )
    ):
        raise PlanIntegrityError(
            "Persisted analysis plan failed its integrity check"
        )


def _encode_bounded_result(result: dict, *, max_bytes: int | None = None) -> bytes:
    blob = json.dumps(
        result,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    limit = max_bytes if max_bytes is not None else getattr(
        get_settings(), "code_analysis_max_result_bytes", 2 * 1024 * 1024
    )
    limit = max(1, int(limit))
    if len(blob) > limit:
        raise ResultContractError("Analysis result exceeds the trusted JSON limit")
    return blob


def _effective_limits(job: dict) -> dict[str, int | float | str]:
    """Use signed per-job quotas, but never exceed today's administrator caps."""
    settings = get_settings()
    snapshot = job.get("resource_limits") or {}
    if not isinstance(snapshot, dict):
        raise PlanIntegrityError("Stored execution resource limits are invalid")

    def bounded_int(key: str, configured: int, minimum: int) -> int:
        try:
            requested = int(snapshot.get(key, configured))
        except (TypeError, ValueError) as exc:
            raise PlanIntegrityError(f"Stored {key} limit is invalid") from exc
        return max(minimum, min(requested, configured))

    def bounded_float(key: str, configured: float, minimum: float) -> float:
        try:
            requested = float(snapshot.get(key, configured))
        except (TypeError, ValueError) as exc:
            raise PlanIntegrityError(f"Stored {key} limit is invalid") from exc
        return max(minimum, min(requested, configured))

    image = str(snapshot.get("runner_image") or settings.code_analysis_runner_image)
    if not re.fullmatch(r"[A-Za-z0-9._/@:-]+", image):
        raise PlanIntegrityError("Stored runner image reference is invalid")
    return {
        "timeout_seconds": bounded_int(
            "timeout_seconds", settings.code_analysis_execution_timeout_seconds, 1
        ),
        "memory_mb": bounded_int(
            "memory_mb", settings.code_analysis_runner_memory_mb, 128
        ),
        "cpu_cores": bounded_float(
            "cpu_cores", settings.code_analysis_runner_cpu_cores, 0.1
        ),
        "pids": bounded_int("pids", settings.code_analysis_runner_pids, 16),
        "max_output_bytes": bounded_int(
            "max_output_bytes", settings.code_analysis_max_output_bytes, 1
        ),
        "max_result_bytes": bounded_int(
            "max_result_bytes", settings.code_analysis_max_result_bytes, 1
        ),
        "runner_image": image,
    }


def _record_reconciled_audits(conn, rows: list[dict]) -> None:
    """Persist terminal evidence for leases whose original worker crashed."""
    settings = get_settings()
    for row in rows:
        contract = conn.execute(
            text(
                """SELECT f.contract_data
                   FROM governance.code_analysis_jobs j
                   JOIN governance.code_analysis_files f ON f.file_id=j.file_id
                   WHERE j.job_id=:job"""
            ),
            {"job": row["job_id"]},
        ).scalar_one()
        audit_id = (
            "caa_"
            + _digest(
                f"{row['job_id']}:reconcile:{row['status']}:{row['attempts']}"
            )[:24]
        )
        conn.execute(
            text(
                """INSERT INTO governance.code_analysis_audits
                  (audit_id,tenant_id,user_id,job_id,status,provider,
                   generation_model,provider_version,candidate_attempts,
                   repair_attempts,plan_hash,data_contract_hash,
                   candidate_code_hash,runner_image_digest,result_hash,
                   trace_id,duration_ms,error_code)
                  VALUES(:id,:tenant,:user,:job,:status,'worker',
                         'reconciliation','builtin',0,0,:plan,:contract,
                         NULL,:image,NULL,:trace,0,:error)
                  ON CONFLICT (audit_id) DO NOTHING"""
            ),
            {
                "id": audit_id,
                "tenant": row["tenant_id"],
                "user": row["user_id"],
                "job": row["job_id"],
                "status": row["status"],
                "plan": row["plan_hash"],
                "contract": compact_fingerprint(contract),
                "image": settings.code_analysis_runner_image,
                "trace": f"trace_code_{row['job_id']}",
                "error": row["error_code"],
            },
        )


def _reconcile_expired_terminal_audits(conn, limit: int = 20) -> int:
    """Backfill audits left pending by a terminal audit database outage."""
    rows = conn.execute(
        text(
            """SELECT j.job_id,j.tenant_id,j.user_id,j.status,j.lease_token,
                      j.plan_hash,j.result_data,j.error_code,f.contract_data,
                      (SELECT a.sha256
                         FROM governance.code_analysis_artifacts a
                        WHERE a.job_id=j.job_id AND a.kind='candidate_code'
                        ORDER BY a.created_at DESC LIMIT 1) AS candidate_hash,
                      (SELECT a.metadata->>'provider'
                         FROM governance.code_analysis_artifacts a
                        WHERE a.job_id=j.job_id AND a.kind='candidate_code'
                        ORDER BY a.created_at DESC LIMIT 1) AS provider
               FROM governance.code_analysis_jobs j
               JOIN governance.code_analysis_files f ON f.file_id=j.file_id
               WHERE j.status IN ('completed','failed','cancelled')
                 AND j.lease_token IS NOT NULL
                 AND j.lease_expires_at < clock_timestamp()
                 AND NOT EXISTS (
                   SELECT 1 FROM governance.code_analysis_audits ca
                    WHERE ca.job_id=j.job_id
                 )
               ORDER BY j.updated_at
               FOR UPDATE OF j SKIP LOCKED
               LIMIT :limit"""
        ),
        {"limit": limit},
    ).mappings().all()
    settings = get_settings()
    reconciled = 0
    for raw in rows:
        row = dict(raw)
        result_hash = (
            _digest_json(row["result_data"])
            if row["result_data"] is not None
            else None
        )
        audit_id = (
            "caa_"
            + _digest(
                f"{row['job_id']}:terminal-reconcile:{row['lease_token']}"
            )[:24]
        )
        conn.execute(
            text(
                """INSERT INTO governance.code_analysis_audits
                  (audit_id,tenant_id,user_id,job_id,status,provider,
                   generation_model,provider_version,candidate_attempts,
                   repair_attempts,plan_hash,data_contract_hash,
                   candidate_code_hash,runner_image_digest,result_hash,
                   trace_id,duration_ms,error_code)
                  VALUES(:id,:tenant,:user,:job,:status,:provider,
                         'reconciliation','builtin',0,0,:plan,:contract,
                         :candidate,:image,:result,:trace,0,:error)
                  ON CONFLICT (audit_id) DO NOTHING"""
            ),
            {
                "id": audit_id,
                "tenant": row["tenant_id"],
                "user": row["user_id"],
                "job": row["job_id"],
                "status": row["status"],
                "provider": row["provider"] or "worker",
                "plan": row["plan_hash"],
                "contract": compact_fingerprint(row["contract_data"]),
                "candidate": row["candidate_hash"],
                "image": settings.code_analysis_runner_image,
                "result": result_hash,
                "trace": f"trace_code_{row['job_id']}",
                "error": row["error_code"],
            },
        )
        released = conn.execute(
            text(
                """UPDATE governance.code_analysis_jobs
                   SET lease_token=NULL,lease_expires_at=NULL,updated_at=now()
                   WHERE job_id=:job AND lease_token=:token
                     AND status=:status
                   RETURNING job_id"""
            ),
            {
                "job": row["job_id"],
                "token": row["lease_token"],
                "status": row["status"],
            },
        ).scalar_one_or_none()
        if released is None:
            raise LeaseLost("Terminal audit reconciliation lost its lease marker")
        reconciled += 1
    return reconciled


def _claim() -> dict | None:
    """Atomically lease one confirmed job; stale leases can be recovered."""
    settings = get_settings()
    max_attempts = max(1, settings.code_analysis_max_job_attempts)
    with engine.begin() as conn:
        _reconcile_expired_terminal_audits(conn)
        # Do not let a cleanup/expiry race start work without a source object.
        conn.execute(
            text("""UPDATE governance.code_analysis_jobs j
                   SET status='expired',error_code='source_expired',updated_at=now()
                   FROM governance.code_analysis_files f
                   WHERE j.file_id=f.file_id
                     AND j.status IN ('awaiting_confirmation','queued')
                     AND (f.purged_at IS NOT NULL OR f.expires_at <= clock_timestamp())""")
        )
        # A crashed owner cannot perform its own terminal transition. Converge
        # stale cancellation and retry exhaustion before leasing new work.
        reconciled_cancelled = conn.execute(
            text(
                """UPDATE governance.code_analysis_jobs
                   SET status='cancelled',lease_token=NULL,lease_expires_at=NULL,
                       error_code=NULL,updated_at=now()
                   WHERE status IN ('generating','validating','running')
                     AND lease_expires_at < clock_timestamp()
                     AND cancel_requested
                   RETURNING job_id,tenant_id,user_id,status,attempts,plan_hash,
                             error_code"""
            )
        ).mappings().all()
        reconciled_failed = conn.execute(
            text(
                """UPDATE governance.code_analysis_jobs
                   SET status='failed',lease_token=NULL,lease_expires_at=NULL,
                       error_code='worker_retry_exhausted',updated_at=now()
                   WHERE status IN ('generating','validating','running')
                     AND lease_expires_at < clock_timestamp()
                     AND NOT cancel_requested
                     AND attempts >= :max_attempts
                   RETURNING job_id,tenant_id,user_id,status,attempts,plan_hash,
                             error_code"""
            ),
            {"max_attempts": max_attempts},
        ).mappings().all()
        _record_reconciled_audits(
            conn,
            [
                *(dict(row) for row in reconciled_cancelled),
                *(dict(row) for row in reconciled_failed),
            ],
        )
        row = conn.execute(text("""
          SELECT j.*, f.object_key AS file_object_key, f.file_type AS file_type,
                 f.contract_data AS file_contract_data, f.sha256 AS file_sha256,
                 f.size_bytes AS file_size_bytes
          FROM governance.code_analysis_jobs j
          JOIN governance.code_analysis_files f ON f.file_id=j.file_id
          WHERE (j.status='queued' OR (j.status IN ('generating','validating','running') AND j.lease_expires_at < clock_timestamp()))
            AND NOT j.cancel_requested
            AND j.attempts < :max_attempts
            AND f.purged_at IS NULL
            AND f.expires_at > clock_timestamp()
          ORDER BY j.created_at FOR UPDATE SKIP LOCKED LIMIT 1
        """), {"max_attempts": max_attempts}).mappings().one_or_none()
        if not row:
            return None
        token = _digest(f"{row['job_id']}:{time.time()}")[:32]
        conn.execute(text("""UPDATE governance.code_analysis_jobs
          SET status='generating', lease_token=:token, lease_expires_at=now()+(:seconds || ' seconds')::interval,
              attempts=attempts+1, updated_at=now() WHERE job_id=:id"""),
          {"id":row["job_id"], "token":token, "seconds":settings.code_analysis_job_lease_seconds})
        claimed = dict(row)
        claimed["lease_token"] = token
        claimed["lease_attempt"] = int(row["attempts"]) + 1
        return claimed


class LeaseLost(RuntimeError):
    """Raised when this worker no longer owns the job's fencing token."""


class SandboxCancelled(RuntimeError):
    pass


class JobDeadlineExceeded(RuntimeError):
    pass


class InputIntegrityError(RuntimeError):
    pass


class PlanIntegrityError(RuntimeError):
    pass


class ResultContractError(RuntimeError):
    pass


class ResultCompletenessError(RuntimeError):
    pass


_ACTIVE_JOB_STATUSES = ("generating", "validating", "running")


def _assert_owned_lease(
    conn,
    job: dict,
    *,
    allow_cancel: bool = False,
) -> dict:
    """Lock and verify the live lease before any state or artifact write."""
    row = conn.execute(
        text(
            """SELECT status,cancel_requested
               FROM governance.code_analysis_jobs
               WHERE job_id=:id AND lease_token=:token
                 AND lease_expires_at > clock_timestamp()
                 AND status IN ('generating','validating','running')
               FOR UPDATE"""
        ),
        {"id": job["job_id"], "token": job["lease_token"]},
    ).mappings().one_or_none()
    if row is None:
        raise LeaseLost("Code analysis lease was lost")
    owned = dict(row)
    if owned["cancel_requested"] and not allow_cancel:
        raise SandboxCancelled("Code analysis was cancelled")
    return owned


def _fenced_job_update(
    conn,
    job: dict,
    statement: str,
    parameters: dict | None = None,
) -> str:
    """Execute an UPDATE ... RETURNING under the current fencing token."""
    values = {
        "id": job["job_id"],
        "token": job["lease_token"],
        **(parameters or {}),
    }
    returned = conn.execute(text(statement), values).scalar_one_or_none()
    if returned is None:
        raise LeaseLost("Code analysis lease was lost during a state transition")
    return str(returned)


def _artifact(conn, job: dict, kind: str, key: str, data: bytes, metadata: dict | None = None) -> dict:
    # Hold the job row lock across object-store and metadata writes.  A stale
    # worker can neither overwrite a current owner's object nor insert metadata
    # after its token has expired or been replaced.
    _assert_owned_lease(conn, job)
    store = get_artifact_store()
    object_key: str | None = None
    try:
        object_key, digest, size = store.put(key, data)
        # Use clock_timestamp(), not transaction-stable now(), so a slow remote
        # object write cannot carry an expired lease into metadata.
        _assert_owned_lease(conn, job)
        artifact_id = f"art_{_digest(object_key + digest)[:24]}"
        conn.execute(text("""INSERT INTO governance.code_analysis_artifacts
          (artifact_id,tenant_id,user_id,job_id,kind,object_key,sha256,size_bytes,metadata)
          VALUES(:id,:tenant,:user,:job,:kind,:key,:sha,:size,CAST(:metadata AS jsonb))"""),
          {"id":artifact_id,"tenant":job["tenant_id"],"user":job["user_id"],"job":job["job_id"],"kind":kind,"key":object_key,"sha":digest,"size":size,"metadata":json.dumps(metadata or {},ensure_ascii=False)})
    except Exception:
        if object_key is not None:
            try:
                store.delete(object_key)
            except Exception as cleanup_exc:
                logger.error(
                    "Artifact compensation failed (%s)",
                    type(cleanup_exc).__name__,
                )
        raise
    return {"artifact_id":artifact_id,"object_key":object_key,"sha256":digest,"size_bytes":size}


class SandboxExecutionError(RuntimeError):
    def __init__(self, message: str, logs: bytes = b"") -> None:
        super().__init__(message)
        self.logs = logs[-100_000:]


def _check_live_lease(job: dict) -> None:
    """Poll cancellation and fencing state while an attached runner executes."""
    with engine.begin() as conn:
        _assert_owned_lease(conn, job)


def _docker(
    *args: str,
    check: bool = True,
    input_data: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        ["docker", *args],
        input=input_data,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and completed.returncode != 0:
        raise SandboxExecutionError(
            f"Docker command failed: {' '.join(args[:2])}",
            completed.stdout,
        )
    return completed


def _runner_image_digest(image: str | None = None) -> str:
    image = image or get_settings().code_analysis_runner_image
    completed = _docker("image", "inspect", "--format", "{{.Id}}", image, check=False)
    return completed.stdout.decode("utf-8", "replace").strip() or image


def _resource_is_orphan(job_id: str, lease_digest: str) -> bool:
    """Fail closed unless the labeled resource no longer has a live owner."""
    with engine.begin() as conn:
        row = conn.execute(
            text(
                """SELECT status,lease_token,
                          lease_expires_at > clock_timestamp() AS lease_live
                   FROM governance.code_analysis_jobs
                   WHERE job_id=:id"""
            ),
            {"id": job_id},
        ).mappings().one_or_none()
    if row is None:
        return True
    return not (
        row["status"] in _ACTIVE_JOB_STATUSES
        and bool(row["lease_live"])
        and bool(row["lease_token"])
        and hmac.compare_digest(
            _digest(str(row["lease_token"])),
            lease_digest,
        )
    )


def _inspect_labeled_resource(kind: str, resource_id: str) -> dict | None:
    inspected = _docker(kind, "inspect", resource_id, check=False)
    if inspected.returncode != 0:
        return None
    try:
        payload = json.loads(inspected.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list) or len(payload) != 1:
        return None
    item = payload[0]
    return item if isinstance(item, dict) else None


def _scavenge_orphaned_runners() -> int:
    """Remove only old, strictly labeled resources whose DB lease is stale."""
    settings = get_settings()
    minimum_age = max(
        300,
        settings.code_analysis_execution_timeout_seconds + 60,
        settings.code_analysis_job_lease_seconds * 2,
    )
    now = int(time.time())
    removed = 0
    for kind in ("container", "volume"):
        if kind == "container":
            listed = _docker(
                "container",
                "ls",
                "--all",
                "--quiet",
                "--filter",
                "label=bizinsight.code-analysis=true",
                check=False,
            )
        else:
            listed = _docker(
                "volume",
                "ls",
                "--quiet",
                "--filter",
                "label=bizinsight.code-analysis=true",
                check=False,
            )
        if listed.returncode != 0:
            continue
        for resource_id in listed.stdout.decode("utf-8", "replace").splitlines():
            resource_id = resource_id.strip()
            if not resource_id:
                continue
            inspected = _inspect_labeled_resource(kind, resource_id)
            if not inspected:
                continue
            labels = (
                (inspected.get("Config") or {}).get("Labels")
                if kind == "container"
                else inspected.get("Labels")
            )
            if not isinstance(labels, dict):
                continue
            job_id = str(labels.get("bizinsight.job-id") or "")
            lease_digest = str(
                labels.get("bizinsight.lease-token-sha256") or ""
            )
            created = str(labels.get("bizinsight.created-at") or "")
            if (
                labels.get("bizinsight.code-analysis") != "true"
                or not job_id
                or not re.fullmatch(r"[0-9a-f]{64}", lease_digest)
                or not created.isdigit()
                or now - int(created) < minimum_age
            ):
                continue
            if not _resource_is_orphan(job_id, lease_digest):
                continue
            if kind == "container":
                removed_result = _docker(
                    "container",
                    "rm",
                    "--force",
                    resource_id,
                    check=False,
                )
            else:
                removed_result = _docker(
                    "volume",
                    "rm",
                    "--force",
                    resource_id,
                    check=False,
                )
            if removed_result.returncode == 0:
                removed += 1
    return removed


def _parse_runner_output(logs: bytes) -> tuple[dict, list[dict], str]:
    decoded = logs.decode("utf-8", "replace")
    result_line = next(
        (line for line in reversed(decoded.splitlines()) if line.startswith(RESULT_PREFIX)),
        None,
    )
    if result_line is None:
        raise SandboxExecutionError("Sandbox runner did not return a result envelope", logs)
    try:
        envelope = json.loads(result_line[len(RESULT_PREFIX):])
    except json.JSONDecodeError as exc:
        raise SandboxExecutionError("Sandbox runner returned malformed JSON", logs) from exc
    result = envelope.get("result")
    artifacts = envelope.get("artifacts", [])
    if not isinstance(result, dict) or not isinstance(artifacts, list):
        raise SandboxExecutionError("Sandbox runner returned an invalid output contract", logs)
    return result, artifacts, str(envelope.get("candidate_log") or "")[:32_000]


def _run_in_sandbox(
    source: bytes,
    extension: str,
    code: str,
    job: dict,
) -> tuple[dict, bytes, list[dict]]:
    """Run via Docker copy/start so containerized workers never bind host paths."""
    settings = get_settings()
    limits = _effective_limits(job)
    job_id = job["job_id"]
    safe_extension = {
        "csv": "csv",
        "xlsx": "xlsx",
        "parquet": "parquet",
    }.get(str(extension))
    if safe_extension is None:
        raise InputIntegrityError("Stored input has an unsupported file type")
    extension = safe_extension
    container_name = f"bizinsight-runner-{re.sub(r'[^a-zA-Z0-9_.-]', '-', job_id)[-40:]}-{_digest(str(time.time()))[:8]}"
    volume_name = f"bizinsight-job-{_digest(container_name)[:24]}"
    created_at = str(int(time.time()))
    lease_digest = _digest(job["lease_token"])
    try:
        _docker(
            "volume", "create",
            "--label", "bizinsight.code-analysis=true",
            "--label", f"bizinsight.job-id={job_id}",
            "--label", f"bizinsight.lease-token-sha256={lease_digest}",
            "--label", f"bizinsight.created-at={created_at}",
            volume_name,
        )
        staging_flags = (
            "run", "--rm", "--network", "none", "--cap-drop", "ALL",
            "--cap-add", "CHOWN",
            "--security-opt", "no-new-privileges",
            "--user", "0:0",
            "--interactive",
            "--entrypoint", "/bin/sh",
            "-v", f"{volume_name}:/staging",
            str(limits["runner_image"]),
        )
        _docker(
            *staging_flags,
            "-c", f"umask 077; cat > /staging/source.{extension}; chown 10001:10001 /staging/source.{extension}",
            input_data=source,
        )
        _docker(
            *staging_flags,
            "-c", "umask 077; cat > /staging/analysis.py; chown 10001:10001 /staging/analysis.py",
            input_data=code.encode("utf-8"),
        )
        _docker(
            "create",
            "--name", container_name,
            "--label", "bizinsight.code-analysis=true",
            "--label", f"bizinsight.job-id={job_id}",
            "--label", f"bizinsight.lease-token-sha256={lease_digest}",
            "--label", f"bizinsight.created-at={created_at}",
            "--network", "none",
            "--read-only",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", str(limits["pids"]),
            "--memory", f"{limits['memory_mb']}m",
            "--cpus", str(limits["cpu_cores"]),
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "--tmpfs", (
                "/work/output:rw,noexec,nosuid,"
                f"size={max(16, min(512, int(limits['max_output_bytes']) // (1024 * 1024) + 8))}m,"
                "uid=10001,gid=10001"
            ),
            "-v", f"{volume_name}:/work/task:ro",
            "--env", f"CODE_ANALYSIS_MAX_OUTPUT_BYTES={limits['max_output_bytes']}",
            "--env", (
                "CODE_ANALYSIS_MAX_RESULT_BYTES="
                f"{limits['max_result_bytes']}"
            ),
            "--env", f"CODE_ANALYSIS_EXECUTION_TIMEOUT_SECONDS={limits['timeout_seconds']}",
            "--env", "CODE_ANALYSIS_CANDIDATE_PATH=/work/task/analysis.py",
            "--env", "MPLCONFIGDIR=/tmp/matplotlib",
            "--env", "MPLBACKEND=Agg",
            "--env", "XDG_CACHE_HOME=/tmp/cache",
            "--env", "OPENBLAS_NUM_THREADS=1",
            str(limits["runner_image"]),
            f"/work/task/source.{extension}",
        )
        with tempfile.TemporaryFile() as output:
            process = subprocess.Popen(
                ["docker", "start", "--attach", container_name],
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + int(limits["timeout_seconds"])
            while process.poll() is None:
                try:
                    _check_live_lease(job)
                except (LeaseLost, SandboxCancelled):
                    _docker("kill", container_name, check=False)
                    process.wait(timeout=10)
                    raise
                if time.monotonic() >= deadline:
                    _docker("kill", container_name, check=False)
                    process.wait(timeout=10)
                    output.seek(0)
                    raise SandboxExecutionError("Sandbox execution timed out", output.read())
                time.sleep(0.5)
            process.wait(timeout=10)
            output.seek(0)
            logs = output.read()
        inspection = _docker("inspect", "--format", "{{.State.ExitCode}}", container_name, check=False)
        exit_code_text = inspection.stdout.decode("utf-8", "replace").strip()
        exit_code = int(exit_code_text) if exit_code_text.isdigit() else process.returncode
        if exit_code != 0:
            raise SandboxExecutionError(f"Sandbox runner failed with exit code {exit_code}", logs)
        result, artifacts, candidate_log = _parse_runner_output(logs)
        audit_logs = candidate_log.encode("utf-8", "replace")[-32_000:]
        return result, audit_logs, artifacts
    finally:
        _docker("rm", "--force", container_name, check=False)
        _docker("volume", "rm", "--force", volume_name, check=False)


def _sanitized_feedback(kind: str, message: str) -> dict:
    value = message[-2_000:]
    value = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[redacted-email]", value)
    value = re.sub(r"(?:[A-Za-z]:\\|/)(?:[^\s:]+[/\\])+[^\s:]*", "[redacted-path]", value)
    return {"failure_type": kind, "message": value}


def _classify_sandbox_failure(exc: SandboxExecutionError) -> str:
    """Classify failures before deciding whether an LLM repair is appropriate."""
    message = str(exc).lower()
    if "timed out" in message or "timeout" in message:
        return "timeout"
    if re.search(r"exit code (?:137|143|125)\b", message):
        return "resource_limit"
    infrastructure_markers = (
        "docker command failed",
        "cannot connect to the docker daemon",
        "is the docker daemon running",
        "no such image",
        "pull access denied",
        "volume is in use",
        "permission denied",
    )
    if any(marker in message for marker in infrastructure_markers):
        return "sandbox_infrastructure"
    if "invalid output contract" in message or "malformed json" in message:
        return "result_contract"
    if "did not return a result envelope" in message:
        return "result_contract"
    return "candidate_bug"


def _repair_fingerprint(kind: str, message: str) -> str:
    """Stable, bounded fingerprint used to stop identical repair loops."""
    feedback = _sanitized_feedback(kind, message)
    return _digest_json(feedback)


def _sanitize_diagnostic_text(message: str) -> str:
    """Keep useful provider diagnostics without persisting secrets or payloads."""
    value = str(message or "").strip()
    value = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/-]+", "Bearer [redacted]", value)
    value = re.sub(
        r"(?i)(api[-_]?key|token|password|secret)\s*[:=]\s*[^\s,;]+",
        r"\1=[redacted]",
        value,
    )
    value = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[redacted-email]", value)
    value = re.sub(r"(?:[A-Za-z]:\\|/)(?:[^\s:]+[/\\])+[^\s:]*", "[redacted-path]", value)
    # Query strings can contain credentials or uploaded identifiers.
    value = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[redacted-query]", value)
    return value[:2_000] or "Provider request failed without a diagnostic message"


def _provider_error_diagnostic(
    exc: Exception,
    *,
    phase: str,
    provider_request_id: str = "",
) -> dict[str, object]:
    """Extract a bounded, non-sensitive summary from an upstream exception."""
    response = getattr(exc, "response", None)
    http_status = getattr(response, "status_code", None)
    if not isinstance(http_status, int):
        http_status = None
    detail = ""
    if response is not None:
        try:
            payload = response.json()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                detail = str(error.get("message") or error.get("detail") or "")
            if not detail:
                detail = str(payload.get("detail") or payload.get("message") or "")
        if not detail:
            try:
                detail = str(response.text or "")
            except Exception:
                detail = ""
    summary = f"{type(exc).__name__}: {detail or str(exc)}"
    safe_diagnostics = getattr(exc, "diagnostics", {})
    if not isinstance(safe_diagnostics, dict):
        safe_diagnostics = {}
    return {
        "phase": phase,
        "http_status": http_status,
        "provider_request_id": provider_request_id,
        "detail": _sanitize_diagnostic_text(summary),
        "diagnostics": safe_diagnostics,
    }


class _LeaseHeartbeat:
    def __init__(self, job: dict, *, deadline_monotonic: float | None = None) -> None:
        self.job = job
        self.deadline_monotonic = deadline_monotonic
        self.stop_event = threading.Event()
        self.lost_event = threading.Event()
        self.cancel_event = threading.Event()
        self.failure: Exception | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "_LeaseHeartbeat":
        _check_live_lease(self.job)
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)

    def checkpoint(self, *, verify: bool = False) -> None:
        if self.deadline_monotonic is not None and time.monotonic() >= self.deadline_monotonic:
            raise JobDeadlineExceeded("Code analysis job deadline exceeded")
        if self.lost_event.is_set():
            detail = f": {self.failure}" if self.failure else ""
            raise LeaseLost(f"Code analysis lease heartbeat stopped{detail}")
        if self.cancel_event.is_set():
            raise SandboxCancelled("Code analysis was cancelled")
        if verify:
            try:
                _check_live_lease(self.job)
            except LeaseLost as exc:
                self.failure = exc
                self.lost_event.set()
                raise
            except SandboxCancelled:
                self.cancel_event.set()
                raise

    def _run(self) -> None:
        settings = get_settings()
        interval = max(5, settings.code_analysis_job_lease_seconds // 3)
        while not self.stop_event.wait(interval):
            if self.deadline_monotonic is not None and time.monotonic() >= self.deadline_monotonic:
                self.failure = JobDeadlineExceeded("Code analysis job deadline exceeded")
                self.lost_event.set()
                return
            try:
                with engine.begin() as conn:
                    row = conn.execute(
                        text("""UPDATE governance.code_analysis_jobs
                          SET lease_expires_at=now()+(:seconds || ' seconds')::interval,
                              updated_at=now()
                          WHERE job_id=:id AND lease_token=:token
                            AND lease_expires_at > clock_timestamp()
                            AND status IN ('generating','validating','running')
                          RETURNING cancel_requested"""),
                        {
                            "id": self.job["job_id"],
                            "token": self.job["lease_token"],
                            "seconds": settings.code_analysis_job_lease_seconds,
                        },
                    ).scalar_one_or_none()
                if row is None:
                    self.lost_event.set()
                    return
                if bool(row):
                    self.cancel_event.set()
                    return
            except Exception as exc:  # fail closed; the lease may now expire
                self.failure = exc
                self.lost_event.set()
                return


def _mark_cancelled(conn, job: dict) -> str:
    return _fenced_job_update(
        conn,
        job,
        """UPDATE governance.code_analysis_jobs
           SET status='cancelled',error_code=NULL,updated_at=now()
           WHERE job_id=:id AND lease_token=:token
             AND lease_expires_at > clock_timestamp()
             AND status IN ('generating','validating','running')
           RETURNING status""",
    )


def _mark_failed_or_cancelled(conn, job: dict, error_code: str) -> str:
    return _fenced_job_update(
        conn,
        job,
        """UPDATE governance.code_analysis_jobs
           SET status=CASE WHEN cancel_requested THEN 'cancelled' ELSE 'failed' END,
               error_code=CASE WHEN cancel_requested THEN NULL ELSE :error END,
               updated_at=now()
           WHERE job_id=:id AND lease_token=:token
             AND lease_expires_at > clock_timestamp()
             AND status IN ('generating','validating','running')
           RETURNING status""",
        {"error": error_code},
    )


def _write_audit_and_release(
    job: dict,
    *,
    status: str,
    provider_name: str,
    generation_model: str,
    provider_version: str,
    provider_request_id: str,
    candidate_attempts: int,
    candidate_hash: str | None,
    runner_digest: str,
    result_hash: str | None,
    trace_id: str,
    duration: int,
    error_code: str | None,
    error_phase: str | None,
    error_http_status: int | None,
    error_detail: str | None,
    error_diagnostics: dict[str, object] | None,
) -> None:
    """Write one terminal audit and release only the matching fencing token."""
    settings = get_settings()
    parameters = {
        "id": (
            "caa_"
            f"{_digest(job['job_id'] + ':' + job['lease_token'])[:24]}"
        ),
        "tenant": job["tenant_id"],
        "user": job["user_id"],
        "job": job["job_id"],
        "token": job["lease_token"],
        "status": status,
        "provider": provider_name or settings.code_analysis_provider,
        "model": generation_model
        or (
            "deterministic"
            if settings.code_analysis_provider == "fake"
            else settings.llm_model
        ),
        "provider_version": provider_version
        or settings.code_analysis_opencode_version,
        "candidate_attempts": candidate_attempts,
        "provider_request_id": provider_request_id or None,
        "repair_attempts": max(0, candidate_attempts - 1),
        "plan": job["plan_hash"],
        "contract": _digest_json(job["file_contract_data"]),
        "input_sha256": str(job.get("file_sha256") or ""),
        "candidate": candidate_hash,
        "image": runner_digest,
        "result": result_hash,
        "trace": trace_id,
        "duration": duration,
        "error": error_code,
        "error_phase": error_phase,
        "error_http_status": error_http_status,
        "error_detail": error_detail,
        "error_diagnostics": json.dumps(error_diagnostics or {}, ensure_ascii=False),
    }
    for attempt in range(3):
        try:
            with engine.begin() as conn:
                # :job / :status 同时作为"写入审计列的列值(text)"与"WHERE 比较(varchar)"
                # 会被 psycopg 判定为参数类型歧义；WHERE 改用独立参数名 expect_job/expect_status。
                audit_params = {**parameters, "expect_job": parameters["job"], "expect_status": parameters["status"]}
                inserted = conn.execute(
                    text(
                        """INSERT INTO governance.code_analysis_audits
                          (audit_id,tenant_id,user_id,job_id,status,provider,
                           generation_model,provider_version,candidate_attempts,
                           repair_attempts,plan_hash,data_contract_hash,
                           candidate_code_hash,runner_image_digest,result_hash,
                           trace_id,duration_ms,error_code,actor_id,initiator_type,
                           permission_decision,provider_request_id,input_sha256,
                           input_tokens,output_tokens,estimated_cost,error_category,
                            error_phase,error_http_status,error_detail,
                            error_diagnostics)
                          SELECT :id,:tenant,:user,:job,:status,:provider,:model,
                                 :provider_version,:candidate_attempts,
                                 :repair_attempts,:plan,:contract,:candidate,
                                 :image,:result,:trace,:duration,:error,:user,
                                 'worker','allow',:provider_request_id,:input_sha256,
                                 0,0,0,:error,:error_phase,:error_http_status,
                                  :error_detail,CAST(:error_diagnostics AS jsonb)
                          FROM governance.code_analysis_jobs
                          WHERE job_id=:expect_job AND lease_token=:token
                            AND status=:expect_status
                          ON CONFLICT (audit_id) DO UPDATE
                            SET audit_id=EXCLUDED.audit_id
                          RETURNING audit_id"""
                    ),
                    audit_params,
                ).scalar_one_or_none()
                if inserted is None:
                    already_committed = conn.execute(
                        text(
                            """SELECT audit_id
                               FROM governance.code_analysis_audits
                               WHERE audit_id=:id AND job_id=:job"""
                        ),
                        parameters,
                    ).scalar_one_or_none()
                    if already_committed is not None:
                        return
                    raise LeaseLost("Lease was lost before the terminal audit")
                released = conn.execute(
                    text(
                        """UPDATE governance.code_analysis_jobs
                           SET lease_token=NULL,lease_expires_at=NULL,
                               updated_at=now()
                           WHERE job_id=:job AND lease_token=:token
                             AND status=:status
                           RETURNING job_id"""
                    ),
                    parameters,
                ).scalar_one_or_none()
                if released is None:
                    raise LeaseLost("Lease was lost before release")
            return
        except OperationalError:
            if attempt == 2:
                logger.error(
                    "Terminal audit persistence unavailable after retries"
                )
                raise
            time.sleep(0.25 * (2**attempt))


def run_one() -> bool:
    job = _claim()
    if not job:
        return False
    started = time.monotonic()
    deadline_monotonic = started + max(1, get_settings().code_analysis_job_deadline_seconds)
    status = "failed"
    error_code: str | None = None
    trace_id = f"trace_code_{job['job_id']}"
    candidate_hash = result_hash = None
    provider_name = ""
    generation_model = ""
    provider_version = ""
    provider_request_id = ""
    candidate_attempts = 0
    provider_context: ProviderContext | None = None
    failure_phase = "initialization"
    error_phase: str | None = None
    error_http_status: int | None = None
    error_detail: str | None = None
    error_diagnostics: dict[str, object] | None = None
    runner_digest = get_settings().code_analysis_runner_image
    node_history: list[dict] = []
    result: dict | None = None
    terminal_owned = False
    try:
        with _LeaseHeartbeat(job, deadline_monotonic=deadline_monotonic) as heartbeat:
            heartbeat.checkpoint(verify=True)
            store = get_artifact_store()
            failure_phase = "input_integrity"
            error_code = "input_integrity_failed"
            source = _load_verified_source(job, store)
            node_history.append(
                {"node": "input_integrity", "summary": "SHA-256 verified"}
            )
            error_code = None
            heartbeat.checkpoint(verify=True)
            plan = job["plan_data"]
            contract = job["file_contract_data"]
            failure_phase = "plan_integrity"
            error_code = "plan_integrity_failed"
            try:
                _verify_plan_integrity(job)
            except PlanIntegrityError:
                node_history.append(
                    {
                        "node": "plan_integrity",
                        "summary": "persisted plan/file contract mismatch",
                    }
                )
                raise
            node_history.append(
                {"node": "plan_integrity", "summary": "fingerprints verified"}
            )
            runner_digest = _runner_image_digest(
                str(_effective_limits(job)["runner_image"])
            )
            error_code = None
            provider = get_provider()
            previous_code: str | None = None
            feedback: dict | None = None
            logs = b""
            generated_artifacts: list[dict] = []
            repair_attempts_used = 0
            security_repair_attempts_used = 0
            seen_repair_failures: set[str] = set()
            job_prefix = (
                f"{owner_artifact_prefix(job['tenant_id'], job['user_id'])}/"
                f"jobs/{job['job_id']}/attempt-{job['lease_attempt']}"
            )
            max_repair_attempts = max(
                0, int(get_settings().code_analysis_max_repair_attempts)
            )
            max_security_repair_attempts = max(
                0, int(get_settings().code_analysis_max_security_repair_attempts)
            )
            max_candidates = 1 + max_repair_attempts
            for candidate_attempt in range(1, max_candidates + 1):
                heartbeat.checkpoint(verify=True)
                failure_phase = "provider_generation"
                provider_context = ProviderContext(
                    job_id=str(job["job_id"]),
                    tenant_id=str(job["tenant_id"]),
                    actor_id=str(job["user_id"]),
                    trace_id=trace_id,
                    plan_hash=str(job["plan_hash"]),
                    data_contract_hash=_digest_json(contract),
                    deadline_monotonic=deadline_monotonic,
                )
                candidate = provider.propose_analysis(
                    plan,
                    contract,
                    context=provider_context,
                    previous_code=previous_code,
                    failure_feedback=feedback,
                )
                heartbeat.checkpoint(verify=True)
                failure_phase = "candidate_policy"
                provider_name = candidate.provider
                generation_model = candidate.model
                provider_version = candidate.provider_version
                provider_request_id = candidate.provider_request_id
                candidate_attempts = candidate_attempt
                candidate_hash = _digest(candidate.code)
                node_history.append(
                    {
                        "node": (
                            "code_agent_generate"
                            if candidate_attempt == 1
                            else "code_agent_repair"
                        ),
                        "summary": (
                            f"{provider_name}, candidate "
                            f"{candidate_attempt}/{max_candidates}, request="
                            f"{provider_request_id or 'none'}"
                        ),
                    }
                )
                code_policy = validate_candidate_code(candidate.code)
                requirement_policy = validate_candidate_requirements(
                    candidate.requirements
                )
                policy_errors = sorted(
                    set(code_policy.errors + requirement_policy.errors)
                )
                policy_valid = code_policy.valid and requirement_policy.valid
                security_review = None
                if policy_valid and get_settings().code_analysis_security_review_enabled:
                    failure_phase = "security_review"
                    security_review = provider.review_analysis(
                        plan,
                        contract,
                        candidate,
                        context=provider_context,
                    )
                    failure_phase = "candidate_policy"
                security_review_valid = (
                    security_review is None
                    or (
                        security_review.decision == "allow"
                        and security_review.risk_level != "high"
                    )
                )
                validation_valid = policy_valid and security_review_valid
                with engine.begin() as conn:
                    _artifact(
                        conn,
                        job,
                        "candidate_code",
                        (
                            f"{job_prefix}/candidate-{candidate_attempt}/analysis.py"
                        ),
                        candidate.code.encode(),
                        {
                            "provider": candidate.provider,
                            "candidate_attempt": candidate_attempt,
                        },
                    )
                    _artifact(
                        conn,
                        job,
                        "validation",
                        (
                            f"{job_prefix}/candidate-{candidate_attempt}/"
                            "validation.json"
                        ),
                        json.dumps(
                            {
                                "valid": validation_valid,
                                "errors": policy_errors,
                                "requirements": candidate.requirements,
                                "security_review": (
                                    security_review.as_dict()
                                    if security_review is not None
                                    else None
                                ),
                            },
                            ensure_ascii=False,
                        ).encode(),
                        {"candidate_attempt": candidate_attempt},
                    )
                    _fenced_job_update(
                        conn,
                        job,
                        """UPDATE governance.code_analysis_jobs
                           SET status='validating',updated_at=now()
                           WHERE job_id=:id AND lease_token=:token
                             AND lease_expires_at > clock_timestamp()
                             AND status IN ('generating','validating','running')
                           RETURNING job_id""",
                    )
                if not policy_valid:
                    error_code = "candidate_policy_rejected"
                    node_history.append(
                        {"node": "candidate_policy", "summary": "rejected"}
                    )
                    fingerprint = _repair_fingerprint(
                        "candidate_policy_rejected", "; ".join(policy_errors)
                    )
                    can_repair = (
                        candidate.provider != "fake"
                        and candidate_attempt < max_candidates
                        and repair_attempts_used < max_repair_attempts
                        and fingerprint not in seen_repair_failures
                    )
                    if can_repair:
                        seen_repair_failures.add(fingerprint)
                        repair_attempts_used += 1
                        previous_code = candidate.code
                        feedback = _sanitized_feedback(
                            error_code,
                            "; ".join(policy_errors),
                        )
                        with engine.begin() as conn:
                            _fenced_job_update(
                                conn,
                                job,
                                """UPDATE governance.code_analysis_jobs
                                   SET status='generating',updated_at=now()
                                   WHERE job_id=:id AND lease_token=:token
                                     AND lease_expires_at > clock_timestamp()
                                     AND status='validating'
                                   RETURNING job_id""",
                            )
                        continue
                    raise RuntimeError(
                        "Candidate code was rejected: " + "; ".join(policy_errors)
                    )
                node_history.append(
                    {"node": "candidate_policy", "summary": "accepted"}
                )
                if security_review is not None:
                    review_summary = (
                        f"{security_review.decision}/{security_review.risk_level}: "
                        f"{security_review.summary}"
                    )
                    node_history.append(
                        {
                            "node": "security_review",
                            "summary": review_summary[:1_200],
                        }
                    )
                    if (
                        security_review.decision != "allow"
                        or security_review.risk_level == "high"
                    ):
                        error_code = "candidate_security_review_rejected"
                        review_feedback = "; ".join(
                            security_review.findings
                        ) or security_review.summary
                        fingerprint = _repair_fingerprint(
                            error_code,
                            f"{security_review.decision}:{security_review.risk_level}:{review_feedback}",
                        )
                        can_repair = (
                            candidate.provider != "fake"
                            and candidate_attempt < max_candidates
                            and security_repair_attempts_used
                            < max_security_repair_attempts
                            and fingerprint not in seen_repair_failures
                        )
                        if can_repair:
                            seen_repair_failures.add(fingerprint)
                            security_repair_attempts_used += 1
                            repair_attempts_used += 1
                            previous_code = candidate.code
                            feedback = _sanitized_feedback(
                                error_code,
                                f"risk={security_review.risk_level}; {review_feedback}",
                            )
                            with engine.begin() as conn:
                                _fenced_job_update(
                                    conn,
                                    job,
                                    """UPDATE governance.code_analysis_jobs
                                       SET status='generating',updated_at=now()
                                       WHERE job_id=:id AND lease_token=:token
                                         AND lease_expires_at > clock_timestamp()
                                         AND status='validating'
                                       RETURNING job_id""",
                                )
                            continue
                        raise RuntimeError(
                            "Candidate failed AI security review: " + review_feedback
                        )
                heartbeat.checkpoint(verify=True)
                failure_phase = "sandbox_execution"
                with engine.begin() as conn:
                    _assert_owned_lease(conn, job)
                    _fenced_job_update(
                        conn,
                        job,
                        """UPDATE governance.code_analysis_jobs
                           SET status='running',
                               started_at=COALESCE(started_at,now()),
                               updated_at=now()
                           WHERE job_id=:id AND lease_token=:token
                             AND lease_expires_at > clock_timestamp()
                             AND status IN ('generating','validating')
                             AND NOT cancel_requested
                           RETURNING job_id""",
                    )
                try:
                    result, logs, generated_artifacts = _run_in_sandbox(
                        source,
                        job["file_type"],
                        candidate.code,
                        job,
                    )
                    completeness = validate_result_completeness(
                        str(plan.get("question") or ""),
                        result,
                        plan.get("expected_output"),
                    )
                    if not completeness.valid:
                        error_code = "result_completeness_failed"
                        missing = ", ".join(completeness.missing)
                        failure_phase = "result_completeness"
                        node_history.append(
                            {
                                "node": "result_completeness",
                                "summary": f"missing: {missing}"[:1_200],
                            }
                        )
                        fingerprint = _repair_fingerprint(error_code, missing)
                        can_repair = (
                            candidate.provider != "fake"
                            and candidate_attempt < max_candidates
                            and repair_attempts_used < max_repair_attempts
                            and fingerprint not in seen_repair_failures
                        )
                        if can_repair:
                            seen_repair_failures.add(fingerprint)
                            repair_attempts_used += 1
                            previous_code = candidate.code
                            feedback = _sanitized_feedback(
                                error_code,
                                f"The sandbox result is missing required question coverage: {missing}. "
                                "Recompute and expose every requested metric in the result.",
                            )
                            with engine.begin() as conn:
                                _fenced_job_update(
                                    conn,
                                    job,
                                    """UPDATE governance.code_analysis_jobs
                                       SET status='generating',updated_at=now()
                                       WHERE job_id=:id AND lease_token=:token
                                         AND lease_expires_at > clock_timestamp()
                                         AND status='running'
                                       RETURNING job_id""",
                                )
                            continue
                        raise ResultCompletenessError(
                            "Sandbox result did not cover the confirmed question: "
                            + missing
                        )
                    node_history.append(
                        {
                            "node": "sandbox_execute",
                            "summary": "completed with network disabled",
                        }
                    )
                    break
                except SandboxExecutionError as exc:
                    logs = exc.logs
                    error_code = "sandbox_execution_failed"
                    failure_class = _classify_sandbox_failure(exc)
                    node_history.append(
                        {
                            "node": "sandbox_execute",
                            "summary": f"failed safely ({failure_class})",
                        }
                    )
                    with engine.begin() as conn:
                        _artifact(
                            conn,
                            job,
                            "runner_log",
                            (
                                f"{job_prefix}/candidate-{candidate_attempt}/"
                                "runner.log"
                            ),
                            logs,
                            {
                                "candidate_attempt": candidate_attempt,
                                "failed": True,
                            },
                        )
                    fingerprint = _repair_fingerprint(
                        failure_class, str(exc)
                    )
                    can_repair = (
                        failure_class in {"candidate_bug", "result_contract"}
                        and candidate.provider != "fake"
                        and candidate_attempt < max_candidates
                        and repair_attempts_used < max_repair_attempts
                        and fingerprint not in seen_repair_failures
                    )
                    if can_repair:
                        seen_repair_failures.add(fingerprint)
                        repair_attempts_used += 1
                        previous_code = candidate.code
                        # Raw runner logs can contain uploaded values. Only the
                        # bounded orchestrator error is returned to the model.
                        feedback = _sanitized_feedback(error_code, str(exc))
                        with engine.begin() as conn:
                            _fenced_job_update(
                                conn,
                                job,
                                """UPDATE governance.code_analysis_jobs
                                   SET status='generating',updated_at=now()
                                   WHERE job_id=:id AND lease_token=:token
                                     AND lease_expires_at > clock_timestamp()
                                     AND status='running'
                                   RETURNING job_id""",
                            )
                        continue
                    raise
            if result is None:
                raise RuntimeError("No candidate completed successfully")
            heartbeat.checkpoint(verify=True)
            # Candidate code cannot attest to its own provenance. Overwrite any
            # model-supplied value with worker-observed, digest-only metadata.
            result["reproducibility"] = {
                "input_sha256": str(job["file_sha256"]),
                "plan_sha256": str(job["plan_hash"]),
                "candidate_code_sha256": candidate_hash,
                "runner_image_digest": runner_digest,
                "provider": provider_name,
                "model": generation_model,
                "provider_version": provider_version,
            }
            error_code = "result_too_large"
            _encode_bounded_result(
                result,
                max_bytes=int(_effective_limits(job)["max_result_bytes"]),
            )
            error_code = None
            with engine.begin() as conn:
                _assert_owned_lease(conn, job)
                chart_metadata: list[dict] = []
                for chart in generated_artifacts:
                    payload = base64.b64decode(
                        str(chart["data_base64"]),
                        validate=True,
                    )
                    if (
                        chart.get("content_type") != "image/png"
                        or not payload.startswith(b"\x89PNG\r\n\x1a\n")
                    ):
                        error_code = "artifact_policy_rejected"
                        raise SandboxExecutionError(
                            "Sandbox returned a non-PNG chart artifact"
                        )
                    safe_name = re.sub(
                        r"[^A-Za-z0-9_.-]",
                        "-",
                        str(chart.get("name") or "chart.png"),
                    )[:120]
                    if not safe_name.lower().endswith(".png"):
                        safe_name = f"{safe_name}.png"
                    saved = _artifact(
                        conn,
                        job,
                        "chart",
                        (
                            f"{job_prefix}/charts/{candidate_hash[:12]}-"
                            f"{int(chart['index'])}-{safe_name}"
                        ),
                        payload,
                        {
                            "content_type": "image/png",
                            "name": safe_name,
                        },
                    )
                    chart_metadata.append(
                        {
                            "artifact_id": saved["artifact_id"],
                            "name": safe_name,
                            "content_type": "image/png",
                        }
                    )
                result["charts"] = chart_metadata
                error_code = "result_too_large"
                result_blob = _encode_bounded_result(
                    result,
                    max_bytes=int(_effective_limits(job)["max_result_bytes"]),
                )
                error_code = None
                result_hash = _digest(result_blob)
                _artifact(
                    conn,
                    job,
                    "result",
                    (
                        f"{job_prefix}/result.json"
                    ),
                    result_blob,
                )
                _artifact(
                    conn,
                    job,
                    "runner_log",
                    (
                        f"{job_prefix}/runner.log"
                    ),
                    logs,
                )
                status = _fenced_job_update(
                    conn,
                    job,
                    """UPDATE governance.code_analysis_jobs
                       SET status='completed',
                           result_data=CAST(:result AS jsonb),
                           completed_at=now(),updated_at=now(),error_code=NULL
                       WHERE job_id=:id AND lease_token=:token
                         AND lease_expires_at > clock_timestamp()
                         AND status='running' AND NOT cancel_requested
                       RETURNING status""",
                    {"result": result_blob.decode()},
                )
            # A repaired candidate may have succeeded after an earlier failed
            # attempt; terminal success should not retain that stale diagnostic.
            error_phase = None
            error_http_status = None
            error_detail = None
            error_diagnostics = None
            terminal_owned = True
            try:
                write_trace(
                    trace_id,
                    {
                        "trace_id": trace_id,
                        "tenant_id": job["tenant_id"],
                        "user_id": job["user_id"],
                        "trace_kind": "code_analysis",
                        "code_analysis_job_id": job["job_id"],
                        "node_history": node_history,
                        "result": result,
                    },
                )
            except Exception:
                # Trace persistence is observability, not a reason to rewrite a
                # successfully fenced terminal result.
                pass
    except LeaseLost:
        status = "lease_lost"
        error_code = "lease_lost"
        terminal_owned = False
    except SandboxCancelled:
        try:
            with engine.begin() as conn:
                status = _mark_cancelled(conn, job)
            error_code = None
            terminal_owned = True
        except LeaseLost:
            status = "lease_lost"
            error_code = "lease_lost"
            terminal_owned = False
    except JobDeadlineExceeded as exc:
        diagnostic = _provider_error_diagnostic(
            exc,
            phase=failure_phase,
            provider_request_id=provider_request_id
            or (provider_context.provider_request_id if provider_context else ""),
        )
        error_phase = str(diagnostic["phase"])
        error_http_status = diagnostic["http_status"]
        error_detail = str(diagnostic["detail"])
        error_diagnostics = diagnostic["diagnostics"]
        try:
            with engine.begin() as conn:
                status = _mark_failed_or_cancelled(conn, job, "job_deadline_exceeded")
            error_code = None if status == "cancelled" else "job_deadline_exceeded"
            terminal_owned = True
        except LeaseLost:
            status = "lease_lost"
            error_code = "lease_lost"
            terminal_owned = False
    except Exception as exc:
        provider_request_id = provider_request_id or (
            provider_context.provider_request_id if provider_context else ""
        )
        diagnostic = _provider_error_diagnostic(
            exc,
            phase=failure_phase,
            provider_request_id=provider_request_id,
        )
        error_phase = str(diagnostic["phase"])
        error_http_status = diagnostic["http_status"]
        error_detail = str(diagnostic["detail"])
        error_diagnostics = diagnostic["diagnostics"]
        try:
            with engine.begin() as conn:
                deadline_error = (
                    "job_deadline_exceeded"
                    if time.monotonic() >= deadline_monotonic
                    else (error_code or "execution_failed")
                )
                status = _mark_failed_or_cancelled(
                    conn,
                    job,
                    deadline_error,
                )
            if status == "cancelled":
                error_code = None
            else:
                error_code = deadline_error
            terminal_owned = True
        except LeaseLost:
            status = "lease_lost"
            error_code = "lease_lost"
            terminal_owned = False
    finally:
        if terminal_owned:
            duration = int((time.monotonic() - started) * 1000)
            # Code Agent uses the worker audit/trace boundary rather than an
            # API response.  Emit the same shared AgentRun protocol here so
            # successful, failed, and cancelled jobs are comparable with the
            # other governed agents.
            run_status = "finished" if status == "completed" else "failed"
            run_initiator = "agent" if status == "completed" else "policy"
            agent_run = make_agent_run(
                run_id=trace_id,
                domain="code",
                decision_driver="workflow_fallback" if provider_name == "fake" else "llm",
                status=run_status,
                termination_initiator=run_initiator,
                termination_reason=error_code or status,
                rounds=candidate_attempts,
                tool_calls=1 if result is not None else 0,
                plan=job.get("plan_data") or {},
                trace=node_history,
            ).model_dump(mode="json")
            try:
                write_trace(
                    trace_id,
                    {
                        "trace_id": trace_id,
                        "tenant_id": job["tenant_id"],
                        "user_id": job["user_id"],
                        "trace_kind": "code_analysis",
                        "code_analysis_job_id": job["job_id"],
                        "node_history": node_history,
                        "agent_run": agent_run,
                        "result": result,
                    },
                )
            except Exception:
                pass
            try:
                _write_audit_and_release(
                    job,
                    status=status,
                    provider_name=provider_name,
                    generation_model=generation_model,
                    provider_version=provider_version,
                    provider_request_id=provider_request_id,
                    candidate_attempts=candidate_attempts,
                    candidate_hash=candidate_hash,
                    runner_digest=runner_digest,
                    result_hash=result_hash,
                    trace_id=trace_id,
                    duration=duration,
                    error_code=error_code,
                    error_phase=error_phase,
                    error_http_status=error_http_status,
                    error_detail=error_detail,
                    error_diagnostics=error_diagnostics,
                )
            except LeaseLost:
                # The winning owner (or retention cleanup) now controls the
                # row. A stale worker must not emit a competing audit.
                pass
            except OperationalError:
                # Retries are exhausted. Keep the terminal lease token in the
                # row as an explicit reconciliation marker; do not terminate
                # the worker process or pretend the audit was persisted.
                pass
    return True


def main() -> None:  # pragma: no cover - deployment loop
    last_cleanup = 0.0
    last_scavenge = 0.0
    while True:
        try:
            if time.monotonic() - last_scavenge > 300:
                _scavenge_orphaned_runners()
                last_scavenge = time.monotonic()
            if time.monotonic() - last_cleanup > 3600:
                cleanup_expired()
                last_cleanup = time.monotonic()
            if not run_one():
                time.sleep(1)
        except SQLAlchemyError as exc:
            if not isinstance(exc, (OperationalError, ProgrammingError)):
                raise
            # Undefined tables during rollout and transient database outages
            # are operational. Log only the exception type (no SQL/DSN) and
            # keep the container alive for the migration/service to recover.
            logger.warning(
                "Code analysis database unavailable (%s); retrying in 5s",
                type(exc).__name__,
            )
            time.sleep(5)


if __name__ == "__main__":
    main()

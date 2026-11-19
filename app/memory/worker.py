"""Lease worker for opt-in LLM memory consolidation.

Run as ``python -m app.memory.worker`` using a database role with BYPASSRLS in
production.  The HTTP API remains RLS-bound; this worker re-establishes the
job's tenant context before writing the pending proposal.
"""
from __future__ import annotations

import hashlib
import logging
import time

from sqlalchemy import create_engine, text

from app.core.config import get_settings
from app.memory.consolidation import ConsolidationUnavailable, propose_memory_consolidation
from app.memory.repository import MemoryRepository
from app.tenancy import TenantContext

logger = logging.getLogger(__name__)


def _engine():
    settings = get_settings()
    return create_engine(
        settings.memory_consolidation_database_url or settings.postgres_url,
        future=True, pool_pre_ping=True, connect_args={"connect_timeout": int(settings.database_timeout_seconds)},
    )


def _claim() -> dict | None:
    settings = get_settings()
    engine = _engine()
    with engine.begin() as conn:
        conn.execute(text("""UPDATE governance.memory_consolidation_jobs
            SET status='queued', lease_token=NULL, lease_expires_at=NULL, updated_at=now()
            WHERE status='generating' AND lease_expires_at < clock_timestamp()
              AND attempts < :max_attempts"""), {"max_attempts": settings.memory_consolidation_max_attempts})
        conn.execute(text("""UPDATE governance.memory_consolidation_jobs
            SET status='failed', error_code='worker_retry_exhausted', updated_at=now(), completed_at=now()
            WHERE status='generating' AND lease_expires_at < clock_timestamp()
              AND attempts >= :max_attempts"""), {"max_attempts": settings.memory_consolidation_max_attempts})
        row = conn.execute(text("""SELECT * FROM governance.memory_consolidation_jobs
            WHERE status='queued' AND attempts < :max_attempts
            ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1"""), {
                "max_attempts": settings.memory_consolidation_max_attempts,
            }).mappings().one_or_none()
        if not row:
            return None
        job = dict(row)
        token = hashlib.sha256(f"{job['job_id']}:{time.time_ns()}".encode()).hexdigest()[:48]
        conn.execute(text("""UPDATE governance.memory_consolidation_jobs
            SET status='generating', attempts=attempts+1, lease_token=:token,
                lease_expires_at=now()+(:seconds || ' seconds')::interval, updated_at=now()
            WHERE job_id=:id"""), {
                "id": job["job_id"], "token": token, "seconds": settings.memory_consolidation_job_lease_seconds,
            })
        job["lease_token"] = token
        return job


def _finish(job: dict, *, status: str, proposed_memory_id: str | None = None, error_code: str | None = None) -> None:
    engine = _engine()
    with engine.begin() as conn:
        updated = conn.execute(text("""UPDATE governance.memory_consolidation_jobs
            SET status=:status, proposed_memory_id=:memory_id, error_code=:error, completed_at=now(),
                lease_token=NULL, lease_expires_at=NULL, updated_at=now()
            WHERE job_id=:id AND lease_token=:token AND status='generating'
            RETURNING job_id"""), {
                "id": job["job_id"], "token": job["lease_token"], "status": status,
                "memory_id": proposed_memory_id, "error": error_code,
            }).scalar_one_or_none()
        if not updated:
            raise RuntimeError("memory consolidation lease was lost")


def run_one() -> bool:
    job = _claim()
    if not job:
        return False
    payload = job["input_payload"]
    context = TenantContext(
        tenant_id=str(job["tenant_id"]), user_id=str(job["user_id"]), role="admin", scope_type="platform",
    )
    try:
        item = propose_memory_consolidation(
            MemoryRepository(context), str(job["scope_fingerprint"]), str(job["actor"]),
            sources=list(payload.get("sources") or []),
            consolidation_job_id=str(job["job_id"]), consolidation_lease_token=str(job["lease_token"]),
        )
        MemoryRepository(context).record_memory_decision(
            action="proposal_completed", actor=str(job["actor"]), memory_id=str(item["memory_id"]),
            reason="durable LLM consolidation proposal completed",
            source_memory_ids=[str(item) for item in payload.get("source_memory_ids", [])],
        )
        _finish(job, status="completed", proposed_memory_id=str(item["memory_id"]))
    except ConsolidationUnavailable as exc:
        _finish(job, status="failed", error_code="proposal_unavailable")
        logger.info("memory consolidation job %s unavailable: %s", job["job_id"], exc)
    except Exception:
        logger.exception("memory consolidation job %s failed", job["job_id"])
        _finish(job, status="failed", error_code="worker_error")
    return True


def main() -> None:  # pragma: no cover - deployment loop
    while True:
        try:
            if not run_one():
                time.sleep(2)
        except Exception:
            logger.exception("memory consolidation worker iteration failed")
            time.sleep(5)


if __name__ == "__main__":
    main()

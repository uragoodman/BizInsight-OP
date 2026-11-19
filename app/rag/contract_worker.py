"""Lease worker for slow RAG release contracts; never runs in an HTTP request."""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import text

from app.data.database import engine
from app.quality.evaluations import finalize_rag_contract
from app.rag.release_controller import block_release_for_quality_gate
from evaluation.retrieval.contracts import run_contracts


def process_once() -> bool:
    token = str(uuid4())
    with engine.begin() as conn:
        job = conn.execute(text("""SELECT * FROM governance.rag_contract_validation_jobs
          WHERE (status='queued' OR (status='running' AND lease_expires_at < now())) AND attempts < 3
          ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1""")).mappings().first()
        if not job: return False
        conn.execute(text("UPDATE governance.rag_contract_validation_jobs SET status='running',attempts=attempts+1,lease_token=:token,lease_expires_at=:expires WHERE job_id=:id"), {"id": job["job_id"], "token": token, "expires": datetime.now(timezone.utc) + timedelta(minutes=3)})
    contracts = run_contracts()
    try:
        run = finalize_rag_contract(job["evaluation_run_id"], contracts)
        if run["gate_status"] == "block": block_release_for_quality_gate(job["release_id"], "rag-contract-worker", list(run["gate_reasons"]))
        with engine.begin() as conn:
            conn.execute(text("UPDATE governance.rag_contract_validation_jobs SET status='completed',completed_at=now(),error_code=:error WHERE job_id=:id AND lease_token=:token"), {"id": job["job_id"], "token": token, "error": contracts.get("error_code")})
    except Exception as exc:
        with engine.begin() as conn:
            conn.execute(text("UPDATE governance.rag_contract_validation_jobs SET status=CASE WHEN attempts >= 3 THEN 'failed' ELSE 'queued' END,error_code=:error,lease_token=NULL,lease_expires_at=NULL,completed_at=CASE WHEN attempts >= 3 THEN now() ELSE NULL END WHERE job_id=:id AND lease_token=:token"), {"id": job["job_id"], "token": token, "error": type(exc).__name__})
    return True


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--once", action="store_true"); args = parser.parse_args()
    while True:
        if args.once:
            process_once(); return
        if not process_once(): time.sleep(1)


if __name__ == "__main__": main()

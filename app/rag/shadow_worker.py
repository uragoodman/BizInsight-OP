"""Lease-based worker for non-blocking stable/candidate retrieval comparisons."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import text

from app.data.database import engine
from app.rag.release_controller import evaluate_online_safety, get_release
from app.rag.retriever import retrieve_for_release
from app.tenancy import TenantContext


def _claim() -> dict | None:
    token = str(uuid4())
    with engine.begin() as conn:
        row = conn.execute(text("""SELECT * FROM governance.rag_shadow_jobs
          WHERE (status='queued' OR (status='running' AND lease_expires_at < now()))
            AND expires_at > now() AND attempts < 3
          ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1""")).mappings().first()
        if not row:
            return None
        conn.execute(text("""UPDATE governance.rag_shadow_jobs SET status='running',attempts=attempts+1,lease_token=:token,
          lease_expires_at=:expires WHERE job_id=:id"""), {"token": token, "expires": datetime.now(timezone.utc) + timedelta(seconds=60), "id": row["job_id"]})
        result = dict(row); result["lease_token"] = token
        return result


def _summary(result) -> dict:
    return {"release_id": (result.filters or {}).get("release_id"), "strategy": result.strategy,
            "latency_ms": None, "empty": not result.hits,
            "source_ids": [hit.citation.source_path for hit in result.hits],
            "citation_count": len(result.citations)}


def process_once() -> bool:
    job = _claim()
    if not job:
        return False
    context = TenantContext(tenant_id=job["tenant_id"], user_id=job["user_id"], role="admin", scope_type="platform")
    request = dict(job["request_context"] or {})
    try:
        stable, candidate = get_release(job["stable_release_id"]), get_release(job["candidate_release_id"])
        if not stable or not candidate:
            raise ValueError("release missing")
        started = time.perf_counter()
        stable_result = retrieve_for_release(job["query_payload"], stable, doc_types=set(request["doc_types"]) if request.get("doc_types") else None, metric_name=request.get("metric_name"), top_k=request.get("top_k"), tenant_context=context)
        stable_summary = _summary(stable_result); stable_summary["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        started = time.perf_counter()
        candidate_result = retrieve_for_release(job["query_payload"], candidate, doc_types=set(request["doc_types"]) if request.get("doc_types") else None, metric_name=request.get("metric_name"), top_k=request.get("top_k"), tenant_context=context)
        candidate_summary = _summary(candidate_result); candidate_summary["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        overlap = len(set(stable_summary["source_ids"]) & set(candidate_summary["source_ids"]))
        comparison = {"top_k_overlap": overlap, "candidate_empty": candidate_summary["empty"], "candidate_citationless": candidate_summary["citation_count"] == 0,
                      "latency_ratio": candidate_summary["latency_ms"] / max(stable_summary["latency_ms"], 1.0)}
        with engine.begin() as conn:
            conn.execute(text("""INSERT INTO governance.rag_shadow_comparisons(comparison_id,job_id,tenant_id,user_id,stable_release_id,candidate_release_id,mode,stable_result,candidate_result,comparison)
              VALUES(:id,:job,:tenant,:user,:stable,:candidate,:mode,CAST(:stable_result AS jsonb),CAST(:candidate_result AS jsonb),CAST(:comparison AS jsonb))"""), {"id": f"rcmp_{uuid4().hex}", "job": job["job_id"], "tenant": job["tenant_id"], "user": job["user_id"], "stable": job["stable_release_id"], "candidate": job["candidate_release_id"], "mode": job["mode"], "stable_result": json.dumps(stable_summary), "candidate_result": json.dumps(candidate_summary), "comparison": json.dumps(comparison)})
            conn.execute(text("UPDATE governance.rag_shadow_jobs SET status='completed',query_payload='',completed_at=now() WHERE job_id=:id AND lease_token=:token"), {"id": job["job_id"], "token": job["lease_token"]})
        evaluate_online_safety(job["candidate_release_id"])
    except Exception as exc:
        with engine.begin() as conn:
            # Persist the failure as a comparison so online gates can measure
            # it, then retry through the same fenced lease path up to 3 times.
            conn.execute(text("""INSERT INTO governance.rag_shadow_comparisons(comparison_id,job_id,tenant_id,user_id,stable_release_id,candidate_release_id,mode,stable_result,candidate_result,comparison)
              VALUES(:id,:job,:tenant,:user,:stable,:candidate,:mode,'{}'::jsonb,CAST(:candidate_result AS jsonb),'{}'::jsonb)
              ON CONFLICT(job_id) DO NOTHING"""), {"id": f"rcmp_{uuid4().hex}", "job": job["job_id"], "tenant": job["tenant_id"], "user": job["user_id"], "stable": job["stable_release_id"], "candidate": job["candidate_release_id"], "mode": job["mode"], "candidate_result": json.dumps({"error": type(exc).__name__})})
            conn.execute(text("""UPDATE governance.rag_shadow_jobs
              SET status=CASE WHEN attempts >= 3 THEN 'failed' ELSE 'queued' END,
                  completed_at=CASE WHEN attempts >= 3 THEN now() ELSE NULL END,
                  lease_token=NULL,lease_expires_at=NULL
              WHERE job_id=:id AND lease_token=:token"""), {"id": job["job_id"], "token": job["lease_token"]})
        evaluate_online_safety(job["candidate_release_id"])
    return True


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--once", action="store_true"); args = parser.parse_args()
    # A worker is intentionally long-lived in Compose.  ``--once`` is useful
    # for a scheduler and for deterministic tests, while the normal process
    # waits for the next leased job instead of exiting when the queue is empty.
    while True:
        processed = process_once()
        if args.once:
            return
        if not processed:
            time.sleep(1)


if __name__ == "__main__":
    main()

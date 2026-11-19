"""Unified quality envelopes; domain tables remain the metric source of truth."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

from sqlalchemy import text

from app.data.database import engine
from app.quality.service import _git_sha

_SOURCES = {"rag_release_evaluation", "rag_shadow_summary", "data_replay_audit", "memory_decision", "benchmark_file", "golden_artifact"}
_SUBJECTS = {"rag_release", "rag_retrieval", "memory_policy", "context_packer", "data_versioning", "benchmark", "agent_evaluation"}
_RAG_THRESHOLDS = {"mrr_drop": 0.01, "p95_ratio": 1.20}


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def create_run(subject_type: str, subject_id: str, actor: str, *, baseline_ref: dict | None = None, candidate_ref: dict | None = None, dataset_ref: str | None = None, dataset_hash: str | None = None, config_snapshot: dict | None = None, runner_version: str = "quality-v1", run_id: str | None = None) -> dict:
    if subject_type not in _SUBJECTS:
        raise ValueError("Unsupported evaluation subject")
    identity = _hash({"subject": subject_type, "id": subject_id, "baseline": baseline_ref or {}, "candidate": candidate_ref or {}, "dataset": dataset_hash, "config": config_snapshot or {}, "commit": _git_sha(), "runner": runner_version})
    run_id = run_id or f"eval_{identity[:32]}"
    with engine.begin() as conn:
        existing = conn.execute(text("SELECT * FROM governance.evaluation_runs WHERE run_id=:id"), {"id": run_id}).mappings().first()
        if existing:
            return dict(existing)
        conn.execute(text("""INSERT INTO governance.evaluation_runs(run_id,subject_type,subject_id,baseline_ref,candidate_ref,dataset_ref,dataset_hash,commit_sha,config_snapshot,runner_version,status,gate_status,created_by)
          VALUES(:id,:subject,:subject_id,CAST(:baseline AS jsonb),CAST(:candidate AS jsonb),:dataset,:hash,:commit,CAST(:config AS jsonb),:runner,'running','not_applicable',:actor)"""), {"id": run_id, "subject": subject_type, "subject_id": subject_id, "baseline": json.dumps(baseline_ref or {}), "candidate": json.dumps(candidate_ref or {}), "dataset": dataset_ref, "hash": dataset_hash, "commit": _git_sha(), "config": json.dumps(config_snapshot or {}), "runner": runner_version, "actor": actor})
    return get_run(run_id) or {}


def add_evidence(run_id: str, source_type: str, source_id: str, role: str, summary: dict, content_hash: str | None = None) -> None:
    if source_type not in _SOURCES or role not in {"baseline", "candidate", "gate", "online", "benchmark"}:
        raise ValueError("Unsupported evaluation evidence")
    with engine.begin() as conn:
        conn.execute(text("""INSERT INTO governance.evaluation_run_evidence(evidence_id,run_id,source_type,source_id,role,summary,content_hash)
          VALUES(:id,:run,:type,:source,:role,CAST(:summary AS jsonb),:hash)
          ON CONFLICT(run_id,source_type,source_id,role) DO NOTHING"""), {"id": f"eev_{uuid4().hex}", "run": run_id, "type": source_type, "source": source_id, "role": role, "summary": json.dumps(summary), "hash": content_hash})


def add_case_results(run_id: str, rows: list[dict]) -> None:
    """Persist queryable case summaries for an Agent evaluation run."""
    if not rows:
        return
    with engine.begin() as conn:
        for row in rows:
            conn.execute(text("""INSERT INTO governance.evaluation_run_cases
              (run_id,arm,case_id,category,provider_called,trace_id,result)
              VALUES(:run,:arm,:case,:category,:provider,:trace,CAST(:result AS jsonb))
              ON CONFLICT(run_id,arm,case_id) DO UPDATE SET
                category=EXCLUDED.category, provider_called=EXCLUDED.provider_called,
                trace_id=EXCLUDED.trace_id, result=EXCLUDED.result"""), {
                "run": run_id,
                "arm": str(row.get("arm") or "agent"),
                "case": str(row.get("case_id") or row.get("id") or "unknown"),
                "category": str(row.get("category") or "unknown"),
                "provider": bool(row.get("provider_called", False)),
                "trace": row.get("trace_id"),
                "result": json.dumps(row.get("result") or {}, ensure_ascii=False, default=str),
            })


def complete_run(run_id: str, *, status: str, gate_status: str, reasons: list[str], metric_deltas: dict | None = None, thresholds: dict | None = None) -> dict:
    if status not in {"passed", "warned", "blocked", "unavailable", "failed"} or gate_status not in {"pass", "warn", "block", "not_applicable"}:
        raise ValueError("Invalid evaluation outcome")
    with engine.begin() as conn:
        conn.execute(text("UPDATE governance.evaluation_runs SET status=:status,gate_status=:gate,gate_reasons=CAST(:reasons AS jsonb),completed_at=now() WHERE run_id=:id"), {"id": run_id, "status": status, "gate": gate_status, "reasons": json.dumps(reasons)})
        if metric_deltas is not None:
            conn.execute(text("""INSERT INTO governance.evaluation_comparisons(comparison_id,run_id,metric_deltas,thresholds,verdict,reasons)
              VALUES(:id,:run,CAST(:deltas AS jsonb),CAST(:thresholds AS jsonb),:verdict,CAST(:reasons AS jsonb))
              ON CONFLICT(run_id) DO UPDATE SET metric_deltas=EXCLUDED.metric_deltas,thresholds=EXCLUDED.thresholds,verdict=EXCLUDED.verdict,reasons=EXCLUDED.reasons"""), {"id": f"ecmp_{uuid4().hex}", "run": run_id, "deltas": json.dumps(metric_deltas), "thresholds": json.dumps(thresholds or {}), "verdict": gate_status if gate_status != "not_applicable" else "warn", "reasons": json.dumps(reasons)})
    return get_run(run_id) or {}


def run_rag_gate(release_id: str, actor: str, candidate: dict, baseline: dict, *, evaluation_id: str, dataset_path: str, contracts: dict | None = None) -> dict:
    path = Path(dataset_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    run = create_run("rag_release", release_id, actor, baseline_ref={"release_id": baseline.get("release_id"), "metrics": baseline}, candidate_ref={"release_id": release_id, "metrics": candidate}, dataset_ref=str(path), dataset_hash=digest, config_snapshot={"thresholds": _RAG_THRESHOLDS})
    add_evidence(run["run_id"], "rag_release_evaluation", evaluation_id, "candidate", {"candidate": candidate, "baseline": baseline})
    add_evidence(run["run_id"], "golden_artifact", f"golden:{digest}", "gate", {"dataset": str(path)}, digest)
    if contracts is not None:
        add_evidence(run["run_id"], "golden_artifact", f"contracts:{contracts['output_hash']}", "gate", {key: value for key, value in contracts.items() if key != "output"}, contracts["output_hash"])
    reasons: list[str] = []
    for name in ("doc_hit@5", "chunk_hit@5"):
        if candidate.get(name) is None or baseline.get(name) is None or candidate[name] < baseline[name]: reasons.append(f"{name} regressed")
    if candidate.get("mrr@5") is None or baseline.get("mrr@5") is None or candidate["mrr@5"] < baseline["mrr@5"] - _RAG_THRESHOLDS["mrr_drop"]: reasons.append("mrr@5 regressed")
    if candidate.get("p95_ms") is None or baseline.get("p95_ms") is None or candidate["p95_ms"] > baseline["p95_ms"] * _RAG_THRESHOLDS["p95_ratio"]: reasons.append("p95 exceeded")
    if contracts and not contracts.get("passed"): reasons.append("RAG retrieval/RLS/citation contract failed")
    deltas = {key: (candidate.get(key) - baseline.get(key)) if isinstance(candidate.get(key), (int, float)) and isinstance(baseline.get(key), (int, float)) else None for key in ("doc_hit@5", "chunk_hit@5", "mrr@5", "p95_ms")}
    if contracts is None and not reasons:
        return complete_run(run["run_id"], status="warned", gate_status="warn", reasons=["RAG contract validation pending"], metric_deltas=deltas, thresholds=_RAG_THRESHOLDS)
    return complete_run(run["run_id"], status="blocked" if reasons else "passed", gate_status="block" if reasons else "pass", reasons=reasons, metric_deltas=deltas, thresholds=_RAG_THRESHOLDS)


def enqueue_rag_contract_validation(release_id: str, run_id: str) -> str:
    job_id = f"rcontract_{uuid4().hex}"
    with engine.begin() as conn:
        row = conn.execute(text("SELECT job_id FROM governance.rag_contract_validation_jobs WHERE evaluation_run_id=:run"), {"run": run_id}).scalar_one_or_none()
        if row: return str(row)
        conn.execute(text("INSERT INTO governance.rag_contract_validation_jobs(job_id,release_id,evaluation_run_id) VALUES(:id,:release,:run)"), {"id": job_id, "release": release_id, "run": run_id})
    return job_id


def finalize_rag_contract(run_id: str, contracts: dict) -> dict:
    run = get_run(run_id)
    if not run: raise ValueError("Evaluation run not found")
    candidate = dict(run["candidate_ref"]).get("metrics", {}); baseline = dict(run["baseline_ref"]).get("metrics", {})
    add_evidence(run_id, "golden_artifact", f"contracts:{contracts['output_hash']}", "gate", {key: value for key, value in contracts.items() if key != "output"}, contracts["output_hash"])
    reasons = []
    for name in ("doc_hit@5", "chunk_hit@5"):
        if candidate.get(name) is None or baseline.get(name) is None or candidate[name] < baseline[name]: reasons.append(f"{name} regressed")
    if candidate.get("mrr@5") is None or baseline.get("mrr@5") is None or candidate["mrr@5"] < baseline["mrr@5"] - _RAG_THRESHOLDS["mrr_drop"]: reasons.append("mrr@5 regressed")
    if candidate.get("p95_ms") is None or baseline.get("p95_ms") is None or candidate["p95_ms"] > baseline["p95_ms"] * _RAG_THRESHOLDS["p95_ratio"]: reasons.append("p95 exceeded")
    if not contracts.get("passed"): reasons.append("RAG retrieval/RLS/citation contract failed")
    deltas = {key: (candidate.get(key) - baseline.get(key)) if isinstance(candidate.get(key), (int, float)) and isinstance(baseline.get(key), (int, float)) else None for key in ("doc_hit@5", "chunk_hit@5", "mrr@5", "p95_ms")}
    return complete_run(run_id, status="blocked" if reasons else "passed", gate_status="block" if reasons else "pass", reasons=reasons, metric_deltas=deltas, thresholds=_RAG_THRESHOLDS)


def rag_gate_passed(release_id: str) -> bool:
    with engine.begin() as conn:
        return bool(conn.execute(text("SELECT 1 FROM governance.evaluation_runs WHERE subject_type='rag_release' AND subject_id=:id AND gate_status='pass' ORDER BY completed_at DESC LIMIT 1"), {"id": release_id}).scalar_one_or_none())


def get_run(run_id: str) -> dict | None:
    with engine.begin() as conn:
        run = conn.execute(text("SELECT * FROM governance.evaluation_runs WHERE run_id=:id"), {"id": run_id}).mappings().first()
        if not run: return None
        evidence = conn.execute(text("SELECT * FROM governance.evaluation_run_evidence WHERE run_id=:id ORDER BY created_at"), {"id": run_id}).mappings().all()
        cases = conn.execute(text("SELECT * FROM governance.evaluation_run_cases WHERE run_id=:id ORDER BY arm,case_id"), {"id": run_id}).mappings().all()
        comparison = conn.execute(text("SELECT * FROM governance.evaluation_comparisons WHERE run_id=:id"), {"id": run_id}).mappings().first()
    return {**dict(run), "evidence": [dict(item) for item in evidence], "cases": [dict(item) for item in cases], "comparison": dict(comparison) if comparison else None}


def list_runs(limit: int = 50) -> list[dict]:
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT * FROM governance.evaluation_runs ORDER BY created_at DESC LIMIT :limit"), {"limit": min(max(limit, 1), 100)}).mappings().all()
    return [dict(row) for row in rows]


def run_subject(subject_type: str, actor: str) -> dict:
    """Run only registered, versioned offline subjects; no caller-controlled paths."""
    from app.quality import subjects
    runners = {"rag_retrieval": subjects.run_rag_retrieval, "memory_policy": subjects.run_memory_policy, "context_packer": subjects.run_context_packer, "data_versioning": subjects.run_data_versioning, "benchmark": subjects.run_benchmark}
    if subject_type not in runners: raise ValueError("Subject must be run through its governed endpoint")
    metrics, path, digest, reasons = runners[subject_type]()
    run = create_run(subject_type, f"{subject_type}-v1", actor, baseline_ref=metrics.get("baseline", {}), candidate_ref=metrics.get("candidate", {}), dataset_ref=str(path) if path else None, dataset_hash=digest, config_snapshot={"runner": subject_type})
    if path: add_evidence(run["run_id"], "benchmark_file" if subject_type == "benchmark" else "golden_artifact", f"{subject_type}:{digest}", "benchmark" if subject_type == "benchmark" else "gate", metrics, digest)
    status = "unavailable" if metrics.get("status") == "unavailable" else ("warned" if reasons else "passed")
    return complete_run(run["run_id"], status=status, gate_status="not_applicable" if status == "unavailable" else ("warn" if reasons else "pass"), reasons=reasons, metric_deltas=metrics.get("deltas"), thresholds={"policy": "offline comparison; never a release gate"} if subject_type == "rag_retrieval" else {})

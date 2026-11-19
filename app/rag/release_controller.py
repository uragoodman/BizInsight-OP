"""Database-backed RAG progressive-delivery control plane.

The controller is deliberately the source of truth: flags decide *who* is in
a rollout, never which artifact is valid or may be promoted.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import text

from app.data.database import engine
from app.observability import record_event
from app.tenancy import TenantContext

_STAGES = {"off", "shadowing", "canary_5", "canary_25", "canary_100"}
_TRANSITIONS = {
    "draft": {"building", "failed"}, "building": {"evaluated", "failed"},
    "evaluated": {"shadowing", "failed"}, "shadowing": {"canary_5", "paused", "promotable"},
    "canary_5": {"canary_25", "paused"}, "canary_25": {"canary_100", "paused"},
    "canary_100": {"promotable", "rolled_back", "paused"}, "promotable": {"published", "paused"},
    "paused": {"shadowing", "rolled_back"}, "published": {"rolled_back"},
}
_DEFAULT_THRESHOLDS = {"p95_ratio": 1.2, "error_rate_delta": 0.01, "empty_rate_delta": 0.01, "citationless_rate_delta": 0.01, "negative_feedback_delta": 0.03}


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _validate_qdrant_release_manifest(spec: dict) -> None:
    """Require a readable, collection-bound manifest before DB release state."""

    reference = Path(str(spec.get("release_manifest_ref") or ""))
    if not reference.is_file():
        raise ValueError(f"Qdrant release manifest does not exist: {reference}")
    try:
        document = json.loads(reference.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Qdrant release manifest must be valid JSON") from exc
    payload = document.get("payload") if isinstance(document, dict) else None
    if not isinstance(payload, dict):
        raise ValueError("Qdrant release manifest is missing payload")
    if payload.get("collection") != spec.get("qdrant_collection"):
        raise ValueError("Qdrant release collection does not match release manifest")
    fingerprint = payload.get("index_fingerprint")
    if not isinstance(fingerprint, dict) or not fingerprint.get("digest"):
        raise ValueError("Qdrant release manifest is missing index_fingerprint")


def transition_allowed(current_status: str, target_stage: str) -> bool:
    """Pure state-machine guard, shared by API and regression tests."""
    return target_stage == "off" or target_stage in _TRANSITIONS.get(current_status, set())


def rollout_assignment(stable: dict | None, candidate: dict | None, context: TenantContext, session_id: str | None, trace_id: str | None) -> dict:
    """Resolve a deterministic, session-sticky rollout without I/O."""
    key = f"{context.tenant_id}:{context.user_id}:{session_id or trace_id or ''}"
    bucket = int(_hash(key)[:8], 16) % 100
    allowlisted = candidate and context.tenant_id in set(candidate.get("tenant_allowlist") or [])
    canary = bool(candidate and candidate["stage"].startswith("canary") and (allowlisted or bucket < int(candidate["canary_percent"])))
    shadow = bool(candidate and (candidate["stage"] == "shadowing" or canary) and (int(_hash(key + ':shadow')[:8], 16) / 0xFFFFFFFF <= float(candidate["shadow_sample_rate"])))
    return {"stable": stable, "candidate": candidate, "bucket": bucket, "canary": canary, "shadow": shadow}


def _scope(conn, context: TenantContext) -> None:
    conn.execute(text("SELECT set_config('app.tenant_id', :v, true)"), {"v": context.tenant_id})
    conn.execute(text("SELECT set_config('app.user_id', :v, true)"), {"v": context.user_id})
    conn.execute(text("SELECT set_config('app.scope_type', :v, true)"), {"v": context.scope_type})


def _decision(conn, release_id: str, action: str, actor: str, reason: str = "", evidence: dict | None = None) -> None:
    conn.execute(text("""INSERT INTO governance.rag_release_decisions(decision_id,release_id,action,actor,reason,evidence)
      VALUES(:id,:release,:action,:actor,:reason,CAST(:evidence AS jsonb))"""), {"id": f"rdec_{uuid4().hex}", "release": release_id, "action": action, "actor": actor, "reason": reason, "evidence": json.dumps(evidence or {})})


def create_release(actor: str, spec: dict, corpus_hash: str, parent_release_id: str | None = None) -> dict:
    required = {"backend", "index_version", "embedding_model", "reranker_strategy", "top_k", "candidate_k"}
    missing = sorted(required - set(spec))
    if missing:
        raise ValueError(f"Release spec is missing: {', '.join(missing)}")
    if spec["backend"] not in {"local_hybrid", "qdrant_vector", "qdrant_hybrid"}:
        raise ValueError("Unsupported RAG backend")
    if spec["backend"] == "local_hybrid" and not spec.get("local_snapshot_ref"):
        raise ValueError("local_hybrid release requires local_snapshot_ref")
    if spec["backend"] in {"qdrant_vector", "qdrant_hybrid"}:
        if not spec.get("qdrant_collection"):
            raise ValueError("Qdrant release requires qdrant_collection")
        if not spec.get("release_manifest_ref"):
            raise ValueError("Qdrant release requires release_manifest_ref")
        _validate_qdrant_release_manifest(spec)
    spec_hash = _hash(spec)
    release_id = f"ragrel_{spec_hash[:32]}"
    with engine.begin() as conn:
        existing = conn.execute(text("SELECT * FROM governance.rag_releases WHERE spec_hash=:hash"), {"hash": spec_hash}).mappings().first()
        if existing:
            return dict(existing)
        conn.execute(text("""INSERT INTO governance.rag_releases(release_id,parent_release_id,status,spec,spec_hash,corpus_hash,created_by)
          VALUES(:id,:parent,'draft',CAST(:spec AS jsonb),:hash,:corpus,:actor)"""), {"id": release_id, "parent": parent_release_id, "spec": json.dumps(spec), "hash": spec_hash, "corpus": corpus_hash, "actor": actor})
        _decision(conn, release_id, "created", actor, evidence={"spec_hash": spec_hash})
        row = conn.execute(text("SELECT * FROM governance.rag_releases WHERE release_id=:id"), {"id": release_id}).mappings().one()
    record_event("rag.release.created", {"rag.release_id": release_id})
    return dict(row)


def get_release(release_id: str) -> dict | None:
    with engine.begin() as conn:
        row = conn.execute(text("SELECT * FROM governance.rag_releases WHERE release_id=:id"), {"id": release_id}).mappings().first()
    return dict(row) if row else None


def published_release() -> dict | None:
    with engine.begin() as conn:
        row = conn.execute(text("SELECT * FROM governance.rag_releases WHERE status='published'")).mappings().first()
    return dict(row) if row else None


def list_releases(limit: int = 50) -> list[dict]:
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT * FROM governance.rag_releases ORDER BY created_at DESC LIMIT :limit"), {"limit": min(max(limit, 1), 100)}).mappings().all()
    return [dict(row) for row in rows]


def release_summary(release_id: str) -> dict:
    with engine.begin() as conn:
        evaluation = conn.execute(text("SELECT * FROM governance.rag_release_evaluations WHERE release_id=:id ORDER BY created_at DESC LIMIT 10"), {"id": release_id}).mappings().all()
        incidents = conn.execute(text("SELECT * FROM governance.rag_release_incidents WHERE release_id=:id ORDER BY created_at DESC LIMIT 20"), {"id": release_id}).mappings().all()
        comparisons = conn.execute(text("SELECT count(*) AS comparisons FROM governance.rag_shadow_comparisons WHERE candidate_release_id=:id"), {"id": release_id}).mappings().one()
    return {"release": get_release(release_id), "evaluations": [dict(row) for row in evaluation], "incidents": [dict(row) for row in incidents], "shadow_comparisons": comparisons["comparisons"]}


def record_artifact(release_id: str, actor: str, artifact: dict) -> dict:
    release = get_release(release_id)
    if not release or release["status"] not in {"draft", "building"}:
        raise ValueError("Release is not buildable")
    required = {"backend", "artifact_hash", "chunk_count", "vector_count", "sparse_vector_count", "build_status"}
    if required - set(artifact):
        raise ValueError("Artifact is incomplete")
    spec = dict(release["spec"] or {})
    if artifact["backend"] != spec.get("backend"):
        raise ValueError("Artifact backend must match immutable release spec")
    if spec.get("local_snapshot_ref") and artifact.get("local_snapshot_ref") != spec["local_snapshot_ref"]:
        raise ValueError("Artifact local snapshot must match immutable release spec")
    if spec.get("qdrant_collection") and artifact.get("qdrant_collection") != spec["qdrant_collection"]:
        raise ValueError("Artifact Qdrant collection must match immutable release spec")
    with engine.begin() as conn:
        conn.execute(text("UPDATE governance.rag_releases SET status='building' WHERE release_id=:id AND status='draft'"), {"id": release_id})
        conn.execute(text("""INSERT INTO governance.rag_release_artifacts(artifact_id,release_id,backend,local_snapshot_ref,qdrant_collection,artifact_hash,chunk_count,vector_count,sparse_vector_count,build_status,validation)
          VALUES(:id,:release,:backend,:local,:collection,:hash,:chunks,:vectors,:sparse,:status,CAST(:validation AS jsonb))"""), {"id": f"rart_{uuid4().hex}", "release": release_id, "backend": artifact["backend"], "local": artifact.get("local_snapshot_ref"), "collection": artifact.get("qdrant_collection"), "hash": artifact["artifact_hash"], "chunks": artifact["chunk_count"], "vectors": artifact["vector_count"], "sparse": artifact["sparse_vector_count"], "status": artifact["build_status"], "validation": json.dumps(artifact.get("validation") or {})})
        _decision(conn, release_id, "built", actor, evidence=artifact)
    return get_release(release_id) or {}


def record_evaluation(release_id: str, actor: str, kind: str, metrics: dict, *, baseline_release_id: str | None, dataset_version: str, passed: bool | None, thresholds: dict | None = None, evaluator: dict | None = None, failure_reason: str = "") -> dict:
    if kind not in {"deterministic", "ragas"}:
        raise ValueError("Unsupported evaluation type")
    status = "unavailable" if passed is None else ("passed" if passed else "failed")
    evaluation_id = f"reval_{uuid4().hex}"
    with engine.begin() as conn:
        if kind == "deterministic":
            # An evaluation is evidence about a concrete artifact, never only
            # about a mutable configuration object.
            artifact = conn.execute(text("""SELECT 1 FROM governance.rag_release_artifacts
                WHERE release_id=:id AND build_status='ready' AND artifact_hash IS NOT NULL
                LIMIT 1"""), {"id": release_id}).scalar_one_or_none()
            if not artifact:
                raise ValueError("A ready, validated release artifact is required before evaluation")
        conn.execute(text("""INSERT INTO governance.rag_release_evaluations(evaluation_id,release_id,baseline_release_id,kind,dataset_version,metrics,thresholds,status,evaluator,failure_reason)
          VALUES(:id,:release,:baseline,:kind,:dataset,CAST(:metrics AS jsonb),CAST(:thresholds AS jsonb),:status,CAST(:evaluator AS jsonb),:reason)"""), {"id": evaluation_id, "release": release_id, "baseline": baseline_release_id, "kind": kind, "dataset": dataset_version, "metrics": json.dumps(metrics), "thresholds": json.dumps(thresholds or _DEFAULT_THRESHOLDS), "status": status, "evaluator": json.dumps(evaluator or {}), "reason": failure_reason})
        if kind == "deterministic":
            target = "evaluated" if passed else "failed"
            conn.execute(text("UPDATE governance.rag_releases SET status=:status WHERE release_id=:id AND status IN ('building','draft','evaluated')"), {"status": target, "id": release_id})
            _decision(conn, release_id, "evaluated", actor, failure_reason, {"passed": passed, "metrics": metrics})
    return {"evaluation_id": evaluation_id, "release_id": release_id, "kind": kind, "status": status}


def block_release_for_quality_gate(release_id: str, actor: str, reasons: list[str]) -> None:
    with engine.begin() as conn:
        release = conn.execute(text("SELECT status FROM governance.rag_releases WHERE release_id=:id FOR UPDATE"), {"id": release_id}).mappings().first()
        if not release:
            raise ValueError("Release not found")
        active = release["status"] in {"shadowing", "canary_5", "canary_25", "canary_100"}
        # The policy update and state transition share one transaction: an
        # asynchronous contract failure can never leave a routable candidate.
        conn.execute(text("UPDATE governance.rag_rollout_policies SET active=false,updated_at=now() WHERE release_id=:id AND active"), {"id": release_id})
        conn.execute(text("UPDATE governance.rag_releases SET status=:status WHERE release_id=:id AND status IN ('draft','building','evaluated','shadowing','canary_5','canary_25','canary_100')"), {"id": release_id, "status": "paused" if active else "failed"})
        _decision(conn, release_id, "paused" if active else "evaluated", actor, "unified quality gate blocked release", {"reasons": reasons, "previous_status": release["status"]})


def set_rollout(release_id: str, actor: str, stage: str, *, shadow_sample_rate: float = 1.0, tenant_allowlist: list[str] | None = None, auto_rollback: bool = True, thresholds: dict | None = None) -> dict:
    if stage not in _STAGES:
        raise ValueError("Unsupported rollout stage")
    release = get_release(release_id)
    if not release:
        raise ValueError("Release not found")
    target = stage
    if not transition_allowed(release["status"], target):
        raise ValueError(f"Cannot transition {release['status']} to {target}")
    effective_thresholds = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}
    if set(effective_thresholds) != set(_DEFAULT_THRESHOLDS) or any(float(value) < 0 for value in effective_thresholds.values()):
        raise ValueError("Invalid rollout thresholds")
    if target != "off":
        from app.quality.evaluations import rag_gate_passed
        if not rag_gate_passed(release_id):
            raise ValueError("A passed unified RAG quality gate is required before rollout")
    percent = {"off": 0, "shadowing": 0, "canary_5": 5, "canary_25": 25, "canary_100": 100}[stage]
    with engine.begin() as conn:
        conn.execute(text("UPDATE governance.rag_rollout_policies SET active=false WHERE release_id=:release AND active"), {"release": release_id})
        conn.execute(text("""INSERT INTO governance.rag_rollout_policies(policy_id,release_id,stage,shadow_sample_rate,canary_percent,tenant_allowlist,thresholds,auto_rollback,created_by)
          VALUES(:id,:release,:stage,:rate,:percent,CAST(:allowlist AS jsonb),CAST(:thresholds AS jsonb),:rollback,:actor)"""), {"id": f"rpol_{uuid4().hex}", "release": release_id, "stage": stage, "rate": shadow_sample_rate, "percent": percent, "allowlist": json.dumps(tenant_allowlist or []), "thresholds": json.dumps(effective_thresholds), "rollback": auto_rollback, "actor": actor})
        if stage != "off":
            conn.execute(text("UPDATE governance.rag_releases SET status=:status WHERE release_id=:id"), {"status": target, "id": release_id})
            _decision(conn, release_id, "shadow_started" if stage == "shadowing" else "canary_started", actor, evidence={"stage": stage, "percent": percent})
    return resolve_rollout(release_id) or {}


def resolve_rollout(candidate_release_id: str) -> dict | None:
    with engine.begin() as conn:
        row = conn.execute(text("SELECT * FROM governance.rag_rollout_policies WHERE release_id=:id AND active ORDER BY updated_at DESC LIMIT 1"), {"id": candidate_release_id}).mappings().first()
    return dict(row) if row else None


def assignment(context: TenantContext, session_id: str | None, trace_id: str | None) -> dict:
    stable = published_release()
    candidate = None
    with engine.begin() as conn:
        row = conn.execute(text("""SELECT r.*, p.stage,p.canary_percent,p.shadow_sample_rate,p.tenant_allowlist
          FROM governance.rag_rollout_policies p JOIN governance.rag_releases r ON r.release_id=p.release_id
          WHERE p.active AND p.stage <> 'off' AND r.status IN ('shadowing','canary_5','canary_25','canary_100') ORDER BY p.updated_at DESC LIMIT 1""")).mappings().first()
    if row:
        candidate = dict(row)
    return rollout_assignment(stable, candidate, context, session_id, trace_id)


def enqueue_shadow(context: TenantContext, query: str, stable_release_id: str, candidate_release_id: str, mode: str, request_context: dict) -> str:
    digest = _hash(query)
    job_id = f"rshadow_{uuid4().hex}"
    with engine.begin() as conn:
        _scope(conn, context)
        conn.execute(text("""INSERT INTO governance.rag_shadow_jobs(job_id,tenant_id,user_id,stable_release_id,candidate_release_id,mode,query_payload,query_hash,request_context,expires_at)
          VALUES(:id,:tenant,:user,:stable,:candidate,:mode,:query,:hash,CAST(:ctx AS jsonb),:expires)"""), {"id": job_id, "tenant": context.tenant_id, "user": context.user_id, "stable": stable_release_id, "candidate": candidate_release_id, "mode": mode, "query": query[:4000], "hash": digest, "ctx": json.dumps(request_context), "expires": datetime.now(timezone.utc) + timedelta(days=7)})
    record_event("rag.shadow.enqueued", {"rag.candidate_release": candidate_release_id, "rag.mode": mode})
    return job_id


def record_feedback(context: TenantContext, trace_id: str, release_id: str, rating: int, category: str | None) -> dict:
    with engine.begin() as conn:
        _scope(conn, context)
        conn.execute(text("""INSERT INTO governance.rag_retrieval_feedback(feedback_id,tenant_id,user_id,trace_id,release_id,rating,category)
          VALUES(:id,:tenant,:user,:trace,:release,:rating,:category)
          ON CONFLICT(tenant_id,user_id,trace_id) DO UPDATE SET rating=EXCLUDED.rating,category=EXCLUDED.category"""), {"id": f"rfeed_{uuid4().hex}", "tenant": context.tenant_id, "user": context.user_id, "trace": trace_id, "release": release_id, "rating": rating, "category": category})
    return {"accepted": True, "release_id": release_id}


def promote(release_id: str, actor: str) -> dict:
    release = get_release(release_id)
    if not release or release["status"] not in {"canary_100", "promotable"}:
        raise ValueError("Only a fully-canary or promotable release can be published")
    with engine.begin() as conn:
        gate = conn.execute(text("""SELECT 1 FROM governance.rag_release_evaluations
            WHERE release_id=:id AND kind='deterministic' AND status='passed'"""), {"id": release_id}).scalar_one_or_none()
        if not gate:
            raise ValueError("A passed deterministic evaluation is required")
        from app.quality.evaluations import rag_gate_passed
        if not rag_gate_passed(release_id):
            raise ValueError("A passed unified RAG quality gate is required")
        incident = conn.execute(text("SELECT 1 FROM governance.rag_release_incidents WHERE release_id=:id AND severity='critical' AND resolved_at IS NULL LIMIT 1"), {"id": release_id}).scalar_one_or_none()
        if incident:
            raise ValueError("Unresolved critical release incident blocks promotion")
        conn.execute(text("UPDATE governance.rag_releases SET status='rolled_back' WHERE status='published'"))
        conn.execute(text("UPDATE governance.rag_releases SET status='published',approved_by=:actor,approved_at=now() WHERE release_id=:id"), {"id": release_id, "actor": actor})
        conn.execute(text("UPDATE governance.rag_rollout_policies SET active=false WHERE release_id=:id AND active"), {"id": release_id})
        _decision(conn, release_id, "promoted", actor)
    _export_manifest(get_release(release_id) or {})
    record_event("rag.release.promoted", {"rag.release_id": release_id})
    return get_release(release_id) or {}


def rollback(release_id: str, actor: str, reason: str = "manual rollback") -> dict:
    release = get_release(release_id)
    if not release:
        raise ValueError("Release not found")
    parent = get_release(str(release["parent_release_id"])) if release.get("parent_release_id") else None
    with engine.begin() as conn:
        conn.execute(text("UPDATE governance.rag_rollout_policies SET active=false WHERE release_id=:id AND active"), {"id": release_id})
        conn.execute(text("UPDATE governance.rag_releases SET status='rolled_back' WHERE release_id=:id"), {"id": release_id})
        if parent:
            conn.execute(text("UPDATE governance.rag_releases SET status='published',approved_by=:actor,approved_at=now() WHERE release_id=:id"), {"id": parent["release_id"], "actor": actor})
        _decision(conn, release_id, "rolled_back", actor, reason, {"parent_release_id": parent["release_id"] if parent else None})
    if parent:
        _export_manifest(get_release(parent["release_id"]) or {})
    return get_release(release_id) or {}


def _export_manifest(release: dict) -> None:
    """Keep the legacy bootstrap manifest in sync without making it authoritative."""
    from app.core.config import get_settings
    if not release:
        return
    target = Path(get_settings().rag_release_manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"releases": [{"version": release["release_id"], "source": "release-controller", "status": "published", "approved_by": release.get("approved_by")}]}, ensure_ascii=False), encoding="utf-8")


def evaluate_online_safety(release_id: str) -> bool:
    """Auto-pause only after the policy's minimum comparison sample is met."""
    with engine.begin() as conn:
        policy = conn.execute(text("SELECT * FROM governance.rag_rollout_policies WHERE release_id=:id AND active"), {"id": release_id}).mappings().first()
        if not policy or not policy["auto_rollback"]:
            return False
        metrics = conn.execute(text("""SELECT count(*) AS samples,
          avg((comparison->>'latency_ratio')::numeric) AS latency_ratio,
          avg(CASE WHEN candidate_result ? 'error' THEN 1 ELSE 0 END) AS error_rate,
          avg(CASE WHEN candidate_result->>'empty'='true' THEN 1 ELSE 0 END) AS empty_rate,
          avg(CASE WHEN candidate_result->>'citation_count'='0' THEN 1 ELSE 0 END) AS citationless_rate
          FROM governance.rag_shadow_comparisons WHERE candidate_release_id=:id"""), {"id": release_id}).mappings().one()
        if metrics["samples"] < policy["min_samples"]:
            return False
        limits = dict(policy["thresholds"] or _DEFAULT_THRESHOLDS)
        unsafe = (float(metrics["latency_ratio"] or 0) > float(limits["p95_ratio"])
                  or float(metrics["error_rate"] or 0) > float(limits["error_rate_delta"])
                  or float(metrics["empty_rate"] or 0) > float(limits["empty_rate_delta"])
                  or float(metrics["citationless_rate"] or 0) > float(limits["citationless_rate_delta"]))
        if not unsafe:
            return False
        conn.execute(text("UPDATE governance.rag_rollout_policies SET active=false,updated_at=now() WHERE policy_id=:id"), {"id": policy["policy_id"]})
        release = conn.execute(text("SELECT parent_release_id FROM governance.rag_releases WHERE release_id=:id FOR UPDATE"), {"id": release_id}).mappings().one()
        # At 5/25% an unsafe candidate is paused.  At 100% it is a real
        # rollback: restore the retained parent as published before returning
        # from this transaction, so every subsequent resolution is stable.
        if policy["stage"] == "canary_100" and release["parent_release_id"]:
            conn.execute(text("UPDATE governance.rag_releases SET status='rolled_back' WHERE release_id=:id"), {"id": release_id})
            conn.execute(text("UPDATE governance.rag_releases SET status='published',approved_by='system',approved_at=now() WHERE release_id=:id"), {"id": release["parent_release_id"]})
            action = "rolled_back"
        else:
            conn.execute(text("UPDATE governance.rag_releases SET status='paused' WHERE release_id=:id AND status LIKE 'canary_%'"), {"id": release_id})
            action = "paused"
        conn.execute(text("""INSERT INTO governance.rag_release_incidents(incident_id,release_id,severity,code,details)
          VALUES(:incident,:release,'critical','online_quality_gate',CAST(:details AS jsonb))"""), {"incident": f"rinc_{uuid4().hex}", "release": release_id, "details": json.dumps(dict(metrics))})
        _decision(conn, release_id, action, "system", "online quality threshold exceeded", dict(metrics))
    if action == "rolled_back":
        _export_manifest(get_release(str(release["parent_release_id"])) or {})
    record_event(f"rag.release.{action}", {"rag.release_id": release_id})
    return True


def retrieve_routed(question: str, *, context: TenantContext | None, session_id: str | None, trace_id: str | None, doc_types=None, metric_name: str | None = None, top_k: int | None = None, authorization_context=None, authorization_boundary=None, current_scope_fingerprint: str | None = None):
    """Resolve stable/candidate once and never let a candidate outage hurt chat."""
    from app.core.config import get_settings
    from app.rag.retriever import retrieve_for_release, retrieve_knowledge
    if not get_settings().rag_release_controller_enabled or context is None:
        return retrieve_knowledge(question, doc_types=doc_types, metric_name=metric_name, top_k=top_k, tenant_context=context, authorization_context=authorization_context, authorization_boundary=authorization_boundary, current_scope_fingerprint=current_scope_fingerprint)
    decision = assignment(context, session_id, trace_id)
    from app.rag.flags import rollout_flags
    flags = rollout_flags({"tenant_id": context.tenant_id, "user_id": context.user_id, "session_id": session_id or trace_id or ""})
    # A flag outage or locally disabled flag may only reduce exposure.  It
    # cannot select a release or bypass the controller's eligibility checks.
    decision["canary"] = bool(decision["canary"] and flags["canary"])
    decision["shadow"] = bool(decision["shadow"] and flags["shadow"])
    stable, candidate = decision["stable"], decision["candidate"]
    if not stable:
        return retrieve_knowledge(question, doc_types=doc_types, metric_name=metric_name, top_k=top_k, tenant_context=context, authorization_context=authorization_context, authorization_boundary=authorization_boundary, current_scope_fingerprint=current_scope_fingerprint)
    chosen = candidate if decision["canary"] and candidate else stable
    try:
        result = retrieve_for_release(question, chosen, doc_types=doc_types, metric_name=metric_name, top_k=top_k, tenant_context=context, authorization_context=authorization_context, authorization_boundary=authorization_boundary, current_scope_fingerprint=current_scope_fingerprint)
    except Exception as exc:
        result = retrieve_for_release(question, stable, doc_types=doc_types, metric_name=metric_name, top_k=top_k, tenant_context=context, authorization_context=authorization_context, authorization_boundary=authorization_boundary, current_scope_fingerprint=current_scope_fingerprint)
        result.filters["candidate_fallback"] = type(exc).__name__
        with engine.begin() as conn:
            conn.execute(text("""INSERT INTO governance.rag_release_incidents(incident_id,release_id,severity,code,details)
              VALUES(:id,:release,'critical','candidate_retrieval_failure',CAST(:details AS jsonb))"""), {"id": f"rinc_{uuid4().hex}", "release": chosen["release_id"], "details": json.dumps({"error": type(exc).__name__})})
    if candidate and decision["shadow"]:
        enqueue_shadow(context, question, stable["release_id"], candidate["release_id"], "canary_compare" if decision["canary"] else "shadow", {"session_id": session_id, "trace_id": trace_id, "doc_types": sorted(doc_types) if doc_types else None, "metric_name": metric_name, "top_k": top_k})
    result.filters["rollout_mode"] = "canary" if decision["canary"] else ("shadow" if decision["shadow"] else "stable")
    return result

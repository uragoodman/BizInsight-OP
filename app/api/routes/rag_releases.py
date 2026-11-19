"""Administrator API for governed RAG releases and rollout decisions."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.auth import require_analysis_write, require_tenant_context
from app.rag import release_controller as controller
from app.tenancy import TenantContext

router = APIRouter(prefix="/rag", tags=["rag-releases"])

class ReleaseCreate(BaseModel):
    spec: dict
    corpus_hash: str = Field(min_length=64, max_length=64)
    parent_release_id: str | None = None

class ArtifactCreate(BaseModel):
    backend: str
    artifact_hash: str
    chunk_count: int = 0
    vector_count: int = 0
    sparse_vector_count: int = 0
    build_status: str = "ready"
    local_snapshot_ref: str | None = None
    qdrant_collection: str | None = None
    validation: dict = Field(default_factory=dict)

class RolloutRequest(BaseModel):
    shadow_sample_rate: float = 1.0
    tenant_allowlist: list[str] = Field(default_factory=list)
    auto_rollback: bool = True
    thresholds: dict[str, float] | None = None

class FeedbackRequest(BaseModel):
    trace_id: str
    release_id: str
    rating: int
    category: str | None = None

def _admin(context: TenantContext) -> None:
    if context.role not in {"admin", "platform_admin"}:
        raise HTTPException(status_code=403, detail="RAG release management requires administrator role")

@router.post("/releases")
def create_release(payload: ReleaseCreate, actor: str = Depends(require_analysis_write), context: TenantContext = Depends(require_tenant_context)) -> dict:
    _admin(context)
    return controller.create_release(actor, payload.spec, payload.corpus_hash, payload.parent_release_id)

@router.get("/releases/{release_id}")
def get_release(release_id: str, context: TenantContext = Depends(require_tenant_context)) -> dict:
    _admin(context)
    row = controller.get_release(release_id)
    if not row:
        raise HTTPException(status_code=404, detail="RAG release not found")
    return row

@router.get("/releases")
def list_releases(context: TenantContext = Depends(require_tenant_context)) -> dict:
    _admin(context)
    return {"items": controller.list_releases()}

@router.get("/releases/{release_id}/shadow-summary")
def shadow_summary(release_id: str, context: TenantContext = Depends(require_tenant_context)) -> dict:
    _admin(context)
    result = controller.release_summary(release_id)
    if not result["release"]:
        raise HTTPException(status_code=404, detail="RAG release not found")
    return result

@router.post("/releases/{release_id}/build")
def build_release(release_id: str, payload: ArtifactCreate, actor: str = Depends(require_analysis_write), context: TenantContext = Depends(require_tenant_context)) -> dict:
    _admin(context)
    try:
        return controller.record_artifact(release_id, actor, payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

@router.post("/releases/{release_id}/evaluate")
def evaluate_release(release_id: str, actor: str = Depends(require_analysis_write), context: TenantContext = Depends(require_tenant_context)) -> dict:
    _admin(context)
    from evaluation.retrieval.runner import evaluate
    release = controller.get_release(release_id)
    if not release:
        raise HTTPException(status_code=404, detail="RAG release not found")
    stable = controller.published_release()
    spec = dict(release["spec"])
    candidate_metrics = evaluate(
        str(spec.get("reranker_strategy") or "rule"),
        top_k=int(spec.get("top_k") or 20),
        backend_override=str(spec.get("backend") or "local_hybrid"),
        candidate_k=int(spec.get("candidate_k") or 0) or None,
        weight_overrides={
            field: float(spec[field])
            for field in ("vector_weight", "sparse_weight", "semantic_weight", "metadata_weight")
            if field in spec
        } or None,
    )
    baseline_spec = dict(stable["spec"]) if stable else spec
    baseline_metrics = evaluate(
        str(baseline_spec.get("reranker_strategy") or "rule"),
        top_k=int(baseline_spec.get("top_k") or 20),
        backend_override=str(baseline_spec.get("backend") or "local_hybrid"),
        candidate_k=int(baseline_spec.get("candidate_k") or 0) or None,
        weight_overrides={
            field: float(baseline_spec[field])
            for field in ("vector_weight", "sparse_weight", "semantic_weight", "metadata_weight")
            if field in baseline_spec
        } or None,
    ) if stable else candidate_metrics
    passed = (
        candidate_metrics["doc_hit@5"] >= baseline_metrics["doc_hit@5"]
        and candidate_metrics["chunk_hit@5"] >= baseline_metrics["chunk_hit@5"]
        and candidate_metrics["mrr@5"] >= baseline_metrics["mrr@5"] - 0.01
        and candidate_metrics["ndcg@5"] >= baseline_metrics["ndcg@5"] - 0.01
        and candidate_metrics["timeout_rate"] <= baseline_metrics["timeout_rate"] + 0.01
        and candidate_metrics["p95_ms"] <= baseline_metrics["p95_ms"] * 1.2
    )
    result = controller.record_evaluation(release_id, actor, "deterministic", {"candidate": candidate_metrics, "baseline": baseline_metrics}, baseline_release_id=stable["release_id"] if stable else None, dataset_version="retrieval-queries-v1", passed=passed)
    from evaluation.retrieval.runner import QUERIES
    from app.quality.evaluations import enqueue_rag_contract_validation, run_rag_gate
    envelope = run_rag_gate(release_id, actor, candidate_metrics, baseline_metrics, evaluation_id=result["evaluation_id"], dataset_path=str(QUERIES))
    contract_job_id = enqueue_rag_contract_validation(release_id, envelope["run_id"])
    if envelope["gate_status"] == "block":
        controller.block_release_for_quality_gate(release_id, actor, list(envelope["gate_reasons"]))
    return {**result, "evaluation_run": envelope, "contract_validation_job_id": contract_job_id}

@router.post("/releases/{release_id}/shadow/start")
@router.post("/releases/{release_id}/canary/{stage}")
def set_rollout(release_id: str, payload: RolloutRequest, stage: str = "shadowing", actor: str = Depends(require_analysis_write), context: TenantContext = Depends(require_tenant_context)) -> dict:
    _admin(context)
    resolved = stage if stage == "shadowing" else f"canary_{stage}"
    try:
        return controller.set_rollout(release_id, actor, resolved, shadow_sample_rate=payload.shadow_sample_rate, tenant_allowlist=payload.tenant_allowlist, auto_rollback=payload.auto_rollback, thresholds=payload.thresholds)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

@router.post("/releases/{release_id}/pause")
def pause(release_id: str, payload: RolloutRequest, actor: str = Depends(require_analysis_write), context: TenantContext = Depends(require_tenant_context)) -> dict:
    _admin(context)
    return controller.set_rollout(release_id, actor, "off", shadow_sample_rate=payload.shadow_sample_rate)

@router.post("/releases/{release_id}/promote")
def promote(release_id: str, actor: str = Depends(require_analysis_write), context: TenantContext = Depends(require_tenant_context)) -> dict:
    _admin(context)
    try:
        return controller.promote(release_id, actor)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

@router.post("/releases/{release_id}/rollback")
def rollback(release_id: str, actor: str = Depends(require_analysis_write), context: TenantContext = Depends(require_tenant_context)) -> dict:
    _admin(context)
    try:
        return controller.rollback(release_id, actor)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

@router.post("/feedback")
def feedback(payload: FeedbackRequest, context: TenantContext = Depends(require_tenant_context)) -> dict:
    if payload.rating not in {-1, 1}:
        raise HTTPException(status_code=422, detail="rating must be -1 or 1")
    return controller.record_feedback(context, payload.trace_id, payload.release_id, payload.rating, payload.category)

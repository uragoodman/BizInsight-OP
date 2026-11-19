"""Administrator API for the unified evaluation evidence envelopes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.auth import require_analysis_write, require_tenant_context
from app.quality import evaluations
from app.tenancy import TenantContext

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


def _admin(context: TenantContext) -> None:
    if context.role != "admin":
        raise HTTPException(status_code=403, detail="Evaluation management requires administrator role")


@router.get("")
def list_evaluations(context: TenantContext = Depends(require_tenant_context)) -> dict:
    _admin(context)
    return {"items": evaluations.list_runs()}


@router.get("/{run_id}")
def get_evaluation(run_id: str, context: TenantContext = Depends(require_tenant_context)) -> dict:
    _admin(context)
    run = evaluations.get_run(run_id)
    if not run: raise HTTPException(status_code=404, detail="Evaluation run not found")
    return run


@router.get("/{run_id}/comparison")
def get_comparison(run_id: str, context: TenantContext = Depends(require_tenant_context)) -> dict:
    _admin(context)
    run = evaluations.get_run(run_id)
    if not run: raise HTTPException(status_code=404, detail="Evaluation run not found")
    return {"run_id": run_id, "comparison": run["comparison"], "gate_status": run["gate_status"], "gate_reasons": run["gate_reasons"]}


@router.post("/{subject_type}")
def run_subject(subject_type: str, actor: str = Depends(require_analysis_write), context: TenantContext = Depends(require_tenant_context)) -> dict:
    _admin(context)
    try:
        return evaluations.run_subject(subject_type, actor)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

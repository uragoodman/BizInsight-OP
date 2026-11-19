"""Governed HTTP entry point for registered randomized experiment estimates."""

from fastapi import APIRouter, Depends, HTTPException

from app.api.auth import require_tenant_context
from app.iam.service import require_permission
from app.schemas.experiments import (
    CausalExperimentEstimateRequest,
    CausalExperimentScopeRequest,
)
from app.schemas.explorer import ControlledAnalysisScope
from app.tenancy import TenantContext, activate_tenant_context

router = APIRouter(prefix="/causal-experiments", tags=["causal-experiments"])


def _scope(payload: CausalExperimentScopeRequest) -> ControlledAnalysisScope:
    return ControlledAnalysisScope.model_validate({
        "seller_id": payload.seller_id,
        "default_start_date": payload.start_date,
        "default_end_date": payload.end_date,
        "metric_name": "gmv",
        "granularity": "month",
        "max_rows": 100_000,
    })


def _authorize_action(context: TenantContext) -> None:
    try:
        require_permission(context, "analyze")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.post("/eligible")
def eligible_experiments(
    payload: CausalExperimentScopeRequest,
    context: TenantContext = Depends(require_tenant_context),
) -> dict[str, object]:
    """List verified randomized experiments the caller can actually analyze."""
    from app.data.repositories.experiments import list_scoped_randomized_experiments

    _authorize_action(context)
    try:
        with activate_tenant_context(context):
            return list_scoped_randomized_experiments(_scope(payload))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/estimate")
def estimate_experiment(
    payload: CausalExperimentEstimateRequest,
    context: TenantContext = Depends(require_tenant_context),
) -> dict[str, object]:
    """Estimate registered ITT; the client cannot choose outcome or estimator settings."""
    from app.data.repositories.experiments import estimate_scoped_randomized_experiment

    _authorize_action(context)
    try:
        with activate_tenant_context(context):
            return estimate_scoped_randomized_experiment(
                _scope(payload), experiment_id=payload.experiment_id,
            )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

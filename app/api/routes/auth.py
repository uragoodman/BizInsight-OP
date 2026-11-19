from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from app.api.auth import _decode_hs256_jwt, require_tenant_context
from app.core.config import get_settings
from app.iam.service import (
    MembershipErasureError,
    AccountErasureError,
    bootstrap,
    create_member,
    create_organization,
    get_membership_erasure_job,
    list_members,
    login,
    logout,
    profile,
    request_membership_erasure,
    request_account_erasure,
    require_permission,
    retry_membership_erasure,
    update_member,
)
from app.security.abac import delete_policy_rule, list_policy_rules, save_policy_rule
from app.schemas.auth import AbacPolicyRequest, BootstrapRequest, LoginRequest, MemberCreateRequest, MemberUpdateRequest, OrganizationCreateRequest
from app.tenancy import TenantContext
from app.usage.service import usage_overview


router = APIRouter(prefix="/auth", tags=["auth"])


class MemberErasureRequest(BaseModel):
    confirmation_phrase: str = Field(min_length=1, max_length=64)
    idempotency_key: str = Field(min_length=1, max_length=128)


class AccountErasureRequest(BaseModel):
    confirmation_phrase: str = Field(min_length=1, max_length=64)
    idempotency_key: str = Field(min_length=1, max_length=128)


@router.post("/bootstrap")
def initial_bootstrap(payload: BootstrapRequest, x_bootstrap_token: str | None = Header(default=None)) -> dict:
    settings = get_settings()
    if not settings.auth_bootstrap_token or x_bootstrap_token != settings.auth_bootstrap_token:
        raise HTTPException(status_code=403, detail="Bootstrap permission denied")
    try:
        bootstrap(email=str(payload.email), password=payload.password, display_name=payload.display_name)
        return {"bootstrapped": True}
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/login")
def password_login(payload: LoginRequest) -> dict:
    settings = get_settings()
    if not settings.tenant_auth_secret:
        raise HTTPException(status_code=503, detail="Password login is not configured")
    try:
        token, context, expires_at = login(email=str(payload.email), password=payload.password, secret=settings.tenant_auth_secret, session_hours=settings.auth_session_hours)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail="Invalid email or password") from exc
    return {"access_token": token, "token_type": "bearer", "expires_at": expires_at.isoformat(), "profile": profile(context)}


@router.post("/logout")
def password_logout(authorization: str | None = Header(default=None)) -> dict:
    settings = get_settings()
    if not authorization or not authorization.startswith("Bearer ") or not settings.tenant_auth_secret:
        raise HTTPException(status_code=401, detail="Bearer session required")
    claims = _decode_hs256_jwt(authorization.removeprefix("Bearer ").strip(), settings.tenant_auth_secret)
    session_id = claims.get("sid")
    if not isinstance(session_id, str):
        raise HTTPException(status_code=401, detail="Password session required")
    logout(session_id)
    return {"logged_out": True}


@router.get("/me")
def me(context: TenantContext = Depends(require_tenant_context)) -> dict:
    return profile(context)


@router.post("/account/erasure", status_code=202)
def erase_my_account(
    payload: AccountErasureRequest,
    context: TenantContext = Depends(require_tenant_context),
) -> dict:
    """Revoke all account access now and queue cross-tenant personal-data erasure."""
    if payload.confirmation_phrase != "DELETE_MY_ACCOUNT_AND_ERASE_DATA":
        raise HTTPException(status_code=422, detail="second_confirmation_required")
    try:
        job = request_account_erasure(context, idempotency_key=payload.idempotency_key)
        return {"job_id": job["job_id"], "status": job["status"], "access_revoked": True}
    except AccountErasureError as exc:
        status_code = 409 if exc.code == "active_account_not_found" else 400
        raise HTTPException(status_code=status_code, detail=exc.code) from exc


@router.get("/usage")
def usage(days: int = 30, limit: int = 20, context: TenantContext = Depends(require_tenant_context)) -> dict:
    """Return only the signed-in user's provider-reported model usage."""
    return usage_overview(tenant_id=context.tenant_id, user_id=context.user_id, days=days, limit=limit)


@router.get("/members")
def members(context: TenantContext = Depends(require_tenant_context)) -> dict:
    try:
        require_permission(context, "manage_members")
        return {"items": list_members(context)}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.post("/organizations")
def add_organization(payload: OrganizationCreateRequest, context: TenantContext = Depends(require_tenant_context)) -> dict:
    try:
        return create_organization(context, tenant_name=payload.tenant_name, email=str(payload.email), display_name=payload.display_name, password=payload.password, grants=[item.model_dump() for item in payload.seller_grants])
    except (PermissionError, ValueError) as exc:
        raise HTTPException(status_code=403 if isinstance(exc, PermissionError) else 400, detail=str(exc)) from exc


@router.post("/members")
def add_member(payload: MemberCreateRequest, context: TenantContext = Depends(require_tenant_context)) -> dict:
    try:
        require_permission(context, "manage_members")
        return create_member(context, email=str(payload.email), display_name=payload.display_name, password=payload.password, role=payload.role, grants=[item.model_dump() for item in payload.seller_grants])
    except (PermissionError, ValueError) as exc:
        raise HTTPException(status_code=403 if isinstance(exc, PermissionError) else 400, detail=str(exc)) from exc


@router.put("/members/{user_id}")
def edit_member(user_id: str, payload: MemberUpdateRequest, context: TenantContext = Depends(require_tenant_context)) -> dict:
    try:
        require_permission(context, "manage_members")
        update_member(context, user_id, role=payload.role, grants=[item.model_dump() for item in payload.seller_grants])
        return {"updated": True}
    except (PermissionError, ValueError) as exc:
        raise HTTPException(status_code=403 if isinstance(exc, PermissionError) else 400, detail=str(exc)) from exc


@router.delete("/members/{user_id}", status_code=202)
def revoke_member(
    user_id: str,
    payload: MemberErasureRequest,
    context: TenantContext = Depends(require_tenant_context),
) -> dict:
    try:
        require_permission(context, "manage_members")
        if payload.confirmation_phrase != "REVOKE_MEMBER_AND_ERASE_MEMORY":
            raise HTTPException(status_code=422, detail="second_confirmation_required")
        job = request_membership_erasure(
            context, user_id, idempotency_key=payload.idempotency_key,
        )
        return {"job_id": job["job_id"], "subject_user_id": job["subject_user_id"],
                "status": job["status"], "membership_revoked": True}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except MembershipErasureError as exc:
        status_code = 404 if exc.code == "active_member_not_found" else 409
        raise HTTPException(status_code=status_code, detail=exc.code) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/members/erasure-jobs/{job_id}")
def member_erasure_job(
    job_id: str,
    context: TenantContext = Depends(require_tenant_context),
) -> dict:
    try:
        require_permission(context, "manage_members")
        job = get_membership_erasure_job(context, job_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="erasure_job_not_found")
    return job


@router.post("/members/erasure-jobs/{job_id}/retry")
def retry_member_erasure_job(
    job_id: str,
    context: TenantContext = Depends(require_tenant_context),
) -> dict:
    try:
        require_permission(context, "manage_members")
        return retry_membership_erasure(context, job_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except MembershipErasureError as exc:
        status_code = 404 if exc.code == "erasure_job_not_found" else 409
        raise HTTPException(status_code=status_code, detail=exc.code) from exc


@router.get("/abac/policies")
def abac_policies(context: TenantContext = Depends(require_tenant_context)) -> dict:
    try:
        return {"items": list_policy_rules(context)}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.put("/abac/policies/{policy_id}")
def put_abac_policy(policy_id: str, payload: AbacPolicyRequest, context: TenantContext = Depends(require_tenant_context)) -> dict:
    try:
        return save_policy_rule(context, policy_id, payload.rule, status=payload.status)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/abac/policies/{policy_id}")
def remove_abac_policy(policy_id: str, context: TenantContext = Depends(require_tenant_context)) -> dict:
    try:
        delete_policy_rule(context, policy_id)
        return {"deleted": True, "policy_id": policy_id}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

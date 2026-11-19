"""Server-side resolution for controlled seller breakdown scopes."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import date
from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.auth.context import AuthorizationContext, AuthorizationDenied
from app.scope.canonical import CanonicalScope, ScopeError
from app.schemas.explorer import SellerScopeProvenance, SignedSellerScopePayload
from app.tenancy import TenantContext


class SellerScopeIntent(BaseModel):
    """Untrusted scope request. Seller IDs are re-authorized before use."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seller_scope_type: Literal["single", "selected", "all"]
    seller_ids: tuple[str, ...] = ()
    start_date: date
    end_date: date
    metric_ids: tuple[str, ...] = ("gmv",)
    dimensions: tuple[str, ...] = ("seller",)
    granularity: str = "day"
    include_unattributed: bool = False

    @model_validator(mode="after")
    def validate_shape(self) -> "SellerScopeIntent":
        ids = tuple(sorted(set(str(value) for value in self.seller_ids if str(value))))
        object.__setattr__(self, "seller_ids", ids)
        if self.end_date < self.start_date:
            raise ValueError("end_date must not precede start_date")
        if self.seller_scope_type == "all" and ids:
            raise ValueError("ALL must use an explicit marker and empty seller_ids")
        if self.seller_scope_type == "single" and len(ids) != 1:
            raise ValueError("single scope requires exactly one seller_id")
        if self.seller_scope_type == "selected" and not ids:
            raise ValueError("selected scope requires seller_ids")
        if self.include_unattributed and self.seller_scope_type != "all":
            raise ValueError("include_unattributed is only valid for ALL scope")
        return self


def issue_seller_scope_payload(
    intent: SellerScopeIntent,
    *,
    tenant: TenantContext,
    ttl_seconds: int = 300,
) -> dict[str, object]:
    """Sign a short-lived scope selection made by the trusted Explorer API."""

    from app.core.config import get_settings

    secret = str(get_settings().tenant_auth_secret or "")
    if not secret:
        raise ScopeError("Seller scope provenance signing is not configured")
    now = datetime.now(timezone.utc)
    expires_epoch = int((now + timedelta(seconds=min(max(int(ttl_seconds), 1), 300))).timestamp())
    expires = datetime.fromtimestamp(expires_epoch, timezone.utc)
    canonical_intent = intent.model_dump(mode="json")
    claims = {
        "v": 1,
        "source": "explorer_selector",
        "tenant_id": tenant.tenant_id,
        "user_id": tenant.user_id,
        "issued_at": int(now.timestamp()),
        "expires_at": expires_epoch,
        "seller_scope_intent": canonical_intent,
    }
    body = _b64url(json.dumps(
        claims, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8"))
    signature = hmac.new(
        _seller_scope_signing_key(secret), body.encode("ascii"), hashlib.sha256,
    ).digest()
    token = body + "." + _b64url(signature)
    return SignedSellerScopePayload(
        seller_scope_intent=canonical_intent,
        provenance=SellerScopeProvenance(
            source="explorer_selector", token=token,
            expires_at=expires.isoformat(),
        ),
    ).model_dump(mode="json")


def verify_seller_scope_payload(
    payload: dict[str, object] | SignedSellerScopePayload,
    *,
    tenant: TenantContext,
) -> SellerScopeIntent:
    """Verify provenance, caller binding, expiry, and exact intent equality."""

    from app.core.config import get_settings

    secret = str(get_settings().tenant_auth_secret or "")
    if not secret:
        raise ScopeError("Seller scope provenance signing is not configured")
    try:
        envelope = payload if isinstance(payload, SignedSellerScopePayload) else SignedSellerScopePayload.model_validate(payload)
        body, encoded_signature = envelope.provenance.token.split(".", 1)
        supplied = _b64url_decode(encoded_signature)
        expected = hmac.new(
            _seller_scope_signing_key(secret), body.encode("ascii"), hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(expected, supplied):
            raise ValueError("invalid signature")
        claims = json.loads(_b64url_decode(body))
        intent = SellerScopeIntent.model_validate(envelope.seller_scope_intent)
        claimed_intent = SellerScopeIntent.model_validate(claims["seller_scope_intent"])
    except Exception as exc:  # noqa: BLE001 - all malformed tokens fail closed
        raise ScopeError("Invalid seller scope provenance") from exc
    now = int(datetime.now(timezone.utc).timestamp())
    if claims.get("v") != 1 or claims.get("source") != "explorer_selector":
        raise ScopeError("Seller scope provenance source is invalid")
    if claims.get("tenant_id") != tenant.tenant_id or claims.get("user_id") != tenant.user_id:
        raise ScopeError("Seller scope provenance belongs to a different identity")
    if not isinstance(claims.get("expires_at"), int) or claims["expires_at"] <= now:
        raise ScopeError("Seller scope provenance has expired")
    if not isinstance(claims.get("issued_at"), int) or claims["issued_at"] > now + 30:
        raise ScopeError("Seller scope provenance issue time is invalid")
    if claims["expires_at"] - claims["issued_at"] > 300:
        raise ScopeError("Seller scope provenance lifetime exceeds the limit")
    if envelope.provenance.source != claims["source"]:
        raise ScopeError("Seller scope provenance source does not match the token")
    expires_at = datetime.fromtimestamp(claims["expires_at"], timezone.utc).isoformat()
    try:
        response_expiry = datetime.fromisoformat(envelope.provenance.expires_at)
        response_expiry = response_expiry.astimezone(timezone.utc).isoformat()
    except ValueError as exc:
        raise ScopeError("Seller scope provenance expiry is invalid") from exc
    if response_expiry != expires_at:
        raise ScopeError("Seller scope provenance expiry does not match the token")
    if _canonical_json(intent.model_dump(mode="json")) != _canonical_json(claimed_intent.model_dump(mode="json")):
        raise ScopeError("Seller scope intent does not match its provenance token")
    return intent


def _seller_scope_signing_key(secret: str) -> bytes:
    return hmac.new(secret.encode("utf-8"), b"seller-scope-provenance-v1", hashlib.sha256).digest()


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def resolve_seller_scope(
    intent: SellerScopeIntent | dict[str, object],
    *,
    tenant: TenantContext,
    authz: AuthorizationContext,
) -> CanonicalScope:
    """Recompute the requested scope from server-owned authorization facts."""

    from app.core.config import get_settings
    from app.metrics.registry import get_registry

    request = intent if isinstance(intent, SellerScopeIntent) else SellerScopeIntent.model_validate(intent)
    settings = get_settings()
    if not settings.seller_dimension_enabled:
        raise ScopeError("Seller dimension execution is disabled")
    if request.seller_scope_type in {"selected", "all"} and not settings.seller_multi_scope_enabled:
        raise ScopeError("Multi-seller scopes are disabled")
    if request.seller_scope_type in {"all", "selected"} and not tenant.is_platform_admin:
        raise ScopeError("Only platform administrators may request selected or ALL seller scopes")
    if request.include_unattributed and not tenant.is_platform_admin:
        raise ScopeError("Only platform administrators may include unattributed rows")
    if len(request.seller_ids) > settings.seller_scope_max_selected_sellers:
        raise ScopeError(
            f"Seller scope exceeds the limit of {settings.seller_scope_max_selected_sellers} sellers"
        )
    if (request.end_date - request.start_date).days > settings.seller_scope_max_date_span_days:
        raise ScopeError(
            f"Seller scope date range must not exceed {settings.seller_scope_max_date_span_days} days"
        )

    if request.seller_scope_type == "single":
        seller_id = request.seller_ids[0]
        if tenant.scope_type == "seller" and seller_id not in tenant.allowed_seller_ids:
            raise ScopeError("Seller tenant cannot access another seller")
        seller_ids = (seller_id,)
    else:
        seller_ids = request.seller_ids

    registry = get_registry()
    if request.dimensions != ("seller",):
        raise ScopeError("Seller breakdown scope must declare only the seller dimension")
    for metric_id in request.metric_ids:
        if metric_id not in {"gmv", "order_count"}:
            raise ScopeError(f"Seller breakdown is not enabled for metric {metric_id!r}")
        if not registry.supports(metric_id, "breakdown", scope="platform") or not registry.is_dimension_allowed(
            metric_id, "seller", scope="platform"
        ):
            raise ScopeError(f"Seller breakdown is not registered for metric {metric_id!r}")

    try:
        authz.require_action("analyze")
        authz.require_metrics(request.metric_ids)
        if tenant.is_platform_admin:
            authz.require_dimensions(("seller",))
        if request.seller_scope_type != "all":
            authz.require_sellers(seller_ids)
        for seller_id in seller_ids:
            if tenant.is_platform_admin and request.seller_scope_type == "selected":
                # Selected is an explicit finite entitlement set. Platform
                # administrators must have a current grant for every member;
                # ALL remains the separate platform-level entitlement.
                grant = authz.seller_grant(seller_id)
                if grant is None:
                    raise ScopeError(f"Selected seller {seller_id!r} has no explicit seller grant")
                grant_start = date.fromisoformat(grant.start_date) if grant.start_date else None
                grant_end = date.fromisoformat(grant.end_date) if grant.end_date else None
                if (grant_start and request.start_date < grant_start) or (grant_end and request.end_date > grant_end):
                    raise ScopeError("Requested date range exceeds seller grant")
                continue
            if tenant.is_platform_admin:
                grant = authz.seller_grant(seller_id)
                if grant is not None:
                    grant_start = date.fromisoformat(grant.start_date) if grant.start_date else None
                    grant_end = date.fromisoformat(grant.end_date) if grant.end_date else None
                    if (grant_start and request.start_date < grant_start) or (grant_end and request.end_date > grant_end):
                        raise ScopeError("Requested date range exceeds seller grant")
                continue
            authz.require_date_range(seller_id, request.start_date, request.end_date)
    except AuthorizationDenied as exc:
        raise ScopeError(str(exc)) from exc

    return CanonicalScope.resolve(
        authz,
        seller_ids=seller_ids,
        seller_scope_type=request.seller_scope_type,
        include_unattributed=request.include_unattributed,
        start_date=request.start_date,
        end_date=request.end_date,
        metric_ids=request.metric_ids,
        dimensions=request.dimensions,
        granularity=request.granularity,
        actions=("analyze",),
    )

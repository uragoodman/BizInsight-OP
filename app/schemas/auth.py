from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    # `.local` is intentionally accepted for offline demo accounts; production
    # deployments may still enforce verified corporate domains upstream.
    email: str = Field(min_length=3, max_length=320, pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    password: str = Field(min_length=1, max_length=256)


class BootstrapRequest(LoginRequest):
    display_name: str = Field(min_length=1, max_length=256)


class SellerGrantRequest(BaseModel):
    seller_id: str = Field(min_length=1, max_length=64)
    start_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")


class MemberCreateRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320, pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    display_name: str = Field(min_length=1, max_length=256)
    password: str = Field(min_length=8, max_length=256)
    role: str = Field(pattern=r"^(admin|analyst|reviewer|viewer)$")
    seller_grants: list[SellerGrantRequest] = Field(default_factory=list, max_length=100)


class MemberUpdateRequest(BaseModel):
    role: str = Field(pattern=r"^(admin|analyst|reviewer|viewer)$")
    seller_grants: list[SellerGrantRequest] = Field(default_factory=list, max_length=100)


class OrganizationCreateRequest(MemberCreateRequest):
    tenant_name: str = Field(min_length=1, max_length=256)


class AbacPolicyRequest(BaseModel):
    rule: dict[str, Any]
    status: Literal["active", "disabled"] = "active"

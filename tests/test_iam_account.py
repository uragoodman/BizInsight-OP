from datetime import datetime, timezone

from app.iam.service import encode_session_token, hash_password, permissions, verify_password
from app.tenancy import SELLER_SCOPE, SellerGrant, TenantContext


def test_scrypt_password_hash_is_salted_and_verifiable() -> None:
    first = hash_password("correct horse battery staple")
    second = hash_password("correct horse battery staple")
    assert first != second
    assert verify_password("correct horse battery staple", first)
    assert not verify_password("wrong password", first)


def test_multi_seller_analyst_permission_and_grants() -> None:
    context = TenantContext(
        tenant_id="tenant-a", user_id="user-a", role="analyst", scope_type=SELLER_SCOPE,
        seller_grants=(SellerGrant("seller-1", "2018-01-01", "2018-12-31"), SellerGrant("seller-2")),
    )
    assert context.allowed_seller_ids == ("seller-1", "seller-2")
    assert context.seller_grant("seller-1").start_date == "2018-01-01"  # type: ignore[union-attr]
    assert {"analyze", "report_export", "code_analysis"}.issubset(permissions(context))
    assert {"code_analysis_read"}.issubset(permissions(context))


def test_code_analysis_roles_have_separate_share_and_audit_permissions() -> None:
    analyst = TenantContext(
        tenant_id="tenant-a", user_id="user-a", role="analyst", scope_type=SELLER_SCOPE,
        seller_grants=(SellerGrant("seller-1"),),
    )
    reviewer = TenantContext(
        tenant_id="tenant-a", user_id="user-r", role="reviewer", scope_type=SELLER_SCOPE,
        seller_grants=(SellerGrant("seller-1"),),
    )
    assert "code_analysis_read" in permissions(analyst)
    assert "code_analysis_share" not in permissions(analyst)
    assert "code_analysis_read" in permissions(reviewer)
    assert "code_analysis" not in permissions(reviewer)


def test_session_token_does_not_embed_authorization_role() -> None:
    token = encode_session_token(session_id="session-id", tenant_id="tenant-a", user_id="user-a", secret="secret", expires_at=datetime.now(timezone.utc))
    assert "analyst" not in token and "seller-1" not in token

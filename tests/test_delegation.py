import pytest

from app.tenancy import SELLER_SCOPE, TenantContext
from app.tenancy.delegation import issue_delegation_token, verify_delegation_token


def test_delegation_token_preserves_tenant_scope_and_rejects_tampering():
    context = TenantContext("tenant-a", "user-a", "analyst", SELLER_SCOPE, "seller-a")
    token = issue_delegation_token(context, correlation_id="trace-a", secret="secret")
    claims = verify_delegation_token(token, secret="secret")
    assert (claims.tenant_id, claims.seller_id, claims.correlation_id) == ("tenant-a", "seller-a", "trace-a")
    with pytest.raises(ValueError):
        verify_delegation_token(token + "x", secret="secret")

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.api import auth


def _request() -> Request:
    return Request({"type": "http", "method": "GET", "path": "/protected", "headers": []})


@pytest.mark.parametrize(
    ("environment", "tenant_auth_required"),
    [("production", False), ("development", True)],
)
def test_legacy_analysis_token_cannot_establish_identity_in_protected_environments(
    monkeypatch, environment: str, tenant_auth_required: bool,
) -> None:
    monkeypatch.setattr(auth, "get_settings", lambda: SimpleNamespace(
        environment=environment,
        tenant_auth_required=tenant_auth_required,
        analysis_write_token="shared-write-token",
        user_id="configured-user",
        tenant_auth_secret="",
    ))
    monkeypatch.setattr(auth, "_audit_denial", lambda *_args, **_kwargs: None)

    with pytest.raises(HTTPException) as exc:
        auth.require_tenant_context(
            _request(),
            authorization=None,
            x_api_key=None,
            x_analysis_token="shared-write-token",
            x_actor_id="caller-chosen-admin",
        )
    assert exc.value.status_code == 401


def test_legacy_analysis_token_remains_available_for_unprotected_local_development(monkeypatch) -> None:
    monkeypatch.setattr(auth, "get_settings", lambda: SimpleNamespace(
        environment="development",
        tenant_auth_required=False,
        analysis_write_token="shared-write-token",
        user_id="configured-user",
        tenant_auth_secret="",
    ))

    context = auth.require_tenant_context(
        _request(),
        authorization=None,
        x_api_key=None,
        x_analysis_token="shared-write-token",
        x_actor_id="local-demo-actor",
    )

    assert context.tenant_id == "platform"
    assert context.user_id == "local-demo-actor"
    assert context.role == "admin"

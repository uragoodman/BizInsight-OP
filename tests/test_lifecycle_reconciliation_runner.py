from __future__ import annotations

from app.tenancy import TenantContext
from scripts import reconcile_chat_lifecycle as runner


def test_runner_reloads_each_member_context_and_aggregates(monkeypatch) -> None:
    contexts = {
        ("tenant-a", "user-a"): TenantContext(
            tenant_id="tenant-a", user_id="user-a", role="owner", scope_type="seller",
            seller_id="seller-a",
        ),
        ("tenant-b", "user-b"): TenantContext(
            tenant_id="tenant-b", user_id="user-b", role="analyst", scope_type="seller",
            seller_id="seller-b",
        ),
    }
    loaded: list[tuple[str, str]] = []
    authorized: list[TenantContext] = []
    monkeypatch.setattr(runner, "active_member_keys", lambda: list(contexts))

    def load_context(*, tenant_id: str, user_id: str):
        loaded.append((tenant_id, user_id))
        return contexts[(tenant_id, user_id)]

    def build_authz(context):
        authorized.append(context)
        return object()

    monkeypatch.setattr(runner, "load_active_tenant_context", load_context)
    monkeypatch.setattr(runner, "build_authorization_context", build_authz)
    monkeypatch.setattr(runner, "run_lifecycle_reconciliation", lambda context, authz: {
        "interrupted": int(context.user_id == "user-a"),
        "orphan_tasks_interrupted": int(context.user_id == "user-b") * 2,
    })
    monkeypatch.setattr(runner, "record_event", lambda *_args, **_kwargs: None)

    result = runner.run_reconciliation_pass()

    assert loaded == list(contexts)
    assert authorized == list(contexts.values())
    assert result == {
        "identities_seen": 2,
        "identities_processed": 2,
        "identities_skipped": 0,
        "identity_errors": 0,
        "executions_interrupted": 1,
        "orphan_tasks_interrupted": 2,
    }


def test_runner_skips_revoked_member_and_continues_after_member_error(monkeypatch) -> None:
    contexts = [
        TenantContext(
            tenant_id="tenant-a", user_id="revoked", role="owner", scope_type="seller",
            seller_id="seller-a",
        ),
        TenantContext(
            tenant_id="tenant-b", user_id="active", role="analyst", scope_type="seller",
            seller_id="seller-b",
        ),
    ]
    keys = [(ctx.tenant_id, ctx.user_id) for ctx in contexts]
    monkeypatch.setattr(runner, "active_member_keys", lambda: keys)
    monkeypatch.setattr(
        runner,
        "load_active_tenant_context",
        lambda *, tenant_id, user_id: None if user_id == "revoked" else contexts[1],
    )
    monkeypatch.setattr(runner, "build_authorization_context", lambda _context: object())

    def reconcile(_context, _authz):
        raise RuntimeError("synthetic test error")

    monkeypatch.setattr(runner, "run_lifecycle_reconciliation", reconcile)
    monkeypatch.setattr(runner, "record_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner.logger, "exception", lambda *_args, **_kwargs: None)

    result = runner.run_reconciliation_pass()

    assert result["identities_seen"] == 2
    assert result["identities_processed"] == 0
    assert result["identities_skipped"] == 1
    assert result["identity_errors"] == 1

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.auth import require_tenant_context
from app.api.routes import auth as auth_routes, memory as memory_routes
from app.main import app
from app.memory import preference_management
from app.tenancy import TenantContext


@pytest.fixture
def preference_client():
    context = TenantContext(
        tenant_id="tenant-api", user_id="user-api", role="admin", scope_type="platform",
    )
    previous = app.dependency_overrides.get(require_tenant_context)
    app.dependency_overrides[require_tenant_context] = lambda: context
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(require_tenant_context, None)
        if previous is not None:
            app.dependency_overrides[require_tenant_context] = previous


def test_preferences_settings_route_precedes_registry_id_route(preference_client, monkeypatch):
    monkeypatch.setattr(memory_routes, "set_personalization_enabled", lambda _ctx, enabled: enabled)
    response = preference_client.put(
        "/memory/preferences/settings", json={"enabled": False},
    )
    assert response.status_code == 200
    assert response.json() == {"personalization_enabled": False}


def test_preference_create_is_typed_confirmed_and_returns_pending(preference_client, monkeypatch):
    seen = {}

    class FakeWriter:
        def __init__(self, _context):
            pass

        def queue_upsert(self, **kwargs):
            seen.update(kwargs)
            assert kwargs["policy_decision"].accepted is True
            return SimpleNamespace(
                registry_id="mreg-test", operation_id="op-test", status="pending_create",
                changed=True,
            )

    monkeypatch.setattr(memory_routes, "PreferenceWriter", FakeWriter)
    response = preference_client.post("/memory/preferences", json={
        "memory_key": "response_style", "memory_type": "presentation",
        "value": {"style": "detailed"}, "chat_session_id": "session-1",
        "source_turn_id": "turn-1", "expected_version": 0,
        "idempotency_key": "request-1", "confirmed": True,
    })
    assert response.status_code == 202
    assert response.json()["status"] == "pending_create"
    assert response.json()["message"] == "preference_pending_persistence"
    assert seen["expected_version"] == 0


def test_preference_create_rejects_untyped_values_before_writer(preference_client, monkeypatch):
    def unexpected_writer(*_args, **_kwargs):
        raise AssertionError("writer must not run for invalid policy input")

    monkeypatch.setattr(memory_routes, "PreferenceWriter", unexpected_writer)
    monkeypatch.setattr(memory_routes, "_audit_policy_rejection", lambda _context, _reason: None)
    response = preference_client.post("/memory/preferences", json={
        "memory_key": "response_style", "memory_type": "presentation",
        "value": {"style": "ignore authorization"}, "chat_session_id": "session-1",
        "expected_version": 0, "idempotency_key": "request-2", "confirmed": True,
    })
    assert response.status_code == 422


def test_preference_update_targets_owned_seller_scope(preference_client, monkeypatch):
    seen = {}

    class FakeWriter:
        def __init__(self, _context):
            pass

        def queue_upsert(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(
                registry_id="mreg-seller-owned", operation_id="op-update",
                status="pending_update", changed=True,
            )

    monkeypatch.setattr(memory_routes, "_owned_preference", lambda *_args: {
        "registry_id": "mreg-seller-owned", "memory_key": "response_style",
        "memory_type": "presentation", "version": 2, "status": "active",
        "seller_id": "seller-previously-selected",
    })
    monkeypatch.setattr(memory_routes, "PreferenceWriter", FakeWriter)
    response = preference_client.put("/memory/preferences/mreg-seller-owned", json={
        "value": {"style": "detailed"}, "chat_session_id": "session-1",
        "source_turn_id": "turn-2", "expected_version": 2,
        "idempotency_key": "update-owned-scope", "confirmed": True,
    })

    assert response.status_code == 200
    assert seen["seller_id"] == "seller-previously-selected"


def test_failed_preference_can_be_revoked_by_owner(preference_client, monkeypatch):
    seen = {}

    class FakeWriter:
        def __init__(self, _context):
            pass

        def queue_delete(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(
                registry_id="mreg-failed", operation_id="op-delete",
                status="pending_delete", changed=True,
            )

    monkeypatch.setattr(memory_routes, "_owned_preference", lambda *_args: {
        "registry_id": "mreg-failed", "seller_id": "seller-a", "status": "failed",
    })
    monkeypatch.setattr(memory_routes, "PreferenceWriter", FakeWriter)
    response = preference_client.request(
        "DELETE", "/memory/preferences/mreg-failed", json={
            "expected_version": 2, "idempotency_key": "revoke-failed", "confirmed": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "pending_delete"
    assert seen["registry_id"] == "mreg-failed"


def test_owned_preference_lookup_uses_tenant_user_ownership_not_active_seller(monkeypatch):
    context = TenantContext(
        tenant_id="tenant-owner", user_id="user-owner", role="admin", scope_type="platform",
    )
    calls = []

    class Result:
        def mappings(self):
            return self

        def first(self):
            return {"registry_id": "mreg-owned", "seller_id": "seller-other"}

    class Connection:
        def execute(self, statement, params):
            calls.append((str(statement), params))
            return Result()

    class Transaction:
        def __enter__(self):
            return Connection()

        def __exit__(self, *_args):
            return False

    class Database:
        def begin(self):
            return Transaction()

    monkeypatch.setattr(memory_routes, "engine", Database())
    monkeypatch.setattr(memory_routes, "set_local_context", lambda *_args, **_kwargs: None)
    row = memory_routes._owned_preference(context, "mreg-owned")

    assert row == {"registry_id": "mreg-owned", "seller_id": "seller-other"}
    sql, params = calls[0]
    assert "tenant_id=:tenant_id" in sql and "user_id=:user_id" in sql
    assert "seller_id IS NOT DISTINCT FROM" not in sql
    assert params == {
        "registry_id": "mreg-owned", "tenant_id": "tenant-owner", "user_id": "user-owner",
    }


def test_preference_retry_route_exposes_requeued_state(preference_client, monkeypatch):
    monkeypatch.setattr(
        memory_routes, "retry_failed_preference", lambda _context, registry_id: "pending_delete",
    )
    response = preference_client.post("/memory/preferences/mreg-failed/retry")
    assert response.status_code == 200
    assert response.json() == {"registry_id": "mreg-failed", "status": "pending_delete"}


def test_preference_reconcile_route_queues_active_registry_only(preference_client, monkeypatch):
    seen = {}

    class FakeWriter:
        def __init__(self, _context):
            pass

        def queue_reconciliation(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(
                registry_id="mreg-active", operation_id="op-active",
                status="pending_update", changed=True,
            )

    monkeypatch.setattr(
        memory_routes, "_owned_preference",
        lambda _context, _registry_id: {"registry_id": "mreg-active", "status": "active", "version": 4},
    )
    monkeypatch.setattr(memory_routes, "PreferenceWriter", FakeWriter)
    response = preference_client.post(
        "/memory/preferences/mreg-active/reconcile",
        json={"expected_version": 4, "idempotency_key": "reconcile-4"},
    )

    assert response.status_code == 202
    assert response.json() == {
        "registry_id": "mreg-active", "operation_id": "op-active",
        "status": "pending_update", "changed": True,
    }
    assert seen == {
        "registry_id": "mreg-active", "expected_version": 4,
        "idempotency_key": "reconcile-4",
    }


@pytest.mark.parametrize(
    ("owned", "expected_status", "expected_code"),
    [
        (None, 404, "preference_not_found"),
        ({"registry_id": "mreg-pending", "status": "pending_update", "version": 2},
         409, "preference_not_reconcilable"),
    ],
)
def test_preference_reconcile_rejects_unowned_or_inflight_registry(
    preference_client, monkeypatch, owned, expected_status, expected_code,
):
    def unexpected_writer(*_args, **_kwargs):
        raise AssertionError("writer must not run for an unowned or in-flight preference")

    monkeypatch.setattr(memory_routes, "_owned_preference", lambda *_args: owned)
    monkeypatch.setattr(memory_routes, "PreferenceWriter", unexpected_writer)
    response = preference_client.post(
        "/memory/preferences/mreg-target/reconcile",
        json={"expected_version": 2, "idempotency_key": "reconcile-guard"},
    )
    assert response.status_code == expected_status
    assert response.json()["detail"] == expected_code


def test_preference_history_is_tenant_scoped_and_metadata_only(preference_client, monkeypatch):
    monkeypatch.setattr(
        memory_routes, "_owned_preference", lambda _context, _registry_id: {"registry_id": "mreg-1"},
    )
    monkeypatch.setattr(memory_routes, "preference_history", lambda *_args: [{
        "event_type": "worker_succeeded", "status": "succeeded", "reason_code": None,
    }])
    response = preference_client.get("/memory/preferences/mreg-1/history")
    assert response.status_code == 200
    assert response.json() == {"items": [{
        "event_type": "worker_succeeded", "status": "succeeded", "reason_code": None,
    }]}


def test_delete_all_requires_explicit_second_confirmation(preference_client, monkeypatch):
    def unexpected_engine(*_args, **_kwargs):
        raise AssertionError("confirmation must be checked before database access")

    monkeypatch.setattr(memory_routes, "engine", unexpected_engine)
    response = preference_client.request(
        "DELETE", "/memory/preferences", json={
            "confirmation_phrase": "delete", "idempotency_key": "delete-all-1",
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "second_confirmation_required"


def test_delete_all_cancels_pending_create_deletes_expired_scope_and_reports_inflight(
    preference_client, monkeypatch,
):
    class Rows:
        def mappings(self):
            return self

        def all(self):
            return [
                {"registry_id": "mreg-pending", "version": 1, "status": "pending_create"},
                {"registry_id": "mreg-updating", "version": 2, "status": "pending_update"},
                {"registry_id": "mreg-expired", "version": 3, "status": "superseded"},
                {"registry_id": "mreg-failed", "version": 4, "status": "failed"},
            ]

    class Connection:
        def execute(self, *_args, **_kwargs):
            return Rows()

    class Transaction:
        def __enter__(self):
            return Connection()

        def __exit__(self, *_args):
            return False

    class Database:
        def begin(self):
            return Transaction()

    calls = []

    class FakeWriter:
        def __init__(self, _context):
            pass

        def queue_delete(self, **kwargs):
            calls.append(kwargs["registry_id"])
            return SimpleNamespace(registry_id=kwargs["registry_id"])

    monkeypatch.setattr(memory_routes, "engine", Database())
    monkeypatch.setattr(memory_routes, "set_local_context", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(memory_routes, "PreferenceWriter", FakeWriter)
    response = preference_client.request("DELETE", "/memory/preferences", json={
        "confirmation_phrase": "DELETE_ALL_PREFERENCES", "idempotency_key": "delete-all-2",
    })

    assert response.status_code == 200
    assert calls == ["mreg-pending", "mreg-expired", "mreg-failed"]
    assert response.json() == {
        "queued_count": 3,
        "deleted_count": 0,
        "pending_registry_ids": ["mreg-pending", "mreg-expired", "mreg-failed"],
        "unfinished": [{"registry_id": "mreg-updating", "reason": "already_pending_update"}],
        "complete": False,
    }


def test_expired_scope_default_remains_visible_for_reconfirmation(monkeypatch):
    context = TenantContext(
        tenant_id="tenant-expired", user_id="user-expired",
        role="admin", scope_type="platform",
    )
    row = {
        "registry_id": "mreg-expired", "seller_id": None,
        "memory_type": "scope_default", "memory_key": "default_date",
        "mem0_memory_id": "mem-expired", "status": "superseded", "version": 1,
        "operation_id": "op-expired", "source_turn_id": "turn-expired",
        "updated_at": datetime.now(timezone.utc),
        "expires_at": datetime.now(timezone.utc) - timedelta(days=1),
    }

    class Result:
        def __init__(self, items):
            self.items = items

        def mappings(self):
            return self

        def all(self):
            return self.items

    class Connection:
        calls = 0

        def execute(self, statement, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                assert "status='superseded'" in str(statement)
                return Result([row])
            return Result([])

    class Transaction:
        def __enter__(self):
            self.connection = Connection()
            return self.connection

        def __exit__(self, *_args):
            return False

    class Database:
        def begin(self):
            return Transaction()

    class FakeAdapter:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_validated(self, *_args, include_expired=False, **_kwargs):
            assert include_expired is True
            return SimpleNamespace(value={"date": "2026-01"})

    monkeypatch.setattr(preference_management, "set_local_context", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        preference_management, "PostgresPreferenceRegistry", lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(preference_management, "Mem0PreferenceAdapter", FakeAdapter)

    items = preference_management.list_preferences(context, engine_override=Database())
    assert items == [{
        "registry_id": "mreg-expired", "memory_key": "default_date",
        "memory_type": "scope_default", "value": {"date": "2026-01"},
        "scope": {"type": "user"}, "status": "superseded",
        "version": 1,
        "source_turn_id": "turn-expired", "requires_confirmation": True,
        "updated_at": row["updated_at"], "expires_at": row["expires_at"],
        "overlay": None,
    }]


def test_preferences_list_all_owned_seller_scopes(monkeypatch):
    context = TenantContext(
        tenant_id="tenant-multi-seller", user_id="user-multi-seller",
        role="admin", scope_type="platform",
    )
    rows = [
        {
            "registry_id": "mreg-a", "seller_id": "seller-a",
            "memory_type": "presentation", "memory_key": "response_style",
            "mem0_memory_id": None, "status": "pending_create", "version": 1,
            "operation_id": "op-a", "source_turn_id": None,
            "updated_at": None, "expires_at": None,
        },
        {
            "registry_id": "mreg-b", "seller_id": "seller-b",
            "memory_type": "presentation", "memory_key": "report_style",
            "mem0_memory_id": None, "status": "pending_create", "version": 1,
            "operation_id": "op-b", "source_turn_id": None,
            "updated_at": None, "expires_at": None,
        },
    ]

    class Result:
        def __init__(self, items):
            self.items = items

        def mappings(self):
            return self

        def all(self):
            return self.items

    executed = []

    class Connection:
        calls = 0

        def execute(self, statement, params=None, **_kwargs):
            self.calls += 1
            executed.append((str(statement), params))
            return Result(rows if self.calls == 1 else [])

    class Transaction:
        def __enter__(self):
            self.connection = Connection()
            return self.connection

        def __exit__(self, *_args):
            return False

    class Database:
        def begin(self):
            return Transaction()

    monkeypatch.setattr(preference_management, "set_local_context", lambda *_args, **_kwargs: None)
    items = preference_management.list_preferences(context, engine_override=Database())

    assert [item["scope"] for item in items] == [
        {"seller_id": "seller-a"}, {"seller_id": "seller-b"},
    ]
    registry_sql, registry_params = executed[0]
    assert "seller_id IS NOT DISTINCT FROM :seller_id" not in registry_sql
    assert registry_params == {"tenant_id": context.tenant_id, "user_id": context.user_id}


def test_preferences_management_reads_mem0_from_recorded_seller_partition(monkeypatch):
    context = TenantContext(
        tenant_id="tenant-multi-seller", user_id="user-multi-seller",
        role="admin", scope_type="platform",
    )
    rows = [
        {
            "registry_id": "mreg-a", "seller_id": "seller-a",
            "memory_type": "presentation", "memory_key": "response_style",
            "mem0_memory_id": "mem-a", "status": "active", "version": 1,
            "operation_id": "op-a", "source_turn_id": None,
            "updated_at": None, "expires_at": None,
        },
        {
            "registry_id": "mreg-b", "seller_id": "seller-b",
            "memory_type": "presentation", "memory_key": "report_style",
            "mem0_memory_id": "mem-b", "status": "active", "version": 1,
            "operation_id": "op-b", "source_turn_id": None,
            "updated_at": None, "expires_at": None,
        },
    ]

    class Result:
        def __init__(self, items):
            self.items = items

        def mappings(self):
            return self

        def all(self):
            return self.items

    class Connection:
        calls = 0

        def execute(self, *_args, **_kwargs):
            self.calls += 1
            return Result(rows if self.calls == 1 else [])

    class Transaction:
        def __enter__(self):
            return Connection()

        def __exit__(self, *_args):
            return False

    class Database:
        def begin(self):
            return Transaction()

    adapter_sellers = []

    class FakeAdapter:
        def __init__(self, scoped_context, *, registry):
            self.seller_id = scoped_context.seller_id
            adapter_sellers.append(self.seller_id)

        def get_validated(self, memory_id, **_kwargs):
            return SimpleNamespace(value={"memory_id": memory_id})

    monkeypatch.setattr(preference_management, "set_local_context", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(preference_management, "PostgresPreferenceRegistry", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(preference_management, "Mem0PreferenceAdapter", FakeAdapter)
    items = preference_management.list_preferences(context, engine_override=Database())

    assert adapter_sellers == ["seller-a", "seller-b"]
    assert [item["value"] for item in items] == [
        {"memory_id": "mem-a"}, {"memory_id": "mem-b"},
    ]


def test_membership_revocation_requires_second_confirmation(preference_client, monkeypatch):
    def unexpected_request(*_args, **_kwargs):
        raise AssertionError("erasure job must not be created without the confirmation phrase")

    monkeypatch.setattr(auth_routes, "require_permission", lambda *_args: None)
    monkeypatch.setattr(auth_routes, "request_membership_erasure", unexpected_request)
    response = preference_client.request(
        "DELETE", "/auth/members/user-target", json={
            "confirmation_phrase": "delete", "idempotency_key": "erase-confirmation",
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "second_confirmation_required"


def test_membership_revocation_returns_durable_job_and_status(preference_client, monkeypatch):
    context = TenantContext(
        tenant_id="tenant-api", user_id="user-api", role="admin", scope_type="platform",
    )
    calls = []
    monkeypatch.setattr(auth_routes, "require_permission", lambda *_args: None)

    def request_erasure(actual_context, subject, *, idempotency_key):
        calls.append((actual_context, subject, idempotency_key))
        return {
            "job_id": "merase-test", "subject_user_id": subject,
            "status": "pending", "created_at": "2026-10-31T00:00:00Z",
        }

    monkeypatch.setattr(auth_routes, "request_membership_erasure", request_erasure)
    monkeypatch.setattr(auth_routes, "get_membership_erasure_job", lambda *_args: {
        "job_id": "merase-test", "subject_user_id": "user-target",
        "status": "completed", "attempts": 2, "last_error_code": None,
    })
    delete_response = preference_client.request(
        "DELETE", "/auth/members/user-target", json={
            "confirmation_phrase": "REVOKE_MEMBER_AND_ERASE_MEMORY",
            "idempotency_key": "erase-member-1",
        },
    )
    assert delete_response.status_code == 202
    assert delete_response.json() == {
        "job_id": "merase-test", "subject_user_id": "user-target",
        "status": "pending", "membership_revoked": True,
    }
    assert calls == [(context, "user-target", "erase-member-1")]

    status_response = preference_client.get("/auth/members/erasure-jobs/merase-test")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "completed"


def test_membership_erasure_dead_letter_has_explicit_retry_route(preference_client, monkeypatch):
    monkeypatch.setattr(auth_routes, "require_permission", lambda *_args: None)
    monkeypatch.setattr(auth_routes, "retry_membership_erasure", lambda *_args: {
        "job_id": "merase-test", "subject_user_id": "user-target",
        "status": "pending", "attempts": 0,
    })
    response = preference_client.post("/auth/members/erasure-jobs/merase-test/retry")
    assert response.status_code == 200
    assert response.json() == {
        "job_id": "merase-test", "subject_user_id": "user-target",
        "status": "pending", "attempts": 0,
    }

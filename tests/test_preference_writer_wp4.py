"""WP4 preference write-path contract tests.

The PostgreSQL integration test runs only against an explicitly supplied,
already migrated disposable database via ``WP4_TEST_POSTGRES_URL``.
"""

from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from sqlalchemy.engine import make_url

import pytest
from sqlalchemy import create_engine, text

from app.memory.preference_writer import (
    PreferenceWriteConflict,
    PreferenceWriteRejected,
    PreferenceWriter,
)
from app.memory.preference_worker import cleanup_expired_state, run_one
from app.memory.preference_worker import _exception_code
from app.memory.preference_worker import _unique_reconciled_hit
from app.memory import preference_worker
from app.memory.preference_provider import Mem0PreferenceProvider
from app.memory.mem0_adapter import Mem0Unavailable
from app.memory.policy import MemoryPolicy
from app.memory.preference_payload import (
    PreferencePayloadError,
    decrypt_preference_payload,
    encrypt_preference_payload,
    payload_key_material,
)
from app.tenancy import TenantContext


class _FakePreferenceAdapter:
    """Deterministic external Mem0 boundary for the PostgreSQL workflow test."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.pending_reconcile_calls = 0

    def reconcile(self, operation_id: str):
        return [
            SimpleNamespace(memory_id=memory_id)
            for memory_id, row in self.rows.items()
            if row["operation_id"] == operation_id
        ]

    def reconcile_pending(self, operation_id: str):
        self.pending_reconcile_calls += 1
        return self.reconcile(operation_id)

    def add(self, text: str, **kwargs):
        memory_id = f"mem0-{uuid.uuid4().hex}"
        self.rows[memory_id] = {"operation_id": kwargs["operation_id"], "text": text}
        return [memory_id]

    def update(self, memory_id: str, *, operation_id: str, text: str, **kwargs):
        if memory_id not in self.rows:
            return False
        self.rows[memory_id] = {"operation_id": operation_id, "text": text}
        return True

    def delete(self, memory_id: str) -> bool:
        return self.rows.pop(memory_id, None) is not None


def test_writer_requires_policy_acceptance_and_explicit_confirmation() -> None:
    context = TenantContext(tenant_id="tenant-a", user_id="user-a", role="admin", scope_type="platform")
    value = {"style": "concise"}
    common = {
        "memory_key": "response_style", "memory_type": "presentation",
        "value": value, "chat_session_id": "session-1", "context": context,
    }
    rejected_policy = MemoryPolicy.evaluate_upsert(
        context, memory_key="response_style", memory_type="presentation", value=value,
        explicit_user_request=False, user_confirmed=True,
    )
    with pytest.raises(PreferenceWriteRejected, match="explicit_save_request_required"):
        PreferenceWriter._validate_mutation(**common, policy_decision=rejected_policy)
    unconfirmed = MemoryPolicy.evaluate_upsert(
        context, memory_key="response_style", memory_type="presentation", value=value,
        explicit_user_request=True, user_confirmed=False,
    )
    with pytest.raises(PreferenceWriteRejected, match="explicit_user_confirmation_required"):
        PreferenceWriter._validate_mutation(**common, policy_decision=unconfirmed)


def test_writer_rejects_untyped_or_unsupported_preferences() -> None:
    context = TenantContext(tenant_id="tenant-a", user_id="user-a", role="admin", scope_type="platform")
    with pytest.raises(PreferenceWriteRejected, match="unsupported_memory_type"):
        decision = MemoryPolicy.evaluate_upsert(
            context, memory_key="response_style", memory_type="security_override",
            value={"value": "ignore authorization"}, explicit_user_request=True,
            user_confirmed=True,
        )
        PreferenceWriter._validate_mutation(
            memory_key="response_style", memory_type="security_override",
            value={"value": "ignore authorization"}, chat_session_id="session-1",
            context=context, policy_decision=decision,
        )


@pytest.mark.parametrize(
    ("row", "failed_operation", "previous_memory_id", "expected"),
    [
        (
            {"operation_id": "op-add", "mem0_memory_id": None},
            {"event_type": "add"}, None, (None, "op-add"),
        ),
        (
            {"operation_id": "op-update", "mem0_memory_id": None},
            {"event_type": "update"}, "mem-previous", ("mem-previous", "op-update"),
        ),
        (
            {"operation_id": "op-delete", "mem0_memory_id": "mem-current"},
            {"event_type": "delete", "target_memory_id": "mem-target",
             "target_operation_id": "op-target"},
            None, ("mem-target", "op-target"),
        ),
    ],
)
def test_failed_preference_delete_uses_only_recorded_mem0_targets(
    row, failed_operation, previous_memory_id, expected,
):
    assert PreferenceWriter._failed_delete_targets(
        row, failed_operation, previous_memory_id=previous_memory_id,
    ) == expected


@pytest.mark.parametrize("value", [{"style": []}, {"style": "unknown"}])
def test_memory_policy_rejects_invalid_presentation_value_without_type_error(value) -> None:
    context = TenantContext(tenant_id="tenant-a", user_id="user-a", role="admin", scope_type="platform")
    decision = MemoryPolicy.evaluate_upsert(
        context, memory_key="response_style", memory_type="presentation", value=value,
        explicit_user_request=True, user_confirmed=True,
    )
    assert decision.accepted is False
    assert decision.reason == "preference_value_schema_invalid"


def test_memory_policy_rejects_invalid_scope_dates_and_reversed_period() -> None:
    context = TenantContext(tenant_id="tenant-a", user_id="user-a", role="admin", scope_type="platform")
    for value in (
        {"date": "2026-99"},
        {"start_date": "2026-08", "end_date": "2026-07"},
    ):
        decision = MemoryPolicy.evaluate_upsert(
            context, memory_key="default_date" if "date" in value else "default_period",
            memory_type="scope_default", value=value,
            explicit_user_request=True, user_confirmed=True,
        )
        assert decision.accepted is False
        assert decision.reason == "preference_value_schema_invalid"
    with pytest.raises(PreferenceWriteRejected, match="typed_preference_value_required"):
        decision = MemoryPolicy.evaluate_upsert(
            context, memory_key="response_style", memory_type="presentation",
            value="free text", explicit_user_request=True, user_confirmed=True,
        )
        PreferenceWriter._validate_mutation(
            memory_key="response_style", memory_type="presentation", value="free text",
            chat_session_id="session-1", context=context, policy_decision=decision,
        )


def test_outbox_payload_encryption_authenticates_identity_and_content() -> None:
    key = "ab" * 32
    encrypted = encrypt_preference_payload(
        {"style": "concise"}, key=key, tenant_id="tenant-a", user_id="user-a",
        operation_id="op-a",
    )
    assert decrypt_preference_payload(
        encrypted.ciphertext, encrypted.nonce, key=key,
        key_version=encrypted.key_version, tenant_id="tenant-a", user_id="user-a",
        operation_id="op-a",
    ) == {"style": "concise"}
    with pytest.raises(PreferencePayloadError, match="payload_authentication_failed"):
        decrypt_preference_payload(
            encrypted.ciphertext, encrypted.nonce, key=key,
            key_version=encrypted.key_version, tenant_id="tenant-b", user_id="user-a",
            operation_id="op-a",
        )
    tampered = encrypted.ciphertext[:-1] + bytes([encrypted.ciphertext[-1] ^ 1])
    with pytest.raises(PreferencePayloadError, match="payload_authentication_failed"):
        decrypt_preference_payload(
            tampered, encrypted.nonce, key=key,
            key_version=encrypted.key_version, tenant_id="tenant-a", user_id="user-a",
            operation_id="op-a",
        )


def test_payload_keyring_keeps_old_versions_readable_during_rotation() -> None:
    keyring = {"v1": "ab" * 32, "v2": "cd" * 32}
    encrypted = encrypt_preference_payload(
        {"style": "balanced"}, key=keyring, key_version="v2",
        tenant_id="tenant-a", user_id="user-a", operation_id="op-rotation",
    )
    assert encrypted.key_version == "v2"
    assert decrypt_preference_payload(
        encrypted.ciphertext, encrypted.nonce, key=keyring,
        key_version="v2", tenant_id="tenant-a", user_id="user-a",
        operation_id="op-rotation",
    ) == {"style": "balanced"}
    with pytest.raises(PreferencePayloadError, match="payload_key_version_unsupported"):
        decrypt_preference_payload(
            encrypted.ciphertext, encrypted.nonce, key={"v1": keyring["v1"]},
            key_version="v2", tenant_id="tenant-a", user_id="user-a",
            operation_id="op-rotation",
        )
    configured, current = payload_key_material(SimpleNamespace(
        memory_preference_payload_keyring='{"v1":"ab", "v2":"cd"}',
        memory_preference_payload_current_key_version="v2",
    ))
    assert configured == {"v1": "ab", "v2": "cd"}
    assert current == "v2"


def test_worker_preserves_safe_runtime_error_codes_without_persisting_raw_messages() -> None:
    assert _exception_code(RuntimeError("mem0_operation_result_not_reconciled")) == (
        "mem0_operation_result_not_reconciled"
    )
    assert _exception_code(RuntimeError("raw value: customer@example.com")) == "RuntimeError"


def test_same_session_pending_overlay_survives_mem0_failure_and_is_returned_first(monkeypatch) -> None:
    context = TenantContext(tenant_id="tenant-a", user_id="user-a", role="admin", scope_type="platform")

    class _Registry:
        def list_pending_overlays(self, *, context, chat_session_id):
            assert chat_session_id == "session-a"
            return [{
                "overlay_id": "ovl-a", "memory_key": "response_style",
                "memory_type": "presentation", "value": {"style": "concise"},
                "source_turn_id": "turn-a", "operation_id": "op-a",
            }]

    class _Adapter:
        def search(self, *args, **kwargs):
            raise Mem0Unavailable("qdrant_unavailable")

    provider = Mem0PreferenceProvider(
        context, adapter=_Adapter(), registry=_Registry(),
        settings=SimpleNamespace(mem0_top_k_per_type=5, mem0_top_k_total=10, mem0_relevance_threshold=0.35),
    )
    monkeypatch.setattr("app.memory.preference_provider.get_personalization_enabled", lambda _context: True)
    result = provider.search(
        need={"memory_types": ["presentation"]},
        state={"user_message": "按我刚说的风格", "chat_session_id": "session-a"},
    )
    assert result == [{
        "memory_key": "response_style", "memory_type": "presentation",
        "value": {"style": "concise"}, "source_ref": "overlay:ovl-a",
        "confidence": 1.0, "source_turn_id": "turn-a", "operation_id": "op-a",
        "source": "pending_overlay",
    }]


def test_disabled_personalization_skips_provider_reads(monkeypatch) -> None:
    context = TenantContext(tenant_id="tenant-a", user_id="user-a", role="admin", scope_type="platform")

    class Registry:
        def ensure_available(self):
            raise AssertionError("disabled personalization must not consult Registry")

    class Adapter:
        def search(self, *args, **kwargs):
            raise AssertionError("disabled personalization must not call Mem0")

    monkeypatch.setattr("app.memory.preference_provider.get_personalization_enabled", lambda _context: False)
    provider = Mem0PreferenceProvider(
        context, adapter=Adapter(), registry=Registry(),
        settings=SimpleNamespace(mem0_top_k_per_type=5, mem0_top_k_total=10, mem0_relevance_threshold=0.35),
    )
    assert provider.search(
        need={"memory_types": ["presentation"]}, state={"user_message": "按我习惯的方式"},
    ) == []


def test_worker_default_adapter_uses_postgres_registry(monkeypatch) -> None:
    from app.memory.preference_registry import PostgresPreferenceRegistry

    db = object()
    lease = {
        "tenant_id": "tenant-a", "user_id": "user-a", "seller_id": None,
        "event_type": "add",
    }
    captured = {}

    class _Adapter:
        def __init__(self, context, *, registry):
            captured["registry"] = registry

    monkeypatch.setattr(preference_worker, "_claim", lambda _db, lease_seconds: lease)
    monkeypatch.setattr(preference_worker, "run_membership_erasure_once", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(preference_worker, "run_account_erasure_once", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(preference_worker, "Mem0PreferenceAdapter", _Adapter)
    monkeypatch.setattr(
        preference_worker, "get_settings",
        lambda: SimpleNamespace(mem0_enabled=True, memory_preference_worker_lease_seconds=60),
    )

    def deliver(_lease, engine, adapter_factory):
        assert engine is db
        context = TenantContext(tenant_id="tenant-a", user_id="user-a", role="admin", scope_type="platform")
        adapter_factory(context)

    monkeypatch.setattr(preference_worker, "_deliver", deliver)
    assert run_one(engine_override=db) is True
    assert isinstance(captured["registry"], PostgresPreferenceRegistry)
    assert captured["registry"]._engine is db


def test_worker_prefers_registry_pending_reconciliation_path() -> None:
    calls = {"pending": 0, "normal": 0}

    class _Adapter:
        def reconcile_pending(self, operation_id):
            calls["pending"] += 1
            return []

        def reconcile(self, operation_id):
            calls["normal"] += 1
            raise AssertionError("worker must not use Registry-gated reconcile")

    assert _unique_reconciled_hit(_Adapter(), "operation-1") is None
    assert calls == {"pending": 1, "normal": 0}
@pytest.mark.skipif(
    not (os.getenv("WP4_TEST_POSTGRES_URL") or os.getenv("POSTGRES_URL")),
    reason="Set WP4_TEST_POSTGRES_URL to a disposable PostgreSQL database migrated to head",
)
def test_registry_overlay_outbox_mutations_are_atomic_and_idempotent(monkeypatch) -> None:
    engine = create_engine(os.getenv("WP4_TEST_POSTGRES_URL") or os.environ["POSTGRES_URL"])
    suffix = uuid.uuid4().hex[:12]
    tenant = f"wp4-{suffix}"
    user = f"wp4-user-{suffix}"
    other_user = f"wp4-other-{suffix}"
    session = f"wp4-session-{suffix}"
    other_session = f"wp4-other-session-{suffix}"
    rls_role = f"wp4_rls_{suffix}"
    worker_role = f"wp4_worker_{suffix}"
    worker_password = uuid.uuid4().hex
    worker_engine = None
    context = TenantContext(tenant_id=tenant, user_id=user, role="admin", scope_type="platform")
    writer = PreferenceWriter(context, engine_override=engine, payload_key="ab" * 32)
    adapter = _FakePreferenceAdapter()

    def adapter_factory(_context):
        return adapter

    try:
        with engine.begin() as connection:
            connection.execute(text(f'CREATE ROLE "{worker_role}" LOGIN PASSWORD \'{worker_password}\' BYPASSRLS'))
            connection.execute(text(f'GRANT USAGE ON SCHEMA governance TO "{worker_role}"'))
            connection.execute(text(f'''GRANT SELECT, UPDATE ON governance.memory_registry,
                governance.memory_preference_overlays, governance.memory_outbox
                TO "{worker_role}"'''))
            connection.execute(text(f'GRANT DELETE ON governance.memory_registry, governance.memory_preference_overlays TO "{worker_role}"'))
            connection.execute(text(f'GRANT DELETE ON governance.memory_outbox TO "{worker_role}"'))
            connection.execute(text(f'GRANT INSERT ON governance.memory_preference_audit TO "{worker_role}"'))
            connection.execute(text(f'GRANT SELECT ON governance.memory_preference_settings TO "{worker_role}"'))
            connection.execute(text(f'GRANT SELECT, UPDATE ON governance.memory_membership_erasure_jobs TO "{worker_role}"'))
            connection.execute(text(f'GRANT SELECT, UPDATE ON governance.account_erasure_jobs TO "{worker_role}"'))
            connection.execute(text("""
                INSERT INTO iam.tenants(tenant_id, tenant_name, scope_type, status)
                VALUES (:tenant, 'WP4 writer test', 'platform', 'active')
            """), {"tenant": tenant})
            connection.execute(text("""
                INSERT INTO iam.users(user_id, display_name, email, status)
                VALUES (:user, 'WP4 writer test', :email, 'active')
            """), {"user": user, "email": f"{user}@example.invalid"})
            connection.execute(text("""
                INSERT INTO iam.tenant_memberships(tenant_id, user_id, role)
                VALUES (:tenant, :user, 'admin')
            """), {"tenant": tenant, "user": user})
            connection.execute(text("""
                INSERT INTO governance.chat_sessions
                    (tenant_id, user_id, chat_session_id, title)
                VALUES (:tenant, :user, :session, 'WP4 writer test')
            """), {"tenant": tenant, "user": user, "session": session})
            connection.execute(text("""
                INSERT INTO governance.memory_preference_settings
                    (tenant_id, user_id, personalization_enabled)
                VALUES (:tenant, :user, false)
            """), {"tenant": tenant, "user": user})
        disabled_policy = MemoryPolicy.evaluate_upsert(
            context, memory_key="response_style", memory_type="presentation",
            value={"style": "concise"}, explicit_user_request=True, user_confirmed=True,
        )
        with pytest.raises(PreferenceWriteRejected, match="personalization_disabled"):
            writer.queue_upsert(
                memory_key="response_style", memory_type="presentation",
                value={"style": "concise"}, chat_session_id=session,
                source_turn_id=f"turn-disabled-{suffix}", expected_version=0,
                idempotency_key=f"disabled-{suffix}", policy_decision=disabled_policy,
            )
        with engine.begin() as connection:
            connection.execute(text("""
                UPDATE governance.memory_preference_settings
                   SET personalization_enabled=true
                 WHERE tenant_id=:tenant AND user_id=:user
            """), {"tenant": tenant, "user": user})
        worker_url = make_url(os.getenv("WP4_TEST_POSTGRES_URL") or os.environ["POSTGRES_URL"]).set(
            username=worker_role, password=worker_password,
        )
        worker_engine = create_engine(worker_url)

        create_policy = MemoryPolicy.evaluate_upsert(
            context, memory_key="response_style", memory_type="presentation",
            value={"style": "concise"}, explicit_user_request=True, user_confirmed=True,
        )
        created = writer.queue_upsert(
            memory_key="response_style", memory_type="presentation",
            value={"style": "concise"}, chat_session_id=session,
            source_turn_id=f"turn-{suffix}", expected_version=0,
            idempotency_key=f"create-{suffix}", policy_decision=create_policy,
        )
        replay = writer.queue_upsert(
            memory_key="response_style", memory_type="presentation",
            value={"style": "concise"}, chat_session_id=session,
            source_turn_id=f"turn-{suffix}", expected_version=0,
            idempotency_key=f"create-{suffix}", policy_decision=create_policy,
        )
        assert created.status == "pending_create"
        assert replay.changed is False
        assert replay.operation_id == created.operation_id
        assert replay.outbox_event_id == created.outbox_event_id
        with engine.begin() as connection:
            encrypted_row = connection.execute(text("""
                SELECT payload_ciphertext, payload_nonce, payload_key_version
                  FROM governance.memory_outbox WHERE event_id=:event_id
            """), {"event_id": created.outbox_event_id}).mappings().one()
            assert b"concise" not in bytes(encrypted_row["payload_ciphertext"])
            assert encrypted_row["payload_key_version"] == "v1"
        assert run_one(engine_override=worker_engine, adapter_factory=adapter_factory) is True
        assert len(adapter.rows) == 1
        assert adapter.pending_reconcile_calls > 0
        with engine.begin() as connection:
            created_row = connection.execute(text("""
                SELECT status, mem0_memory_id FROM governance.memory_registry
                 WHERE registry_id=:registry_id
            """), {"registry_id": created.registry_id}).mappings().one()
            overlay_row = connection.execute(text("""
                SELECT status, value FROM governance.memory_preference_overlays
                 WHERE overlay_id=:overlay_id
            """), {"overlay_id": created.overlay_id}).mappings().one()
            assert created_row["status"] == "active"
            assert created_row["mem0_memory_id"] in adapter.rows
            first_memory_id = created_row["mem0_memory_id"]
            assert overlay_row["status"] == "cleared"
            assert overlay_row["value"] == {}

        reconciliation = writer.queue_reconciliation(
            registry_id=created.registry_id, expected_version=1,
            idempotency_key=f"reconcile-{suffix}",
        )
        reconciliation_replay = writer.queue_reconciliation(
            registry_id=created.registry_id, expected_version=1,
            idempotency_key=f"reconcile-{suffix}",
        )
        assert reconciliation.status == "pending_update"
        assert reconciliation_replay.changed is False
        with engine.begin() as connection:
            reconcile_row = connection.execute(text("""
                SELECT event_type, overlay_id FROM governance.memory_outbox
                 WHERE event_id=:event_id
            """), {"event_id": reconciliation.outbox_event_id}).mappings().one()
            assert reconcile_row["event_type"] == "reconcile"
            assert reconcile_row["overlay_id"] is None
        assert run_one(engine_override=worker_engine, adapter_factory=adapter_factory) is True
        assert len([row for row in adapter.rows.values() if row["operation_id"] == created.operation_id]) == 1
        with engine.begin() as connection:
            reconciled = connection.execute(text("""
                SELECT r.status, o.status AS outbox_status
                  FROM governance.memory_registry r
                  JOIN governance.memory_outbox o ON o.registry_id=r.registry_id
                 WHERE o.event_id=:event_id
            """), {"event_id": reconciliation.outbox_event_id}).mappings().one()
            assert reconciled["status"] == "active"
            assert reconciled["outbox_status"] == "succeeded"

        # Inject the exact external-success / Registry-commit crash boundary.
        crash_policy = MemoryPolicy.evaluate_upsert(
            context, memory_key="response_language", memory_type="presentation",
            value={"language": "zh-CN"}, explicit_user_request=True, user_confirmed=True,
        )
        crash_operation = writer.queue_upsert(
            memory_key="response_language", memory_type="presentation",
            value={"language": "zh-CN"}, chat_session_id=session,
            source_turn_id=f"turn-crash-{suffix}", expected_version=0,
            idempotency_key=f"crash-{suffix}", policy_decision=crash_policy,
        )
        complete = preference_worker._complete

        def fail_after_external_write(*args, **kwargs):
            raise RuntimeError("injected_registry_commit_failure")

        monkeypatch.setattr(preference_worker, "_complete", fail_after_external_write)
        assert run_one(engine_override=worker_engine, adapter_factory=adapter_factory) is True
        assert len([row for row in adapter.rows.values() if row["operation_id"] == crash_operation.operation_id]) == 1
        monkeypatch.setattr(preference_worker, "_complete", complete)
        with engine.begin() as connection:
            connection.execute(text("""
                UPDATE governance.memory_outbox SET next_attempt_at=now()
                 WHERE event_id=:event_id AND status='pending'
            """), {"event_id": crash_operation.outbox_event_id})
        assert run_one(engine_override=worker_engine, adapter_factory=adapter_factory) is True
        assert len([row for row in adapter.rows.values() if row["operation_id"] == crash_operation.operation_id]) == 1
        with engine.begin() as connection:
            recovered = connection.execute(text("""
                SELECT r.status, r.mem0_memory_id, o.status AS outbox_status
                  FROM governance.memory_registry r
                  JOIN governance.memory_outbox o USING (registry_id)
                 WHERE r.registry_id=:registry_id
            """), {"registry_id": crash_operation.registry_id}).mappings().one()
            assert recovered["status"] == "active"
            assert recovered["outbox_status"] == "succeeded"
            assert recovered["mem0_memory_id"] in adapter.rows

        update_policy = MemoryPolicy.evaluate_upsert(
            context, memory_key="response_style", memory_type="presentation",
            value={"style": "detailed"}, explicit_user_request=True, user_confirmed=True,
        )
        updated = writer.queue_upsert(
            memory_key="response_style", memory_type="presentation",
            value={"style": "detailed"}, chat_session_id=session,
            source_turn_id=f"turn-2-{suffix}", expected_version=1,
            idempotency_key=f"update-{suffix}", policy_decision=update_policy,
        )
        assert updated.status == "pending_update"
        assert run_one(engine_override=worker_engine, adapter_factory=adapter_factory) is True
        assert len(adapter.rows) == 2
        with engine.begin() as connection:
            states = connection.execute(text("""
                SELECT registry_id, status FROM governance.memory_registry
                 WHERE tenant_id=:tenant AND user_id=:user
                   AND memory_key='response_style' ORDER BY version
            """), {"tenant": tenant, "user": user}).mappings().all()
            assert [(row["registry_id"], row["status"]) for row in states] == [
                (created.registry_id, "superseded"), (updated.registry_id, "active")
            ]
            outbox = connection.execute(text("""
                SELECT event_type, target_memory_id, status
                  FROM governance.memory_outbox WHERE event_id=:event_id
            """), {"event_id": updated.outbox_event_id}).mappings().one()
            assert dict(outbox) == {
                "event_type": "update", "target_memory_id": first_memory_id, "status": "succeeded"
            }
            assert "payload" not in {
                row["column_name"] for row in connection.execute(text("""
                    SELECT column_name FROM information_schema.columns
                     WHERE table_schema='governance' AND table_name='memory_outbox'
                """)).mappings().all()
            }

        with pytest.raises(PreferenceWriteConflict, match="registry_version_conflict"):
            writer.queue_upsert(
                memory_key="response_style", memory_type="presentation",
                value={"style": "balanced"}, chat_session_id=session,
                source_turn_id=f"turn-3-{suffix}", expected_version=1,
                idempotency_key=f"stale-{suffix}", policy_decision=MemoryPolicy.evaluate_upsert(
                    context, memory_key="response_style", memory_type="presentation",
                    value={"style": "balanced"}, explicit_user_request=True, user_confirmed=True,
                ),
            )

        deleted = writer.queue_delete(
            registry_id=updated.registry_id, expected_version=2,
            idempotency_key=f"delete-{suffix}", policy_decision=MemoryPolicy.evaluate_delete(
                context, registry_id=updated.registry_id, user_confirmed=True,
            ),
        )
        assert run_one(engine_override=worker_engine, adapter_factory=adapter_factory) is True
        assert [row["operation_id"] for row in adapter.rows.values()] == [crash_operation.operation_id]
        delete_replay = writer.queue_delete(
            registry_id=updated.registry_id, expected_version=2,
            idempotency_key=f"delete-{suffix}", policy_decision=MemoryPolicy.evaluate_delete(
                context, registry_id=updated.registry_id, user_confirmed=True,
            ),
        )
        assert deleted.status == "pending_delete"
        assert delete_replay.changed is False
        with engine.begin() as connection:
            row = connection.execute(text("""
                SELECT status, operation_id FROM governance.memory_registry
                 WHERE registry_id=:registry_id
            """), {"registry_id": updated.registry_id}).mappings().one()
            assert row["status"] == "deleted"
            assert row["operation_id"] == deleted.operation_id
            counts = connection.execute(text("""
                SELECT
                    (SELECT count(*) FROM governance.memory_preference_overlays
                      WHERE tenant_id=:tenant AND user_id=:user) AS overlays,
                    (SELECT count(*) FROM governance.memory_outbox
                      WHERE tenant_id=:tenant AND user_id=:user) AS outbox,
                    (SELECT count(*) FROM governance.memory_preference_audit
                      WHERE tenant_id=:tenant AND user_id=:user) AS audit
            """), {"tenant": tenant, "user": user}).mappings().one()
            assert counts["overlays"] == 3
            assert counts["outbox"] == 4
            assert counts["audit"] == 11

        session_close_policy = MemoryPolicy.evaluate_upsert(
            context, memory_key="report_style", memory_type="presentation",
            value={"format": "table"}, explicit_user_request=True, user_confirmed=True,
        )
        after_session_close = writer.queue_upsert(
            memory_key="report_style", memory_type="presentation",
            value={"format": "table"}, chat_session_id=session,
            source_turn_id=f"turn-session-close-{suffix}", expected_version=0,
            idempotency_key=f"session-close-{suffix}", policy_decision=session_close_policy,
        )
        with engine.begin() as connection:
            connection.execute(text("""
                DELETE FROM governance.chat_sessions
                 WHERE tenant_id=:tenant AND user_id=:user AND chat_session_id=:session
            """), {"tenant": tenant, "user": user, "session": session})
            pending = connection.execute(text("""
                SELECT overlay_id, payload_ciphertext FROM governance.memory_outbox
                 WHERE event_id=:event_id
            """), {"event_id": after_session_close.outbox_event_id}).mappings().one()
            assert pending["overlay_id"] is None
            assert pending["payload_ciphertext"]
        assert run_one(engine_override=worker_engine, adapter_factory=adapter_factory) is True
        with engine.begin() as connection:
            closed_session_result = connection.execute(text("""
                SELECT status FROM governance.memory_registry WHERE registry_id=:registry_id
            """), {"registry_id": after_session_close.registry_id}).scalar_one()
            assert closed_session_result == "active"

        # A user can revoke a just-created preference before the worker delivers it.
        scope_session = f"wp4-scope-session-{suffix}"
        with engine.begin() as connection:
            connection.execute(text("""
                INSERT INTO governance.chat_sessions
                    (tenant_id, user_id, chat_session_id, title)
                VALUES (:tenant, :user, :session, 'WP4 scope cancellation test')
            """), {"tenant": tenant, "user": user, "session": scope_session})
        scope_policy = MemoryPolicy.evaluate_upsert(
            context, memory_key="default_period", memory_type="scope_default",
            value={"start_date": "2026-07", "end_date": "2026-09"},
            explicit_user_request=True, user_confirmed=True,
        )
        pending_scope = writer.queue_upsert(
            memory_key="default_period", memory_type="scope_default",
            value={"start_date": "2026-07", "end_date": "2026-09"},
            chat_session_id=scope_session, source_turn_id=f"turn-scope-{suffix}",
            expected_version=0, idempotency_key=f"scope-create-{suffix}",
            policy_decision=scope_policy,
        )
        pending_delete = writer.queue_delete(
            registry_id=pending_scope.registry_id, expected_version=1,
            idempotency_key=f"scope-delete-{suffix}",
            policy_decision=MemoryPolicy.evaluate_delete(
                context, registry_id=pending_scope.registry_id, user_confirmed=True,
            ),
        )
        assert pending_delete.status == "pending_delete"
        assert run_one(engine_override=worker_engine, adapter_factory=adapter_factory) is True
        with engine.begin() as connection:
            cancelled_registry_status = connection.execute(text("""
                SELECT status FROM governance.memory_registry WHERE registry_id=:registry_id
            """), {"registry_id": pending_scope.registry_id}).scalar_one()
            cancelled_rows = connection.execute(text("""
                SELECT event_type, status FROM governance.memory_outbox
                 WHERE registry_id=:registry_id ORDER BY created_at
            """), {"registry_id": pending_scope.registry_id}).mappings().all()
            assert cancelled_registry_status == "deleted"
            assert [(row["event_type"], row["status"]) for row in cancelled_rows] == [
                ("add", "succeeded"), ("delete", "succeeded"),
            ]

        expired_registry_id = f"mreg-expired-{suffix}"
        expired_operation_id = f"op-expired-{suffix}"
        expired_overlay_id = f"ovl-expired-{suffix}"
        with engine.begin() as connection:
            connection.execute(text("""
                INSERT INTO governance.memory_registry
                    (registry_id, tenant_id, user_id, memory_type, memory_key,
                     status, version, operation_id, expires_at)
                VALUES (:registry_id, :tenant, :user, 'scope_default', 'default_date',
                        'active', 1, :operation_id, now()-interval '1 day')
            """), {
                "registry_id": expired_registry_id, "tenant": tenant, "user": user,
                "operation_id": expired_operation_id,
            })
            connection.execute(text("""
                INSERT INTO governance.memory_preference_overlays
                    (overlay_id, tenant_id, user_id, chat_session_id, registry_id,
                     memory_key, memory_type, value, operation_id, version,
                     status, expires_at)
                VALUES (:overlay_id, :tenant, :user, :session, :registry_id,
                        'default_date', 'scope_default', '{"date":"2026-01"}'::jsonb,
                        :operation_id, 1, 'pending', now()-interval '1 day')
            """), {
                "overlay_id": expired_overlay_id, "tenant": tenant, "user": user,
                "session": scope_session, "registry_id": expired_registry_id,
                "operation_id": expired_operation_id,
            })
        cleanup_expired_state(engine)
        with engine.begin() as connection:
            assert connection.execute(text("""
                SELECT status FROM governance.memory_registry WHERE registry_id=:registry_id
            """), {"registry_id": expired_registry_id}).scalar_one() == "superseded"
            expired_overlay = connection.execute(text("""
                SELECT status, value FROM governance.memory_preference_overlays
                 WHERE overlay_id=:overlay_id
            """), {"overlay_id": expired_overlay_id}).mappings().one()
            assert expired_overlay["status"] == "expired"
            assert expired_overlay["value"] == {}
            connection.execute(text("""
                UPDATE governance.memory_registry SET mem0_memory_id='mem0-expired-scope'
                 WHERE registry_id=:registry_id
            """), {"registry_id": expired_registry_id})

        reconfirm_policy = MemoryPolicy.evaluate_upsert(
            context, memory_key="default_date", memory_type="scope_default",
            value={"date": "2026-02"}, explicit_user_request=True, user_confirmed=True,
        )
        reconfirmed = writer.queue_upsert(
            memory_key="default_date", memory_type="scope_default",
            value={"date": "2026-02"}, chat_session_id=scope_session,
            source_turn_id=f"turn-reconfirm-{suffix}", expected_version=1,
            idempotency_key=f"reconfirm-{suffix}", policy_decision=reconfirm_policy,
        )
        assert reconfirmed.status == "pending_update"
        with engine.begin() as connection:
            reconfirm_outbox = connection.execute(text("""
                SELECT event_type, target_memory_id, target_operation_id
                  FROM governance.memory_outbox WHERE event_id=:event_id
            """), {"event_id": reconfirmed.outbox_event_id}).mappings().one()
            assert dict(reconfirm_outbox) == {
                "event_type": "update", "target_memory_id": "mem0-expired-scope",
                "target_operation_id": expired_operation_id,
            }
            expired_delete_id = f"mreg-expired-delete-{suffix}"
            expired_delete_operation = f"op-expired-delete-{suffix}"
            connection.execute(text("""
                INSERT INTO governance.memory_registry
                    (registry_id, tenant_id, user_id, memory_type, memory_key,
                     mem0_memory_id, status, version, operation_id, expires_at)
                    VALUES (:registry_id, :tenant, :user, 'scope_default', 'default_period',
                            'mem0-expired-delete', 'superseded', 2, :operation_id,
                        now()-interval '1 day')
            """), {
                "registry_id": expired_delete_id, "tenant": tenant, "user": user,
                "operation_id": expired_delete_operation,
            })
            connection.execute(text("""
                INSERT INTO governance.memory_registry
                    (registry_id, tenant_id, user_id, memory_type, memory_key,
                     status, version, operation_id, created_at, updated_at)
                    VALUES (:registry_id, :tenant, :user, 'analysis_method', 'analysis_steps',
                        'deleted', 1, :operation_id,
                        now()-interval '181 days', now()-interval '181 days')
            """), {
                "registry_id": f"mreg-retention-{suffix}", "tenant": tenant,
                "user": user, "operation_id": f"op-retention-{suffix}",
            })
            connection.execute(text("""
                INSERT INTO governance.memory_preference_overlays
                    (overlay_id, tenant_id, user_id, chat_session_id, registry_id,
                     memory_key, memory_type, value, operation_id, version,
                     status, created_at, expires_at)
                VALUES (:overlay_id, :tenant, :user, :session, :registry_id,
                        'analysis_steps', 'analysis_method', '{}'::jsonb,
                        :operation_id, 1, 'cleared', now()-interval '181 days', now()-interval '180 days')
            """), {
                "overlay_id": f"ovl-retention-{suffix}", "tenant": tenant,
                "user": user, "session": scope_session,
                "registry_id": f"mreg-retention-{suffix}",
                "operation_id": f"op-retention-overlay-{suffix}",
            })
        expired_delete = writer.queue_delete(
            registry_id=expired_delete_id, expected_version=2,
            idempotency_key=f"delete-expired-{suffix}",
            policy_decision=MemoryPolicy.evaluate_delete(
                context, registry_id=expired_delete_id, user_confirmed=True,
            ),
        )
        assert expired_delete.status == "pending_delete"
        with engine.begin() as connection:
            expired_delete_outbox = connection.execute(text("""
                SELECT target_memory_id, target_operation_id
                  FROM governance.memory_outbox WHERE event_id=:event_id
            """), {"event_id": expired_delete.outbox_event_id}).mappings().one()
            assert dict(expired_delete_outbox) == {
                "target_memory_id": "mem0-expired-delete",
                "target_operation_id": expired_delete_operation,
            }
        cleanup_expired_state(engine)
        with engine.begin() as connection:
            retained_registry = connection.execute(text("""
                SELECT 1 FROM governance.memory_registry WHERE registry_id=:registry_id
            """), {"registry_id": f"mreg-retention-{suffix}"}).scalar_one_or_none()
            assert retained_registry is None

        barrier = Barrier(2)
        concurrent_values = ({"language": "zh-TW"}, {"language": "en-US"})

        def concurrent_correction(index: int):
            value = concurrent_values[index]
            policy = MemoryPolicy.evaluate_upsert(
                context, memory_key="response_language", memory_type="presentation",
                value=value, explicit_user_request=True, user_confirmed=True,
            )
            barrier.wait(timeout=10)
            try:
                return writer.queue_upsert(
                    memory_key="response_language", memory_type="presentation",
                    value=value, chat_session_id=scope_session,
                    source_turn_id=f"turn-concurrent-{index}-{suffix}", expected_version=1,
                    idempotency_key=f"concurrent-{index}-{suffix}", policy_decision=policy,
                )
            except PreferenceWriteConflict:
                return "conflict"

        with ThreadPoolExecutor(max_workers=2) as pool:
            concurrent_results = list(pool.map(concurrent_correction, (0, 1)))
        assert sum(result != "conflict" for result in concurrent_results) == 1
        assert concurrent_results.count("conflict") == 1

        # Owner management must continue to target the original seller partition
        # when the authenticated management context itself is platform scoped.
        seller_scope = f"seller-{suffix}"
        seller_policy = MemoryPolicy.evaluate_upsert(
            context, memory_key="report_style", memory_type="presentation",
            value={"format": "bullets"}, explicit_user_request=True, user_confirmed=True,
        )
        seller_owned = writer.queue_upsert(
            memory_key="report_style", memory_type="presentation",
            value={"format": "bullets"}, chat_session_id=scope_session,
            source_turn_id=f"turn-seller-{suffix}", expected_version=0,
            idempotency_key=f"seller-create-{suffix}", policy_decision=seller_policy,
            seller_id=seller_scope,
        )

        def run_until_registry_status(registry_id: str, expected_status: str) -> None:
            for _ in range(25):
                with engine.begin() as connection:
                    current_status = connection.execute(text("""
                        SELECT status FROM governance.memory_registry WHERE registry_id=:registry_id
                    """), {"registry_id": registry_id}).scalar_one()
                if current_status == expected_status:
                    return
                run_one(engine_override=worker_engine, adapter_factory=adapter_factory)
            raise AssertionError(f"registry {registry_id} did not reach {expected_status}")

        run_until_registry_status(seller_owned.registry_id, "active")
        seller_reconcile = writer.queue_reconciliation(
            registry_id=seller_owned.registry_id, expected_version=1,
            idempotency_key=f"seller-reconcile-{suffix}",
        )
        run_until_registry_status(seller_owned.registry_id, "active")
        seller_delete = writer.queue_delete(
            registry_id=seller_owned.registry_id, expected_version=1,
            idempotency_key=f"seller-delete-{suffix}", policy_decision=MemoryPolicy.evaluate_delete(
                context, registry_id=seller_owned.registry_id, user_confirmed=True,
            ),
        )
        run_until_registry_status(seller_owned.registry_id, "deleted")
        with engine.begin() as connection:
            seller_events = connection.execute(text("""
                SELECT seller_id, event_type FROM governance.memory_outbox
                 WHERE tenant_id=:tenant AND user_id=:user
                   AND registry_id=:registry_id ORDER BY created_at
            """), {
                "tenant": tenant, "user": user, "registry_id": seller_owned.registry_id,
            }).mappings().all()
            # A reconciliation may reuse the original successful Outbox row,
            # and timestamps can tie at transaction precision. Assert the
            # persisted event set rather than relying on created_at ordering.
            assert sorted((row["seller_id"], row["event_type"]) for row in seller_events) == [
                (seller_scope, "delete"),
                (seller_scope, "reconcile"),
            ]
            assert seller_reconcile.status == "pending_update"
            assert seller_delete.status == "pending_delete"

        # A dead-letter add remains revocable: deletion must reconcile its
        # original operation ID even when Registry never received a Mem0 ID.
        failed_registry_id = f"mreg-failed-revoke-{suffix}"
        failed_operation_id = f"op-failed-revoke-{suffix}"
        failed_overlay_id = f"ovl-failed-revoke-{suffix}"
        with engine.begin() as connection:
            connection.execute(text("""
                INSERT INTO governance.memory_registry
                    (registry_id, tenant_id, user_id, seller_id, memory_type,
                     memory_key, status, version, operation_id)
                VALUES (:registry_id, :tenant, :user, :seller, 'presentation',
                        'response_language', 'failed', 1, :operation_id)
            """), {
                "registry_id": failed_registry_id, "tenant": tenant, "user": user,
                "seller": seller_scope, "operation_id": failed_operation_id,
            })
            connection.execute(text("""
                INSERT INTO governance.memory_preference_overlays
                    (overlay_id, tenant_id, user_id, seller_id, chat_session_id,
                     registry_id, memory_key, memory_type, value, operation_id,
                     version, status, error_code)
                VALUES (:overlay_id, :tenant, :user, :seller, :session, :registry_id,
                        'response_language', 'presentation', '{"language":"zh-CN"}'::jsonb,
                        :operation_id, 1, 'failed', 'mem0_timeout')
            """), {
                "overlay_id": failed_overlay_id, "tenant": tenant, "user": user,
                "seller": seller_scope, "session": scope_session,
                "registry_id": failed_registry_id, "operation_id": failed_operation_id,
            })
            connection.execute(text("""
                INSERT INTO governance.memory_outbox
                    (event_id, operation_id, idempotency_key, tenant_id, user_id,
                     seller_id, event_type, registry_id, overlay_id, registry_version,
                     payload_ciphertext, payload_nonce, payload_key_version,
                     status, attempts, max_attempts, last_error_code)
                VALUES (:event_id, :operation_id, :idempotency_key, :tenant, :user,
                        :seller, 'add', :registry_id, :overlay_id, 1,
                        :ciphertext, :nonce, 'v1', 'dead_letter', 5, 5, 'mem0_timeout')
            """), {
                "event_id": f"mout-failed-revoke-{suffix}",
                "operation_id": failed_operation_id,
                "idempotency_key": f"failed-revoke-add-{suffix}",
                "tenant": tenant, "user": user, "seller": seller_scope,
                "registry_id": failed_registry_id, "overlay_id": failed_overlay_id,
                "ciphertext": b"encrypted-value", "nonce": b"nonce",
            })
        revoked_failed = writer.queue_delete(
            registry_id=failed_registry_id, expected_version=1,
            idempotency_key=f"failed-revoke-delete-{suffix}",
            policy_decision=MemoryPolicy.evaluate_delete(
                context, registry_id=failed_registry_id, user_confirmed=True,
            ),
        )
        with engine.begin() as connection:
            delete_targets = connection.execute(text("""
                SELECT target_memory_id, target_operation_id
                  FROM governance.memory_outbox WHERE event_id=:event_id
            """), {"event_id": revoked_failed.outbox_event_id}).mappings().one()
            assert dict(delete_targets) == {
                "target_memory_id": None, "target_operation_id": failed_operation_id,
            }
            cleared = connection.execute(text("""
                SELECT o.payload_ciphertext, v.status AS overlay_status, v.value
                  FROM governance.memory_outbox o
                  JOIN governance.memory_preference_overlays v USING (overlay_id)
                 WHERE o.event_id=:event_id
            """), {"event_id": f"mout-failed-revoke-{suffix}"}).mappings().one()
            assert cleared["payload_ciphertext"] is None
            assert cleared["overlay_status"] == "superseded"
            assert cleared["value"] == {}
        run_until_registry_status(failed_registry_id, "deleted")

        with engine.begin() as connection:
            connection.execute(text(f'CREATE ROLE "{rls_role}" NOLOGIN NOSUPERUSER NOBYPASSRLS'))
            connection.execute(text(f'GRANT USAGE ON SCHEMA governance TO "{rls_role}"'))
            connection.execute(text(f'''GRANT SELECT ON governance.memory_registry,
                governance.memory_preference_overlays, governance.memory_outbox,
                governance.memory_preference_audit, governance.memory_preference_settings
                TO "{rls_role}"'''))
            connection.execute(text("""
                INSERT INTO iam.users(user_id, display_name, email, status)
                VALUES (:user, 'WP4 RLS user', :email, 'active')
            """), {"user": other_user, "email": f"{other_user}@example.invalid"})
            connection.execute(text("""
                INSERT INTO governance.chat_sessions
                    (tenant_id, user_id, chat_session_id, title)
                VALUES (:tenant, :user, :session, 'WP4 RLS session')
            """), {"tenant": tenant, "user": other_user, "session": other_session})
            connection.execute(text("""
                INSERT INTO governance.memory_registry
                    (registry_id, tenant_id, user_id, memory_type, memory_key,
                     status, version, operation_id)
                VALUES (:registry_id, :tenant, :user, 'presentation', 'response_style',
                        'pending_create', 1, :operation_id)
            """), {
                "registry_id": f"mreg-other-{suffix}", "tenant": tenant,
                "user": other_user, "operation_id": f"op-other-{suffix}",
            })
            connection.execute(text("""
                INSERT INTO governance.memory_preference_overlays
                    (overlay_id, tenant_id, user_id, chat_session_id, registry_id,
                     memory_key, memory_type, value, operation_id, version)
                VALUES (:overlay_id, :tenant, :user, :session, :registry_id,
                        'response_style', 'presentation', '{"style":"concise"}'::jsonb,
                        :operation_id, 1)
            """), {
                "overlay_id": f"ovl-other-{suffix}", "tenant": tenant,
                "user": other_user, "session": other_session,
                "registry_id": f"mreg-other-{suffix}", "operation_id": f"op-other-{suffix}",
            })
            connection.execute(text("""
                INSERT INTO governance.memory_outbox
                    (event_id, operation_id, idempotency_key, tenant_id, user_id,
                     event_type, registry_id, registry_version)
                VALUES (:event_id, :operation_id, :idempotency_key, :tenant, :user,
                        'add', :registry_id, 1)
            """), {
                "event_id": f"mout-other-{suffix}", "operation_id": f"op-other-{suffix}",
                "idempotency_key": f"idem-other-{suffix}", "tenant": tenant,
                "user": other_user, "registry_id": f"mreg-other-{suffix}",
            })
            connection.execute(text("""
                INSERT INTO governance.memory_preference_audit
                    (audit_id, tenant_id, user_id, operation_id, event_type,
                     actor_user_id, status)
                VALUES (:audit_id, :tenant, :user, :operation_id, 'queued', :user, 'pending')
            """), {
                "audit_id": f"maud-other-{suffix}", "tenant": tenant,
                "user": other_user, "operation_id": f"op-other-{suffix}",
            })
            connection.execute(text("""
                INSERT INTO governance.memory_preference_settings
                    (tenant_id, user_id, personalization_enabled)
                VALUES (:tenant, :user, false)
            """), {"tenant": tenant, "user": other_user})

        with engine.connect() as connection:
            with connection.begin():
                connection.execute(text(f'SET LOCAL ROLE "{rls_role}"'))
                for visible_user, expected in ((user, (0, 0, 0, 0, 0)), (other_user, (1, 1, 1, 1, 1))):
                    connection.execute(text("SELECT set_config('app.tenant_id', :tenant, true)"), {"tenant": tenant})
                    connection.execute(text("SELECT set_config('app.user_id', :visible_user, true)"), {"visible_user": visible_user})
                    counts = connection.execute(text("""
                        SELECT
                            (SELECT count(*) FROM governance.memory_registry WHERE user_id=:target_user),
                            (SELECT count(*) FROM governance.memory_preference_overlays WHERE user_id=:target_user),
                            (SELECT count(*) FROM governance.memory_outbox WHERE user_id=:target_user),
                            (SELECT count(*) FROM governance.memory_preference_audit WHERE user_id=:target_user),
                            (SELECT count(*) FROM governance.memory_preference_settings WHERE user_id=:target_user)
                    """), {"target_user": other_user}).one()
                    assert counts == expected
    finally:
        if worker_engine is not None:
            worker_engine.dispose()
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM governance.memory_preference_audit WHERE tenant_id=:tenant"), {"tenant": tenant})
            connection.execute(text("DELETE FROM governance.memory_preference_settings WHERE tenant_id=:tenant"), {"tenant": tenant})
            connection.execute(text("DELETE FROM governance.memory_outbox WHERE tenant_id=:tenant"), {"tenant": tenant})
            connection.execute(text("DELETE FROM governance.memory_preference_overlays WHERE tenant_id=:tenant"), {"tenant": tenant})
            connection.execute(text("DELETE FROM governance.memory_registry WHERE tenant_id=:tenant"), {"tenant": tenant})
            connection.execute(text("DELETE FROM governance.chat_sessions WHERE tenant_id=:tenant"), {"tenant": tenant})
            connection.execute(text("DELETE FROM iam.tenant_memberships WHERE tenant_id=:tenant"), {"tenant": tenant})
            connection.execute(text("DELETE FROM iam.users WHERE user_id IN (:user, :other_user)"), {"user": user, "other_user": other_user})
            role_exists = connection.execute(
                text("SELECT 1 FROM pg_roles WHERE rolname=:role"), {"role": rls_role}
            ).scalar_one_or_none()
            if role_exists:
                connection.execute(text(f'REVOKE SELECT ON governance.memory_registry, governance.memory_preference_overlays, governance.memory_outbox, governance.memory_preference_audit, governance.memory_preference_settings FROM "{rls_role}"'))
                connection.execute(text(f'REVOKE USAGE ON SCHEMA governance FROM "{rls_role}"'))
                connection.execute(text(f'DROP ROLE "{rls_role}"'))
            worker_exists = connection.execute(
                text("SELECT 1 FROM pg_roles WHERE rolname=:role"), {"role": worker_role}
            ).scalar_one_or_none()
            if worker_exists:
                connection.execute(text(f'REVOKE SELECT, UPDATE ON governance.memory_registry, governance.memory_preference_overlays, governance.memory_outbox FROM "{worker_role}"'))
                connection.execute(text(f'REVOKE SELECT ON governance.memory_preference_settings FROM "{worker_role}"'))
                connection.execute(text(f'REVOKE DELETE ON governance.memory_outbox FROM "{worker_role}"'))
                connection.execute(text(f'REVOKE DELETE ON governance.memory_registry, governance.memory_preference_overlays FROM "{worker_role}"'))
                connection.execute(text(f'REVOKE SELECT, UPDATE ON governance.memory_membership_erasure_jobs FROM "{worker_role}"'))
                connection.execute(text(f'REVOKE SELECT, UPDATE ON governance.account_erasure_jobs FROM "{worker_role}"'))
                connection.execute(text(f'REVOKE INSERT ON governance.memory_preference_audit FROM "{worker_role}"'))
                connection.execute(text(f'REVOKE USAGE ON SCHEMA governance FROM "{worker_role}"'))
                connection.execute(text(f'DROP ROLE "{worker_role}"'))
            connection.execute(text("DELETE FROM governance.memory_preference_audit WHERE tenant_id=:tenant"), {"tenant": tenant})
            connection.execute(text("DELETE FROM governance.memory_outbox WHERE tenant_id=:tenant"), {"tenant": tenant})
            connection.execute(text("DELETE FROM governance.memory_preference_overlays WHERE tenant_id=:tenant"), {"tenant": tenant})
            connection.execute(text("DELETE FROM governance.memory_registry WHERE tenant_id=:tenant"), {"tenant": tenant})
            connection.execute(text("DELETE FROM governance.chat_sessions WHERE tenant_id=:tenant"), {"tenant": tenant})
            connection.execute(text("DELETE FROM iam.tenant_memberships WHERE tenant_id=:tenant"), {"tenant": tenant})
            connection.execute(text("DELETE FROM iam.users WHERE user_id=:user"), {"user": user})
            connection.execute(text("DELETE FROM iam.tenants WHERE tenant_id=:tenant"), {"tenant": tenant})
        engine.dispose()

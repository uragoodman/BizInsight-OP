"""Opt-in real PostgreSQL + Mem0 OSS + Qdrant preference worker E2E test."""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from app.api.auth import require_tenant_context
from app.api.routes import memory as memory_routes
from app.iam import service as iam_service
from app.main import app
from app.memory.mem0_adapter import Mem0PreferenceAdapter, Mem0Timeout
from app.memory.preference_provider import Mem0PreferenceProvider
from app.memory.preference_registry import PostgresPreferenceRegistry
from app.memory.preference_management import retry_failed_preference
from app.memory import preference_management, preference_writer
from app.memory import preference_worker
from app.memory.preference_worker import run_one
from app.tenancy import TenantContext
from app.observability import snapshot_metrics


pytestmark = pytest.mark.skipif(
    not all(os.getenv(key) for key in (
        "WP4_TEST_POSTGRES_URL", "MEM0_TEST_QDRANT_URL", "MEM0_TEST_HMAC_SECRET",
        "MEMORY_PREFERENCE_PAYLOAD_KEY",
    )),
    reason="Requires disposable PostgreSQL, Qdrant server, HMAC secret, and payload key",
)


def test_real_worker_crash_reconcile_update_delete_and_overlay(monkeypatch):
    from qdrant_client import QdrantClient

    database = create_engine(os.environ["WP4_TEST_POSTGRES_URL"], pool_pre_ping=True)
    suffix = uuid.uuid4().hex[:10]
    api_role = f"wp4_api_{suffix}"
    api_password = uuid.uuid4().hex
    api_engine = None
    tenant, user = f"wp4-real-{suffix}", f"wp4-real-user-{suffix}"
    manager = f"wp4-real-manager-{suffix}"
    session = f"wp4-real-session-{suffix}"
    context = TenantContext(tenant_id=tenant, user_id=user, role="admin", scope_type="platform")
    collection = f"wp4_worker_{suffix}"
    settings = SimpleNamespace(
        mem0_qdrant_collection=collection,
        mem0_qdrant_url=os.environ["MEM0_TEST_QDRANT_URL"],
        mem0_qdrant_api_key=os.getenv("MEM0_TEST_QDRANT_API_KEY", ""),
        mem0_embedding_model=os.getenv("MEM0_TEST_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
        mem0_embedding_dims=int(os.getenv("MEM0_TEST_EMBEDDING_DIMS", "512")),
        mem0_hmac_secret=os.environ["MEM0_TEST_HMAC_SECRET"], mem0_scope_version="v1",
        mem0_history_db_path="", mem0_search_timeout_seconds=5.0,
        mem0_write_timeout_seconds=10.0, mem0_breaker_failures=5,
        mem0_breaker_window_seconds=30.0, mem0_top_k_per_type=5,
        mem0_top_k_total=10, mem0_relevance_threshold=0.0,
        mem0_llm_model="", mem0_llm_api_key="", mem0_llm_base_url="",
        deepseek_api_key="", deepseek_base_url="",
        qdrant_url=os.environ["MEM0_TEST_QDRANT_URL"],
        qdrant_collection="bizinsight_knowledge", rag_qdrant_active_alias="",
        metric_semantic_qdrant_collection="metric_semantic_catalog_v1",
    )
    adapter: Mem0PreferenceAdapter | None = None
    qdrant = QdrantClient(url=os.environ["MEM0_TEST_QDRANT_URL"], timeout=10)
    worker_complete = preference_worker._complete
    try:
        # Wait for the CI service endpoint without masking an unavailable service.
        deadline = time.monotonic() + 30
        while True:
            try:
                qdrant.get_collections()
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.5)

        with database.begin() as connection:
            connection.execute(text("""
                INSERT INTO iam.tenants(tenant_id, tenant_name, scope_type, status)
                VALUES (:tenant, 'WP4 real worker E2E', 'platform', 'active')
            """), {"tenant": tenant})
            connection.execute(text("""
                INSERT INTO iam.users(user_id, display_name, email, status)
                VALUES (:user, 'WP4 real worker E2E', :email, 'active')
            """), {"user": user, "email": f"{user}@example.invalid"})
            connection.execute(text("""
                INSERT INTO iam.users(user_id, display_name, email, status)
                VALUES (:user, 'WP4 real worker E2E manager', :email, 'active')
            """), {"user": manager, "email": f"{manager}@example.invalid"})
            connection.execute(text("""
                INSERT INTO iam.tenant_memberships(tenant_id, user_id, role)
                VALUES (:tenant, :user, 'admin')
            """), {"tenant": tenant, "user": user})
            connection.execute(text("""
                INSERT INTO iam.tenant_memberships(tenant_id, user_id, role)
                VALUES (:tenant, :user, 'admin')
            """), {"tenant": tenant, "user": manager})
            connection.execute(text("""
                INSERT INTO governance.chat_sessions(tenant_id, user_id, chat_session_id, title)
                VALUES (:tenant, :user, :session, 'WP4 real worker E2E')
            """), {"tenant": tenant, "user": user, "session": session})
            connection.execute(text(
                f'CREATE ROLE "{api_role}" LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEROLE NOINHERIT PASSWORD \'{api_password}\''
            ))
            connection.execute(text(f'GRANT USAGE ON SCHEMA governance TO "{api_role}"'))
            connection.execute(text(f'GRANT USAGE ON SCHEMA iam TO "{api_role}"'))
            connection.execute(text(f'''GRANT SELECT, UPDATE ON iam.tenant_memberships,
                iam.auth_sessions TO "{api_role}"'''))
            connection.execute(text(f'''GRANT SELECT, INSERT ON governance.memory_membership_erasure_jobs
                TO "{api_role}"'''))
            connection.execute(text(f'''GRANT SELECT, UPDATE ON governance.chat_sessions TO "{api_role}"'''))
            connection.execute(text(f'''GRANT SELECT, INSERT, UPDATE ON governance.memory_registry,
                governance.memory_preference_overlays TO "{api_role}"'''))
            connection.execute(text(f'''GRANT SELECT, INSERT, UPDATE ON governance.memory_outbox,
                governance.memory_preference_settings TO "{api_role}"'''))
            connection.execute(text(f'''GRANT SELECT, INSERT ON governance.memory_preference_audit
                TO "{api_role}"'''))

        api_engine = create_engine(
            make_url(os.environ["WP4_TEST_POSTGRES_URL"]).set(
                username=api_role, password=api_password,
            ), pool_pre_ping=True,
        )

        registry = PostgresPreferenceRegistry(context, engine_override=database)
        monkeypatch.setattr(preference_writer, "engine", api_engine)
        monkeypatch.setattr(memory_routes, "engine", api_engine)
        monkeypatch.setattr(preference_management, "engine", api_engine)
        monkeypatch.setattr(iam_service, "engine", api_engine)

        previous_context_override = app.dependency_overrides.get(require_tenant_context)

        manager_context = TenantContext(
            tenant_id=tenant, user_id=manager, role="admin", scope_type="platform",
        )

        def api_request(
            method: str, path: str, *, payload: dict | None = None,
            request_context: TenantContext | None = None,
        ):
            app.dependency_overrides[require_tenant_context] = lambda: request_context or context
            try:
                with TestClient(app) as client:
                    return client.request(method, path, json=payload)
            finally:
                if previous_context_override is None:
                    app.dependency_overrides.pop(require_tenant_context, None)
                else:
                    app.dependency_overrides[require_tenant_context] = previous_context_override

        def outbox_event_id(registry_id: str, operation_id: str) -> str:
            with database.begin() as connection:
                return str(connection.execute(text("""
                    SELECT event_id FROM governance.memory_outbox
                     WHERE registry_id=:registry_id AND operation_id=:operation_id
                     ORDER BY created_at DESC LIMIT 1
                """), {
                    "registry_id": registry_id, "operation_id": operation_id,
                }).scalar_one())

        def adapter_factory(current_context):
            nonlocal adapter
            adapter = Mem0PreferenceAdapter(current_context, settings=settings, registry=registry)
            return adapter

        monkeypatch.setattr(
            "app.memory.preference_provider.get_personalization_enabled", lambda _context: True,
        )
        create_response = api_request("POST", "/memory/preferences", payload={
            "memory_key": "response_style", "memory_type": "presentation",
            "value": {"style": "concise"}, "chat_session_id": session,
            "source_turn_id": f"turn-{suffix}", "expected_version": 0,
            "idempotency_key": f"real-create-{suffix}", "confirmed": True,
        })
        assert create_response.status_code == 202, create_response.text
        created = SimpleNamespace(**create_response.json())
        created.outbox_event_id = outbox_event_id(created.registry_id, created.operation_id)
        pending_response = api_request(
            "GET", f"/memory/preferences?chat_session_id={session}",
        )
        assert pending_response.status_code == 200
        assert pending_response.json()["items"][0]["status"] == "pending_create"
        provider = Mem0PreferenceProvider(
            context, settings=settings, registry=registry,
            adapter=adapter_factory(context),
        )

        def resolve_preference_to_triage():
            from app.context.compiler import compile_turn_context
            from app.context.orchestrator import ContextOrchestrator

            message = "按我平时的方式回答"
            state = compile_turn_context({
                "user_message": message,
                "preference_provider": provider,
                "memory_enabled": True,
                "chat_session_id": session,
                "current_query_scope_fingerprint": f"wp4-e2e-{suffix}",
                "node_history": [],
            })
            triage = ContextOrchestrator().build_triage_context(
                state, instruction=message, model="wp4-e2e-test-model",
            )
            return state, triage

        pending = provider.search(
            need={"memory_types": ["presentation"]},
            state={"user_message": "按我惯用的回答风格", "chat_session_id": session},
        )
        assert pending and pending[0]["source"] == "pending_overlay"
        pending_state, pending_triage = resolve_preference_to_triage()
        assert pending_state["preference_context"]["status"] == "accepted"
        assert any(
            candidate.get("source_ref", "").startswith("overlay:")
            for candidate in pending_state["preference_context"]["items"]
        )
        assert any(
            item.source.value == "mem0" and item.trust_tier.value == "preference"
            and '"style":"concise"' in item.content.replace(" ", "")
            for item in pending_triage.bundle.items
        )

        original_add = Mem0PreferenceAdapter.add

        def fail_mem0_add(_adapter, *_args, **_kwargs):
            raise Mem0Timeout("add_timeout")

        monkeypatch.setattr(Mem0PreferenceAdapter, "add", fail_mem0_add)
        assert run_one(engine_override=database, adapter_factory=adapter_factory) is True
        with database.begin() as connection:
            timeout_state = connection.execute(text("""
                SELECT r.status, o.status AS outbox_status, o.last_error_code
                  FROM governance.memory_registry r
                  JOIN governance.memory_outbox o ON o.registry_id=r.registry_id
                 WHERE r.registry_id=:registry_id AND o.event_id=:event_id
            """), {
                "registry_id": created.registry_id,
                "event_id": created.outbox_event_id,
            }).mappings().one()
            assert timeout_state["status"] == "pending_create"
            assert timeout_state["outbox_status"] == "pending"
            assert timeout_state["last_error_code"] == "add_timeout"
            connection.execute(text("""
                UPDATE governance.memory_outbox SET next_attempt_at=now()
                 WHERE event_id=:event_id AND status='pending'
            """), {"event_id": created.outbox_event_id})
        monkeypatch.setattr(Mem0PreferenceAdapter, "add", original_add)

        def fail_before_registry_commit(*_args, **_kwargs):
            raise RuntimeError("injected_registry_commit_failure")

        monkeypatch.setattr(preference_worker, "_complete", fail_before_registry_commit)
        assert run_one(engine_override=database, adapter_factory=adapter_factory) is True
        assert adapter is not None
        assert len(adapter.reconcile_pending(created.operation_id)) == 1
        # Simulate a worker process/connection restart; CI also restarts its
        # disposable PostgreSQL service to verify the durable recovery boundary.
        database.dispose()
        postgres_container_id = os.getenv("WP4_TEST_POSTGRES_CONTAINER_ID")
        if os.getenv("GITHUB_ACTIONS", "").lower() == "true":
            assert postgres_container_id, "CI must identify its disposable PostgreSQL container"
        if postgres_container_id:
            subprocess.run(
                ["docker", "restart", postgres_container_id],
                check=True, capture_output=True, text=True, timeout=60,
            )
        database = create_engine(os.environ["WP4_TEST_POSTGRES_URL"], pool_pre_ping=True)
        deadline = time.monotonic() + 60
        while True:
            try:
                with database.begin() as connection:
                    connection.execute(text("SELECT 1"))
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.5)
        monkeypatch.setattr(preference_worker, "_complete", worker_complete)
        with database.begin() as connection:
            connection.execute(text("""
                UPDATE governance.memory_outbox SET next_attempt_at=now()
                 WHERE event_id=:event_id AND status='pending'
            """), {"event_id": created.outbox_event_id})
        assert run_one(engine_override=database, adapter_factory=adapter_factory) is True
        assert len(adapter.reconcile_pending(created.operation_id)) == 1
        with database.begin() as connection:
            active = connection.execute(text("""
                SELECT status, mem0_memory_id FROM governance.memory_registry
                 WHERE registry_id=:registry_id
            """), {"registry_id": created.registry_id}).mappings().one()
            assert active["status"] == "active"
            assert active["mem0_memory_id"]
        assert adapter.get_validated(
            str(active["mem0_memory_id"]), operation_id=created.operation_id,
            memory_type="presentation", memory_key="response_style",
        ) is not None
        persisted_state, persisted_triage = resolve_preference_to_triage()
        assert persisted_state["preference_context"]["status"] == "accepted"
        assert not any(
            candidate.get("source") == "pending_overlay"
            for candidate in persisted_state["preference_context"]["items"]
        )
        assert any(
            item.source.value == "mem0" and item.trust_tier.value == "preference"
            and '"style":"concise"' in item.content.replace(" ", "")
            for item in persisted_triage.bundle.items
        )
        reconcile_response = api_request(
            "POST", f"/memory/preferences/{created.registry_id}/reconcile", payload={
                "expected_version": 1,
                "idempotency_key": f"real-reconcile-{suffix}",
            },
        )
        assert reconcile_response.status_code == 202, reconcile_response.text
        reconciliation = SimpleNamespace(**reconcile_response.json())
        reconciliation.outbox_event_id = outbox_event_id(
            reconciliation.registry_id, reconciliation.operation_id,
        )
        assert reconciliation.status == "pending_update"
        # Expired leases must be recovered from durable state before delivery.
        with database.begin() as connection:
            connection.execute(text("""
                UPDATE governance.memory_outbox
                   SET status='leased', attempts=1, lease_token='expired-lease',
                       lease_expires_at=now()-interval '1 second'
                 WHERE event_id=:event_id AND status='pending'
            """), {"event_id": reconciliation.outbox_event_id})
        assert run_one(engine_override=database, adapter_factory=adapter_factory) is True
        assert len(adapter.reconcile_pending(created.operation_id)) == 1

        update_response = api_request("PUT", f"/memory/preferences/{created.registry_id}", payload={
            "value": {"style": "detailed"}, "chat_session_id": session,
            "source_turn_id": f"turn-update-{suffix}", "expected_version": 1,
            "idempotency_key": f"real-update-{suffix}", "confirmed": True,
        })
        assert update_response.status_code == 200, update_response.text
        updated = SimpleNamespace(**update_response.json())
        assert run_one(engine_override=database, adapter_factory=adapter_factory) is True
        assert adapter.reconcile_pending(created.operation_id) == []
        assert len(adapter.reconcile_pending(updated.operation_id)) == 1

        delete_response = api_request(
            "DELETE", f"/memory/preferences/{updated.registry_id}", payload={
                "expected_version": 2, "idempotency_key": f"real-delete-{suffix}",
                "confirmed": True,
            },
        )
        assert delete_response.status_code == 200, delete_response.text
        deleted = SimpleNamespace(**delete_response.json())
        deleted.outbox_event_id = outbox_event_id(deleted.registry_id, deleted.operation_id)
        delete_memory = Mem0PreferenceAdapter.delete

        def fail_qdrant_delete(_adapter, _memory_id):
            raise RuntimeError("injected_delete_failure")

        monkeypatch.setattr(Mem0PreferenceAdapter, "delete", fail_qdrant_delete)
        dead_letter_before = snapshot_metrics().get("memory.preference_dead_letter_total", 0)
        for attempt in range(5):
            if attempt:
                with database.begin() as connection:
                    connection.execute(text("""
                        UPDATE governance.memory_outbox SET next_attempt_at=now()
                         WHERE event_id=:event_id AND status='pending'
                    """), {"event_id": deleted.outbox_event_id})
            assert run_one(engine_override=database, adapter_factory=adapter_factory) is True
        assert adapter.reconcile_pending(updated.operation_id)
        with database.begin() as connection:
            failed = connection.execute(text("""
                SELECT r.status, o.status AS outbox_status
                  FROM governance.memory_registry r
                  JOIN governance.memory_outbox o ON o.registry_id=r.registry_id
                 WHERE r.registry_id=:registry_id AND o.event_id=:event_id
            """), {
                "registry_id": deleted.registry_id, "event_id": deleted.outbox_event_id,
            }).mappings().one()
            assert failed["status"] == "failed"
            assert failed["outbox_status"] == "dead_letter"
        assert snapshot_metrics()["memory.preference_dead_letter_total"] == dead_letter_before + 1
        assert retry_failed_preference(context, deleted.registry_id, engine_override=database) == "pending_delete"
        monkeypatch.setattr(Mem0PreferenceAdapter, "delete", delete_memory)
        worker_complete = preference_worker._complete

        def fail_after_external_delete(*_args, **_kwargs):
            raise RuntimeError("injected_delete_registry_commit_failure")

        monkeypatch.setattr(preference_worker, "_complete", fail_after_external_delete)
        assert run_one(engine_override=database, adapter_factory=adapter_factory) is True
        assert adapter.reconcile_pending(updated.operation_id) == []
        with database.begin() as connection:
            connection.execute(text("""
                UPDATE governance.memory_outbox SET next_attempt_at=now()
                 WHERE event_id=:event_id AND status='pending'
            """), {"event_id": deleted.outbox_event_id})
        monkeypatch.setattr(preference_worker, "_complete", worker_complete)
        assert run_one(engine_override=database, adapter_factory=adapter_factory) is True
        assert adapter.reconcile_pending(updated.operation_id) == []
        assert provider.search(
            need={"memory_types": ["presentation"]},
            state={"user_message": "按我惯用的回答风格", "chat_session_id": session},
        ) == []
        with database.begin() as connection:
            final_status = connection.execute(text("""
                SELECT status FROM governance.memory_registry WHERE registry_id=:registry_id
            """), {"registry_id": deleted.registry_id}).scalar_one()
            assert final_status == "deleted"

        # Tenant membership revocation is immediate, while durable erasure
        # drains Mem0 first and then clears PostgreSQL memory stores.
        revoke_pref = api_request("POST", "/memory/preferences", payload={
            "memory_key": "response_language", "memory_type": "presentation",
            "value": {"language": "zh-CN"}, "chat_session_id": session,
            "source_turn_id": f"turn-revoke-{suffix}", "expected_version": 0,
            "idempotency_key": f"real-revoke-preference-{suffix}", "confirmed": True,
        })
        assert revoke_pref.status_code == 202, revoke_pref.text
        revoke_preference = SimpleNamespace(**revoke_pref.json())
        assert run_one(engine_override=database, adapter_factory=adapter_factory) is True
        assert len(adapter.reconcile_pending(revoke_preference.operation_id)) == 1
        # Seed the PostgreSQL-only memory stores so completion proves they are purged.
        with database.begin() as connection:
            connection.execute(text("""
                INSERT INTO governance.memory_sessions
                    (tenant_id, user_id, memory_session_id, session_summary,
                     last_scope_fingerprint, last_seller_id, last_start_date,
                     last_end_date, last_trace_id, expires_at)
                VALUES (:tenant, :user, :memory_session, 'private summary',
                        :fingerprint, '', '2026-10-01', '2026-10-31', :trace, now()+interval '1 day')
            """), {
                "tenant": tenant, "user": user, "memory_session": f"mem-session-{suffix}",
                "fingerprint": "a" * 64, "trace": f"trace-{suffix}",
            })
            connection.execute(text("""
                INSERT INTO governance.analysis_memories
                    (memory_id, tenant_id, user_id, seller_id, start_date, end_date,
                     scope_fingerprint, task_type, insight_text, evidence_ids, source_trace_id)
                VALUES (:memory_id, :tenant, :user, 'seller-erasure', '2026-10-01',
                        '2026-10-31', :fingerprint, 'metric_query', 'private insight',
                        '[]'::jsonb, :trace)
            """), {
                "memory_id": f"analysis-memory-{suffix}", "tenant": tenant,
                "user": user, "fingerprint": "b" * 64,
                "trace": f"analysis-trace-{suffix}",
            })
        expired_scope_response = api_request("POST", "/memory/preferences", payload={
            "memory_key": "default_date", "memory_type": "scope_default",
            "value": {"date": "2026-10"}, "chat_session_id": session,
            "source_turn_id": f"turn-expired-scope-{suffix}", "expected_version": 0,
            "idempotency_key": f"real-expired-scope-{suffix}", "confirmed": True,
        })
        assert expired_scope_response.status_code == 202, expired_scope_response.text
        expired_scope = SimpleNamespace(**expired_scope_response.json())
        assert run_one(engine_override=database, adapter_factory=adapter_factory) is True
        with database.begin() as connection:
            connection.execute(text("""
                UPDATE governance.memory_registry
                   SET status='superseded', expires_at=now()-interval '1 day'
                 WHERE registry_id=:registry_id
            """), {"registry_id": expired_scope.registry_id})
        revoke_response = api_request(
            "DELETE", f"/auth/members/{user}", request_context=manager_context,
            payload={
                "confirmation_phrase": "REVOKE_MEMBER_AND_ERASE_MEMORY",
                "idempotency_key": f"real-member-erasure-{suffix}",
            },
        )
        assert revoke_response.status_code == 202, revoke_response.text
        erasure_job_id = revoke_response.json()["job_id"]
        with database.begin() as connection:
            assert connection.execute(text("""
                SELECT revoked_at IS NOT NULL FROM iam.tenant_memberships
                 WHERE tenant_id=:tenant AND user_id=:user
            """), {"tenant": tenant, "user": user}).scalar_one() is True
        # Drain both the active preference and the expired latest scope default,
        # then verify local memories are cleared before the job completes.
        for _ in range(8):
            with database.begin() as connection:
                connection.execute(text("""
                    UPDATE governance.memory_membership_erasure_jobs
                       SET next_attempt_at=now() WHERE job_id=:job_id AND status='pending'
                """), {"job_id": erasure_job_id})
            run_one(engine_override=database, adapter_factory=adapter_factory)
            with database.begin() as connection:
                erasure_status = connection.execute(text("""
                    SELECT status FROM governance.memory_membership_erasure_jobs
                     WHERE job_id=:job_id
                """), {"job_id": erasure_job_id}).scalar_one()
            if erasure_status == "completed":
                break
        assert erasure_status == "completed"
        assert adapter.reconcile_pending(revoke_preference.operation_id) == []
        assert adapter.reconcile_pending(expired_scope.operation_id) == []
        status_response = api_request(
            "GET", f"/auth/members/erasure-jobs/{erasure_job_id}",
            request_context=manager_context,
        )
        assert status_response.status_code == 200, status_response.text
        assert status_response.json()["status"] == "completed"
        with database.begin() as connection:
            assert connection.execute(text("""
                SELECT count(*) FROM governance.memory_sessions
                 WHERE tenant_id=:tenant AND user_id=:user
            """), {"tenant": tenant, "user": user}).scalar_one() == 0
            assert connection.execute(text("""
                SELECT count(*) FROM governance.analysis_memories
                 WHERE tenant_id=:tenant AND user_id=:user
            """), {"tenant": tenant, "user": user}).scalar_one() == 0
            assert connection.execute(text("""
                SELECT count(*) FROM governance.memory_preference_settings
                 WHERE tenant_id=:tenant AND user_id=:user
            """), {"tenant": tenant, "user": user}).scalar_one() == 0
    finally:
        monkeypatch.setattr(preference_worker, "_complete", worker_complete)
        try:
            if qdrant.collection_exists(collection):
                qdrant.delete_collection(collection)
        finally:
            qdrant.close()
            if api_engine is not None:
                api_engine.dispose()
            with database.begin() as connection:
                connection.execute(text("DELETE FROM governance.memory_preference_audit WHERE tenant_id=:tenant"), {"tenant": tenant})
                connection.execute(text("DELETE FROM governance.memory_membership_erasure_jobs WHERE tenant_id=:tenant"), {"tenant": tenant})
                connection.execute(text("DELETE FROM governance.memory_outbox WHERE tenant_id=:tenant"), {"tenant": tenant})
                connection.execute(text("DELETE FROM governance.memory_preference_overlays WHERE tenant_id=:tenant"), {"tenant": tenant})
                connection.execute(text("DELETE FROM governance.memory_registry WHERE tenant_id=:tenant"), {"tenant": tenant})
                connection.execute(text("DELETE FROM governance.memory_sessions WHERE tenant_id=:tenant"), {"tenant": tenant})
                connection.execute(text("DELETE FROM governance.analysis_memories WHERE tenant_id=:tenant"), {"tenant": tenant})
                connection.execute(text("DELETE FROM governance.memory_preference_settings WHERE tenant_id=:tenant"), {"tenant": tenant})
                connection.execute(text("DELETE FROM governance.chat_sessions WHERE tenant_id=:tenant"), {"tenant": tenant})
                connection.execute(text("DELETE FROM iam.tenant_memberships WHERE tenant_id=:tenant"), {"tenant": tenant})
                connection.execute(text("DELETE FROM iam.users WHERE user_id IN (:user,:manager)"), {"user": user, "manager": manager})
                connection.execute(text("DELETE FROM iam.tenants WHERE tenant_id=:tenant"), {"tenant": tenant})
                role_exists = connection.execute(text(
                    "SELECT 1 FROM pg_roles WHERE rolname=:role"
                ), {"role": api_role}).scalar_one_or_none()
                if role_exists:
                    connection.execute(text(f'DROP OWNED BY "{api_role}"'))
                    connection.execute(text(f'DROP ROLE "{api_role}"'))
            database.dispose()

"""Opt-in WP6 worker stop/restart replay drill on an isolated PostgreSQL clone.

Simulates a stopped worker: a preference is queued while ``MEM0_ENABLED=false``
(worker must not claim or touch the database), then the worker is "restarted"
with a fresh engine and a deterministic adapter and must deliver the pending
event exactly once, keeping the Outbox/Registry consistent.
"""

from __future__ import annotations

import os
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from app.memory import preference_worker
from app.memory.policy import MemoryPolicy
from app.memory.preference_writer import PreferenceWriter
from app.tenancy import TenantContext


pytestmark = pytest.mark.skipif(
    not os.getenv("ISOLATED_TEST_POSTGRES_ADMIN_URL"),
    reason="Set ISOLATED_TEST_POSTGRES_ADMIN_URL to run the WP6 worker replay drill",
)

PAYLOAD_KEY = "ab" * 32


class _FakeAdapter:
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}

    def reconcile_pending(self, operation_id: str):
        return [
            SimpleNamespace(memory_id=memory_id)
            for memory_id, row in self.rows.items()
            if row["operation_id"] == operation_id
        ]

    def add(self, text: str, **kwargs):
        memory_id = f"mem0-{uuid.uuid4().hex}"
        self.rows[memory_id] = {"operation_id": kwargs["operation_id"], "text": text}
        return [memory_id]

    def update(self, memory_id: str, **kwargs):
        if memory_id not in self.rows:
            return False
        self.rows[memory_id] = {"operation_id": kwargs["operation_id"], "text": kwargs["text"]}
        return True

    def delete(self, memory_id: str) -> bool:
        return self.rows.pop(memory_id, None) is not None


def test_worker_stop_and_restart_replays_pending_outbox_once(monkeypatch) -> None:
    admin_url = make_url(os.environ["ISOLATED_TEST_POSTGRES_ADMIN_URL"])
    admin = create_engine(admin_url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    template = os.getenv("WP2_TEST_POSTGRES_TEMPLATE", "bizinsight")
    database = f"wp6_worker_{uuid.uuid4().hex[:10]}"
    clone_url = admin_url.set(database=database).render_as_string(hide_password=False)
    engine = None
    try:
        with admin.connect() as connection:
            if not connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname=:name"), {"name": template},
            ).scalar():
                pytest.skip(f"template database {template!r} does not exist")
            connection.execute(text(f'CREATE DATABASE "{database}" TEMPLATE "{template}"'))

        engine = create_engine(clone_url)
        suffix = uuid.uuid4().hex[:10]
        tenant, user, session = f"wp6-worker-{suffix}", f"wp6-worker-user-{suffix}", f"wp6-worker-session-{suffix}"
        context = TenantContext(tenant_id=tenant, user_id=user, role="admin", scope_type="platform")
        with engine.begin() as connection:
            connection.execute(text("""
                INSERT INTO iam.tenants(tenant_id, tenant_name, scope_type, status)
                VALUES (:tenant, 'wp6 worker replay', 'platform', 'active')
            """), {"tenant": tenant})
            connection.execute(text("""
                INSERT INTO iam.users(user_id, display_name, email, status)
                VALUES (:user, 'wp6 worker replay', :email, 'active')
            """), {"user": user, "email": f"{user}@example.invalid"})
            connection.execute(text("""
                INSERT INTO iam.tenant_memberships(tenant_id, user_id, role)
                VALUES (:tenant, :user, 'admin')
            """), {"tenant": tenant, "user": user})
            connection.execute(text("""
                INSERT INTO governance.chat_sessions(tenant_id, user_id, chat_session_id, title)
                VALUES (:tenant, :user, :session, 'wp6 worker replay')
            """), {"tenant": tenant, "user": user, "session": session})

        writer = PreferenceWriter(context, engine_override=engine, payload_key=PAYLOAD_KEY)
        decision = MemoryPolicy.evaluate_upsert(
            context, memory_key="response_style", memory_type="presentation",
            value={"style": "concise"}, explicit_user_request=True, user_confirmed=True,
        )
        assert decision.accepted, decision.reason
        queued = writer.queue_upsert(
            memory_key="response_style", memory_type="presentation",
            value={"style": "concise"}, chat_session_id=session,
            source_turn_id=f"turn-{suffix}", expected_version=0,
            idempotency_key=f"replay-{suffix}", policy_decision=decision,
        )
        assert queued.status == "pending_create"

        # Worker is "stopped": Mem0 disabled => no claim and no database access.
        monkeypatch.setattr(
            preference_worker, "get_settings",
            lambda: SimpleNamespace(
                mem0_enabled=False, memory_preference_worker_lease_seconds=60,
                memory_preference_payload_key=PAYLOAD_KEY,
            ),
        )
        monkeypatch.setattr(
            preference_worker, "_worker_engine",
            lambda: (_ for _ in ()).throw(AssertionError("disabled worker must not connect")),
        )
        assert preference_worker.run_one(engine_override=engine) is False

        with engine.connect() as connection:
            pending = connection.execute(text("""
                SELECT status FROM governance.memory_outbox WHERE event_id=:event
            """), {"event": queued.outbox_event_id}).scalar_one()
            registry = connection.execute(text("""
                SELECT status FROM governance.memory_registry WHERE registry_id=:registry
            """), {"registry": queued.registry_id}).scalar_one()
        assert pending == "pending"
        assert registry == "pending_create"

        # Worker "restarts": new engine (fresh connections) + Mem0 enabled.
        engine.dispose()
        engine = create_engine(clone_url)
        monkeypatch.setattr(
            preference_worker, "get_settings",
            lambda: SimpleNamespace(
                mem0_enabled=True, memory_preference_worker_lease_seconds=60,
                memory_preference_payload_key=PAYLOAD_KEY,
            ),
        )
        adapter = _FakeAdapter()
        assert preference_worker.run_one(
            engine_override=engine, adapter_factory=lambda _context: adapter,
        ) is True
        assert len(adapter.rows) == 1

        with engine.connect() as connection:
            outbox = connection.execute(text("""
                SELECT status FROM governance.memory_outbox WHERE event_id=:event
            """), {"event": queued.outbox_event_id}).scalar_one()
            active = connection.execute(text("""
                SELECT status, mem0_memory_id FROM governance.memory_registry
                 WHERE registry_id=:registry
            """), {"registry": queued.registry_id}).mappings().one()
        assert outbox == "succeeded"
        assert active["status"] == "active"
        assert active["mem0_memory_id"] in adapter.rows

        # No pending work remains; a second run must not create duplicates.
        assert preference_worker.run_one(
            engine_override=engine, adapter_factory=lambda _context: adapter,
        ) is False
        assert len(adapter.rows) == 1
    finally:
        if engine is not None:
            engine.dispose()
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)'))
        admin.dispose()

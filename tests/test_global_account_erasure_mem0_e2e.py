"""Cross-tenant account erasure against disposable PostgreSQL and real Qdrant/Mem0."""

from __future__ import annotations

import os
import time
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text

from app.iam import service as iam_service
from app.memory.mem0_adapter import Mem0PreferenceAdapter
from app.memory.preference_registry import PostgresPreferenceRegistry
from app.memory import preference_worker
from app.tenancy import TenantContext


REQUIRED_ENV = (
    "GLOBAL_ACCOUNT_ERASURE_TEST_URL",
    "MEM0_TEST_QDRANT_URL",
    "MEM0_TEST_HMAC_SECRET",
)
pytestmark = pytest.mark.skipif(
    not all(os.getenv(name) for name in REQUIRED_ENV),
    reason="Requires a disposable migrated PostgreSQL DB and real Qdrant/Mem0 settings",
)


def test_global_account_erasure_deletes_real_mem0_preferences_across_tenants(monkeypatch):
    from qdrant_client import QdrantClient

    database = create_engine(os.environ["GLOBAL_ACCOUNT_ERASURE_TEST_URL"], pool_pre_ping=True)
    suffix = uuid.uuid4().hex[:10]
    user_id = f"erase-mem0-user-{suffix}"
    tenants = [f"erase-mem0-tenant-a-{suffix}", f"erase-mem0-tenant-b-{suffix}"]
    collection = f"global_erase_{suffix}"
    qdrant = QdrantClient(url=os.environ["MEM0_TEST_QDRANT_URL"], timeout=10)
    settings = SimpleNamespace(
        mem0_qdrant_collection=collection,
        mem0_qdrant_url=os.environ["MEM0_TEST_QDRANT_URL"],
        mem0_qdrant_api_key=os.getenv("MEM0_TEST_QDRANT_API_KEY", ""),
        mem0_embedding_model=os.getenv("MEM0_TEST_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
        mem0_embedding_dims=int(os.getenv("MEM0_TEST_EMBEDDING_DIMS", "512")),
        mem0_hmac_secret=os.environ["MEM0_TEST_HMAC_SECRET"],
        mem0_scope_version="v1", mem0_history_db_path="",
        mem0_search_timeout_seconds=5.0, mem0_write_timeout_seconds=10.0,
        mem0_breaker_failures=5, mem0_breaker_window_seconds=30.0,
        mem0_top_k_per_type=5, mem0_top_k_total=10, mem0_relevance_threshold=0.0,
        mem0_llm_model="", mem0_llm_api_key="", mem0_llm_base_url="",
        deepseek_api_key="", deepseek_base_url="",
        qdrant_url=os.environ["MEM0_TEST_QDRANT_URL"],
        qdrant_collection="bizinsight_knowledge", rag_qdrant_active_alias="",
        metric_semantic_qdrant_collection="metric_semantic_catalog_v1",
    )
    original_engine = iam_service.engine
    operations: list[tuple[str, str, str, str]] = []
    anonymous_id = None
    job_id = None

    def adapter_factory(context: TenantContext) -> Mem0PreferenceAdapter:
        return Mem0PreferenceAdapter(
            context, settings=settings,
            registry=PostgresPreferenceRegistry(context, engine_override=database),
        )

    adapters: dict[str, Mem0PreferenceAdapter] = {}
    try:
        with database.begin() as connection:
            for tenant in tenants:
                connection.execute(text("""
                    INSERT INTO iam.tenants(tenant_id,tenant_name,scope_type,status)
                    VALUES (:tenant,'Mem0 erasure integration','platform','active')
                """), {"tenant": tenant})
            connection.execute(text("""
                INSERT INTO iam.users(user_id,display_name,email,password_hash,status)
                VALUES (:user,'Mem0 erasure test',:email,'scrypt$fake','active')
            """), {"user": user_id, "email": f"{user_id}@example.invalid"})
            for tenant in tenants:
                connection.execute(text("""
                    INSERT INTO iam.tenant_memberships(tenant_id,user_id,role)
                    VALUES (:tenant,:user,'admin')
                """), {"tenant": tenant, "user": user_id})

        # Create one genuine Mem0 vector and matching active Registry row in each tenant.
        for index, tenant in enumerate(tenants):
            context = TenantContext(
                tenant_id=tenant, user_id=user_id, role="admin", scope_type="platform",
            )
            adapter = adapter_factory(context)
            adapters[tenant] = adapter
            operation_id = f"global-erasure-{suffix}-{index}"
            memory_id = adapter.add(
                "回答风格偏好：保持简洁。", memory_key="response_style",
                memory_type="presentation", operation_id=operation_id,
                value={"style": "concise"}, infer=False,
            )[0]
            registry_id = f"mreg-{suffix}-{index}"
            operations.append((tenant, operation_id, memory_id, registry_id))
            with database.begin() as connection:
                connection.execute(text("""
                    INSERT INTO governance.memory_registry(
                        registry_id,tenant_id,user_id,seller_id,memory_type,memory_key,
                        mem0_memory_id,status,version,operation_id,user_confirmed
                    ) VALUES (:registry,:tenant,:user,NULL,'presentation','response_style',
                        :memory_id,'active',1,:operation,true)
                """), {
                    "registry": registry_id, "tenant": tenant,
                    "user": user_id, "memory_id": memory_id, "operation": operation_id,
                })

        monkeypatch.setattr(iam_service, "engine", database)
        context = TenantContext(
            tenant_id=tenants[0], user_id=user_id, role="admin", scope_type="platform",
        )
        requested = iam_service.request_account_erasure(
            context, idempotency_key=f"global-mem0-erase-{suffix}",
        )
        job_id = str(requested["job_id"])
        with database.begin() as connection:
            anonymous_id = connection.execute(text("""
                SELECT anonymous_user_id FROM governance.account_erasure_jobs
                 WHERE job_id=:job
            """), {"job": job_id}).scalar_one()

        # Each worker iteration may queue a delete, deliver an Outbox row, or
        # finish one tenant job. Keep driving the real durable coordinator.
        for _ in range(32):
            preference_worker.run_one(
                engine_override=database, adapter_factory=adapter_factory,
            )
            with database.begin() as connection:
                status = connection.execute(text("""
                    SELECT status FROM governance.account_erasure_jobs WHERE job_id=:job
                """), {"job": job_id}).scalar_one()
            if status == "completed":
                break
            # The coordinator deliberately delays a retry when tenant erasure
            # has more work; drive the real backoff instead of busy-spinning.
            time.sleep(1.05)
        assert status == "completed"
        # The collection is unique to this test, so point count zero proves the
        # global erasure removed the real vectors from every tenant scope.
        assert qdrant.count(collection_name=collection, exact=True).count == 0

        for tenant, operation_id, _memory_id, registry_id in operations:
            context = TenantContext(
                tenant_id=tenant, user_id=user_id, role="admin", scope_type="platform",
            )
            adapter = adapters[tenant]
            assert adapter.reconcile_pending(operation_id) == []
            with database.begin() as connection:
                row = connection.execute(text("""
                    SELECT user_id,status,mem0_memory_id,value_hash
                      FROM governance.memory_registry WHERE registry_id=:registry
                """), {"registry": registry_id}).mappings().one()
            assert row["user_id"] == anonymous_id
            assert row["status"] == "deleted"
            assert row["mem0_memory_id"] is None
            assert row["value_hash"] is None

        with database.begin() as connection:
            pending_outbox = connection.execute(text("""
                SELECT count(*) FROM governance.memory_outbox
                 WHERE user_id=:user AND tenant_id=ANY(:tenants)
            """), {"user": user_id, "tenants": tenants}).scalar_one()
            residual_overlays = connection.execute(text("""
                SELECT count(*) FROM governance.memory_preference_overlays
                 WHERE user_id=:user AND tenant_id=ANY(:tenants)
            """), {"user": user_id, "tenants": tenants}).scalar_one()
        assert pending_outbox == 0
        assert residual_overlays == 0

    finally:
        monkeypatch.setattr(iam_service, "engine", original_engine)
        try:
            qdrant.delete_collection(collection_name=collection)
        except Exception:
            pass
        # The test is also safe on a reusable isolated database: remove only
        # rows carrying this invocation's unique tenant/user/job identifiers.
        try:
            with database.begin() as connection:
                connection.execute(text("DELETE FROM governance.audit_events WHERE tenant_id=ANY(:tenants)"), {"tenants": tenants})
                connection.execute(text("DELETE FROM governance.memory_outbox WHERE tenant_id=ANY(:tenants)"), {"tenants": tenants})
                connection.execute(text("DELETE FROM governance.memory_preference_audit WHERE tenant_id=ANY(:tenants)"), {"tenants": tenants})
                connection.execute(text("DELETE FROM governance.memory_preference_overlays WHERE tenant_id=ANY(:tenants)"), {"tenants": tenants})
                connection.execute(text("DELETE FROM governance.memory_registry WHERE tenant_id=ANY(:tenants)"), {"tenants": tenants})
                connection.execute(text("DELETE FROM governance.memory_membership_erasure_jobs WHERE tenant_id=ANY(:tenants)"), {"tenants": tenants})
                if job_id:
                    connection.execute(text("DELETE FROM governance.account_erasure_jobs WHERE job_id=:job"), {"job": job_id})
                connection.execute(text("DELETE FROM iam.tenant_memberships WHERE tenant_id=ANY(:tenants)"), {"tenants": tenants})
                connection.execute(text("DELETE FROM iam.auth_sessions WHERE tenant_id=ANY(:tenants)"), {"tenants": tenants})
                connection.execute(text("DELETE FROM iam.api_credentials WHERE tenant_id=ANY(:tenants)"), {"tenants": tenants})
                connection.execute(text("DELETE FROM iam.users WHERE user_id IN (:user,:anonymous)"), {"user": user_id, "anonymous": anonymous_id})
                connection.execute(text("DELETE FROM iam.tenants WHERE tenant_id=ANY(:tenants)"), {"tenants": tenants})
        finally:
            database.dispose()

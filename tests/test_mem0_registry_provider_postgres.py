"""Real PostgreSQL Registry + real Mem0/Qdrant provider integration.

Requires both:
  WP3_REGISTRY_TEST_DATABASE_URL  migrated disposable database (>= 20261030_57)
  MEM0_TEST_QDRANT_URL            real Qdrant server
  MEM0_TEST_HMAC_SECRET           test-only opaque-scope secret
"""

from __future__ import annotations

import os
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text

from app.context.compiler import compile_turn_context
from app.context.orchestrator import ContextOrchestrator
from app.memory.mem0_adapter import Mem0PreferenceAdapter
from app.memory.preference_provider import Mem0PreferenceProvider
from app.memory.preference_registry import PostgresPreferenceRegistry
from app.tenancy import TenantContext


pytestmark = pytest.mark.skipif(
    not os.getenv("WP3_REGISTRY_TEST_DATABASE_URL")
    or not os.getenv("MEM0_TEST_QDRANT_URL")
    or not os.getenv("MEM0_TEST_HMAC_SECRET"),
    reason="Set WP3_REGISTRY_TEST_DATABASE_URL, MEM0_TEST_QDRANT_URL and MEM0_TEST_HMAC_SECRET",
)


def _settings(collection: str) -> SimpleNamespace:
    return SimpleNamespace(
        mem0_qdrant_collection=collection,
        mem0_qdrant_url=os.environ["MEM0_TEST_QDRANT_URL"],
        mem0_qdrant_api_key=os.getenv("MEM0_TEST_QDRANT_API_KEY", ""),
        mem0_embedding_model=os.getenv("MEM0_TEST_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
        mem0_embedding_dims=int(os.getenv("MEM0_TEST_EMBEDDING_DIMS", "512")),
        mem0_hmac_secret=os.environ["MEM0_TEST_HMAC_SECRET"],
        mem0_scope_version="v1",
        mem0_history_db_path=os.getenv("MEM0_TEST_HISTORY_DB", ""),
        mem0_search_timeout_seconds=5.0,
        mem0_write_timeout_seconds=10.0,
        mem0_breaker_failures=5,
        mem0_breaker_window_seconds=30.0,
        mem0_top_k_per_type=5,
        mem0_top_k_total=10,
        mem0_relevance_threshold=0.0,
        mem0_llm_model="",
        mem0_llm_api_key="",
        mem0_llm_base_url="",
        deepseek_api_key="",
        deepseek_base_url="",
        qdrant_url=os.environ["MEM0_TEST_QDRANT_URL"],
        qdrant_collection="bizinsight_knowledge",
        rag_qdrant_active_alias="",
        metric_semantic_qdrant_collection="metric_semantic_catalog_v1",
    )


def test_real_registry_allows_only_active_provider_candidate(monkeypatch):
    from qdrant_client import QdrantClient

    db_url = os.environ["WP3_REGISTRY_TEST_DATABASE_URL"]
    qdrant = QdrantClient(url=os.environ["MEM0_TEST_QDRANT_URL"], timeout=10)
    engine = create_engine(db_url)
    collection = f"wp3_registry_provider_{uuid.uuid4().hex[:8]}"
    suffix = uuid.uuid4().hex[:10]
    tenant = f"wp3-reg-provider-{suffix}"
    user = f"wp3-reg-provider-user-{suffix}"
    context = TenantContext(tenant_id=tenant, user_id=user, role="admin", scope_type="platform")
    settings = _settings(collection)
    registry = PostgresPreferenceRegistry(context, engine_override=engine)
    adapter = Mem0PreferenceAdapter(context, settings=settings, registry=registry)
    try:
        with engine.begin() as connection:
            connection.execute(text("""
                INSERT INTO iam.tenants(tenant_id, tenant_name, scope_type, status)
                VALUES (:tenant, 'wp3 registry provider', 'platform', 'active')
            """), {"tenant": tenant})
            connection.execute(text("""
                INSERT INTO iam.users(user_id, display_name, email, status)
                VALUES (:user, 'wp3 registry provider', :email, 'active')
            """), {"user": user, "email": f"{user}@example.invalid"})
            connection.execute(text("""
                INSERT INTO iam.tenant_memberships(tenant_id, user_id, role)
                VALUES (:tenant, :user, 'admin')
            """), {"tenant": tenant, "user": user})

        memory_ids = adapter.add(
            "回答尽量简洁", memory_key="response_style", memory_type="presentation",
            operation_id=f"op-provider-{suffix}", value={"style": "concise"}, infer=False,
        )
        assert memory_ids
        with engine.begin() as connection:
            connection.execute(text("""
                INSERT INTO governance.memory_registry
                    (registry_id, tenant_id, user_id, seller_id, memory_type,
                     memory_key, mem0_memory_id, status, version, operation_id,
                     user_confirmed)
                VALUES (:registry_id, :tenant, :user, NULL, 'presentation',
                        'response_style', :memory_id, 'active', 1, :operation_id, true)
            """), {
                "registry_id": f"mreg_{suffix}", "tenant": tenant, "user": user,
                "memory_id": memory_ids[0], "operation_id": f"op-provider-{suffix}",
            })

        provider = Mem0PreferenceProvider(
            context, settings=settings, adapter=adapter, registry=registry,
        )
        state = compile_turn_context({
            "user_message": "按我平时的方式回答",
            "previous_turn_context": {"authorized_metrics": [{"metric_id": "gmv"}]},
            "current_query_scope_fingerprint": "scope-provider",
            "preference_provider": provider,
            "memory_enabled": True,
            "node_history": [],
        })
        assert state["preference_context"]["accepted"], state["preference_context"]
        triage = ContextOrchestrator().build_triage_context(
            state, instruction="按我平时的方式回答", model="test-model",
        )
        assert any(
            item.source.value == "mem0" and item.trust_tier.value == "preference"
            for item in triage.bundle.items
        )
    finally:
        try:
            with engine.begin() as connection:
                connection.execute(text("DELETE FROM governance.memory_registry WHERE tenant_id=:tenant"), {"tenant": tenant})
                connection.execute(text("DELETE FROM iam.tenant_memberships WHERE tenant_id=:tenant"), {"tenant": tenant})
                connection.execute(text("DELETE FROM iam.users WHERE user_id=:user"), {"user": user})
                connection.execute(text("DELETE FROM iam.tenants WHERE tenant_id=:tenant"), {"tenant": tenant})
        finally:
            if qdrant.collection_exists(collection):
                qdrant.delete_collection(collection)
            qdrant.close()
            engine.dispose()

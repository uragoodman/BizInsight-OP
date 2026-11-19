"""Real Mem0 OSS 2.2.1 + Qdrant 1.15.1 server contract for WP3.

This is an opt-in integration test.  It uses the real Mem0 SDK, real Qdrant
server, real BAAI embedding model and a disposable collection. The Registry is
an allow-all test double that isolates the SDK/vector-store contract. A separate
WP4 real-worker E2E test covers PostgreSQL Registry and Outbox integration.
"""

from __future__ import annotations

import os
import uuid
from types import SimpleNamespace

import pytest

from app.memory.mem0_adapter import Mem0PreferenceAdapter
from app.memory.preference_registry import RegistryDecision
from app.tenancy import TenantContext


pytestmark = pytest.mark.skipif(
    not os.getenv("MEM0_TEST_QDRANT_URL") or not os.getenv("MEM0_TEST_HMAC_SECRET"),
    reason="Set MEM0_TEST_QDRANT_URL and MEM0_TEST_HMAC_SECRET for real Mem0/Qdrant verification",
)


@pytest.fixture(autouse=True)
def _personalization_enabled(monkeypatch):
    # This file verifies the Mem0/Qdrant contract, not the PostgreSQL-backed
    # personalization toggle. Keep the contract deterministic without a DB.
    monkeypatch.setattr(
        "app.memory.preference_provider.get_personalization_enabled",
        lambda _context: True,
    )


class _AllowRegistry:
    def validate(self, **_: object) -> RegistryDecision:
        return RegistryDecision(True)


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
        mem0_llm_model=os.getenv("MEM0_TEST_LLM_MODEL", ""),
        mem0_llm_api_key=os.getenv("MEM0_TEST_LLM_API_KEY", ""),
        mem0_llm_base_url=os.getenv("MEM0_TEST_LLM_BASE_URL", ""),
        deepseek_api_key="",
        deepseek_base_url="",
        qdrant_url=os.environ["MEM0_TEST_QDRANT_URL"],
        qdrant_collection="bizinsight_knowledge",
        rag_qdrant_active_alias="",
        metric_semantic_qdrant_collection="metric_semantic_catalog_v1",
    )


@pytest.fixture()
def real_qdrant_collection():
    from qdrant_client import QdrantClient

    collection = f"wp3_contract_{uuid.uuid4().hex[:10]}"
    url = os.environ["MEM0_TEST_QDRANT_URL"]
    client = QdrantClient(url=url, timeout=10)
    try:
        yield collection, client
    finally:
        try:
            if client.collection_exists(collection):
                client.delete_collection(collection)
        finally:
            client.close()


def _context(user: str = "u-a", tenant: str = "tenant-a") -> TenantContext:
    return TenantContext(tenant_id=tenant, user_id=user, role="admin", scope_type="platform")


def test_server_payload_index_filters_and_cross_user_isolation(real_qdrant_collection):
    collection, qdrant = real_qdrant_collection
    settings = _settings(collection)
    a = Mem0PreferenceAdapter(_context("u-a"), settings=settings, registry=_AllowRegistry())
    b = Mem0PreferenceAdapter(_context("u-b"), settings=settings, registry=_AllowRegistry())
    ids_a = a.add(
        "偏好简洁的中文回答", memory_key="response_style", memory_type="presentation",
        operation_id="op-shared", value={"style": "concise"}, infer=False,
    )
    ids_b = b.add(
        "偏好先结论后表格", memory_key="response_style", memory_type="presentation",
        operation_id="op-shared", value={"style": "table"}, infer=False,
    )
    assert ids_a and ids_b
    health = a.health()
    assert health["available"] is True
    assert health["collection"] == collection
    assert "operation_id" in health["payload_indexes"]
    assert "keyword" in health["payload_index_types"]["operation_id"]
    assert health["sdk_version"] == "2.2.1"
    assert health["qdrant_client_version"] == "1.16.2"

    hits_a = a.search("回答风格", memory_types=["presentation"], threshold=0.0)
    hits_b = b.search("回答风格", memory_types=["presentation"], threshold=0.0)
    assert [hit.memory_id for hit in hits_a] == ids_a
    assert [hit.memory_id for hit in hits_b] == ids_b
    assert ids_a[0] not in [hit.memory_id for hit in hits_b]

    # Filters must apply before top_k: fill the user with unrelated operations,
    # then retrieve the one target with top_k=1.
    for index in range(10):
        a.add(
            f"填充偏好 {index}", memory_key=f"filler_{index}", memory_type="presentation",
            operation_id=f"op-filler-{index}", value={"index": index}, infer=False,
        )
    limited = a._memory.get_all(
        filters={"user_id": a.scope.entity_id, "operation_id": "op-shared"},
        top_k=1,
    )
    assert len(limited.get("results") or []) == 1

    # Direct Qdrant filtering by operation_id alone is not an auth boundary.
    from qdrant_client import models

    points, _ = qdrant.scroll(
        collection,
        scroll_filter=models.Filter(must=[
            models.FieldCondition(
                key="operation_id", match=models.MatchValue(value="op-shared"),
            ),
        ]),
        limit=20,
        with_payload=True,
    )
    assert {str(point.payload.get("user_id")) for point in points} == {
        a.scope.entity_id, b.scope.entity_id,
    }

    # A wrong entity scope in metadata must be rejected before Bundle assembly.
    a._memory.add(
        "错误的 scope 偏好", user_id=a.scope.entity_id,
        metadata={
            "operation_id": "op-wrong-scope", "memory_type": "presentation",
            "memory_key": "response_style", "memory_status": "active",
            "entity_scope": b.scope.entity_id, "scope_version": "v1",
            "seller_scope": "",
        }, infer=False,
    )
    assert all(hit.metadata.get("entity_scope") == a.scope.entity_id for hit in a.search(
        "错误 scope", memory_types=["presentation"], threshold=0.0,
    ))


def test_update_delete_reconcile_and_duplicate_cleanup(real_qdrant_collection):
    collection, _ = real_qdrant_collection
    adapter = Mem0PreferenceAdapter(
        _context("u-a"), settings=_settings(collection), registry=_AllowRegistry(),
    )
    first = adapter.add(
        "偏好简洁", memory_key="response_style", memory_type="presentation",
        operation_id="op-dup", value={"style": "concise"}, infer=False,
    )[0]
    duplicate = adapter.add(
        "偏好简洁", memory_key="response_style", memory_type="presentation",
        operation_id="op-dup", value={"style": "concise"}, infer=False,
    )[0]
    assert first != duplicate
    assert len(adapter.reconcile("op-dup")) == 2
    assert adapter.update(first, text="偏好非常简洁", value={"style": "very_concise"}) is True
    assert adapter.search("非常简洁", memory_types=["presentation"], threshold=0.0)[0].value == {
        "style": "very_concise",
    }
    cleaned = adapter.delete_operation("op-dup", keep_memory_id=first)
    assert cleaned["deleted_count"] == 1
    assert cleaned["remaining_memory_ids"] == [first]
    assert adapter.delete(first) is True
    assert adapter.reconcile("op-dup") == []


def test_inactive_and_expired_records_are_not_returned(real_qdrant_collection):
    collection, _ = real_qdrant_collection
    adapter = Mem0PreferenceAdapter(
        _context("u-a"), settings=_settings(collection), registry=_AllowRegistry(),
    )
    active = adapter.add(
        "偏好简洁", memory_key="response_style", memory_type="presentation",
        operation_id="op-status", value={"style": "concise"}, infer=False,
    )[0]
    adapter.update(active, memory_status="revoked")
    assert adapter.search("简洁", memory_types=["presentation"], threshold=0.0) == []

    # Expired rows are rejected even if the status is active.
    rows = adapter._memory.add(
        "过期偏好", user_id=adapter.scope.entity_id,
        metadata={
            "operation_id": "op-expired", "memory_type": "presentation",
            "memory_key": "response_style", "memory_status": "active",
            "entity_scope": adapter.scope.entity_id, "scope_version": "v1",
            "seller_scope": "", "expires_at": "2000-01-01T00:00:00+00:00",
            "value_json": '{"style": "stale"}',
        }, infer=False,
    )
    assert rows
    assert adapter.search("过期", memory_types=["presentation"], threshold=0.0) == []


def test_mem0_infer_true_real_sdk_path(real_qdrant_collection):
    if not os.getenv("MEM0_TEST_LLM_MODEL") or not os.getenv("MEM0_TEST_LLM_API_KEY"):
        pytest.skip("Set MEM0_TEST_LLM_MODEL and MEM0_TEST_LLM_API_KEY for infer=True verification")
    collection, _ = real_qdrant_collection
    adapter = Mem0PreferenceAdapter(
        _context("u-infer"), settings=_settings(collection), registry=_AllowRegistry(),
    )
    ids = adapter.add(
        "我以后希望回答尽量简洁，先给结论。",
        memory_key="response_style", memory_type="presentation",
        operation_id="op-infer", value={"style": "concise"}, infer=True,
    )
    assert ids
    reader_count = len(adapter.reconcile("op-infer"))
    assert reader_count >= 1


def test_real_provider_flows_to_wp2_resolver_and_wp1_triage_bundle(real_qdrant_collection):
    from app.context.compiler import compile_turn_context
    from app.context.orchestrator import ContextOrchestrator
    from app.memory.preference_provider import Mem0PreferenceProvider

    collection, _ = real_qdrant_collection
    settings = _settings(collection)
    context = _context("u-provider")
    adapter = Mem0PreferenceAdapter(context, settings=settings, registry=_AllowRegistry())
    adapter.add(
        "回答尽量简洁，先给结论", memory_key="response_style",
        memory_type="presentation", operation_id="op-provider",
        value={"style": "concise", "structure": "conclusion_first"}, infer=False,
    )
    provider = Mem0PreferenceProvider(
        context, settings=settings, adapter=adapter, registry=_AllowRegistry(),
    )
    state = compile_turn_context({
        "user_message": "按我平时的方式回答",
        "previous_turn_context": {
            "source_turn_id": "turn-provider-prior",
            "authorized_metrics": [{"metric_id": "gmv"}],
            "context_slots": {"metric": {"value": "gmv", "source": "current_turn"}},
        },
        "preference_provider": provider,
        "current_query_scope_fingerprint": "scope-provider",
        "memory_enabled": True,
        "node_history": [],
    })
    preference = state["preference_context"]
    assert preference["status"] == "accepted", preference
    assert preference["items"][0]["memory_type"] == "presentation"
    assert preference["items"][0]["value"]["style"] == "concise"
    triage = ContextOrchestrator().build_triage_context(
        state, instruction="按我平时的方式回答", model="test-model",
    )
    assert any(
        item.source.value == "mem0" and item.trust_tier.value == "preference"
        for item in triage.bundle.items
    )
    assert state["turn_resolution"]["preference"]["accepted"]
    assert state["context_compilation"]["known_slots"]["metric"]["value"] == "gmv"


def test_rag_and_metric_collections_remain_untouched(real_qdrant_collection):
    collection, qdrant = real_qdrant_collection
    names = [item.name for item in qdrant.get_collections().collections]
    reserved = [name for name in ("bizinsight_knowledge", "metric_semantic_catalog_v1") if name in names]
    before = {
        name: qdrant.count(collection_name=name, exact=True).count
        for name in reserved
    }
    adapter = Mem0PreferenceAdapter(
        _context("u-isolation"), settings=_settings(collection), registry=_AllowRegistry(),
    )
    adapter.add(
        "偏好简洁", memory_key="response_style", memory_type="presentation",
        operation_id="op-isolation", value={"style": "concise"}, infer=False,
    )
    after = {
        name: qdrant.count(collection_name=name, exact=True).count
        for name in reserved
    }
    assert before == after
    assert collection not in {"bizinsight_knowledge", "metric_semantic_catalog_v1"}


def test_context_compiler_uses_production_provider_factory(real_qdrant_collection):
    from unittest.mock import patch

    from app.context.compiler import compile_turn_context
    from app.memory.preference_provider import Mem0PreferenceProvider

    collection, _ = real_qdrant_collection
    settings = _settings(collection)
    context = _context("u-factory")
    adapter = Mem0PreferenceAdapter(context, settings=settings, registry=_AllowRegistry())
    adapter.add(
        "回答尽量简洁", memory_key="response_style", memory_type="presentation",
        operation_id="op-factory", value={"style": "concise"}, infer=False,
    )
    provider = Mem0PreferenceProvider(
        context, settings=settings, adapter=adapter, registry=_AllowRegistry(),
    )
    with patch("app.memory.preference_provider.provider_for_context", return_value=provider):
        state = compile_turn_context({
            "user_message": "按我平时的方式回答",
            "previous_turn_context": {"authorized_metrics": [{"metric_id": "gmv"}]},
            "current_query_scope_fingerprint": "scope-factory",
            "memory_enabled": True,
            "node_history": [],
        })
    assert state["preference_context"]["accepted"]
    assert state["preference_context"]["items"][0]["memory_type"] == "presentation"

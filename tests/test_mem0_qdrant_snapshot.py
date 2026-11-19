"""Real Qdrant snapshot/restore drill for the isolated Mem0 collection."""

from __future__ import annotations

import io
import os
import uuid
from types import SimpleNamespace

import pytest
import requests

from app.memory.mem0_adapter import Mem0PreferenceAdapter
from app.memory.preference_registry import RegistryDecision
from app.tenancy import TenantContext


pytestmark = pytest.mark.skipif(
    not os.getenv("MEM0_TEST_QDRANT_URL") or not os.getenv("MEM0_TEST_HMAC_SECRET"),
    reason="Set MEM0_TEST_QDRANT_URL and MEM0_TEST_HMAC_SECRET for snapshot/restore drill",
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


def test_snapshot_delete_restore_preserves_mem0_collection():
    from qdrant_client import QdrantClient

    url = os.environ["MEM0_TEST_QDRANT_URL"].rstrip("/")
    collection = f"wp3_snapshot_{uuid.uuid4().hex[:10]}"
    client = QdrantClient(url=url, timeout=30)
    context = TenantContext(tenant_id="snap-tenant", user_id="snap-user", role="admin", scope_type="platform")
    adapter = Mem0PreferenceAdapter(
        context, settings=_settings(collection), registry=_AllowRegistry(),
    )
    snapshot_name = None
    try:
        ids = adapter.add(
            "偏好简洁", memory_key="response_style", memory_type="presentation",
            operation_id="op-snapshot", value={"style": "concise"}, infer=False,
        )
        assert ids
        before = client.count(collection_name=collection, exact=True).count
        assert before == 1
        snapshot = client.create_snapshot(collection, wait=True)
        assert snapshot is not None and snapshot.name
        snapshot_name = snapshot.name
        restored = f"{collection}_restored"
        snapshot_url = f"{url}/collections/{collection}/snapshots/{snapshot_name}"
        snapshot_response = requests.get(snapshot_url, timeout=30)
        snapshot_response.raise_for_status()
        upload_response = requests.post(
            f"{url}/collections/{restored}/snapshots/upload",
            params={"priority": "snapshot", "wait": "true"},
            files={
                "snapshot": (
                    snapshot_name,
                    io.BytesIO(snapshot_response.content),
                    "application/octet-stream",
                ),
            },
            timeout=60,
        )
        upload_response.raise_for_status()
        assert upload_response.json().get("result") is True
        assert client.collection_exists(restored)
        assert client.count(collection_name=restored, exact=True).count == 1
        info = client.get_collection(restored)
        assert "operation_id" in info.payload_schema
        points, _ = client.scroll(restored, limit=10, with_payload=True)
        assert len(points) == 1
        assert points[0].payload.get("operation_id") == "op-snapshot"
        assert points[0].payload.get("entity_scope") == adapter.scope.entity_id
        client.delete_collection(collection)
        assert not client.collection_exists(collection)
    finally:
        try:
            for name in (collection, f"{collection}_restored"):
                if client.collection_exists(name):
                    client.delete_collection(name)
        finally:
            client.close()

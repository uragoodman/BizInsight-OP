"""RAG 建库门禁测试。

这些测试只使用 Qdrant in-memory client，验证安全默认值，不连接 Docker 中的真实
Qdrant，也不会修改项目的持久化 collection。
"""

from __future__ import annotations

import hashlib

from qdrant_client import QdrantClient, models

from app.rag.indexing import build_qdrant_index_native
from app.rag.schemas import KnowledgeChunk


def _chunks() -> list[KnowledgeChunk]:
    return [
        KnowledgeChunk(
            chunk_id="docs/knowledge/metrics/gmv.md#chunk-1",
            doc_id="docs/knowledge/metrics/gmv.md",
            title="GMV",
            text="GMV 是成交金额。",
            source_path="docs/knowledge/metrics/gmv.md",
            doc_type="metric",
            metric_name="gmv",
            tags=["metric", "gmv"],
            document_hash=hashlib.sha256("GMV 是成交金额。".encode()).hexdigest(),
            index_version="test-v1",
        )
    ]


def _embed(text: str) -> list[float]:
    return [1.0, 0.0, 0.0, 0.0]


def test_build_requires_explicit_collection_name(monkeypatch) -> None:
    monkeypatch.setattr("app.rag.indexing._write_manifest", lambda *args, **kwargs: None)
    result = build_qdrant_index_native(client=QdrantClient(":memory:"), chunks=_chunks(), embed_fn=_embed)
    assert result["status"] == "safety_error"
    assert "collection_name" in str(result["error"])


def test_dry_run_does_not_create_or_write_collection(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.rag.indexing._write_manifest", lambda *args, **kwargs: None)
    client = QdrantClient(":memory:")
    result = build_qdrant_index_native(
        collection_name="test_hybrid_v2",
        client=client,
        chunks=_chunks(),
        embed_fn=_embed,
        sparse_model_path=tmp_path / "sparse.json",
        dry_run=True,
    )
    assert result["status"] == "dry_run"
    assert result["write_performed"] is False
    assert not client.collection_exists("test_hybrid_v2")
    client.close()


def test_existing_collection_fails_without_destructive_opt_in(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.rag.indexing._write_manifest", lambda *args, **kwargs: None)
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="test_existing",
        vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
    )
    result = build_qdrant_index_native(
        collection_name="test_existing",
        client=client,
        chunks=_chunks(),
        embed_fn=_embed,
        sparse_model_path=tmp_path / "sparse.json",
    )
    assert result["status"] == "safety_error"
    assert client.collection_exists("test_existing")
    assert client.count("test_existing").count == 0
    client.close()


def test_destructive_rebuild_requires_snapshot_and_manifest(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.rag.indexing._write_manifest", lambda *args, **kwargs: None)
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="test_existing",
        vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
    )
    result = build_qdrant_index_native(
        collection_name="test_existing",
        client=client,
        chunks=_chunks(),
        embed_fn=_embed,
        sparse_model_path=tmp_path / "sparse.json",
        allow_destructive_rebuild=True,
        snapshot_ref="snapshot-test",
        release_manifest_path=tmp_path / "missing-manifest.json",
    )
    assert result["status"] == "safety_error"
    assert client.collection_exists("test_existing")
    client.close()


def test_destructive_rebuild_rejects_invalid_release_manifest(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.rag.indexing._write_manifest", lambda *args, **kwargs: None)
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="test_existing",
        vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
    )
    manifest = tmp_path / "release.json"
    manifest.write_text("not-json", encoding="utf-8")
    result = build_qdrant_index_native(
        collection_name="test_existing",
        client=client,
        chunks=_chunks(),
        embed_fn=_embed,
        sparse_model_path=tmp_path / "sparse.json",
        allow_destructive_rebuild=True,
        snapshot_ref="snapshot-test",
        release_manifest_path=manifest,
    )
    assert result["status"] == "safety_error"
    assert client.collection_exists("test_existing")
    client.close()

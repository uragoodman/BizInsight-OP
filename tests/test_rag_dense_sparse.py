"""Native Qdrant dense+sparse 构建与原生 hybrid 检索测试。

使用 ``QdrantClient(":memory:")`` 与确定性 fake dense 编码，全程离线、不依赖外部
Qdrant 服务或 embedding 模型，验证 multimodal collection 构建、RRF 融合检索与
metadata filter 下推。
"""
from __future__ import annotations

import hashlib
import math

from app.rag.indexing import build_qdrant_index_native
from app.rag.retriever import _retrieve_from_qdrant_native
from app.rag.schemas import KnowledgeChunk
from app.rag import sparse as sparse_features

DIM = 64


def _token_vec(token: str, dim: int = DIM) -> list[float]:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    vec = [digest[i % len(digest)] / 255.0 - 0.5 for i in range(dim)]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def make_fake_embed(dim: int = DIM):
    """确定性 dense 编码：共享词项越多的文本，向量余弦越接近。"""

    def embed(text: str) -> list[float]:
        tokens = list(set(sparse_features.tokenize(text)))
        if not tokens:
            return [0.0] * dim
        vec = [0.0] * dim
        for token in tokens:
            token_vec = _token_vec(token, dim)
            for i in range(dim):
                vec[i] += token_vec[i]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    return embed


def _chunk(
    chunk_id: str,
    title: str,
    text: str,
    doc_type: str = "metric",
    metric_name: str | None = None,
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("#")[0],
        title=title,
        text=text,
        source_path=chunk_id,
        doc_type=doc_type,
        metric_name=metric_name,
        tags=[doc_type],
        section=None,
        document_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        index_version="test-v1",
    )


CHUNKS = [
    _chunk("docs/knowledge/metrics/gmv.md#chunk-1", "GMV 指标定义",
           "GMV 成交金额 指标定义。GMV 等于商品成交总金额，衡量经营规模。", "metric", "gmv"),
    _chunk("docs/knowledge/metrics/order_count.md#chunk-1", "订单量指标定义",
           "订单量 指标定义。订单量统计成交订单数量。", "metric", "order_count"),
    _chunk("docs/knowledge/sop/anomaly.md#chunk-1", "异常分析 SOP",
           "异常分析 SOP。指标出现异常波动时，按步骤排查根因。", "sop", None),
    _chunk("docs/knowledge/templates/report.md#chunk-1", "周报模板",
           "周报模板。周报包含 gmv 订单量 客单价 汇总。", "template", None),
]


def _build_native(monkeypatch, tmp_path):
    """构建 in-memory native collection，返回 (client, model_path, embed)。"""
    from qdrant_client import QdrantClient

    # 隔离增量 manifest，避免测试写真实 data/rag/manifest.json。
    monkeypatch.setattr("app.rag.indexing._write_manifest", lambda *a, **k: None)
    client = QdrantClient(":memory:")
    model_path = tmp_path / "sparse-model.json"
    embed = make_fake_embed()
    result = build_qdrant_index_native(
        collection_name="test_knowledge_hybrid",
        client=client, embed_fn=embed, sparse_model_path=model_path, chunks=CHUNKS,
    )
    assert result["status"] == "ok", result
    return client, model_path, embed


def test_build_creates_native_dense_sparse_collection(monkeypatch, tmp_path):
    client, model_path, _ = _build_native(monkeypatch, tmp_path)
    try:
        assert client.count("test_knowledge_hybrid").count == len(CHUNKS)
    finally:
        client.close()
    assert model_path.exists()
    payload = model_path.read_text(encoding="utf-8")
    assert "vocab" in payload and "idf" in payload


def test_shared_term_query_recalls_target_chunk(monkeypatch, tmp_path):
    client, model_path, embed = _build_native(monkeypatch, tmp_path)
    try:
        result = _retrieve_from_qdrant_native(
            "gmv 成交金额 怎么定义", doc_types=None, metric_name=None, top_k=3,
            embed_fn=embed, client=client, sparse_model_path=model_path,
            collection_name="test_knowledge_hybrid",
        )
        assert result is not None
        assert result.strategy == "qdrant_hybrid"
        assert result.filters["native_dense_sparse"] is True
        assert result.filters["fusion"] == "RRF"
        chunk_ids = [hit.chunk_id for hit in result.hits]
        assert "docs/knowledge/metrics/gmv.md#chunk-1" in chunk_ids
        assert chunk_ids[0] == "docs/knowledge/metrics/gmv.md#chunk-1"
    finally:
        client.close()


def test_metadata_filter_narrows_to_metric(monkeypatch, tmp_path):
    client, model_path, embed = _build_native(monkeypatch, tmp_path)
    try:
        result = _retrieve_from_qdrant_native(
            "gmv 成交金额", top_k=5, doc_types={"metric"}, metric_name="gmv",
            embed_fn=embed, client=client, sparse_model_path=model_path,
            collection_name="test_knowledge_hybrid",
        )
        assert result is not None
        assert all(hit.citation.doc_type == "metric" for hit in result.hits)
        assert all(hit.citation.metric_name == "gmv" for hit in result.hits)
    finally:
        client.close()


def test_out_of_vocab_query_returns_none_for_sparse(monkeypatch, tmp_path):
    client, model_path, embed = _build_native(monkeypatch, tmp_path)
    try:
        # 英文长串不在中文/英文词表，sparse 特征为空，native 查询应返回 None 触发降级。
        result = _retrieve_from_qdrant_native(
            "zzzzzzqqqqqq", doc_types=None, metric_name=None, top_k=3,
            embed_fn=embed, client=client, sparse_model_path=model_path,
            collection_name="test_knowledge_hybrid",
        )
        assert result is None
    finally:
        client.close()


def test_sparse_model_roundtrip_and_positive_weights(tmp_path):
    vocab, idf = sparse_features.build_vocab(CHUNKS, min_df=1)
    assert len(vocab) > 0
    target = tmp_path / "model.json"
    sparse_features.save_sparse_model(vocab, idf, target)
    loaded = sparse_features.load_sparse_model(target)
    assert loaded is not None
    restored_vocab, restored_idf = loaded
    assert restored_vocab == vocab
    assert restored_idf == idf

    vector = sparse_features.sparse_embed("gmv 成交金额", vocab, idf)
    assert vector["indices"]
    assert len(vector["indices"]) == len(vector["values"])
    assert all(value > 0 for value in vector["values"])

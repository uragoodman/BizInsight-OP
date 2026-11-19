from __future__ import annotations

from pathlib import Path

from app.rag.fingerprint import build_index_fingerprint, corpus_digest, sparse_model_digest
from app.rag.config import get_rag_settings
from app.rag.indexing import _artifact_path, _select_changed_chunks, check_collection_fingerprint
from app.rag.pipeline import build_document_chunks
from app.rag.schemas import KnowledgeChunk, KnowledgeDocument


def _chunk(doc_id: str, text: str) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=f"{doc_id}#chunk-1",
        doc_id=doc_id,
        title=doc_id,
        text=text,
        source_path=f"{doc_id}.md",
        document_hash=f"hash-{doc_id}",
        pipeline_version="section-v1",
        content_hash=f"content-{text}",
    )


def test_corpus_digest_is_order_independent() -> None:
    left = [_chunk("b", "B"), _chunk("a", "A")]
    right = [_chunk("a", "A"), _chunk("b", "B")]
    assert corpus_digest(left) == corpus_digest(right)


def test_fingerprint_changes_when_schema_or_pipeline_changes() -> None:
    common = dict(
        collection="demo_v2",
        vector_names=["default"],
        distance="Cosine",
        dimension=4,
        embedding_model="BAAI/bge-small-zh-v1.5",
        embedding_model_revision="rev-1",
        indexing_code_version="indexing-v2",
        chunk_parameters={"chunk_size": 512, "chunk_overlap": 80},
        tokenizer_id="huggingface:BAAI/bge-small-zh-v1.5",
        sparse_vocab_hash=None,
        corpus_hash="corpus-1",
        index_version="rag-v1",
    )
    first = build_index_fingerprint(index_mode="dense", chunk_pipeline_version="section-v1", **common)
    second = build_index_fingerprint(index_mode="dense", chunk_pipeline_version="sentence-v2", **common)
    assert first["digest"] != second["digest"]
    assert first["payload"]["chunk_pipeline_version"] == "section-v1"


def test_sparse_model_digest_includes_idf() -> None:
    assert sparse_model_digest({"gmv": 0}, {0: 1.0}) != sparse_model_digest({"gmv": 0}, {0: 2.0})


def test_custom_collection_gets_isolated_manifest_path() -> None:
    base = _artifact_path(
        Path(get_rag_settings().manifest_path),
        "collection-for-test",
    )
    assert "collection-for-test" in base.name
    assert base != Path(get_rag_settings().manifest_path)


def test_legacy_manifest_without_pipeline_version_is_rebuilt() -> None:
    chunk = _chunk("doc", "new")
    settings = get_rag_settings()
    chunk = chunk.model_copy(update={
        "document_hash": "hash-doc",
        "index_version": settings.index_version,
    })
    assert _select_changed_chunks(
        [chunk],
        {
            "documents": {
                "doc": {
                    "document_hash": chunk.document_hash,
                    "index_version": chunk.index_version,
                    "embedding_model": settings.embedding_model,
                }
            }
        },
    ) == [chunk]


def test_sentence_v2_uses_same_final_chunk_contract_for_all_backends() -> None:
    document = KnowledgeDocument(
        doc_id="docs/demo.md",
        title="Demo",
        text="# Demo\n\n## Definition\n第一句。第二句。第三句。",
        source_path="docs/demo.md",
        document_hash="doc-hash",
        index_version="rag-v1",
    )
    chunks = build_document_chunks(
        document,
        pipeline_version="sentence-v2",
        chunk_size=8,
        chunk_overlap=2,
        tokenizer=lambda value: list(value),
    )
    assert chunks
    assert all(chunk.pipeline_version == "sentence-v2" for chunk in chunks)
    assert all(chunk.content_hash for chunk in chunks)
    assert all(chunk.source_start is not None and chunk.source_end is not None for chunk in chunks)
    assert all(chunk.source_start < chunk.source_end for chunk in chunks)
    assert all("#sentence-v2#" in chunk.chunk_id for chunk in chunks)
    assert len({(chunk.source_start, chunk.source_end) for chunk in chunks}) > 1


def test_collection_without_fingerprint_is_explicitly_legacy(monkeypatch) -> None:
    monkeypatch.setattr("app.rag.indexing._load_manifest", lambda _name: {"schema_version": 1})
    result = check_collection_fingerprint(object(), "legacy", index_mode="dense")
    assert result["status"] == "legacy_unverified"


def test_collection_fingerprint_detects_schema_mismatch(monkeypatch) -> None:
    class Params:
        vectors = type("Vector", (), {"size": 3, "distance": type("Distance", (), {"value": "Cosine"})()})()
        sparse_vectors = None

    class Client:
        def get_collection(self, _name):
            return type("Info", (), {"config": type("Config", (), {"params": Params()})()})()

    settings = get_rag_settings()
    fingerprint = build_index_fingerprint(
        index_mode="dense", collection="demo", vector_names=["default"],
        distance="Cosine", dimension=4, embedding_model=settings.embedding_model,
        embedding_model_revision=settings.embedding_model_revision,
        indexing_code_version="indexing-v2", chunk_pipeline_version=settings.chunk_pipeline_version,
        chunk_parameters={"chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap},
        tokenizer_id=f"huggingface:{settings.embedding_model}", sparse_vocab_hash=None,
        corpus_hash="corpus", index_version=settings.index_version,
    )
    monkeypatch.setattr("app.rag.indexing._load_manifest", lambda _name: {"index_fingerprint": fingerprint})
    result = check_collection_fingerprint(Client(), "demo", index_mode="dense")
    assert result["status"] == "mismatch"
    assert "dimension" in result["different_fields"]

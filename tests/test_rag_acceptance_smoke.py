"""Small, non-destructive acceptance smoke suite for the RAG release plan.

These tests deliberately use temporary files, an in-memory Qdrant instance, or
pure functions.  They verify the release gates without touching the Docker
Qdrant collection or PostgreSQL database used by the development environment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from app.rag.backup import export_collection, restore_export, verify_export
from app.rag.fingerprint import build_index_fingerprint
from app.rag.release_controller import (
    _validate_qdrant_release_manifest,
    rollout_assignment,
    transition_allowed,
)
from app.rag.schemas import RagCitation, RagSearchHit
from app.tenancy import TenantContext
from evaluation.retrieval import runner
from evaluation.retrieval.metrics import chunk_hit_at_k, doc_hit_at_k, mrr_at_k, ndcg_at_k
from scripts import backup_postgres
from scripts import restore_postgres


def _qdrant_source() -> QdrantClient:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="source",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    client.upsert(
        collection_name="source",
        points=[models.PointStruct(id=1, vector=[1.0, 0.0], payload={"text": "GMV"})],
    )
    return client


def test_qdrant_export_is_verifiable_and_does_not_overwrite(tmp_path: Path) -> None:
    client = _qdrant_source()
    try:
        output = tmp_path / "qdrant.jsonl"
        result = export_collection(client, collection_name="source", output_path=output)
        assert result["exported_points"] == 1
        assert verify_export(output)["point_count"] == 1
        with pytest.raises(FileExistsError):
            export_collection(client, collection_name="source", output_path=output)
    finally:
        client.close()


def test_qdrant_restore_uses_an_isolated_collection(tmp_path: Path) -> None:
    client = _qdrant_source()
    try:
        output = tmp_path / "qdrant.jsonl"
        export_collection(client, collection_name="source", output_path=output)
        restored = restore_export(client, input_path=output, target_collection="source_restore")
        assert restored["point_count"] == 1
        assert client.count("source_restore").count == 1
    finally:
        client.close()


def test_qdrant_restore_refuses_existing_target(tmp_path: Path) -> None:
    client = _qdrant_source()
    try:
        output = tmp_path / "qdrant.jsonl"
        export_collection(client, collection_name="source", output_path=output)
        with pytest.raises(FileExistsError, match="isolated collection"):
            restore_export(client, input_path=output, target_collection="source")
    finally:
        client.close()


def test_postgres_backup_dry_run_redacts_connection_url(tmp_path: Path, monkeypatch) -> None:
    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("dry-run must not execute pg_dump")

    monkeypatch.setattr(backup_postgres.subprocess, "run", unexpected_run)
    result = backup_postgres.create_backup(
        database_url="postgresql://user:secret@localhost/bizinsight",
        output=tmp_path / "backup.dump",
        dry_run=True,
    )
    assert result["status"] == "dry_run"
    assert "secret" not in json.dumps(result)


def test_postgres_restore_requires_isolated_database(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "backup.dump"
    source.write_bytes(b"backup")
    monkeypatch.setattr(
        restore_postgres,
        "verify_backup",
        lambda _path: {"sha256": "a" * 64},
    )
    with pytest.raises(ValueError, match="isolated target database"):
        restore_postgres.restore_backup(
            input_path=source,
            database_url="postgresql://user:secret@localhost/bizinsight",
            dry_run=True,
        )


def test_postgres_restore_dry_run_is_isolated_and_redacted(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "backup.dump"
    source.write_bytes(b"backup")
    monkeypatch.setattr(
        restore_postgres,
        "verify_backup",
        lambda _path: {"sha256": "b" * 64},
    )
    result = restore_postgres.restore_backup(
        input_path=source,
        database_url="postgresql://user:secret@localhost/bizinsight_restore",
        dry_run=True,
    )
    assert result["target_database"] == "bizinsight_restore"
    assert "secret" not in json.dumps(result)


def _fingerprint(**overrides):
    values = {
        "index_mode": "dense",
        "collection": "knowledge_v2",
        "vector_names": ["dense"],
        "distance": "Cosine",
        "dimension": 512,
        "embedding_model": "BAAI/bge-small-zh-v1.5",
        "embedding_model_revision": "rev-1",
        "indexing_code_version": "commit-1",
        "chunk_pipeline_version": "section-v1",
        "chunk_parameters": {"chunk_size": 512, "chunk_overlap": 64},
        "tokenizer_id": "bge-tokenizer",
        "sparse_vocab_hash": None,
        "corpus_hash": "corpus-1",
        "index_version": "rag-v2",
    }
    values.update(overrides)
    return build_index_fingerprint(**values)


def test_fingerprint_is_stable_and_changes_for_pipeline_revision() -> None:
    assert _fingerprint()["digest"] == _fingerprint()["digest"]
    assert _fingerprint(chunk_pipeline_version="sentence-v2")["digest"] != _fingerprint()["digest"]


def test_release_manifest_must_bind_collection_and_fingerprint(tmp_path: Path) -> None:
    manifest = tmp_path / "release.json"
    spec = {
        "backend": "qdrant_vector",
        "qdrant_collection": "knowledge_v2",
        "release_manifest_ref": str(manifest),
    }
    manifest.write_text(
        json.dumps({
            "payload": {
                "collection": "knowledge_v2",
                "index_fingerprint": {"digest": "abc"},
            },
        }),
        encoding="utf-8",
    )
    _validate_qdrant_release_manifest(spec)
    manifest.write_text(json.dumps({"payload": {"collection": "other"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="collection does not match"):
        _validate_qdrant_release_manifest(spec)


def test_rollout_transition_and_assignment_are_safe() -> None:
    assert transition_allowed("evaluated", "shadowing")
    assert not transition_allowed("evaluated", "canary_100")
    candidate = {
        "stage": "canary_25",
        "canary_percent": 25,
        "shadow_sample_rate": 1.0,
        "tenant_allowlist": [],
    }
    context = TenantContext(tenant_id="tenant-a", user_id="user-a", role="admin", scope_type="platform")
    first = rollout_assignment(None, candidate, context, "session-1", "trace-1")
    second = rollout_assignment(None, candidate, context, "session-1", "trace-2")
    assert first["bucket"] == second["bucket"]
    assert first["canary"] == second["canary"]


def _hit(source_path: str, text: str) -> RagSearchHit:
    return RagSearchHit(
        chunk_id="chunk-1",
        text=text,
        score=1.0,
        citation=RagCitation(
            source_path=source_path,
            title="",
            doc_type="metric",
            metric_name="gmv",
            chunk_id="chunk-1",
            section=None,
            index_version="rag-v2",
        ),
    )


def test_retrieval_metrics_reward_correct_document_and_evidence() -> None:
    doc = "docs/knowledge/metrics/gmv.md"
    case = {"expected_doc_ids": [doc], "expected_chunk_snippets": ["成交金额"]}
    hits = [_hit("other.md", "x"), _hit(doc, "成交金额")]
    assert not doc_hit_at_k(hits[:1], case, 1)
    assert doc_hit_at_k(hits, case, 2)
    assert chunk_hit_at_k(hits, case, 2)
    assert mrr_at_k(hits, case, 5) == 0.5
    assert ndcg_at_k(hits, case, 5) > 0


def test_evaluation_artifact_binds_dataset_and_runtime_hash(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        runner,
        "evaluate_all",
        lambda **_kwargs: {"rule": {"mrr@5": 1.0}},
    )
    output, payload = runner.write_results(tmp_path / "retrieval.json")
    assert output.exists()
    assert len(payload["dataset_hash"]) == 64
    assert len(payload["runtime_config_hash"]) == 64
    assert payload["runtime_config"]["candidate_k"] >= payload["runtime_config"]["top_k"]


def test_weight_grid_restores_runtime_defaults(monkeypatch) -> None:
    settings = runner.get_settings()
    original = settings.rag_vector_weight
    monkeypatch.setattr(
        runner,
        "evaluate",
        lambda strategy, top_k, backend_override: {
            "strategy": strategy,
            "top_k": top_k,
            "backend": backend_override,
        },
    )
    result = runner.evaluate_weight_grid([{"vector_weight": 0.8}], top_k=5)
    assert len(result) == 1
    assert len(result[0]["runtime_config_hash"]) == 64
    assert settings.rag_vector_weight == original

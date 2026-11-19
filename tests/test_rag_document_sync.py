from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from llama_index.core.embeddings import MockEmbedding
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, select

import llama_index.embeddings.huggingface as hf_embeddings

from app.rag import indexing
from app.rag.document_sync import (
    _sync_native_candidate,
    enqueue_reconciliation,
    process_one_reconciliation,
)
from app.rag.sync_store import RagDocumentSyncStore, corpus_snapshot


def _patch_embedding_and_manifests(monkeypatch, manifests: dict[str, dict]) -> None:
    monkeypatch.setattr(
        hf_embeddings,
        "HuggingFaceEmbedding",
        lambda model_name: MockEmbedding(embed_dim=8),
    )
    monkeypatch.setattr(
        indexing,
        "_load_manifest",
        lambda collection_name=None: manifests.get(
            collection_name, {"schema_version": 2, "documents": {}}
        ),
    )
    monkeypatch.setattr(
        indexing,
        "_write_manifest",
        lambda manifest, collection_name=None: manifests.__setitem__(collection_name, manifest),
    )


def test_sync_queue_deduplicates_and_supersedes_stale_snapshots() -> None:
    engine = create_engine("sqlite:///:memory:")
    store = RagDocumentSyncStore(engine, create_schema=True)
    old = store.enqueue(collection_name="knowledge", corpus_hash="a" * 64, source_revision="r1")
    assert store.enqueue(collection_name="knowledge", corpus_hash="a" * 64, source_revision="r1") == old
    new = store.enqueue(collection_name="knowledge", corpus_hash="b" * 64, source_revision="r2")
    assert old != new
    with engine.connect() as conn:
        statuses = dict(conn.execute(
            store.jobs.select().with_only_columns(store.jobs.c.corpus_hash, store.jobs.c.status)
        ).all())
    assert statuses == {"a" * 64: "superseded", "b" * 64: "queued"}
    claimed = store.claim(worker_id="worker-1")
    assert claimed["job_id"] == new
    assert claimed["attempts"] == 1
    assert store.renew_lease(new, "worker-1") is True
    store.complete(new)
    assert store.claim(worker_id="worker-2") is None
    engine.dispose()


def test_retry_and_source_tombstones_are_persisted() -> None:
    engine = create_engine("sqlite:///:memory:")
    store = RagDocumentSyncStore(engine, create_schema=True)
    job_id = store.enqueue(collection_name="knowledge", corpus_hash="c" * 64, source_revision="r1")
    job = store.claim(worker_id="worker")
    store.fail(job, "temporary qdrant failure", retry_delay_seconds=0)
    retry = store.claim(worker_id="worker-2")
    assert retry["job_id"] == job_id
    assert retry["status"] == "running"
    store.record_snapshot(
        collection_name="knowledge",
        source_revision="r1",
        documents={"docs/a.md": "a" * 64, "docs/b.md": "b" * 64},
        succeeded=True,
    )
    store.record_snapshot(
        collection_name="knowledge",
        source_revision="r2",
        documents={"docs/a.md": "c" * 64},
        succeeded=True,
    )
    sources = {row["source_id"]: row for row in store.list_sources("knowledge")}
    assert sources["docs/a.md"]["indexed_hash"] == "c" * 64
    assert sources["docs/a.md"]["status"] == "indexed"
    assert sources["docs/b.md"]["status"] == "deleted"
    assert sources["docs/b.md"]["indexed_hash"] is None
    engine.dispose()


def test_expired_final_worker_lease_becomes_terminal(tmp_path: Path) -> None:
    engine = create_engine("sqlite:///:memory:")
    store = RagDocumentSyncStore(engine, create_schema=True)
    job_id = store.enqueue(collection_name="knowledge", corpus_hash="d" * 64, source_revision="r1")
    with engine.begin() as conn:
        conn.execute(store.jobs.update().where(store.jobs.c.job_id == job_id).values(max_attempts=1))
    claimed = store.claim(worker_id="crashed-worker", lease_seconds=1)
    assert claimed and claimed["attempts"] == 1
    with engine.begin() as conn:
        conn.execute(store.jobs.update().where(store.jobs.c.job_id == job_id).values(
            lease_expires_at=claimed["updated_at"] - timedelta(seconds=1),
        ))
    assert store.claim(worker_id="recovery-worker") is None
    with engine.connect() as conn:
        row = conn.execute(select(store.jobs).where(store.jobs.c.job_id == job_id)).mappings().one()
    assert row["status"] == "failed"
    assert "lease expired" in row["last_error"]
    engine.dispose()


def test_document_sync_end_to_end_add_update_delete_retry_and_alias(
    tmp_path: Path, monkeypatch,
) -> None:
    root = tmp_path / "knowledge"
    root.mkdir()
    first = root / "gmv.md"
    second = root / "sop.md"
    first.write_text("# GMV\n\nFirst definition", encoding="utf-8")
    second.write_text("# SOP\n\nOperational guidance", encoding="utf-8")

    engine = create_engine("sqlite:///:memory:")
    store = RagDocumentSyncStore(engine, create_schema=True)
    digest, snapshot = corpus_snapshot(str(root))
    assert digest and set(snapshot) == {"gmv.md", "sop.md"}
    qdrant = QdrantClient(":memory:")
    manifests: dict[str, dict] = {}
    _patch_embedding_and_manifests(monkeypatch, manifests)

    def indexer(*, root: str, collection_name: str):
        return indexing.build_qdrant_index(root=root, collection_name=collection_name, client=qdrant)

    first_job = enqueue_reconciliation(
        root=str(root), collection_name="knowledge_v1", store=store, revision="git-a",
    )
    first_result = process_one_reconciliation(
        root=str(root), collection_name="knowledge_v1", store=store,
        worker_id="worker-1", indexer=indexer, collection_resolver=lambda name: name,
    )
    assert first_result["status"] == "ok", first_result
    assert first_result["job_id"] == first_job["job_id"]
    assert qdrant.count("knowledge_v1", exact=True).count == 2

    first.write_text("# GMV\n\nUpdated definition with more detail", encoding="utf-8")
    updated_job = enqueue_reconciliation(
        root=str(root), collection_name="knowledge_v1", store=store, revision="git-b",
    )
    update_result = process_one_reconciliation(
        root=str(root), collection_name="knowledge_v1", store=store,
        worker_id="worker-1", indexer=indexer, collection_resolver=lambda name: name,
    )
    assert updated_job["job_id"] != first_job["job_id"]
    assert update_result["changed_documents"] == 1
    assert qdrant.count("knowledge_v1", exact=True).count == 2
    points, _ = qdrant.scroll("knowledge_v1", with_payload=True, limit=10)
    gmv_points = [point for point in points if point.payload.get("source_path", "").endswith("gmv.md")]
    assert len(gmv_points) == 1
    assert "Updated definition" in json.loads(gmv_points[0].payload["_node_content"])["text"]

    second.unlink()
    enqueue_reconciliation(
        root=str(root), collection_name="knowledge_v1", store=store, revision="git-c",
    )
    deletion_result = process_one_reconciliation(
        root=str(root), collection_name="knowledge_v1", store=store,
        worker_id="worker-1", indexer=indexer, collection_resolver=lambda name: name,
    )
    assert deletion_result["deleted_documents"] == 1
    assert qdrant.count("knowledge_v1", exact=True).count == 1
    source_rows = {row["source_id"]: row for row in store.list_sources("knowledge_v1")}
    deleted_source = next(path for path in source_rows if path.endswith("sop.md"))
    assert source_rows[deleted_source]["status"] == "deleted"

    # Reconciliation is idempotent: the same snapshot does not create work or
    # duplicate Qdrant points after a retry.
    repeated = enqueue_reconciliation(
        root=str(root), collection_name="knowledge_v1", store=store, revision="git-c",
    )
    with engine.connect() as conn:
        completed_job_id = conn.execute(select(store.jobs.c.job_id).where(
            store.jobs.c.corpus_hash == repeated["corpus_hash"],
            store.jobs.c.collection_name == "knowledge_v1",
        )).scalar_one()
    assert repeated["job_id"] == completed_job_id
    assert process_one_reconciliation(
        root=str(root), collection_name="knowledge_v1", store=store,
        worker_id="worker-1", indexer=indexer, collection_resolver=lambda name: name,
    ) is None
    assert qdrant.count("knowledge_v1", exact=True).count == 1

    # Full index releases are built alongside the active collection and then
    # switched by one atomic alias update.
    release_result = indexing.build_qdrant_index(
        root=str(root), collection_name="knowledge_v2", client=qdrant,
    )
    assert release_result["status"] == "ok"
    digest = manifests["knowledge_v2"]["index_fingerprint"]["payload"]["corpus_digest"]
    monkeypatch.setattr(indexing, "_current_corpus_digest", lambda root=None: digest)
    publication = indexing.switch_collection_alias(qdrant, "knowledge_active", "knowledge_v2")
    assert publication["previous_collection"] is None
    assert indexing.resolve_collection_alias(qdrant, "knowledge_active") == "knowledge_v2"
    assert indexing.check_collection_fingerprint(
        qdrant, "knowledge_active", index_mode="dense",
    )["status"] == "match"

    # Native hybrid changes rebuild a versioned dense+sparse collection and
    # publish the stable alias; its sparse model artifact travels with it.
    corpus_hash, _ = corpus_snapshot(str(root))
    native_candidate = f"hybrid_active_sync_{corpus_hash[:16]}"
    monkeypatch.setattr(
        indexing, "_sparse_model_path",
        lambda collection_name=None: tmp_path / f"{collection_name}.sparse.json",
    )
    monkeypatch.setattr(
        indexing, "_current_corpus_digest",
        lambda root=None: manifests[native_candidate]["index_fingerprint"]["payload"]["corpus_digest"],
    )

    def native_builder(*, root: str, collection_name: str, client):
        return indexing.build_qdrant_index_native(
            root=root,
            collection_name=collection_name,
            client=client,
            embed_fn=lambda text: [0.25] * 8,
            sparse_model_path=tmp_path / f"{collection_name}.sparse.json",
        )

    native_result = _sync_native_candidate(
        str(root), "hybrid_active", corpus_hash,
        native_indexer=native_builder,
        client_factory=lambda: qdrant,
    )
    assert native_result["status"] == "ok"
    assert native_result["publication"]["index_mode"] == "dense_sparse"
    assert native_result["publication"]["alias"] == "hybrid_active"

    qdrant.close()
    engine.dispose()


def test_dense_index_recovers_after_partial_update_write(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "knowledge"
    root.mkdir()
    document = root / "metric.md"
    document.write_text("# GMV\n\nVersion one", encoding="utf-8")
    qdrant = QdrantClient(":memory:")
    manifests: dict[str, dict] = {}
    _patch_embedding_and_manifests(monkeypatch, manifests)
    monkeypatch.setattr(
        indexing, "_sync_state_path", lambda collection: tmp_path / f"{collection}.sync-state.json",
    )

    assert indexing.build_qdrant_index(
        root=str(root), collection_name="recoverable", client=qdrant,
    )["status"] == "ok"
    assert qdrant.count("recoverable", exact=True).count == 1

    document.write_text("# GMV\n\nVersion two after interruption", encoding="utf-8")
    delete = qdrant.delete
    failed_once = False

    def fail_stale_delete(*args, **kwargs):
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise RuntimeError("simulated interruption during stale point cleanup")
        return delete(*args, **kwargs)

    monkeypatch.setattr(qdrant, "delete", fail_stale_delete)
    interrupted = indexing.build_qdrant_index(
        root=str(root), collection_name="recoverable", client=qdrant,
    )
    assert interrupted["status"] == "index_error"
    assert (tmp_path / "recoverable.sync-state.json").exists()
    # Stable chunk IDs make the upsert idempotent even though stale cleanup
    # failed; the persisted sync state makes the next pass repeat cleanup.
    assert qdrant.count("recoverable", exact=True).count == 1

    monkeypatch.setattr(qdrant, "delete", delete)
    recovered = indexing.build_qdrant_index(
        root=str(root), collection_name="recoverable", client=qdrant,
    )
    assert recovered["status"] == "ok"
    assert qdrant.count("recoverable", exact=True).count == 1
    assert not (tmp_path / "recoverable.sync-state.json").exists()
    points, _ = qdrant.scroll("recoverable", with_payload=True, limit=10)
    assert "Version two after interruption" in json.loads(points[0].payload["_node_content"])["text"]
    qdrant.close()


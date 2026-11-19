from __future__ import annotations

"""Document change detection, durable reconciliation, and local file watching."""

from pathlib import Path
import subprocess
import threading
import time
import random
from typing import Callable
import uuid

from app.core.config import get_settings
from app.rag.config import get_rag_settings
from app.rag.indexing import (
    IndexSafetyError,
    build_qdrant_index,
    build_qdrant_index_native,
    resolve_collection_alias,
    switch_collection_alias,
)
from app.rag.sync_store import RagDocumentSyncStore, corpus_snapshot


def source_revision(root: str, corpus_hash: str) -> str:
    """Use the checked-out Git revision when available, otherwise content digest."""
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(root).resolve()), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        revision = result.stdout.strip()
        if revision:
            return revision[:128]
    except (OSError, subprocess.SubprocessError):
        pass
    return corpus_hash


def resolve_active_collection(collection_name: str) -> str:
    """Resolve a stable query alias to the physical collection used for writes."""
    from qdrant_client import QdrantClient

    client = QdrantClient(url=get_settings().qdrant_url)
    try:
        return resolve_collection_alias(client, collection_name)
    finally:
        client.close()


def _sync_native_candidate(
    root: str,
    alias_name: str,
    corpus_hash: str,
    *,
    native_indexer: Callable[..., dict[str, object]] = build_qdrant_index_native,
    client_factory: Callable[[], object] | None = None,
) -> dict[str, object]:
    """Full-build native dense+sparse artifacts and atomically publish the alias."""
    if not alias_name:
        return {
            "status": "safety_error",
            "error": "RAG_BACKEND=qdrant_hybrid requires RAG_QDRANT_ACTIVE_ALIAS for versioned full rebuilds",
        }
    from qdrant_client import QdrantClient

    suffix = corpus_hash[:16]
    candidate = f"{alias_name[:120]}_sync_{suffix}"
    client = client_factory() if client_factory else QdrantClient(url=get_settings().qdrant_url)
    try:
        if client.collection_exists(candidate):
            try:
                publication = switch_collection_alias(client, alias_name, candidate)
                return {
                    "status": "ok",
                    "collection": candidate,
                    "publication": publication,
                    "reused_candidate": True,
                }
            except IndexSafetyError:
                if resolve_collection_alias(client, alias_name) == candidate:
                    raise
                # Candidate names are deterministic and owned by this corpus
                # digest. A partial build can be discarded and retried safely.
                client.delete_collection(candidate)
        result = native_indexer(root=root, collection_name=candidate, client=client)
        if result.get("status") != "ok":
            return result
        publication = switch_collection_alias(client, alias_name, candidate)
        return {**result, "publication": publication}
    finally:
        client.close()


def enqueue_reconciliation(
    *,
    root: str | None = None,
    collection_name: str | None = None,
    store: RagDocumentSyncStore | None = None,
    revision: str | None = None,
    retry_failed: bool = False,
) -> dict[str, object]:
    settings = get_settings()
    root = root or settings.rag_knowledge_root
    collection_name = collection_name or settings.rag_qdrant_active_alias or settings.qdrant_collection
    store = store or RagDocumentSyncStore()
    digest, documents = corpus_snapshot(root)
    source_rev = (revision or source_revision(root, digest))[:128]
    job_id = store.enqueue(
        collection_name=collection_name,
        corpus_hash=digest,
        source_revision=source_rev,
        retry_failed=retry_failed,
    )
    return {
        "job_id": job_id,
        "collection": collection_name,
        "corpus_hash": digest,
        "source_revision": source_rev,
        "documents": len(documents),
    }


def process_one_reconciliation(
    *,
    root: str | None = None,
    collection_name: str | None = None,
    store: RagDocumentSyncStore | None = None,
    worker_id: str | None = None,
    indexer: Callable[..., dict[str, object]] = build_qdrant_index,
    collection_resolver: Callable[[str], str] = resolve_active_collection,
    native_indexer: Callable[..., dict[str, object]] = build_qdrant_index_native,
) -> dict[str, object] | None:
    """Claim and process one durable job; retries are safe and at-least-once."""
    settings = get_settings()
    root = root or settings.rag_knowledge_root
    collection_name = collection_name or settings.rag_qdrant_active_alias or settings.qdrant_collection
    store = store or RagDocumentSyncStore()
    worker_id = worker_id or f"rag-sync-{uuid.uuid4().hex[:12]}"
    job = store.claim(worker_id=worker_id)
    if job is None:
        return None

    with store.collection_lock(collection_name) as acquired:
        if not acquired:
            store.fail(job, "another worker is synchronizing this collection", retry_delay_seconds=3)
            return {"status": "retrying", "job_id": job["job_id"], "reason": "collection_locked"}

        latest_hash, documents = corpus_snapshot(root)
        if latest_hash != job["corpus_hash"]:
            store.supersede(str(job["job_id"]))
            queued = enqueue_reconciliation(
                root=root,
                collection_name=collection_name,
                store=store,
            )
            return {"status": "superseded", "job_id": job["job_id"], "queued_job_id": queued["job_id"]}

        heartbeat_stop = threading.Event()

        def heartbeat() -> None:
            while not heartbeat_stop.wait(60):
                if not store.renew_lease(str(job["job_id"]), worker_id, lease_seconds=300):
                    return

        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()
        try:
            rag_backend = get_rag_settings().backend
            if rag_backend == "qdrant_hybrid":
                result = _sync_native_candidate(
                    root, get_rag_settings().active_alias, str(job["corpus_hash"]),
                    native_indexer=native_indexer,
                )
            else:
                write_collection = collection_resolver(collection_name)
                result = indexer(root=root, collection_name=write_collection)
        except Exception as exc:
            result = {"status": "index_error", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1)
        status = str(result.get("status") or "index_error")
        succeeded = status in {"ok", "skipped", "empty"}
        error = "" if succeeded else str(result.get("error") or status)
        store.record_snapshot(
            collection_name=collection_name,
            source_revision=str(job["source_revision"]),
            documents=documents,
            succeeded=succeeded,
            error=error,
        )
        if succeeded:
            store.complete(str(job["job_id"]))
            return {**result, "job_id": job["job_id"], "status": status}
        retry_delay = min(900, 5 * (2 ** max(0, int(job["attempts"]) - 1)))
        retry_delay += random.randint(0, min(10, max(1, retry_delay // 4)))
        store.fail(
            job, error, retry_delay_seconds=retry_delay,
            permanent=(status == "safety_error"),
        )
        return {**result, "job_id": job["job_id"], "status": "retrying", "index_status": status}


def run_worker(
    *,
    root: str | None = None,
    collection_name: str | None = None,
    poll_seconds: float = 2.0,
    reconcile_seconds: float = 60.0,
    once: bool = False,
    store: RagDocumentSyncStore | None = None,
    on_result: Callable[[dict[str, object]], None] | None = None,
) -> list[dict[str, object]]:
    """Poll durable work and periodically reconcile missed file events."""
    store = store or RagDocumentSyncStore()
    results: list[dict[str, object]] = []

    def report(result: dict[str, object]) -> None:
        if once:
            results.append(result)
        if on_result is not None:
            on_result(result)
    last_reconcile = 0.0
    while True:
        now = time.monotonic()
        if now - last_reconcile >= reconcile_seconds:
            report(enqueue_reconciliation(
                root=root, collection_name=collection_name, store=store,
            ))
            last_reconcile = now
        result = process_one_reconciliation(
            root=root, collection_name=collection_name, store=store,
        )
        if result is not None:
            report(result)
        if once:
            return results
        time.sleep(poll_seconds)


def watch_knowledge_files(
    *,
    root: str | None = None,
    collection_name: str | None = None,
    debounce_seconds: float = 1.0,
    with_worker: bool = False,
    store: RagDocumentSyncStore | None = None,
) -> None:
    """Watch local Markdown edits; production should also run the periodic worker."""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError as exc:  # pragma: no cover - optional platform dependency
        raise RuntimeError("Install watchdog from requirements-rag-qdrant-runtime.txt") from exc

    settings = get_settings()
    root = str(Path(root or settings.rag_knowledge_root).resolve())
    store = store or RagDocumentSyncStore()
    enqueue_reconciliation(root=root, collection_name=collection_name, store=store)
    timer: threading.Timer | None = None
    lock = threading.Lock()

    def schedule() -> None:
        nonlocal timer
        with lock:
            if timer is not None:
                timer.cancel()
            timer = threading.Timer(
                debounce_seconds,
                lambda: enqueue_reconciliation(
                    root=root, collection_name=collection_name, store=store,
                ),
            )
            timer.daemon = True
            timer.start()

    class Handler(FileSystemEventHandler):
        def on_any_event(self, event):  # noqa: ANN001
            if event.is_directory:
                return
            paths = [getattr(event, "src_path", ""), getattr(event, "dest_path", "")]
            if any(path and path.lower().endswith(".md") for path in paths):
                schedule()

    observer = Observer()
    observer.schedule(Handler(), root, recursive=True)
    observer.start()
    worker = None
    if with_worker:
        worker = threading.Thread(
            target=run_worker,
            kwargs={"root": root, "collection_name": collection_name, "store": store},
            daemon=True,
        )
        worker.start()
    try:
        while observer.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join(timeout=5)
        if timer is not None:
            timer.cancel()



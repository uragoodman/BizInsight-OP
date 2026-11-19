"""Build a no-secret release manifest for a concrete RAG artifact."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.rag.config import get_rag_settings
from app.rag.fingerprint import corpus_digest
from app.rag.indexing import _load_manifest, _sparse_model_path, load_knowledge_chunks


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return value


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _vector_schema(collection_info: Any) -> dict[str, Any]:
    config = getattr(collection_info, "config", None)
    params = getattr(config, "params", None)
    vectors = getattr(params, "vectors", None)
    sparse = getattr(params, "sparse_vectors", None)

    def vector_value(value: Any) -> Any:
        value = _jsonable(value)
        if isinstance(value, dict):
            return {
                key: value.get(key)
                for key in ("size", "distance", "datatype", "multivector_config")
                if value.get(key) is not None
            }
        return value

    if isinstance(vectors, dict):
        dense_schema = {str(name): vector_value(value) for name, value in vectors.items()}
    else:
        dense_schema = vector_value(vectors)
    if isinstance(sparse, dict):
        sparse_schema = {str(name): vector_value(value) for name, value in sparse.items()}
    else:
        sparse_schema = vector_value(sparse)
    return {"dense": dense_schema, "sparse": sparse_schema}


def build_release_manifest(
    *,
    collection_name: str,
    snapshot_ref: str,
    client=None,
    knowledge_root: str | None = None,
    backup_ref: str | None = None,
    backup_sha256: str | None = None,
) -> dict[str, Any]:
    """Read current artifacts and return a release manifest without writes."""

    if not collection_name.strip():
        raise ValueError("collection_name is required")
    if not snapshot_ref.strip():
        raise ValueError("snapshot_ref is required")

    rag_settings = get_rag_settings()
    app_settings = get_settings()
    collection_info = client.get_collection(collection_name) if client is not None else None
    point_count = None
    if client is not None:
        point_count = int(client.count(collection_name).count)

    chunks = load_knowledge_chunks(knowledge_root)
    index_manifest = _load_manifest(collection_name)
    sparse_path = _sparse_model_path(collection_name)
    sparse_asset = {
        "path": str(sparse_path),
        "sha256": _sha256_file(sparse_path),
        "exists": sparse_path.exists(),
    }
    payload: dict[str, Any] = {
        "manifest_schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "app_version": app_settings.app_version,
        "rag_backend": rag_settings.backend,
        "collection": collection_name,
        "snapshot_ref": snapshot_ref,
        "backup_ref": backup_ref,
        "backup_sha256": backup_sha256,
        "point_count": point_count,
        "collection_schema": _vector_schema(collection_info) if collection_info else None,
        "index_manifest_schema_version": index_manifest.get("schema_version"),
        "index_fingerprint": index_manifest.get("index_fingerprint"),
        "embedding_model": rag_settings.embedding_model,
        "embedding_model_revision": rag_settings.embedding_model_revision,
        "index_version": rag_settings.index_version,
        "chunk_pipeline_version": rag_settings.chunk_pipeline_version,
        "corpus_digest": corpus_digest(chunks),
        "document_hashes": sorted({
            str(chunk.document_hash)
            for chunk in chunks
            if chunk.document_hash
        }),
        "sparse_asset": sparse_asset,
        "runtime_config": {
            "top_k": rag_settings.top_k,
            "candidate_k": rag_settings.candidate_k,
            "reranker_top_k": rag_settings.reranker_top_k,
            "reranker_strategy": rag_settings.reranker_strategy,
            "vector_weight": rag_settings.vector_weight,
            "sparse_weight": rag_settings.sparse_weight,
            "semantic_weight": rag_settings.semantic_weight,
            "metadata_weight": rag_settings.metadata_weight,
        },
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "release_id": "ragrel-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24],
        "payload": payload,
    }


def write_release_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    """Write a manifest atomically enough for a local release record."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return target

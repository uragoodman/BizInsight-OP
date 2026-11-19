"""Deterministic fingerprints for RAG index compatibility checks."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


FINGERPRINT_ALGORITHM = "sha256"
FINGERPRINT_SCHEMA_VERSION = 1
SPARSE_FIELD_COMPOSITION = ("title", "section", "text", "tags")


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def corpus_digest(chunks: Iterable[object]) -> str:
    """Hash the complete document snapshot in deterministic path order."""

    documents: dict[str, str] = {}
    for chunk in chunks:
        doc_id = str(getattr(chunk, "doc_id", ""))
        document_hash = str(getattr(chunk, "document_hash", "") or "")
        documents[doc_id] = document_hash
    return _digest(sorted(documents.items()))


def sparse_model_digest(vocab: dict[str, int], idf: dict[int, float]) -> str:
    """Hash the exact sparse vocabulary and IDF values used by the index."""

    return _digest({
        "vocab": dict(sorted(vocab.items())),
        "idf": {str(key): idf[key] for key in sorted(idf)},
    })


def build_index_fingerprint(
    *,
    index_mode: str,
    collection: str,
    vector_names: list[str],
    distance: str,
    dimension: int | None,
    embedding_model: str,
    embedding_model_revision: str,
    indexing_code_version: str,
    chunk_pipeline_version: str,
    chunk_parameters: dict[str, Any],
    tokenizer_id: str,
    sparse_vocab_hash: str | None,
    corpus_hash: str,
    index_version: str,
) -> dict[str, Any]:
    """Return both the canonical payload and its hash for release manifests."""

    payload = {
        "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "index_mode": index_mode,
        "collection": collection,
        "vector_names": sorted(vector_names),
        "distance": distance,
        "dimension": dimension,
        "embedding_model": embedding_model,
        "embedding_model_revision": embedding_model_revision,
        "indexing_code_version": indexing_code_version,
        "chunk_pipeline_version": chunk_pipeline_version,
        "chunk_parameters": chunk_parameters,
        "tokenizer_id": tokenizer_id,
        "sparse_vocab_hash": sparse_vocab_hash,
        "sparse_field_composition": list(SPARSE_FIELD_COMPOSITION),
        "corpus_digest": corpus_hash,
        "index_version": index_version,
    }
    return {
        "algorithm": FINGERPRINT_ALGORITHM,
        "payload": payload,
        "digest": _digest(payload),
    }

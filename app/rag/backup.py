"""Non-destructive Qdrant point export and offline verification helpers."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_collection(
    client,
    *,
    collection_name: str,
    output_path: str | Path,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Export every point, vector and payload without mutating Qdrant.

    The first JSONL record describes the collection schema.  Remaining records
    are complete Qdrant points and can be replayed into an explicitly new
    collection by a separate restore tool.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"backup output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    info = client.get_collection(collection_name)
    schema = _jsonable(getattr(getattr(info, "config", None), "params", None))
    point_count = int(client.count(collection_name).count)
    temporary = target.with_suffix(target.suffix + ".tmp")
    opener = gzip.open if target.suffix == ".gz" else open
    exported = 0
    try:
        with opener(temporary, "wt", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "kind": "qdrant_point_export",
                "schema_version": 1,
                "collection": collection_name,
                "point_count": point_count,
                "collection_params": schema,
            }, ensure_ascii=False, sort_keys=True) + "\n")
            offset = None
            while True:
                points, offset = client.scroll(
                    collection_name=collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_vectors=True,
                    with_payload=True,
                )
                for point in points:
                    stream.write(json.dumps({
                        "id": _jsonable(getattr(point, "id", None)),
                        "vector": _jsonable(getattr(point, "vector", None)),
                        "payload": _jsonable(getattr(point, "payload", None)),
                    }, ensure_ascii=False, sort_keys=True) + "\n")
                    exported += 1
                if offset is None or not points:
                    break
        temporary.replace(target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return {
        "path": str(target),
        "collection": collection_name,
        "point_count": point_count,
        "exported_points": exported,
        "sha256": _sha256(target),
    }


def verify_export(path: str | Path) -> dict[str, Any]:
    """Verify header, point count and duplicate point IDs without restoring."""

    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    header: dict[str, Any] | None = None
    point_ids: set[str] = set()
    point_count = 0
    with opener(source, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = json.loads(line)
            if line_number == 1:
                header = record
                if header.get("kind") != "qdrant_point_export":
                    raise ValueError("unsupported Qdrant export header")
                continue
            point_id = json.dumps(record.get("id"), sort_keys=True)
            if point_id in point_ids:
                raise ValueError(f"duplicate point id in export: {record.get('id')!r}")
            point_ids.add(point_id)
            point_count += 1
    if header is None:
        raise ValueError("empty Qdrant export")
    expected = header.get("point_count")
    if expected is not None and int(expected) != point_count:
        raise ValueError(f"point count mismatch: header={expected}, actual={point_count}")
    return {
        "valid": True,
        "collection": header.get("collection"),
        "point_count": point_count,
        "sha256": _sha256(source),
    }


def restore_export(
    client,
    *,
    input_path: str | Path,
    target_collection: str,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Restore an export only into a non-existing target collection.

    This is intentionally create-only: restoring over an existing collection
    is rejected so recovery testing cannot destroy the only copy.
    """

    from qdrant_client import models

    source = Path(input_path)
    verification = verify_export(source)
    if client.collection_exists(target_collection):
        raise FileExistsError(
            f"restore target already exists; choose an isolated collection: {target_collection}"
        )
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as stream:
        header = json.loads(next(stream))
        params = header.get("collection_params") or {}
        dense_raw = params.get("vectors")
        sparse_raw = params.get("sparse_vectors")

        def vector_params(value):
            return models.VectorParams(
                size=int(value["size"]),
                distance=models.Distance(value["distance"]),
            )

        vectors_config = (
            {name: vector_params(value) for name, value in dense_raw.items()}
            if isinstance(dense_raw, dict) and "size" not in dense_raw
            else vector_params(dense_raw)
        )
        sparse_config = None
        if isinstance(sparse_raw, dict):
            sparse_config = {
                name: models.SparseVectorParams(**{
                    key: item[key] for key in ("index", "modifier") if key in item
                })
                for name, item in sparse_raw.items()
            }
        client.create_collection(
            collection_name=target_collection,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config,
        )
        points = []
        for line in stream:
            record = json.loads(line)
            vector = record.get("vector")
            if isinstance(vector, dict):
                vector = {
                    name: models.SparseVector(**value)
                    if isinstance(value, dict) and "indices" in value
                    else value
                    for name, value in vector.items()
                }
            points.append(models.PointStruct(
                id=record.get("id"), vector=vector, payload=record.get("payload"),
            ))
            if len(points) >= batch_size:
                client.upsert(collection_name=target_collection, points=points)
                points = []
        if points:
            client.upsert(collection_name=target_collection, points=points)
    return {
        "collection": target_collection,
        "point_count": verification["point_count"],
        "source_sha256": verification["sha256"],
    }

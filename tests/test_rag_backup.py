from __future__ import annotations

import json
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from app.rag.backup import export_collection, restore_export, verify_export


class _Point:
    def __init__(self, point_id: str):
        self.id = point_id
        self.vector = {"default": [1.0, 0.0]}
        self.payload = {"chunk_id": point_id}


class _Info:
    config = type("Config", (), {"params": {"vectors": {"size": 2}}})()


class _Client:
    def get_collection(self, _name):
        return _Info()

    def count(self, _name):
        return type("Count", (), {"count": 2})()

    def scroll(self, collection_name, limit, offset, with_vectors, with_payload):
        if offset is None:
            return ([_Point("a")], "next")
        return ([_Point("b")], None)


def test_qdrant_export_is_complete_and_verifiable(tmp_path: Path) -> None:
    result = export_collection(
        _Client(), collection_name="old", output_path=tmp_path / "qdrant.jsonl",
    )
    assert result["exported_points"] == 2
    verified = verify_export(tmp_path / "qdrant.jsonl")
    assert verified["valid"] is True
    assert verified["point_count"] == 2


def test_qdrant_export_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "qdrant.jsonl"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_collection(_Client(), collection_name="old", output_path=output)


def test_qdrant_export_restore_round_trip_is_create_only(tmp_path: Path) -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="source",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    client.upsert(
        collection_name="source",
        points=[models.PointStruct(id=1, vector=[1.0, 0.0], payload={"text": "GMV"})],
    )
    output = tmp_path / "source.jsonl"
    export_collection(client, collection_name="source", output_path=output)
    result = restore_export(client, input_path=output, target_collection="source_restore")
    assert result["point_count"] == 1
    assert client.count("source_restore").count == 1
    with pytest.raises(FileExistsError):
        restore_export(client, input_path=output, target_collection="source_restore")
    client.close()

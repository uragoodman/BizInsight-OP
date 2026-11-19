from __future__ import annotations

from pathlib import Path

from app.rag.release_manifest import build_release_manifest, write_release_manifest


class _Info:
    config = type("Config", (), {
        "params": type("Params", (), {
            "vectors": {"dense": type("Vector", (), {"size": 4, "distance": "Cosine"})()},
            "sparse_vectors": {"sparse": type("Sparse", (), {})()},
        })(),
    })()
    points_count = 1


class _Client:
    def get_collection(self, _name):
        return _Info()

    def count(self, _name):
        return type("Count", (), {"count": 1})()


def test_release_manifest_binds_collection_schema_and_runtime_config(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("app.rag.release_manifest.load_knowledge_chunks", lambda _root=None: [])
    monkeypatch.setattr("app.rag.release_manifest._load_manifest", lambda _collection: {
        "schema_version": 2,
        "index_fingerprint": {"digest": "fingerprint"},
    })
    manifest = build_release_manifest(
        collection_name="bizinsight_knowledge_v2",
        snapshot_ref="snapshot-1",
        client=_Client(),
    )
    assert manifest["payload"]["collection"] == "bizinsight_knowledge_v2"
    assert manifest["payload"]["point_count"] == 1
    assert manifest["payload"]["index_fingerprint"]["digest"] == "fingerprint"
    assert manifest["payload"]["runtime_config"]["candidate_k"] >= 1

    target = write_release_manifest(tmp_path / "release.json", manifest)
    assert target.exists()

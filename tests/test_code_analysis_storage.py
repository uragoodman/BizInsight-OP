"""Focused tests for the Phase 11 object-storage boundary."""

from __future__ import annotations

import pytest

from app.code_analysis.storage import LocalArtifactStore, owner_artifact_prefix


def test_owner_prefix_is_stable_path_safe_and_non_identifying() -> None:
    prefix = owner_artifact_prefix("../租户/alpha", "user@example.com")
    assert prefix == owner_artifact_prefix("../租户/alpha", "user@example.com")
    assert prefix.startswith("tenants/")
    assert "租户" not in prefix
    assert "example.com" not in prefix
    assert ".." not in prefix


def test_local_store_rejects_traversal_and_bounds_reads(tmp_path) -> None:
    store = LocalArtifactStore(str(tmp_path))
    with pytest.raises(ValueError, match="Invalid artifact key"):
        store.put("../outside.txt", b"secret")

    key, digest, size = store.put("tenants/a/users/b/source.csv", b"abc")
    assert key == "tenants/a/users/b/source.csv"
    assert len(digest) == 64
    assert size == 3
    assert store.get(key, max_bytes=3) == b"abc"
    with pytest.raises(ValueError, match="allowed read size"):
        store.get(key, max_bytes=2)

"""Fast tests for Phase 11 model-proxy and artifact-store boundaries."""

from pathlib import Path
import tempfile

import pytest
from fastapi import HTTPException

from app.code_analysis.model_proxy import CompletionRequest, _validated_messages
from app.code_analysis.storage import LocalArtifactStore, owner_artifact_prefix


def test_model_proxy_drops_unregistered_openai_extension_fields() -> None:
    payload = CompletionRequest.model_validate(
        {
            "model": "client-selected-model",
            "messages": [{"role": "user", "content": "analyze"}],
            "tools": [{"type": "function", "function": {"name": "shell"}}],
            "tool_choice": "auto",
        }
    )
    serialized = payload.model_dump()
    assert "tools" not in serialized
    assert "tool_choice" not in serialized


def test_model_proxy_rejects_tool_messages_and_oversized_prompts() -> None:
    with pytest.raises(HTTPException) as role_error:
        _validated_messages([{"role": "tool", "content": "result"}])
    assert role_error.value.status_code == 422

    with pytest.raises(HTTPException) as size_error:
        _validated_messages([{"role": "user", "content": "x" * 1_000_001}])
    assert size_error.value.status_code == 413


def test_owner_prefix_does_not_expose_claims_or_allow_path_injection() -> None:
    prefix = owner_artifact_prefix("../../tenant@example.com", "用户/../admin")
    assert ".." not in prefix
    assert "tenant@example.com" not in prefix
    assert "用户" not in prefix
    assert prefix.startswith("tenants/")


def test_local_artifact_store_is_atomic_and_enforces_read_bound() -> None:
    Path("data").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phase11-test-", dir="data") as temporary:
        store = LocalArtifactStore(temporary)
        key, digest, size = store.put("tenants/a/users/b/result.json", b"12345")
        assert key == "tenants/a/users/b/result.json"
        assert size == 5
        assert len(digest) == 64
        assert store.get(key, max_bytes=5) == b"12345"
        with pytest.raises(ValueError, match="allowed read size"):
            store.get(key, max_bytes=4)
        with pytest.raises(ValueError, match="Invalid artifact key"):
            store.put("../escape", b"x")

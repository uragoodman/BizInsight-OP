"""Tests for the narrow OpenAI-compatible model-proxy request contract."""

from __future__ import annotations

import pytest
import httpx
from fastapi import HTTPException

from app.code_analysis.model_proxy import (
    CompletionRequest,
    _backoff_seconds,
    _retry_after_seconds,
    _upstream_error,
    _upstream_url,
    _validated_messages,
)


def test_proxy_request_drops_unregistered_extension_fields() -> None:
    payload = CompletionRequest.model_validate(
        {
            "model": "ignored-by-proxy",
            "messages": [{"role": "user", "content": "analyze"}],
            "tools": [{"type": "function", "function": {"name": "shell"}}],
            "upstream_url": "http://attacker.invalid",
        }
    )
    dumped = payload.model_dump()
    assert "tools" not in dumped
    assert "upstream_url" not in dumped


def test_proxy_messages_are_bounded_and_tool_roles_are_rejected() -> None:
    assert _validated_messages([{"role": "user", "content": "ok"}]) == [
        {"role": "user", "content": "ok"}
    ]
    with pytest.raises(HTTPException) as role_error:
        _validated_messages([{"role": "tool", "content": "secret"}])
    assert role_error.value.status_code == 422
    with pytest.raises(HTTPException) as size_error:
        _validated_messages([{"role": "user", "content": "x" * 1_000_001}])
    assert size_error.value.status_code == 413


def test_proxy_upstream_path_is_fixed_to_chat_completions() -> None:
    assert _upstream_url("https://provider.example/v1") == (
        "https://provider.example/v1/chat/completions"
    )
    assert _upstream_url("https://provider.example/v1/chat/completions") == (
        "https://provider.example/v1/chat/completions"
    )


def test_proxy_prefers_provider_retry_after_header() -> None:
    response = httpx.Response(429, headers={"Retry-After": "7"})
    assert _retry_after_seconds(response) == 7.0
    assert _backoff_seconds(response, attempt=0, base=2.0) == 7.0


def test_proxy_classifies_auth_and_rate_limit_failures() -> None:
    auth_error = _upstream_error(httpx.Response(401))
    rate_error = _upstream_error(httpx.Response(429))
    assert auth_error.status_code == 502
    assert auth_error.detail == "Model provider authentication failed"
    assert rate_error.status_code == 429
    assert rate_error.detail == "Model provider is rate limited"
    assert rate_error.headers == {"Retry-After": "1"}

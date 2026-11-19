"""OAuth-scoped HTTP transport for remote MCP servers."""

from __future__ import annotations

from typing import Protocol, Any

import httpx

from app.core.config import get_settings
from app.resilience import ExternalServiceError, execute_with_resilience


class TokenProvider(Protocol):
    def get_token(self, audience: str, scope: str) -> str: ...


class EnvironmentTokenProvider:
    def get_token(self, audience: str, scope: str) -> str:
        token = get_settings().remote_mcp_oauth_token
        if not token:
            raise ExternalServiceError("remote_mcp", "oauth_missing", "Remote MCP OAuth token is not configured.")
        return token


def call_remote_mcp_tool(tool_name: str, arguments: dict[str, Any], *, request_id: str, idempotency_key: str, token_provider: TokenProvider | None = None) -> dict[str, Any]:
    settings = get_settings()
    if not settings.remote_mcp_url:
        raise ExternalServiceError("remote_mcp", "url_missing", "Remote MCP URL is not configured.")
    token = (token_provider or EnvironmentTokenProvider()).get_token(settings.remote_mcp_audience, settings.remote_mcp_oauth_scope)
    headers = {
        "Authorization": f"Bearer {token}", "X-Request-ID": request_id,
        "Idempotency-Key": idempotency_key, "X-OAuth-Scope": settings.remote_mcp_oauth_scope,
    }
    payload = {"tool_name": tool_name, "arguments": arguments}
    def request() -> dict[str, Any]:
        try:
            response = httpx.post(settings.remote_mcp_url, json=payload, headers=headers, timeout=settings.mcp_timeout_seconds)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            raise ExternalServiceError("remote_mcp", "http_error", str(exc)) from exc
    return execute_with_resilience(request, "remote_mcp", timeout_seconds=settings.mcp_timeout_seconds)

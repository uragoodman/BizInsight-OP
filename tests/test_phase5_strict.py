from __future__ import annotations

import json

import pytest

from app.agents.planner import validate_or_fallback
from app.mcp.remote_client import call_remote_mcp_tool
from app.rag.release import load_release_manifest, published_rag_version
from app.security.governance import ToolPermissionPolicy


def test_planner_rejects_invalid_topology() -> None:
    plan = validate_or_fallback({"agents": ["root_cause"], "dependencies": {"root_cause": ["data_query"]}}, "anomaly_analysis")
    assert [agent.value for agent in plan.agents] == ["data_query", "root_cause"]


def test_release_manifest_only_selects_published(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"releases": [{"version": "canary", "source": "x", "status": "canary"}, {"version": "stable", "source": "x", "status": "published", "approved_by": "reviewer"}]}), encoding="utf-8")
    assert load_release_manifest(str(path)).published().version == "stable"


def test_default_deny_policy(tmp_path) -> None:
    path = tmp_path / "policy.json"
    path.write_text("{}", encoding="utf-8")
    assert not ToolPermissionPolicy(str(path)).allows("unknown", "viewer", "data_query", "query_gmv")


def test_platform_scope_uses_platform_grant_for_tenant_membership(tmp_path) -> None:
    # Isolate the policy's built-in defaults from deployment-specific policy files.
    policy = ToolPermissionPolicy(str(tmp_path / "missing-policy.json"))
    assert policy.allows(
        "tenant-example",
        "admin",
        "data_query",
        "compare_breakdown_by_dimension",
        scope_type="platform",
    )
    assert not policy.allows(
        "tenant-example",
        "admin",
        "data_query",
        "compare_breakdown_by_dimension",
        scope_type="seller",
    )


def test_remote_mcp_requires_oauth_configuration(monkeypatch) -> None:
    monkeypatch.setattr("app.mcp.remote_client.get_settings", lambda: type("Settings", (), {"remote_mcp_url": "http://example.test", "remote_mcp_audience": "api", "remote_mcp_oauth_scope": "read", "remote_mcp_oauth_token": "", "mcp_timeout_seconds": 1})())
    with pytest.raises(Exception):
        call_remote_mcp_tool("query_gmv", {}, request_id="r", idempotency_key="i")


def test_remote_mcp_forwards_scope_and_idempotency(monkeypatch) -> None:
    captured = {}
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"status": "ok"}
    class Token:
        def get_token(self, audience, scope):
            captured["audience"], captured["scope"] = audience, scope
            return "fixture-token"
    settings = type("Settings", (), {"remote_mcp_url": "http://fixture.test/mcp", "remote_mcp_audience": "analytics", "remote_mcp_oauth_scope": "biz.read", "remote_mcp_oauth_token": "", "mcp_timeout_seconds": 1})()
    monkeypatch.setattr("app.mcp.remote_client.get_settings", lambda: settings)
    monkeypatch.setattr("app.mcp.remote_client.httpx.post", lambda url, **kwargs: captured.update({"url": url, **kwargs}) or Response())
    assert call_remote_mcp_tool("query_gmv", {"metric_date": "2018-08-10"}, request_id="request-1", idempotency_key="idem-1", token_provider=Token()) == {"status": "ok"}
    assert captured["scope"] == "biz.read"
    assert captured["headers"]["Idempotency-Key"] == "idem-1"

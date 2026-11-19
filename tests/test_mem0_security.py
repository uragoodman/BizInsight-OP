"""WP3 security/failure tests that do not require a running Mem0 collection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.context.compiler import compile_turn_context
from app.memory.mem0_adapter import (
    Mem0ConfigurationError,
    Mem0PreferenceAdapter,
    Mem0Timeout,
    Mem0Unavailable,
    build_opaque_scope,
)
from app.memory.preference_provider import (
    Mem0PreferenceProvider,
    UnavailablePreferenceProvider,
    provider_for_context,
)
from app.memory.preference_registry import NullPreferenceRegistry, RegistryUnavailable
from app.tenancy import TenantContext


def _context(tenant: str = "t1", user: str = "u1") -> TenantContext:
    return TenantContext(tenant_id=tenant, user_id=user, role="admin", scope_type="platform")


def _settings(**overrides) -> SimpleNamespace:
    values = dict(
        mem0_qdrant_collection="bizinsight_user_preferences_v1",
        mem0_qdrant_url="http://127.0.0.1:6333",
        mem0_qdrant_api_key="",
        mem0_embedding_model="BAAI/bge-small-zh-v1.5",
        mem0_embedding_dims=512,
        mem0_hmac_secret="0123456789abcdef0123456789abcdef",
        mem0_scope_version="v1",
        mem0_history_db_path="",
        mem0_search_timeout_seconds=1.5,
        mem0_write_timeout_seconds=5.0,
        mem0_breaker_failures=5,
        mem0_breaker_window_seconds=30.0,
        mem0_top_k_per_type=5,
        mem0_top_k_total=10,
        mem0_relevance_threshold=0.35,
        mem0_llm_model="",
        mem0_llm_api_key="",
        mem0_llm_base_url="",
        deepseek_api_key="",
        deepseek_base_url="",
        qdrant_url="http://127.0.0.1:6333",
        qdrant_collection="bizinsight_knowledge",
        rag_qdrant_active_alias="",
        metric_semantic_qdrant_collection="metric_semantic_catalog_v1",
        mem0_enabled=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_opaque_scope_is_server_derived_and_isolated_by_tenant_user_seller():
    secret = "0123456789abcdef0123456789abcdef"
    a = build_opaque_scope(_context("t1", "u1"), hmac_secret=secret, scope_version="v1")
    b = build_opaque_scope(_context("t1", "u2"), hmac_secret=secret, scope_version="v1")
    c = build_opaque_scope(_context("t2", "u1"), hmac_secret=secret, scope_version="v1")
    v2 = build_opaque_scope(_context("t1", "u1"), hmac_secret=secret, scope_version="v2")
    assert a.entity_id.startswith("mem0_")
    assert len({a.entity_id, b.entity_id, c.entity_id, v2.entity_id}) == 4
    assert "t1" not in a.entity_id and "u1" not in a.entity_id


def test_missing_registry_fails_closed_instead_of_trusting_mem0_metadata():
    with pytest.raises(RegistryUnavailable) as exc:
        NullPreferenceRegistry().validate(
            context=_context(), entity_scope="mem0_x", memory_id="mem-1",
            operation_id="op-1", memory_type="presentation", memory_key="style",
        )
    assert exc.value.args[0] == "registry_not_configured"


def test_explicit_empty_type_scope_does_not_widen_to_all_types():
    from app.memory.mem0_adapter import _normalise_types

    assert _normalise_types(None) == ["analysis_method", "presentation", "scope_default"]
    assert _normalise_types([]) == []
    assert _normalise_types(["not-a-type"]) == []


def test_adapter_rejects_reserved_rag_and_metric_collections():
    for collection in ("bizinsight_knowledge", "metric_semantic_catalog_v1"):
        with pytest.raises(Mem0ConfigurationError):
            Mem0PreferenceAdapter(
                _context(), settings=_settings(mem0_qdrant_collection=collection),
            )


def test_invalid_mem0_hit_is_audited_without_logging_memory_content(monkeypatch):
    from app.memory import mem0_adapter

    events = []
    monkeypatch.setattr(mem0_adapter, "record_event", lambda name, attrs: events.append((name, attrs)))
    adapter = Mem0PreferenceAdapter(
        _context(), settings=_settings(), registry=NullPreferenceRegistry(), trace_id="trace-safe",
    )
    hit = adapter._validated_hit({
        "id": "mem-secret-id",
        "user_id": "mem0_someone-else",
        "memory": "private preference text",
        "metadata": {
            "memory_type": "presentation", "memory_key": "response_style",
            "entity_scope": "mem0_someone-else", "scope_version": "v1", "seller_scope": "",
            "memory_status": "active",
        },
    }, expected_types={"presentation"})

    assert hit is None
    assert events == [(
        "mem0.preference_hit_rejected",
        {
            "trace_id": "trace-safe",
            "mem0.rejection_reason": "entity_scope_mismatch",
            "mem0.collection": "bizinsight_user_preferences_v1",
        },
    )]
    assert "private preference text" not in repr(events)
    assert "mem-secret-id" not in repr(events)


def test_provider_disabled_does_not_create_mem0_client(monkeypatch):
    assert provider_for_context(_context(), settings=_settings(mem0_enabled=False)) is None


def test_provider_missing_tenant_context_degrades_explicitly(monkeypatch):
    provider = provider_for_context(None, settings=_settings())
    assert isinstance(provider, UnavailablePreferenceProvider)
    with pytest.raises(Mem0Unavailable) as exc:
        provider.search(need={}, state={})
    assert exc.value.reason == "trusted_tenant_context_missing"


def test_missing_production_registry_degrades_without_loading_mem0(monkeypatch):
    monkeypatch.setattr("app.memory.preference_provider.get_personalization_enabled", lambda _context: True)
    class MissingRegistry:
        def ensure_available(self):
            raise RegistryUnavailable("registry_table_missing")

    provider = Mem0PreferenceProvider(
        _context(), settings=_settings(), adapter=object(), registry=MissingRegistry(),
    )
    state = compile_turn_context({
        "user_message": "按我平时的方式分析 GMV",
        "previous_turn_context": {"authorized_metrics": [{"metric_id": "gmv"}]},
        "current_query_scope_fingerprint": "scope-a",
        "preference_provider": provider,
        "node_history": [],
    })
    assert state["preference_context"]["availability"] == "partial_provider_error"
    assert state["preference_context"]["provider_errors"][0]["error"] == "registry_table_missing"
    assert state["turn_resolution"]["source_needs"][0]["trigger_ids"] == ["P1"]


def test_timeout_provider_degrades_and_source_need_is_visible():
    class TimeoutProvider:
        def search(self, *, need, state):
            raise Mem0Timeout("search_timeout")

    state = compile_turn_context({
        "user_message": "按我平时的方式分析 GMV",
        "previous_turn_context": {"authorized_metrics": [{"metric_id": "gmv"}]},
        "current_query_scope_fingerprint": "scope-a",
        "preference_provider": TimeoutProvider(),
        "node_history": [],
    })
    context = state["preference_context"]
    assert context["availability"] == "partial_provider_error"
    assert context["provider_errors"][0]["error"] == "search_timeout"
    assert state["turn_resolution"]["source_needs"][0]["trigger_ids"] == ["P1"]
    assert state["turn_resolution"]["preference"]["provider_errors"][0]["reason"] == "search_timeout"


def test_non_trigger_does_not_call_provider():
    class CountingProvider:
        def __init__(self):
            self.calls = 0
        def search(self, *, need, state):
            self.calls += 1
            return []

    provider = CountingProvider()
    state = compile_turn_context({
        "user_message": "那它有哪些限制？",
        "previous_turn_context": {
            "authorized_metrics": [{"metric_id": "gmv"}],
            "context_slots": {"metric": {"value": "gmv", "source": "current_turn"}},
        },
        "current_query_scope_fingerprint": "scope-a",
        "preference_provider": provider,
        "node_history": [],
    })
    assert provider.calls == 0
    assert state["preference_context"]["status"] == "not_requested"


def test_p2_and_p3_are_explicit_trigger_types():
    p2 = compile_turn_context({
        "user_message": "请生成一份报告",
        "preference_provider": None,
        "memory_enabled": True,
        "node_history": [],
    })["preference_context"]
    assert p2["source_needs"][0]["trigger_ids"] == ["P2"]
    assert p2["source_needs"][0]["memory_types"] == ["presentation"]

    p3 = compile_turn_context({
        "user_message": "我有哪些偏好？",
        "preference_provider": None,
        "memory_enabled": True,
        "node_history": [],
    })["preference_context"]
    assert p3["source_needs"] == []
    assert p3["status"] == "not_requested"

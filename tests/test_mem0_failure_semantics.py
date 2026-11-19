"""WP3 failure/retry semantics with a deterministic in-process Mem0 double.

The real SDK/Qdrant path is covered separately by
``tests/test_mem0_qdrant_contract.py``.  These tests exercise adapter behavior
that would otherwise require fault injection into Mem0/Qdrant: timeouts,
circuit breaking, and cleanup after a simulated Registry commit failure.
"""

from __future__ import annotations

import time
import uuid
from types import SimpleNamespace

import pytest

from app.memory.mem0_adapter import (
    Mem0CircuitOpen,
    Mem0PreferenceAdapter,
    Mem0Timeout,
    Mem0Unavailable,
)
from app.memory.preference_registry import RegistryDecision
from app.tenancy import TenantContext


class _AllowRegistry:
    def validate(self, **_: object) -> RegistryDecision:
        return RegistryDecision(True)


class _FakeMemory:
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.search_calls = 0
        self.delete_calls: list[str] = []
        self.delete_all_calls = 0
        self.get_all_calls: list[dict] = []
        self.reject_operation_id_filter = False
        self.sleep_seconds = 0.0
        self.add_sleep_seconds = 0.0
        self.raise_search = False
        self.vector_store = SimpleNamespace(client=object())

    def add(self, text, *, user_id=None, metadata=None, infer=False):
        if self.add_sleep_seconds:
            time.sleep(self.add_sleep_seconds)
        memory_id = f"mem-{uuid.uuid4().hex[:8]}"
        self.rows[memory_id] = {
            "id": memory_id, "memory": text, "user_id": user_id,
            "metadata": dict(metadata or {}),
        }
        return {"results": [{"id": memory_id, "memory": text, "event": "ADD"}]}

    def search(self, query, *, top_k=20, filters=None, threshold=0.1, rerank=False):
        self.search_calls += 1
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        if self.raise_search:
            raise RuntimeError("injected search failure")
        filters = filters or {}
        rows = [
            row for row in self.rows.values()
            if all(
                row.get("user_id") == value
                if key == "user_id"
                else (row.get("metadata") or {}).get(key) == value
                for key, value in filters.items()
            )
        ]
        return {"results": rows[:top_k]}

    def get_all(self, *, filters=None, top_k=20, show_expired=False):
        filters = filters or {}
        self.get_all_calls.append(dict(filters))
        if self.reject_operation_id_filter and "operation_id" in filters:
            raise RuntimeError("operation_id metadata filter is unsupported")
        rows = [
            row for row in self.rows.values()
            if all(
                row.get("user_id") == value
                if key == "user_id"
                else (row.get("metadata") or {}).get(key) == value
                for key, value in filters.items()
            )
        ]
        return {"results": rows[:top_k]}

    def get(self, memory_id):
        return self.rows.get(memory_id)

    def update(self, memory_id, text=None, metadata=None, **kwargs):
        row = self.rows[memory_id]
        if text is not None:
            row["memory"] = text
        if metadata is not None:
            row["metadata"] = dict(metadata)
        return {"id": memory_id}

    def delete(self, memory_id):
        self.delete_calls.append(memory_id)
        self.rows.pop(memory_id, None)
        return True

    def delete_all(self, **kwargs):
        self.delete_all_calls += 1
        raise AssertionError("adapter must not call delete_all(metadata=...)")


def _adapter(memory: _FakeMemory, collection: str | None = None) -> Mem0PreferenceAdapter:
    context = TenantContext(tenant_id="t1", user_id="u1", role="admin", scope_type="platform")
    settings = SimpleNamespace(
        mem0_qdrant_collection=collection or f"wp3_failure_{uuid.uuid4().hex[:8]}",
        mem0_qdrant_url="http://127.0.0.1:6333",
        mem0_qdrant_api_key="",
        mem0_embedding_model="BAAI/bge-small-zh-v1.5",
        mem0_embedding_dims=512,
        mem0_hmac_secret="0123456789abcdef0123456789abcdef",
        mem0_scope_version="v1",
        mem0_history_db_path="",
        mem0_search_timeout_seconds=0.02,
        mem0_write_timeout_seconds=0.02,
        mem0_breaker_failures=2,
        mem0_breaker_window_seconds=30.0,
        mem0_top_k_per_type=5,
        mem0_top_k_total=10,
        mem0_relevance_threshold=0.0,
        mem0_llm_model="",
        mem0_llm_api_key="",
        mem0_llm_base_url="",
        deepseek_api_key="",
        deepseek_base_url="",
        qdrant_url="http://127.0.0.1:6333",
        qdrant_collection="bizinsight_knowledge",
        rag_qdrant_active_alias="",
        metric_semantic_qdrant_collection="metric_semantic_catalog_v1",
    )
    return Mem0PreferenceAdapter(
        context, settings=settings, memory=memory, registry=_AllowRegistry(),
    )


def test_search_timeout_is_a_safe_mem0_timeout():
    memory = _FakeMemory()
    memory.sleep_seconds = 0.2
    adapter = _adapter(memory)
    with pytest.raises(Mem0Timeout) as exc:
        adapter.search("回答风格", memory_types=["presentation"])
    assert exc.value.reason == "search_timeout"


def test_breaker_opens_after_repeated_failures():
    memory = _FakeMemory()
    memory.raise_search = True
    adapter = _adapter(memory)
    for _ in range(2):
        with pytest.raises(Exception):
            adapter.search("回答风格", memory_types=["presentation"])
    with pytest.raises(Mem0CircuitOpen):
        adapter.search("回答风格", memory_types=["presentation"])
    assert memory.search_calls == 2


def test_add_timeout_is_not_retried_without_reconciliation():
    memory = _FakeMemory()
    memory.add_sleep_seconds = 0.2
    adapter = _adapter(memory)
    with pytest.raises(Mem0Timeout) as exc:
        adapter.add(
            "偏好简洁", memory_key="style", memory_type="presentation",
            operation_id="op-add-timeout", value={"style": "concise"}, infer=False,
        )
    assert exc.value.reason == "add_timeout"
    assert memory.rows == {}
    assert memory.search_calls == 0


def test_duplicate_cleanup_uses_memory_ids_and_never_delete_all():
    memory = _FakeMemory()
    adapter = _adapter(memory)
    first = adapter.add(
        "偏好简洁", memory_key="style", memory_type="presentation",
        operation_id="op-retry", value={"style": "concise"}, infer=False,
    )[0]
    second = adapter.add(
        "偏好简洁", memory_key="style", memory_type="presentation",
        operation_id="op-retry", value={"style": "concise"}, infer=False,
    )[0]
    assert first != second
    assert len(adapter.reconcile("op-retry")) == 2
    result = adapter.delete_operation("op-retry", keep_memory_id=first)
    assert result["deleted_count"] == 1
    assert memory.delete_calls == [second]
    assert memory.delete_all_calls == 0


def test_update_changes_operation_id_for_outbox_reconciliation():
    memory = _FakeMemory()
    adapter = _adapter(memory)
    memory_id = adapter.add(
        "回答简洁", memory_key="style", memory_type="presentation",
        operation_id="op-before-update", value={"style": "concise"}, infer=False,
    )[0]
    assert adapter.update(
        memory_id, operation_id="op-update", text="回答详细",
        value={"style": "detailed"},
    ) is True
    assert adapter.reconcile("op-before-update") == []
    hits = adapter.reconcile("op-update")
    assert [hit.memory_id for hit in hits] == [memory_id]
    assert hits[0].value == {"style": "detailed"}


def test_management_can_retrieve_expired_scope_default_for_reconfirmation():
    memory = _FakeMemory()
    adapter = _adapter(memory)
    memory_id = adapter.add(
        "默认日期", memory_key="default_date", memory_type="scope_default",
        operation_id="op-expired-scope", value={"date": "2026-01"}, infer=False,
    )[0]
    memory.rows[memory_id]["metadata"]["expires_at"] = "2000-01-01T00:00:00+00:00"

    assert adapter.get_validated(
        memory_id, operation_id="op-expired-scope",
        memory_type="scope_default", memory_key="default_date",
    ) is None
    hit = adapter.get_validated(
        memory_id, operation_id="op-expired-scope",
        memory_type="scope_default", memory_key="default_date", include_expired=True,
    )
    assert hit is not None
    assert hit.value == {"date": "2026-01"}


def test_adapter_reconcile_supports_external_write_recovery_contract():
    """Inject the WP3 external-write/Registry-commit failure boundary.

    This validates recoverability primitives, not the production WP4
    Writer/Outbox state machine.
    """
    memory = _FakeMemory()
    adapter = _adapter(memory)
    operation_id = "op-commit-fail"

    def write_then_commit_registry() -> None:
        adapter.add(
            "偏好简洁", memory_key="style", memory_type="presentation",
            operation_id=operation_id, value={"style": "concise"}, infer=False,
        )
        # Fault injection at the exact external-success/local-commit boundary.
        raise RuntimeError("injected_registry_commit_failure")

    with pytest.raises(RuntimeError, match="injected_registry_commit_failure"):
        write_then_commit_registry()

    recovered = adapter.reconcile(operation_id)
    assert len(recovered) == 1
    claimed_id = recovered[0].memory_id

    # Simulate a retry after the caller could not observe the previous result.
    # Reconciliation must enumerate both records and remove the duplicate by ID.
    adapter.add(
        "偏好简洁", memory_key="style", memory_type="presentation",
        operation_id=operation_id, value={"style": "concise"}, infer=False,
    )
    recovered = adapter.reconcile(operation_id)
    assert len(recovered) == 2
    cleanup = adapter.delete_operation(operation_id, keep_memory_id=claimed_id)
    assert cleanup["deleted_count"] == 1
    assert [hit.memory_id for hit in adapter.reconcile(operation_id)] == [claimed_id]
    assert memory.delete_all_calls == 0


def test_pending_reconcile_recovers_before_registry_reference_is_committed():
    """Outbox recovery must find Mem0 rows while Registry memory_id is NULL."""
    memory = _FakeMemory()
    adapter = _adapter(memory)
    memory_id = adapter.add(
        "偏好简洁", memory_key="style", memory_type="presentation",
        operation_id="op-pending", value={"style": "concise"}, infer=False,
    )[0]

    class _RegistryNotCommitted:
        def validate(self, **kwargs):
            from app.memory.preference_registry import RegistryDecision
            return RegistryDecision(False, "registry_record_missing")

    adapter.registry = _RegistryNotCommitted()
    assert adapter.reconcile("op-pending") == []
    pending = adapter.reconcile_pending("op-pending")
    assert [hit.memory_id for hit in pending] == [memory_id]
    assert pending[0].value == {"style": "concise"}


def test_pending_reconcile_rejects_wrong_operation_id_even_if_provider_ignores_filter():
    memory = _FakeMemory()
    adapter = _adapter(memory)
    adapter.add(
        "偏好简洁", memory_key="style", memory_type="presentation",
        operation_id="op-other", value={"style": "concise"}, infer=False,
    )
    memory.get_all = lambda filters, top_k=20: list(memory.rows.values())
    assert adapter.reconcile_pending("op-requested") == []


def test_operation_id_filter_unavailable_fails_closed_without_broad_fallback():
    memory = _FakeMemory()
    adapter = _adapter(memory)
    memory.rows["mem-owned"] = {
        "id": "mem-owned",
        "memory": "偏好简洁",
        "user_id": adapter.scope.entity_id,
        "metadata": {
            "operation_id": "op-no-filter",
            "memory_type": "presentation",
            "memory_key": "style",
            "entity_scope": adapter.scope.entity_id,
            "seller_scope": adapter.scope.seller_scope,
            "scope_version": adapter.scope.scope_version,
            "memory_status": "active",
            "value_json": {"style": "concise"},
        },
    }
    memory.reject_operation_id_filter = True

    with pytest.raises(Mem0Unavailable) as exc:
        adapter.reconcile("op-no-filter")

    assert exc.value.reason == "reconcile_failed"
    assert memory.get_all_calls == [{
        "user_id": adapter.scope.entity_id,
        "operation_id": "op-no-filter",
    }]
    assert memory.search_calls == 0
    assert memory.delete_calls == []


def test_empty_explicit_type_scope_queries_nothing():
    memory = _FakeMemory()
    adapter = _adapter(memory)
    assert adapter.search("回答风格", memory_types=[]) == []
    assert memory.search_calls == 0

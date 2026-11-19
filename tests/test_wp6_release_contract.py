"""WP6 release and rollback contract tests; no external services required."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.routes.chat import _memory_session_for_request
from app.context.orchestrator import ContextOrchestrator, Source
from app.memory import preference_worker
from app.memory.preference_provider import provider_for_context
from app.tenancy import TenantContext


def test_durable_chat_does_not_alias_chat_session_to_legacy_memory_session():
    payload = SimpleNamespace(
        chat_session_id="chat-session",
        memory_session_id=None,
        use_current_explorer_context=True,
        explorer_context={"seller_id": "seller-a"},
    )

    assert _memory_session_for_request(payload) is None


def test_legacy_memory_only_request_keeps_compatibility_session_id():
    payload = SimpleNamespace(
        chat_session_id=None,
        memory_session_id="legacy-session",
        use_current_explorer_context=False,
        explorer_context=None,
    )

    assert _memory_session_for_request(payload) == "legacy-session"


def test_durable_chat_rejects_conflicting_legacy_session_id():
    payload = SimpleNamespace(
        chat_session_id="chat-session",
        memory_session_id="different-session",
        use_current_explorer_context=True,
        explorer_context={"seller_id": "seller-a"},
    )

    with pytest.raises(HTTPException) as error:
        _memory_session_for_request(payload)

    assert error.value.status_code == 400
    assert error.value.detail == "memory_session_id_must_match_chat_session_id"


def test_mem0_rollback_preserves_core_conversation_and_task_bundle():
    state = {
        "trace_id": "trace-wp6",
        "task_id": "task-wp6",
        "chat_session_id": "session-wp6",
        "current_query_scope_fingerprint": "scope-wp6",
        "user_message": "继续看一下订单量",
        "context_compilation": {
            "standalone_question": "继续看一下订单量",
            "known_slots": {"metric": {"value": "order_count"}},
            "needs_clarification": False,
            "relation": "continue",
            "task_progress_context_used": True,
        },
        "previous_turn_context": {
            "source_turn_id": "turn-prev-wp6",
            "context_slots": {
                "metric": {"value": "order_count"},
                "date": {"value": "2026-07"},
            },
            "recent_committed_turns": [{
                "turn_id": "turn-prev-wp6",
                "question": "分析七月订单量",
                "answer": "订单量已完成查询",
            }],
        },
        "task_progress_context": {
            "objective": "分析七月订单量",
            "scope_fingerprint": "scope-wp6",
            "latest_trace_id": "trace-prev-wp6",
            "task_state_version": 2,
        },
        "preference_context": {
            "status": "unavailable",
            "availability": "mem0_disabled",
            "items": [],
        },
        "turn_resolution": {"relation": "continue"},
    }

    triage = ContextOrchestrator().build_triage_context(
        state,
        instruction=state["user_message"],
        system_prompt="route",
        model="test-model",
    )

    sources = {item.source for item in triage.bundle.items}
    assert triage.route_slots["metric"] == "order_count"
    assert Source.CONVERSATION in sources
    assert Source.TASK in sources
    assert Source.MEM0 not in sources
    assert "分析七月订单量" in triage.bundle.text


def test_mem0_rollback_prevents_worker_claim_without_touching_database(monkeypatch):
    monkeypatch.setattr(
        preference_worker,
        "get_settings",
        lambda: SimpleNamespace(mem0_enabled=False),
    )

    # Erasure jobs still run while personalization is disabled; only the
    # preference Outbox claim must be skipped and must not touch the DB.
    monkeypatch.setattr(preference_worker, "_worker_engine", lambda: object())
    monkeypatch.setattr(preference_worker, "run_membership_erasure_once", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(preference_worker, "run_account_erasure_once", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        preference_worker, "_claim",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("disabled Mem0 worker must not claim preference Outbox")
        ),
    )

    assert preference_worker.run_one() is False


def test_mem0_rollback_disables_provider_construction():
    context = TenantContext(
        tenant_id="tenant-wp6", user_id="user-wp6", role="admin", scope_type="platform",
    )

    provider = provider_for_context(
        context,
        settings=SimpleNamespace(mem0_enabled=False),
    )

    assert provider is None

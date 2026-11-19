from __future__ import annotations

from app.core.execution_context import turn_soft_limit_reached


def test_turn_soft_limit_uses_request_usage_snapshot() -> None:
    state = {
        "runtime_budget": {"soft_tokens": 100},
        "llm_usage": {"input_tokens": 45, "output_tokens": 54},
    }
    assert turn_soft_limit_reached(state) is False

    state["llm_usage"]["output_tokens"] = 55
    assert turn_soft_limit_reached(state) is True


def test_turn_soft_limit_is_disabled_without_a_configured_soft_limit() -> None:
    assert turn_soft_limit_reached(None) is False
    assert turn_soft_limit_reached({"runtime_budget": {}, "llm_usage": {"input_tokens": 999}}) is False

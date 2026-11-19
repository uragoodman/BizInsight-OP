from types import SimpleNamespace

from app.core.execution_context import activate_state, reset_state
from app.llm.client import _record_usage


def test_provider_reported_usage_is_bound_to_current_user_and_trace(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr("app.llm.client.record_usage", lambda **kwargs: captured.update(kwargs))
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8),
        choices=[SimpleNamespace(finish_reason="stop")],
    )
    state = {
        "tenant_id": "tenant-a", "user_id": "user-a", "trace_id": "trace-a",
        "chat_session_id": "session-a", "runtime_budget": {}, "errors": [],
    }
    token = activate_state(state)
    try:
        _record_usage(response, "model-a", "chat")
    finally:
        reset_state(token)
    assert captured == {
        "tenant_id": "tenant-a", "user_id": "user-a", "trace_id": "trace-a",
        "chat_session_id": "session-a", "operation": "chat", "model": "model-a",
        "input_tokens": 12, "output_tokens": 8, "estimated_cost": 0.0,
        "task_id": None, "evaluation_run_id": None, "route_profile": "standard",
        "provider_id": None, "fallback_from": None,
    }
    assert state["llm_usage"] == {"input_tokens": 12, "output_tokens": 8, "estimated_cost": 0.0}

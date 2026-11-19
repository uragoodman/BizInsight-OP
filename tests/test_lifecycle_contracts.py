import pytest

from app.chat_lifecycle.errors import ExecutionErrorCode
from app.chat_lifecycle.state_machine import (
    ExecutionStatus,
    TurnStatus,
    assert_execution_transition,
    assert_turn_transition,
)


def test_turn_terminal_states_are_monotonic() -> None:
    assert_turn_transition(TurnStatus.QUEUED, TurnStatus.RUNNING)
    assert_turn_transition(TurnStatus.RUNNING, TurnStatus.ABSTAINED)
    with pytest.raises(ValueError, match="Terminal state"):
        assert_turn_transition(TurnStatus.ABSTAINED, TurnStatus.RUNNING)


def test_execution_supports_hitl_suspend_and_resume() -> None:
    assert_execution_transition(ExecutionStatus.QUEUED, ExecutionStatus.RUNNING)
    assert_execution_transition(ExecutionStatus.RUNNING, ExecutionStatus.SUSPENDED)
    assert_execution_transition(ExecutionStatus.SUSPENDED, ExecutionStatus.RUNNING)
    assert_execution_transition(ExecutionStatus.RUNNING, ExecutionStatus.COMPLETED)


def test_execution_error_codes_are_stable() -> None:
    assert ExecutionErrorCode.AUTH_STALE.value == "AUTH_STALE"
    assert ExecutionErrorCode.TOOL_VALIDATION_FAILED.value == "TOOL_VALIDATION_FAILED"

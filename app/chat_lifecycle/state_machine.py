"""Centralized monotonic Turn/Execution state machines."""

from enum import StrEnum


class TurnStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    ABSTAINED = "abstained"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class ExecutionStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    ABSTAINED = "abstained"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class InteractionState(StrEnum):
    NONE = "none"
    AWAITING_SCOPE = "awaiting_scope"
    AWAITING_METRIC_CONFIRMATION = "awaiting_metric_confirmation"
    AWAITING_HUMAN_INPUT = "awaiting_human_input"


_TERMINAL_TURN = {
    TurnStatus.COMPLETED,
    TurnStatus.ABSTAINED,
    TurnStatus.FAILED,
    TurnStatus.CANCELLED,
    TurnStatus.INTERRUPTED,
}
_TERMINAL_EXECUTION = {
    ExecutionStatus.COMPLETED,
    ExecutionStatus.ABSTAINED,
    ExecutionStatus.FAILED,
    ExecutionStatus.CANCELLED,
    ExecutionStatus.INTERRUPTED,
}

_TURN_TRANSITIONS = {
    TurnStatus.QUEUED: {TurnStatus.RUNNING, TurnStatus.CANCELLED, TurnStatus.FAILED},
    TurnStatus.RUNNING: _TERMINAL_TURN,
}
_EXECUTION_TRANSITIONS = {
    ExecutionStatus.QUEUED: {ExecutionStatus.RUNNING, ExecutionStatus.CANCELLED, ExecutionStatus.FAILED},
    ExecutionStatus.RUNNING: {ExecutionStatus.SUSPENDED, *_TERMINAL_EXECUTION},
    ExecutionStatus.SUSPENDED: {
        ExecutionStatus.RUNNING,
        ExecutionStatus.CANCELLED,
        ExecutionStatus.FAILED,
        ExecutionStatus.ABSTAINED,
        ExecutionStatus.INTERRUPTED,
    },
}


def _assert_transition(current: StrEnum, target: StrEnum, transitions: dict[StrEnum, set[StrEnum]]) -> None:
    if current == target:
        return
    if current in _TERMINAL_TURN or current in _TERMINAL_EXECUTION:
        raise ValueError(f"Terminal state cannot transition: {current} -> {target}")
    if target not in transitions.get(current, set()):
        raise ValueError(f"Invalid state transition: {current} -> {target}")


def assert_turn_transition(current: TurnStatus | str, target: TurnStatus | str) -> None:
    _assert_transition(TurnStatus(current), TurnStatus(target), _TURN_TRANSITIONS)


def assert_execution_transition(current: ExecutionStatus | str, target: ExecutionStatus | str) -> None:
    _assert_transition(ExecutionStatus(current), ExecutionStatus(target), _EXECUTION_TRANSITIONS)

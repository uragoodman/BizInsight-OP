"""Opt-in transport/thinking presets for OpenAI-compatible LLM calls.

Operation assignments below are proposals only. Existing callers do not
consult this table; callers must explicitly pass a preset to ``llm_extract``
or ``llm_chat`` before these settings affect a request.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal


Transport = Literal["stream", "nonstream"]
Thinking = Literal["on", "off", "provider_default"]


@dataclass(frozen=True)
class InteractionPolicy:
    name: str
    transport: Transport
    thinking: Thinking
    connect_timeout_seconds: float
    idle_timeout_seconds: float | None
    total_timeout_seconds: float
    retry_before_first_token: int = 1
    provider_fallback_enabled: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Interaction policy name must not be empty")
        if self.transport not in {"stream", "nonstream"}:
            raise ValueError("Unsupported interaction transport")
        if self.thinking not in {"on", "off", "provider_default"}:
            raise ValueError("Unsupported thinking mode")
        timeouts = (self.connect_timeout_seconds, self.total_timeout_seconds)
        if any(not isfinite(value) or value <= 0 for value in timeouts):
            raise ValueError("Interaction policy timeouts must be positive")
        if self.transport == "stream":
            if (
                self.idle_timeout_seconds is None
                or not isfinite(self.idle_timeout_seconds)
                or self.idle_timeout_seconds <= 0
            ):
                raise ValueError("Streaming policies require a positive idle timeout")
        elif self.idle_timeout_seconds is not None:
            raise ValueError("Non-streaming policies cannot set an idle timeout")
        if (
            isinstance(self.retry_before_first_token, bool)
            or not isinstance(self.retry_before_first_token, int)
            or self.retry_before_first_token < 0
        ):
            raise ValueError("retry_before_first_token must be a non-negative integer")
        if not isinstance(self.provider_fallback_enabled, bool):
            raise ValueError("provider_fallback_enabled must be a boolean")


POLICIES: dict[str, InteractionPolicy] = {
    "stream_thinking": InteractionPolicy(
        name="stream_thinking",
        transport="stream",
        thinking="on",
        connect_timeout_seconds=10.0,
        idle_timeout_seconds=30.0,
        total_timeout_seconds=180.0,
    ),
    "stream_fast": InteractionPolicy(
        name="stream_fast",
        transport="stream",
        thinking="off",
        connect_timeout_seconds=10.0,
        idle_timeout_seconds=20.0,
        total_timeout_seconds=90.0,
    ),
    "nonstream_fast": InteractionPolicy(
        name="nonstream_fast",
        transport="nonstream",
        thinking="off",
        connect_timeout_seconds=10.0,
        idle_timeout_seconds=None,
        total_timeout_seconds=25.0,
    ),
    "nonstream_thinking": InteractionPolicy(
        name="nonstream_thinking",
        transport="nonstream",
        thinking="on",
        connect_timeout_seconds=10.0,
        idle_timeout_seconds=None,
        total_timeout_seconds=180.0,
    ),
}


# Review proposal only: these point-name assignments are deliberately not
# consulted by llm_extract, llm_chat, or existing business call sites.
OPERATION_POLICY_PROPOSAL: dict[str, str] = {
    "classify.task_type": "nonstream_fast",
    "extract.data_query_plan": "nonstream_fast",
    "extract.relation_resolution": "nonstream_fast",
    "extract.semantic_verifier": "nonstream_fast",
    "extract.semantic_interpreter": "stream_fast",
    "critic.groundedness": "nonstream_fast",
    "critic.completeness": "nonstream_fast",
    "critic.consistency": "nonstream_fast",
    "critic.combined": "nonstream_fast",
    "insight.generate": "stream_thinking",
    "insight.review": "nonstream_thinking",
    "memory.consolidate": "nonstream_thinking",
    "agent_action.diagnostic": "stream_thinking",
    "agent_action.insight": "stream_thinking",
    "agent_action.open_diagnosis": "stream_thinking",
    "chat.answer": "nonstream_fast",
    "chat.recommendation": "nonstream_fast",
    "chat.connectivity_test": "nonstream_fast",
    "report.generate": "stream_thinking",
    "eval.ragas_judge": "nonstream_thinking",
}


OPERATION_PROFILE_PROPOSAL: dict[str, str] = {
    "classify.task_type": "fast",
    "extract.data_query_plan": "fast",
    "extract.relation_resolution": "fast",
    "extract.semantic_verifier": "fast",
    "extract.semantic_interpreter": "standard",
    "critic.groundedness": "fast",
    "critic.completeness": "fast",
    "critic.consistency": "fast",
    "critic.combined": "fast",
    "insight.generate": "quality",
    "insight.review": "quality",
    "memory.consolidate": "quality",
    "agent_action.diagnostic": "standard",
    "agent_action.insight": "standard",
    "agent_action.open_diagnosis": "standard",
    "chat.answer": "standard",
    "chat.recommendation": "standard",
    "chat.connectivity_test": "standard",
    "report.generate": "quality",
    "eval.ragas_judge": "quality",
}


def policy_for(operation: str) -> InteractionPolicy | None:
    """Return the proposed policy for an operation, if one is listed.

    This is a lookup helper, not an automatic default. No existing client path
    calls it; the policy takes effect only when a caller explicitly passes the
    returned object to an LLM call.
    """

    preset_name = OPERATION_POLICY_PROPOSAL.get(operation)
    return POLICIES.get(preset_name) if preset_name is not None else None


def resolve_policy(value: InteractionPolicy | str | None) -> InteractionPolicy | None:
    if value is None:
        return None
    if isinstance(value, InteractionPolicy):
        return value
    if isinstance(value, str):
        try:
            return POLICIES[value]
        except KeyError as exc:
            raise ValueError(f"Unknown LLM interaction policy: {value}") from exc
    raise TypeError("interaction_policy must be a preset name, InteractionPolicy, or None")

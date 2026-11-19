"""Internal per-graph policy for optional Critic LLM request thinking mode."""

from __future__ import annotations

from enum import Enum


class CriticLLMThinkingMode(str, Enum):
    PROVIDER_DEFAULT = "provider_default"
    NON_THINKING_EXPERIMENT = "non_thinking_experiment"

    @classmethod
    def parse(cls, value: "CriticLLMThinkingMode | str") -> "CriticLLMThinkingMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except (TypeError, ValueError) as exc:
            choices = ", ".join(mode.value for mode in cls)
            raise ValueError(f"Invalid internal Critic thinking mode {value!r}; expected one of: {choices}") from exc

    @property
    def enable_thinking(self) -> bool | None:
        return False if self is self.NON_THINKING_EXPERIMENT else None

    def trace_summary(self) -> dict[str, object]:
        return {
            "mode": self.value,
            "scope": "optional_llm_critic_reviews",
            "enable_thinking": self.enable_thinking,
            "sdk_parameter": "extra_body.enable_thinking" if self.enable_thinking is not None else "omitted_provider_default",
        }

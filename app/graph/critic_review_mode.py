"""Internal evaluation modes for optional Critic LLM reviews."""

from enum import Enum


class CriticLLMReviewMode(str, Enum):
    FULL = "full"
    CRITIC_LLM_REVIEW_OFF = "critic_llm_review_off"
    ADAPTIVE = "adaptive"
    COMBINED_REVIEW_EXPERIMENT = "combined_review_experiment"

    @classmethod
    def parse(cls, value: "CriticLLMReviewMode | str") -> "CriticLLMReviewMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except (TypeError, ValueError) as exc:
            choices = ", ".join(mode.value for mode in cls)
            raise ValueError(f"Invalid internal Critic review mode {value!r}; expected one of: {choices}") from exc

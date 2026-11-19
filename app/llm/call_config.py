"""Internal, immutable per-request LLM call policies.

These policies are passed by trusted server code into the graph state. They are
not part of any public request schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class LLMCallConfig:
    mode: Literal["normal", "provider_ab_experiment"]
    provider_id: str | None
    sdk_max_retries: int
    application_attempts: int
    fallback_enabled: bool
    timeout_seconds: float
    max_completion_tokens: int | None = None
    connect_retry_attempts: int = 0
    connect_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"normal", "provider_ab_experiment"}:
            raise ValueError("Unsupported internal LLM call configuration mode.")
        if (self.sdk_max_retries < 0 or self.application_attempts < 1 or self.timeout_seconds <= 0
                or self.connect_retry_attempts < 0
                or (self.connect_timeout_seconds is not None and self.connect_timeout_seconds <= 0)):
            raise ValueError("Invalid retry or timeout values in internal LLM call configuration.")
        if self.mode == "provider_ab_experiment":
            if (not self.provider_id or self.sdk_max_retries != 0 or self.application_attempts != 1
                    or self.fallback_enabled or self.connect_retry_attempts > 1):
                raise ValueError("Provider A/B mode requires one fixed provider, SDK retries=0, application attempts=1, at most one explicit connection retry, and fallback disabled.")
            if self.max_completion_tokens is None or self.max_completion_tokens < 1:
                raise ValueError("Provider A/B mode requires a positive max_completion_tokens limit.")
            if self.connect_timeout_seconds is not None and self.connect_timeout_seconds > self.timeout_seconds:
                raise ValueError("Connection timeout cannot exceed the total model timeout.")
        elif not self.fallback_enabled:
            raise ValueError("Fallback can only be disabled in Provider A/B mode.")
        elif self.max_completion_tokens is not None:
            raise ValueError("The normal call policy cannot override the configured max_tokens behavior.")
        elif self.connect_retry_attempts or self.connect_timeout_seconds is not None:
            raise ValueError("Connection-specific retry settings are reserved for explicit provider experiments.")

    @classmethod
    def normal(cls, settings: Any) -> "LLMCallConfig":
        # OpenAI SDK's current default is two retries. Pin that value explicitly
        # so the trace records the effective setting without changing behavior.
        return cls(
            mode="normal",
            provider_id=None,
            sdk_max_retries=2,
            application_attempts=int(settings.llm_resilience_attempts),
            fallback_enabled=True,
            timeout_seconds=float(settings.llm_timeout_seconds),
        )

    @classmethod
    def provider_ab_experiment(
        cls, settings: Any, *, provider_id: str = "primary", max_completion_tokens: int = 4096,
        timeout_seconds: float | None = None, connect_retry_attempts: int = 0,
        connect_timeout_seconds: float | None = None,
    ) -> "LLMCallConfig":
        return cls(
            mode="provider_ab_experiment",
            provider_id=provider_id,
            sdk_max_retries=0,
            application_attempts=1,
            fallback_enabled=False,
            # Formal experiments pass the frozen timeout explicitly. Other
            # internal callers retain the configured default when omitted.
            timeout_seconds=float(
                settings.llm_timeout_seconds if timeout_seconds is None else timeout_seconds
            ),
            max_completion_tokens=max_completion_tokens,
            connect_retry_attempts=connect_retry_attempts,
            connect_timeout_seconds=connect_timeout_seconds,
        )

    def to_state(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "provider_id": self.provider_id,
            "sdk_max_retries": self.sdk_max_retries,
            "application_attempts": self.application_attempts,
            "fallback_enabled": self.fallback_enabled,
            "timeout_seconds": self.timeout_seconds,
            "max_completion_tokens": self.max_completion_tokens,
            "connect_retry_attempts": self.connect_retry_attempts,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "model_ids": [],
        }

    @classmethod
    def from_state(cls, value: Any) -> "LLMCallConfig":
        if not isinstance(value, dict):
            raise ValueError("Invalid internal LLM call configuration in request state.")
        return cls(
            mode=value.get("mode"),
            provider_id=value.get("provider_id"),
            sdk_max_retries=int(value.get("sdk_max_retries")),
            application_attempts=int(value.get("application_attempts")),
            fallback_enabled=bool(value.get("fallback_enabled")),
            timeout_seconds=float(value.get("timeout_seconds")),
            max_completion_tokens=(
                int(value["max_completion_tokens"]) if value.get("max_completion_tokens") is not None else None
            ),
            connect_retry_attempts=int(value.get("connect_retry_attempts", 0)),
            connect_timeout_seconds=(
                float(value["connect_timeout_seconds"])
                if value.get("connect_timeout_seconds") is not None else None
            ),
        )

    def trace_summary(self, model_ids: list[str] | None = None) -> dict[str, Any]:
        models = sorted(set(model_ids or []))
        return {
            "mode": self.mode,
            "provider_id": self.provider_id or "configured_provider_chain",
            "model": models,
            "sdk_max_retries": self.sdk_max_retries,
            "application_attempts": self.application_attempts,
            "fallback_enabled": self.fallback_enabled,
            "timeout_seconds": self.timeout_seconds,
            "timeout_owner": "sdk_transport" if self.mode == "provider_ab_experiment" else "sdk_transport_and_resilience",
            "max_completion_tokens": self.max_completion_tokens,
            "connect_retry_attempts": self.connect_retry_attempts,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "retry_scope": "connect_establishment_only" if self.connect_retry_attempts else "none",
            "effective_max_sdk_invocations_per_logical_call": self.application_attempts * (1 + self.connect_retry_attempts),
        }

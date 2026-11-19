"""Fail-closed flag boundary for RAG progressive delivery.

The release controller remains authoritative for release identity and approval.
This module only supplies the operational kill switches.  If Unleash (or the
OpenFeature fallback) is unavailable every candidate-facing flag evaluates to
``False``.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.core.config import get_settings


class _UnleashAdapter:
    """Small provider adapter; the rest of the app only sees this API."""

    def __init__(self, url: str, token: str, app_name: str, refresh_interval: int) -> None:
        from UnleashClient import UnleashClient  # type: ignore[import-not-found]
        self._client = UnleashClient(
            url=url, app_name=app_name, custom_headers={"Authorization": token},
            refresh_interval=refresh_interval, disable_metrics=True,
        )
        self._client.initialize_client()

    def get_boolean_value(self, key: str, _default: bool, context: dict[str, str]) -> bool:
        # The fallback is deliberately false: a stale/unreachable Unleash
        # server must not expose a candidate release.
        return bool(self._client.is_enabled(key, context=context, fallback_function=lambda: False))

    def destroy(self) -> None:
        self._client.destroy()


@lru_cache(maxsize=1)
def _client() -> Any | None:
    settings = get_settings()
    if settings.rag_unleash_url and settings.rag_unleash_api_token:
        try:
            return _UnleashAdapter(
                settings.rag_unleash_url, settings.rag_unleash_api_token,
                settings.rag_unleash_app_name, settings.rag_unleash_refresh_interval_seconds,
            )
        except Exception:
            return None
    try:
        from openfeature import api  # type: ignore[import-not-found]
        return api.get_client("bizinsight-rag")
    except Exception:
        return None


def enabled(key: str, default: bool, context: dict[str, str]) -> bool:
    """Return an operational flag, never allowing an unsafe fallback.

    The local adapter is intentionally restrictive: configuration may enable
    the controller itself, but an unavailable remote provider cannot turn on
    Shadow or Canary.  An absent provider never exposes a candidate to a real
    request.
    """
    client = _client()
    if client is not None:
        try:
            return bool(client.get_boolean_value(key, default, context))
        except Exception:
            return False
    # Development configuration may document an intended flag state, but an
    # absent provider must never expose a candidate to a real request.
    return False


def bootstrap() -> bool:
    """Initialize the configured remote provider without making startup fatal."""
    return _client() is not None


def shutdown() -> None:
    client = _client()
    if client is not None and hasattr(client, "destroy"):
        try:
            client.destroy()
        finally:
            _client.cache_clear()


def rollout_flags(context: dict[str, str]) -> dict[str, bool]:
    settings = get_settings()
    # Defaults deliberately disable candidate execution.  The controller's
    # database policy still decides eligibility after these kill switches.
    return {
        "shadow": enabled("rag.shadow.enabled", bool(settings.rag_shadow_enabled), context),
        "canary": enabled("rag.canary.enabled", bool(settings.rag_canary_enabled), context),
        "ragas": enabled("rag.ragas.enabled", bool(settings.rag_ragas_enabled), context),
        "phoenix": enabled("rag.phoenix.enabled", bool(settings.rag_phoenix_enabled), context),
    }

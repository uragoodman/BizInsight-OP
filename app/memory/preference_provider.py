"""WP3 preference provider wired into the WP2 source-need resolver.

The provider is intentionally thin: it translates a WP2 preference need into
one or more isolated Mem0 searches and returns only validated candidates.  All
scope, SDK, timeout, circuit-breaker and Registry checks live in the adapter.
"""

from __future__ import annotations

from typing import Any

from app.core.config import get_settings
from app.memory.mem0_adapter import (
    Mem0Error,
    Mem0PreferenceAdapter,
    Mem0Unavailable,
)
from app.memory.preference_registry import PostgresPreferenceRegistry
from app.memory.preference_management import get_personalization_enabled
from app.tenancy import TenantContext, current_tenant_context


class UnavailablePreferenceProvider:
    """Explicit degradation object when Mem0 is disabled or misconfigured."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def search(self, *, need: dict[str, Any], state: dict[str, Any]) -> list[Any]:
        raise Mem0Unavailable(self.reason)


def _query_for_type(query: str, memory_type: str) -> str:
    """Use the user's own wording; type-specific hints biased cross-type recall."""
    return query


def _query_for_key(query: str, memory_key: str) -> str:
    """Add a deterministic key label only after Resolver explicitly allows that key."""
    hint = {
        "response_style": "回答风格",
        "report_style": "报告排版格式",
        "response_language": "输出语言",
        "analysis_method": "分析习惯",
        "analysis_steps": "分析步骤",
        "default_period": "默认统计周期",
        "default_date": "默认统计日期",
    }.get(memory_key)
    return f"{query} {hint}" if hint else query


class Mem0PreferenceProvider:
    def __init__(
        self,
        context: TenantContext,
        *,
        settings: Any | None = None,
        adapter: Mem0PreferenceAdapter | None = None,
        registry: Any | None = None,
    ) -> None:
        self.context = context
        self.settings = settings or get_settings()
        self._adapter = adapter
        self._registry = registry if registry is not None else PostgresPreferenceRegistry(context)

    def search(self, *, need: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
        if not get_personalization_enabled(self.context):
            return []
        query = str(state.get("user_message") or "").strip()
        if not query:
            return []
        ensure_available = getattr(self._registry, "ensure_available", None)
        if callable(ensure_available):
            ensure_available()
        types = list(need.get("memory_types") or [])
        max_per_type = min(
            5, int(need.get("top_k_per_type") or getattr(self.settings, "mem0_top_k_per_type", 5))
        )
        max_total = min(
            10, int(need.get("top_k_total") or getattr(self.settings, "mem0_top_k_total", 10))
        )
        overlay_candidates: list[dict[str, Any]] = []
        list_overlays = getattr(self._registry, "list_pending_overlays", None)
        if callable(list_overlays):
            overlay_type_counts: dict[str, int] = {}
            for overlay in list_overlays(
                context=self.context,
                chat_session_id=str(state.get("chat_session_id") or ""),
            ):
                if types and str(overlay.get("memory_type") or "") not in types:
                    continue
                overlay_type = str(overlay.get("memory_type") or "")
                if len(overlay_candidates) >= max_total:
                    break
                if overlay_type_counts.get(overlay_type, 0) >= max_per_type:
                    continue
                value = overlay.get("value")
                if not isinstance(value, dict) or not value:
                    continue
                overlay_candidates.append({
                    "memory_key": str(overlay["memory_key"]),
                    "memory_type": str(overlay["memory_type"]),
                    "value": value,
                    "source_ref": f"overlay:{overlay['overlay_id']}",
                    "confidence": 1.0,
                    "source_turn_id": overlay.get("source_turn_id"),
                    "operation_id": overlay.get("operation_id"),
                    "source": "pending_overlay",
                })
                overlay_type_counts[overlay_type] = overlay_type_counts.get(overlay_type, 0) + 1
        adapter = self._adapter or Mem0PreferenceAdapter(
            self.context, settings=self.settings, registry=self._registry,
            trace_id=str(state.get("trace_id") or ""),
        )
        threshold = float(
            need.get("relevance_threshold")
            or getattr(self.settings, "mem0_relevance_threshold", 0.55)
        )
        per_type = int(
            need.get("top_k_per_type") or getattr(self.settings, "mem0_top_k_per_type", 5)
        )
        total = int(
            need.get("top_k_total") or getattr(self.settings, "mem0_top_k_total", 10)
        )
        hits_by_id: dict[str, Any] = {}
        first_error: Mem0Error | None = None
        search_types = types or [None]
        allowed_keys = need.get("allowed_keys") or {}
        for memory_type in search_types:
            keys = list(allowed_keys.get(memory_type) or []) if memory_type else []
            for memory_key in keys or [None]:
                try:
                    hits = adapter.search(
                        _query_for_key(query, str(memory_key)) if memory_key else query,
                        memory_types=[memory_type] if memory_type else [],
                        memory_keys=[str(memory_key)] if memory_key else None,
                        top_k_per_type=per_type,
                        top_k_total=min(total, per_type), threshold=threshold,
                    )
                except Mem0Error as exc:
                    first_error = first_error or exc
                    continue
                for hit in hits:
                    hits_by_id[hit.memory_id] = hit
        ordered = sorted(
            hits_by_id.values(), key=lambda hit: (hit.score or 0.0), reverse=True,
        )
        selected: list[Any] = []
        type_counts: dict[str, int] = {}
        for hit in ordered:
            if len(selected) >= total:
                break
            hit_type = str(getattr(hit, "memory_type", "") or "")
            if type_counts.get(hit_type, 0) >= per_type:
                continue
            selected.append(hit)
            type_counts[hit_type] = type_counts.get(hit_type, 0) + 1
        mem0_candidates = [hit.as_candidate() for hit in selected]
        if first_error is not None and not selected and not overlay_candidates:
            raise first_error
        # Same-session pending writes override stale Mem0 copies for that key.
        overlay_keys = {item["memory_key"] for item in overlay_candidates}
        return [item for item in mem0_candidates if item.get("memory_key") not in overlay_keys] + overlay_candidates


def provider_for_context(
    context: TenantContext | None = None,
    *,
    settings: Any | None = None,
) -> Any | None:
    """Return the production preference provider for the trusted request.

    Disabled/misconfigured Mem0 returns an explicit unavailable provider so
    WP2 records a degradation reason rather than silently pretending that no
    preference exists.
    """
    settings = settings or get_settings()
    if not bool(getattr(settings, "mem0_enabled", False)):
        return None
    context = context or current_tenant_context()
    if context is None:
        return UnavailablePreferenceProvider("trusted_tenant_context_missing")
    try:
        return Mem0PreferenceProvider(context, settings=settings)
    except Mem0Error as exc:
        return UnavailablePreferenceProvider(getattr(exc, "reason", "mem0_unavailable"))
    except Exception:  # noqa: BLE001 - personalisation must degrade, never block analysis
        return UnavailablePreferenceProvider("mem0_provider_init_failed")

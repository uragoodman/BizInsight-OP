"""Bounded, tenant-isolated source-data snapshots for analysis-memory replay."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from app.core.config import get_settings


class SourceSnapshotError(ValueError):
    pass


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _metric_definitions(metric_tags: list[str]) -> dict[str, object]:
    """Freeze the relevant registry declarations alongside the observed result."""
    try:
        from app.metrics.registry import get_registry

        registry = get_registry()
        result: dict[str, object] = {}
        for metric in sorted(set(metric_tags)):
            definition = registry.get(metric)
            result[metric] = definition.model_dump(mode="json") if hasattr(definition, "model_dump") else str(definition)
        return result
    except Exception:
        # Snapshotting the governed tool output remains useful if registry warmup
        # is unavailable; the empty declaration is still part of the hash.
        return {}


def build_source_snapshot(state: dict, *, scope_fingerprint: str, metric_tags: list[str]) -> dict:
    """Return a deterministic, bounded snapshot of authorised tool output.

    It stores no prompt, chain-of-thought, or untrusted browser payload.  The
    result plus evidence bundle payloads are enough to replay what the memory
    writer observed, while upstream CDC remains a future stronger source.
    """
    tool_result = state.get("tool_result") or {}
    payload = {
        "schema_version": "memory-source-snapshot-v1",
        "scope_fingerprint": scope_fingerprint,
        "task_type": str(state.get("task_type") or "metric_query"),
        "tool_result": tool_result,
        "evidence_bundles": [
            {
                "evidence_id": item.get("evidence_id"), "source_type": item.get("source_type"),
                "source_ref": item.get("source_ref"), "content": item.get("content"),
            }
            for item in state.get("evidence_bundles", []) if isinstance(item, dict)
        ],
        "metric_definitions": _metric_definitions(metric_tags),
    }
    encoded = _json(payload).encode("utf-8")
    if not tool_result:
        raise SourceSnapshotError("controlled tool output is required for a replayable memory")
    if len(encoded) > get_settings().memory_source_snapshot_max_bytes:
        raise SourceSnapshotError("source snapshot exceeds configured size bound")
    source_data_version = hashlib.sha256(encoded).hexdigest()
    metric_definition_version = hashlib.sha256(_json(payload["metric_definitions"]).encode("utf-8")).hexdigest()
    return {
        "snapshot_id": f"msnap_{source_data_version[:32]}", "source_data_version": source_data_version,
        "scope_fingerprint": scope_fingerprint, "source_trace_id": str(state["trace_id"]),
        "metric_definition_version": metric_definition_version, "snapshot_payload": payload,
        "payload_bytes": len(encoded),
    }

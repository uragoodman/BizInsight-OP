"""Durable queue contracts for optional memory-consolidation proposals."""
from __future__ import annotations

import hashlib
import json
from uuid import uuid4

from app.core.config import get_settings
from app.memory.consolidation import ConsolidationUnavailable
from app.memory.repository import MemoryRepository

_SOURCE_FIELDS = (
    "memory_id", "seller_id", "start_date", "end_date", "scope_fingerprint", "metric_tags", "task_type",
    "insight_text", "limitations", "evidence_ids", "source_trace_id", "review_reason",
    "source_data_version", "metric_definition_version",
)


def enqueue_consolidation_job(repository: MemoryRepository, scope_fingerprint: str, actor: str) -> dict:
    settings = get_settings()
    if not settings.memory_llm_consolidation_enabled:
        raise ConsolidationUnavailable("LLM memory consolidation is disabled")
    sources = repository.list_active_memories_for_scope(scope_fingerprint, settings.memory_llm_consolidation_max_claims)
    if len(sources) < 2:
        raise ConsolidationUnavailable("At least two active memories in the same scope are required")
    frozen_sources = [{key: row.get(key) for key in _SOURCE_FIELDS} for row in sources]
    payload = {
        "schema_version": "memory-consolidation-job-v1", "scope_fingerprint": scope_fingerprint,
        "source_memory_ids": [str(row["memory_id"]) for row in frozen_sources], "sources": frozen_sources,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return repository.create_consolidation_job(
        f"mcon_{uuid4().hex}", actor, scope_fingerprint, payload, hashlib.sha256(raw.encode()).hexdigest(),
    )

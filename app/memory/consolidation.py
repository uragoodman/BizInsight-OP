"""Operator-triggered, opt-in LLM memory consolidation proposals.

This module deliberately does not own storage, permissions, or lifecycle changes.
It reads only active memories inside an already RLS-scoped repository and can only
create a `pending_review` record for a human to decide on.
"""
from __future__ import annotations

import hashlib
import json
from uuid import uuid4

from app.core.config import get_settings
from app.llm.client import llm_extract
from app.memory.contracts import AnalysisMemory, MemoryConsolidationProposal
from app.memory.embeddings import EMBEDDING_MODEL, EMBEDDING_VERSION, embed_texts
from app.memory.repository import MemoryRepository
from app.memory.service import normalized_claim_hash


class ConsolidationUnavailable(ValueError):
    pass


def propose_memory_consolidation(
    repository: MemoryRepository, scope_fingerprint: str, actor: str, sources: list[dict] | None = None,
    consolidation_job_id: str | None = None, consolidation_lease_token: str | None = None,
) -> dict:
    settings = get_settings()
    if not settings.memory_llm_consolidation_enabled:
        raise ConsolidationUnavailable("LLM memory consolidation is disabled")

    if sources is None:
        sources = repository.list_active_memories_for_scope(
            scope_fingerprint, limit=settings.memory_llm_consolidation_max_claims,
        )
    if len(sources) < 2:
        repository.record_memory_decision(
            action="proposal_skipped", actor=actor, reason="at least two active memories are required",
            source_memory_ids=[str(row["memory_id"]) for row in sources],
        )
        raise ConsolidationUnavailable("At least two active memories in the same scope are required")

    # A scope fingerprint is deliberately conservative, but task type is also a
    # semantic boundary.  Do not ask the model to combine unlike tasks.
    task_type = str(sources[0]["task_type"])
    sources = [row for row in sources if str(row["task_type"]) == task_type]
    if len(sources) < 2:
        raise ConsolidationUnavailable("At least two active memories with the same task type are required")
    source_ids = [str(row["memory_id"]) for row in sources]
    version_pairs = {(row.get("source_data_version"), row.get("metric_definition_version")) for row in sources}
    if len(version_pairs) != 1:
        repository.record_memory_decision(
            action="proposal_skipped", actor=actor,
            reason="source-data or metric-definition versions differ", source_memory_ids=source_ids,
        )
        raise ConsolidationUnavailable("Source-data and metric-definition versions must match")
    allowed_evidence = {str(evidence) for row in sources for evidence in row["evidence_ids"]}
    prompt_sources = [{
        "memory_id": row["memory_id"], "insight_text": row["insight_text"],
        "limitations": row["limitations"], "evidence_ids": row["evidence_ids"],
        "review_reason": row.get("review_reason") or "",
        "source_data_version": row.get("source_data_version"),
        "metric_definition_version": row.get("metric_definition_version"),
    } for row in sources]
    response = llm_extract(
        system_prompt=(
            "You consolidate approved business-memory claims. Return only a concise, evidence-bound "
            "proposal. Do not invent facts, numbers, dates, evidence IDs, or causal certainty. "
            "Use only evidence_ids supplied in the input."
        ),
        user_message=json.dumps({"approved_memories": prompt_sources}, ensure_ascii=False),
        model=settings.memory_llm_consolidation_model or None,
        temperature=0.0,
        quality=True,
    )
    try:
        proposal = MemoryConsolidationProposal.model_validate(response or {})
        if not set(proposal.evidence_ids).issubset(allowed_evidence):
            raise ValueError("proposal contains evidence IDs outside its approved sources")
    except Exception as exc:
        repository.record_memory_decision(
            action="proposal_failed", actor=actor, reason=str(exc)[:500], source_memory_ids=source_ids,
        )
        raise ConsolidationUnavailable("LLM did not produce a valid evidence-bound proposal") from exc

    source_key = ":".join(sorted(source_ids))
    proposal_hash = hashlib.sha256(source_key.encode()).hexdigest()
    template = sources[0]
    metric_tags = sorted({str(tag) for row in sources for tag in row["metric_tags"]})[:8]
    memory = AnalysisMemory(
        memory_id=f"mem_prop_{uuid4().hex}", seller_id=str(template["seller_id"]),
        start_date=template["start_date"], end_date=template["end_date"],
        scope_fingerprint=scope_fingerprint, metric_tags=metric_tags, task_type=task_type,
        insight_text=proposal.insight_text, limitations=proposal.limitations,
        evidence_ids=proposal.evidence_ids, source_trace_id=f"memory_consolidation:{proposal_hash}",
        status="pending_review", source_data_version=template.get("source_data_version"),
        metric_definition_version=template.get("metric_definition_version"),
        normalized_claim_hash=normalized_claim_hash(proposal.insight_text, task_type, scope_fingerprint, metric_tags),
    )
    vector = embed_texts([memory.insight_text])
    inserted = repository.insert_analysis_memory(
        memory, embedding=vector[0] if vector else None,
        embedding_model=EMBEDDING_MODEL if vector else None,
        embedding_version=EMBEDDING_VERSION if vector else None,
        consolidation_job_id=consolidation_job_id, consolidation_lease_token=consolidation_lease_token,
    )
    repository.record_memory_decision(
        action="proposal_created" if inserted else "proposal_skipped", actor=actor, memory_id=memory.memory_id,
        reason="LLM consolidation proposal; human review required", source_memory_ids=source_ids,
        rule_scores={"source_count": len(sources), "proposal_hash": proposal_hash},
    )
    if not inserted:
        raise ConsolidationUnavailable("An equivalent active or pending proposal already exists")
    return repository.get_analysis_memory(memory.memory_id) or {}

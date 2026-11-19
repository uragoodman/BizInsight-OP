"""WP1 Context Orchestrator.

Builds the per-agent ContextBundle from already-authorised state.  It does not
read business data, grant permissions, call Mem0, or write memory.  The bundle
is a governed context container, never an authorization credential.

Design notes:
- Triage receives only conversation/task/preference candidates.
- Specialists receive only the sources listed in AGENT_POLICIES.
- Tool evidence is added only after ``refresh_evidence`` (called by the graph
  wrapper around data_query/policy_execute).
- Trace metadata contains item ids, sources, trust tiers, source refs, token
  estimates and omission reasons; it never contains item content.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable
from uuid import uuid4

from app.context.packer import (
    ContextItem as PackerContextItem,
    ContextSource as PackerSource,
    TrustLevel,
    estimate_tokens,
    pack_for_llm,
    rendered_item_tokens,
)
from app.observability import record_event, record_measurement

POLICY_VERSION = "context-orchestrator-v1"
_MAX_EVENTS = 24
_MAX_ITEMS_PER_BUNDLE = 12
_MAX_TEXT = 1600
_MAX_REFS = 8

# EvidenceBundle.source_type values that represent current-turn tool execution.
# RAG/report/root_cause/explorer bundles are reference material, not current
# tool evidence and must not be labelled current_evidence.
CURRENT_EVIDENCE_SOURCE_TYPES = frozenset({
    "mcp", "investigation_action", "deterministic_attribution", "diagnostic_query",
})
_FAILED_EVIDENCE_STATUSES = frozenset({"failed", "execution_failed", "error"})


def _evidence_status(item: dict[str, Any]) -> str:
    content = item.get("content")
    if isinstance(content, dict):
        result = content.get("result")
        if isinstance(result, dict):
            return str(result.get("status") or "")
        return str(content.get("status") or "")
    return ""


def _is_current_evidence(item: Any) -> bool:
    if not isinstance(item, dict) or not item.get("evidence_id"):
        return False
    if str(item.get("source_type") or "") not in CURRENT_EVIDENCE_SOURCE_TYPES:
        return False
    if item.get("trusted") is not True:
        return False
    return _evidence_status(item) not in _FAILED_EVIDENCE_STATUSES


class Source(str, Enum):
    CONVERSATION = "conversation"
    TASK = "task"
    MEM0 = "mem0"
    ANALYSIS_MEMORY = "analysis_memory"
    RAG = "rag"
    METRIC_REGISTRY = "metric_registry"
    TOOL_EVIDENCE = "tool_evidence"


class TrustTier(str, Enum):
    PREFERENCE = "preference"
    HISTORICAL_REFERENCE = "historical_reference"
    CONTROLLED_DEFINITION = "controlled_definition"
    RETRIEVED_DOCUMENT = "retrieved_document"
    CURRENT_EVIDENCE = "current_evidence"


SOURCE_TIER: dict[Source, TrustTier] = {
    Source.CONVERSATION: TrustTier.HISTORICAL_REFERENCE,
    Source.TASK: TrustTier.HISTORICAL_REFERENCE,
    Source.MEM0: TrustTier.PREFERENCE,
    Source.ANALYSIS_MEMORY: TrustTier.HISTORICAL_REFERENCE,
    Source.RAG: TrustTier.RETRIEVED_DOCUMENT,
    Source.METRIC_REGISTRY: TrustTier.CONTROLLED_DEFINITION,
    Source.TOOL_EVIDENCE: TrustTier.CURRENT_EVIDENCE,
}

PACKER_SOURCE: dict[Source, PackerSource] = {
    Source.CONVERSATION: PackerSource.CONVERSATION,
    Source.TASK: PackerSource.TASK_STATE,
    Source.MEM0: PackerSource.PREFERENCE,
    Source.ANALYSIS_MEMORY: PackerSource.ANALYSIS_MEMORY,
    Source.RAG: PackerSource.RAG_EVIDENCE,
    Source.METRIC_REGISTRY: PackerSource.METRIC_REGISTRY,
    Source.TOOL_EVIDENCE: PackerSource.TOOL_EVIDENCE,
}

PACKER_TRUST: dict[TrustTier, TrustLevel] = {
    TrustTier.PREFERENCE: TrustLevel.UNTRUSTED_REFERENCE,
    TrustTier.HISTORICAL_REFERENCE: TrustLevel.UNTRUSTED_REFERENCE,
    TrustTier.CONTROLLED_DEFINITION: TrustLevel.TRUSTED_FACT,
    TrustTier.RETRIEVED_DOCUMENT: TrustLevel.UNTRUSTED_REFERENCE,
    TrustTier.CURRENT_EVIDENCE: TrustLevel.TRUSTED_FACT,
}


@dataclass(frozen=True)
class AgentPolicy:
    sources: frozenset[Source]
    tiers: frozenset[TrustTier]
    model_role: str
    include_answer_under_review: bool = False
    # Soft item-count ceiling: required items are retained, and the token
    # budget then decides whether the required set can be sent safely.
    max_items: int = _MAX_ITEMS_PER_BUNDLE


AGENT_POLICIES: dict[str, AgentPolicy] = {
    "triage": AgentPolicy(
        sources=frozenset({Source.CONVERSATION, Source.TASK, Source.MEM0}),
        tiers=frozenset({TrustTier.PREFERENCE, TrustTier.HISTORICAL_REFERENCE}),
        model_role="classify",
    ),
    "metric_definition": AgentPolicy(
        sources=frozenset({Source.RAG, Source.METRIC_REGISTRY}),
        tiers=frozenset({TrustTier.RETRIEVED_DOCUMENT, TrustTier.CONTROLLED_DEFINITION}),
        model_role="extract",
    ),
    "data_query": AgentPolicy(
        sources=frozenset({Source.CONVERSATION, Source.TASK, Source.METRIC_REGISTRY}),
        tiers=frozenset({TrustTier.HISTORICAL_REFERENCE, TrustTier.CONTROLLED_DEFINITION}),
        model_role="extract",
    ),
    "diagnostic": AgentPolicy(
        sources=frozenset({Source.TASK, Source.METRIC_REGISTRY, Source.TOOL_EVIDENCE}),
        tiers=frozenset({
            TrustTier.HISTORICAL_REFERENCE, TrustTier.CONTROLLED_DEFINITION,
            TrustTier.CURRENT_EVIDENCE,
        }),
        model_role="extract",
    ),
    "report": AgentPolicy(
        sources=frozenset({
            Source.CONVERSATION, Source.TASK, Source.ANALYSIS_MEMORY,
            Source.RAG, Source.METRIC_REGISTRY, Source.TOOL_EVIDENCE,
        }),
        tiers=frozenset({
            TrustTier.HISTORICAL_REFERENCE, TrustTier.RETRIEVED_DOCUMENT,
            TrustTier.CONTROLLED_DEFINITION, TrustTier.CURRENT_EVIDENCE,
        }),
        model_role="chat",
    ),
    "answer": AgentPolicy(
        sources=frozenset({
            Source.CONVERSATION, Source.TASK, Source.ANALYSIS_MEMORY,
            Source.METRIC_REGISTRY, Source.TOOL_EVIDENCE, Source.MEM0,
        }),
        tiers=frozenset({
            TrustTier.PREFERENCE, TrustTier.HISTORICAL_REFERENCE,
            TrustTier.CONTROLLED_DEFINITION, TrustTier.CURRENT_EVIDENCE,
        }),
        model_role="chat",
    ),
    "critic": AgentPolicy(
        sources=frozenset({Source.TOOL_EVIDENCE, Source.METRIC_REGISTRY}),
        tiers=frozenset({TrustTier.CURRENT_EVIDENCE, TrustTier.CONTROLLED_DEFINITION}),
        model_role="extract",
        include_answer_under_review=True,
    ),
}


@dataclass(frozen=True)
class OrchestratedItem:
    item_id: str
    source: Source
    trust_tier: TrustTier
    content: str
    source_refs: tuple[str, ...]
    scope_fingerprint: str
    version: str
    required: bool
    priority: int
    created_at: str = ""
    token_estimate: int = 0
    evidence_ids: tuple[str, ...] = ()
    sensitivity: str = "internal"

    def metadata(self, *, included: bool, omission_reason: str | None) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "source": self.source.value,
            "trust_tier": self.trust_tier.value,
            "source_refs": list(self.source_refs),
            "scope_fingerprint": self.scope_fingerprint,
            "version": self.version,
            "required": self.required,
            "priority": self.priority,
            "created_at": self.created_at,
            "token_estimate": self.token_estimate,
            "sensitivity": self.sensitivity,
            "evidence_ids": list(self.evidence_ids),
            "included": included,
            "omission_reason": omission_reason,
        }


@dataclass
class ContextBundle:
    bundle_id: str
    turn_id: str
    task_id: str | None
    agent: str
    resolved_request: dict[str, Any]
    scope_reference: str
    items: list[OrchestratedItem]
    text: str
    budget: dict[str, Any]
    assembly_policy_version: str
    omitted: list[dict[str, Any]]
    included_item_ids: list[str]
    included_sources: list[str]
    trace_id: str
    created_at: str
    answer_under_review: dict[str, Any] | None = None
    bundle_schema_version: str = "context-bundle-v1"
    task_type: str | None = None

    def metadata(self) -> dict[str, Any]:
        omitted_by_id = {
            str(item.get("item_id")): str(item.get("reason"))
            for item in self.omitted
            if item.get("item_id")
        }
        included = set(self.included_item_ids)
        answer_summary = (
            {
                "present": True,
                "length": len(str((self.answer_under_review or {}).get("content") or "")),
                "evidence_ids": list((self.answer_under_review or {}).get("evidence_ids") or []),
            }
            if self.answer_under_review
            else {"present": False, "length": 0, "evidence_ids": []}
        )
        inherited_fields = []
        for field in self.resolved_request.get("inherited_fields") or []:
            if not isinstance(field, dict):
                continue
            inherited_fields.append({
                key: field.get(key)
                for key in (
                    "field", "source", "source_turn_id", "source_turn_id_status",
                    "source_task_state_version",
                )
                if field.get(key) is not None
            })
        resolved_request_metadata = {
            key: self.resolved_request.get(key)
            for key in (
                "ref", "storage", "schema_version", "turn_id", "client_turn_id",
                "relation", "confidence", "needs_clarification",
                "clarification_reason", "source_turn_id", "request_contract_hash",
            )
            if self.resolved_request.get(key) is not None
        }
        resolved_request_metadata["explicit_fields"] = list(
            self.resolved_request.get("explicit_fields") or []
        )
        resolved_request_metadata["inherited_fields"] = inherited_fields
        resolved_request_metadata["user_change_fields"] = [
            str(item.get("field")) for item in (self.resolved_request.get("user_changes") or [])
            if isinstance(item, dict) and item.get("field")
        ]
        resolved_request_metadata["candidate_fields"] = [
            str(item.get("field")) for item in (self.resolved_request.get("candidate_fields") or [])
            if isinstance(item, dict) and item.get("field")
        ]
        resolved_request_metadata["source_need_triggers"] = [
            trigger
            for need in (self.resolved_request.get("source_needs") or [])
            if isinstance(need, dict)
            for trigger in (need.get("trigger_ids") or [])
        ]
        task_relation = self.resolved_request.get("task_relation") or {}
        resolved_request_metadata["task_relation"] = {
            key: task_relation.get(key)
            for key in (
                "action", "task_id", "decision_driver", "candidate_task_ids",
                "scope_conflict", "ambiguous",
            )
            if task_relation.get(key) is not None
        }
        return {
            "bundle_schema_version": self.bundle_schema_version,
            "bundle_id": self.bundle_id,
            "turn_id": self.turn_id,
            "task_id": self.task_id,
            "agent": self.agent,
            "task_type": self.task_type,
            # Request text and inherited values are available to the runtime
            # consumer but stay out of trace metadata.
            "resolved_request": resolved_request_metadata,
            "scope_reference": self.scope_reference,
            "assembly_policy_version": self.assembly_policy_version,
            "trace_id": self.trace_id,
            "created_at": self.created_at,
            "budget": self.budget,
            "included_item_ids": list(self.included_item_ids),
            "included_sources": list(self.included_sources),
            "items": [
                item.metadata(included=item.item_id in included, omission_reason=omitted_by_id.get(item.item_id))
                for item in self.items
            ],
            "omitted": self.omitted,
            "answer_under_review": answer_summary,
        }


@dataclass
class TriageContext:
    bundle: ContextBundle
    route_slots: dict[str, Any]
    task_relation: str
    needs_clarification: bool
    source_turn_id: str | None
    preference_source_needs: list[dict[str, Any]] = field(default_factory=list)
    preference_candidates: list[dict[str, Any]] = field(default_factory=list)
    candidate_analysis_plan: dict[str, Any] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        plan = dict(self.candidate_analysis_plan or {})
        return {
            "bundle": self.bundle.metadata(),
            "route_slots": self.route_slots,
            "task_relation": self.task_relation,
            "needs_clarification": self.needs_clarification,
            "source_turn_id": self.source_turn_id,
            "preference_source_needs": list(self.preference_source_needs),
            "preference_candidates": list(self.preference_candidates),
            "candidate_analysis_plan": {
                "schema_version": plan.get("schema_version"),
                "status": plan.get("status"),
                "candidate_ids": list(plan.get("candidate_ids") or []),
                "step_fields": [
                    item.get("field") for item in (plan.get("steps") or [])
                    if isinstance(item, dict) and item.get("field")
                ],
                "applied": bool(plan.get("applied")),
                "requires_server_application": bool(plan.get("requires_server_application")),
            },
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_rank(value: object) -> float:
    """Return an ordering value for source timestamps; unknown dates sort last."""
    if not value:
        return 0.0
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _task_relevance(agent: str, task_type: str, relation: str, source: Source) -> int:
    """Rank sources by how directly they support this agent's current task."""
    task_sources = {
        "metric_definition": {Source.RAG: 3, Source.METRIC_REGISTRY: 4},
        "metric_query": {
            Source.CONVERSATION: 2, Source.TASK: 2,
            Source.METRIC_REGISTRY: 4,
        },
        "anomaly_analysis": {
            Source.TASK: 3, Source.METRIC_REGISTRY: 3,
            Source.ANALYSIS_MEMORY: 2, Source.TOOL_EVIDENCE: 4,
        },
        "report": {
            Source.ANALYSIS_MEMORY: 3, Source.RAG: 3,
            Source.METRIC_REGISTRY: 3, Source.TOOL_EVIDENCE: 4,
        },
    }
    score = task_sources.get(task_type, {}).get(source, 0)
    if source is Source.TOOL_EVIDENCE and agent in {"answer", "critic", "diagnostic", "report"}:
        score = max(score, 4)
    if source in {Source.CONVERSATION, Source.TASK, Source.ANALYSIS_MEMORY}:
        if relation in {"continue", "clarify"}:
            score += 2
    return score


def _refs(*values: object) -> tuple[str, ...]:
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text[:160])
    return tuple(out[:_MAX_REFS])


def _text(value: object, limit: int | None = _MAX_TEXT) -> str:
    raw = str(value or "")
    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw if limit is None else raw[:limit]


def _json_text(value: object, limit: int = _MAX_TEXT) -> str:
    try:
        return _text(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str), limit)
    except Exception:
        return _text(value, limit)


class ContextOrchestrator:
    """Builds TriageContext / specialist ContextBundle / Evidence Refresh."""

    def __init__(self, policy_version: str = POLICY_VERSION) -> None:
        self.policy_version = policy_version

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_triage_context(
        self,
        state: dict[str, Any],
        *,
        instruction: str | None = None,
        system_prompt: str = "",
        model: str | None = None,
    ) -> TriageContext:
        bundle = self.build_specialist_bundle(
            state, "triage",
            instruction=instruction, system_prompt=system_prompt, model=model,
            include_instruction=True,
        )
        compilation = state.get("context_compilation") or {}
        slots = compilation.get("known_slots") or {}
        route_slots = {
            name: slot.get("value")
            for name, slot in slots.items()
            if name in {"metric", "date", "dimension", "operation"} and isinstance(slot, dict) and slot.get("value")
        }
        triage = TriageContext(
            bundle=bundle,
            route_slots=route_slots,
            task_relation=str(bundle.resolved_request.get("relation") or "new"),
            needs_clarification=bool(compilation.get("needs_clarification")),
            source_turn_id=compilation.get("source_turn_id"),
            preference_source_needs=list(
                (state.get("turn_resolution") or {}).get("source_needs") or []
            ),
            preference_candidates=list(
                ((state.get("preference_context") or {}).get("accepted") or [])
            ),
            candidate_analysis_plan=dict(
                (state.get("preference_context") or {}).get("candidate_analysis_plan") or {}
            ),
        )
        state["triage_context"] = triage.metadata()
        return triage

    def build_specialist_bundle(
        self,
        state: dict[str, Any],
        agent: str,
        *,
        instruction: str | None = None,
        system_prompt: str = "",
        model: str | None = None,
        include_instruction: bool = False,
        extra_reserved_tokens: int = 0,
        additional_items: Iterable[OrchestratedItem] | None = None,
        replace_sources: Iterable[Source] = (),
    ) -> ContextBundle:
        policy = AGENT_POLICIES.get(agent)
        if policy is None:
            raise ValueError(f"Unknown context agent: {agent}")

        collected, source_omissions = self._collect(state, policy, agent=agent)
        replaced_sources = set(replace_sources)
        if replaced_sources:
            collected = [item for item in collected if item.source not in replaced_sources]
            replaced_names = {source.value for source in replaced_sources}
            source_omissions = [
                omission for omission in source_omissions
                if omission.get("source") not in replaced_names
            ]
        supplemental = list(additional_items or [])
        collected.extend(supplemental)
        supplemented_sources = {
            *(item.source.value for item in supplemental),
            *(source.value for source in replaced_sources),
        }
        source_omissions = [
            omission for omission in source_omissions
            if omission.get("source") not in supplemented_sources
        ]
        rejected = [
            item for item in collected
            if item.source not in policy.sources or item.trust_tier not in policy.tiers
        ]
        eligible = [
            item for item in collected
            if item.source in policy.sources and item.trust_tier in policy.tiers
        ]
        tier_rank = {
            TrustTier.CURRENT_EVIDENCE: 5,
            TrustTier.CONTROLLED_DEFINITION: 4,
            TrustTier.HISTORICAL_REFERENCE: 3,
            TrustTier.RETRIEVED_DOCUMENT: 2,
            TrustTier.PREFERENCE: 1,
        }
        task_type = str(state.get("task_type") or "")
        relation = str(
            (state.get("turn_resolution") or {}).get("relation")
            or (state.get("context_compilation") or {}).get("relation")
            or "new"
        )

        def selection_key(item: OrchestratedItem) -> tuple[Any, ...]:
            return (
                -tier_rank[item.trust_tier],
                -_timestamp_rank(item.created_at),
                -_task_relevance(agent, task_type, relation, item.source),
                -item.priority,
                item.item_id,
            )

        required_items = sorted(
            (item for item in eligible if item.required), key=selection_key,
        )
        optional_items = sorted(
            (item for item in eligible if not item.required), key=selection_key,
        )
        optional_capacity = max(0, policy.max_items - len(required_items))
        items = required_items + optional_items[:optional_capacity]
        selected_ids = {item.item_id for item in items}
        truncated = [item for item in eligible if item.item_id not in selected_ids]
        estimate_model = model or ""
        estimated_by_id: dict[str, int] = {}
        for item in collected:
            estimated_by_id[item.item_id] = rendered_item_tokens(
                self._to_packer_item(item), estimate_model,
            )
        pre_pack_omitted = [
            {"item_id": item.item_id, "source": item.source.value,
             "reason": "policy_source_or_tier", "token_estimate": estimated_by_id.get(item.item_id, 0)}
            for item in rejected
        ] + [
            {"item_id": item.item_id, "source": item.source.value,
             "reason": "max_items_exceeded", "token_estimate": estimated_by_id.get(item.item_id, 0)}
            for item in truncated
        ]

        pack_items = [self._to_packer_item(item) for item in items]
        instruction_text = _text(instruction if instruction is not None else self._resolved_text(state), None)
        instruction_tokens = estimate_tokens(instruction_text, model) if instruction_text else 0
        extra_reserved = 0 if include_instruction else instruction_tokens
        extra_reserved += max(0, int(extra_reserved_tokens))
        if include_instruction and instruction_text:
            pack_items.insert(0, PackerContextItem(
                item_id="instruction",
                source=PackerSource.CURRENT_QUESTION,
                content=instruction_text,
                priority=100,
                required=True,
                trust_level=TrustLevel.USER_INPUT,
            ))

        packed = pack_for_llm(
            agent,
            pack_items,
            system_prompt=system_prompt,
            model=model,
            state=state,
            extra_reserved_tokens=extra_reserved,
        )
        estimated_by_id = {
            item.item_id: rendered_item_tokens(self._to_packer_item(item), packed.budget.model)
            for item in collected
        }
        for omission in pre_pack_omitted:
            omission["token_estimate"] = estimated_by_id.get(str(omission.get("item_id") or ""), 0)
        pack_by_id = {pack_item.item_id: pack_item for pack_item in pack_items}
        items = [
            replace(
                item,
                token_estimate=rendered_item_tokens(
                    pack_by_id[item.item_id], packed.budget.model,
                ) if item.item_id in pack_by_id else estimate_tokens(item.content, packed.budget.model),
            )
            for item in items
        ]
        included = set(packed.included_item_ids)
        bundle_included_ids = [item_id for item_id in packed.included_item_ids if item_id != "instruction"]
        packer_omitted = []
        for omission in packed.omitted:
            item_id = str(omission.get("item_id") or "")
            item = next((candidate for candidate in eligible if candidate.item_id == item_id), None)
            packer_omitted.append({
                "item_id": item_id,
                "source": str(omission.get("source") or (item.source.value if item else "unknown")),
                "reason": str(omission.get("reason") or "omitted"),
                "token_estimate": estimated_by_id.get(item_id, 0),
            })
        state.setdefault("context_source_availability", {})[agent] = source_omissions
        omitted = pre_pack_omitted + packer_omitted
        answer_text = str(state.get("answer") or "")
        answer_under_review = None
        if policy.include_answer_under_review and answer_text.strip():
            current_evidence_ids = [
                str(item.get("evidence_id"))
                for item in (state.get("evidence_bundles") or [])
                if _is_current_evidence(item)
            ]
            answer_under_review = {
                "content": answer_text,
                "evidence_ids": list(dict.fromkeys(current_evidence_ids)),
            }

        from app.context.turn_resolver import client_turn_id, durable_turn_id

        bundle_turn_id = durable_turn_id(state)
        resolution = state.get("turn_resolution") or {}
        compilation = state.get("context_compilation") or {}
        if not isinstance(resolution, dict):
            resolution = {}
        relation = str(resolution.get("relation") or compilation.get("relation") or "new")
        relation = {
            "independent": "new",
            "continue": "continue",
        }.get(relation, relation)
        if resolution.get("needs_clarification") or compilation.get("needs_clarification"):
            relation = "clarify"
        if relation not in {"continue", "new", "parallel_new", "clarify"}:
            relation = "new"
        resolved_request = {
            "ref": f"turn-resolution:{bundle_turn_id}",
            "storage": "chat_turns.context_snapshot.turn_resolution",
            "schema_version": str(resolution.get("schema_version") or "turn-resolution-v1"),
            "turn_id": bundle_turn_id,
            "client_turn_id": client_turn_id(state),
            "text": str(resolution.get("resolved_text") or compilation.get("standalone_question") or state.get("user_message") or ""),
            "relation": relation,
            "explicit_fields": list(resolution.get("explicit_fields") or []),
            "inherited_fields": list(resolution.get("inherited_fields") or []),
            "user_changes": list(resolution.get("user_changes") or []),
            "candidate_fields": list(resolution.get("candidate_fields") or []),
            "scope_candidates": list(resolution.get("scope_candidates") or []),
            "source_needs": list(resolution.get("source_needs") or []),
            "task_relation": dict(resolution.get("task_relation") or {}),
            "confidence": resolution.get("confidence"),
            "needs_clarification": bool(resolution.get("needs_clarification") or compilation.get("needs_clarification")),
            "clarification_reason": resolution.get("clarification_reason") or compilation.get("clarification_reason"),
            "source_turn_id": resolution.get("source_turn_id") or compilation.get("source_turn_id"),
            "request_contract_hash": str(resolution.get("request_contract_hash") or state.get("request_contract_hash") or ""),
        }
        bundle = ContextBundle(
            bundle_id=f"ctx_{uuid4().hex[:12]}",
            bundle_schema_version="context-bundle-v1",
            turn_id=bundle_turn_id,
            task_id=str(state["task_id"]) if state.get("task_id") else None,
            agent=agent,
            resolved_request=resolved_request,
            scope_reference=str(state.get("current_query_scope_fingerprint") or ""),
            items=items,
            text=packed.text,
            budget={
                **packed.budget.model_dump(),
                "estimated_tokens": packed.estimated_tokens,
                "optional_candidates": packed.optional_candidates,
                "optional_included": packed.optional_included,
                "over_budget_required": packed.over_budget_required,
                "token_counting_method": packed.token_counting_method,
                "instruction_included": bool(include_instruction and instruction_text),
                "extra_reserved_tokens": extra_reserved,
            },
            assembly_policy_version=self.policy_version,
            omitted=omitted,
            included_item_ids=bundle_included_ids,
            included_sources=list(packed.included_sources),
            trace_id=str(state.get("trace_id") or ""),
            created_at=_now(),
            answer_under_review=answer_under_review,
            task_type=str(state.get("task_type") or "") or None,
        )
        self._record(state, bundle)
        return bundle

    def refresh_evidence(self, state: dict[str, Any]) -> dict[str, Any]:
        """Mark the current-turn evidence as available for downstream agents."""
        evidence = [
            item for item in (state.get("evidence_bundles") or [])
            if _is_current_evidence(item)
        ]
        evidence_ids = list(dict.fromkeys(str(item["evidence_id"]) for item in evidence))
        refresh = {
            "available": bool(evidence_ids),
            "evidence_ids": evidence_ids,
            "count": len(evidence_ids),
            "refreshed_at": _now(),
            "tool_call_count": len(state.get("tool_calls") or []),
        }
        state["evidence_refresh"] = refresh
        record_event("context.evidence_refresh", {
            "trace_id": state.get("trace_id"),
            "evidence.count": refresh["count"],
            "tool_call.count": refresh["tool_call_count"],
        })
        return refresh

    def packer_items(self, bundle: ContextBundle) -> list[PackerContextItem]:
        """Return only included Bundle items in the existing Packer contract."""
        included = set(bundle.included_item_ids)
        return [
            self._to_packer_item(item)
            for item in bundle.items
            if item.item_id in included
        ]

    # ------------------------------------------------------------------
    # Source readers (read only already-authorized state)
    # ------------------------------------------------------------------
    def _collect(
        self, state: dict[str, Any], policy: AgentPolicy, *, agent: str,
    ) -> tuple[list[OrchestratedItem], list[dict[str, Any]]]:
        items: list[OrchestratedItem] = []
        produced: set[Source] = set()
        if Source.CONVERSATION in policy.sources:
            items.extend(self._conversation_items(state))
        if Source.TASK in policy.sources:
            items.extend(self._task_items(state))
        if Source.ANALYSIS_MEMORY in policy.sources:
            items.extend(self._analysis_memory_items(state))
        if Source.RAG in policy.sources:
            items.extend(self._rag_items(state))
        if Source.METRIC_REGISTRY in policy.sources:
            items.extend(self._registry_items(state))
        if Source.TOOL_EVIDENCE in policy.sources:
            items.extend(self._tool_evidence_items(state))
        if Source.MEM0 in policy.sources:
            items.extend(self._mem0_items(state, agent=agent))
        produced = {item.source for item in items}

        omissions: list[dict[str, Any]] = []
        for source in sorted(policy.sources, key=lambda item: item.value):
            if source in produced:
                continue
            if source is Source.MEM0:
                preference = state.get("preference_context") or {}
                reason = str(
                    preference.get("availability")
                    or ("not_requested" if preference.get("status") == "not_requested" else "not_configured")
                )
            else:
                reason = "empty_or_unavailable"
            omissions.append({"source": source.value, "reason": reason})
        return items, omissions

    def _conversation_items(self, state: dict[str, Any]) -> list[OrchestratedItem]:
        items: list[OrchestratedItem] = []
        scope = str(state.get("current_query_scope_fingerprint") or "")
        version = str(state.get("workflow_version") or "state")
        previous = state.get("previous_turn_context") or {}
        if isinstance(previous, dict) and previous:
            slots = previous.get("context_slots") or {}
            metric = (slots.get("metric") or {}).get("value")
            date_value = (slots.get("date") or {}).get("value")
            dimension = (slots.get("dimension") or {}).get("value")
            parts = []
            if metric:
                parts.append(f"metric={metric}")
            if date_value:
                parts.append(f"date={date_value}")
            if dimension:
                parts.append(f"dimension={dimension}")
            if parts:
                items.append(self._item(
                    "ctx.prev_turn", Source.CONVERSATION, TrustTier.HISTORICAL_REFERENCE,
                    "; ".join(parts), _refs(previous.get("source_turn_id")), scope, version,
                    required=False, priority=60, created_at=previous.get("created_at"),
                ))
        summary_source = previous.get("conversation_summary") or state.get("session_memory_context")
        if not summary_source and not state.get("analysis_memories"):
            # Backward-compatible fallback for direct callers; once analysis
            # memories are present, memory_context is a combined field and must
            # not be relabelled as a conversation summary.
            summary_source = state.get("memory_context")
        summary = _text(summary_source, 1200)
        if summary:
            items.append(self._item(
                "ctx.session_summary", Source.CONVERSATION, TrustTier.HISTORICAL_REFERENCE,
                summary, _refs(
                    previous.get("summary_covered_from_turn_id"),
                    previous.get("summary_covered_to_turn_id"),
                    previous.get("summary_policy_version"),
                    previous.get("summary_generated_at"),
                ), scope, version,
                required=False, priority=40,
                created_at=previous.get("summary_generated_at") or (
                    summary_source.get("created_at") if isinstance(summary_source, dict) else None
                ),
            ))
        open_questions = previous.get("open_questions") or []
        if isinstance(open_questions, list) and open_questions:
            rendered_questions = [
                item for item in open_questions[:10]
                if isinstance(item, dict) and item.get("field")
            ]
            if rendered_questions:
                items.append(self._item(
                    "ctx.session_open_questions", Source.CONVERSATION,
                    TrustTier.HISTORICAL_REFERENCE,
                    _json_text(rendered_questions, 1200),
                    _refs(*[
                        item.get("source_turn_id") for item in rendered_questions
                        if item.get("source_turn_id")
                    ]), scope, version, required=False, priority=65,
                    created_at=previous.get("open_questions_updated_at"),
                ))
        recent_turns = previous.get("recent_committed_turns") or []
        if isinstance(recent_turns, list) and recent_turns:
            lines = []
            turn_refs = []
            for turn in recent_turns[-20:]:
                if not isinstance(turn, dict):
                    continue
                question = _text(turn.get("question"), 120)
                answer = _text(turn.get("answer"), 160)
                if not question:
                    continue
                lines.append(f"用户：{question}" + (f"；答复：{answer}" if answer else ""))
                turn_refs.append(str(turn.get("turn_id") or ""))
            recent_text = "最近提交轮次（历史参考，不是当前数据证据）：" + "\n".join(lines)
            recent_text = _text(recent_text, 2400)
            if recent_text:
                items.append(self._item(
                    "ctx.recent_committed_turns", Source.CONVERSATION,
                    TrustTier.HISTORICAL_REFERENCE, recent_text,
                    _refs(*turn_refs), scope, version, required=False, priority=50,
                ))
        claims = state.get("historical_claims_context") or []
        if claims:
            text = "; ".join(_text(item.get("value"), 160) for item in claims[:5] if isinstance(item, dict) and item.get("value"))
            if text:
                items.append(self._item(
                    "ctx.historical_claims", Source.ANALYSIS_MEMORY, TrustTier.HISTORICAL_REFERENCE,
                    text, _refs(previous.get("source_turn_id"), state.get("task_id")), scope, version,
                    required=False, priority=55,
                    created_at=max(
                        (str(claim.get("created_at") or "") for claim in claims
                         if isinstance(claim, dict)),
                        default="",
                    ),
                ))
        hypotheses = state.get("historical_hypothesis_context") or {}
        if isinstance(hypotheses, dict) and hypotheses:
            text = "; ".join(
                f"{metric}: " + ", ".join(
                    f"{_text(step.get('hypothesis'), 80)}({step.get('verdict')})"
                    for step in chain[:3] if isinstance(step, dict)
                )
                for metric, chain in list(hypotheses.items())[:2] if isinstance(chain, list)
            )
            if text:
                items.append(self._item(
                    "ctx.historical_hypotheses", Source.ANALYSIS_MEMORY, TrustTier.HISTORICAL_REFERENCE,
                    text, _refs(previous.get("source_turn_id"), state.get("task_id")), scope, version,
                    required=False, priority=50,
                    created_at=max(
                        (str(step.get("created_at") or "") for chain in hypotheses.values()
                         if isinstance(chain, list) for step in chain if isinstance(step, dict)),
                        default="",
                    ),
                ))
        return items

    def _task_items(self, state: dict[str, Any]) -> list[OrchestratedItem]:
        progress = state.get("task_progress_context") or {}
        if not isinstance(progress, dict) or not progress:
            return []
        # Never revive task progress that the compiler rejected as
        # scope/data-version incompatible.  In the graph the compiler sets
        # task_progress_context_used; direct callers fall back to the same
        # fingerprint/version comparison.
        compilation = state.get("context_compilation") or {}
        if not compilation.get("task_progress_context_used"):
            current_scope = str(state.get("current_query_scope_fingerprint") or "")
            current_version = str(state.get("data_version_manifest_hash") or "")
            if not (
                current_scope and current_version
                and str(progress.get("scope_fingerprint") or "") == current_scope
                and str(progress.get("data_version_manifest_hash") or "") == current_version
            ):
                return []
        parts = []
        objective = _text(progress.get("objective"), 400)
        if objective:
            parts.append(f"objective={objective}")
        steps = progress.get("completed_steps") or []
        if steps:
            parts.append("completed=" + " → ".join(_text(step, 60) for step in steps[-12:]))
        pending = progress.get("pending_questions") or []
        if pending:
            parts.append("pending=" + ", ".join(_text(item, 80) for item in pending[:5]))
        if not parts:
            return []
        return [self._item(
            "task.progress", Source.TASK, TrustTier.HISTORICAL_REFERENCE,
            "; ".join(parts),
            _refs(state.get("task_id"), progress.get("latest_trace_id")),
            str(progress.get("scope_fingerprint") or state.get("current_query_scope_fingerprint") or ""),
            str(progress.get("schema_version") or "task-state-v1"),
            required=False, priority=55,
            created_at=progress.get("updated_at") or progress.get("created_at"),
        )]

    def _analysis_memory_items(self, state: dict[str, Any]) -> list[OrchestratedItem]:
        items = []
        for row in (state.get("analysis_memories") or [])[:3]:
            if not isinstance(row, dict) or not row.get("insight_text"):
                continue
            content = _text(row.get("insight_text"), 400)
            limitations = _text(row.get("limitations"), 160)
            if limitations:
                content += f" (限制: {limitations})"
            items.append(self._item(
                f"memory.{row.get('memory_id') or uuid4().hex[:8]}",
                Source.ANALYSIS_MEMORY, TrustTier.HISTORICAL_REFERENCE,
                content,
                _refs(row.get("memory_id"), row.get("source_trace_id")),
                str(row.get("scope_fingerprint") or state.get("current_query_scope_fingerprint") or ""),
                str(row.get("source_data_version") or "memory"),
                required=False, priority=45,
                evidence_ids=tuple(str(item) for item in (row.get("evidence_ids") or [])[:10]),
                sensitivity="internal",
                created_at=row.get("created_at") or row.get("updated_at"),
            ))
        return items

    def _rag_items(self, state: dict[str, Any]) -> list[OrchestratedItem]:
        rag_result = state.get("rag_result") or {}
        context = _text(rag_result.get("context") if isinstance(rag_result, dict) else "", 1600)
        if not context:
            return []
        refs = _refs(
            *[
                citation.get("chunk_id") or citation.get("document_id") or citation.get("doc_id")
                for citation in (state.get("rag_citations") or [])[:8]
                if isinstance(citation, dict)
            ],
            state.get("rag_release_version"),
        )
        return [self._item(
            "rag.context", Source.RAG, TrustTier.RETRIEVED_DOCUMENT,
            context, refs,
            str(state.get("current_query_scope_fingerprint") or ""),
            str(state.get("rag_release_version") or "rag"),
            required=False, priority=50,
        )]

    def _registry_items(self, state: dict[str, Any]) -> list[OrchestratedItem]:
        authority = state.get("metric_authoritative_definition")
        if not isinstance(authority, dict) or not authority:
            return []
        metric_id = str(authority.get("metric_id") or authority.get("metric_name") or "")
        return [self._item(
            f"registry.{metric_id or 'metric'}",
            Source.METRIC_REGISTRY, TrustTier.CONTROLLED_DEFINITION,
            _json_text(authority, 1200),
            _refs(metric_id, authority.get("definition_version") or authority.get("version")),
            str(state.get("current_query_scope_fingerprint") or ""),
            str(authority.get("definition_version") or authority.get("version") or "registry"),
            required=False, priority=70,
            created_at=authority.get("updated_at") or authority.get("created_at"),
        )]

    def _tool_evidence_items(self, state: dict[str, Any]) -> list[OrchestratedItem]:
        items: list[OrchestratedItem] = []
        investigation_context = state.get("critic_investigation_context")
        if investigation_context:
            if isinstance(investigation_context, dict):
                # Keep a complete, compact projection instead of slicing JSON
                # at an arbitrary character offset. A truncated object can no
                # longer be parsed or safely interpreted as current evidence.
                included_fields = (
                    "target", "plan_steps", "validated_numeric_decompositions",
                    "verified_evidence_ids", "investigation_limitations",
                    "terminal_reason", "global_stop", "business_causality",
                    "conflicting_evidence_ids",
                    # Registered numeric decomposition context fields. Keep
                    # these server-derived review inputs visible to the Critic;
                    # omitting them leaves the prompt without the verified
                    # evidence and the operation's business-causality boundary.
                    "operation_intent", "task_type", "target_metric_id", "method_id",
                    "scope", "baseline_date", "current_date",
                    "decomposition_review_status", "verified_numeric_evidence",
                    "limitations", "stop_reason",
                )
                projection = {
                    "projection_schema": "critic-investigation-summary-v1",
                    **{
                        key: investigation_context[key]
                        for key in included_fields
                        if key in investigation_context
                    },
                    "omitted_fields": [
                        key for key in investigation_context
                        if key not in included_fields and key != "schema_version"
                    ],
                }
                content = _json_text(projection, None)
            elif isinstance(investigation_context, list):
                content = _json_text(investigation_context, None)
            else:
                content = _text(investigation_context, 2400)
            if content:
                items.append(self._item(
                    "evidence.investigation_context", Source.TOOL_EVIDENCE,
                    TrustTier.CURRENT_EVIDENCE, content,
                    _refs("investigation_context"),
                    str(state.get("current_query_scope_fingerprint") or ""),
                    str(state.get("data_version_manifest_hash") or "current"),
                    required=True, priority=85,
                    content_limit=12000,
                ))
        refresh = state.get("evidence_refresh") or {}
        if not refresh.get("available"):
            return items
        refreshed_ids = list(dict.fromkeys(str(item) for item in (refresh.get("evidence_ids") or []) if item))
        refreshed_set = set(refreshed_ids)
        refreshed = {
            str(row.get("evidence_id")): row
            for row in (state.get("evidence_bundles") or [])
            if isinstance(row, dict) and str(row.get("evidence_id") or "") in refreshed_set
        }
        for evidence_id in refreshed_ids:
            row = refreshed.get(evidence_id)
            if not row:
                continue
            if not _is_current_evidence(row):
                continue
            raw_content = row.get("content") or {}
            if isinstance(raw_content, dict) and "result" in raw_content:
                prompt_content = {
                    key: raw_content.get(key)
                    for key in ("metric_id", "query_type", "result")
                    if raw_content.get(key) is not None
                }
            elif isinstance(raw_content, dict):
                prompt_content = {
                    key: value for key, value in raw_content.items()
                    if key not in {"question", "original_query", "request_id", "prompt", "instructions"}
                }
            else:
                prompt_content = {}
            content = " | ".join(filter(None, [
                evidence_id,
                str(row.get("source_type") or ""),
                _text(row.get("source_ref"), 120),
                _json_text(prompt_content, 12000),
            ]))
            items.append(self._item(
                f"evidence.{evidence_id}", Source.TOOL_EVIDENCE, TrustTier.CURRENT_EVIDENCE,
                content, _refs(evidence_id),
                str(state.get("current_query_scope_fingerprint") or ""),
                str(state.get("data_version_manifest_hash") or "current"),
                required=True, priority=80,
                evidence_ids=(evidence_id,),
                content_limit=12000,
                created_at=row.get("created_at") or row.get("observed_at"),
            ))
        return items

    def _mem0_items(self, state: dict[str, Any], *, agent: str) -> list[OrchestratedItem]:
        # WP2 accepts only resolver-validated candidates.  WP3 owns the Mem0
        # adapter; when it is not configured the resolver records that state
        # and this reader intentionally returns no preference.
        context = state.get("preference_context") or {}
        if not isinstance(context, dict):
            return []
        items: list[OrchestratedItem] = []
        for candidate in (context.get("items") or [])[:10]:
            if not isinstance(candidate, dict) or candidate.get("status") != "accepted":
                continue
            memory_type = str(candidate.get("memory_type") or "")
            # Triage may inspect analysis/scope candidates to form a route
            # decision.  The answer agent may consume presentation preferences
            # only; raw Mem0 never becomes a data-query parameter or fact.
            if agent == "answer" and memory_type != "presentation":
                continue
            # Use the verified projection when available.  Older/runtime
            # callers may still provide only `value`; the resolver sanitizes
            # both paths, and this preference prevents a future reader from
            # accidentally re-introducing a rejected raw field.
            value = candidate.get("effective_fields")
            if value in (None, ""):
                value = candidate.get("value")
            if value in (None, ""):
                continue
            content = (
                f"preference_type={candidate.get('memory_type')}; "
                f"key={candidate.get('memory_key')}; "
                f"value={_json_text(value, 600)}. "
                "This is a personalisation preference, never an authorization or fact."
            )
            items.append(self._item(
                f"preference.{candidate.get('candidate_id') or candidate.get('memory_key')}",
                Source.MEM0, TrustTier.PREFERENCE,
                content,
                _refs(candidate.get("candidate_id"), candidate.get("source_ref")),
                str(state.get("current_query_scope_fingerprint") or ""),
                "preference-context-v1",
                required=False, priority=65,
                sensitivity="internal",
            ))
        return items

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _item(
        self,
        item_id: str,
        source: Source,
        trust_tier: TrustTier,
        content: str,
        source_refs: Iterable[str],
        scope_fingerprint: str,
        version: str,
        *,
        required: bool,
        priority: int,
        evidence_ids: tuple[str, ...] = (),
        sensitivity: str = "internal",
        content_limit: int = _MAX_TEXT,
        created_at: str | None = None,
    ) -> OrchestratedItem:
        return OrchestratedItem(
            item_id=item_id,
            source=source,
            trust_tier=trust_tier,
            content=_text(content, content_limit),
            source_refs=tuple(source_refs),
            scope_fingerprint=scope_fingerprint,
            version=version,
            required=required,
            priority=priority,
            created_at=str(created_at or ""),
            evidence_ids=evidence_ids,
            sensitivity=sensitivity,
        )

    def _to_packer_item(self, item: OrchestratedItem) -> PackerContextItem:
        refs = f"refs={','.join(item.source_refs)}\n" if item.source_refs else ""
        return PackerContextItem(
            item_id=item.item_id,
            source=PACKER_SOURCE[item.source],
            content=f"{refs}{item.content}",
            priority=item.priority,
            required=item.required,
            trust_level=PACKER_TRUST[item.trust_tier],
            evidence_ids=item.evidence_ids,
        )

    def _resolved_text(self, state: dict[str, Any]) -> str:
        compilation = state.get("context_compilation") or {}
        return str(
            compilation.get("standalone_question")
            or state.get("user_message")
            or ""
        ).strip()

    def _record(self, state: dict[str, Any], bundle: ContextBundle) -> None:
        metadata = bundle.metadata()
        state.setdefault("context_bundles", {})[bundle.agent] = metadata
        events = state.setdefault("context_orchestration_events", [])
        events.append(metadata)
        if len(events) > _MAX_EVENTS:
            del events[:-_MAX_EVENTS]
        summary = state.setdefault("context_orchestration_summary", {
            "bundles": 0, "items": 0, "included": 0, "omitted": 0,
            "required_overflows": 0, "input_tokens": 0,
        })
        summary["bundles"] = int(summary.get("bundles", 0)) + 1
        summary["items"] = int(summary.get("items", 0)) + len(bundle.items)
        summary["included"] = int(summary.get("included", 0)) + len(bundle.included_item_ids)
        summary["omitted"] = int(summary.get("omitted", 0)) + len(bundle.omitted)
        summary["required_overflows"] = int(summary.get("required_overflows", 0)) + int(
            bool((bundle.budget or {}).get("over_budget_required"))
        )
        summary["input_tokens"] = int(summary.get("input_tokens", 0)) + int(
            (bundle.budget or {}).get("estimated_tokens") or 0
        )
        record_event("context.orchestrated", {
            "trace_id": bundle.trace_id,
            "context.agent": bundle.agent,
            "context.bundle_id": bundle.bundle_id,
            "context.policy_version": bundle.assembly_policy_version,
            "context.items": len(bundle.items),
            "context.included": len(bundle.included_item_ids),
            "context.omitted": len(bundle.omitted),
            "context.input_tokens": (bundle.budget or {}).get("estimated_tokens"),
            "context.over_budget_required": (bundle.budget or {}).get("over_budget_required"),
        })
        record_measurement(
            "context_bundle_tokens",
            float((bundle.budget or {}).get("estimated_tokens") or 0),
            labels={"agent": bundle.agent},
        )
        record_measurement(
            "context_bundle_omitted_items", float(len(bundle.omitted)),
            labels={"agent": bundle.agent},
        )

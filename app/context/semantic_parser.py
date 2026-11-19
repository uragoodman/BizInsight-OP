"""Deterministic Mention → Entity → Operation → Frame → Role → Relation parser."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from app.context.dialogue_context import DialogueContextResolution, InheritedField
from app.context.dimension_intent import classify_dimension_mentions
from app.context.entity_resolver import (
    EntityResolution,
    detect_entity_mentions,
    resolve_entity,
)
from app.context.relation_schema import make_relation, unique_relations
from app.context.request_frame import (
    RequestFrame,
    build_request_frame,
    operation_spans,
)
from app.context.semantic_contract import (
    Entity,
    Mention,
    Operation,
    ReferenceIntegrityIssue,
    Role,
    SemanticContract,
    Span,
    reference_integrity_issues,
)

SemanticStatus = Literal["CONFIRMED", "AMBIGUOUS", "UNSUPPORTED", "MISSING", "CONFLICT"]
_SUPPORTED_OPERATION_IDS = frozenset(
    {
        "segment", "aggregate", "compare", "filter", "sort", "stat", "trend",
        "contribution", "breakdown", "definition", "enumerate", "explain_cause",
        "metric_query", "rfm", "continue_analysis", "explain_rfm", "simulate_rfm",
        "inspect_customer",
    }
)
_SWITCH_MARKER = re.compile(
    r"不做这个了|不要这个|换成|改成|改为|改查|改看|转而|另一个|重新查|重新看|先不做|不继续|"
    r"(?:不要|不做|别再)\s*(?:继续|再)?\s*(?:做|进行|执行)?\s*(?:R\s*F\s*M|分层|这个)|\bswitch to\b",
    re.I,
)
_GROUP_PREFIX = re.compile(r"(?:按|按照|根据|依据|分别|各个?|每个?|基于|以)\s*$", re.I)
_FILTER_PREFIX = re.compile(r"(?:只看|仅看|筛选|过滤|限定|where)\s*$", re.I)
_REFERENCE_PREFIX = re.compile(r"(?:这个|那个|上述|前述|剛才的|刚才的|其|它的)\s*$", re.I)


def _operation_mentions(question: str) -> tuple[Mention, ...]:
    result: list[Mention] = []
    for text, start, end in operation_spans(question):
        result.append(
            Mention(
                text=text,
                span=Span(text=text, start=start, end=end),
                candidate_types=("operation",),
            )
        )
    return tuple(result)


def _merge_mentions(mentions: Sequence[Mention]) -> tuple[Mention, ...]:
    merged: dict[tuple[int, int], tuple[str, set[str]]] = {}
    for mention in mentions:
        key = (mention.span.start, mention.span.end)
        current = merged.get(key)
        if current is None:
            merged[key] = (mention.text, set(mention.candidate_types))
        else:
            current[1].update(mention.candidate_types)
    return tuple(
        Mention(
            text=text,
            span=Span(text=text, start=start, end=end),
            candidate_types=tuple(sorted(types)),
        )
        for (start, end), (text, types) in sorted(merged.items())
    )


def _operation_payload(value: Any) -> Operation | None:
    if isinstance(value, Operation):
        return value if value.id in _SUPPORTED_OPERATION_IDS else None
    if isinstance(value, Mapping):
        operation_id = value.get("id") or value.get("operation")
        if not operation_id or str(operation_id) not in _SUPPORTED_OPERATION_IDS:
            return None
        mode = value.get("mode") if value.get("mode") in {"EXECUTE", "EXPLAIN", "ASK"} else "EXECUTE"
        polarity = value.get("polarity") if value.get("polarity") in {"POSITIVE", "NEGATIVE"} else "POSITIVE"
        label = value.get("label")
        return Operation(
            id=str(operation_id),
            label=str(label) if label is not None else None,
            mode=mode,
            polarity=polarity,
        )
    if isinstance(value, str) and value.strip():
        operation_id = value.strip()
        if operation_id not in _SUPPORTED_OPERATION_IDS:
            return None
        return Operation(id=operation_id, label=None, mode="EXECUTE", polarity="POSITIVE")
    return None


def _entity_from_inherited(item: InheritedField) -> Entity | None:
    value = item.value
    if isinstance(value, Mapping):
        entity_id = value.get("id") or value.get("entity") or value.get("value")
        entity_type = value.get("type")
        alias_type = value.get("alias_type")
        confidence = value.get("confidence", 0.75)
        if not entity_id:
            return None
        if entity_type not in {
            "metric", "dimension", "business_entity", "semantic_factor", "time", "scope"
        }:
            entity_type = None
        if entity_type:
            return Entity(
                id=str(entity_id),
                type=entity_type,
                alias_type=alias_type if alias_type in {"generic", "specific"} else None,
                confidence=confidence,
            )
        value = entity_id
    entity_id = str(value).strip() if value is not None else ""
    if not entity_id:
        return None
    known = {
        "customer": "business_entity",
        "order": "business_entity",
        "product": "business_entity",
        "purchase_frequency": "semantic_factor",
        "monetary": "semantic_factor",
        "customer_value": "semantic_factor",
        "rfm": "semantic_factor",
        "gmv": "metric",
        "order_count": "metric",
        "customer_count": "metric",
        "previous_month": "time",
        "current_month": "time",
        "previous_week": "time",
        "current_week": "time",
        "previous_year": "time",
        "today": "time",
        "yesterday": "time",
        "seller_scope": "scope",
        "platform_scope": "scope",
    }
    if item.field in {"time", "time_scope", "date", "date_scope"}:
        if entity_id in known and known[entity_id] == "time":
            return Entity(id=entity_id, type="time", alias_type="specific", confidence=0.75)
        if re.fullmatch(
            r"\d{4}-\d{2}-\d{2}(?:\s*(?:至|到|~|–|-)\s*\d{4}-\d{2}-\d{2})?",
            entity_id,
        ):
            return Entity(id=entity_id, type="time", alias_type="specific", confidence=0.75)
        return None
    if item.field in {"scope", "seller_scope", "platform_scope"} and entity_id in {
        "seller_scope", "platform_scope"
    }:
        return Entity(id=entity_id, type="scope", alias_type="generic", confidence=0.75)
    entity_type = known.get(entity_id)
    if entity_type is None:
        try:
            from app.metrics.registry import get_registry

            if any(str(item.metric_id) == entity_id for item in get_registry().list()):
                entity_type = "metric"
            elif get_registry().dimension(entity_id) is not None:
                entity_type = "dimension"
        except Exception:  # noqa: BLE001 - unknown inherited values fail closed
            entity_type = None
    if entity_type is None:
        return None
    return Entity(id=entity_id, type=entity_type, alias_type="generic", confidence=0.75)


def _inherited_records(
    fields: Sequence[InheritedField], entities: Sequence[Entity], operations: Sequence[Operation]
) -> list[dict[str, Any]]:
    entity_ids = {item.id for item in entities}
    operation_ids = {item.id for item in operations}
    records: list[dict[str, Any]] = []
    for item in fields:
        if not item.validated:
            continue
        value: str | None = None
        if item.field in {"operation", "operations"}:
            operation = _operation_payload(item.value)
            if operation and operation.id in operation_ids:
                value = operation.id
        else:
            entity = _entity_from_inherited(item)
            if entity and entity.id in entity_ids:
                value = entity.id
        if value is not None:
            records.append(
                {
                    "field": item.field,
                    "value": value,
                    "source_turn": item.source_turn,
                    "reason": item.reason,
                    "validated": True,
                }
            )
    return records


def _current_explicit_entity_ids(entities: Sequence[Entity]) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {"metric": set(), "business_entity": set(), "dimension": set()}
    for entity in entities:
        if entity.type in groups:
            groups[entity.type].add(entity.id)
    return groups


def _validate_dialogue_act(
    question: str,
    context: DialogueContextResolution | None,
    current_entities: Sequence[Entity],
) -> tuple[str, bool]:
    if context is None:
        return "NEW", False
    provisional = context.provisional_dialogue_act
    if provisional == "SWITCH" or _SWITCH_MARKER.search(question):
        return "SWITCH", False
    if provisional != "CONTINUE":
        return provisional, False

    inherited_by_type: dict[str, set[str]] = {"metric": set(), "business_entity": set(), "dimension": set()}
    for item in context.inherited_fields:
        entity = _entity_from_inherited(item)
        if entity and entity.type in inherited_by_type:
            inherited_by_type[entity.type].add(entity.id)
    current_by_type = _current_explicit_entity_ids(current_entities)
    for entity_type, current_ids in current_by_type.items():
        inherited_ids = inherited_by_type[entity_type]
        if current_ids and inherited_ids and current_ids.isdisjoint(inherited_ids):
            return "SWITCH", False
    return "CONTINUE", True


def _dimension_roles(
    question: str,
    mention_entities: Sequence[tuple[Mention, Entity]],
    frame: RequestFrame,
) -> dict[str, str]:
    dimensions = [
        (entity.id, mention.text, mention.span.start, mention.span.end)
        for mention, entity in mention_entities
        if entity.type in {"dimension", "business_entity"}
    ]
    metric_spans = [
        (mention.span.start, mention.span.end)
        for mention, entity in mention_entities
        if entity.type == "metric"
    ]
    intent = "retrieve_metric_definition" if frame.request_mode == "EXPLAIN" else "analyze_business_data"
    try:
        assignments = classify_dimension_mentions(
            question,
            dimensions,
            intent=intent,
            metric_spans=metric_spans,
        )
    except Exception:  # noqa: BLE001 - local role heuristics below remain available
        assignments = []
    roles: dict[str, str] = {}
    for assignment in assignments:
        role = {
            "grouping": "segment_basis",
            "filter": "filter",
            "comparison": "comparison_target",
            "reference": "reference",
        }.get(assignment.role)
        if role:
            roles[assignment.dimension] = role
    return roles


def _has_prefix(question: str, mention: Mention, pattern: re.Pattern[str]) -> bool:
    left = question[max(0, mention.span.start - 24) : mention.span.start]
    left = re.split(r"[，,。.!！?？；;\n]", left)[-1]
    return bool(pattern.search(left))


def _classify_roles(
    question: str,
    frame: RequestFrame,
    entities: Sequence[Entity],
    mention_entities: Sequence[tuple[Mention, Entity]],
) -> tuple[Role, ...]:
    dimension_role_map = _dimension_roles(question, mention_entities, frame)
    operation_ids = {operation.id for operation in frame.operations}
    is_segment = "segment" in operation_ids or any(
        operation.id == "rfm"
        and operation.polarity == "POSITIVE"
        and frame.modality == "FACTUAL"
        for operation in frame.operations
    )
    roles: dict[tuple[str, str], Role] = {}

    for mention, entity in mention_entities:
        role: str | None = None
        if _has_prefix(question, mention, _FILTER_PREFIX):
            role = "filter"
        elif _has_prefix(question, mention, _REFERENCE_PREFIX):
            role = "reference"
        elif entity.id in dimension_role_map:
            role = dimension_role_map[entity.id]
        elif is_segment and entity.type == "semantic_factor":
            role = "segment_basis"
        elif entity.type == "time":
            role = "filter"
        elif entity.type == "metric":
            role = "analysis_subject"
        elif entity.type == "business_entity" and is_segment:
            role = "analysis_subject"
        elif entity.type == "business_entity" and any(
            operation.id in {"stat", "metric_query", "aggregate", "trend", "compare", "filter"}
            for operation in frame.operations
        ):
            role = "analysis_subject"
        elif entity.type == "semantic_factor":
            role = "segment_basis" if is_segment else None

        if role:
            roles[(entity.id, role)] = Role(entity=entity.id, role=role)

    # In a phrase such as “按客户统计订单量”, dimension_intent sees `customer`
    # as an explicit grouping axis. If role assignment is unresolved, the
    # direct lexical prefix remains authoritative.
    for mention, entity in mention_entities:
        if entity.type in {"business_entity", "dimension"} and _has_prefix(question, mention, _GROUP_PREFIX):
            roles[(entity.id, "segment_basis")] = Role(entity=entity.id, role="segment_basis")

    return tuple(roles.values())


def _extract_relations(roles: Sequence[Role], operations: Sequence[Operation]) -> tuple:
    by_role: dict[str, list[str]] = {}
    for role in roles:
        by_role.setdefault(role.role, []).append(role.entity)
    subjects = by_role.get("analysis_subject", [])
    if not subjects:
        subjects = [entity for role in roles if role.role == "segment_basis" for entity in [role.entity]][:1]
    subject = subjects[0] if subjects else None
    relations = []
    if subject:
        for entity_id in by_role.get("segment_basis", []):
            if entity_id != subject:
                relation_name = "AGGREGATE_BY" if any(item.id == "aggregate" for item in operations) else "GROUP_BY"
                relations.append(make_relation(subject, relation_name, entity_id))
        for entity_id in by_role.get("filter", []):
            if entity_id != subject:
                relations.append(make_relation(subject, "FILTER_BY", entity_id))
        for entity_id in by_role.get("comparison_target", []):
            if entity_id != subject:
                relations.append(make_relation(subject, "COMPARE_WITH", entity_id))
        for entity_id in by_role.get("reference", []):
            if entity_id != subject:
                relations.append(make_relation(subject, "REFERS_TO", entity_id))
    return unique_relations(relations)


def _semantic_status(
    question: str,
    frame: RequestFrame,
    resolutions: Sequence[EntityResolution],
    entities: Sequence[Entity],
    roles: Sequence[Role],
    relations: Sequence,
    integrity_issues: Sequence[ReferenceIntegrityIssue],
    dialogue_act: str,
    inherited_fields: Sequence[InheritedField],
) -> SemanticStatus:
    if integrity_issues:
        return "AMBIGUOUS" if any(item.code != "missing_role_entity" for item in integrity_issues) else "MISSING"
    if dialogue_act == "AMBIGUOUS":
        return "AMBIGUOUS"
    if any(item.ambiguous for item in resolutions):
        return "AMBIGUOUS"
    if any(item.errors for item in resolutions):
        if entities:
            return "AMBIGUOUS"
    operation_ids = {item.id for item in frame.operations}
    if not entities and not frame.operations:
        return "MISSING"
    if "segment" in operation_ids and not any(item.role == "segment_basis" for item in roles):
        return "MISSING"
    if "metric_query" in operation_ids and not any(item.type == "metric" for item in entities):
        return "MISSING"
    if "CONTINUE" == dialogue_act and not inherited_fields and not entities:
        return "MISSING"
    if not frame.operations and frame.request_mode != "EXPLAIN" and not any(
        item.type in {"metric", "semantic_factor"} for item in entities
    ):
        return "UNSUPPORTED"
    return "CONFIRMED"


def _validated_inherited_fields(
    context: DialogueContextResolution | None, accepted: bool
) -> tuple[InheritedField, ...]:
    if not context or not accepted:
        return ()
    validated: list[InheritedField] = []
    for item in context.inherited_fields:
        if item.field in {"operation", "operations"}:
            if _operation_payload(item.value) is None:
                continue
        elif _entity_from_inherited(item) is None:
            continue
        validated.append(item.model_copy(update={"validated": True}))
    return tuple(validated)


def parse_semantics(
    question: str,
    *,
    dialogue_context: DialogueContextResolution | None = None,
) -> SemanticContract:
    """Parse a turn into a frozen semantic contract with no route/template data."""

    raw_question = str(question or "")
    raw_entity_mentions = detect_entity_mentions(raw_question)
    entity_resolutions = tuple(
        resolve_entity(mention, question=raw_question) for mention in raw_entity_mentions
    )
    current_entities = tuple(
        resolution.selected for resolution in entity_resolutions if resolution.selected is not None
    )
    provisional = dialogue_context.provisional_dialogue_act if dialogue_context else "NEW"
    validated_act, accept_inheritance = _validate_dialogue_act(
        raw_question, dialogue_context, current_entities
    )
    accepted_fields = _validated_inherited_fields(dialogue_context, accept_inheritance)
    rejected_inherited = bool(
        dialogue_context
        and accept_inheritance
        and len(accepted_fields) < len(dialogue_context.inherited_fields)
    )

    inherited_entities: list[Entity] = []
    inherited_operations: list[Operation] = []
    unresolved_inherited = rejected_inherited
    for item in accepted_fields:
        if item.field in {"operation", "operations"}:
            operation = _operation_payload(item.value)
            if operation:
                inherited_operations.append(operation)
            continue
        entity = _entity_from_inherited(item)
        if entity is None:
            unresolved_inherited = True
        else:
            inherited_entities.append(entity)

    frame = build_request_frame(
        raw_question,
        provisional_dialogue_act=provisional,
        validated_dialogue_act=validated_act,
        inherited_operations=inherited_operations if accept_inheritance else (),
    )
    # The request-frame parser sees only the current turn. A context resolver's
    # route fields are deliberately never read here.
    operations = tuple(frame.operations)

    mention_operations = _operation_mentions(raw_question)
    mentions = _merge_mentions((*raw_entity_mentions, *mention_operations))
    entities_by_id: dict[str, Entity] = {}
    for entity in (*current_entities, *inherited_entities):
        previous = entities_by_id.get(entity.id)
        if previous is None or float(entity.confidence or 0.0) > float(previous.confidence or 0.0):
            entities_by_id[entity.id] = entity
    for resolution in entity_resolutions:
        if resolution.ambiguous:
            for candidate in resolution.candidates:
                entities_by_id.setdefault(candidate.id, candidate)
    entities = tuple(entities_by_id.values())

    mention_entities: list[tuple[Mention, Entity]] = []
    for mention, resolution in zip(raw_entity_mentions, entity_resolutions):
        if resolution.selected:
            mention_entities.append((mention, resolution.selected))
    mention_entities.extend(
        (Mention(text=item.id, span=Span(text=item.id, start=0, end=len(item.id)), candidate_types=(item.type,)), item)
        for item in inherited_entities
        if item.id not in {entity.id for entity in current_entities}
    )

    roles = _classify_roles(raw_question, frame, entities, mention_entities)
    relations = _extract_relations(roles, operations)
    integrity_issues = reference_integrity_issues(
        entities=entities, roles=roles, relations=relations
    )
    status = _semantic_status(
        raw_question,
        frame,
        entity_resolutions,
        entities,
        roles,
        relations,
        integrity_issues,
        validated_act,
        accepted_fields,
    )
    if unresolved_inherited:
        status = "AMBIGUOUS"

    current_source_ids = sorted(entity.id for entity in current_entities)
    inherited_records = _inherited_records(accepted_fields, entities, operations)
    if unresolved_inherited:
        status = "AMBIGUOUS"
    current_operation_ids = {
        operation.id for operation in _candidate_operation_ids(operations, raw_question)
    }
    context_sources = {
        "current_turn": sorted(set(current_source_ids) | current_operation_ids),
        "inherited": inherited_records,
    }
    return SemanticContract(
        mentions=mentions,
        entities=entities,
        operations=operations,
        request_mode=frame.request_mode,
        dialogue_act_provisional=provisional,
        dialogue_act=validated_act,
        modality=frame.modality,
        roles=roles,
        relations=relations,
        semantic_status=status,
        context_sources=context_sources,
        frozen=True,
    )


def _candidate_operation_ids(
    operations: Sequence[Operation], question: str
) -> tuple[Operation, ...]:
    """Return current-turn operation IDs without exposing route metadata."""

    current_ids = {item[2] for item in _candidate_operation_specs(question)}
    return tuple(item for item in operations if item.id in current_ids)


def _candidate_operation_specs(question: str) -> tuple[tuple[int, int, str, str, str, str], ...]:
    # Kept behind a local import to avoid exposing request-frame internals as a
    # second parser API. The import is deterministic and side-effect free.
    from app.context.request_frame import _candidate_operations

    return tuple(_candidate_operations(question))

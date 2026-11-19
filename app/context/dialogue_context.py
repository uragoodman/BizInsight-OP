"""Resolve cross-turn semantic context without deciding a production route."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DialogueAct = Literal["NEW", "CONTINUE", "SWITCH", "AMBIGUOUS"]


class DialogueContextModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InheritedField(DialogueContextModel):
    field: str
    value: Any
    source_turn: str
    reason: str
    validated: bool = False


class CurrentField(DialogueContextModel):
    field: str
    value: Any


class DialogueContextResolution(DialogueContextModel):
    provisional_dialogue_act: DialogueAct
    inherited_fields: tuple[InheritedField, ...]
    current_turn_fields: tuple[CurrentField, ...]
    routing_context: dict[str, Any]


_SWITCH = re.compile(
    r"不做这个了|不要这个|换成|改成|改为|改查|改看|转而|另一个|重新查|重新看|"
    r"先不做|不继续|(?:不要|不做|别再)\s*(?:继续|再)?\s*(?:做|进行|执行)?\s*(?:R\s*F\s*M|分层|这个)|\bswitch to\b",
    re.I,
)
_MULTI_REFERENCE = re.compile(r"它们|他们|这些|那些|这两个|上述两者|它们俩|\bthose\b|\bthem\b", re.I)
_CONTINUATION = re.compile(
    r"^\s*(?:那|那么|然后|继续|再|还有)|这个|那个|它|其|上个月|上月|上周|这周|"
    r"上季度|上季度|去年|当天|当日|\bthat\b|\bit\b",
    re.I,
)
_TIME_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("previous_month", re.compile(r"上个月|上月|last month", re.I)),
    ("current_month", re.compile(r"本月|这个月|this month", re.I)),
    ("previous_week", re.compile(r"上周|上星期|last week", re.I)),
    ("current_week", re.compile(r"本周|这周|this week", re.I)),
    ("previous_year", re.compile(r"去年|last year", re.I)),
    ("today", re.compile(r"今天|当日|today", re.I)),
    ("yesterday", re.compile(r"昨天|yesterday", re.I)),
)


def _turn_id(turn: Mapping[str, Any], fallback: str) -> str:
    for key in ("turn_id", "source_turn", "id"):
        value = turn.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return fallback


def _is_completed(turn: Mapping[str, Any]) -> bool:
    if turn.get("completed") is True:
        return True
    status = str(turn.get("status") or turn.get("turn_status") or "").casefold()
    return status in {"completed", "complete", "done", "succeeded"}


def _semantic_fields(turn: Mapping[str, Any]) -> list[tuple[str, Any]]:
    fields: list[tuple[str, Any]] = []
    for container_key in (
        "semantic_fields", "fields", "context_slots", "semantic_context", "semantic_contract"
    ):
        raw = turn.get(container_key)
        if not isinstance(raw, Mapping):
            continue
        for name, value in raw.items():
            if container_key == "semantic_contract" and name not in {"entities", "operations"}:
                continue
            if isinstance(value, Mapping) and "value" in value:
                value = value["value"]
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                field_name = "operation" if name == "operations" else str(name)
                for item in value:
                    if field_name == "entities" and isinstance(item, Mapping):
                        entity_type = item.get("type")
                        field_name_for_entity = (
                            str(entity_type)
                            if entity_type in {
                                "metric", "dimension", "business_entity", "semantic_factor", "time", "scope"
                            }
                            else "entity"
                        )
                        fields.append((field_name_for_entity, item))
                    else:
                        fields.append((field_name, item))
            else:
                fields.append((str(name), value))

    entities = turn.get("entities")
    if isinstance(entities, Sequence) and not isinstance(entities, (str, bytes)):
        for value in entities:
            if isinstance(value, Mapping):
                entity_id = value.get("id") or value.get("entity") or value.get("value")
                entity_type = value.get("type")
                if entity_id:
                    field = (
                        str(entity_type)
                        if entity_type in {
                            "metric", "dimension", "business_entity", "semantic_factor", "time", "scope"
                        }
                        else "entity"
                    )
                    fields.append((field, entity_id))
            elif isinstance(value, str) and value.strip():
                fields.append(("entity", value.strip()))

    operations = turn.get("operations")
    if isinstance(operations, Sequence) and not isinstance(operations, (str, bytes)):
        for operation in operations:
            if isinstance(operation, Mapping) and operation.get("id"):
                fields.append(("operation", dict(operation)))
            elif isinstance(operation, str):
                fields.append(("operation", operation))

    deduplicated: list[tuple[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for field, value in fields:
        signature = (field, repr(value))
        if signature not in seen:
            seen.add(signature)
            deduplicated.append((field, value))
    return deduplicated


def _current_fields(question: str) -> tuple[CurrentField, ...]:
    result: list[CurrentField] = []
    for canonical, pattern in _TIME_PATTERNS:
        match = pattern.search(question)
        if match:
            result.append(CurrentField(field="time_scope", value=canonical))
    if re.search(r"客户|顾客|\bcustomer\b", question, flags=re.I):
        result.append(CurrentField(field="business_entity", value="customer"))
    if re.search(r"订单量|订单数|order_count", question, flags=re.I):
        result.append(CurrentField(field="metric", value="order_count"))
    if re.search(r"GMV|成交总额|交易额|成交额", question, flags=re.I):
        result.append(CurrentField(field="metric", value="gmv"))
    for match in re.finditer(r"(?<![A-Za-z0-9_])R\s*F\s*M(?![A-Za-z0-9_])", question, flags=re.I):
        result.append(CurrentField(field="operation", value="rfm"))
    return tuple(result)


def resolve_dialogue_context(
    question: str,
    *,
    history: Sequence[dict] = (),
    routing_context: dict | None = None,
) -> DialogueContextResolution:
    """Return tentative inheritance candidates and isolated routing metadata."""

    current_fields = _current_fields(str(question or ""))
    safe_routing_context: dict[str, Any] = {}
    provided_route = dict(routing_context or {})

    history_turns = [item for item in history if isinstance(item, Mapping)]
    latest: Mapping[str, Any] | None = history_turns[-1] if history_turns else None
    semantic_turn: Mapping[str, Any] | None = next(
        (item for item in reversed(history_turns) if _semantic_fields(item)), latest
    )
    source_turn = ""
    fields: list[tuple[str, Any]] = []
    completed = False
    if semantic_turn is not None:
        source_turn = _turn_id(semantic_turn, f"turn-{len(history)}")
        fields = _semantic_fields(semantic_turn)
    if latest is not None:
        completed = _is_completed(latest) or (
            semantic_turn is not None and _is_completed(semantic_turn)
        )

    history_routing = latest.get("routing_context") if latest is not None else None
    if (not isinstance(history_routing, Mapping) or not history_routing) and semantic_turn is not None:
        history_routing = semantic_turn.get("routing_context")
    if not isinstance(history_routing, Mapping):
        history_routing = {}
    route = provided_route.get("previous_route") or history_routing.get("previous_route")
    if route is None and latest is not None:
        route = latest.get("previous_route") or latest.get("route")
    route_source = (
        provided_route.get("source_turn")
        or history_routing.get("source_turn")
        or source_turn
        or None
    )
    continuation_status = (
        provided_route.get("continuation_status") or history_routing.get("continuation_status")
    )
    completed = completed or str(continuation_status or "").casefold() in {
        "completed", "complete", "done", "succeeded"
    }

    # Only routing_context is allowed to retain route identifiers. The history
    # route is copied here and never enters inherited_fields/current_turn_fields.
    safe_routing_context["previous_route"] = route
    safe_routing_context["source_turn"] = str(route_source) if route_source is not None else None

    explicit_switch = bool(_SWITCH.search(question or ""))
    continuation_cue = bool(_CONTINUATION.search(question or ""))
    current_ids_by_type: dict[str, set[str]] = {"metric": set(), "business_entity": set()}
    for item in current_fields:
        if item.field in current_ids_by_type:
            current_ids_by_type[item.field].add(str(item.value))
    historical_ids_by_field: dict[str, set[str]] = {}
    possible_referents: set[str] = set()
    for field, value in fields:
        if field in {"operation", "operations"}:
            continue
        raw_value = value.get("id") or value.get("entity") or value.get("value") if isinstance(value, Mapping) else value
        if raw_value is None:
            continue
        normalized = str(raw_value)
        historical_ids_by_field.setdefault(field, set()).add(normalized)
        if field in {
            "reference", "referent", "reference_candidate", "reference_candidates", "ambiguous_referent"
        }:
            possible_referents.add(normalized)
    historical_metric_ids = set(historical_ids_by_field.get("metric", set()))
    historical_business_ids = (
        set(historical_ids_by_field.get("business_entity", set()))
        | set(historical_ids_by_field.get("analysis_subject", set()))
    )
    historical_business_ids -= historical_metric_ids
    explicit_entity_conflict = any(
        current_ids_by_type[entity_type]
        and historical_ids
        and current_ids_by_type[entity_type].isdisjoint(historical_ids)
        for entity_type, historical_ids in (
            ("metric", historical_metric_ids),
            ("business_entity", historical_business_ids),
        )
    )
    # Multiple history facts (for example, customer + GMV + segment) are not
    # automatically ambiguous. Only explicit anaphora candidates, or multiple
    # values competing for the same singular field, block inheritance.
    singular_fields = {
        "metric", "business_entity", "analysis_subject", "reference", "referent",
        "reference_candidate", "reference_candidates", "ambiguous_referent",
    }
    multiple_referents = bool(_MULTI_REFERENCE.search(question or "")) or len(possible_referents) > 1 or any(
        field in singular_fields and len(values) > 1
        for field, values in historical_ids_by_field.items()
    )

    if explicit_switch or explicit_entity_conflict:
        provisional: DialogueAct = "SWITCH"
        continuation_status = "SWITCH"
    elif not latest or not fields:
        provisional = "AMBIGUOUS" if continuation_cue else "NEW"
        continuation_status = continuation_status or provisional
    elif completed and continuation_cue:
        provisional = "AMBIGUOUS"
        continuation_status = "AMBIGUOUS"
    elif completed:
        provisional = "NEW"
        continuation_status = "NEW"
    elif multiple_referents and continuation_cue:
        provisional = "AMBIGUOUS"
        continuation_status = "AMBIGUOUS"
    elif continuation_cue:
        provisional = "CONTINUE"
        continuation_status = "CONTINUE"
    else:
        provisional = "NEW"
        continuation_status = "NEW"

    inherited: tuple[InheritedField, ...] = ()
    if provisional == "CONTINUE":
        inherited = tuple(
            InheritedField(
                field=field,
                value=value,
                source_turn=source_turn,
                reason="anaphora_resolution" if continuation_cue else "same_task_context",
            )
            for field, value in fields
        )

    safe_routing_context["continuation_status"] = continuation_status or provisional
    return DialogueContextResolution(
        provisional_dialogue_act=provisional,
        inherited_fields=inherited,
        current_turn_fields=current_fields,
        routing_context=safe_routing_context,
    )

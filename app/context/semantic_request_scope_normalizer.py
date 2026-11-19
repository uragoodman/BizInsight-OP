"""Narrow deterministic normalization for cross-clause operation negation."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


_CURRENT_TURN_GROUNDING_SOURCES = frozenset({
    "exact_alias",
    "catalog_synonym",
    "llm_inferred",
})


def _span_pair(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, Mapping):
        return None
    start = value.get("start")
    end = value.get("end")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        return None
    return start, end


def _sequence(value: Any) -> list[Any] | None:
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def _items_by_id(items: list[Any]) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        item_id = item.get("id")
        if isinstance(item_id, str) and item_id:
            result.setdefault(item_id, []).append(item)
    return result


def _unique_by_id(
    items_by_id: Mapping[str, list[Mapping[str, Any]]], item_id: Any,
) -> Mapping[str, Any] | None:
    if not isinstance(item_id, str) or not item_id:
        return None
    matches = items_by_id.get(item_id, [])
    return matches[0] if len(matches) == 1 else None


def _span_inside(
    span: tuple[int, int] | None,
    container: tuple[int, int] | None,
    source_length: int,
) -> bool:
    if span is None or container is None:
        return False
    start, end = span
    outer_start, outer_end = container
    return end <= source_length and outer_start <= start and end <= outer_end


def _valid_clause_partition(
    clauses: list[Any], source_length: int,
) -> tuple[dict[str, list[Mapping[str, Any]]], dict[str, tuple[int, int]]] | None:
    clauses_by_id = _items_by_id(clauses)
    if (
        not clauses
        or len(clauses_by_id) != len(clauses)
        or any(len(matches) != 1 for matches in clauses_by_id.values())
    ):
        return None

    spans_by_id: dict[str, tuple[int, int]] = {}
    ordered_spans: list[tuple[int, int]] = []
    for clause_id, matches in clauses_by_id.items():
        clause = matches[0]
        span = _span_pair(clause.get("span"))
        if span is None or span[1] > source_length:
            return None
        spans_by_id[clause_id] = span
        ordered_spans.append(span)

    cursor = 0
    for start, end in sorted(ordered_spans):
        if start != cursor:
            return None
        cursor = end
    if cursor != source_length:
        return None
    return clauses_by_id, spans_by_id


def normalize_cross_clause_negative_request_scope(
    candidate: dict[str, Any], source_text: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert only an explicit, unique cross-clause negative target to REQUEST.

    This function does not infer polarity, operation kind, or target concepts.
    It edits only ``clause_scope``, ``source_clause_id``, and
    ``source_turn_id`` when a current-turn target mention and negation cue are
    uniquely grounded in a valid clause partition. It is intended to run after
    coordinate normalization and before Validator. Unclear candidates are
    returned unchanged for the schema and Validator to handle fail-closed.
    """

    if not isinstance(candidate, dict):
        raise TypeError("candidate must be a dict")
    if not isinstance(source_text, str):
        raise TypeError("source_text must be a string")

    normalized = deepcopy(candidate)
    operations = _sequence(normalized.get("operations"))
    stats: dict[str, Any] = {
        "enabled": True,
        "operations_seen": len(operations) if operations is not None else 0,
        "negative_same_clause_examined": 0,
        "request_scope_rewrites": 0,
    }
    clauses = _sequence(normalized.get("clauses"))
    mentions = _sequence(normalized.get("mentions"))
    turn_id = normalized.get("turn_id")
    if (
        operations is None
        or clauses is None
        or mentions is None
        or not isinstance(turn_id, str)
        or not turn_id
        or not source_text
    ):
        return normalized, stats

    partition = _valid_clause_partition(clauses, len(source_text))
    if partition is None:
        return normalized, stats
    clauses_by_id, clause_spans = partition
    mentions_by_id = _items_by_id(mentions)

    for operation in operations:
        if not isinstance(operation, dict):
            continue
        if (
            operation.get("polarity") != "NEGATIVE"
            or operation.get("clause_scope") != "SAME_CLAUSE"
        ):
            continue
        stats["negative_same_clause_examined"] += 1

        operation_clause_id = operation.get("clause_id")
        if not isinstance(operation_clause_id, str) or not operation_clause_id:
            continue
        operation_clause = _unique_by_id(clauses_by_id, operation_clause_id)
        operation_clause_span = clause_spans.get(operation_clause_id)
        operation_span = _span_pair(operation.get("span"))
        cue_span = _span_pair(operation.get("negation_cue_span"))
        if (
            operation_clause is None
            or operation_clause_span is None
            or not _span_inside(operation_span, operation_clause_span, len(source_text))
            or not _span_inside(cue_span, operation_clause_span, len(source_text))
            or operation.get("source_clause_id") is not None
            or operation.get("source_turn_id") is not None
        ):
            continue

        raw_target_ids = _sequence(operation.get("target_mention_ids"))
        raw_target_concepts = _sequence(operation.get("target_concept_ids"))
        if (
            raw_target_ids is None
            or len(raw_target_ids) != 1
            or not isinstance(raw_target_ids[0], str)
            or raw_target_concepts is None
            or len(raw_target_concepts) != 1
            or not isinstance(raw_target_concepts[0], str)
            or not raw_target_concepts[0]
        ):
            continue

        mention = _unique_by_id(mentions_by_id, raw_target_ids[0])
        if mention is None:
            continue
        mention_clause_id = mention.get("clause_id")
        if not isinstance(mention_clause_id, str) or not mention_clause_id:
            continue
        mention_clause = _unique_by_id(clauses_by_id, mention_clause_id)
        mention_clause_span = clause_spans.get(mention_clause_id)
        mention_span = _span_pair(mention.get("span"))
        surface = mention.get("surface")
        concept_id = mention.get("concept_id")
        if (
            mention_clause is None
            or mention_clause_span is None
            or mention_clause_id == operation_clause_id
            or not _span_inside(mention_span, mention_clause_span, len(source_text))
            or not isinstance(surface, str)
            or source_text[mention_span[0] : mention_span[1]] != surface
            or mention.get("source_turn_id") != turn_id
            or mention.get("grounding_source") not in _CURRENT_TURN_GROUNDING_SOURCES
            or not isinstance(concept_id, str)
            or not concept_id
            or raw_target_concepts[0] != concept_id
        ):
            continue

        mentioned_arguments = operation.get("mentioned_arguments", {})
        argument_negations = operation.get("argument_negations", [])
        if (
            not isinstance(mentioned_arguments, Mapping)
            or mentioned_arguments
            or not isinstance(argument_negations, (list, tuple))
            or argument_negations
        ):
            continue

        operation["clause_scope"] = "REQUEST"
        operation["source_clause_id"] = operation_clause_id
        operation["source_turn_id"] = turn_id
        stats["request_scope_rewrites"] += 1

    return normalized, stats

"""Deterministically repair Candidate references and coordinate metadata.

Repairs require either an exact wire reference or a unique same-clause mention.
Unresolved and ambiguous values remain untouched so validation can fail closed.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from app.context.semantic_catalog_v2 import DEFAULT_SEMANTIC_CATALOG, SemanticCatalog


_CLAUSE_DELIMITERS = frozenset("，,。；;！!？?\n")


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _span_pair(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, Mapping):
        return None
    start = _as_int(value.get("start"))
    end = _as_int(value.get("end"))
    if start is None or end is None:
        return None
    return start, end


def _span_dict(start: int, end: int) -> dict[str, int]:
    return {"start": start, "end": end}


def _inside_text(span: Any, source_text: str) -> bool:
    pair = _span_pair(span)
    if pair is None:
        return False
    start, end = pair
    return 0 <= start < end <= len(source_text)


def _clause_span(clause: Mapping[str, Any] | None) -> tuple[int, int] | None:
    if clause is None:
        return None
    return _span_pair(clause.get("span"))


def _fallback_clause_spans(source_text: str) -> list[tuple[int, int]]:
    if not source_text:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    for index, char in enumerate(source_text):
        if char in _CLAUSE_DELIMITERS:
            spans.append((start, index + 1))
            start = index + 1
    if start < len(source_text):
        spans.append((start, len(source_text)))
    return spans or [(0, len(source_text))]


def _map_clause_references(value: Any, clause_id_map: Mapping[str, str]) -> None:
    if isinstance(value, dict):
        for key, child in tuple(value.items()):
            if key in {"clause_id", "source_clause_id"} and isinstance(child, str):
                value[key] = clause_id_map.get(child, child)
            else:
                _map_clause_references(child, clause_id_map)
    elif isinstance(value, list):
        for child in value:
            _map_clause_references(child, clause_id_map)


def _known_mention_ids_in_reference(value: Any, known_mention_ids: set[str]) -> set[str]:
    matches: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, str):
            if item in known_mention_ids:
                matches.add(item)
        elif isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return matches


def _operation_is_ungrounded(
    operation: Mapping[str, Any], clause_ids_with_mentions: set[Any],
) -> bool:
    """Whether a positive operation has no referential anchor at all.

    A positive operation with an implicit target must carry a grounded
    argument; when its clause contains no mention and its argument map is
    empty, the candidate is a stray constraint (for example “只描述统计变化”)
    that cannot execute or be validated.  Dropping it is safer than borrowing
    a target from another clause.
    """

    if str(operation.get("polarity") or "POSITIVE") == "NEGATIVE":
        return False
    target_mentions = operation.get("target_mention_ids")
    if isinstance(target_mentions, (list, tuple)) and target_mentions:
        return False
    target_concepts = operation.get("target_concept_ids")
    if not isinstance(target_concepts, (list, tuple)) or not target_concepts:
        return False
    arguments = operation.get("mentioned_arguments")
    if isinstance(arguments, Mapping) and any(
        isinstance(values, (list, tuple)) and values
        for values in arguments.values()
    ):
        return False
    return operation.get("clause_id") not in clause_ids_with_mentions


def normalize_pre_schema_references(candidate: dict) -> tuple[dict, dict[str, Any]]:
    """Normalize only unambiguous raw reference shapes before Pydantic validation.

    This repairs role references that use the RelationEndpoint wire shape and
    links an implicit operation target only when each target concept has one
    current-turn mention in the same clause. Ambiguous or malformed values are
    left unchanged for schema validation to reject.
    """

    if not isinstance(candidate, dict):
        raise TypeError("candidate must be a dict")
    normalized = deepcopy(candidate)
    stats = {
        "enabled": True,
        "role_references_normalized": 0,
        "role_references_ambiguous": 0,
        "target_mentions_attached": 0,
        "target_concepts_derived": 0,
        "target_reference_ambiguous": 0,
        "ungrounded_operations_dropped": 0,
    }
    mentions = normalized.get("mentions")
    if not isinstance(mentions, list):
        mentions = []
    turn_id = normalized.get("turn_id")
    mentions_by_id = {
        mention.get("id"): mention
        for mention in mentions
        if isinstance(mention, Mapping) and isinstance(mention.get("id"), str)
    }

    roles = normalized.get("roles")
    if isinstance(roles, list):
        for role in roles:
            if not isinstance(role, dict):
                continue
            reference = role.get("mention_id")
            if isinstance(reference, str):
                continue
            role_clause_id = role.get("clause_id")
            role_concept_id = role.get("concept_id")
            if not isinstance(role_clause_id, str) or not isinstance(role_concept_id, str):
                continue
            matching_ids = _known_mention_ids_in_reference(reference, set(mentions_by_id))
            if len(matching_ids) == 1:
                mention_id = next(iter(matching_ids))
                mention = mentions_by_id[mention_id]
                if (
                    mention.get("clause_id") == role_clause_id
                    and mention.get("concept_id") == role_concept_id
                ):
                    role["mention_id"] = mention_id
                    stats["role_references_normalized"] += 1
            elif len(matching_ids) > 1:
                stats["role_references_ambiguous"] += 1

    operations = normalized.get("operations")
    if not isinstance(operations, list):
        return normalized, stats
    clause_ids_with_mentions = {
        mention.get("clause_id")
        for mention in mentions
        if isinstance(mention, Mapping)
    }
    processed_operations: list[dict[str, Any]] = []
    ungrounded_operations: list[dict[str, Any]] = []
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        processed_operations.append(operation)
        raw_target_ids = operation.get("target_mention_ids", [])
        target_concepts = operation.get("target_concept_ids")
        clause_id = operation.get("clause_id")
        if (
            target_concepts in ([], ())
            and isinstance(raw_target_ids, (list, tuple))
            and raw_target_ids
            and all(isinstance(value, str) for value in raw_target_ids)
            and isinstance(clause_id, str)
            and isinstance(turn_id, str)
        ):
            referenced_mentions = [mentions_by_id.get(mention_id) for mention_id in raw_target_ids]
            if all(
                isinstance(mention, Mapping)
                and mention.get("clause_id") == clause_id
                and mention.get("source_turn_id") == turn_id
                and mention.get("grounding_source") != "inherited"
                and isinstance(mention.get("concept_id"), str)
                and mention.get("concept_id")
                for mention in referenced_mentions
            ):
                derived_concepts = list(dict.fromkeys(
                    mention["concept_id"] for mention in referenced_mentions
                ))
                operation["target_concept_ids"] = derived_concepts
                target_concepts = derived_concepts
                stats["target_concepts_derived"] += len(derived_concepts)

        if raw_target_ids not in ([], ()):
            continue
        if (
            not isinstance(target_concepts, (list, tuple))
            or not target_concepts
            or not all(isinstance(value, str) and value for value in target_concepts)
            or len(set(target_concepts)) != len(target_concepts)
            or not isinstance(clause_id, str)
            or not isinstance(turn_id, str)
        ):
            continue

        target_ids: list[str] = []
        ambiguous = False
        for concept_id in target_concepts:
            matches = [
                mention
                for mention in mentions
                if isinstance(mention, Mapping)
                and mention.get("concept_id") == concept_id
                and mention.get("clause_id") == clause_id
                and mention.get("source_turn_id") == turn_id
                and mention.get("grounding_source") != "inherited"
                and isinstance(mention.get("id"), str)
            ]
            if len(matches) != 1:
                ambiguous = len(matches) > 1
                target_ids = []
                break
            target_ids.append(matches[0]["id"])
        if target_ids:
            operation["target_mention_ids"] = target_ids
            stats["target_mentions_attached"] += len(target_ids)
        elif ambiguous:
            stats["target_reference_ambiguous"] += 1
        if _operation_is_ungrounded(operation, clause_ids_with_mentions):
            ungrounded_operations.append(operation)

    ungrounded_ids = {id(item) for item in ungrounded_operations}
    kept_operations = [
        operation for operation in processed_operations
        if id(operation) not in ungrounded_ids
    ]
    if ungrounded_operations and kept_operations:
        normalized["operations"] = kept_operations
        stats["ungrounded_operations_dropped"] = len(ungrounded_operations)
    return normalized, stats


def _normalize_clauses(
    clauses: list[dict[str, Any]], source_text: str, stats: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    text_length = len(source_text)
    if not clauses:
        normalized = [{"id": "c1", "span": _span_dict(0, text_length)}]
        stats["clauses_fixed"] += 1
        stats["clauses_fallback_split"] += 1
        return normalized, {}

    ordered = sorted(
        enumerate(clauses),
        key=lambda item: (
            _span_pair(item[1].get("span"))[0]
            if _span_pair(item[1].get("span")) is not None else -1,
            item[0],
        ),
    )
    original_ids = [item.get("id") for item in clauses]
    ids_unique = all(isinstance(value, str) for value in original_ids) and len(set(original_ids)) == len(original_ids)
    order_correct = [item[0] for item in ordered] == list(range(len(clauses)))

    invalid_raw_span = any(
        (pair := _span_pair(clause.get("span"))) is None or pair[1] <= pair[0]
        for clause in clauses
    )
    interior_starts = [
        min(max(_span_pair(clause.get("span"))[0], 0), text_length)
        for _index, clause in ordered[1:]
        if _span_pair(clause.get("span")) is not None
    ]
    ordered_pairs = [_span_pair(clause.get("span")) for _index, clause in ordered]
    overlapping_or_reversed = any(
        previous is not None and current is not None
        and (current[0] < previous[1] or current[1] < previous[1])
        for previous, current in zip(ordered_pairs, ordered_pairs[1:])
    )
    boundaries = [0, *interior_starts, text_length]
    duplicate_or_empty = (
        len(boundaries) != len(clauses) + 1
        or any(boundaries[index] >= boundaries[index + 1] for index in range(len(boundaries) - 1))
    )
    fallback = invalid_raw_span or duplicate_or_empty or overlapping_or_reversed

    if fallback:
        normalized_spans = _fallback_clause_spans(source_text)
        stats["clauses_fallback_split"] += 1
    else:
        normalized_spans = [
            (boundaries[index], boundaries[index + 1])
            for index in range(len(clauses))
        ]

    keep_ids = ids_unique and order_correct and len(normalized_spans) == len(clauses)
    normalized: list[dict[str, Any]] = []
    for index, (span_start, span_end) in enumerate(normalized_spans):
        clause_id = ordered[index][1]["id"] if keep_ids else f"c{index + 1}"
        normalized.append({"id": clause_id, "span": _span_dict(span_start, span_end)})
        if index >= len(ordered):
            stats["clauses_fixed"] += 1
        else:
            old_span = _span_pair(ordered[index][1].get("span"))
            if old_span != (span_start, span_end):
                stats["clauses_fixed"] += 1
            if ordered[index][1].get("id") != clause_id:
                stats["clause_ids_reassigned"] += 1

    normalized_by_position = normalized
    id_map: dict[str, str] = {}
    id_counts = {value: original_ids.count(value) for value in original_ids if isinstance(value, str)}
    for _old_index, old_clause in ordered:
        old_id = old_clause.get("id")
        if not isinstance(old_id, str) or old_id in id_map or id_counts.get(old_id) != 1:
            continue
        old_pair = _span_pair(old_clause.get("span"))
        old_start = min(max(old_pair[0], 0), text_length) if old_pair is not None else 0
        nearest = min(
            enumerate(normalized_by_position),
            key=lambda item: (abs(item[1]["span"]["start"] - old_start), item[0]),
        )[1]
        id_map[old_id] = nearest["id"]
    return normalized, id_map


def _adjust_boundaries_for_mentions(
    clauses: list[dict[str, Any]], mentions: list[dict[str, Any]],
    source_text: str, stats: dict[str, Any],
) -> list[dict[str, Any]]:
    """Keep a resolvable mention from straddling a clause edge."""

    if len(clauses) < 2:
        return clauses
    clause_index = {str(clause["id"]): index for index, clause in enumerate(clauses)}
    exact_anchors: list[tuple[int, int, int]] = []
    for mention in mentions:
        if mention.get("grounding_source") == "inherited":
            continue
        surface = mention.get("surface")
        index = clause_index.get(str(mention.get("clause_id", "")))
        if index is None or not isinstance(surface, str) or not surface:
            continue
        pair = _span_pair(mention.get("span"))
        old_start = pair[0] if pair is not None else 0
        if pair is None or not _inside_text(mention.get("span"), source_text) or source_text[pair[0]:pair[1]] != surface:
            clause_bounds = _clause_span(clauses[index])
            local_matches = (
                _find_occurrences(source_text, surface, *clause_bounds)
                if clause_bounds is not None else []
            )
            matches = local_matches or _find_occurrences(source_text, surface)
            if not matches:
                continue
            start = min(matches, key=lambda item: (abs(item - old_start), item))
            pair = (start, start + len(surface))
        exact_anchors.append((index, pair[0], pair[1]))

    for boundary_index in range(len(clauses) - 1):
        previous_bounds = _clause_span(clauses[boundary_index])
        next_bounds = _clause_span(clauses[boundary_index + 1])
        if previous_bounds is None or next_bounds is None:
            continue
        boundary = previous_bounds[1]
        lower = max((end for owner, start, end in exact_anchors
                     if owner == boundary_index and start < boundary < end), default=previous_bounds[0] + 1)
        upper = min((start for owner, start, end in exact_anchors
                     if owner == boundary_index + 1 and start < boundary < end), default=next_bounds[1] - 1)
        if lower > upper:
            continue
        new_boundary = min(max(boundary, lower), upper)
        if new_boundary == lower and lower > boundary and new_boundary < len(source_text):
            if source_text[new_boundary] in _CLAUSE_DELIMITERS:
                new_boundary += 1
        if not previous_bounds[0] < new_boundary < next_bounds[1]:
            continue
        if new_boundary != boundary:
            clauses[boundary_index]["span"]["end"] = new_boundary
            clauses[boundary_index + 1]["span"]["start"] = new_boundary
            stats["clauses_fixed"] += 1
            stats["clause_boundaries_adjusted_for_mentions"] += 1
    return clauses


def _find_occurrences(text: str, surface: str, start: int = 0, end: int | None = None) -> list[int]:
    if not surface:
        return []
    stop = len(text) if end is None else min(end, len(text))
    cursor = max(0, start)
    matches: list[int] = []
    while cursor <= stop - len(surface):
        found = text.find(surface, cursor, stop)
        if found < 0:
            break
        matches.append(found)
        cursor = found + 1
    return matches


def _owner_clause(
    span: tuple[int, int], clauses_by_id: Mapping[str, Mapping[str, Any]],
) -> str | None:
    start, end = span
    owners = [
        clause_id
        for clause_id, clause in clauses_by_id.items()
        if (bounds := _clause_span(clause)) is not None
        and bounds[0] <= start < end <= bounds[1]
    ]
    return owners[0] if len(owners) == 1 else None


def _resolve_mention_spans(
    mentions: list[dict[str, Any]],
    clauses: list[dict[str, Any]],
    source_text: str,
    stats: dict[str, Any],
) -> list[dict[str, Any]]:
    clauses_by_id = {str(clause["id"]): clause for clause in clauses}
    for mention in mentions:
        if mention.get("grounding_source") == "inherited":
            if mention.get("span") is not None:
                mention["span"] = None
                stats["mention_spans_fixed"] += 1
            continue

        surface = mention.get("surface")
        if not isinstance(surface, str) or not surface:
            stats["mention_spans_unresolved"] += 1
            stats["unresolved_surface"].append(str(mention.get("id", "")))
            continue

        old_pair = _span_pair(mention.get("span"))
        old_start = old_pair[0] if old_pair is not None else 0
        current_clause = clauses_by_id.get(str(mention.get("clause_id", "")))
        chosen_start: int | None = None

        if _inside_text(mention.get("span"), source_text):
            start, end = old_pair  # type: ignore[misc]
            if source_text[start:end] == surface:
                chosen_start = start

        if chosen_start is None:
            clause_bounds = _clause_span(current_clause)
            local_matches = (
                _find_occurrences(source_text, surface, *clause_bounds)
                if clause_bounds is not None else []
            )
            matches = local_matches or _find_occurrences(source_text, surface)
            if matches:
                chosen_start = min(matches, key=lambda item: (abs(item - old_start), item))

        if chosen_start is None:
            stats["mention_spans_unresolved"] += 1
            stats["unresolved_surface"].append(str(mention.get("id", "")))
            continue

        chosen = (chosen_start, chosen_start + len(surface))
        if old_pair != chosen:
            mention["span"] = _span_dict(*chosen)
            stats["mention_spans_fixed"] += 1

        owner_id = _owner_clause(chosen, clauses_by_id)
        if owner_id is not None and mention.get("clause_id") != owner_id:
            mention["clause_id"] = owner_id
            stats["mention_clause_remapped"] += 1
    return mentions


def _merge_clause_boundaries_inside_mentions(
    clauses: list[dict[str, Any]],
    mentions: list[dict[str, Any]],
    source_text: str,
    stats: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Merge adjacent clauses only when an exact mention crosses their edge."""

    boundaries_to_merge: set[int] = set()
    for mention in mentions:
        if mention.get("grounding_source") == "inherited":
            continue
        pair = _span_pair(mention.get("span"))
        surface = mention.get("surface")
        if (
            pair is None
            or not _inside_text(mention.get("span"), source_text)
            or not isinstance(surface, str)
            or not surface
            or source_text[pair[0]:pair[1]] != surface
            or any(char in _CLAUSE_DELIMITERS for char in surface)
        ):
            continue
        for boundary_index, clause in enumerate(clauses[:-1]):
            boundary = _clause_span(clause)
            if boundary is not None and pair[0] < boundary[1] < pair[1]:
                boundaries_to_merge.add(boundary_index)

    if not boundaries_to_merge:
        return clauses, {}

    merged: list[dict[str, Any]] = []
    clause_id_map: dict[str, str] = {}
    for index, clause in enumerate(clauses):
        if merged and index - 1 in boundaries_to_merge:
            survivor = merged[-1]
            survivor["span"]["end"] = clause["span"]["end"]
            clause_id_map[str(clause["id"])] = str(survivor["id"])
            stats["clauses_merged"] += 1
        else:
            merged.append(deepcopy(clause))
    stats["clause_boundaries_merged"] += len(boundaries_to_merge)
    stats["clauses_fixed"] += len(boundaries_to_merge)
    return merged, clause_id_map


def _remap_rfm_method_operations(
    operations: list[dict[str, Any]],
    mentions: list[dict[str, Any]],
    remapped_mention_ids: set[str],
    stats: dict[str, Any],
) -> None:
    mentions_by_id = {
        str(mention.get("id", "")): mention
        for mention in mentions
        if isinstance(mention, Mapping)
    }
    for operation in operations:
        if (
            operation.get("kind") != "SEGMENT"
            or "CUSTOMER" not in operation.get("target_concept_ids", ())
            or operation.get("target_mention_ids")
        ):
            continue
        arguments = operation.get("mentioned_arguments")
        method_values = arguments.get("method") if isinstance(arguments, Mapping) else None
        if not isinstance(method_values, list):
            continue
        rfm_mentions = []
        for value in method_values:
            if (
                not isinstance(value, Mapping)
                or value.get("concept_id") != "RFM"
                or value.get("stance") != "ADOPTED"
            ):
                continue
            mention_id = str(value.get("mention_id", ""))
            mention = mentions_by_id.get(mention_id)
            if (
                mention is not None
                and mention.get("concept_id") == "RFM"
                and mention_id in remapped_mention_ids
            ):
                rfm_mentions.append(mention)
        target_clause_ids = {
            mention.get("clause_id")
            for mention in rfm_mentions
            if isinstance(mention.get("clause_id"), str)
        }
        if len(rfm_mentions) == 1 and len(target_clause_ids) == 1:
            target_clause_id = next(iter(target_clause_ids))
            if operation.get("clause_id") != target_clause_id:
                operation["clause_id"] = target_clause_id
                stats["operation_clause_remapped"] += 1


def _normalize_grounding(
    mentions: list[dict[str, Any]], catalog: SemanticCatalog, stats: dict[str, Any],
) -> list[dict[str, Any]]:
    for mention in mentions:
        if mention.get("grounding_source") == "inherited":
            continue
        concept = catalog.concept(str(mention.get("concept_id", "")))
        if concept is None:
            continue
        surface = mention.get("surface")
        if surface in concept.exact_aliases:
            expected = "exact_alias"
        elif surface in concept.synonyms:
            expected = "catalog_synonym"
        else:
            expected = "llm_inferred"
        if mention.get("grounding_source") != expected:
            mention["grounding_source"] = expected
            stats["grounding_fixed"] += 1
    return mentions


def _clamp_to_bounds(
    span: Any, bounds: tuple[int, int], *, fallback: tuple[int, int] | None = None,
) -> tuple[dict[str, int], bool]:
    lower, upper = bounds
    if upper <= lower:
        return _span_dict(lower, upper), _span_pair(span) != (lower, upper)
    pair = _span_pair(span)
    if pair is None or pair[1] <= pair[0]:
        pair = fallback or bounds
    start, end = pair
    new_start = min(max(start, lower), upper - 1)
    new_end = min(max(end, new_start + 1), upper)
    result = _span_dict(new_start, new_end)
    return result, pair != (new_start, new_end) or _span_pair(span) != (new_start, new_end)


def _operation_fallback_span(
    operation: Mapping[str, Any], mentions_by_id: Mapping[str, Mapping[str, Any]],
    bounds: tuple[int, int],
) -> tuple[int, int]:
    spans = []
    for mention_id in operation.get("target_mention_ids", ()):
        mention = mentions_by_id.get(str(mention_id))
        pair = _span_pair(mention.get("span")) if mention is not None else None
        if pair is not None and bounds[0] <= pair[0] < pair[1] <= bounds[1]:
            spans.append(pair)
    if spans:
        return min(item[0] for item in spans), max(item[1] for item in spans)
    return bounds


def _clamp_operation_spans(
    operations: list[dict[str, Any]],
    clauses: list[dict[str, Any]],
    mentions: list[dict[str, Any]],
    stats: dict[str, Any],
) -> list[dict[str, Any]]:
    clauses_by_id = {str(clause["id"]): clause for clause in clauses}
    mentions_by_id = {str(mention.get("id", "")): mention for mention in mentions}
    for operation in operations:
        clause = clauses_by_id.get(str(operation.get("clause_id", "")))
        bounds = _clause_span(clause)
        if bounds is None:
            continue
        fallback = _operation_fallback_span(operation, mentions_by_id, bounds)
        normalized, changed = _clamp_to_bounds(operation.get("span"), bounds, fallback=fallback)
        if changed:
            operation["span"] = normalized
            stats["operation_spans_clamped"] += 1

        arguments = operation.get("mentioned_arguments") or {}
        if isinstance(arguments, Mapping):
            for values in arguments.values():
                if not isinstance(values, list):
                    continue
                for value in values:
                    if not isinstance(value, dict) or value.get("evidence_span") is None:
                        continue
                    mention = mentions_by_id.get(str(value.get("mention_id", "")))
                    mention_span = _span_pair(mention.get("span")) if mention is not None else None
                    argument_evidence, argument_changed = _clamp_to_bounds(
                        value.get("evidence_span"), bounds,
                        fallback=mention_span or _span_pair(operation["span"]),
                    )
                    if argument_changed:
                        value["evidence_span"] = argument_evidence
                        stats["evidence_spans_clamped"] += 1
                        stats["argument_evidence_spans_clamped"] += 1

        evidence = operation.get("evidence_spans")
        if not isinstance(evidence, list) or not evidence:
            operation["evidence_spans"] = [deepcopy(operation["span"])]
            stats["evidence_spans_clamped"] += 1
        else:
            repaired = []
            for span in evidence:
                fixed, span_changed = _clamp_to_bounds(span, bounds, fallback=_span_pair(operation["span"]))
                repaired.append(fixed)
                if span_changed:
                    stats["evidence_spans_clamped"] += 1
            operation["evidence_spans"] = repaired

        cue = operation.get("negation_cue_span")
        if cue is not None:
            fixed, changed = _clamp_to_bounds(cue, bounds, fallback=_span_pair(operation["span"]))
            if changed:
                operation["negation_cue_span"] = fixed
                stats["negation_spans_clamped"] += 1

        for negation in operation.get("argument_negations", ()):
            source_clause = clauses_by_id.get(str(negation.get("source_clause_id", "")))
            source_bounds = _clause_span(source_clause)
            if source_bounds is None:
                continue
            cue_span, cue_changed = _clamp_to_bounds(
                negation.get("cue_span"), source_bounds, fallback=_span_pair(operation["span"]),
            )
            if cue_changed:
                negation["cue_span"] = cue_span
                stats["argument_negation_spans_clamped"] += 1
            negation_evidence = negation.get("evidence_spans")
            if not isinstance(negation_evidence, list) or not negation_evidence:
                negation["evidence_spans"] = [deepcopy(cue_span)]
                stats["argument_negation_spans_clamped"] += 1
            else:
                fixed_evidence = []
                for span in negation_evidence:
                    fixed, evidence_changed = _clamp_to_bounds(
                        span, source_bounds, fallback=_span_pair(cue_span),
                    )
                    fixed_evidence.append(fixed)
                    if evidence_changed:
                        stats["argument_negation_spans_clamped"] += 1
                negation["evidence_spans"] = fixed_evidence
    return operations


def _clamp_unowned_evidence(candidate: dict[str, Any], source_text: str, stats: dict[str, Any]) -> None:
    if not source_text:
        return
    bounds = (0, len(source_text))
    for item in candidate.get("intent_candidates", ()):
        spans = item.get("evidence_spans", ())
        fixed_spans = []
        for span in spans:
            fixed, changed = _clamp_to_bounds(span, bounds, fallback=bounds)
            fixed_spans.append(fixed)
            if changed:
                stats["intent_evidence_spans_clamped"] += 1
        if isinstance(spans, list):
            item["evidence_spans"] = fixed_spans
    for item in candidate.get("ambiguities", ()):
        spans = item.get("spans", ())
        fixed_spans = []
        for span in spans:
            fixed, changed = _clamp_to_bounds(span, bounds, fallback=bounds)
            fixed_spans.append(fixed)
            if changed:
                stats["ambiguity_spans_clamped"] += 1
        if isinstance(spans, list):
            item["spans"] = fixed_spans


def _semantic_signature(candidate: Mapping[str, Any]) -> Any:
    coordinate_or_grounding_fields = {
        "ambiguity_spans", "clause_id", "clauses", "cue_span", "evidence_span",
        "evidence_spans", "grounding_source", "negation_cue_span", "source_clause_id",
        "span", "spans",
    }

    def strip_coordinates(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: strip_coordinates(child)
                for key, child in value.items()
                if key not in coordinate_or_grounding_fields
            }
        if isinstance(value, (list, tuple)):
            return [strip_coordinates(child) for child in value]
        return value

    return strip_coordinates(candidate)


def normalize_candidate(
    candidate: dict, source_text: str, catalog: SemanticCatalog = DEFAULT_SEMANTIC_CATALOG,
) -> tuple[dict, dict]:
    """Return a coordinate-normalized Candidate and auditable repair counters.

    Semantic fields are guarded by an internal before/after signature. Unresolved
    surfaces are left untouched so the Validator can continue to fail closed.
    """

    if not isinstance(candidate, dict):
        raise TypeError("candidate must be a dict")
    normalized = deepcopy(candidate)
    stats: dict[str, Any] = {
        "enabled": True,
        "clauses_fixed": 0,
        "clauses_merged": 0,
        "clauses_fallback_split": 0,
        "clause_ids_reassigned": 0,
        "clause_boundaries_adjusted_for_mentions": 0,
        "clause_boundaries_merged": 0,
        "mention_spans_fixed": 0,
        "mention_spans_unresolved": 0,
        "mention_clause_remapped": 0,
        "grounding_fixed": 0,
        "operation_clause_remapped": 0,
        "operation_spans_clamped": 0,
        "evidence_spans_clamped": 0,
        "argument_evidence_spans_clamped": 0,
        "negation_spans_clamped": 0,
        "argument_negation_spans_clamped": 0,
        "intent_evidence_spans_clamped": 0,
        "ambiguity_spans_clamped": 0,
        # Store identifiers only; raw surfaces can contain user text.
        "unresolved_surface": [],
    }
    before_signature = _semantic_signature(candidate)

    old_clauses = deepcopy(normalized.get("clauses", []))
    clauses, clause_id_map = _normalize_clauses(old_clauses, source_text, stats)
    normalized["clauses"] = clauses
    _map_clause_references(normalized, clause_id_map)

    mentions = normalized.get("mentions", [])
    if not isinstance(mentions, list):
        mentions = []
        normalized["mentions"] = mentions
    clauses = _adjust_boundaries_for_mentions(clauses, mentions, source_text, stats)
    normalized["clauses"] = clauses
    _resolve_mention_spans(mentions, clauses, source_text, stats)
    _normalize_grounding(mentions, catalog, stats)

    clauses, merge_map = _merge_clause_boundaries_inside_mentions(
        clauses, mentions, source_text, stats,
    )
    remapped_mention_ids: set[str] = set()
    if merge_map:
        normalized["clauses"] = clauses
        for operation in normalized.get("operations", []):
            old_clause_id = operation.get("clause_id")
            if isinstance(old_clause_id, str) and merge_map.get(old_clause_id, old_clause_id) != old_clause_id:
                stats["operation_clause_remapped"] += 1
        _map_clause_references(normalized, merge_map)
        clauses_by_id = {str(clause["id"]): clause for clause in clauses}
        for mention in mentions:
            pair = _span_pair(mention.get("span"))
            owner_id = _owner_clause(pair, clauses_by_id) if pair is not None else None
            if owner_id is not None and mention.get("clause_id") != owner_id:
                mention["clause_id"] = owner_id
                stats["mention_clause_remapped"] += 1
                remapped_mention_ids.add(str(mention.get("id", "")))

    mentions_by_id = {str(item.get("id", "")): item for item in mentions}
    for role in normalized.get("roles", []):
        mention = mentions_by_id.get(str(role.get("mention_id", "")))
        if mention is not None and role.get("clause_id") != mention.get("clause_id"):
            role["clause_id"] = mention["clause_id"]
    _remap_rfm_method_operations(
        normalized.get("operations", []), mentions, remapped_mention_ids, stats,
    )
    _clamp_operation_spans(
        normalized.get("operations", []), clauses, mentions, stats,
    )
    _clamp_unowned_evidence(normalized, source_text, stats)

    if _semantic_signature(normalized) != before_signature:
        raise RuntimeError("semantic_span_resolver_changed_semantic_fields")
    stats["unresolved_surface_count"] = stats["mention_spans_unresolved"]
    return normalized, stats

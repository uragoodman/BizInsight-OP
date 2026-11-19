"""Relation predicates and deterministic relation construction helpers."""

from __future__ import annotations

from typing import Literal

from app.context.semantic_contract import Relation

RelationPredicate = Literal[
    "GROUP_BY",
    "FILTER_BY",
    "COMPARE_WITH",
    "AGGREGATE_BY",
    "REFERS_TO",
    "INHERITS_FROM",
]


def make_relation(subject: str, predicate: RelationPredicate, object: str) -> Relation:
    """Build a validated relation without adding implicit semantic facts."""

    return Relation(subject=subject, predicate=predicate, object=object)


def unique_relations(relations: tuple[Relation, ...] | list[Relation]) -> tuple[Relation, ...]:
    """Deduplicate relations while preserving the first extracted occurrence."""

    seen: set[tuple[str, str, str]] = set()
    result: list[Relation] = []
    for relation in relations:
        key = (relation.subject, relation.predicate, relation.object)
        if key not in seen:
            seen.add(key)
            result.append(relation)
    return tuple(result)

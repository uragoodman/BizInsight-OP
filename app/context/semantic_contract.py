"""Frozen, template-independent semantic contract models.

This module deliberately has no dependency on the analysis template registry.
The contract captures what the user said, not which implementation might serve
the request.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Span(ContractModel):
    """A half-open span in the original Python Unicode string."""

    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_coordinates(self) -> Span:
        if self.end < self.start:
            raise ValueError("span end must be greater than or equal to start")
        if self.end - self.start != len(self.text):
            raise ValueError("span length must match its Unicode text length")
        return self


class Mention(ContractModel):
    text: str
    span: Span
    candidate_types: tuple[str, ...]

    @model_validator(mode="after")
    def validate_text_matches_span(self) -> Mention:
        if self.text != self.span.text:
            raise ValueError("mention text must match the original span text")
        return self


class Entity(ContractModel):
    id: str
    type: Literal[
        "metric", "dimension", "business_entity", "semantic_factor", "time", "scope"
    ]
    alias_type: Literal["generic", "specific"] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class Operation(ContractModel):
    id: str
    label: str | None = None
    mode: Literal["EXECUTE", "EXPLAIN", "ASK"]
    polarity: Literal["POSITIVE", "NEGATIVE"]


class Role(ContractModel):
    entity: str
    role: Literal[
        "analysis_subject", "segment_basis", "filter", "comparison_target", "reference"
    ]


class Relation(ContractModel):
    subject: str
    predicate: Literal[
        "GROUP_BY",
        "FILTER_BY",
        "COMPARE_WITH",
        "AGGREGATE_BY",
        "REFERS_TO",
        "INHERITS_FROM",
    ]
    object: str


class ReferenceIntegrityIssue(ContractModel):
    code: Literal["missing_role_entity", "missing_relation_entity"]
    location: str
    entity_id: str
    message: str


def reference_integrity_issues(
    *, entities: tuple[Entity, ...], roles: tuple[Role, ...], relations: tuple[Relation, ...]
) -> tuple[ReferenceIntegrityIssue, ...]:
    """Return structured issues for role/relation references to absent entities."""

    entity_ids = {entity.id for entity in entities}
    issues: list[ReferenceIntegrityIssue] = []
    for index, role in enumerate(roles):
        if role.entity not in entity_ids:
            issues.append(
                ReferenceIntegrityIssue(
                    code="missing_role_entity",
                    location=f"roles.{index}.entity",
                    entity_id=role.entity,
                    message=f"role references unknown entity {role.entity!r}",
                )
            )
    for index, relation in enumerate(relations):
        for endpoint in ("subject", "object"):
            entity_id = getattr(relation, endpoint)
            if entity_id not in entity_ids:
                issues.append(
                    ReferenceIntegrityIssue(
                        code="missing_relation_entity",
                        location=f"relations.{index}.{endpoint}",
                        entity_id=entity_id,
                        message=f"relation references unknown entity {entity_id!r}",
                    )
                )
    return tuple(issues)


def _template_leak_paths(value: Any, path: str = "$") -> list[str]:
    """Find routing/template fields or IDs in a value destined for the contract."""

    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).casefold()
            child_path = f"{path}.{key}"
            if "template" in key_text or key_text in {
                "route", "route_id", "previous_route", "routing_context"
            }:
                paths.append(child_path)
            paths.extend(_template_leak_paths(item, child_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            paths.extend(_template_leak_paths(item, f"{path}.{index}"))
    elif isinstance(value, str):
        if re.search(r"\btemplate\s*:\s*[A-Za-z0-9_.:-]+", value, flags=re.IGNORECASE):
            paths.append(path)
        if "template_capability" in value.casefold():
            paths.append(path)
    return paths


class SemanticContract(ContractModel):
    mentions: tuple[Mention, ...]
    entities: tuple[Entity, ...]
    operations: tuple[Operation, ...]
    request_mode: Literal["EXECUTE", "EXPLAIN", "ASK"]
    dialogue_act_provisional: Literal["NEW", "CONTINUE", "SWITCH", "AMBIGUOUS"]
    dialogue_act: Literal["NEW", "CONTINUE", "SWITCH", "AMBIGUOUS"] | None
    modality: Literal["FACTUAL", "HYPOTHETICAL"]
    roles: tuple[Role, ...]
    relations: tuple[Relation, ...]
    semantic_status: Literal["CONFIRMED", "AMBIGUOUS", "UNSUPPORTED", "MISSING", "CONFLICT"]
    context_sources: dict[str, Any]
    frozen: bool = True

    @model_validator(mode="after")
    def validate_contract_integrity(self) -> SemanticContract:
        if not self.frozen:
            raise ValueError("semantic contracts must be frozen")
        issues = reference_integrity_issues(
            entities=self.entities, roles=self.roles, relations=self.relations
        )
        if issues:
            details = "; ".join(
                f"{issue.location}: {issue.message}" for issue in issues
            )
            raise ValueError(f"reference_integrity: {details}")
        inherited = self.context_sources.get("inherited", ())
        if isinstance(inherited, (list, tuple)):
            unvalidated = [
                index
                for index, item in enumerate(inherited)
                if not isinstance(item, dict) or item.get("validated") is not True
            ]
            if unvalidated:
                raise ValueError(
                    "inherited_context_unvalidated at: "
                    + ", ".join(f"context_sources.inherited.{index}" for index in unvalidated)
                )
        leak_paths = _template_leak_paths(self.model_dump(mode="python"))
        if leak_paths:
            raise ValueError(
                "template or routing data is forbidden in SemanticContract at: "
                + ", ".join(leak_paths)
            )
        return self

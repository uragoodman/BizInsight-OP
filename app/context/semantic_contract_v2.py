"""Schema for the template-independent Semantic Contract v2.1.

The candidate models describe interpreter output. They intentionally omit
``semantic_status`` and ``effective_arguments``; only the validator module may
produce the frozen output model that contains those fields.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ContractVersion = Literal["semantic-contract-v2.1"]
ParseStatus = Literal["PARSED", "AMBIGUOUS", "UNSUPPORTED", "INCOMPLETE"]
SemanticStatus = Literal["CONFIRMED", "AMBIGUOUS", "UNSUPPORTED", "MISSING", "CONFLICT"]
RequestDomain = Literal[
    "GENERAL_CONVERSATION",
    "BUSINESS_ANALYSIS",
    "BUSINESS_QUERY",
    "BUSINESS_REPORT",
]
OperationKind = Literal[
    "SEGMENT",
    "ANALYZE",
    "FILTER",
    "SORT",
    "AGGREGATE",
    "QUERY",
    "EXPLAIN",
    "EXPLAIN_CAUSE",
    "COMPARE",
    "TREND",
    "CONTRIBUTION",
    "BREAKDOWN",
    "ENUMERATE",
]
ArgumentPath = Literal[
    "basis",
    "topic",
    "metric",
    "dimension",
    "field",
    "value",
    "key",
    "direction",
    "group_by",
    "comparison_target",
    "time_grain",
    "outcome",
    "entity",
    "filter",
    "method",
]
ClauseScope = Literal["SAME_CLAUSE", "REQUEST"]
ArgumentStance = Literal["ADOPTED", "MENTIONED_ONLY", "EXCLUDED"]
GroundingSource = Literal["exact_alias", "catalog_synonym", "llm_inferred", "inherited"]
RoleName = Literal[
    "analysis_subject",
    "segment_basis",
    "filter_field",
    "comparison_target",
    "reference",
    "time_axis",
    "outcome",
]
RelationPredicate = Literal[
    "GROUP_BY",
    "FILTER_BY",
    "COMPARE_WITH",
    "AGGREGATE_BY",
    "REFERS_TO",
    "INHERITS_FROM",
    "CAUSE_OF",
    "BREAKS_DOWN_BY",
]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class Span(ContractModel):
    """Zero-based, half-open Python Unicode-codepoint coordinates."""

    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_nonempty(self) -> Span:
        if self.end <= self.start:
            raise ValueError("span end must be greater than span start")
        return self


class Clause(ContractModel):
    id: str = Field(min_length=1)
    span: Span


class Mention(ContractModel):
    id: str = Field(min_length=1)
    clause_id: str = Field(min_length=1)
    span: Span | None
    surface: str = Field(min_length=1)
    concept_id: str = Field(min_length=1)
    candidate_types: tuple[str, ...] = ()
    grounding_source: GroundingSource
    source_turn_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_grounding_shape(self) -> Mention:
        if self.grounding_source == "inherited" and self.span is not None:
            raise ValueError("inherited mentions do not claim a current-turn span")
        if self.grounding_source != "inherited" and self.span is None:
            raise ValueError("current-turn mentions require a source span")
        return self


class ArgumentValue(ContractModel):
    mention_id: str = Field(min_length=1)
    concept_id: str = Field(min_length=1)
    stance: ArgumentStance
    evidence_span: Span | None = None

    @model_validator(mode="after")
    def validate_adoption_evidence(self) -> ArgumentValue:
        if self.stance == "ADOPTED" and self.evidence_span is None:
            raise ValueError("adopted arguments require positive evidence")
        return self


class FrozenArgumentMap(dict[ArgumentPath, tuple[ArgumentValue, ...]]):
    """JSON-compatible mapping that prevents post-validation mutation."""

    def __init__(self, values: dict[ArgumentPath, tuple[ArgumentValue, ...]]):
        dict.__init__(self, values)

    @staticmethod
    def _immutable(*_args, **_kwargs):
        raise TypeError("effective_arguments is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


class ArgumentNegation(ContractModel):
    id: str = Field(min_length=1)
    source_clause_id: str = Field(min_length=1)
    source_turn_id: str = Field(min_length=1)
    cue_span: Span
    argument_path: ArgumentPath
    target_value_mention_id: str = Field(min_length=1)
    evidence_spans: tuple[Span, ...] = Field(min_length=1)


class OperationCandidate(ContractModel):
    id: str = Field(min_length=1)
    clause_id: str = Field(min_length=1)
    span: Span
    kind: OperationKind
    target_mention_ids: tuple[str, ...] = Field(min_length=0)
    target_concept_ids: tuple[str, ...] = Field(min_length=1)
    mentioned_arguments: dict[ArgumentPath, tuple[ArgumentValue, ...]] = Field(default_factory=dict)
    argument_negations: tuple[ArgumentNegation, ...] = ()
    mode: Literal["EXECUTE", "EXPLAIN", "ASK"]
    polarity: Literal["POSITIVE", "NEGATIVE"]
    clause_scope: ClauseScope | None = None
    negation_cue_span: Span | None = None
    # Request-wide operation negation is valid only when its source is explicit.
    source_clause_id: str | None = None
    source_turn_id: str | None = None
    evidence_spans: tuple[Span, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_polarity_fields(self) -> OperationCandidate:
        if not self.target_mention_ids:
            has_grounded_argument = any(
                values for values in self.mentioned_arguments.values() if values
            )
            if not self.target_concept_ids or not has_grounded_argument:
                raise ValueError("missing_target_reference")
        if self.polarity == "NEGATIVE":
            if self.clause_scope is None or self.negation_cue_span is None:
                raise ValueError("negative operations require an explicit clause_scope and cue span")
            if (self.source_clause_id is None) != (self.source_turn_id is None):
                raise ValueError("negative operation source clause and turn must be provided together")
            if self.argument_negations:
                raise ValueError("argument negations may only attach to positive operations")
            if self.mentioned_arguments:
                raise ValueError("negative operations must have an empty argument map")
        elif (
            self.clause_scope is not None
            or self.negation_cue_span is not None
            or self.source_clause_id is not None
            or self.source_turn_id is not None
        ):
            raise ValueError("positive operations cannot carry operation-level negation fields")
        return self


class Operation(OperationCandidate):
    """Validator-produced operation with computed effective arguments."""

    effective_arguments: dict[ArgumentPath, tuple[ArgumentValue, ...]]

    @model_validator(mode="after")
    def freeze_effective_arguments(self) -> Operation:
        object.__setattr__(
            self,
            "effective_arguments",
            FrozenArgumentMap(
                {
                    path: tuple(values)
                    for path, values in self.effective_arguments.items()
                }
            ),
        )
        return self


class RequestFrame(ContractModel):
    request_mode: Literal["EXECUTE", "EXPLAIN", "ASK"]
    modality: Literal["FACTUAL", "HYPOTHETICAL"]
    dialogue_act_provisional: Literal["NEW", "CONTINUE", "SWITCH", "AMBIGUOUS"]
    dialogue_act_validated: Literal["NEW", "CONTINUE", "SWITCH", "AMBIGUOUS"] | None


class IntentCandidate(ContractModel):
    intent_id: str = Field(min_length=1)
    evidence_spans: tuple[Span, ...] = Field(min_length=1)
    confidence_class: Literal["HIGH", "MEDIUM", "LOW"]


class Ambiguity(ContractModel):
    code: str = Field(min_length=1)
    spans: tuple[Span, ...] = ()
    clarification_question_candidate: str | None = None


class ContextSource(ContractModel):
    source_turn_id: str = Field(min_length=1)
    mention_ids: tuple[str, ...] = ()
    relation: Literal["EXPLICIT_REFERENCE", "INHERITED_CONTEXT"]


class InterpreterMetadata(ContractModel):
    model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    elapsed_ms: int = Field(ge=0)


class MentionEndpoint(ContractModel):
    endpoint_type: Literal["mention_id"]
    value: str = Field(min_length=1)


class ConceptEndpoint(ContractModel):
    endpoint_type: Literal["concept_id"]
    value: str = Field(min_length=1)


class RoleEndpoint(ContractModel):
    endpoint_type: Literal["role_id"]
    value: str = Field(min_length=1)


RelationEndpoint = Annotated[
    MentionEndpoint | ConceptEndpoint | RoleEndpoint,
    Field(discriminator="endpoint_type"),
]


class Role(ContractModel):
    id: str = Field(min_length=1)
    clause_id: str = Field(min_length=1)
    mention_id: str = Field(min_length=1)
    concept_id: str = Field(min_length=1)
    name: RoleName


class Relation(ContractModel):
    id: str = Field(min_length=1)
    clause_id: str = Field(min_length=1)
    predicate: RelationPredicate
    subject: RelationEndpoint
    object: RelationEndpoint


class SemanticContractCandidateV21(ContractModel):
    """Interpreter-shaped input. Extra fields such as effective_arguments fail closed."""

    contract_version: ContractVersion
    turn_id: str = Field(min_length=1)
    parse_status: ParseStatus
    domain: RequestDomain
    clauses: tuple[Clause, ...]
    mentions: tuple[Mention, ...]
    operations: tuple[OperationCandidate, ...]
    request_frame: RequestFrame
    roles: tuple[Role, ...] = ()
    relations: tuple[Relation, ...] = ()
    intent_candidates: tuple[IntentCandidate, ...] = ()
    primary_intent: str | None = None
    ambiguities: tuple[Ambiguity, ...] = ()
    context_sources: tuple[ContextSource, ...] = ()
    catalog_snapshot_hash: str = Field(min_length=1)


class SemanticContractV21(ContractModel):
    """Frozen validator output; semantic_status is assigned only by Validator."""

    contract_version: ContractVersion
    turn_id: str = Field(min_length=1)
    parse_status: ParseStatus
    semantic_status: SemanticStatus
    domain: RequestDomain
    clauses: tuple[Clause, ...]
    mentions: tuple[Mention, ...]
    operations: tuple[Operation, ...]
    request_frame: RequestFrame
    roles: tuple[Role, ...] = ()
    relations: tuple[Relation, ...] = ()
    intent_candidates: tuple[IntentCandidate, ...] = ()
    primary_intent: str | None = None
    ambiguities: tuple[Ambiguity, ...] = ()
    context_sources: tuple[ContextSource, ...] = ()
    catalog_snapshot_hash: str = Field(min_length=1)
    interpreter: InterpreterMetadata | None = None


class TemplateConfirmation(ContractModel):
    """Structured UI selection/confirmation envelope, separate from language routing."""

    protocol_version: Literal["template-confirmation-v1"]
    action: Literal["SELECT_TEMPLATE", "CONFIRM_PLAN", "CANCEL_PLAN"]
    current_turn_id: str = Field(min_length=1)
    plan_hash: str = Field(min_length=1)
    template_id: str | None = None
    candidate_set_hash: str | None = None

    @model_validator(mode="after")
    def validate_action_fields(self) -> TemplateConfirmation:
        if self.action == "SELECT_TEMPLATE" and (not self.template_id or not self.candidate_set_hash):
            raise ValueError("template selection requires template_id and candidate_set_hash")
        if self.action != "SELECT_TEMPLATE" and (self.template_id is not None or self.candidate_set_hash is not None):
            raise ValueError("template_id and candidate_set_hash are only valid when selecting a template")
        return self


class CapabilityReference(ContractModel):
    capability_id: str = Field(min_length=1)


class CapabilityDecisionBasis(ContractModel):
    contract_version: ContractVersion
    catalog_snapshot_hash: str = Field(min_length=1)
    operation_ids: tuple[str, ...] = ()
    clause_ids: tuple[str, ...] = ()
    capability_ids: tuple[CapabilityReference, ...]
    effective_arguments_digest: str | None = None
    rejected_candidates: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_unique_references(self) -> CapabilityDecisionBasis:
        capability_ids = [item.capability_id for item in self.capability_ids]
        if len(capability_ids) != len(set(capability_ids)):
            raise ValueError("decision_basis capability references must be unique")
        if len(self.operation_ids) != len(set(self.operation_ids)):
            raise ValueError("decision_basis operation IDs must be unique")
        if len(self.clause_ids) != len(set(self.clause_ids)):
            raise ValueError("decision_basis clause IDs must be unique")
        return self


class SealedRouteDecision(ContractModel):
    """Typed, catalog-bound internal decision required by plan creation.

    It is not an API payload or an authorization credential. Only a future
    server-side semantic router may construct one after validation.
    """

    contract_version: ContractVersion
    catalog_snapshot_hash: str = Field(min_length=64, max_length=64)
    route_decision_status: Literal["SELECTED"]
    route_id: str = Field(min_length=1)
    capability_id: str = Field(min_length=1)
    template_id: str = Field(min_length=1)
    operation_ids: tuple[str, ...] = Field(min_length=1)
    clause_ids: tuple[str, ...] = Field(min_length=1)
    effective_arguments_digest: str = Field(min_length=64, max_length=64)
    decision_basis: CapabilityDecisionBasis
    seal_digest: str = Field(min_length=64, max_length=64)

    @classmethod
    def issue(cls, **payload) -> SealedRouteDecision:
        basis = payload.get("decision_basis")
        normalized = {
            **payload,
            "decision_basis": basis.model_dump(mode="json") if isinstance(basis, BaseModel) else basis,
        }
        serialized = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        return cls(**payload, seal_digest=hashlib.sha256(serialized.encode("utf-8")).hexdigest())

    @model_validator(mode="after")
    def validate_decision_basis(self) -> SealedRouteDecision:
        basis = self.decision_basis
        if basis.contract_version != self.contract_version:
            raise ValueError("sealed decision contract version does not match its basis")
        if basis.catalog_snapshot_hash != self.catalog_snapshot_hash:
            raise ValueError("sealed decision catalog hash does not match its basis")
        if basis.operation_ids != self.operation_ids or basis.clause_ids != self.clause_ids:
            raise ValueError("sealed decision IDs do not match its decision basis")
        if basis.effective_arguments_digest != self.effective_arguments_digest:
            raise ValueError("sealed decision argument digest does not match its decision basis")
        if tuple(item.capability_id for item in basis.capability_ids) != (self.capability_id,):
            raise ValueError("sealed decision must cite exactly its selected capability")
        if not self.verify_seal():
            raise ValueError("sealed decision integrity digest is invalid")
        return self

    def seal_payload(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "catalog_snapshot_hash": self.catalog_snapshot_hash,
            "route_decision_status": self.route_decision_status,
            "route_id": self.route_id,
            "capability_id": self.capability_id,
            "template_id": self.template_id,
            "operation_ids": self.operation_ids,
            "clause_ids": self.clause_ids,
            "effective_arguments_digest": self.effective_arguments_digest,
            "decision_basis": self.decision_basis.model_dump(mode="json"),
        }

    def verify_seal(self) -> bool:
        serialized = json.dumps(self.seal_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest() == self.seal_digest

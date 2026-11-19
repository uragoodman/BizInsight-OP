"""Deterministic validation and freezing for Semantic Contract v2.1."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.context.semantic_catalog_v2 import DEFAULT_SEMANTIC_CATALOG, SemanticCatalog
from app.context.semantic_contract_v2 import (
    ArgumentStance,
    Operation,
    SemanticContractCandidateV21,
    SemanticContractV21,
    TemplateConfirmation,
)


_PARSE_TO_SEMANTIC_STATUS: dict[str, str] = {
    "PARSED": "CONFIRMED",
    "AMBIGUOUS": "AMBIGUOUS",
    "UNSUPPORTED": "UNSUPPORTED",
    "INCOMPLETE": "MISSING",
}


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1)
    location: str = Field(min_length=1)
    severity: Literal["MISSING", "AMBIGUOUS", "CONFLICT"]


class SemanticContractValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract: SemanticContractV21
    issues: tuple[ValidationIssue, ...] = ()
    audit_issues: tuple[ValidationIssue, ...] = ()


ConfirmationOutcome = Literal[
    "CONFIRMED",
    "TEMPLATE_SELECTED",
    "CANCELLED",
    "STALE_PLAN_HASH",
    "STALE_CURRENT_TURN",
    "INVALID_SELECTION",
    "UNAUTHORIZED_TEMPLATE",
    "BLOCKED_BY_CURRENT_TURN_NEGATIVE",
    "NEEDS_SEMANTIC_VALIDATION",
]

TEMPLATE_CONFIRMATION_WIRED = False
VALIDATOR_ISSUE_CODES: tuple[str, ...] = (
    "argument_concept_domain_mismatch",
    "argument_concept_reference_mismatch",
    "argument_evidence_outside_operation_clause",
    "argument_negation_cue_outside_source_clause",
    "argument_negation_evidence_outside_source_clause",
    "catalog_snapshot_mismatch",
    "cross_turn_argument_negation_requires_clarification",
    "current_mention_turn_mismatch",
    "duplicate_id",
    "empty_inherited_context_reference",
    "empty_text_has_clauses",
    "exact_alias_not_in_catalog",
    "excluded_value_missing_negation_evidence",
    "general_domain_has_operations",
    "inherited_mention_same_turn",
    "intent_id_used_as_concept",
    "invalid_argument_path",
    "invalid_clause_partition",
    "invalid_context_mention_reference",
    "mention_outside_owned_clause",
    "missing_clause_partition",
    "missing_target_reference",
    "negation_cue_outside_owned_clause",
    "negation_target_not_excluded_argument",
    "negative_operation_source_missing",
    "negative_target_outside_owned_clause",
    "operation_evidence_outside_owned_clause",
    "operation_outside_owned_clause",
    "parsed_business_contract_has_no_operations",
    "positive_negative_argument_conflict",
    "positive_negative_operation_conflict",
    "primary_intent_operation_mismatch",
    "role_clause_reference_mismatch",
    "role_concept_reference_mismatch",
    "span_out_of_range",
    "span_surface_mismatch",
    "synonym_not_in_catalog",
    "target_concept_reference_mismatch",
    "unknown_clause_reference",
    "unknown_concept_reference",
    "unknown_intent_reference",
    "unknown_mention_reference",
    "unknown_role_reference",
    "unreferenced_inherited_mention",
    "untrusted_inherited_context_reference",
)


def _inside(inner_start: int, inner_end: int, outer_start: int, outer_end: int) -> bool:
    return outer_start <= inner_start and inner_end <= outer_end


def _span_in_text(span, source_text: str) -> bool:
    return span.end <= len(source_text)


def _add_issue(
    issues: list[ValidationIssue], code: str, location: str, severity: str = "MISSING"
) -> None:
    issues.append(ValidationIssue(code=code, location=location, severity=severity))


def _validate_concept_reference(
    issues: list[ValidationIssue],
    catalog: SemanticCatalog,
    concept_id: str,
    location: str,
) -> None:
    concept_exists = catalog.concept(concept_id) is not None
    if not concept_exists:
        _add_issue(issues, "unknown_concept_reference", location)
    # Some catalog identifiers intentionally exist in both collections. Only
    # flag an intent ID here when it is not also a registered concept.
    if not concept_exists and concept_id in catalog.intent_ids:
        _add_issue(issues, "intent_id_used_as_concept", location, "CONFLICT")


def _add_audit_issue(
    audit_issues: list[ValidationIssue], code: str, location: str, severity: str = "MISSING"
) -> None:
    audit_issues.append(ValidationIssue(code=code, location=location, severity=severity))


def validate_semantic_contract(
    candidate: SemanticContractCandidateV21 | Mapping[str, object],
    *,
    source_text: str,
    catalog: SemanticCatalog = DEFAULT_SEMANTIC_CATALOG,
    trusted_prior_contract: SemanticContractV21 | None = None,
) -> SemanticContractValidationResult:
    """Validate spans/references and compute validator-owned fields.

    ``source_text`` is read only for coordinate and surface equality checks. It
    never supplies or repairs a Mention, Operation, polarity, or argument.
    """

    parsed = (
        candidate
        if isinstance(candidate, SemanticContractCandidateV21)
        else SemanticContractCandidateV21.model_validate(candidate)
    )
    issues: list[ValidationIssue] = []
    audit_issues: list[ValidationIssue] = []

    clauses_by_id = {item.id: item for item in parsed.clauses}
    mentions_by_id = {item.id: item for item in parsed.mentions}
    roles_by_id = {item.id: item for item in parsed.roles}
    registered_method_concepts = {
        concept_id
        for capability in catalog.capabilities
        for concept_id in capability.method_concept_ids
    }

    if parsed.catalog_snapshot_hash != catalog.snapshot_hash():
        _add_issue(issues, "catalog_snapshot_mismatch", "catalog_snapshot_hash")

    for collection_name, values, id_attribute in (
        ("clauses", parsed.clauses, "id"),
        ("mentions", parsed.mentions, "id"),
        ("operations", parsed.operations, "id"),
        ("roles", parsed.roles, "id"),
        ("relations", parsed.relations, "id"),
        ("intent_candidates", parsed.intent_candidates, "intent_id"),
    ):
        ids = [getattr(item, id_attribute) for item in values]
        if len(ids) != len(set(ids)):
            _add_issue(issues, "duplicate_id", collection_name)

    # Clauses are an ordered, gap-free, non-overlapping partition of the full
    # original string. Delimiters and whitespace belong to one adjacent clause.
    if not source_text and parsed.clauses:
        _add_issue(issues, "empty_text_has_clauses", "clauses")
    elif source_text:
        if not parsed.clauses:
            _add_issue(issues, "missing_clause_partition", "clauses")
        cursor = 0
        for index, clause in enumerate(parsed.clauses):
            if clause.span.start != cursor:
                _add_issue(issues, "invalid_clause_partition", f"clauses.{index}.span")
            if not _span_in_text(clause.span, source_text):
                _add_issue(issues, "span_out_of_range", f"clauses.{index}.span")
            cursor = clause.span.end
        if cursor != len(source_text):
            _add_issue(issues, "invalid_clause_partition", "clauses")

    for index, mention in enumerate(parsed.mentions):
        location = f"mentions.{index}"
        clause = clauses_by_id.get(mention.clause_id)
        if clause is None:
            _add_issue(issues, "unknown_clause_reference", f"{location}.clause_id")
        _validate_concept_reference(issues, catalog, mention.concept_id, f"{location}.concept_id")
        if mention.grounding_source == "inherited":
            if mention.source_turn_id == parsed.turn_id:
                _add_issue(issues, "inherited_mention_same_turn", f"{location}.source_turn_id")
        elif mention.source_turn_id != parsed.turn_id:
            _add_issue(issues, "current_mention_turn_mismatch", f"{location}.source_turn_id")
        if mention.span is not None:
            if not _span_in_text(mention.span, source_text):
                _add_issue(issues, "span_out_of_range", f"{location}.span")
            elif source_text[mention.span.start : mention.span.end] != mention.surface:
                _add_issue(issues, "span_surface_mismatch", f"{location}.surface")
            if clause is not None and not _inside(
                mention.span.start, mention.span.end, clause.span.start, clause.span.end
            ):
                _add_issue(issues, "mention_outside_owned_clause", f"{location}.span")
        concept = catalog.concept(mention.concept_id)
        if concept is not None:
            if mention.grounding_source == "exact_alias" and mention.surface not in concept.exact_aliases:
                _add_issue(issues, "exact_alias_not_in_catalog", f"{location}.surface")
            elif mention.grounding_source == "catalog_synonym" and mention.surface not in concept.synonyms:
                _add_issue(issues, "synonym_not_in_catalog", f"{location}.surface")
            # llm_inferred is accepted only for the explicit mention/concept
            # pair above. The validator never creates a missing mention.

    prior_mentions = (
        {mention.id: mention for mention in trusted_prior_contract.mentions}
        if trusted_prior_contract is not None
        else {}
    )
    inherited_context_refs: set[tuple[str, str]] = set()
    for index, context in enumerate(parsed.context_sources):
        for mention_id in context.mention_ids:
            mention = mentions_by_id.get(mention_id)
            if context.relation == "INHERITED_CONTEXT":
                prior_mention = prior_mentions.get(mention_id)
                if (
                    mention is None
                    or mention.grounding_source != "inherited"
                    or mention.source_turn_id != context.source_turn_id
                    or prior_mention is None
                    or prior_mention.source_turn_id != context.source_turn_id
                    or prior_mention.concept_id != mention.concept_id
                    or trusted_prior_contract is None
                    or trusted_prior_contract.semantic_status != "CONFIRMED"
                ):
                    _add_issue(issues, "untrusted_inherited_context_reference", f"context_sources.{index}")
                else:
                    inherited_context_refs.add((context.source_turn_id, mention_id))
            elif mention is None or mention.source_turn_id != context.source_turn_id:
                _add_issue(issues, "invalid_context_mention_reference", f"context_sources.{index}")
        if context.relation == "INHERITED_CONTEXT" and not context.mention_ids:
            _add_issue(issues, "empty_inherited_context_reference", f"context_sources.{index}")
    for index, mention in enumerate(parsed.mentions):
        if mention.grounding_source == "inherited" and (
            mention.source_turn_id,
            mention.id,
        ) not in inherited_context_refs:
            _add_issue(issues, "unreferenced_inherited_mention", f"mentions.{index}")

    value_stances: dict[tuple[str, str, str], set[ArgumentStance]] = defaultdict(set)
    negated_values: dict[tuple[str, str], set[str]] = defaultdict(set)
    for op_index, operation in enumerate(parsed.operations):
        location = f"operations.{op_index}"
        clause = clauses_by_id.get(operation.clause_id)
        if clause is None:
            _add_issue(issues, "unknown_clause_reference", f"{location}.clause_id")
        if not _span_in_text(operation.span, source_text):
            _add_issue(issues, "span_out_of_range", f"{location}.span")
        if clause is not None and not _inside(
            operation.span.start, operation.span.end, clause.span.start, clause.span.end
        ):
            _add_issue(issues, "operation_outside_owned_clause", f"{location}.span")
        for evidence_index, span in enumerate(operation.evidence_spans):
            if not _span_in_text(span, source_text):
                _add_issue(issues, "span_out_of_range", f"{location}.evidence_spans.{evidence_index}")
            if clause is not None and not _inside(
                span.start, span.end, clause.span.start, clause.span.end
            ):
                _add_issue(issues, "operation_evidence_outside_owned_clause", f"{location}.evidence_spans.{evidence_index}")
        if operation.negation_cue_span is not None:
            if not _span_in_text(operation.negation_cue_span, source_text):
                _add_issue(issues, "span_out_of_range", f"{location}.negation_cue_span")
            if clause is not None and not _inside(
                operation.negation_cue_span.start,
                operation.negation_cue_span.end,
                clause.span.start,
                clause.span.end,
            ):
                _add_issue(issues, "negation_cue_outside_owned_clause", f"{location}.negation_cue_span")
        if operation.polarity == "NEGATIVE":
            if operation.clause_scope == "REQUEST":
                if (
                    operation.source_clause_id is None
                    or operation.source_turn_id is None
                ):
                    _add_issue(
                        issues,
                        "negative_operation_source_missing",
                        f"{location}.source_clause_id",
                        "MISSING",
                    )
                elif (
                    operation.source_clause_id != operation.clause_id
                    or operation.source_turn_id != parsed.turn_id
                ):
                    _add_issue(
                        issues,
                        "negative_operation_source_missing",
                        f"{location}.source_clause_id",
                        "CONFLICT",
                    )
            elif operation.source_clause_id is not None and (
                operation.source_clause_id != operation.clause_id
                or operation.source_turn_id != parsed.turn_id
            ):
                _add_issue(
                    issues,
                    "negative_operation_source_missing",
                    f"{location}.source_clause_id",
                    "CONFLICT",
                )

        target_concepts: list[str] = []
        for mention_id in operation.target_mention_ids:
            mention = mentions_by_id.get(mention_id)
            if mention is None:
                _add_issue(issues, "unknown_mention_reference", f"{location}.target_mention_ids")
            else:
                target_concepts.append(mention.concept_id)
                if operation.polarity == "NEGATIVE" and operation.clause_scope == "SAME_CLAUSE":
                    if mention.clause_id != operation.clause_id:
                        _add_issue(
                            issues,
                            "negative_target_outside_owned_clause",
                            f"{location}.target_mention_ids",
                            "CONFLICT",
                        )
        if operation.target_mention_ids:
            if set(target_concepts) != set(operation.target_concept_ids):
                _add_issue(issues, "target_concept_reference_mismatch", f"{location}.target_concept_ids")
        else:
            valid_argument_reference = any(
                value.stance == "ADOPTED"
                and (mention := mentions_by_id.get(value.mention_id)) is not None
                and mention.concept_id == value.concept_id
                and (path != "method" or value.concept_id in registered_method_concepts)
                for path, values in operation.mentioned_arguments.items()
                for value in values
            )
            if not operation.target_concept_ids or not valid_argument_reference:
                _add_issue(issues, "missing_target_reference", f"{location}.target_mention_ids")
        for concept_id in operation.target_concept_ids:
            _validate_concept_reference(issues, catalog, concept_id, f"{location}.target_concept_ids")

        allowed_paths = set(catalog.operation_paths(operation.kind))
        for path, values in operation.mentioned_arguments.items():
            if path not in allowed_paths:
                _add_issue(issues, "invalid_argument_path", f"{location}.mentioned_arguments.{path}")
            concept_domain = catalog.operation_path_concept_domain(operation.kind, path)
            for value_index, value in enumerate(values):
                value_location = f"{location}.mentioned_arguments.{path}.{value_index}"
                if concept_domain is not None and value.concept_id not in concept_domain:
                    _add_issue(
                        issues,
                        "argument_concept_domain_mismatch",
                        f"{value_location}.concept_id",
                        "CONFLICT",
                    )
                mention = mentions_by_id.get(value.mention_id)
                if mention is None:
                    _add_issue(issues, "unknown_mention_reference", f"{value_location}.mention_id")
                elif mention.concept_id != value.concept_id:
                    _add_issue(issues, "argument_concept_reference_mismatch", f"{value_location}.concept_id")
                _validate_concept_reference(issues, catalog, value.concept_id, f"{value_location}.concept_id")
                if path == "method" and value.concept_id not in registered_method_concepts:
                    _add_issue(issues, "unknown_concept_reference", f"{value_location}.concept_id")
                if value.evidence_span is not None:
                    if not _span_in_text(value.evidence_span, source_text):
                        _add_issue(issues, "span_out_of_range", f"{value_location}.evidence_span")
                    if value.stance == "ADOPTED" and clause is not None and not _inside(
                        value.evidence_span.start,
                        value.evidence_span.end,
                        clause.span.start,
                        clause.span.end,
                    ):
                        _add_issue(issues, "argument_evidence_outside_operation_clause", f"{value_location}.evidence_span")
                value_stances[(operation.id, path, value.concept_id)].add(value.stance)

        for negation_index, negation in enumerate(operation.argument_negations):
            neg_location = f"{location}.argument_negations.{negation_index}"
            source_clause = clauses_by_id.get(negation.source_clause_id)
            if source_clause is None:
                _add_issue(issues, "unknown_clause_reference", f"{neg_location}.source_clause_id")
            if negation.source_turn_id != parsed.turn_id:
                _add_issue(
                    issues,
                    "cross_turn_argument_negation_requires_clarification",
                    f"{neg_location}.source_turn_id",
                    "AMBIGUOUS",
                )
            if negation.source_turn_id == parsed.turn_id:
                if not _span_in_text(negation.cue_span, source_text):
                    _add_issue(issues, "span_out_of_range", f"{neg_location}.cue_span")
                if source_clause is not None and not _inside(
                    negation.cue_span.start,
                    negation.cue_span.end,
                    source_clause.span.start,
                    source_clause.span.end,
                ):
                    _add_issue(issues, "argument_negation_cue_outside_source_clause", f"{neg_location}.cue_span")
            for evidence_index, span in enumerate(negation.evidence_spans):
                if negation.source_turn_id == parsed.turn_id:
                    if not _span_in_text(span, source_text):
                        _add_issue(issues, "span_out_of_range", f"{neg_location}.evidence_spans.{evidence_index}")
                    if source_clause is not None and not _inside(
                        span.start, span.end, source_clause.span.start, source_clause.span.end
                    ):
                        _add_issue(
                            issues,
                            "argument_negation_evidence_outside_source_clause",
                            f"{neg_location}.evidence_spans.{evidence_index}",
                        )
            if negation.argument_path not in allowed_paths:
                _add_issue(issues, "invalid_argument_path", f"{neg_location}.argument_path")
            target_value_mention = mentions_by_id.get(negation.target_value_mention_id)
            if target_value_mention is None:
                _add_issue(
                    issues,
                    "unknown_mention_reference",
                    f"{neg_location}.target_value_mention_id",
                )
            matching_values = [
                value
                for value in operation.mentioned_arguments.get(negation.argument_path, ())
                if value.mention_id == negation.target_value_mention_id
            ]
            if len(matching_values) != 1 or matching_values[0].stance != "EXCLUDED":
                _add_issue(issues, "negation_target_not_excluded_argument", f"{neg_location}.target_value_mention_id")
            else:
                negated_values[(operation.id, negation.argument_path)].add(
                    negation.target_value_mention_id
                )

        if operation.polarity == "POSITIVE":
            for path, values in operation.mentioned_arguments.items():
                for value in values:
                    if value.stance == "EXCLUDED" and not any(
                        item.argument_path == path
                        and item.target_value_mention_id == value.mention_id
                        for item in operation.argument_negations
                    ):
                        _add_issue(issues, "excluded_value_missing_negation_evidence", f"{location}.mentioned_arguments.{path}")

    for (operation_id, path, _concept_id), stances in value_stances.items():
        if "ADOPTED" in stances and "EXCLUDED" in stances:
            _add_issue(
                issues,
                "positive_negative_argument_conflict",
                f"operations.{operation_id}.mentioned_arguments.{path}",
                "CONFLICT",
            )

    for index, left in enumerate(parsed.operations):
        for right in parsed.operations[index + 1 :]:
            if (
                left.clause_id == right.clause_id
                and left.kind == right.kind
                and left.polarity != right.polarity
                and set(left.target_concept_ids) & set(right.target_concept_ids)
            ):
                _add_issue(
                    issues,
                    "positive_negative_operation_conflict",
                    f"operations.{left.id},{right.id}",
                    "CONFLICT",
                )

    for index, role in enumerate(parsed.roles):
        location = f"roles.{index}"
        _validate_concept_reference(issues, catalog, role.concept_id, f"{location}.concept_id")
        mention = mentions_by_id.get(role.mention_id)
        if mention is None:
            _add_issue(issues, "unknown_mention_reference", f"{location}.mention_id")
        elif mention.concept_id != role.concept_id:
            _add_issue(issues, "role_concept_reference_mismatch", f"{location}.concept_id")
        elif mention.clause_id != role.clause_id:
            _add_issue(issues, "role_clause_reference_mismatch", f"{location}.clause_id")
        if role.clause_id not in clauses_by_id:
            _add_issue(issues, "unknown_clause_reference", f"{location}.clause_id")

    for index, relation in enumerate(parsed.relations):
        if relation.clause_id not in clauses_by_id:
            _add_issue(issues, "unknown_clause_reference", f"relations.{index}.clause_id")
        for endpoint_name in ("subject", "object"):
            endpoint = getattr(relation, endpoint_name)
            path = f"relations.{index}.{endpoint_name}"
            if endpoint.endpoint_type == "mention_id" and endpoint.value not in mentions_by_id:
                _add_issue(issues, "unknown_mention_reference", path)
            elif endpoint.endpoint_type == "concept_id":
                _validate_concept_reference(issues, catalog, endpoint.value, path)
            elif endpoint.endpoint_type == "role_id" and endpoint.value not in roles_by_id:
                _add_issue(issues, "unknown_role_reference", path)

    for index, item in enumerate(parsed.intent_candidates):
        if item.intent_id not in catalog.intent_ids:
            _add_issue(issues, "unknown_intent_reference", f"intent_candidates.{index}.intent_id")
        for span_index, span in enumerate(item.evidence_spans):
            if not _span_in_text(span, source_text):
                _add_issue(issues, "span_out_of_range", f"intent_candidates.{index}.evidence_spans.{span_index}")
    if parsed.primary_intent is not None:
        if parsed.primary_intent not in catalog.intent_ids:
            _add_issue(issues, "unknown_intent_reference", "primary_intent")
        else:
            requirement = catalog.primary_intent_requirement(parsed.primary_intent)
            if requirement is None or not any(
                operation.kind in requirement.operation_kinds
                and bool(set(operation.target_concept_ids) & set(requirement.target_concept_ids))
                for operation in parsed.operations
            ):
                _add_audit_issue(
                    audit_issues,
                    "primary_intent_operation_mismatch",
                    "primary_intent",
                    "CONFLICT",
                )

    if parsed.domain == "GENERAL_CONVERSATION" and parsed.operations:
        _add_issue(issues, "general_domain_has_operations", "operations")
    if parsed.parse_status == "PARSED" and parsed.domain != "GENERAL_CONVERSATION" and not parsed.operations:
        _add_issue(issues, "parsed_business_contract_has_no_operations", "operations")

    computed_operations: list[Operation] = []
    for operation in parsed.operations:
        effective_arguments = {
            path: tuple(
                value
                for value in values
                if value.stance == "ADOPTED"
                and value.mention_id not in negated_values.get((operation.id, path), set())
            )
            for path, values in operation.mentioned_arguments.items()
        }
        operation_data = operation.model_dump(mode="python")
        operation_data["effective_arguments"] = effective_arguments
        computed_operations.append(Operation.model_validate(operation_data))

    conflict = any(issue.severity == "CONFLICT" for issue in issues)
    ambiguous = any(issue.severity == "AMBIGUOUS" for issue in issues)
    missing = any(issue.severity == "MISSING" for issue in issues)
    if conflict:
        semantic_status = "CONFLICT"
    elif ambiguous:
        semantic_status = "AMBIGUOUS"
    elif missing:
        semantic_status = "MISSING"
    else:
        semantic_status = _PARSE_TO_SEMANTIC_STATUS[parsed.parse_status]

    contract_data = parsed.model_dump(mode="python")
    contract_data["semantic_status"] = semantic_status
    contract_data["operations"] = tuple(computed_operations)
    contract = SemanticContractV21.model_validate(contract_data)
    return SemanticContractValidationResult(
        contract=contract,
        issues=tuple(issues),
        audit_issues=tuple(audit_issues),
    )


def resolve_template_confirmation(
    confirmation: TemplateConfirmation,
    *,
    expected_plan_hash: str,
    expected_current_turn_id: str,
    current_turn_contract: SemanticContractV21 | None,
    has_user_text: bool,
    plan_template_id: str,
    candidate_template_ids: tuple[str, ...] = (),
    expected_candidate_set_hash: str | None = None,
    visible_template_ids: frozenset[str] = frozenset(),
    authorized_template_ids: frozenset[str] = frozenset(),
    allow_protocol_resolution: bool = False,
) -> ConfirmationOutcome:
    """Apply UI confirmation precedence against server-owned plan/candidate state.

    Candidate membership, hash, visibility, and authorization are never taken
    from the confirmation payload itself. A current-turn negative operation
    vetoes selection or confirmation; text that has not been validated cannot
    confirm or select a plan.
    """

    if not TEMPLATE_CONFIRMATION_WIRED and not allow_protocol_resolution:
        return "NEEDS_SEMANTIC_VALIDATION"
    if confirmation.plan_hash != expected_plan_hash:
        return "STALE_PLAN_HASH"
    if confirmation.action == "CANCEL_PLAN":
        return "CANCELLED"
    if confirmation.current_turn_id != expected_current_turn_id:
        return "STALE_CURRENT_TURN"
    if confirmation.action == "SELECT_TEMPLATE":
        if (
            confirmation.candidate_set_hash != expected_candidate_set_hash
            or confirmation.template_id not in candidate_template_ids
        ):
            return "INVALID_SELECTION"
        selected_template_id = str(confirmation.template_id)
    else:
        selected_template_id = plan_template_id
    if selected_template_id not in visible_template_ids:
        return "UNAUTHORIZED_TEMPLATE"
    if selected_template_id not in authorized_template_ids:
        return "UNAUTHORIZED_TEMPLATE"
    if has_user_text and current_turn_contract is None:
        return "NEEDS_SEMANTIC_VALIDATION"
    if current_turn_contract is not None:
        if current_turn_contract.turn_id != expected_current_turn_id:
            return "NEEDS_SEMANTIC_VALIDATION"
        if any(operation.polarity == "NEGATIVE" for operation in current_turn_contract.operations):
            return "BLOCKED_BY_CURRENT_TURN_NEGATIVE"
        if current_turn_contract.semantic_status != "CONFIRMED":
            return "NEEDS_SEMANTIC_VALIDATION"
    if confirmation.action == "SELECT_TEMPLATE":
        return "TEMPLATE_SELECTED"
    return "CONFIRMED"

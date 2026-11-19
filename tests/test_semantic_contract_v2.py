from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.context.semantic_contract_v2 import (
    ArgumentValue,
    Clause,
    ConceptEndpoint,
    MentionEndpoint,
    OperationCandidate,
    Relation,
    RequestFrame,
    Role,
    RoleEndpoint,
    SemanticContractCandidateV21,
    Span,
    TemplateConfirmation,
)


def test_v21_input_schema_is_strict_and_keeps_validator_owned_fields_out() -> None:
    values = {
        "contract_version": "semantic-contract-v2.1",
        "turn_id": "turn-1",
        "parse_status": "PARSED",
        "domain": "BUSINESS_ANALYSIS",
        "clauses": (Clause(id="c1", span=Span(start=0, end=1)),),
        "mentions": (),
        "operations": (),
        "request_frame": RequestFrame(
            request_mode="EXECUTE",
            modality="FACTUAL",
            dialogue_act_provisional="NEW",
            dialogue_act_validated="NEW",
        ),
        "catalog_snapshot_hash": "catalog-hash",
    }
    with pytest.raises(ValidationError):
        SemanticContractCandidateV21.model_validate({**values, "semantic_status": "CONFIRMED"})
    with pytest.raises(ValidationError):
        SemanticContractCandidateV21.model_validate(
            {
                **values,
                "operations": (
                    {
                        "id": "op1",
                        "clause_id": "c1",
                        "span": {"start": 0, "end": 1},
                        "kind": "ANALYZE",
                        "target_mention_ids": ["m1"],
                        "target_concept_ids": ["CUSTOMER"],
                        "mentioned_arguments": {},
                        "argument_negations": [],
                        "effective_arguments": {},
                        "mode": "EXECUTE",
                        "polarity": "POSITIVE",
                        "evidence_spans": [{"start": 0, "end": 1}],
                    },
                ),
            }
        )


def test_operation_kind_and_argument_path_names_are_closed_and_uppercase() -> None:
    with pytest.raises(ValidationError):
        OperationCandidate(
            id="op1",
            clause_id="c1",
            span=Span(start=0, end=1),
            kind="segment",
            target_mention_ids=("m1",),
            target_concept_ids=("CUSTOMER",),
            mode="EXECUTE",
            polarity="POSITIVE",
            evidence_spans=(Span(start=0, end=1),),
        )


def test_negative_operation_cannot_carry_an_argument_map() -> None:
    with pytest.raises(ValidationError, match="negative operations must have an empty argument map"):
        OperationCandidate(
            id="op1",
            clause_id="c1",
            span=Span(start=0, end=1),
            kind="SEGMENT",
            target_mention_ids=("m1",),
            target_concept_ids=("CUSTOMER",),
            mentioned_arguments={
                "basis": (
                    ArgumentValue(
                        mention_id="m2",
                        concept_id="MONETARY",
                        stance="EXCLUDED",
                    ),
                ),
            },
            mode="EXECUTE",
            polarity="NEGATIVE",
            clause_scope="SAME_CLAUSE",
            negation_cue_span=Span(start=0, end=1),
            evidence_spans=(Span(start=0, end=1),),
        )
    with pytest.raises(ValidationError):
        OperationCandidate(
            id="op1",
            clause_id="c1",
            span=Span(start=0, end=1),
            kind="SEGMENT",
            target_mention_ids=("m1",),
            target_concept_ids=("CUSTOMER",),
            mentioned_arguments={"unsupported_path": ()},
            mode="EXECUTE",
            polarity="POSITIVE",
            evidence_spans=(Span(start=0, end=1),),
        )


def test_request_frame_does_not_duplicate_operation_modes() -> None:
    values = {
        "request_mode": "EXECUTE",
        "modality": "FACTUAL",
        "dialogue_act_provisional": "NEW",
        "dialogue_act_validated": "NEW",
        "operation_modes": ["EXECUTE"],
    }
    with pytest.raises(ValidationError):
        RequestFrame.model_validate(values)


def test_template_confirmation_protocol_requires_server_resolvable_selection_fields() -> None:
    with pytest.raises(ValidationError, match="template selection requires template_id and candidate_set_hash"):
        TemplateConfirmation(
            protocol_version="template-confirmation-v1",
            action="SELECT_TEMPLATE",
            current_turn_id="turn-1",
            plan_hash="plan-hash",
        )
    with pytest.raises(ValidationError, match="template_id and candidate_set_hash are only valid"):
        TemplateConfirmation(
            protocol_version="template-confirmation-v1",
            action="CONFIRM_PLAN",
            current_turn_id="turn-1",
            plan_hash="plan-hash",
            template_id="cohort_retention",
        )


def test_relation_endpoint_has_an_explicit_reference_type() -> None:
    assert MentionEndpoint(endpoint_type="mention_id", value="m1").endpoint_type == "mention_id"
    assert ConceptEndpoint(endpoint_type="concept_id", value="CUSTOMER").endpoint_type == "concept_id"
    assert RoleEndpoint(endpoint_type="role_id", value="r1").endpoint_type == "role_id"
    with pytest.raises(ValidationError):
        MentionEndpoint(endpoint_type="analysis_subject", value="CUSTOMER")


def test_v2_role_and_relation_fields_separate_role_names_from_endpoints() -> None:
    role = Role(
        id="r1",
        clause_id="c1",
        mention_id="m1",
        concept_id="CUSTOMER",
        name="analysis_subject",
    )
    relation = Relation(
        id="rel1",
        clause_id="c1",
        predicate="GROUP_BY",
        subject=RoleEndpoint(endpoint_type="role_id", value="r1"),
        object=ConceptEndpoint(endpoint_type="concept_id", value="PURCHASE_FREQUENCY"),
    )
    assert role.name == "analysis_subject"
    assert relation.subject.endpoint_type == "role_id"
    with pytest.raises(ValidationError):
        Relation(
            id="rel2",
            clause_id="c1",
            predicate="GROUP_BY",
            subject={"endpoint_type": "analysis_subject", "value": "CUSTOMER"},
            object={"endpoint_type": "concept_id", "value": "PURCHASE_FREQUENCY"},
        )
    with pytest.raises(ValidationError):
        Role(
            id="r2",
            clause_id="c1",
            mention_id="m1",
            concept_id="CUSTOMER",
            name="unregistered_role",
        )
    with pytest.raises(ValidationError):
        Relation(
            id="rel3",
            clause_id="c1",
            predicate="UNREGISTERED_PREDICATE",
            subject=MentionEndpoint(endpoint_type="mention_id", value="m1"),
            object=ConceptEndpoint(endpoint_type="concept_id", value="PURCHASE_FREQUENCY"),
        )

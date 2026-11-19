from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.context.semantic_catalog_v2 import DEFAULT_SEMANTIC_CATALOG
from app.context.semantic_contract_v2 import (
    Ambiguity,
    ArgumentNegation,
    ArgumentValue,
    Clause,
    ConceptEndpoint,
    ContextSource,
    IntentCandidate,
    Mention,
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
from app.context.semantic_contract_validator import (
    VALIDATOR_ISSUE_CODES,
    resolve_template_confirmation,
    validate_semantic_contract,
)


TURN_ID = "turn-current"


def _span(start: int, end: int) -> Span:
    return Span(start=start, end=end)


def _find(text: str, surface: str, start: int = 0, end: int | None = None) -> Span:
    location = text.index(surface, start, end)
    return _span(location, location + len(surface))


def _mention(
    text: str,
    spec: dict,
    clause_by_id: dict[str, Clause],
    default_turn_id: str = TURN_ID,
) -> Mention:
    clause = clause_by_id[spec["clause_id"]]
    span = (
        spec["span"]
        if "span" in spec
        else _find(text, spec["surface"], clause.span.start, clause.span.end)
    )
    return Mention(
        id=spec["id"],
        clause_id=spec["clause_id"],
        span=span,
        surface=spec["surface"],
        concept_id=spec["concept_id"],
        candidate_types=spec.get("candidate_types", ()),
        grounding_source=spec.get("grounding_source", "exact_alias"),
        source_turn_id=spec.get("source_turn_id", default_turn_id),
    )


def _contract(
    text: str,
    clause_ranges: tuple[tuple[int, int], ...],
    mention_specs: tuple[dict, ...],
    operation_specs: tuple[dict, ...],
    *,
    parse_status: str = "PARSED",
    domain: str = "BUSINESS_ANALYSIS",
    context_sources: tuple[dict, ...] = (),
    turn_id: str = TURN_ID,
    primary_intent: str | None = None,
) -> SemanticContractCandidateV21:
    clauses = tuple(
        Clause(id=f"c{index + 1}", span=_span(start, end))
        for index, (start, end) in enumerate(clause_ranges)
    )
    clause_by_id = {clause.id: clause for clause in clauses}
    mentions = tuple(_mention(text, spec, clause_by_id, turn_id) for spec in mention_specs)
    operations = []
    for spec in operation_specs:
        clause = clause_by_id[spec["clause_id"]]
        operation_span = _span(clause.span.start, clause.span.end)
        values = {}
        for path, argument_specs in spec.get("mentioned_arguments", {}).items():
            values[path] = tuple(
                ArgumentValue(
                    mention_id=value["mention_id"],
                    concept_id=value["concept_id"],
                    stance=value["stance"],
                    evidence_span=(
                        operation_span if value["stance"] == "ADOPTED" else None
                    ),
                )
                for value in argument_specs
            )
        argument_negations = tuple(
            ArgumentNegation(
                id=item.get("id", f"neg-{index + 1}"),
                source_clause_id=item["source_clause_id"],
                source_turn_id=item.get("source_turn_id", TURN_ID),
                cue_span=_find(
                    text,
                    item["cue"],
                    clause_by_id[item["source_clause_id"]].span.start,
                    clause_by_id[item["source_clause_id"]].span.end,
                ),
                argument_path=item["argument_path"],
                target_value_mention_id=item["target_value_mention_id"],
                evidence_spans=(
                    _find(
                        text,
                        item["cue"],
                        clause_by_id[item["source_clause_id"]].span.start,
                        clause_by_id[item["source_clause_id"]].span.end,
                    ),
                ),
            )
            for index, item in enumerate(spec.get("argument_negations", ()))
        )
        cue_span = _find(
            text,
            spec["cue"],
            clause.span.start,
            clause.span.end,
        ) if spec.get("polarity") == "NEGATIVE" else None
        operations.append(
            OperationCandidate(
                id=spec["id"],
                clause_id=spec["clause_id"],
                span=operation_span,
                kind=spec["kind"],
                target_mention_ids=tuple(spec["target_mention_ids"]),
                target_concept_ids=tuple(spec["target_concept_ids"]),
                mentioned_arguments=values,
                argument_negations=argument_negations,
                mode=spec.get("mode", "EXECUTE"),
                polarity=spec.get("polarity", "POSITIVE"),
                clause_scope=spec.get("clause_scope"),
                negation_cue_span=cue_span,
                source_clause_id=spec.get(
                    "source_clause_id",
                    spec["clause_id"] if spec.get("clause_scope") == "REQUEST" else None,
                ),
                source_turn_id=spec.get(
                    "source_turn_id",
                    turn_id if spec.get("clause_scope") == "REQUEST" else None,
                ),
                evidence_spans=(operation_span,),
            )
        )
    return SemanticContractCandidateV21(
        contract_version="semantic-contract-v2.1",
        turn_id=turn_id,
        parse_status=parse_status,
        domain=domain,
        clauses=clauses,
        mentions=mentions,
        operations=tuple(operations),
        request_frame=RequestFrame(
            request_mode="EXECUTE",
            modality="FACTUAL",
            dialogue_act_provisional="NEW",
            dialogue_act_validated="NEW",
        ),
        primary_intent=primary_intent,
        context_sources=tuple(ContextSource.model_validate(item) for item in context_sources),
        catalog_snapshot_hash=DEFAULT_SEMANTIC_CATALOG.snapshot_hash(),
    )


def _one_clause(text: str) -> tuple[tuple[int, int], ...]:
    return ((0, len(text)),)


def _two_clauses(text: str) -> tuple[tuple[int, int], ...]:
    delimiter = text.index("，")
    return ((0, delimiter + 1), (delimiter + 1, len(text)))


def _run(
    candidate: SemanticContractCandidateV21,
    text: str,
    *,
    trusted_prior_contract=None,
):
    return validate_semantic_contract(
        candidate,
        source_text=text,
        trusted_prior_contract=trusted_prior_contract,
    )


def test_not_do_lifecycle_is_a_negative_lifecycle_operation() -> None:
    text = "不做生命周期分析"
    candidate = _contract(
        text,
        _one_clause(text),
        ({"id": "m1", "clause_id": "c1", "surface": "生命周期", "concept_id": "LIFECYCLE"},),
        (
            {
                "id": "op1",
                "clause_id": "c1",
                "kind": "ANALYZE",
                "target_mention_ids": ("m1",),
                "target_concept_ids": ("LIFECYCLE",),
                "polarity": "NEGATIVE",
                "clause_scope": "SAME_CLAUSE",
                "cue": "不做",
            },
        ),
    )
    result = _run(candidate, text)
    assert result.contract.semantic_status == "CONFIRMED"
    assert result.contract.operations[0].polarity == "NEGATIVE"
    assert result.contract.operations[0].effective_arguments == {}


def test_postposed_cross_clause_negation_uses_request_scope() -> None:
    text = "做客户生命周期分析，不要做"
    candidate = _contract(
        text,
        _two_clauses(text),
        ({"id": "m_lifecycle", "clause_id": "c1", "surface": "生命周期", "concept_id": "LIFECYCLE"},),
        ({
            "id": "op_lifecycle", "clause_id": "c2", "kind": "ANALYZE",
            "target_mention_ids": ("m_lifecycle",), "target_concept_ids": ("LIFECYCLE",),
            "polarity": "NEGATIVE", "clause_scope": "REQUEST", "cue": "不要做",
        },),
    )

    result = _run(candidate, text)

    assert not result.issues
    assert result.contract.operations[0].kind == "ANALYZE"
    assert result.contract.operations[0].target_concept_ids == ("LIFECYCLE",)
    assert result.contract.operations[0].polarity == "NEGATIVE"
    assert result.contract.operations[0].clause_scope == "REQUEST"


def test_lifecycle_negative_does_not_erase_positive_customer_segmentation() -> None:
    text = "不做生命周期分析，按购买频次和金额给客户分层"
    ranges = _two_clauses(text)
    candidate = _contract(
        text,
        ranges,
        (
            {"id": "m_life", "clause_id": "c1", "surface": "生命周期", "concept_id": "LIFECYCLE"},
            {"id": "m_frequency", "clause_id": "c2", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_money", "clause_id": "c2", "surface": "金额", "concept_id": "MONETARY"},
            {"id": "m_customer", "clause_id": "c2", "surface": "客户", "concept_id": "CUSTOMER"},
        ),
        (
            {
                "id": "op_life",
                "clause_id": "c1",
                "kind": "ANALYZE",
                "target_mention_ids": ("m_life",),
                "target_concept_ids": ("LIFECYCLE",),
                "polarity": "NEGATIVE",
                "clause_scope": "SAME_CLAUSE",
                "cue": "不做",
            },
            {
                "id": "op_segment",
                "clause_id": "c2",
                "kind": "SEGMENT",
                "target_mention_ids": ("m_customer",),
                "target_concept_ids": ("CUSTOMER",),
                "mentioned_arguments": {
                    "basis": (
                        {"mention_id": "m_frequency", "concept_id": "PURCHASE_FREQUENCY", "stance": "ADOPTED"},
                        {"mention_id": "m_money", "concept_id": "MONETARY", "stance": "ADOPTED"},
                    ),
                },
            },
        ),
    )
    result = _run(candidate, text)
    assert result.contract.semantic_status == "CONFIRMED"
    assert [operation.polarity for operation in result.contract.operations] == ["NEGATIVE", "POSITIVE"]
    assert {value.concept_id for value in result.contract.operations[1].effective_arguments["basis"]} == {
        "PURCHASE_FREQUENCY",
        "MONETARY",
    }


def test_cross_clause_argument_negation_removes_only_monetary_basis() -> None:
    text = "不要用金额做分层依据，但仍然按购买频次给客户分层"
    ranges = _two_clauses(text)
    candidate = _contract(
        text,
        ranges,
        (
            {"id": "m_money", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
            {"id": "m_frequency", "clause_id": "c2", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_customer", "clause_id": "c2", "surface": "客户", "concept_id": "CUSTOMER"},
        ),
        (
            {
                "id": "op_segment",
                "clause_id": "c2",
                "kind": "SEGMENT",
                "target_mention_ids": ("m_customer",),
                "target_concept_ids": ("CUSTOMER",),
                "mentioned_arguments": {
                    "basis": (
                        {"mention_id": "m_money", "concept_id": "MONETARY", "stance": "EXCLUDED"},
                        {"mention_id": "m_frequency", "concept_id": "PURCHASE_FREQUENCY", "stance": "ADOPTED"},
                    ),
                },
                "argument_negations": (
                    {
                        "source_clause_id": "c1",
                        "cue": "不要",
                        "argument_path": "basis",
                        "target_value_mention_id": "m_money",
                    },
                ),
            },
        ),
    )
    result = _run(candidate, text)
    assert result.contract.semantic_status == "CONFIRMED"
    assert [value.concept_id for value in result.contract.operations[0].effective_arguments["basis"]] == [
        "PURCHASE_FREQUENCY"
    ]


def test_negative_sort_does_not_become_segment_basis_negation() -> None:
    text = "不要按金额排序，按购买频次给客户分层"
    candidate = _contract(
        text,
        _two_clauses(text),
        (
            {"id": "m_money", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
            {"id": "m_frequency", "clause_id": "c2", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_customer", "clause_id": "c2", "surface": "客户", "concept_id": "CUSTOMER"},
        ),
        (
            {
                "id": "op_sort",
                "clause_id": "c1",
                "kind": "SORT",
                "target_mention_ids": ("m_money",),
                "target_concept_ids": ("MONETARY",),
                "polarity": "NEGATIVE",
                "clause_scope": "SAME_CLAUSE",
                "cue": "不要",
            },
            {
                "id": "op_segment",
                "clause_id": "c2",
                "kind": "SEGMENT",
                "target_mention_ids": ("m_customer",),
                "target_concept_ids": ("CUSTOMER",),
                "mentioned_arguments": {
                    "basis": (
                        {"mention_id": "m_frequency", "concept_id": "PURCHASE_FREQUENCY", "stance": "ADOPTED"},
                    ),
                },
            },
        ),
    )
    result = _run(candidate, text)
    assert result.contract.semantic_status == "CONFIRMED"
    assert result.contract.operations[0].kind == "SORT"
    assert [value.concept_id for value in result.contract.operations[1].effective_arguments["basis"]] == [
        "PURCHASE_FREQUENCY"
    ]


def test_rfm_negative_and_cohort_replacement_are_separate_operations() -> None:
    text = "RFM 分群不做，改用 Cohort 留存分析"
    candidate = _contract(
        text,
        _two_clauses(text),
        (
            {"id": "m_rfm", "clause_id": "c1", "surface": "RFM", "concept_id": "RFM"},
            {"id": "m_cohort", "clause_id": "c2", "surface": "Cohort 留存分析", "concept_id": "COHORT_RETENTION"},
        ),
        (
            {
                "id": "op_rfm",
                "clause_id": "c1",
                "kind": "SEGMENT",
                "target_mention_ids": ("m_rfm",),
                "target_concept_ids": ("RFM",),
                "polarity": "NEGATIVE",
                "clause_scope": "SAME_CLAUSE",
                "cue": "不做",
            },
            {
                "id": "op_cohort",
                "clause_id": "c2",
                "kind": "TREND",
                "target_mention_ids": ("m_cohort",),
                "target_concept_ids": ("COHORT_RETENTION",),
            },
        ),
    )
    result = _run(candidate, text)
    assert result.contract.semantic_status == "CONFIRMED"
    assert [(item.kind, item.polarity) for item in result.contract.operations] == [
        ("SEGMENT", "NEGATIVE"),
        ("TREND", "POSITIVE"),
    ]


def test_rfm_positive_with_a_negative_other_analysis_operation() -> None:
    text = "RFM 客户分群，不要做别的分析"
    candidate = _contract(
        text,
        _two_clauses(text),
        (
            {"id": "m_rfm", "clause_id": "c1", "surface": "RFM", "concept_id": "RFM"},
            {"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},
            {"id": "m_other", "clause_id": "c2", "surface": "别的分析", "concept_id": "OTHER"},
        ),
        (
            {
                "id": "op_rfm",
                "clause_id": "c1",
                "kind": "SEGMENT",
                "target_mention_ids": ("m_customer",),
                "target_concept_ids": ("CUSTOMER",),
                "mentioned_arguments": {
                    "method": ({"mention_id": "m_rfm", "concept_id": "RFM", "stance": "ADOPTED"},),
                },
            },
            {
                "id": "op_other",
                "clause_id": "c2",
                "kind": "ANALYZE",
                "target_mention_ids": ("m_other",),
                "target_concept_ids": ("OTHER",),
                "polarity": "NEGATIVE",
                "clause_scope": "SAME_CLAUSE",
                "cue": "不要",
            },
        ),
    )
    result = _run(candidate, text)
    assert result.contract.semantic_status == "CONFIRMED"
    assert [(item.target_concept_ids, item.polarity) for item in result.contract.operations] == [
        (("CUSTOMER",), "POSITIVE"),
        (("OTHER",), "NEGATIVE"),
    ]
    assert [value.concept_id for value in result.contract.operations[0].effective_arguments["method"]] == ["RFM"]


def test_rfm_positive_without_customer_mention_uses_method_as_grounded_target_reference() -> None:
    text = "RFM 分群"
    candidate = _contract(
        text,
        _one_clause(text),
        ({"id": "m_rfm", "clause_id": "c1", "surface": "RFM", "concept_id": "RFM"},),
        ({
            "id": "op_rfm",
            "clause_id": "c1",
            "kind": "SEGMENT",
            "target_mention_ids": (),
            "target_concept_ids": ("CUSTOMER",),
            "mentioned_arguments": {
                "basis": (),
                "method": ({"mention_id": "m_rfm", "concept_id": "RFM", "stance": "ADOPTED"},),
            },
        },),
    )

    result = _run(candidate, text)

    assert result.contract.semantic_status == "CONFIRMED"
    operation = result.contract.operations[0]
    assert operation.target_mention_ids == ()
    assert operation.target_concept_ids == ("CUSTOMER",)
    assert [value.concept_id for value in operation.effective_arguments["method"]] == ["RFM"]
    assert operation.effective_arguments["basis"] == ()
    assert not result.issues


def test_implicit_customer_target_can_be_grounded_by_adopted_basis_argument() -> None:
    text = "按购买频次分层"
    candidate = _contract(
        text,
        _one_clause(text),
        ({"id": "m_frequency", "clause_id": "c1", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},),
        ({
            "id": "op_segment",
            "clause_id": "c1",
            "kind": "SEGMENT",
            "target_mention_ids": (),
            "target_concept_ids": ("CUSTOMER",),
            "mentioned_arguments": {
                "basis": ({"mention_id": "m_frequency", "concept_id": "PURCHASE_FREQUENCY", "stance": "ADOPTED"},),
            },
        },),
    )

    result = _run(candidate, text)

    assert result.contract.semantic_status == "CONFIRMED"
    operation = result.contract.operations[0]
    assert operation.target_mention_ids == ()
    assert operation.target_concept_ids == ("CUSTOMER",)
    assert [value.concept_id for value in operation.effective_arguments["basis"]] == ["PURCHASE_FREQUENCY"]
    assert not result.issues


def test_cross_clause_replacement_keeps_only_final_operation_in_replacement_clause() -> None:
    text = "按金额给客户分层，但改按购买频次"
    split = text.index("，") + 1
    candidate = _contract(
        text,
        ((0, split), (split, len(text))),
        ({"id": "m_frequency", "clause_id": "c2", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},),
        ({
            "id": "op_segment",
            "clause_id": "c2",
            "kind": "SEGMENT",
            "target_mention_ids": (),
            "target_concept_ids": ("CUSTOMER",),
            "mentioned_arguments": {
                "basis": ({"mention_id": "m_frequency", "concept_id": "PURCHASE_FREQUENCY", "stance": "ADOPTED"},),
            },
        },),
    )

    result = _run(candidate, text)

    assert result.contract.semantic_status == "CONFIRMED"
    assert len(result.contract.operations) == 1
    operation = result.contract.operations[0]
    assert operation.clause_id == "c2"
    assert operation.kind == "SEGMENT"
    assert operation.target_concept_ids == ("CUSTOMER",)
    assert [value.concept_id for value in operation.effective_arguments["basis"]] == ["PURCHASE_FREQUENCY"]
    assert not operation.argument_negations
    assert not result.issues


@pytest.mark.parametrize(
    ("stance", "argument_concept_id"),
    [
        ("EXCLUDED", "PURCHASE_FREQUENCY"),
        ("MENTIONED_ONLY", "PURCHASE_FREQUENCY"),
        ("ADOPTED", "RECENCY"),
    ],
)
def test_empty_target_requires_an_adopted_grounded_argument_reference(
    stance: str, argument_concept_id: str,
) -> None:
    text = "购买频次"
    candidate = _contract(
        text,
        _one_clause(text),
        ({"id": "m_frequency", "clause_id": "c1", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},),
        ({
            "id": "op_segment",
            "clause_id": "c1",
            "kind": "SEGMENT",
            "target_mention_ids": (),
            "target_concept_ids": ("CUSTOMER",),
            "mentioned_arguments": {
                "basis": ({"mention_id": "m_frequency", "concept_id": argument_concept_id, "stance": stance},),
            },
        },),
    )

    result = _run(candidate, text)

    assert "missing_target_reference" in {issue.code for issue in result.issues}


def test_method_argument_must_reference_a_registered_method_concept() -> None:
    text = "RFM 客户分群"
    candidate = _contract(
        text,
        _one_clause(text),
        (
            {"id": "m_rfm", "clause_id": "c1", "surface": "RFM", "concept_id": "RFM"},
            {"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},
        ),
        ({
            "id": "op_rfm",
            "clause_id": "c1",
            "kind": "SEGMENT",
            "target_mention_ids": ("m_customer",),
            "target_concept_ids": ("CUSTOMER",),
            "mentioned_arguments": {
                "method": ({"mention_id": "m_customer", "concept_id": "CUSTOMER", "stance": "ADOPTED"},),
            },
        },),
    )

    result = _run(candidate, text)

    assert any(
        issue.code == "unknown_concept_reference"
        and issue.location.endswith("mentioned_arguments.method.0.concept_id")
        for issue in result.issues
    )


def test_empty_target_mentions_without_method_fail_schema_validation() -> None:
    text = "RFM 分群"
    with pytest.raises(ValidationError, match="missing_target_reference"):
        _contract(
            text,
            _one_clause(text),
            ({"id": "m_rfm", "clause_id": "c1", "surface": "RFM", "concept_id": "RFM"},),
            ({
                "id": "op_rfm",
                "clause_id": "c1",
                "kind": "SEGMENT",
                "target_mention_ids": (),
                "target_concept_ids": ("CUSTOMER",),
            },),
        )


def test_empty_target_mentions_without_any_argument_fail_schema_validation() -> None:
    text = "不做 RFM"
    with pytest.raises(ValidationError, match="missing_target_reference"):
        _contract(
            text,
            _one_clause(text),
            (),
            ({
                "id": "op_negative_rfm",
                "clause_id": "c1",
                "kind": "SEGMENT",
                "target_mention_ids": (),
                "target_concept_ids": ("RFM",),
                "polarity": "NEGATIVE",
                "clause_scope": "SAME_CLAUSE",
                "cue": "不做",
            },),
        )


def test_empty_target_method_reference_must_be_a_registered_analysis_method() -> None:
    text = "客户 RFM 分群"
    candidate = _contract(
        text,
        _one_clause(text),
        ({"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},),
        ({
            "id": "op_segment",
            "clause_id": "c1",
            "kind": "SEGMENT",
            "target_mention_ids": (),
            "target_concept_ids": ("CUSTOMER",),
            "mentioned_arguments": {
                "method": ({"mention_id": "m_customer", "concept_id": "CUSTOMER", "stance": "ADOPTED"},),
            },
        },),
    )

    result = _run(candidate, text)

    assert "missing_target_reference" in {issue.code for issue in result.issues}
    assert "unknown_concept_reference" in {issue.code for issue in result.issues}


def test_reminder_wording_is_represented_as_positive_evidence() -> None:
    text = "不要忘记按购买频次和金额给客户分层"
    candidate = _contract(
        text,
        _one_clause(text),
        (
            {"id": "m_frequency", "clause_id": "c1", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_money", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
            {"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},
        ),
        (
            {
                "id": "op_segment",
                "clause_id": "c1",
                "kind": "SEGMENT",
                "target_mention_ids": ("m_customer",),
                "target_concept_ids": ("CUSTOMER",),
                "mentioned_arguments": {
                    "basis": (
                        {"mention_id": "m_frequency", "concept_id": "PURCHASE_FREQUENCY", "stance": "ADOPTED"},
                        {"mention_id": "m_money", "concept_id": "MONETARY", "stance": "ADOPTED"},
                    ),
                },
            },
        ),
    )
    result = _run(candidate, text)
    assert result.contract.semantic_status == "CONFIRMED"
    assert result.contract.operations[0].polarity == "POSITIVE"
    assert len(result.contract.operations[0].effective_arguments["basis"]) == 2


def test_effective_arguments_can_be_empty_after_explicit_exclusion() -> None:
    text = "不要用金额做分层依据，给客户分层"
    candidate = _contract(
        text,
        _two_clauses(text),
        (
            {"id": "m_money", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
            {"id": "m_customer", "clause_id": "c2", "surface": "客户", "concept_id": "CUSTOMER"},
        ),
        (
            {
                "id": "op_segment",
                "clause_id": "c2",
                "kind": "SEGMENT",
                "target_mention_ids": ("m_customer",),
                "target_concept_ids": ("CUSTOMER",),
                "mentioned_arguments": {
                    "basis": (
                        {"mention_id": "m_money", "concept_id": "MONETARY", "stance": "EXCLUDED"},
                    ),
                },
                "argument_negations": (
                    {
                        "source_clause_id": "c1",
                        "cue": "不要",
                        "argument_path": "basis",
                        "target_value_mention_id": "m_money",
                    },
                ),
            },
        ),
    )
    result = _run(candidate, text)
    assert result.contract.operations[0].effective_arguments["basis"] == ()


def test_mentioned_only_argument_is_not_treated_as_adopted() -> None:
    text = "提到金额，给客户分层"
    candidate = _contract(
        text,
        _two_clauses(text),
        (
            {"id": "m_money", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
            {"id": "m_customer", "clause_id": "c2", "surface": "客户", "concept_id": "CUSTOMER"},
        ),
        (
            {
                "id": "op_segment",
                "clause_id": "c2",
                "kind": "SEGMENT",
                "target_mention_ids": ("m_customer",),
                "target_concept_ids": ("CUSTOMER",),
                "mentioned_arguments": {
                    "basis": (
                        {"mention_id": "m_money", "concept_id": "MONETARY", "stance": "MENTIONED_ONLY"},
                    ),
                },
            },
        ),
    )
    result = _run(candidate, text)
    assert result.contract.operations[0].effective_arguments["basis"] == ()


def test_current_clause_can_explicitly_exclude_an_inherited_argument_mention() -> None:
    text = "不要用它做分层依据，但按购买频次给客户分层"
    ranges = _two_clauses(text)
    prior_text = "按金额给客户分层"
    prior_candidate = _contract(
        prior_text,
        _one_clause(prior_text),
        (
            {"id": "m_money", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
        ),
        (
            {
                "id": "op_prior_segment",
                "clause_id": "c1",
                "kind": "SEGMENT",
                "target_mention_ids": ("m_money",),
                "target_concept_ids": ("MONETARY",),
                "mentioned_arguments": {
                    "basis": (
                        {"mention_id": "m_money", "concept_id": "MONETARY", "stance": "ADOPTED"},
                    ),
                },
            },
        ),
        turn_id="turn-prior",
    )
    trusted_prior_contract = _run(prior_candidate, prior_text).contract
    candidate = _contract(
        text,
        ranges,
        (
            {
                "id": "m_money",
                "clause_id": "c1",
                "surface": "金额",
                "concept_id": "MONETARY",
                "grounding_source": "inherited",
                "source_turn_id": "turn-prior",
                "span": None,
            },
            {"id": "m_frequency", "clause_id": "c2", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_customer", "clause_id": "c2", "surface": "客户", "concept_id": "CUSTOMER"},
        ),
        (
            {
                "id": "op_segment",
                "clause_id": "c2",
                "kind": "SEGMENT",
                "target_mention_ids": ("m_customer",),
                "target_concept_ids": ("CUSTOMER",),
                "mentioned_arguments": {
                    "basis": (
                        {"mention_id": "m_money", "concept_id": "MONETARY", "stance": "EXCLUDED"},
                        {"mention_id": "m_frequency", "concept_id": "PURCHASE_FREQUENCY", "stance": "ADOPTED"},
                    ),
                },
                "argument_negations": (
                    {
                        "source_clause_id": "c1",
                        "cue": "不要",
                        "argument_path": "basis",
                        "target_value_mention_id": "m_money",
                    },
                ),
            },
        ),
        context_sources=(
            {"source_turn_id": "turn-prior", "mention_ids": ("m_money",), "relation": "INHERITED_CONTEXT"},
        ),
    )
    result = _run(candidate, text, trusted_prior_contract=trusted_prior_contract)
    assert result.contract.semantic_status == "CONFIRMED"
    assert [value.concept_id for value in result.contract.operations[0].effective_arguments["basis"]] == [
        "PURCHASE_FREQUENCY"
    ]


def test_inherited_mention_requires_a_trusted_prior_contract() -> None:
    text = "按购买频次给客户分层"
    candidate = _contract(
        text,
        _one_clause(text),
        (
            {
                "id": "m_money",
                "clause_id": "c1",
                "surface": "金额",
                "concept_id": "MONETARY",
                "grounding_source": "inherited",
                "source_turn_id": "turn-prior",
                "span": None,
            },
            {"id": "m_frequency", "clause_id": "c1", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},
        ),
        (
            {
                "id": "op_segment",
                "clause_id": "c1",
                "kind": "SEGMENT",
                "target_mention_ids": ("m_customer",),
                "target_concept_ids": ("CUSTOMER",),
                "mentioned_arguments": {
                    "basis": (
                        {"mention_id": "m_frequency", "concept_id": "PURCHASE_FREQUENCY", "stance": "ADOPTED"},
                    ),
                },
            },
        ),
        context_sources=(
            {"source_turn_id": "turn-prior", "mention_ids": ("m_money",), "relation": "INHERITED_CONTEXT"},
        ),
    )
    result = _run(candidate, text)
    assert result.contract.semantic_status == "MISSING"
    assert any(issue.code == "untrusted_inherited_context_reference" for issue in result.issues)


def test_argument_exclusion_only_applies_to_its_argument_path() -> None:
    text = "不要把金额作为字段，同时将金额用作筛选值"
    candidate = _contract(
        text,
        _one_clause(text),
        (
            {"id": "m_money", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
        ),
        (
            {
                "id": "op_filter",
                "clause_id": "c1",
                "kind": "FILTER",
                "target_mention_ids": ("m_money",),
                "target_concept_ids": ("MONETARY",),
                "mentioned_arguments": {
                    "field": (
                        {"mention_id": "m_money", "concept_id": "MONETARY", "stance": "EXCLUDED"},
                    ),
                    "value": (
                        {"mention_id": "m_money", "concept_id": "MONETARY", "stance": "ADOPTED"},
                    ),
                },
                "argument_negations": (
                    {
                        "source_clause_id": "c1",
                        "cue": "不要",
                        "argument_path": "field",
                        "target_value_mention_id": "m_money",
                    },
                ),
            },
        ),
    )
    result = _run(candidate, text)
    assert result.contract.semantic_status == "CONFIRMED"
    effective = result.contract.operations[0].effective_arguments
    assert effective["field"] == ()
    assert [item.mention_id for item in effective["value"]] == ["m_money"]


def test_validator_rejects_argument_path_not_allowed_for_operation_kind() -> None:
    text = "按购买频次給客户分层"
    candidate = _contract(
        text,
        _one_clause(text),
        (
            {"id": "m_frequency", "clause_id": "c1", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},
        ),
        (
            {
                "id": "op_segment",
                "clause_id": "c1",
                "kind": "SEGMENT",
                "target_mention_ids": ("m_customer",),
                "target_concept_ids": ("CUSTOMER",),
                "mentioned_arguments": {
                    "key": (
                        {"mention_id": "m_frequency", "concept_id": "PURCHASE_FREQUENCY", "stance": "ADOPTED"},
                    ),
                },
            },
        ),
    )
    result = _run(candidate, text)
    assert result.contract.semantic_status == "MISSING"
    assert any(issue.code == "invalid_argument_path" for issue in result.issues)


def test_same_clause_positive_and_negative_argument_assertions_are_conflict() -> None:
    text = "按金额给客户分层，但排除金额"
    candidate = _contract(
        text,
        _one_clause(text),
        (
            {"id": "m_money_pos", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
            {"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},
        ),
        (
            {
                "id": "op_segment",
                "clause_id": "c1",
                "kind": "SEGMENT",
                "target_mention_ids": ("m_customer",),
                "target_concept_ids": ("CUSTOMER",),
                "mentioned_arguments": {
                    "basis": (
                        {"mention_id": "m_money_pos", "concept_id": "MONETARY", "stance": "ADOPTED"},
                        {"mention_id": "m_money_pos", "concept_id": "MONETARY", "stance": "EXCLUDED"},
                    ),
                },
                "argument_negations": (
                    {
                        "source_clause_id": "c1",
                        "cue": "排除",
                        "argument_path": "basis",
                        "target_value_mention_id": "m_money_pos",
                    },
                ),
            },
        ),
    )
    result = _run(candidate, text)
    assert result.contract.semantic_status == "CONFLICT"
    assert any(issue.code == "positive_negative_argument_conflict" for issue in result.issues)


def test_same_clause_positive_and_negative_operations_on_same_target_are_conflict() -> None:
    text = "按金额排序，也不要按金额排序"
    candidate = _contract(
        text,
        _one_clause(text),
        ({"id": "m_money", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},),
        (
            {
                "id": "op_positive",
                "clause_id": "c1",
                "kind": "SORT",
                "target_mention_ids": ("m_money",),
                "target_concept_ids": ("MONETARY",),
            },
            {
                "id": "op_negative",
                "clause_id": "c1",
                "kind": "SORT",
                "target_mention_ids": ("m_money",),
                "target_concept_ids": ("MONETARY",),
                "polarity": "NEGATIVE",
                "clause_scope": "SAME_CLAUSE",
                "cue": "不要",
            },
        ),
    )
    result = _run(candidate, text)
    assert result.contract.semantic_status == "CONFLICT"
    assert any(issue.code == "positive_negative_operation_conflict" for issue in result.issues)


def test_cross_turn_argument_negation_requires_clarification() -> None:
    text = "不要用金额做分层依据，但仍按购买频次给客户分层"
    ranges = _two_clauses(text)
    candidate = _contract(
        text,
        ranges,
        (
            {"id": "m_money", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
            {"id": "m_frequency", "clause_id": "c2", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_customer", "clause_id": "c2", "surface": "客户", "concept_id": "CUSTOMER"},
        ),
        (
            {
                "id": "op_segment",
                "clause_id": "c2",
                "kind": "SEGMENT",
                "target_mention_ids": ("m_customer",),
                "target_concept_ids": ("CUSTOMER",),
                "mentioned_arguments": {
                    "basis": (
                        {"mention_id": "m_money", "concept_id": "MONETARY", "stance": "EXCLUDED"},
                        {"mention_id": "m_frequency", "concept_id": "PURCHASE_FREQUENCY", "stance": "ADOPTED"},
                    ),
                },
                "argument_negations": (
                    {
                        "source_clause_id": "c1",
                        "source_turn_id": "turn-prior",
                        "cue": "不要",
                        "argument_path": "basis",
                        "target_value_mention_id": "m_money",
                    },
                ),
            },
        ),
    )
    result = _run(candidate, text)
    assert result.contract.semantic_status == "AMBIGUOUS"
    assert any(
        issue.code == "cross_turn_argument_negation_requires_clarification"
        for issue in result.issues
    )


def test_clause_partition_must_cover_the_source_without_gaps() -> None:
    text = "不做生命周期分析"
    candidate = _contract(
        text,
        ((0, 4), (5, len(text))),
        ({"id": "m1", "clause_id": "c1", "surface": "生命", "concept_id": "LIFECYCLE", "grounding_source": "llm_inferred"},),
        (),
        parse_status="INCOMPLETE",
    )
    result = _run(candidate, text)
    assert result.contract.semantic_status == "MISSING"
    assert any(issue.code == "invalid_clause_partition" for issue in result.issues)


def test_span_and_role_relation_references_are_validated() -> None:
    text = "按购买频次给客户分层"
    candidate = _contract(
        text,
        _one_clause(text),
        (
            {"id": "m_frequency", "clause_id": "c1", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},
        ),
        (
            {
                "id": "op_segment",
                "clause_id": "c1",
                "kind": "SEGMENT",
                "target_mention_ids": ("m_customer",),
                "target_concept_ids": ("CUSTOMER",),
                "mentioned_arguments": {
                    "basis": (
                        {"mention_id": "m_frequency", "concept_id": "PURCHASE_FREQUENCY", "stance": "ADOPTED"},
                    ),
                },
            },
        ),
    )
    candidate = candidate.model_copy(
        update={
            "mentions": tuple(
                item.model_copy(update={"span": _span(0, 4)}) if item.id == "m_customer" else item
                for item in candidate.mentions
            ),
            "roles": (
                Role(
                    id="r_customer",
                    clause_id="c1",
                    mention_id="m_customer",
                    concept_id="CUSTOMER",
                    name="analysis_subject",
                ),
            ),
            "relations": (
                Relation(
                    id="rel1",
                    clause_id="c1",
                    predicate="GROUP_BY",
                    subject=RoleEndpoint(endpoint_type="role_id", value="r_customer"),
                    object=ConceptEndpoint(endpoint_type="concept_id", value="PURCHASE_FREQUENCY"),
                ),
                Relation(
                    id="rel2",
                    clause_id="c1",
                    predicate="FILTER_BY",
                    subject=MentionEndpoint(endpoint_type="mention_id", value="missing"),
                    object=ConceptEndpoint(endpoint_type="concept_id", value="CUSTOMER"),
                ),
            ),
        }
    )
    result = _run(candidate, text)
    assert result.contract.semantic_status == "MISSING"
    assert any(issue.code == "span_surface_mismatch" for issue in result.issues)
    assert any(issue.code == "unknown_mention_reference" for issue in result.issues)


def test_explicit_llm_grounding_is_allowed_but_validator_never_creates_mentions() -> None:
    text = "按购频给客户分层"
    candidate = _contract(
        text,
        _one_clause(text),
        (
            {"id": "m_frequency", "clause_id": "c1", "surface": "购频", "concept_id": "PURCHASE_FREQUENCY", "grounding_source": "llm_inferred"},
            {"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},
        ),
        (
            {
                "id": "op_segment",
                "clause_id": "c1",
                "kind": "SEGMENT",
                "target_mention_ids": ("m_customer",),
                "target_concept_ids": ("CUSTOMER",),
                "mentioned_arguments": {
                    "basis": (
                        {"mention_id": "m_frequency", "concept_id": "PURCHASE_FREQUENCY", "stance": "ADOPTED"},
                    ),
                },
            },
        ),
    )
    result = _run(candidate, text)
    assert result.contract.semantic_status == "CONFIRMED"
    assert {mention.id for mention in result.contract.mentions} == {"m_frequency", "m_customer"}

    incomplete = candidate.model_copy(
        update={
            "mentions": tuple(item for item in candidate.mentions if item.id != "m_frequency"),
        }
    )
    missing_result = _run(incomplete, text)
    assert missing_result.contract.semantic_status == "MISSING"
    assert {mention.id for mention in missing_result.contract.mentions} == {"m_customer"}


@pytest.mark.parametrize(
    ("parse_status", "expected"),
    (("PARSED", "CONFIRMED"), ("AMBIGUOUS", "AMBIGUOUS"), ("UNSUPPORTED", "UNSUPPORTED"), ("INCOMPLETE", "MISSING")),
)
def test_parse_status_has_one_validator_mapping(parse_status: str, expected: str) -> None:
    text = "不做生命周期分析"
    candidate = _contract(
        text,
        _one_clause(text),
        ({"id": "m1", "clause_id": "c1", "surface": "生命周期", "concept_id": "LIFECYCLE"},),
        (
            {
                "id": "op1",
                "clause_id": "c1",
                "kind": "ANALYZE",
                "target_mention_ids": ("m1",),
                "target_concept_ids": ("LIFECYCLE",),
                "polarity": "NEGATIVE",
                "clause_scope": "SAME_CLAUSE",
                "cue": "不做",
            },
        ),
        parse_status=parse_status,
    )
    assert _run(candidate, text).contract.semantic_status == expected


def test_primary_intent_mismatch_is_audit_only_and_does_not_gate_contract() -> None:
    text = "客户分层"
    candidate = _contract(
        text,
        _one_clause(text),
        ({"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},),
        ({"id": "op1", "clause_id": "c1", "kind": "SEGMENT", "target_mention_ids": ("m_customer",), "target_concept_ids": ("CUSTOMER",)},),
        primary_intent="COHORT_RETENTION",
    )
    result = _run(candidate, text)
    assert result.contract.semantic_status == "CONFIRMED"
    assert not any(issue.code == "primary_intent_operation_mismatch" for issue in result.issues)
    audit_issue = next(
        issue for issue in result.audit_issues
        if issue.code == "primary_intent_operation_mismatch"
    )
    assert audit_issue.severity == "CONFLICT"


def test_same_clause_negative_operation_cannot_target_another_clause() -> None:
    text = "不要做分析，生命周期"
    candidate = _contract(
        text,
        _two_clauses(text),
        ({"id": "m_lifecycle", "clause_id": "c2", "surface": "生命周期", "concept_id": "LIFECYCLE"},),
        ({
            "id": "op_negative", "clause_id": "c1", "kind": "ANALYZE",
            "target_mention_ids": ("m_lifecycle",), "target_concept_ids": ("LIFECYCLE",),
            "polarity": "NEGATIVE", "clause_scope": "SAME_CLAUSE", "cue": "不要",
        },),
    )
    result = _run(candidate, text)
    assert result.contract.semantic_status == "CONFLICT"
    assert any(issue.code == "negative_target_outside_owned_clause" for issue in result.issues)


def test_request_scope_negative_operation_requires_explicit_source_binding() -> None:
    text = "不要做生命周期分析"
    missing = _contract(
        text,
        _one_clause(text),
        ({"id": "m1", "clause_id": "c1", "surface": "生命周期", "concept_id": "LIFECYCLE"},),
        ({
            "id": "op1", "clause_id": "c1", "kind": "ANALYZE",
            "target_mention_ids": ("m1",), "target_concept_ids": ("LIFECYCLE",),
            "polarity": "NEGATIVE", "clause_scope": "REQUEST", "cue": "不要",
            "source_clause_id": None, "source_turn_id": None,
        },),
    )
    missing_result = _run(missing, text)
    assert missing_result.contract.semantic_status == "MISSING"
    assert any(issue.code == "negative_operation_source_missing" for issue in missing_result.issues)

    explicit = missing.model_copy(update={
        "operations": (missing.operations[0].model_copy(update={"source_clause_id": "c1", "source_turn_id": TURN_ID}),),
    })
    explicit_result = _run(explicit, text)
    assert explicit_result.contract.semantic_status == "CONFIRMED"
    assert not any(issue.code == "negative_operation_source_missing" for issue in explicit_result.issues)


def test_request_scope_negative_operation_cross_clause_requires_explicit_source_binding() -> None:
    text = "不要做分析，生命周期"
    clauses = _two_clauses(text)
    mentions = ({"id": "m_lifecycle", "clause_id": "c2", "surface": "生命周期", "concept_id": "LIFECYCLE"},)
    operation = {
        "id": "op_negative",
        "clause_id": "c1",
        "kind": "ANALYZE",
        "target_mention_ids": ("m_lifecycle",),
        "target_concept_ids": ("LIFECYCLE",),
        "polarity": "NEGATIVE",
        "clause_scope": "REQUEST",
        "cue": "不要",
        "source_clause_id": None,
        "source_turn_id": None,
    }
    missing = _contract(text, clauses, mentions, (operation,))
    missing_result = _run(missing, text)
    assert missing_result.contract.semantic_status == "MISSING"
    assert any(issue.code == "negative_operation_source_missing" for issue in missing_result.issues)

    explicitly_bound = _contract(
        text,
        clauses,
        mentions,
        ({**operation, "source_clause_id": "c1", "source_turn_id": TURN_ID},),
    )
    bound_result = _run(explicitly_bound, text)
    assert bound_result.contract.semantic_status == "CONFIRMED"
    assert not any(issue.code == "negative_operation_source_missing" for issue in bound_result.issues)


def test_same_turn_argument_negation_can_cross_clauses_but_cross_turn_requires_clarification() -> None:
    same_text = "按金额分层，不要金额"
    same = _contract(
        same_text,
        _one_clause(same_text),
        (
            {"id": "m_money", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
            {"id": "m_customer", "clause_id": "c1", "surface": "分层", "concept_id": "CUSTOMER", "grounding_source": "llm_inferred"},
        ),
        ({
            "id": "op1", "clause_id": "c1", "kind": "SEGMENT", "target_mention_ids": ("m_customer",),
            "target_concept_ids": ("CUSTOMER",),
            "mentioned_arguments": {"basis": ({"mention_id": "m_money", "concept_id": "MONETARY", "stance": "EXCLUDED"},)},
            "argument_negations": ({"source_clause_id": "c1", "cue": "不要", "argument_path": "basis", "target_value_mention_id": "m_money"},),
        },),
    )
    same_result = _run(same, same_text)
    assert not same_result.issues

    cross_text = "按购买频次给客户分层，金额，不要使用它"
    first = cross_text.index("，") + 1
    second = cross_text.index("，", first) + 1
    ranges = ((0, first), (first, second), (second, len(cross_text)))
    cross = _contract(
        cross_text,
        ranges,
        (
            {"id": "m_frequency", "clause_id": "c1", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},
            {"id": "m_money", "clause_id": "c2", "surface": "金额", "concept_id": "MONETARY"},
        ),
        ({
            "id": "op1", "clause_id": "c1", "kind": "SEGMENT", "target_mention_ids": ("m_customer",),
            "target_concept_ids": ("CUSTOMER",),
            "mentioned_arguments": {"basis": (
                {"mention_id": "m_frequency", "concept_id": "PURCHASE_FREQUENCY", "stance": "ADOPTED"},
                {"mention_id": "m_money", "concept_id": "MONETARY", "stance": "EXCLUDED"},
            )},
            "argument_negations": ({"source_clause_id": "c3", "cue": "不要", "argument_path": "basis", "target_value_mention_id": "m_money"},),
        },),
    )
    cross_result = _run(cross, cross_text)
    assert not cross_result.issues
    assert cross_result.contract.semantic_status == "CONFIRMED"


def test_sealed_route_decision_rejects_forged_or_tampered_payload() -> None:
    from app.context.semantic_catalog_v2 import DEFAULT_SEMANTIC_CATALOG
    from app.context.semantic_contract_v2 import CapabilityDecisionBasis, CapabilityReference, SealedRouteDecision
    from app.complex_analysis.service import _sealed_template_from_route_decision

    with pytest.raises(ValidationError):
        SealedRouteDecision.model_validate({"route_id": "customer_analysis", "template_id": "cohort_retention"})
    with pytest.raises(ValueError, match="typed server-sealed"):
        _sealed_template_from_route_decision({"route_decision_status": "SELECTED", "template_id": "cohort_retention"})
    capability = next(item for item in DEFAULT_SEMANTIC_CATALOG.capabilities if item.capability_id == "customer_cohort_retention")
    catalog_hash = DEFAULT_SEMANTIC_CATALOG.snapshot_hash()
    basis = CapabilityDecisionBasis(
        contract_version="semantic-contract-v2.1",
        catalog_snapshot_hash=catalog_hash,
        operation_ids=("op1",),
        clause_ids=("c1",),
        capability_ids=(CapabilityReference(capability_id=capability.capability_id),),
        effective_arguments_digest="a" * 64,
    )
    sealed = SealedRouteDecision.issue(
        contract_version="semantic-contract-v2.1",
        catalog_snapshot_hash=catalog_hash,
        route_decision_status="SELECTED",
        route_id=capability.route_id,
        capability_id=capability.capability_id,
        template_id=capability.template_id,
        operation_ids=("op1",),
        clause_ids=("c1",),
        effective_arguments_digest="a" * 64,
        decision_basis=basis,
    )
    assert sealed.verify_seal()
    with pytest.raises(ValueError, match="integrity check failed"):
        _sealed_template_from_route_decision(sealed.model_copy(update={"template_id": "rfm_segmentation"}))


def test_plan_creation_without_a_sealed_decision_fails_closed() -> None:
    from app.complex_analysis import service

    assert not hasattr(service, "matched_templates")
    assert not hasattr(service, "matched_chat_templates")
    with pytest.raises(ValueError, match="sealed route_decision is required"):
        service.create_plan_v2(None, "RFM 客户分群", {}, route_decision=None)
    with pytest.raises(ValueError, match="server-sealed route_decision is required"):
        service.create_plan(None, "RFM 客户分群", {})


def test_template_confirmation_checks_hash_then_current_turn_negative() -> None:
    text = "不做生命周期分析"
    negative_candidate = _contract(
        text,
        _one_clause(text),
        ({"id": "m1", "clause_id": "c1", "surface": "生命周期", "concept_id": "LIFECYCLE"},),
        (
            {
                "id": "op1",
                "clause_id": "c1",
                "kind": "ANALYZE",
                "target_mention_ids": ("m1",),
                "target_concept_ids": ("LIFECYCLE",),
                "polarity": "NEGATIVE",
                "clause_scope": "SAME_CLAUSE",
                "cue": "不做",
            },
        ),
    )
    current_contract = _run(negative_candidate, text).contract
    confirmation = TemplateConfirmation(
        protocol_version="template-confirmation-v1",
        action="CONFIRM_PLAN",
        current_turn_id=TURN_ID,
        plan_hash="hash-1",
    )
    template_ids = frozenset({"cohort_retention"})
    resolution_context = {
        "expected_current_turn_id": TURN_ID,
        "current_turn_contract": current_contract,
        "has_user_text": True,
        "plan_template_id": "cohort_retention",
        "visible_template_ids": template_ids,
        "authorized_template_ids": template_ids,
        "allow_protocol_resolution": True,
    }
    assert resolve_template_confirmation(
        confirmation,
        expected_plan_hash="stale-hash",
        **resolution_context,
    ) == "STALE_PLAN_HASH"
    assert resolve_template_confirmation(
        confirmation,
        expected_plan_hash="hash-1",
        **resolution_context,
    ) == "BLOCKED_BY_CURRENT_TURN_NEGATIVE"
    button_only = confirmation.model_copy(update={"current_turn_id": "turn-button"})
    assert resolve_template_confirmation(
        button_only,
        expected_plan_hash="hash-1",
        expected_current_turn_id="turn-button",
        current_turn_contract=None,
        has_user_text=False,
        plan_template_id="cohort_retention",
        visible_template_ids=template_ids,
        authorized_template_ids=template_ids,
        allow_protocol_resolution=True,
    ) == "CONFIRMED"
    assert resolve_template_confirmation(
        button_only,
        expected_plan_hash="hash-1",
        expected_current_turn_id="turn-button",
        current_turn_contract=None,
        has_user_text=True,
        plan_template_id="cohort_retention",
        visible_template_ids=template_ids,
        authorized_template_ids=template_ids,
        allow_protocol_resolution=True,
    ) == "NEEDS_SEMANTIC_VALIDATION"
    assert resolve_template_confirmation(
        confirmation,
        expected_plan_hash="hash-1",
        **{**resolution_context, "has_user_text": False},
    ) == "BLOCKED_BY_CURRENT_TURN_NEGATIVE"
    stale_turn = confirmation.model_copy(update={"current_turn_id": "turn-other"})
    assert resolve_template_confirmation(
        stale_turn,
        expected_plan_hash="hash-1",
        **{**resolution_context, "has_user_text": False},
    ) == "STALE_CURRENT_TURN"


def test_template_selection_checks_server_candidate_hash_visibility_and_authorization() -> None:
    selection = TemplateConfirmation(
        protocol_version="template-confirmation-v1",
        action="SELECT_TEMPLATE",
        current_turn_id=TURN_ID,
        plan_hash="plan-hash",
        template_id="cohort_retention",
        candidate_set_hash="candidate-hash",
    )
    context = {
        "expected_plan_hash": "plan-hash",
        "expected_current_turn_id": TURN_ID,
        "current_turn_contract": None,
        "has_user_text": False,
        "plan_template_id": "cohort_retention",
        "candidate_template_ids": ("cohort_retention", "rfm_segmentation"),
        "expected_candidate_set_hash": "candidate-hash",
        "visible_template_ids": frozenset({"cohort_retention", "rfm_segmentation"}),
        "authorized_template_ids": frozenset({"cohort_retention"}),
        "allow_protocol_resolution": True,
    }
    assert resolve_template_confirmation(selection, **context) == "TEMPLATE_SELECTED"
    assert resolve_template_confirmation(
        selection,
        **{
            **context,
            "candidate_template_ids": ("rfm_segmentation",),
        },
    ) == "INVALID_SELECTION"
    assert resolve_template_confirmation(
        selection,
        **{
            **context,
            "visible_template_ids": frozenset({"rfm_segmentation"}),
            "authorized_template_ids": frozenset({"cohort_retention"}),
        },
    ) == "UNAUTHORIZED_TEMPLATE"
    assert resolve_template_confirmation(
        selection.model_copy(update={"candidate_set_hash": "forged-hash"}), **context
    ) == "INVALID_SELECTION"
    assert resolve_template_confirmation(
        selection.model_copy(update={"template_id": "rfm_segmentation"}), **context
    ) == "UNAUTHORIZED_TEMPLATE"


def test_phase0_template_confirmation_is_unwired_and_fails_closed_by_default() -> None:
    from app.context.semantic_contract_validator import TEMPLATE_CONFIRMATION_WIRED

    confirmation = TemplateConfirmation(
        protocol_version="template-confirmation-v1",
        action="CONFIRM_PLAN",
        current_turn_id=TURN_ID,
        plan_hash="plan-hash",
    )
    assert TEMPLATE_CONFIRMATION_WIRED is False
    assert resolve_template_confirmation(
        confirmation,
        expected_plan_hash="plan-hash",
        expected_current_turn_id=TURN_ID,
        current_turn_contract=None,
        has_user_text=False,
        plan_template_id="cohort_retention",
        visible_template_ids=frozenset({"cohort_retention"}),
        authorized_template_ids=frozenset({"cohort_retention"}),
    ) == "NEEDS_SEMANTIC_VALIDATION"


def test_template_confirmation_cancel_and_nonconfirmed_text_precedence() -> None:
    text = "按购买频次给客户分层"
    ambiguous_candidate = _contract(
        text,
        _one_clause(text),
        (
            {"id": "m_frequency", "clause_id": "c1", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},
        ),
        (
            {
                "id": "op_segment",
                "clause_id": "c1",
                "kind": "SEGMENT",
                "target_mention_ids": ("m_customer",),
                "target_concept_ids": ("CUSTOMER",),
                "mentioned_arguments": {
                    "basis": (
                        {"mention_id": "m_frequency", "concept_id": "PURCHASE_FREQUENCY", "stance": "ADOPTED"},
                    ),
                },
            },
        ),
        parse_status="AMBIGUOUS",
    )
    current_contract = _run(ambiguous_candidate, text).contract
    template_ids = frozenset({"cohort_retention"})
    base = {
        "expected_plan_hash": "plan-hash",
        "expected_current_turn_id": TURN_ID,
        "current_turn_contract": current_contract,
        "has_user_text": True,
        "plan_template_id": "cohort_retention",
        "visible_template_ids": template_ids,
        "authorized_template_ids": template_ids,
        "allow_protocol_resolution": True,
    }
    confirmation = TemplateConfirmation(
        protocol_version="template-confirmation-v1",
        action="CONFIRM_PLAN",
        current_turn_id=TURN_ID,
        plan_hash="plan-hash",
    )
    assert resolve_template_confirmation(confirmation, **base) == "NEEDS_SEMANTIC_VALIDATION"
    cancel = TemplateConfirmation(
        protocol_version="template-confirmation-v1",
        action="CANCEL_PLAN",
        current_turn_id="stale-turn",
        plan_hash="plan-hash",
    )
    assert resolve_template_confirmation(cancel, **base) == "CANCELLED"
    assert resolve_template_confirmation(
        cancel.model_copy(update={"plan_hash": "stale-hash"}), **base
    ) == "STALE_PLAN_HASH"


def test_validator_has_static_boundaries_and_remains_bounded() -> None:
    validator_path = Path(__file__).parents[1] / "app" / "context" / "semantic_contract_validator.py"
    catalog_path = Path(__file__).parents[1] / "app" / "context" / "semantic_catalog_v2.py"
    service_path = Path(__file__).parents[1] / "app" / "complex_analysis" / "service.py"
    chat_path = Path(__file__).parents[1] / "app" / "api" / "routes" / "chat.py"
    chat_schema_path = Path(__file__).parents[1] / "app" / "schemas" / "chat.py"
    validator_tree = ast.parse(validator_path.read_text(encoding="utf-8"))
    validator_imports = {
        alias.name
        for node in ast.walk(validator_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(validator_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "re" not in validator_imports
    assert not any(name.endswith("entity_resolver") or name.endswith("negative_intent") for name in validator_imports)
    assert not any(name.startswith("app.llm") or name.startswith("app.data.database") for name in validator_imports)
    registered_issue_literals = {
        node.args[1].value
        for node in ast.walk(validator_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"_add_issue", "_add_audit_issue"}
        and len(node.args) > 1
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    }
    assert registered_issue_literals == set(VALIDATOR_ISSUE_CODES)
    catalog_source = catalog_path.read_text(encoding="utf-8")
    assert "TEMPLATE_TRIGGER_PHRASES" not in catalog_source
    assert "explicit_template_matches" not in catalog_source
    service_tree = ast.parse(service_path.read_text(encoding="utf-8"))
    create_plan = next(
        node
        for node in service_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "create_plan_v2"
    )
    assert "matched_templates" not in {
        node.func.id
        for node in ast.walk(create_plan)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    chat_source = chat_path.read_text(encoding="utf-8")
    assert "_pre_graph_complex_templates" not in chat_source
    assert "matched_chat_templates" not in chat_source
    assert "complex_template_confirmation" not in chat_source
    assert "create_plan_v2(" not in chat_source
    schema_source = chat_schema_path.read_text(encoding="utf-8")
    assert "complex_template_confirmation" not in schema_source
    assert not (Path(__file__).parents[1] / "app" / "complex_analysis" / "router.py").exists()
    assert not (Path(__file__).parents[1] / "app" / "complex_analysis" / "routes.py").exists()
    assert not (Path(__file__).parents[1] / "app" / "complex_analysis" / "selector.py").exists()
    v2_paths = (
        Path(__file__).parents[1] / "app" / "context" / "semantic_contract_v2.py",
        catalog_path,
        validator_path,
    )
    for path in v2_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                argument_names = {
                    item.arg
                    for item in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
                }
                assert not argument_names & {"question", "user_message", "raw_text"}, path.name
    intent_router_path = Path(__file__).parents[1] / "app" / "graph" / "nodes" / "intent_router.py"
    intent_router_tree = ast.parse(intent_router_path.read_text(encoding="utf-8"))
    intent_router_imports = {
        alias.name
        for node in ast.walk(intent_router_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(intent_router_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(name == "re" or name.endswith(".re") for name in intent_router_imports)
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "re"
        for node in ast.walk(intent_router_tree)
    )
    intent_router_source = intent_router_path.read_text(encoding="utf-8")
    assert "app.complex_analysis.router" not in intent_router_source
    assert "app.complex_analysis.selector" not in intent_router_source
    assert "negative_intent" not in intent_router_source
    assert "entity_resolver" not in intent_router_source
    assert "startswith(" not in intent_router_source
    assert "endswith(" not in intent_router_source


def test_v1_semantic_status_cannot_enter_the_v21_validator() -> None:
    text = "客户分层"
    candidate = _contract(
        text,
        _one_clause(text),
        ({"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},),
        ({"id": "op1", "clause_id": "c1", "kind": "SEGMENT", "target_mention_ids": ("m_customer",), "target_concept_ids": ("CUSTOMER",)},),
        primary_intent="CUSTOMER_SEGMENTATION",
    )
    v1_payload = candidate.model_dump(mode="python")
    v1_payload["contract_version"] = "semantic-contract-v1"
    v1_payload["semantic_status"] = "CONFIRMED"
    with pytest.raises(ValidationError):
        validate_semantic_contract(v1_payload, source_text=text)


def test_v21_candidate_schema_rejects_v1_semantic_status_field() -> None:
    text = "客户分层"
    candidate = _contract(
        text,
        _one_clause(text),
        ({"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},),
        ({"id": "op1", "clause_id": "c1", "kind": "SEGMENT", "target_mention_ids": ("m_customer",), "target_concept_ids": ("CUSTOMER",)},),
    )
    payload = candidate.model_dump(mode="python")
    payload["semantic_status"] = "CONFIRMED"
    with pytest.raises(ValidationError):
        SemanticContractCandidateV21.model_validate(payload)


def test_effective_arguments_are_deeply_immutable_after_validation() -> None:
    text = "按购买频次给客户分层"
    candidate = _contract(
        text,
        _one_clause(text),
        (
            {"id": "m_frequency", "clause_id": "c1", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},
        ),
        ({
            "id": "op1", "clause_id": "c1", "kind": "SEGMENT",
            "target_mention_ids": ("m_customer",), "target_concept_ids": ("CUSTOMER",),
            "mentioned_arguments": {"basis": ({"mention_id": "m_frequency", "concept_id": "PURCHASE_FREQUENCY", "stance": "ADOPTED"},)},
        },),
        primary_intent="CUSTOMER_SEGMENTATION",
    )
    result = _run(candidate, text)
    arguments = result.contract.operations[0].effective_arguments
    with pytest.raises(TypeError, match="immutable"):
        arguments["basis"] = ()
    with pytest.raises(TypeError, match="immutable"):
        arguments.update({"basis": ()})
    with pytest.raises(ValidationError):
        result.contract.operations[0].effective_arguments["basis"][0].stance = "EXCLUDED"


def test_same_contract_replay_is_deterministic_for_100_runs() -> None:
    text = "按购买频次给客户分层"
    candidate = _contract(
        text,
        _one_clause(text),
        (
            {"id": "m_frequency", "clause_id": "c1", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},
        ),
        ({
            "id": "op1", "clause_id": "c1", "kind": "SEGMENT",
            "target_mention_ids": ("m_customer",), "target_concept_ids": ("CUSTOMER",),
            "mentioned_arguments": {"basis": ({"mention_id": "m_frequency", "concept_id": "PURCHASE_FREQUENCY", "stance": "ADOPTED"},)},
        },),
        primary_intent="CUSTOMER_SEGMENTATION",
    )
    expected = _run(candidate, text).contract.model_dump(mode="json")
    for _ in range(99):
        assert _run(candidate, text).contract.model_dump(mode="json") == expected


def test_adversarial_fixture_has_expected_contract_and_status_for_each_case() -> None:
    fixture_path = Path(__file__).parents[1] / "evaluation" / "datasets" / "complex_semantic_v341_adversarial.jsonl"
    cases = [json.loads(line) for line in fixture_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(cases) >= 12
    assert len({case["case_id"] for case in cases}) == len(cases)
    required_categories = {
        "same_clause_negation", "postposed_negation", "reminder_negation", "cross_clause_replacement",
        "argument_negation", "sort_vs_basis", "lifecycle_negation", "typo", "reference",
        "general_conversation", "prompt_injection", "hypothetical", "cross_turn_negation",
    }
    assert required_categories <= {case["category"] for case in cases}
    for case in cases:
        assert case["text"]
        assert case["expected_status"] in {"CONFIRMED", "AMBIGUOUS", "UNSUPPORTED", "MISSING", "CONFLICT"}
        assert isinstance(case["expected_contract"], dict) and case["expected_contract"]


def _validator_issue_case(code: str) -> tuple[SemanticContractCandidateV21, str]:
    """Build malformed structured candidates; no surface-based parser is used."""
    text = "按购买频次给客户分层"
    mentions = (
        {"id": "m_frequency", "clause_id": "c1", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
        {"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},
    )
    operations = ({
        "id": "op1", "clause_id": "c1", "kind": "SEGMENT",
        "target_mention_ids": ("m_customer",), "target_concept_ids": ("CUSTOMER",),
        "mentioned_arguments": {"basis": ({"mention_id": "m_frequency", "concept_id": "PURCHASE_FREQUENCY", "stance": "ADOPTED"},)},
    },)

    def build(
        source: str = text,
        *,
        clause_ranges: tuple[tuple[int, int], ...] | None = None,
        mention_specs: tuple[dict, ...] = mentions,
        operation_specs: tuple[dict, ...] = operations,
        domain: str = "BUSINESS_ANALYSIS",
        parse_status: str = "PARSED",
        context_sources: tuple[dict, ...] = (),
        primary_intent: str | None = None,
    ) -> SemanticContractCandidateV21:
        return _contract(
            source,
            clause_ranges if clause_ranges is not None else _one_clause(source),
            mention_specs,
            operation_specs,
            domain=domain,
            parse_status=parse_status,
            context_sources=context_sources,
            primary_intent=primary_intent,
        )

    def replace_operation(candidate, index: int = 0, **updates):
        values = list(candidate.operations)
        values[index] = values[index].model_copy(update=updates)
        return candidate.model_copy(update={"operations": tuple(values)})

    if code == "argument_concept_domain_mismatch":
        source = "按金额从高到低排序"
        return build(
            source,
            mention_specs=(
                {"id": "m_monetary", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
            ),
            operation_specs=(
                {
                    "id": "op_sort",
                    "clause_id": "c1",
                    "kind": "SORT",
                    "target_mention_ids": ("m_monetary",),
                    "target_concept_ids": ("MONETARY",),
                    "mentioned_arguments": {
                        "key": ({"mention_id": "m_monetary", "concept_id": "MONETARY", "stance": "ADOPTED"},),
                        "direction": ({"mention_id": "m_monetary", "concept_id": "MONETARY", "stance": "ADOPTED"},),
                    },
                },
            ),
        ), source

    if code == "missing_target_reference":
        return build(
            operation_specs=({
                **operations[0],
                "target_mention_ids": (),
                "mentioned_arguments": {
                    "method": ({
                        "mention_id": "missing-method-mention",
                        "concept_id": "RFM",
                        "stance": "ADOPTED",
                    },),
                },
            },),
        ), text

    candidate = build(primary_intent="CUSTOMER_SEGMENTATION")
    if code == "argument_concept_reference_mismatch":
        operation = candidate.operations[0]
        value = operation.mentioned_arguments["basis"][0].model_copy(update={"concept_id": "MONETARY"})
        return replace_operation(candidate, mentioned_arguments={"basis": (value,)}), text
    if code == "argument_evidence_outside_operation_clause":
        source = "按购买频次给客户分层，继续"
        split = source.index("，") + 1
        candidate = build(source, clause_ranges=((0, split), (split, len(source))))
        operation = candidate.operations[0]
        value = operation.mentioned_arguments["basis"][0].model_copy(update={"evidence_span": _span(split, len(source))})
        return replace_operation(candidate, mentioned_arguments={"basis": (value,)}), source
    if code in {"argument_negation_cue_outside_source_clause", "argument_negation_evidence_outside_source_clause"}:
        source = "不要金额，按购买频次给客户分层"
        candidate = build(
            source,
            clause_ranges=_two_clauses(source),
            mention_specs=(
                {"id": "m_money", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
                {"id": "m_frequency", "clause_id": "c2", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
                {"id": "m_customer", "clause_id": "c2", "surface": "客户", "concept_id": "CUSTOMER"},
            ),
            operation_specs=({
                "id": "op1", "clause_id": "c2", "kind": "SEGMENT", "target_mention_ids": ("m_customer",),
                "target_concept_ids": ("CUSTOMER",), "mentioned_arguments": {"basis": (
                    {"mention_id": "m_money", "concept_id": "MONETARY", "stance": "EXCLUDED"},
                    {"mention_id": "m_frequency", "concept_id": "PURCHASE_FREQUENCY", "stance": "ADOPTED"},
                )}, "argument_negations": ({
                    "source_clause_id": "c1", "cue": "不要", "argument_path": "basis", "target_value_mention_id": "m_money",
                },),
            },),
        )
        outside = _find(source, "购买频次")
        if code.endswith("cue_outside_source_clause"):
            negation = candidate.operations[0].argument_negations[0].model_copy(update={"cue_span": outside})
        else:
            negation = candidate.operations[0].argument_negations[0].model_copy(update={"evidence_spans": (outside,)})
        return replace_operation(candidate, argument_negations=(negation,)), source
    if code == "catalog_snapshot_mismatch":
        return candidate.model_copy(update={"catalog_snapshot_hash": "0" * 64}), text
    if code == "cross_turn_argument_negation_requires_clarification":
        source = "不要用金额做分层依据，但仍按购买频次给客户分层"
        return build(
            source, clause_ranges=_two_clauses(source),
            mention_specs=(
                {"id": "m_money", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
                {"id": "m_frequency", "clause_id": "c2", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
                {"id": "m_customer", "clause_id": "c2", "surface": "客户", "concept_id": "CUSTOMER"},
            ),
            operation_specs=({
                **operations[0], "clause_id": "c2", "mentioned_arguments": {"basis": (
                    {"mention_id": "m_money", "concept_id": "MONETARY", "stance": "EXCLUDED"},
                    {"mention_id": "m_frequency", "concept_id": "PURCHASE_FREQUENCY", "stance": "ADOPTED"},
                )}, "argument_negations": ({
                    "source_clause_id": "c1", "source_turn_id": "turn-prior", "cue": "不要", "argument_path": "basis", "target_value_mention_id": "m_money",
                },),
            },),
        ), source
    if code == "current_mention_turn_mismatch":
        mentions_updated = tuple(item.model_copy(update={"source_turn_id": "other-turn"}) if item.id == "m_customer" else item for item in candidate.mentions)
        return candidate.model_copy(update={"mentions": mentions_updated}), text
    if code == "duplicate_id":
        return candidate.model_copy(update={"mentions": (*candidate.mentions, candidate.mentions[0])}), text
    if code == "empty_inherited_context_reference":
        return build(context_sources=({"source_turn_id": "prior", "mention_ids": (), "relation": "INHERITED_CONTEXT"},)), text
    if code == "empty_text_has_clauses":
        return candidate, ""
    if code in {"exact_alias_not_in_catalog", "synonym_not_in_catalog"}:
        source = "虚构分层"
        grounding = "exact_alias" if code.startswith("exact") else "catalog_synonym"
        return build(
            source,
            mention_specs=({"id": "m_customer", "clause_id": "c1", "surface": "虚构", "concept_id": "CUSTOMER", "grounding_source": grounding},),
            operation_specs=({"id": "op1", "clause_id": "c1", "kind": "SEGMENT", "target_mention_ids": ("m_customer",), "target_concept_ids": ("CUSTOMER",)},),
        ), source
    if code == "excluded_value_missing_negation_evidence":
        operation = candidate.operations[0]
        value = operation.mentioned_arguments["basis"][0].model_copy(update={"stance": "EXCLUDED", "evidence_span": None})
        return replace_operation(candidate, mentioned_arguments={"basis": (value,)}), text
    if code == "general_domain_has_operations":
        return candidate.model_copy(update={"domain": "GENERAL_CONVERSATION"}), text
    if code == "inherited_mention_same_turn":
        mentions_updated = tuple(
            item.model_copy(update={"grounding_source": "inherited", "span": None}) if item.id == "m_customer" else item
            for item in candidate.mentions
        )
        return candidate.model_copy(update={"mentions": mentions_updated}), text
    if code == "invalid_argument_path":
        operation = candidate.operations[0]
        values = {**operation.mentioned_arguments, "key": operation.mentioned_arguments["basis"]}
        return replace_operation(candidate, mentioned_arguments=values), text
    if code == "invalid_clause_partition":
        return candidate.model_copy(update={"clauses": (Clause(id="c1", span=_span(0, 3)),)}), text
    if code == "invalid_context_mention_reference":
        return build(context_sources=({"source_turn_id": "other-turn", "mention_ids": ("m_customer",), "relation": "EXPLICIT_REFERENCE"},)), text
    if code == "mention_outside_owned_clause":
        source = "客户，购买频次给客户分层"
        first = source.index("，") + 1
        last_customer = source.rindex("客户")
        return build(
            source, clause_ranges=((0, first), (first, len(source))),
            mention_specs=(
                {"id": "m_frequency", "clause_id": "c2", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
                {"id": "m_customer", "clause_id": "c2", "surface": "客户", "concept_id": "CUSTOMER", "span": _span(0, 2)},
            ),
            operation_specs=({**operations[0], "clause_id": "c2"},),
        ), source
    if code == "missing_clause_partition":
        return candidate.model_copy(update={"clauses": ()}), text
    if code == "negation_cue_outside_owned_clause":
        source = "不要做，生命周期"
        candidate = build(
            source, clause_ranges=_two_clauses(source),
            mention_specs=({"id": "m_lifecycle", "clause_id": "c1", "surface": "做", "concept_id": "LIFECYCLE", "grounding_source": "llm_inferred"},),
            operation_specs=({"id": "op1", "clause_id": "c1", "kind": "ANALYZE", "target_mention_ids": ("m_lifecycle",), "target_concept_ids": ("LIFECYCLE",), "polarity": "NEGATIVE", "clause_scope": "SAME_CLAUSE", "cue": "不要"},),
        )
        return replace_operation(candidate, negation_cue_span=_find(source, "生命周期")), source
    if code == "negation_target_not_excluded_argument":
        source = "不要金额作为依据"
        return build(
            source,
            mention_specs=({"id": "m_money", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},),
            operation_specs=({"id": "op1", "clause_id": "c1", "kind": "FILTER", "target_mention_ids": ("m_money",), "target_concept_ids": ("MONETARY",), "mentioned_arguments": {"field": ({"mention_id": "m_money", "concept_id": "MONETARY", "stance": "ADOPTED"},)}, "argument_negations": ({"source_clause_id": "c1", "cue": "不要", "argument_path": "field", "target_value_mention_id": "m_money"},)},),
        ), source
    if code == "negative_operation_source_missing":
        source = "不要做生命周期分析"
        return build(
            source,
            mention_specs=({"id": "m1", "clause_id": "c1", "surface": "生命周期", "concept_id": "LIFECYCLE"},),
            operation_specs=({"id": "op1", "clause_id": "c1", "kind": "ANALYZE", "target_mention_ids": ("m1",), "target_concept_ids": ("LIFECYCLE",), "polarity": "NEGATIVE", "clause_scope": "REQUEST", "source_clause_id": None, "source_turn_id": None, "cue": "不要"},),
        ), source
    if code == "negative_target_outside_owned_clause":
        source = "不要做，生命周期"
        return build(
            source, clause_ranges=_two_clauses(source),
            mention_specs=({"id": "m1", "clause_id": "c2", "surface": "生命周期", "concept_id": "LIFECYCLE"},),
            operation_specs=({"id": "op1", "clause_id": "c1", "kind": "ANALYZE", "target_mention_ids": ("m1",), "target_concept_ids": ("LIFECYCLE",), "polarity": "NEGATIVE", "clause_scope": "SAME_CLAUSE", "cue": "不要"},),
        ), source
    if code == "operation_evidence_outside_owned_clause":
        source = "按购买频次给客户分层，继续"
        candidate = build(source, clause_ranges=((0, 12), (12, len(source))))
        return replace_operation(candidate, evidence_spans=(_span(12, len(source)),)), source
    if code == "operation_outside_owned_clause":
        source = "按购买频次给客户分层，继续"
        candidate = build(source, clause_ranges=((0, 12), (12, len(source))))
        return replace_operation(candidate, span=_span(12, len(source))), source
    if code == "parsed_business_contract_has_no_operations":
        return build(operation_specs=()), text
    if code == "positive_negative_argument_conflict":
        source = "按金额给客户分层，不用金额"
        return build(
            source,
            mention_specs=({"id": "m_money", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"}, {"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"}),
            operation_specs=({"id": "op1", "clause_id": "c1", "kind": "SEGMENT", "target_mention_ids": ("m_customer",), "target_concept_ids": ("CUSTOMER",), "mentioned_arguments": {"basis": ({"mention_id": "m_money", "concept_id": "MONETARY", "stance": "ADOPTED"}, {"mention_id": "m_money", "concept_id": "MONETARY", "stance": "EXCLUDED"})}, "argument_negations": ({"source_clause_id": "c1", "cue": "不用", "argument_path": "basis", "target_value_mention_id": "m_money"},)},),
        ), source
    if code == "positive_negative_operation_conflict":
        source = "按金额排序，不要按金额排序"
        candidate = build(
            source,
            mention_specs=({"id": "m_money", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},),
            operation_specs=(
                {"id": "positive", "clause_id": "c1", "kind": "SORT", "target_mention_ids": ("m_money",), "target_concept_ids": ("MONETARY",)},
                {"id": "negative", "clause_id": "c1", "kind": "SORT", "target_mention_ids": ("m_money",), "target_concept_ids": ("MONETARY",), "polarity": "NEGATIVE", "clause_scope": "SAME_CLAUSE", "cue": "不要"},
            ),
        )
        return candidate, source
    if code == "primary_intent_operation_mismatch":
        return candidate.model_copy(update={"primary_intent": "COHORT_RETENTION"}), text
    if code in {"role_clause_reference_mismatch", "role_concept_reference_mismatch"}:
        source = "客户分层，继续"
        candidate = build(
            source,
            clause_ranges=_two_clauses(source),
            mention_specs=({"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},),
            operation_specs=({"id": "op1", "clause_id": "c1", "kind": "SEGMENT", "target_mention_ids": ("m_customer",), "target_concept_ids": ("CUSTOMER",)},),
        )
        role = Role(id="r1", clause_id="c1", mention_id="m_customer", concept_id="CUSTOMER", name="analysis_subject")
        updates = {
            "role_clause_reference_mismatch": {"clause_id": "c2"},
            "role_concept_reference_mismatch": {"concept_id": "MONETARY"},
        }[code]
        role = role.model_copy(update=updates)
        return candidate.model_copy(update={"roles": (role,)}), source
    if code == "span_out_of_range":
        mentions_updated = tuple(item.model_copy(update={"span": _span(0, len(text) + 5)}) if item.id == "m_customer" else item for item in candidate.mentions)
        return candidate.model_copy(update={"mentions": mentions_updated}), text
    if code == "span_surface_mismatch":
        mentions_updated = tuple(item.model_copy(update={"span": _find(text, "频次")}) if item.id == "m_customer" else item for item in candidate.mentions)
        return candidate.model_copy(update={"mentions": mentions_updated}), text
    if code == "target_concept_reference_mismatch":
        return replace_operation(candidate, target_concept_ids=("MONETARY",)), text
    if code == "unknown_clause_reference":
        return replace_operation(candidate, clause_id="missing-clause"), text
    if code == "unknown_concept_reference":
        mentions_updated = tuple(item.model_copy(update={"concept_id": "UNKNOWN_CONCEPT"}) if item.id == "m_customer" else item for item in candidate.mentions)
        return candidate.model_copy(update={"mentions": mentions_updated}), text
    if code == "intent_id_used_as_concept":
        mentions_updated = tuple(
            item.model_copy(update={"concept_id": "CUSTOMER_SEGMENTATION"})
            if item.id == "m_customer" else item
            for item in candidate.mentions
        )
        operations_updated = tuple(
            item.model_copy(update={"target_concept_ids": ("CUSTOMER_SEGMENTATION",)})
            if item.id == "op1" else item
            for item in candidate.operations
        )
        return candidate.model_copy(update={"mentions": mentions_updated, "operations": operations_updated}), text
    if code == "unknown_intent_reference":
        item = IntentCandidate(intent_id="UNKNOWN_INTENT", evidence_spans=(_find(text, "客户"),), confidence_class="HIGH")
        return candidate.model_copy(update={"intent_candidates": (item,)}), text
    if code == "unknown_mention_reference":
        return replace_operation(candidate, target_mention_ids=("missing-mention",)), text
    if code == "unknown_role_reference":
        relation = Relation(id="r1", clause_id="c1", predicate="GROUP_BY", subject=RoleEndpoint(endpoint_type="role_id", value="missing-role"), object=ConceptEndpoint(endpoint_type="concept_id", value="CUSTOMER"))
        return candidate.model_copy(update={"relations": (relation,)}), text
    if code == "unreferenced_inherited_mention":
        inherited = candidate.mentions[0].model_copy(update={"grounding_source": "inherited", "source_turn_id": "turn-prior", "span": None})
        return candidate.model_copy(update={"mentions": (inherited, *candidate.mentions[1:])}), text
    if code == "untrusted_inherited_context_reference":
        inherited = candidate.mentions[0].model_copy(update={"grounding_source": "inherited", "source_turn_id": "turn-prior", "span": None})
        context = ContextSource(source_turn_id="turn-prior", mention_ids=(inherited.id,), relation="INHERITED_CONTEXT")
        return candidate.model_copy(update={"mentions": (inherited, *candidate.mentions[1:]), "context_sources": (context,)}), text
    raise AssertionError(f"No issue trigger fixture registered for {code}")


@pytest.mark.parametrize("issue_code", VALIDATOR_ISSUE_CODES)
def test_each_validator_issue_code_has_a_triggering_fixture(issue_code: str) -> None:
    candidate, source_text = _validator_issue_case(issue_code)
    result = _run(candidate, source_text)
    assert issue_code in {
        issue.code for issue in (*result.issues, *result.audit_issues)
    }, issue_code


def test_argument_concept_domain_mismatch_is_a_conflict() -> None:
    candidate, source_text = _validator_issue_case("argument_concept_domain_mismatch")

    result = _run(candidate, source_text)

    issue = next(issue for issue in result.issues if issue.code == "argument_concept_domain_mismatch")
    assert issue.severity == "CONFLICT"
    assert result.contract.semantic_status == "CONFLICT"


def test_intent_id_used_as_concept_is_a_conflict() -> None:
    candidate, source_text = _validator_issue_case("intent_id_used_as_concept")

    result = _run(candidate, source_text)

    issue = next(issue for issue in result.issues if issue.code == "intent_id_used_as_concept")
    assert issue.severity == "CONFLICT"
    assert result.contract.semantic_status == "CONFLICT"


@pytest.mark.parametrize(
    ("direction_surface", "direction_concept_id"),
    [
        ("从低到高", "SORT_ASCENDING"),
        ("从高到低", "SORT_DESCENDING"),
    ],
)
def test_sort_direction_accepts_grounded_concepts_in_catalog_domain(
    direction_surface: str, direction_concept_id: str
) -> None:
    source = f"按金额{direction_surface}排序"
    candidate = _contract(
        source,
        _one_clause(source),
        (
            {"id": "m_monetary", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
            {"id": "m_direction", "clause_id": "c1", "surface": direction_surface, "concept_id": direction_concept_id},
        ),
        ({
            "id": "op_sort",
            "clause_id": "c1",
            "kind": "SORT",
            "target_mention_ids": ("m_monetary",),
            "target_concept_ids": ("MONETARY",),
            "mentioned_arguments": {
                "key": ({"mention_id": "m_monetary", "concept_id": "MONETARY", "stance": "ADOPTED"},),
                "direction": ({"mention_id": "m_direction", "concept_id": direction_concept_id, "stance": "ADOPTED"},),
            },
        },),
    )

    result = _run(candidate, source)

    assert not result.issues
    assert result.contract.semantic_status == "CONFIRMED"


def test_sort_direction_may_be_omitted_when_user_does_not_specify_it() -> None:
    source = "按金额排序"
    candidate = _contract(
        source,
        _one_clause(source),
        ({"id": "m_monetary", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},),
        ({
            "id": "op_sort",
            "clause_id": "c1",
            "kind": "SORT",
            "target_mention_ids": ("m_monetary",),
            "target_concept_ids": ("MONETARY",),
            "mentioned_arguments": {
                "key": ({"mention_id": "m_monetary", "concept_id": "MONETARY", "stance": "ADOPTED"},),
            },
        },),
    )

    result = _run(candidate, source)

    assert not result.issues
    assert "direction" not in result.contract.operations[0].mentioned_arguments


def test_ambiguous_sort_direction_uses_ambiguous_parse_and_asks_for_clarification() -> None:
    source = "按金额按正反方向排序"
    candidate = _contract(
        source,
        _one_clause(source),
        ({"id": "m_monetary", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},),
        ({
            "id": "op_sort",
            "clause_id": "c1",
            "kind": "SORT",
            "target_mention_ids": ("m_monetary",),
            "target_concept_ids": ("MONETARY",),
            "mentioned_arguments": {
                "key": ({"mention_id": "m_monetary", "concept_id": "MONETARY", "stance": "ADOPTED"},),
            },
        },),
        parse_status="AMBIGUOUS",
    )
    candidate = candidate.model_copy(update={
        "request_frame": candidate.request_frame.model_copy(update={"request_mode": "ASK"}),
        "ambiguities": (
            Ambiguity(
                code="ambiguous_sort_direction",
                spans=(_find(source, "正反方向"),),
                clarification_question_candidate="请问希望按升序还是降序排序？",
            ),
        ),
    })

    result = _run(candidate, source)

    assert not result.issues
    assert result.contract.semantic_status == "AMBIGUOUS"
    assert result.contract.request_frame.request_mode == "ASK"

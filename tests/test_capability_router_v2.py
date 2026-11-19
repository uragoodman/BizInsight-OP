from __future__ import annotations

import inspect

import pytest

from app.complex_analysis.capability_router import route_semantic_contract
from app.context.semantic_catalog_v2 import DEFAULT_SEMANTIC_CATALOG, RelationRequirement
from app.context.semantic_contract_v2 import (
    ArgumentNegation,
    ArgumentValue,
    Clause,
    Mention,
    OperationCandidate,
    RequestFrame,
    Role,
    SemanticContractCandidateV21,
    Span,
)
from app.context.semantic_contract_validator import validate_semantic_contract


TURN_ID = "turn-router"


def _span(start: int, end: int) -> Span:
    return Span(start=start, end=end)


def _find(text: str, surface: str, start: int, end: int) -> Span:
    location = text.index(surface, start, end)
    return _span(location, location + len(surface))


def _candidate(
    text: str,
    clause_ranges: tuple[tuple[int, int], ...],
    mention_specs: tuple[dict, ...],
    operation_specs: tuple[dict, ...],
    *,
    role_specs: tuple[dict, ...] = (),
    parse_status: str = "PARSED",
    domain: str = "BUSINESS_ANALYSIS",
    request_mode: str = "EXECUTE",
    modality: str = "FACTUAL",
    turn_id: str = TURN_ID,
) -> SemanticContractCandidateV21:
    clauses = tuple(
        Clause(id=f"c{index + 1}", span=_span(start, end))
        for index, (start, end) in enumerate(clause_ranges)
    )
    clause_by_id = {clause.id: clause for clause in clauses}
    mentions = tuple(
        Mention(
            id=spec["id"],
            clause_id=spec["clause_id"],
            span=_find(
                text,
                spec["surface"],
                clause_by_id[spec["clause_id"]].span.start,
                clause_by_id[spec["clause_id"]].span.end,
            ),
            surface=spec["surface"],
            concept_id=spec["concept_id"],
            grounding_source=spec.get("grounding_source", "exact_alias"),
            source_turn_id=turn_id,
        )
        for spec in mention_specs
    )
    mention_by_id = {mention.id: mention for mention in mentions}

    operations = []
    for spec in operation_specs:
        clause = clause_by_id[spec["clause_id"]]
        operation_span = _span(clause.span.start, clause.span.end)
        mentioned_arguments = {}
        for path, values in spec.get("mentioned_arguments", {}).items():
            mentioned_arguments[path] = tuple(
                ArgumentValue(
                    mention_id=value["mention_id"],
                    concept_id=value["concept_id"],
                    stance=value["stance"],
                    evidence_span=operation_span if value["stance"] == "ADOPTED" else None,
                )
                for value in values
            )
        argument_negations = tuple(
            ArgumentNegation(
                id=item.get("id", f"neg-{index + 1}"),
                source_clause_id=item["source_clause_id"],
                source_turn_id=turn_id,
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
        cue = spec.get("cue")
        operations.append(
            OperationCandidate(
                id=spec["id"],
                clause_id=spec["clause_id"],
                span=operation_span,
                kind=spec["kind"],
                target_mention_ids=tuple(spec.get("target_mention_ids", ())),
                target_concept_ids=tuple(spec["target_concept_ids"]),
                mentioned_arguments=mentioned_arguments,
                argument_negations=argument_negations,
                mode=spec.get("mode", "EXECUTE"),
                polarity=spec.get("polarity", "POSITIVE"),
                clause_scope=spec.get("clause_scope"),
                negation_cue_span=(
                    _find(text, cue, clause.span.start, clause.span.end)
                    if spec.get("polarity") == "NEGATIVE" and isinstance(cue, str)
                    else None
                ),
                source_clause_id=(
                    spec.get("source_clause_id", spec["clause_id"])
                    if spec.get("clause_scope") == "REQUEST"
                    else None
                ),
                source_turn_id=turn_id if spec.get("clause_scope") == "REQUEST" else None,
                evidence_spans=(operation_span,),
            )
        )

    roles = tuple(
        Role(
            id=spec["id"],
            clause_id=mention_by_id[spec["mention_id"]].clause_id,
            mention_id=spec["mention_id"],
            concept_id=mention_by_id[spec["mention_id"]].concept_id,
            name=spec["name"],
        )
        for spec in role_specs
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
            request_mode=request_mode,
            modality=modality,
            dialogue_act_provisional="NEW",
            dialogue_act_validated="NEW",
        ),
        roles=roles,
        catalog_snapshot_hash=DEFAULT_SEMANTIC_CATALOG.snapshot_hash(),
    )


def _route(
    text: str,
    clause_ranges: tuple[tuple[int, int], ...],
    mention_specs: tuple[dict, ...],
    operation_specs: tuple[dict, ...],
    *,
    role_specs: tuple[dict, ...] = (),
    parse_status: str = "PARSED",
    domain: str = "BUSINESS_ANALYSIS",
    request_mode: str = "EXECUTE",
    modality: str = "FACTUAL",
):
    candidate = _candidate(
        text,
        clause_ranges,
        mention_specs,
        operation_specs,
        role_specs=role_specs,
        parse_status=parse_status,
        domain=domain,
        request_mode=request_mode,
        modality=modality,
    )
    validated = validate_semantic_contract(
        candidate,
        source_text=text,
        catalog=DEFAULT_SEMANTIC_CATALOG,
    )
    return validated.contract, route_semantic_contract(validated.contract)


def test_selects_rfm_with_two_bases() -> None:
    text = "按购买频次和金额给客户分层"
    contract, outcome = _route(
        text,
        ((0, len(text)),),
        (
            {"id": "m_frequency", "clause_id": "c1", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_monetary", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
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
                        {"mention_id": "m_monetary", "concept_id": "MONETARY", "stance": "ADOPTED"},
                    ),
                },
            },
        ),
        role_specs=(
            {"id": "r_basis_frequency", "mention_id": "m_frequency", "name": "segment_basis"},
            {"id": "r_basis_monetary", "mention_id": "m_monetary", "name": "segment_basis"},
        ),
    )

    assert contract.semantic_status == "CONFIRMED"
    assert outcome.status == "SELECTED"
    assert outcome.route_id == "customer_analysis"
    assert [item.capability_id for item in outcome.capability_ids] == ["customer_rfm_segmentation"]
    assert outcome.clarification_required is False
    assert outcome.decision_basis.operation_ids == ("op_segment",)
    assert outcome.decision_basis.clause_ids == ("c1",)
    assert outcome.decision_basis.effective_arguments_digest


def test_rfm_with_one_basis_needs_context() -> None:
    text = "按购买频次给客户分层"
    _, outcome = _route(
        text,
        ((0, len(text)),),
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
        role_specs=({"id": "r_basis", "mention_id": "m_frequency", "name": "segment_basis"},),
    )

    assert outcome.status == "NEEDS_CONTEXT"
    assert outcome.clarification_required is True
    assert "insufficient_basis" in " ".join(outcome.decision_basis.rejection_reasons)


def test_explicit_rfm_without_basis_needs_context() -> None:
    text = "RFM 分群"
    _, outcome = _route(
        text,
        ((0, len(text)),),
        ({"id": "m_rfm", "clause_id": "c1", "surface": "RFM", "concept_id": "RFM"},),
        (
            {
                "id": "op_segment",
                "clause_id": "c1",
                "kind": "SEGMENT",
                "target_mention_ids": (),
                "target_concept_ids": ("CUSTOMER",),
                "mentioned_arguments": {
                    "method": ({"mention_id": "m_rfm", "concept_id": "RFM", "stance": "ADOPTED"},),
                },
            },
        ),
    )

    assert outcome.status == "NEEDS_CONTEXT"
    assert outcome.clarification_required is True


def test_customer_segmentation_with_only_excluded_basis_needs_context() -> None:
    text = "给客户做分层，先不要按金额"
    contract, outcome = _route(
        text,
        ((0, len(text)),),
        (
            {"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},
            {"id": "m_monetary", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
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
                        {"mention_id": "m_monetary", "concept_id": "MONETARY", "stance": "EXCLUDED"},
                    ),
                },
                "argument_negations": (
                    {
                        "source_clause_id": "c1",
                        "cue": "不要",
                        "argument_path": "basis",
                        "target_value_mention_id": "m_monetary",
                    },
                ),
            },
        ),
        role_specs=({"id": "r_basis", "mention_id": "m_monetary", "name": "segment_basis"},),
    )

    assert contract.semantic_status == "CONFIRMED"
    assert outcome.status == "NEEDS_CONTEXT"
    assert outcome.clarification_required is True
    assert "customer_rfm_segmentation:missing_required_arguments" in outcome.decision_basis.rejection_reasons


def test_same_clause_negative_rfm_vetoes_segmentation() -> None:
    text = "按购买频次和金额给客户分层 RFM 不要"
    _, outcome = _route(
        text,
        ((0, len(text)),),
        (
            {"id": "m_frequency", "clause_id": "c1", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_monetary", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
            {"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},
            {"id": "m_rfm", "clause_id": "c1", "surface": "RFM", "concept_id": "RFM"},
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
                        {"mention_id": "m_monetary", "concept_id": "MONETARY", "stance": "ADOPTED"},
                    ),
                },
            },
            {
                "id": "op_neg_rfm",
                "clause_id": "c1",
                "kind": "SEGMENT",
                "target_mention_ids": ("m_rfm",),
                "target_concept_ids": ("RFM",),
                "polarity": "NEGATIVE",
                "clause_scope": "SAME_CLAUSE",
                "cue": "不要",
            },
        ),
        role_specs=(
            {"id": "r_basis_frequency", "mention_id": "m_frequency", "name": "segment_basis"},
            {"id": "r_basis_monetary", "mention_id": "m_monetary", "name": "segment_basis"},
        ),
    )

    assert outcome.status == "NO_MATCH"
    assert "customer_rfm_segmentation" in outcome.decision_basis.rejected_candidates
    assert "vetoed_by_negative_operation" in " ".join(outcome.decision_basis.rejection_reasons)


def test_request_scope_negative_vetoes_earlier_lifecycle() -> None:
    text = "给客户做生命周期分析，先不要"
    _, outcome = _route(
        text,
        ((0, 11), (11, len(text))),
        (
            {"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},
            {"id": "m_lifecycle", "clause_id": "c1", "surface": "生命周期", "concept_id": "LIFECYCLE"},
        ),
        (
            {
                "id": "op_analyze",
                "clause_id": "c1",
                "kind": "ANALYZE",
                "target_mention_ids": ("m_lifecycle",),
                "target_concept_ids": ("LIFECYCLE",),
            },
            {
                "id": "op_neg_lifecycle",
                "clause_id": "c2",
                "kind": "ANALYZE",
                "target_mention_ids": ("m_lifecycle",),
                "target_concept_ids": ("LIFECYCLE",),
                "polarity": "NEGATIVE",
                "clause_scope": "REQUEST",
                "cue": "不要",
            },
        ),
    )

    assert outcome.status == "NO_MATCH"
    assert "customer_lifecycle" in outcome.decision_basis.rejected_candidates


def test_reminder_is_not_negation() -> None:
    text = "不要忘记按购买频次和金额给客户分层"
    _, outcome = _route(
        text,
        ((0, len(text)),),
        (
            {"id": "m_frequency", "clause_id": "c1", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_monetary", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
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
                        {"mention_id": "m_monetary", "concept_id": "MONETARY", "stance": "ADOPTED"},
                    ),
                },
            },
        ),
        role_specs=(
            {"id": "r_basis_frequency", "mention_id": "m_frequency", "name": "segment_basis"},
            {"id": "r_basis_monetary", "mention_id": "m_monetary", "name": "segment_basis"},
        ),
    )

    assert outcome.status == "SELECTED"


def test_sort_negation_does_not_veto_segmentation() -> None:
    text = "不按金额排序，按购买频次和金额给客户分层"
    first_end = text.index("，") + 1
    _, outcome = _route(
        text,
        ((0, first_end), (first_end, len(text))),
        (
            {"id": "m_monetary_1", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
            {"id": "m_frequency", "clause_id": "c2", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_monetary_2", "clause_id": "c2", "surface": "金额", "concept_id": "MONETARY"},
            {"id": "m_customer", "clause_id": "c2", "surface": "客户", "concept_id": "CUSTOMER"},
        ),
        (
            {
                "id": "op_neg_sort",
                "clause_id": "c1",
                "kind": "SORT",
                "target_mention_ids": ("m_monetary_1",),
                "target_concept_ids": ("MONETARY",),
                "polarity": "NEGATIVE",
                "clause_scope": "SAME_CLAUSE",
                "cue": "不",
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
                        {"mention_id": "m_monetary_2", "concept_id": "MONETARY", "stance": "ADOPTED"},
                    ),
                },
            },
        ),
        role_specs=(
            {"id": "r_basis_frequency", "mention_id": "m_frequency", "name": "segment_basis"},
            {"id": "r_basis_monetary", "mention_id": "m_monetary_2", "name": "segment_basis"},
        ),
    )

    assert outcome.status == "SELECTED"
    assert "customer_rfm_segmentation" not in outcome.decision_basis.rejected_candidates


def test_cross_clause_replacement_selects_cohort() -> None:
    text = "RFM 分群不做，改用 Cohort 留存分析"
    first_end = text.index("，") + 1
    _, outcome = _route(
        text,
        ((0, first_end), (first_end, len(text))),
        (
            {"id": "m_rfm", "clause_id": "c1", "surface": "RFM", "concept_id": "RFM"},
            {"id": "m_cohort", "clause_id": "c2", "surface": "Cohort 留存分析", "concept_id": "COHORT_RETENTION"},
        ),
        (
            {
                "id": "op_neg_rfm",
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

    assert outcome.status == "SELECTED"
    assert [item.capability_id for item in outcome.capability_ids] == ["customer_cohort_retention"]


def test_general_conversation_has_no_capability() -> None:
    text = "你好"
    _, outcome = _route(
        text,
        ((0, len(text)),),
        (),
        (),
        domain="GENERAL_CONVERSATION",
    )

    assert outcome.status == "GENERAL_CONVERSATION"
    assert outcome.capability_ids == ()
    assert outcome.decision_basis.capability_ids == ()


def test_unsupported_contract_is_unsupported() -> None:
    text = "做生命周期成本分析"
    _, outcome = _route(text, ((0, len(text)),), (), (), parse_status="UNSUPPORTED")

    assert outcome.status == "UNSUPPORTED"
    assert outcome.route_id is None


def test_missing_contract_needs_context() -> None:
    text = "给客户做分层"
    _, outcome = _route(text, ((0, len(text)),), (), ())

    assert outcome.status == "NEEDS_CONTEXT"
    assert outcome.clarification_required is True
    assert "semantic_status_missing" in outcome.decision_basis.rejection_reasons


def test_multiple_capabilities_are_ambiguous() -> None:
    text = "按购买频次和金额给客户分层，并做 Cohort 留存分析"
    first_end = text.index("，") + 1
    _, outcome = _route(
        text,
        ((0, first_end), (first_end, len(text))),
        (
            {"id": "m_frequency", "clause_id": "c1", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_monetary", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
            {"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},
            {"id": "m_cohort", "clause_id": "c2", "surface": "Cohort 留存分析", "concept_id": "COHORT_RETENTION"},
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
                        {"mention_id": "m_monetary", "concept_id": "MONETARY", "stance": "ADOPTED"},
                    ),
                },
            },
            {
                "id": "op_cohort",
                "clause_id": "c2",
                "kind": "TREND",
                "target_mention_ids": ("m_cohort",),
                "target_concept_ids": ("COHORT_RETENTION",),
            },
        ),
        role_specs=(
            {"id": "r_basis_frequency", "mention_id": "m_frequency", "name": "segment_basis"},
            {"id": "r_basis_monetary", "mention_id": "m_monetary", "name": "segment_basis"},
        ),
    )

    assert outcome.status == "AMBIGUOUS"
    assert outcome.clarification_required is True


def test_route_is_deterministic() -> None:
    text = "按购买频次和金额给客户分层"
    contract, first = _route(
        text,
        ((0, len(text)),),
        (
            {"id": "m_frequency", "clause_id": "c1", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_monetary", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
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
                        {"mention_id": "m_monetary", "concept_id": "MONETARY", "stance": "ADOPTED"},
                    ),
                },
            },
        ),
        role_specs=(
            {"id": "r_basis_frequency", "mention_id": "m_frequency", "name": "segment_basis"},
            {"id": "r_basis_monetary", "mention_id": "m_monetary", "name": "segment_basis"},
        ),
    )
    second = route_semantic_contract(contract)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_router_signature_has_no_text_parameters() -> None:
    parameters = inspect.signature(route_semantic_contract).parameters
    assert list(parameters) == ["contract", "catalog"]
    assert not {"question", "user_message", "raw_text"} & set(parameters)


def test_explain_only_does_not_select_a_capability() -> None:
    text = "客户生命周期分析有用吗"
    _, outcome = _route(
        text,
        ((0, len(text)),),
        (
            {"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},
            {"id": "m_lifecycle", "clause_id": "c1", "surface": "生命周期", "concept_id": "LIFECYCLE"},
        ),
        (
            {
                "id": "op_explain",
                "clause_id": "c1",
                "kind": "ANALYZE",
                "target_mention_ids": ("m_lifecycle",),
                "target_concept_ids": ("LIFECYCLE",),
                "mode": "EXPLAIN",
            },
        ),
        request_mode="EXPLAIN",
    )

    assert outcome.status == "NO_MATCH"
    assert "no_executable_operation" in outcome.decision_basis.rejection_reasons


def test_hypothetical_never_selects_a_capability() -> None:
    text = "按购买频次和金额给客户分层"
    _, outcome = _route(
        text,
        ((0, len(text)),),
        (
            {"id": "m_frequency", "clause_id": "c1", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_monetary", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
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
                        {"mention_id": "m_monetary", "concept_id": "MONETARY", "stance": "ADOPTED"},
                    ),
                },
            },
        ),
        role_specs=(
            {"id": "r_basis_frequency", "mention_id": "m_frequency", "name": "segment_basis"},
            {"id": "r_basis_monetary", "mention_id": "m_monetary", "name": "segment_basis"},
        ),
        modality="HYPOTHETICAL",
    )

    assert outcome.status == "NO_MATCH"
    assert "hypothetical_not_executable" in outcome.decision_basis.rejection_reasons


def test_ask_mode_never_selects_a_capability() -> None:
    text = "按购买频次和金额给客户分层"
    _, outcome = _route(
        text,
        ((0, len(text)),),
        (
            {"id": "m_frequency", "clause_id": "c1", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_monetary", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
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
                        {"mention_id": "m_monetary", "concept_id": "MONETARY", "stance": "ADOPTED"},
                    ),
                },
            },
        ),
        role_specs=(
            {"id": "r_basis_frequency", "mention_id": "m_frequency", "name": "segment_basis"},
            {"id": "r_basis_monetary", "mention_id": "m_monetary", "name": "segment_basis"},
        ),
        request_mode="ASK",
    )

    assert outcome.status == "NO_MATCH"
    assert "request_mode_ask" in outcome.decision_basis.rejection_reasons


def test_mixed_explain_and_execute_keeps_the_execute_intent() -> None:
    text = "解释生命周期，再按购买频次和金额给客户分层"
    first_end = text.index("，") + 1
    _, outcome = _route(
        text,
        ((0, first_end), (first_end, len(text))),
        (
            {"id": "m_lifecycle", "clause_id": "c1", "surface": "生命周期", "concept_id": "LIFECYCLE"},
            {"id": "m_frequency", "clause_id": "c2", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_monetary", "clause_id": "c2", "surface": "金额", "concept_id": "MONETARY"},
            {"id": "m_customer", "clause_id": "c2", "surface": "客户", "concept_id": "CUSTOMER"},
        ),
        (
            {
                "id": "op_explain",
                "clause_id": "c1",
                "kind": "ANALYZE",
                "target_mention_ids": ("m_lifecycle",),
                "target_concept_ids": ("LIFECYCLE",),
                "mode": "EXPLAIN",
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
                        {"mention_id": "m_monetary", "concept_id": "MONETARY", "stance": "ADOPTED"},
                    ),
                },
            },
        ),
        role_specs=(
            {"id": "r_basis_frequency", "mention_id": "m_frequency", "name": "segment_basis"},
            {"id": "r_basis_monetary", "mention_id": "m_monetary", "name": "segment_basis"},
        ),
        request_mode="EXPLAIN",
    )

    assert outcome.status == "SELECTED"
    assert [item.capability_id for item in outcome.capability_ids] == ["customer_rfm_segmentation"]



def test_intent_router_shadow_records_router_observation(monkeypatch) -> None:
    from types import SimpleNamespace

    from app.graph.nodes import intent_router as intent_router_node

    text = "按购买频次和金额给客户分层"
    contract, _ = _route(
        text,
        ((0, len(text)),),
        (
            {"id": "m_frequency", "clause_id": "c1", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_monetary", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
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
                        {"mention_id": "m_monetary", "concept_id": "MONETARY", "stance": "ADOPTED"},
                    ),
                },
            },
        ),
        role_specs=(
            {"id": "r_basis_frequency", "mention_id": "m_frequency", "name": "segment_basis"},
            {"id": "r_basis_monetary", "mention_id": "m_monetary", "name": "segment_basis"},
        ),
    )
    settings = SimpleNamespace(
        complex_route_mode="shadow",
        semantic_interpreter_mode="shadow",
        complex_selector_mode="off",
    )
    monkeypatch.setattr(intent_router_node, "get_settings", lambda: settings)
    state = {
        "semantic_contract": contract.model_dump(mode="json"),
        "node_history": [],
    }

    result = intent_router_node.intent_router(state)

    assert result["complex_route_decision"] == {}
    assert result["routing_context"]["capability_router"]["status"] == "routed"
    assert result["routing_context"]["capability_router"]["outcome"]["status"] == "SELECTED"
    basis = result["routing_context"]["capability_router"]["outcome"]["decision_basis"]
    assert basis["operation_ids"] != ["op_segment"]
    assert all(len(value) == 64 for value in basis["operation_ids"] + basis["clause_ids"])


def test_router_rejects_contract_anchored_to_another_catalog_snapshot() -> None:
    text = "按购买频次和金额给客户分层"
    contract, _ = _route(
        text,
        ((0, len(text)),),
        (
            {"id": "m_frequency", "clause_id": "c1", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_monetary", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
            {"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},
        ),
        ({
            "id": "op_segment", "clause_id": "c1", "kind": "SEGMENT",
            "target_mention_ids": ("m_customer",), "target_concept_ids": ("CUSTOMER",),
            "mentioned_arguments": {"basis": (
                {"mention_id": "m_frequency", "concept_id": "PURCHASE_FREQUENCY", "stance": "ADOPTED"},
                {"mention_id": "m_monetary", "concept_id": "MONETARY", "stance": "ADOPTED"},
            )},
        },),
        role_specs=(
            {"id": "r_basis_frequency", "mention_id": "m_frequency", "name": "segment_basis"},
            {"id": "r_basis_monetary", "mention_id": "m_monetary", "name": "segment_basis"},
        ),
    )
    mismatched = contract.model_copy(update={"catalog_snapshot_hash": "0" * 64})
    with pytest.raises(ValueError, match="catalog snapshot"):
        route_semantic_contract(mismatched)


def test_router_rejects_unknown_target_concepts_even_when_known_target_is_present() -> None:
    text = "按购买频次和金额给客户分层"
    contract, _ = _route(
        text,
        ((0, len(text)),),
        (
            {"id": "m_frequency", "clause_id": "c1", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_monetary", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
            {"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},
        ),
        ({
            "id": "op_segment", "clause_id": "c1", "kind": "SEGMENT",
            "target_mention_ids": ("m_customer",), "target_concept_ids": ("CUSTOMER",),
            "mentioned_arguments": {"basis": (
                {"mention_id": "m_frequency", "concept_id": "PURCHASE_FREQUENCY", "stance": "ADOPTED"},
                {"mention_id": "m_monetary", "concept_id": "MONETARY", "stance": "ADOPTED"},
            )},
        },),
    )
    bad_operation = contract.operations[0].model_copy(
        update={"target_concept_ids": ("CUSTOMER", "OUTSIDE_CATALOG")}
    )
    unanchored = contract.model_copy(update={"operations": (bad_operation,)})
    with pytest.raises(ValueError, match="outside the router catalog"):
        route_semantic_contract(unanchored)


@pytest.mark.parametrize("endpoint_type", ["role_id", "mention_id", "concept_id"])
def test_required_relations_resolve_all_endpoint_types(endpoint_type: str) -> None:
    from app.context.semantic_contract_v2 import ConceptEndpoint, MentionEndpoint, Relation, RoleEndpoint

    text = "按购买频次和金额给客户分层"
    contract, _ = _route(
        text,
        ((0, len(text)),),
        (
            {"id": "m_frequency", "clause_id": "c1", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_monetary", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
            {"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},
        ),
        ({
            "id": "op_segment", "clause_id": "c1", "kind": "SEGMENT",
            "target_mention_ids": ("m_customer",), "target_concept_ids": ("CUSTOMER",),
            "mentioned_arguments": {"basis": (
                {"mention_id": "m_frequency", "concept_id": "PURCHASE_FREQUENCY", "stance": "ADOPTED"},
                {"mention_id": "m_monetary", "concept_id": "MONETARY", "stance": "ADOPTED"},
            )},
        },),
        role_specs=(
            {"id": "r_basis_frequency", "mention_id": "m_frequency", "name": "segment_basis"},
            {"id": "r_basis_monetary", "mention_id": "m_monetary", "name": "segment_basis"},
            {"id": "r_customer", "mention_id": "m_customer", "name": "analysis_subject"},
        ),
    )
    base_capability = next(
        item for item in DEFAULT_SEMANTIC_CATALOG.capabilities
        if item.capability_id == "customer_rfm_segmentation"
    )
    relation_requirement = RelationRequirement(
        subject_role="segment_basis", predicate="GROUP_BY", object_role="analysis_subject",
    )
    updated_capability = base_capability.model_copy(
        update={"required_relations": (relation_requirement,)}
    )
    custom_capabilities = tuple(
        updated_capability if item.capability_id == updated_capability.capability_id else item
        for item in DEFAULT_SEMANTIC_CATALOG.capabilities
    )
    custom_catalog = DEFAULT_SEMANTIC_CATALOG.model_copy(update={"capabilities": custom_capabilities})

    if endpoint_type == "role_id":
        subject = RoleEndpoint(endpoint_type="role_id", value="r_basis_frequency")
        object_ = RoleEndpoint(endpoint_type="role_id", value="r_customer")
    elif endpoint_type == "mention_id":
        subject = MentionEndpoint(endpoint_type="mention_id", value="m_frequency")
        object_ = MentionEndpoint(endpoint_type="mention_id", value="m_customer")
    else:
        subject = ConceptEndpoint(endpoint_type="concept_id", value="PURCHASE_FREQUENCY")
        object_ = ConceptEndpoint(endpoint_type="concept_id", value="CUSTOMER")
    relation = Relation(
        id="r_group_by", clause_id="c1", predicate="GROUP_BY", subject=subject, object=object_,
    )
    custom_contract = contract.model_copy(update={
        "catalog_snapshot_hash": custom_catalog.snapshot_hash(),
        "relations": (relation,),
    })

    outcome = route_semantic_contract(custom_contract, custom_catalog)

    assert outcome.status == "SELECTED"
    assert [item.capability_id for item in outcome.capability_ids] == ["customer_rfm_segmentation"]


def test_required_relation_with_unresolved_endpoint_has_deterministic_rejection() -> None:
    from app.context.semantic_contract_v2 import MentionEndpoint, Relation

    text = "按购买频次和金额给客户分层"
    contract, _ = _route(
        text,
        ((0, len(text)),),
        (
            {"id": "m_frequency", "clause_id": "c1", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_monetary", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
            {"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},
        ),
        ({
            "id": "op_segment", "clause_id": "c1", "kind": "SEGMENT",
            "target_mention_ids": ("m_customer",), "target_concept_ids": ("CUSTOMER",),
            "mentioned_arguments": {"basis": (
                {"mention_id": "m_frequency", "concept_id": "PURCHASE_FREQUENCY", "stance": "ADOPTED"},
                {"mention_id": "m_monetary", "concept_id": "MONETARY", "stance": "ADOPTED"},
            )},
        },),
        role_specs=(
            {"id": "r_basis_frequency", "mention_id": "m_frequency", "name": "segment_basis"},
            {"id": "r_basis_monetary", "mention_id": "m_monetary", "name": "segment_basis"},
            {"id": "r_customer", "mention_id": "m_customer", "name": "analysis_subject"},
        ),
    )
    base_capability = next(
        item for item in DEFAULT_SEMANTIC_CATALOG.capabilities
        if item.capability_id == "customer_rfm_segmentation"
    )
    requirement = RelationRequirement(
        subject_role="segment_basis", predicate="GROUP_BY", object_role="analysis_subject",
    )
    updated_capability = base_capability.model_copy(update={"required_relations": (requirement,)})
    custom_catalog = DEFAULT_SEMANTIC_CATALOG.model_copy(update={
        "capabilities": tuple(
            updated_capability if item.capability_id == updated_capability.capability_id else item
            for item in DEFAULT_SEMANTIC_CATALOG.capabilities
        ),
    })
    relation = Relation(
        id="r_group_by", clause_id="c1", predicate="GROUP_BY",
        subject=MentionEndpoint(endpoint_type="mention_id", value="not-a-role-mention"),
        object=MentionEndpoint(endpoint_type="mention_id", value="m_customer"),
    )
    custom_contract = contract.model_copy(update={
        "catalog_snapshot_hash": custom_catalog.snapshot_hash(),
        "relations": (relation,),
    })

    outcome = route_semantic_contract(custom_contract, custom_catalog)

    assert outcome.status == "NO_MATCH"
    assert "customer_rfm_segmentation:missing_required_relation:GROUP_BY" in outcome.decision_basis.rejection_reasons
    assert "customer_rfm_segmentation" in outcome.decision_basis.rejected_candidates


def test_router_outcomes_remain_catalog_bounded_and_template_free() -> None:
    text = "按购买频次和金额给客户分层"
    _, outcome = _route(
        text,
        ((0, len(text)),),
        (
            {"id": "m_frequency", "clause_id": "c1", "surface": "购买频次", "concept_id": "PURCHASE_FREQUENCY"},
            {"id": "m_monetary", "clause_id": "c1", "surface": "金额", "concept_id": "MONETARY"},
            {"id": "m_customer", "clause_id": "c1", "surface": "客户", "concept_id": "CUSTOMER"},
        ),
        ({
            "id": "op_segment", "clause_id": "c1", "kind": "SEGMENT",
            "target_mention_ids": ("m_customer",), "target_concept_ids": ("CUSTOMER",),
            "mentioned_arguments": {"basis": (
                {"mention_id": "m_frequency", "concept_id": "PURCHASE_FREQUENCY", "stance": "ADOPTED"},
                {"mention_id": "m_monetary", "concept_id": "MONETARY", "stance": "ADOPTED"},
            )},
        },),
        role_specs=(
            {"id": "r_basis_frequency", "mention_id": "m_frequency", "name": "segment_basis"},
            {"id": "r_basis_monetary", "mention_id": "m_monetary", "name": "segment_basis"},
        ),
    )
    catalog = DEFAULT_SEMANTIC_CATALOG
    route_ids = {item.route_id for item in catalog.routes}
    capabilities = {item.capability_id for item in catalog.capabilities}
    assert outcome.route_id in route_ids
    assert {item.capability_id for item in outcome.capability_ids} <= capabilities
    assert {item.capability_id for item in outcome.route_capability_ids} <= capabilities
    assert {item.capability_id for item in outcome.decision_basis.capability_ids} <= capabilities
    assert set(outcome.decision_basis.rejected_candidates) <= capabilities
    assert "template_id" not in outcome.model_dump(mode="json")


def test_generic_customer_retention_exposes_two_non_executable_candidates() -> None:
    text = "客户留存分析"
    _, outcome = _route(
        text,
        ((0, len(text)),),
        ({"id": "m_retention", "clause_id": "c1", "surface": text,
          "concept_id": "CUSTOMER_RETENTION"},),
        ({"id": "op_retention", "clause_id": "c1", "kind": "ANALYZE",
          "target_mention_ids": ("m_retention",),
          "target_concept_ids": ("CUSTOMER_RETENTION",)},),
    )

    assert outcome.status == "AMBIGUOUS"
    assert outcome.route_id is None
    assert outcome.capability_ids == ()
    assert {
        item.capability_id for item in outcome.candidate_capability_ids
    } == {
        "customer_retention_cohort_choice",
        "customer_retention_lifecycle_choice",
    }
    assert all(item.template_id for item in DEFAULT_SEMANTIC_CATALOG.capabilities
               if item.capability_id in {
                   ref.capability_id for ref in outcome.candidate_capability_ids
               })


def test_ambiguous_router_candidates_require_full_visibility_and_allowlisting(monkeypatch) -> None:
    from types import SimpleNamespace

    from app.analysis.templates import registry
    from app.graph.nodes.intent_router import _safe_template_candidates

    text = "客户留存分析"
    _, outcome = _route(
        text,
        ((0, len(text)),),
        ({"id": "m_retention", "clause_id": "c1", "surface": text,
          "concept_id": "CUSTOMER_RETENTION"},),
        ({"id": "op_retention", "clause_id": "c1", "kind": "ANALYZE",
          "target_mention_ids": ("m_retention",),
          "target_concept_ids": ("CUSTOMER_RETENTION",)},),
    )
    monkeypatch.setattr(registry, "list_chat_visible_templates", lambda: [
        {"name": "cohort_retention", "display_name": "Cohort 留存分析"},
        {"name": "customer_lifecycle", "display_name": "客户生命周期分析"},
    ])
    settings = SimpleNamespace(complex_route_canary_templates=[
        "cohort_retention", "customer_lifecycle",
    ])

    candidates = _safe_template_candidates(outcome, settings)
    assert candidates == [
        {"template_id": "cohort_retention", "display_name": "Cohort 留存分析"},
        {"template_id": "customer_lifecycle", "display_name": "客户生命周期分析"},
    ]
    settings.complex_route_canary_templates = ["cohort_retention"]
    assert _safe_template_candidates(outcome, settings) == []


def test_gmv_diagnostics_has_a_registered_deterministic_route() -> None:
    text = "GMV 诊断"
    _, outcome = _route(
        text,
        ((0, len(text)),),
        ({"id": "m_diagnostics", "clause_id": "c1", "surface": text,
          "concept_id": "GMV_GROWTH_DIAGNOSTICS"},),
        ({"id": "op_diagnostics", "clause_id": "c1", "kind": "ANALYZE",
          "target_mention_ids": ("m_diagnostics",),
          "target_concept_ids": ("GMV_GROWTH_DIAGNOSTICS",)},),
    )

    assert outcome.status == "SELECTED"
    assert outcome.route_id == "growth_analysis"
    assert [item.capability_id for item in outcome.capability_ids] == ["gmv_growth_diagnostics"]

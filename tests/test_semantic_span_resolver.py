from __future__ import annotations

from copy import deepcopy

from app.context.semantic_catalog_v2 import DEFAULT_SEMANTIC_CATALOG
from app.context.semantic_span_resolver import (
    normalize_candidate,
    normalize_pre_schema_references,
)


def _candidate(text: str, *, clauses=None, mentions=None, operations=None) -> dict:
    return {
        "clauses": clauses if clauses is not None else [{"id": "c1", "span": {"start": 0, "end": len(text)}}],
        "mentions": mentions or [],
        "operations": operations or [],
        "roles": [],
        "relations": [],
        "intent_candidates": [],
        "ambiguities": [],
        "contract_version": "semantic-contract-v2.1",
        "turn_id": "turn-current",
        "parse_status": "PARSED",
        "domain": "BUSINESS_ANALYSIS",
        "request_frame": {"request_mode": "EXECUTE", "modality": "FACTUAL"},
        "primary_intent": None,
        "catalog_snapshot_hash": DEFAULT_SEMANTIC_CATALOG.snapshot_hash(),
    }


def _mention(
    mention_id: str, surface: str, concept_id: str, start: int, end: int,
    *, clause_id: str = "c1", grounding_source: str = "llm_inferred",
    source_turn_id: str = "turn-current",
) -> dict:
    return {
        "id": mention_id,
        "clause_id": clause_id,
        "span": {"start": start, "end": end},
        "surface": surface,
        "concept_id": concept_id,
        "candidate_types": [],
        "grounding_source": grounding_source,
        "source_turn_id": source_turn_id,
    }


def test_clause_end_off_by_one_and_two_clause_tail_are_repaired():
    text = "客户分层，按金额"
    candidate = _candidate(
        text,
        clauses=[
            {"id": "c1", "span": {"start": 0, "end": 4}},
            {"id": "c2", "span": {"start": 5, "end": len(text) - 1}},
        ],
    )

    normalized, stats = normalize_candidate(candidate, text)

    assert normalized["clauses"] == [
        {"id": "c1", "span": {"start": 0, "end": 5}},
        {"id": "c2", "span": {"start": 5, "end": len(text)}},
    ]
    assert stats["clauses_fixed"] == 2


def test_missing_clauses_create_one_full_text_clause():
    text = "客户分层"

    normalized, stats = normalize_candidate(_candidate(text, clauses=[]), text)

    assert normalized["clauses"] == [{"id": "c1", "span": {"start": 0, "end": len(text)}}]
    assert stats["clauses_fallback_split"] == 1


def test_reversed_clause_order_is_sorted_and_clause_references_follow_ids():
    text = "客户，金额"
    candidate = _candidate(
        text,
        clauses=[
            {"id": "later", "span": {"start": 3, "end": len(text)}},
            {"id": "earlier", "span": {"start": 0, "end": 3}},
        ],
        mentions=[
            _mention("m_customer", "客户", "CUSTOMER", 0, 2, clause_id="earlier"),
            _mention("m_money", "金额", "MONETARY", 3, 5, clause_id="later"),
        ],
    )

    normalized, stats = normalize_candidate(candidate, text)

    assert normalized["clauses"] == [
        {"id": "c1", "span": {"start": 0, "end": 3}},
        {"id": "c2", "span": {"start": 3, "end": len(text)}},
    ]
    assert [mention["clause_id"] for mention in normalized["mentions"]] == ["c1", "c2"]
    assert stats["clause_ids_reassigned"] == 2


def test_duplicate_clause_coordinates_fall_back_to_punctuation_split():
    text = "客户，按金额"
    candidate = _candidate(
        text,
        clauses=[
            {"id": "c1", "span": {"start": 0, "end": 2}},
            {"id": "c2", "span": {"start": 0, "end": 2}},
        ],
    )

    normalized, stats = normalize_candidate(candidate, text)

    assert normalized["clauses"] == [
        {"id": "c1", "span": {"start": 0, "end": 3}},
        {"id": "c2", "span": {"start": 3, "end": len(text)}},
    ]
    assert stats["clauses_fallback_split"] == 1


def test_overlapping_clause_coordinates_fall_back_to_punctuation_split():
    text = "客户，金额"
    candidate = _candidate(
        text,
        clauses=[
            {"id": "c1", "span": {"start": 0, "end": 4}},
            {"id": "c2", "span": {"start": 3, "end": len(text)}},
        ],
    )

    normalized, stats = normalize_candidate(candidate, text)

    assert normalized["clauses"] == [
        {"id": "c1", "span": {"start": 0, "end": 3}},
        {"id": "c2", "span": {"start": 3, "end": len(text)}},
    ]
    assert stats["clauses_fallback_split"] == 1


def test_empty_clause_coordinate_falls_back_to_punctuation_split():
    text = "客户，金额"
    candidate = _candidate(
        text,
        clauses=[
            {"id": "c1", "span": {"start": 0, "end": 0}},
            {"id": "c2", "span": {"start": 0, "end": len(text)}},
        ],
    )

    normalized, stats = normalize_candidate(candidate, text)

    assert normalized["clauses"] == [
        {"id": "c1", "span": {"start": 0, "end": 3}},
        {"id": "c2", "span": {"start": 3, "end": len(text)}},
    ]
    assert stats["clauses_fallback_split"] == 1


def test_exact_mention_anchor_moves_a_bad_clause_boundary():
    text = "客户，RFM分群，后续"
    candidate = _candidate(
        text,
        clauses=[
            {"id": "c1", "span": {"start": 0, "end": 5}},
            {"id": "c2", "span": {"start": 5, "end": len(text)}},
        ],
        mentions=[_mention("m_rfm", "RFM分群", "RFM", 3, 8, clause_id="c1")],
    )

    normalized, stats = normalize_candidate(candidate, text)

    assert normalized["clauses"] == [
        {"id": "c1", "span": {"start": 0, "end": 9}},
        {"id": "c2", "span": {"start": 9, "end": len(text)}},
    ]
    assert normalized["mentions"][0]["clause_id"] == "c1"
    assert stats["clause_boundaries_adjusted_for_mentions"] == 1


def test_mention_span_is_found_inside_its_clause_and_alias_source_is_fixed():
    text = "按客户分层"
    candidate = _candidate(
        text,
        mentions=[_mention("m_customer", "客户", "CUSTOMER", 2, 4)],
    )

    normalized, stats = normalize_candidate(candidate, text)

    assert normalized["mentions"][0]["span"] == {"start": 1, "end": 3}
    assert normalized["mentions"][0]["grounding_source"] == "exact_alias"
    assert stats["mention_spans_fixed"] == 1
    assert stats["grounding_fixed"] == 1


def test_duplicate_surface_uses_the_nearest_occurrence():
    text = "客户和客户"
    candidate = _candidate(
        text,
        mentions=[_mention("m_customer", "客户", "CUSTOMER", 4, 6)],
    )

    normalized, stats = normalize_candidate(candidate, text)

    assert normalized["mentions"][0]["span"] == {"start": 3, "end": 5}
    assert normalized["mentions"][0]["clause_id"] == "c1"
    assert stats["mention_spans_fixed"] == 1
    assert stats["mention_clause_remapped"] == 0


def test_unique_global_surface_can_remap_mention_to_owning_clause():
    text = "金额客户"
    candidate = _candidate(
        text,
        clauses=[
            {"id": "c1", "span": {"start": 0, "end": 2}},
            {"id": "c2", "span": {"start": 2, "end": len(text)}},
        ],
        mentions=[_mention("m_customer", "客户", "CUSTOMER", 0, 1, clause_id="c1")],
    )

    normalized, stats = normalize_candidate(candidate, text)

    assert normalized["mentions"][0]["span"] == {"start": 2, "end": 4}
    assert normalized["mentions"][0]["clause_id"] == "c2"
    assert stats["mention_spans_fixed"] == 1
    assert stats["mention_clause_remapped"] == 1


def test_unresolved_surface_is_preserved_and_reported_without_raw_text():
    text = "客户分层"
    mention = _mention("m_missing", "不存在的词", "CUSTOMER", 0, 1)
    candidate = _candidate(text, mentions=[mention])

    normalized, stats = normalize_candidate(candidate, text)

    assert normalized["mentions"][0] == mention
    assert stats["mention_spans_unresolved"] == 1
    assert stats["unresolved_surface"] == ["m_missing"]
    assert stats["unresolved_surface_count"] == 1
    assert "不存在的词" not in repr(stats)


def test_exact_alias_synonym_and_inferred_grounding_are_normalized():
    text = "客户复购频次经营额"
    candidate = _candidate(
        text,
        mentions=[
            _mention("m_customer", "客户", "CUSTOMER", 0, 2, grounding_source="catalog_synonym"),
            _mention("m_frequency", "复购频次", "PURCHASE_FREQUENCY", 2, 6),
            _mention("m_gmv", "经营额", "GMV", 6, 9, grounding_source="exact_alias"),
        ],
    )

    normalized, stats = normalize_candidate(candidate, text)

    assert [item["grounding_source"] for item in normalized["mentions"]] == [
        "exact_alias", "catalog_synonym", "llm_inferred",
    ]
    assert stats["grounding_fixed"] == 3


def test_unknown_concept_grounding_metadata_is_left_unchanged():
    text = "客户"
    mention = _mention("m_unknown", "客户", "UNKNOWN_CONCEPT", 0, 2)
    candidate = _candidate(text, mentions=[mention])

    normalized, stats = normalize_candidate(candidate, text)

    assert normalized["mentions"][0]["concept_id"] == "UNKNOWN_CONCEPT"
    assert normalized["mentions"][0]["grounding_source"] == "llm_inferred"
    assert stats["grounding_fixed"] == 0


def test_inherited_mentions_keep_null_span_and_are_not_reparsed():
    text = "继续分析"
    mention = _mention(
        "m_prior", "客户", "CUSTOMER", 0, 1,
        grounding_source="inherited", source_turn_id="turn-prior",
    )
    mention["span"] = None
    candidate = _candidate(text, mentions=[mention])

    normalized, stats = normalize_candidate(candidate, text)

    assert normalized["mentions"][0] == mention
    assert stats["mention_spans_fixed"] == 0
    assert stats["mention_spans_unresolved"] == 0
    assert stats["grounding_fixed"] == 0


def test_operation_evidence_and_negation_spans_are_clamped_to_their_clauses():
    text = "客户，金额"
    candidate = _candidate(
        text,
        clauses=[{"id": "c1", "span": {"start": 0, "end": 2}}],
        operations=[
            {
                "id": "op_negative", "clause_id": "c1", "kind": "SEGMENT",
                "span": {"start": 3, "end": 12},
                "target_mention_ids": [], "target_concept_ids": ["RFM"],
                "mentioned_arguments": {}, "argument_negations": [],
                "mode": "EXECUTE", "polarity": "NEGATIVE", "clause_scope": "SAME_CLAUSE",
                "negation_cue_span": {"start": 7, "end": 8},
                "evidence_spans": [{"start": -1, "end": 1}],
            },
            {
                "id": "op_positive", "clause_id": "c1", "kind": "SEGMENT",
                "span": {"start": 0, "end": 1},
                "target_mention_ids": [], "target_concept_ids": ["CUSTOMER"],
                "mentioned_arguments": {"basis": [{
                    "mention_id": "m_amount", "concept_id": "MONETARY", "stance": "ADOPTED",
                    "evidence_span": {"start": 6, "end": 8},
                }]}, "argument_negations": [{
                    "id": "neg1", "source_clause_id": "c1", "source_turn_id": "turn-current",
                    "cue_span": {"start": 7, "end": 8}, "argument_path": "basis",
                    "target_value_mention_id": "m_amount", "evidence_spans": [{"start": 6, "end": 8}],
                }],
                "mode": "EXECUTE", "polarity": "POSITIVE", "evidence_spans": [{"start": 0, "end": 1}],
            },
        ],
    )

    normalized, stats = normalize_candidate(candidate, text)

    assert normalized["clauses"] == [{"id": "c1", "span": {"start": 0, "end": len(text)}}]
    negative, positive = normalized["operations"]
    assert negative["span"] == {"start": 3, "end": len(text)}
    assert negative["negation_cue_span"] == {"start": len(text) - 1, "end": len(text)}
    assert negative["evidence_spans"] == [{"start": 0, "end": 1}]
    assert positive["argument_negations"][0]["cue_span"] == {"start": len(text) - 1, "end": len(text)}
    assert positive["argument_negations"][0]["evidence_spans"] == [
        {"start": len(text) - 1, "end": len(text)},
    ]
    assert positive["mentioned_arguments"]["basis"][0]["evidence_span"] == {
        "start": len(text) - 1, "end": len(text),
    }
    assert stats["operation_spans_clamped"] == 1
    assert stats["evidence_spans_clamped"] == 2
    assert stats["argument_evidence_spans_clamped"] == 1
    assert stats["negation_spans_clamped"] == 1
    assert stats["argument_negation_spans_clamped"] == 2


def test_normalization_preserves_semantic_fields_including_rfm_method_and_basis():
    text = "按购买频次和金额做 RFM 客户分群"
    candidate = _candidate(
        text,
        clauses=[{"id": "c1", "span": {"start": 1, "end": len(text) - 1}}],
        mentions=[
            _mention("m_frequency", "购买频次", "PURCHASE_FREQUENCY", 0, 4),
            _mention("m_money", "金额", "MONETARY", 5, 7),
            _mention("m_rfm", "RFM", "RFM", 9, 12),
            _mention("m_customer", "客户", "CUSTOMER", 13, 15),
        ],
        operations=[{
            "id": "op_rfm", "clause_id": "c1", "kind": "SEGMENT",
            "span": {"start": 1, "end": len(text) - 1},
            "target_mention_ids": ["m_customer"], "target_concept_ids": ["CUSTOMER"],
            "mentioned_arguments": {
                "basis": [
                    {"mention_id": "m_frequency", "concept_id": "PURCHASE_FREQUENCY", "stance": "ADOPTED", "evidence_span": {"start": 0, "end": 4}},
                    {"mention_id": "m_money", "concept_id": "MONETARY", "stance": "ADOPTED", "evidence_span": {"start": 5, "end": 7}},
                ],
                "method": [{"mention_id": "m_rfm", "concept_id": "RFM", "stance": "ADOPTED", "evidence_span": {"start": 9, "end": 12}}],
            },
            "argument_negations": [], "mode": "EXECUTE", "polarity": "POSITIVE",
            "evidence_spans": [{"start": 0, "end": len(text) - 1}],
        }],
    )
    before = deepcopy(candidate)

    normalized, _stats = normalize_candidate(candidate, text)

    assert normalized["operations"][0]["kind"] == before["operations"][0]["kind"]
    assert normalized["operations"][0]["polarity"] == before["operations"][0]["polarity"]
    assert normalized["operations"][0]["target_concept_ids"] == before["operations"][0]["target_concept_ids"]
    assert normalized["operations"][0]["mentioned_arguments"] == before["operations"][0]["mentioned_arguments"]
    assert normalized["operations"][0]["mentioned_arguments"]["method"][0]["concept_id"] == "RFM"
    assert normalized["operations"][0]["mentioned_arguments"]["basis"][0]["concept_id"] == "PURCHASE_FREQUENCY"


def test_pre_schema_reference_normalization_repairs_exact_role_endpoint_and_unique_target():
    text = "客户分层"
    candidate = _candidate(
        text,
        mentions=[_mention("m_customer", "客户", "CUSTOMER", 0, 2)],
        operations=[{
            "id": "op_explain", "clause_id": "c1", "kind": "EXPLAIN",
            "target_mention_ids": [], "target_concept_ids": ["CUSTOMER"],
        }],
    )
    candidate["roles"] = [{
        "id": "r_subject", "clause_id": "c1",
        "mention_id": {"endpoint_type": "mention_id", "value": "m_customer"},
        "concept_id": "CUSTOMER", "name": "analysis_subject",
    }, {
        "id": "r_subject_copy", "clause_id": "c1",
        "mention_id": ["m_customer"],
        "concept_id": "CUSTOMER", "name": "analysis_subject",
    }, {
        "id": "r_subject_nested", "clause_id": "c1",
        "mention_id": {"mention": {"id": "m_customer", "surface": "客户"}},
        "concept_id": "CUSTOMER", "name": "analysis_subject",
    }]

    normalized, stats = normalize_pre_schema_references(candidate)

    assert normalized["roles"][0]["mention_id"] == "m_customer"
    assert normalized["roles"][1]["mention_id"] == "m_customer"
    assert normalized["roles"][2]["mention_id"] == "m_customer"
    assert normalized["operations"][0]["target_mention_ids"] == ["m_customer"]
    assert stats == {
        "enabled": True,
        "role_references_normalized": 3,
        "role_references_ambiguous": 0,
        "target_mentions_attached": 1,
        "target_concepts_derived": 0,
        "target_reference_ambiguous": 0,
        "ungrounded_operations_dropped": 0,
    }


def test_ungrounded_constraint_operation_is_dropped_instead_of_borrowing_a_target():
    text = "对比 GMV，只描述统计变化"
    boundary = text.index("，") + 1
    candidate = _candidate(
        text,
        clauses=[
            {"id": "c1", "span": {"start": 0, "end": boundary}},
            {"id": "c2", "span": {"start": boundary, "end": len(text)}},
        ],
        mentions=[_mention("m_gmv", "GMV", "GMV", 3, 6)],
        operations=[{
            "id": "op_compare", "clause_id": "c1", "kind": "COMPARE",
            "target_mention_ids": ["m_gmv"], "target_concept_ids": ["GMV"],
        }, {
            "id": "op_constraint", "clause_id": "c2", "kind": "ANALYZE",
            "target_mention_ids": [], "target_concept_ids": ["GMV"],
            "mentioned_arguments": {},
        }],
    )

    normalized, stats = normalize_pre_schema_references(candidate)

    assert [operation["id"] for operation in normalized["operations"]] == ["op_compare"]
    assert stats["ungrounded_operations_dropped"] == 1


def test_negative_operation_without_target_is_left_for_schema_failure():
    text = "对比 GMV，不要做"
    boundary = text.index("，") + 1
    candidate = _candidate(
        text,
        clauses=[
            {"id": "c1", "span": {"start": 0, "end": boundary}},
            {"id": "c2", "span": {"start": boundary, "end": len(text)}},
        ],
        mentions=[_mention("m_gmv", "GMV", "GMV", 3, 6)],
        operations=[{
            "id": "op_compare", "clause_id": "c1", "kind": "COMPARE",
            "target_mention_ids": ["m_gmv"], "target_concept_ids": ["GMV"],
        }, {
            "id": "op_veto", "clause_id": "c2", "kind": "COMPARE",
            "target_mention_ids": [], "target_concept_ids": ["GMV"],
            "polarity": "NEGATIVE",
        }],
    )

    normalized, stats = normalize_pre_schema_references(candidate)

    assert [operation["id"] for operation in normalized["operations"]] == [
        "op_compare", "op_veto",
    ]
    assert stats["ungrounded_operations_dropped"] == 0


def test_pre_schema_role_reference_with_multiple_existing_ids_is_left_unchanged():
    text = "客户和金额"
    candidate = _candidate(
        text,
        mentions=[
            _mention("m_customer", "客户", "CUSTOMER", 0, 2),
            _mention("m_money", "金额", "MONETARY", 3, 5),
        ],
    )
    candidate["roles"] = [{
        "id": "r_subject", "clause_id": "c1",
        "mention_id": {"left": "m_customer", "right": "m_money"},
        "concept_id": "CUSTOMER", "name": "analysis_subject",
    }]

    normalized, stats = normalize_pre_schema_references(candidate)

    assert normalized["roles"][0]["mention_id"] == {
        "left": "m_customer", "right": "m_money",
    }
    assert stats["role_references_normalized"] == 0
    assert stats["role_references_ambiguous"] == 1


def test_pre_schema_role_reference_does_not_cross_clause_or_concept_boundaries():
    text = "客户和金额"
    candidate = _candidate(
        text,
        mentions=[
            _mention("m_customer", "客户", "CUSTOMER", 0, 2),
            _mention("m_money", "金额", "MONETARY", 3, 5),
        ],
    )
    candidate["roles"] = [{
        "id": "r_subject", "clause_id": "c1",
        "mention_id": {"mention": {"id": "m_money"}},
        "concept_id": "CUSTOMER", "name": "analysis_subject",
    }]

    normalized, stats = normalize_pre_schema_references(candidate)

    assert normalized["roles"][0]["mention_id"] == {"mention": {"id": "m_money"}}
    assert stats["role_references_normalized"] == 0


def test_pre_schema_reference_normalization_leaves_ambiguous_targets_unresolved():
    text = "客户和客户"
    candidate = _candidate(
        text,
        mentions=[
            _mention("m_customer_1", "客户", "CUSTOMER", 0, 2),
            _mention("m_customer_2", "客户", "CUSTOMER", 3, 5),
        ],
        operations=[{
            "id": "op_explain", "clause_id": "c1", "kind": "EXPLAIN",
            "target_mention_ids": [], "target_concept_ids": ["CUSTOMER"],
        }],
    )

    normalized, stats = normalize_pre_schema_references(candidate)

    assert normalized["operations"][0]["target_mention_ids"] == []
    assert stats["target_mentions_attached"] == 0
    assert stats["target_reference_ambiguous"] == 1


def test_pre_schema_target_concept_is_derived_from_same_clause_target_mention():
    text = "解释客户"
    candidate = _candidate(
        text,
        mentions=[_mention("m_customer", "客户", "CUSTOMER", 2, 4)],
        operations=[{
            "id": "op_explain", "clause_id": "c1", "kind": "EXPLAIN",
            "target_mention_ids": ["m_customer"], "target_concept_ids": [],
        }],
    )

    normalized, stats = normalize_pre_schema_references(candidate)

    assert normalized["operations"][0]["target_concept_ids"] == ["CUSTOMER"]
    assert stats["target_concepts_derived"] == 1
    assert stats["target_mentions_attached"] == 0


def test_crossing_rfm_mention_merges_clause_and_remaps_implicit_rfm_operation():
    text = "Now RFM plan"
    candidate = _candidate(
        text,
        clauses=[
            {"id": "c1", "span": {"start": 0, "end": 4}},
            {"id": "c2", "span": {"start": 4, "end": 5}},
            {"id": "c3", "span": {"start": 5, "end": len(text)}},
        ],
        mentions=[_mention("m_rfm", "RFM", "RFM", 0, 3, clause_id="c1")],
        operations=[{
            "id": "op_rfm", "clause_id": "c1", "kind": "SEGMENT",
            "span": {"start": 0, "end": 4},
            "target_mention_ids": [], "target_concept_ids": ["CUSTOMER"],
            "mentioned_arguments": {"method": [{
                "mention_id": "m_rfm", "concept_id": "RFM", "stance": "ADOPTED",
                "evidence_span": {"start": 4, "end": 7},
            }]},
            "argument_negations": [], "mode": "EXECUTE", "polarity": "POSITIVE",
            "evidence_spans": [{"start": 0, "end": 4}],
        }],
    )

    normalized, stats = normalize_candidate(candidate, text)

    assert normalized["clauses"] == [
        {"id": "c1", "span": {"start": 0, "end": 4}},
        {"id": "c2", "span": {"start": 4, "end": len(text)}},
    ]
    assert normalized["mentions"][0]["clause_id"] == "c2"
    assert normalized["operations"][0]["clause_id"] == "c2"
    assert stats["clauses_merged"] == 1
    assert stats["operation_clause_remapped"] == 1

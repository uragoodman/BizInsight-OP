from __future__ import annotations

from copy import deepcopy

import pytest

from app.context.semantic_request_scope_normalizer import (
    normalize_cross_clause_negative_request_scope,
)


def _candidate(
    question: str = "按金额排序客户，不是按金额分层",
    *,
    operation_clause_id: str = "c2",
) -> dict:
    comma_end = question.index("，") + 1
    customer_start = question.index("客户")
    cue_start = question.index("不是")
    clauses = [
        {"id": "c1", "span": {"start": 0, "end": comma_end}},
        {"id": "c2", "span": {"start": comma_end, "end": len(question)}},
    ]
    clause = next(item for item in clauses if item["id"] == operation_clause_id)
    return {
        "turn_id": "turn-1",
        "clauses": clauses,
        "mentions": [{
            "id": "m_customer",
            "clause_id": "c1",
            "span": {"start": customer_start, "end": customer_start + 2},
            "surface": "客户",
            "concept_id": "CUSTOMER",
            "grounding_source": "llm_inferred",
            "source_turn_id": "turn-1",
        }],
        "operations": [{
            "id": "op_segment",
            "clause_id": operation_clause_id,
            "span": clause["span"],
            "kind": "SEGMENT",
            "target_mention_ids": ["m_customer"],
            "target_concept_ids": ["CUSTOMER"],
            "mentioned_arguments": {},
            "argument_negations": [],
            "mode": "ASK",
            "polarity": "NEGATIVE",
            "clause_scope": "SAME_CLAUSE",
            "negation_cue_span": {"start": cue_start, "end": cue_start + 2},
            "source_clause_id": None,
            "source_turn_id": None,
        }],
    }


def test_normalizes_only_scope_and_source_for_unique_grounded_cross_clause_target():
    question = "按金额排序客户，不是按金额分层"
    candidate = _candidate(question)
    original = deepcopy(candidate)

    normalized, stats = normalize_cross_clause_negative_request_scope(candidate, question)

    operation = normalized["operations"][0]
    original_operation = original["operations"][0]
    assert operation["clause_scope"] == "REQUEST"
    assert operation["source_clause_id"] == "c2"
    assert operation["source_turn_id"] == "turn-1"
    assert {
        key: value for key, value in operation.items()
        if key not in {"clause_scope", "source_clause_id", "source_turn_id"}
    } == {
        key: value for key, value in original_operation.items()
        if key not in {"clause_scope", "source_clause_id", "source_turn_id"}
    }
    assert candidate == original
    assert stats == {
        "enabled": True,
        "operations_seen": 1,
        "negative_same_clause_examined": 1,
        "request_scope_rewrites": 1,
    }


def test_does_not_rewrite_same_clause_negative_target():
    question = "不是按金额分层客户，按金额排序"
    candidate = _candidate(question, operation_clause_id="c1")

    normalized, stats = normalize_cross_clause_negative_request_scope(candidate, question)

    assert normalized == candidate
    assert stats["request_scope_rewrites"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda item: item["operations"][0].update(polarity="POSITIVE"),
        lambda item: item["operations"][0].update(target_mention_ids=["m_customer", "m_customer"]),
        lambda item: item["operations"][0].update(target_concept_ids=["MONETARY"]),
        lambda item: item["mentions"][0].update(
            grounding_source="inherited", span=None, source_turn_id="previous-turn",
        ),
        lambda item: item["mentions"][0].update(source_turn_id="previous-turn"),
        lambda item: item["operations"][0].update(
            negation_cue_span={"start": 0, "end": 1},
        ),
        lambda item: item["operations"][0].update(source_clause_id="c1"),
        lambda item: item["operations"][0].update(
            mentioned_arguments={"basis": [{"mention_id": "m_customer"}]},
        ),
        lambda item: item["clauses"][1]["span"].update(start=1),
    ],
    ids=[
        "positive-operation", "multiple-target-mentions", "concept-mismatch",
        "inherited-mention", "cross-turn-mention", "cue-outside-operation-clause",
        "source-already-populated", "argument-map-present", "invalid-clause-partition",
    ],
)
def test_unclear_or_ineligible_candidate_is_left_unchanged(mutation):
    question = "按金额排序客户，不是按金额分层"
    candidate = _candidate(question)
    mutation(candidate)
    original = deepcopy(candidate)

    normalized, stats = normalize_cross_clause_negative_request_scope(candidate, question)

    assert normalized == original
    assert stats["request_scope_rewrites"] == 0

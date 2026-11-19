from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.context.semantic_contract import (
    Entity,
    Mention,
    Operation,
    Relation,
    Role,
    SemanticContract,
    Span,
)


def _contract(**overrides) -> SemanticContract:
    values = {
        "mentions": (),
        "entities": (),
        "operations": (Operation(id="segment", mode="EXECUTE", polarity="POSITIVE"),),
        "request_mode": "EXECUTE",
        "dialogue_act_provisional": "NEW",
        "dialogue_act": "NEW",
        "modality": "FACTUAL",
        "roles": (),
        "relations": (),
        "semantic_status": "CONFIRMED",
        "context_sources": {"current_turn": [], "inherited": []},
        "frozen": True,
    }
    values.update(overrides)
    return SemanticContract(**values)


def test_span_uses_original_unicode_half_open_coordinates() -> None:
    text = "按购买频次和金额给客户分层"
    expectations = (("购买频次", 1, 5), ("金额", 6, 8), ("客户", 9, 11), ("分层", 11, 13))
    for value, start, end in expectations:
        span = Span(text=value, start=start, end=end)
        assert text[span.start : span.end] == value


def test_mention_requires_span_text_to_match() -> None:
    with pytest.raises(ValidationError):
        Mention(text="订单量", span=Span(text="订单", start=0, end=2), candidate_types=("metric",))


def test_roles_and_relations_require_existing_entities() -> None:
    with pytest.raises(ValidationError) as error:
        _contract(roles=(Role(entity="missing", role="analysis_subject"),))
    assert any("reference_integrity" in item["msg"] for item in error.value.errors())

    with pytest.raises(ValidationError) as error:
        _contract(relations=(Relation(subject="missing", predicate="GROUP_BY", object="also_missing"),))
    assert any("reference_integrity" in item["msg"] for item in error.value.errors())


def test_operation_does_not_need_to_be_an_entity() -> None:
    contract = _contract()
    assert contract.operations[0].id == "segment"
    assert contract.entities == ()


def test_serialized_contract_rejects_template_and_route_data() -> None:
    with pytest.raises(ValidationError):
        _contract(context_sources={"previous_route": "template:rfm_segmentation"})

    serialized = json.dumps(_contract().model_dump(mode="json"), ensure_ascii=False)
    assert "template:" not in serialized


def test_contract_rejects_unvalidated_inherited_context() -> None:
    with pytest.raises(ValidationError):
        _contract(
            context_sources={
                "current_turn": [],
                "inherited": [
                    {
                        "field": "metric",
                        "value": "gmv",
                        "source_turn": "turn-1",
                        "reason": "anaphora_resolution",
                        "validated": False,
                    }
                ],
            }
        )

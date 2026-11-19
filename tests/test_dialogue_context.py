from __future__ import annotations

from app.context.dialogue_context import resolve_dialogue_context
from app.context.semantic_parser import parse_semantics


def _history(*, completed: bool = False) -> list[dict]:
    turn = {
        "turn_id": "turn-1",
        "semantic_fields": {
            "analysis_subject": "customer",
            "segment_basis": "purchase_frequency",
            "operation": {"id": "segment", "mode": "EXECUTE", "polarity": "POSITIVE"},
        },
    }
    if completed:
        turn["status"] = "completed"
    return [turn]


def test_elliptical_time_followup_is_only_a_provisional_continuation() -> None:
    resolution = resolve_dialogue_context(
        "那上个月呢",
        history=_history(),
        routing_context={
            "previous_route": "template:rfm_segmentation",
            "source_turn": "turn-1",
        },
    )

    assert resolution.provisional_dialogue_act == "CONTINUE"
    assert resolution.inherited_fields
    assert all(item.validated is False for item in resolution.inherited_fields)
    assert resolution.routing_context["previous_route"] == "template:rfm_segmentation"
    assert all("template:" not in str(item.value) for item in resolution.inherited_fields)

    contract = parse_semantics("那上个月呢", dialogue_context=resolution)
    assert contract.dialogue_act_provisional == "CONTINUE"
    assert contract.dialogue_act == "CONTINUE"
    assert contract.context_sources["inherited"]
    assert all(item["validated"] is True for item in contract.context_sources["inherited"])
    assert "template:" not in contract.model_dump_json()


def test_explicit_switch_clears_inherited_fields() -> None:
    resolution = resolve_dialogue_context("不做这个了，查订单量", history=_history())
    assert resolution.provisional_dialogue_act == "SWITCH"
    assert resolution.inherited_fields == ()

    contract = parse_semantics("不做这个了，查订单量", dialogue_context=resolution)
    assert contract.dialogue_act == "SWITCH"
    assert {entity.id for entity in contract.entities} == {"order_count"}
    assert not contract.context_sources["inherited"]


def test_parser_validates_inherited_fields_individually() -> None:
    resolution = resolve_dialogue_context(
        "那上个月呢",
        history=[
            {
                "turn_id": "turn-1",
                "semantic_fields": {
                    "analysis_subject": "customer",
                    "unsupported_slot": "unknown-value",
                    "operations": [
                        {"id": "segment", "mode": "EXECUTE", "polarity": "POSITIVE"},
                        {"id": "template:rfm_segmentation", "mode": "EXECUTE", "polarity": "POSITIVE"},
                    ],
                },
            }
        ],
    )
    assert resolution.provisional_dialogue_act == "CONTINUE"
    assert all(item.validated is False for item in resolution.inherited_fields)

    contract = parse_semantics("那上个月呢", dialogue_context=resolution)
    assert {item["field"] for item in contract.context_sources["inherited"]} == {
        "analysis_subject", "operation"
    }
    assert all(item["validated"] is True for item in contract.context_sources["inherited"])
    assert "unknown-value" not in {item.id for item in contract.entities}
    assert contract.semantic_status == "AMBIGUOUS"


def test_completed_history_and_multiple_referents_do_not_inherit() -> None:
    completed = resolve_dialogue_context("那上个月呢", history=_history(completed=True))
    assert completed.provisional_dialogue_act == "AMBIGUOUS"
    assert completed.inherited_fields == ()

    multiple = resolve_dialogue_context(
        "那上个月呢",
        history=[
            {
                "turn_id": "turn-1",
                "semantic_fields": {"reference_candidates": ["customer", "seller"]},
            }
        ],
    )
    assert multiple.provisional_dialogue_act == "AMBIGUOUS"
    assert multiple.inherited_fields == ()


def test_resolver_skips_nonsemantic_history_tail_but_reads_its_routing_context() -> None:
    resolution = resolve_dialogue_context(
        "那上个月呢",
        history=[
            {
                "turn_id": "turn-1",
                "semantic_fields": {"analysis_subject": "customer"},
                "routing_context": {"previous_route": "template:rfm_segmentation"},
            },
            {"turn_id": "turn-1-answer", "role": "assistant", "content": "上轮已完成。"},
        ],
    )
    assert resolution.provisional_dialogue_act == "CONTINUE"
    assert {item.source_turn for item in resolution.inherited_fields} == {"turn-1"}
    assert resolution.routing_context["previous_route"] == "template:rfm_segmentation"


def test_plural_reference_and_negative_old_operation_block_inheritance() -> None:
    plural = resolve_dialogue_context("它们上个月呢", history=_history())
    assert plural.provisional_dialogue_act == "AMBIGUOUS"
    assert plural.inherited_fields == ()

    negative = resolve_dialogue_context(
        "不要做 RFM，查询订单量",
        history=[
            {
                "turn_id": "turn-1",
                "semantic_fields": {"analysis_subject": "customer", "operation": "rfm"},
            }
        ],
    )
    assert negative.provisional_dialogue_act == "SWITCH"
    assert negative.inherited_fields == ()

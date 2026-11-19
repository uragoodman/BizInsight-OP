from __future__ import annotations

from app.context.dialogue_context import resolve_dialogue_context
from app.context.semantic_parser import parse_semantics


def _entity(contract, entity_id: str):
    return next(item for item in contract.entities if item.id == entity_id)


def test_customer_segmentation_resolves_factors_and_spans() -> None:
    question = "按购买频次和金额给客户分层"
    contract = parse_semantics(question)
    expected_spans = {"购买频次": (1, 5), "金额": (6, 8), "客户": (9, 11), "分层": (11, 13)}
    actual_spans = {
        mention.text: (mention.span.start, mention.span.end)
        for mention in contract.mentions
        if mention.text in expected_spans
    }
    assert actual_spans == expected_spans
    assert _entity(contract, "customer").type == "business_entity"
    assert _entity(contract, "purchase_frequency").type == "semantic_factor"
    assert _entity(contract, "monetary").type == "semantic_factor"
    assert [item.id for item in contract.operations] == ["segment"]
    assert {(item.entity, item.role) for item in contract.roles} == {
        ("customer", "analysis_subject"),
        ("purchase_frequency", "segment_basis"),
        ("monetary", "segment_basis"),
    }
    assert {(item.subject, item.predicate, item.object) for item in contract.relations} == {
        ("customer", "GROUP_BY", "purchase_frequency"),
        ("customer", "GROUP_BY", "monetary"),
    }


def test_customer_count_is_metric_without_group_by() -> None:
    contract = parse_semantics("统计客户数量")
    assert {item.id for item in contract.entities} == {"customer_count"}
    assert not contract.relations
    assert contract.semantic_status == "CONFIRMED"


def test_order_count_grouped_by_customer_is_not_customer_count() -> None:
    contract = parse_semantics("按客户统计订单量")
    assert {item.id for item in contract.entities} == {"customer", "order_count"}
    assert ("order_count", "GROUP_BY", "customer") in {
        (item.subject, item.predicate, item.object) for item in contract.relations
    }
    assert "customer_count" not in {item.id for item in contract.entities}


def test_gmv_grouped_by_customer() -> None:
    contract = parse_semantics("按客户统计 GMV")
    assert {item.id for item in contract.entities} == {"customer", "gmv"}
    assert ("gmv", "GROUP_BY", "customer") in {
        (item.subject, item.predicate, item.object) for item in contract.relations
    }


def test_explain_rfm_does_not_create_execution_operation() -> None:
    contract = parse_semantics("什么是 RFM 分层")
    assert contract.request_mode == "EXPLAIN"
    assert [(item.id, item.mode) for item in contract.operations] == [("definition", "EXPLAIN")]
    assert contract.semantic_status == "CONFIRMED"


def test_compound_explain_and_query_preserves_both_operations() -> None:
    contract = parse_semantics("什么是 RFM，顺便查订单量")
    assert contract.request_mode == "EXPLAIN"
    assert [(item.id, item.mode) for item in contract.operations] == [
        ("definition", "EXPLAIN"),
        ("metric_query", "ASK"),
    ]
    assert {item.id for item in contract.entities} >= {"rfm", "order_count"}


def test_compound_negative_and_positive_operations_are_preserved() -> None:
    contract = parse_semantics("不要做 RFM，查询订单量")
    assert contract.request_mode == "ASK"
    assert [(item.id, item.mode, item.polarity) for item in contract.operations] == [
        ("rfm", "EXECUTE", "NEGATIVE"),
        ("metric_query", "ASK", "POSITIVE"),
    ]
    assert contract.semantic_status == "CONFIRMED"


def test_hypothetical_rfm_is_not_a_factual_execution() -> None:
    contract = parse_semantics("如果做 RFM 会怎样")
    assert contract.modality == "HYPOTHETICAL"
    assert contract.operations[0].id == "rfm"
    assert contract.operations[0].mode == "ASK"


def test_cause_question_is_ask_and_detects_explain_cause() -> None:
    contract = parse_semantics("为什么客户价值下降")
    assert contract.request_mode == "ASK"
    assert [item.id for item in contract.operations] == ["explain_cause"]


def test_parser_marks_inherited_fields_validated_only_after_compatibility_check() -> None:
    resolution = resolve_dialogue_context(
        "那上个月呢",
        history=[
            {
                "turn_id": "turn-1",
                "semantic_fields": {
                    "analysis_subject": "customer",
                    "segment_basis": "purchase_frequency",
                    "operation": {"id": "segment", "mode": "EXECUTE", "polarity": "POSITIVE"},
                },
            }
        ],
    )
    assert resolution.provisional_dialogue_act == "CONTINUE"
    assert all(item.validated is False for item in resolution.inherited_fields)
    contract = parse_semantics("那上个月呢", dialogue_context=resolution)
    assert contract.dialogue_act == "CONTINUE"
    assert {item.id for item in contract.entities} >= {"customer", "purchase_frequency", "previous_month"}
    assert contract.context_sources["inherited"]


def test_contract_is_template_free_even_when_routing_context_has_a_template_id() -> None:
    resolution = resolve_dialogue_context(
        "那上个月呢",
        history=[
            {
                "turn_id": "turn-1",
                "semantic_fields": {"analysis_subject": "customer", "segment_basis": "purchase_frequency"},
            }
        ],
        routing_context={"previous_route": "template:rfm_segmentation"},
    )
    contract = parse_semantics("那上个月呢", dialogue_context=resolution)
    serialized = contract.model_dump_json()
    assert "template:" not in serialized
    assert "rfm_segmentation" not in serialized


def test_insufficient_segment_basis_is_missing_not_confirmed() -> None:
    contract = parse_semantics("给客户分层")
    assert contract.semantic_status == "MISSING"

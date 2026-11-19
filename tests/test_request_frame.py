from __future__ import annotations

from app.context.request_frame import build_request_frame


def test_definition_is_explain_without_execution_operation() -> None:
    frame = build_request_frame("什么是 RFM 分层")
    assert frame.request_mode == "EXPLAIN"
    assert [(item.id, item.mode) for item in frame.operations] == [("definition", "EXPLAIN")]


def test_compound_definition_keeps_query_in_a_later_clause() -> None:
    frame = build_request_frame("什么是 RFM，顺便查订单量")
    assert frame.request_mode == "EXPLAIN"
    assert [(item.id, item.mode) for item in frame.operations] == [
        ("definition", "EXPLAIN"),
        ("metric_query", "ASK"),
    ]

    single_clause = build_request_frame("什么是 RFM 分层")
    assert [item.id for item in single_clause.operations] == ["definition"]


def test_negative_rfm_and_positive_metric_query_have_independent_modes() -> None:
    frame = build_request_frame("不要做 RFM，查询订单量")
    assert frame.request_mode == "ASK"
    assert [(item.id, item.mode, item.polarity) for item in frame.operations] == [
        ("rfm", "EXECUTE", "NEGATIVE"),
        ("metric_query", "ASK", "POSITIVE"),
    ]


def test_request_mode_is_ask_when_all_execute_operations_are_negative() -> None:
    frame = build_request_frame("不要做 RFM")
    assert frame.request_mode == "ASK"
    assert [(item.id, item.mode, item.polarity) for item in frame.operations] == [
        ("rfm", "EXECUTE", "NEGATIVE")
    ]


def test_conditional_rfm_is_hypothetical() -> None:
    frame = build_request_frame("如果做 RFM 会怎样")
    assert frame.modality == "HYPOTHETICAL"
    assert frame.operations[0].id == "rfm"
    assert frame.operations[0].mode == "ASK"


def test_cause_question_is_ask_with_explain_cause_operation() -> None:
    frame = build_request_frame("为什么客户价值下降")
    assert frame.request_mode == "ASK"
    assert [item.id for item in frame.operations] == ["explain_cause"]


def test_compound_cause_question_keeps_query_in_a_later_clause() -> None:
    frame = build_request_frame("为什么 GMV 下降，顺便查订单量")
    assert frame.request_mode == "ASK"
    assert [item.id for item in frame.operations] == ["explain_cause", "metric_query"]

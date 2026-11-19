"""Regression coverage for unconsumed comparison dimensions (P4-KD-001)."""

from __future__ import annotations

import pytest

from app.context.compiler import (
    _unrepresented_comparison_dimensions,
    compile_turn_context,
)
from app.context.dimension_intent import RoleAssignment
from app.core.config import Settings
from app.graph.workflow import _route_from_triage


_REPLAY_DATES = (
    "2016-10-17", "2016-11-24", "2017-01-02", "2017-02-09", "2017-03-20",
    "2017-04-27", "2017-06-05", "2017-07-13", "2017-08-20", "2017-09-28",
)
_EXTENSION_DATES = (
    "2016-10-18", "2016-11-11", "2016-12-07", "2016-12-31", "2017-01-26",
    "2017-02-20", "2017-03-17", "2017-04-11", "2017-05-07", "2017-05-31",
    "2017-06-26", "2017-07-21", "2017-08-15", "2017-09-09", "2017-10-05",
    "2017-10-29", "2017-11-24", "2017-12-19", "2018-01-13", "2018-02-07",
    "2018-03-05", "2018-03-29", "2018-04-24", "2018-05-19", "2018-06-13",
    "2018-07-08", "2018-08-03", "2018-08-27", "2018-09-22", "2018-10-16",
)
_P4_CASES = tuple(
    [(f"p4pilot-{index:03d}", day) for index, day in enumerate(_REPLAY_DATES, 21)]
    + [
        (f"l3-open_combo-{index:03d}", day)
        for index, day in enumerate(_EXTENSION_DATES, 1)
    ]
)


@pytest.fixture(autouse=True)
def _disable_semantic_verifier(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: Settings(
            semantic_verifier_mode="off",
            semantic_verifier_send_raw_question=False,
        ),
    )


def _compile_state(question: str, trace_id: str) -> dict:
    return compile_turn_context({
        "user_message": question,
        "trace_id": trace_id,
        "preference_context": {"source_needs": [], "candidate_analysis_plan": None},
    })


def _compile(question: str, trace_id: str) -> dict:
    return _compile_state(question, trace_id)["context_compilation"]


@pytest.mark.parametrize(
    ("case_id", "day"),
    _P4_CASES,
    ids=[case_id for case_id, _ in _P4_CASES],
)
def test_p4_open_category_payment_comparison_requests_fail_closed(case_id: str, day: str) -> None:
    state = _compile_state(
        f"按商品品类查看 {day} GMV，并比较支付方式",
        trace_id=case_id,
    )
    compiled = state["context_compilation"]

    roles = {
        item["dimension"]: item["role"]
        for item in compiled.get("dimension_roles") or []
    }
    assert roles["product_category"] == "grouping"
    assert roles["payment_type"] == "comparison"
    assert compiled["needs_clarification"] is True
    assert compiled["clarification_reason"] == "comparison_dimension_not_supported"

    contract = compiled["request_contract"]
    assert contract["schema_version"] == "analysis-request-contract-v1"
    assert contract["parse_status"] == "clarification_required"
    assert contract["uncertainty_reason"] == "comparison_dimension_not_supported"
    assert contract["dimension"] == "product_category"

    state["task_type"] = "metric_query"
    state["next_agent"] = "data_query"
    assert _route_from_triage(state) == "context_clarification"


@pytest.mark.parametrize(
    ("question", "expected_dimension"),
    [
        ("按商品品类查看 2016-10-18 GMV", "product_category"),
        ("按支付方式查看 2016-10-18 GMV", "payment_type"),
    ],
)
def test_single_dimension_breakdown_remains_resolved(
    question: str,
    expected_dimension: str,
) -> None:
    compiled = _compile(question, trace_id=f"single-{expected_dimension}")

    assert compiled["needs_clarification"] is False
    assert compiled["known_slots"]["dimension"]["value"] == expected_dimension
    assert compiled["request_contract"]["parse_status"] == "resolved"


def test_comparison_dimension_guard_accepts_future_multi_axis_contract_shape() -> None:
    assignment = RoleAssignment(
        dimension="future_dimension",
        start=10,
        end=20,
        role="comparison",
        evidence=("comparison_context",),
    )

    assert _unrepresented_comparison_dimensions(
        [assignment], {"dimension": "product_category"},
    ) == ("future_dimension",)
    assert _unrepresented_comparison_dimensions(
        [assignment], {
            "dimensions": [
                {"dimension_id": "product_category", "role": "grouping"},
                {"dimension_id": "future_dimension", "role": "comparison"},
            ],
        },
    ) == ()

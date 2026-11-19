import json
from pathlib import Path

import pytest

from app.context.compiler import (
    _comparison_resolution,
    _has_unnegated_marker,
    compile_turn_context,
    context_clarification,
)
from app.graph.workflow import _route_from_triage


def _use_gmv_registry(monkeypatch):
    monkeypatch.setattr(
        "app.context.compiler._metric_labels",
        lambda: {"gmv": "GMV", "average_order_value": "客单价"},
    )


def test_context_compiler_resolves_one_same_scope_metric(monkeypatch):
    _use_gmv_registry(monkeypatch)
    state = {
        "user_message": "那它有哪些限制？",
        "previous_turn_context": {
            "authorized_metrics": [{"metric_id": "gmv"}],
        },
    }

    result = compile_turn_context(state)["context_compilation"]

    assert result["relation"] == "continue"
    assert result["standalone_question"] == "那GMV有哪些限制？"
    assert result["known_slots"]["metric"]["value"] == "gmv"
    assert result["known_slots"]["metric"]["source"] == "previous_turn_context"
    assert result["known_slots"]["metric"]["confidence"] == 0.96
    assert result["resolved_references"][0]["text"] == "它"
    assert result["resolved_references"][0]["source"] == "previous_turn_context"
    assert result["needs_clarification"] is False


def test_context_compiler_resolves_elliptical_breakdown_request(monkeypatch):
    _use_gmv_registry(monkeypatch)
    state = {
        "user_message": "那按品类拆开",
        "previous_turn_context": {
            "authorized_metrics": [{"metric_id": "gmv"}],
        },
    }

    result = compile_turn_context(state)["context_compilation"]

    assert result["standalone_question"] == "GMV按品类拆开"
    assert result["resolved_references"][0]["text"] == "那"
    assert result["needs_clarification"] is False


def test_context_compiler_uses_validated_explorer_metric_without_chat_history(monkeypatch):
    _use_gmv_registry(monkeypatch)
    state = {
        "user_message": "\u90a3\u6309\u54c1\u7c7b\u62c6\u5f00",
        "use_current_explorer_context": True,
        "controlled_analysis_scope": {
            "metric_name": "gmv",
            "allowed_metrics": ["gmv", "average_order_value"],
            "default_start_date": "2018-08-01",
            "default_end_date": "2018-08-31",
        },
        "current_query_scope_fingerprint": "scope-fingerprint",
    }

    result = compile_turn_context(state)["context_compilation"]

    assert result["standalone_question"] == (
        "GMV\u6309\u54c1\u7c7b\u62c6\u5f00\uff0c\u65e5\u671f "
        "2018-08-01 \u81f3 2018-08-31"
    )
    assert result["resolved_references"][0]["source"] == "current_explorer_scope"
    assert result["known_slots"]["metric"] == {
        "value": "gmv",
        "source": "current_explorer_scope",
        "confidence": 0.99,
        "source_turn_id": None,
    }
    assert result["known_slots"]["dimension"]["value"] == "product_category"
    assert result["known_slots"]["date"] == {
        "value": "2018-08-01 \u81f3 2018-08-31",
        "source": "current_explorer_scope",
        "confidence": 0.99,
        "source_turn_id": None,
    }
    assert result["needs_clarification"] is False
    assert result["scope_fingerprint"] == "scope-fingerprint"


def test_context_compiler_does_not_use_unallowed_explorer_metric(monkeypatch):
    _use_gmv_registry(monkeypatch)
    state = {
        "user_message": "\u90a3\u6309\u54c1\u7c7b\u62c6\u5f00",
        "use_current_explorer_context": True,
        "controlled_analysis_scope": {
            "metric_name": "gmv",
            "allowed_metrics": ["average_order_value"],
        },
    }

    result = compile_turn_context(state)["context_compilation"]

    assert result["needs_clarification"] is True
    assert result["clarification_reason"] == "no_same_scope_metric_context"


def test_context_compiler_does_not_use_raw_explorer_payload(monkeypatch):
    _use_gmv_registry(monkeypatch)
    state = {
        "user_message": "\u90a3\u6309\u54c1\u7c7b\u62c6\u5f00",
        "use_current_explorer_context": True,
        "explorer_context": {"metric_name": "gmv", "allowed_metrics": ["gmv"]},
    }

    result = compile_turn_context(state)["context_compilation"]

    assert result["needs_clarification"] is True
    assert result["clarification_reason"] == "no_same_scope_metric_context"


def test_context_compiler_uses_explorer_metric_for_definition_but_not_query_dates(monkeypatch):
    _use_gmv_registry(monkeypatch)
    state = {
        "user_message": "\u5b83\u6709\u54ea\u4e9b\u9650\u5236\uff1f",
        "use_current_explorer_context": True,
        "controlled_analysis_scope": {
            "metric_name": "gmv",
            "allowed_metrics": ["gmv"],
            "default_start_date": "2018-08-01",
            "default_end_date": "2018-08-31",
        },
    }

    result = compile_turn_context(state)["context_compilation"]

    assert result["known_slots"]["metric"]["value"] == "gmv"
    assert result["intent"] == "retrieve_metric_definition"
    assert "date" not in result["known_slots"]
    assert result["needs_clarification"] is False


def test_current_explorer_selection_overrides_old_default_metric(monkeypatch):
    _use_gmv_registry(monkeypatch)
    monkeypatch.setattr(
        "app.context.compiler._metric_labels",
        lambda: {"gmv": "GMV", "order_count": "order_count"},
    )
    state = {
        "user_message": "\u90a3\u6309\u5468\u770b\u8d8b\u52bf",
        "use_current_explorer_context": True,
        "controlled_analysis_scope": {
            "metric_name": "order_count",
            "allowed_metrics": ["gmv", "order_count"],
            "default_start_date": "2018-08-01",
            "default_end_date": "2018-08-31",
        },
        "previous_turn_context": {
            "authorized_metrics": [{"metric_id": "gmv"}],
            "context_slots": {"metric": {"value": "gmv"}},
            "source_context_snapshot": {"scope": {}, "metric_name": "gmv"},
        },
    }

    result = compile_turn_context(state)["context_compilation"]

    assert result["known_slots"]["metric"]["value"] == "order_count"
    assert result["known_slots"]["metric"]["source"] == "current_explorer_scope"
    assert result["known_slots"]["metric"]["source_turn_id"] is None
    assert result["resolved_references"][0]["value"] == "order_count"
    assert result["known_slots"]["operation"]["value"] == "trend"


def test_same_explorer_selection_preserves_more_specific_chat_metric(monkeypatch):
    _use_gmv_registry(monkeypatch)
    monkeypatch.setattr(
        "app.context.compiler._metric_labels",
        lambda: {"gmv": "GMV", "order_count": "order_count"},
    )
    state = {
        "user_message": "\u90a3\u6309\u5468\u770b\u8d8b\u52bf",
        "use_current_explorer_context": True,
        "controlled_analysis_scope": {
            "metric_name": "gmv",
            "allowed_metrics": ["gmv", "order_count"],
            "default_start_date": "2018-08-01",
            "default_end_date": "2018-08-31",
        },
        "previous_turn_context": {
            "authorized_metrics": [{"metric_id": "order_count"}],
            "context_slots": {"metric": {"value": "order_count"}},
            "source_context_snapshot": {"scope": {}, "metric_name": "gmv"},
        },
    }

    result = compile_turn_context(state)["context_compilation"]

    assert result["known_slots"]["metric"]["value"] == "order_count"
    assert result["known_slots"]["metric"]["source"] == "previous_turn_context"


def test_context_compiler_clarifies_multiple_candidate_metrics(monkeypatch):
    _use_gmv_registry(monkeypatch)
    state = {
        "user_message": "那它有哪些限制？",
        "previous_turn_context": {
            "authorized_metrics": [
                {"metric_id": "gmv"},
                {"metric_id": "average_order_value"},
            ],
        },
    }

    result = compile_turn_context(state)["context_compilation"]

    assert result["standalone_question"] == state["user_message"]
    assert result["needs_clarification"] is True
    assert result["clarification_reason"] == "multiple_prior_metrics"
    assert _route_from_triage({"context_compilation": result, "errors": []}) == "context_clarification"


def test_context_compiler_clarifies_reference_without_prior_context(monkeypatch):
    _use_gmv_registry(monkeypatch)
    state = {"user_message": "那它有哪些限制？"}

    result = compile_turn_context(state)["context_compilation"]

    assert result["needs_clarification"] is True
    assert result["clarification_reason"] == "no_same_scope_metric_context"
    clarified = context_clarification(state)
    assert clarified["metric_planning_decision"]["status"] == "clarification_required"
    assert clarified["answer"]


def test_ambiguous_metric_clarification_preserves_registered_candidates():
    state = {
        "context_compilation": {
            "clarification_reason": "ambiguous_metric",
            "missing_slots": ["metric"],
            "metric_candidates": [
                {"metric_id": "gmv", "label": "GMV"},
                {"metric_id": "average_order_value", "label": "客单价"},
            ],
        }
    }

    clarified = context_clarification(state)

    assert clarified["metric_planning_decision"]["ambiguous_candidates"] == [
        {"metric_id": "gmv", "display_name": "GMV"},
        {"metric_id": "average_order_value", "display_name": "客单价"},
    ]


def test_comparison_dimension_clarification_has_stable_code_and_actionable_chinese_copy():
    state = {
        "context_compilation": {
            "clarification_reason": "comparison_dimension_not_supported",
            "missing_slots": ["comparison_dimension"],
        }
    }

    clarified = context_clarification(state)

    decision = clarified["metric_planning_decision"]
    assert decision["reason"] == "comparison_dimension_not_supported"
    assert decision["reason_code"] == "comparison_dimension_not_supported"
    assert "商品品类" in clarified["answer"]
    assert "支付方式" in clarified["answer"]
    assert "单独查询" in clarified["answer"]


def test_context_compiler_does_not_treat_this_month_as_a_reference(monkeypatch):
    _use_gmv_registry(monkeypatch)
    state = {"user_message": "这个月 GMV 有哪些变化？"}

    result = compile_turn_context(state)["context_compilation"]

    assert result["relation"] == "independent"
    assert result["standalone_question"] == state["user_message"]
    assert result["needs_clarification"] is False


def test_context_compiler_preserves_natural_month_contribution_roles(monkeypatch):
    monkeypatch.setattr(
        "app.context.compiler._metric_labels",
        lambda: {"freight_rate": "运费率"},
    )
    state = {
        "user_message": (
            "请比较平台运费率在2017年3月与2017年4月的变化，按 month_position "
            "说明各位置的统计贡献；只陈述数值，不推断业务因果。"
        )
    }

    result = compile_turn_context(state)["context_compilation"]

    assert result["known_slots"]["dimension"]["value"] == "month_position"
    assert result["operation_intent"]["kind"] == "dimensional_statistical_contribution"
    assert result["operation_intent"]["comparison_periods"] == {
        "baseline": {"start_date": "2017-03-01", "end_date": "2017-03-31"},
        "current": {"start_date": "2017-04-01", "end_date": "2017-04-30"},
        "granularity": "month",
    }


def _prior_completed_month_comparison(*, scope="scope-a", version="manifest-a", query_type="metric_compare"):
    return {
        "source_turn_id": "turn-prior-comparison",
        "scope_fingerprint": scope,
        "authorized_metrics": [{"metric_id": "gmv"}],
        "context_slots": {"metric": {"value": "gmv", "source": "current_turn"}},
        "result_context": {
            "version": "result-context-v1",
            "source_turn_id": "turn-prior-comparison",
            "source_trace_id": "trace-prior-comparison",
            "scope_fingerprint": scope,
            "data_version_manifest_hash": version,
            "metric_id": "gmv",
            "query_type": query_type,
            "result_status": "completed",
            "comparison": {
                "metric_name": "gmv", "baseline_value": 100.0,
                "current_value": 120.0, "delta_value": 20.0,
                "baseline_period": "2018-07-01 至 2018-07-31",
                "current_period": "2018-08-01 至 2018-08-31",
                "evidence_ids": ["ev-prior-comparison"],
            },
            "query_period": {
                "baseline_start": "2018-07-01", "baseline_end": "2018-07-31",
                "current_start": "2018-08-01", "current_end": "2018-08-31",
            },
            "evidence_ids": ["ev-prior-comparison"],
        },
    }


def test_context_compiler_inherits_trusted_comparison_period_before_contract_seal(monkeypatch):
    _use_gmv_registry(monkeypatch)
    previous = _prior_completed_month_comparison()
    state = {
        "user_message": "再按商品品类解释这个变化的统计贡献，只说查询支持的内容。",
        "previous_turn_context": previous,
        "chat_session_id": "session-a",
        "current_query_scope_fingerprint": "scope-a",
        "data_version_manifest_hash": "manifest-a",
    }

    compilation = compile_turn_context(state)["context_compilation"]
    contract = compilation["request_contract"]

    assert compilation["relation"] == "continue"
    assert compilation["needs_clarification"] is False
    assert compilation["known_slots"]["metric"]["value"] == "gmv"
    assert compilation["known_slots"]["dimension"]["value"] == "product_category"
    assert compilation["known_slots"]["comparison_period"]["source"] == "previous_result_context"
    assert compilation["known_slots"]["comparison_period"]["source_turn_id"] == "turn-prior-comparison"
    assert compilation["operation_intent"]["kind"] == "dimensional_statistical_contribution"
    expected_periods = {
        "baseline": {"start_date": "2018-07-01", "end_date": "2018-07-31"},
        "current": {"start_date": "2018-08-01", "end_date": "2018-08-31"},
        "granularity": "month",
    }
    assert compilation["operation_intent"]["comparison_periods"] == expected_periods
    assert contract["comparison"] == {"kind": "period_pair", **expected_periods}
    assert contract["required_tool"] == "compare_breakdown_by_dimension_period"
    assert contract["parse_status"] == "resolved"
    assert "source_turn_id" not in contract
    assert "previous_result_context" in compilation["context_sources"]
    from app.context.request_contract import validate_compiled_request_contract
    assert validate_compiled_request_contract(compilation) == (True, None)


@pytest.mark.parametrize(
    ("question", "previous", "current_scope", "current_version", "expected_inherit"),
    [
        (
            "再按商品品类解释这个变化的统计贡献。",
            _prior_completed_month_comparison(), "scope-a", "manifest-a", True,
        ),
        (
            "再按商品品类解释这个变化的统计贡献。",
            _prior_completed_month_comparison(scope="scope-old"), "scope-a", "manifest-a", False,
        ),
        (
            "再按商品品类解释这个变化的统计贡献。",
            _prior_completed_month_comparison(version="manifest-old"), "scope-a", "manifest-a", False,
        ),
        (
            "再按商品品类解释这个变化的统计贡献。",
            _prior_completed_month_comparison(query_type="metric_breakdown"), "scope-a", "manifest-a", False,
        ),
        (
            "按商品品类解释 GMV 的统计贡献。",
            _prior_completed_month_comparison(), "scope-a", "manifest-a", False,
        ),
        (
            "再按商品品类解释这个变化的统计贡献，2018年9月。",
            _prior_completed_month_comparison(), "scope-a", "manifest-a", False,
        ),
        (
            "再按商品品类解释这个变化的统计贡献，比较2018-08-31。",
            _prior_completed_month_comparison(), "scope-a", "manifest-a", False,
        ),
    ],
)
def test_context_compiler_period_inheritance_requires_explicit_trusted_continuation(
    monkeypatch, question, previous, current_scope, current_version, expected_inherit,
):
    _use_gmv_registry(monkeypatch)
    compilation = compile_turn_context({
        "user_message": question,
        "previous_turn_context": previous,
        "chat_session_id": "session-a",
        "current_query_scope_fingerprint": current_scope,
        "data_version_manifest_hash": current_version,
    })["context_compilation"]

    inherited = (
        compilation.get("inherited_comparison_source_turn_id") == "turn-prior-comparison"
        and compilation.get("known_slots", {}).get("comparison_period", {}).get("source")
        == "previous_result_context"
    )
    assert inherited is expected_inherit
    if expected_inherit:
        assert compilation["request_contract"]["comparison"].get("kind") == "period_pair"
        assert compilation["request_contract"]["parse_status"] == "resolved"
        assert compilation["request_contract"]["required_tool"] == "compare_breakdown_by_dimension_period"
    else:
        assert compilation.get("inherited_comparison_source_turn_id") is None
        assert compilation.get("known_slots", {}).get("comparison_period", {}).get("source") != "previous_result_context"
        if "2018年9月" in question or "2018-08-31" in question:
            assert compilation["request_contract"]["parse_status"] == "clarification_required"


def test_context_compiler_persists_prior_metric_slot_for_rag_followups(monkeypatch):
    _use_gmv_registry(monkeypatch)
    state = {
        "user_message": "那它有哪些限制？",
        "previous_turn_context": {
            "context_slots": {"metric": {"value": "gmv", "source": "current_turn"}},
        },
    }

    result = compile_turn_context(state)["context_compilation"]

    assert result["standalone_question"] == "那GMV有哪些限制？"
    assert result["resolved_references"][0]["source"] == "previous_turn_context"


def test_context_compiler_extracts_typed_current_turn_slots(monkeypatch):
    _use_gmv_registry(monkeypatch)
    state = {"user_message": "2026-09-01 按地区拆分 GMV"}

    result = compile_turn_context(state)["context_compilation"]

    assert result["known_slots"]["metric"]["value"] == "gmv"
    assert result["known_slots"]["date"]["value"] == "2026-09-01"
    # The compiler now resolves natural-language geography to the Registry's
    # canonical customer_state dimension instead of emitting a non-registered
    # generic `region` token.
    assert result["known_slots"]["dimension"]["value"] == "customer_state"
    assert result["known_slots"]["operation"]["value"] == "breakdown"


def test_context_compiler_does_not_read_delivered_order_scope_as_order_count_metric():
    """A bare “订单” before 的 + target metric is scope wording, not a second metric."""
    cases = (
        ("成交总额", "gmv"),
        ("客单价", "avg_order_value"),
        ("运费率", "freight_rate"),
    )
    for target_label, expected_metric in cases:
        compilation = compile_turn_context({
            "user_message": (
                f"请比较 2017-01-05 与 2017-01-06 平台已交付订单的{target_label}，"
                "按已注册分解方法拆分数值变化贡献；只作数值解释，不推断业务因果。"
            ),
        })["context_compilation"]

        assert compilation["needs_clarification"] is False
        assert compilation["known_slots"]["metric"]["value"] == expected_metric
        assert compilation["operation_intent"]["target_metric_id"] == expected_metric


def test_context_compiler_does_not_bind_bare_order_semantic_term():
    compilation = compile_turn_context({
        "user_message": "请比较 2017-01-05 与 2017-01-06 平台已交付订单的变化。",
    })["context_compilation"]

    assert compilation["needs_clarification"] is False
    assert "metric" not in compilation["known_slots"]


def test_data_followup_inherits_prior_date_only_from_same_scope_context(monkeypatch):
    _use_gmv_registry(monkeypatch)
    state = {
        "user_message": "那按品类拆开",
        "previous_turn_context": {
            "authorized_metrics": [{"metric_id": "gmv"}],
            "context_slots": {"date": {"value": "2018-08-29", "source": "current_turn"}},
        },
    }

    result = compile_turn_context(state)["context_compilation"]

    assert "2018-08-29" in result["standalone_question"]
    assert result["known_slots"]["date"]["source"] == "previous_turn_context"
    assert result["known_slots"]["dimension"]["value"] == "product_category"


def _trusted_prior_context(**slots):
    return {
        "source_turn_id": "prior-turn",
        "scope_fingerprint": "scope-a",
        "context_slots": slots,
    }


def test_explicit_metric_followup_inherits_same_scope_date(monkeypatch):
    _use_gmv_registry(monkeypatch)
    monkeypatch.setattr(
        "app.context.compiler._metric_labels",
        lambda: {"gmv": "GMV", "order_count": "订单量"},
    )
    state = {
        "user_message": "那订单量呢？",
        "current_query_scope_fingerprint": "scope-a",
        "previous_turn_context": _trusted_prior_context(
            metric={"value": "gmv", "source": "current_turn"},
            date={"value": "2018-08-20", "source": "current_turn"},
        ),
    }

    compilation = compile_turn_context(state)["context_compilation"]

    assert compilation["relation"] == "continue"
    assert compilation["known_slots"]["metric"] == {
        "value": "order_count",
        "source": "current_turn",
        "confidence": 1.0,
        "source_turn_id": None,
    }
    assert compilation["known_slots"]["date"]["value"] == "2018-08-20"
    assert compilation["known_slots"]["date"]["source"] == "previous_turn_context"
    assert compilation["known_slots"]["date"]["source_turn_id"] == "prior-turn"
    assert "2018-08-20" in compilation["standalone_question"]
    assert compilation["request_contract"]["metric_id"] == "order_count"


def test_result_reference_followup_inherits_trusted_date(monkeypatch):
    _use_gmv_registry(monkeypatch)
    monkeypatch.setattr(
        "app.context.compiler._metric_labels",
        lambda: {"gmv": "GMV", "order_count": "订单量"},
    )
    state = {
        "user_message": "刚才查询的 GMV 数值是多少？请基于上一轮结果回答。",
        "current_query_scope_fingerprint": "scope-a",
        "previous_turn_context": _trusted_prior_context(
            metric={"value": "gmv", "source": "current_turn"},
            date={"value": "2018-08-20", "source": "current_turn"},
            operation={"value": "value", "source": "current_turn"},
        ),
    }

    compilation = compile_turn_context(state)["context_compilation"]

    assert compilation["relation"] == "continue"
    assert compilation["known_slots"]["metric"]["source"] == "current_turn"
    assert compilation["known_slots"]["date"]["value"] == "2018-08-20"
    assert compilation["known_slots"]["date"]["source"] == "previous_turn_context"
    assert "2018-08-20" in compilation["standalone_question"]


def test_same_day_dimension_switch_inherits_date_but_uses_current_dimensions(monkeypatch):
    _use_gmv_registry(monkeypatch)
    state = {
        "user_message": "同一天改按支付方式拆分 GMV。",
        "current_query_scope_fingerprint": "scope-a",
        "previous_turn_context": _trusted_prior_context(
            metric={"value": "gmv", "source": "current_turn"},
            date={"value": "2018-08-20", "source": "current_turn"},
            dimension={"value": "product_category", "source": "current_turn"},
            operation={"value": "breakdown", "source": "current_turn"},
        ),
    }

    compilation = compile_turn_context(state)["context_compilation"]

    assert compilation["relation"] == "continue"
    assert compilation["known_slots"]["date"]["value"] == "2018-08-20"
    assert compilation["known_slots"]["dimension"]["value"] == "payment_type"
    assert compilation["known_slots"]["dimension"]["source"] == "current_turn"
    assert "2018-08-20" in compilation["standalone_question"]


def test_explicit_followup_date_wins_over_previous_date(monkeypatch):
    _use_gmv_registry(monkeypatch)
    state = {
        "user_message": "那 GMV 呢？改为 2018-08-21。",
        "current_query_scope_fingerprint": "scope-a",
        "previous_turn_context": _trusted_prior_context(
            metric={"value": "gmv", "source": "current_turn"},
            date={"value": "2018-08-20", "source": "current_turn"},
        ),
    }

    compilation = compile_turn_context(state)["context_compilation"]

    assert compilation["relation"] == "continue"
    assert compilation["known_slots"]["date"]["value"] == "2018-08-21"
    assert compilation["known_slots"]["date"]["source"] == "current_turn"
    assert "2018-08-20" not in compilation["standalone_question"]


def test_followup_without_trusted_predecessor_does_not_invent_date(monkeypatch):
    _use_gmv_registry(monkeypatch)
    monkeypatch.setattr(
        "app.context.compiler._metric_labels",
        lambda: {"gmv": "GMV", "order_count": "订单量"},
    )

    compilation = compile_turn_context({
        "user_message": "那订单量呢？",
        "current_query_scope_fingerprint": "scope-a",
    })["context_compilation"]

    assert compilation["relation"] == "continue"
    assert "date" not in compilation["known_slots"]
    assert "2018-" not in compilation["standalone_question"]


def test_previous_turn_from_different_scope_is_not_inherited(monkeypatch):
    _use_gmv_registry(monkeypatch)
    monkeypatch.setattr(
        "app.context.compiler._metric_labels",
        lambda: {"gmv": "GMV", "order_count": "订单量"},
    )

    compilation = compile_turn_context({
        "user_message": "那订单量呢？",
        "current_query_scope_fingerprint": "scope-b",
        "previous_turn_context": _trusted_prior_context(
            metric={"value": "gmv", "source": "current_turn"},
            date={"value": "2018-08-20", "source": "current_turn"},
        ),
    })["context_compilation"]

    assert compilation["relation"] == "continue"
    assert "date" not in compilation["known_slots"]
    assert compilation["source_turn_id"] is None


def test_inherited_date_outside_current_scope_requires_clarification(monkeypatch):
    _use_gmv_registry(monkeypatch)
    monkeypatch.setattr(
        "app.context.compiler._metric_labels",
        lambda: {"gmv": "GMV", "order_count": "订单量"},
    )

    compilation = compile_turn_context({
        "user_message": "那按品类拆开",
        "current_query_scope_fingerprint": "scope-a",
        "use_current_explorer_context": True,
        "controlled_analysis_scope": {
            "default_start_date": "2018-08-21",
            "default_end_date": "2018-08-31",
            "allowed_metrics": ["gmv"],
            "allowed_dimensions": ["product_category"],
        },
        "previous_turn_context": _trusted_prior_context(
            metric={"value": "gmv", "source": "current_turn"},
            date={"value": "2018-08-20", "source": "current_turn"},
        ),
    })["context_compilation"]

    assert compilation["needs_clarification"] is True
    assert compilation["clarification_reason"] == "date_outside_locked_scope"
    assert "date" not in compilation["known_slots"]
    assert compilation["request_contract"]["parse_status"] == "clarification_required"


def test_explicit_metric_does_not_inherit_unsupported_prior_dimension(monkeypatch):
    _use_gmv_registry(monkeypatch)
    monkeypatch.setattr(
        "app.context.compiler._metric_labels",
        lambda: {"gmv": "GMV", "order_count": "订单量"},
    )

    compilation = compile_turn_context({
        "user_message": "那订单量呢？",
        "current_query_scope_fingerprint": "scope-a",
        "previous_turn_context": _trusted_prior_context(
            metric={"value": "gmv", "source": "current_turn"},
            date={"value": "2018-08-20", "source": "current_turn"},
            dimension={"value": "payment_type", "source": "current_turn"},
            operation={"value": "breakdown", "source": "current_turn"},
        ),
    })["context_compilation"]

    assert compilation["needs_clarification"] is True
    assert compilation["clarification_reason"] == "dimension_not_supported_by_metric"
    assert "dimension" not in compilation["known_slots"]
    assert compilation["scope_candidates"][-1]["field"] == "dimension"


def test_explicit_metric_does_not_inherit_unsupported_prior_operation(monkeypatch):
    _use_gmv_registry(monkeypatch)
    monkeypatch.setattr(
        "app.context.compiler._metric_labels",
        lambda: {"gmv": "GMV", "order_count": "订单量"},
    )

    compilation = compile_turn_context({
        "user_message": "那订单量呢？",
        "current_query_scope_fingerprint": "scope-a",
        "controlled_analysis_scope": {
            "seller_id": "seller-1",
            "allowed_metrics": ["order_count"],
        },
        "previous_turn_context": _trusted_prior_context(
            metric={"value": "gmv", "source": "current_turn"},
            operation={"value": "compare", "source": "current_turn"},
        ),
    })["context_compilation"]

    assert compilation["needs_clarification"] is True
    assert compilation["clarification_reason"] == "capability_not_enabled"
    assert "operation" not in compilation["known_slots"]


def test_single_explicit_month_is_a_period_slot_not_a_point_date(monkeypatch):
    _use_gmv_registry(monkeypatch)

    compilation = compile_turn_context({
        "user_message": "查询 2018 年 8 月的 GMV。",
    })["context_compilation"]

    assert compilation["needs_clarification"] is False
    assert compilation["known_slots"]["period"]["value"] == {
        "start_date": "2018-08-01", "end_date": "2018-08-31",
    }
    assert "date" not in compilation["known_slots"]
    assert compilation["request_contract"]["comparison"]["kind"] == "none"


def test_period_comparison_followup_combines_same_scope_prior_period(monkeypatch):
    _use_gmv_registry(monkeypatch)
    monkeypatch.setattr(
        "app.context.compiler._metric_labels",
        lambda: {"gmv": "GMV", "order_count": "订单量"},
    )
    state = {
        "user_message": "和 2018 年 7 月相比呢？请比较 GMV 并说明变化方向。",
        "current_query_scope_fingerprint": "scope-a",
        "previous_turn_context": _trusted_prior_context(
            metric={"value": "gmv", "source": "current_turn"},
            period={
                "value": {"start_date": "2018-08-01", "end_date": "2018-08-31"},
                "source": "current_turn",
            },
        ),
    }

    compilation = compile_turn_context(state)["context_compilation"]

    expected = {
        "kind": "period_pair",
        "baseline": {"start_date": "2018-07-01", "end_date": "2018-07-31"},
        "current": {"start_date": "2018-08-01", "end_date": "2018-08-31"},
        "granularity": "month",
    }
    assert compilation["relation"] == "continue"
    assert compilation["needs_clarification"] is False
    assert compilation["request_contract"]["comparison"] == expected
    assert compilation["operation_intent"]["comparison_periods"] == {
        "baseline": expected["baseline"],
        "current": expected["current"],
        "granularity": "month",
    }


def test_explicit_two_month_comparison_does_not_inherit_prior_period(monkeypatch):
    _use_gmv_registry(monkeypatch)
    state = {
        "user_message": "比较 2018 年 7 月与 2018 年 8 月的 GMV。",
        "current_query_scope_fingerprint": "scope-a",
        "previous_turn_context": _trusted_prior_context(
            metric={"value": "gmv", "source": "current_turn"},
            period={
                "value": {"start_date": "2018-09-01", "end_date": "2018-09-30"},
                "source": "current_turn",
            },
        ),
    }

    compilation = compile_turn_context(state)["context_compilation"]

    assert compilation["needs_clarification"] is False
    assert compilation["request_contract"]["comparison"] == {
        "kind": "period_pair",
        "baseline": {"start_date": "2018-07-01", "end_date": "2018-07-31"},
        "current": {"start_date": "2018-08-01", "end_date": "2018-08-31"},
        "granularity": "month",
    }


def test_period_comparison_followup_fails_closed_without_matching_prior(monkeypatch):
    _use_gmv_registry(monkeypatch)
    monkeypatch.setattr(
        "app.context.compiler._metric_labels",
        lambda: {"gmv": "GMV", "order_count": "订单量"},
    )
    query = "和 2018 年 7 月相比呢？请比较 GMV。"

    for previous, current_scope in (
        (None, "scope-a"),
        (_trusted_prior_context(
            metric={"value": "gmv", "source": "current_turn"},
            period={"value": {"start_date": "2018-08-01", "end_date": "2018-08-31"}},
        ), "scope-b"),
        (_trusted_prior_context(
            metric={"value": "order_count", "source": "current_turn"},
            period={"value": {"start_date": "2018-08-01", "end_date": "2018-08-31"}},
        ), "scope-a"),
    ):
        compilation = compile_turn_context({
            "user_message": query,
            "current_query_scope_fingerprint": current_scope,
            "previous_turn_context": previous or {},
        })["context_compilation"]

        assert compilation["needs_clarification"] is True
        assert compilation["clarification_reason"] == "comparison_period_context_unavailable"
        assert compilation["request_contract"]["parse_status"] == "clarification_required"


def test_explorer_same_day_followup_inherits_exact_date_not_range_end(monkeypatch):
    _use_gmv_registry(monkeypatch)
    monkeypatch.setattr(
        "app.context.compiler._metric_labels",
        lambda: {"gmv": "GMV", "order_count": "订单量"},
    )
    base = {
        "user_message": "那这家店铺当天的订单量呢？",
        "current_query_scope_fingerprint": "scope-a",
        "use_current_explorer_context": True,
        "controlled_analysis_scope": {
            "seller_id": "seller-a",
            "default_start_date": "2018-01-01",
            "default_end_date": "2018-08-31",
            "allowed_metrics": ["gmv", "order_count"],
        },
    }

    inherited = compile_turn_context({
        **base,
        "previous_turn_context": _trusted_prior_context(
            metric={"value": "gmv", "source": "current_turn"},
            date={"value": "2018-08-31", "source": "current_turn"},
        ),
    })["context_compilation"]

    assert inherited["needs_clarification"] is False
    assert inherited["known_slots"]["date"]["value"] == "2018-08-31"
    assert inherited["known_slots"]["date"]["source"] == "previous_turn_context"
    assert "2018-01-01 至 2018-08-31" not in inherited["standalone_question"]

    for prior_date in (None, "2018-01-01 至 2018-08-31"):
        previous = _trusted_prior_context(
            metric={"value": "gmv", "source": "current_turn"},
            **({"date": {"value": prior_date, "source": "current_explorer_scope"}} if prior_date else {}),
        )
        unresolved = compile_turn_context({
            **base, "previous_turn_context": previous,
        })["context_compilation"]
        assert unresolved["needs_clarification"] is True
        assert unresolved["clarification_reason"] == "exact_date_context_unavailable"
        assert "date" not in unresolved["known_slots"]
        assert "2018-08-31" not in unresolved["standalone_question"]


def test_contextual_breakdown_keeps_user_preposition_after_metric_expansion(monkeypatch):
    _use_gmv_registry(monkeypatch)
    result = compile_turn_context({
        "user_message": "按品类拆开看看",
        "previous_turn_context": {
            "authorized_metrics": [{"metric_id": "gmv"}],
            "context_slots": {"date": {"value": "2018-08-01", "source": "current_turn"}},
        },
    })["context_compilation"]

    assert result["standalone_question"].startswith("GMV按品类拆开看看")
    assert result["known_slots"]["metric"]["value"] == "gmv"
    assert result["known_slots"]["dimension"]["value"] == "product_category"


def test_task_progress_is_loaded_only_for_matching_scope_and_data_version(monkeypatch):
    _use_gmv_registry(monkeypatch)
    task_progress = {
        "scope_fingerprint": "scope-a",
        "data_version_manifest_hash": "manifest-a",
        "approved_claims": [{"value": "已审核历史结论", "source_evidence_ids": ["ev-1"]}],
        "evidence_refs": [{"evidence_id": "ev-1", "trusted": True}],
        "hypothesis_chain_summary": {"gmv": [{"hypothesis": "品类变化", "verdict": "supported"}]},
    }
    state = {
        "user_message": "继续分析",
        "current_query_scope_fingerprint": "scope-a",
        "data_version_manifest_hash": "manifest-a",
        "task_progress_context": task_progress,
    }
    result = compile_turn_context(state)
    assert result["context_compilation"]["task_progress_context_used"] is True
    assert result["historical_claims_context"][0]["value"] == "已审核历史结论"
    assert result["historical_evidence_context"][0]["evidence_id"] == "ev-1"

    changed = compile_turn_context({
        "user_message": "继续分析",
        "current_query_scope_fingerprint": "scope-a",
        "data_version_manifest_hash": "manifest-b",
        "task_progress_context": task_progress,
    })
    assert changed["context_compilation"]["task_progress_context_used"] is False
    assert changed.get("historical_claims_context", []) == []

    unversioned = compile_turn_context({
        "user_message": "继续分析",
        "current_query_scope_fingerprint": "scope-a",
        "data_version_manifest_hash": "",
        "task_progress_context": {**task_progress, "data_version_manifest_hash": ""},
    })
    assert unversioned["context_compilation"]["task_progress_context_used"] is False
    assert unversioned.get("historical_claims_context", []) == []


def test_context_compiler_evaluation_cases(monkeypatch):
    _use_gmv_registry(monkeypatch)
    cases = json.loads(
        (Path(__file__).parents[1] / "docs/testing/context-compiler-eval.json").read_text(encoding="utf-8")
    )["cases"]
    for case in cases:
        result = compile_turn_context({
            "user_message": case["question"],
            "previous_turn_context": case.get("prior") or {},
        })["context_compilation"]
        expected = case["expected"]
        if "relation" in expected:
            assert result["relation"] == expected["relation"], case["id"]
        if "needs_clarification" in expected:
            assert result["needs_clarification"] == expected["needs_clarification"], case["id"]
        if "clarification_reason" in expected:
            assert result["clarification_reason"] == expected["clarification_reason"], case["id"]
        for slot_name in ("metric", "date", "dimension", "operation"):
            if slot_name in expected:
                assert result["known_slots"][slot_name]["value"] == expected[slot_name], case["id"]


def test_ordered_trend_then_contribution_is_preserved_as_two_required_steps():
    question = (
        "请调查平台订单量从2017-04-07到2017-04-08的变化：先核对趋势，"
        "再按product_category查看注册支持的统计贡献；若证据不足请明确停止，不作业务因果断言。"
    )
    compilation = compile_turn_context({"user_message": question})["context_compilation"]

    assert compilation["needs_clarification"] is False
    assert compilation["operation_intent"]["kind"] == "dimensional_statistical_contribution"
    assert compilation["operation_intent"]["required_evidence_sequence"] == [
        "trend", "dimension_contribution",
    ]
    assert compilation["request_contract"]["required_evidence_sequence"] == [
        "trend", "dimension_contribution",
    ]
    assert compilation["request_contract"]["metric_id"] == "order_count"
    assert compilation["request_contract"]["dimension"] == "product_category"
    assert compilation["request_contract"]["comparison"] == {
        "kind": "day_pair", "baseline_date": "2017-04-07", "current_date": "2017-04-08",
    }
    assert _route_from_triage({
        "task_type": "anomaly_analysis", "next_agent": "data_query",
        "context_compilation": compilation,
        "request_contract_hash": compilation["request_contract_hash"],
    }) == "premise_check"


def test_unordered_trend_and_contribution_still_requires_clarification():
    compilation = compile_turn_context({
        "user_message": "比较平台订单量2017-04-07和2017-04-08的趋势及product_category统计贡献。",
    })["context_compilation"]

    assert compilation["needs_clarification"] is True
    assert compilation["clarification_reason"] == "ambiguous_operation"
    assert compilation["request_contract"]["parse_status"] == "clarification_required"


@pytest.mark.parametrize(
    ("question", "expected_baseline", "expected_current"),
    [
        (
            "比较 2018 年 7 月和 8 月的 GMV",
            {"start_date": "2018-07-01", "end_date": "2018-07-31"},
            {"start_date": "2018-08-01", "end_date": "2018-08-31"},
        ),
        (
            "比较 2018 年的 7 月和 8 月的 GMV",
            {"start_date": "2018-07-01", "end_date": "2018-07-31"},
            {"start_date": "2018-08-01", "end_date": "2018-08-31"},
        ),
        (
            "比较 2017 年 12 月和 2018 年 1 月的 GMV",
            {"start_date": "2017-12-01", "end_date": "2017-12-31"},
            {"start_date": "2018-01-01", "end_date": "2018-01-31"},
        ),
    ],
)
def test_month_comparison_inherits_a_single_explicit_year(
    question, expected_baseline, expected_current,
):
    comparison, status, reason, _tokens = _comparison_resolution(question)

    assert status == "resolved"
    assert reason is None
    assert comparison == {
        "kind": "period_pair",
        "baseline": expected_baseline,
        "current": expected_current,
        "granularity": "month",
    }


def test_inherited_year_month_breakdown_is_contribution_not_a_followup():
    from app.context.request_contract import validate_compiled_request_contract

    compilation = compile_turn_context({
        "user_message": (
            "比较 2018 年 7 月与 8 月去重订单量，按商品品类拆分变化；"
            "这里要看订单数，不是商品件数。"
        ),
    })["context_compilation"]

    comparison = compilation["request_contract"]["comparison"]
    assert comparison == {
        "kind": "period_pair",
        "baseline": {"start_date": "2018-07-01", "end_date": "2018-07-31"},
        "current": {"start_date": "2018-08-01", "end_date": "2018-08-31"},
        "granularity": "month",
    }
    assert compilation["needs_clarification"] is False
    assert compilation["operation_intent"]["kind"] == "dimensional_statistical_contribution"
    assert compilation["operation_intent"]["dimension"] == "product_category"
    assert compilation["request_contract"]["metric_id"] == "order_count"
    assert validate_compiled_request_contract(compilation) == (True, None)


def test_negated_business_mechanism_keeps_period_breakdown_as_contribution():
    from app.context.request_contract import validate_compiled_request_contract

    compilation = compile_turn_context({
        "user_message": (
            "比较 2018 年 7 月与 8 月的客单价，并按月初、月中、月末展示各段数值变化。"
            "说明最大变化出现在哪里，不推断业务机制。"
        ),
    })["context_compilation"]

    assert compilation["needs_clarification"] is False
    assert compilation["operation_intent"]["kind"] == "dimensional_statistical_contribution"
    assert compilation["operation_intent"]["dimension"] == "month_position"
    assert compilation["request_contract"]["operation_intent"] == (
        "dimensional_statistical_contribution"
    )
    assert validate_compiled_request_contract(compilation) == (True, None)


def test_causal_marker_negation_is_clause_local():
    assert _has_unnegated_marker("不同品类导致 GMV 下降。", ("导致",)) is True
    assert _has_unnegated_marker("说明变化，不推断业务机制。", ("业务机制",)) is False
    assert _has_unnegated_marker("不要分析业务机制，只看数值。", ("业务机制",)) is False


def test_unresolved_sealed_contract_reports_parser_reason_not_period_drift():
    from app.context.request_contract import (
        seal_request_contract,
        validate_compiled_request_contract,
    )

    contract = seal_request_contract({
        "operation_intent": "business_causal_investigation",
        "metric_id": "gmv",
        "dimension": "product_category",
        "comparison": {"kind": "period_pair", "granularity": "month"},
        "parse_status": "clarification_required",
        "uncertainty_reason": "comparison_period_context_unavailable",
    })
    compilation = {
        "request_contract": contract,
        "request_contract_hash": contract["contract_hash"],
        "operation_intent": {"kind": "business_causal_investigation"},
        "known_slots": {
            "metric": {"value": "gmv"},
            "dimension": {"value": "product_category"},
        },
    }

    assert validate_compiled_request_contract(compilation) == (
        False, "comparison_period_context_unavailable",
    )


def test_registered_dimension_without_metric_capability_is_rejected_before_period_clarification():
    from app.context.compiler import context_clarification

    compilation = compile_turn_context({
        "user_message": (
            "将 2018 年 8 月 GMV 变化按支付方式拆解统计贡献。"
            "如果系统未注册这种归因维度，请先说明并停止查询。"
        ),
    })["context_compilation"]

    assert compilation["needs_clarification"] is True
    assert compilation["clarification_reason"] == "dimension_not_supported_by_metric"
    assert compilation["request_contract"]["parse_status"] == "clarification_required"
    assert compilation["request_contract"]["uncertainty_reason"] == (
        "dimension_not_supported_by_metric"
    )
    assert compilation["request_contract"]["dimension"] is None
    assert "dimension" in compilation["missing_slots"]

    clarified = context_clarification({"context_compilation": compilation})
    assert "未声明该维度的统计贡献能力" in clarified["answer"]

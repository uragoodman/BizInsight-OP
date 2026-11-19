"""WP2 Turn Resolver acceptance tests.

These tests exercise the actual compiler/resolver and task-service decisions.
They do not replace the Mem0 adapter: when no WP3 provider exists the resolver
must expose an explicit ``provider_unavailable`` need rather than inventing a
preference.
"""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import patch

from app.context.compiler import compile_turn_context
from app.context.orchestrator import ContextOrchestrator
from app.context.turn_resolver import (
    detect_preference_source_needs,
    evaluate_preference_candidates as _evaluate_preference_candidates,
    resolve_preference_context,
)
from app.core.config import get_settings
from app.tasks import service


_ALL_PREFERENCE_SOURCE_NEEDS = [
    {
        "source": "mem0", "rules_version": "preference-intent-rules-v4",
        "memory_types": ["presentation"],
        "allowed_keys": {"presentation": ["response_style", "response_language", "report_style"]},
        "allowed_fields": {"presentation": ["style", "language", "format"]},
    },
    {
        "source": "mem0", "rules_version": "preference-intent-rules-v4",
        "memory_types": ["analysis_method"],
        "allowed_keys": {"analysis_method": ["analysis_method", "analysis_steps"]},
        "allowed_fields": {"analysis_method": ["metric", "dimension", "operation", "date"]},
    },
    {
        "source": "mem0", "rules_version": "preference-intent-rules-v4",
        "memory_types": ["scope_default"],
        "allowed_keys": {"scope_default": ["default_period", "default_date"]},
        "allowed_fields": {"scope_default": ["start_date", "end_date", "date"]},
    },
]


def evaluate_preference_candidates(state, candidates, compilation=None, *, source_needs=None):
    """Use a broad, explicit v2 need in legacy unit fixtures unless overridden."""
    return _evaluate_preference_candidates(
        state,
        candidates,
        compilation,
        source_needs=(
            _ALL_PREFERENCE_SOURCE_NEEDS if source_needs is None else source_needs
        ),
    )


def _prior(*, metric: str = "gmv", date_value: str = "2018-08-01 至 2018-08-31") -> dict:
    return {
        "source_turn_id": "turn-prior",
        "authorized_metrics": [{"metric_id": metric}],
        "context_slots": {
            "metric": {"value": metric, "source": "current_turn"},
            "date": {"value": date_value, "source": "current_turn"},
        },
        "scope_fingerprint": "scope-a",
    }


def _compile(message: str, **overrides) -> tuple[dict, dict, dict]:
    state = {
        "user_message": message,
        "previous_turn_context": _prior(),
        "current_query_scope_fingerprint": "scope-a",
        "node_history": [],
        **overrides,
    }
    compiled = compile_turn_context(state)
    return compiled["context_compilation"], compiled["turn_resolution"], compiled["preference_context"]


def test_inherited_fields_and_explicit_correction_keep_provenance() -> None:
    compilation, resolution, _ = _compile("换成 7 月，按品类拆开")

    assert compilation["known_slots"]["metric"]["source"] == "previous_turn_context"
    assert compilation["known_slots"]["metric"]["source_turn_id"] == "turn-prior"
    assert compilation["known_slots"]["date"]["source"] == "current_turn_correction"
    assert compilation["known_slots"]["dimension"]["value"] == "product_category"
    assert compilation["user_changes"] == [{
        "field": "date",
        "from": "2018-08-01 至 2018-08-31",
        "to": "2018-07-01 至 2018-07-31",
        "source": "current_turn_correction",
        "reason": "explicit_user_correction",
    }]
    assert resolution["explicit_fields"] == ["date", "dimension", "operation"]
    inherited = {item["field"]: item for item in resolution["inherited_fields"]}
    assert inherited["metric"]["source"] == "previous_turn_context"
    assert inherited["metric"]["source_turn_id"] == "turn-prior"
    assert "date" not in inherited
    assert resolution["user_changes"][0]["field"] == "date"


def test_east_china_reference_inherits_then_rejects_unsupported_scope_filter() -> None:
    compilation, resolution, preference = _compile("那华东呢")

    assert compilation["relation"] == "continue"
    assert compilation["known_slots"]["metric"]["value"] == "gmv"
    assert compilation["known_slots"]["date"]["value"] == "2018-08-01 至 2018-08-31"
    assert "region" not in compilation["known_slots"]
    assert compilation["needs_clarification"] is True
    assert compilation["clarification_reason"] == "region_filter_not_supported"
    assert compilation["scope_candidates"][0]["field"] == "region"
    assert compilation["scope_candidates"][0]["status"] == "unsupported"
    assert resolution["relation"] == "clarify"
    assert resolution["confidence"] <= 0.4
    assert preference["status"] == "not_requested"
    assert resolution["source_needs"] == []

    state = {
        "user_message": "那华东呢",
        "context_compilation": compilation,
        "turn_resolution": resolution,
        "preference_context": preference,
        "current_query_scope_fingerprint": "scope-a",
    }
    triage = ContextOrchestrator().build_triage_context(
        state, instruction="那华东呢", model="test-model",
    )
    assert triage.preference_source_needs == []
    assert state["context_source_availability"]["triage"] == [
        {"source": "conversation", "reason": "empty_or_unavailable"},
        {"source": "task", "reason": "empty_or_unavailable"},
    ] or any(
        row["source"] == "mem0" and row["reason"] == "not_requested"
        for row in state["context_source_availability"]["triage"]
    )


def test_personalisation_request_emits_source_need_without_fake_preference() -> None:
    compilation, resolution, preference = _compile(
        "按我平时的方式回答",
        preference_context={},
        turn_relation_decision={},
    )

    assert compilation["known_slots"]["metric"]["value"] == "gmv"
    assert preference["status"] == "provider_unavailable"
    assert preference["availability"] == "provider_unavailable"
    assert preference["accepted"] == []
    assert preference["items"] == []
    assert preference["source_needs"][0]["trigger_ids"] == ["P1"]
    assert preference["source_needs"][0]["status"] == "unavailable"
    assert preference["retrieval"]["threshold"] == 0.55
    assert preference["retrieval"]["top_k_total"] == 10
    assert preference["retrieval"]["hit_count"] == 0
    assert preference["retrieval"]["provider_calls"] == 0
    assert resolution["source_needs"][0]["memory_types"] == ["presentation"]
    assert resolution["source_needs"][0]["allowed_keys"] == {
        "presentation": ["response_language", "response_style"],
    }
    assert resolution["source_needs"][0]["rules_version"] == "preference-intent-rules-v4"
    assert resolution["preference"]["status"] == "provider_unavailable"
    assert resolution["preference"]["retrieval"]["threshold"] == 0.55
    assert resolution["preference"]["retrieval"]["hit_count"] == 0

    state = {
        "user_message": "按我平时的方式回答",
        "context_compilation": compilation,
        "turn_resolution": resolution,
        "preference_context": preference,
        "current_query_scope_fingerprint": "scope-a",
    }
    triage = ContextOrchestrator().build_triage_context(
        state, instruction="按我平时的方式回答", model="test-model",
    )
    assert triage.preference_source_needs[0]["trigger_ids"] == ["P1"]
    assert triage.preference_candidates == []
    assert any(
        row["source"] == "mem0" and row["reason"] == "provider_unavailable"
        for row in state["context_source_availability"]["triage"]
    )


def test_preference_management_alone_does_not_request_mem0_search() -> None:
    assert detect_preference_source_needs({
        "user_message": "我有哪些偏好？",
        "chat_session_id": "session-1",
    }) == []


def test_preference_need_trace_records_rules_without_preference_values(monkeypatch) -> None:
    from app.context import turn_resolver

    events = []
    monkeypatch.setattr(turn_resolver, "record_event", lambda name, attrs: events.append((name, attrs)))
    turn_resolver.resolve_preference_context({
        "user_message": "按我平时的分析习惯分析 GMV",
        "chat_session_id": "session-1",
        "current_query_scope_fingerprint": "scope-1",
    }, provider=None)

    name, attrs = next(event for event in events if event[0] == "context.preference_source_decision")
    assert attrs["preference.rules_versions"] == "preference-intent-rules-v4"
    assert attrs["preference.intent_ids"] == "analysis_method"
    assert attrs["preference.allowed_keys"] == "analysis_method:analysis_method;analysis_method:analysis_steps"
    assert attrs["preference.allowed_fields"] == "analysis_method:dimension;analysis_method:operation"
    assert attrs["preference.overridden_fields"] == ""
    assert "偏好" not in repr(attrs)


def test_preference_intent_needs_are_multi_label_and_field_scoped() -> None:
    # Table-driven source-need contract tests. Current-turn fields are included
    # explicitly so they must be represented as overridden, not recalled.
    cases = [
        (
            "按我平时的方式回答",
            {},
            ["presentation"],
            {"presentation": ["response_language", "response_style"]},
            {"presentation": ["language", "style"]},
            [],
            ["P1"],
            ["presentation"],
        ),
        (
            "请生成一份报告",
            {},
            ["presentation"],
            {"presentation": ["report_style"]},
            {"presentation": ["format"]},
            [],
            ["P2"],
            ["presentation"],
        ),
        (
            "按我平时的方式分析",
            {},
            ["analysis_method"],
            {"analysis_method": ["analysis_method", "analysis_steps"]},
            {"analysis_method": ["dimension", "operation"]},
            [],
            ["P1"],
            ["analysis_method"],
        ),
        (
            "按我平时的方式简短回答，并按商品品类拆解",
            {"dimension": {"value": "product_category", "source": "current_turn"}},
            ["presentation"],
            {"presentation": ["response_language"]},
            {"presentation": ["language"]},
            ["analysis_method.dimension", "presentation.style"],
            ["P1"],
            ["analysis_method", "presentation"],
        ),
        (
            "按我平时的分析习惯分析，并使用我的默认统计周期",
            {},
            ["analysis_method", "scope_default"],
            {
                "analysis_method": ["analysis_method", "analysis_steps"],
                "scope_default": ["default_period"],
            },
            {
                "analysis_method": ["dimension", "operation"],
                "scope_default": ["end_date", "start_date"],
            },
            [],
            ["P1"],
            ["analysis_method", "scope_default"],
        ),
        (
            "报告按我平时格式，回答用英文",
            {},
            ["presentation"],
            {"presentation": ["report_style"]},
            {"presentation": ["format"]},
            ["presentation.language"],
            ["P1", "P2"],
            ["presentation"],
        ),
        (
            "按惯例处理",
            {},
            [],
            {},
            {},
            [],
            [],
            [],
        ),
        (
            "按华东的默认范围分析",
            {},
            [],
            {},
            {},
            [],
            [],
            [],
        ),
        (
            "按我平时的方式回答，并告诉我有哪些偏好",
            {},
            ["presentation"],
            {"presentation": ["response_language", "response_style"]},
            {"presentation": ["language", "style"]},
            [],
            ["P1"],
            ["presentation"],
        ),
    ]
    for message, slots, memory_types, allowed_keys, allowed_fields, overridden, triggers, intents in cases:
        state = {
            "user_message": message,
            "chat_session_id": "session-1",
            "current_query_scope_fingerprint": "scope-1",
            "context_compilation": {"known_slots": slots},
        }
        needs = detect_preference_source_needs(state)
        if not memory_types:
            assert needs == [], message
            continue
        assert len(needs) == 1, message
        need = needs[0]
        assert need["memory_types"] == memory_types, message
        assert need["allowed_keys"] == allowed_keys, message
        assert need["allowed_fields"] == allowed_fields, message
        assert need["overridden_fields"] == overridden, message
        assert need["trigger_ids"] == triggers, message
        assert need["rules_version"] == "preference-intent-rules-v4", message
        assert need["intent_ids"] == intents, message
        assert "allowed_values" in need and isinstance(need["allowed_values"], dict), message
        assert "blocked_fields" in need and isinstance(need["blocked_fields"], dict), message
        if "有哪些偏好" in message:
            assert need["other_intents"] == ["preference_management"], message


def test_preference_intent_synonyms_and_negative_trigger_boundaries() -> None:
    cases = [
        ("保持我常用的回答方式", ["presentation"]),
        ("按我平常的回答方式回复", ["presentation"]),
        ("按我平时的回答方式，GMV 是多少", ["presentation"]),
        ("按我平常的回答方式，update GMV", ["presentation"]),
        ("按我平时比较喜欢的回答方式回答", ["presentation"]),
        ("按我平时比较容易理解的回答方式回答", ["presentation"]),
        ("按我平时比较关注的方式回答", ["presentation"]),
        ("GMV 这个指标是多少", []),
        ("今天的日期是多少", []),
        ("update", []),
        ("按我平时的回答方式，按我平时比较喜欢的方式分析 GMV", ["presentation", "analysis_method"]),
        ("按我平时的回答方式，按我平时比较容易理解的方式分析 GMV", ["presentation", "analysis_method"]),
        ("按我平时用的 last month", []),
    ]
    for message, expected_types in cases:
        needs = detect_preference_source_needs({
            "user_message": message,
            "chat_session_id": "session-1",
            "context_compilation": {
                "known_slots": {
                    "metric": {"value": "gmv", "source": "current_turn"},
                },
            },
        })
        actual_types = needs[0]["memory_types"] if needs else []
        assert actual_types == sorted(expected_types), message


def test_preference_intent_metric_and_date_facets_are_explicitly_scoped() -> None:
    cases = [
        ("按我平时的默认指标是什么", ["analysis_method"], {"analysis_method": ["metric"]}),
        ("按我平时常看的指标", ["analysis_method"], {"analysis_method": ["metric"]}),
        ("按我平时的指标偏好", ["analysis_method"], {"analysis_method": ["metric"]}),
        ("按我平时默认的分析指标", ["analysis_method"], {"analysis_method": ["metric"]}),
        ("按我平时的分析日期习惯", ["analysis_method"], {"analysis_method": ["date"]}),
        ("按我平时分析时常用的日期", ["analysis_method"], {"analysis_method": ["date"]}),
        ("按我平时分析常用的日期", ["analysis_method"], {"analysis_method": ["date"]}),
        ("按我平时常用的日期", ["scope_default"], {"scope_default": ["date"]}),
        ("按我平时用的 default date", ["scope_default"], {"scope_default": ["date"]}),
    ]
    for message, expected_types, expected_fields in cases:
        needs = detect_preference_source_needs({
            "user_message": message,
            "chat_session_id": "session-1",
            "context_compilation": {"known_slots": {}},
        })
        assert len(needs) == 1, message
        need = needs[0]
        assert need["memory_types"] == expected_types, message
        assert need["allowed_fields"] == expected_fields, message
        if "metric" in repr(expected_fields):
            assert need["allowed_keys"]["analysis_method"] == ["analysis_method", "analysis_steps"]


def test_negative_preference_intent_and_lookup_are_separate_from_apply() -> None:
    cases = [
        ("不要按我平时的方式回答", []),
        ("别按我平时的分析习惯分析", []),
        ("不要按我平时的默认指标", []),
        ("don't use my usual answer style", []),
        ("我不想按我平时的方式回答", []),
    ]
    for message, expected in cases:
        assert detect_preference_source_needs({
            "user_message": message,
            "chat_session_id": "session-1",
            "context_compilation": {"known_slots": {}},
        }) == expected, message

    partial = detect_preference_source_needs({
        "user_message": "不要简洁，按我平时的回答方式回答",
        "chat_session_id": "session-1",
        "context_compilation": {"known_slots": {}},
    })
    assert partial[0]["allowed_fields"] == {"presentation": ["language", "style"]}
    assert partial[0]["excluded_values"] == {"presentation.style": ["concise"]}
    assert partial[0]["overridden_fields"] == []

    lookup_state = {
        "user_message": "我的默认指标是什么",
        "chat_session_id": "session-1",
        "context_compilation": {"known_slots": {"metric": {"value": "gmv", "source": "task_state"}}},
    }
    assert detect_preference_source_needs(lookup_state) == []
    resolved = resolve_preference_context(lookup_state, provider=None)
    assert resolved["status"] == "registry_lookup_required"
    assert resolved["lookup_request"] == {
        "source": "preference_registry", "mode": "lookup",
        "memory_type": "analysis_method", "memory_keys": ["analysis_method", "analysis_steps"],
        "fields": ["metric"],
    }
    assert resolved["retrieval"]["provider_calls"] == 0


def test_negative_constraints_v2_cover_mixed_categories_and_fail_closed() -> None:
    cases = [
        ("不要分析，按我平时的方式回答，GMV 是多少", ["presentation"], ["analysis_method"]),
        ("不需要按我平时的方式回答", [], ["presentation"]),
        ("不必按我平时的方式回答", [], ["presentation"]),
        ("无需按我平时的方式回答", [], ["presentation"]),
        ("never use my usual answer style", [], ["presentation"]),
        ("avoid my usual answer style", [], ["presentation"]),
        ("not use my usual answer style", [], ["presentation"]),
        ("no longer use my usual answer style", [], ["presentation"]),
        ("without using my usual answer style", [], ["presentation"]),
        ("avoidance my usual answer style", ["presentation"], []),
        ("notable my usual answer style", ["presentation"], []),
        ("如果我说‘不要分析’，你会怎么做？", [], []),
        ("不要不按我平时的方式回答", [], []),
        ("don't not use my usual answer style", [], []),
    ]
    for message, expected_types, expected_blocked in cases:
        state = {
            "user_message": message,
            "chat_session_id": "session-negative-v2",
            "context_compilation": {"known_slots": {}},
        }
        needs = detect_preference_source_needs(state)
        actual_types = needs[0]["memory_types"] if needs else []
        assert actual_types == expected_types, message
        if needs:
            assert needs[0]["rules_version"] == "preference-intent-rules-v4"
            assert needs[0]["blocked_categories"] == {
                item: "explicit_negative_instruction" for item in expected_blocked
            }, message
        else:
            resolved = resolve_preference_context(state, provider=None)
            assert sorted(resolved["intent_decision"]["blocked_categories"]) == sorted(expected_blocked), message
            assert resolved["retrieval"]["provider_calls"] == 0


def test_shared_negative_parser_handles_conjunctions_and_separates_query_params() -> None:
    cases = [
        ("不要分析并按我平时的方式回答", ["presentation"], ["analysis_method"], {}),
        ("不要用表格并按我平时的格式生成报告", ["presentation"], [], {"presentation.format": ["table"]}),
        ("不要用表格而报告按我平时的格式", ["presentation"], [], {"presentation.format": ["table"]}),
        ("不要用表格并报告按我平时的格式", ["presentation"], [], {"presentation.format": ["table"]}),
        ("不要用表格而使用我平时的格式", ["presentation"], [], {"presentation.format": ["table"]}),
        ("不要分析而采用我平时的方式回答", ["presentation"], ["analysis_method"], {}),
        ("不要分析而采用我平时的分析方式", ["analysis_method"], [], {}),
        ("不要分析而选择我的分析习惯", ["analysis_method"], [], {}),
        ("不要用表格而换成我平时的格式", ["presentation"], [], {"presentation.format": ["table"]}),
        ("不要分析而生成为我平时格式的报告", ["presentation"], ["analysis_method"], {}),
        ("不要分析而汇报我平时格式的报告", ["presentation"], ["analysis_method"], {}),
        ("不要用表格而汇报我平时的格式", ["presentation"], [], {"presentation.format": ["table"]}),
        ("不要简洁并保持我常用的回答方式", ["presentation"], [], {"presentation.style": ["concise"]}),
    ]
    for message, expected_types, expected_blocked, expected_excluded in cases:
        state = {
            "user_message": message,
            "chat_session_id": "session-negative-conjunction",
            "context_compilation": {"known_slots": {}},
        }
        needs = detect_preference_source_needs(state)
        assert [kind for need in needs for kind in need["memory_types"]] == expected_types, message
        if needs:
            assert needs[0]["blocked_categories"] == {
                kind: "explicit_negative_instruction" for kind in expected_blocked
            }, message
            assert needs[0]["excluded_values"] == expected_excluded, message

    assert detect_preference_source_needs({
        "user_message": "不要分析并且不要总结",
        "chat_session_id": "session-negative-repeat",
        "context_compilation": {"known_slots": {}},
    }) == []

    for message in (
        "不要分析且按我平时的方式回答",
        "不要分析而按我平时的方式回答",
    ):
        need = detect_preference_source_needs({
            "user_message": message,
            "chat_session_id": "session-bare-conjunction",
            "context_compilation": {"known_slots": {}},
        })
        assert len(need) == 1, message
        assert need[0]["memory_types"] == ["presentation"], message
        assert need[0]["blocked_categories"] == {
            "analysis_method": "explicit_negative_instruction",
        }, message


def test_double_negation_variants_fail_closed_in_compiler_and_preference_resolver() -> None:
    for message in (
        "不是不按我平时的方式回答",
        "并非不按我平时的方式回答",
    ):
        compiled = compile_turn_context({"user_message": message, "node_history": []})["context_compilation"]
        assert compiled["needs_clarification"] is True, message
        assert compiled["clarification_reason"] == "unsupported_negative_instruction", message
        assert detect_preference_source_needs({
            "user_message": message,
            "chat_session_id": "session-double-negative-v4",
            "context_compilation": compiled,
        }) == [], message


def test_negative_query_parameters_are_separate_from_preference_exclusions(monkeypatch) -> None:
    monkeypatch.setattr("app.context.compiler._metric_labels", lambda: {"gmv": "GMV"})
    negated_compare = compile_turn_context({
        "user_message": "不要同比",
        "node_history": [],
    })
    assert "operation" not in negated_compare["context_compilation"]["known_slots"]
    assert negated_compare["context_compilation"]["negative_query_constraints"]["params"] == {
        "operation": ["compare"],
    }
    english_substring = compile_turn_context({
        "user_message": "avoidance comparative analysis, GMV",
        "node_history": [],
    })["context_compilation"]
    assert english_substring["negative_query_constraints"]["params"] == {}
    assert "operation" not in english_substring["known_slots"]

    negated_date = compile_turn_context({
        "user_message": "不要最近一个月",
        "node_history": [],
    })
    assert "date" not in negated_date["context_compilation"]["known_slots"]
    assert negated_date["context_compilation"]["negative_query_constraints"]["params"] == {
        "date": ["last_30_days"],
    }
    negated_calendar_month = compile_turn_context({
        "user_message": "不要上个月",
        "node_history": [],
    })["context_compilation"]
    assert negated_calendar_month["negative_query_constraints"]["params"] == {
        "date": ["last_month"],
    }

    negated_dimension = compile_turn_context({
        "user_message": "不要按品类拆解 GMV",
        "node_history": [],
    })
    slots = negated_dimension["context_compilation"]["known_slots"]
    assert slots.get("metric", {}).get("value") == "gmv"
    assert "dimension" not in slots
    assert "operation" not in slots
    assert negated_dimension["context_compilation"]["negative_query_constraints"]["params"]["dimension"] == [
        "product_category",
    ]

    unsupported_double_negative = compile_turn_context({
        "user_message": "不要不按品类拆解 GMV",
        "node_history": [],
    })["context_compilation"]
    assert unsupported_double_negative["needs_clarification"] is True
    assert unsupported_double_negative["clarification_reason"] == "unsupported_negative_instruction"
    assert "dimension" not in unsupported_double_negative["known_slots"]

    # Negating the use of historical analysis preferences does not negate an
    # explicitly requested comparison later in that same clause.
    explicit_compare = compile_turn_context({
        "user_message": "不要按我平时的分析习惯做同比，GMV是多少",
        "node_history": [],
    })
    assert explicit_compare["context_compilation"]["known_slots"]["operation"]["value"] == "compare"
    assert explicit_compare["context_compilation"]["negative_query_constraints"]["params"] == {}
    assert detect_preference_source_needs({
        "user_message": "不要按我平时的分析习惯做同比，GMV是多少",
        "chat_session_id": "session-negative-preference-vs-query",
        "context_compilation": {"known_slots": {}},
    }) == []
    mixed = detect_preference_source_needs({
        "user_message": "不要按我平时的分析习惯做同比，但是按我平时的分析方式按品类拆解 GMV",
        "chat_session_id": "session-negative-explicit-operation",
        "context_compilation": {"known_slots": {}},
    })
    assert len(mixed) == 1
    assert mixed[0]["memory_types"] == ["analysis_method"]
    assert mixed[0]["negative_query_params"] == {}
    assert mixed[0]["excluded_values"] == {}
    assert "analysis_method.operation" in mixed[0]["overridden_fields"]

    preference_after_negative_operation = detect_preference_source_needs({
        "user_message": "不要同比，按我平时的分析习惯分析 GMV",
        "chat_session_id": "session-negative-operation-with-preference",
        "context_compilation": {"known_slots": {}},
    })
    assert len(preference_after_negative_operation) == 1
    need = preference_after_negative_operation[0]
    assert need["negative_query_params"] == {"operation": ["compare"]}
    assert "analysis_method.operation" not in need["blocked_fields"]
    assert "operation" in need["allowed_fields"]["analysis_method"]
    assert need["excluded_values"]["analysis_method.operation"] == ["compare"]
    decision = resolve_preference_context({
        "user_message": "不要同比",
        "chat_session_id": "session-negative-query-trace",
    }, provider=None)["intent_decision"]
    assert decision["negative_query_param_fields"] == ["operation"]
    assert decision["negative_query_param_counts"] == {"operation": 1}


def test_negative_presentation_values_are_excluded_not_whole_fields() -> None:
    report_state = {
        "user_message": "不要用表格，报告按我平时的格式",
        "chat_session_id": "session-negative-format",
        "context_compilation": {"known_slots": {}},
    }
    report_need = detect_preference_source_needs(report_state)[0]
    assert report_need["allowed_keys"] == {"presentation": ["report_style"]}
    assert report_need["allowed_fields"] == {"presentation": ["format"]}
    assert report_need["excluded_values"] == {"presentation.format": ["table"]}

    candidates = [
        {"memory_key": "report_style", "memory_type": "presentation", "value": {"format": "table"}, "confidence": 0.95},
        {"memory_key": "report_style", "memory_type": "presentation", "value": {"format": "bullets"}, "confidence": 0.95},
    ]
    evaluated = _evaluate_preference_candidates(
        report_state, candidates, {"known_slots": {}}, source_needs=[report_need],
    )
    assert len(evaluated["accepted"]) == 1
    assert evaluated["items"][0]["effective_fields"] == {"format": "bullets"}
    assert next(item for item in evaluated["rejected"] if item["memory_key"] == "report_style")["reason"] == "source_need_value_excluded"

    style_state = {
        "user_message": "不要简洁，按我平时的回答方式回答",
        "chat_session_id": "session-negative-style",
        "context_compilation": {"known_slots": {}},
    }
    style_need = detect_preference_source_needs(style_state)[0]
    assert style_need["allowed_fields"]["presentation"] == ["language", "style"]
    assert style_need["excluded_values"] == {"presentation.style": ["concise"]}
    style_candidates = [
        {"memory_key": "response_style", "memory_type": "presentation", "value": {"style": "concise"}, "confidence": 0.95},
        {"memory_key": "response_style", "memory_type": "presentation", "value": {"style": "detailed"}, "confidence": 0.95},
    ]
    style_result = _evaluate_preference_candidates(
        style_state, style_candidates, {"known_slots": {}}, source_needs=[style_need],
    )
    assert len(style_result["accepted"]) == 1
    assert style_result["items"][0]["effective_fields"] == {"style": "detailed"}
    assert any(item["reason"] == "source_need_value_excluded" for item in style_result["rejected"])


def test_negative_preference_does_not_apply_analysis_dimension_or_remove_explicit_operation() -> None:
    message = "不要分析，按我平时的方式回答，GMV 是多少"
    state = {
        "user_message": message,
        "chat_session_id": "session-no-analysis-preference",
        "current_query_scope_fingerprint": "scope-a",
        "controlled_analysis_scope": {
            "allowed_metrics": ["gmv"], "allowed_dimensions": ["product_category"],
        },
        "context_compilation": {"known_slots": {}},
    }
    needs = detect_preference_source_needs(state)
    assert [need["memory_types"] for need in needs] == [["presentation"]]
    hostile_candidates = [{
        "memory_key": "analysis_method", "memory_type": "analysis_method",
        "value": {"dimension": "product_category"}, "confidence": 0.99,
    }]
    evaluated = _evaluate_preference_candidates(
        state, hostile_candidates, {"known_slots": {}}, source_needs=needs,
    )
    assert evaluated["accepted"] == []
    assert evaluated["rejected"][0]["reason"] == "source_need_type_not_allowed"

    compiled = compile_turn_context({
        "user_message": message,
        "previous_turn_context": {},
        "current_query_scope_fingerprint": "scope-a",
        "node_history": [],
        "preference_context": {**evaluated, "source_needs": needs},
    })
    contract = compiled["context_compilation"]["request_contract"]
    assert contract.get("dimension") is None
    assert contract.get("operation_intent") == "ordinary_metric_query"
    assert "dimension" not in compiled["context_compilation"].get("known_slots", {})

    explicit_comparison = compile_turn_context({
        "user_message": "不要按我平时的分析习惯做同比，GMV是多少",
        "previous_turn_context": {},
        "current_query_scope_fingerprint": "scope-a",
        "node_history": [],
    })
    assert explicit_comparison["context_compilation"]["known_slots"]["operation"]["value"] == "compare"
    assert detect_preference_source_needs({
        "user_message": "不要按我平时的分析习惯做同比，GMV是多少",
        "chat_session_id": "session-explicit-compare",
        "context_compilation": {"known_slots": {}},
    }) == []


def test_negative_dimension_and_metric_exclude_only_matching_preference_values(monkeypatch) -> None:
    monkeypatch.setattr("app.context.turn_resolver._metric_allowed", lambda metric_id, state: (True, None))
    dimension_state = {
        "user_message": "不要按品类拆解，按我平时的分析方式分析 GMV",
        "chat_session_id": "session-negative-dimension-preference",
        "current_query_scope_fingerprint": "scope-a",
        "controlled_analysis_scope": {
            "allowed_metrics": ["gmv"],
            "allowed_dimensions": ["product_category", "seller", "payment_type"],
        },
        "context_compilation": {"known_slots": {}},
    }
    dimension_need = detect_preference_source_needs(dimension_state)[0]
    assert dimension_need["allowed_fields"]["analysis_method"] == ["dimension", "operation"]
    assert dimension_need["excluded_values"]["analysis_method.dimension"] == ["product_category"]
    assert "analysis_method.dimension" not in dimension_need["blocked_fields"]
    decision = resolve_preference_context(dimension_state, provider=None)["intent_decision"]
    assert decision["excluded_value_fields"] == [
        "analysis_method.dimension", "analysis_method.operation",
    ]
    assert decision["excluded_value_counts"] == {
        "analysis_method.dimension": 1,
        "analysis_method.operation": 1,
    }
    assert "product_category" not in repr(decision)
    dimension_eval = _evaluate_preference_candidates(
        dimension_state,
        [{
            "memory_key": "analysis_method", "memory_type": "analysis_method",
            "value": {"dimension": "商品品类", "operation": "trend"}, "confidence": 0.95,
        }],
        {"known_slots": {}}, source_needs=[dimension_need],
    )
    assert dimension_eval["accepted"][0]["status"] == "partially_accepted"
    assert dimension_eval["effects"][0]["fields"] == {"operation": "trend"}
    assert dimension_eval["accepted"][0]["field_rejections"] == [
        {"field": "dimension", "reason": "source_need_value_excluded"},
    ]
    alternative_dimension = _evaluate_preference_candidates(
        dimension_state,
        [{
            "memory_key": "analysis_method", "memory_type": "analysis_method",
            "source_ref": "alternative-dimension",
            "value": {"dimension": "payment_type"}, "confidence": 0.95,
        }],
        {"known_slots": {}}, source_needs=[dimension_need],
    )
    assert alternative_dimension["effects"][0]["fields"] == {"dimension": "payment_type"}
    from app.context.turn_resolver import apply_candidate_analysis_plan

    monkeypatch.setattr("app.context.turn_resolver._dimension_allowed", lambda dimension, metric_ids, state: (True, None))
    compilation = {
        "needs_clarification": False,
        "known_slots": {},
        "request_contract": {"metric_id": "gmv", "dimension": None},
    }
    applied = apply_candidate_analysis_plan(
        dimension_state,
        compilation,
        {**alternative_dimension, "source_needs": [dimension_need]},
    )
    assert applied["candidate_analysis_plan"]["applied"] is True
    assert compilation["request_contract"]["dimension"] == "payment_type"

    monkeypatch.setattr("app.context.compiler._metric_labels", lambda: {"gmv": "GMV", "orders": "订单数"})
    metric_message = "不要 GMV，按我平时的默认指标分析"
    metric_compilation = compile_turn_context({
        "user_message": metric_message,
        "node_history": [],
    })["context_compilation"]
    metric_state = {
        "user_message": metric_message,
        "chat_session_id": "session-negative-metric-preference",
        "current_query_scope_fingerprint": "scope-a",
        "controlled_analysis_scope": {"allowed_metrics": ["gmv", "orders"]},
        "context_compilation": metric_compilation,
    }
    metric_need = detect_preference_source_needs(metric_state)[0]
    assert metric_need["allowed_fields"]["analysis_method"] == ["metric"]
    assert metric_need["excluded_values"]["analysis_method.metric"] == ["gmv"]
    metric_eval = _evaluate_preference_candidates(
        metric_state,
        [{
            "memory_key": "analysis_method", "memory_type": "analysis_method",
            "value": {"metric": ["gmv", "orders"]}, "confidence": 0.95,
        }],
        {"known_slots": {}}, source_needs=[metric_need],
    )
    assert metric_eval["accepted"][0]["status"] == "accepted"
    assert metric_eval["effects"][0]["fields"] == {"metric": ["orders"]}


def test_negative_relative_date_excludes_matching_scope_period_but_allows_other_period() -> None:
    state = {
        "user_message": "不要最近一个月，按我平时常用的时间范围",
        "chat_session_id": "session-negative-scope-period",
        "scope_reference_date": "2026-11-01",
        "controlled_analysis_scope": {
            "default_start_date": "2026-01-01", "default_end_date": "2026-12-31",
        },
        "context_compilation": {"known_slots": {}},
    }
    need = detect_preference_source_needs(state)[0]
    assert need["memory_types"] == ["scope_default"]
    assert need["allowed_fields"]["scope_default"] == ["end_date", "start_date"]
    assert need["excluded_values"]["scope_default.period"] == ["last_30_days"]
    candidates = [
        {
            "memory_key": "default_period", "memory_type": "scope_default",
            "source_ref": "last-month-period",
            "value": {"start_date": "2026-10-01", "end_date": "2026-10-31"}, "confidence": 0.95,
        },
        {
            "memory_key": "default_period", "memory_type": "scope_default",
            "source_ref": "other-period",
            "value": {"start_date": "2026-08-01", "end_date": "2026-08-31"}, "confidence": 0.95,
        },
    ]
    evaluated = _evaluate_preference_candidates(
        state, candidates, {"known_slots": {}}, source_needs=[need],
    )
    assert len(evaluated["accepted"]) == 1
    assert evaluated["accepted"][0]["source_ref"] == "other-period"
    assert evaluated["items"][0]["effective_fields"] == {"date": "2026-08-31"}
    assert any(
        item["reason"] == "source_need_value_excluded"
        for item in evaluated["rejected"]
    )


def test_conflicting_value_instruction_suppresses_lookup_and_traces_only_safe_metadata(monkeypatch) -> None:
    from app.context import turn_resolver

    state = {
        "user_message": "不要用表格，报告用表格",
        "chat_session_id": "session-conflicting-format",
    }
    assert detect_preference_source_needs(state) == []
    events = []
    monkeypatch.setattr(turn_resolver, "record_event", lambda name, attrs: events.append((name, attrs)))
    result = resolve_preference_context(state, provider=None)
    assert result["intent_decision"]["conflicts"] == ["presentation.format"]
    attrs = next(attrs for name, attrs in events if name == "context.preference_source_decision")
    assert attrs["preference.conflict_fields"] == "presentation.format"
    assert "table" not in repr(attrs)

    allowed_state = {
        "user_message": "不要用表格，报告按我平时的格式",
        "chat_session_id": "session-safe-exclusion-trace",
    }
    need_context = resolve_preference_context(allowed_state, provider=None)
    from app.context.turn_resolver import _preference_metadata

    metadata = _preference_metadata(need_context)
    assert metadata["source_needs"][0]["excluded_value_fields"] == ["presentation.format"]
    assert metadata["source_needs"][0]["excluded_value_counts"] == {"presentation.format": 1}
    assert "table" not in repr(metadata)


def test_default_lookup_apply_phrases_and_explicit_iso_date_override() -> None:
    cases = [
        ("用我的默认统计周期", "scope_default", ["end_date", "start_date"]),
        ("用我的默认日期", "scope_default", ["date"]),
        ("按我平时默认指标分析", "analysis_method", ["metric"]),
    ]
    for message, memory_type, fields in cases:
        needs = detect_preference_source_needs({
            "user_message": message,
            "chat_session_id": "session-1",
            "context_compilation": {"known_slots": {}},
        })
        assert len(needs) == 1, message
        assert needs[0]["memory_types"] == [memory_type], message
        assert needs[0]["allowed_fields"][memory_type] == fields, message
        assert needs[0]["rules_version"] == "preference-intent-rules-v4"
        assert "overridden_fields" in needs[0] and "blocked_fields" in needs[0]

    explicit = detect_preference_source_needs({
        "user_message": "按我平时常用的时间范围，2026-10-01 至 2026-10-31",
        "chat_session_id": "session-1",
        "context_compilation": {"known_slots": {}},
    })
    assert explicit == []


def test_preference_intent_ignores_adjectival_compare_and_explicit_period() -> None:
    for message in (
        "按我平时比较喜欢的方式分析 GMV",
        "按我平时比较容易理解的方式分析 GMV",
    ):
        needs = detect_preference_source_needs({
            "user_message": message,
            "chat_session_id": "session-1",
            "context_compilation": {"known_slots": {}},
        })
        assert needs[0]["memory_types"] == ["analysis_method"], message
        assert "analysis_method.operation" not in needs[0]["overridden_fields"], message
        assert "dimension" in needs[0]["allowed_fields"]["analysis_method"], message

    assert detect_preference_source_needs({
        "user_message": "按我平时用的 default period last month",
        "chat_session_id": "session-1",
        "context_compilation": {"known_slots": {}},
    }) == []


def test_comparative_adjective_in_answer_preference_does_not_request_analysis_memory() -> None:
    for message in (
        "按我平时比较喜欢的回答方式回答",
        "按我平时比较容易理解的回答方式回答",
        "按我平时比较关注的方式回答",
    ):
        needs = detect_preference_source_needs({
            "user_message": message,
            "chat_session_id": "session-1",
            "context_compilation": {"known_slots": {}},
        })
        assert len(needs) == 1, message
        assert needs[0]["memory_types"] == ["presentation"], message


def test_compiler_does_not_turn_comparative_adjective_into_compare_operation() -> None:
    for message in (
        "按我平时比较喜欢的回答方式回答",
        "按我平时比较容易理解的回答方式回答",
        "按我平时比较注重的方式回答",
        "按我平时比较看重的方式回答",
    ):
        state = compile_turn_context({
            "user_message": message,
            "chat_session_id": "session-1",
            "node_history": [],
        })
        compilation = state["context_compilation"]
        assert "operation" not in compilation["known_slots"], message
        assert compilation["operation_intent"]["kind"] == "ordinary_metric_query", message

    for message in ("比较一下 GMV 和订单", "对比两期 GMV"):
        state = compile_turn_context({"user_message": message, "node_history": []})
        assert state["context_compilation"]["known_slots"]["operation"]["value"] == "compare"


def test_clarification_turn_suppresses_all_preference_retrieval() -> None:
    provider_calls = []

    class PreferenceProvider:
        def search(self, *, need, state):
            provider_calls.append(need)
            return []

    compilation, resolution, preference = _compile(
        "按我平时的方式回答，按华东的默认范围分析 GMV",
        preference_provider=PreferenceProvider(),
    )

    assert compilation["needs_clarification"] is True
    assert resolution["relation"] == "clarify"
    assert preference["source_needs"] == []
    assert preference["retrieval"]["provider_calls"] == 0
    assert provider_calls == []


def test_unsupported_region_preference_wording_still_clarifies() -> None:
    compilation, resolution, preference = _compile("按华东的默认范围分析 GMV")

    assert compilation["needs_clarification"] is True
    assert compilation["clarification_reason"] == "region_filter_not_supported"
    assert resolution["relation"] == "clarify"
    assert preference["source_needs"] == []


def test_explicit_operation_and_default_period_are_not_recalled() -> None:
    needs = detect_preference_source_needs({
        "user_message": "按我平时的分析习惯做同比，并默认最近一个月",
        "chat_session_id": "session-1",
        "context_compilation": {
            "known_slots": {
                "operation": {"value": "compare", "source": "current_turn"},
                "date": {"value": "最近一个月", "source": "current_turn"},
            },
        },
    })
    # The text names both intents, but also fully specifies both values. Reading
    # those same historical fields would be redundant and could violate the
    # current-turn-wins rule.
    assert needs == []


def test_task_state_fields_are_blocked_from_preference_need() -> None:
    needs = detect_preference_source_needs({
        "user_message": "按我平时的分析习惯分析",
        "chat_session_id": "session-1",
        "context_compilation": {
            "known_slots": {
                "dimension": {"value": "product_category", "source": "task_state"},
            },
        },
    })
    assert needs[0]["allowed_fields"]["analysis_method"] == ["operation"]
    assert needs[0]["blocked_fields"] == {
        "analysis_method.dimension": "task_state_value_present",
    }


def test_preference_needs_keep_durable_and_legacy_memory_gate_contract() -> None:
    durable = detect_preference_source_needs({
        "user_message": "按我平时的方式回答",
        "chat_session_id": "session-1",
        "memory_enabled": False,
    })
    legacy_disabled = detect_preference_source_needs({
        "user_message": "按我平时的方式回答",
        "memory_enabled": False,
    })
    assert durable[0]["memory_types"] == ["presentation"]
    assert legacy_disabled == []


def test_preference_candidates_cannot_change_authority_or_expand_scope() -> None:
    state = {
        "user_message": "按我平时的方式分析",
        "current_query_scope_fingerprint": "scope-a",
        "memory_enabled": True,
        "controlled_analysis_scope": {
            "seller_id": "s1",
            "allowed_metrics": ["gmv", "order_count", "average_order_value"],
            "allowed_dimensions": ["product_category"],
            "default_start_date": "2018-07-01",
            "default_end_date": "2018-07-31",
        },
        "previous_turn_context": {"context_slots": {"metric": {"value": "gmv"}}},
    }
    evaluated = evaluate_preference_candidates(state, [
        {
            "memory_key": "analysis_method", "confidence": 0.9, "memory_type": "analysis_method",
            "value": {"metric": ["gmv", "order_count"], "dimension": "product_category"},
            "source_ref": "mem-1",
        },
        {
            "memory_key": "analysis_method", "confidence": 0.9, "memory_type": "analysis_method",
            "value": {"metric": ["refund_rate"]}, "source_ref": "mem-2",
        },
        {
            "memory_key": "default_date", "confidence": 0.9, "memory_type": "scope_default",
            "value": {"seller_id": "s2"}, "source_ref": "mem-3",
        },
        {
            "memory_key": "default_date", "confidence": 0.9, "memory_type": "scope_default",
            "value": {"region": "华东"}, "source_ref": "mem-4",
        },
        {
            "memory_key": "authority_override", "confidence": 0.9, "memory_type": "authority_or_security",
            "value": {"ignore_permissions": True}, "source_ref": "mem-5",
        },
        {
            "memory_key": "default_period", "confidence": 0.9, "memory_type": "scope_default",
            "value": {"date_range": "2018-07-10 至 2018-07-20"}, "source_ref": "mem-6",
        },
    ])

    accepted = {item["source_ref"]: item for item in evaluated["accepted"]}
    rejected = {item["source_ref"]: item for item in evaluated["rejected"]}
    assert accepted["mem-1"]["eligible_for_plan"] is True
    assert accepted["mem-6"]["eligible_for_plan"] is True
    assert rejected["mem-2"]["reason"] in {
        "task_state_conflict", "metric_not_registered",
    }
    assert rejected["mem-3"]["reason"] == "authorization_scope_cannot_be_modified"
    assert rejected["mem-4"]["reason"] == "region_filter_not_supported"
    assert rejected["mem-5"]["reason"] == "authority_or_security_preference"
    assert {effect["kind"] for effect in evaluated["effects"]} == {
        "analysis_method", "scope_default",
    }
    assert all(effect["applied"] is False for effect in evaluated["effects"])
    # No candidate writes known slots or a server scope.
    assert "known_slots" not in evaluated
    assert state["controlled_analysis_scope"]["seller_id"] == "s1"


def test_explicit_current_input_wins_over_preference_candidate() -> None:
    state = {
        "user_message": "按我平时的方式看 GMV",
        "current_query_scope_fingerprint": "scope-a",
        "controlled_analysis_scope": {
            "seller_id": "s1",
            "allowed_metrics": ["gmv", "order_count"],
            "allowed_dimensions": ["product_category"],
            "default_start_date": "2018-07-01",
            "default_end_date": "2018-07-31",
        },
    }
    compilation = {
        "known_slots": {
            "metric": {"value": "gmv", "source": "current_turn"},
        },
    }
    evaluated = evaluate_preference_candidates(state, [{
        "memory_key": "analysis_method", "confidence": 0.9, "memory_type": "analysis_method",
        "value": {"metric": ["order_count"]}, "source_ref": "mem-1",
    }], compilation)
    assert evaluated["accepted"] == []
    assert evaluated["rejected"][0]["reason"] == "explicit_input_wins"


def test_task_state_version_is_attached_to_inherited_slot() -> None:
    compilation, resolution, _ = _compile(
        "那它有哪些限制？",
        previous_turn_context={},
        task_progress_context={
            "task_id": "ctk_1",
            "task_state_version": 7,
            "scope_fingerprint": "scope-a",
            "data_version_manifest_hash": "manifest-a",
            "latest_turn_id": "turn-task",
            "context_slots": {"metric": {
                "value": "gmv", "source": "current_turn",
                "source_turn_id": "turn-task",
            }},
        },
        data_version_manifest_hash="manifest-a",
    )
    assert compilation["known_slots"]["metric"]["source"] == "task_state"
    assert compilation["known_slots"]["metric"]["source_task_state_version"] == 7
    assert compilation["known_slots"]["metric"]["source_turn_id"] == "turn-task"
    inherited = {item["field"]: item for item in resolution["inherited_fields"]}
    assert inherited["metric"]["source_task_state_version"] == 7
    assert inherited["metric"]["source_turn_id"] == "turn-task"


def test_low_confidence_reference_without_context_clarifies() -> None:
    state = compile_turn_context({
        "user_message": "那它有哪些限制？",
        "previous_turn_context": {},
        "node_history": [],
    })
    compilation = state["context_compilation"]
    resolution = state["turn_resolution"]
    assert compilation["needs_clarification"] is True
    assert compilation["clarification_reason"] == "no_same_scope_metric_context"
    assert resolution["relation"] == "clarify"
    assert resolution["needs_clarification"] is True
    assert resolution["confidence"] <= 0.5


def _task(task_id: str, title: str, status: str = "running", *, version: int = 3) -> dict:
    return {
        "task_id": task_id,
        "status": status,
        "title": title,
        "objective": title,
        "scope_data": {
            "seller_id": "s1", "start_date": "2018-08-01", "end_date": "2018-08-31",
        },
        "task_state": {
            "scope_fingerprint": "scope-a",
            "data_version_manifest_hash": "manifest-a",
            "context_slots": {
                "metric": {"value": "gmv", "source": "current_turn"},
                "date": {"value": "2018-08-01 至 2018-08-31", "source": "current_turn"},
            },
        },
        "task_state_version": version,
    }


def _run_relation(
    tasks: list[dict], message: str, *, requested: str | None = None,
    scope: dict | None = None, decision: dict | None = None,
    llm_enabled: bool = True,
) -> dict:
    data = {"tasks": tasks, "current_task": tasks[0] if tasks else None, "memory": {}}
    with patch.object(service, "task_context", lambda context, session_id: data), \
         patch.object(service, "get_task", lambda context, task_id: next(
             (task for task in tasks if task["task_id"] == task_id), None)), \
         patch.object(service, "prior_evidence_ids", lambda *a, **k: []), \
         patch.object(service, "resolve_relation_llm", lambda *a, **k: decision), \
         patch.object(service.ChatHistoryRepository, "latest_completed_context", return_value={}), \
         patch.object(get_settings(), "relation_resolver_llm_enabled", llm_enabled):
        return service.relation(
            None, "session-1", message, scope, requested,
            scope_fingerprint="scope-a", expected_data_version="manifest-a",
        )


_SCOPE = {"seller_id": "s1", "start_date": "2018-08-01", "end_date": "2018-08-31"}


def test_multiple_active_tasks_require_clarification() -> None:
    tasks = [_task("ctk_1", "GMV 分析"), _task("ctk_2", "退款分析")]
    result = _run_relation(tasks, "继续")
    assert result["action"] == "clarify"
    assert result["reason"] == "multiple_active_tasks"
    relation = result["context_resolution"]["task_relation"]
    assert relation["candidate_task_ids"] == ["ctk_1", "ctk_2"]
    assert relation["ambiguous"] is True


def test_message_can_explicitly_select_one_of_multiple_tasks() -> None:
    tasks = [_task("ctk_1", "GMV 分析"), _task("ctk_2", "退款分析")]
    result = _run_relation(tasks, "继续退款分析")
    assert result["action"] == "continue"
    assert result["task_id"] == "ctk_2"
    assert result["context_resolution"]["task_relation"]["reason"] == "explicit_task_reference"


def test_task_scope_conflict_requires_clarification_not_silent_continue() -> None:
    task = _task("ctk_1", "GMV 分析")
    result = _run_relation(
        [task], "继续",
        scope={"seller_id": "s1", "start_date": "2018-07-01", "end_date": "2018-07-31"},
        decision={"action": "CONTINUE", "task_id": "ctk_1", "reusable_evidence_ids": [], "reason": "continue"},
    )
    assert result["action"] == "clarify"
    assert result["reason"] == "task_scope_conflict"
    assert result["context_resolution"]["task_relation"]["scope_conflict"] is True


def test_full_topic_switch_can_start_new_task() -> None:
    task = _task("ctk_1", "GMV 分析")
    result = _run_relation(
        [task], "分析退款率",
        scope=_SCOPE,
        decision={"action": "NEW_TASK", "task_id": None, "reusable_evidence_ids": [], "reason": "new topic"},
    )
    assert result["action"] == "new"
    assert result["reason"] == "llm_new_task"
    assert result.get("task_id") is None


def test_task_state_snapshot_keeps_bounded_slots() -> None:
    state = service._bounded_task_state(
        {
            "task_state_version": 2,
            "task_state": {
                "scope_fingerprint": "scope-a",
                "data_version_manifest_hash": "manifest-a",
                "context_slots": {"metric": {"value": "gmv"}},
            },
        },
        {
            "scope_fingerprint": "scope-a",
            "data_version_manifest_hash": "manifest-a",
            "context_slots": {
                "metric": {"value": "gmv", "source": "current_turn"},
                "date": {"value": "2018-08-01", "source": "current_turn"},
            },
        },
        status="completed", trace_id="trace-new", turn_id="turn-new",
    )
    assert state["state_version"] == 3
    assert state["context_slots"]["metric"]["value"] == "gmv"
    assert state["context_slots"]["metric"]["source_turn_id"] == "turn-new"
    assert state["context_slots"]["date"]["value"] == "2018-08-01"
    assert state["latest_turn_id"] == "turn-new"


def test_trace_strips_preference_values_but_keeps_source_need(monkeypatch, tmp_path) -> None:
    import app.harness.storage as storage

    monkeypatch.setattr(get_settings(), "trace_backend", "file")
    monkeypatch.setattr(get_settings(), "trace_dir", str(tmp_path))
    storage.write_trace("trace-wp2", {
        "trace_id": "trace-wp2",
        "tenant_id": "tenant",
        "user_id": "user",
        "node_history": [],
        "turn_resolution": {"source_needs": [{"trigger_ids": ["P1"]}]},
        "preference_context": {
            "status": "accepted",
            "source_needs": [{"trigger_ids": ["P1"]}],
            "items": [{"value": "secret-preference-text"}],
            "candidate_analysis_plan": {
                "status": "validated_candidate",
                "steps": [{"field": "dimension", "value": "secret-plan-value"}],
            },
        },
    })
    trace = json.loads((tmp_path / "trace-wp2.json").read_text(encoding="utf-8"))
    assert trace["state_snapshot"]["turn_resolution"]["source_needs"][0]["trigger_ids"] == ["P1"]
    assert "items" not in trace["state_snapshot"]["preference_context"]
    assert trace["state_snapshot"]["preference_context"]["candidate_analysis_plan"]["step_fields"] == ["dimension"]
    assert "secret-preference-text" not in json.dumps(trace, ensure_ascii=False)
    assert "secret-plan-value" not in json.dumps(trace, ensure_ascii=False)


def test_parallel_new_task_relation_is_preserved_in_turn_resolution() -> None:
    state = compile_turn_context({
        "user_message": "再看看 7 月的订单量",
        "previous_turn_context": _prior(),
        "turn_relation_decision": {
            "action": "new",
            "decision": "PARALLEL_NEW_TASK",
            "task_id": None,
            "reason": "parallel",
            "context_resolution": {
                "task_relation": {
                    "candidate_task_ids": ["ctk_1"],
                    "decision_driver": "llm",
                },
            },
        },
        "node_history": [],
    })
    assert state["turn_resolution"]["relation"] == "parallel_new"
    assert state["turn_resolution"]["task_relation"]["action"] == "parallel_new"
    assert state["turn_resolution"]["task_relation"]["decision_driver"] == "llm"


def test_stale_task_state_slots_are_not_inherited_after_scope_change() -> None:
    state = compile_turn_context({
        "user_message": "那它有哪些限制？",
        "previous_turn_context": {},
        "current_query_scope_fingerprint": "scope-new",
        "data_version_manifest_hash": "manifest-new",
        "task_progress_context": {
            "task_id": "ctk_1",
            "task_state_version": 4,
            "scope_fingerprint": "scope-old",
            "data_version_manifest_hash": "manifest-old",
            "context_slots": {"metric": {"value": "gmv"}},
        },
        "node_history": [],
    })
    assert "metric" not in state["context_compilation"]["known_slots"]
    assert state["context_compilation"]["needs_clarification"] is True
    assert state["turn_resolution"]["needs_clarification"] is True
    assert state["turn_resolution"]["inherited_fields"] == []


def test_task_state_direct_relation_is_not_silently_used_for_new_scope() -> None:
    stale = _task("ctk_1", "GMV 分析")
    stale["task_state"]["scope_fingerprint"] = "scope-old"
    stale["task_state"]["data_version_manifest_hash"] = "manifest-old"
    active = service._compatible_task_progress(
        stale, scope_fingerprint="scope-new", expected_data_version="manifest-new",
    )
    assert "context_slots" not in active
    assert active["task_id"] == "ctk_1"


def test_accepted_candidate_does_not_widen_final_request_contract_without_server_apply() -> None:
    preference = {
        "schema_version": "preference-context-v1",
        "status": "accepted",
        "availability": "available",
        "source_needs": [{
            "source": "mem0", "rules_version": "preference-intent-rules-v4",
            "memory_types": ["analysis_method"],
            "allowed_keys": {"analysis_method": ["analysis_method"]},
            "allowed_fields": {"analysis_method": ["dimension"]},
        }],
        "accepted": [{
            "candidate_id": "pref_1", "memory_key": "analysis_method",
            "memory_type": "analysis_method", "status": "accepted",
            "eligible_for_plan": True,
        }],
        "rejected": [],
        "items": [],
        "effects": [{
            "candidate_id": "pref_1", "memory_key": "analysis_method",
            "memory_type": "analysis_method", "kind": "analysis_method",
            "applies_to": "candidate_plan", "eligible_for_plan": True,
            "applied": False, "fields": {"dimension": "product_category"},
        }],
    }
    state = compile_turn_context({
        "user_message": "按我平时的方式分析 GMV",
        "previous_turn_context": _prior(),
        "preference_context": preference,
        "node_history": [],
    })
    compilation = state["context_compilation"]
    assert compilation["known_slots"]["metric"]["value"] == "gmv"
    assert "dimension" not in compilation["known_slots"]
    assert compilation["request_contract"]["metric_id"] == "gmv"
    assert compilation["request_contract"]["dimension"] is None
    assert state["turn_resolution"]["candidate_fields"] == []
    assert state["turn_resolution"]["preference"]["effects"][0]["applied"] is False


def test_correction_outside_locked_scope_is_rejected_before_tool_plan() -> None:
    compilation, resolution, _ = _compile(
        "换成 7 月",
        controlled_analysis_scope={
            "seller_id": "s1",
            "allowed_metrics": ["gmv"],
            "allowed_dimensions": ["product_category"],
            "default_start_date": "2018-08-01",
            "default_end_date": "2018-08-31",
        },
    )
    assert compilation["known_slots"]["date"]["source"] == "current_turn_correction"
    assert compilation["needs_clarification"] is True
    assert compilation["clarification_reason"] == "date_outside_locked_scope"
    assert compilation["request_contract"]["parse_status"] == "clarification_required"
    assert compilation["scope_candidates"][0]["status"] == "rejected"
    assert resolution["relation"] == "clarify"


def test_explicit_metric_correction_is_recorded_as_user_change() -> None:
    compilation, resolution, _ = _compile("改成订单量")
    assert compilation["known_slots"]["metric"]["value"] == "order_count"
    assert compilation["known_slots"]["metric"]["source"] == "current_turn"
    assert compilation["user_changes"][0] == {
        "field": "metric",
        "from": "gmv",
        "to": "order_count",
        "source": "current_turn",
        "reason": "explicit_user_correction",
    }
    assert resolution["user_changes"][0]["field"] == "metric"
    assert "metric" in resolution["explicit_fields"]


def test_deterministic_fallback_starts_new_task_for_independent_explicit_question() -> None:
    task = _task("ctk_1", "GMV 分析")
    result = _run_relation(
        [task], "分析 GMV", scope=_SCOPE, decision=None, llm_enabled=False,
    )
    assert result["action"] == "new"
    assert result["reason"] == "independent_topic"


def test_graph_init_defaults_still_produce_a_preference_source_need() -> None:
    from app.graph.workflow import init_state

    state = init_state({
        "user_message": "按我平时的方式回答",
        "trace_id": "trace-init",
        "chat_session_id": "session-init",
        "previous_turn_context": _prior(),
    })
    assert state["preference_context"] == {}
    compiled = compile_turn_context(state)
    assert compiled["preference_context"]["source_needs"][0]["trigger_ids"] == ["P1"]
    assert compiled["turn_resolution"]["source_needs"][0]["trigger_ids"] == ["P1"]


def test_validated_preference_is_assembled_as_preference_tier_in_bundle() -> None:
    state = {
        "user_message": "按我平时的方式",
        "trace_id": "trace-pref-bundle",
        "current_query_scope_fingerprint": "scope-a",
        "context_compilation": {"needs_clarification": False},
        "turn_resolution": {
            "relation": "continue",
            "source_needs": [{
                "source": "mem0", "trigger_ids": ["P1"],
                "memory_types": ["presentation"], "status": "requested",
            }],
            "explicit_fields": [], "inherited_fields": [],
        },
        "preference_context": {
            "schema_version": "preference-context-v1",
            "status": "accepted",
            "availability": "available",
            "source_needs": [],
            "accepted": [{
                "candidate_id": "pref_style", "memory_key": "response_style",
                "memory_type": "presentation", "status": "accepted",
                "eligible_for_plan": False,
            }],
            "rejected": [],
            "items": [{
                "candidate_id": "pref_style", "memory_key": "response_style",
                "memory_type": "presentation", "source_ref": "mem-style",
                "status": "accepted", "value": "先结论后表格",
            }],
            "candidate_analysis_plan": {
                "schema_version": "preference-candidate-plan-v1",
                "status": "validated_candidate",
                "candidate_ids": ["pref_style"],
                "steps": [{"field": "dimension", "value": "product_category", "candidate_id": "pref_1"}],
                "applied": False,
                "requires_server_application": True,
            },
        },
    }
    triage = ContextOrchestrator().build_triage_context(
        state, instruction="按我平时的方式", model="test-model",
    )
    preference_items = [item for item in triage.bundle.items if item.source.value == "mem0"]
    assert len(preference_items) == 1
    assert preference_items[0].trust_tier.value == "preference"
    assert preference_items[0].item_id == "preference.pref_style"
    assert triage.metadata()["preference_source_needs"][0]["trigger_ids"] == ["P1"]
    assert triage.metadata()["preference_candidates"][0]["candidate_id"] == "pref_style"
    assert triage.metadata()["candidate_analysis_plan"]["step_fields"] == ["dimension"]
    assert triage.metadata()["candidate_analysis_plan"]["applied"] is False
    assert not any(
        row["source"] == "mem0"
        for row in state["context_source_availability"]["triage"]
    )


def test_answer_bundle_receives_only_presentation_preferences() -> None:
    state = {
        "trace_id": "trace-answer-pref",
        "current_query_scope_fingerprint": "scope-a",
        "preference_context": {
            "items": [
                {"candidate_id": "pref_pres", "memory_key": "style",
                 "memory_type": "presentation", "status": "accepted", "value": "concise"},
                {"candidate_id": "pref_method", "memory_key": "method",
                 "memory_type": "analysis_method", "status": "accepted",
                 "value": {"dimension": "product_category"}},
            ],
        },
        "turn_resolution": {"relation": "continue"},
    }
    bundle = ContextOrchestrator().build_specialist_bundle(
        state, "answer", instruction="q", model="test-model",
    )
    ids = {item.item_id for item in bundle.items if item.source.value == "mem0"}
    assert ids == {"preference.pref_pres"}


def test_missing_preference_confidence_is_rejected_by_default_threshold() -> None:
    state = {
        "user_message": "按我平时的方式",
        "current_query_scope_fingerprint": "scope-a",
        "controlled_analysis_scope": {
            "seller_id": "s1",
            "allowed_metrics": ["gmv"],
            "allowed_dimensions": ["product_category"],
            "default_start_date": "2018-07-01",
            "default_end_date": "2018-07-31",
        },
    }
    evaluated = evaluate_preference_candidates(state, [{
        "memory_key": "response_style", "memory_type": "presentation", "value": "concise",
    }])
    assert evaluated["accepted"] == []
    assert evaluated["rejected"][0]["reason"] == "below_relevance_threshold"


def test_scope_default_cannot_override_explicit_or_inherited_date() -> None:
    state = {
        "user_message": "按我平时的方式，7 月 10 日",
        "current_query_scope_fingerprint": "scope-a",
        "controlled_analysis_scope": {
            "seller_id": "s1",
            "allowed_metrics": ["gmv"],
            "allowed_dimensions": ["product_category"],
            "default_start_date": "2018-07-01",
            "default_end_date": "2018-07-31",
        },
        "previous_turn_context": {"context_slots": {
            "date": {"value": "2018-07-10 至 2018-07-10"},
        }},
    }
    explicit_compilation = {"known_slots": {
        "date": {"value": "2018-07-10 至 2018-07-10", "source": "current_turn"},
    }}
    explicit_conflict = evaluate_preference_candidates(state, [{
        "memory_key": "default_period", "memory_type": "scope_default",
        "value": {"date_range": "2018-07-01 至 2018-07-31"},
        "confidence": 0.9,
    }], explicit_compilation)
    assert explicit_conflict["accepted"] == []
    assert explicit_conflict["rejected"][0]["reason"] == "explicit_input_wins"

    inherited_conflict = evaluate_preference_candidates(state, [{
        "memory_key": "default_period", "memory_type": "scope_default",
        "value": {"date_range": "2018-07-01 至 2018-07-31"},
        "confidence": 0.9,
    }], {"known_slots": {}})
    assert inherited_conflict["accepted"] == []
    assert inherited_conflict["rejected"][0]["reason"] == "task_state_conflict"


def test_validated_analysis_preference_produces_candidate_analysis_plan() -> None:
    state = {
        "user_message": "按我平时的方式分析",
        "current_query_scope_fingerprint": "scope-a",
        "controlled_analysis_scope": {
            "seller_id": "s1",
            "allowed_metrics": ["gmv", "order_count"],
            "allowed_dimensions": ["product_category"],
            "default_start_date": "2018-07-01",
            "default_end_date": "2018-07-31",
        },
    }
    evaluated = evaluate_preference_candidates(state, [{
        "memory_key": "analysis_method", "memory_type": "analysis_method",
        "value": {"metric": ["gmv", "order_count"], "dimension": "product_category"},
        "confidence": 0.9,
    }])
    plan = evaluated["candidate_analysis_plan"]
    assert plan["status"] == "validated_candidate"
    assert plan["applied"] is False
    assert plan["requires_server_application"] is True
    assert {step["field"] for step in plan["steps"]} == {"metric", "dimension"}
    assert all(step["source"] == "mem0_preference" for step in plan["steps"])


def test_unversioned_task_state_is_not_inherited() -> None:
    state = compile_turn_context({
        "user_message": "那它有哪些限制？",
        "previous_turn_context": {},
        "current_query_scope_fingerprint": "scope-a",
        "data_version_manifest_hash": "manifest-a",
        "task_progress_context": {
            "task_id": "ctk_1",
            "task_state_version": 4,
            "scope_fingerprint": "scope-a",
            "context_slots": {"metric": {"value": "gmv"}},
        },
        "node_history": [],
    })
    assert "metric" not in state["context_compilation"]["known_slots"]
    assert state["context_compilation"]["needs_clarification"] is True


def test_turn_resolution_exposes_candidate_plan_without_preference_values() -> None:
    preference = {
        "schema_version": "preference-context-v1",
        "status": "accepted",
        "availability": "available",
        "source_needs": [{
            "source": "mem0", "rules_version": "preference-intent-rules-v4",
            "memory_types": ["analysis_method"],
            "allowed_keys": {"analysis_method": ["analysis_method"]},
            "allowed_fields": {"analysis_method": ["dimension"]},
        }],
        "accepted": [],
        "rejected": [],
        "items": [],
        "candidate_analysis_plan": {
            "schema_version": "preference-candidate-plan-v1",
            "status": "validated_candidate",
            "candidate_ids": ["pref_1"],
            "steps": [{
                "field": "dimension", "value": "product_category", "candidate_id": "pref_1",
                "memory_type": "analysis_method", "memory_key": "analysis_method",
                "rules_version": "preference-intent-rules-v4",
            }],
            "applied": False,
            "requires_server_application": True,
        },
    }
    state = compile_turn_context({
        "user_message": "按我平时的方式分析 GMV",
        "previous_turn_context": _prior(),
        "current_query_scope_fingerprint": "scope-a",
        "controlled_analysis_scope": {
            "allowed_metrics": ["gmv"],
            "allowed_dimensions": ["product_category"],
        },
        "preference_context": preference,
        "node_history": [],
    })
    plan = state["turn_resolution"]["preference"]["candidate_analysis_plan"]
    assert plan["status"] == "applied"
    assert plan["step_fields"] == ["dimension"]
    assert plan["applied"] is True
    assert plan["requires_server_application"] is False
    assert "product_category" not in repr(plan)
    assert state["context_compilation"]["request_contract"]["dimension"] == "product_category"
    assert state["context_compilation"]["request_contract"]["preference_application"] == {
        "source": "validated_mem0_candidate",
        "candidate_ids": ["pref_1"],
        "fields": ["dimension"],
    }
    assert state["context_compilation"]["known_slots"]["dimension"]["source"] == "validated_preference_candidate"
    from app.context.request_contract import validate_compiled_request_contract

    assert validate_compiled_request_contract(
        state["context_compilation"],
        state_contract_hash=state["request_contract_hash"],
        require_state_anchor=True,
    ) == (True, None)


def test_scope_default_conflicts_with_task_state_date() -> None:
    state = {
        "user_message": "按我平时的方式",
        "current_query_scope_fingerprint": "scope-a",
        "data_version_manifest_hash": "manifest-a",
        "controlled_analysis_scope": {
            "seller_id": "s1",
            "allowed_metrics": ["gmv"],
            "allowed_dimensions": ["product_category"],
            "default_start_date": "2018-07-01",
            "default_end_date": "2018-07-31",
        },
        "task_progress_context": {
            "scope_fingerprint": "scope-a",
            "data_version_manifest_hash": "manifest-a",
            "context_slots": {
                "date": {"value": "2018-07-10 至 2018-07-10"},
            },
        },
    }
    evaluated = evaluate_preference_candidates(state, [{
        "memory_key": "default_period", "memory_type": "scope_default",
        "value": {"date_range": "2018-07-01 至 2018-07-31"},
        "confidence": 0.9,
    }])
    assert evaluated["accepted"] == []
    assert evaluated["rejected"][0]["reason"] == "task_state_conflict"


def test_preference_provider_retrieval_metrics_are_recorded() -> None:
    class _Provider:
        def search(self, *, need, state):
            return [{
                "memory_key": "analysis_method", "memory_type": "analysis_method",
                "value": {"dimension": "product_category"}, "confidence": 0.9,
                "source_ref": "provider-hit-1",
            }]

    state = {
        "user_message": "按我平时的方式分析",
        "current_query_scope_fingerprint": "scope-a",
        "memory_enabled": True,
        "controlled_analysis_scope": {
            "seller_id": "s1",
            "allowed_metrics": ["gmv"],
            "allowed_dimensions": ["product_category"],
            "default_start_date": "2018-07-01",
            "default_end_date": "2018-07-31",
        },
    }
    context = resolve_preference_context(
        state, provider=_Provider(), compilation={"known_slots": {}},
    )
    assert context["retrieval"]["provider_calls"] == 1
    assert context["retrieval"]["hit_count"] == 1
    assert context["retrieval"]["top_k_total"] == 10
    assert context["retrieval"]["threshold"] == 0.55
    assert context["retrieval"]["elapsed_ms"] >= 0
    assert context["candidate_analysis_plan"]["status"] == "validated_candidate"


def test_chat_api_passes_relation_decision_to_workflow() -> None:
    from fastapi.testclient import TestClient

    from app.api.auth import require_tenant_context
    from app.main import app
    from app.tenancy import TenantContext

    context = TenantContext(
        tenant_id="platform", user_id="wp2-api-test", role="admin", scope_type="platform",
    )
    decision = {
        "action": "continue",
        "task_id": "ctk_12345678",
        "reason": "resolver_continue",
        "resolved_message": "继续分析",
        "context_resolution": {
            "turn_resolution": {
                "source_needs": [{"source": "mem0", "trigger_ids": ["P1"]}],
            },
        },
    }
    captured = {}

    def fake_workflow(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return {
            "task_type": "metric_query", "answer": "ok", "trace_id": "trace-api",
            "error": False, "errors": [], "clarification": None, "presentation": None,
        }

    app.dependency_overrides[require_tenant_context] = lambda: context
    try:
        with patch("app.api.routes.chat.relation", return_value=decision), \
             patch("app.api.routes.chat.update_task", return_value={}), \
             patch("app.api.routes.chat.record_task_progress", return_value={}), \
             patch("app.api.routes.chat.run_chat_workflow", side_effect=fake_workflow):
            response = TestClient(app).post("/chat", json={
                "message": "继续分析",
                "chat_session_id": "session-api",
                "task_id": "ctk_12345678",
            })
    finally:
        app.dependency_overrides.pop(require_tenant_context, None)

    assert response.status_code == 200, response.text
    assert captured["turn_relation_decision"] is decision
    assert captured["args"][10] == "ctk_12345678"


def test_rejected_preference_fields_never_reach_triage_prompt() -> None:
    state = {
        "user_message": "按我平时的方式分析",
        "current_query_scope_fingerprint": "scope-a",
        "memory_enabled": True,
        "controlled_analysis_scope": {
            "seller_id": "s1",
            "allowed_metrics": ["gmv"],
            "allowed_dimensions": ["product_category"],
            "default_start_date": "2018-07-01",
            "default_end_date": "2018-07-31",
        },
    }
    evaluated = evaluate_preference_candidates(state, [{
        "memory_key": "analysis_method", "memory_type": "analysis_method",
        "value": {
            "dimension": "product_category",
            "seller_id": "s2",
            "region": "华东",
        },
        "confidence": 0.9,
    }])
    assert evaluated["accepted"]
    rendered_items = evaluated["items"]
    assert all("seller_id" not in json.dumps(item, ensure_ascii=False) for item in rendered_items)
    assert all("华东" not in json.dumps(item, ensure_ascii=False) for item in rendered_items)

    triage_state = {
        "user_message": "按我平时的方式分析",
        "trace_id": "trace-pref-sanitized",
        "current_query_scope_fingerprint": "scope-a",
        "context_compilation": {"needs_clarification": False},
        "turn_resolution": {"relation": "continue", "source_needs": []},
        "preference_context": {
            **evaluated,
            "availability": "available",
            "source_needs": [],
        },
    }
    triage = ContextOrchestrator().build_triage_context(
        triage_state, instruction="按我平时的方式分析", model="test-model",
    )
    assert "seller_id" not in triage.bundle.text
    assert "s2" not in triage.bundle.text
    assert "华东" not in triage.bundle.text


def test_candidate_type_key_and_field_must_match_source_need() -> None:
    state = {"user_message": "按我平时的分析日期习惯", "current_query_scope_fingerprint": "scope-a"}
    presentation_need = [{
        "source": "mem0", "rules_version": "preference-intent-rules-v4",
        "memory_types": ["presentation"],
        "allowed_keys": {"presentation": ["response_style"]},
        "allowed_fields": {"presentation": ["style"]},
    }]
    type_mismatch = evaluate_preference_candidates(state, [{
        "memory_key": "analysis_method", "memory_type": "analysis_method",
        "value": {"dimension": "product_category"}, "confidence": 0.9,
    }], source_needs=presentation_need)
    assert type_mismatch["accepted"] == []
    assert type_mismatch["rejected"][0]["reason"] == "source_need_type_not_allowed"

    key_mismatch_need = [{
        "source": "mem0", "rules_version": "preference-intent-rules-v4",
        "memory_types": ["analysis_method"],
        "allowed_keys": {"analysis_method": ["analysis_method"]},
        "allowed_fields": {"analysis_method": ["dimension"]},
    }]
    key_mismatch = evaluate_preference_candidates(state, [{
        "memory_key": "analysis_steps", "memory_type": "analysis_method",
        "value": {"dimension": "product_category"}, "confidence": 0.9,
    }], source_needs=key_mismatch_need)
    assert key_mismatch["accepted"] == []
    assert key_mismatch["rejected"][0]["reason"] == "source_need_key_not_allowed"

    date_only_need = [{
        "source": "mem0", "rules_version": "preference-intent-rules-v4",
        "memory_types": ["analysis_method"],
        "allowed_keys": {"analysis_method": ["analysis_method"]},
        "allowed_fields": {"analysis_method": ["date"]},
    }]
    projected = evaluate_preference_candidates(
        {**state, "controlled_analysis_scope": {
            "default_start_date": "2018-07-01", "default_end_date": "2018-07-31",
        }},
        [{
            "memory_key": "analysis_method", "memory_type": "analysis_method",
            "value": {"date": "2018-07-10", "dimension": "product_category"},
            "confidence": 0.9,
        }],
        source_needs=date_only_need,
    )
    assert projected["accepted"]
    assert projected["effects"][0]["fields"] == {"date": "2018-07-10"}
    assert projected["accepted"][0]["field_rejections"] == [
        {"field": "dimension", "reason": "outside_source_need"},
    ]


def test_apply_candidate_analysis_plan_fails_closed_without_matching_field_need() -> None:
    from app.context.turn_resolver import apply_candidate_analysis_plan

    compilation = {
        "needs_clarification": False,
        "known_slots": {},
        "request_contract": {"metric_id": "gmv", "dimension": None},
    }
    state = {"controlled_analysis_scope": {
        "allowed_metrics": ["gmv"], "allowed_dimensions": ["product_category"],
    }}
    plan = {
        "status": "validated_candidate", "applied": False,
        "steps": [{
            "field": "dimension", "value": "product_category", "candidate_id": "pref-1",
            "memory_type": "analysis_method", "memory_key": "analysis_method",
            "rules_version": "preference-intent-rules-v4",
        }],
    }
    preference = {
        "source_needs": [{
            "source": "mem0", "rules_version": "preference-intent-rules-v4",
            "memory_types": ["analysis_method"],
            "allowed_keys": {"analysis_method": ["analysis_method"]},
            "allowed_fields": {"analysis_method": ["date"]},
        }],
        "candidate_analysis_plan": plan,
    }
    result = apply_candidate_analysis_plan(state, compilation, preference)
    assert result["candidate_analysis_plan"]["applied"] is False
    assert result["candidate_analysis_plan"]["steps"][0]["application_reason"] == "source_need_field_not_allowed"
    assert compilation["request_contract"]["dimension"] is None
    assert "dimension" not in compilation["known_slots"]


def test_apply_candidate_analysis_plan_rechecks_excluded_dimension_value() -> None:
    from app.context.turn_resolver import apply_candidate_analysis_plan

    compilation = {
        "needs_clarification": False,
        "known_slots": {},
        "request_contract": {"metric_id": "gmv", "dimension": None},
    }
    preference = {
        "source_needs": [{
            "source": "mem0", "rules_version": "preference-intent-rules-v4",
            "memory_types": ["analysis_method"],
            "allowed_keys": {"analysis_method": ["analysis_method"]},
            "allowed_fields": {"analysis_method": ["dimension"]},
            "excluded_values": {"analysis_method.dimension": ["product_category"]},
        }],
        "candidate_analysis_plan": {
            "status": "validated_candidate", "applied": False,
            "steps": [{
                "field": "dimension", "value": "product_category", "candidate_id": "pref-excluded",
                "memory_type": "analysis_method", "memory_key": "analysis_method",
                "rules_version": "preference-intent-rules-v4",
            }],
        },
    }
    result = apply_candidate_analysis_plan({"controlled_analysis_scope": {
        "allowed_metrics": ["gmv"], "allowed_dimensions": ["product_category"],
    }}, compilation, preference)
    assert result["candidate_analysis_plan"]["applied"] is False
    assert result["candidate_analysis_plan"]["steps"][0]["application_reason"] == "source_need_value_excluded"
    assert compilation["request_contract"]["dimension"] is None
    assert "dimension" not in compilation["known_slots"]


def test_apply_candidate_analysis_plan_fails_closed_for_missing_stale_or_wrong_type_need() -> None:
    from app.context.turn_resolver import apply_candidate_analysis_plan

    rejected_needs = [
        ([], "source_need_missing"),
        ([{
            "source": "mem0", "rules_version": "preference-intent-rules-v0",
            "memory_types": ["analysis_method"],
            "allowed_keys": {"analysis_method": ["analysis_method"]},
            "allowed_fields": {"analysis_method": ["dimension"]},
        }], "source_need_rules_version_mismatch"),
        ([{
            "source": "mem0", "rules_version": "preference-intent-rules-v1",
            "memory_types": ["analysis_method"],
            "allowed_keys": {"analysis_method": ["analysis_method"]},
            "allowed_fields": {"analysis_method": ["dimension"]},
        }], "source_need_rules_version_mismatch"),
        ([{
            "source": "mem0", "rules_version": "preference-intent-rules-v4",
            "memory_types": ["presentation"],
            "allowed_keys": {"presentation": ["response_style"]},
            "allowed_fields": {"presentation": ["style"]},
        }], "source_need_type_not_allowed"),
    ]
    for source_needs, expected_reason in rejected_needs:
        compilation = {
            "needs_clarification": False,
            "known_slots": {},
            "request_contract": {"metric_id": "gmv", "dimension": None},
        }
        preference = {
            "source_needs": source_needs,
            "candidate_analysis_plan": {
                "status": "validated_candidate", "applied": False,
                "steps": [{
                    "field": "dimension", "value": "product_category", "candidate_id": "pref-1",
                    "memory_type": "analysis_method", "memory_key": "analysis_method",
                    "rules_version": "preference-intent-rules-v4",
                }],
            },
        }
        result = apply_candidate_analysis_plan({"controlled_analysis_scope": {
            "allowed_metrics": ["gmv"], "allowed_dimensions": ["product_category"],
        }}, compilation, preference)
        assert result["candidate_analysis_plan"]["steps"][0]["application_reason"] == expected_reason
        assert result["candidate_analysis_plan"]["applied"] is False
        assert compilation["request_contract"]["dimension"] is None


def test_old_task_slot_without_source_turn_is_not_relabelled_as_latest_turn() -> None:
    state = compile_turn_context({
        "user_message": "那它有哪些限制？",
        "previous_turn_context": {"source_turn_id": "turn-prior"},
        "current_query_scope_fingerprint": "scope-a",
        "data_version_manifest_hash": "manifest-a",
        "task_progress_context": {
            "task_id": "ctk_1",
            "task_state_version": 9,
            "scope_fingerprint": "scope-a",
            "data_version_manifest_hash": "manifest-a",
            "latest_turn_id": "turn-latest",
            "context_slots": {"metric": {"value": "gmv", "source": "current_turn"}},
        },
        "node_history": [],
    })
    slot = state["context_compilation"]["known_slots"]["metric"]
    assert slot["source"] == "task_state"
    assert slot["source_turn_id"] is None
    inherited = state["turn_resolution"]["inherited_fields"][0]
    assert inherited["source_turn_id"] is None
    assert inherited["source_turn_id_status"] == "unknown"
    assert inherited["source_task_state_version"] == 9
    assert inherited.get("source_turn_id") != "turn-latest"


def test_chat_stream_passes_relation_decision_to_stream_workflow() -> None:
    from fastapi.testclient import TestClient

    from app.api.auth import require_tenant_context
    from app.main import app
    from app.tenancy import TenantContext

    context = TenantContext(
        tenant_id="platform", user_id="wp2-stream-test", role="admin", scope_type="platform",
    )
    decision = {
        "action": "continue",
        "task_id": "ctk_87654321",
        "reason": "resolver_continue",
        "resolved_message": "继续分析",
        "context_resolution": {"turn_resolution": {"source_needs": []}},
    }
    captured = {}

    def fake_stream(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        yield {
            "event": "completed", "sequence_number": 1, "task_type": "metric_query",
            "answer": "stream-ok", "trace_id": "trace-stream", "error": False,
            "errors": [], "clarification": None, "presentation": None,
        }

    app.dependency_overrides[require_tenant_context] = lambda: context
    try:
        with patch("app.api.routes.chat.relation", return_value=decision), \
             patch("app.api.routes.chat.update_task", return_value={}), \
             patch("app.api.routes.chat.record_task_progress", return_value={}), \
             patch("app.api.routes.chat.stream_chat_workflow", side_effect=fake_stream):
            response = TestClient(app).post("/chat/stream", json={
                "message": "继续分析",
                "chat_session_id": "session-stream",
                "task_id": "ctk_87654321",
            })
    finally:
        app.dependency_overrides.pop(require_tenant_context, None)

    assert response.status_code == 200, response.text
    assert "stream-ok" in response.text
    assert captured["turn_relation_decision"] is decision


def test_agent_state_declares_turn_identity_for_graph_persistence() -> None:
    from app.graph.state import AGENTSTATE_KEYS

    assert {"task_id", "chat_session_id", "chat_turn_id"} <= AGENTSTATE_KEYS


def test_turn_resolution_uses_graph_task_and_session_identity() -> None:
    state = compile_turn_context({
        "user_message": "那它有哪些限制？",
        "previous_turn_context": _prior(),
        "task_id": "ctk_graph_identity",
        "chat_session_id": "session_graph_identity",
        "chat_turn_id": "turn_graph_identity",
        "node_history": [],
    })
    resolution = state["turn_resolution"]
    assert resolution["task_id"] == "ctk_graph_identity"
    assert resolution["session_id"] == "session_graph_identity"


def test_init_state_preserves_turn_identity_for_langgraph() -> None:
    from app.graph.workflow import init_state

    state = init_state({
        "user_message": "q", "trace_id": "trace",
        "task_id": "ctk_init", "chat_session_id": "session_init",
        "chat_turn_id": "turn_init",
    })
    assert state["task_id"] == "ctk_init"
    assert state["chat_session_id"] == "session_init"
    assert state["chat_turn_id"] == "turn_init"


def test_durable_turn_id_wins_over_client_turn_id_and_is_used_by_bundle() -> None:
    state = compile_turn_context({
        "user_message": "那它有哪些限制？",
        "previous_turn_context": _prior(),
        "persistence_turn_id": "persist-durable-1",
        "chat_turn_id": "client-idempotency-1",
        "chat_session_id": "session-1",
        "task_id": "ctk_1",
        "node_history": [],
    })
    resolution = state["turn_resolution"]
    assert resolution["turn_id"] == "persist-durable-1"
    assert resolution["client_turn_id"] == "client-idempotency-1"
    bundle = ContextOrchestrator().build_specialist_bundle(
        state, "triage", instruction="q", model="test-model",
    )
    assert bundle.turn_id == "persist-durable-1"
    assert bundle.resolved_request["client_turn_id"] == "client-idempotency-1"


def test_client_turn_id_is_the_fallback_canonical_turn_id() -> None:
    state = compile_turn_context({
        "user_message": "那它有哪些限制？",
        "previous_turn_context": _prior(),
        "chat_turn_id": "client-only-1",
        "trace_id": "trace-fallback-1",
        "node_history": [],
    })
    assert state["turn_resolution"]["turn_id"] == "client-only-1"
    bundle = ContextOrchestrator().build_specialist_bundle(
        state, "triage", instruction="那它有哪些限制？", model="test-model",
    )
    assert bundle.turn_id == "client-only-1"
    assert bundle.resolved_request["client_turn_id"] == "client-only-1"


def test_run_chat_workflow_puts_both_turn_ids_in_graph_state(monkeypatch) -> None:
    from app.graph import workflow as wf

    captured = {}

    class FakeGraph:
        def invoke(self, state, config=None, durability=None):
            captured["state"] = state
            return {**state, "task_type": "metric_definition", "answer": "ok"}

    monkeypatch.setattr(wf, "build_chat_graph", lambda **kwargs: FakeGraph())
    monkeypatch.setattr(wf, "_persist_chat_turn", lambda *a, **k: None)
    wf.run_chat_workflow(
        "q", None, None, None, False, None, False, None,
        "session-1", "client-1", "ctk_1",
        persistence_turn_id="persist-1", turn_relation_decision={"action": "continue"},
    )
    state = captured["state"]
    assert state["chat_turn_id"] == "client-1"
    assert state["persistence_turn_id"] == "persist-1"
    assert state["chat_session_id"] == "session-1"
    assert state["task_id"] == "ctk_1"


def test_stream_chat_workflow_puts_both_turn_ids_in_graph_state(monkeypatch) -> None:
    from app.graph import workflow as wf

    captured = {}

    class FakeGraph:
        def stream(self, state, config=None, stream_mode=None, durability=None):
            captured["state"] = state
            yield {"init_state": {"trace_id": state.get("trace_id")}}

    monkeypatch.setattr(wf, "build_chat_graph", lambda **kwargs: FakeGraph())
    monkeypatch.setattr(wf, "_persist_chat_turn", lambda *a, **k: None)
    list(wf.stream_chat_workflow(
        "q", None, None, None, False, None, False, None,
        "session-stream", "client-stream", "ctk_stream",
        persistence_turn_id="persist-stream", emit_started=False,
        turn_relation_decision={"action": "continue"},
    ))
    state = captured["state"]
    assert state["chat_turn_id"] == "client-stream"
    assert state["persistence_turn_id"] == "persist-stream"
    assert state["chat_session_id"] == "session-stream"
    assert state["task_id"] == "ctk_stream"


def test_compiler_resolution_flows_into_wp1_triage_bundle_with_provenance() -> None:
    """Exercise the WP2 -> WP1 seam using the real compiler and orchestrator."""
    state = compile_turn_context({
        "user_message": "换成 7 月，按品类拆开",
        "previous_turn_context": _prior(),
        "current_query_scope_fingerprint": "scope-a",
        "persistence_turn_id": "durable-seam-turn",
        "chat_turn_id": "client-seam-key",
        "chat_session_id": "session-seam",
        "task_id": "ctk-seam",
        "node_history": [],
    })

    triage = ContextOrchestrator().build_triage_context(
        state, instruction="换成 7 月，按品类拆开", model="test-model",
    )
    resolution = state["turn_resolution"]
    bundle = triage.bundle

    assert resolution["turn_id"] == bundle.turn_id == "durable-seam-turn"
    assert resolution["client_turn_id"] == bundle.resolved_request["client_turn_id"] == "client-seam-key"
    assert bundle.task_id == "ctk-seam"
    assert bundle.resolved_request["text"] == state["context_compilation"]["standalone_question"]
    assert bundle.resolved_request["relation"] == "continue"
    assert triage.route_slots["metric"] == "gmv"
    assert triage.route_slots["date"] == "2018-07-01 至 2018-07-31"
    assert triage.route_slots["dimension"] == "product_category"

    inherited = {item["field"]: item for item in bundle.resolved_request["inherited_fields"]}
    assert inherited["metric"]["source"] == "previous_turn_context"
    assert inherited["metric"]["source_turn_id"] == "turn-prior"
    assert bundle.resolved_request["user_changes"][0]["field"] == "date"

    recorded = state["context_bundles"]["triage"]["resolved_request"]
    assert recorded["turn_id"] == "durable-seam-turn"
    assert recorded["client_turn_id"] == "client-seam-key"
    assert "text" not in recorded  # Trace metadata must not copy the request body.


def test_resolved_preference_candidate_flows_to_triage_and_applies_only_validated_dimension() -> None:
    """A WP2 candidate reaches WP1 and only its need-approved dimension is applied."""
    class PreferenceProvider:
        def search(self, *, need, state):
            if "analysis_method" not in (need.get("memory_types") or []):
                return []
            return [{
                "candidate_id": "pref-analysis-1",
                "memory_key": "analysis_method",
                "memory_type": "analysis_method",
                "value": {"metric": ["gmv"], "dimension": "product_category"},
                "confidence": 0.9,
                "source_ref": "registry-pref-1",
            }]

    state = compile_turn_context({
        "user_message": "按我平时的方式分析",
        "previous_turn_context": _prior(),
        "current_query_scope_fingerprint": "scope-a",
        "scope_type": "platform",
        "preference_provider": PreferenceProvider(),
        "persistence_turn_id": "durable-pref-turn",
        "node_history": [],
    })

    triage = ContextOrchestrator().build_triage_context(
        state, instruction="按我平时的方式分析", model="test-model",
    )
    plan = triage.candidate_analysis_plan

    assert triage.preference_source_needs[0]["trigger_ids"] == ["P1"]
    assert plan["status"] == "applied"
    assert {step["field"] for step in plan["steps"]} == {"dimension"}
    assert plan["applied"] is True
    assert plan["requires_server_application"] is False
    assert triage.metadata()["candidate_analysis_plan"]["step_fields"] == ["dimension"]
    assert any(
        item.source.value == "mem0" and item.trust_tier.value == "preference"
        for item in triage.bundle.items
    )
    # Only the registry-validated dimension is applied; the preference cannot
    # replace the inherited metric or widen the authorization scope.
    assert state["context_compilation"]["request_contract"].get("dimension") == "product_category"
    assert state["context_compilation"]["request_contract"].get("metric_id") == "gmv"
    assert state["context_compilation"]["request_contract"].get("preference_application", {}).get("fields") == ["dimension"]
    assert state["context_compilation"]["known_slots"]["dimension"]["source"] == "validated_preference_candidate"
    assert state["context_compilation"]["known_slots"]["metric"]["source"] == "previous_turn_context"


def test_applied_preference_dimension_reaches_query_planner() -> None:
    from app.agents.data_query_agent import plan_confirmed_metric_queries

    class PreferenceProvider:
        def search(self, *, need, state):
            if "analysis_method" not in (need.get("memory_types") or []):
                return []
            return [{
                "memory_key": "analysis_method", "memory_type": "analysis_method",
                "value": {"dimension": "product_category"}, "confidence": 0.9,
                "source_ref": "registry-pref-planner",
            }]

    state = compile_turn_context({
        "user_message": "按我平时的方式分析 GMV",
        "previous_turn_context": _prior(),
        "current_query_scope_fingerprint": "scope-a",
        "preference_provider": PreferenceProvider(),
        "node_history": [],
    })
    contract = state["context_compilation"]["request_contract"]
    assert contract.get("dimension") == "product_category"
    plan = plan_confirmed_metric_queries(
        "按我平时的方式分析 GMV", scope="platform", allowed_metrics={"gmv"},
        request_contract=contract,
        scope_data={"seller_id": "seller-1", "start_date": "2018-08-01", "end_date": "2018-08-31"},
    )
    assert plan.decision.query_type == "metric_breakdown"
    assert plan.decision.dimension == "product_category"
    assert all(item.dimension == "product_category" for item in plan.plans)


def test_preference_candidate_does_not_apply_without_a_resolved_metric() -> None:
    class PreferenceProvider:
        def search(self, *, need, state):
            return [{
                "memory_key": "analysis_method", "memory_type": "analysis_method",
                "value": {"dimension": "product_category"}, "confidence": 0.9,
                "source_ref": "registry-pref-no-metric",
            }]

    state = compile_turn_context({
        "user_message": "按我平时的方式分析",
        "current_query_scope_fingerprint": "scope-a",
        "controlled_analysis_scope": {
            "allowed_metrics": ["gmv"],
            "allowed_dimensions": ["product_category"],
        },
        "preference_provider": PreferenceProvider(),
        "node_history": [],
    })
    plan = state["preference_context"]["candidate_analysis_plan"]
    assert state["context_compilation"]["request_contract"].get("dimension") is None
    assert plan["applied"] is False
    assert plan["status"] == "not_applied_metric_required"
    assert plan["steps"][0]["application_reason"] == "metric_required_before_preference_application"


def test_validated_preference_dimension_reaches_executable_breakdown_plan(monkeypatch) -> None:
    from app.agents.data_query_agent import plan_confirmed_metric_queries
    from app.context.request_contract import validate_compiled_request_contract

    class PreferenceProvider:
        def search(self, *, need, state):
            return [{
                "memory_key": "analysis_method", "memory_type": "analysis_method",
                "value": {"dimension": "product_category"}, "confidence": 0.93,
                "source_ref": "registry-pref-breakdown",
            }]

    question = "按我平时的分析方式查询 2018-08-01 的 GMV"
    state = compile_turn_context({
        "user_message": question,
        "current_query_scope_fingerprint": "scope-a",
        "scope_type": "platform",
        "preference_provider": PreferenceProvider(),
        "node_history": [],
    })
    compilation = state["context_compilation"]
    assert compilation["request_contract"].get("dimension") == "product_category"
    assert validate_compiled_request_contract(
        compilation,
        state_contract_hash=state["request_contract_hash"],
        require_state_anchor=True,
    ) == (True, None)

    batch = plan_confirmed_metric_queries(
        question,
        scope="platform",
        allowed_metrics={"gmv"},
        scope_data={"tenant_id": "t1", "user_id": "u1", "permissions": "admin", "scope_type": "platform"},
        operation_intent=compilation["operation_intent"],
        request_contract=compilation["request_contract"],
    )

    assert batch.decision.status == "confirmed"
    assert batch.decision.query_type == "metric_breakdown"
    assert batch.decision.dimension == "product_category"
    assert len(batch.plans) == 1
    plan = batch.plans[0]
    assert plan.query_type == "metric_breakdown"
    assert plan.dimension == "product_category"
    assert plan.metric_name == "gmv"
    assert plan.metric_date == "2018-08-01"
    assert plan.tool_name == "breakdown_by_dimension"
    assert plan.authorized_metric is not None
    assert plan.authorized_metric.metric_id == "gmv"

    # Cross the real query gateway with only the Registry executor replaced:
    # the captured call is the actual operation/argument contract sent to it.
    from app.metrics.registry import MetricRegistry
    from app.mcp.client import query_business_data
    from app.tenancy import TenantContext, activate_tenant_context
    from types import SimpleNamespace

    execution_calls = []

    def fake_execute(_registry, metric_id, **kwargs):
        execution_calls.append({"metric_id": metric_id, **kwargs})
        return {"status": "ok", "items": [], "source_table": "fixture"}

    monkeypatch.setattr(MetricRegistry, "execute", fake_execute)
    monkeypatch.setattr("app.mcp.client.resolve_expression", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "app.auth.context.build_authorization_context",
        lambda *_args, **_kwargs: SimpleNamespace(
            authz_digest="authz-test",
            permission_version="permission-v1",
            policy_set_version="policy-v1",
            authz_ref="authz-test-ref",
            require_action=lambda *_args, **_kwargs: None,
            require_metrics=lambda *_args, **_kwargs: None,
            require_dimensions=lambda *_args, **_kwargs: None,
            require_tool=lambda *_args, **_kwargs: None,
        ),
    )
    with activate_tenant_context(TenantContext(
        tenant_id="t1", user_id="u1", role="admin", scope_type="platform",
    )):
        result = query_business_data(
            "metric_query", question, tenant_id="t1", role="admin", plan=plan,
        )

    assert result["result"]["status"] == "ok", result["result"].get("reason")
    assert result["tool_name"] == "registry_breakdown"
    assert result["arguments"]["dimension"] == "product_category"
    assert result["arguments"]["metric_date"] == "2018-08-01"
    assert execution_calls == [{
        "metric_id": "gmv",
        "operation": "breakdown",
        "metric_date": date(2018, 8, 1),
        "start_date": date(2018, 8, 1),
        "end_date": date(2018, 8, 1),
        "left_date": None,
        "right_date": None,
        "granularity": "day",
        "dimension": "product_category",
        "limit": 10,
    }]


def test_preference_dimension_still_fails_closed_at_metric_registry_boundary() -> None:
    from app.agents.data_query_agent import plan_confirmed_metric_queries
    from app.context.request_contract import seal_request_contract

    question = "按我平时的分析方式查询 2018-08-01 的 GMV"
    state = compile_turn_context({
        "user_message": question,
        "current_query_scope_fingerprint": "scope-a",
        "controlled_analysis_scope": {
            "seller_id": "seller-1",
            "allowed_metrics": ["gmv"],
            "allowed_dimensions": ["seller_id"],
        },
        "node_history": [],
    })
    compilation = state["context_compilation"]
    # Simulate stale/misconfigured server-side scope metadata. The planner must
    # still validate the candidate against the active metric Registry.
    contract = dict(compilation["request_contract"])
    contract["dimension"] = "seller_id"
    contract["preference_application"] = {
        "source": "validated_mem0_candidate",
        "candidate_ids": ["candidate-1"],
        "fields": ["dimension"],
    }
    contract = seal_request_contract(contract)

    batch = plan_confirmed_metric_queries(
        question,
        scope="platform",
        allowed_metrics={"gmv"},
        scope_data={"tenant_id": "t1", "user_id": "u1", "scope_type": "platform"},
        operation_intent=compilation["operation_intent"],
        request_contract=contract,
    )

    assert batch.decision.status == "unsupported"
    assert batch.plans == []
    assert "不支持按" in batch.decision.reason


def test_candidate_dimension_is_not_applied_when_selected_scope_lacks_breakdown() -> None:
    from app.context.turn_resolver import _dimension_allowed

    allowed, reason = _dimension_allowed("product_category", {"gmv"}, {
        "controlled_analysis_scope": {
            "seller_id": "seller-1",
            "allowed_metrics": ["gmv"],
            "allowed_dimensions": ["product_category"],
        },
    })
    assert allowed is False
    assert reason == "dimension_not_supported_by_metric"


def test_explicit_dimension_wins_over_analysis_preference() -> None:
    calls = []

    class PreferenceProvider:
        def search(self, *, need, state):
            calls.append(need)
            return [{
                "memory_key": "analysis_method", "memory_type": "analysis_method",
                "value": {"dimension": "payment_type"}, "confidence": 0.9,
                "source_ref": "registry-pref-dimension",
            }]

    state = compile_turn_context({
        "user_message": "按品类拆开，并按我平时的方式分析 GMV",
        "previous_turn_context": _prior(),
        "current_query_scope_fingerprint": "scope-a",
        "controlled_analysis_scope": {
            "allowed_metrics": ["gmv"],
            "allowed_dimensions": ["product_category", "payment_type"],
        },
        "preference_provider": PreferenceProvider(),
        "node_history": [],
    })
    compilation = state["context_compilation"]
    assert compilation["known_slots"]["dimension"]["value"] == "product_category"
    assert compilation["known_slots"]["dimension"]["source"] == "current_turn"
    assert compilation["request_contract"]["dimension"] == "product_category"
    assert "preference_application" not in compilation["request_contract"]
    assert calls == []
    assert state["preference_context"]["status"] == "not_requested"

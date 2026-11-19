from app.context.result_context import (
    build_result_context,
    build_result_followup_answer,
    detect_result_followup,
    safe_result_context,
)
from app.graph.nodes.answer import _make_facts_from_tool


def _breakdown_result() -> dict:
    return {
        "trace_id": "trace-prior",
        "persistence_turn_id": "turn-prior",
        "current_query_scope_fingerprint": "scope-a",
        "data_version_manifest_hash": "v1",
        "query_results": [{
            "evidence_id": "ev-prior",
            "result": {
                "breakdown_result": {
                    "dimension": "product_category",
                    "metric_name": "gmv",
                    "items": [
                        {"dimension_value": "books", "value": 30.0},
                        {"dimension_value": "home", "value": 50.0},
                        {"dimension_value": "health", "value": 20.0},
                    ],
                },
            },
        }],
        "analysis_presentation": {
            "status": "completed",
            "semantic_interpretation": {"metric_id": "gmv"},
            "facts": [],
            "changes": [],
        },
    }


def _comparison_result() -> dict:
    return {
        "trace_id": "trace-compare",
        "persistence_turn_id": "turn-compare",
        "current_query_scope_fingerprint": "scope-a",
        "data_version_manifest_hash": "v1",
        "query_results": [{"evidence_id": "ev-compare", "result": {}}],
        "analysis_presentation": {
            "status": "completed",
            "semantic_interpretation": {"metric_id": "gmv"},
            "facts": [],
            "changes": [{
                "metric_name": "gmv",
                "baseline_value": 100.0,
                "current_value": 120.0,
                "delta_value": 20.0,
                "delta_ratio": 20.0,
                "unit": "BRL",
                "baseline_period": "2018-07-01 至 2018-07-31",
                "current_period": "2018-08-01 至 2018-08-31",
                "evidence_ids": ["ev-compare"],
            }],
        },
    }


def _value_result() -> dict:
    return {
        "trace_id": "trace-value",
        "persistence_turn_id": "turn-value",
        "current_query_scope_fingerprint": "scope-a",
        "data_version_manifest_hash": "v1",
        "query_results": [{"evidence_id": "ev-value", "result": {}}],
        "analysis_presentation": {
            "status": "completed",
            "semantic_interpretation": {"metric_id": "gmv"},
            "facts": [{
                "metric_name": "gmv",
                "display_name": "GMV",
                "value": 16220.0,
                "unit": "BRL",
                "date_or_range": "2018-08-20",
                "evidence_ids": ["ev-value"],
            }],
            "changes": [],
        },
    }


def _multi_metric_breakdown_result() -> dict:
    result = _breakdown_result()
    result["query_results"] = [
        {
            "evidence_id": "ev-gmv",
            "arguments": {"metric_name": "gmv", "query_type": "metric_breakdown"},
            "result": {"breakdown_result": {
                "metric_name": "gmv",
                "dimension": "product_category",
                "unit": "BRL",
                "items": [
                    {"dimension_value": "A", "value": 100.0},
                    {"dimension_value": "B", "value": 50.0},
                ],
            }},
        },
        {
            "evidence_id": "ev-orders",
            "arguments": {"metric_name": "order_count", "query_type": "metric_breakdown"},
            "result": {"breakdown_result": {
                "metric_name": "order_count",
                "dimension": "product_category",
                "unit": "orders",
                "items": [
                    {"dimension_value": "A", "value": 3.0},
                    {"dimension_value": "B", "value": 200.0},
                ],
            }},
        },
    ]
    result["analysis_presentation"]["semantic_interpretation"]["metric_id"] = "gmv"
    return result


def _no_data_result() -> dict:
    return {
        "trace_id": "trace-no-data",
        "persistence_turn_id": "turn-no-data",
        "current_query_scope_fingerprint": "scope-a",
        "data_version_manifest_hash": "v1",
        "query_results": [{
            "evidence_id": "ev-no-data",
            "arguments": {
                "metric_name": "gmv",
                "query_type": "metric_value",
                "metric_date": "2018-10-30",
            },
            "result": {
                "status": "not_found",
                "reason_code": "raw_reason_must_not_be_persisted",
                "reason": "arbitrary source message must not be persisted",
                "parsed": {
                    "metric_name": "gmv",
                    "query_type": "metric_value",
                    "metric_date": "2018-10-30",
                },
            },
        }],
        "analysis_presentation": {
            "status": "completed",
            "semantic_interpretation": {"metric_id": "gmv"},
            "facts": [],
            "changes": [],
        },
    }


def _state(context: dict, kind: str, **extra) -> dict:
    return {
        "trace_id": "trace-followup",
        "context_compilation": {
            "result_followup": {"kind": kind, **extra},
        },
        "result_followup_context": context,
    }


def test_build_result_context_keeps_bounded_rows_and_scope() -> None:
    context = build_result_context(_breakdown_result())

    assert context is not None
    assert context["version"] == "result-context-v2"
    assert context["metric_id"] == "gmv"
    assert context["dimension"] == "product_category"
    assert [row["key"] for row in context["rows"]] == ["books", "home", "health"]
    assert context["scope_fingerprint"] == "scope-a"
    assert context["source_trace_id"] == "trace-prior"
    assert "question" not in context


def test_build_and_validate_no_data_context_keeps_only_safe_status_and_period() -> None:
    context = build_result_context(_no_data_result())

    assert context is not None
    assert context["metric_id"] == "gmv"
    assert context["query_type"] == "metric_value"
    assert context["result_status"] == "not_found"
    assert context["result_reason"] == "data_not_found"
    assert context["query_period"] == {"metric_date": "2018-10-30"}
    assert context["evidence_ids"] == []
    assert "raw_reason_must_not_be_persisted" not in repr(context)
    assert "arbitrary source message" not in repr(context)

    safe = safe_result_context(context)
    assert safe is not None
    assert safe["result_status"] == "not_found"
    assert safe["query_period"] == {"metric_date": "2018-10-30"}


def test_detect_no_data_next_step_requires_a_current_same_scope_snapshot() -> None:
    context = build_result_context(_no_data_result())
    assert context is not None
    previous = {
        "source_turn_id": "turn-no-data",
        "scope_fingerprint": "scope-a",
        "result_context": context,
    }

    spec, trusted, reason = detect_result_followup("那下一步怎么办？", previous, {
        "current_query_scope_fingerprint": "scope-a",
        "data_version_manifest_hash": "v1",
    })
    assert spec == {"kind": "no_data_next_step"}
    assert reason is None
    assert trusted is not None
    assert trusted["result_status"] == "not_found"

    spec, trusted, reason = detect_result_followup("那下一步怎么办？", previous, {
        "current_query_scope_fingerprint": "scope-b",
        "data_version_manifest_hash": "v1",
    })
    assert spec == {"kind": "no_data_next_step"}
    assert trusted is None
    assert reason == "previous_no_data_context_unavailable"

    spec, trusted, reason = detect_result_followup("那下一步怎么办？", None, {
        "current_query_scope_fingerprint": "scope-a",
        "data_version_manifest_hash": "v1",
    })
    assert spec == {"kind": "no_data_next_step"}
    assert trusted is None
    assert reason == "previous_no_data_context_unavailable"


def test_safe_result_context_rejects_legacy_and_malformed_rows() -> None:
    assert safe_result_context({
        "version": "result-context-v1",
        "metric_id": "gmv",
        "rows": [{"key": "books", "value": 10}],
        "result_status": "completed",
    }) is None

    context = build_result_context(_breakdown_result())
    assert context is not None
    assert safe_result_context(context) is not None
    context["rows"][0]["value"] = "not-a-number"
    assert safe_result_context(context) is None


def test_detect_result_followup_for_extremum_and_top_n() -> None:
    previous = {"source_turn_id": "turn-prior", "scope_fingerprint": "scope-a", "result_context": build_result_context(_breakdown_result())}
    state = {
        "current_query_scope_fingerprint": "scope-a",
        "data_version_manifest_hash": "v1",
    }

    extremum, context, reason = detect_result_followup("哪个品类最高？", previous, state)
    assert reason is None
    assert extremum == {"kind": "extremum", "direction": "max"}
    assert context is not None

    top_n, context, reason = detect_result_followup("只看 Top 3。", previous, state)
    assert reason is None
    assert top_n == {"kind": "top_n", "n": 3}
    assert context is not None


def test_detect_result_followup_for_comparison_and_provenance() -> None:
    state = {
        "current_query_scope_fingerprint": "scope-a",
        "data_version_manifest_hash": "v1",
    }
    comparison_previous = {"source_turn_id": "turn-compare", "scope_fingerprint": "scope-a", "result_context": build_result_context(_comparison_result())}
    direction, context, reason = detect_result_followup("哪个月更高？", comparison_previous, state)
    assert reason is None
    assert direction == {"kind": "comparison_direction"}
    assert context and context["comparison"]["current_value"] == 120.0

    value_previous = {"source_turn_id": "turn-value", "scope_fingerprint": "scope-a", "result_context": build_result_context(_value_result())}
    provenance, context, reason = detect_result_followup("这个数字的证据是什么？", value_previous, state)
    assert reason is None
    assert provenance == {"kind": "provenance"}
    assert context and context["facts"][0]["value"] == 16220.0


def test_detect_result_followup_fails_closed_on_scope_change() -> None:
    previous = {"source_turn_id": "turn-prior", "scope_fingerprint": "scope-a", "result_context": build_result_context(_breakdown_result())}
    state = {
        "current_query_scope_fingerprint": "scope-b",
        "data_version_manifest_hash": "v1",
    }

    spec, context, reason = detect_result_followup("哪个品类最高？", previous, state)
    assert spec == {"kind": "extremum", "direction": "max"}
    assert context is None
    assert reason == "previous_result_context_unavailable"


def test_multi_metric_result_followup_requires_metric_and_keeps_evidence_units_isolated() -> None:
    result_context = build_result_context(_multi_metric_breakdown_result())
    assert result_context is not None
    assert result_context["metric_id"] == ""
    assert result_context["metric_ids"] == ["gmv", "order_count"]
    assert {row["metric_id"] for row in result_context["rows"]} == {"gmv", "order_count"}
    previous = {
        "source_turn_id": "turn-prior",
        "scope_fingerprint": "scope-a",
        "result_context": result_context,
    }
    state = {
        "current_query_scope_fingerprint": "scope-a",
        "data_version_manifest_hash": "v1",
    }

    spec, context, reason = detect_result_followup("哪个品类最高？", previous, state)
    assert spec == {"kind": "extremum", "direction": "max"}
    assert context is None
    assert reason == "previous_result_metric_ambiguous"

    spec, gmv_context, reason = detect_result_followup("哪个品类的GMV最高？", previous, state)
    assert spec == {"kind": "extremum", "direction": "max"}
    assert reason is None
    assert gmv_context is not None and gmv_context["metric_id"] == "gmv"
    gmv_answer = build_result_followup_answer(_state(gmv_context, "extremum", direction="max"))
    assert "A（100 BRL）" in gmv_answer["answer"]
    assert all(
        record["content"]["metric_id"] == "gmv"
        and record["content"]["source_evidence_ids"] == ["ev-gmv"]
        for record in gmv_answer["evidence_bundles"]
    )

    spec, order_context, reason = detect_result_followup("哪个品类的订单数最高？", previous, state)
    assert spec == {"kind": "extremum", "direction": "max"}
    assert reason is None
    assert order_context is not None and order_context["metric_id"] == "order_count"
    order_answer = build_result_followup_answer(_state(order_context, "extremum", direction="max"))
    assert "B（200 orders）" in order_answer["answer"]
    assert all(
        record["content"]["metric_id"] == "order_count"
        and record["content"]["source_evidence_ids"] == ["ev-orders"]
        for record in order_answer["evidence_bundles"]
    )


def test_answer_from_extremum_and_top_n_is_bounded() -> None:
    context = build_result_context(_breakdown_result())
    assert context is not None
    home_evidence_id = next(row["evidence_ids"][0] for row in context["rows"] if row["key"] == "home")

    extremum = build_result_followup_answer(_state(context, "extremum", direction="max"))
    assert extremum["analysis_presentation"]["status"] == "completed"
    assert "home" in extremum["answer"]
    assert "商品品类" in extremum["answer"]
    assert extremum["analysis_presentation"]["facts"][0]["evidence_ids"] == [home_evidence_id]
    assert extremum["evidence_bundles"][0]["source_type"] == "result_context"
    assert extremum["evidence_bundles"][0]["content"]["source_evidence_ids"] == ["ev-prior"]
    assert extremum["claims"]

    top_n = build_result_followup_answer(_state(context, "top_n", n=2))
    assert "home" in top_n["answer"] and "books" in top_n["answer"]
    assert "health" not in top_n["answer"]


def test_answer_from_comparison_and_provenance() -> None:
    comparison_context = build_result_context(_comparison_result())
    assert comparison_context is not None
    comparison = build_result_followup_answer(_state(comparison_context, "comparison_direction"))
    assert "2018-08-01 至 2018-08-31" in comparison["answer"]
    assert "前者更高" in comparison["answer"]
    assert comparison["analysis_presentation"]["changes"][0]["evidence_ids"] == comparison_context["comparison"]["evidence_ids"]

    value_context = build_result_context(_value_result())
    assert value_context is not None
    provenance = build_result_followup_answer(_state(value_context, "provenance"))
    assert "16220" in provenance["answer"]
    assert "证据" in provenance["answer"]
    assert "trace trace-value" in provenance["answer"]
    assert provenance["analysis_presentation"]["facts"][0]["value"] == 16220.0


def test_report_action_uses_same_scope_result_period_facts_and_real_evidence():
    previous_result = _breakdown_result()
    previous_result["query_results"][0]["arguments"] = {"metric_date": "2018-08-20"}
    context = build_result_context(previous_result)
    assert context is not None
    assert context["query_period"] == {"metric_date": "2018-08-20"}
    previous = {
        "source_turn_id": "turn-prior",
        "scope_fingerprint": "scope-a",
        "result_context": context,
    }
    state_scope = {
        "current_query_scope_fingerprint": "scope-a",
        "data_version_manifest_hash": "v1",
    }

    spec, trusted, reason = detect_result_followup(
        "把刚才结果整理成简短经营报告，并给一项可以验证的下一步建议。",
        previous,
        state_scope,
    )
    assert spec == {"kind": "report_action"}
    assert reason is None
    assert trusted is not None

    answered = build_result_followup_answer(_state(trusted, "report_action"))
    presentation = answered["analysis_presentation"]
    assert "简短经营摘要" in answered["answer"]
    assert "2018-08-20" in answered["answer"]
    assert "home" in answered["answer"]
    assert "行动建议" in answered["answer"]
    assert "不是完整经营报告" in answered["answer"]
    assert presentation["status"] == "completed"
    assert presentation["facts"]
    expected_evidence = {item for row in context["rows"] for item in row["evidence_ids"]}
    assert all(fact["evidence_ids"][0] in expected_evidence for fact in presentation["facts"])
    assert len(presentation["recommendations"]) == 1
    assert presentation["recommendations"][0]["evidence_ids"] == next(
        row["evidence_ids"] for row in context["rows"] if row["key"] == "home"
    )
    assert answered["claims"]
    assert all(set(claim["source_evidence_ids"]) <= expected_evidence for claim in answered["claims"])
    assert "因果" in " ".join(presentation["limitations"])


def test_report_action_fails_closed_without_same_scope_result_evidence():
    context = build_result_context(_breakdown_result())
    assert context is not None
    context["data_version_manifest_hash"] = "data-v1"
    previous = {"source_turn_id": "turn-prior", "scope_fingerprint": "scope-a", "result_context": context}
    spec, trusted, reason = detect_result_followup(
        "整理成简短经营报告。",
        previous,
        {"current_query_scope_fingerprint": "scope-b", "data_version_manifest_hash": "v1"},
    )
    assert spec == {"kind": "report_action"}
    assert trusted is None
    assert reason == "previous_result_context_unavailable"

    spec, trusted, reason = detect_result_followup(
        "整理成简短经营报告。",
        previous,
        {"current_query_scope_fingerprint": "scope-a", "data_version_manifest_hash": "data-v2"},
    )
    assert spec == {"kind": "report_action"}
    assert trusted is None
    assert reason == "previous_result_context_unavailable"


def test_no_data_metric_result_does_not_become_a_numeric_fact():
    facts, evidence_ids = _make_facts_from_tool(
        {
            "tool_name": "registry_query",
            "result": {
                "status": "not_found",
                "metric_result": {
                    "metric_name": "gmv",
                    "status": "not_found",
                    "metric_date": "2018-10-30",
                    "value": None,
                    "unit": "BRL",
                },
            },
        },
        [{"evidence_id": "ev-no-data", "source_type": "mcp", "source_ref": "registry_query"}],
    )

    assert facts == []
    assert evidence_ids == ["ev-no-data"]


def test_compiler_detects_result_followup_and_inherits_metric() -> None:
    from app.context.compiler import compile_turn_context

    previous = {
        "source_turn_id": "turn-prior",
        "scope_fingerprint": "scope-a",
        "context_slots": {"metric": {"value": "gmv", "source": "current_turn"}},
        "result_context": build_result_context(_breakdown_result()),
    }
    compilation = compile_turn_context({
        "user_message": "哪个品类最高？",
        "current_query_scope_fingerprint": "scope-a",
        "data_version_manifest_hash": "v1",
        "previous_turn_context": previous,
    })["context_compilation"]

    assert compilation["result_followup"]["kind"] == "extremum"
    assert compilation["needs_clarification"] is False
    assert compilation["relation"] == "continue"
    assert compilation["intent"] == "follow_up"
    assert compilation["known_slots"]["metric"]["value"] == "gmv"
    assert compilation["known_slots"]["dimension"]["value"] == "product_category"


def test_triage_routes_report_action_without_llm_classification(monkeypatch):
    from app.context.compiler import compile_turn_context
    from app.graph.nodes import triage as triage_module

    question = "把刚才结果整理成简短经营报告，并给一项可以验证的下一步建议。"
    context = build_result_context(_breakdown_result())
    assert context is not None
    previous = {
        "source_turn_id": "turn-prior",
        "scope_fingerprint": "scope-a",
        "context_slots": {"metric": {"value": "gmv", "source": "current_turn"}},
        "result_context": context,
    }
    compilation = compile_turn_context({
        "user_message": question,
        "current_query_scope_fingerprint": "scope-a",
        "data_version_manifest_hash": "v1",
        "previous_turn_context": previous,
    })["context_compilation"]
    assert compilation["result_followup"]["kind"] == "report_action"
    assert compilation["needs_clarification"] is False

    def fail_if_classified(*_args, **_kwargs):
        raise AssertionError("validated result follow-ups must not call the LLM classifier")

    monkeypatch.setattr(triage_module, "llm_classify", fail_if_classified)
    result = triage_module.triage({
        "user_message": question,
        "context_compilation": compilation,
        "previous_turn_context": previous,
        "current_query_scope_fingerprint": "scope-a",
        "data_version_manifest_hash": "v1",
        "errors": [],
        "node_history": [],
        "handoffs": [],
    })

    assert result["task_type"] == "metric_query"
    assert result["next_agent"] == "data_query"


def test_data_query_skips_business_query_for_result_followup() -> None:
    from app.graph.nodes.data_query import data_query

    context = build_result_context(_breakdown_result())
    assert context is not None
    state = {
        "user_message": "哪个品类最高？",
        "task_type": "metric_query",
        "trace_id": "trace-followup",
        "current_query_scope_fingerprint": "scope-a",
        "data_version_manifest_hash": "v1",
        "previous_turn_context": {
            "source_turn_id": "turn-prior",
            "scope_fingerprint": "scope-a",
            "result_context": context,
        },
        "context_compilation": {
            "result_followup": {"kind": "extremum", "direction": "max"},
            "request_contract": None,
        },
        "runtime_budget": {},
        "tool_calls": [],
        "query_results": [],
        "node_history": [],
    }

    result = data_query(state)

    assert result["query_results"] == []
    assert result["tool_calls"] == []
    assert result["result_followup_context"]["metric_id"] == "gmv"
    assert result["metric_planning_decision"]["status"] == "confirmed"


def test_data_query_skips_business_query_for_report_action_followup():
    from app.graph.nodes.data_query import data_query

    context = build_result_context(_breakdown_result())
    assert context is not None
    state = {
        "user_message": "把刚才结果整理成简短经营报告，并给一项可以验证的下一步建议。",
        "task_type": "metric_query",
        "trace_id": "trace-report-action",
        "current_query_scope_fingerprint": "scope-a",
        "data_version_manifest_hash": "v1",
        "previous_turn_context": {
            "source_turn_id": "turn-prior",
            "scope_fingerprint": "scope-a",
            "result_context": context,
        },
        "context_compilation": {
            "result_followup": {"kind": "report_action"},
            "request_contract": None,
        },
        "runtime_budget": {},
        "tool_calls": [],
        "query_results": [],
        "node_history": [],
    }

    result = data_query(state)

    assert result["query_results"] == []
    assert result["tool_calls"] == []
    assert result["result_followup_context"]["metric_id"] == "gmv"
    assert result["result_followup"]["kind"] == "report_action"
    assert result["metric_planning_decision"]["status"] == "confirmed"


def test_no_data_next_step_answers_deterministically_without_facts_or_queries() -> None:
    from app.context.compiler import compile_turn_context
    from app.graph.nodes.data_query import data_query

    context = build_result_context(_no_data_result())
    assert context is not None
    previous = {
        "source_turn_id": "turn-no-data",
        "scope_fingerprint": "scope-a",
        "result_context": context,
    }
    compiled = compile_turn_context({
        "user_message": "那下一步怎么办？",
        "current_query_scope_fingerprint": "scope-a",
        "data_version_manifest_hash": "v1",
        "previous_turn_context": previous,
    })["context_compilation"]
    assert compiled["result_followup"]["kind"] == "no_data_next_step"
    assert compiled["needs_clarification"] is False

    queried = data_query({
        "user_message": "那下一步怎么办？",
        "task_type": "metric_query",
        "trace_id": "trace-next-step",
        "current_query_scope_fingerprint": "scope-a",
        "data_version_manifest_hash": "v1",
        "previous_turn_context": previous,
        # This unit isolates the data-query follow-up branch. The full graph's
        # planner signs/anchors the compiler contract before entering it.
        "context_compilation": {**compiled, "request_contract": None},
        "runtime_budget": {},
        "tool_calls": [],
        "query_results": [],
        "node_history": [],
    })
    assert queried["query_results"] == []
    assert queried["tool_calls"] == []
    assert queried["result_followup_context"]["result_status"] == "not_found"
    assert queried["metric_planning_decision"]["query_type"] == "metric_value"

    answered = build_result_followup_answer(queried)
    assert "2018-10-30" in answered["answer"]
    assert all(word in answered["answer"] for word in ("日期范围", "数据覆盖日期", "筛选范围"))
    assert "没有触发新查询" in answered["answer"]
    assert answered["analysis_presentation"]["facts"] == []
    assert answered["claims"] == []
    assert answered["evidence_bundles"] == []
    from app.schemas.events import SSE_EVENT_ADAPTER

    terminal = SSE_EVENT_ADAPTER.validate_python({
        "event": "completed",
        "task_type": "metric_query",
        "answer": answered["answer"],
        "trace_id": "trace-next-step",
        "presentation": answered["analysis_presentation"],
    })
    assert terminal.event == "completed"
    assert [item.text for item in terminal.presentation.recommendations] == [
        "扩大日期范围后重试",
        "确认数据源覆盖的日期",
        "检查指标和筛选范围",
    ]

import pytest

from app.context.compiler import compile_turn_context
from app.graph.nodes import triage as triage_module


def _compile(question: str):
    return compile_turn_context({
        "user_message": question,
        "tenant_id": "platform",
        "user_id": "offline-test",
        "user_role": "admin",
        "scope_type": "platform",
    })


def test_registered_calculation_comparison_is_not_misrouted_to_definition(monkeypatch):
    question = (
        "请按平台平均送达时长的注册口径比较2017-08-20与2018-10-18；"
        "若任一比较日期没有可用的已加载数据，请明确说明数据不足。"
    )
    state = _compile(question)
    compilation = state["context_compilation"]

    assert compilation["intent"] == "analyze_business_data"
    assert compilation["known_slots"]["operation"]["value"] == "compare"
    assert compilation["request_contract"]["comparison"] == {
        "kind": "day_pair",
        "baseline_date": "2017-08-20",
        "current_date": "2018-10-18",
    }
    monkeypatch.setattr(triage_module, "llm_classify", lambda *_args, **_kwargs: "metric_query")
    result = triage_module.triage(state)
    assert result["task_type"] == "metric_query"
    assert result["next_agent"] == "data_query"


def test_standalone_metric_calculation_contract_question_stays_definition(monkeypatch):
    question = "平台平均送达时长的注册口径是什么？"
    state = _compile(question)
    assert state["context_compilation"]["intent"] == "retrieve_metric_definition"
    monkeypatch.setattr(triage_module, "llm_classify", lambda *_args, **_kwargs: "metric_definition")
    result = triage_module.triage(state)
    assert result["task_type"] == "metric_definition"
    assert result["next_agent"] == "metric_definition"


@pytest.mark.parametrize(
    "question",
    [
        "GMV 是什么？",
        "列出我能访问的店铺",
        "生成本周经营分析报告",
    ],
)
def test_registered_task_classifier_owns_definition_catalog_and_report_routing(
    monkeypatch, question: str,
) -> None:
    calls: list[str] = []

    def classify(_system: str, packed_context: str, _allowed: list[str], **kwargs) -> str:
        calls.append(packed_context)
        assert kwargs == {
            "operation": "classify.task_type",
            "interaction_policy": "nonstream_fast",
        }
        return "metric_query"

    monkeypatch.setattr(triage_module, "llm_classify", classify)
    result = triage_module.triage({"user_message": question})

    assert calls
    assert result["task_type"] == "metric_query"
    assert result["next_agent"] == "data_query"

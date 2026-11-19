from __future__ import annotations

from app.analysis.templates import registry as template_registry
from app.graph.nodes.answer import _summarize_metric_query, generate_answer


def test_generate_answer_does_not_execute_template_with_locked_scope(monkeypatch) -> None:
    calls: list[str] = []

    def tracked_get_template(name: str):
        calls.append(name)
        raise RuntimeError("template execution must require a confirmed plan")

    monkeypatch.setattr(template_registry, "get_template", tracked_get_template)
    result = generate_answer({
        "task_type": "complex_analysis",
        "user_message": "请做 RFM 客户分群",
        "use_current_explorer_context": True,
        "controlled_analysis_scope": {
            "seller_id": "seller-1",
            "allowed_metrics": ["gmv"],
            "allowed_dimensions": [],
            "default_start_date": "2025-01-01",
            "default_end_date": "2025-01-31",
            "metric_name": "gmv",
            "granularity": "day",
        },
        "tool_result": {"result": {"status": "clarification_required", "parsed": {}}},
        "errors": [{"code": "token_budget_exceeded"}],
        "evidence_bundles": [],
        "analysis_memories": [],
        "node_history": [],
    })

    assert calls == []
    assert result["analysis_presentation"]["template_analyses"] == []
    assert "已完成白名单分析" not in result["answer"]
    assert "已在当前锁定范围内完成白名单分析" not in result["answer"]


def test_legacy_template_only_result_says_plan_confirmation_is_required() -> None:
    answer = _summarize_metric_query({
        "result": {
            "status": "ok",
            "template_only": True,
            "templates": ["rfm_segmentation"],
        },
    })

    assert "需要先创建并确认分析计划" in answer
    assert "已完成白名单分析" not in answer

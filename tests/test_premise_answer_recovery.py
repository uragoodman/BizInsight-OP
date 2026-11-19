import pytest

from app.graph.nodes.answer import _build_recommendations, _confirmed_premise_direction_text


def test_confirmed_premise_correction_renders_actual_direction() -> None:
    note = _confirmed_premise_direction_text({
        "premise_check": {
            "status": "confirmed_by_user",
            "claimed_direction": "down",
            "actual_direction": "up",
            "metric_name": "gmv",
            "metric_display_name": "成交总额",
            "baseline_date": "2018-08-19",
            "current_date": "2018-08-20",
        },
    })

    assert "成交总额" in note
    assert "实际上升" in note
    assert "并非下降" in note


def test_premise_direction_note_requires_confirmed_mismatch() -> None:
    assert _confirmed_premise_direction_text({
        "premise_check": {
            "status": "mismatch", "claimed_direction": "down", "actual_direction": "up",
            "metric_name": "gmv", "baseline_date": "2018-08-19", "current_date": "2018-08-20",
        },
    }) == ""
    assert _confirmed_premise_direction_text({
        "premise_check": {
            "status": "confirmed_by_user", "claimed_direction": "up", "actual_direction": "up",
            "metric_name": "gmv", "baseline_date": "2018-08-19", "current_date": "2018-08-20",
        },
    }) == ""


def test_optional_recommendation_context_budget_failure_does_not_fail_answer(monkeypatch) -> None:
    from app.context.packer import ContextBudgetExceeded

    class OverBudgetOrchestrator:
        def build_specialist_bundle(self, *_args, **_kwargs):
            raise ContextBudgetExceeded("optional recommendation context exceeds budget")

    monkeypatch.setattr("app.context.orchestrator.ContextOrchestrator", OverBudgetOrchestrator)
    monkeypatch.setattr("app.llm.routing.model_for", lambda _task: ("test-model", "test-provider"))
    monkeypatch.setattr("app.llm.client.llm_chat", lambda **_kwargs: pytest.fail("LLM must not run without a packed bundle"))

    recommendations = _build_recommendations(
        {"contribution": [{"dimension": "product_category", "top_contributor": "books"}]},
        ["ev_contribution"],
        {"evidence_refresh": {"evidence_ids": ["ev_contribution"]}},
    )

    assert recommendations == []

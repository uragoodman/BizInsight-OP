"""Registry-driven semantic metric confirmation tests."""

import pytest

from app.agents.data_query_agent import plan_confirmed_metric_queries
from app.graph.nodes.data_query import data_query
from app.graph.nodes.root_cause import root_cause
from app.graph.nodes.answer import generate_answer
from app.graph.nodes.critic import critic


@pytest.fixture(autouse=True)
def _disable_external_semantic_recall(monkeypatch):
    """Keep planner unit tests independent of Qdrant and local HF model startup."""
    monkeypatch.setattr("app.metrics.semantic_catalog.retrieve_metric_candidates", lambda *_args, **_kwargs: [])


def test_llm_confirmation_builds_multiple_registry_plans(monkeypatch) -> None:
    captured = {}

    def fake_extract(*_args, **kwargs):
        captured.update(kwargs)
        return {
            "status": "confirmed", "confirmed_metrics": ["gmv", "avg_order_value"],
            "ambiguous_candidates": [], "query_type": "metric_trend", "dimension": None,
            "granularity": "month", "reason": "明确提及两个指标", "suggestions": [],
        }

    monkeypatch.setattr("app.llm.client.llm_extract", fake_extract)
    batch = plan_confirmed_metric_queries("看 GMV 和客单价趋势")
    assert batch.decision.status == "confirmed"
    assert [item.metric_name for item in batch.plans] == ["gmv", "avg_order_value"]
    assert all(item.tool_name is None for item in batch.plans)
    assert captured["operation"] == "extract.data_query_plan"
    assert captured["interaction_policy"] == "nonstream_fast"


def test_catalog_exposes_yaml_planning_metadata_without_execution_details() -> None:
    from app.agents.data_query_agent import build_planning_catalog

    entry = next(item for item in build_planning_catalog() if item["metric_id"] == "gmv")
    assert entry["examples"]
    assert entry["limitations"]
    assert "handler_key" not in entry and "sql" not in entry


def test_llm_scope_escape_becomes_clarification(monkeypatch) -> None:
    monkeypatch.setattr("app.llm.client.llm_extract", lambda *_args, **_kwargs: {
        "status": "confirmed", "confirmed_metrics": ["gmv", "order_count"],
        "ambiguous_candidates": [], "query_type": "metric_value", "reason": "", "suggestions": [],
    })
    batch = plan_confirmed_metric_queries("查询 GMV 和订单量", scope="seller", allowed_metrics={"gmv"})
    assert batch.decision.status == "clarification_required"
    assert not batch.plans


def test_llm_unavailable_allows_only_registered_exact_phrase(monkeypatch) -> None:
    monkeypatch.setattr("app.llm.client.llm_extract", lambda *_args, **_kwargs: None)
    assert plan_confirmed_metric_queries("查询 GMV").decision.status == "confirmed"
    assert plan_confirmed_metric_queries("看看销售表现").decision.status == "clarification_required"


def test_high_confidence_semantic_mapping_still_requires_confirmation(monkeypatch) -> None:
    monkeypatch.setattr("app.llm.client.llm_extract", lambda *_args, **_kwargs: {
        "status": "confirmed",
        "confirmed_metrics": ["gmv"],
        "ambiguous_candidates": [],
        "query_type": "metric_breakdown",
        "dimension": "product_category",
        "confidence": 0.94,
        "semantic_interpretation": {
            "source_text": "销售表现较低的商品",
            "metric_id": "gmv",
            "display_name": "成交总额",
            "dimension": "product_category",
            "dimension_label": "商品品类",
            "interpretation": "比较各商品品类的成交总额",
            "confidence": 0.94,
            "reason": "问题要求比较商品表现",
            "requires_confirmation": False,
        },
    })
    batch = plan_confirmed_metric_queries("找出销售表现较低的商品", scope="platform")
    # The model's confidence is informational only.  A non-exact mapping
    # cannot create an executable plan until the user confirms it.
    assert batch.decision.status == "clarification_required"
    assert batch.decision.reason_code == "invalid_llm_output"
    assert not batch.plans


def test_low_confidence_semantic_mapping_requires_confirmation(monkeypatch) -> None:
    monkeypatch.setattr("app.llm.client.llm_extract", lambda *_args, **_kwargs: {
        "status": "confirmed", "confirmed_metrics": ["review_score_avg"],
        "ambiguous_candidates": [], "query_type": "metric_value",
        "confidence": 0.72, "reason": "可能是评分指标", "suggestions": [],
    })
    batch = plan_confirmed_metric_queries("低分的评价", scope="platform")
    assert batch.decision.status == "clarification_required"
    assert batch.decision.reason_code == "invalid_llm_output"
    assert not batch.plans


def test_exact_review_metric_with_unsupported_product_dimension_does_not_query() -> None:
    # ``评分`` is only a semantic term; use the registered exact alias so this
    # test reaches deterministic capability validation without asking the LLM
    # to authorize a fuzzy metric mapping.
    batch = plan_confirmed_metric_queries("本月平均评分最低的商品有哪些", scope="seller")
    assert batch.decision.status == "unsupported"
    assert batch.decision.reason_code == "unsupported_dimension"
    assert "不支持按商品品类拆分" in batch.decision.reason
    assert not batch.plans


def test_answer_and_presentation_keep_semantic_interpretation() -> None:
    state = generate_answer({
        "task_type": "metric_query",
        "user_message": "判断低分评价的原因",
        "metric_planning_decision": {
            "status": "unsupported",
            "confirmed_metrics": ["review_score_avg"],
            "reason": "当前不支持按商品品类拆分",
            "reason_code": "unsupported_dimension",
            "semantic_interpretation": {
                "source_text": "低分的评价",
                "metric_id": "review_score_avg",
                "display_name": "平均评分",
                "dimension": "product_category",
                "dimension_label": "商品品类",
                "interpretation": "比较商品品类的平均评分",
                "confidence": 0.92,
                "reason": "问题关注评价分数",
                "requires_confirmation": False,
            },
        },
        "tool_result": {
            "result": {
                "status": "unsupported",
                "clarification": {"reason": "当前不支持按商品品类拆分"},
            }
        },
        "evidence_bundles": [], "analysis_memories": [], "node_history": [],
    })
    assert "本次分析口径" in state["answer"]
    assert state["analysis_presentation"]["semantic_interpretation"]["metric_id"] == "review_score_avg"


def test_answer_presentation_includes_successful_multi_dimension_axes(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.graph.nodes.answer._grounded_narrative",
        lambda _state, _evidence, answer: (answer, "skipped_in_test"),
    )
    state = generate_answer({
        "task_type": "metric_query",
        "user_message": "按商品品类和客户州分析 GMV",
        "metric_planning_decision": {
            "status": "confirmed",
            "semantic_interpretation": {
                "source_text": "商品品类和客户州",
                "metric_id": "gmv",
                "display_name": "成交总额",
                "dimension": "product_category",
                "dimension_label": "商品品类",
                "interpretation": "按商品品类和客户州分析成交总额",
            },
        },
        "tool_result": {
            "result": {
                "status": "ok",
                "multi_dimension_breakdown_result": {
                    "status": "ok",
                    "dimensions": ["customer_state", "product_category"],
                    "items": [],
                },
            },
        },
        "evidence_bundles": [], "analysis_memories": [], "node_history": [],
    })

    card = state["analysis_presentation"]["semantic_interpretation"]
    assert card["dimension"] == "product_category"
    assert card["dimensions"] == ["customer_state", "product_category"]
    from app.schemas.analysis_presentation import AnalysisPresentation

    presentation = AnalysisPresentation.model_validate(state["analysis_presentation"])
    assert presentation.semantic_interpretation.dimensions == ["customer_state", "product_category"]


def test_exact_registry_phrase_overrides_overly_conservative_llm_clarification(monkeypatch) -> None:
    """Explicit registered metrics must not depend on one stochastic LLM verdict."""
    from app.agents.data_query_agent import build_planning_catalog

    monkeypatch.setattr("app.llm.client.llm_extract", lambda *_args, **_kwargs: {
        "status": "clarification_required", "confirmed_metrics": [],
        "ambiguous_candidates": [], "query_type": "metric_value",
        "reason": "model was overly conservative", "suggestions": [],
    })
    freight = next(item for item in build_planning_catalog(scope="seller") if item["metric_id"] == "freight_value")
    batch = plan_confirmed_metric_queries(f"{freight['phrases'][0]} compare", scope="seller")

    assert batch.decision.status == "confirmed"
    assert batch.decision.confirmed_metrics == ["freight_value"]
    assert batch.decision.query_type == "metric_compare"


def test_exact_metric_cannot_be_replaced_by_wrong_llm_metric(monkeypatch) -> None:
    """An exact user phrase must win even if the model returns another ID."""
    monkeypatch.setattr("app.llm.client.llm_extract", lambda *_args, **_kwargs: {
        "status": "confirmed", "confirmed_metrics": ["avg_approval_lead_hours"],
        "ambiguous_candidates": [], "query_type": "metric_trend",
        "reason": "incorrect model guess", "suggestions": [],
    })

    batch = plan_confirmed_metric_queries("查询 2018-08-10 的 GMV", scope="platform")

    assert batch.decision.status == "confirmed"
    assert batch.decision.confirmed_metrics == ["gmv"]
    assert batch.decision.query_type == "metric_value"
    assert batch.plans[0].metric_name == "gmv"
    assert batch.plans[0].metric_date == "2018-08-10"


def test_invalid_explicit_iso_date_is_clarified_without_query() -> None:
    batch = plan_confirmed_metric_queries("查询 2018-02-30 的 GMV", scope="platform")

    assert batch.decision.status == "clarification_required"
    assert "不是有效日历日期" in batch.decision.reason
    assert batch.plans == []


def test_mixed_exact_and_semantic_metrics_do_not_authorize_model_ids(monkeypatch) -> None:
    """One exact phrase must not authorize other IDs returned by the model."""
    from app.mcp.schemas import MetricConfirmationCandidate

    monkeypatch.setattr(
        "app.agents.data_query_agent._semantic_candidates_from_llm",
        lambda *_args, **_kwargs: [MetricConfirmationCandidate(
            candidate_no=0,
            metric_id="review_score_avg",
            display_name="平均评分",
            relation="related_but_different",
            reason="相关候选，需确认",
            evidence="低分评论",
            confidence=0.82,
        )],
    )
    monkeypatch.setattr(
        "app.llm.client.llm_extract",
        lambda *_args, **_kwargs: {
            "status": "confirmed",
            "confirmed_metrics": ["bad_review_rate", "review_score_avg"],
            "ambiguous_candidates": [],
            "query_type": "metric_trend",
            "reason": "模型返回了混合指标",
            "suggestions": [],
        },
    )

    batch = plan_confirmed_metric_queries(
        "比较本月和上月差评率，并分析低分评论的评价质量",
        scope="seller",
    )

    assert batch.decision.status == "clarification_required"
    assert batch.decision.reason_code == "semantic_confirmation_required"
    assert batch.decision.confirmed_metrics == ["bad_review_rate"]
    assert [item.metric_id for item in batch.decision.ambiguous_candidates] == ["review_score_avg"]
    assert batch.plans == []


def test_clarification_does_not_call_data_gateway(monkeypatch) -> None:
    monkeypatch.setattr("app.graph.nodes.data_query.plan_confirmed_metric_queries", lambda *_args, **_kwargs: __import__("app.mcp.schemas", fromlist=["MetricQueryBatchPlan", "MetricPlanningDecision"]).MetricQueryBatchPlan(
        decision=__import__("app.mcp.schemas", fromlist=["MetricPlanningDecision"]).MetricPlanningDecision(
            status="clarification_required", reason="指标名称不明确"
        )
    ))
    called = []
    monkeypatch.setattr("app.graph.nodes.data_query.query_business_data", lambda **_kwargs: called.append(True))
    state = data_query({"user_message": "查 GMVV", "trace_id": "planner-test"})
    assert not called
    assert state["tool_result"]["result"]["status"] == "clarification_required"


def test_clarification_is_not_rejected_as_an_unsupported_claim() -> None:
    state = generate_answer({
        "task_type": "metric_query", "tool_result": {
            "result": {"status": "clarification_required", "clarification": {"reason": "请确认指标"}}
        },
        "evidence_bundles": [], "analysis_memories": [], "node_history": [],
    })
    assert state["claims"] == []
    reviewed = critic(state)
    assert reviewed["critic_verdict"] == "approved"


def test_batch_root_cause_creates_one_track_per_metric() -> None:
    result = root_cause({
        "query_results": [
            {"result": {"parsed": {"metric_name": "gmv", "metric_date": "2018-08-10"}}},
            {"result": {"parsed": {"metric_name": "avg_order_value", "metric_date": "2018-08-10"}}},
        ],
        "tool_result": {}, "controlled_analysis_scope": {"end_date": "2018-08-10"},
        "node_history": [], "evidence_bundles": [],
    })
    assert len(result["root_cause_tracks"]) == 2
    assert all(track["max_followups"] == 5 for track in result["root_cause_tracks"])
    assert result["pending_query"]["metric_id"] == "gmv"


def test_batch_root_cause_uses_four_followup_budget_above_three_metrics() -> None:
    metric_ids = ["gmv", "avg_order_value", "order_count", "freight_value"]
    result = root_cause({
        "query_results": [{"result": {"parsed": {"metric_name": metric, "metric_date": "2018-08-10"}}} for metric in metric_ids],
        "tool_result": {}, "controlled_analysis_scope": {"end_date": "2018-08-10"},
        "node_history": [], "evidence_bundles": [],
    })
    assert all(track["max_followups"] == 4 for track in result["root_cause_tracks"])

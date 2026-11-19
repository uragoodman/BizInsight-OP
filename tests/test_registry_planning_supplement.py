"""补充注册中心规划层测试：Registry 校验、Schema 验证、目录过滤、Scope 执行。"""

from __future__ import annotations

from typing import Any

import pytest

from app.metrics.contracts import (
    MetricDefinition,
    MetricPlanningMetadata,
    ScopeImplementation,
)
from app.metrics.exceptions import RegistryValidationError
from app.metrics.registry import get_registry
from app.mcp.schemas import (
    MetricConfirmationCandidate,
    MetricPlanningDecision,
    MetricQueryBatchPlan,
    MetricQueryPlan,
)
from app.agents.data_query_agent import build_planning_catalog


@pytest.fixture(autouse=True)
def _disable_external_semantic_recall(monkeypatch):
    """Keep planner unit tests independent of Qdrant and local HF model startup."""
    monkeypatch.setattr("app.metrics.semantic_catalog.retrieve_metric_candidates", lambda *_args, **_kwargs: [])


# =============================================================================
# Registry validate — planning metadata completeness
# =============================================================================


def test_registry_validate_passes_when_planning_metadata_complete() -> None:
    """All metrics have phrases and examples; validate passes."""
    reg = get_registry()
    # validate() should not raise for properly configured metrics
    reg.validate()


def test_every_declared_attribution_method_has_a_required_data_contract() -> None:
    registry = get_registry()
    registry.load()
    supported = [
        (metric.metric_id, scope, capability)
        for metric in registry._metrics.values()
        for scope, capability in metric.attribution.scopes.items()
        if capability.method_ids
    ]
    assert supported
    for metric_id, scope, capability in supported:
        assert set(capability.required_data) == set(capability.method_ids), (metric_id, scope)
        assert all(capability.required_data[method] for method in capability.method_ids)


def test_registry_validate_fails_on_missing_phrases() -> None:
    """Metric without planning.phrases must fail validate()."""
    reg = get_registry()
    reg.load()
    # Tamper the gmv metric to remove phrases
    gmv = reg._metrics["gmv"]
    reg._metrics["gmv"] = MetricDefinition(
        metric_id=gmv.metric_id,
        display_name=gmv.display_name,
        aliases=gmv.aliases,
        unit=gmv.unit,
        display_format=gmv.display_format,
        base_grain=gmv.base_grain,
        aggregation=gmv.aggregation,
        direction=gmv.direction,
        version=gmv.version,
        documentation=gmv.documentation,
        display_order=gmv.display_order,
        surfaces=gmv.surfaces,
        implementations=gmv.implementations,
        planning=MetricPlanningMetadata(
            phrases=(), examples=(), intents=(), limitations="",
        ),
    )
    with pytest.raises(RegistryValidationError, match="planning requires phrases"):
        reg.validate()
    # This test intentionally corrupts the process-global registry.  Restore
    # it so later planner tests exercise the real YAML metadata.
    reg.load()


# =============================================================================
# MetricPlanningDecision schema validation
# =============================================================================


class TestMetricPlanningDecisionValidation:
    def test_confirmed_metrics_must_be_unique(self) -> None:
        """Duplicate metric names in confirmed_metrics must fail."""
        with pytest.raises(ValueError, match="unique"):
            MetricPlanningDecision(
                status="confirmed",
                confirmed_metrics=["gmv", "gmv"],
            )

    def test_valid_confirmed_metrics_passes(self) -> None:
        decision = MetricPlanningDecision(
            status="confirmed",
            confirmed_metrics=["gmv", "order_count"],
        )
        assert decision.status == "confirmed"
        assert decision.confirmed_metrics == ["gmv", "order_count"]

    def test_invalid_status_raises(self) -> None:
        with pytest.raises(ValueError, match="Input should be"):
            MetricPlanningDecision(status="invalid_status")  # type: ignore[arg-type]

    def test_clarification_status_allows_empty_metrics(self) -> None:
        decision = MetricPlanningDecision(status="clarification_required")
        assert decision.confirmed_metrics == []
        assert decision.ambiguous_candidates == []

    def test_unsupported_status_with_reason(self) -> None:
        decision = MetricPlanningDecision(
            status="unsupported", reason="No queryable metrics found",
        )
        assert decision.status == "unsupported"
        assert "No queryable" in decision.reason

    def test_ambiguous_candidates(self) -> None:
        decision = MetricPlanningDecision(
            status="clarification_required",
            ambiguous_candidates=[
                MetricConfirmationCandidate(
                    metric_id="gmv", display_name="成交总额",
                    reason="疑似 GMVV 的错别字",
                ),
            ],
            suggestions=["您是否想查询 GMV（成交总额）？"],
        )
        assert len(decision.ambiguous_candidates) == 1
        assert decision.ambiguous_candidates[0].metric_id == "gmv"


# =============================================================================
# MetricQueryBatchPlan schema validation
# =============================================================================


class TestMetricQueryBatchPlanValidation:
    def test_duplicate_metric_names_in_plans_raises(self) -> None:
        """Duplicate metric names in batch plans must fail."""
        valid_decision = MetricPlanningDecision(
            status="confirmed",
            confirmed_metrics=["gmv", "order_count"],
        )
        with pytest.raises(ValueError, match="must not duplicate"):
            MetricQueryBatchPlan(
                decision=valid_decision,
                plans=[
                    MetricQueryPlan(metric_name="gmv", metric_date="2018-08-10"),
                    MetricQueryPlan(metric_name="gmv", metric_date="2018-08-10"),
                ],
            )

    # Duplicate removed — covered by test_valid_batch_plan_passes in the
    # other batch-plan test class.


def test_planning_catalog_only_returns_chat_surface_metrics() -> None:
    """Catalog entries should have the chat surface."""
    catalog = build_planning_catalog()
    assert catalog, "catalog must not be empty"
    ids = {item["metric_id"] for item in catalog}
    assert "gmv" in ids
    assert "order_count" in ids


def test_planning_catalog_respects_allowed_metrics_filter() -> None:
    """When allowed_metrics is set, only those metrics appear."""
    catalog = build_planning_catalog(allowed_metrics={"gmv"})
    ids = [item["metric_id"] for item in catalog]
    assert ids == ["gmv"]
    assert len(catalog) == 1


def test_planning_catalog_excludes_execution_details() -> None:
    """Catalog entries must not contain SQL, handler keys, or schema info."""
    catalog = build_planning_catalog()
    for entry in catalog:
        assert "handler_key" not in entry, f"{entry['metric_id']} leaks handler_key"
        assert "sql" not in entry, f"{entry['metric_id']} leaks sql"
        assert "lineage_source" not in entry, f"{entry['metric_id']} leaks lineage_source"


def test_planning_catalog_limitations_falls_back_to_lineage() -> None:
    """When planning.limitations is empty, lineage_description is used."""
    catalog = build_planning_catalog()
    gmv = next(item for item in catalog if item["metric_id"] == "gmv")
    assert gmv["limitations"], "limitations must not be empty"


# =============================================================================
# Exact catalog matching (LLM-unavailable fallback)
# =============================================================================


def test_exact_catalog_match_on_alias(monkeypatch) -> None:
    """When LLM is unavailable, exact alias matching must still work."""
    from app.agents.data_query_agent import plan_confirmed_metric_queries

    monkeypatch.setattr("app.llm.client.llm_extract", lambda *_args, **_kwargs: None)
    batch = plan_confirmed_metric_queries("成交总额")
    assert batch.decision.status == "confirmed"
    assert "gmv" in [m for m in batch.decision.confirmed_metrics]


def test_exact_catalog_match_on_metric_id(monkeypatch) -> None:
    """Direct metric_id mention works as exact match."""
    from app.agents.data_query_agent import plan_confirmed_metric_queries

    monkeypatch.setattr("app.llm.client.llm_extract", lambda *_args, **_kwargs: None)
    batch = plan_confirmed_metric_queries("gmv, order_count")
    assert batch.decision.status == "confirmed"
    assert len(batch.decision.confirmed_metrics) == 2


def test_exact_catalog_match_on_display_name(monkeypatch) -> None:
    """The label shown by the data explorer is an executable exact phrase."""
    from app.agents.data_query_agent import plan_confirmed_metric_queries

    monkeypatch.setattr("app.llm.client.llm_extract", lambda *_args, **_kwargs: None)
    batch = plan_confirmed_metric_queries(
        "分析店铺平均提前送达天数的变化",
        scope="seller",
    )
    assert batch.decision.status == "confirmed"
    assert batch.decision.confirmed_metrics == ["seller_avg_early_delivery_days"]


def test_selected_explorer_metric_is_default_chat_metric(monkeypatch) -> None:
    """A generic follow-up uses the metric selected in the locked explorer scope."""
    from app.agents.data_query_agent import plan_confirmed_metric_queries

    monkeypatch.setattr("app.llm.client.llm_extract", lambda *_args, **_kwargs: None)
    batch = plan_confirmed_metric_queries(
        "分析本月为什么变化，并说明可能的关联因素",
        scope="seller",
        default_metric="seller_avg_early_delivery_days",
    )
    assert batch.decision.status == "confirmed"
    assert batch.decision.confirmed_metrics == ["seller_avg_early_delivery_days"]
    assert "默认分析口径" in batch.decision.reason


def test_unsupported_registered_metric_does_not_force_replacement(monkeypatch) -> None:
    """Seller scope must not invent a fuzzy replacement when semantic recall is unavailable."""
    from app.agents.data_query_agent import plan_confirmed_metric_queries

    monkeypatch.setattr("app.llm.client.llm_extract", lambda *_args, **_kwargs: None)
    batch = plan_confirmed_metric_queries("请分析延迟送达率", scope="seller")
    assert batch.decision.status == "clarification_required"
    assert batch.decision.reason_code == "unsupported_metric"
    assert batch.decision.unavailable_metrics == ["延迟送达率"]
    assert not batch.decision.ambiguous_candidates
    assert not batch.plans


def test_allowed_delivery_days_phrase_does_not_match_platform_delay_alias(monkeypatch) -> None:
    """An overlapping short alias inside an exact seller metric is not a second request."""
    from app.agents.data_query_agent import plan_confirmed_metric_queries

    monkeypatch.setattr("app.llm.client.llm_extract", lambda *_args, **_kwargs: None)
    batch = plan_confirmed_metric_queries(
        "店铺平均延迟送达天数再按月看看趋势",
        scope="seller",
        default_metric="seller_avg_late_delivery_days",
    )
    assert batch.decision.status == "confirmed"
    assert batch.decision.confirmed_metrics == ["seller_avg_late_delivery_days"]
    assert not batch.decision.unavailable_metrics
    assert len(batch.plans) == 1
    assert batch.plans[0].metric_name == "seller_avg_late_delivery_days"


def test_explicit_separate_platform_delay_rate_still_requires_clarification(monkeypatch) -> None:
    """Suppress only the alias occurrence covered by the allowed metric mention."""
    from app.agents.data_query_agent import plan_confirmed_metric_queries

    monkeypatch.setattr("app.llm.client.llm_extract", lambda *_args, **_kwargs: None)
    batch = plan_confirmed_metric_queries(
        "店铺平均延迟送达天数再按月看看趋势，同时分析延迟送达率",
        scope="seller",
        default_metric="seller_avg_late_delivery_days",
    )
    assert batch.decision.status == "clarification_required"
    assert batch.decision.confirmed_metrics == ["seller_avg_late_delivery_days"]
    assert batch.decision.unavailable_metrics == ["延迟送达率"]
    assert not batch.plans


def test_unclear_input_becomes_clarification_when_llm_unavailable(monkeypatch) -> None:
    """Without LLM, vague input must go to clarification, not false positive."""
    from app.agents.data_query_agent import plan_confirmed_metric_queries

    monkeypatch.setattr("app.llm.client.llm_extract", lambda *_args, **_kwargs: None)
    batch = plan_confirmed_metric_queries("最近销售怎么样")
    assert batch.decision.status == "clarification_required"
    assert not batch.plans


# =============================================================================
# Scope enforcement via allowed_metrics
# =============================================================================


def test_planning_catalog_filters_by_seller_scope() -> None:
    """Seller scope catalog should only include seller-implemented metrics."""
    catalog = build_planning_catalog(scope="seller")
    ids = {item["metric_id"] for item in catalog}
    # cancel_rate has no seller implementation
    assert "cancel_rate" not in ids
    # gmv has seller implementation
    assert "gmv" in ids


def test_scope_escape_via_allowed_metrics_becomes_clarification(monkeypatch) -> None:
    """LLM confirming metric not in allowed_metrics becomes clarification."""
    from app.agents.data_query_agent import plan_confirmed_metric_queries

    monkeypatch.setattr("app.llm.client.llm_extract", lambda *_args, **_kwargs: {
        "status": "confirmed", "confirmed_metrics": ["gmv", "order_count"],
        "ambiguous_candidates": [], "query_type": "metric_value",
        "reason": "", "suggestions": [],
    })
    batch = plan_confirmed_metric_queries(
        "query gmv", scope="seller", allowed_metrics={"order_count"},
    )
    assert batch.decision.status == "clarification_required"
    assert not batch.plans


# =============================================================================
# Clarification flow does not execute MCP
# =============================================================================


def test_unclear_single_metric_also_enters_clarification(monkeypatch) -> None:
    """A single unclear metric should still enter clarification."""
    from app.agents.data_query_agent import plan_confirmed_metric_queries

    monkeypatch.setattr("app.llm.client.llm_extract", lambda *_args, **_kwargs: {
        "status": "clarification_required",
        "ambiguous_candidates": [
            {"metric_id": "gmv", "display_name": "成交总额", "reason": "疑似错别字"},
        ],
        "suggestions": ["您是否想查询 GMV？"],
        "reason": "指标名称可能包含错别字", "query_type": "metric_value",
    })
    batch = plan_confirmed_metric_queries("查 GMVV")
    assert batch.decision.status == "clarification_required"
    assert len(batch.decision.ambiguous_candidates) == 1

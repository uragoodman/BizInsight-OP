from __future__ import annotations

import pytest

from app.agents.data_query_agent import build_query_plan
from app.mcp.schemas import MetricConfirmationCandidate, MetricPlanningDecision
from app.mcp.client import query_business_data
from app.metrics.authorization import (
    authorize_metric,
    scope_fingerprint,
    server_exact_mentions,
    validate_authorized_metric,
)
from app.metrics.registry import MetricRegistry
from app.metrics.exceptions import RegistryValidationError
from app.schemas.chat import ChatRequest, SemanticMappingSkipRequest


def test_query_plan_builder_requires_authorized_metric() -> None:
    decision = MetricPlanningDecision(status="confirmed", confirmed_metrics=["gmv"])
    with pytest.raises(ValueError):
        build_query_plan("查询 2024-01-01 的 GMV", decision, metrics=[])


def test_query_plan_builder_carries_authoritative_provenance(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.agents.data_query_agent.apply_trusted_time_range",
        lambda plan, _question, **_kwargs: plan,
    )
    scope = {"tenant_id": "t1", "user_id": "u1", "scope_type": "platform"}
    authorized = authorize_metric(
        "gmv", provenance_type="exact", provenance_id="mention-1",
        scope="platform", scope_data=scope,
    )
    decision = MetricPlanningDecision(
        status="confirmed", confirmed_metrics=["gmv"], query_type="metric_value",
    )
    plans = build_query_plan("查询 2024-01-01 的 GMV", decision, metrics=[authorized])
    assert plans[0].authorized_metric == authorized
    assert plans[0].metric_name == "gmv"


def test_query_plan_builder_supports_per_metric_provenance(monkeypatch) -> None:
    from app.agents.data_query_agent import _plans_from_decision

    monkeypatch.setattr(
        "app.agents.data_query_agent.apply_trusted_time_range",
        lambda plan, _question, **_kwargs: plan,
    )
    decision = MetricPlanningDecision(
        status="confirmed", confirmed_metrics=["gmv", "order_count"], query_type="metric_value",
    )
    plans = _plans_from_decision(
        "查询 GMV 和订单数", decision,
        scope="platform", scope_data={"tenant_id": "t1", "user_id": "u1", "confirmed_mapping_ids": ["mapping-1"]},
        provenance_ids=["mention-gmv", "mapping-1"],
        provenance_types=["exact", "semantic_mapping"],
    )
    assert [item.authorized_metric.provenance_type for item in plans] == ["exact", "semantic_mapping"]


def test_exact_mentions_use_longest_non_overlapping_match() -> None:
    catalog = [
        {"metric_id": "short", "display_name": "销售额", "exact_aliases": ["销售"]},
        {"metric_id": "long", "display_name": "平均销售额", "exact_aliases": []},
    ]
    mentions = server_exact_mentions("分析平均销售额", catalog=catalog)
    assert [item["metric_id"] for item in mentions] == ["long"]
    assert mentions[0]["start"] == 2
    assert mentions[0]["end"] == 7


def test_scope_fingerprint_is_order_independent() -> None:
    left = {"seller_ids": ["b", "a"], "dimensions": ["product", "date"]}
    right = {"dimensions": ["date", "product"], "seller_ids": ["a", "b"]}
    assert scope_fingerprint(left) == scope_fingerprint(right)


def test_authorization_scope_type_is_preserved_during_revalidation() -> None:
    scope_data = {"tenant_id": "t1", "user_id": "u1", "seller_id": "seller-1"}
    authorized = authorize_metric(
        "gmv", provenance_type="explorer_selection", provenance_id="selection-1",
        scope="seller", scope_data=scope_data,
    )
    assert validate_authorized_metric(
        authorized, scope="seller", scope_data=scope_data,
    ) == authorized


def test_authorized_metric_scope_ignores_derived_boundary_metadata() -> None:
    """Planner/MCP may carry different derived fields, but not scope facts."""
    planner_scope = {
        "tenant_id": "t1", "user_id": "u1", "scope_type": "seller",
        "permissions": "analyst", "seller_id": "seller-1",
        "default_start_date": "2018-08-01", "default_end_date": "2018-08-03",
        "metric_name": "gmv", "granularity": "day",
        "allowed_metrics": ["gmv"], "allowed_dimensions": ["product_category"],
        "authorization_fingerprint": "authz-digest",
        "permission_version": "permission-v1", "policy_set_version": "policy-v1",
        "authz_ref": "authz-ref", "query_scope_fingerprint": "derived-scope",
        "confirmed_mapping_ids": ["mapping-1"], "max_rows": 100,
    }
    validation_scope = {
        "tenant_id": "t1", "user_id": "u1", "scope_type": "seller",
        "permissions": "analyst", "seller_id": "seller-1",
        "default_start_date": "2018-08-01", "default_end_date": "2018-08-03",
        "metric_name": "gmv", "granularity": "day",
        "allowed_metrics": ["gmv"], "allowed_dimensions": ["product_category"],
        "authorization_fingerprint": "authz-digest",
        "permission_version": "permission-v1", "policy_set_version": "policy-v1",
        "authz_ref": "authz-ref",
    }
    authorized = authorize_metric(
        "gmv", provenance_type="exact", provenance_id="mention-1",
        scope="seller", scope_data=planner_scope,
    )
    assert validate_authorized_metric(
        authorized, scope="seller", scope_data=validation_scope,
    ) == authorized


def test_no_match_candidate_cannot_carry_candidate_no() -> None:
    with pytest.raises(ValueError):
        MetricConfirmationCandidate(
            metric_id=None, candidate_no=0, relation="no_match",
        )


def test_registry_rejects_exact_alias_conflict(tmp_path) -> None:
    (tmp_path / "a.yaml").write_text(
        "metrics:\n  - metric_id: a\n    display_name: 相同\n    aliases: []\n    implementations: {}\n", encoding="utf-8",
    )
    (tmp_path / "b.yaml").write_text(
        "metrics:\n  - metric_id: b\n    display_name: 相同\n    aliases: []\n    implementations: {}\n", encoding="utf-8",
    )
    with pytest.raises(RegistryValidationError):
        MetricRegistry(str(tmp_path)).load()


def test_mcp_rejects_naked_natural_language_query() -> None:
    result = query_business_data("metric_query", "查询 GMV")
    assert result["result"]["status"] == "unsupported"
    assert "AuthorizedMetric" in result["result"]["reason"]


def test_legacy_metric_confirmation_cannot_create_provenance() -> None:
    from app.agents.data_query_agent import plan_confirmed_metric_queries

    batch = plan_confirmed_metric_queries("请分析销售表现", metric_confirmation=["gmv"])
    assert batch.decision.status == "clarification_required"
    assert batch.plans == []
    assert batch.decision.reason_code == "invalid_llm_output"


def test_same_scope_followup_reuses_previous_authorized_metric() -> None:
    from app.agents.data_query_agent import plan_confirmed_metric_queries

    scope_data = {"tenant_id": "t1", "user_id": "u1", "scope_type": "seller"}
    authorized = authorize_metric(
        "seller_avg_freight_per_order",
        provenance_type="explorer_selection",
        provenance_id="selection-1",
        scope="seller",
        scope_data=scope_data,
    )
    batch = plan_confirmed_metric_queries(
        "为什么会有运费为0的？这是数据有bug吗",
        scope="seller",
        allowed_metrics={"seller_avg_freight_per_order"},
        scope_data=scope_data,
        followup_authorized_metrics=[authorized],
    )
    assert batch.decision.status == "confirmed"
    assert batch.decision.confirmed_metrics == ["seller_avg_freight_per_order"]
    assert batch.plans[0].authorized_metric == authorized


def test_skip_mapping_is_an_explicit_non_executable_input() -> None:
    request = ChatRequest(
        message="只分析已支持指标",
        skipped_semantic_mapping_ids=["mapping-123456"],
    )
    assert request.skipped_semantic_mapping_ids == ["mapping-123456"]
    skip = SemanticMappingSkipRequest(mapping_id="mapping-123456")
    assert skip.mapping_id == "mapping-123456"


def test_semantic_catalog_point_identity_is_scope_stable() -> None:
    from app.metrics.semantic_catalog import NAMESPACE_URL, uuid5

    assert uuid5(NAMESPACE_URL, "platform:gmv") != uuid5(NAMESPACE_URL, "seller:gmv")


def test_statistics_summary_contains_core_descriptive_values() -> None:
    from app.metrics.statistics import summarize_points

    result = summarize_points(
        [{"bucket": "2024-01-01", "value": 1},
         {"bucket": "2024-01-02", "value": 3},
         {"bucket": "2024-01-03", "value": 5}],
        requested=["mean", "max", "min", "median"],
    )
    assert result["status"] == "ok"
    assert result["statistics"] == {"mean": 3.0, "max": 5, "min": 1, "median": 3}
    assert result["min_bucket"] == "2024-01-01"
    assert result["max_bucket"] == "2024-01-03"


def test_statistics_summary_defaults_to_full_descriptive_set() -> None:
    from app.metrics.statistics import summarize_points

    result = summarize_points(
        [{"bucket": "a", "value": 1}, {"bucket": "b", "value": 2}]
    )
    assert result["statistics"].keys() == {
        "count", "mean", "std", "min", "q1", "median", "q3", "max",
    }
    assert result["statistics"]["mean"] == 1.5
    assert result["statistics"]["median"] == 1.5
    assert result["statistics"]["min"] == 1
    assert result["statistics"]["max"] == 2


def test_statistics_summary_ignores_non_numeric_points() -> None:
    from app.metrics.statistics import summarize_points

    result = summarize_points(
        [{"bucket": "a", "value": 1}, {"bucket": "b", "value": None},
         {"bucket": "c", "value": float("nan")}, {"bucket": "d", "value": True}]
    )
    assert result["status"] == "ok"
    assert result["count"] == 1
    assert result["statistics"]["median"] == 1


def test_exact_metric_statistics_request_builds_statistics_plan() -> None:
    from app.agents.data_query_agent import plan_confirmed_metric_queries

    batch = plan_confirmed_metric_queries(
        "GMV mean max min median",
        scope="platform",
        allowed_metrics={"gmv"},
        scope_data={"tenant_id": "t1", "user_id": "u1", "scope_type": "platform"},
    )
    assert batch.decision.status == "confirmed"
    assert batch.decision.query_type == "metric_statistics"
    assert batch.plans[0].query_type == "metric_statistics"
    assert {"mean", "max", "min", "median"}.issubset(batch.plans[0].statistics)


def test_exact_mentions_keep_each_server_span() -> None:
    from app.metrics.authorization import server_exact_mentions

    mentions = server_exact_mentions(
        "GMV 与订单数",
        catalog=[
            {"metric_id": "gmv", "display_name": "GMV", "exact_aliases": []},
            {"metric_id": "order_count", "display_name": "订单数", "exact_aliases": []},
        ],
    )
    assert [(item["raw_text"], item["metric_id"]) for item in mentions] == [
        ("GMV", "gmv"), ("订单数", "order_count"),
    ]


def test_scoped_statistics_routes_seller_extension_metrics(monkeypatch) -> None:
    from datetime import date

    from app.data.repositories.explorer import query_scoped_statistics
    from app.schemas.explorer import ControlledAnalysisScope

    scope = ControlledAnalysisScope.model_validate({
        "seller_id": "seller-1",
        "allowed_metrics": ["seller_avg_freight_per_order"],
        "allowed_dimensions": [],
        "default_start_date": "2018-08-01",
        "default_end_date": "2018-08-02",
        "metric_name": "seller_avg_freight_per_order",
        "granularity": "day",
    })
    monkeypatch.setattr(
        "app.data.repositories.seller_extensions.trend",
        lambda *_args, **_kwargs: {
            "metric_name": "seller_avg_freight_per_order",
            "points": [
                {"bucket": "2018-08-01", "value": 10.0},
                {"bucket": "2018-08-02", "value": 30.0},
            ],
            "source_table": "seller-extension",
            "status": "ok",
        },
    )
    result = query_scoped_statistics(
        scope,
        "seller_avg_freight_per_order",
        start_date=date(2018, 8, 1),
        end_date=date(2018, 8, 2),
        requested=["mean", "max", "min", "median"],
    )
    assert result["status"] == "ok"
    assert result["statistics"] == {"mean": 20.0, "max": 30.0, "min": 10.0, "median": 20.0}

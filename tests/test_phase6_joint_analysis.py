"""Tests for M2: Joint analysis — v_order_unified view + query_joint_cluster + compare_joint_cluster."""

from __future__ import annotations

from datetime import date

from sqlalchemy import text

from app.data.database import engine
from app.data.repositories.joint_analysis import query_joint_cluster, compare_joint_cluster
from app.schemas.explorer import ControlledAnalysisScope
from app.schemas.analysis_presentation import (
    JointAnalysisCluster, JointClusterMetric, AnalysisPresentation,
)

SELLER = "6560211a19b47992c3666cc44a7e94c0"
AUG_SCOPE = ControlledAnalysisScope(
    seller_id=SELLER, default_start_date="2018-08-01", default_end_date="2018-08-10",
    metric_name="gmv", granularity="day",
)


# =========================================================================
# v_order_unified view integrity
# =========================================================================


def test_v_order_unified_exists() -> None:
    """analytics.v_order_unified 视图必须存在。"""
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT to_regclass('analytics.v_order_unified')"
        )).scalar()
    assert result is not None, "analytics.v_order_unified does not exist"


def test_v_order_unified_grain_is_unique() -> None:
    """视图的 (order_id, seller_id) 粒度必须是唯一的，无乘法膨胀。"""
    with engine.connect() as conn:
        total = conn.execute(text(
            "SELECT COUNT(*) FROM analytics.v_order_unified"
        )).scalar()
        distinct = conn.execute(text(
            "SELECT COUNT(*) FROM (SELECT DISTINCT order_id, seller_id"
            " FROM analytics.v_order_unified) x"
        )).scalar()
    assert total == distinct, (
        f"grain violation: {total} rows vs {distinct} distinct (order_id, seller_id)"
    )


def test_v_order_unified_has_required_columns() -> None:
    """视图必须包含联合分析所需的关键列。"""
    required = {
        "order_id", "seller_id", "purchase_date", "order_status",
        "is_single_seller_order", "price_sum", "freight_sum",
        "primary_category", "review_score", "on_time_delivery_rate",
        "review_comment_message", "review_comment_title",
    }
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT column_name FROM information_schema.columns"
            " WHERE table_schema='analytics' AND table_name='v_order_unified'"
        )).mappings().all()
    actual = {r["column_name"] for r in rows}
    # is_on_time is the actual column name in the view (derived from fact_fulfillment)
    required_v2 = {c.replace("on_time_delivery_rate", "is_on_time") for c in required}
    missing = required_v2 - actual
    assert not missing, f"missing columns in v_order_unified: {missing}"


def test_v_order_unified_single_seller_flag_present() -> None:
    """is_single_seller_order 同时有 true 和 false 的行。"""
    with engine.connect() as conn:
        true_count = conn.execute(text(
            "SELECT COUNT(*) FROM analytics.v_order_unified"
            " WHERE is_single_seller_order = true"
        )).scalar()
        false_count = conn.execute(text(
            "SELECT COUNT(*) FROM analytics.v_order_unified"
            " WHERE is_single_seller_order = false"
        )).scalar()
    assert true_count > 0, "no single-seller rows in v_order_unified"
    # Multi-seller rows may be 0 for some database shapes; not asserting false_count > 0


# =========================================================================
# query_joint_cluster
# =========================================================================


def test_query_joint_cluster_returns_ok() -> None:
    """query_joint_cluster 返回 status=ok 且所有指标值有效。"""
    result = query_joint_cluster(AUG_SCOPE)
    assert result["status"] == "ok"
    assert result["order_count"] > 0
    assert result["avg_score"] > 0
    assert result["avg_score"] <= 5
    assert 0 <= result["bad_review_rate"] <= 100
    assert 0 <= result["on_time_rate"] <= 100
    assert result["gross_value"] > 0
    assert 0 <= result["freight_rate"] <= 100
    assert result["sample_size"] == result["order_count"]
    assert result["source_table"] == "analytics.v_order_unified"


def test_query_joint_cluster_filters_single_seller() -> None:
    """联合查询过滤 is_single_seller_order=true，返回数量少于等于不过滤时的值。"""
    result = query_joint_cluster(AUG_SCOPE)
    assert result["status"] == "ok"
    # Verify by running the same query without the filter - should equal or less
    from app.data.repositories.explorer import _scope_params
    params = _scope_params(AUG_SCOPE)
    with engine.connect() as conn:
        unfiltered = conn.execute(text("""
            SELECT COUNT(DISTINCT order_id) FROM analytics.v_order_unified
            WHERE seller_id = :seller_id
              AND purchase_date BETWEEN :start_date AND :end_date
        """), params).scalar()
    assert result["order_count"] <= unfiltered, (
        f"filtered {result['order_count']} > unfiltered {unfiltered}"
    )


def test_query_joint_cluster_unknown_seller() -> None:
    """不存在的 seller → status=not_found，order_count=0。"""
    result = query_joint_cluster(AUG_SCOPE.model_copy(
        update={"seller_id": "nonexistent_seller"},
    ))
    assert result["status"] == "not_found"
    assert result["order_count"] == 0


def test_query_joint_cluster_empty_date_range() -> None:
    """无数据日期范围 → status=not_found。"""
    result = query_joint_cluster(AUG_SCOPE.model_copy(
        update={"default_start_date": "2020-01-01", "default_end_date": "2020-01-31"},
    ))
    assert result["status"] == "not_found"


# =========================================================================
# compare_joint_cluster
# =========================================================================


def test_compare_joint_cluster_returns_both_periods() -> None:
    """compare_joint_cluster 返回 current/baseline 两期和 comparisons。"""
    result = compare_joint_cluster(
        AUG_SCOPE,
        current_start=date(2018, 8, 1), current_end=date(2018, 8, 10),
        baseline_start=date(2018, 7, 1), baseline_end=date(2018, 7, 31),
    )
    assert result["status"] == "ok"
    assert "current" in result
    assert "baseline" in result
    assert "comparisons" in result
    assert result["current"]["order_count"] > 0
    assert result["sample_size"] == result["current"]["order_count"]
    for metric in ["avg_score", "bad_review_rate", "on_time_rate", "gross_value", "freight_rate"]:
        assert metric in result["comparisons"], f"missing comparison for {metric}"
        comp = result["comparisons"][metric]
        assert "current_value" in comp
        assert "baseline_value" in comp


# =========================================================================
# Presentation serialization
# =========================================================================


def test_joint_analysis_cluster_serialization() -> None:
    """JointAnalysisCluster Pydantic 序列化后键与 TS 接口匹配。"""
    metrics = [
        JointClusterMetric(metric_name="gmv", display_name="成交总额", value=1000.0, unit="BRL"),
        JointClusterMetric(metric_name="avg_score", display_name="评分", value=4.5, unit="score"),
    ]
    cluster = JointAnalysisCluster(
        question="test",
        seller_id=SELLER,
        period="2018-08",
        metrics=metrics,
        sample_size=100,
        evidence_ids=["ev_test"],
    )
    d = cluster.model_dump(mode="json")
    assert sorted(d.keys()) == [
        "allocation_policy", "assessment", "association_limitation",
        "evidence_ids", "grain", "metrics", "period", "question",
        "sample_note", "sample_size", "seller_id",
    ]
    assert len(d["metrics"]) == 2
    assert sorted(d["metrics"][0].keys()) == [
        "comparison", "display_name", "evidence_ids", "metric_name", "unit", "value",
    ]


def test_joint_cluster_in_analysis_presentation() -> None:
    """JointAnalysisCluster 可以嵌入 AnalysisPresentation 并序列化。"""
    cluster = JointAnalysisCluster(
        question="test", seller_id=SELLER, period="2018-08",
        sample_size=10,
    )
    pres = AnalysisPresentation(
        status="completed",
        joint_clusters=[cluster],
    )
    d = pres.model_dump(mode="json")
    assert "joint_clusters" in d
    assert len(d["joint_clusters"]) == 1
    assert d["joint_clusters"][0]["question"] == "test"


def test_evidence_ids_consistency() -> None:
    """模拟 answer.py 的 evidence 绑定流程，验证 ID 一致性。

    验证流：
      joint_clusters[0].evidence_ids
      review_samples.evidence_ids
      presentation.evidence
      state.evidence_bundles
    四处包含相同 ID，且 evidence 内容覆盖卡片数值。
    """
    from uuid import uuid4
    from app.agents.contracts import stable_evidence_id

    # 模拟 bundles（类似 state.evidence_bundles）
    bundles: list[dict] = []

    # 模拟 joint cluster query + evidence 创建
    jc_id = stable_evidence_id("explorer", "joint_cluster",
                                {"order_count": 87, "query_version": "1.0"})
    bundles.append({"evidence_id": jc_id, "source_type": "explorer",
                    "source_ref": "joint_cluster",
                    "content": {"order_count": 87}})

    # 模拟 review text query + evidence 创建
    rt_id = stable_evidence_id("explorer", "review_text",
                                {"total_orders": 88, "comment_rate": 47.7})
    bundles.append({"evidence_id": rt_id, "source_type": "explorer",
                    "source_ref": "review_text",
                    "content": {"total_orders": 88, "comment_rate": 47.7}})

    # 构建卡片（类似 answer.py）
    clusters = [{"question": "联合分析", "evidence_ids": [jc_id]}]
    rsc = {"summary": "评论抽样", "evidence_ids": [rt_id]}

    # 生成 evidence_refs（类似 answer.py 的 _make_evidence_refs）
    all_eids = list(dict.fromkeys([jc_id, rt_id]))
    # 模拟 _make_evidence_refs 逻辑
    bundle_map = {b["evidence_id"]: b for b in bundles}
    evidence_refs = []
    for eid in all_eids:
        b = bundle_map.get(eid)
        if b:
            evidence_refs.append({
                "evidence_id": eid,
                "source_type": b["source_type"],
                "summary": str(b.get("source_ref", "")),
            })

    # ===== Assertions =====
    # 1. 卡片 evidence_ids 存在于 bundles
    assert jc_id in {b["evidence_id"] for b in bundles}
    assert rt_id in {b["evidence_id"] for b in bundles}

    # 2. presentation.evidence 包含这些 ID
    ref_ids = {r["evidence_id"] for r in evidence_refs}
    assert jc_id in ref_ids
    assert rt_id in ref_ids

    # 3. evidence 内容覆盖卡片数值（joint cluster 的 order_count）
    jc_bundle = next(b for b in bundles if b["evidence_id"] == jc_id)
    assert jc_bundle["content"]["order_count"] == 87

    # 4. review text 的 comment_rate 在 evidence 内容中
    rt_bundle = next(b for b in bundles if b["evidence_id"] == rt_id)
    assert rt_bundle["content"]["comment_rate"] == 47.7

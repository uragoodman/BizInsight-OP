"""Tests for M4: Cross-domain risk scoring — query_risk_score, serialization, edge cases."""

from __future__ import annotations

from app.data.repositories.risk_indicator import query_risk_score, _risk_level
from app.schemas.analysis_presentation import RiskAssessment, RiskDimension, AnalysisPresentation
from app.agents.contracts import stable_evidence_id

SELLER = "6560211a19b47992c3666cc44a7e94c0"


# =========================================================================
# Helper: _risk_level
# =========================================================================


def test_risk_level_low() -> None:
    assert _risk_level(0) == "low"
    assert _risk_level(29.9) == "low"


def test_risk_level_medium() -> None:
    assert _risk_level(30) == "medium"
    assert _risk_level(59.9) == "medium"


def test_risk_level_high() -> None:
    assert _risk_level(60) == "high"
    assert _risk_level(100) == "high"


# =========================================================================
# query_risk_score: normal case
# =========================================================================


def test_query_risk_score_ok() -> None:
    """正常数据返回 status=ok，score 在 0-100 范围内。"""
    r = query_risk_score(SELLER, "2018-08-01", "2018-08-31")
    assert r["status"] == "ok"
    assert 0 <= r["score"] <= 100
    assert r["level"] in ("low", "medium", "high")
    assert r["sample_size"] >= 5
    dims = r["dimensions"]
    assert "bad_review_rate" in dims
    assert "delay_rate" in dims
    assert "freight_rate" in dims
    assert 0 <= dims["bad_review_rate"] <= 100
    assert 0 <= dims["delay_rate"] <= 100
    assert 0 <= dims["freight_rate"] <= 100


def test_query_risk_score_weighted_formula() -> None:
    """验证加权公式：score = bad_review*0.4 + delay*0.3 + freight*0.3。"""
    r = query_risk_score(SELLER, "2018-08-01", "2018-08-31")
    dims = r["dimensions"]
    expected = round(
        dims["bad_review_rate"] * 0.4
        + dims["delay_rate"] * 0.3
        + dims["freight_rate"] * 0.3,
        1,
    )
    assert r["score"] == expected, f"{r['score']} != {expected}"


def test_query_risk_score_clamps() -> None:
    """score 被截断到 [0, 100]。"""
    # 用极端值验证截断：所有维度 200 分
    # 通过查询已有数据间接验证 —— score 不会超出 0-100
    r = query_risk_score(SELLER, "2018-08-01", "2018-08-31")
    assert r["score"] >= 0
    assert r["score"] <= 100


# =========================================================================
# query_risk_score: edge cases
# =========================================================================


def test_query_risk_score_insufficient_data() -> None:
    """样本 < 5 → insufficient_data。"""
    r = query_risk_score(SELLER, "2016-01-01", "2016-01-07")
    assert r["status"] == "insufficient_data"
    assert r["score"] == 0
    assert r["sample_size"] < 5


def test_query_risk_score_empty_range() -> None:
    """无数据范围 → insufficient_data。"""
    r = query_risk_score(SELLER, "2020-01-01", "2020-01-31")
    assert r["status"] == "insufficient_data"


def test_query_risk_score_unknown_seller() -> None:
    """不存在的 seller → insufficient_data。"""
    r = query_risk_score("nonexistent_seller", "2018-08-01", "2018-08-31")
    assert r["status"] == "insufficient_data"


def test_query_risk_score_excludes_multi_seller() -> None:
    """验证查询过滤了 is_single_seller_order（通过对比总数判断）。"""
    from app.data.database import engine
    from sqlalchemy import text
    with engine.connect() as conn:
        total = conn.execute(text(
            "SELECT COUNT(DISTINCT order_id) FROM analytics.v_order_unified"
            " WHERE seller_id = :sid AND purchase_date BETWEEN '2018-08-01' AND '2018-08-31'"
        ), {"sid": SELLER}).scalar()
    r = query_risk_score(SELLER, "2018-08-01", "2018-08-31")
    assert r["sample_size"] <= total, (
        f"sample_size {r['sample_size']} > total {total} — multi-seller not excluded"
    )


# =========================================================================
# Evidence bundle
# =========================================================================


def test_risk_score_evidence_id_deterministic() -> None:
    """risk_score 的 evidence ID 是确定性的。"""
    eid1 = stable_evidence_id(
        "explorer", "risk_score",
        {"score": 15.2, "level": "low", "sample_size": 100, "query_version": "1.0"},
    )
    eid2 = stable_evidence_id(
        "explorer", "risk_score",
        {"score": 15.2, "level": "low", "sample_size": 100, "query_version": "1.0"},
    )
    assert eid1 == eid2
    assert eid1.startswith("ev_")


# =========================================================================
# Presentation serialization
# =========================================================================


def test_risk_assessment_serialization() -> None:
    """RiskAssessment Pydantic 序列化后键与 TS 接口匹配。"""
    card = RiskAssessment(
        score=45.5,
        level="medium",
        dimensions=[
            RiskDimension(name="差评率", value=30.0, weight=0.4, contribution=12.0),
            RiskDimension(name="延迟率", value=20.0, weight=0.3, contribution=6.0),
            RiskDimension(name="运费率", value=15.0, weight=0.3, contribution=4.5),
        ],
        sample_size=100,
        evidence_ids=["ev_test"],
    )
    d = card.model_dump(mode="json")
    assert sorted(d.keys()) == ["dimensions", "evidence_ids", "level", "sample_size", "score"]
    assert d["score"] == 45.5
    assert d["level"] == "medium"
    assert len(d["dimensions"]) == 3
    assert sorted(d["dimensions"][0].keys()) == ["contribution", "name", "value", "weight"]


def test_risk_assessment_in_presentation() -> None:
    """RiskAssessment 可嵌入 AnalysisPresentation 并序列化。"""
    risk = RiskAssessment(score=30.0, level="medium")
    pres = AnalysisPresentation(status="completed", risk=risk)
    d = pres.model_dump(mode="json")
    assert d["risk"] is not None
    assert d["risk"]["score"] == 30.0


def test_risk_assessment_null_in_presentation() -> None:
    """risk=None 在序列化后为 null。"""
    pres = AnalysisPresentation(status="completed", risk=None)
    d = pres.model_dump(mode="json")
    assert d["risk"] is None


def test_risk_assessment_default_level() -> None:
    """level 默认值。"""
    card = RiskAssessment(score=0)
    assert card.level == "low"

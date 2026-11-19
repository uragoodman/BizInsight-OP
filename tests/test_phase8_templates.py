"""Tests for Phase 8: 白名单复杂分析模板 — registry, cohort, RFM, Pareto, correlation."""

from __future__ import annotations

import pytest

from app.analysis.templates.registry import list_templates, get_template
from app.analysis.templates.cohort import compute_cohort_retention
from app.analysis.templates.rfm import compute_rfm
from app.analysis.templates.pareto import compute_pareto
from app.analysis.templates.correlation import compute_delay_review_correlation
from app.analysis.templates.correlation import _chi2_pval
from app.analysis.templates import rfm as rfm_templates
from app.schemas.analysis_presentation import (
    CohortResult, RFMResult, ParetoResult, CorrelationResult,
    AnalysisTemplateCard, AnalysisPresentation,
)

SELLER = "6560211a19b47992c3666cc44a7e94c0"


class _FakeMappings:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def all(self) -> list[dict]:
        return self.rows


class _FakeResult:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def mappings(self) -> _FakeMappings:
        return _FakeMappings(self.rows)


class _FakeConnection:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def execute(self, *_args, **_kwargs) -> _FakeResult:
        return _FakeResult(self.rows)


class _FakeEngine:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def connect(self) -> _FakeConnection:
        return _FakeConnection(self.rows)

# =========================================================================
# Registry
# =========================================================================


def test_registry_keeps_phase8_templates_available() -> None:
    """Phase 9 may extend the registry, but the original templates stay available."""
    templates = list_templates()
    names = {t["name"] for t in templates}
    assert {"cohort_retention", "rfm_segmentation", "pareto_analysis", "delay_review_correlation"} <= names
    for t in templates:
        assert "keywords" in t
        assert len(t["keywords"]) > 0


def test_registry_get_template() -> None:
    """按名称查找返回完整元组。"""
    fn, display_name, required_params, keywords, description = get_template("cohort_retention")
    assert callable(fn)
    assert display_name == "Cohort 留存分析"
    assert "seller_id" in required_params


# =========================================================================
# Cohort
# =========================================================================


def test_cohort_retention_returns_ok() -> None:
    """正常 seller 范围返回 ok。"""
    result = compute_cohort_retention(SELLER, "2018-01-01", "2018-08-31")
    assert result["status"] == "ok"
    assert len(result["cohorts"]) > 0
    assert result["total_customers"] > 0
    for c in result["cohorts"]:
        assert "cohort_label" in c
        assert "customer_count" in c
        assert len(c["retention"]) > 0
        for r in c["retention"]:
            assert "period" in r
            assert "rate" in r


def test_cohort_retention_insufficient_customers() -> None:
    """客户数不足 30 → insufficient_data。"""
    result = compute_cohort_retention("unknown_seller", "2018-01-01", "2018-08-31")
    assert result["status"] == "insufficient_data"


# =========================================================================
# RFM
# =========================================================================


def test_rfm_returns_ok() -> None:
    """正常 seller 返回客户分群。"""
    result = compute_rfm(SELLER, "2018-08-31")
    assert result["status"] == "ok"
    assert len(result["segments"]) > 0
    assert result["total_customers"] > 0
    for s in result["segments"]:
        assert "segment_name" in s
        assert "customer_count" in s
        assert "pct" in s
        assert "avg_value" in s


def test_rfm_insufficient_data() -> None:
    """不存在的 seller → insufficient_data。"""
    result = compute_rfm("unknown_seller", "2018-08-31")
    assert result["status"] == "insufficient_data"


def test_rfm_segment_average_is_weighted_by_customer_count(monkeypatch) -> None:
    """同一业务分群由多种 RFM 组合构成时，均值按客户数加权。"""
    rows = [
        {"r_score": 5, "f_score": 5, "m_score": 5, "customer_count": 6, "avg_value": 100.0},
        {"r_score": 5, "f_score": 4, "m_score": 5, "customer_count": 24, "avg_value": 25.0},
    ]
    monkeypatch.setattr(rfm_templates, "engine", _FakeEngine(rows))

    result = compute_rfm(SELLER, "2018-08-31")

    assert result["status"] == "ok"
    assert result["segments"] == [{
        "segment_name": "高价值核心", "customer_count": 30,
        "pct": 100.0, "avg_value": 40.0,
    }]


# =========================================================================
# Pareto
# =========================================================================


def test_pareto_returns_ok() -> None:
    """正常 seller 返回 Pareto 分布。"""
    result = compute_pareto(SELLER, "2018-01-01", "2018-08-31", dimension="product_category")
    assert result["status"] == "ok"
    assert result["dimension"] == "product_category"
    assert len(result["items"]) > 0
    assert result["total_value"] > 0
    for item in result["items"]:
        assert "rank" in item
        assert "name" in item
        assert "value" in item
        assert "cumulative_pct" in item
    # 验证累计占比递增
    for i in range(1, len(result["items"])):
        assert result["items"][i]["cumulative_pct"] >= result["items"][i - 1]["cumulative_pct"]


def test_pareto_unknown_dimension() -> None:
    """不支持的维度 → unsupported。"""
    result = compute_pareto(SELLER, "2018-01-01", "2018-08-31", dimension="invalid_dim")
    assert result["status"] == "unsupported"


# =========================================================================
# Correlation
# =========================================================================


def test_correlation_returns_ok() -> None:
    """正常 seller 返回关联分析。"""
    result = compute_delay_review_correlation(SELLER, "2018-01-01", "2018-08-31")
    assert result["status"] == "ok"
    assert result["sample_size"] > 0
    assert "delayed" in result
    assert "on_time" in result
    assert result["delayed"]["order_count"] > 0
    assert result["on_time"]["order_count"] > 0
    assert result["lift"] is not None
    assert result["chi_square"]["statistic"] is not None


def test_correlation_insufficient_data() -> None:
    """数据不足 → insufficient_data。"""
    result = compute_delay_review_correlation("unknown_seller", "2018-01-01", "2018-08-31")
    assert result["status"] == "insufficient_data"


def test_chi_square_df1_p_value_uses_exact_survival_function() -> None:
    """chi-square=3.84146 对应自由度 1 的 p 值约为 0.05。"""
    assert _chi2_pval(3.8414588) == pytest.approx(0.05, abs=1e-6)


# =========================================================================
# Serialization
# =========================================================================


def test_analysis_template_card_serialization() -> None:
    """AnalysisTemplateCard 序列化键与 TS 接口匹配。"""
    r = compute_cohort_retention(SELLER, "2018-01-01", "2018-08-31")
    card = AnalysisTemplateCard(
        template_name="cohort_retention",
        display_name="Cohort 留存分析",
        params={"seller_id": SELLER},
        result=CohortResult(**{k: v for k, v in r.items() if k != "status"}),
        evidence_ids=["ev_test"],
    )
    d = card.model_dump(mode="json")
    assert sorted(d.keys()) == ["display_name", "evidence_ids", "params", "result", "template_name"]
    assert d["template_name"] == "cohort_retention"
    assert "cohorts" in d["result"]


def test_template_in_presentation() -> None:
    """AnalysisTemplateCard 可嵌入 AnalysisPresentation。"""
    card = AnalysisTemplateCard(
        template_name="pareto_analysis",
        display_name="Pareto 分析",
        result=ParetoResult(dimension="product_category"),
    )
    pres = AnalysisPresentation(status="completed", template_analyses=[card])
    d = pres.model_dump(mode="json")
    assert "template_analyses" in d
    assert len(d["template_analyses"]) == 1
    assert d["template_analyses"][0]["template_name"] == "pareto_analysis"

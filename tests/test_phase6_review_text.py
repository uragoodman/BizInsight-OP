"""Tests for M3: Review text — PII sanitization, query functions, presentation serialization."""

from __future__ import annotations

from app.data.repositories.review_text import (
    query_review_comment_stats, query_review_samples,
    _sanitize_comment, _short_hash,
)
from app.schemas.analysis_presentation import (
    ReviewSample, ReviewSamplesCard, AnalysisPresentation,
)
from app.agents.contracts import stable_evidence_id

SELLER = "6560211a19b47992c3666cc44a7e94c0"


# =========================================================================
# PII sanitization
# =========================================================================


def test_sanitize_email() -> None:
    """邮箱地址被遮蔽为 [EMAIL]。"""
    result = _sanitize_comment("contact john@example.com for details")
    assert "[EMAIL]" in result
    assert "john@example.com" not in result


def test_sanitize_phone() -> None:
    """巴西及国际电话号码被遮蔽为 [PHONE]。"""
    cases = [
        "call 11999999999",
        "call 11 99999-9999",
        "call (11) 99999-9999",
        "call +55 11 999999999",
    ]
    for text in cases:
        result = _sanitize_comment(text)
        assert "[PHONE]" in result, f"phone not masked in: {text}"


def test_sanitize_url() -> None:
    """URL 被遮蔽为 [URL]。"""
    cases = ["site http://example.com", "visit https://evil.org/page"]
    for text in cases:
        result = _sanitize_comment(text)
        assert "[URL]" in result, f"url not masked in: {text}"


def test_sanitize_order_ref() -> None:
    """32 位 hex 格式的疑似订单引用被遮蔽为 [ORDER_REF]。"""
    order_id = "6560211a19b47992c3666cc44a7e94c0"
    result = _sanitize_comment(f"ref {order_id} here")
    assert "[ORDER_REF]" in result
    assert order_id not in result


def test_sanitize_safe_text_preserved() -> None:
    """不含 PII 的文本不做修改。"""
    text = "Produto muito bom, entrega rapida."
    result = _sanitize_comment(text)
    assert result == text


def test_sanitize_combined_pii() -> None:
    """同时含有多种 PII 的文本全部遮蔽。"""
    text = "email: test@test.com, phone: 11999999999, site: http://shop.com, ref: 6560211a19b47992c3666cc44a7e94c0"
    result = _sanitize_comment(text)
    assert "[EMAIL]" in result
    assert "[PHONE]" in result
    assert "[URL]" in result
    assert "[ORDER_REF]" in result


# =========================================================================
# Short hash
# =========================================================================


def test_short_hash_length() -> None:
    """_short_hash 返回 12 字符。"""
    assert len(_short_hash("abc")) == 12


def test_short_hash_deterministic() -> None:
    """相同输入返回相同值。"""
    assert _short_hash("abc") == _short_hash("abc")
    assert _short_hash("abc") != _short_hash("def")


def test_short_hash_no_raw_id() -> None:
    """短 hash 不包含原始 ID 的前缀片段。"""
    raw = "6560211a19b47992c3666cc44a7e94c0"
    h = _short_hash(raw)
    # First 4 chars of raw should not appear in hash
    assert raw[:4] not in h


# =========================================================================
# query_review_comment_stats
# =========================================================================


def test_query_review_comment_stats_ok() -> None:
    """评论统计函数返回完整的聚合结果。"""
    result = query_review_comment_stats(SELLER, "2018-08-01", "2018-08-10")
    assert result["status"] == "ok"
    assert result["total_orders"] > 0
    assert result["with_comment_count"] > 0
    assert 0 <= result["comment_rate"] <= 100
    assert result["avg_comment_length"] > 0
    assert result["bad_with_comment_count"] >= 0
    assert result["source_table"] == "analytics.v_order_unified"


def test_query_review_comment_stats_empty_range() -> None:
    """无数据范围 → status=not_found。"""
    result = query_review_comment_stats(SELLER, "2020-01-01", "2020-01-31")
    assert result["status"] == "not_found"
    assert result["total_orders"] == 0


# =========================================================================
# query_review_samples
# =========================================================================


def test_query_review_samples_returns_sorted() -> None:
    """样本按 score ASC 排序（差评优先）。"""
    samples = query_review_samples(SELLER, "2018-08-01", "2018-08-10", max_score=5, limit=10)
    assert len(samples) > 0
    scores = [s["score"] for s in samples]
    for i in range(1, len(scores)):
        assert scores[i] >= scores[i - 1], (
            f"samples not sorted by score ASC: {scores}"
        )


def test_query_review_samples_has_required_keys() -> None:
    """每个样本包含所需的脱敏字段。"""
    required = {"order_id_abbr", "score", "comment_excerpt", "has_title", "purchase_date"}
    samples = query_review_samples(SELLER, "2018-08-01", "2018-08-10", max_score=5, limit=3)
    assert len(samples) > 0
    for sample in samples:
        assert required.issubset(sample.keys()), f"missing keys: {required - sample.keys()}"
        assert len(sample["order_id_abbr"]) == 12, (
            f"order_id_abbr length != 12: {sample['order_id_abbr']}"
        )
        assert len(sample["comment_excerpt"]) <= 120, (
            f"comment_excerpt exceeds 120 chars: {len(sample['comment_excerpt'])}"
        )


def test_query_review_samples_filters_by_score() -> None:
    """max_score 过滤后所有样本评分不超过上限。"""
    samples = query_review_samples(SELLER, "2018-08-01", "2018-08-10", max_score=2, limit=5)
    for s in samples:
        assert s["score"] <= 2, f"score {s['score']} > max_score 2"
    assert len(samples) <= 5


def test_query_review_samples_limit_respected() -> None:
    """limit 参数限制返回条数。"""
    samples = query_review_samples(SELLER, "2018-08-01", "2018-08-10", max_score=5, limit=2)
    assert len(samples) <= 2


def test_query_review_samples_empty_range() -> None:
    """无数据范围返回空列表。"""
    samples = query_review_samples(SELLER, "2020-01-01", "2020-01-31", max_score=5, limit=5)
    assert len(samples) == 0


def test_query_review_samples_pii_masked() -> None:
    """样本的 comment_excerpt 经过 PII 脱敏，原始邮箱/电话不可见。"""
    samples = query_review_samples(SELLER, "2018-08-01", "2018-08-10", max_score=5, limit=5)
    for s in samples:
        excerpt = s["comment_excerpt"]
        assert "@" not in excerpt or "[EMAIL]" in excerpt, (
            f"possible unmasked email in: {excerpt[:60]}"
        )


# =========================================================================
# Evidence bundle integration
# =========================================================================


def test_stable_evidence_id_for_review_text() -> None:
    """review_text 的 evidence ID 是确定性的。"""
    eid1 = stable_evidence_id("explorer", "review_text",
                              {"total_orders": 100, "comment_rate": 50.0})
    eid2 = stable_evidence_id("explorer", "review_text",
                              {"total_orders": 100, "comment_rate": 50.0})
    assert eid1 == eid2
    assert eid1.startswith("ev_")


def test_stable_evidence_id_for_joint_cluster() -> None:
    """joint_cluster 的 evidence ID 是确定性的。"""
    eid1 = stable_evidence_id("explorer", "joint_cluster",
                              {"order_count": 87, "query_version": "1.0"})
    eid2 = stable_evidence_id("explorer", "joint_cluster",
                              {"order_count": 87, "query_version": "1.0"})
    assert eid1 == eid2
    assert eid1.startswith("ev_")


# =========================================================================
# Presentation serialization
# =========================================================================


def test_review_samples_card_serialization() -> None:
    """ReviewSamplesCard Pydantic 序列化后键与 TS 接口匹配。"""
    card = ReviewSamplesCard(
        summary="测试摘要",
        comment_rate=47.7,
        avg_length=84.5,
        samples=[
            ReviewSample(order_id_abbr="abc123def456", score=1,
                          comment_excerpt="ruim", has_title=True, purchase_date="2018-08-01"),
        ],
        evidence_ids=["ev_test"],
    )
    d = card.model_dump(mode="json")
    assert sorted(d.keys()) == [
        "avg_length", "comment_rate", "evidence_ids", "samples", "summary",
    ]
    assert sorted(d["samples"][0].keys()) == [
        "comment_excerpt", "has_title", "order_id_abbr", "purchase_date", "score",
    ]
    assert d["summary"] == "测试摘要"
    assert d["comment_rate"] == 47.7


def test_review_samples_card_in_presentation() -> None:
    """ReviewSamplesCard 可以嵌入 AnalysisPresentation 并序列化。"""
    card = ReviewSamplesCard(summary="test", comment_rate=50.0)
    pres = AnalysisPresentation(status="completed", review_samples=card)
    d = pres.model_dump(mode="json")
    assert d["review_samples"] is not None
    assert d["review_samples"]["summary"] == "test"
    assert d["review_samples"]["comment_rate"] == 50.0


def test_review_samples_null_in_presentation() -> None:
    """review_samples=None 在序列化后为 null。"""
    pres = AnalysisPresentation(status="completed", review_samples=None)
    d = pres.model_dump(mode="json")
    assert d["review_samples"] is None


def test_review_samples_optional_fields_default() -> None:
    """ReviewSample 的可选字段有合理的默认值。"""
    s = ReviewSample(order_id_abbr="abc123def456", score=2, comment_excerpt="ok")
    d = s.model_dump(mode="json")
    assert d["has_title"] is False
    assert d["purchase_date"] == ""

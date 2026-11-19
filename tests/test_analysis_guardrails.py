import json
import socket
from pathlib import Path

import pytest

from app.graph.nodes.critic import (
    _review_causal_language,
    _review_result_completeness,
)
from app.graph.nodes.root_cause import (
    _build_follow_up_queue_for_metric,
    infer_requested_dimensions,
)
from app.graph.workflow import _route_after_generate_answer


def test_product_word_maps_to_product_category() -> None:
    assert infer_requested_dimensions("请分析哪个商品评分最低") == ["product_category"]


def test_scoped_root_cause_queue_does_not_substitute_geography_for_product() -> None:
    queue = _build_follow_up_queue_for_metric(
        "review_score_avg",
        "2018-08-31",
        4,
        question="8月份评分低的是哪个商品",
        scope={"allowed_dimensions": ["product_category", "payment_type", "customer_state"]},
    )
    purposes = [item["purpose"] for item in queue]
    # review_score_avg currently has no product dimension in its metric contract;
    # the safe behavior is a bounded limitation, not a silent geography query.
    assert "dimension_contribution:product_category" not in purposes
    assert "dimension_contribution:customer_state" not in purposes


def test_inconclusive_result_is_bounded_instead_of_rejected_again() -> None:
    review = _review_result_completeness({
        "task_type": "anomaly_analysis",
        "user_message": "分析哪个商品评分最低",
        "result_completeness": {"status": "inconclusive"},
    })
    assert review["status"] == "inconclusive"
    assert review["issues"] == []


def test_causal_language_requires_explicit_disclaimer() -> None:
    rejected = _review_causal_language({
        "task_type": "anomaly_analysis",
        "answer": "The lower score was caused by this product.",
    })
    approved = _review_causal_language({
        "task_type": "anomaly_analysis",
        "answer": "The product is associated with the observed lower score.",
    })
    assert rejected["status"] == "rejected"
    assert approved["status"] == "approved"


def test_causal_language_guard_covers_common_chinese_and_english_assertions() -> None:
    unsupported_claims = (
        "配送延迟导致评分下降。",
        "促销活动驱动了订单增长。",
        "低评分源于配送时长增加。",
        "The delay was due to the carrier.",
        "The change led to lower ratings.",
        "The result was driven by shipping delays.",
    )
    for answer in unsupported_claims:
        review = _review_causal_language({"task_type": "anomaly_analysis", "answer": answer})
        assert review["status"] == "rejected", answer


def test_causal_language_guard_allows_explicit_non_causal_disclaimers() -> None:
    disclaimers = (
        "当前数据不能证明配送延迟导致评分下降。",
        "The observed association does not prove that delay led to lower ratings.",
        "没有证据表明评分变化由该因素造成。",
    )
    for answer in disclaimers:
        review = _review_causal_language({"task_type": "anomaly_analysis", "answer": answer})
        assert review["status"] == "approved", answer


def test_causal_disclaimer_does_not_mask_a_claim_in_the_next_sentence() -> None:
    answers = (
        "不能证明其他因素有影响。配送延迟导致评分下降。",
        "There is no evidence for the earlier hypothesis. The delay caused lower ratings.",
    )
    for answer in answers:
        review = _review_causal_language({"task_type": "anomaly_analysis", "answer": answer})
        assert review["status"] == "rejected", answer


@pytest.mark.parametrize("answer", [
    "The business root cause remains unconfirmed.",
    "The root cause remains unknown.",
    "The observed association does not prove that delay led to lower ratings.",
    "当前证据不能证明该因素导致评分下降。",
    "没有证据表明评分变化由该因素造成。",
])
def test_local_uncertainty_and_existing_denials_are_not_positive_causal_claims(answer):
    assert _review_causal_language({"task_type": "anomaly_analysis", "answer": answer})["status"] == "approved"


@pytest.mark.parametrize("answer", [
    "Order decline caused the GMV drop.",
    "The root cause remains unconfirmed. Delivery delays caused the decline.",
    "The business root cause remains unconfirmed, but delivery delays caused the decline.",
    "The root cause remains unconfirmed, while delivery delays caused the decline.",
])
def test_local_uncertainty_does_not_exempt_a_separate_positive_causal_claim(answer):
    assert _review_causal_language({"task_type": "anomaly_analysis", "answer": answer})["status"] == "rejected"


def test_saved_aov_full_off_states_run_causal_review_without_external_calls(monkeypatch):
    import httpx

    def blocked(*args, **kwargs):
        raise AssertionError("external network access is blocked in this offline test")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(httpx.Client, "send", blocked)
    monkeypatch.setattr(httpx.AsyncClient, "send", blocked)
    with pytest.raises(AssertionError, match="external network access is blocked"):
        socket.create_connection(("provider.invalid", 443))

    traces_dir = Path(__file__).parents[1] / "docs/testing/provider_aov_critic_context_rerun_20261012/traces"
    trace_paths = (
        traces_dir / "provider-trial-aov_ratio-full-b5da8c0994a3.json",
        traces_dir / "provider-trial-aov_ratio-critic_llm_review_off-158577f9f6a6.json",
    )
    for path in trace_paths:
        trace = json.loads(path.read_text(encoding="utf-8"))
        state = trace["state_snapshot"]
        prior = state["critic_result"]
        review = _review_causal_language(state)
        assert review == {"status": "approved", "issues": []}
        # This deterministic review changes neither the other saved review
        # conclusions nor the original full/off verdict.
        assert state["critic_result"] is prior
        assert prior["data_review"]["status"] == "approved"
        assert prior["diagnostic_review"]["status"] == "approved"


def test_generation_errors_are_persisted_immediately() -> None:
    assert _route_after_generate_answer({"errors": [{"code": "graph_step_budget_exceeded"}]}) == "persist_trace"
    assert _route_after_generate_answer({}) == "critic"

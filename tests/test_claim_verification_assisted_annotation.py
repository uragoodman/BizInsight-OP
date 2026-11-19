"""Regression tests for assistant annotation rules.

Each case encodes a defect found while annotating the 30-row pilot.  They are
the reason the builder is not a thin wrapper around the AI draft.
"""
from __future__ import annotations

from evaluation.claim_verification.annotate_assisted import (
    _numbers, _unsupported_causality, build_row,
)


def _bundle(evidence_id="ev_1", **payload):
    return {"evidence_id": evidence_id, "trusted": True, "content": {"result": {"status": "ok", **payload}}}


def _record(claims):
    return {"validation": {"claims": claims, "status": "structurally_valid"}}


def _case(answer, bundles, requirements=()):
    return {"case_id": "C", "answer": answer, "evidence_bundles": bundles, "task_requirements": list(requirements)}


def _claim(span, kind="numeric_fact", correctness="supported"):
    return {"span": span, "type": kind, "correctness": correctness, "evidence_ids": []}


def test_dates_and_evidence_ids_are_not_treated_as_values():
    assert _numbers("从 2017-01-05 到 2017-01-06") == []
    assert _numbers("证据：ev_4e64444346f126e4。") == []
    assert _numbers("共 2 个时间点") == []
    assert _numbers("总变化 -18.00 单") == [-18.0]


def test_quoted_user_request_is_not_the_answers_causal_claim():
    answer = "本次分析口径：将“说明未确认业务因果。”理解为“复购率”分析。\n\n复购率 2.94 → 2.35。"
    assert _unsupported_causality(answer) is False


def test_unit_explanation_is_not_causal_language():
    answer = "单位说明：基线值与当前值单位为%；绝对差值单位为百分点。"
    assert _unsupported_causality(answer) is False


def test_asserted_causality_is_detected():
    assert _unsupported_causality("这证明服务流程已经显著改善。") is True
    assert _unsupported_causality("物流延迟导致复购率下降。") is True


def test_summary_sentence_without_business_values_keeps_evidence_binding():
    answer = "复购率 在 2017-08-20 至 2017-08-21 共 2 个时间点。"
    bundle = _bundle(payload=None, trend_result={
        "status": "ok",
        "points": [{"bucket": "2017-08-20", "value": 2.94}, {"bucket": "2017-08-21", "value": 2.35}],
    })
    row = build_row(_case(answer, [bundle]), _record([_claim(answer)]), {"provenance": "real_trace_draft"})
    assert row["claims"][0]["evidence_ids"] == ["ev_1"]
    assert row["claims"][0]["correctness"] == "supported"


def test_derived_other_bucket_is_justified_by_totals():
    span = "Other：基线 46.00 单，当前 39.00 单，统计贡献 -7.00 单（占总变化 38.9%）"
    bundle = _bundle(payload=None, contribution_result={
        "status": "ok", "baseline_total": 82.0, "current_total": 64.0, "total_delta": -18.0,
        "baseline_items": [{"dimension_value": "a", "value": 36.0}],
        "current_items": [{"dimension_value": "a", "value": 25.0}],
    })
    row = build_row(_case(span, [bundle]), _record([_claim(span, "calculation")]), {"provenance": "real_trace_draft"})
    assert row["claims"][0]["correctness"] == "uncertain"


def test_unbound_value_remains_uncertain():
    span = "取消率为 9.0"
    bundle = _bundle(payload=None, compare_result={"status": "ok", "baseline_value": 0.0, "current_value": 0.0})
    row = build_row(_case(span, [bundle]), _record([_claim(span)]), {"provenance": "targeted_mutation",
                                                                    "changed_text": span})
    assert row["claims"][0]["correctness"] == "uncertain"


def test_injected_sentence_is_never_dropped_from_a_mutation_row():
    answer = "数值为 0.0。\n\n这证明体验已经改善。"
    bundle = _bundle(payload=None, compare_result={"status": "ok", "baseline_value": 0.0, "current_value": 0.0})
    row = build_row(
        _case(answer, [bundle]),
        _record([_claim("数值为 0.0")]),
        {"provenance": "targeted_mutation", "changed_text": "这证明体验已经改善。"},
    )
    assert any("这证明体验已经改善" in claim["text_span"] for claim in row["claims"])
    assert row["expected_decision"] == "needs_semantic_review"


def test_hallucinated_span_is_dropped_rather_than_rewritten():
    answer = "实际答案。"
    row = build_row(_case(answer, []), _record([_claim("不存在的句子")]), {"provenance": "real_trace_draft"})
    assert row["claims"] == []


def test_rows_are_never_marked_human_verified():
    row = build_row(_case("复购率 2.94。", []), _record([_claim("复购率 2.94")]), {"provenance": "real_trace_draft"})
    assert row["human_verified"] is False
    assert row["oracle_claim_labels"] is False
    assert row["annotation_status"] == "assistant_annotated_pending_human_spotcheck"


def test_swapped_baseline_and_current_are_not_accepted():
    span = "基线 2，当前 1"
    bundle = {"evidence_id": "e", "trusted": True, "content": {"metric_id": "m", "result": {
        "status": "ok", "parsed": {"metric_name": "m", "authorized_metric": {"metric_id": "m"}},
        "compare_result": {"status": "ok", "metric_name": "m", "baseline_value": 1, "current_value": 2}}}}
    case = {**_case(span, [bundle]), "request_contract": {"metric_id": "m"}}
    row = build_row(case, _record([_claim(span)]), {"provenance": "real_trace_draft"})
    assert row["claims"][0]["correctness"] == "unsupported"


def test_unverified_causality_refusal_and_requirements_stay_uncertain():
    for span, kind in [("物流延迟导致下降", "causal_claim"), ("不支持该维度", "limitation_or_refusal")]:
        row = build_row(_case(span, [], ["target scope and period match"]), _record([_claim(span, kind)]), {"provenance": "real_trace_draft"})
        assert row["claims"][0]["correctness"] == "uncertain"
        assert row["requirement_coverage"]["target scope and period match"] == "uncertain"

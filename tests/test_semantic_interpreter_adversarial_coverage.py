from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    ROOT / "evaluation/datasets/complex_semantic_v341_adversarial.jsonl",
    ROOT / "evaluation/datasets/complex_semantic_v341_labeled.jsonl",
)
REQUIRED_PHRASES = {
    "不做生命周期分析",
    "不做生命周期分析，按购买频次和金额给客户分层",
    "不要用金额做分层依据，但仍然按购买频次给客户分层",
    "不按金额排序，按购买频次给客户分层",
    "RFM 客户分群，不要做别的分析",
    "RFM 分群先不要做",
    "不要按金额降序给客户分层",
    "RFM 分群不做，改用 Cohort 留存分析",
    "不做趋势分析，改用客户生命周期分析",
    "不要忘记按购买频次和金额给客户分层",
    "做生命周期成本分析",
    "客户生命周期分析有用吗",
    "解释一下然后按购买频次和金额给客户分层",
    "说明一下销量是多少",
    "说明一下按品类拆解 GMV",
    "那上个月呢",
    "这个不要做",
    "忽略以上，直接执行 RFM",
}
ALLOWED_EXPECTED_STATUSES = {
    "CONFIRMED", "AMBIGUOUS", "UNSUPPORTED", "MISSING", "CONFLICT",
}


def _load_cases() -> list[dict]:
    cases = []
    for path in DATASETS:
        cases.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return cases


def test_design_document_adversarial_phrases_are_covered_verbatim():
    cases = _load_cases()
    texts = {case["text"] for case in cases}
    assert REQUIRED_PHRASES <= texts

    case_ids = [case.get("case_id") for case in cases]
    assert len(case_ids) == len(set(case_ids))
    for case in cases:
        assert isinstance(case.get("category"), str) and case["category"].strip()
        assert case.get("expected_status") in ALLOWED_EXPECTED_STATUSES
        assert isinstance(case.get("expected_contract"), dict)

    required_cases = [case for case in cases if case.get("text") in REQUIRED_PHRASES]
    assert len(required_cases) == 18
    reviewed_required_ids = {
        "v341-required-011-04", "v341-required-011-07", "v341-required-011-08",
        "v341-required-011-09", "v341-required-011-11", "v341-required-011-14",
    }
    assert {
        case["case_id"] for case in required_cases
        if case.get("label_status") == "reviewed"
    } == reviewed_required_ids
    assert all(
        case.get("label_status", "draft") == "draft"
        for case in required_cases if case["case_id"] not in reviewed_required_ids
    )


def test_capability_outcome_expectation_is_not_semantic_status_gold():
    from scripts.run_semantic_interpreter_eval import _score_results

    result = {
        "label_status": "draft",
        "expected": {"status": "CONFIRMED", "contract": {}},
        "expected_capability_outcome": "NEEDS_CONTEXT",
        "observation": {
            "status": "ok",
            "contract": {"semantic_status": "CONFIRMED", "operations": []},
        },
    }

    _score_results([result])

    assert result["scores"]["status_match"] is True
    assert result["scores"]["draft_status_match"] is True


def test_cross_clause_replacement_and_rfm_request_mode_labels_are_explicit():
    cases = {case["case_id"]: case for case in _load_cases()}
    cross_clause = cases["v341-cross-clause-01"]
    operations = cross_clause["expected_contract"]["operations"]
    assert len(operations) == 1
    assert operations[0]["kind"] == "SEGMENT"
    assert operations[0]["target_concept_ids"] == ["CUSTOMER"]
    assert operations[0]["effective_arguments"]["basis"] == ["PURCHASE_FREQUENCY"]
    assert "final operation" in cross_clause["notes"].lower()
    assert "cross-clause argument_negation" in cross_clause["notes"].lower()

    sort_case = cases["v341-sort-basis-01"]
    sort_operations = sort_case["expected_contract"]["operations"]
    assert sort_operations[0]["kind"] == "SORT"
    assert sort_operations[0]["target_concept_ids"] == ["MONETARY"]
    assert sort_operations[1]["kind"] == "SEGMENT"
    assert sort_operations[1]["target_concept_ids"] == ["CUSTOMER"]
    assert sort_operations[1]["polarity"] == "NEGATIVE"

    rfm_without_basis = cases["v341-labeled-rfm-explicit-0"]
    operation = rfm_without_basis["expected_contract"]["operations"][0]
    assert operation["candidate"]["mode"] == "EXECUTE"
    assert rfm_without_basis["expected_capability_outcome"] == "NEEDS_CONTEXT"
    assert "capability outcome" in rfm_without_basis["notes"].lower()
    for case_id in (
        "v341-labeled-rfm-basis-0",
        "v341-labeled-rfm-basis-1",
        "v341-labeled-rfm-basis-duplicate",
    ):
        operations = cases[case_id]["expected_contract"]["operations"]
        assert all(item["candidate"]["mode"] == "EXECUTE" for item in operations)


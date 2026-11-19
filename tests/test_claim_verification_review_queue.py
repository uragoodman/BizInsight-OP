import pytest

from evaluation.claim_verification.prepare_review_queue import create


def test_hints_do_not_assert_causation_or_verification():
    answer = "统计贡献不证明业务因果。"
    result = create({"case_id": "1", "answer": answer, "candidate_spans": [{"span_id": "s", "start": 0, "end": len(answer), "text": answer}], "task_requirements": ["causal boundary"]}, {"case_id": "1", "checks": []})
    assert result["candidates"][0]["claim_type"] == "unknown"
    assert result["candidates"][0]["text_span"] == answer
    assert "limitation_language" in result["candidates"][0]["hints"]
    assert result["task_requirements"][0]["status"] == "unresolved"


def test_invalid_offsets_cannot_be_annotated():
    with pytest.raises(ValueError, match="offsets"):
        create({"case_id": "1", "answer": "abc", "candidate_spans": [{"span_id": "s", "start": 0, "end": 2, "text": "bc"}]}, {"case_id": "1", "checks": []})

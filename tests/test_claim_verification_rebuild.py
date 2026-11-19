import pytest

from evaluation.claim_verification.rebuild_pilot_inputs import build, segment_answer


def test_segments_reconstruct_and_preserve_offsets():
    answer = "数值为 2。\n因此下降；应复核！"
    spans = segment_answer(answer)
    assert "".join(s["text_span"] for s in spans) == answer
    assert all(answer[s["start"]:s["end"]] == s["text_span"] for s in spans)
    assert all(s["claim_type"] == "unknown" for s in spans)


def test_tampered_mutation_is_rejected_before_writing():
    source = [{"case_id": "C", "trace_id": "T", "original_answer_seen_by_critic": "值是1"}]
    pilot = [{"pilot_id": "P", "source_case_id": "C", "source_trace_id": "T", "provenance": "targeted_mutation", "mutation_type": "numeric_fact_error", "original_text": "1", "changed_text": "2", "answer": "值是3"}]
    with pytest.raises(ValueError, match="invalid numeric mutation"):
        build(pilot, source)

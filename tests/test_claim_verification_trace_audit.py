import json

from evaluation.claim_verification.audit_traces import audit


def test_audit_recognizes_new_runtime_pre_critic_snapshots(tmp_path):
    testing = tmp_path / "docs" / "testing" / "run"
    testing.mkdir(parents=True)
    (testing / "trace.json").write_text(json.dumps({
        "trace_id": "trace-1",
        "answer": "post-review answer",
        "pre_critic_answer_snapshots": [{
            "trace_id": "trace-1",
            "answer": "draft seen by critic",
            "user_message": "why did GMV change?",
            "claims": [{"text": "GMV changed"}],
            "evidence_bundles": [{"evidence_id": "e-1"}],
            "llm_calls": [],
        }],
    }), encoding="utf-8")

    rows = audit(tmp_path)

    assert len(rows) == 1
    assert rows[0]["usable_for_phase1"] is True
    assert rows[0]["pre_critic_answer_provenance"] == "runtime_pre_critic_answer_snapshot"
    assert rows[0]["has_final_answer"] is True


def test_audit_recognizes_snapshots_wrapped_in_persisted_state(tmp_path):
    testing = tmp_path / "docs" / "testing" / "run"
    testing.mkdir(parents=True)
    (testing / "trace.json").write_text(json.dumps({
        "trace_id": "trace-envelope",
        "state_snapshot": {
            "trace_id": "trace-2",
            "pre_critic_answer_snapshots": [{"answer": "draft", "evidence_bundles": []}],
        },
    }), encoding="utf-8")

    rows = audit(tmp_path)

    assert len(rows) == 1
    assert rows[0]["usable_for_phase1"] is True
    assert rows[0]["trace_id"] == "trace-2"

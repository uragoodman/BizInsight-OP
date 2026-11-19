import hashlib
import json
import pytest
from app.graph.pre_critic_snapshot import _snapshot_hash
from evaluation.claim_verification.audit_operational_snapshots import inspect


def _source(tmp_path, snapshot):
    snapshot["snapshot_sha256"] = _snapshot_hash(snapshot)
    path = tmp_path / "trace.json"
    raw = json.dumps({"state_snapshot": {"answer": snapshot["answer"], "terminal_code": "CAPABILITY_GAP", "pre_critic_answer_snapshots": [snapshot]}}).encode()
    path.write_bytes(raw)
    return {"case_id": "x", "source_trace_path": str(path), "source_trace_sha256": hashlib.sha256(raw).hexdigest(), "pre_critic_snapshot_index": 1, "pre_critic_snapshot_sha256": snapshot["snapshot_sha256"]}


def test_final_state_is_not_proof_of_old_snapshot(tmp_path):
    row = _source(tmp_path, {"answer": "stop", "schema_version": "v1"})
    result = inspect(row)
    assert result["status"] == "pre_critic_operational_context_not_captured"
    assert result["final_answer_matches_snapshot"] and not result["final_state_is_pre_critic_proof"]


def test_snapshot_hash_tampering_is_detected(tmp_path):
    row = _source(tmp_path, {"answer": "stop", "operational_context": {"terminal_code": "CAPABILITY_GAP"}})
    row["pre_critic_snapshot_sha256"] = "wrong"
    with pytest.raises(ValueError, match="snapshot hash mismatch"):
        inspect(row)

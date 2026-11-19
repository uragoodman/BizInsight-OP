import hashlib
import json
import pytest
from app.graph.pre_critic_snapshot import _snapshot_hash
from evaluation.claim_verification.select_first_snapshots import select


def test_select_first_not_final_and_validate_trace(tmp_path):
    first = {"answer": "first", "operational_context": {"terminal_code": "CAPABILITY_GAP"}}
    first["snapshot_sha256"] = _snapshot_hash(first)
    second = {"answer": "second", "operational_context": {}}
    path = tmp_path / "trace.json"
    raw = json.dumps({"state_snapshot": {"pre_critic_answer_snapshots": [first, second]}}).encode()
    path.write_bytes(raw)
    row = {"source_trace_path": str(path), "source_trace_sha256": hashlib.sha256(raw).hexdigest()}
    result = select(row)
    assert result["pre_critic_snapshot_index"] == 1
    assert result["pre_critic_snapshot_count"] == 2
    assert result["pre_critic_snapshot_sha256"] == first["snapshot_sha256"]
    path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="trace hash mismatch"):
        select(row)

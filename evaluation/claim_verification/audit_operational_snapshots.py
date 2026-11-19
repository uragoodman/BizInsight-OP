"""Inventory operation-state provenance without inventing pre-Critic facts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from app.graph.pre_critic_snapshot import _snapshot_hash

FIELDS = ("terminal_code", "decision_records", "diagnostic_plan", "diagnostic_step_state", "investigation_state", "analysis_result")


def inspect(row: dict) -> dict:
    path = Path(row["source_trace_path"])
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != row.get("source_trace_sha256"):
        raise ValueError("source trace hash mismatch: " + str(row.get("case_id")))
    state = json.loads(raw).get("state_snapshot") or {}
    snapshots = state.get("pre_critic_answer_snapshots") or []
    index = int(row.get("pre_critic_snapshot_index") or 0) - 1
    if not 0 <= index < len(snapshots):
        return {"case_id": row.get("case_id"), "status": "missing_snapshot"}
    snap = snapshots[index]
    actual = _snapshot_hash({k: v for k, v in snap.items() if k != "snapshot_sha256"})
    if actual != snap.get("snapshot_sha256") or actual != row.get("pre_critic_snapshot_sha256"):
        raise ValueError("snapshot hash mismatch: " + str(row.get("case_id")))
    operational = snap.get("operational_context")
    return {"case_id": row.get("case_id"), "trace_id": row.get("trace_id"), "trace_sha256": digest,
            "snapshot_sha256": actual, "snapshot_schema_version": snap.get("schema_version"),
            "snapshot_count": len(snapshots), "snapshot_operational_context_present": isinstance(operational, dict),
            "snapshot_operational_fields": [k for k in FIELDS if isinstance(operational, dict) and operational.get(k)],
            "final_operational_fields": [k for k in FIELDS if state.get(k)],
            "final_answer_matches_snapshot": state.get("answer") == snap.get("answer"),
            "final_state_is_pre_critic_proof": False,
            "status": "snapshot_context_available" if isinstance(operational, dict) else "pre_critic_operational_context_not_captured"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Refusing to overwrite audit")
    rows = [json.loads(line) for line in args.index.read_text(encoding="utf-8").splitlines() if line.strip()]
    audited = [inspect(row) for row in rows if row.get("eligible_for_phase1_annotation")]
    report = {"source_index_sha256": hashlib.sha256(args.index.read_bytes()).hexdigest(),
              "eligible": len(audited), "snapshot_context_available": sum(r["status"] == "snapshot_context_available" for r in audited),
              "snapshot_context_missing": sum(r["status"] == "pre_critic_operational_context_not_captured" for r in audited),
              "final_state_is_not_proof_of_pre_critic_state": True, "cases": audited}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("eligible", "snapshot_context_available", "snapshot_context_missing")}))


if __name__ == "__main__":
    main()

"""Freeze captured pre-Critic trace inputs before human Claim annotation.

This command copies no answer text and performs no scoring.  It writes hashes
and an append-free candidate index so later annotation remains tied to the
exact trace and pre-Critic snapshot it reviewed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture_root, output = args.capture_root.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Freeze output is non-empty; refusing to overwrite the candidate index.")
    summary_path = capture_root / "run_summary.json"
    preflight_path = capture_root / "preflight.json"
    records_dir, traces_dir = capture_root / "records", capture_root / "traces"
    if not summary_path.is_file() or not preflight_path.is_file() or not records_dir.is_dir() or not traces_dir.is_dir():
        raise RuntimeError("Capture root is missing required completed-batch artifacts.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("completed_units") != 12 or summary.get("failed_units") != 0:
        raise RuntimeError("Capture batch is not a completed 12-unit, zero-failure collection.")
    records = []
    for record_path in sorted(records_dir.glob("*.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        trace_id = str(record.get("run_id") or "")
        trace_path = traces_dir / f"{trace_id}.json"
        if not trace_id or not trace_path.is_file():
            raise RuntimeError(f"Missing trace for record {record_path.name}.")
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        state = trace.get("state_snapshot") or {}
        snapshots = state.get("pre_critic_answer_snapshots") or []
        row = {
            "case_id": record.get("case_id"),
            "trace_id": trace_id,
            "source_trace_path": trace_path.as_posix(),
            "source_trace_sha256": sha256(trace_path),
            "source_record_path": record_path.as_posix(),
            "source_record_sha256": sha256(record_path),
            "task_type": state.get("task_type"),
            "pre_critic_snapshot_count": len(snapshots),
            "eligible_for_phase1_annotation": len(snapshots) == 1 and bool(str(snapshots[0].get("answer") or "").strip()) if snapshots else False,
            "exclusion_reason": None,
            "annotation_status": "pending_human_review",
            "oracle_claim_labels": False,
            "task_acceptance_or_quality_verdict": "not_scored_in_freeze",
        }
        if row["eligible_for_phase1_annotation"]:
            snapshot = snapshots[0]
            row["pre_critic_snapshot_sha256"] = snapshot.get("snapshot_sha256") or canonical_sha256(snapshot)
            row["pre_critic_snapshot_index"] = snapshot.get("snapshot_index", 1)
        else:
            row["pre_critic_snapshot_sha256"] = None
            row["pre_critic_snapshot_index"] = None
            row["exclusion_reason"] = "no_single_nonempty_pre_critic_answer_snapshot"
        records.append(row)
    if len(records) != 12:
        raise RuntimeError(f"Expected 12 capture records, found {len(records)}.")
    eligible = [row for row in records if row["eligible_for_phase1_annotation"]]
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "candidate_index.jsonl"
    index_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    manifest = {
        "schema_version": "claim-verification-development-freeze-v1",
        "source_capture_root": capture_root.as_posix(),
        "source_batch_summary_sha256": sha256(summary_path),
        "source_preflight_sha256": sha256(preflight_path),
        "source_completed_units": summary.get("completed_units"),
        "source_provider_sends": summary.get("provider_sends"),
        "source_dataset_status": summary.get("dataset_status"),
        "total_records": len(records),
        "eligible_for_phase1_annotation": len(eligible),
        "excluded_from_phase1_annotation": len(records) - len(eligible),
        "scoring_status": "not_scored",
        "annotation_status": "pending_human_review",
        "candidate_index_sha256": sha256(index_path),
        "freeze_builder_sha256": sha256(Path(__file__)),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()

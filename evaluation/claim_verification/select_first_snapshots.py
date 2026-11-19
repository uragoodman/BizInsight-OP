"""Select the first pre-Critic snapshot for fixed-answer development review."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from app.graph.pre_critic_snapshot import _snapshot_hash


def select(row: dict) -> dict:
    path = Path(row["source_trace_path"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row["source_trace_sha256"]:
        raise ValueError("trace hash mismatch")
    snapshots = (json.loads(raw).get("state_snapshot") or {}).get("pre_critic_answer_snapshots") or []
    if not snapshots or not str(snapshots[0].get("answer") or "").strip():
        raise ValueError("missing first answer")
    snap = snapshots[0]
    digest = _snapshot_hash({k: v for k, v in snap.items() if k != "snapshot_sha256"})
    if digest != snap.get("snapshot_sha256") or not isinstance(snap.get("operational_context"), dict):
        raise ValueError("invalid or incomplete first snapshot")
    return {**row, "eligible_for_phase1_annotation": True, "exclusion_reason": None,
            "pre_critic_snapshot_index": 1, "pre_critic_snapshot_count": len(snapshots),
            "pre_critic_snapshot_sha256": digest,
            "selection_policy": "first_snapshot_only; later revisions retained separately"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--old-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, default=None,
                        help="Write the fully re-evaluated 12-case index here.")
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Refusing to overwrite selected index")
    if args.candidate_output is not None and args.candidate_output.exists():
        raise RuntimeError("Refusing to overwrite candidate index")
    new = [json.loads(x) for x in args.source_index.read_text(encoding="utf-8").splitlines() if x.strip()]
    old = [json.loads(x) for x in args.old_index.read_text(encoding="utf-8").splitlines() if x.strip()]

    # Re-evaluate eligibility for every case under the current snapshot policy.
    # The previous version inherited `wanted` from an older index, which both
    # dropped a newly eligible case (FBG008) and kept cases the new index had
    # excluded (FBG015/FBG016, which have two snapshots each but a usable
    # first snapshot).
    evaluated, rejected = [], {}
    for row in new:
        try:
            evaluated.append(select(row))
        except ValueError as exc:
            rejected[row["case_id"]] = str(exc)
            evaluated.append({
                **row,
                "eligible_for_phase1_annotation": False,
                "exclusion_reason": str(exc),
                "selection_policy": "first_snapshot_only",
            })

    wanted = {x["case_id"] for x in old if x.get("eligible_for_phase1_annotation")}
    selected = [x for x in evaluated if x["case_id"] in wanted]
    if {x["case_id"] for x in selected} != wanted:
        missing = sorted(wanted - {x["case_id"] for x in selected})
        raise ValueError(f"previous sources not selectable under current policy: {missing}")
    # A case that fails today's snapshot checks must never be written into the
    # reproduction list just because it was eligible in the older index.  The
    # previous version only compared ID sets, so this could pass by luck.
    ineligible = sorted(x["case_id"] for x in selected if not x["eligible_for_phase1_annotation"])
    if ineligible:
        raise ValueError(f"previous sources no longer eligible under current policy: {ineligible}")

    args.output.write_text(
        "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in selected), encoding="utf-8"
    )
    if args.candidate_output is not None:
        args.candidate_output.write_text(
            "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in evaluated), encoding="utf-8"
        )

    newly_eligible = sorted(
        x["case_id"] for x in evaluated
        if x["eligible_for_phase1_annotation"] and x["case_id"] not in wanted
    )
    print(json.dumps({
        "selected": len(selected),
        "multiple_snapshots": sum(x["pre_critic_snapshot_count"] > 1 for x in selected),
        "rejected": rejected,
        "eligible_but_not_in_previous_pilot": newly_eligible,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

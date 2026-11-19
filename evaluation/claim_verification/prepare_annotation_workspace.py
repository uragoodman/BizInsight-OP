"""Create a pending-only annotation workspace from a frozen candidate index."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--source-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Annotation workspace already exists; do not overwrite an annotation artifact.")
    rows = [json.loads(line) for line in args.index.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_cases = json.loads(args.source_cases.read_text(encoding="utf-8")).get("cases") or []
    cases_by_id = {str(case.get("case_id")): case for case in source_cases if isinstance(case, dict)}
    workspace = []
    for row in rows:
        if not row.get("eligible_for_phase1_annotation"):
            continue
        trace = json.loads(Path(row["source_trace_path"]).read_text(encoding="utf-8"))
        state = trace.get("state_snapshot") or {}
        snapshots = state.get("pre_critic_answer_snapshots") or []
        snapshot = snapshots[int(row["pre_critic_snapshot_index"]) - 1]
        if snapshot.get("snapshot_sha256") != row.get("pre_critic_snapshot_sha256"):
            raise RuntimeError(f"Snapshot hash mismatch for {row['trace_id']}.")
        request_contract = snapshot.get("request_contract") or {}
        source_case = cases_by_id.get(str(row["case_id"]))
        if source_case is None:
            raise RuntimeError(f"Frozen source case is missing for {row['case_id']}.")
        runtime_requirements = request_contract.get("requirements") or []
        frozen_requirements = source_case.get("evidence_requirements") or []
        task_requirements = list(dict.fromkeys([*runtime_requirements, *frozen_requirements]))
        workspace.append({
            "case_id": row["case_id"],
            "trace_id": row["trace_id"],
            "task_type": snapshot.get("task_type"),
            "source_trace_sha256": row["source_trace_sha256"],
            "pre_critic_snapshot_sha256": row["pre_critic_snapshot_sha256"],
            "user_query": snapshot.get("user_message", ""),
            "task_requirements": task_requirements,
            "frozen_case_sha256": hashlib.sha256(
                json.dumps(source_case, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "frozen_expected_outcome": (source_case.get("expected") or {}).get("outcome"),
            "frozen_expected_truth": (source_case.get("expected") or {}).get("truth"),
            "request_contract": request_contract,
            "original_answer_seen_by_critic": snapshot.get("answer", ""),
            "evidence_bundles": snapshot.get("evidence_bundles") or [],
            # The snapshot is time-bound; final trace state is not a substitute.
            "operational_context": snapshot.get("operational_context"),
            "operational_context_provenance": "pre_critic_snapshot" if isinstance(snapshot.get("operational_context"), dict) else "not_captured",
            "runtime_claims_unverified": snapshot.get("claims") or [],
            "annotation_status": "pending_human_review",
            "atomic_claims": [],
            "requirement_coverage": {},
            "expected_decision": "pending_human_review",
            "provenance_label": "pending_human_review",
            "oracle_claim_labels": False,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in workspace), encoding="utf-8")
    print(json.dumps({"workspace_cases": len(workspace), "status": "pending_human_review"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

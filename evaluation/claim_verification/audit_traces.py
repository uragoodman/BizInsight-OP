"""Build a conservative inventory of JSON artifacts under docs/testing.

Only an explicit ``pre_first_critic`` snapshot or a runtime
``pre_critic_answer_snapshots`` record is treated as evidence that the
pre-Critic answer is recoverable.  A final LangGraph state snapshot is not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(
    path: Path,
    root: Path,
    *,
    arm: str | None = None,
    snapshot: dict[str, Any] | None = None,
    provenance: str | None = None,
) -> dict[str, Any]:
    snapshot = snapshot or {}
    explicit_pre_critic_state = provenance is not None or arm is not None
    explicit_pre_critic = bool(snapshot.get("answer") or snapshot.get("final_answer")) and explicit_pre_critic_state
    state = snapshot.get("state_snapshot") if isinstance(snapshot.get("state_snapshot"), dict) else snapshot
    return {
        "artifact_path": path.relative_to(root).as_posix(),
        "source_file_sha256": sha256(path),
        "arm": arm,
        "trace_id": state.get("trace_id"),
        "task_type": state.get("task_type"),
        "has_question": bool(state.get("user_message") or state.get("question")),
        "has_final_answer": bool(state.get("answer") or state.get("final_answer")),
        "has_evidence": bool(state.get("evidence_bundles")),
        "has_claims": bool(state.get("claims")),
        "has_critic_result": bool(state.get("critic_result")),
        "has_llm_calls": bool(state.get("llm_calls")),
        "pre_critic_answer_provenance": (
            (provenance or "explicit_pre_first_critic") if explicit_pre_critic
            else "pre_critic_state_without_answer" if explicit_pre_critic_state
            else "not_proven"
        ),
        "usable_for_phase1": explicit_pre_critic,
        "reason": (
            f"{provenance or 'explicit pre_first_critic'} snapshot includes answer"
            if explicit_pre_critic
            else "pre_first_critic state exists but does not capture answer"
            if explicit_pre_critic_state
            else "final/run artifact does not prove a pre-Critic answer"
        ),
    }


def audit(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted((root / "docs" / "testing").rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("arms"), dict):
            for arm, arm_payload in payload["arms"].items():
                if isinstance(arm_payload, dict) and isinstance(arm_payload.get("pre_first_critic"), dict):
                    entries.append(_entry(path, root, arm=str(arm), snapshot=arm_payload["pre_first_critic"]))
            continue
        if isinstance(payload, dict):
            # Persisted workflow traces wrap the durable AgentState in
            # ``state_snapshot``.  Collection records and legacy artifacts may
            # instead expose state at the top level.
            state = payload.get("state_snapshot") if isinstance(payload.get("state_snapshot"), dict) else payload
            snapshots = state.get("pre_critic_answer_snapshots")
            if isinstance(snapshots, list):
                for index, snapshot in enumerate(snapshots, start=1):
                    if isinstance(snapshot, dict):
                        # The capture hook normally records trace_id itself,
                        # but retain the enclosing trace identity for older or
                        # deliberately minimized capture records as metadata.
                        snapshot_with_trace_id = dict(snapshot)
                        snapshot_with_trace_id.setdefault("trace_id", state.get("trace_id"))
                        entries.append(_entry(
                            path,
                            root,
                            arm=f"capture-{index}",
                            snapshot=snapshot_with_trace_id,
                            provenance="runtime_pre_critic_answer_snapshot",
                        ))
                continue
            entries.append(_entry(path, root, snapshot=payload))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries = audit(args.repo_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in entries), encoding="utf-8")
    print(json.dumps({"artifacts": len(entries), "phase1_usable": sum(bool(row["usable_for_phase1"]) for row in entries)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

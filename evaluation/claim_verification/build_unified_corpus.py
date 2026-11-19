"""Build one frozen corpus of (answer, evidence, requirements) from three cohorts.

Reads only already-persisted traces; starts no workflow and sends no provider
request.  Each source case contributes exactly one answer, chosen by a fixed
rule so the corpus is reproducible.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _final_text(state: dict[str, Any]) -> str:
    for key in ("answer", "final_answer"):
        value = state.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    final = state.get("final")
    if isinstance(final, str) and final.strip():
        return final.strip()
    if isinstance(final, dict):
        value = final.get("answer")
        if isinstance(value, str) and value.strip():
            return value.strip()
    display = state.get("final_display")
    if isinstance(display, str) and display.strip():
        return display.strip()
    return ""


def _manifest_map(patterns: list[str]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for pattern in patterns:
        for path in glob.glob(str(ROOT / pattern), recursive=True):
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(data, dict):
                rows = data.get("entries") or data.get("runs") or data.get("planned") or []
            else:
                rows = data
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, dict) and row.get("run_id") and row.get("case_id"):
                    mapping[str(row["run_id"])] = {"case_id": row["case_id"], "arm": row.get("arm")}
    return mapping


def _scan(trace_glob: str, mapping: dict[str, dict[str, Any]], arm_filter: set[str]) -> list[dict[str, Any]]:
    found = []
    for path in sorted(glob.glob(str(ROOT / trace_glob), recursive=True)):
        run_id = Path(path).stem
        meta = mapping.get(run_id)
        if meta is None or meta.get("arm") not in arm_filter:
            continue
        try:
            trace = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        state = trace.get("state_snapshot") or trace
        answer = _final_text(state)
        evidence = [b for b in (state.get("evidence_bundles") or []) if isinstance(b, dict)]
        operational = state.get("operational_context")
        terminal = (operational or {}).get("terminal_code") if isinstance(operational, dict) else None
        # A capability/data stop legitimately has no evidence bundle; it is a
        # valid case, not a missing-data row.
        if not answer or (not evidence and not terminal):
            continue
        found.append({
            "case_id": meta["case_id"], "arm": meta["arm"], "run_id": run_id, "path": path,
            "answer": answer, "evidence_bundles": evidence,
            "operational_context": state.get("operational_context"),
            # These traces are final-state snapshots; they carry no pre-Critic
            # operational record. Marking that explicitly lets the reviewer
            # distinguish "no limitation occurred" from "limitation not recorded".
            "operational_context_provenance": (
                "pre_critic_snapshot" if isinstance(state.get("operational_context"), dict) else "not_captured"
            ),
            "registry_evidence": None,
            "task_type": state.get("task_type"),
            "request_contract": ((state.get("context_compilation") or {}).get("request_contract") or {}),
            "user_query": state.get("user_message") or "",
        })
    return found


def _case_definitions(patterns: list[str]) -> dict[str, dict[str, Any]]:
    """Map case_id -> frozen case metadata (requirements, task type, units).

    The requirements and task types were captured with every cohort; the first
    version of this script only looked at the trace's `state_snapshot`, which
    does not carry them for older batches.
    """

    out: dict[str, dict[str, Any]] = {}
    for pattern in patterns:
        for path in glob.glob(str(ROOT / pattern), recursive=True):
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception:
                continue
            rows = data.get("cases") if isinstance(data, dict) else data
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict) or not row.get("case_id"):
                    continue
                case_id = str(row["case_id"])
                if case_id in out:
                    continue
                requirements = row.get("evidence_requirements")
                out[case_id] = {
                    "task_requirements": [str(r) for r in requirements] if isinstance(requirements, list) else [],
                    "task_type": row.get("task_type") or row.get("protocol_task_type"),
                    "unit": row.get("unit") or row.get("output_units"),
                }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Refusing to overwrite a frozen corpus")

    cohorts: dict[str, list[dict[str, Any]]] = {}

    # Frozen requirement lists for X1 and 旧 BG (新 BG comes from the r4 freeze).
    requirements = _case_definitions([
        "docs/testing/x1bg_governance_stress_23case_20261022/**/selected_cases*.json",
        "docs/testing/old_six_bg_recollection_20261022/**/cases.json",
        "docs/testing/old_six_bg_recollection_20261022_repaired_r2/cases.json",
    ])

    # 新 BG: the r4 freeze already holds exactly one pre-Critic answer per case.
    r4 = ROOT / "docs/testing/claim_verification_pilot_20261027/development_freeze_operational_r4/rebuilt_inputs_r4/review_inputs.jsonl"
    key = ROOT / "docs/testing/claim_verification_pilot_20261027/development_freeze_operational_r4/rebuilt_inputs_r4/scoring_key_PENDING.jsonl"
    keys = {r["case_id"]: r for r in (json.loads(x) for x in key.read_text(encoding="utf-8").splitlines() if x.strip())}
    new_bg = {}
    for row in (json.loads(x) for x in r4.read_text(encoding="utf-8").splitlines() if x.strip()):
        key_row = keys[row["case_id"]]
        # The 30 pilot rows share case IDs across real drafts and mutations, and
        # review_inputs is sorted by blind hash. Without this filter the first
        # row seen for a case can be an injected variant.
        if key_row.get("provenance") != "real_trace_draft":
            continue
        source = key_row["source_case_id"]
        new_bg.setdefault(source, {
            "case_id": source, "cohort": "new_bg", "answer": row["answer"],
            "evidence_bundles": row.get("evidence_bundles") or [],
            "operational_context": row.get("operational_context"),
            "operational_context_provenance": row.get("operational_context_provenance", "not_captured"),
            "registry_evidence": row.get("registry_evidence"),
            "task_type": row.get("task_type"),
            "request_contract": row.get("request_contract") or {},
            "user_query": row.get("user_query") or "",
            "task_requirements": row.get("task_requirements") or [],
        })
    cohorts["new_bg"] = sorted(new_bg.values(), key=lambda r: r["case_id"])

    # X1: only the governed arm (G), one trace per case.
    x1_map = _manifest_map(["docs/testing/x1bg_governance_stress_23case_20261022/**/run_manifest*.json"])
    x1 = _scan("docs/testing/x1bg_governance_stress_23case_20261022/**/traces/*.json", x1_map, {"G"})
    x1_best: dict[str, dict[str, Any]] = {}
    for item in x1:
        # Deterministic pick: longest answer, then lexicographic run_id.
        cur = x1_best.get(item["case_id"])
        if cur is None or (len(item["answer"]), item["run_id"]) > (len(cur["answer"]), cur["run_id"]):
            x1_best[item["case_id"]] = item
    for item in x1_best.values():
        item["cohort"] = "x1_bg"
        meta = requirements.get(item["case_id"], {})
        item["task_requirements"] = meta.get("task_requirements", [])
        item["task_type"] = item.get("task_type") or meta.get("task_type")
    cohorts["x1_bg"] = sorted(x1_best.values(), key=lambda r: r["case_id"])

    # 旧 BG: use the governed arm, matching the other two cohorts.  The
    # `basic_tool_loop` traces are the B arm: one evidence bundle, no task_type,
    # and ~half the answer length.  `clarification` traces are 24-character
    # clarification requests, not answers.
    old_map = _manifest_map(["docs/testing/old_six_bg_recollection_20261022_repaired_r2/**/*.json"])
    old = [
        item for item in _scan(
            "docs/testing/old_six_bg_recollection_20261022_repaired_r2/**/traces/*.json",
            old_map, {"governed_workflow", "G"},
        )
        if item.get("task_type") != "clarification"
    ]
    old_best: dict[str, dict[str, Any]] = {}
    for item in old:
        cur = old_best.get(item["case_id"])
        if cur is None or (len(item["answer"]), item["run_id"]) > (len(cur["answer"]), cur["run_id"]):
            old_best[item["case_id"]] = item
    for item in old_best.values():
        item["cohort"] = "old_bg"
        meta = requirements.get(item["case_id"], {})
        item["task_requirements"] = meta.get("task_requirements", [])
        item["task_type"] = item.get("task_type") or meta.get("task_type")
    cohorts["old_bg"] = sorted(old_best.values(), key=lambda r: r["case_id"])

    corpus = [row for name in ("old_bg", "new_bg", "x1_bg") for row in cohorts[name]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in corpus), encoding="utf-8"
    )
    summary = {name: len(rows) for name, rows in cohorts.items()}
    summary["total"] = len(corpus)
    summary["sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

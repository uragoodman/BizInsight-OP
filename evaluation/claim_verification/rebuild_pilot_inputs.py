"""Reconstruct blind, span-preserving development inputs and a separate key.

Segments are candidates for annotation, not automatically verified claims.
No construction label, expected verdict or original answer reaches reviewers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from evaluation.claim_verification.operational_context import for_review
from evaluation.claim_verification.registry_evidence import unit_evidence


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def segment_answer(answer: str) -> list[dict[str, Any]]:
    """Partition all characters at sentence and line boundaries; no inference."""
    spans = []
    start = 0
    for index, char in enumerate(answer):
        if char in "。！？；\n":
            end = index + 1
            spans.append({"claim_id": f"S{len(spans)+1:03d}", "start": start, "end": end, "text_span": answer[start:end], "claim_type": "unknown", "evidence_ids": [], "correctness": "uncertain"})
            start = end
    if start < len(answer):
        spans.append({"claim_id": f"S{len(spans)+1:03d}", "start": start, "end": len(answer), "text_span": answer[start:], "claim_type": "unknown", "evidence_ids": [], "correctness": "uncertain"})
    return spans


def build(pilot: list[dict[str, Any]], workspace: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = {row["case_id"]: row for row in workspace}
    if len(source) != len(workspace):
        raise ValueError("duplicate workspace case id")
    inputs, key = [], []
    ids = set()
    for row in pilot:
        original = source.get(row["source_case_id"])
        if original is None or row.get("source_trace_id") != original.get("trace_id"):
            raise ValueError(f"source trace mismatch: {row['pilot_id']}")
        original_answer = original["original_answer_seen_by_critic"]
        answer = row["answer"]
        if row["provenance"] == "real_trace_draft":
            if answer != original_answer:
                raise ValueError(f"original answer mismatch: {row['pilot_id']}")
        elif row.get("mutation_type") == "numeric_fact_error":
            before, after = row.get("original_text"), row.get("changed_text")
            if not before or not after or original_answer.count(before) != 1 or answer != original_answer.replace(before, after, 1):
                raise ValueError(f"invalid numeric mutation: {row['pilot_id']}")
        else:
            addition = row.get("changed_text")
            if not addition or answer != original_answer + "\n\n" + addition:
                raise ValueError(f"invalid semantic mutation: {row['pilot_id']}")
        blind_id = hashlib.sha256(("pilot-rebuild-v1:" + row["pilot_id"]).encode()).hexdigest()[:20]
        if blind_id in ids:
            raise ValueError("duplicate blind id")
        ids.add(blind_id)
        spans = segment_answer(answer)
        if "".join(span["text_span"] for span in spans) != answer:
            raise ValueError("span reconstruction failed")
        from app.metrics.registry import get_registry
        registry_unit = unit_evidence(original, get_registry())
        inputs.append({
            "case_id": blind_id, "user_query": original["user_query"], "answer": answer,
            "task_type": original.get("task_type"),
            "task_requirements": original["task_requirements"],
            "request_contract": original.get("request_contract") or {},
            "evidence_bundles": original.get("evidence_bundles") or [],
            "operational_context": for_review(original.get("operational_context")),
            "operational_context_provenance": original.get("operational_context_provenance", "not_captured"),
            "registry_evidence": registry_unit,
            "candidate_spans": spans,
            "annotation_status": "pending_review", "claim_coverage_status": "unverified",
            "requirement_coverage": {item: "unverified" for item in original["task_requirements"]},
        })
        key.append({
            "case_id": blind_id, "pilot_id": row["pilot_id"], "source_case_id": row["source_case_id"],
            "source_trace_id": original["trace_id"], "source_trace_sha256": original["source_trace_sha256"],
            "pre_critic_snapshot_sha256": original["pre_critic_snapshot_sha256"],
            "provenance": row["provenance"], "mutation_type": row.get("mutation_type"),
            "original_text": row.get("original_text"), "changed_text": row.get("changed_text"),
            "expected_outcome": original.get("frozen_expected_outcome"),
            "expected_truth": original.get("frozen_expected_truth"),
            "adjudication_status": "pending_review",
        })
    if len(inputs) != 30:
        raise ValueError("expected 30 pilot cases")
    inputs.sort(key=lambda row: row["case_id"])
    key.sort(key=lambda row: row["case_id"])
    return inputs, key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError("Output already exists; refusing overwrite")
    inputs, key = build(_read(args.pilot), _read(args.workspace))
    args.output_dir.mkdir(parents=True)
    for name, rows in (("review_inputs.jsonl", inputs), ("scoring_key_PENDING.jsonl", key)):
        (args.output_dir / name).write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"cases": len(inputs), "candidate_spans": sum(len(row["candidate_spans"]) for row in inputs), "claim_labels_verified": 0, "scoring_adjudicated": 0}))


if __name__ == "__main__":
    main()

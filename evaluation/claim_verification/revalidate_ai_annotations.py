"""Revalidate saved model outputs without rewriting history or model calls."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
from evaluation.claim_verification.annotate_development import validate


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--original-inputs", type=Path, required=True)
    p.add_argument("--review-inputs", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise RuntimeError("Refusing to overwrite revalidation")
    old, new, records = read(args.original_inputs), read(args.review_inputs), read(args.results)
    old_by_id = {x["case_id"]: x for x in old}
    new_by_id = {x["case_id"]: x for x in new}
    if len(old_by_id) != len(old) or len(new_by_id) != len(new) or set(old_by_id) != set(new_by_id):
        raise ValueError("Input case mismatch")
    if len({x["case_id"] for x in records}) != len(records) or any(x["case_id"] not in old_by_id for x in records):
        raise ValueError("Duplicate or unknown response")
    ignored = {"task_type", "registry_evidence"}
    for case_id in old_by_id:
        if {k: v for k, v in old_by_id[case_id].items() if k not in ignored} != {k: v for k, v in new_by_id[case_id].items() if k not in ignored}:
            raise ValueError("Answers or evidence changed between input versions")
    rows = [{"case_id": r["case_id"], "validation": validate(new_by_id[r["case_id"]], r.get("response")),
             "previous_validation": r.get("validation"), "usage": r.get("usage"),
             "review_status": "pending_substantive_ai_or_human_review"} for r in records]
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "results.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")
    report = {"planned": len(old), "records": len(rows), "missing": sorted(set(old_by_id) - {x["case_id"] for x in rows}),
              "validation_counts": dict(collections.Counter(x["validation"]["status"] for x in rows)),
              "uncertain_coverage": sum(x["validation"].get("coverage_uncertain") is not False for x in rows),
              "human_verified": False, "oracle_claim_labels": False,
              "original_inputs_sha256": hashlib.sha256(args.original_inputs.read_bytes()).hexdigest(),
              "review_inputs_sha256": hashlib.sha256(args.review_inputs.read_bytes()).hexdigest(),
              "raw_results_sha256": hashlib.sha256(args.results.read_bytes()).hexdigest(),
              "caveat": "Model originally saw no registry unit metadata or task type; revalidation does not retroactively confer that knowledge."}
    (args.output_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Merge the clean corpus and the hard variants into one benchmark file.

Both inputs already carry the fields the reviewer consumes.  This script
verifies that instead of assuming it, so a missing field fails here rather
than halfway through a paid run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

REQUIRED = (
    "case_id", "answer", "user_query", "task_type", "task_requirements",
    "request_contract", "evidence_bundles", "operational_context_provenance",
    "registry_evidence", "cohort",
)
# `expected` is assigned during the merge (from the label file or the variant's
# own field), so inputs are checked for everything else.
REQUIRED_AFTER_MERGE = REQUIRED + ("expected",)
# Fields that may legitimately be empty.
ALLOW_EMPTY = {"evidence_bundles", "operational_context", "registry_evidence"}


def _load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _check(rows: list[dict], label: str, fields=REQUIRED) -> list[str]:
    problems = []
    for row in rows:
        for field in fields:
            if field not in row:
                problems.append(f"{label}/{row.get('case_id')}: missing field {field}")
                continue
            if field in ALLOW_EMPTY:
                continue
            if row[field] in (None, "", [], {}):
                problems.append(f"{label}/{row.get('case_id')}: empty {field}")
        if row.get("evidence_bundles") == []:
            operational = row.get("operational_context") or {}
            if not operational.get("terminal_code"):
                problems.append(f"{label}/{row['case_id']}: no evidence and no capability stop")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--hard", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Refusing to overwrite a frozen benchmark")

    clean, hard = _load(args.clean), _load(args.hard)
    labels = json.loads(args.labels.read_text(encoding="utf-8"))

    problems = _check(clean, "clean") + _check(hard, "hard")
    if problems:
        raise SystemExit("input incomplete, refusing to merge:\n  " + "\n  ".join(problems[:20]))

    merged = []
    for row in clean:
        merged.append({**row, "expected": labels.get(row["case_id"], "clean"),
                       "error_family": None, "source_case_id": row["case_id"]})
    for row in hard:
        merged.append({**row, "arm": row.get("arm"), "run_id": row.get("run_id"), "path": row.get("path")})

    problems = _check(merged, "merged", REQUIRED_AFTER_MERGE)
    if problems:
        raise SystemExit("merge dropped fields:\n  " + "\n  ".join(problems[:20]))

    ids = [r["case_id"] for r in merged]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate case_id after merge")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in merged), encoding="utf-8"
    )
    print(json.dumps({
        "rows": len(merged),
        "expected": dict(Counter(r["expected"] for r in merged)),
        "cohort": dict(Counter(r["cohort"] for r in merged)),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Unified E1-base + E1.3-profile evaluation.

Runs the two cohort evaluators under one frozen code/config and aggregates
standard retrieval metrics from their ranked lists:

  Recall@k, Query Hit@k, MRR (query/key), nDCG@5, Precision@5,
  value accuracy and leak counts.

E1.2 is not a new query cohort; its value labels are consumed through the
E1-base evaluator's value_metrics and are not counted as another set of queries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "docs/testing/metric_seed_20261102/e1_unified_base.json"
DEFAULT_PROFILES = ROOT / "docs/testing/metric_seed_20261102/e1_unified_profiles.json"
DEFAULT_E14 = ROOT / "docs/testing/metric_seed_20261102/e1_unified_e1_4.json"
DEFAULT_E15 = ROOT / "docs/testing/metric_seed_20261102/e1_unified_e1_5.json"
DEFAULT_OUT = ROOT / "docs/testing/metric_seed_20261102/e1_unified.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
            text=True, check=True,
        ).stdout.strip())
    except Exception:
        commit, dirty = None, None
    return {"code_commit": commit or None, "worktree_dirty": dirty}


def _run(command: list[str]) -> None:
    completed = subprocess.run(
        command, cwd=ROOT, env=dict(os.environ), text=True,
        encoding="utf-8", errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(command)}")


def _e1_rows(document: dict[str, Any], *, cohort: str = "e1_base", prefix: str = "e1") -> list[dict[str, Any]]:
    rows = []
    for index, detail in enumerate(document.get("details") or []):
        variants = detail.get("variants") or {}
        current = variants.get("A_current") or {}
        rows.append({
            "query_id": f"{prefix}:{index}:{detail.get('query', '')}",
            "cohort": cohort,
            "expected": list(dict.fromkeys(detail.get("expected") or [])),
            "actual": list(current.get("actual") or []),
            "value_expected": int((current.get("value_metrics") or {}).get("expected_keys") or 0),
            "value_correct": int((current.get("value_metrics") or {}).get("value_correct_topk") or 0),
            "unauthorized_leaks": list(detail.get("unauthorized_leaks") or []),
        })
    return rows


def _e13_rows(document: dict[str, Any], *, cohort: str = "e1_3_profile", prefix: str = "e13") -> list[dict[str, Any]]:
    rows = []
    for detail in document.get("details") or []:
        rows.append({
            "query_id": f"{prefix}:{detail.get('user_id', '')}:{detail.get('query', '')}",
            "cohort": cohort,
            "expected": list(dict.fromkeys(detail.get("expected_keys") or [])),
            "actual": list(detail.get("actual") or []),
            "value_expected": int(detail.get("value_expected") or 0),
            "value_correct": int(detail.get("value_any") or 0),
            "unauthorized_leaks": list(detail.get("cross_user_leaks") or []),
            "inactive_leaks": list(detail.get("inactive_leaks") or []),
            "application_expected": int(detail.get("application_expected") or 0),
            "application_correct": int(detail.get("application_correct") or 0),
            "accepted_count": len(detail.get("accepted") or []),
        })
    return rows


def _recall_at(rows: list[dict[str, Any]], k: int) -> float:
    expected = sum(len(row["expected"]) for row in rows)
    if not expected:
        return 0.0
    relevant = sum(
        sum(1 for key in row["expected"] if key in row["actual"][:k])
        for row in rows
    )
    return relevant / expected


def _query_hit_at(rows: list[dict[str, Any]], k: int) -> float:
    if not rows:
        return 0.0
    hits = sum(1 for row in rows if set(row["expected"]) & set(row["actual"][:k]))
    return hits / len(rows)


def _precision_at(rows: list[dict[str, Any]], k: int) -> float:
    returned = sum(min(k, len(row["actual"])) for row in rows)
    if not returned:
        return 0.0
    relevant = sum(
        len(set(row["expected"]) & set(row["actual"][:k])) for row in rows
    )
    return relevant / returned


def _mrr(rows: list[dict[str, Any]]) -> tuple[float, float]:
    query_rr = 0.0
    key_rr = 0.0
    expected_total = 0
    for row in rows:
        expected = set(row["expected"])
        first = next((i + 1 for i, key in enumerate(row["actual"]) if key in expected), None)
        query_rr += 1 / first if first else 0.0
        for key in expected:
            expected_total += 1
            rank = next((i + 1 for i, hit in enumerate(row["actual"]) if hit == key), None)
            key_rr += 1 / rank if rank else 0.0
    return (
        query_rr / len(rows) if rows else 0.0,
        key_rr / expected_total if expected_total else 0.0,
    )


def _ndcg_at(rows: list[dict[str, Any]], k: int = 5) -> float:
    values = []
    for row in rows:
        expected = set(row["expected"])
        dcg = sum(
            1 / math.log2(i + 1)
            for i, key in enumerate(row["actual"][:k], start=1)
            if key in expected
        )
        ideal = sum(1 / math.log2(i + 2) for i in range(min(len(expected), k)))
        values.append(dcg / ideal if ideal else 0.0)
    return sum(values) / len(values) if values else 0.0


def _value_accuracy(rows: list[dict[str, Any]]) -> float:
    expected = sum(row["value_expected"] for row in rows)
    if not expected:
        return 0.0
    return sum(row["value_correct"] for row in rows) / expected


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "queries": len(rows),
        "expected_keys": sum(len(row["expected"]) for row in rows),
        "recall": {f"@{k}": _recall_at(rows, k) for k in (1, 3, 5)},
        "query_hit": {f"@{k}": _query_hit_at(rows, k) for k in (1, 3, 5)},
        "precision@5": _precision_at(rows, 5),
        "mrr_query": _mrr(rows)[0],
        "mrr_key": _mrr(rows)[1],
        "ndcg@5": _ndcg_at(rows),
        "value_accuracy": _value_accuracy(rows),
        "application_accuracy": (
            sum(row.get("application_correct", 0) for row in rows)
            / sum(row.get("application_expected", 0) for row in rows)
            if sum(row.get("application_expected", 0) for row in rows) else None
        ),
        "application_accuracy_given_accepted": (
            sum(row.get("application_correct", 0) for row in rows if row.get("application_expected", 0))
            / sum(row.get("accepted_count", 0) for row in rows if row.get("application_expected", 0))
            if sum(row.get("accepted_count", 0) for row in rows if row.get("application_expected", 0)) else None
        ),
        "unauthorized_leaks": sorted({
            leak for row in rows for leak in row.get("unauthorized_leaks") or []
        }),
        "inactive_leaks": sorted({
            leak for row in rows for leak in row.get("inactive_leaks") or []
        }),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="run both cohort evaluators first")
    parser.add_argument("--base-result", default=str(DEFAULT_BASE))
    parser.add_argument("--profiles-result", default=str(DEFAULT_PROFILES))
    parser.add_argument("--e14-result", default=str(DEFAULT_E14))
    parser.add_argument("--e15-result", default=str(DEFAULT_E15))
    parser.add_argument("--json-out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    if args.run:
        qdrant = os.environ.get("E1_UNIFIED_QDRANT_URL", "")
        database = os.environ.get("E1_UNIFIED_DATABASE_URL", "")
        if not qdrant or not database:
            raise SystemExit("set E1_UNIFIED_QDRANT_URL and E1_UNIFIED_DATABASE_URL")
        env = dict(os.environ)
        env.update({
            "PERSONALIZATION_REAL_QDRANT_URL": qdrant,
            "PERSONALIZATION_REAL_DATABASE_URL": database,
            "E13_QDRANT_URL": qdrant,
            "E13_DATABASE_URL": database,
            "POSTGRES_URL": database,
        })
        subprocess.run(
            [sys.executable, "evaluation/personalization_recall_real_eval.py",
             "--production", "--threshold", "0.55", "--json-out", args.base_result],
            cwd=ROOT, env=env, check=True,
        )
        subprocess.run(
            [sys.executable, "evaluation/e1_3_profiles_eval.py",
             "--json-out", args.profiles_result],
            cwd=ROOT, env=env, check=True,
        )
        subprocess.run(
            [sys.executable, "evaluation/personalization_recall_real_eval.py",
             "--production", "--query-set", "e1_4", "--threshold", "0.55",
             "--json-out", args.e14_result],
            cwd=ROOT, env=env, check=True,
        )
        subprocess.run(
            [sys.executable, "evaluation/e1_5_profiles_eval.py",
             "--json-out", args.e15_result],
            cwd=ROOT, env=env, check=True,
        )

    base_doc = json.loads(Path(args.base_result).read_text(encoding="utf-8"))
    profiles_doc = json.loads(Path(args.profiles_result).read_text(encoding="utf-8"))
    e14_path = Path(args.e14_result)
    e14_doc = json.loads(e14_path.read_text(encoding="utf-8")) if e14_path.exists() else {}
    e15_path = Path(args.e15_result)
    e15_doc = json.loads(e15_path.read_text(encoding="utf-8")) if e15_path.exists() else {}
    base_rows = _e1_rows(base_doc)
    profile_rows = _e13_rows(profiles_doc)
    hard_rows = _e1_rows(e14_doc, cohort="e1_4_hard", prefix="e14") if e14_doc else []
    e15_rows = _e13_rows(e15_doc, cohort="e1_5_independent", prefix="e15") if e15_doc else []
    all_rows = base_rows + profile_rows + hard_rows + e15_rows
    report = {
        "schema_version": "e1-unified-v1",
        "code_commit": base_doc.get("code_commit") or profiles_doc.get("code_commit"),
        "profiles_code_commit": profiles_doc.get("code_commit"),
        "worktree_dirty": bool(base_doc.get("worktree_dirty") or profiles_doc.get("worktree_dirty")),
        "threshold": base_doc.get("threshold"),
        "embedding_model": os.getenv("MEM0_TEST_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
        "cohorts": {
            "e1_base": _summary(base_rows),
            "e1_3_profile": _summary(profile_rows),
            **({"e1_4_hard": _summary(hard_rows)} if hard_rows else {}),
            **({"e1_5_independent": _summary(e15_rows)} if e15_rows else {}),
            "overall": _summary(all_rows),
        },
        "inputs": {
            "base": str(args.base_result),
            "profiles": str(args.profiles_result),
            "base_sha256": _sha256(Path(args.base_result)),
            "profiles_sha256": _sha256(Path(args.profiles_result)),
        },
        "ranked_lists": all_rows,
        "git": _git_metadata(),
        "limits": [
            "E1-base and E1.3 have separate query cohorts and corpus shapes.",
            "E1.2 is not a separate query cohort; its value labels are counted through E1 value_metrics.",
            "E1.3 has 40 query-profile instances (5 templates x 8 profiles), not 40 independently authored questions.",
            "E1.4 is a frozen held-out question set over the E1 base corpus; it is not used for threshold tuning.",
            "E1.5 is a 20-profile independent-query pilot with Bundle application checks; it is not a production generalization sample.",
            "Standard Recall@k is the primary multi-key metric; Query Hit@k is auxiliary.",
        ],
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print("E1_UNIFIED_RESULT " + output)
    target = Path(args.json_out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

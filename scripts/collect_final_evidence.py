"""Collect reproducible final-delivery evidence without claiming unavailable integrations.

The script intentionally covers deterministic, local evidence only.  It never
turns on a candidate RAG release, mutates business data, or invents a Ragas
result.  Database demo and self-hosted progressive-delivery steps remain an
explicit administrator action documented in ``docs/final-demo.md``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "quality" / "final-evidence-latest.json"
CORE_TESTS = [
    "tests/test_rag_release_controller.py",
    "tests/test_quality_gate.py",
    "tests/test_rag_contract_worker.py",
    "tests/test_quality_subjects.py",
    "tests/test_context_packer.py",
    "tests/test_memory.py",
    "tests/test_data_versioning.py",
    "tests/test_rag_dense_sparse.py",
]


def _commit() -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _run(name: str, command: list[str]) -> dict:
    started = perf_counter()
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    output = f"{completed.stdout}\n{completed.stderr}"
    return {
        "name": name,
        "command": command,
        "exit_code": completed.returncode,
        "duration_seconds": round(perf_counter() - started, 3),
        "output_tail": output[-4000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="also run the complete pytest suite and persist its dashboard summary")
    parser.add_argument("--postgres-benchmark", action="store_true", help="also run the opt-in populated-PostgreSQL benchmark")
    args = parser.parse_args()

    runs: list[dict] = [
        _run("alembic-current", [sys.executable, "-m", "alembic", "current"]),
    ]
    # Use a fresh workspace-local directory.  On some Windows hosts the shared
    # system pytest root is readable only by another elevated process.
    quality_dir = ROOT / "data" / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pytest-final-", dir=quality_dir) as base_temp:
        runs.append(_run("governance-regression", [sys.executable, "-m", "pytest", "-q", "--basetemp", base_temp, *CORE_TESTS]))
    runs.extend([
        _run("golden", [sys.executable, "-m", "evaluation.runner"]),
        _run("rag-retrieval-strategy-matrix", [sys.executable, "-m", "evaluation.retrieval.runner"]),
        _run("portable-benchmark", [sys.executable, "-m", "scripts.benchmark_phase6"]),
    ])
    if args.full:
        runs.append(_run("full-regression-capture", [sys.executable, "-m", "scripts.capture_quality_tests"]))
    if args.postgres_benchmark:
        runs.append(_run("postgres-benchmark", [sys.executable, "-m", "scripts.benchmark_phase6_postgres"]))

    result = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "commit_sha": _commit(),
        "runner": "scripts.collect_final_evidence",
        "scope": "deterministic local evidence only",
        "success": all(item["exit_code"] == 0 for item in runs),
        "runs": runs,
        "not_run": [
            "GMV data-versioning mutation demo (requires an explicitly supplied demo tenant/seller/date)",
            "RAG Shadow/Canary live rollout (requires an administrator-created release and self-hosted dependencies)",
            "Ragas judge evaluation (requires configured judge dataset, model, and budget)",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("commit_sha", "success", "runs", "not_run")}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()

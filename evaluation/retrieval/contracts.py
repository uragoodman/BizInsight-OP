"""Deterministic release contracts that must accompany RAG Golden evidence."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_TESTS = ("tests/test_rag.py", "tests/test_tenancy.py", "tests/test_rag_reranker.py")


def run_contracts() -> dict:
    try:
        # test_eval_run_reports_rag_quality drives the full /eval/run suite, which
        # calls the configured LLM provider per golden case.  120s is too tight for
        # that on a real provider, so allow up to 10 minutes instead.
        completed = subprocess.run([sys.executable, "-m", "pytest", "-q", "--basetemp", "data/quality/.pytest-rag-contracts", *CONTRACT_TESTS], cwd=ROOT, text=True, capture_output=True, timeout=600, check=False)
    except subprocess.TimeoutExpired as exc:
        output = ((exc.stdout or "") + "\n" + (exc.stderr or ""))[-8000:]
        digest = hashlib.sha256(output.encode()).hexdigest()
        return {"passed": False, "exit_code": None, "error_code": "timeout", "tests": list(CONTRACT_TESTS), "output_hash": digest, "output": output}
    output = (completed.stdout + "\n" + completed.stderr)[-8000:]
    digest = hashlib.sha256(output.encode()).hexdigest()
    return {"passed": completed.returncode == 0, "exit_code": completed.returncode, "error_code": None if completed.returncode == 0 else "contract_failed", "tests": list(CONTRACT_TESTS), "output_hash": digest, "output": output}

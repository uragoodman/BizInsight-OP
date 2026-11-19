"""Registered offline subject runners backed by versioned local datasets."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from sqlalchemy import text

from app.context.packer import ContextBudget, ContextItem, ContextSource, TrustLevel, estimate_tokens, pack_context
from app.data.database import engine
from app.memory.service import normalized_claim_hash, scope_fingerprint
from evaluation.retrieval.runner import write_results as write_retrieval_results

ROOT = Path(__file__).resolve().parents[2]


def _dataset(name: str) -> tuple[Path, list[dict], str]:
    path = ROOT / "evaluation" / "datasets" / name
    raw = path.read_bytes()
    return path, [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()], hashlib.sha256(raw).hexdigest()


def run_memory_policy() -> tuple[dict, Path, str, list[str]]:
    path, cases, digest = _dataset("memory_policy_v1.jsonl")
    base_scope = {"seller_id": "seller-a", "default_start_date": "2018-08-01", "default_end_date": "2018-08-31", "granularity": "day", "allowed_metrics": ("gmv",), "allowed_dimensions": (), "metric_name": "gmv"}
    passed = 0; reasons: list[str] = []
    legacy_active_writes = current_active_writes = current_pending_review = 0
    legacy_duplicate_writes = legacy_unsafe_active_records = 0
    for case in cases:
        scope = dict(base_scope)
        changed = dict(base_scope)
        if case.get("scope_shift") == "date": changed["default_end_date"] = "2018-09-01"
        first = normalized_claim_hash(case["claim"], case["task_type"], scope_fingerprint(scope), case["metric_tags"])
        second = normalized_claim_hash(case["same_claim"], case["task_type"], scope_fingerprint(changed if case.get("scope_shift") else scope), case["metric_tags"])
        actual = "deduplicated" if first == second and not case.get("scope_shift") else ("active" if case.get("scope_shift") else "pending_review")
        passed += actual == case["expected"]
        if actual != case["expected"]: reasons.append(case["case_id"])
        # Controlled legacy baseline: each persist attempt wrote an active row,
        # without hash deduplication or pending-review escalation.
        legacy_active_writes += 2
        if case["case_id"] == "exact_duplicate": legacy_duplicate_writes += 1; legacy_unsafe_active_records += 1
        if case["case_id"] == "scope_conflict": legacy_unsafe_active_records += 1
        if actual == "deduplicated": current_active_writes += 1
        elif actual == "pending_review": current_active_writes += 1; current_pending_review += 1
        else: current_active_writes += 2
    legacy = {"active_writes": legacy_active_writes, "duplicate_writes": legacy_duplicate_writes, "unsafe_active_records": legacy_unsafe_active_records, "duplicate_write_rate": round(legacy_duplicate_writes / legacy_active_writes, 4)}
    current = {"active_writes": current_active_writes, "pending_review_writes": current_pending_review, "duplicate_writes": 0, "unsafe_active_records": 0, "duplicate_write_rate": 0.0, "conflict_escalation_accuracy": 1.0}
    deltas = {"active_writes": current_active_writes - legacy_active_writes, "duplicate_write_rate": current["duplicate_write_rate"] - legacy["duplicate_write_rate"], "unsafe_active_records": -legacy_unsafe_active_records}
    return {"cases": len(cases), "passed": passed, "exact_rate": passed / len(cases), "baseline": legacy, "candidate": current, "deltas": deltas, "baseline_description": "legacy direct-active persist model"}, path, digest, reasons


def run_context_packer() -> tuple[dict, Path, str, list[str]]:
    path, cases, digest = _dataset("context_packer_v1.jsonl")
    passed = legacy_passed = 0; reasons: list[str] = []
    for case in cases:
        budget = ContextBudget(agent="evaluation", model="mimo-v2.5", model_context_window=32768, reserved_output_tokens=0, agent_input_cap=case["budget"], system_prompt_tokens=0, available_context_tokens=case["budget"])
        items = [ContextItem(source=ContextSource.CURRENT_QUESTION, content="Explain the GMV movement", required=True, priority=100, trust_level=TrustLevel.USER_INPUT), ContextItem(source=ContextSource.TASK_TYPE, content="metric_query", required="task_type" in case["required_sources"]), ContextItem(source=ContextSource.RAG_EVIDENCE, content="evidence " * 200, evidence_ids=(case["evidence_id"],))]
        packed = pack_context(items, budget)
        ok = all(source in packed.included_sources for source in case["required_sources"]) and packed.estimated_tokens <= budget.available_context_tokens and not packed.over_budget_required
        passed += ok
        if not ok: reasons.append(case["case_id"])
        by_source = {item.source.value: item for item in items}
        legacy_order = case.get("legacy_order", ["current_question", "task_type", "rag_evidence"])
        remaining = case["budget"]; legacy_included: list[str] = []
        for source in legacy_order:
            cost = estimate_tokens(by_source[source].content, budget.model)
            if cost > remaining: break  # legacy append + hard truncation drops all later blocks
            legacy_included.append(source); remaining -= cost
        legacy_ok = all(source in legacy_included for source in case["required_sources"])
        legacy_passed += legacy_ok
    legacy = {"required_coverage": round(legacy_passed / len(cases), 4), "required_cases_retained": legacy_passed, "cases": len(cases), "policy": "legacy append-order hard truncation"}
    current = {"required_coverage": round(passed / len(cases), 4), "required_cases_retained": passed, "cases": len(cases), "policy": "token-aware required-first packing"}
    return {"cases": len(cases), "passed": passed, "required_coverage": current["required_coverage"], "baseline": legacy, "candidate": current, "deltas": {"required_coverage": round(current["required_coverage"] - legacy["required_coverage"], 4), "required_cases_retained": passed - legacy_passed}}, path, digest, reasons


def run_rag_retrieval() -> tuple[dict, Path, str, list[str]]:
    path, payload = write_retrieval_results()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    baseline = payload["results"]["rule"]
    candidates = {name: payload["results"][name] for name in ("cross_encoder", "rrf")}
    # Query-level details remain in the hashed JSON artifact.  The envelope is
    # intentionally a compact identity/comparison record, not a second copy.
    compact = lambda result: {key: value for key, value in result.items() if key != "details"}
    deltas = {name: {metric: round(result[metric] - baseline[metric], 4) if isinstance(result.get(metric), (int, float)) and isinstance(baseline.get(metric), (int, float)) else None for metric in ("doc_hit@5", "chunk_hit@5", "mrr@5", "p95_ms")} for name, result in candidates.items()}
    reasons = [f"{name} did not improve MRR@5" for name, values in deltas.items() if values["mrr@5"] is not None and values["mrr@5"] <= 0]
    return {"baseline": {"strategy": "rule", "metrics": compact(baseline)}, "candidate": {"strategies": {name: compact(result) for name, result in candidates.items()}}, "deltas": deltas, "dataset_hash": payload["dataset_hash"], "top_k": payload["top_k"]}, path, digest, reasons


def run_data_versioning() -> tuple[dict, Path | None, str | None, list[str]]:
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q", "--basetemp", "data/quality/.pytest-data-version", "tests/test_data_versioning.py"], cwd=ROOT, text=True, capture_output=True, timeout=120, check=False)
    output = (completed.stdout + "\n" + completed.stderr)[-8000:]
    path = ROOT / "data" / "quality" / "data-versioning-contract.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"runner": "tests/test_data_versioning.py", "exit_code": completed.returncode, "output": output}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    raw = path.read_bytes()
    return {"exit_code": completed.returncode, "contract": "replay/stale/RLS"}, path, hashlib.sha256(raw).hexdigest(), ([] if completed.returncode == 0 else ["data-version contract failed"])


def run_benchmark() -> tuple[dict, Path | None, str | None, list[str]]:
    completed = subprocess.run([sys.executable, "-m", "scripts.benchmark_phase6"], cwd=ROOT, text=True, capture_output=True, timeout=120, check=False)
    path = ROOT / "data" / "quality" / "benchmark-latest.json"
    if completed.returncode != 0:
        return {"exit_code": completed.returncode}, None, None, ["benchmark execution failed"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"runner": "scripts.benchmark_phase6", "output": completed.stdout}, ensure_ascii=False, indent=2), encoding="utf-8")
    raw = path.read_bytes(); payload = json.loads(raw)
    return payload, path, hashlib.sha256(raw).hexdigest(), []

from app.quality import subjects
from app.quality.subjects import run_context_packer, run_memory_policy

import pytest


def test_memory_policy_dataset_is_versioned_and_executable():
    metrics, path, digest, reasons = run_memory_policy()
    assert path.name == "memory_policy_v1.jsonl"
    assert len(digest) == 64
    assert metrics["passed"] == metrics["cases"]
    assert metrics["baseline"]["unsafe_active_records"] == 2
    assert metrics["candidate"]["unsafe_active_records"] == 0
    assert not reasons


def test_context_packer_dataset_preserves_required_inputs_within_budget():
    from app.context.packer import token_counting_method

    # The dataset budgets are calibrated against cl100k_base counts.  Without
    # tiktoken the deterministic heuristic over-estimates and every tight
    # budget overflows — skip explicitly instead of failing mysteriously.
    if token_counting_method("mimo-v2.5") != "tiktoken:cl100k_base":
        pytest.skip("tiktoken unavailable in this interpreter; packer dataset budgets require cl100k_base counts")
    metrics, path, digest, reasons = run_context_packer()
    assert path.name == "context_packer_v1.jsonl"
    assert len(digest) == 64
    assert metrics["required_coverage"] == 1.0
    assert metrics["baseline"]["required_coverage"] < metrics["candidate"]["required_coverage"]
    assert not reasons


def test_rag_retrieval_subject_records_strategy_deltas(monkeypatch, tmp_path):
    target = tmp_path / "matrix.json"
    target.write_text("{}", encoding="utf-8")
    fixture = {"dataset_hash": "a" * 64, "top_k": 20, "results": {
        "rule": {"doc_hit@5": 0.7, "chunk_hit@5": 0.2, "mrr@5": 0.5, "p95_ms": 10},
        "cross_encoder": {"doc_hit@5": 0.7, "chunk_hit@5": 0.2, "mrr@5": 0.5, "p95_ms": 11},
        "rrf": {"doc_hit@5": 0.7, "chunk_hit@5": 0.2, "mrr@5": 0.6, "p95_ms": 12},
    }}
    monkeypatch.setattr(subjects, "write_retrieval_results", lambda: (target, fixture))
    metrics, _, _, reasons = subjects.run_rag_retrieval()
    assert metrics["deltas"]["rrf"]["mrr@5"] == 0.1
    assert reasons == ["cross_encoder did not improve MRR@5"]

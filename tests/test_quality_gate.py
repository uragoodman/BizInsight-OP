from app.quality import evaluations


def test_rag_gate_passes_when_metrics_and_contracts_are_healthy(monkeypatch, tmp_path):
    dataset = tmp_path / "queries.jsonl"; dataset.write_text('{"query_id":"q"}\n', encoding="utf-8")
    captured = {}
    monkeypatch.setattr(evaluations, "create_run", lambda *_args, **_kwargs: {"run_id": "run"})
    monkeypatch.setattr(evaluations, "add_evidence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(evaluations, "complete_run", lambda _id, **kwargs: captured.update(kwargs) or kwargs)
    metrics = {"doc_hit@5": 1.0, "chunk_hit@5": 1.0, "mrr@5": 0.9, "p95_ms": 100.0}
    evaluations.run_rag_gate("candidate", "admin", metrics, metrics, evaluation_id="e1", dataset_path=str(dataset), contracts={"passed": True, "output_hash": "a" * 64})
    assert captured["gate_status"] == "pass"


def test_rag_gate_blocks_contract_failure_even_when_metrics_are_equal(monkeypatch, tmp_path):
    dataset = tmp_path / "queries.jsonl"; dataset.write_text('{"query_id":"q"}\n', encoding="utf-8")
    captured = {}
    monkeypatch.setattr(evaluations, "create_run", lambda *_args, **_kwargs: {"run_id": "run"})
    monkeypatch.setattr(evaluations, "add_evidence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(evaluations, "complete_run", lambda _id, **kwargs: captured.update(kwargs) or kwargs)
    metrics = {"doc_hit@5": 1.0, "chunk_hit@5": 1.0, "mrr@5": 0.9, "p95_ms": 100.0}
    evaluations.run_rag_gate("candidate", "admin", metrics, metrics, evaluation_id="e1", dataset_path=str(dataset), contracts={"passed": False, "output_hash": "b" * 64})
    assert captured["gate_status"] == "block"
    assert "contract failed" in captured["reasons"][0]


def test_finalize_rag_contract_turns_pending_metric_gate_into_pass(monkeypatch):
    metrics = {"doc_hit@5": 1.0, "chunk_hit@5": 1.0, "mrr@5": 0.9, "p95_ms": 100.0}
    captured = {}
    monkeypatch.setattr(evaluations, "get_run", lambda _id: {"run_id": "run", "candidate_ref": {"metrics": metrics}, "baseline_ref": {"metrics": metrics}})
    monkeypatch.setattr(evaluations, "add_evidence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(evaluations, "complete_run", lambda _id, **kwargs: captured.update(kwargs) or kwargs)
    result = evaluations.finalize_rag_contract("run", {"passed": True, "output_hash": "c" * 64})
    assert result["gate_status"] == "pass"
    assert captured["status"] == "passed"


def test_finalize_rag_contract_blocks_a_late_contract_failure(monkeypatch):
    metrics = {"doc_hit@5": 1.0, "chunk_hit@5": 1.0, "mrr@5": 0.9, "p95_ms": 100.0}
    captured = {}
    monkeypatch.setattr(evaluations, "get_run", lambda _id: {"run_id": "run", "candidate_ref": {"metrics": metrics}, "baseline_ref": {"metrics": metrics}})
    monkeypatch.setattr(evaluations, "add_evidence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(evaluations, "complete_run", lambda _id, **kwargs: captured.update(kwargs) or kwargs)
    result = evaluations.finalize_rag_contract("run", {"passed": False, "output_hash": "d" * 64})
    assert result["gate_status"] == "block"
    assert captured["status"] == "blocked"

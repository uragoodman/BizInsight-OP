from __future__ import annotations

from app.rag import contract_worker


def test_contract_worker_blocks_release_when_late_contract_fails(monkeypatch) -> None:
    """The worker must invoke the transactional rollout shutdown on a block."""
    updates: list[str] = []
    blocked: list[tuple[str, str, list[str]]] = []

    class Result:
        def __init__(self, job=None): self.job = job
        def mappings(self): return self
        def first(self): return self.job

    class Connection:
        def execute(self, statement, *_args, **_kwargs):
            sql = str(statement)
            updates.append(sql)
            if "SELECT * FROM governance.rag_contract_validation_jobs" in sql:
                return Result({"job_id": "job-1", "release_id": "candidate", "evaluation_run_id": "run-1"})
            return Result()

    class Begin:
        def __enter__(self): return Connection()
        def __exit__(self, *_args): return False

    class Engine:
        def begin(self): return Begin()

    monkeypatch.setattr(contract_worker, "engine", Engine())
    monkeypatch.setattr(contract_worker, "run_contracts", lambda: {"passed": False, "output_hash": "e" * 64, "error_code": "contract_failed"})
    monkeypatch.setattr(contract_worker, "finalize_rag_contract", lambda *_args: {"gate_status": "block", "gate_reasons": ["RLS contract failed"]})
    monkeypatch.setattr(contract_worker, "block_release_for_quality_gate", lambda *args: blocked.append(args))
    assert contract_worker.process_once() is True
    assert blocked == [("candidate", "rag-contract-worker", ["RLS contract failed"])]
    assert any("SET status='completed'" in sql for sql in updates)

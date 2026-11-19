"""Security regressions for the governed Phase 11 code runner."""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

from app.code_analysis.integrity import plan_fingerprint
from app.code_analysis.policy import validate_candidate_code

worker = None
_worker_skip_reason = ""
if sys.version_info < (3, 10):
    _worker_skip_reason = "worker application requires Python 3.10+"
else:
    try:
        from app.code_analysis import worker
    except ModuleNotFoundError as exc:
        if exc.name != "psycopg":
            raise
        _worker_skip_reason = "worker tests require the psycopg runtime"

requires_worker = pytest.mark.skipif(
    worker is None,
    reason=_worker_skip_reason,
)


def _load_runner_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "deploy"
        / "analysis-runner"
        / "runner.py"
    )
    spec = importlib.util.spec_from_file_location("phase11_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_policy_allows_only_the_supplied_input_path():
    safe = """\
import pandas as pd

def analyze(input_path, output_dir):
    frame = pd.read_csv(input_path)
    return {"summary": {"rows": len(frame)}, "charts": [], "limitations": []}
"""
    assert validate_candidate_code(safe).valid

    arbitrary = safe.replace("pd.read_csv(input_path)", 'pd.read_csv("/etc/passwd")')
    result = validate_candidate_code(arbitrary)
    assert not result.valid
    assert any("provided input_path" in error for error in result.errors)


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            """\
import pandas as pd
def analyze(input_path, output_dir):
    pd.io.common.os.system("id")
    return {}
""",
            "Unsafe module",
        ),
        (
            """\
import numpy as np
def analyze(input_path, output_dir):
    return {"summary": np.load(input_path)}
""",
            "Reader is not allowed",
        ),
        (
            """\
import pandas as pd
def analyze(input_path, output_dir):
    reader = pd.read_csv
    return {"summary": reader("/etc/passwd")}
""",
            "Aliasing sensitive callable",
        ),
    ],
)
def test_policy_rejects_reexported_process_and_reader_bypasses(
    source: str,
    expected: str,
):
    result = validate_candidate_code(source)
    assert not result.valid
    assert any(expected in error for error in result.errors)


def test_policy_rejects_import_time_side_effects_and_output():
    source = """\
print("runs before analyze")
def analyze(input_path, output_dir):
    return {}
"""
    result = validate_candidate_code(source)
    assert not result.valid
    assert any("Top-level function call" in error for error in result.errors)


def test_policy_allows_only_headless_matplotlib_top_level_setup():
    safe = """\
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def analyze(input_path, output_dir):
    return {"summary": {}, "charts": [], "limitations": []}
"""
    assert validate_candidate_code(safe).valid

    unsafe_backend = safe.replace('matplotlib.use("Agg")', 'matplotlib.use("TkAgg")')
    result = validate_candidate_code(unsafe_backend)
    assert not result.valid
    assert any("matplotlib.use" in error for error in result.errors)

    unsafe_argument = safe.replace('matplotlib.use("Agg")', 'matplotlib.use(backend_name)')
    result = validate_candidate_code(unsafe_argument)
    assert not result.valid
    assert any("matplotlib.use" in error for error in result.errors)


def test_runner_captures_and_bounds_import_time_output(tmp_path, capsys):
    runner = _load_runner_module()
    candidate = tmp_path / "candidate.py"
    candidate.write_text(
        """\
print("A" * 40000)
def analyze(input_path, output_dir):
    print("B" * 40000)
    return {"summary": {}, "charts": [], "limitations": []}
""",
        encoding="utf-8",
    )
    source = tmp_path / "source.csv"
    source.write_text("x\n1\n", encoding="utf-8")

    result, captured = runner._execute_candidate(
        str(candidate),
        str(source),
        tmp_path,
    )

    assert result["summary"] == {}
    assert len(captured) == 32_000
    assert set(captured) == {"A"}
    assert capsys.readouterr().out == ""


def test_runner_bounds_candidate_exception_rendering(tmp_path):
    candidate = tmp_path / "candidate.py"
    candidate.write_text(
        'raise RuntimeError("Z" * 10000000)\n',
        encoding="utf-8",
    )
    runner_path = (
        Path(__file__).resolve().parents[1]
        / "deploy"
        / "analysis-runner"
        / "runner.py"
    )
    completed = subprocess.run(
        [sys.executable, str(runner_path), str(tmp_path / "source.csv")],
        env={
            **os.environ,
            "CODE_ANALYSIS_CANDIDATE_PATH": str(candidate),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 1
    assert len(completed.stderr) <= 2_100
    assert b"runner error (RuntimeError)" in completed.stderr


def test_runner_rejects_svg_and_fake_png(tmp_path):
    runner = _load_runner_module()
    svg = tmp_path / "chart.svg"
    svg.write_text("<svg/>", encoding="utf-8")
    with pytest.raises(RuntimeError, match="only PNG"):
        runner._collect_outputs({"charts": [str(svg)]}, tmp_path)

    fake_png = tmp_path / "chart.png"
    fake_png.write_bytes(b"not a png")
    with pytest.raises(RuntimeError, match="valid PNG"):
        runner._collect_outputs({"charts": [str(fake_png)]}, tmp_path)


def test_runner_enforces_result_contract_and_json_limit(monkeypatch):
    runner = _load_runner_module()
    valid = {
        "conclusion": "The evidence supports the result.",
        "method": "descriptive",
        "summary": {"rows": 2},
        "evidence": [],
        "charts": [],
        "limitations": [],
        "recommendations": [],
    }
    runner._validate_result_contract(valid)
    with pytest.raises(RuntimeError, match="conclusion"):
        runner._validate_result_contract({**valid, "conclusion": ""})
    with pytest.raises(RuntimeError, match="evidence"):
        runner._validate_result_contract({**valid, "evidence": {}})
    with pytest.raises(RuntimeError, match="summary"):
        runner._validate_result_contract({**valid, "summary": {}})
    with pytest.raises(RuntimeError, match="reserved"):
        runner._validate_result_contract(
            {**valid, "reproducibility": {"candidate_claim": True}}
        )
    monkeypatch.setattr(runner, "MAX_RESULT_BYTES", 64)
    with pytest.raises(RuntimeError, match="result limit"):
        runner._validate_result_size({**valid, "conclusion": "x" * 100})


def test_fake_provider_emits_the_complete_candidate_result_contract(tmp_path):
    from app.code_analysis.providers import FakeCodeAgentProvider

    source = tmp_path / "input.csv"
    source.write_text("amount,category\n10,a\n20,b\n", encoding="utf-8")
    artifact = FakeCodeAgentProvider().propose_analysis(
        {"question": "Summarize sales"},
        {"file_type": "csv"},
    )
    namespace: dict = {}
    exec(compile(artifact.code, "analysis.py", "exec"), namespace)
    result = namespace["analyze"](str(source), str(tmp_path))

    runner = _load_runner_module()
    runner._validate_result_contract(result)
    assert result["conclusion"].strip()
    assert set(result) == {
        "conclusion",
        "method",
        "summary",
        "evidence",
        "charts",
        "limitations",
        "recommendations",
    }


def test_provider_context_is_server_scoped_and_not_a_user_token():
    from app.code_analysis.providers import FakeCodeAgentProvider, ProviderContext

    context = ProviderContext(
        job_id="job-1",
        tenant_id="tenant-1",
        actor_id="user-1",
        trace_id="trace-1",
        plan_hash="p" * 64,
        data_contract_hash="c" * 64,
    )
    artifact = FakeCodeAgentProvider().propose_analysis(
        {"question": "Summarize sales"}, {"file_type": "csv"}, context=context
    )
    assert artifact.provider == "fake"
    assert not hasattr(artifact, "access_token")


class _MemoryStore:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.max_bytes: int | None = None

    def get(self, _key: str, *, max_bytes: int | None = None) -> bytes:
        self.max_bytes = max_bytes
        return self.payload


@requires_worker
def test_worker_bounds_and_verifies_source_before_execution():
    payload = b"x\n1\n"
    store = _MemoryStore(payload)
    job = {
        "file_object_key": "opaque/source.csv",
        "file_size_bytes": len(payload),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
    }
    assert worker._load_verified_source(job, store) == payload
    assert store.max_bytes == len(payload)

    job["file_sha256"] = "0" * 64
    with pytest.raises(worker.InputIntegrityError, match="SHA-256"):
        worker._load_verified_source(job, store)


@requires_worker
def test_worker_verifies_plan_file_and_contract_fingerprints():
    contract = {"file_type": "csv", "columns": [{"name": "x"}]}
    file_sha = hashlib.sha256(b"x\n1\n").hexdigest()
    plan = {
        "question": "summarize",
        "file_sha256": file_sha,
        "data_contract": contract,
    }
    job = {
        "plan_data": plan,
        "plan_hash": plan_fingerprint(plan),
        "file_sha256": file_sha,
        "file_contract_data": contract,
        "file_type": "csv",
    }
    worker._verify_plan_integrity(job)
    with pytest.raises(worker.PlanIntegrityError):
        worker._verify_plan_integrity(
            {
                **job,
                "file_contract_data": {
                    "file_type": "csv",
                    "columns": [{"name": "tampered"}],
                },
            }
        )


@requires_worker
def test_artifact_put_is_compensated_when_post_write_fence_is_lost(
    monkeypatch,
):
    calls = 0
    deleted: list[str] = []

    class Store:
        def put(self, key: str, data: bytes):
            return key, hashlib.sha256(data).hexdigest(), len(data)

        def delete(self, key: str) -> None:
            deleted.append(key)

    def fence(_conn, _job, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise worker.LeaseLost("expired during put")
        return {"status": "running", "cancel_requested": False}

    monkeypatch.setattr(worker, "get_artifact_store", lambda: Store())
    monkeypatch.setattr(worker, "_assert_owned_lease", fence)
    with pytest.raises(worker.LeaseLost):
        worker._artifact(
            object(),
            {"job_id": "job", "tenant_id": "tenant", "user_id": "user"},
            "result",
            "opaque/result.json",
            b"{}",
        )
    assert calls == 2
    assert deleted == ["opaque/result.json"]


@requires_worker
def test_claim_caps_crash_retries_and_returns_lease_attempt(monkeypatch):
    row = {
        "job_id": "caj_test",
        "attempts": 2,
    }

    class Result:
        def __init__(self, value=None) -> None:
            self.value = value

        def mappings(self):
            return self

        def one_or_none(self):
            return self.value

        def all(self):
            return [] if self.value is None else self.value

    class Connection:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        def execute(self, statement, parameters=None):
            sql = str(statement)
            values = parameters or {}
            self.calls.append((sql, values))
            if "SELECT j.*" in sql:
                return Result(row)
            return Result()

    connection = Connection()

    class Begin:
        def __enter__(self):
            return connection

        def __exit__(self, *_args):
            return False

    class Engine:
        def begin(self):
            return Begin()

    monkeypatch.setattr(worker, "engine", Engine())
    claimed = worker._claim()
    assert claimed is not None
    assert claimed["lease_attempt"] == 3
    assert any(
        "worker_retry_exhausted" in sql and values["max_attempts"] == 3
        for sql, values in connection.calls
        if "max_attempts" in values
    )
    assert any("j.attempts < :max_attempts" in sql for sql, _ in connection.calls)

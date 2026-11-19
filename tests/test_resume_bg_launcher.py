"""Offline process-boundary checks for the resume B/G PowerShell launcher helper."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "Invoke-PythonLogged.ps1"


def _ps_executable() -> str:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if not executable:
        pytest.skip("PowerShell is unavailable; production launcher helper cannot be exercised here.")
    return executable


def _quote_ps(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _run_actual_helper(tmp_path: Path, *, mode: str, timeout_ms: int = 8000) -> tuple[dict, str, str]:
    python = shutil.which("python") or shutil.which("py")
    if not python:
        pytest.skip("Python executable not found on PATH for subprocess helper test.")
    child = tmp_path / "child.py"
    child.write_text(
        "import pathlib, sys, time\n"
        "mode, result, marker = sys.argv[1:4]\n"
        "if mode == 'warning_result':\n"
        "    print('registry warning on stderr', file=sys.stderr, flush=True)\n"
        "    time.sleep(1.25)\n"
        "    pathlib.Path(result).write_text('{\"ok\":true}', encoding='utf-8')\n"
        "elif mode == 'nonzero':\n"
        "    print('child diagnostic on stderr', file=sys.stderr, flush=True)\n"
        "    pathlib.Path(result).write_text('diagnostic saved', encoding='utf-8')\n"
        "    raise SystemExit(7)\n"
        "elif mode == 'missing':\n"
        "    print('clean exit without result', file=sys.stderr, flush=True)\n"
        "elif mode == 'sleep':\n"
        "    pathlib.Path(marker).write_text('started', encoding='utf-8')\n"
        "    time.sleep(30)\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "logs"
    output_dir.mkdir()
    result_path = tmp_path / "result.json"
    marker_path = tmp_path / "started.marker"
    report_path = tmp_path / "process_result.json"
    harness = tmp_path / "invoke_helper.ps1"
    harness.write_text(
        f". {_quote_ps(HELPER)}\n"
        f"$result = Invoke-PythonLogged -PythonPath {_quote_ps(python)} `\n"
        f"  -Arguments @({_quote_ps(child)}, {_quote_ps(mode)}, {_quote_ps(result_path)}, {_quote_ps(marker_path)}) `\n"
        f"  -WorkingDirectory {_quote_ps(tmp_path)} -OutputDirectory {_quote_ps(output_dir)} `\n"
        f"  -Name 'case' -TimeoutMs {int(timeout_ms)} -ExpectedResultPath {_quote_ps(result_path)}\n"
        f"$result | ConvertTo-Json -Compress | Set-Content -LiteralPath {_quote_ps(report_path)} -Encoding utf8\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [_ps_executable(), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(harness)],
        cwd=ROOT, capture_output=True, text=True, timeout=20,
    )
    assert completed.returncode == 0, completed.stderr + "\n" + completed.stdout
    raw = report_path.read_text(encoding="utf-8-sig")
    return json.loads(raw), completed.stdout, completed.stderr


def test_warning_stderr_and_wait_progress_do_not_contaminate_exit_result(tmp_path):
    result, stdout, _stderr = _run_actual_helper(tmp_path, mode="warning_result")
    assert result["status"] == "success"
    assert result["exit_code"] == 0 and result["exited"] is True
    assert result["result_present"] is True
    assert "registry warning on stderr" in Path(result["stderr_path"]).read_text(encoding="utf-8")
    assert "WAITING case" in stdout
    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8")) == {"ok": True}


def test_nonzero_exit_is_preserved_with_stderr_and_diagnostic_result(tmp_path):
    result, _stdout, _stderr = _run_actual_helper(tmp_path, mode="nonzero")
    assert result["status"] == "nonzero_exit"
    assert result["exit_code"] == 7 and result["exited"] is True
    assert result["result_present"] is True
    assert "child diagnostic on stderr" in Path(result["stderr_path"]).read_text(encoding="utf-8")
    assert (tmp_path / "result.json").read_text(encoding="utf-8") == "diagnostic saved"


def test_zero_exit_without_required_result_is_not_accepted(tmp_path):
    result, _stdout, _stderr = _run_actual_helper(tmp_path, mode="missing")
    assert result["exit_code"] == 0 and result["exited"] is True
    assert result["result_present"] is False
    assert result["status"] == "result_missing"


def test_timeout_kills_and_confirms_child_exit_before_return(tmp_path):
    result, _stdout, _stderr = _run_actual_helper(tmp_path, mode="sleep", timeout_ms=700)
    assert result["timed_out"] is True
    assert result["exited"] is True
    assert result["status"] == "timed_out"
    assert (tmp_path / "started.marker").is_file()
    assert result["result_present"] is False

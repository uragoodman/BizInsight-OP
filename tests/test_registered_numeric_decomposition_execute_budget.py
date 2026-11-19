from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/testing/registered_numeric_decomposition_critic_ablation_20261015"
RUNNER_PATH = OUT / "run_experiment.py"
_SPEC = importlib.util.spec_from_file_location("registered_numeric_execute_runner", RUNNER_PATH)
assert _SPEC and _SPEC.loader
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def _resolved(auth_a: float = 5.0, auth_b: float = 5.0) -> dict:
    manifest = json.loads((OUT / "freeze_manifest.json").read_text(encoding="utf-8"))
    manifest["actual_provider_authorization_cny"] = auth_a
    manifest["current_authorized_spend_cny"] = auth_b
    return {
        "manifest": manifest,
        "freeze_manifest_sha256": runner._canonical_sha(manifest),
        "schedule": json.loads((OUT / "run_manifest_12.json").read_text(encoding="utf-8")),
        "cost": json.loads((OUT / "cost_estimate.json").read_text(encoding="utf-8")),
    }


def _ledgers(tmp_path: Path, resolved: dict, limit: float = 5.0):
    from evaluation.provider_candidate_batch import CostPolicy, EstimatedCostLedger, RunLedger

    args = dict(resolved["cost"]["policy"])
    args["cumulative_estimate_limit_cny"] = limit
    policy = CostPolicy(**args)
    return policy, RunLedger(tmp_path / "run_events.jsonl"), EstimatedCostLedger(policy, tmp_path / "cost_reservations.jsonl")


def test_two_selected_units_fit_five_cny_and_execute_reuses_same_selection(tmp_path):
    resolved = _resolved()
    policy, ledger, cost_ledger = _ledgers(tmp_path, resolved)
    selection = runner._build_execute_selection(
        resolved, ledger=ledger, cost_ledger=cost_ledger, max_units=2,
        estimate_limit_cny=5.0, provider_authorization_cny=5.0,
    )

    assert selection["selected_run_ids"] == list(runner.REQUIRED_FIRST_PAIR_RUN_IDS)
    assert selection["full_plan_estimate"]["runs"] == 12
    assert selection["full_plan_estimate"]["estimated_cny"] == pytest.approx(27.94455144)
    assert selection["selected_new_batch_estimate"]["runs"] == 2
    assert selection["selected_reservation_total_cny"] == pytest.approx(4.65742524)
    assert selection["cumulative_reserved_after_cny"] == pytest.approx(4.65742524)

    # The preflight-produced selection is the exact object the Execute path
    # verifies again before touching PostgreSQL.
    execute_selection = runner._build_execute_selection(
        resolved, ledger=ledger, cost_ledger=cost_ledger, max_units=2,
        estimate_limit_cny=5.0, provider_authorization_cny=5.0,
    )
    runner._verify_execute_selection({"execute_selection": selection}, execute_selection)
    assert execute_selection["selection_sha256"] == selection["selection_sha256"]


def test_twelve_unit_plan_is_rejected_at_five_cny(tmp_path):
    resolved = _resolved()
    _policy, ledger, cost_ledger = _ledgers(tmp_path, resolved)
    with pytest.raises(RuntimeError, match="12-run reservation estimate"):
        runner._build_execute_selection(
            resolved, ledger=ledger, cost_ledger=cost_ledger, max_units=12,
            estimate_limit_cny=5.0, provider_authorization_cny=5.0,
        )


def test_existing_reservations_count_against_the_same_cumulative_limit(tmp_path):
    resolved = _resolved()
    policy, ledger, cost_ledger = _ledgers(tmp_path, resolved)
    cost_ledger.reserve("previously-reserved-run")
    with pytest.raises(RuntimeError, match="Cumulative estimated reservations"):
        runner._build_execute_selection(
            resolved, ledger=ledger, cost_ledger=cost_ledger, max_units=2,
            estimate_limit_cny=5.0, provider_authorization_cny=5.0,
        )


@pytest.mark.parametrize("auth_a,auth_b", [(5.0, 4.0), (0.0, 5.0)])
def test_conflicting_frozen_authorization_fields_are_rejected(tmp_path, auth_a, auth_b):
    resolved = _resolved(auth_a, auth_b)
    _policy, ledger, cost_ledger = _ledgers(tmp_path, resolved)
    with pytest.raises(RuntimeError, match="authorization fields.*conflict"):
        runner._build_execute_selection(
            resolved, ledger=ledger, cost_ledger=cost_ledger, max_units=2,
            estimate_limit_cny=5.0, provider_authorization_cny=5.0,
        )


def _write_authorized_child(tmp_path: Path, *, auth_a: float = 5.0, auth_b: float = 5.0) -> Path:
    manifest = json.loads((OUT / "freeze_manifest.json").read_text(encoding="utf-8"))
    manifest["actual_provider_authorization_cny"] = auth_a
    manifest["current_authorized_spend_cny"] = auth_b
    path = tmp_path / "authorized_child_manifest.json"
    data = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    (tmp_path / "freeze_manifest.sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return path


def _powershell() -> Path:
    return Path(os.environ.get("SystemRoot", r"C:\Windows")) / r"System32\WindowsPowerShell\v1.0\powershell.exe"


def test_powershell_z_preflight_uses_python_pair_selection_without_database(tmp_path):
    if not _powershell().is_file():
        pytest.skip("Windows PowerShell executable is unavailable.")
    formal_root = OUT / "formal_execution"
    if formal_root.exists():
        pytest.skip("Formal execution state exists; refusing to inspect or modify it in this isolated test.")
    child_manifest = _write_authorized_child(tmp_path)
    script = OUT / "run_experiment.ps1"
    try:
        completed = subprocess.run(
            [str(_powershell()), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
             "-Mode", "Preflight", "-FreezeManifest", str(child_manifest), "-MaxUnits", "2",
             "-EstimateLimitCny", "5", "-ProviderAuthorizationCny", "5"],
            cwd=ROOT, capture_output=True, text=True, timeout=60,
        )
    finally:
        # The real preflight constructor creates only this empty parent folder.
        # Remove it only when it is still empty; never touch run or cost ledgers.
        if formal_root.exists() and not any(formal_root.iterdir()):
            formal_root.rmdir()
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["execute_selection"]["selected_run_ids"] == list(runner.REQUIRED_FIRST_PAIR_RUN_IDS)
    assert payload["execute_selection"]["selected_reservation_total_cny"] == pytest.approx(4.65742524)
    assert payload["execute_selection"]["full_plan_estimate"]["runs"] == 12
    assert payload["started"] == payload["provider_sends"] == 0
    assert payload["actual_database_started"] is False


def test_powershell_rejects_conflicting_auth_fields_before_database(tmp_path):
    if not _powershell().is_file():
        pytest.skip("Windows PowerShell executable is unavailable.")
    child_manifest = _write_authorized_child(tmp_path, auth_a=5.0, auth_b=4.0)
    completed = subprocess.run(
        [str(_powershell()), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(OUT / "run_experiment.ps1"),
         "-Mode", "Preflight", "-FreezeManifest", str(child_manifest), "-MaxUnits", "2",
         "-EstimateLimitCny", "5", "-ProviderAuthorizationCny", "5"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode != 0
    assert "authorization fields conflict" in completed.stderr
    assert "isolated_postgres_start" not in completed.stdout + completed.stderr


def test_powershell_rejects_twelve_units_under_five_cny_without_database(tmp_path):
    if not _powershell().is_file():
        pytest.skip("Windows PowerShell executable is unavailable.")
    formal_root = OUT / "formal_execution"
    if formal_root.exists():
        pytest.skip("Formal execution state exists; refusing to inspect or modify it in this isolated test.")
    child_manifest = _write_authorized_child(tmp_path)
    completed = subprocess.run(
        [str(_powershell()), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(OUT / "run_experiment.ps1"),
         "-Mode", "Preflight", "-FreezeManifest", str(child_manifest), "-MaxUnits", "12",
         "-EstimateLimitCny", "5", "-ProviderAuthorizationCny", "5"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode != 0
    assert "shared_frozen_input_preflight" in completed.stderr
    if formal_root.exists():
        assert not any(formal_root.iterdir())  # Python selected no units and wrote no ledger/reservation.
        formal_root.rmdir()

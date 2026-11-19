import json

import pytest

from evaluation.resume_formal_bg_executor import (
    assert_runtime_freeze_complete,
    assert_run_ids_not_previously_used,
    historical_run_id_occurrences,
    runtime_freeze_hashes,
)


def test_formal_run_guard_scans_all_historical_ledger_types(tmp_path):
    first = tmp_path / "batch_a" / "events.jsonl"
    second = tmp_path / "batch_b" / "cost_reservations.jsonl"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text(json.dumps({"run_id": "used", "event": "completed"}) + "\n", encoding="utf-8")
    second.write_text(json.dumps({"run_id": "reserved", "reserved_cny": 1.0}) + "\n", encoding="utf-8")

    found = historical_run_id_occurrences(["used", "reserved", "new"], [tmp_path])
    assert set(found) == {"used", "reserved"}
    assert found["used"][0]["event"] == "completed"
    assert found["reserved"][0]["path"].endswith("cost_reservations.jsonl")
    with pytest.raises(RuntimeError, match="already exist"):
        assert_run_ids_not_previously_used(["used"], [tmp_path])
    assert assert_run_ids_not_previously_used(["new"], [tmp_path]) == {}


def test_formal_run_guard_fails_closed_on_corrupt_historical_ledger(tmp_path):
    ledger = tmp_path / "corrupt" / "run_events.jsonl"
    ledger.parent.mkdir()
    ledger.write_text('{"run_id":"safe","event":"started"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed run-history ledger"):
        assert_run_ids_not_previously_used(["new"], [tmp_path])


def test_formal_run_guard_allows_only_unused_ids_and_excludes_active_ledgers(tmp_path):
    active_events = tmp_path / "current" / "events.jsonl"
    active_costs = tmp_path / "current" / "cost_reservations.jsonl"
    active_events.parent.mkdir()
    active_events.write_text('{"run_id":"resume","event":"cost_reserved"}\n', encoding="utf-8")
    active_costs.write_text('{"run_id":"resume","reserved_cny":1.0}\n', encoding="utf-8")
    assert assert_run_ids_not_previously_used(
        ["resume"], [tmp_path], active_ledger_path=active_events,
        active_cost_ledger_path=active_costs,
    ) == {}


def test_runtime_freeze_inventory_contains_application_and_evaluation_code(tmp_path):
    app_file = tmp_path / "app" / "context" / "request_contract.py"
    eval_file = tmp_path / "evaluation" / "scorer.py"
    app_file.parent.mkdir(parents=True)
    eval_file.parent.mkdir(parents=True)
    config_file = tmp_path / "config" / "app.yaml"
    secret_file = tmp_path / "config" / "demo-accounts" / ".env"
    prompt_file = tmp_path / "docs" / "knowledge" / "agent.md"
    migration_file = tmp_path / "migrations" / "versions" / "001.py"
    config_file.parent.mkdir(parents=True)
    secret_file.parent.mkdir(parents=True)
    prompt_file.parent.mkdir(parents=True)
    migration_file.parent.mkdir(parents=True)
    app_file.write_text("contract = 1\n", encoding="utf-8")
    eval_file.write_text("score = 1\n", encoding="utf-8")
    config_file.write_text("setting: 1\n", encoding="utf-8")
    secret_file.write_text("API_KEY=do-not-hash\n", encoding="utf-8")
    prompt_file.write_text("review instructions\n", encoding="utf-8")
    migration_file.write_text("upgrade()\n", encoding="utf-8")
    first = runtime_freeze_hashes(tmp_path)
    assert set(first) == {
        "app/context/request_contract.py", "evaluation/scorer.py",
        "config/app.yaml", "docs/knowledge/agent.md", "migrations/versions/001.py",
    }
    assert "config/demo-accounts/.env" not in first
    assert assert_runtime_freeze_complete(tmp_path, first) == first
    with pytest.raises(ValueError, match="omits runtime"):
        assert_runtime_freeze_complete(tmp_path, {key: value for key, value in first.items() if key != "config/app.yaml"})
    app_file.write_text("contract = 2\n", encoding="utf-8")
    second = runtime_freeze_hashes(tmp_path)
    assert first["app/context/request_contract.py"] != second["app/context/request_contract.py"]
    with pytest.raises(ValueError, match="do not match"):
        assert_runtime_freeze_complete(tmp_path, first)


def test_history_guard_detects_duplicate_run_id_across_nested_experiment_ledgers(tmp_path):
    ledger_one = tmp_path / "exp_a" / "pair_1" / "run_events.jsonl"
    ledger_two = tmp_path / "exp_b" / "results" / "cost_reservations.jsonl"
    ledger_one.parent.mkdir(parents=True)
    ledger_two.parent.mkdir(parents=True)
    ledger_one.write_text(json.dumps({"run_id": "collision", "event": "completed"}) + "\n", encoding="utf-8")
    ledger_two.write_text(json.dumps({"run_id": "collision", "reserved_cny": 1.25}) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already exist"):
        assert_run_ids_not_previously_used(["collision"], [tmp_path])
    rows = historical_run_id_occurrences(["collision"], [tmp_path])["collision"]
    assert len(rows) == 2


def test_history_guard_reads_cost_ledger_json_snapshots(tmp_path):
    path = tmp_path / "older_experiment" / "cost_ledger_baseline.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"reservations": [
        {"run_id": "reserved-earlier", "reserved_cny": 2.5},
    ]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="already exist"):
        assert_run_ids_not_previously_used(["reserved-earlier"], [tmp_path])

    versioned = tmp_path / "later_experiment" / "run_events_v3.jsonl"
    versioned.parent.mkdir(parents=True)
    versioned.write_text(json.dumps({"run_id": "completed-v3", "event": "completed"}) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already exist"):
        assert_run_ids_not_previously_used(["completed-v3"], [tmp_path])


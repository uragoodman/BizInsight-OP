import json

from evaluation.claim_verification.freeze_development_inputs import main


def test_freeze_indexes_only_nonempty_single_precritic_snapshot(tmp_path, monkeypatch):
    capture = tmp_path / "capture"
    (capture / "records").mkdir(parents=True)
    (capture / "traces").mkdir()
    (capture / "run_summary.json").write_text(json.dumps({"completed_units": 12, "failed_units": 0}), encoding="utf-8")
    (capture / "preflight.json").write_text("{}", encoding="utf-8")
    for index in range(12):
        run_id = f"run-{index}"
        (capture / "records" / f"{run_id}.json").write_text(json.dumps({"run_id": run_id, "case_id": f"C{index}"}), encoding="utf-8")
        snapshots = [{"answer": "draft", "snapshot_sha256": f"hash-{index}"}] if index < 11 else []
        (capture / "traces" / f"{run_id}.json").write_text(json.dumps({"state_snapshot": {"task_type": "metric_query", "pre_critic_answer_snapshots": snapshots}}), encoding="utf-8")
    output = tmp_path / "freeze"
    monkeypatch.setattr("sys.argv", ["freeze", "--capture-root", str(capture), "--output", str(output)])

    main()

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (output / "candidate_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert manifest["eligible_for_phase1_annotation"] == 11
    excluded = [row for row in rows if row["eligible_for_phase1_annotation"] is False]
    assert len(excluded) == 1
    assert excluded[0]["exclusion_reason"] == "no_single_nonempty_pre_critic_answer_snapshot"

from __future__ import annotations

import pytest

from evaluation.resume_formal_bg_executor import select_pending_manifest_entries


def _entries():
    return [
        {"run_id": "g-1", "case_id": "case-1", "arm": "G"},
        {"run_id": "b-1", "case_id": "case-1", "arm": "B"},
        {"run_id": "g-2", "case_id": "case-2", "arm": "G"},
        {"run_id": "b-2", "case_id": "case-2", "arm": "B"},
    ]


def test_explicit_selection_runs_only_requested_governed_units_in_frozen_order():
    entries = _entries()
    selected = select_pending_manifest_entries(
        entries, {}, max_units=2, run_ids=["g-1", "g-2"],
    )
    assert [row["run_id"] for row in selected] == ["g-1", "g-2"]
    assert all(row["arm"] == "G" for row in selected)


def test_default_selection_preserves_frozen_prefix_behavior():
    selected = select_pending_manifest_entries(_entries(), {}, max_units=2)
    assert [row["run_id"] for row in selected] == ["g-1", "b-1"]


@pytest.mark.parametrize(
    "run_ids,max_units,latest,error,match",
    [
        (["g-1"], 2, {}, ValueError, "match max_units"),
        (["g-1", "g-1"], 2, {}, ValueError, "unique"),
        (["unknown"], 1, {}, ValueError, "frozen manifest"),
        (["b-1", "g-1"], 2, {}, ValueError, "manifest order"),
        (["g-1"], 1, {"g-1": {"event": "started"}}, RuntimeError, "retry is forbidden"),
    ],
)
def test_explicit_selection_fails_closed_for_invalid_or_started_units(run_ids, max_units, latest, error, match):
    with pytest.raises(error, match=match):
        select_pending_manifest_entries(
            _entries(), latest, max_units=max_units, run_ids=run_ids,
        )

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.merge_semantic_interpreter_reviews import main, merge_reviews
from scripts.evaluate_semantic_interpreter_clarification import (
    _render_report,
    evaluate_clarification,
)
from scripts.build_semantic_interpreter_reviewed_evidence import build_reviewed_evidence


def _result(case_id: str, status: str, *, operation: dict | None = None) -> dict:
    operations = [operation] if operation else []
    return {
        "case_id": case_id,
        "label_status": "draft",
        "expected": {"status": status, "contract": {"operations": operations}},
        "observation": {
            "status": "ok",
            "issue_codes": [],
            "contract": {"semantic_status": status, "operations": operations},
        },
        "scores": {
            "field_correct": 2,
            "field_total": 3,
            "draft_status_match": True,
        },
    }


def _decision(case_id: str, status: str, *, operations: list[dict] | None = None) -> dict:
    return {
        "case_id": case_id,
        "reviewer": "reviewer-1",
        "reviewed_at": "2026-11-13T10:00:00+08:00",
        "reviewed_status": status,
        "reviewed_operations": operations or [],
        "reviewed_clarification_required": status != "CONFIRMED",
        "notes": "人工逐项核对。",
    }


def _files(tmp_path, payload: dict, decisions: list[dict]):
    evaluation = tmp_path / "evaluation.json"
    decision_path = tmp_path / "decisions.jsonl"
    evaluation.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    decision_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in decisions),
        encoding="utf-8",
    )
    return evaluation, decision_path


def test_empty_decisions_leave_every_case_draft_and_keep_draft_metrics():
    original = {"meta": {"run": "fixture"}, "results": [_result("case-1", "CONFIRMED")]}
    before = deepcopy(original["results"][0]["scores"])

    merged = merge_reviews(original, {})

    assert merged["review_merge"]["decision_count"] == 0
    assert merged["review_metrics"]["reviewed"]["case_count"] == 0
    assert merged["review_metrics"]["reviewed"]["status_accuracy"] is None
    assert merged["review_metrics"]["draft"]["case_count"] == 1
    assert merged["review_metrics"]["draft"]["field_accuracy"] == pytest.approx(2 / 3)
    assert merged["results"][0]["label_status"] == "draft"
    assert "review" not in merged["results"][0]
    assert merged["results"][0]["scores"] == before
    assert original["results"][0]["scores"] == before


def test_two_decisions_merge_and_review_metrics_are_separate_from_remaining_draft():
    segment = {
        "kind": "SEGMENT",
        "target_concept_ids": ["CUSTOMER"],
        "polarity": "NEGATIVE",
    }
    payload = {
        "meta": {},
        "results": [
            _result("case-1", "CONFIRMED", operation=segment),
            _result("case-2", "AMBIGUOUS"),
            _result("case-3", "CONFIRMED"),
        ],
    }
    decisions = {
        "case-1": _decision("case-1", "CONFIRMED", operations=[segment]),
        "case-2": _decision("case-2", "AMBIGUOUS"),
    }

    merged = merge_reviews(payload, decisions)

    reviewed = merged["review_metrics"]["reviewed"]
    assert reviewed["case_count"] == 2
    assert reviewed["schema_valid_rate"] == 1.0
    assert reviewed["status_accuracy"] == 1.0
    assert reviewed["operation_exact_match_rate"] == 1.0
    assert merged["review_metrics"]["draft"]["case_count"] == 1
    assert [item["label_status"] for item in merged["results"]] == ["reviewed", "reviewed", "draft"]
    assert merged["results"][0]["review"] == decisions["case-1"]
    assert merged["results"][2]["scores"] == payload["results"][2]["scores"]


def test_clarification_metrics_use_reviewed_gold_and_group_by_category():
    results = [
        _result("case-tp", "AMBIGUOUS"),
        _result("case-tn", "CONFIRMED"),
        _result("case-fp-ambiguity", "CONFIRMED"),
        _result("case-fn", "CONFIRMED"),
        _result("case-fp-status", "UNSUPPORTED"),
    ]
    for result in results:
        result["category"] = "negation" if result["case_id"] != "case-fp-status" else "unsupported"
    results[2]["observation"]["contract"]["ambiguities"] = [{"code": "clarification_required"}]
    payload = {"results": results}
    clarifications = {
        "case-tp": True,
        "case-tn": False,
        "case-fp-ambiguity": False,
        "case-fn": True,
        "case-fp-status": False,
    }
    reviewed_statuses = {
        "case-tp": "AMBIGUOUS",
        "case-fp-status": "UNSUPPORTED",
    }
    decisions = {
        case_id: _decision(
            case_id,
            reviewed_statuses.get(case_id, "CONFIRMED"),
        ) | {"reviewed_clarification_required": gold}
        for case_id, gold in clarifications.items()
    }
    for result in results:
        decision = decisions[result["case_id"]]
        result["label_status"] = "reviewed"
        result["review"] = decision

    metrics = evaluate_clarification(payload, clarifications)

    assert metrics["reviewed_case_count"] == 5
    assert metrics["overall"]["case_count"] == 5
    assert metrics["overall"]["true_positive"] == 1
    assert metrics["overall"]["false_positive"] == 2
    assert metrics["overall"]["false_negative"] == 1
    assert metrics["overall"]["true_negative"] == 1
    assert metrics["overall"]["precision"] == pytest.approx(0.3333)
    assert metrics["overall"]["recall"] == pytest.approx(0.5)
    assert metrics["overall"]["f1"] == pytest.approx(0.4)
    assert metrics["by_category"]["unsupported"]["false_positive"] == 1
    report = _render_report(
        metrics,
        eval_path=Path("merged.json"),
        decisions_path=Path("decisions.jsonl"),
    )
    assert "| unsupported | 1 | 0 | 1 | 0 | 0 | 0.0 | null | 0.0 |" in report


def test_compact_evidence_keeps_full_evaluation_and_only_selected_reviewed_labels(tmp_path):
    evaluation_path, decisions_path = _files(
        tmp_path,
        {
            "meta": {"environment": {"prompt_version": "semantic-interpreter-v3.6"}},
            "results": [_result("case-reviewed", "CONFIRMED"), _result("case-draft", "CONFIRMED")],
        },
        [_decision("case-reviewed", "CONFIRMED")],
    )

    evidence, decisions = build_reviewed_evidence(evaluation_path, decisions_path)

    assert len(evidence["results"]) == 2
    assert [item["label_status"] for item in evidence["results"]] == ["reviewed", "draft"]
    assert len(decisions) == 1
    assert evidence["review_metrics"]["reviewed"]["case_count"] == 1


@pytest.mark.parametrize(
    ("mutate", "error_fragment"),
    [
        (lambda item: item.update(case_id="unknown-case"), "unknown case_id"),
        (None, "Duplicate decision"),
        (lambda item: item.update(reviewed_status="NOT_A_STATUS"), "invalid reviewed_status"),
        (lambda item: item.update(reviewed_at="2026-11-13T10:00:00"), "must include a timezone"),
        (
            lambda item: item.update(reviewed_operations=[{
                "kind": "SEGMENT", "target_concept_ids": ["CUSTOMER"], "polarity": "MAYBE",
            }]),
            "invalid polarity",
        ),
    ],
)
def test_cli_rejects_invalid_decisions(tmp_path, capsys, mutate, error_fragment):
    payload = {"results": [_result("case-1", "CONFIRMED")]}
    decision = _decision("case-1", "CONFIRMED")
    decisions = [decision]
    if mutate is None:
        decisions.append(_decision("case-1", "CONFIRMED"))
    else:
        mutate(decision)
    evaluation, decision_path = _files(tmp_path, payload, decisions)

    result = main([
        "--evaluation", str(evaluation),
        "--decisions", str(decision_path),
        "--output", str(tmp_path / "merged.json"),
    ])

    assert result == 2
    assert error_fragment in capsys.readouterr().err

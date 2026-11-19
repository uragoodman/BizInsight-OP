"""P3 evaluation contract: rules-only metrics and explicitly unrun LLM groups."""

from pathlib import Path
import pytest

from evaluation.dimension_role_eval import run

DATASET = Path(__file__).resolve().parents[1] / "evaluation" / "datasets" / "dimension_role_cases_v1.jsonl"


def test_p3_evaluator_reports_rules_only_and_does_not_fake_llm_groups(tmp_path) -> None:
    payload = run(DATASET, tmp_path / "dimension-role-p3.json")
    comparison = payload["summary"]["p3_comparison"]

    assert comparison["dataset_count"] == 49
    rules = comparison["A_rules_only"]
    assert rules["mention_f1"] >= 0.9
    assert rules["role_f1"] >= 0.9
    assert rules["metrics_scope"] == "role_and_mention_only"
    assert rules["coverage"] is None
    assert rules["wrong_contract_rate"] is None
    assert rules["wrong_execution_rate"] is None
    assert comparison["legacy_P3_A_rules_metrics"]["wrong_execution_note"].startswith("Proxy")
    assert comparison["B_llm_only"]["status"] == "not_run"
    assert comparison["C_hybrid"]["status"] == "not_run"
    assert comparison["B_llm_only"]["mention_f1"] is None
    assert comparison["C_hybrid"]["mention_f1"] is None
    for group in (comparison["B_llm_only"], comparison["C_hybrid"]):
        assert group["metrics_scope"] == "role_and_mention_only"
        assert group["coverage"] is None
        assert group["wrong_contract_rate"] is None
        assert group["wrong_execution_rate"] is None


@pytest.mark.parametrize(
    ("enable_llm_groups", "approval_marker", "raw_question_opt_in", "expected_reason"),
    [
        (False, "approved", True, "cli_opt_in_missing"),
        (True, "approved", False, "external_question_opt_in_not_enabled"),
        (True, "", True, "human_approval_marker_missing"),
    ],
)
def test_p3_llm_comparison_requires_every_explicit_gate(
    tmp_path,
    enable_llm_groups,
    approval_marker,
    raw_question_opt_in,
    expected_reason,
) -> None:
    payload = run(
        DATASET,
        tmp_path / "dimension-role-p3-gated.json",
        comparison="all",
        enable_llm_groups=enable_llm_groups,
        approval_marker=approval_marker,
        raw_question_opt_in=raw_question_opt_in,
    )
    comparison = payload["summary"]["p3_comparison"]

    assert comparison["requested_comparison"] == "all"
    assert comparison["llm_group_gates_satisfied"] is False
    assert comparison["B_llm_only"]["status"] == "not_run"
    assert comparison["B_llm_only"]["reason"] == expected_reason
    assert comparison["C_hybrid"]["status"] == "not_run"
    assert comparison["C_hybrid"]["reason"] == expected_reason


def test_p3_all_gates_without_real_adapter_stays_not_run(tmp_path) -> None:
    payload = run(
        DATASET,
        tmp_path / "dimension-role-p3-no-adapter.json",
        comparison="all",
        enable_llm_groups=True,
        approval_marker="human-approved-eval-run",
        raw_question_opt_in=True,
    )
    comparison = payload["summary"]["p3_comparison"]

    assert comparison["llm_group_gates_satisfied"] is True
    assert comparison["B_llm_only"]["status"] == "not_run"
    assert comparison["B_llm_only"]["reason"] == "comparison_adapter_not_implemented"
    assert comparison["C_hybrid"]["status"] == "not_run"
    assert comparison["C_hybrid"]["reason"] == "comparison_adapter_not_implemented"
    for group in (comparison["B_llm_only"], comparison["C_hybrid"]):
        assert group["metrics_scope"] == "role_and_mention_only"
        assert group["coverage"] is None
        assert group["wrong_contract_rate"] is None
        assert group["wrong_execution_rate"] is None


def test_p3_synthetic_adapter_is_not_accepted_as_real_adapter(tmp_path) -> None:
    class SyntheticAdapter:
        is_real_adapter = False

        def predict_llm_only(self, _request):
            raise AssertionError("synthetic adapter must not be called by the real B/C runner")

        def predict_verifier(self, _request):
            raise AssertionError("synthetic adapter must not be called by the real B/C runner")

    payload = run(
        DATASET,
        tmp_path / "dimension-role-p3-synthetic.json",
        comparison="all",
        enable_llm_groups=True,
        approval_marker="human-approved-eval-run",
        raw_question_opt_in=True,
        comparison_adapter=SyntheticAdapter(),
        authorization_preflight=lambda *_: True,
        hybrid_support_preflight=lambda *_: True,
    )
    comparison = payload["summary"]["p3_comparison"]

    assert comparison["B_llm_only"]["status"] == "not_run"
    assert comparison["B_llm_only"]["reason"] == "comparison_adapter_not_implemented"
    assert comparison["C_hybrid"]["status"] == "not_run"
    assert comparison["C_hybrid"]["reason"] == "comparison_adapter_not_implemented"

"""Offline P3b role comparison and Hybrid arbitration safety tests.

All predictors in this module are synthetic fixtures. Their scores are contract
tests for the harness and are never evidence of real LLM quality.
"""

from __future__ import annotations

from evaluation.dimension_role_comparison import compare, hybrid_roles, score_roles


def _case(case_id: str = "one", preflight_failed: bool = False) -> dict:
    text = "店铺按品类分析"
    return {
        "case_id": case_id,
        "input": text,
        "mentions": [
            {"id": "m1", "raw_text": "店铺", "start": 0, "end": 2, "canonical": "seller"},
            {"id": "m2", "raw_text": "品类", "start": 3, "end": 5, "canonical": "product_category"},
        ],
        "roles": {"seller": "reference", "product_category": "grouping"},
        "preflight_failed": preflight_failed,
    }


def _row(mention: dict, role: str) -> dict:
    return {
        "id": mention["id"],
        "start": mention["start"],
        "end": mention["end"],
        "canonical": mention["canonical"],
        "role": role,
    }


def _rule_rows(case: dict) -> list[dict]:
    roles = {"seller": "reference", "product_category": "grouping"}
    return [_row(item, roles[item["canonical"]]) for item in case["mentions"]]


def _gold_rows(case: dict) -> list[dict]:
    return [_row(item, case["roles"][item["canonical"]]) for item in case["mentions"]]


def test_hybrid_only_relabels_existing_eligible_mentions_by_id_and_exact_span():
    case = _case()
    rules = _rule_rows(case)
    verifier = [_row(case["mentions"][0], "grouping")]

    result = hybrid_roles(
        rules,
        verifier,
        mentions={item["id"]: item for item in case["mentions"]},
        risk_tier="open",
        eligible_mentions=["m1"],
        blocked=False,
    )

    assert result == [_row(case["mentions"][0], "grouping"), rules[1]]


def test_hybrid_rejects_unknown_id_span_change_and_extra_fields_as_a_whole():
    case = _case()
    rules = _rule_rows(case)
    mentions = {item["id"]: item for item in case["mentions"]}
    invalid_responses = [
        [_row(case["mentions"][0], "grouping"), {**_row(case["mentions"][1], "grouping"), "id": "new-dimension"}],
        [{**_row(case["mentions"][0], "grouping"), "start": 1}],
        [{**_row(case["mentions"][0], "grouping"), "canonical": "order_count"}],
        [{**_row(case["mentions"][0], "grouping"), "scope": {"seller_id": "other"}}],
        [{**_row(case["mentions"][0], "grouping"), "role": "new_role"}],
    ]
    for response in invalid_responses:
        assert hybrid_roles(
            rules, response, mentions=mentions, risk_tier="conflict", eligible_mentions=["m1", "m2"], blocked=False,
        ) == rules


def test_hybrid_never_deletes_unmentioned_rule_labels_or_unapproved_dimensions():
    case = _case()
    rules = _rule_rows(case)
    mentions = {item["id"]: item for item in case["mentions"]}

    result = hybrid_roles(
        rules,
        [_row(case["mentions"][0], "grouping")],
        mentions=mentions,
        risk_tier="elevated",
        eligible_mentions=["m1", "m2"],
        blocked=False,
    )

    assert result == [_row(case["mentions"][0], "grouping"), rules[1]]
    assert len(result) == len(rules)
    assert {item["id"] for item in result} == {"m1", "m2"}


def test_hybrid_preserves_rules_for_high_denied_scope_date_and_support_failures():
    case = _case()
    rules = _rule_rows(case)
    mentions = {item["id"]: item for item in case["mentions"]}
    proposed = [_row(case["mentions"][0], "grouping")]

    for risk_tier, blocked in [
        ("high", False),
        ("denied", True),
        ("scope_failed", True),
        ("date_failed", True),
        ("support_failed", True),
    ]:
        assert hybrid_roles(
            rules,
            proposed,
            mentions=mentions,
            risk_tier=risk_tier,
            eligible_mentions=["m1"],
            blocked=blocked,
        ) == rules


def test_hybrid_unclear_abstain_and_ineligible_mentions_keep_rule_roles():
    case = _case()
    rules = _rule_rows(case)
    mentions = {item["id"]: item for item in case["mentions"]}

    assert hybrid_roles(
        rules,
        [_row(case["mentions"][0], "unclear")],
        mentions=mentions,
        risk_tier="open",
        eligible_mentions=["m1"],
        blocked=False,
    ) == rules
    assert hybrid_roles(
        rules,
        {"roles": [], "abstain": True},
        mentions=mentions,
        risk_tier="open",
        eligible_mentions=["m1"],
        blocked=False,
    ) == rules
    assert hybrid_roles(
        rules,
        [_row(case["mentions"][0], "grouping")],
        mentions=mentions,
        risk_tier="low",
        eligible_mentions=["m1"],
        blocked=False,
    ) == rules
    # Rule/verifier agreement is idempotent.
    assert hybrid_roles(
        rules,
        [_row(case["mentions"][1], "grouping")],
        mentions=mentions,
        risk_tier="conflict",
        eligible_mentions=["m2"],
        blocked=False,
    ) == rules


def test_compare_marks_fake_predictors_synthetic_and_reports_only_role_metrics():
    cases = [_case("a"), _case("b")]
    result = compare(
        cases,
        rules_predictor=_rule_rows,
        llm_predictor=_gold_rows,
        verifier_predictor=_gold_rows,
        risk_by_case=lambda _: "open",
        blocked_by_case=lambda _: False,
        post_hybrid_preflight=lambda _case, _roles: True,
    )

    assert result["dataset_count"] == 2
    assert result["evaluated_case_count"] == 2
    assert result["blocked_case_count"] == 0
    assert result["synthetic_test_only"] is True
    assert result["result_provenance"] == "synthetic_test_only"
    assert result["A_rules_only"]["case_count"] == 2
    assert result["B_llm_only"]["case_count"] == 2
    assert result["C_hybrid"]["case_count"] == 2
    assert result["B_llm_only"]["role_f1"] == 1.0
    assert result["C_hybrid"]["role_f1"] >= result["A_rules_only"]["role_f1"]
    for group in ("A_rules_only", "B_llm_only", "C_hybrid"):
        assert result[group]["metrics_scope"] == "role_and_mention_only"
        assert result[group]["coverage"] is None
        assert result[group]["wrong_contract_rate"] is None
        assert result[group]["wrong_execution_rate"] is None
    assert "does not call a provider" in result["note"]


def test_comparison_does_not_call_b_or_verifier_predictors_when_preflight_blocks():
    cases = [_case("authz", preflight_failed=True), _case("date", preflight_failed=True)]
    b_calls = []
    verifier_calls = []

    def forbidden_b(_case):
        b_calls.append(True)
        raise AssertionError("B adapter must not run for blocked scope")

    def forbidden_verifier(_case):
        verifier_calls.append(True)
        raise AssertionError("Verifier must not run for blocked scope")

    result = compare(
        cases,
        rules_predictor=_rule_rows,
        llm_predictor=forbidden_b,
        verifier_predictor=forbidden_verifier,
        risk_by_case=lambda _: "open",
        blocked_by_case=lambda case: case["preflight_failed"],
    )

    assert not b_calls
    assert not verifier_calls
    assert result["evaluated_case_count"] == 0
    assert result["blocked_case_count"] == 2
    assert result["B_llm_only"]["case_count"] == 2
    assert result["B_llm_only"]["mention_recall"] == 0.0
    assert result["B_llm_only"]["false_negative_roles"] == 4


def test_missing_authorization_preflight_fails_closed_before_b_or_verifier():
    case = _case("missing-preflight")
    calls = []

    def forbidden(_case):
        calls.append(True)
        raise AssertionError("model predictor must not run without trusted preflight")

    result = compare(
        [case],
        rules_predictor=_rule_rows,
        llm_predictor=forbidden,
        verifier_predictor=forbidden,
        risk_by_case=lambda _: "open",
    )

    assert calls == []
    assert result["evaluated_case_count"] == 0
    assert result["blocked_case_count"] == 1


def test_hybrid_preflight_exception_restores_rules():
    case = _case("preflight-error")
    rules = _rule_rows(case)

    def broken_recheck(_case, _roles):
        raise PermissionError("support snapshot is unavailable")

    result = compare(
        [case],
        rules_predictor=lambda _: rules,
        llm_predictor=_gold_rows,
        verifier_predictor=lambda _: [_row(case["mentions"][0], "grouping")],
        risk_by_case=lambda _: "open",
        blocked_by_case=lambda _: False,
        post_hybrid_preflight=broken_recheck,
    )

    assert result["C_hybrid"]["role_f1"] == result["A_rules_only"]["role_f1"]


def test_hybrid_post_support_authorization_recheck_failure_restores_rules():
    case = _case()
    rule = _rule_rows(case)
    result = compare(
        [case],
        rules_predictor=lambda _: rule,
        llm_predictor=_gold_rows,
        verifier_predictor=lambda _: [_row(case["mentions"][0], "grouping")],
        risk_by_case=lambda _: "open",
        blocked_by_case=lambda _: False,
        post_hybrid_preflight=lambda _case, _roles: False,
    )

    assert result["C_hybrid"]["role_f1"] == result["A_rules_only"]["role_f1"]


def test_shared_scorer_counts_missing_gold_and_extra_compiled_spans():
    case = _case("score")
    first = case["mentions"][0]
    extra = {"id": "compiled-fp-1", "start": 2, "end": 3, "canonical": "region", "role": "filter"}
    scores = score_roles(
        [case],
        {"score": [_row(first, "reference"), extra]},
        allow_unmatched_mentions=True,
    )

    assert scores["case_count"] == 1
    assert scores["false_negative_roles"] == 1
    assert scores["false_positive_roles"] == 1
    assert scores["mention_precision"] == 0.5
    assert scores["mention_recall"] == 0.5
    assert scores["metrics_scope"] == "role_and_mention_only"

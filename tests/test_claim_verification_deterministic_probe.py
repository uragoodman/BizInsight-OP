from evaluation.claim_verification.deterministic_probe import probe


def _case(answer):
    return {
        "case_id": "blind", "answer": answer, "request_contract": {"metric_id": "m"},
        "evidence_bundles": [{"evidence_id": "e", "trusted": True, "content": {"metric_id": "m", "result": {
            "status": "ok", "parsed": {"metric_name": "m", "authorized_metric": {"metric_id": "m"}},
            "compare_result": {"status": "ok", "metric_name": "m", "baseline_date": "2020-01-01", "baseline_value": 1, "current_date": "2020-01-02", "current_value": 2},
        }}}],
    }


def test_table_mismatch_compares_to_authorized_evidence():
    result = probe(_case("| 基线日 | 2020-01-01 | 9 |"))
    assert result["status"] == "mismatch_found"
    assert result["checks"][0]["expected"] == 1
    assert result["whole_answer_verified"] is False


def test_unsupported_text_never_becomes_pass():
    result = probe(_case("数值是 9，因此用户购买力恢复。"))
    assert result["status"] == "unverified"
    assert result["checks"] == []


def test_component_share_is_not_parsed_as_total_change():
    case = _case("占总变化 38.9%")
    result = case["evidence_bundles"][0]["content"]["result"]
    result["contribution_result"] = {"status": "ok", "metric_name": "m", "total_delta": -18}
    result["compare_result"] = None
    assert probe(case)["checks"] == []


def test_trend_value_mismatch_checks_date_against_evidence():
    case = _case("从 2020-01-01 的 9.0 到 2020-01-02 的 2.0")
    result = case["evidence_bundles"][0]["content"]["result"]
    result["trend_result"] = {"status": "ok", "metric_name": "m", "points": [{"bucket": "2020-01-01", "value": 1}, {"bucket": "2020-01-02", "value": 2}]}
    result["compare_result"] = None
    checked = probe(case)
    assert checked["status"] == "mismatch_found"
    assert len(checked["checks"]) == 2


def test_ambiguous_duplicate_trend_date_is_not_checked():
    case = _case("2020-01-01：9")
    result = case["evidence_bundles"][0]["content"]["result"]
    result["trend_result"] = {"status": "ok", "metric_name": "m", "points": [{"bucket": "2020-01-01", "value": 1}, {"bucket": "2020-01-01", "value": 2}]}
    result["compare_result"] = None
    assert probe(case)["status"] == "unverified"


def test_trend_and_contribution_evidence_do_not_hide_total_mismatch():
    case = _case("总变化 +18 单")
    result = case["evidence_bundles"][0]["content"]["result"]
    result["compare_result"] = None
    result["contribution_result"] = {"status": "ok", "metric_name": "m", "total_delta": -18}
    case["evidence_bundles"].append({"evidence_id": "trend", "trusted": True, "content": {"metric_id": "m", "result": {"status": "ok", "parsed": {"metric_name": "m", "authorized_metric": {"metric_id": "m"}}, "trend_result": {"status": "ok", "metric_name": "m", "points": []}}}})
    assert probe(case)["status"] == "mismatch_found"


def test_untrusted_wrong_metric_and_duplicate_ids_are_not_evidence():
    import copy
    case = _case("| 基线日 | 2020-01-01 | 9 |")
    case["evidence_bundles"][0]["trusted"] = False
    assert probe(case)["status"] == "unverified"
    case = _case("| 基线日 | 2020-01-01 | 9 |")
    case["evidence_bundles"][0]["content"]["metric_id"] = "other"
    assert probe(case)["status"] == "unverified"
    case = _case("| 基线日 | 2020-01-01 | 9 |")
    case["evidence_bundles"].append(copy.deepcopy(case["evidence_bundles"][0]))
    assert probe(case)["status"] == "unverified"


def test_plain_role_lines_compare_date_and_role():
    result = probe(_case("- 基线日 2020-01-01：9\n- 当前日 2020-01-02：2"))
    assert result["status"] == "mismatch_found"
    assert result["checks"][0]["evidence_field"] == "baseline_value"


def test_markdown_bold_date_and_value_match_point_evidence():
    case = _case("在 **2020-01-01** 的数值为 **9.0**")
    result = case["evidence_bundles"][0]["content"]["result"]
    result["compare_result"] = None
    result["metric_result"] = {"status": "ok", "metric_name": "m", "metric_date": "2020-01-01", "value": 1.0}
    assert probe(case)["status"] == "mismatch_found"


def _authorized_percent_case(answer):
    from app.metrics.registry import get_registry

    registry = get_registry()
    metric = "freight_rate"
    return {
        "case_id": "percent-unit-review",
        "answer": answer,
        "request_contract": {"metric_id": metric},
        "evidence_bundles": [{
            "evidence_id": "ev-authorized-metric",
            "trusted": True,
            "content": {"metric_id": metric, "result": {
                "status": "ok",
                "parsed": {"metric_name": metric, "authorized_metric": {
                    "metric_id": metric, "catalog_digest": registry.catalog_digest,
                }},
                "trend_result": {"status": "ok", "metric_name": metric, "points": []},
            }},
        }],
    }


def test_registered_percent_delta_cannot_be_labeled_percent_instead_of_points():
    result = probe(_authorized_percent_case("变化 +0.84 个百分点；绝对差值单位为%。"))
    assert result["status"] == "mismatch_found"
    unit_check = next(check for check in result["checks"] if check["evidence_field"] == "absolute_change_unit")
    assert unit_check["actual"] == "%"
    assert unit_check["expected"] == "percentage_points"
    assert unit_check["evidence_id"] == "ev-authorized-metric"


def test_registered_percent_delta_accepts_points_and_keeps_relative_percent_distinct():
    result = probe(_authorized_percent_case(
        "绝对差值单位为百分点；数值贡献 0.84 个百分点；相对变化率为 5.6%。"
    ))
    assert not any(check["evidence_field"] == "absolute_change_unit" and not check["matched"]
                   for check in result["checks"])


def test_percent_delta_unit_check_requires_successful_trusted_authorized_evidence():
    case = _authorized_percent_case("绝对差值单位为%。")
    case["evidence_bundles"][0]["trusted"] = False
    assert probe(case)["status"] == "unverified"
    case = _authorized_percent_case("绝对差值单位为%。")
    case["evidence_bundles"][0]["content"]["result"]["parsed"]["authorized_metric"]["catalog_digest"] = "wrong"
    assert probe(case)["status"] == "unverified"

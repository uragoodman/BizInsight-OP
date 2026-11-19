from app.security.result_policy import ResultPolicyDenied, apply_result_policy


def test_result_policy_suppresses_explicit_small_cells(monkeypatch):
    monkeypatch.setenv("RESULT_POLICY_MIN_CELL_SIZE", "5")
    from app.core.config import get_settings
    get_settings.cache_clear()
    decision = apply_result_policy({"sample_size": 2, "value": 12, "count": 2})
    assert decision.payload["value"] is None
    assert decision.payload["suppressed"] is True
    assert decision.suppressed_cells == 1
    get_settings.cache_clear()


def test_result_policy_masks_and_requires_export_approval(monkeypatch):
    monkeypatch.setenv("RESULT_POLICY_MASK_FIELDS_JSON", '{"email":"[REDACTED]"}')
    from app.core.config import get_settings
    get_settings.cache_clear()
    decision = apply_result_policy({"email": "a@example.com", "value": 3})
    assert decision.payload["email"] == "[REDACTED]"
    try:
        apply_result_policy({"value": 1}, action="export")
    except ResultPolicyDenied as exc:
        assert exc.code == "EXPORT_APPROVAL_REQUIRED"
    else:
        raise AssertionError("export must require approval")
    get_settings.cache_clear()


def test_result_policy_suppresses_high_dimensional_derived_detail(monkeypatch):
    monkeypatch.setenv("RESULT_POLICY_MAX_DERIVED_DIMENSIONS", "2")
    from app.core.config import get_settings
    get_settings.cache_clear()
    decision = apply_result_policy({
        "dimension_count": 3,
        "dimensions": ["seller", "product", "day"],
        "value": 42,
    })
    assert decision.payload["value"] is None
    assert decision.payload["suppression_reason"] == "derived_output_privacy"
    assert decision.suppressed_cells == 1
    get_settings.cache_clear()

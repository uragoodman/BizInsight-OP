from app.security.result_policy import apply_result_policy


def test_result_policy_keeps_legacy_payloads_compatible():
    result = apply_result_policy({"metric_name": "gmv", "value": 12})
    assert result.payload == {"metric_name": "gmv", "value": 12}

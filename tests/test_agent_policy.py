from evaluation.agent_policy import _select_candidate
from evaluation.scenario_library import BASE
from evaluation.scenario_library_v5 import generate_scenarios


def test_same_direction_larger_confounded_metric_does_not_override_direct_gmv_factor():
    # Use an actual generated counterexample: delay and GMV both fall, and
    # delay's percentage change is larger than the direct AOV factor.
    spec = next(item for item in generate_scenarios() if item.pattern == "same_direction_noise")
    aov_change = 100 * (spec.values["avg_order_value"] / BASE["avg_order_value"] - 1)
    delay_change = 100 * (spec.values["delivery_delay_rate"] / BASE["delivery_delay_rate"] - 1)
    assert delay_change < aov_change < 0
    candidates = [
        (abs(delay_change), "delivery_delay_rate", True),
        (abs(aov_change), "avg_order_value", True),
    ]

    selected = _select_candidate(candidates, target_metric="gmv")

    assert selected == (abs(aov_change), "avg_order_value", True)


def test_direct_gmv_factor_moving_against_target_does_not_mask_aligned_related_signal():
    candidates = [
        (20.0, "order_count", False),
        (15.0, "delivery_delay_rate", True),
    ]

    selected = _select_candidate(candidates, target_metric="gmv")

    assert selected == (15.0, "delivery_delay_rate", True)


def test_gmv_direct_factor_preference_is_not_applied_to_other_targets():
    candidates = [
        (70.0, "delivery_delay_rate", True),
        (45.0, "avg_order_value", True),
    ]

    selected = _select_candidate(candidates, target_metric="seller_avg_freight_per_order")

    assert selected == (70.0, "delivery_delay_rate", True)

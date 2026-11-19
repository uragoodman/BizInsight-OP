import pytest

from app.analysis.attribution_methods import AttributionDataError, resolve_product_factor_values
from app.metrics.contracts import AttributionFactorSpec


def test_product_factor_values_resolve_from_registered_additive_sources():
    specs = (
        AttributionFactorSpec(factor_id="items", kind="sum", metric_id="item_count"),
        AttributionFactorSpec(
            factor_id="unit_price", kind="ratio_of_sums",
            numerator_metric="revenue", denominator_metric="item_count",
        ),
    )

    baseline, current = resolve_product_factor_values(
        specs,
        {"item_count": (10, 12), "revenue": (50, 72)},
    )

    assert baseline == {"items": 10.0, "unit_price": 5.0}
    assert current == {"items": 12.0, "unit_price": 6.0}
    assert baseline["items"] * baseline["unit_price"] == 50
    assert current["items"] * current["unit_price"] == 72


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), float("-inf")])
def test_product_factor_values_reject_non_finite_and_boolean_sources(bad):
    specs = (
        AttributionFactorSpec(factor_id="units", kind="sum", metric_id="units"),
        AttributionFactorSpec(
            factor_id="price", kind="ratio_of_sums",
            numerator_metric="revenue", denominator_metric="units",
        ),
    )
    with pytest.raises(AttributionDataError):
        resolve_product_factor_values(specs, {"units": (10, 12), "revenue": (bad, 72)})


def test_product_factor_values_reject_zero_ratio_denominator():
    specs = (
        AttributionFactorSpec(factor_id="units", kind="sum", metric_id="units"),
        AttributionFactorSpec(
            factor_id="unit_price", kind="ratio_of_sums",
            numerator_metric="revenue", denominator_metric="units",
        ),
    )
    with pytest.raises(AttributionDataError, match="positive denominators"):
        resolve_product_factor_values(specs, {"units": (0, 12), "revenue": (0, 72)})

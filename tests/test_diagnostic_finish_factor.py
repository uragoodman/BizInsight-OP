from __future__ import annotations

import pytest

from app.graph.nodes.diagnostic import _single_factor_explaining_delta


@pytest.mark.parametrize(
    ("components", "delta", "expected"),
    [
        ({"order_count": -400.0, "avg_order_value": 0.0}, -400.0, "order_count"),
        ({"order_count": 0.0, "avg_order_value": 500.0}, 500.0, "avg_order_value"),
        ({"order_count": -200.0, "avg_order_value": -200.0}, -400.0, None),
        ({"order_count": 0.01, "avg_order_value": -0.01}, 0.0, None),
        ({"order_count": 0.0, "avg_order_value": 0.0}, 0.01, None),
        ({"order_count": float("nan"), "avg_order_value": 0.0}, 1.0, None),
    ],
)
def test_single_numerical_factor_requires_unique_material_contribution(
    components: dict[str, float], delta: float, expected: str | None,
) -> None:
    assert _single_factor_explaining_delta(components, delta) == expected

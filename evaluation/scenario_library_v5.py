"""Expanded, parameterized root-cause scenarios for the v5 evaluation set.

These cases extend the frozen v4 100-case suite. They vary causal magnitudes,
confounder strength, missing-data depth, and dimension concentration; dates are
only identifiers and are not treated as scenario variation by themselves.
"""

from __future__ import annotations

from datetime import date, timedelta

from evaluation.scenario_library import BASE, ScenarioSpec


def _spread(low: float, high: float, count: int) -> list[float]:
    if count == 1:
        return [low]
    return [low + (high - low) * index / (count - 1) for index in range(count)]


def _case(
    index: int,
    pattern: str,
    order: float,
    aov: float,
    *,
    delay: float = 1.0,
    expect: dict,
    missing_tail: int = 0,
    dimension_share: float | None = None,
    question: str | None = None,
) -> ScenarioSpec:
    day = (date(2017, 1, 1) + timedelta(days=index)).isoformat()
    values = {
        "gmv": round(BASE["order_count"] * order * BASE["avg_order_value"] * aov * (0.5 if dimension_share is not None else 1.0), 2),
        "order_count": round(BASE["order_count"] * order, 2),
        "avg_order_value": round(BASE["avg_order_value"] * aov, 2),
        "delivery_delay_rate": round(BASE["delivery_delay_rate"] * delay, 2),
    }
    if question is None:
        question = f"为什么 {day} 的 GMV 下降了？"
    spec = ScenarioSpec(
        scenario_id=f"v5-{pattern}-{index:03d}",
        scenario_date=day,
        pattern=pattern,
        query=question,
        expect=expect,
        values=values,
        missing_tail=missing_tail,
        concentrated_dimension=dimension_share is not None,
    )
    # Kept as an explicit per-case fixture attribute; scenario specs are not
    # serialized with arbitrary values, so the fixture table reads this map.
    spec.dimension_share = dimension_share
    return spec


def generate_scenarios() -> list[ScenarioSpec]:
    """Generate 255 distinct long-tail cases across 14 patterns."""
    specs: list[ScenarioSpec] = []
    i = 0

    def add(pattern: str, order: float, aov: float, **kwargs) -> None:
        nonlocal i
        specs.append(_case(i, pattern, order, aov, **kwargs))
        i += 1

    # Single-factor effects test a range of magnitudes and both directions.
    for aov in _spread(0.35, 0.82, 20):
        add("aov_drop", 1.0, aov, expect={"found_factor": "avg_order_value"})
    for order in _spread(0.35, 0.82, 20):
        add("order_drop", order, 1.0, expect={"found_factor": "order_count"})
    for aov in _spread(1.18, 1.85, 20):
        add("aov_rise", 1.0, aov, expect={"found_factor": "avg_order_value"},
            question=f"解释 {date(2017, 1, 1) + timedelta(days=i)} GMV 上升的主要因素")
    for order in _spread(1.18, 1.85, 20):
        add("order_rise", order, 1.0, expect={"found_factor": "order_count"},
            question=f"解释 {date(2017, 1, 1) + timedelta(days=i)} GMV 上升的主要因素")

    # Correlated noise moves more than the real factor; light noise moves less.
    for aov, delay in zip(_spread(0.38, 0.76, 20), _spread(2.0, 5.0, 20)):
        add("delay_noise", 1.0, aov, delay=delay,
            expect={"found_factor": "avg_order_value"},
            question=f"{date(2017, 1, 1) + timedelta(days=i)} GMV 下降且延迟送达率升高，判断主要原因")
    for aov, delay in zip(_spread(0.38, 0.78, 10), _spread(1.03, 1.25, 10)):
        add("delay_light", 1.0, aov, delay=delay,
            expect={"found_factor": "avg_order_value"},
            question=f"{date(2017, 1, 1) + timedelta(days=i)} \u8bf7\u89e3\u91ca GMV \u53d8\u5316\u539f\u56e0\uff0c\u5ef6\u8fdf\u7387\u53ea\u6709\u8f7b\u5fae\u53d8\u5316")

    # Same-direction confounder: delay falls more than GMV/AOV, but AOV is
    # a direct GMV factor in this offline rule, while delay is only correlated.
    for aov, delay in ((0.35, 0.12), (0.55, 0.22), (0.75, 0.30)):
        add("same_direction_noise", 1.0, aov, delay=delay,
            expect={"found_factor": "avg_order_value"},
            question=f"{date(2017, 1, 1) + timedelta(days=i)} \u8bf7\u5f52\u56e0 GMV \u4e0b\u964d\uff0c\u5ef6\u8fdf\u9001\u8fbe\u7387\u4e5f\u4e0b\u964d\u4e14\u53d8\u5316\u66f4\u5927\u7684\u60c5\u51b5")

    # Both factors contribute, with distinct pairings and ordering.
    for index in range(24):
        order = 0.38 + 0.019 * index
        aov = 0.84 - 0.017 * index
        add("both_move", order, aov,
            expect={"found_factor": ["avg_order_value", "order_count"]})

    # Opposing movements test net direction versus factor attribution.
    for index in range(20):
        order = 1.12 + 0.014 * index
        aov = 0.68 - 0.014 * index
        add("opposing", order, aov, expect={"found_factor": "avg_order_value"},
            question=f"{date(2017, 1, 1) + timedelta(days=i)} GMV 下滑但订单量增长，归因于什么")
    for index in range(16):
        order = 0.68 - 0.014 * index
        aov = 1.10 + 0.012 * index
        add("reverse_opposing", order, aov, expect={"found_factor": "order_count"},
            question=f"{date(2017, 1, 1) + timedelta(days=i)} GMV 下滑但客单价增长，归因于什么")

    # Near-flat combinations remain under the materiality boundary.
    for index in range(20):
        order = 0.94 + 0.006 * index
        aov = 1.04 - 0.004 * index
        add("flat", order, aov, expect={"abstain": True},
            question=f"{date(2017, 1, 1) + timedelta(days=i)} 的 GMV 是否有值得归因的异常？")

    # Incomplete history varies both the missing-tail depth and observed values.
    for index in range(20):
        order = 0.97 + 0.002 * index
        aov = 0.96 + 0.001 * index
        add("incomplete", order, aov, missing_tail=1 + index % 6,
            expect={"abstain": True},
            question=f"{date(2017, 1, 1) + timedelta(days=i)} GMV 数据有缺口，能否判断原因？")

    # Below the legacy 30% cutoff but above the current 10% floor.
    for index in range(20):
        magnitude = 0.12 + 0.008 * index
        if index % 2 == 0:
            add("sub_threshold", 1.0, 1.0 - magnitude,
                expect={"found_factor": "avg_order_value"},
                question=f"{date(2017, 1, 1) + timedelta(days=i)} GMV 下降幅度不大，仍应找原因吗？")
        else:
            add("sub_threshold", 1.0 - magnitude, 1.0,
                expect={"found_factor": "order_count"},
                question=f"{date(2017, 1, 1) + timedelta(days=i)} 小幅订单变化是否解释 GMV 下降？")

    # Dimension-only explanations vary the dominant category's share.
    for share in _spread(0.55, 0.95, 22):
        add("dimension_driven", 1.0, 1.0,
            expect={"found_dimension": True}, dimension_share=share,
            question=f"{date(2017, 1, 1) + timedelta(days=i)} GMV \u4e0b\u964d\u7684\u539f\u56e0\u662f\u4ec0\u4e48\uff1f\u8bf7\u67e5\u627e GMV \u8d21\u732e\u6700\u5927\u7684\u7ef4\u5ea6")

    assert len(specs) == 255, len(specs)
    assert len({spec.scenario_date for spec in specs}) == len(specs)
    assert len({spec.scenario_id for spec in specs}) == len(specs)
    return specs


def scenario_dicts() -> list[dict]:
    return [
        {
            "id": spec.scenario_id,
            "category": "root_cause_longtail",
            "pattern": spec.pattern,
            "query": spec.query,
            "expect": spec.expect,
        }
        for spec in generate_scenarios()
    ]


def truth_tables() -> tuple[dict[str, dict[str, float]], dict[str, int], dict[str, float]]:
    specs = generate_scenarios()
    truth = {spec.scenario_date: spec.values for spec in specs}
    incomplete = {spec.scenario_date: spec.missing_tail for spec in specs if spec.missing_tail}
    concentrated = {
        spec.scenario_date: float(spec.dimension_share)
        for spec in specs if spec.dimension_share is not None
    }
    return truth, incomplete, concentrated

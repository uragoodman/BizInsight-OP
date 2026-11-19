"""Parameterized root-cause scenario library (review follow-up: dataset scale).

Hand-written golden scenarios are great for narrative but too few for
statistics — with n=3, one judgment flip moves completion by 33 points.  This
module generates 82 long-tail scenarios from a small set of *patterns*
(single-factor drops/rises, correlated-but-not-causal distractors, opposing
factors, both-factors, sub-threshold moves, flat, incomplete data,
dimension-driven), each with explicit ground truth and an expect block that
``scorers.score_scenario`` understands.

One scenario = one date.  ``evaluation/fixtures.py`` merges the generated
dates into its truth tables, and ``scripts/generate_rootcause_dataset.py``
writes the golden cases from the same definitions — a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

BASE = {"gmv": 1000.0, "order_count": 100.0, "avg_order_value": 10.0, "delivery_delay_rate": 10.0}

# Patterns that the threshold rule cannot solve by construction — the whole
# point of the long-tail report is to show where the rule breaks and whether
# model judgment does better.
RULE_BREAKING_PATTERNS = {"delay_noise", "sub_threshold", "both_move", "opposing", "reverse_opposing"}


@dataclass
class ScenarioSpec:
    scenario_id: str
    scenario_date: str
    pattern: str
    query: str
    expect: dict
    values: dict = field(default_factory=dict)
    missing_tail: int = 0
    concentrated_dimension: bool = False

    @property
    def gmv(self) -> float:
        return self.values.get("gmv", BASE["gmv"])


def _q_drop(scenario_date: str) -> str:
    return f"为什么 {scenario_date} 的 GMV 下降了？"


def _q_rise(scenario_date: str) -> str:
    return f"为什么 {scenario_date} 的 GMV 上涨了？"


def _q_status(scenario_date: str) -> str:
    return f"{scenario_date} 的 GMV 情况怎么样，有异常吗？"


def _q_noise(scenario_date: str, alias: str, direction: str) -> str:
    return f"{scenario_date} GMV 下降了，{alias}也{direction}了，为什么？"


def _spec(index: int, pattern: str, order_m: float, aov_m: float, delay_m: float = 1.0,
          *, expect: dict, query: str | None = None, missing_tail: int = 0,
          concentrated_dimension: bool = False) -> ScenarioSpec:
    # Dates must sit inside the real data availability range
    # (2016-09-15 … 2018-08-29) or time_expression rejects the question before
    # the loop ever runs.  The hand-written scenarios occupy 2018-08-20…28.
    scenario_date = (date(2018, 1, 8) + timedelta(days=index)).isoformat()
    values = {
        # A concentrated dimension scenario represents a GMV decline whose
        # unit-level cause is not captured by the cross-metric factors.
        "gmv": round(BASE["order_count"] * order_m * BASE["avg_order_value"] * aov_m * (0.5 if concentrated_dimension else 1.0), 2),
        "order_count": round(BASE["order_count"] * order_m, 2),
        "avg_order_value": round(BASE["avg_order_value"] * aov_m, 2),
        "delivery_delay_rate": round(BASE["delivery_delay_rate"] * delay_m, 2),
    }
    return ScenarioSpec(
        scenario_id=f"gen-{pattern}-{index:02d}",
        scenario_date=scenario_date,
        pattern=pattern,
        query=query or _q_drop(scenario_date),
        expect=expect,
        values=values,
        missing_tail=missing_tail,
        concentrated_dimension=concentrated_dimension,
    )


def generate_scenarios() -> list[ScenarioSpec]:
    """82 generated long-tail scenarios across 12 patterns."""
    specs: list[ScenarioSpec] = []
    push = specs.append
    index = iter(range(100))

    def next_index() -> int:
        return next(index)

    # --- single-factor drops (rule-solvable): aov × {0.65 … 0.30}
    for multiplier in (0.65, 0.55, 0.50, 0.45, 0.35, 0.30):
        push(_spec(next_index(), "aov_drop", 1.0, multiplier,
                   expect={"found_factor": "avg_order_value"}))
    # --- single-factor drops: order × {0.65 … 0.30}
    for multiplier in (0.65, 0.55, 0.50, 0.45, 0.35, 0.30):
        push(_spec(next_index(), "order_drop", multiplier, 1.0,
                   expect={"found_factor": "order_count"}))
    # --- single-factor rises (gmv up, rule-solvable)
    for multiplier in (1.40, 1.60, 1.25, 1.80, 1.35, 1.50):
        push(_spec(next_index(), "aov_rise", 1.0, multiplier,
                   expect={"found_factor": "avg_order_value"}))
    for multiplier in (1.40, 1.60, 1.25, 1.80, 1.35, 1.50):
        spec = _spec(next_index(), "order_rise", multiplier, 1.0,
                     expect={"found_factor": "order_count"})
        spec.query = _q_rise(spec.scenario_date)
        push(spec)
    # --- correlated-but-not-causal: delay distractor moves MORE than the true factor
    for aov_m, delay_m in ((0.60, 4.0), (0.50, 4.0), (0.50, 2.5), (0.60, 2.5), (0.45, 3.0), (0.50, 3.5), (0.40, 5.0), (0.55, 3.0), (0.45, 4.0), (0.65, 2.8)):
        spec = _spec(next_index(), "delay_noise", 1.0, aov_m, delay_m,
                     expect={"found_factor": "avg_order_value"})
        spec.query = _q_noise(spec.scenario_date, "延迟送达率", "升高")
        push(spec)
    # --- distractor moves less than the true factor (rule-solvable)
    for aov_m, delay_m in ((0.50, 1.20), (0.60, 1.15), (0.45, 1.10), (0.55, 1.25), (0.40, 1.20), (0.65, 1.10)):
        spec = _spec(next_index(), "delay_light", 1.0, aov_m, delay_m,
                     expect={"found_factor": "avg_order_value"})
        spec.query = _q_noise(spec.scenario_date, "延迟送达率", "升高")
        push(spec)
    # --- both factors move: either is an acceptable answer
    for order_m, aov_m in ((0.80, 0.50), (0.50, 0.80), (0.65, 0.65), (0.60, 0.55), (0.75, 0.60), (0.55, 0.70), (0.70, 0.70), (0.85, 0.55)):
        push(_spec(next_index(), "both_move", order_m, aov_m,
                   expect={"found_factor": ["avg_order_value", "order_count"]}))
    # --- opposing: one factor pulls gmv down while the other rises
    for order_m, aov_m in ((1.50, 0.40), (1.30, 0.45), (1.80, 0.30), (1.25, 0.60), (1.60, 0.35), (1.40, 0.50), (1.20, 0.55), (1.70, 0.40)):
        push(_spec(next_index(), "opposing", order_m, aov_m,
                   expect={"found_factor": "avg_order_value"}))
    for order_m, aov_m in ((0.50, 1.60), (0.40, 1.50), (0.55, 1.70), (0.45, 1.80)):
        scenario_index = next_index()
        scenario_date = (date(2018, 1, 8) + timedelta(days=scenario_index)).isoformat()
        direction_query = _q_rise(scenario_date) if order_m * aov_m > 1.0 else _q_drop(scenario_date)
        push(_spec(scenario_index, "reverse_opposing", order_m, aov_m,
                   expect={"found_factor": "order_count"}, query=direction_query))
    # --- flat: nothing material → correct ABSTAIN
    for order_m, aov_m in ((1.0, 1.0), (1.02, 1.0), (1.0, 0.98), (1.01, 0.99), (1.03, 0.98), (0.99, 1.02), (1.04, 0.97), (0.98, 1.03)):
        push(_spec(next_index(), "flat", order_m, aov_m,
                   expect={"abstain": True}, query=None))
    # --- incomplete data: latest observations missing → correct ABSTAIN
    for _ in range(5):
        push(_spec(next_index(), "incomplete", 1.0, 1.0,
                   expect={"abstain": True}, query=_q_status(""),
                   missing_tail=2))
    # --- sub-threshold: a real factor below the 30% materiality rule
    for aov_m in (0.75, 0.80):
        push(_spec(next_index(), "sub_threshold", 1.0, aov_m,
                   expect={"found_factor": "avg_order_value"}))
    for order_m in (0.85, 0.80):
        push(_spec(next_index(), "sub_threshold", order_m, 1.0,
                   expect={"found_factor": "order_count"}))
    # --- dimension-driven: cross metrics flat, one category explains it all
    for _ in range(5):
        push(_spec(next_index(), "dimension_driven", 1.0, 1.0,
                   expect={"found_dimension": True}, concentrated_dimension=True))
    # fix query dates for the status/incomplete templates that used a placeholder
    for spec in specs:
        if spec.pattern in {"incomplete", "flat"}:
            spec.query = _q_status(spec.scenario_date)
        if spec.pattern == "aov_rise" and "上涨" not in spec.query:
            spec.query = _q_rise(spec.scenario_date)
    return specs


def core_scenarios() -> list[ScenarioSpec]:
    """Nine additional core-gate scenarios with deliberately varied outcomes."""
    specs = [
        _spec(100, "core_order_drop", 0.60, 1.0,
              expect={"found_factor": "order_count"}),
        _spec(101, "core_aov_rise", 1.0, 1.40,
              expect={"found_factor": "avg_order_value"}, query=None),
        _spec(102, "core_order_rise", 1.40, 1.0,
              expect={"found_factor": "order_count"}, query=None),
        _spec(103, "core_both_move", 0.80, 0.60,
              expect={"found_factor": ["avg_order_value", "order_count"]}),
        _spec(104, "core_opposing", 1.50, 0.40,
              expect={"found_factor": "avg_order_value"}),
        _spec(105, "core_reverse_opposing", 0.50, 1.60,
              expect={"found_factor": "order_count"}),
        _spec(106, "core_flat", 1.0, 1.0,
              expect={"abstain": True}, query=None),
        _spec(107, "core_incomplete", 1.0, 1.0,
              expect={"abstain": True}, query=None, missing_tail=2),
        _spec(108, "core_dimension_driven", 1.0, 1.0,
              expect={"found_dimension": True}, concentrated_dimension=True),
    ]
    for spec in specs:
        spec.scenario_id = spec.scenario_id.replace("gen-", "core-", 1)
        if spec.pattern in {"core_flat", "core_incomplete"}:
            spec.query = _q_status(spec.scenario_date)
        elif spec.pattern in {"core_aov_rise", "core_order_rise"}:
            spec.query = _q_rise(spec.scenario_date)
    return specs


def scenario_dicts() -> list[dict]:
    """Return core-gate and long-tail cases from one source of truth."""
    core = [
        {
            "id": spec.scenario_id,
            "category": "root_cause_scenario",
            "pattern": spec.pattern,
            "query": spec.query,
            "expect": spec.expect,
        }
        for spec in core_scenarios()
    ]
    longtail = [
        {
            "id": spec.scenario_id,
            "category": "root_cause_longtail",
            "pattern": spec.pattern,
            "query": spec.query,
            "expect": spec.expect,
        }
        for spec in generate_scenarios()
    ]
    return core + longtail


def truth_tables() -> tuple[dict[str, dict[str, float]], set[str], set[str]]:
    """(date → metric end values, incomplete dates, concentrated-dimension dates)."""
    truth: dict[str, dict[str, float]] = {}
    incomplete: set[str] = set()
    concentrated: set[str] = set()
    for spec in [*core_scenarios(), *generate_scenarios()]:
        truth[spec.scenario_date] = spec.values
        if spec.missing_tail:
            incomplete.add(spec.scenario_date)
        if spec.concentrated_dimension:
            concentrated.add(spec.scenario_date)
    return truth, incomplete, concentrated

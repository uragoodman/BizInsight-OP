"""Offline fixture dependencies for deterministic Golden evaluation."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import re
from uuid import uuid4

from app.analysis.contracts import MetricPoint, MetricSeries, PeriodComparison, PeriodSpec
from app.core.config import get_settings
from app.mcp.client import _build_arguments_from_plan
from app.mcp.schemas import MetricQueryPlan
from app.rag.schemas import RagCitation, RagSearchHit, RagSearchResult
from evaluation.scenario_library import truth_tables
from evaluation.scenario_library_v5 import truth_tables as v5_truth_tables

_GENERATED_TRUTH, _GENERATED_INCOMPLETE, _GENERATED_CONCENTRATED = truth_tables()
_V5_TRUTH, _V5_INCOMPLETE_TAILS, _V5_CONCENTRATION_SHARES = v5_truth_tables()
_ROOT = Path(__file__).resolve().parents[1]
_HOLDOUT_FACTS_PATH = _ROOT / "evaluation" / "fixtures" / "provider_holdout_olist_v1.json"
_HOLDOUT_DATASET_PATH = _ROOT / "evaluation" / "datasets" / "provider_holdout_olist_v1.json"
_HOLDOUT_FACTS = json.loads(_HOLDOUT_FACTS_PATH.read_text(encoding="utf-8"))["facts_by_day"] if _HOLDOUT_FACTS_PATH.exists() else {}
_HOLDOUT_DATES = {
    str(case.get("source_ground_truth", {}).get("current_date"))
    for case in json.loads(_HOLDOUT_DATASET_PATH.read_text(encoding="utf-8")).get("cases", [])
} if _HOLDOUT_DATASET_PATH.exists() else set()
_GENERATED_DATES = tuple(sorted({*_GENERATED_TRUTH, *_V5_TRUTH}))
# Dates whose latest trend observations are missing (incomplete-data scenarios).
_INCOMPLETE_DATES = {"2018-08-27"} | set(_GENERATED_INCOMPLETE) | set(_V5_INCOMPLETE_TAILS)
_INCOMPLETE_TAILS = {day: 2 for day in _INCOMPLETE_DATES} | _V5_INCOMPLETE_TAILS
# Dates where one dimension category concentrates the whole move (dimension-
# driven scenarios); every other scenario date refutes all dimensions.
_CONCENTRATION_SHARES = {
    **{day: 1.0 for day in {"2018-08-26", *_GENERATED_CONCENTRATED}},
    **_V5_CONCENTRATION_SHARES,
}


def fixture_rag(query: str, **_: object) -> RagSearchResult:
    citation = RagCitation(source_path="docs/knowledge/metrics/gmv.md", title="GMV", doc_type="metric", metric_name="gmv", chunk_id="gmv#1", index_version="golden-v1")
    return RagSearchResult(query=query, backend="fixture", strategy="local_hybrid", top_k=1, index_version="golden-v1", hits=[RagSearchHit(chunk_id="gmv#1", text="GMV is gross merchandise value.", score=1.0, citation=citation)], context="GMV is gross merchandise value.")


# ---------------------------------------------------------------------------
# Phase 0 golden scenarios: deterministic "the deterministic rule gives up" data.
#
# Each scenario date encodes a distinct ground truth on the SAME observable
# surface (GMV down, every single-attribute dimension contribution refuted):
#   - 2018-08-20  A: multiplicative factor — orders flat, AOV halved (real cause).
#   - 2018-08-21  B: no material cause — all cross-metrics flat (should ABSTAIN).
#   - 2018-08-22  C: linked anomaly — delivery_delay_rate spikes same day.
# The baseline Workflow only ever queries GMV trend + GMV dimensions, so it
# stops with ``no_material_contributor_in_examined_dimensions`` on all three.
# Cross-metric ground truth is served here so the future agent can find it.
# ---------------------------------------------------------------------------

SCENARIO_DATES: tuple[str, ...] = (
    "2018-08-20", "2018-08-21", "2018-08-22",
    # Long-tail scenarios (AGENT_PLAN review follow-up #4/#7): rule-solvable
    # and rule-breaking patterns reported separately from the frozen gate set.
    "2018-08-23", "2018-08-24", "2018-08-25", "2018-08-26", "2018-08-27", "2018-08-28",
    *_GENERATED_DATES, *_HOLDOUT_DATES,
)


def _case_pair_values(metric: str, current: str, baseline: str) -> tuple[float | None, float | None] | None:
    """Return a pair from one case's facts, keeping Olist on its own source."""
    if current in _HOLDOUT_DATES:
        current_fact, baseline_fact = _HOLDOUT_FACTS.get(current), _HOLDOUT_FACTS.get(baseline)
        if current_fact and baseline_fact:
            current_value, baseline_value = current_fact.get(metric), baseline_fact.get(metric)
            return (
                float(current_value) if current_value is not None else None,
                float(baseline_value) if baseline_value is not None else None,
            )
    if current in _SCENARIO_TRUTH:
        current_value = _daily_metric_value(metric, current, case_day=current)
        # Synthetic dates identify independent cases.  The case's baseline
        # comes from its own configured base even if the preceding date names
        # a different case.
        baseline_value = _daily_metric_value(metric, baseline, case_day=current)
        missing = _missing_case_tail(metric, current, baseline, current)
        return (
            None if current in missing else current_value,
            None if baseline in missing else baseline_value,
        )
    return None


def _metric_definition(metric: str):
    from app.metrics.registry import get_registry

    return get_registry().get(metric)


def _base_metric_value(metric: str) -> float:
    if metric in _SCENARIO_BASE:
        return float(_SCENARIO_BASE[metric])
    if metric == "freight_value":
        return float(_SCENARIO_BASE["gmv"] * 0.1)
    return 10.0


def _daily_metric_value(metric: str, day: str, *, case_day: str | None = None) -> float | None:
    """Resolve one synthetic daily fact from the scenario's shared truth row."""
    if day in _HOLDOUT_FACTS and (case_day is None or case_day in _HOLDOUT_DATES):
        fact = _HOLDOUT_FACTS[day]
        value = fact.get(metric)
        if value is None and metric == "freight_value":
            return None  # the Olist snapshot does not include this component
        if value is not None:
            return float(value)

    anchor = case_day or day
    truth = _SCENARIO_TRUTH.get(anchor, {}) if day == anchor else {}
    definition = _metric_definition(metric)
    aggregation = definition.period_aggregation
    if aggregation is not None:
        numerator = _daily_metric_value(aggregation.numerator_metric, day, case_day=case_day)
        denominator = _daily_metric_value(aggregation.denominator_metric, day, case_day=case_day)
        if numerator is None or denominator in (None, 0):
            return None
        return round(numerator / denominator * aggregation.multiplier, 6)
    if metric == "freight_value":
        gmv = _daily_metric_value("gmv", day, case_day=case_day)
        return None if gmv is None else round(gmv * 0.1, 6)
    value = truth.get(metric)
    return float(value) if value is not None else _base_metric_value(metric)


def _missing_case_tail(metric: str, case_day: str, start: str, end: str) -> set[str]:
    if metric != "gmv" or case_day not in _INCOMPLETE_DATES:
        return set()
    count = min(_INCOMPLETE_TAILS.get(case_day, 1), (date.fromisoformat(end) - date.fromisoformat(start)).days + 1)
    return {
        (date.fromisoformat(end) - timedelta(days=offset)).isoformat()
        for offset in range(count)
    }


def _synthetic_points(metric: str, start: str, end: str, *, case_day: str | None = None) -> list[dict]:
    start_date, end_date = date.fromisoformat(start), date.fromisoformat(end)
    anchor = case_day or end
    holdout_source = anchor in _HOLDOUT_DATES
    missing = _missing_case_tail(metric, anchor, start, end) if anchor in _INCOMPLETE_DATES else set()
    points: list[dict] = []
    current = start_date
    while current <= end_date:
        bucket = current.isoformat()
        if bucket in missing:
            value = None
        elif holdout_source:
            fact = _HOLDOUT_FACTS.get(bucket, {})
            raw_value = fact.get(metric)
            value = float(raw_value) if raw_value is not None else None
        else:
            value = _daily_metric_value(metric, bucket, case_day=anchor)
        points.append({"bucket": bucket, "value": value})
        current += timedelta(days=1)
    return points


def _period_metric_value(metric: str, period: PeriodSpec, *, case_day: str | None = None) -> float | None:
    """Aggregate synthetic period facts according to the metric registry."""
    from statistics import fmean

    anchor = case_day or period.end_date.isoformat()
    points = _synthetic_points(metric, period.start_date.isoformat(), period.end_date.isoformat(), case_day=anchor)
    values = [float(point["value"]) for point in points if point["value"] is not None]
    definition = _metric_definition(metric)
    if definition.period_aggregation is not None:
        spec = definition.period_aggregation
        numerator = _period_metric_value(spec.numerator_metric, period, case_day=anchor)
        denominator = _period_metric_value(spec.denominator_metric, period, case_day=anchor)
        if numerator is None or denominator in (None, 0):
            return None
        return round(numerator / denominator * spec.multiplier, 2)
    if definition.aggregation == "sum":
        # A partial sum would misrepresent a period whose configured case tail
        # is missing, so only return a total when every requested day exists.
        if len(values) != len(points):
            return None
        return round(sum(values), 2)
    if definition.aggregation == "average":
        return round(fmean(values), 2) if values else None
    if definition.aggregation == "distinct_count":
        return round(values[0], 2) if len(points) == 1 and values else None
    if definition.aggregation == "weighted_ratio":
        return None
    return round(values[-1], 2) if values else None


def _holdout_contribution(metric: str, current: str, baseline: str) -> dict | None:
    current_fact, baseline_fact = _HOLDOUT_FACTS.get(current), _HOLDOUT_FACTS.get(baseline)
    if not current_fact or not baseline_fact or metric != "gmv":
        return None
    current_items = current_fact.get("customer_state", {})
    baseline_items = baseline_fact.get("customer_state", {})
    values = sorted(set(current_items) | set(baseline_items))
    current_total, baseline_total = float(current_fact["gmv"]), float(baseline_fact["gmv"])
    return {
        "status": "ok",
        "contribution_result": {
            "status": "ok", "metric_name": metric, "current_date": current,
            "baseline_date": baseline,
            "current_items": [{"dimension_value": value, "value": float(current_items.get(value, 0))} for value in values],
            "baseline_items": [{"dimension_value": value, "value": float(baseline_items.get(value, 0))} for value in values],
            "total_delta": round(current_total - baseline_total, 2),
            "source_table": "olist-public.order_items+orders+customers",
        },
    }

_SCENARIO_TRUTH: dict[str, dict[str, float]] = {
    # Shared legacy query default, used only when a non-case request omits dates.
    "2025-03-31": {"gmv": 600.0, "order_count": 100.0, "avg_order_value": 6.0, "delivery_delay_rate": 10.0},
    # per-date end values for the cross-metric time series the agent will query
    "2018-08-20": {"gmv": 500.0, "order_count": 100.0, "avg_order_value": 5.0, "delivery_delay_rate": 10.0},
    "2018-08-21": {"gmv": 800.0, "order_count": 100.0, "avg_order_value": 10.0, "delivery_delay_rate": 10.0},
    "2018-08-22": {"gmv": 600.0, "order_count": 120.0, "avg_order_value": 10.0, "delivery_delay_rate": 40.0},
    # 08-23 two factors move together (order -20%, AOV -50% → gmv -60%)
    "2018-08-23": {"gmv": 400.0, "order_count": 80.0, "avg_order_value": 5.0, "delivery_delay_rate": 10.0},
    # 08-24 opposing factors: order +50%, AOV -60% → gmv -40%
    "2018-08-24": {"gmv": 600.0, "order_count": 150.0, "avg_order_value": 4.0, "delivery_delay_rate": 10.0},
    # 08-25 correlated-but-not-cause: delay rate spikes, but gmv is explained by AOV
    "2018-08-25": {"gmv": 500.0, "order_count": 100.0, "avg_order_value": 5.0, "delivery_delay_rate": 40.0},
    # 08-26 dimension-driven: cross metrics flat, one category concentrates the drop
    "2018-08-26": {"gmv": 500.0, "order_count": 100.0, "avg_order_value": 10.0, "delivery_delay_rate": 10.0},
    # 08-27 incomplete data: the latest gmv observations are missing
    "2018-08-27": {"gmv": 1000.0, "order_count": 100.0, "avg_order_value": 10.0, "delivery_delay_rate": 10.0},
    # 08-28 multiple significant factors needing ordering (AOV -50% > order -25%)
    "2018-08-28": {"gmv": 375.0, "order_count": 75.0, "avg_order_value": 5.0, "delivery_delay_rate": 10.0},
} | _GENERATED_TRUTH | _V5_TRUTH

# value defaults for trend bases, so a flat/unchanged series is reproducible
_SCENARIO_BASE: dict[str, float] = {
    "gmv": 1000.0, "order_count": 100.0, "avg_order_value": 10.0, "delivery_delay_rate": 10.0,
}

_METRIC_HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("delivery_delay_rate", ("延迟送达率", "延误送达率", "延迟率", "delivery_delay_rate")),
    ("avg_order_value", ("客单价", "avg_order_value", "aov")),
    ("order_count", ("订单量", "订单数", "order_count", "订单")),
    ("gmv", ("gmv", "成交额", "成交总额")),
]
_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def _metric_from_question(question: str) -> str | None:
    """Resolve the metric by the EARLIEST hint occurrence in the question.

    Executor questions embed the user message after the typed query prefix
    ("查询 order_count 趋势 …<user message>"), so first-mention wins — a
    distractor mentioned later in the user message must not hijack every
    query on the wire.
    """
    lowered = question.casefold().replace(" ", "")
    best: tuple[int, str] | None = None
    for metric_id, hints in _METRIC_HINTS:
        for hint in hints:
            index = lowered.find(str(hint).casefold().replace(" ", ""))
            if index >= 0 and (best is None or index < best[0]):
                best = (index, metric_id)
    return best[1] if best else None


def _scenario_date(question: str) -> str | None:
    for raw in _ISO_DATE.findall(question):
        if raw in SCENARIO_DATES:
            return raw
    return None


# Number of dimension values used to construct a REFUTED contribution.  The
# movement is spread evenly across many segments so no single segment explains
# >=1% of the drop (top contribution ratio = 0.5 / 100 = 0.5%).
_REFUTED_SEGMENTS = 200


def _refuted_contribution(
    metric: str,
    current_date: str,
    baseline_date: str,
    *,
    baseline_total: float,
    current_total: float,
) -> dict:
    """Evenly allocate one case's totals so no segment dominates its change."""
    def allocate(total: float) -> list[dict]:
        per_segment = round(float(total) / _REFUTED_SEGMENTS, 2)
        values = [per_segment] * (_REFUTED_SEGMENTS - 1)
        values.append(round(float(total) - sum(values), 2))
        return [{"dimension_value": f"K{index}", "value": value} for index, value in enumerate(values)]

    current_items = allocate(current_total)
    baseline_items = allocate(baseline_total)
    total_delta = round(sum(item["value"] for item in current_items) - sum(item["value"] for item in baseline_items), 2)
    return {
        "status": "ok",
        "contribution_result": {
            "status": "ok", "metric_name": metric,
            "current_date": current_date, "baseline_date": baseline_date,
            "current_items": current_items, "baseline_items": baseline_items,
            "total_delta": total_delta, "source_table": "golden.fixture",
        },
    }


def _concentrated_contribution(
    metric: str, current_date: str, baseline_date: str, share: float = 1.0,
    *, current_total: float, baseline_total: float,
) -> dict:
    """Allocate the case's observed delta according to its configured share."""
    # The scenario controls the share of the total movement explained by the
    # leading category. Both totals remain positive and the second category
    # provides a genuine competing contribution.
    share = max(0.51, min(1.0, float(share)))
    first_baseline = round(float(baseline_total) * share, 2)
    second_baseline = round(float(baseline_total) - first_baseline, 2)
    first_current = round(float(current_total) * share, 2)
    second_current = round(float(current_total) - first_current, 2)
    return {
        "status": "ok",
        "contribution_result": {
            "status": "ok", "metric_name": metric,
            "current_date": current_date, "baseline_date": baseline_date,
            "current_items": [{"dimension_value": "A", "value": first_current}, {"dimension_value": "B", "value": second_current}],
            "baseline_items": [{"dimension_value": "A", "value": first_baseline}, {"dimension_value": "B", "value": second_baseline}],
            "total_delta": round(float(current_total) - float(baseline_total), 2), "source_table": "golden.fixture",
        },
    }


def _scenario_dimension_result(metric: str, current: str, baseline: str) -> dict:
    """Build a dimension result from the same fact pair used by trend/compare."""
    pair = _case_pair_values(metric, current, baseline)
    if pair is None or pair[0] is None or pair[1] is None:
        return {
            "status": "incomplete",
            "contribution_result": {
                "status": "incomplete", "metric_name": metric,
                "current_date": current, "baseline_date": baseline,
                "reason": "required_case_observation_missing", "source_table": "golden.fixture",
            },
        }
    current_total, baseline_total = pair
    share = _CONCENTRATION_SHARES.get(current)
    if share is not None:
        return _concentrated_contribution(
            metric, current, baseline, share,
            current_total=current_total, baseline_total=baseline_total,
        )
    return _refuted_contribution(
        metric, current, baseline,
        current_total=current_total, baseline_total=baseline_total,
    )


def fixture_mcp(
    task_type: str,
    question: str,
    plan: object | None = None,
    *,
    request_id: str | None = None,
    idempotency_key: str | None = None,
    **_: object,
) -> dict:
    """Scenario-aware MCP fixture. Non-scenario calls keep the legacy default."""
    plan_model: MetricQueryPlan | None = None
    if plan is not None:
        if isinstance(plan, MetricQueryPlan):
            plan_model = plan
        else:
            # Serialize the caller's plan through the same public schema used by
            # query_business_data. This preserves the supplied authorization
            # object and leaves it absent when the caller did not supply one.
            plan_model = MetricQueryPlan.model_validate(plan)

    planned_type = plan_model.query_type if plan_model is not None else None
    def plan_value(name: str):
        return getattr(plan_model, name, None) if plan_model is not None else None

    def response(tool_name: str, arguments: dict, result: dict) -> dict:
        """Build the same trace-facing envelope as query_business_data."""
        actual_request_id = request_id or str(uuid4())
        actual_idempotency_key = idempotency_key or actual_request_id
        server = "mock-mcp"
        actual_tool_name = tool_name
        actual_arguments = arguments
        normalized_result = dict(result)

        if plan_model is not None:
            if plan_model.query_type == "metric_contribution":
                # Match the public contribution gateway's dedicated route;
                # contributions must not be serialized through the legacy
                # Registry breakdown argument builder.
                from app.mcp.client import _platform_contribution_route

                routed_tool, routed_arguments, rejection = _platform_contribution_route(plan_model)
                if rejection:
                    server = "metric-registry-planner"
                    actual_tool_name = ""
                    actual_arguments = {}
                    normalized_result = {
                        "status": "unsupported",
                        "parsed": plan_model.model_dump(mode="json"),
                        "reason": rejection,
                    }
                else:
                    server = "mock-mcp"
                    actual_tool_name = str(routed_tool)
                    actual_arguments = dict(routed_arguments or {})
                    normalized_result["parsed"] = plan_model.model_dump(mode="json")
            else:
                operation = {
                    "metric_value": "value",
                    "metric_trend": "trend",
                    "metric_compare": "compare",
                    "metric_breakdown": "breakdown",
                    "metric_statistics": "statistics",
                }.get(plan_model.query_type, "value")
                server = "metric-registry"
                actual_tool_name = f"registry_{operation}"
                actual_arguments = _build_arguments_from_plan(plan_model)
                normalized_result["parsed"] = plan_model.model_dump(mode="json")

            # The real registry wrapper rejects a named metric plan without a
            # server-created authorization object. Preserve that limitation in
            # offline execution; never manufacture an AuthorizedMetric.
            if (
                plan_model.metric_name and plan_model.authorized_metric is None
                and normalized_result.get("status") != "unsupported"
            ):
                normalized_result = {
                    "status": "unsupported",
                    "parsed": plan_model.model_dump(mode="json"),
                    "reason": "QueryPlan 缺少服务端 AuthorizedMetric，已阻止执行。",
                }

        return {
            "task_type": task_type,
            "question": question,
            "server": server,
            "tool_name": actual_tool_name,
            "arguments": actual_arguments,
            "request_id": actual_request_id,
            "idempotency_key": actual_idempotency_key,
            "data": [],
            "result": normalized_result,
        }
    # Trend planner questions contain a full start/end range followed by the
    # original user request. Use the latest scenario date (the trend endpoint)
    # so an earlier date that happens to be another fixture case cannot hijack
    # the current case's truth table. Point comparisons keep current-date first.
    scenario_dates = _ISO_DATE.findall(question)
    if planned_type == "metric_trend":
        scenario = next((day for day in reversed(scenario_dates) if day in SCENARIO_DATES), None)
    else:
        scenario = _scenario_date(question)
    metric = _metric_from_question(question) or "gmv"

    # The graph passes the deterministic registry plan to the gateway.  The
    # offline fixture must honor that plan so it tests the same evidence shape
    # as the real dispatch path instead of defaulting to query_gmv.
    if planned_type == "metric_compare" and len(_ISO_DATE.findall(question)) >= 2:
        dates = _ISO_DATE.findall(question)
        current = plan_value("current_date") or plan_value("left_date") or dates[0]
        baseline = plan_value("baseline_date") or plan_value("right_date") or dates[1]
        pair = _case_pair_values(metric, current, baseline)
        current_value, baseline_value = pair or (
            _daily_metric_value(metric, current), _daily_metric_value(metric, baseline),
        )
        delta = round(current_value - baseline_value, 2) if current_value is not None and baseline_value is not None else None
        return response(
            "compare_metric", {},
            {"status": "ok", "parsed": {"query_type": "metric_compare"},
                       "compare_result": {"status": "ok" if delta is not None else "not_found", "metric_name": metric, "left_date": current,
                                          "right_date": baseline, "left_value": current_value,
                                          "right_value": baseline_value,
                                          "delta_value": delta,
                                          "delta_ratio": round(delta * 100 / baseline_value, 2) if delta is not None and baseline_value else None,
                                           "source_table": "olist-public.orders+order_items" if current in _HOLDOUT_DATES else "golden.fixture"}},
        )
    if planned_type == "metric_statistics":
        from app.metrics.statistics import summarize_points

        dates = _ISO_DATE.findall(question)
        end = plan_value("end_date") or next((day for day in reversed(dates) if day in SCENARIO_DATES), None) or "2025-03-31"
        start = plan_value("start_date") or end
        points = _synthetic_points(metric, start, end, case_day=end if end in _SCENARIO_TRUTH or end in _HOLDOUT_DATES else None)
        summary = summarize_points(points, requested=plan_value("statistics"))
        return response(
            "query_metric_statistics", {},
            {
                "status": "ok", "parsed": {"query_type": "metric_statistics"},
                "statistics_result": {
                    "metric_name": metric, "start_date": start, "end_date": end,
                    "granularity": plan_value("granularity") or "day",
                    "distribution_basis": "time_bucket_values",
                    "source_table": "olist-public.orders+order_items" if end in _HOLDOUT_DATES else "golden.fixture",
                    **summary,
                },
            },
        )

    if planned_type == "metric_contribution":
        dates = _ISO_DATE.findall(question)
        current = plan_value("current_date") or plan_value("metric_date") or (dates[0] if dates else scenario)
        baseline = plan_value("baseline_date") or (dates[1] if len(dates) >= 2 else (date.fromisoformat(current) - timedelta(days=1)).isoformat())
        holdout_result = _holdout_contribution(metric, current, baseline) if scenario in _HOLDOUT_DATES else None
        result = holdout_result or _scenario_dimension_result(metric, current, baseline)
        return response("compare_breakdown_by_dimension", {}, result)

    # Trend queries use the same daily fact resolver as comparisons and periods.
    if planned_type == "metric_trend" or "trend" in question.casefold() or (not scenario_dates and "gmv" in question.casefold() and "7" in question):
        dates = _ISO_DATE.findall(question)
        end = plan_value("end_date") or next((day for day in reversed(dates) if day in SCENARIO_DATES), None)
        if end is None:
            end = "2025-03-31"
        start = plan_value("start_date")
        if start is None:
            start = (date.fromisoformat(end) - timedelta(days=1)).isoformat() if end in SCENARIO_DATES or end in _HOLDOUT_DATES else end
        anchor = end if end in _SCENARIO_TRUTH or end in _HOLDOUT_DATES else None
        if plan_value("start_date") is None and not dates:
            start = (date.fromisoformat(end) - timedelta(days=6)).isoformat()
        elif plan_value("start_date") is None and end == "2025-03-31":
            start = (date.fromisoformat(end) - timedelta(days=6)).isoformat()
        points = _synthetic_points(metric, start, end, case_day=anchor)
        source = "olist-public.orders+order_items" if end in _HOLDOUT_DATES else "golden.fixture"
        trend = {
            "metric_name": metric, "start_date": start, "end_date": end,
            "granularity": plan_value("granularity") or "day", "source_table": source,
            "points": points,
        }
        return response(
            "query_metric_trend", {},
            {"status": "ok", "parsed": {"query_type": "metric_trend"},
             "trend_result": trend},
        )

    # Dimension-contribution queries on scenario dates are all refuted —
    # except the dimension-driven scenario, where one category explains it all.
    if "贡献度" in question and scenario:
        dates = _ISO_DATE.findall(question)
        current = dates[0] if dates else scenario
        baseline = dates[1] if len(dates) >= 2 else (date.fromisoformat(current) - timedelta(days=1)).isoformat()
        holdout_result = _holdout_contribution(metric, current, baseline) if scenario in _HOLDOUT_DATES else None
        result = holdout_result or _scenario_dimension_result(metric, current, baseline)
        return response("compare_breakdown_by_dimension", {}, result)

    # English-language evaluation inputs use the same governed evidence
    # shapes as the historical Chinese fixtures.
    if ("contribution" in question.casefold() or "breakdown" in question.casefold()) and scenario:
        dates = _ISO_DATE.findall(question)
        current = dates[0] if dates else scenario
        baseline = dates[1] if len(dates) >= 2 else (date.fromisoformat(current) - timedelta(days=1)).isoformat()
        holdout_result = _holdout_contribution(metric, current, baseline) if scenario in _HOLDOUT_DATES else None
        result = holdout_result or _scenario_dimension_result(metric, current, baseline)
        return response("compare_breakdown_by_dimension", {}, result)

    if "compare" in question.casefold() and len(_ISO_DATE.findall(question)) >= 2:
        dates = _ISO_DATE.findall(question)
        current, baseline = dates[0], dates[1]
        pair = _case_pair_values(metric, current, baseline)
        current_value, baseline_value = pair or (
            _daily_metric_value(metric, current), _daily_metric_value(metric, baseline),
        )
        delta = round(current_value - baseline_value, 2) if current_value is not None and baseline_value is not None else None
        return response(
            "compare_metric", {},
            {"status": "ok", "parsed": {"query_type": "metric_compare"},
                       "compare_result": {"status": "ok" if delta is not None else "not_found", "metric_name": metric, "left_date": current,
                                          "right_date": baseline, "left_value": current_value,
                                          "right_value": baseline_value,
                                          "delta_value": delta,
                                          "delta_ratio": round(delta * 100 / baseline_value, 2) if delta is not None and baseline_value else None,
                                           "source_table": "olist-public.orders+order_items" if current in _HOLDOUT_DATES else "golden.fixture"}},
        )

    # Default: legacy single-value result, scenario-aware for the future agent.
    metric_date = _ISO_DATE.findall(question)[0] if _ISO_DATE.search(question) else "2018-08-10"
    missing_value = bool(
        scenario
        and metric_date in _missing_case_tail(metric, scenario, metric_date, metric_date)
    )
    value = (
        None if missing_value
        else _daily_metric_value(metric, scenario, case_day=scenario) if scenario
        else 100.0
    )
    return response(
        "query_gmv", {"metric_date": metric_date},
        {"status": "not_found" if missing_value else "ok", "parsed": {"query_type": "metric_value"},
         "metric_result": {"metric_name": metric, "metric_date": metric_date, "value": value,
                           "source_table": "olist-public.orders+order_items" if metric_date in _HOLDOUT_DATES else "golden.fixture"}},
    )


def fixture_series(metric_name: str, period: PeriodSpec, **_: object) -> MetricSeries:
    start, end = period.start_date.isoformat(), period.end_date.isoformat()
    points = _synthetic_points(metric_name, start, end)
    metric = _metric_definition(metric_name)
    components: dict[str, float] = {}
    if metric.period_aggregation is not None:
        aggregation = metric.period_aggregation
        numerator_points = _synthetic_points(aggregation.numerator_metric, start, end)
        denominator_points = _synthetic_points(aggregation.denominator_metric, start, end)
        numerator = [point["value"] for point in numerator_points if point["value"] is not None]
        denominator = [point["value"] for point in denominator_points if point["value"] is not None]
        if len(numerator) == len(points) and len(denominator) == len(points):
            components = {"numerator": round(sum(numerator), 2), "denominator": round(sum(denominator), 2)}
        else:
            points = [{**point, "value": None} for point in points]
    source = "olist-public.orders+order_items" if any(day in _HOLDOUT_FACTS for day in (start, end)) else "golden.fixture"
    freshness_days = [date.fromisoformat(point["bucket"]) for point in points if point["value"] is not None]
    return MetricSeries(
        metric_name=metric_name, unit=metric.unit or "units", period=period,
        source_refs=[source], definition_version=metric.version,
        freshness_at=(datetime.combine(max(freshness_days), time.max, tzinfo=timezone.utc) if freshness_days else None),
        points=[MetricPoint(bucket=date.fromisoformat(point["bucket"]), value=point["value"]) for point in points],
        component_totals=components,
    )


def fixture_period_value(metric_name: str, period: PeriodSpec, **kwargs: object) -> float | None:
    return _period_metric_value(metric_name, period, case_day=kwargs.get("case_day"))


def fixture_comparison(
    metric_name: str, current: PeriodSpec, baseline: PeriodSpec, comparison_type: str, **_: object,
) -> PeriodComparison:
    case_day = current.end_date.isoformat() if current.end_date.isoformat() in _SCENARIO_TRUTH or current.end_date.isoformat() in _HOLDOUT_DATES else None
    current_value = fixture_period_value(metric_name, current, case_day=case_day)
    baseline_value = fixture_period_value(metric_name, baseline, case_day=case_day)
    delta = round(current_value - baseline_value, 2) if current_value is not None and baseline_value is not None else None
    return PeriodComparison.model_validate({
        "comparison_type": comparison_type, "current_period": current, "baseline_period": baseline,
        "current_value": current_value, "baseline_value": baseline_value, "delta_value": delta,
        "delta_ratio": round(delta * 100 / baseline_value, 2) if delta is not None and baseline_value else None,
        "status": "not_found" if current_value is None or baseline_value is None else "zero_baseline" if baseline_value == 0 else "ok",
    })


def fixture_period_breakdown(
    metric_name: str, current_start: date, current_end: date,
    baseline_start: date, baseline_end: date, dimension: str, limit: int,
) -> dict:
    del dimension, limit
    current = PeriodSpec(start_date=current_start, end_date=current_end)
    baseline = PeriodSpec(start_date=baseline_start, end_date=baseline_end)
    definition = _metric_definition(metric_name)
    if metric_name == "gmv" and (current_end.isoformat() in _HOLDOUT_DATES or baseline_end.isoformat() in _HOLDOUT_DATES):
        def holdout_groups(period: PeriodSpec) -> dict[str, float] | None:
            groups: dict[str, float] = {}
            day = period.start_date
            while day <= period.end_date:
                fact = _HOLDOUT_FACTS.get(day.isoformat())
                if fact is None or "gmv" not in fact:
                    return None
                for label, value in (fact.get("customer_state") or {}).items():
                    groups[label] = groups.get(label, 0.0) + float(value)
                day += timedelta(days=1)
            return groups

        current_groups, baseline_groups = holdout_groups(current), holdout_groups(baseline)
        if current_groups is None or baseline_groups is None:
            return {"status": "incomplete", "reason": "olist_observation_missing", "source_table": "olist-public.order_items+orders+customers"}
        labels = sorted(set(current_groups) | set(baseline_groups))
        current_items = [{"dimension_value": label, "value": round(current_groups.get(label, 0.0), 2)} for label in labels]
        baseline_items = [{"dimension_value": label, "value": round(baseline_groups.get(label, 0.0), 2)} for label in labels]
        current_total = round(sum(item["value"] for item in current_items), 2)
        baseline_total = round(sum(item["value"] for item in baseline_items), 2)
        return {
            "status": "ok", "aggregation": "sum", "current_items": current_items,
            "baseline_items": baseline_items, "current_total": current_total,
            "baseline_total": baseline_total, "total_delta": round(current_total - baseline_total, 2),
            "source_table": "olist-public.order_items+orders+customers",
        }
    case_day = current.end_date.isoformat() if current.end_date.isoformat() in _SCENARIO_TRUTH or current.end_date.isoformat() in _HOLDOUT_DATES else None
    current_total = fixture_period_value(metric_name, current, case_day=case_day)
    baseline_total = fixture_period_value(metric_name, baseline, case_day=case_day)
    if current_total is None or baseline_total is None:
        return {"status": "incomplete", "reason": "required_period_observation_missing", "source_table": "golden.fixture"}
    delta = round(current_total - baseline_total, 2)
    if definition.aggregation != "sum":
        # Ratio/average values are not additive. Preserve registered period
        # totals and expose the aggregation class; do not invent a sum-to-total
        # dimension reconciliation for them.
        return {
            "status": "ok", "aggregation": definition.aggregation,
            "current_items": [], "baseline_items": [],
            "current_total": current_total, "baseline_total": baseline_total,
            "total_delta": delta,
            "period_aggregation": definition.period_aggregation.__dict__ if definition.period_aggregation else None,
            "source_table": "olist-public.orders+order_items" if current_end.isoformat() in _HOLDOUT_DATES else "golden.fixture",
        }
    scenario = current_end.isoformat()
    share = _CONCENTRATION_SHARES.get(scenario)
    if share is not None:
        result = _concentrated_contribution(
            metric_name, scenario, baseline_end.isoformat(), share,
            current_total=current_total, baseline_total=baseline_total,
        )["contribution_result"]
        return {
            **result, "aggregation": definition.aggregation,
            "current_total": current_total, "baseline_total": baseline_total,
        }
    current_items = _refuted_contribution(
        metric_name, scenario, baseline_end.isoformat(),
        current_total=current_total, baseline_total=baseline_total,
    )["contribution_result"]["current_items"]
    baseline_items = _refuted_contribution(
        metric_name, scenario, baseline_end.isoformat(),
        current_total=current_total, baseline_total=baseline_total,
    )["contribution_result"]["baseline_items"]
    return {
        "status": "ok", "aggregation": definition.aggregation,
        "current_items": current_items, "baseline_items": baseline_items,
        "current_total": current_total, "baseline_total": baseline_total,
        "total_delta": delta,
        "source_table": "olist-public.orders+order_items" if current_end.isoformat() in _HOLDOUT_DATES else "golden.fixture",
    }


@contextmanager
def offline_dependencies():
    """Replace RAG/MCP gateways only; graph orchestration stays real."""
    with TemporaryDirectory() as temporary:
        settings = get_settings()
        with (
            # The Golden suite must never make a provider request.  Patch both
            # module-level imports and the client functions imported lazily by
            # the planner/answer nodes.
            patch("app.graph.nodes.triage.llm_classify", lambda *args, **kwargs: ""),
            patch("app.agents.critic_agent.llm_extract", lambda *args, **kwargs: None),
            patch.multiple(
                "app.llm.client",
                llm_extract=lambda *args, **kwargs: None,
                llm_agent_action=lambda *args, **kwargs: None,
                llm_chat=lambda *args, **kwargs: "",
            ),
            patch("app.graph.nodes.metric_definition.retrieve_knowledge", fixture_rag),
            patch.multiple(
                "app.graph.nodes.data_query",
                query_business_data=fixture_mcp,
                query_scoped_business_data=fixture_mcp,
                data_time_bounds=lambda: (date(2016, 9, 15), date(2018, 8, 29)),
            ),
            patch("app.graph.nodes.diagnostic.query_business_data", fixture_mcp),
            # branch_evidence imports the gateway at module level; without this
            # patch its scenario follow-ups hit the real (offline-unavailable)
            # MCP client and every contribution silently becomes INCONCLUSIVE.
            patch("app.graph.nodes.branch_evidence.query_business_data", fixture_mcp),
            patch("app.graph.nodes.branch_evidence.query_scoped_business_data", fixture_mcp),
            # Date resolution moved out of the graph node into the deterministic
            # query planner/root-cause helpers.  Patch the active import sites,
            # rather than retaining a mock for a symbol the node no longer has.
            patch("app.agents.data_query_agent.get_latest_metric_date", lambda: "2025-03-31"),
            patch("app.analysis.reports.get_latest_metric_date", lambda: "2025-03-31"),
            patch.multiple(
                "app.analysis.periodic_reports",
                query_metric_series=fixture_series,
                aggregate_metric_period=fixture_period_value,
                compare_periods=fixture_comparison,
                compare_breakdown_by_dimension_period=fixture_period_breakdown,
            ),
            # The decorator's globals live in report_snapshot.py and open an
            # MVCC transaction before the patched query functions run.
            patch("app.data.report_snapshot.report_read_snapshot", nullcontext),
            # Report artifact creation captures immutable attribution snapshots
            # after assembling the report. Keep that secondary read on the
            # same offline fixture boundary; otherwise Golden reaches the live
            # PostgreSQL repository and silently falls back to an unstructured
            # Markdown draft when the connection times out.
            patch("app.analysis.report_alignment.capture_report_attribution_snapshots", lambda report: report),
            patch("app.agents.alert_agent.query_metric_series", fixture_series),
            patch("app.core.time_expression.data_time_bounds", lambda: (date(2016, 9, 15), date(2018, 8, 29))),
            patch.object(settings, "phase6_store_path", str(Path(temporary) / "phase6.sqlite")),
            patch.object(settings, "report_chart_dir", str(Path(temporary) / "charts")),
        ):
            yield

"""Independently audit Y2-11..14 against the locally cached Olist v2 CSVs.

This reads raw CSVs only. It does not start a database, invoke the application
query layer, or call a model/provider. Output is a source-fact audit, not a
reviewer verdict or replacement for the preserved historical traces.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "kagglehub/datasets/olistbr/brazilian-ecommerce/versions/2"
PLAN = ROOT / "docs/testing/claim_verification_pilot_20261027/y1gb_y2gb_20261028/Y2GB_CASE_PLAN.jsonl"
CENT = Decimal("0.01")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def round2(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def hours(delta) -> Decimal:
    # Exact conversion from timedelta microseconds; avoid binary-float drift.
    return Decimal(delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds) / Decimal(3_600_000_000)


def days(delta) -> Decimal:
    return Decimal(delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds) / Decimal(86_400_000_000)


def load_orders() -> dict[str, dict[str, Any]]:
    path = DATA / "olist_orders_dataset.csv"
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            row["purchase_dt"] = parse_dt(row.get("order_purchase_timestamp"))
            row["carrier_dt"] = parse_dt(row.get("order_delivered_carrier_date"))
            row["delivered_dt"] = parse_dt(row.get("order_delivered_customer_date"))
            rows[row["order_id"]] = row
    return rows


def audit(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite raw-stat audit: {output}")
    orders_path = DATA / "olist_orders_dataset.csv"
    items_path = DATA / "olist_order_items_dataset.csv"
    orders = load_orders()
    item_counts: Counter[str] = Counter()
    with items_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            item_counts[row["order_id"]] += 1

    transit_by_day: dict[str, list[Decimal]] = {}
    for row in orders.values():
        purchase, carrier, delivered = row["purchase_dt"], row["carrier_dt"], row["delivered_dt"]
        if purchase is None or carrier is None or delivered is None:
            continue
        # Matches migration 20260723_05: per-order transit duration is rounded
        # to 2 decimals before AVG; bucket key is purchase_date.
        value = round2(hours(delivered - carrier))
        transit_by_day.setdefault(purchase.date().isoformat(), []).append(value)

    transit_days = {}
    for day in [f"2017-01-{n:02d}" for n in range(1, 7)]:
        values = transit_by_day.get(day, [])
        transit_days[day] = {
            "eligible_order_count": len(values),
            "sum_of_per_order_hours_rounded_to_2dp": str(sum(values, Decimal(0))) if values else None,
            "average_hours_unrounded": str(sum(values, Decimal(0)) / len(values)) if values else None,
            "query_value_hours_rounded_to_2dp": str(round2(sum(values, Decimal(0)) / len(values))) if values else None,
            "status_semantics": "ok" if values else "not_found/null",
        }
    jan5 = Decimal(transit_days["2017-01-05"]["query_value_hours_rounded_to_2dp"])
    jan6 = Decimal(transit_days["2017-01-06"]["query_value_hours_rounded_to_2dp"])
    pooled = sum((Decimal(transit_days[d]["sum_of_per_order_hours_rounded_to_2dp"]) for d in ("2017-01-05", "2017-01-06")), Decimal(0)) / Decimal(
        sum(transit_days[d]["eligible_order_count"] for d in ("2017-01-05", "2017-01-06"))
    )
    simple_daily = (jan5 + jan6) / Decimal(2)

    lead_days = {}
    for day in ("2017-08-20", "2018-10-18"):
        rows = [r for r in orders.values() if r["purchase_dt"] is not None and r["purchase_dt"].date().isoformat() == day]
        delivered = [r for r in rows if r.get("order_status") == "delivered" and r["delivered_dt"] is not None]
        values = [days(r["delivered_dt"] - r["purchase_dt"]) for r in delivered]
        lead_days[day] = {
            "all_purchase_order_count": len(rows),
            "order_status_counts": dict(sorted(Counter(r.get("order_status") for r in rows).items())),
            "eligible_delivered_order_count": len(values),
            "sum_lead_days": str(sum(values, Decimal(0))) if values else None,
            "average_lead_days_unrounded": str(sum(values, Decimal(0)) / len(values)) if values else None,
            "query_value_days_rounded_to_2dp": str(round2(sum(values, Decimal(0)) / len(values))) if values else None,
            "status_semantics": "ok" if values else "not_found/null",
        }

    item_day = "2017-01-06"
    eligible_item_orders = [r for r in orders.values() if r["purchase_dt"] is not None
                            and r["purchase_dt"].date().isoformat() == item_day
                            and r.get("order_status") == "delivered"]
    distribution = Counter(item_counts.get(r["order_id"], 0) for r in eligible_item_orders)
    item_sum = sum(item_counts.get(r["order_id"], 0) for r in eligible_item_orders)

    result = {
        "audit_version": "y2gb-raw-source-statistics-r1",
        "dataset": {
            "local_cache_path": "kagglehub/datasets/olistbr/brazilian-ecommerce/versions/2",
            "dataset_id_as_cached": "olistbr/brazilian-ecommerce",
            "cached_version": "2",
            "upstream_metadata_available_in_cache": False,
            "source_files": {
                orders_path.name: {"sha256": sha(orders_path), "rows": len(orders)},
                items_path.name: {"sha256": sha(items_path), "rows": sum(item_counts.values())},
            },
        },
        "method_sources": {
            "transit_metric": "migrations/versions/20260723_05_olist_payment_fulfillment.py inserts delivery_transit_hours as ROUND((delivered_customer-carrier)/3600, 2), grouped by DATE(order_purchase_timestamp); config/metrics/fulfillment.yaml registers order-grain average.",
            "lead_metric": "app/data/repositories/daily_metrics.py::_query_avg_delivery_lead_time averages delivered_customer-purchase timestamps in days for delivered rows with non-null delivery time, grouped by purchase date.",
            "items_metric": "app/data/repositories/daily_metrics.py::_query_avg_items_per_order averages per-order item_count over analytics.v_delivered_orders for purchase_date.",
            "missing_bucket": "app/data/repositories/daily_metrics.py::query_metric_trend emits each day; missing raw metric value is represented as None, not numeric zero.",
        },
        "Y2-11": {
            "date_buckets": {day: transit_days[day] for day in ("2017-01-05", "2017-01-06")},
            "pooled_two_day_average_hours_unrounded": str(pooled),
            "pooled_two_day_query_value_hours_rounded_to_2dp": str(round2(pooled)),
            "simple_average_of_rounded_daily_means_hours": str(simple_daily),
            "absolute_difference_pooled_minus_simple_hours": str(pooled - simple_daily),
            "conclusion": "raw facts verify unequal eligible populations (32 vs 4) and different pooled (125.91 h) vs unweighted daily mean (118.35 h).",
        },
        "Y2-12": {
            "jan1_to_jan6_transit_buckets": transit_days,
            "missing_days": [day for day, facts in transit_days.items() if facts["eligible_order_count"] == 0],
            "conclusion": "the original FBG057 Jan5-Jan6 interval has no missing bucket; Jan1-Jan4 are genuinely empty buckets in the same raw source. This supports the missing-vs-zero mechanism but must be scope-labeled as supplementary facts outside the original two-day interval.",
        },
        "Y2-13": {
            "dates": lead_days,
            "conclusion": "raw source confirms baseline has 103 eligible delivered orders and rounds to 10.01 days; current date has no eligible delivered orders and must remain not_found/null, so a two-day mean cannot be calculated.",
        },
        "Y2-14": {
            "metric_date": item_day,
            "eligible_delivered_order_count": len(eligible_item_orders),
            "total_order_items": item_sum,
            "mean_items_per_order": str(Decimal(item_sum) / Decimal(len(eligible_item_orders))) if eligible_item_orders else None,
            "query_mean_rounded_to_2dp": str(round2(Decimal(item_sum) / Decimal(len(eligible_item_orders)))) if eligible_item_orders else None,
            "per_order_item_count_distribution": {str(k): v for k, v in sorted(distribution.items())},
            "orders_not_exactly_one_item": sum(v for k, v in distribution.items() if k != 1),
            "conclusion": "the original appended assertion that every Jan6 delivered order had one item is empirically true here (4/4); it is invalid as a seeded false factual claim. Revise the challenge assertion or leave the original candidate blocked.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.output)
    print(json.dumps({"output": str(args.output), "provider_calls": 0, "database_calls": 0,
                      "Y2-11": result["Y2-11"]["conclusion"],
                      "Y2-12": result["Y2-12"]["conclusion"],
                      "Y2-13": result["Y2-13"]["conclusion"],
                      "Y2-14": result["Y2-14"]["conclusion"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

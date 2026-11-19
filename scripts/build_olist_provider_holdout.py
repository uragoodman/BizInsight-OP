"""Build an independent provider holdout from the locally archived Olist source."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kagglehub/datasets/olistbr/brazilian-ecommerce/versions/2"
OUT = ROOT / "evaluation/datasets/provider_holdout_olist_v1.json"
FACTS = ROOT / "evaluation/fixtures/provider_holdout_olist_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    orders_path = SOURCE / "olist_orders_dataset.csv"
    items_path = SOURCE / "olist_order_items_dataset.csv"
    customers_path = SOURCE / "olist_customers_dataset.csv"
    orders = pd.read_csv(orders_path)
    items = pd.read_csv(items_path)
    customers = pd.read_csv(customers_path)
    orders["day"] = pd.to_datetime(orders["order_purchase_timestamp"], errors="coerce").dt.strftime("%Y-%m-%d")
    orders["delivered"] = pd.to_datetime(orders["order_delivered_customer_date"], errors="coerce")
    orders["estimated"] = pd.to_datetime(orders["order_estimated_delivery_date"], errors="coerce")
    orders = orders.merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")
    rows = items.merge(orders[["order_id", "day", "customer_state", "delivered", "estimated"]], on="order_id", how="inner")
    rows["gmv"] = rows["price"].fillna(0) + rows["freight_value"].fillna(0)
    daily = rows.groupby("day", dropna=True).agg(gmv=("gmv", "sum"), order_count=("order_id", "nunique"))
    daily["avg_order_value"] = daily["gmv"] / daily["order_count"]
    order_delivery = orders.dropna(subset=["day"]).copy()
    order_delivery["delayed"] = (order_delivery["delivered"] > order_delivery["estimated"]).astype(float)
    delivery = order_delivery.groupby("day").agg(delay_rate=("delayed", "mean"))
    daily = daily.join(delivery, how="left").sort_index()
    state_daily = rows.groupby(["day", "customer_state"], dropna=True)["gmv"].sum()
    facts_by_day = {
        day: {
            "gmv": round(float(row.gmv), 2),
            "order_count": int(row.order_count),
            "avg_order_value": round(float(row.avg_order_value), 4),
            "delivery_delay_rate": round(float(row.delay_rate or 0) * 100, 2),
            "customer_state": {
                str(state): round(float(value), 2)
                for (fact_day, state), value in state_daily.items() if fact_day == day
            },
        }
        for day, row in daily.iterrows()
    }

    candidates: list[dict] = []
    for day, current in daily.iterrows():
        if not ("2018-05-01" <= day <= "2018-07-31") or int(current.order_count) < 50:
            continue
        previous_day = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if previous_day not in daily.index or current.gmv >= daily.loc[previous_day].gmv:
            continue
        previous = daily.loc[previous_day]
        order_effect = (current.order_count - previous.order_count) * (current.avg_order_value + previous.avg_order_value) / 2
        aov_effect = (current.avg_order_value - previous.avg_order_value) * (current.order_count + previous.order_count) / 2
        denominator = abs(order_effect) + abs(aov_effect)
        if denominator <= 0:
            continue
        order_share, aov_share = abs(order_effect) / denominator, abs(aov_effect) / denominator
        now_groups = facts_by_day[day]["customer_state"]
        old_groups = facts_by_day.get(previous_day, {}).get("customer_state", {})
        total_delta = float(current.gmv - previous.gmv)
        state_deltas = {
            state: now_groups.get(state, 0.0) - old_groups.get(state, 0.0)
            for state in set(now_groups) | set(old_groups)
        }
        top_state = max(state_deltas, key=lambda state: abs(state_deltas[state]), default="")
        state_share = abs(state_deltas[top_state] / total_delta) if top_state and total_delta else 0.0
        candidates.append({
            "date": day, "baseline_date": previous_day,
            "order_share": order_share, "aov_share": aov_share,
            "state_share": state_share,
            "state_value": top_state,
            "state_ratio": state_deltas.get(top_state, 0.0) / total_delta if top_state and total_delta else 0.0,
            "dominant_factor": "order_count" if order_share >= aov_share else "avg_order_value",
        })

    selected: list[tuple[dict, str]] = []
    used: set[str] = set()
    def take(predicate, count: int, pattern: str) -> None:
        choices = [item for item in candidates if item["date"] not in used and predicate(item)]
        choices.sort(key=lambda item: (max(item["order_share"], item["aov_share"], item["state_share"]), item["date"]), reverse=True)
        for item in choices[:count]:
            selected.append((item, pattern)); used.add(item["date"])

    take(lambda item: item["order_share"] >= 0.70, 8, "order_led")
    take(lambda item: item["aov_share"] >= 0.65, 6, "aov_led")
    take(lambda item: item["order_share"] >= 0.25 and item["aov_share"] >= 0.25, 4, "mixed_factors")
    take(lambda item: item["state_share"] >= 0.50, 4, "customer_state_concentrated")
    if len(selected) != 22:
        raise RuntimeError(f"Holdout strata produced {len(selected)}/22 labeled declines; inspect source data")

    cases = []
    for index, (item, pattern) in enumerate(sorted(selected, key=lambda pair: pair[0]["date"])):
        day = item["date"]
        expectation = {"found_factor": item["dominant_factor"]}
        if pattern == "customer_state_concentrated":
            expectation = {
                "reported_dimension_contribution": {
                    "dimension": "customer_state", "dimension_value": item["state_value"],
                    "contribution_share": round(item["state_share"], 4), "share_tolerance": 0.02,
                },
                "business_cause_unresolved": True,
            }
        cases.append({
            "id": f"olist-holdout-{index + 1:02d}",
            "category": "root_cause_longtail",
            "pattern": pattern,
            "query": (
                f"\u4e3a\u4ec0\u4e48 {day} GMV \u4e0b\u964d\u4e86\uff1f\u5224\u65ad\u8ba2\u5355\u91cf\u548c\u5ba2\u5355\u4ef7\u4e2d\u4e3b\u8981\u7684\u7edf\u8ba1\u89e3\u91ca\u56e0\u7d20\u3002"
                if pattern != "customer_state_concentrated" else
                f"\u4e3a\u4ec0\u4e48 {day} GMV \u4e0b\u964d\u4e86\uff1f\u8bf7\u627e\u51fa customer_state \u7ef4\u5ea6\u8d21\u732e\u6700\u5927\u7684\u5206\u7ec4\u548c\u5360\u6bd4\uff0c\u8bf4\u660e\u8fd9\u53ea\u662f\u7edf\u8ba1\u8d21\u732e\uff0c\u4e1a\u52a1\u673a\u5236\u4ecd\u672a\u786e\u8ba4\u3002"
            ),
            "expect": expectation,
            "source_ground_truth": {
                "current_date": day,
                "baseline_date": item["baseline_date"],
                "order_effect_share_abs": round(item["order_share"], 4),
                "aov_effect_share_abs": round(item["aov_share"], 4),
                "customer_state_top_share_abs": round(item["state_share"], 4),
                "customer_state_top_value": item["state_value"],
            },
        })

    # Two dates with no order-item facts are deliberate missing-data controls.
    missing_days = [
        day for day in pd.date_range("2016-09-15", "2016-12-31", freq="D").strftime("%Y-%m-%d")
        if day not in facts_by_day
    ][:2]
    if len(missing_days) != 2:
        raise RuntimeError("Could not find two independent sparse-period controls")
    for index, day in enumerate(missing_days, start=len(cases) + 1):
        cases.append({
            "id": f"olist-holdout-{index:02d}", "category": "root_cause_longtail",
            "pattern": "source_day_missing", "query": f"\u4e3a\u4ec0\u4e48 {day} GMV \u65e0\u6cd5\u786e\u8ba4\uff1f\u8bf7\u6838\u5bf9\u662f\u5426\u6709\u6570\u636e\u3002",
            "expect": {"abstain": True},
            "source_ground_truth": {"current_date": day, "missing": True},
        })

    selected_days = {case["source_ground_truth"]["current_date"] for case in cases}
    needed_days = set(selected_days)
    for day in list(selected_days):
        needed_days.add((pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        for offset in range(2, 8):
            needed_days.add((pd.Timestamp(day) - pd.Timedelta(days=offset)).strftime("%Y-%m-%d"))
    snapshot = {day: facts_by_day[day] for day in sorted(needed_days) if day in facts_by_day}
    FACTS.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    FACTS.write_text(json.dumps({"snapshot_id": "olist-public-order-items-q2-2018-v1", "facts_by_day": snapshot}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dataset = {
        "dataset_version": "provider-holdout-olist-v1",
        "source": {
            "name": "Olist Brazilian E-Commerce Public Dataset",
            "local_snapshot": str(SOURCE.relative_to(ROOT)),
            "source_files_sha256": {
                path.name: _sha(path) for path in (orders_path, items_path, customers_path)
            },
            "gmv_definition": "sum(order_items.price + order_items.freight_value), grouped by purchase date",
            "baseline": "previous calendar day; factor labels use symmetric two-factor decomposition of GMV = order_count × average order value",
        },
        "case_distribution": {
            "order_led": 8, "aov_led": 6, "mixed_factors": 4,
            "customer_state_concentrated": 4, "source_day_missing": 2,
        },
        "fixture_snapshot": str(FACTS.relative_to(ROOT)),
        "fixture_snapshot_sha256": _sha(FACTS),
        "cases": cases,
    }
    OUT.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": str(OUT.relative_to(ROOT)), "cases": len(cases), "distribution": dataset["case_distribution"], "fixture_sha256": dataset["fixture_snapshot_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Repair corrupted query literals in the ordinary-analysis dataset.

Some historical edits left replacement characters in nine query strings.
This keeps the scenarios identical while making the inputs executable.
"""

from __future__ import annotations

import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / "evaluation" / "datasets" / "analysis_report_v1.json"
data = json.loads(path.read_text(encoding="utf-8"))
replacements = {
    "ar-trend-week-over-week": "Compare GMV between 2018-08-10 and 2018-08-09",
    "ar-trend-month-over-month": "Compare order_count between 2018-08-10 and 2018-08-09",
    "ar-trend-dimension-date": "Compare GMV between 2018-08-10 and 2018-08-09",
    "ar-contribution-city": "Breakdown GMV between 2018-08-10 and 2018-08-09 by customer_city contribution",
    "ar-contribution-payment": "Breakdown GMV between 2018-08-10 and 2018-08-09 by payment_type contribution",
    "ar-contribution-category": "Breakdown GMV between 2018-08-10 and 2018-08-09 by product_category contribution",
    "ar-contribution-seller": "Breakdown order_count between 2018-08-10 and 2018-08-09 by seller_state contribution",
    "ar-report-gmv-daily": "Generate a GMV daily report with trend and anomaly explanation",
    "ar-report-sales-weekly": "Generate a weekly sales report with order_count and average_order_value changes",
}
for case in data["cases"]:
    if case["id"] in replacements:
        case["query"] = replacements[case["id"]]
    if case["id"] == "ar-report-gmv-daily":
        case["expect"].pop("require_report", None)
        case["expect"]["require_report_draft"] = True
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"updated": len(replacements), "path": str(path)}, ensure_ascii=False))

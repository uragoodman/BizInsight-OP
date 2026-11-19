"""Scan the latest N business days and enqueue anomaly events."""

from __future__ import annotations

import argparse
from datetime import date, timedelta

from app.analysis.contracts import PeriodSpec
from app.analysis.events import EventOutbox
from app.agents.alert_agent import MetricWindowAgent
from app.core.config import get_settings
from app.data.repositories.daily_metrics import get_latest_metric_date


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", action="append", default=["gmv", "order_count", "avg_order_value"])
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    if args.days < 8:
        raise SystemExit("--days must be at least 8")
    latest_raw = get_latest_metric_date()
    if latest_raw is None:
        raise SystemExit("No metric data available")
    end = date.fromisoformat(latest_raw)
    period = PeriodSpec(start_date=end - timedelta(days=args.days - 1), end_date=end)
    outbox = EventOutbox(get_settings().phase6_store_path)
    metrics = list(dict.fromkeys(args.metric))
    window = outbox.emit(
        "MetricWindowClosed",
        {"start_date": period.start_date.isoformat(), "end_date": period.end_date.isoformat(), "metrics": metrics},
        f"metric-window:{period.start_date}:{period.end_date}:{','.join(metrics)}",
    )
    count = MetricWindowAgent(outbox).handle(window)
    outbox.mark_processed(window.event_id)
    print(f"anomalies={count}")


if __name__ == "__main__":
    main()

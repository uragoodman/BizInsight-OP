"""Bounded/continuous permission-event outbox worker."""

from __future__ import annotations

import argparse
import time

from app.core.config import get_settings
from app.iam.permission_events import process_permission_events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=None)
    args = parser.parse_args()
    interval = max(1, args.interval_seconds or get_settings().permission_event_worker_interval_seconds)
    while True:
        print(process_permission_events(), flush=True)
        if args.once:
            return
        time.sleep(interval)


if __name__ == "__main__":
    main()

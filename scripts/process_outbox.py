"""Deliver pending Phase 6 events through the local adapter."""

from app.analysis.events import EventOutbox, LocalNotificationAdapter, process_pending_events
from app.core.config import get_settings


def main() -> None:
    outbox = EventOutbox(get_settings().phase6_store_path)
    adapter = LocalNotificationAdapter()
    processed = process_pending_events(outbox, adapter)
    print(f"processed={processed}")


if __name__ == "__main__":
    main()

"""Phase 7 tenant outbox worker entry point.

Deployments provide an adapter that resolves tenant secret references in their
secret manager.  This executable deliberately uses the recording adapter only
for safe local smoke checks; it never sends tenant data to an external service.
"""
from app.tenancy.notifications import RecordingNotificationAdapter, process_tenant_outbox


def main() -> None:
    adapter = RecordingNotificationAdapter(deliveries=[])
    processed = process_tenant_outbox(adapter)
    print({"processed": processed, "deliveries": len(adapter.deliveries), "adapter": "recording"})


if __name__ == "__main__":
    main()

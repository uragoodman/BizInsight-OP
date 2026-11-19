"""PostgreSQL integration proof for the tenant outbox worker."""
import os

import pytest
from sqlalchemy import text

from app.data.database import engine
from app.tenancy.notifications import RecordingNotificationAdapter, process_tenant_outbox


pytestmark = pytest.mark.skipif(not os.getenv("POSTGRES_URL"), reason="requires PostgreSQL")


@pytest.fixture(autouse=True)
def isolate_outbox_rows():
    """Keep the worker's global pending-row scan inside this disposable DB.

    The worker intentionally scans the whole queue, so stale pending events in
    a copied database can otherwise be delivered by these tests. The shard
    runner guarantees this engine targets a unique codex_test_* database.
    """
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM governance.tenant_outbox WHERE status IN ('pending','processing')"))
        connection.execute(text("""DELETE FROM governance.tenant_delivery_audit
            WHERE tenant_id IN ('seller-tenant-1','seller-tenant-no-grant')"""))
        connection.execute(text("""DELETE FROM governance.tenant_outbox
            WHERE tenant_id IN ('seller-tenant-1','seller-tenant-no-grant')"""))
    yield
    with engine.begin() as connection:
        connection.execute(text("""DELETE FROM governance.tenant_delivery_audit
            WHERE tenant_id IN ('seller-tenant-1','seller-tenant-no-grant')"""))
        connection.execute(text("""DELETE FROM governance.tenant_outbox
            WHERE tenant_id IN ('seller-tenant-1','seller-tenant-no-grant')"""))
        connection.execute(text("""DELETE FROM iam.tenant_notification_channels
            WHERE tenant_id IN ('seller-tenant-1','seller-tenant-no-grant')"""))
        connection.execute(text("""DELETE FROM iam.tenant_seller_accounts
            WHERE tenant_id IN ('seller-tenant-1','seller-tenant-no-grant')"""))
        connection.execute(text("""DELETE FROM iam.tenants
            WHERE tenant_id IN ('seller-tenant-1','seller-tenant-no-grant')"""))


def test_tenant_outbox_is_delivered_and_audited_once():
    with engine.begin() as conn:
        conn.execute(text("""INSERT INTO iam.tenants(tenant_id,tenant_name,scope_type,status)
            VALUES ('seller-tenant-1','phase7 test seller','seller','active')
            ON CONFLICT(tenant_id) DO UPDATE SET status='active'"""))
        conn.execute(text("""INSERT INTO iam.tenant_seller_accounts(tenant_id,seller_id)
            VALUES ('seller-tenant-1','phase7-outbox-seller')
            ON CONFLICT(tenant_id) DO UPDATE SET seller_id=EXCLUDED.seller_id"""))
        conn.execute(text("""INSERT INTO iam.tenant_notification_channels(tenant_id,channel,secret_reference)
            VALUES ('seller-tenant-1','email','secret://phase7-test')
            ON CONFLICT (tenant_id,channel) DO UPDATE SET enabled=true"""))
        conn.execute(text("""INSERT INTO governance.tenant_outbox(tenant_id,event_id,idempotency_key,event_type,payload_json)
            VALUES ('seller-tenant-1','phase7-outbox-event','phase7-outbox-key','ReportPublished',CAST(:payload AS jsonb))
            ON CONFLICT (tenant_id,idempotency_key) DO UPDATE SET status='pending',attempts=0,last_error=''"""),
            {"payload": '{"report_id":"phase7"}'})
    adapter = RecordingNotificationAdapter(deliveries=[])
    assert process_tenant_outbox(adapter) == 1
    assert len(adapter.deliveries) == 1
    with engine.connect() as conn:
        status = conn.execute(text("""SELECT status FROM governance.tenant_outbox
            WHERE tenant_id='seller-tenant-1' AND event_id='phase7-outbox-event'""")).scalar_one()
        audit = conn.execute(text("""SELECT status FROM governance.tenant_delivery_audit
            WHERE tenant_id='seller-tenant-1' AND event_id='phase7-outbox-event' AND channel='email'""")).scalar_one()
    assert (status, audit) == ("processed", "delivered")
    assert process_tenant_outbox(adapter) == 0


def test_seller_outbox_without_seller_grant_is_dead_lettered_not_stuck_processing():
    with engine.begin() as conn:
        conn.execute(text("""INSERT INTO iam.tenants(tenant_id,tenant_name,scope_type,status)
            VALUES ('seller-tenant-no-grant','invalid seller test','seller','active')
            ON CONFLICT(tenant_id) DO UPDATE SET status='active'"""))
        conn.execute(text("""INSERT INTO governance.tenant_outbox
            (tenant_id,event_id,idempotency_key,event_type,payload_json,status,attempts)
            VALUES ('seller-tenant-no-grant','phase7-orphan-event','phase7-orphan-key',
                    'ReportPublished',CAST('{"report_id":"phase7-orphan"}' AS jsonb),'pending',0)
            ON CONFLICT (tenant_id,idempotency_key) DO UPDATE
                SET status='pending',attempts=0,last_error=''"""))

    adapter = RecordingNotificationAdapter(deliveries=[])
    assert process_tenant_outbox(adapter, max_attempts=1) == 0
    assert adapter.deliveries == []
    with engine.connect() as conn:
        status, last_error = conn.execute(text("""SELECT status,last_error FROM governance.tenant_outbox
            WHERE tenant_id='seller-tenant-no-grant' AND event_id='phase7-orphan-event'""")).one()
    assert status == "dead_letter"
    assert last_error == "ValueError"

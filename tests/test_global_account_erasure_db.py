"""Opt-in cross-tenant erasure integration against a disposable migrated DB."""

from __future__ import annotations

import os
import uuid
from datetime import date

import pytest
from sqlalchemy import create_engine, text

from app.iam import service as iam_service
from app.memory import preference_worker
from app.tenancy import TenantContext


@pytest.mark.skipif(
    not os.getenv("GLOBAL_ACCOUNT_ERASURE_TEST_URL"),
    reason="Set GLOBAL_ACCOUNT_ERASURE_TEST_URL to a disposable DB migrated to head",
)
def test_global_account_erasure_revokes_then_erases_across_tenants(monkeypatch):
    engine = create_engine(os.environ["GLOBAL_ACCOUNT_ERASURE_TEST_URL"], pool_pre_ping=True)
    suffix = uuid.uuid4().hex[:12]
    user_id = f"erase-user-{suffix}"
    tenants = [f"erase-tenant-a-{suffix}", f"erase-tenant-b-{suffix}"]
    job_id = None
    anonymous_id = None
    object_keys = [f"tests/{suffix}/source.csv", f"tests/{suffix}/artifact.json"]

    class Store:
        def __init__(self):
            self.deleted = []

        def delete(self, key):
            self.deleted.append(key)

    store = Store()
    monkeypatch.setattr("app.code_analysis.storage.get_artifact_store", lambda: store)
    try:
        with engine.begin() as connection:
            for tenant in tenants:
                connection.execute(text("""
                    INSERT INTO iam.tenants(tenant_id,tenant_name,scope_type,status)
                    VALUES (:tenant,'erasure test','platform','active')
                """), {"tenant": tenant})
            connection.execute(text("""
                INSERT INTO iam.users(user_id,display_name,email,password_hash,status)
                VALUES (:user,'Erasure Test',:email,'scrypt$fake','active')
            """), {"user": user_id, "email": f"{user_id}@example.invalid"})
            for tenant in tenants:
                connection.execute(text("""
                    INSERT INTO iam.tenant_memberships(tenant_id,user_id,role)
                    VALUES (:tenant,:user,'admin')
                """), {"tenant": tenant, "user": user_id})
                connection.execute(text("""
                    INSERT INTO iam.auth_sessions(session_id,tenant_id,user_id,expires_at)
                    VALUES (:session,:tenant,:user,now()+interval '1 hour')
                """), {"session": f"sid-{tenant}", "tenant": tenant, "user": user_id})
                connection.execute(text("""
                    INSERT INTO iam.api_credentials(credential_id,tenant_id,user_id,key_hash)
                    VALUES (:credential,:tenant,:user,:hash)
                """), {"credential": f"cred-{tenant}", "tenant": tenant, "user": user_id,
                       "hash": uuid.uuid4().hex})
                session_id = f"chat-{tenant}"
                turn_id = f"turn-{tenant}"
                connection.execute(text("""
                    INSERT INTO governance.chat_sessions(tenant_id,user_id,chat_session_id,title)
                    VALUES (:tenant,:user,:session,'private conversation')
                """), {"tenant": tenant, "user": user_id, "session": session_id})
                connection.execute(text("""
                    INSERT INTO governance.chat_turns(turn_id,tenant_id,user_id,chat_session_id,
                        question,answer,task_type,trace_id)
                    VALUES (:turn,:tenant,:user,:session,'private question','private answer','metric_query',:trace)
                """), {"turn": turn_id, "tenant": tenant, "user": user_id,
                       "session": session_id, "trace": f"trace-{tenant}"})
                connection.execute(text("""
                    INSERT INTO governance.conversation_tasks(task_id,tenant_id,user_id,chat_session_id,
                        title,objective,status)
                    VALUES (:task,:tenant,:user,:session,'private task','private objective','completed')
                """), {"task": f"task-{tenant}", "tenant": tenant, "user": user_id,
                       "session": session_id})
                connection.execute(text("""
                    INSERT INTO governance.tenant_traces(tenant_id,user_id,trace_id,trace_json)
                    VALUES (:tenant,:user,:trace,CAST(:body AS jsonb))
                """), {"tenant": tenant, "user": user_id, "trace": f"trace-{tenant}",
                       "body": '{"private":"trace body"}'})
                connection.execute(text("""
                    INSERT INTO governance.memory_sessions(tenant_id,user_id,memory_session_id,
                        session_summary,last_scope_fingerprint,last_seller_id,last_start_date,
                        last_end_date,last_trace_id,expires_at)
                    VALUES (:tenant,:user,:memory_session,'private memory',:fingerprint,'seller',
                        :start,:end,:trace,now()+interval '1 day')
                """), {"tenant": tenant, "user": user_id,
                       "memory_session": f"memory-{tenant}", "fingerprint": "0" * 64,
                       "start": date.today(), "end": date.today(), "trace": f"trace-{tenant}"})
                for index, memory_key in enumerate(("presentation_a", "presentation_b")):
                    connection.execute(text("""
                        INSERT INTO governance.memory_registry(registry_id,tenant_id,user_id,
                            seller_id,memory_type,memory_key,status,operation_id,value_hash)
                        VALUES (:registry,:tenant,:user,'seller','presentation',:memory_key,
                            'deleted',:operation,:value_hash)
                    """), {"registry": f"registry-{index}-{tenant}", "tenant": tenant,
                           "user": user_id, "memory_key": memory_key,
                           "operation": f"op-{index}-{tenant}", "value_hash": str(index) * 64})
                connection.execute(text("""
                    INSERT INTO governance.audit_events(tenant_id,user_id,event_type,outcome,
                        detail,request_path,correlation_id)
                    VALUES (:tenant,:user,'chat.completed','success','private detail',
                        '/chat/private','private-correlation')
                """), {"tenant": tenant, "user": user_id})
                connection.execute(text("""
                    INSERT INTO governance.code_analysis_files(file_id,tenant_id,user_id,
                        original_name,file_type,content_type,object_key,sha256,size_bytes,
                        contract_data,expires_at)
                    VALUES (:file,:tenant,:user,'private.csv','csv','text/csv',:key,:sha,8,
                        '{}'::jsonb,now()+interval '1 day')
                """), {"file": f"file-{tenant}", "tenant": tenant, "user": user_id,
                       "key": object_keys[tenants.index(tenant)], "sha": "a" * 64})

        monkeypatch.setattr(iam_service, "engine", engine)
        context = TenantContext(tenant_id=tenants[0], user_id=user_id, role="admin", scope_type="platform")
        requested = iam_service.request_account_erasure(context, idempotency_key=f"erase-{suffix}")
        duplicate = iam_service.request_account_erasure(context, idempotency_key=f"erase-{suffix}")
        job_id = requested["job_id"]
        assert duplicate["job_id"] == job_id
        assert requested["status"] == "pending"
        with engine.begin() as connection:
            anonymous_id = connection.execute(text("""
                SELECT anonymous_user_id FROM governance.account_erasure_jobs WHERE job_id=:job
            """), {"job": job_id}).scalar_one()
            assert connection.execute(text("SELECT status FROM iam.users WHERE user_id=:user"),
                                      {"user": user_id}).scalar_one() == "erasure_pending"
            assert connection.execute(text("SELECT count(*) FROM iam.auth_sessions WHERE user_id=:user AND revoked_at IS NOT NULL"),
                                      {"user": user_id}).scalar_one() == 2
            assert connection.execute(text("SELECT count(*) FROM iam.api_credentials WHERE user_id=:user AND revoked_at IS NOT NULL"),
                                      {"user": user_id}).scalar_one() == 2

        for _ in range(4):
            if not preference_worker.run_membership_erasure_once(engine):
                break
        assert preference_worker.run_account_erasure_once(engine)
        assert set(store.deleted) == set(object_keys)
        with engine.begin() as connection:
            account = connection.execute(text("""
                SELECT subject_user_id,status FROM governance.account_erasure_jobs WHERE job_id=:job
            """), {"job": job_id}).mappings().one()
            assert account["subject_user_id"] == anonymous_id
            assert account["status"] == "completed"
            assert connection.execute(text("SELECT count(*) FROM iam.users WHERE user_id=:anonymous AND status='erased' AND password_hash IS NULL"),
                                      {"anonymous": anonymous_id}).scalar_one() == 1
            for table in ("chat_sessions", "chat_turns", "conversation_tasks", "tenant_traces", "memory_sessions", "code_analysis_files"):
                count = connection.execute(text(f"SELECT count(*) FROM governance.{table} WHERE user_id=:user"),
                                           {"user": user_id}).scalar_one()
                assert count == 0, f"{table} retained rows for erased user"
            registry_audit = connection.execute(text("""
                SELECT memory_key,value_hash,mem0_memory_id FROM governance.memory_registry
                 WHERE user_id=:anonymous ORDER BY memory_key
            """), {"anonymous": anonymous_id}).mappings().all()
            assert [row["memory_key"] for row in registry_audit] == [
                "presentation_a", "presentation_a", "presentation_b", "presentation_b",
            ]
            assert all(row["value_hash"] is None and row["mem0_memory_id"] is None for row in registry_audit)
            audit = connection.execute(text("""
                SELECT user_id,detail,request_path,correlation_id FROM governance.audit_events
                 WHERE tenant_id=:tenant AND event_type='chat.completed'
            """), {"tenant": tenants[0]}).mappings().one()
            assert audit["user_id"] == anonymous_id
            assert audit["detail"] == "" and audit["request_path"] is None
            assert audit["correlation_id"] is None
    finally:
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM governance.audit_events WHERE tenant_id=ANY(:tenants)"), {"tenants": tenants})
            connection.execute(text("DELETE FROM governance.memory_registry WHERE tenant_id=ANY(:tenants)"), {"tenants": tenants})
            connection.execute(text("DELETE FROM governance.memory_membership_erasure_jobs WHERE tenant_id=ANY(:tenants)"), {"tenants": tenants})
            if job_id is not None:
                connection.execute(text("DELETE FROM governance.account_erasure_jobs WHERE job_id=:job"), {"job": job_id})
            connection.execute(text("DELETE FROM iam.users WHERE user_id IN (:user,:anonymous)"),
                               {"user": user_id, "anonymous": anonymous_id})
            connection.execute(text("DELETE FROM iam.tenants WHERE tenant_id=ANY(:tenants)"), {"tenants": tenants})
        engine.dispose()

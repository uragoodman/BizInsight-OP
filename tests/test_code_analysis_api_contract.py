"""Fast, database-free contract tests for the Phase 11 API boundary."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.api.auth import require_tenant_context
from app.api.routes import code_analysis as routes
from app.code_analysis import service
from app.tenancy import SELLER_SCOPE, SellerGrant, TenantContext


class _Result:
    def __init__(self, *, one=None, rows=None, scalar=None) -> None:
        self._one = one
        self._rows = rows or []
        self._scalar = scalar

    def mappings(self):
        return self

    def one_or_none(self):
        return self._one

    def all(self):
        return self._rows

    def scalar_one_or_none(self):
        return self._scalar

    def scalar_one(self):
        return 0 if self._scalar is None else self._scalar


class _Transaction:
    def __init__(self, conn) -> None:
        self.conn = conn
        self.exit_exception = None

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, _exc, _tb):
        self.exit_exception = exc_type
        return False


class _Engine:
    def __init__(self, conn) -> None:
        self.transaction = _Transaction(conn)

    def begin(self):
        return self.transaction


def _context() -> TenantContext:
    return TenantContext(
        tenant_id="phase11-contract",
        user_id="contract-user",
        role="admin",
        scope_type="platform",
    )


def test_code_analysis_read_routes_require_read_permission():
    context = TenantContext(
        tenant_id="tenant-a",
        user_id="viewer-1",
        role="viewer",
        scope_type=SELLER_SCOPE,
        seller_grants=(SellerGrant("seller-1"),),
    )
    with pytest.raises(HTTPException) as exc:
        routes.jobs(context=context)
    assert exc.value.status_code == 403


def test_upload_response_includes_file_type(monkeypatch):
    class Conn:
        def execute(self, *_args, **_kwargs):
            return _Result()

    class Store:
        key = ""

        def put(self, key: str, raw: bytes):
            self.key = key
            return key, "a" * 64, len(raw)

    store = Store()
    monkeypatch.setattr(service, "engine", _Engine(Conn()))
    monkeypatch.setattr(service, "get_artifact_store", lambda: store)

    uploaded = service.upload_file(
        _context(), "sales.csv", "text/csv", b"month,gmv\n2026-01,10\n"
    )

    assert uploaded["file_type"] == "csv"
    assert uploaded["data_contract"]["file_type"] == "csv"
    assert "phase11-contract" not in store.key
    assert "contract-user" not in store.key


def test_csv_contract_profiles_types_numeric_stats_and_masks_sensitive_samples():
    settings = SimpleNamespace(code_analysis_max_columns=20, code_analysis_max_rows=100)

    contract = service._csv_contract(
        b"amount,active,occurred_at,email\n1.5,true,2026-01-01,a@example.com\n2.5,false,2026-01-02,b@example.com\n",
        settings,
    )

    columns = {item["name"]: item for item in contract["columns"]}
    assert columns["amount"] == {
        "name": "amount",
        "type": "number",
        "null_rate": 0.0,
        "sensitive": False,
        "min": 1.5,
        "max": 2.5,
        "mean": 2.0,
    }
    assert columns["active"]["type"] == "boolean"
    assert columns["occurred_at"]["type"] == "datetime"
    assert columns["email"]["sensitive"] is True
    assert contract["samples"][0]["email"] == "[masked]"


def test_upload_quota_is_checked_before_object_write(monkeypatch):
    raw = b"x\n1\n"

    class Conn:
        def execute(self, statement, _params=None):
            if "COALESCE(sum(size_bytes),0)" in str(statement):
                return _Result(scalar=100)
            return _Result()

    class Store:
        put_called = False

        def put(self, _key: str, _raw: bytes):
            self.put_called = True
            raise AssertionError("quota must be checked before storage write")

    settings = SimpleNamespace(
        code_analysis_max_upload_bytes=100,
        code_analysis_max_columns=20,
        code_analysis_max_rows=100,
        code_analysis_retention_hours=24,
        code_analysis_max_stored_upload_bytes_per_user=102,
    )
    store = Store()
    monkeypatch.setattr(service, "engine", _Engine(Conn()))
    monkeypatch.setattr(service, "get_settings", lambda: settings)
    monkeypatch.setattr(service, "get_artifact_store", lambda: store)

    with pytest.raises(ValueError, match="quota"):
        service.upload_file(_context(), "quota.csv", "text/csv", raw)

    assert store.put_called is False


@pytest.mark.parametrize(
    ("active", "daily", "message"),
    [(3, 0, "Active analysis job limit"), (0, 20, "Daily analysis job limit")],
)
def test_create_job_enforces_atomic_active_and_daily_quotas(
    monkeypatch, active, daily, message
):
    class Conn:
        def execute(self, statement, _params=None):
            sql = str(statement)
            if "SELECT file_id,sha256,contract_data,expires_at,purged_at" in sql:
                return _Result(
                    one={
                        "file_id": "file_quota",
                        "sha256": "d" * 64,
                        "contract_data": {"file_type": "csv"},
                        "expires_at": datetime.now(timezone.utc) + timedelta(hours=1),
                        "purged_at": None,
                    }
                )
            if "status IN ('awaiting_confirmation'" in sql:
                return _Result(scalar=active)
            if "date_trunc('day'" in sql:
                return _Result(scalar=daily)
            if "INSERT INTO governance.code_analysis_jobs" in sql:
                raise AssertionError("quota failure must happen before insert")
            return _Result()

    settings = SimpleNamespace(
        code_analysis_retention_hours=24,
        code_analysis_max_active_jobs_per_user=3,
        code_analysis_max_jobs_per_user_per_day=20,
    )
    monkeypatch.setattr(service, "engine", _Engine(Conn()))
    monkeypatch.setattr(service, "get_settings", lambda: settings)
    with pytest.raises(ValueError, match=message):
        service.create_job(_context(), "file_quota", "Analyze")


def test_expired_confirmation_commits_terminal_status_before_raising(monkeypatch):
    class Conn:
        updated = False

        def execute(self, statement, _params=None):
            sql = str(statement)
            if "FROM governance.code_analysis_jobs j" in sql:
                return _Result(
                    one={
                        "status": "awaiting_confirmation",
                        "plan_hash": "a" * 64,
                        "plan_data": {"question": "expired"},
                        "resource_limits": {},
                        "expires_at": datetime.now(timezone.utc) - timedelta(seconds=1),
                        "file_expires_at": datetime.now(timezone.utc) + timedelta(hours=1),
                        "purged_at": None,
                    }
                )
            if "SET status='expired'" in sql:
                self.updated = True
            return _Result()

    conn = Conn()
    fake_engine = _Engine(conn)
    monkeypatch.setattr(service, "engine", fake_engine)

    with pytest.raises(ValueError, match="expired"):
        service.confirm_job(_context(), "caj_expired", "a" * 64)

    assert conn.updated is True
    assert fake_engine.transaction.exit_exception is None


def test_confirmation_rejects_database_plan_hash_drift(monkeypatch):
    class Conn:
        def execute(self, statement, _params=None):
            if "FROM governance.code_analysis_jobs j" in str(statement):
                return _Result(
                    one={
                        "status": "awaiting_confirmation",
                        "plan_hash": "0" * 64,
                        "plan_data": {"question": "Database content changed"},
                        "resource_limits": {},
                        "expires_at": datetime.now(timezone.utc) + timedelta(minutes=5),
                        "file_expires_at": datetime.now(timezone.utc) + timedelta(hours=1),
                        "purged_at": None,
                    }
                )
            return _Result()

    monkeypatch.setattr(service, "engine", _Engine(Conn()))

    with pytest.raises(ValueError, match="integrity"):
        service.confirm_job(_context(), "caj_drift", "0" * 64)


def test_upload_compensates_object_when_database_insert_fails(monkeypatch):
    class Conn:
        def execute(self, statement, _params=None):
            if "INSERT INTO governance.code_analysis_files" in str(statement):
                raise RuntimeError("database unavailable")
            return _Result()

    class Store:
        deleted: list[str] = []

        def put(self, key: str, raw: bytes):
            return key, "c" * 64, len(raw)

        def delete(self, key: str):
            self.deleted.append(key)

    store = Store()
    monkeypatch.setattr(service, "engine", _Engine(Conn()))
    monkeypatch.setattr(service, "get_artifact_store", lambda: store)

    with pytest.raises(RuntimeError, match="database unavailable"):
        service.upload_file(_context(), "failed.csv", "text/csv", b"x\n1\n")

    assert len(store.deleted) == 1
    assert store.deleted[0].endswith("/source.csv")


def test_job_detail_projects_actual_model_audit_fields(monkeypatch):
    class Conn:
        audit_sql = ""

        def execute(self, statement, _params=None):
            sql = str(statement)
            if "SELECT * FROM governance.code_analysis_jobs" in sql:
                return _Result(one={"job_id": "caj_model", "status": "completed"})
            if "FROM governance.code_analysis_artifacts" in sql:
                return _Result(rows=[])
            if "FROM governance.code_analysis_audits" in sql:
                self.audit_sql = sql
                return _Result(
                    one={
                        "status": "completed",
                        "provider": "opencode",
                        "generation_model": "mimo-v2.5",
                        "provider_version": "1.18.5",
                        "candidate_attempts": 2,
                        "repair_attempts": 1,
                        "duration_ms": 123,
                        "created_at": datetime.now(timezone.utc),
                    }
                )
            return _Result()

    conn = Conn()
    monkeypatch.setattr(service, "engine", _Engine(conn))

    job = service.get_job(_context(), "caj_model")

    assert job is not None
    assert job["latest_audit"]["generation_model"] == "mimo-v2.5"
    assert job["latest_audit"]["repair_attempts"] == 1
    assert "generation_model" in conn.audit_sql
    assert "provider_version" in conn.audit_sql


def test_cleanup_retains_database_reference_when_object_delete_fails_then_retries(
    monkeypatch, caplog
):
    class Conn:
        deleted = False

        def execute(self, statement, _params=None):
            sql = str(statement)
            if "SELECT file_id,object_key" in sql:
                return _Result(
                    rows=[]
                    if self.deleted
                    else [{"file_id": "file_retry", "object_key": "source-key"}]
                )
            if "SET purged_at=now()" in sql:
                self.deleted = True
            return _Result()

    class Store:
        failed = False

        def delete(self, key: str) -> None:
            if key == "source-key" and not self.failed:
                self.failed = True
                raise OSError("temporary object-store failure")

    conn = Conn()
    store = Store()
    monkeypatch.setattr(service, "engine", _Engine(conn))
    monkeypatch.setattr(service, "get_artifact_store", lambda: store)

    assert service.cleanup_expired() == 0
    assert conn.deleted is False
    assert "purge will be retried" in caplog.text
    assert service.cleanup_expired() == 1
    assert conn.deleted is True


def test_revise_creates_new_immutable_version_and_supersedes_parent(monkeypatch):
    captured: dict[str, object] = {}

    class Conn:
        def execute(self, statement, params=None):
            sql = str(statement)
            if "FROM governance.code_analysis_jobs j" in sql:
                return _Result(
                    one={
                        "status": "awaiting_confirmation",
                        "expires_at": datetime.now(timezone.utc)
                        + timedelta(minutes=10),
                        "file_id": "file_source",
                        "chat_session_id": "session-1",
                        "source_turn_id": "turn-1",
                        "question": "Old question",
                        "plan_data": {
                            "question": "Old question",
                            "file_id": "file_source",
                            "file_sha256": "b" * 64,
                            "data_contract": {"file_type": "csv"},
                            "retention_hours": 24,
                            "expected_output": ["conclusion"],
                        },
                        "revision": 1,
                        "resource_limits": {},
                        "file_expires_at": datetime.now(timezone.utc)
                        + timedelta(hours=1),
                        "purged_at": None,
                    }
                )
            if "SET status='superseded'" in sql:
                captured["superseded"] = params["id"]
            if "INSERT INTO governance.code_analysis_jobs" in sql:
                captured["insert"] = dict(params)
            return _Result()

    monkeypatch.setattr(service, "engine", _Engine(Conn()))
    monkeypatch.setattr(service, "_id", lambda _prefix: "caj_revision_2")
    monkeypatch.setattr(
        service,
        "get_job",
        lambda _context, job_id: {
            "job_id": job_id,
            "status": "awaiting_confirmation",
            "revision": 2,
        },
    )

    revised = service.revise_job(_context(), "caj_revision_1", "New question")

    assert revised["job_id"] == "caj_revision_2"
    assert captured["superseded"] == "caj_revision_1"
    inserted = captured["insert"]
    assert inserted["parent"] == "caj_revision_1"
    assert inserted["turn"] == "turn-1"
    assert inserted["revision"] == 2
    assert json.loads(inserted["plan"])["question"] == "New question"


def test_artifact_download_uses_chart_metadata(monkeypatch):
    payload = b"\x89PNG\r\n"

    class Conn:
        def execute(self, statement, _params=None):
            if "SELECT kind,object_key,sha256,size_bytes,metadata" in str(statement):
                return _Result(
                    one={
                        "kind": "chart",
                        "object_key": "chart-key",
                        "sha256": __import__("hashlib").sha256(payload).hexdigest(),
                        "size_bytes": len(payload),
                        "metadata": {
                            "content_type": "image/png",
                            "name": "趋势 图.png",
                        },
                    }
                )
            return _Result()

    class Store:
        def get(self, _key: str, *, max_bytes: int | None = None) -> bytes:
            assert max_bytes == len(payload)
            return payload

    monkeypatch.setattr(routes, "engine", _Engine(Conn()))
    monkeypatch.setattr(routes, "get_artifact_store", lambda: Store())

    response = routes.artifact("caj_chart", "art_chart", _context())

    assert response.media_type == "image/png"
    assert response.body == payload
    assert "filename*=UTF-8''" in response.headers["content-disposition"]
    assert (
        "%E8%B6%8B%E5%8A%BF%20%E5%9B%BE.png" in response.headers["content-disposition"]
    )
    assert response.headers["cache-control"] == "private, no-store"


def test_status_stream_has_heartbeats_and_distinct_terminal_event(monkeypatch):
    records = [
        {"job_id": "caj_stream", "status": "queued", "error_code": None},
        {"job_id": "caj_stream", "status": "queued", "error_code": None},
        {
            "job_id": "caj_stream",
            "status": "completed",
            "error_code": None,
            "result_data": {"summary": {}},
        },
    ]

    def fake_get_job(_context, _job_id):
        return records.pop(0) if len(records) > 1 else records[0]

    monkeypatch.setattr(routes, "get_job", fake_get_job)
    monkeypatch.setattr(routes.time, "sleep", lambda _seconds: None)
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[require_tenant_context] = _context

    response = TestClient(app).get("/code-analysis/jobs/caj_stream/events")

    assert response.status_code == 200
    assert '"event": "heartbeat"' in response.text
    assert '"event": "completed"' in response.text
    assert '"sequence_number":' in response.text

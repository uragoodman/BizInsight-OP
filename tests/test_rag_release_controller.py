from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from app.rag import flags
from app.rag import release_controller as controller
from app.rag.retriever import retrieve_for_release
from app.tenancy import TenantContext


def _context(tenant: str = "tenant-a") -> TenantContext:
    return TenantContext(tenant_id=tenant, user_id="user-a", role="admin", scope_type="platform")


def test_rollout_state_machine_rejects_skipped_or_illegal_transitions() -> None:
    assert controller.transition_allowed("evaluated", "shadowing")
    assert controller.transition_allowed("canary_25", "canary_100")
    assert not controller.transition_allowed("evaluated", "canary_5")
    assert not controller.transition_allowed("published", "canary_25")


def test_canary_bucket_is_sticky_and_allowlist_overrides_percentage() -> None:
    candidate = {"stage": "canary_5", "canary_percent": 5, "shadow_sample_rate": 1.0, "tenant_allowlist": []}
    first = controller.rollout_assignment({"release_id": "stable"}, candidate, _context(), "session-1", "trace-1")
    second = controller.rollout_assignment({"release_id": "stable"}, candidate, _context(), "session-1", "trace-2")
    assert first["bucket"] == second["bucket"]
    assert first["canary"] == second["canary"]
    allowlisted = controller.rollout_assignment(None, {**candidate, "tenant_allowlist": ["tenant-a"]}, _context(), "session-2", "trace-3")
    assert allowlisted["canary"] is True


def test_openfeature_absence_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    flags._client.cache_clear()
    monkeypatch.setattr(flags, "_client", lambda: None)
    assert flags.enabled("rag.canary.enabled", True, {}) is False
    assert flags.rollout_flags({"tenant_id": "t", "user_id": "u"})["canary"] is False


def test_unleash_bootstrap_can_enable_a_remote_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def initialize_client(self):
            return None
        def is_enabled(self, key, context, fallback_function):
            return key == "rag.canary.enabled" and context["tenant_id"] == "tenant-a"
        def destroy(self):
            return None

    settings = types.SimpleNamespace(
        rag_unleash_url="http://unleash:4242/api", rag_unleash_api_token="token",
        rag_unleash_app_name="bizinsight", rag_unleash_refresh_interval_seconds=5,
    )
    flags._client.cache_clear()
    monkeypatch.setattr(flags, "get_settings", lambda: settings)
    monkeypatch.setitem(sys.modules, "UnleashClient", types.SimpleNamespace(UnleashClient=FakeClient))
    assert flags.bootstrap() is True
    assert flags.enabled("rag.canary.enabled", False, {"tenant_id": "tenant-a"}) is True
    assert flags.enabled("rag.shadow.enabled", True, {"tenant_id": "tenant-a"}) is False
    flags.shutdown()


def test_release_retrieval_refuses_mutable_assets() -> None:
    with pytest.raises(ValueError, match="local_snapshot_ref"):
        retrieve_for_release("what is GMV", {"spec": {"backend": "local_hybrid"}})
    with pytest.raises(ValueError, match="qdrant_collection"):
        retrieve_for_release("what is GMV", {"spec": {"backend": "qdrant_hybrid"}})


def test_qdrant_release_manifest_must_bind_collection_and_fingerprint(tmp_path: Path) -> None:
    manifest = tmp_path / "release.json"
    spec = {
        "backend": "qdrant_vector",
        "qdrant_collection": "knowledge_v2",
        "release_manifest_ref": str(manifest),
    }
    manifest.write_text(json.dumps({"payload": {"collection": "other"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="collection does not match"):
        controller._validate_qdrant_release_manifest(spec)

    manifest.write_text(
        json.dumps({"payload": {"collection": "knowledge_v2", "index_fingerprint": {}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing index_fingerprint"):
        controller._validate_qdrant_release_manifest(spec)

    manifest.write_text(
        json.dumps({
            "payload": {
                "collection": "knowledge_v2",
                "index_fingerprint": {"digest": "abc"},
            },
        }),
        encoding="utf-8",
    )
    controller._validate_qdrant_release_manifest(spec)


def test_promote_requires_a_passed_deterministic_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        def scalar_one_or_none(self):
            return None

    class Connection:
        def execute(self, *_args, **_kwargs):
            return Result()

    class Begin:
        def __enter__(self):
            return Connection()
        def __exit__(self, *_args):
            return False

    class Engine:
        def begin(self):
            return Begin()

    monkeypatch.setattr(controller, "engine", Engine())
    monkeypatch.setattr(controller, "get_release", lambda _id: {"release_id": "candidate", "status": "canary_100"})
    with pytest.raises(ValueError, match="deterministic evaluation"):
        controller.promote("candidate", "admin")


def test_rollout_rejects_unknown_thresholds_before_database_write(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(controller, "get_release", lambda _id: {"release_id": "candidate", "status": "evaluated"})
    with pytest.raises(ValueError, match="Invalid rollout thresholds"):
        controller.set_rollout("candidate", "admin", "shadowing", thresholds={"made_up": 1.0})


def test_rollout_requires_unified_gate_before_database_write(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.quality import evaluations
    monkeypatch.setattr(controller, "get_release", lambda _id: {"release_id": "candidate", "status": "evaluated"})
    monkeypatch.setattr(evaluations, "rag_gate_passed", lambda _id: False)
    with pytest.raises(ValueError, match="unified RAG quality gate"):
        controller.set_rollout("candidate", "admin", "shadowing")


def test_promote_requires_unified_gate_after_deterministic_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.quality import evaluations
    class Result:
        def scalar_one_or_none(self): return 1
    class Connection:
        def execute(self, *_args, **_kwargs): return Result()
    class Begin:
        def __enter__(self): return Connection()
        def __exit__(self, *_args): return False
    class Engine:
        def begin(self): return Begin()
    monkeypatch.setattr(controller, "engine", Engine())
    monkeypatch.setattr(controller, "get_release", lambda _id: {"release_id": "candidate", "status": "canary_100"})
    monkeypatch.setattr(evaluations, "rag_gate_passed", lambda _id: False)
    with pytest.raises(ValueError, match="unified RAG quality gate"):
        controller.promote("candidate", "admin")


def test_promote_rejects_unresolved_critical_incident(monkeypatch: pytest.MonkeyPatch) -> None:
    """A passed deterministic/unified gate never overrides a critical incident."""
    from app.quality import evaluations

    class Result:
        def __init__(self, value): self.value = value
        def scalar_one_or_none(self): return self.value

    class Connection:
        def execute(self, statement, *_args, **_kwargs):
            sql = str(statement)
            if "rag_release_evaluations" in sql:
                return Result(1)
            if "rag_release_incidents" in sql:
                return Result(1)
            return Result(None)

    class Begin:
        def __enter__(self): return Connection()
        def __exit__(self, *_args): return False

    class Engine:
        def begin(self): return Begin()

    monkeypatch.setattr(controller, "engine", Engine())
    monkeypatch.setattr(controller, "get_release", lambda _id: {"release_id": "candidate", "status": "canary_100"})
    monkeypatch.setattr(evaluations, "rag_gate_passed", lambda _id: True)
    with pytest.raises(ValueError, match="Unresolved critical"):
        controller.promote("candidate", "admin")


def test_quality_block_pauses_active_rollout_and_deactivates_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Late asynchronous contract failures must make a candidate unroutable."""
    statements: list[tuple[str, dict]] = []

    class MappingResult:
        def mappings(self): return self
        def first(self): return {"status": "canary_25"}

    class Connection:
        def execute(self, statement, params=None):
            statements.append((str(statement), params or {}))
            if "SELECT status FROM governance.rag_releases" in str(statement):
                return MappingResult()
            return MappingResult()

    class Begin:
        def __enter__(self): return Connection()
        def __exit__(self, *_args): return False

    class Engine:
        def begin(self): return Begin()

    monkeypatch.setattr(controller, "engine", Engine())
    controller.block_release_for_quality_gate("candidate", "contract-worker", ["citation contract failed"])
    policy_updates = [item for item in statements if "UPDATE governance.rag_rollout_policies" in item[0]]
    release_updates = [item for item in statements if "UPDATE governance.rag_releases SET status" in item[0]]
    assert policy_updates and policy_updates[0][1]["id"] == "candidate"
    assert release_updates and release_updates[0][1]["status"] == "paused"


def test_assignment_ignores_active_policy_for_non_routable_release(monkeypatch: pytest.MonkeyPatch) -> None:
    """Policy state alone cannot re-enable a failed/paused candidate."""
    seen: list[str] = []

    class MappingResult:
        def mappings(self): return self
        def first(self): return None

    class Connection:
        def execute(self, statement, *_args, **_kwargs):
            seen.append(str(statement))
            return MappingResult()

    class Begin:
        def __enter__(self): return Connection()
        def __exit__(self, *_args): return False

    class Engine:
        def begin(self): return Begin()

    monkeypatch.setattr(controller, "engine", Engine())
    monkeypatch.setattr(controller, "published_release", lambda: {"release_id": "stable"})
    route = controller.assignment(_context(), "session", "trace")
    assert route["candidate"] is None
    assert any("r.status IN ('shadowing','canary_5','canary_25','canary_100')" in sql for sql in seen)


def test_canary_100_online_gate_restores_parent_release(monkeypatch: pytest.MonkeyPatch) -> None:
    executed: list[str] = []

    class MappingResult:
        def __init__(self, row): self.row = row
        def mappings(self): return self
        def first(self): return self.row
        def one(self): return self.row

    class Connection:
        def execute(self, statement, *_args, **_kwargs):
            sql = str(statement)
            executed.append(sql)
            if "FROM governance.rag_rollout_policies" in sql:
                return MappingResult({"policy_id": "policy", "auto_rollback": True, "min_samples": 100, "stage": "canary_100", "thresholds": controller._DEFAULT_THRESHOLDS})
            if "FROM governance.rag_shadow_comparisons" in sql:
                return MappingResult({"samples": 100, "latency_ratio": 1.4, "error_rate": 0, "empty_rate": 0, "citationless_rate": 0})
            if "SELECT parent_release_id" in sql:
                return MappingResult({"parent_release_id": "stable"})
            return MappingResult(None)

    class Begin:
        def __enter__(self): return Connection()
        def __exit__(self, *_args): return False

    class Engine:
        def begin(self): return Begin()

    exports: list[str] = []
    monkeypatch.setattr(controller, "engine", Engine())
    monkeypatch.setattr(controller, "get_release", lambda release_id: {"release_id": release_id})
    monkeypatch.setattr(controller, "_export_manifest", lambda release: exports.append(release["release_id"]))
    monkeypatch.setattr(controller, "record_event", lambda *_args, **_kwargs: None)
    assert controller.evaluate_online_safety("candidate") is True
    assert exports == ["stable"]
    assert any("status='published'" in sql for sql in executed)

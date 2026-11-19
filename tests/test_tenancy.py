from datetime import date
from uuid import uuid4

import pytest

from app.analysis.contracts import PeriodSpec
from app.analysis.targets import MetricTarget
from app.harness.storage import read_trace, write_trace
from app.rag.schemas import KnowledgeChunk, RagCitation, RagRetrievalSignals, RagSearchHit, RagSearchResult
from app.rag.retriever import retrieve_knowledge
from app.tenancy import SELLER_SCOPE, TenantContext, activate_tenant_context, current_tenant_context


def tenant(tenant_id: str = "seller-a") -> TenantContext:
    return TenantContext(tenant_id=tenant_id, user_id="user-a", role="analyst", scope_type=SELLER_SCOPE, seller_id="seller-id-a")


def test_tenant_context_is_immutable_and_scoped():
    context = tenant()
    assert context.as_state()["seller_id"] == "seller-id-a"
    with activate_tenant_context(context):
        assert current_tenant_context() == context
    assert current_tenant_context() is None


def test_seller_scope_requires_a_real_seller_id():
    try:
        TenantContext(tenant_id="a", user_id="u", role="analyst", scope_type=SELLER_SCOPE)
    except ValueError as exc:
        assert "seller_id" in str(exc)
    else:
        raise AssertionError("seller scope without seller_id must fail")


def test_trace_read_rejects_another_seller(monkeypatch, tmp_path):
    monkeypatch.setattr("app.harness.storage._trace_dir", lambda: tmp_path)
    write_trace("tenant-trace", {"tenant_id": "seller-a", "node_history": []})
    assert read_trace("tenant-trace", tenant("seller-a")) is not None
    assert read_trace("tenant-trace", tenant("seller-b")) is None


def test_qdrant_search_merges_only_current_tenant_private_knowledge(monkeypatch):
    context = tenant()
    global_hit = RagSearchHit(
        chunk_id="global", text="global", score=1.0,
        citation=RagCitation(source_path="global.md", title="global", doc_type="metric"),
        signals=RagRetrievalSignals(rerank_score=1.0),
    )
    monkeypatch.setattr("app.rag.retriever.get_rag_settings", lambda: type("Config", (), {"backend": "qdrant", "top_k": 3, "candidate_k": 3, "reranker_top_k": 3})())
    monkeypatch.setattr("app.rag.retriever._retrieve_from_qdrant", lambda *args, **kwargs: RagSearchResult(query="x", backend="qdrant", strategy="qdrant_vector", top_k=3, hits=[global_hit]))
    private = KnowledgeChunk(chunk_id="tenant:seller-a:private", doc_id="private", title="private", text="private", source_path="private.md", doc_type="metric")
    private_hit = RagSearchHit(chunk_id=private.chunk_id, text="private", score=2.0, citation=RagCitation(source_path="private.md", title="private", doc_type="metric"), signals=RagRetrievalSignals(rerank_score=2.0))
    monkeypatch.setattr("app.rag.retriever._tenant_private_chunks", lambda received: [private])
    monkeypatch.setattr("app.rag.retriever._retrieve_local", lambda *args, **kwargs: RagSearchResult(query="x", backend="local", top_k=3, hits=[private_hit]))
    first = retrieve_knowledge("x", tenant_context=context, reranker_strategy="rule")
    assert [hit.chunk_id for hit in first.hits] == ["tenant:seller-a:private", "global"]


def test_seller_report_contains_target_comparison(monkeypatch):
    from app.tenancy.reports import build_seller_periodic_report
    context = tenant()
    values = {"2018-08-01:2018-08-31": 120.0, "2018-07-01:2018-07-31": 100.0, "2017-08-01:2017-08-31": 90.0}
    monkeypatch.setattr("app.tenancy.reports.query_seller_metric", lambda _c, _m, start, end: {"value": values[f"{start}:{end}"]})
    target = MetricTarget(metric_name="sales_gmv", period_start=date(2018, 8, 1), period_end=date(2018, 8, 31), target_value=100, version="v1", effective_at=date(2018, 8, 1), created_by="user", scope="tenant")
    report = build_seller_periodic_report(context, PeriodSpec(start_date=date(2018, 8, 1), end_date=date(2018, 8, 31), granularity="month"), ["sales_gmv"], target_lookup=lambda *_: target, include_charts=False)
    assert report.metrics[0].target_comparison is not None
    assert report.metrics[0].target_comparison.attainment_rate == 120.0


def test_seller_legacy_sales_gmv_resolves_to_registry_definition():
    from app.analysis.semantic import get_metric_definition
    definition = get_metric_definition("sales_gmv", "seller")
    assert definition.name == "gmv"
    assert definition.unit == "BRL"


class _FakeMappings:
    def __init__(self, rows): self._rows = rows

    def __iter__(self): return iter(self._rows)


class _FakeResult:
    def __init__(self, rows): self._rows = rows

    def mappings(self): return _FakeMappings(self._rows)


class _FakeConnection:
    """记录 set_config 与 INSERT 调用的假连接，模拟 RLS 会话变量行为。"""

    def __init__(self, tenant_ids):
        self.tenant_ids = tenant_ids
        self.configs = []
        self.inserted = []

    def execute(self, statement, params=None):
        sql = str(statement)
        if "set_config" in sql:
            self.configs.append(params)
        elif "SELECT DISTINCT tenant_id" in sql:
            return _FakeResult([{"tenant_id": item} for item in self.tenant_ids])
        elif "INSERT INTO analytics.tenant_metric_targets" in sql:
            self.inserted.append(params)
        return _FakeResult([])


class _FakeBegin:
    def __init__(self, connection): self._connection = connection

    def __enter__(self): return self._connection

    def __exit__(self, *exc_info): return False


class _FakeEngine:
    def __init__(self, connection): self._connection = connection

    def begin(self): return _FakeBegin(self._connection)


def _platform_context() -> TenantContext:
    return TenantContext(tenant_id="tenant-platform", user_id="admin-a", role="admin", scope_type="platform")


def _seller_target(seller_id: str = "seller-id-a") -> MetricTarget:
    return MetricTarget(
        metric_name="sales_gmv", period_start=date(2025, 8, 1), period_end=date(2025, 8, 31),
        target_value=100, version="v1", effective_at=date(2025, 7, 15), created_by="admin-a",
        scope="tenant", seller_id=seller_id,
    )


def test_platform_target_store_writes_one_row_per_mapped_tenant(monkeypatch):
    from app.tenancy.target_store import PlatformTargetStore

    connection = _FakeConnection(["tenant-a", "tenant-b"])
    monkeypatch.setattr("app.tenancy.target_store.engine", _FakeEngine(connection))
    monkeypatch.setattr("app.tenancy.target_store.get_metric_definition", lambda *_: None)
    PlatformTargetStore(_platform_context()).put(_seller_target())
    # RLS WITH CHECK 要求 tenant_id 等于会话变量：先开平台可见性，再逐租户切换写入。
    scopes = [params["scope"] for params in connection.configs if params and "scope" in params]
    tenants = [params["tenant"] for params in connection.configs if params and "tenant" in params]
    assert scopes == ["platform"]
    assert tenants == ["tenant-a", "tenant-b"]
    assert len(connection.inserted) == 2
    assert {row["tenant"] for row in connection.inserted} == {"tenant-a", "tenant-b"}


def test_platform_target_store_rejects_unmapped_seller(monkeypatch):
    from app.tenancy.target_store import PlatformTargetStore

    connection = _FakeConnection([])
    monkeypatch.setattr("app.tenancy.target_store.engine", _FakeEngine(connection))
    monkeypatch.setattr("app.tenancy.target_store.get_metric_definition", lambda *_: None)
    with pytest.raises(ValueError, match="not mapped"):
        PlatformTargetStore(_platform_context()).put(_seller_target())
    assert connection.inserted == []


def test_target_store_routing_keeps_admin_and_seller_on_same_table(monkeypatch, tmp_path):
    from app.analysis.targets import TargetStore
    from app.api.routes.analysis import _target_store
    from app.core.config import get_settings
    from app.tenancy.target_store import PlatformTargetStore, TenantTargetStore

    monkeypatch.setattr(get_settings(), "phase6_store_path", str(tmp_path / "routing.sqlite"))
    assert isinstance(_target_store(tenant()), TenantTargetStore)
    assert isinstance(_target_store(_platform_context(), "seller-id-a"), PlatformTargetStore)
    # 全局目标（不带 seller_id）仍走 legacy 存储，平台报告的目标比较不受影响。
    assert isinstance(_target_store(_platform_context()), TargetStore)


def test_platform_target_roundtrip_on_live_postgres():
    """真库端到端：管理员写入 → 平台视角可见 → 清理。数据库不可达或无映射数据时跳过。"""
    from sqlalchemy import text

    from app.data.database import engine
    from app.tenancy.target_store import PlatformTargetStore

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("live PostgreSQL is not reachable")
    with engine.connect() as connection:
        mapping = connection.execute(
            text("SELECT tenant_id, seller_id FROM iam.tenant_seller_accounts ORDER BY tenant_id, seller_id LIMIT 1")
        ).mappings().first()
    if mapping is None:
        pytest.skip("no seller-tenant mapping seeded in iam.tenant_seller_accounts")

    version = f"v-itest-{uuid4().hex[:8]}"
    period = PeriodSpec(start_date=date(2025, 8, 1), end_date=date(2025, 8, 31))
    target = MetricTarget(
        metric_name="sales_gmv", period_start=period.start_date, period_end=period.end_date,
        target_value=100, version=version, effective_at=date(2025, 7, 15), created_by="itest",
        scope="tenant", seller_id=mapping["seller_id"],
    )
    store = PlatformTargetStore(_platform_context())
    try:
        store.put(target)
        assert store.get("sales_gmv", period, mapping["seller_id"]) is not None
        # 平台管理员列表视角（GET /analysis/targets 相同的 RLS 配置）必须能看到刚写入的行。
        with engine.connect() as connection:
            connection.execute(text("SELECT set_config('app.scope_type','platform',true)"))
            count = connection.execute(
                text("SELECT count(*) FROM analytics.tenant_metric_targets WHERE seller_id=:seller AND version=:version"),
                {"seller": mapping["seller_id"], "version": version},
            ).scalar_one()
        assert count >= 1
    finally:
        with engine.begin() as connection:
            connection.execute(text("SELECT set_config('app.scope_type','platform',true)"))
            connection.execute(
                text("DELETE FROM analytics.tenant_metric_targets WHERE seller_id=:seller AND version=:version"),
                {"seller": mapping["seller_id"], "version": version},
            )

"""GMV 数据版本契约与可重放快照测试：版本隔离、RLS 跨租户、replay unavailable 语义。

依赖一个已 seed 的 PostgreSQL（与 tests/test_memory_semantic.py 相同）。
每个用例使用独立租户 ID，避免重复运行残留数据互相干扰。
"""
from __future__ import annotations

from datetime import date
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from app.analysis.contracts import InsightAngle, InsightArtifact, InsightBrief, PeriodSpec, PeriodicReport, ReportMetric, ReportScope
from app.api.routes.data_versions import ReplayRequest, replay
from app.data.database import engine
from app.data_versions.service import (
    get_manifest,
    impact_memories,
    query_gmv_manifest,
    resolve_gmv_manifest,
    seal_gmv_version,
)
from app.memory.contracts import AnalysisMemory
from app.memory.repository import MemoryRepository
from app.tenancy import TenantContext
from app.tenancy.report_store import TenantReportStore

_USER = "dv-test-user"


def _ctx(tenant_id: str) -> TenantContext:
    return TenantContext(tenant_id=tenant_id, user_id=_USER, role="admin", scope_type="platform")


def _ensure_tenant(tenant_id: str, seller_id: str) -> None:
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO iam.tenants(tenant_id,tenant_name,scope_type) VALUES(:t,:t,'seller') ON CONFLICT (tenant_id) DO NOTHING"), {"t": tenant_id})
        conn.execute(text("INSERT INTO iam.users(user_id,display_name,email) VALUES(:u,:u,:email) ON CONFLICT (user_id) DO NOTHING"), {"u": _USER, "email": f"{_USER}@example.com"})
        conn.execute(text("INSERT INTO iam.tenant_memberships(tenant_id,user_id,role) VALUES(:t,:u,'admin') ON CONFLICT (tenant_id,user_id) DO NOTHING"), {"t": tenant_id, "u": _USER})
        conn.execute(text("INSERT INTO iam.tenant_seller_accounts(tenant_id,seller_id) VALUES(:t,:s) ON CONFLICT (tenant_id) DO NOTHING"), {"t": tenant_id, "s": seller_id})


def _seed_facts(tenant_id: str, seller_id: str, rows: list[tuple[str, int, str, float]]) -> None:
    with engine.begin() as conn:
        conn.execute(text("SELECT set_config('app.tenant_id', :t, true)"), {"t": tenant_id})
        conn.execute(text("SELECT set_config('app.scope_type', 'platform', true)"))
        for order_id, item_id, purchase_date, price in rows:
            conn.execute(text(
                """INSERT INTO analytics.tenant_fact_order_items
                   (tenant_id,order_id,order_item_id,seller_id,customer_unique_id,purchase_date,price,freight_value,order_status)
                   VALUES(:t,:o,:oi,:s,'cust',:d,:p,0.0,'delivered')"""
            ), {"t": tenant_id, "o": order_id, "oi": item_id, "s": seller_id, "d": purchase_date, "p": price})


def _insert_memory(tenant_id: str, seller_id: str, *, source_data_version: str, start: date, end: date) -> str:
    memory_id = f"mem_dv_{uuid4().hex[:12]}"
    MemoryRepository(_ctx(tenant_id)).insert_analysis_memory(AnalysisMemory(
        memory_id=memory_id, seller_id=seller_id, start_date=start, end_date=end,
        scope_fingerprint="c" * 64, metric_tags=["gmv"], task_type="anomaly_analysis",
        insight_text=f"versioned claim {memory_id}", evidence_ids=["ev_abc123"],
        source_trace_id=f"trace_dv_{uuid4().hex[:8]}", source_data_version=source_data_version,
        metric_definition_version="m" * 64, normalized_claim_hash=uuid4().hex * 2,
    ))
    return memory_id


def test_manifest_isolation_v1_vs_v2() -> None:
    tenant = f"dv-iso-{uuid4().hex[:6]}"
    seller = f"s-iso-{uuid4().hex[:6]}"
    _ensure_tenant(tenant, seller)
    _seed_facts(tenant, seller, [("o1", 1, date(2018, 8, 1), 10.0), ("o2", 1, date(2018, 8, 2), 5.0)])
    ctx = _ctx(tenant)

    v1 = seal_gmv_version(ctx)
    m1 = resolve_gmv_manifest(ctx, {"seller_id": seller}, "f" * 64)
    value1 = query_gmv_manifest(ctx, m1["manifest_id"], seller, date(2018, 8, 1), date(2018, 8, 2))["value"]
    assert value1 == 15.0

    # 幂等：同一内容重复导入返回同一版本，不生成无意义新版本。
    assert seal_gmv_version(ctx)["version_id"] == v1["version_id"]

    # 模拟一次数据修订：单行加价 1 元后重新 seal。
    with engine.begin() as conn:
        conn.execute(text("SELECT set_config('app.tenant_id', :t, true)"), {"t": tenant})
        conn.execute(text("SELECT set_config('app.scope_type', 'platform', true)"))
        conn.execute(text("""UPDATE analytics.tenant_fact_order_items SET price=price+1
            WHERE tenant_id=:t AND order_id='o1'"""), {"t": tenant})
    v2 = seal_gmv_version(ctx, parent_version_id=v1["version_id"])
    assert v2["version_id"] != v1["version_id"]

    m2 = resolve_gmv_manifest(ctx, {"seller_id": seller}, "f" * 64)
    assert m2["manifest_id"] != m1["manifest_id"]
    value2 = query_gmv_manifest(ctx, m2["manifest_id"], seller, date(2018, 8, 1), date(2018, 8, 2))["value"]
    assert value2 == 16.0

    # 版本隔离：旧 manifest 重放仍返回旧值，不受最新版本影响。
    assert query_gmv_manifest(ctx, m1["manifest_id"], seller, date(2018, 8, 1), date(2018, 8, 2))["value"] == 15.0


def test_impact_memories_marks_only_overlapping_and_matching_seller() -> None:
    tenant = f"dv-impact-{uuid4().hex[:6]}"
    seller = f"s-impact-{uuid4().hex[:6]}"
    _ensure_tenant(tenant, seller)
    _seed_facts(tenant, seller, [("o1", 1, date(2018, 8, 1), 10.0)])
    ctx = _ctx(tenant)

    v1 = seal_gmv_version(ctx)
    m1 = resolve_gmv_manifest(ctx, {"seller_id": seller}, "e" * 64)
    overlap_id = _insert_memory(tenant, seller, source_data_version=m1["manifest_hash"], start=date(2018, 8, 1), end=date(2018, 8, 7))
    disjoint_id = _insert_memory(tenant, seller, source_data_version=m1["manifest_hash"], start=date(2018, 9, 1), end=date(2018, 9, 7))

    with engine.begin() as conn:
        conn.execute(text("SELECT set_config('app.tenant_id', :t, true)"), {"t": tenant})
        conn.execute(text("SELECT set_config('app.scope_type', 'platform', true)"))
        conn.execute(text("""UPDATE analytics.tenant_fact_order_items SET price=price+1
            WHERE tenant_id=:t AND order_id='o1'"""), {"t": tenant})
    v2 = seal_gmv_version(ctx, parent_version_id=v1["version_id"])

    impacted = impact_memories(ctx, v2["version_id"])
    impacted_ids = {row["memory_id"] for row in impacted}
    # 日期相交且同 seller → 受影响；日期不相交 → 不受影响。
    assert overlap_id in impacted_ids
    assert disjoint_id not in impacted_ids

    impacted_apply = impact_memories(ctx, v2["version_id"], apply=True)
    assert any(row["memory_id"] == overlap_id for row in impacted_apply)
    assert MemoryRepository(ctx).get_analysis_memory(overlap_id)["status"] == "stale"
    assert MemoryRepository(ctx).get_analysis_memory(disjoint_id)["status"] == "active"


def test_rls_cross_tenant_cannot_see_manifest_or_query() -> None:
    tenant_a = f"dv-a-{uuid4().hex[:6]}"
    tenant_b = f"dv-b-{uuid4().hex[:6]}"
    seller_a = f"s-a-{uuid4().hex[:6]}"
    seller_b = f"s-b-{uuid4().hex[:6]}"
    _ensure_tenant(tenant_a, seller_a)
    _ensure_tenant(tenant_b, seller_b)
    _seed_facts(tenant_a, seller_a, [("o1", 1, date(2018, 8, 1), 10.0)])

    ctx_a = _ctx(tenant_a)
    v1 = seal_gmv_version(ctx_a)
    m1 = resolve_gmv_manifest(ctx_a, {"seller_id": seller_a}, "d" * 64)

    # 租户 B 看不到租户 A 的 manifest（RLS 强制隔离）。
    assert get_manifest(_ctx(tenant_b), m1["manifest_hash"]) is None
    # 租户 B 无法用 A 的 manifest 查询数据。
    with pytest.raises(ValueError):
        query_gmv_manifest(_ctx(tenant_b), m1["manifest_id"], seller_b, date(2018, 8, 1), date(2018, 8, 2))


def test_replay_unavailable_for_memory_without_valid_manifest() -> None:
    tenant = f"dv-replay-{uuid4().hex[:6]}"
    seller = f"s-replay-{uuid4().hex[:6]}"
    _ensure_tenant(tenant, seller)
    ctx = _ctx(tenant)

    # 记忆的 source_data_version 不是任何已登记 manifest → 诚实返回 unavailable。
    memory_id = _insert_memory(tenant, seller, source_data_version=uuid4().hex * 2, start=date(2018, 8, 1), end=date(2018, 8, 7))
    result = replay(ReplayRequest(artifact_type="analysis_memory", artifact_id=memory_id, mode="exact"), "system", ctx)
    assert result["status"] == "unavailable"


def test_replay_unavailable_for_legacy_report_and_enum_validation() -> None:
    tenant = f"dv-rep-{uuid4().hex[:6]}"
    seller = f"s-rep-{uuid4().hex[:6]}"
    _ensure_tenant(tenant, seller)
    ctx = _ctx(tenant)

    # 未版本化报告：data_version_manifest_hash 为空 → replay 不伪装成功。
    with engine.begin() as conn:
        conn.execute(text("SELECT set_config('app.tenant_id', :t, true)"), {"t": tenant})
        conn.execute(text("SELECT set_config('app.scope_type', 'platform', true)"))
        conn.execute(text(
            """INSERT INTO governance.tenant_reports(tenant_id,report_id,version,status,content,content_hash,report_json,actor,data_version_manifest_id,data_version_manifest_hash)
               VALUES(:t,'r_dv_legacy',1,'draft','x','y','{}','system',NULL,NULL)"""
        ), {"t": tenant})
    result = replay(ReplayRequest(artifact_type="report", artifact_id="r_dv_legacy", mode="exact"), "system", ctx)
    assert result["status"] == "unavailable"
    assert "manifest" in result["reason"]

    result = replay(ReplayRequest(artifact_type="insight", artifact_id="r_dv_legacy", mode="compare_latest"), "system", ctx)
    assert result["status"] == "unavailable"

    # 非法 artifact_type / mode → 422。
    for bad in (
        ReplayRequest(artifact_type="bogus", artifact_id="x", mode="exact"),
        ReplayRequest(artifact_type="report", artifact_id="x", mode="sideways"),
    ):
        with pytest.raises(HTTPException) as excinfo:
            replay(bad, "system", ctx)
        assert excinfo.value.status_code == 422


def test_report_and_insight_persist_manifest_lineage() -> None:
    """报告草稿 / generating 占位 / finalize 洞察都写入 manifest 谱系列。"""
    from datetime import datetime, timezone

    tenant = f"dv-lineage-{uuid4().hex[:6]}"
    seller = f"s-lineage-{uuid4().hex[:6]}"
    _ensure_tenant(tenant, seller)
    ctx = _ctx(tenant)
    manifest_id, manifest_hash = f"mfest_{uuid4().hex[:44]}", uuid4().hex * 2
    report_id = f"r_dv_lineage_{uuid4().hex[:6]}"

    report = PeriodicReport(
        report_id=report_id, report_type="weekly",
        period=PeriodSpec(start_date=date(2018, 8, 1), end_date=date(2018, 8, 7)),
        scope=ReportScope(scope_type="seller", seller_id=seller, seller_label=seller),
        title="t", summary="s",
        metrics=[ReportMetric(metric_name="gmv", value=100.0, unit="元", evidence_ids=["ev_gmv_series"])],
        evidence_records={"ev_gmv_series": {"kind": "metric_series", "metric_name": "gmv"}},
        lineage_refs=["tenant_gmv_items -> gmv"],
        data_freshness_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    store = TenantReportStore(ctx)
    stored = store.create_draft(report, "content", "system",
                                data_version_manifest_id=manifest_id, data_version_manifest_hash=manifest_hash)
    assert stored["data_version_manifest_id"] == manifest_id
    assert stored["data_version_manifest_hash"] == manifest_hash

    brief = InsightBrief(angle=InsightAngle.OVERVIEW)
    placeholder = store.create_generating(report_id, 1, brief, "system",
                                          data_version_manifest_id=manifest_id, data_version_manifest_hash=manifest_hash)
    assert placeholder["data_version_manifest_hash"] == manifest_hash

    artifact = InsightArtifact(
        report_id=report_id, report_version=1, insight_version=int(placeholder["insight_version"]),
        status="draft", brief=brief, input_snapshot_hash="i" * 64, prompt_version="v1",
        insights=[], model="test", data_version_manifest_id=manifest_id, data_version_manifest_hash=manifest_hash,
    )
    store.finalize_insight(report_id, 1, int(placeholder["insight_version"]), artifact)
    finalized = store.get_insight(report_id, 1, int(placeholder["insight_version"]))
    assert finalized["data_version_manifest_hash"] == manifest_hash

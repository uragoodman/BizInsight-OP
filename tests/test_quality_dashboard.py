# ============================================================
# 质量仪表盘服务单元测试
# 作用：验证 app/quality/service.py 的快照捕获、Trace 版本分桶与指标汇总逻辑。
# 测试策略：不依赖真实数据库/trace——用 monkeypatch 替换文件系统根目录与
#           trace/用量数据源，纯内存验证聚合与序列化行为。
# ============================================================
from __future__ import annotations

import json
from decimal import Decimal

from app.quality import service
from app.tenancy import PLATFORM_SCOPE, SELLER_SCOPE, TenantContext


def _context() -> TenantContext:
    """构造一个平台管理员租户上下文（质量快照所需的默认 scope）。"""
    return TenantContext("platform", "admin", "admin", PLATFORM_SCOPE)


def test_capture_quality_snapshot_keeps_versioned_history(tmp_path, monkeypatch) -> None:
    """验证：捕获快照会保留版本化历史，且 Decimal 成本可序列化。

    测试作用：覆盖 capture_snapshot → dashboard 的主链路。
    主要验证点：
        1. 把文件根目录替换为临时目录，写入一份 Golden 评测证据；
        2. 捕获快照后：golden 只暴露白名单字段，缺失的元数据字段为 None；
        3. dashboard 返回 latest + history，历史数量递增；
        4. 聚合成本（Decimal("0.12")）能正常写入 JSON，不会触发序列化异常。
    """
    # 把文件根目录替换为临时目录，避免测试写坏真实 data/。
    monkeypatch.setattr(service, "ROOT", tmp_path)
    monkeypatch.setattr(service, "QUALITY_DIR", tmp_path / "quality")
    # 固定 SHA 与假数据源，隔离 git/数据库等外部依赖。
    monkeypatch.setattr(service, "_git_sha", lambda: "abc123")
    monkeypatch.setattr(service, "list_traces", lambda _context: [])
    monkeypatch.setattr(service, "usage_overview", lambda **_kwargs: {"summary": {"calls": 0, "total_tokens": 0, "estimated_cost": Decimal("0.12")}})
    # 准备一份 Golden 评测证据（缺失的元数据字段应被白名单过滤为 None）。
    (tmp_path / "data" / "evaluations").mkdir(parents=True)
    (tmp_path / "data" / "evaluations" / "latest.json").write_text(
        json.dumps({"cases": 3, "passed": 3, "overall_rate": 1.0}), encoding="utf-8"
    )

    saved = service.capture_snapshot(_context())
    report = service.dashboard(_context())

    assert saved["golden"] == {"generated_at": None, "commit_sha": None, "dataset_version": None, "cases": 3, "passed": 3, "overall_rate": 1.0}
    assert report["latest"]["traces"]["terminal_rate"] is None
    assert len(report["history"]) == 1
    assert "0.12" in (tmp_path / "quality" / "snapshots.json").read_text(encoding="utf-8")


def test_dashboard_does_not_create_snapshot_directory_on_read(tmp_path, monkeypatch) -> None:
    """GET dashboard must work when the evidence directory is read-only/missing."""
    monkeypatch.setattr(service, "QUALITY_DIR", tmp_path / "quality")
    monkeypatch.setattr(service, "list_traces", lambda _context: [])
    monkeypatch.setattr(service, "usage_overview", lambda **_kwargs: {"summary": {}})
    monkeypatch.setattr(service, "insight_summary", lambda: {"samples": 0})

    report = service.dashboard(_context())

    assert report["history"] == []
    assert not (tmp_path / "quality").exists()


def test_dashboard_history_is_isolated_by_scope_and_tenant(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(service, "QUALITY_DIR", tmp_path / "quality")
    platform = _context()
    tenant_a = TenantContext("tenant-a", "owner-a", "owner", SELLER_SCOPE, seller_id="seller-a")
    tenant_b = TenantContext("tenant-b", "owner-b", "owner", SELLER_SCOPE, seller_id="seller-b")
    snapshots = {
        ("platform", "platform", "admin"): {"scope": "platform", "tenant_id": "platform", "user_id": "admin", "captured_at": "platform-snapshot"},
        ("tenant", "tenant-a", "owner-a"): {"scope": "tenant", "tenant_id": "tenant-a", "user_id": "owner-a", "captured_at": "tenant-a-snapshot"},
        ("tenant", "tenant-a", "owner-a-2"): {"scope": "tenant", "tenant_id": "tenant-a", "user_id": "owner-a-2", "captured_at": "tenant-a-other-user-snapshot"},
        ("tenant", "tenant-b", "owner-b"): {"scope": "tenant", "tenant_id": "tenant-b", "user_id": "owner-b", "captured_at": "tenant-b-snapshot"},
    }
    monkeypatch.setattr(
        service,
        "build_snapshot",
        lambda context: {
            "scope": "platform" if context.is_platform_admin else "tenant",
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
        },
    )
    path = tmp_path / "quality" / "snapshots.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"items": list(snapshots.values())}), encoding="utf-8")

    assert [item["tenant_id"] for item in service.dashboard(tenant_a)["history"]] == ["tenant-a"]
    assert [item["tenant_id"] for item in service.dashboard(tenant_b)["history"]] == ["tenant-b"]
    assert [item["tenant_id"] for item in service.dashboard(platform)["history"]] == ["platform"]


def test_tenant_quality_snapshot_does_not_expose_global_insight_aggregate(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(service, "ROOT", tmp_path)
    monkeypatch.setattr(service, "QUALITY_DIR", tmp_path / "quality")
    monkeypatch.setattr(service, "_git_sha", lambda: "abc123")
    monkeypatch.setattr(service, "trace_summary", lambda _context: {})
    monkeypatch.setattr(service, "usage_overview", lambda **_kwargs: {"summary": {}})
    monkeypatch.setattr(service, "insight_summary", lambda: {"samples": 99})

    tenant = TenantContext("tenant-a", "owner-a", "owner", SELLER_SCOPE, seller_id="seller-a")
    snapshot = service.build_snapshot(tenant)

    assert snapshot["tenant_id"] == "tenant-a"
    assert snapshot["user_id"] == "owner-a"
    assert snapshot["insights"] == {"samples": 0}


def test_trace_summary_separates_legacy_and_versioned_generations(monkeypatch) -> None:
    """验证：Trace 摘要能把 legacy 与版本化分代分开统计，且可被 JSON 序列化。

    测试作用：覆盖 trace_summary 的分桶与指标计算。
    主要验证点：
        1. 传入一条 legacy（无 trace_version）与一条 trace-v2 Trace，应产出 2 个分桶；
        2. 默认选中最新分代（selected_bucket != legacy），统计数据只统计该桶；
        3. 时长按节点列表求和（10+15=25ms），Critic 拒绝率 1.0；
        4. 错误码（tool_timeout）进入 reason_counts；
        5. 选中桶随全部分桶一起返回时不会出现自引用（json.dumps 必须成功）。
    """
    # 构造一条 legacy（无 trace_version）与一条 trace-v2 Trace。
    traces = {
        "legacy": {"state_snapshot": {"critic_verdict": "approved", "node_timings": {"triage": 4}}},
        "v2": {"trace_version": {"trace_kind": "agent_chat", "generation": "trace-v2", "schema_version": "2.0", "workflow_version": "governed-agent-v1", "prompt_version": "v1", "rag_release_version": "r1", "permission_policy_version": "v1"}, "state_snapshot": {"critic_verdict": "rejected", "node_timings": [{"node": "triage", "duration_ms": 10}, {"node": "critic", "duration_ms": 15}], "critic_decisions": [{"revision_target": "report", "data_status": "missing", "llm_issues": ["unsupported claim"]}], "errors": [{"code": "tool_timeout"}] }},
    }
    # 用假数据源替换 trace 读取，验证分桶与指标计算。
    monkeypatch.setattr(service, "list_traces", lambda _context: [{"trace_id": key} for key in traces])
    monkeypatch.setattr(service, "read_trace", lambda trace_id, _context: traces[trace_id])

    # 执行聚合，随后校验分桶、指标与自引用守卫。
    summary = service.trace_summary(_context())

    assert summary["selected_bucket"] != "legacy"
    assert len(summary["buckets"]) == 2
    assert summary["traces"] == 1
    assert summary["critic_reject_rate"] == 1.0
    assert summary["graph_latency_ms"]["p95"] == 25.0
    assert summary["reason_counts"]["error:tool_timeout"] == 1
    # 选中桶与全部分桶同时返回，此断言防呆：避免把对象嵌入自身导致 JSON 序列化失败。
    json.dumps(summary)

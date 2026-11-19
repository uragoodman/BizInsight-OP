"""AI 洞察层测试：选材、白名单校验、生成流水线、SQLite 制品存取。

覆盖用户收敛后的设计要点：
- 逐条白名单校验（uncited_evidence / factual_without_evidence / evidence_type_mismatch）
- 校验/审核不通过也保存 rejected 审计记录（不静默丢弃模型输出）
- LLM 不可用抛 InsightUnavailableError（报告确定性层不受影响）
- 同报告版本 + 同输入哈希的 draft 幂等复用
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from app.analysis import insights as insights_module
from app.analysis.attribution import build_dimension_data_gap_evidence, query_attribution_evidence
from app.analysis.contracts import (
    Insight,
    InsightAngle,
    InsightArtifact,
    InsightAudience,
    InsightBrief,
    InsightCriticVerdict,
    InsightStatementType,
    InsightStyle,
    MetricPoint,
    MetricSeries,
    PeriodComparison,
    PeriodSpec,
    PeriodicReport,
    ReportMetric,
)
from app.analysis.insights import (
    InsightUnavailableError,
    build_insight_context,
    generate_insight_artifact,
    validate_insights,
)
from app.analysis.periodic_reports import build_periodic_report
from app.analysis.publishing import ReportStore
from app.core.config import get_settings


def _series() -> MetricSeries:
    period = PeriodSpec(start_date=date(2018, 7, 30), end_date=date(2018, 8, 5))
    return MetricSeries(
        metric_name="gmv", unit="元", period=period,
        points=[MetricPoint(bucket=date(2018, 7, 30), value=100.0)],
        source_refs=["fact_order_items"], definition_version="v1",
        freshness_at=datetime(2018, 8, 5, 23, 59, 59, tzinfo=timezone.utc),
    )


def _report() -> PeriodicReport:
    series = _series()
    metric = ReportMetric(
        metric_name="gmv", value=100.0, unit="元",
        period_over_period=PeriodComparison(
            comparison_type="period_over_period", current_period=series.period,
            baseline_period=series.period, current_value=100.0, baseline_value=120.0,
            delta_value=-20.0, delta_ratio=-16.67, status="ok",
        ),
        evidence_ids=["ev_gmv_series"],
    )
    return PeriodicReport(
        report_id="report_test", report_type="weekly", period=series.period,
        title="经营分析周报", summary="覆盖 1 个核心指标。", metrics=[metric],
        evidence_records={"ev_gmv_series": {"kind": "metric_series", "metric_name": "gmv", "value": 100.0}},
        lineage_refs=["fact_order_items -> gmv"],
        data_freshness_at=datetime(2018, 8, 5, 23, 59, 59, tzinfo=timezone.utc),
    )


def test_build_insight_context_overview_selects_metric_evidence() -> None:
    report = _report()
    selected, kind_map, context = build_insight_context(report, InsightBrief(angle=InsightAngle.OVERVIEW))
    assert "ev_gmv_series" in selected
    assert kind_map["ev_gmv_series"] == "metric_series"
    assert context["metrics"][0]["metric_name"] == "gmv"
    # 可见证据白名单必须落在报告证据注册表内
    assert set(selected).issubset(set(report.evidence_records.keys()))


def test_validate_insights_accepts_grounded_factual() -> None:
    report = _report()
    brief = InsightBrief(min_words=5, max_words=200)
    selected, kind_map, _ = build_insight_context(report, brief)
    insight = Insight(
        title="GMV 环比下降", finding="GMV 环比下降 16.67%，从 120 降至 100。",
        evidence_ids=["ev_gmv_series"], statement_type=InsightStatementType.FACTUAL, confidence=0.8,
    )
    assert validate_insights([insight], brief, selected, kind_map) == []


def test_validate_insights_rejects_uncited_evidence() -> None:
    report = _report()
    brief = InsightBrief(min_words=5, max_words=200)
    selected, kind_map, _ = build_insight_context(report, brief)
    insight = Insight(
        title="渠道归因", finding="GMV 下降源于华东渠道问题。",
        evidence_ids=["ev_unknown"], statement_type=InsightStatementType.FACTUAL, confidence=0.9,
    )
    errors = validate_insights([insight], brief, selected, kind_map)
    assert any("uncited_evidence" in err for err in errors)


def test_validate_insights_requires_evidence_for_factual() -> None:
    report = _report()
    brief = InsightBrief(min_words=5, max_words=200)
    selected, kind_map, _ = build_insight_context(report, brief)
    insight = Insight(
        title="需复核", finding="GMV 环比下降，需要复核。",
        evidence_ids=[], statement_type=InsightStatementType.FACTUAL, confidence=0.8,
    )
    errors = validate_insights([insight], brief, selected, kind_map)
    assert any("factual_without_evidence" in err for err in errors)


def test_validate_insights_treats_word_range_as_soft_guidance() -> None:
    report = _report()
    brief = InsightBrief(min_words=5, max_words=10)
    selected, kind_map, _ = build_insight_context(report, brief)
    insight = Insight(
        title="GMV 趋势",
        finding="这是一条略长于请求范围但仍然有明确证据支持的经营洞察。",
        evidence_ids=["ev_gmv_series"],
        statement_type=InsightStatementType.OBSERVATION,
        confidence=0.8,
    )
    assert validate_insights([insight], brief, selected, kind_map) == []


def test_data_gap_evidence_can_support_only_a_data_limitation_statement() -> None:
    brief = InsightBrief(min_words=5, max_words=200)
    gap_id = "ev_data_gap"
    insight = Insight(
        title="维度数据限制",
        finding="基期没有可用于品类比较的分组数据，因此本次无法完成该维度归因。",
        evidence_ids=[gap_id],
        statement_type=InsightStatementType.OBSERVATION,
        confidence=0.8,
    )

    assert validate_insights(
        [insight], brief, [gap_id], {gap_id: "attribution_data_gap"},
    ) == []
    unsupported_business_claim = Insight(
        title="GMV 变化",
        finding="GMV 本月下降 30%，主要由品类贡献。",
        evidence_ids=[gap_id],
        statement_type=InsightStatementType.FACTUAL,
        confidence=0.8,
    )
    assert any("evidence_type_mismatch" in error for error in validate_insights(
        [unsupported_business_claim], brief, [gap_id], {gap_id: "attribution_data_gap"},
    ))


def test_data_gap_lifecycle_accepts_limitation_and_rejects_business_claim(monkeypatch) -> None:
    report = _report()
    brief = InsightBrief(
        angle=InsightAngle.CONTRIBUTION, min_words=5, max_words=200,
    )
    gap_id, gap = build_dimension_data_gap_evidence(
        report, metric_id="gmv", metric_unit="BRL", dimension="product_category",
        baseline_period=report.period, period_statuses={"comparison": "not_found"},
        trace_id="trace-data-gap-lifecycle",
    )
    monkeypatch.setattr(
        "app.agents.insight_agent.query_attribution_evidence_agent",
        lambda *_args, **_kwargs: ([], [], {gap_id: gap}, "workflow_fallback", None),
    )
    monkeypatch.setattr(insights_module, "critic_review", lambda *_args, **_kwargs: (True, []))

    monkeypatch.setattr(insights_module, "llm_extract", lambda *a, **k: {
        "insights": [{
            "title": "维度数据限制",
            "finding": "基期没有可用于品类比较的分组数据，因此本次无法完成该维度归因。",
            "evidence_ids": [gap_id], "statement_type": "observation", "confidence": 0.8,
        }]
    })
    limitation_artifact = generate_insight_artifact(report, 1, brief, "tester")
    assert limitation_artifact.status == "draft"
    assert gap_id in limitation_artifact.selected_evidence_ids

    monkeypatch.setattr(insights_module, "llm_extract", lambda *a, **k: {
        "insights": [{
            "title": "GMV 变化",
            "finding": "GMV 本月下降 30%，主要由品类贡献。",
            "evidence_ids": [gap_id], "statement_type": "factual", "confidence": 0.8,
        }]
    })
    unsupported_artifact = generate_insight_artifact(report, 1, brief, "tester")
    assert unsupported_artifact.status == "rejected"
    assert any("evidence_type_mismatch" in error for error in unsupported_artifact.validation_errors)


def test_generate_insight_artifact_draft_when_valid(monkeypatch) -> None:
    report = _report()
    monkeypatch.setattr(insights_module, "llm_extract", lambda *a, **k: {
        "insights": [{
            "title": "GMV 环比下降", "finding": "GMV 环比下降 16.67%，从 120 降至 100。",
            "evidence_ids": ["ev_gmv_series"], "statement_type": "factual", "confidence": 0.8,
            "recommendation": "复核大额退货订单。",
        }]
    })
    # Critic 调用复用同一 mock：无 "approved" 键 → fail-closed pending_review。
    artifact = generate_insight_artifact(report, 1, InsightBrief(min_words=5, max_words=200), "tester")
    assert artifact.status == "pending_review"
    assert artifact.critic_verdict is None
    assert artifact.input_snapshot_hash
    assert len(artifact.insights) == 1
    assert artifact.model


def test_generate_insight_artifact_rejected_keeps_audit(monkeypatch) -> None:
    report = _report()
    monkeypatch.setattr(insights_module, "llm_extract", lambda *a, **k: {
        "insights": [{
            "title": "渠道归因", "finding": "GMV 下降源于华东渠道问题。",
            "evidence_ids": ["ev_nonexistent"], "statement_type": "factual", "confidence": 0.9,
        }]
    })
    artifact = generate_insight_artifact(report, 1, InsightBrief(min_words=5, max_words=200), "tester")
    assert artifact.status == "rejected"
    assert artifact.insights  # 模型输出仍被保留，作为审计证据
    assert any("uncited_evidence" in err for err in artifact.validation_errors)


def test_generate_insight_artifact_unavailable_when_llm_down(monkeypatch) -> None:
    report = _report()
    monkeypatch.setattr(insights_module, "llm_extract", lambda *a, **k: None)
    try:
        generate_insight_artifact(report, 1, InsightBrief(), "tester")
    except InsightUnavailableError:
        return
    raise AssertionError("expected InsightUnavailableError")


def test_sqlite_store_insight_roundtrip_and_dedup(tmp_path) -> None:
    store = ReportStore(str(tmp_path / "phase6.sqlite"))
    artifact = InsightArtifact(
        report_id="report_x", report_version=1, insight_version=0, status="draft",
        brief=InsightBrief(angle=InsightAngle.OVERVIEW, min_words=5),
        selected_evidence_ids=["ev1"], input_snapshot_hash="abc",
        insights=[Insight(title="t", finding="GMV 环比下降 16.67%。", evidence_ids=["ev1"],
                          statement_type=InsightStatementType.FACTUAL, confidence=0.8)],
        model="mimo-v2.5", prompt_version="insights-v1", actor="tester",
    )
    row = store.create_insight(artifact)
    assert row["insight_version"] == 1
    assert row["status"] == "draft"
    assert isinstance(row["insights_json"], list)
    # 同报告版本 + 同哈希 + draft → 幂等复用，不新增版本
    assert store.create_insight(artifact)["insight_version"] == 1
    assert len(store.list_insights("report_x", 1)) == 1
    # 不同输入哈希 → 新 insight_version
    artifact2 = artifact.model_copy(update={
        "input_snapshot_hash": "def",
        "brief": InsightBrief(angle=InsightAngle.ANOMALY, min_words=5),
    })
    assert store.create_insight(artifact2)["insight_version"] == 2
    assert len(store.list_insights("report_x", 1)) == 2
    assert store.get_insight("report_x", 1, 2)["status"] == "draft"


def _insight_artifact(report_id: str, report_version: int, hash_: str, critic: InsightCriticVerdict) -> InsightArtifact:
    return InsightArtifact(
        report_id=report_id, report_version=report_version, insight_version=0,
        status="draft" if critic == InsightCriticVerdict.APPROVED else "rejected",
        brief=InsightBrief(angle=InsightAngle.OVERVIEW, min_words=5),
        selected_evidence_ids=["ev1"], input_snapshot_hash=hash_,
        insights=[Insight(title="t", finding="GMV 环比下降 16.67%。", evidence_ids=["ev1"],
                          statement_type=InsightStatementType.FACTUAL, confidence=0.8)],
        model="mimo-v2.5", prompt_version="insights-v1", critic_verdict=critic, actor="tester",
    )


def test_sqlite_publish_pins_latest_approved_insight(tmp_path) -> None:
    store = ReportStore(str(tmp_path / "pin.sqlite"))
    report = _report()
    store.create_draft(report, "content", "system")
    store.review(report.report_id, 1, "approve", "reviewer", "ok")
    approved_version = store.create_insight(
        _insight_artifact(report.report_id, 1, "aaa", InsightCriticVerdict.APPROVED)
    )["insight_version"]
    # 人工审核通过后才可被 pin
    store.review_insight(report.report_id, 1, approved_version, "approve", "reviewer", "verified")
    # 更新的 rejected 洞察不应被 pin
    store.create_insight(_insight_artifact(report.report_id, 1, "bbb", InsightCriticVerdict.REJECTED))
    published = store.publish(report.report_id, 1, "publisher")
    assert published["status"] == "published"
    assert published["pinned_insight_version"] == approved_version
    # 状态机闭环：被钉选的洞察从 approved → published（同事务）
    assert store.get_insight(report.report_id, 1, approved_version)["status"] == "published"


def test_sqlite_publish_without_approved_insight_pins_null(tmp_path) -> None:
    store = ReportStore(str(tmp_path / "pin-null.sqlite"))
    report = _report()
    store.create_draft(report, "content", "system")
    store.review(report.report_id, 1, "approve", "reviewer", "ok")
    published = store.publish(report.report_id, 1, "publisher")
    assert published["status"] == "published"
    assert published["pinned_insight_version"] is None


def test_published_report_pins_insight_reads_pinned_blocks_generation(tmp_path, monkeypatch) -> None:
    """端到端：发布钉死 → 已发布只读 pinned → 禁止对 published 生成。"""
    from fastapi.testclient import TestClient

    from app.main import app

    period = PeriodSpec(start_date=date(2025, 1, 6), end_date=date(2025, 1, 12), granularity="week")
    fixture_report = build_periodic_report(period, ["gmv"], fetcher=lambda metric, day: {"value": 100.0, "source_table": "fixture.metrics"}, include_charts=False)
    monkeypatch.setattr("app.api.routes.analysis.build_periodic_report", lambda *args, **kwargs: fixture_report)
    settings = get_settings()
    monkeypatch.setattr(settings, "phase6_store_path", str(tmp_path / "api.sqlite"))
    client = TestClient(app)
    created = client.post("/analysis/reports", json={
        "start_date": "2025-01-06", "end_date": "2025-01-12",
        "granularity": "week", "metric_names": ["gmv"], "include_charts": False,
    })
    assert created.status_code == 200
    payload = created.json()
    report_id, version = payload["report"]["report_id"], payload["version"]
    store = ReportStore(str(tmp_path / "api.sqlite"))
    approved_version = store.create_insight(
        _insight_artifact(report_id, version, "aaa", InsightCriticVerdict.APPROVED)
    )["insight_version"]
    # 人工审核通过后才可被 pin
    store.review_insight(report_id, version, approved_version, "approve", "reviewer", "ok")
    assert client.post(f"/analysis/reports/{report_id}/review", json={
        "version": version, "action": "approve", "actor": "reviewer", "reason": "ok",
    }).status_code == 200
    published = client.post(f"/analysis/reports/{report_id}/publish", json={"version": version, "actor": "publisher"})
    assert published.status_code == 200
    assert published.json()["pinned_insight_version"] == approved_version
    # 发布后禁止直接生成洞察
    blocked = client.post(f"/analysis/reports/{report_id}/versions/{version}/insights", json={"angle": "overview"})
    assert blocked.status_code == 409
    # 已发布报告读端只返回钉死版本
    items = client.get(f"/analysis/reports/{report_id}/versions/{version}/insights").json()
    assert len(items) == 1
    assert items[0]["insight_version"] == approved_version


def _mock_llm(gen_result, critic_result):
    """区分生成与 Critic 两次调用的 mock：Critic prompt 含'每条洞察引用的证据内容'。"""
    def fake(system: str, user: str, **kwargs):
        if "每条洞察引用的证据内容" in user:
            return critic_result
        return gen_result
    return fake


def _multiplicative_report() -> PeriodicReport:
    period = PeriodSpec(start_date=date(2018, 7, 30), end_date=date(2018, 8, 5))

    def metric(name: str, current: float, baseline: float) -> ReportMetric:
        return ReportMetric(
            metric_name=name, value=current, unit="单" if name == "order_count" else "元",
            period_over_period=PeriodComparison(
                comparison_type="period_over_period", current_period=period, baseline_period=period,
                current_value=current, baseline_value=baseline, delta_value=round(current - baseline, 2),
                delta_ratio=round((current - baseline) * 100 / baseline, 2) if baseline else None, status="ok",
            ),
            evidence_ids=[f"ev_{name}_series"],
        )

    metrics = [metric("gmv", 100.0, 120.0), metric("order_count", 20.0, 25.0), metric("avg_order_value", 5.0, 4.8)]
    return PeriodicReport(
        report_id="report_decomp", report_type="weekly", period=period,
        title="经营分析周报", summary="summary", metrics=metrics,
        evidence_records={
            name: {"kind": "metric_series", "metric_name": name, "value": item.value}
            for name, item in zip(("gmv", "order_count", "avg_order_value"), metrics)
        },
        lineage_refs=["fact -> gmv"], data_freshness_at=datetime(2018, 8, 5, 23, 59, 59, tzinfo=timezone.utc),
    )


def test_contribution_angle_runs_attribution_and_drafts(monkeypatch) -> None:
    """E2E 场景 1：变化归因角度触发受控补查，LLM 引用补查证据，Critic 放行 → draft。"""
    report = _multiplicative_report()
    # The report fixture has no pinned breakdown snapshots; insight attribution
    # must preserve explicit data gaps and avoid querying live source tables.
    # Phase 8 的证据 Agent 默认启用；本测试针对确定性管线，让 Agent 退回固定选材（隔离真实 provider）。
    monkeypatch.setattr("app.llm.client.llm_agent_action", lambda *a, **k: None)
    brief = InsightBrief(angle=InsightAngle.CONTRIBUTION, min_words=5, max_words=200)
    _, decompositions, attribution_evidence = query_attribution_evidence(report, brief)
    assert decompositions  # 乘法分解可由事实快照计算，无需 DB
    decomp_eid = decompositions[0].evidence_ids[0]
    monkeypatch.setattr(insights_module, "llm_extract", _mock_llm(
        {"insights": [{
            "title": "GMV 下降拆解", "finding": "GMV 下降主要来自订单量下滑，客单价小幅抬升部分对冲。",
            "evidence_ids": [decomp_eid], "statement_type": "factual", "confidence": 0.8,
        }]},
        {"approved": True, "issues": []},
    ))
    artifact = generate_insight_artifact(report, 1, brief, "tester")
    assert artifact.status == "draft"
    assert artifact.critic_verdict == InsightCriticVerdict.APPROVED
    assert artifact.attribution_evidence  # 补查证据已写入洞察输入快照
    assert decomp_eid in artifact.selected_evidence_ids
    assert artifact.insights[0].evidence_ids == [decomp_eid]
    data_gaps = [
        (evidence_id, item) for evidence_id, item in artifact.attribution_evidence.items()
        if item.get("kind") == "data_gap"
    ]
    assert data_gaps
    gap_id, data_gap = data_gaps[0]
    assert gap_id in artifact.selected_evidence_ids
    typed_gap = data_gap["investigation_evidence"]
    assert typed_gap["evidence_type"] == "attribution_data_gap"
    assert typed_gap["trace_id"]
    assert typed_gap["scope_fingerprint"]
    assert typed_gap["data_quality"]["status"] == "snapshot_unavailable"
    assert artifact.investigation_summary
    assert any(
        item["failure_code"] == "DATA_INSUFFICIENT"
        for item in artifact.investigation_summary["action_results"]
    )


def test_fake_evidence_id_rejected_server_side(monkeypatch) -> None:
    """E2E 场景 2：模型引用伪造 evidence_id → 服务端白名单 rejected。"""
    report = _report()
    monkeypatch.setattr(insights_module, "llm_extract", _mock_llm(
        {"insights": [{
            "title": "伪造证据", "finding": "GMV 下降源于华东渠道问题。",
            "evidence_ids": ["ev_fake_not_exist"], "statement_type": "factual", "confidence": 0.9,
        }]},
        {"approved": True, "issues": []},
    ))
    artifact = generate_insight_artifact(report, 1, InsightBrief(min_words=5, max_words=200), "tester")
    assert artifact.status == "rejected"
    assert any("uncited_evidence" in err for err in artifact.validation_errors)


def test_causal_claim_without_evidence_critic_rejected(monkeypatch) -> None:
    """Deterministic validation rejects causal claims even if an LLM critic would approve."""
    report = _report()
    critic_calls: list[bool] = []

    def fake_llm(system: str, user: str, **kwargs):
        if "每条洞察引用的证据内容" in user:
            critic_calls.append(True)
            return {"approved": True, "issues": []}
        return {"insights": [{
            "title": "因果断言", "finding": "华东渠道问题导致 GMV 下降 16.67%。",
            "evidence_ids": ["ev_gmv_series"], "statement_type": "factual", "confidence": 0.9,
        }]}

    monkeypatch.setattr(insights_module, "llm_extract", fake_llm)
    artifact = generate_insight_artifact(report, 1, InsightBrief(min_words=5, max_words=200), "tester")
    assert artifact.status == "rejected"
    assert artifact.critic_verdict is None
    assert artifact.validation_errors == ["insight[0].unsupported_causal_assertion"]
    assert critic_calls == []


def test_critic_unavailable_pending_review_and_report_still_publishable(monkeypatch, tmp_path) -> None:
    """E2E 场景 4：Critic 不可用 → 洞察 pending_review（不视为通过）；确定性报告仍可发布，pin null。"""
    report = _report()
    monkeypatch.setattr(insights_module, "llm_extract", _mock_llm(
        {"insights": [{
            "title": "GMV 下降", "finding": "GMV 环比下降 16.67%，从 120 降至 100。",
            "evidence_ids": ["ev_gmv_series"], "statement_type": "factual", "confidence": 0.8,
        }]},
        None,  # Critic 不可用 → fail-closed
    ))
    artifact = generate_insight_artifact(report, 1, InsightBrief(min_words=5, max_words=200), "tester")
    assert artifact.status == "pending_review"
    assert artifact.critic_verdict is None
    store = ReportStore(str(tmp_path / "critic-down.sqlite"))
    store.create_draft(report, "content", "system")
    store.review(report.report_id, 1, "approve", "reviewer", "ok")
    published = store.publish(report.report_id, 1, "publisher")
    assert published["status"] == "published"
    assert published["pinned_insight_version"] is None


def test_store_generating_finalize_flow(tmp_path) -> None:
    store = ReportStore(str(tmp_path / "async.sqlite"))
    brief = InsightBrief(angle=InsightAngle.OVERVIEW, min_words=5)
    placeholder = store.create_generating("report_a", 1, brief, "tester")
    insight_version = int(placeholder["insight_version"])
    assert placeholder["status"] == "generating"
    assert placeholder["insights_json"] == []
    # 双击防重：同 (report_id, rv, brief) 复用占位行，不起新任务
    assert int(store.create_generating("report_a", 1, brief, "tester")["insight_version"]) == insight_version
    artifact = InsightArtifact(
        report_id="report_a", report_version=1, insight_version=insight_version, status="draft",
        brief=brief, selected_evidence_ids=["ev1"], input_snapshot_hash="hash",
        insights=[Insight(title="t", finding="GMV 环比下降。", evidence_ids=["ev1"],
                          statement_type=InsightStatementType.FACTUAL, confidence=0.8)],
        model="m", prompt_version="insights-v1",
        investigation_summary={
            "metric_id": "gmv", "scope_fingerprint": "scope-1",
            "stop_reason": {"code": "ENOUGH_EVIDENCE"}, "actions_used": 2,
            "remaining_budget": {"queries": 1}, "branches": [],
            "action_results": [], "policy_rejections": [],
        },
        duration_ms=1234, input_tokens=100, output_tokens=50, estimated_cost=0.001, actor="tester",
    )
    row = store.finalize_insight("report_a", 1, insight_version, artifact)
    assert row["status"] == "draft"
    assert row["duration_ms"] == 1234
    assert row["input_tokens"] == 100
    assert row["output_tokens"] == 50
    assert row["investigation_summary"]["stop_reason"]["code"] == "ENOUGH_EVIDENCE"
    assert row["investigation_summary"]["actions_used"] == 2
    # 已非 generating 的洞察不能再 finalize
    try:
        store.finalize_insight("report_a", 1, insight_version, artifact)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_store_sync_insight_persists_investigation_summary_and_migrates_old_schema(tmp_path) -> None:
    db_path = tmp_path / "investigation-summary.sqlite"
    store = ReportStore(str(db_path))
    summary = {
        "metric_id": "gmv", "scope_fingerprint": "scope-1",
        "stop_reason": {"code": "CAPABILITY_GAP"}, "actions_used": 1,
        "remaining_budget": {"queries": 0}, "branches": [],
        "action_results": [], "policy_rejections": [],
    }
    artifact = _insight_artifact("report-summary", 1, "summary-hash", InsightCriticVerdict.APPROVED)
    artifact = artifact.model_copy(update={"investigation_summary": summary})
    row = store.create_insight(artifact)
    assert row["investigation_summary"] == summary

    # Simulate a database created before this field existed, then run the normal migration path.
    with store._connection() as connection:
        connection.execute("ALTER TABLE report_insights DROP COLUMN investigation_summary")
    migrated = ReportStore(str(db_path))
    with migrated._connection() as connection:
        columns = {record[1] for record in connection.execute("PRAGMA table_info(report_insights)")}
    assert "investigation_summary" in columns
    assert migrated.get_insight("report-summary", 1, int(row["insight_version"]))["investigation_summary"] is None


def test_store_generating_fail_on_worker_error(tmp_path) -> None:
    store = ReportStore(str(tmp_path / "async2.sqlite"))
    brief = InsightBrief(angle=InsightAngle.OVERVIEW, min_words=5)
    placeholder = store.create_generating("report_b", 1, brief, "tester")
    insight_version = int(placeholder["insight_version"])
    row = store.fail_insight("report_b", 1, insight_version, "worker_error:ConnectionError")
    assert row["status"] == "rejected"
    assert any("worker_error" in str(item) for item in row["validation_errors"])


def test_background_task_finalizes_placeholder(monkeypatch, tmp_path) -> None:
    """后台任务入口：generating 占位 → 生成 → finalize（含观测字段）。"""
    store = ReportStore(str(tmp_path / "async3.sqlite"))
    report = _report()
    store.create_draft(report, "content", "system")
    brief = InsightBrief(angle=InsightAngle.OVERVIEW, min_words=5, max_words=200)
    placeholder = store.create_generating(report.report_id, 1, brief, "tester")
    insight_version = int(placeholder["insight_version"])
    monkeypatch.setattr(insights_module, "llm_extract", _mock_llm(
        {"insights": [{
            "title": "t", "finding": "GMV 环比下降 16.67%，从 120 降至 100。",
            "evidence_ids": ["ev_gmv_series"], "statement_type": "factual", "confidence": 0.8,
        }]},
        {"approved": True, "issues": []},
    ))
    insights_module._background_insight_task(
        store, report.report_id, 1, insight_version, brief, "tester", {"tenant_id": "", "user_id": ""},
    )
    row = store.get_insight(report.report_id, 1, insight_version)
    assert row["status"] == "draft"
    assert isinstance(row.get("duration_ms"), int)
    assert row["duration_ms"] >= 0
    # 生成阶段进度已写入 DB（SSE 轮询数据源）
    progress = row.get("progress") or {}
    assert progress.get("stage") in {"attribution", "llm_generation", "critic"}


def test_store_update_insight_progress(tmp_path) -> None:
    store = ReportStore(str(tmp_path / "progress.sqlite"))
    brief = InsightBrief(angle=InsightAngle.OVERVIEW, min_words=5)
    placeholder = store.create_generating("report_p", 1, brief, "tester")
    insight_version = int(placeholder["insight_version"])
    store.update_insight_progress("report_p", 1, insight_version, "llm_generation", "调用 LLM 生成结构化洞察")
    row = store.get_insight("report_p", 1, insight_version)
    assert row["progress"]["stage"] == "llm_generation"
    # 已终态的洞察不写进度（WHERE status='generating'）
    artifact = InsightArtifact(
        report_id="report_p", report_version=1, insight_version=insight_version, status="rejected",
        brief=brief, selected_evidence_ids=[], input_snapshot_hash="h",
        insights=[], model="m", prompt_version="insights-v1", actor="tester",
    )
    store.finalize_insight("report_p", 1, insight_version, artifact)
    store.update_insight_progress("report_p", 1, insight_version, "critic", "不应写入")
    assert store.get_insight("report_p", 1, insight_version)["progress"]["stage"] == "llm_generation"


def test_insight_events_sse_terminal_and_404(tmp_path, monkeypatch) -> None:
    """SSE 端点：终态洞察立即推送 completed；不存在的洞察 404。"""
    from fastapi.testclient import TestClient

    from app.main import app

    period = PeriodSpec(start_date=date(2025, 1, 6), end_date=date(2025, 1, 12), granularity="week")
    fixture_report = build_periodic_report(period, ["gmv"], fetcher=lambda metric, day: {"value": 100.0, "source_table": "fixture.metrics"}, include_charts=False)
    monkeypatch.setattr("app.api.routes.analysis.build_periodic_report", lambda *args, **kwargs: fixture_report)
    settings = get_settings()
    monkeypatch.setattr(settings, "phase6_store_path", str(tmp_path / "sse.sqlite"))
    monkeypatch.setattr(insights_module, "llm_extract", _mock_llm(
        {"insights": [{
            "title": "建议", "finding": "建议复核数据采集流程，确保 GMV 指标口径一致后再发布结论。",
            "evidence_ids": [], "statement_type": "recommendation", "confidence": 0.6,
        }]},
        {"approved": True, "issues": []},
    ))
    client = TestClient(app)
    created = client.post("/analysis/reports", json={
        "start_date": "2025-01-06", "end_date": "2025-01-12",
        "granularity": "week", "metric_names": ["gmv"], "include_charts": False,
    })
    payload = created.json()
    report_id, version = payload["report"]["report_id"], payload["version"]
    resp = client.post(f"/analysis/reports/{report_id}/versions/{version}/insights?sync=true", json={"angle": "overview", "min_words": 5})
    assert resp.status_code == 200
    insight_version = int(resp.json()["insight_version"])
    # 终态洞察：SSE 立即推送 completed 并关闭
    events = client.get(f"/analysis/reports/{report_id}/versions/{version}/insights/{insight_version}/events")
    assert events.status_code == 200
    assert events.headers["content-type"].startswith("text/event-stream")
    assert '"event": "completed"' in events.text
    # 不存在的洞察 → 404
    missing = client.get(f"/analysis/reports/{report_id}/versions/{version}/insights/999/events")
    assert missing.status_code == 404


def test_insights_sync_route_returns_terminal(tmp_path, monkeypatch) -> None:
    """sync=true 参数：POST 同步返回终态洞察（测试/调试路径）。"""
    from fastapi.testclient import TestClient

    from app.main import app

    period = PeriodSpec(start_date=date(2025, 1, 6), end_date=date(2025, 1, 12), granularity="week")
    fixture_report = build_periodic_report(period, ["gmv"], fetcher=lambda metric, day: {"value": 100.0, "source_table": "fixture.metrics"}, include_charts=False)
    monkeypatch.setattr("app.api.routes.analysis.build_periodic_report", lambda *args, **kwargs: fixture_report)
    settings = get_settings()
    monkeypatch.setattr(settings, "phase6_store_path", str(tmp_path / "sync.sqlite"))
    monkeypatch.setattr(insights_module, "llm_extract", _mock_llm(
        {"insights": [{
            "title": "建议", "finding": "建议复核数据采集流程，确保 GMV 指标口径一致后再发布结论。",
            "evidence_ids": [], "statement_type": "recommendation", "confidence": 0.6,
        }]},
        {"approved": True, "issues": []},
    ))
    client = TestClient(app)
    created = client.post("/analysis/reports", json={
        "start_date": "2025-01-06", "end_date": "2025-01-12",
        "granularity": "week", "metric_names": ["gmv"], "include_charts": False,
    })
    payload = created.json()
    report_id, version = payload["report"]["report_id"], payload["version"]
    resp = client.post(f"/analysis/reports/{report_id}/versions/{version}/insights?sync=true", json={"angle": "overview", "min_words": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"draft", "pending_review", "rejected"}
    assert isinstance(body.get("insight_version"), int)


def test_brief_audience_style_affect_snapshot_hash() -> None:
    """P3：受众/风格进入输入快照哈希——不同约束 = 不同输入，幂等语义保持一致。"""
    report = _report()
    brief_a = InsightBrief(angle=InsightAngle.OVERVIEW, audience=InsightAudience.EXECUTIVE, style=InsightStyle.FORMAL, min_words=5)
    brief_b = InsightBrief(angle=InsightAngle.OVERVIEW, audience=InsightAudience.OPERATIONS, style=InsightStyle.CONCISE, min_words=5)
    selected, _, _ = build_insight_context(report, brief_a)
    assert insights_module.input_snapshot_hash(report.report_id, 1, brief_a, selected) != insights_module.input_snapshot_hash(report.report_id, 1, brief_b, selected)


def test_investigation_summary_affects_insight_idempotency_hash() -> None:
    report = _report()
    brief = InsightBrief(angle=InsightAngle.CONTRIBUTION, min_words=5)
    selected, _, _ = build_insight_context(report, brief)
    enough = {"stop_reason": {"code": "ENOUGH_EVIDENCE"}, "actions_used": 2}
    capability_limited = {"stop_reason": {"code": "CAPABILITY_GAP"}, "actions_used": 2}
    assert insights_module.input_snapshot_hash(
        report.report_id, 1, brief, selected, enough,
    ) != insights_module.input_snapshot_hash(
        report.report_id, 1, brief, selected, capability_limited,
    )


def test_generation_prompt_includes_audience_and_style() -> None:
    """P3：受众/风格指令进入生成提示词。"""
    report = _report()
    brief = InsightBrief(angle=InsightAngle.OVERVIEW, audience=InsightAudience.EXECUTIVE, style=InsightStyle.FORMAL, min_words=5)
    selected, kind_map, context = build_insight_context(report, brief)
    prompt = insights_module._generation_prompt(brief, selected, kind_map, context)
    assert "受众为管理层" in prompt
    assert "风格正式" in prompt


def test_generate_with_audience_style_still_drafts(monkeypatch) -> None:
    """P3：带受众/风格的 Brief 走完整生成仍可通过。"""
    report = _report()
    monkeypatch.setattr(insights_module, "llm_extract", _mock_llm(
        {"insights": [{
            "title": "t", "finding": "GMV 环比下降 16.67%，从 120 降至 100。",
            "evidence_ids": ["ev_gmv_series"], "statement_type": "factual", "confidence": 0.8,
        }]},
        {"approved": True, "issues": []},
    ))
    brief = InsightBrief(angle=InsightAngle.OVERVIEW, audience=InsightAudience.EXECUTIVE, style=InsightStyle.FORMAL, min_words=5, max_words=200)
    artifact = generate_insight_artifact(report, 1, brief, "tester")
    assert artifact.status == "draft"
    assert artifact.brief.audience == InsightAudience.EXECUTIVE
    assert artifact.brief.style == InsightStyle.FORMAL

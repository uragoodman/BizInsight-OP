"""Phase 6 semantic, report, approval, and outbox APIs."""

import json
import time
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse, StreamingResponse
from sqlalchemy import text

from app.analysis.contracts import Insight, InsightAngle, InsightAudience, InsightBrief, InsightStyle, PeriodicReport, PeriodSpec
from app.analysis.insights import InsightUnavailableError, generate_insight_artifact, submit_insight_task
from app.analysis.periodic_reports import PLATFORM_REPORT_SUPPORTED_METRICS, build_periodic_report, report_to_markdown
from app.analysis.seller_periodic_reports import SELLER_REPORT_SUPPORTED_METRICS, build_seller_scoped_periodic_report
from app.analysis.publishing import ReportStore, report_risks
from app.analysis.semantic import get_metric_definition, list_metric_definitions
from app.analysis.targets import MetricTarget, TargetStore
from app.core.config import get_settings
from app.schemas.analysis import InsightBriefRequest, InsightReviewRequest, MetricTargetRequest, PeriodicReportRequest, ReportPublishRequest, ReportReviewRequest
from app.api.auth import require_actor_match, require_analysis_write, require_tenant_context
from app.tenancy import TenantContext, activate_tenant_context
from app.tenancy.report_store import TenantReportStore
from app.tenancy.target_store import PlatformTargetStore, TenantTargetStore
from app.data.repositories.explorer import get_accessible_seller
from app.data.database import engine
from app.iam.service import require_permission


router = APIRouter(prefix="/analysis", tags=["analysis"])


def _store(context: TenantContext):
    # Seller reports/outbox must never fall back to local SQLite: that would
    # make horizontally deployed workers share no authorization boundary.
    return TenantReportStore(context) if context.scope_type == "seller" else ReportStore(get_settings().phase6_store_path)


def _target_store(context: TenantContext, seller_id: str | None = None):
    if context.scope_type == "seller":
        return TenantTargetStore(context)
    if context.is_platform_admin and seller_id:
        # 平台管理员配置的店铺目标与卖家租户共用同一张 RLS 表；写入、列表、
        # 删除必须落在同一存储，否则管理员写入的行在列表和删除里永远看不到。
        return PlatformTargetStore(context)
    return TargetStore(get_settings().phase6_store_path)


def _report_payload(stored: dict[str, object]) -> dict[str, object]:
    payload = stored.get("report_json", {})
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("Stored report has an invalid report payload")
    return payload


def _effective_report_scope(payload: PeriodicReportRequest, context: TenantContext) -> tuple[str, str | None, str | None]:
    """Resolve report scope from trusted identity, never from client claims alone."""
    if context.scope_type == "seller":
        seller_id = payload.seller_id or context.seller_id
        if not seller_id:
            raise ValueError("Select one of your authorized sellers")
        # Do not rely on the repository lookup as the only authorization
        # guard.  A seller tenant may now have several grants, but a caller
        # still may select only an ID carried by its trusted context.
        if str(seller_id) not in context.allowed_seller_ids:
            raise ValueError("You cannot request another seller's report scope")
        seller = get_accessible_seller(context, str(seller_id))
        return "seller", str(seller_id), str(seller["label"])
    if payload.scope_type == "platform":
        return "platform", None, None
    seller = get_accessible_seller(context, str(payload.seller_id))
    return "seller", str(seller["seller_id"]), str(seller["label"])


@router.get("/reports")
def list_reports(limit: int = 50, tenant_context: TenantContext = Depends(require_tenant_context)) -> list[dict[str, object]]:
    return _store(tenant_context).list(max(1, min(limit, 100)))


@router.get("/reports/capabilities")
def report_capabilities(tenant_context: TenantContext = Depends(require_tenant_context)) -> dict[str, object]:
    if tenant_context.scope_type == "seller":
        if len(tenant_context.allowed_seller_ids) == 1:
            seller = get_accessible_seller(tenant_context, tenant_context.allowed_seller_ids[0])
            return {"allowed_scopes": ["seller"], "locked_seller": seller}
        return {"allowed_scopes": ["seller"], "locked_seller": None}
    return {"allowed_scopes": ["platform", "seller"], "locked_seller": None}


@router.get("/semantic/metrics")
def semantic_metrics(scope: str | None = Query(default=None, pattern="^(platform|seller)$"), tenant_context: TenantContext = Depends(require_tenant_context)) -> list[dict[str, object]]:
    effective_scope = "seller" if tenant_context.scope_type == "seller" else scope or "platform"
    platform_items = {item["metric_name"]: item for item in list_metric_definitions("platform")}
    seller_items = {item["metric_name"]: item for item in list_metric_definitions("seller")}
    names = set(platform_items) | set(seller_items)
    result = []
    for metric_name in sorted(names):
        report_scopes = []
        explorer_scopes = []
        surfaces = set()
        for scope_name, items in (("platform", platform_items), ("seller", seller_items)):
            item = items.get(metric_name)
            if not item:
                continue
            surfaces.update(item.get("surfaces", []))
            if {"value", "trend"}.issubset(set(item.get("capabilities", []))) and "explorer" in item.get("surfaces", []):
                explorer_scopes.append(scope_name)
            if "report" in item.get("surfaces", []):
                if scope_name == "platform" and metric_name in PLATFORM_REPORT_SUPPORTED_METRICS:
                    report_scopes.append(scope_name)
                if scope_name == "seller" and metric_name in SELLER_REPORT_SUPPORTED_METRICS:
                    report_scopes.append(scope_name)
        if not report_scopes and not explorer_scopes:
            continue
        item = dict(platform_items.get(metric_name) or seller_items[metric_name])
        item["surfaces"] = sorted(surfaces)
        item["report_available_scopes"] = report_scopes
        item["explorer_available_scopes"] = explorer_scopes
        # `available` remains report-specific for backwards compatibility with
        # the report page; Explorer uses its surface and explorer scopes.
        item["available_scopes"] = report_scopes
        item["available"] = effective_scope in report_scopes
        item["report_available"] = effective_scope in report_scopes
        item["explorer_available"] = effective_scope in explorer_scopes
        result.append(item)
    return result


@router.get("/semantic/metrics/{metric_name}")
def semantic_metric(metric_name: str, tenant_context: TenantContext = Depends(require_tenant_context)) -> dict[str, object]:
    try:
        definition = get_metric_definition(metric_name, tenant_context.scope_type)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return definition.__dict__ | {"lineage_ref": definition.lineage_ref}


@router.put("/targets/{metric_name}")
def put_metric_target(metric_name: str, payload: MetricTargetRequest, actor: str = Depends(require_analysis_write), tenant_context: TenantContext = Depends(require_tenant_context)) -> dict[str, object]:
    try:
        require_actor_match(actor, payload.created_by)
        target_data = payload.model_dump()
        if tenant_context.scope_type == "seller" or target_data.get("seller_id"):
            # 店铺目标（卖家租户自配或平台管理员代配）都落 RLS 表的 tenant 作用域；
            # 全局目标保持 legacy global 作用域供平台报告比较。
            target_data["scope"] = "tenant"
        target = _target_store(tenant_context, target_data.get("seller_id")).put(MetricTarget(metric_name=metric_name, **target_data))
        return {
            "metric_name": target.metric_name, "period_start": target.period_start,
            "period_end": target.period_end, "target_value": target.target_value,
            "version": target.version, "effective_at": target.effective_at,
            "created_by": target.created_by, "scope": target.scope, "seller_id": target.seller_id,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/targets")
def list_metric_targets(metric_name: str | None = None, seller_id: str | None = None, version: str | None = None, period_start: str | None = None, period_end: str | None = None, offset: int = Query(default=0, ge=0), limit: int = Query(default=20, ge=1, le=100), tenant_context: TenantContext = Depends(require_tenant_context)) -> dict[str, object]:
    try:
        from datetime import date
        if tenant_context.is_platform_admin:
            filters = ["TRUE"]; params: dict[str, object] = {"offset": offset, "limit": limit}
            if seller_id: filters.append("seller_id=:seller_id"); params["seller_id"] = seller_id
            if metric_name: filters.append("metric_name=:metric_name"); params["metric_name"] = metric_name
            if version: filters.append("version=:version"); params["version"] = version
            if period_start: filters.append("period_start>=:period_start"); params["period_start"] = date.fromisoformat(period_start)
            if period_end: filters.append("period_end<=:period_end"); params["period_end"] = date.fromisoformat(period_end)
            where = " AND ".join(filters)
            with engine.begin() as conn:
                conn.execute(text("SELECT set_config('app.scope_type','platform',true)"))
                rows = conn.execute(text(f"SELECT tenant_id,seller_id,metric_name,period_start,period_end,target_value,version,effective_at,created_by FROM analytics.tenant_metric_targets WHERE {where} ORDER BY period_start DESC,effective_at DESC,version DESC LIMIT :limit OFFSET :offset"), params).mappings().all()
                total = conn.execute(text(f"SELECT count(*) FROM analytics.tenant_metric_targets WHERE {where}"), params).scalar_one()
            items = [dict(row) | {"scope": "tenant"} for row in rows]
            return {"items": items, "total": int(total), "next_offset": offset + limit if offset + limit < total else None}
        items, total = _target_store(tenant_context).list(metric_name=metric_name, seller_id=seller_id, version=version, period_start=date.fromisoformat(period_start) if period_start else None, period_end=date.fromisoformat(period_end) if period_end else None, offset=offset, limit=limit)
        return {"items": [item.__dict__ for item in items], "total": total, "next_offset": offset + limit if offset + limit < total else None}
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.delete("/targets/{metric_name}")
def delete_metric_target(metric_name: str, period_start: str, period_end: str, version: str, seller_id: str | None = None, actor: str = Depends(require_analysis_write), tenant_context: TenantContext = Depends(require_tenant_context)) -> dict[str, object]:
    try:
        from datetime import date
        if tenant_context.is_platform_admin and seller_id:
            with engine.begin() as conn:
                conn.execute(text("SELECT set_config('app.scope_type','platform',true)"))
                result = conn.execute(text("DELETE FROM analytics.tenant_metric_targets WHERE seller_id=:seller AND metric_name=:metric AND period_start=:start AND period_end=:end AND version=:version"), {"seller": seller_id, "metric": metric_name, "start": date.fromisoformat(period_start), "end": date.fromisoformat(period_end), "version": version})
            deleted = result.rowcount > 0
        elif tenant_context.scope_type == "seller":
            deleted = _target_store(tenant_context).delete(seller_id=seller_id or tenant_context.seller_id or "", metric_name=metric_name, period_start=date.fromisoformat(period_start), period_end=date.fromisoformat(period_end), version=version)
        else:
            deleted = _target_store(tenant_context).delete(metric_name=metric_name, period_start=date.fromisoformat(period_start), period_end=date.fromisoformat(period_end), version=version, seller_id=seller_id)
        return {"deleted": deleted}
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/targets/{metric_name}")
def get_metric_target(metric_name: str, start_date: str, end_date: str, scope: str = "global", seller_id: str | None = None, tenant_context: TenantContext = Depends(require_tenant_context)) -> dict[str, object]:
    try:
        period = PeriodSpec.model_validate({"start_date": start_date, "end_date": end_date})
        store = _target_store(tenant_context, seller_id)
        target = store.get(metric_name, period, seller_id) if isinstance(store, (PlatformTargetStore, TenantTargetStore)) else store.get(metric_name, period, scope)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if target is None:
        raise HTTPException(status_code=404, detail="Metric target not found")
    return target.__dict__


@router.post("/reports")
def create_periodic_report(payload: PeriodicReportRequest, actor: str = Depends(require_analysis_write), tenant_context: TenantContext = Depends(require_tenant_context)) -> dict[str, object]:
    try:
        require_permission(tenant_context, "report_export")
        period = PeriodSpec(start_date=payload.start_date, end_date=payload.end_date, granularity=payload.granularity)
        scope_type, seller_id, seller_label = _effective_report_scope(payload, tenant_context)
        if scope_type == "seller":
            unsupported = [name for name in payload.metric_names if name not in SELLER_REPORT_SUPPORTED_METRICS]
            if unsupported:
                supported = ", ".join(sorted(SELLER_REPORT_SUPPORTED_METRICS))
                raise ValueError(
                    f"Seller reports do not support these metrics yet: {', '.join(unsupported)}. "
                    f"Available seller report metrics: {supported}"
                )
            grant = tenant_context.seller_grant(str(seller_id))
            grant_start = date.fromisoformat(grant.start_date) if grant and grant.start_date else None
            grant_end = date.fromisoformat(grant.end_date) if grant and grant.end_date else None
            if grant and ((grant_start and period.start_date < grant_start) or (grant_end and period.end_date > grant_end)):
                raise ValueError("Requested report period is outside your seller data entitlement")
        else:
            unsupported = [name for name in payload.metric_names if name not in PLATFORM_REPORT_SUPPORTED_METRICS]
            if unsupported:
                raise ValueError(
                    f"Platform reports do not support these metrics yet: {', '.join(unsupported)}"
                )
        chart_output_dir = str(Path(get_settings().report_chart_dir) / tenant_context.tenant_id)
        with activate_tenant_context(tenant_context):
            if scope_type == "seller":
                report = build_seller_scoped_periodic_report(
                    str(seller_id), str(seller_label), period, payload.metric_names,
                    target_lookup=lambda metric_name, report_period: _target_store(tenant_context).get(metric_name, report_period, seller_id),
                    include_charts=payload.include_charts, chart_output_dir=chart_output_dir,
                )
            else:
                report = build_periodic_report(
                    period, payload.metric_names, include_charts=payload.include_charts,
                    target_lookup=lambda metric_name, report_period: _target_store(tenant_context).get(metric_name, report_period),
                    chart_output_dir=chart_output_dir,
                )
        report.risks = report_risks(report)
        content = report_to_markdown(report)
        stored = _store(tenant_context).create_draft(report, content, actor=actor)
        return {"report": report.model_dump(mode="json"), "content": content, "version": stored["version"], "status": stored["status"]}
    except (ValueError, KeyError, PermissionError) as exc:
        # Keep the client response concise, but retain enough non-sensitive
        # request context to diagnose report-generation validation failures.
        print(
            "[REPORT_CREATE_FAILED] "
            f"scope={payload.scope_type} seller={payload.seller_id or '-'} "
            f"period={payload.start_date}~{payload.end_date} "
            f"granularity={payload.granularity} metrics={','.join(payload.metric_names)} "
            f"error={type(exc).__name__}: {exc}"
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/reports/{report_id}/export", response_class=PlainTextResponse)
def export_report(
    report_id: str,
    version: int | None = None,
    tenant_context: TenantContext = Depends(require_tenant_context),
) -> PlainTextResponse:
    """Export only a published, approved report artifact.

    Draft/edited content is intentionally not exportable.  The report store
    remains the approval authority; this endpoint is merely the last policy
    boundary before bytes leave the service.
    """

    try:
        require_permission(tenant_context, "report_export")
        stored = _store(tenant_context).get(report_id, version)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if stored.get("status") != "published":
        raise HTTPException(status_code=409, detail="Only published reports can be exported")
    content = str(stored.get("content") or "")
    return PlainTextResponse(
        content,
        headers={
            "Content-Disposition": f'attachment; filename="{report_id}.md"',
            "X-Result-Policy": "approved-published-artifact",
        },
    )


@router.get("/reports/{report_id}")
def get_report(report_id: str, version: int | None = None, tenant_context: TenantContext = Depends(require_tenant_context)) -> dict[str, object]:
    try:
        return _store(tenant_context).get(report_id, version)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/reports/{report_id}/review")
def review_report(report_id: str, payload: ReportReviewRequest, actor: str = Depends(require_analysis_write), tenant_context: TenantContext = Depends(require_tenant_context)) -> dict[str, object]:
    try:
        require_actor_match(actor, payload.actor)
        approval = _store(tenant_context).review(
            report_id, payload.version, payload.action, payload.actor, payload.reason, payload.edited_content
        )
        return approval.model_dump(mode="json")
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/reports/{report_id}/publish")
def publish_report(report_id: str, payload: ReportPublishRequest, actor: str = Depends(require_analysis_write), tenant_context: TenantContext = Depends(require_tenant_context)) -> dict[str, object]:
    try:
        require_actor_match(actor, payload.actor)
        return _store(tenant_context).publish(report_id, payload.version, payload.actor)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/reports/{report_id}/versions/{report_version}/insights")
def create_report_insights(
    report_id: str,
    report_version: int,
    payload: InsightBriefRequest,
    sync: bool = Query(default=False, description="同步生成（测试/调试用）；生产走异步 generating 占位行"),
    actor: str = Depends(require_analysis_write),
    tenant_context: TenantContext = Depends(require_tenant_context),
) -> dict[str, object]:
    """按指定报告版本生成 AI 洞察。

    默认异步：先落 generating 占位行秒回，后台线程池执行归因补查 + LLM 生成 + Critic，
    完成后 finalize 为 draft / rejected / pending_review；前端轮询 GET 直到离开 generating。
    显式携带 report_version，避免"点生成时 v1、处理期间变 v2"导致的洞察与事实不匹配。
    """
    try:
        require_permission(tenant_context, "report_export")
        store = _store(tenant_context)
        stored = store.get(report_id, report_version)
        if stored.get("status") == "published":
            raise ValueError("已发布报告不能直接生成洞察；请创建新报告版本重新审核发布，或使用发布后 AI 补充说明机制")
        if stored.get("status") not in {"draft", "approved"}:
            raise ValueError("Insights can only be generated for an active report version")
        brief = InsightBrief(
            angle=InsightAngle(payload.angle), audience=InsightAudience(payload.audience),
            style=InsightStyle(payload.style), max_insights=payload.max_insights,
            min_words=payload.min_words, max_words=payload.max_words,
            include_recommendations=payload.include_recommendations,
            focus_metrics=payload.focus_metrics, dimension=payload.dimension,
        )
        if sync:
            report = PeriodicReport.model_validate(_report_payload(stored))
            artifact = generate_insight_artifact(
                report, report_version, brief, actor,
                data_version_manifest_id=stored.get("data_version_manifest_id"),
                data_version_manifest_hash=stored.get("data_version_manifest_hash"),
            )
            # 同输入同哈希：draft 结果幂等复用；rejected 允许重试生成新版本。
            existing = store.latest_insight_by_hash(report_id, report_version, artifact.input_snapshot_hash)
            if existing and existing.get("status") == "draft":
                return existing
            return store.create_insight(artifact)
        # The tenant-backed store persists data-version manifests, while the
        # legacy platform SQLite store does not have those columns yet.
        # Keep the common insight workflow compatible with both stores.
        if isinstance(store, TenantReportStore):
            placeholder = store.create_generating(
                report_id, report_version, brief, actor,
                data_version_manifest_id=stored.get("data_version_manifest_id"),
                data_version_manifest_hash=stored.get("data_version_manifest_hash"),
            )
        else:
            placeholder = store.create_generating(report_id, report_version, brief, actor)
        submit_insight_task(
            store, report_id, report_version, int(placeholder["insight_version"]),
            brief, actor,
            {"tenant_id": tenant_context.tenant_id, "user_id": tenant_context.user_id},
            tenant_context=tenant_context,
        )
        return placeholder
    except InsightUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, PermissionError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/reports/{report_id}/versions/{report_version}/insights")
def list_report_insights(
    report_id: str,
    report_version: int,
    limit: int = Query(default=20, ge=1, le=100),
    tenant_context: TenantContext = Depends(require_tenant_context),
) -> list[dict[str, object]]:
    """已发布报告只返回发布时钉死的洞察版本；草稿/已审核返回最新列表。"""
    store = _store(tenant_context)
    try:
        report = store.get(report_id, report_version)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if report.get("status") == "published":
        pinned = report.get("pinned_insight_version")
        if pinned is None:
            return []
        try:
            return [store.get_insight(report_id, report_version, int(pinned))]
        except KeyError:
            return []
    return store.list_insights(report_id, report_version, limit)


@router.post("/reports/{report_id}/versions/{report_version}/insights/{insight_version}/review")
def review_report_insight(
    report_id: str,
    report_version: int,
    insight_version: int,
    payload: InsightReviewRequest,
    actor: str = Depends(require_analysis_write),
    tenant_context: TenantContext = Depends(require_tenant_context),
) -> dict[str, object]:
    """人工审核洞察：approve / reject / edit（编辑后通过）。

    发布报告后洞察冻结，不能再审核；pin 只选已人工 approved 的洞察。
    """
    try:
        require_actor_match(actor, payload.actor)
        store = _store(tenant_context)
        report = store.get(report_id, report_version)
        if report.get("status") == "published":
            raise ValueError("已发布报告的洞察已冻结，不能审核")
        edited_insights = None
        if payload.action == "edit":
            try:
                edited_insights = [Insight.model_validate(item).model_dump(mode="json") for item in payload.edited_insights or []]
            except Exception as exc:  # noqa: BLE001 - 转成 409 语义
                raise ValueError(f"edited_insights 校验失败：{exc}") from exc
        return store.review_insight(
            report_id, report_version, insight_version, payload.action, payload.actor, payload.reason, edited_insights,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, PermissionError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _sse_line(event: str, **payload: object) -> str:
    """把事件编码为一条 SSE data 行（客户端按行解析 data: 前缀）。"""
    return "data: " + json.dumps({"event": event, **payload}, ensure_ascii=False, default=str) + "\n\n"


@router.get("/reports/{report_id}/versions/{report_version}/insights/{insight_version}/events")
def insight_progress_events(
    report_id: str,
    report_version: int,
    insight_version: int,
    tenant_context: TenantContext = Depends(require_tenant_context),
) -> StreamingResponse:
    """洞察生成进度 SSE：轮询 DB 的 progress 列，实时推送阶段进度与终态。

    事件序列：stage（attribution → llm_generation → critic）→ completed / failed / timeout。
    与 Code Agent 任务事件同模式：服务端不持连接状态，重连也能续上（进度在 DB）。
    """
    store = _store(tenant_context)
    try:
        store.get_insight(report_id, report_version, insight_version)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    _TERMINAL_STATUSES = {"draft", "rejected", "approved", "published", "pending_review"}

    def stream():
        previous_stage: str | None = None
        previous_status: str | None = None
        sequence = 0
        for _ in range(180):  # 3 分钟 @ 1s 轮询
            try:
                row = store.get_insight(report_id, report_version, insight_version)
            except KeyError:
                sequence += 1
                yield _sse_line("failed", detail="Insight not found", sequence_number=sequence)
                return
            status = str(row.get("status") or "")
            progress = row.get("progress")
            stage = progress.get("stage") if isinstance(progress, dict) else None
            if stage and stage != previous_stage:
                sequence += 1
                yield _sse_line("stage", stage=stage, message=progress.get("message", ""), sequence_number=sequence)
                previous_stage = stage
            if status != previous_status:
                sequence += 1
                yield _sse_line("status", insight_version=insight_version, status=status, sequence_number=sequence)
                previous_status = status
            if status in _TERMINAL_STATUSES:
                sequence += 1
                yield _sse_line("completed", insight=row, sequence_number=sequence)
                return
            sequence += 1
            yield _sse_line("heartbeat", sequence_number=sequence)
            time.sleep(1)
        sequence += 1
        yield _sse_line("timeout", detail="Insight generation still running; poll the insights list", sequence_number=sequence)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/events/outbox")
def pending_events(limit: int = 100, tenant_context: TenantContext = Depends(require_tenant_context)) -> list[dict[str, object]]:
    store = _store(tenant_context)
    if isinstance(store, TenantReportStore):
        return store.pending_events(limit)
    return [event.model_dump(mode="json") for event in store.outbox.pending(limit)]

# ============================================================
# 质量仪表盘 HTTP 路由
# 作用：对外暴露「质量仪表盘」的只读查看与快照捕获两个接口。
# 所有接口都要求租户上下文，并用 manage_usage 权限做二次校验——
# 质量证据属于平台级敏感信息，仅授权管理员可读。
# 具体证据聚合逻辑在 app/quality/service.py，本模块只做薄封装。
# ============================================================
from fastapi import APIRouter, Depends, HTTPException

from app.api.auth import require_tenant_context
from app.iam.service import require_permission
from app.quality.service import capture_snapshot, dashboard
from app.tenancy import TenantContext

# 质量相关路由统一挂在 /api/quality 前缀下。
router = APIRouter(prefix="/api/quality", tags=["quality"])


@router.get("")
def get_dashboard(context: TenantContext = Depends(require_tenant_context)) -> dict:
    """读取当前质量仪表盘数据（最新快照 + 最近 20 条历史快照）。

    函数作用：返回质量仪表盘页面所需的全部展示数据。
    输入：context —— 由 require_tenant_context 注入的租户上下文（含 tenant_id/user_id/角色）。
    输出：dict —— {latest: 最新快照, history: 最近的历史快照列表}。
    主要流程：
        1. 先校验权限：无 manage_usage 权限时抛 PermissionError；
        2. 校验通过后调用 service.dashboard() 组装数据并返回；
        3. 权限不足时把 PermissionError 转换为 403 HTTP 响应。
    """
    try:
        # 权限校验：无 manage_usage 权限时抛 PermissionError。
        require_permission(context, "manage_usage")
        # 权限通过后读取质量仪表盘数据。
        return dashboard(context)
    except PermissionError as exc:
        # 把权限异常转成 403 HTTP 响应。
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.post("/snapshots")
def create_snapshot(context: TenantContext = Depends(require_tenant_context)) -> dict:
    """手动触发一次质量快照捕获并持久化到本地 JSON。

    函数作用：把当前时刻的 Golden/性能基准/测试/Trace/用量证据固化为一个
    可审计的快照，便于跨时间对比质量变化。
    输入：context —— 租户上下文（同上）。
    输出：dict —— 刚捕获的完整快照内容。
    主要流程：
        1. 校验 manage_usage 权限；
        2. 调用 service.capture_snapshot() 生成快照并追加写入 snapshots.json；
        3. 权限不足时返回 403。
    """
    try:
        # 权限校验：无 manage_usage 权限时抛 PermissionError。
        require_permission(context, "manage_usage")
        # 权限通过后捕获并持久化快照。
        return capture_snapshot(context)
    except PermissionError as exc:
        # 把权限异常转成 403 HTTP 响应。
        raise HTTPException(status_code=403, detail=str(exc)) from exc

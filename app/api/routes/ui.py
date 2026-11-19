# ============================================================
# 前端托管路由模块
# 作用：本模块负责把前端 Vue 应用（已构建的 web/dist 产物）通过
# FastAPI 提供给浏览器访问。项目不单独启动 Nginx，而是由后端直接
# 托管静态资源，因此需要这两个路由来完成“首页入口 + SPA 回退”。
# ============================================================
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

# 声明一组 / 前缀下的 UI 相关路由（仅用于 OpenAPI 标签分组，无实际前缀）。
router = APIRouter(tags=["ui"])

# 定位前端构建产物目录：本文件位于 app/api/routes/，向上 3 级即项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
# 前端构建产物根目录（npm run build 的输出位置）。
SPA_ROOT = PROJECT_ROOT / "web" / "dist"
# SPA 唯一入口 HTML 文件，Vue 打包后生成的 index.html。
SPA_ENTRY = SPA_ROOT / "index.html"


@router.get("/", include_in_schema=False, response_model=None)
def index() -> FileResponse | JSONResponse:
    """托管前端首页（构建产物的 index.html）。

    函数作用：处理对站点根路径 "/" 的请求，返回 Vue 应用入口页。
    输入：无（仅 GET 请求，无路径参数）。
    输出：FileResponse —— 前端 index.html；或 JSONResponse —— 503 提示未构建。
    主要流程：
        1. 检查 SPA_ENTRY（web/dist/index.html）是否已存在（是否执行过 npm run build）；
        2. 若存在，直接返回该文件作为响应；
        3. 若不存在，返回 503 状态码，提示先在 web/ 目录执行构建命令。
    """
    # 已构建过前端（存在 index.html）：直接返回入口页。
    if SPA_ENTRY.exists():
        return FileResponse(str(SPA_ENTRY))
    # 未构建：返回 503 提示，避免浏览器拿到空页面。
    return JSONResponse(
        status_code=503,
        content={"detail": "Frontend build not found. Run: cd web && npm run build"},
    )


@router.get("/{spa_path:path}", include_in_schema=False, response_model=None)
def spa_fallback(spa_path: str) -> FileResponse | JSONResponse:
    """SPA 路由回退：让 Vue Router 的浏览器刷新/深链接不吞掉后端 API。

    函数作用：Vue 是单页应用，路由由前端 router 接管。当用户直接刷新
    /reports、/explorer 等前端路由时，后端没有对应路径，必须回退到
    index.html，由前端 router 自行渲染。同时要保证真正的后端 API 前缀
    不受影响，否则会误把 /api/... 之类的请求也回退成 index.html。
    输入：spa_path —— URL 中 "/" 之后的完整路径字符串。
    输出：FileResponse —— 回退到前端入口页；HTTPException —— 404（若命中 API 前缀）。
    主要流程：
        1. 判断 spa_path 是否以后端 API 前缀开头（api/chat/analysis/traces 等）；
        2. 若是 API 前缀，抛出 404，避免前端入口把后端接口吞掉；
        3. 否则调用 index() 返回 index.html，交给前端路由处理该深链接。
    """
    # 命中后端 API 前缀：抛 404，防止 SPA 入口吞掉真实接口。
    # GET /chat and GET /knowledge are browser routes. Their API operations use
    # POST /chat and /knowledge/* respectively, so allow only these exact SPA
    # roots while keeping nested API paths protected from the HTML fallback.
    if spa_path in {"chat", "knowledge"}:
        return index()
    if spa_path.startswith(("api", "chat/", "analysis", "data-explorer", "causal-experiments", "knowledge/", "traces", "health", "metrics", "docs", "openapi")):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not found")
    # 其余路径交给 index() 返回 index.html，由前端路由处理深链接。
    return index()

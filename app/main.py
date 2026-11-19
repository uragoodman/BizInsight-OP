# ============================================================
# FastAPI 应用入口
# 作用：组装整个后端应用——注册 HTTP 追踪中间件、挂载前端静态资源、
#       注册全部业务 API 路由，并生成供 uvicorn 启动的 app 实例。
# 启动方式：python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 18080
# ============================================================
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes.chat import router as chat_router
from app.api.routes.chat_history import router as chat_history_router
from app.api.routes.analysis import router as analysis_router
from app.api.routes.eval import router as eval_router
from app.api.routes.explorer import router as explorer_router
from app.api.routes.health import router as health_router
from app.api.routes.knowledge import router as knowledge_router
from app.api.routes.metrics import router as metrics_router
from app.api.routes.memory import router as memory_router
from app.api.routes.complex_analysis import router as complex_analysis_router
from app.api.routes.code_analysis import router as code_analysis_router
from app.api.routes.auth import router as auth_router
from app.api.routes.traces import router as traces_router
from app.api.routes.ui import router as ui_router
from app.api.routes.quality import router as quality_router
from app.api.routes.data_versions import router as data_versions_router
from app.api.routes.rag_releases import router as rag_releases_router
from app.api.routes.evaluations import router as evaluations_router
from app.api.routes.causal_experiments import router as causal_experiments_router
from app.core.config import get_settings
from app.observability.middleware import trace_request
from pathlib import Path


# 前端构建产物目录（Vite 输出），仅用于挂载 /assets 静态资源。
SPA_ROOT = Path(__file__).resolve().parents[1] / "web" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期钩子：启动时加载并校验指标注册表。

    函数作用：在应用启动阶段预加载全部指标定义并做合法性校验，
    让配置错误尽早暴露，而不是等到首次查询才失败。
    输入：app —— FastAPI 实例（生命周期钩子签名要求，此处未直接使用）。
    输出：无（yield 后进入正常运行期）。
    主要流程：
        1. 从 metrics.registry 获取全局指标注册表单例；
        2. load() 加载全部指标定义；
        3. validate() 校验定义合法性——异常会在启动阶段直接抛出。
    """
    from app.metrics.registry import get_registry

    registry = get_registry()
    # 加载全部指标定义。
    registry.load()
    # 校验定义合法性，启动阶段即暴露配置错误。
    registry.validate()
    from app.agents.data_query_agent import validate_value_tool_mapping
    validate_value_tool_mapping()
    from app.data_versions.service import validate_gmv_dependency_contract
    validate_gmv_dependency_contract()
    from app.rag.flags import bootstrap as bootstrap_rag_flags, shutdown as shutdown_rag_flags
    bootstrap_rag_flags()
    try:
        yield
    finally:
        shutdown_rag_flags()


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用实例。

    函数作用：把中间件、静态资源与全部业务路由组装成可启动的应用。
    输入：无。
    输出：FastAPI —— 配置完成的 app 实例。
    主要流程：
        1. 读取设置，创建 FastAPI 应用（标题/版本/生命周期钩子）；
        2. 挂载 HTTP 追踪中间件 trace_request（OpenTelemetry/指标采集）；
        3. 若前端 /assets 目录存在则挂载静态资源（Vite 构建产物）；
        4. 依次注册健康检查、聊天、分析、认证、质量仪表盘等全部业务路由；
        5. 最后注册 SPA 回退路由——其 catch-all 路径必须排在最后，否则会遮蔽 API。
    """
    settings = get_settings()
    # 创建应用实例：设置标题/版本/生命周期钩子。
    app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
    # 挂载 HTTP 追踪中间件（OpenTelemetry/指标采集）。
    app.middleware("http")(trace_request)
    # Vite 会把哈希后的 JS/CSS 产物放在 /assets 下。
    assets = SPA_ROOT / "assets"
    # 存在构建产物时挂载静态资源目录。
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="spa-assets")
    # 注册全部业务路由（按依赖顺序排列）。
    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(chat_history_router)
    app.include_router(analysis_router)
    app.include_router(traces_router)
    app.include_router(eval_router)
    app.include_router(explorer_router)
    app.include_router(knowledge_router)
    app.include_router(metrics_router)
    app.include_router(memory_router)
    app.include_router(complex_analysis_router)
    app.include_router(code_analysis_router)
    app.include_router(auth_router)
    app.include_router(quality_router)
    app.include_router(data_versions_router)
    app.include_router(rag_releases_router)
    app.include_router(evaluations_router)
    app.include_router(causal_experiments_router)
    # 最后注册 SPA 回退路由，避免其 catch-all 路径遮蔽 API。
    app.include_router(ui_router)
    return app


app = create_app()

"""数据库连接工厂。

该文件负责集中创建数据访问层使用的核心连接对象，避免各模块重复初始化：

- `engine` 用于执行 SQL 和表读写。
- `SessionLocal` 用于需要 ORM Session 的场景。

处理能力主要是根据项目配置生成稳定的 SQLAlchemy 连接入口。
"""

import re

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings


settings = get_settings()
"""当前运行环境对应的配置对象。"""


def _rls_role_statement(role_name: str) -> str:
    """Build SET ROLE SQL only from a validated PostgreSQL identifier."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}", role_name):
        raise ValueError("DB_RLS_ROLE must be a simple PostgreSQL role identifier")
    return f'SET ROLE "{role_name}"'

engine = create_engine(
    settings.postgres_url,
    future=True,
    pool_pre_ping=True,
    connect_args={"connect_timeout": int(settings.database_timeout_seconds)},
)
"""全局 SQLAlchemy Engine，用于仓储层和初始化流程访问数据库。"""

if settings.db_use_rls_role:
    # superuser 连接会绕过 RLS，因此每次从连接池取用连接（含复用）都显式切换到
    # 非 superuser 角色，使行级安全策略真正按 tenant 过滤。connect 事件只在新
    # 连接时触发，池内复用的连接会丢失角色，所以挂在 checkout 上。角色由
    # scripts/setup_rls_role.py 创建并授权；关闭开关即回到直连。
    @event.listens_for(engine, "checkout")
    def _set_rls_role(dbapi_connection, connection_record, connection_proxy) -> None:  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute(_rls_role_statement(settings.db_rls_role))
        finally:
            cursor.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
"""全局 Session 工厂，用于按需创建数据库会话对象。"""

"""Dedicated database engine for the Phase 11 trust boundary.

The normal application engine may use a migration/owner login in development.
Code Analysis can instead be pointed at a least-privilege API or worker login
without forcing the rest of the application to change connection roles.
"""

from sqlalchemy import create_engine

from app.core.config import get_settings


settings = get_settings()
engine = create_engine(
    settings.code_analysis_database_url or settings.postgres_url,
    future=True,
    pool_pre_ping=True,
    connect_args={"connect_timeout": int(settings.database_timeout_seconds)},
)

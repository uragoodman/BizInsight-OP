"""Execution guards shared by every whitelist SQL template."""

from sqlalchemy import text

TEMPLATE_STATEMENT_TIMEOUT_MS = 30_000
TEMPLATE_LOCK_TIMEOUT_MS = 2_000


def apply_statement_timeout(conn) -> None:
    """Limit the current PostgreSQL transaction without changing global DB settings."""
    conn.execute(text("SELECT set_config('statement_timeout', :timeout, true)"), {
        "timeout": str(TEMPLATE_STATEMENT_TIMEOUT_MS),
    })
    conn.execute(text("SELECT set_config('lock_timeout', :timeout, true)"), {
        "timeout": str(TEMPLATE_LOCK_TIMEOUT_MS),
    })

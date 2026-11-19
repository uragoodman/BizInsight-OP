"""Shared helpers for tests that replace repository/database engines."""
from __future__ import annotations

import importlib
from typing import Any


_ENGINE_MODULES = (
    "app.data.database",
    "app.data.repositories.daily_metrics",
    "app.data.repositories.explorer",
    "app.data.repositories.seller_extensions",
    "app.core.time_expression",
)


def bind_test_engine(monkeypatch: Any, engine: Any) -> None:
    """Bind one test engine across all modules that cache the shared engine.

    The real repositories import ``app.data.database.engine`` into module
    globals. Tests that substitute an engine must replace all those aliases
    together; patching only one repository leaves time bounds or SQL paths on
    the developer/default connection.
    """
    # Import first while the canonical engine still points at its original
    # object; otherwise a later import would cache an already-patched alias
    # that monkeypatch cannot restore.
    modules = [importlib.import_module(module_name) for module_name in _ENGINE_MODULES]
    for module in modules:
        monkeypatch.setattr(module, "engine", engine)

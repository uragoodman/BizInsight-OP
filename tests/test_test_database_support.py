from __future__ import annotations

import importlib

import pytest

from test_support.database import bind_test_engine


def test_bind_test_engine_replaces_every_cached_repository_alias() -> None:
    replacement = object()
    module_names = (
        "app.data.database",
        "app.data.repositories.daily_metrics",
        "app.data.repositories.explorer",
        "app.data.repositories.seller_extensions",
        "app.core.time_expression",
    )
    modules = [importlib.import_module(name) for name in module_names]
    originals = [module.engine for module in modules]
    with pytest.MonkeyPatch.context() as monkeypatch:
        bind_test_engine(monkeypatch, replacement)
        assert all(module.engine is replacement for module in modules)

    assert all(module.engine is original for module, original in zip(modules, originals))

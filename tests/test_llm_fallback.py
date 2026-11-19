"""P1.2 多提供商回退测试：可恢复错误切备用、不可恢复不切换、账本记录 provider/fallback。"""

from __future__ import annotations

from types import SimpleNamespace

from app.llm import client as llm_client


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
        self.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)


class _PrimaryClient:
    def __init__(self, exc: Exception) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(exc)))


class _BackupClient:
    def __init__(self) -> None:
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        return _FakeResponse("backup-ok")


def _install(monkeypatch, primary_exc, with_backup=True):
    backup = _BackupClient() if with_backup else None
    providers = [llm_client._Provider("primary", _PrimaryClient(primary_exc), lambda op, ex, q: ("primary-model", "standard"))]
    if backup is not None:
        providers.append(llm_client._Provider("backup", backup, lambda op, ex, q: ("backup-model", "standard")))
    monkeypatch.setattr(llm_client, "_providers", lambda: providers)
    # 只有 RuntimeError 算"可恢复"；ValueError（业务错误）不算。
    monkeypatch.setattr(llm_client, "_RECOVERABLE_ERRORS", (RuntimeError,))
    monkeypatch.setattr(llm_client, "_quota_available", lambda: True)
    monkeypatch.setattr(llm_client, "execute_with_resilience", lambda fn, *a, **k: fn())
    return backup


def test_llm_chat_falls_back_on_recoverable_error(monkeypatch) -> None:
    backup = _install(monkeypatch, RuntimeError("primary down"))
    captured: dict = {}
    def fake_record_usage(response, model, operation, profile="standard", provider_id=None, fallback_from=None, *, call_record=None):
        captured.update(model=model, provider_id=provider_id, fallback_from=fallback_from)
    monkeypatch.setattr(llm_client, "_record_usage", fake_record_usage)

    reply = llm_client.llm_chat("sys", "hi")
    assert reply == "backup-ok"
    assert backup.calls == 1  # 备用 provider 被调用一次
    # 账本记录：命中的 provider 是 backup，回退源是 primary
    assert captured["provider_id"] == "backup"
    assert captured["fallback_from"] == "primary"
    assert captured["model"] == "backup-model"


def test_llm_chat_non_recoverable_error_does_not_fallback(monkeypatch) -> None:
    backup = _install(monkeypatch, ValueError("bad request"))
    monkeypatch.setattr(llm_client, "_record_usage", lambda *a, **k: None)

    reply = llm_client.llm_chat("sys", "hi")
    assert reply == ""  # 不可恢复错误 → 直接失败，不切换
    assert backup.calls == 0  # 备用 provider 未被调用


def test_llm_extract_falls_back_on_recoverable_error(monkeypatch) -> None:
    backup = _install(monkeypatch, RuntimeError("primary down"))
    monkeypatch.setattr(llm_client, "_record_usage", lambda *a, **k: None)

    llm_client.llm_extract("sys", "{}")
    assert backup.calls == 1  # 可恢复故障 → 备用 provider 被调用


def test_fallback_source_index() -> None:
    assert llm_client._fallback_source(0) is None
    assert llm_client._fallback_source(1) == "primary"
    assert llm_client._fallback_source(2) == "primary"

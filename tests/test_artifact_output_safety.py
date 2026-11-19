from pathlib import Path

import pytest

from _artifact_safety import isolated_output_path


def test_optional_test_output_override_is_limited_to_system_temp(monkeypatch, tmp_path):
    from tempfile import gettempdir

    safe = Path(gettempdir()) / "bizinsight-test-output" / "result.json"
    monkeypatch.setenv("PLATFORM_COMPARE_RESULT_PATH", str(safe))
    assert isolated_output_path("PLATFORM_COMPARE_RESULT_PATH") == safe.resolve()

    repo_history = Path(__file__).resolve().parents[1] / "docs" / "testing" / "old.json"
    monkeypatch.setenv("PLATFORM_COMPARE_RESULT_PATH", str(repo_history))
    with pytest.raises(RuntimeError, match="system temporary directory"):
        isolated_output_path("PLATFORM_COMPARE_RESULT_PATH")

    monkeypatch.delenv("PLATFORM_COMPARE_RESULT_PATH")
    assert isolated_output_path("PLATFORM_COMPARE_RESULT_PATH") is None

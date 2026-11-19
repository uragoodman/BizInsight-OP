from __future__ import annotations

from pathlib import Path

import pytest

from scripts.backup_postgres import create_backup
from scripts.restore_postgres import restore_backup


def test_postgres_backup_dry_run_does_not_execute_or_expose_url(tmp_path: Path) -> None:
    result = create_backup(
        database_url="postgresql+psycopg://user:secret@localhost/db",
        output=tmp_path / "db.dump",
        dry_run=True,
    )
    assert result["status"] == "dry_run"
    assert "secret" not in str(result)


def test_postgres_restore_requires_isolated_database(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "db.dump"
    source.write_bytes(b"backup")
    monkeypatch.setattr(
        "scripts.restore_postgres.verify_backup",
        lambda _path: {"valid": True, "sha256": "a" * 64},
    )
    with pytest.raises(ValueError, match="isolated"):
        restore_backup(
            input_path=source,
            database_url="postgresql://user:secret@localhost/bizinsight",
            dry_run=True,
        )


def test_postgres_restore_dry_run_redacts_url(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "db.dump"
    source.write_bytes(b"backup")
    monkeypatch.setattr(
        "scripts.restore_postgres.verify_backup",
        lambda _path: {"valid": True, "sha256": "a" * 64},
    )
    result = restore_backup(
        input_path=source,
        database_url="postgresql://user:secret@localhost/bizinsight_restore",
        dry_run=True,
    )
    assert result["status"] == "dry_run"
    assert "secret" not in str(result)

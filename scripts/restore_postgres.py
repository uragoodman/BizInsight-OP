"""Restore a PostgreSQL logical backup into an explicitly isolated database.

The helper refuses production-looking database names and never adds
``--clean``/``--create`` to pg_restore.  The target database must already be
created by the operator as an isolated restore target.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from scripts.backup_postgres import verify_backup


def _target_database_name(database_url: str) -> str:
    parsed = urlparse(database_url.replace("postgresql+psycopg://", "postgresql://", 1))
    return (parsed.path or "").lstrip("/").strip().lower()


def restore_backup(
    *, input_path: str | Path, database_url: str, dry_run: bool = False,
) -> dict:
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    database = _target_database_name(database_url)
    if not database or not (database.startswith("restore_") or database.endswith("_restore")):
        raise ValueError(
            "PostgreSQL restore requires an isolated target database named restore_* or *_restore"
        )
    verification = verify_backup(source)
    command = [
        "pg_restore", "--exit-on-error", "--no-owner", "--no-privileges",
        "--dbname", database_url.replace("postgresql+psycopg://", "postgresql://", 1),
        str(source),
    ]
    if dry_run:
        return {
            "status": "dry_run",
            "target_database": database,
            "source_sha256": verification["sha256"],
            "command": [*command[:-2], "<redacted>", str(source)],
        }
    subprocess.run(command, check=True)
    return {
        "status": "restored",
        "target_database": database,
        "source_sha256": verification["sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="恢复 PostgreSQL 备份到隔离数据库")
    parser.add_argument("--input", required=True)
    parser.add_argument("--database-url", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = restore_backup(
        input_path=args.input,
        database_url=args.database_url or get_settings().postgres_url,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

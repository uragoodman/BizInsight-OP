"""Safe PostgreSQL logical backup helper for RAG release preparation.

The command never drops databases, volumes or schemas. It refuses to overwrite
an existing archive and records only non-secret release metadata beside it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings


def _normalized_url(value: str) -> str:
    return value.replace("postgresql+psycopg://", "postgresql://", 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _alembic_revision() -> str | None:
    try:
        result = subprocess.run(
            ["alembic", "current"], check=True,
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def create_backup(*, database_url: str, output: str | Path, dry_run: bool = False, release_id: str | None = None) -> dict:
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"backup output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "pg_dump", "--format=custom", "--no-owner", "--no-privileges",
        "--file", str(target), "--dbname", _normalized_url(database_url),
    ]
    if dry_run:
        return {"status": "dry_run", "output": str(target), "command": [*command[:-1], "<redacted>"]}
    subprocess.run(command, check=True)
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(target),
        "sha256": _sha256(target),
        "format": "custom",
        "includes": ["schema", "data"],
        "release_id": release_id,
        "git_commit": _git_commit(),
        "alembic_revision": _alembic_revision(),
    }
    metadata_path = target.with_suffix(target.suffix + ".manifest.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def verify_backup(path: str | Path) -> dict:
    source = Path(path)
    subprocess.run(["pg_restore", "--list", str(source)], check=True, capture_output=True, text=True)
    return {"valid": True, "path": str(source), "sha256": _sha256(source)}


def main() -> None:
    parser = argparse.ArgumentParser(description="PostgreSQL schema+data logical backup")
    parser.add_argument("--output", required=True)
    parser.add_argument("--database-url", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--release-id", default="")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_backup(args.output), ensure_ascii=False))
        return
    result = create_backup(
        database_url=args.database_url or get_settings().postgres_url,
        output=args.output,
        dry_run=args.dry_run,
        release_id=args.release_id or None,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

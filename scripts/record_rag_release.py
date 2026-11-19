"""Record a no-secret release manifest for an existing Qdrant collection.

This command is read-only against Qdrant. It does not create snapshots,
delete collections, create collections or upsert points.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.rag.release_manifest import build_release_manifest, write_release_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="记录 RAG 发布前 manifest")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--snapshot-ref", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backup-ref", default="")
    parser.add_argument("--backup-sha256", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from qdrant_client import QdrantClient

    client = QdrantClient(url=get_settings().qdrant_url)
    try:
        manifest = build_release_manifest(
            collection_name=args.collection,
            snapshot_ref=args.snapshot_ref,
            client=client,
            backup_ref=args.backup_ref or None,
            backup_sha256=args.backup_sha256 or None,
        )
    finally:
        client.close()

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    target = write_release_manifest(args.output, manifest)
    print(json.dumps({"status": "ok", "output": str(target), "release_id": manifest["release_id"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

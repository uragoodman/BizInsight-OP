"""Restore a Qdrant export into a new, non-existing collection only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.rag.backup import restore_export


def main() -> None:
    parser = argparse.ArgumentParser(description="将 Qdrant 导出恢复到隔离 collection")
    parser.add_argument("--input", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--url", default="")
    args = parser.parse_args()
    from qdrant_client import QdrantClient

    client = QdrantClient(url=args.url or get_settings().qdrant_url)
    try:
        result = restore_export(
            client,
            input_path=args.input,
            target_collection=args.collection,
        )
    finally:
        client.close()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

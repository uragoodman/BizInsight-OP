"""Export a complete Qdrant collection to an external JSONL(.gz) backup."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.rag.backup import export_collection, verify_export


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 Qdrant collection 备份")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        print(json.dumps(verify_export(args.output), ensure_ascii=False))
        return

    from qdrant_client import QdrantClient

    client = QdrantClient(url=get_settings().qdrant_url)
    try:
        result = export_collection(
            client,
            collection_name=args.collection,
            output_path=args.output,
            batch_size=args.batch_size,
        )
    finally:
        client.close()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Build the independent Qdrant metric semantic catalog.

Run this after Qdrant and the configured embedding model are available.  The
query path never creates the collection on demand: an absent or stale catalog
fails closed instead of silently falling back to fuzzy string replacement.
"""

from __future__ import annotations

import argparse
import json

from app.metrics.semantic_catalog import index_metric_semantic_catalog


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("platform", "seller"), default="platform")
    args = parser.parse_args()
    print(json.dumps(index_metric_semantic_catalog(scope=args.scope), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

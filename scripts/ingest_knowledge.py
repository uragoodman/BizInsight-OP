"""知识库入库脚本。

本脚本是 RAG 数据库操作的命令行入口，用于把 `docs/knowledge` 下的 markdown
文档切分、向量化并写入 Qdrant。

使用场景：
- 首次启动 Qdrant 后，执行
  `python scripts/ingest_knowledge.py --collection bizinsight_knowledge_dense_v2`
  建立一个显式版本化 collection。
- 修改指标定义、SOP 或报告模板后，重新执行该脚本刷新索引。
- 需要 native dense+sparse multimodal collection 时执行
  `python scripts/ingest_knowledge.py --native --collection bizinsight_knowledge_hybrid_v2`。

调用链路：
- `discover_knowledge_files` 负责发现 markdown 文件。
- `build_qdrant_index` 负责切 chunk、生成 embedding、写入 Qdrant（LlamaIndex dense-only）。
- `build_qdrant_index_native` 负责构建同一 collection 的 dense+sparse 原生索引。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.indexing import (
    build_qdrant_index,
    build_qdrant_index_native,
    discover_knowledge_files,
    switch_collection_alias,
)


def main() -> None:
    """执行知识库索引构建。

    参数：通过命令行显式指定目标 collection；`--native` 选择原生 dense+sparse 构建。
    返回值：`None`。执行结果会打印到终端。

    具体操作：
    1. 统计当前知识库 markdown 文件数量。
    2. 调用 `build_qdrant_index` 或 `build_qdrant_index_native` 写入 Qdrant。
    3. 打印索引构建结果，包括 collection、chunk 数量和状态。
    """

    parser = argparse.ArgumentParser(description="知识库入库脚本")
    parser.add_argument(
        "--native", action="store_true",
        help="构建 native dense+sparse collection（默认使用 LlamaIndex dense-only 构建）",
    )
    parser.add_argument(
        "--publish-alias",
        help="dense index alias to switch atomically after a successful validated build",
    )
    parser.add_argument(
        "--collection", required=True,
        help="显式指定目标 collection；新索引必须使用 versioned collection",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只检查目标 collection 和建库计划，不执行任何写入或删除",
    )
    parser.add_argument(
        "--allow-destructive-rebuild", action="store_true",
        help="显式允许删除已有 collection；必须同时提供 snapshot 和 release manifest",
    )
    parser.add_argument(
        "--snapshot-ref",
        help="破坏性重建前已完成的 Qdrant snapshot 标识",
    )
    parser.add_argument(
        "--release-manifest",
        help="与 snapshot 绑定的 release manifest 路径",
    )
    args = parser.parse_args()

    if args.allow_destructive_rebuild:
        if not args.snapshot_ref or not args.release_manifest:
            parser.error(
                "--allow-destructive-rebuild requires --snapshot-ref and --release-manifest"
            )
        if not Path(args.release_manifest).is_file():
            parser.error(f"release manifest does not exist: {args.release_manifest}")

    files = discover_knowledge_files()
    print(f"Discovered {len(files)} knowledge files.")
    common = {
        "collection_name": args.collection,
        "dry_run": args.dry_run,
        "allow_destructive_rebuild": args.allow_destructive_rebuild,
        "snapshot_ref": args.snapshot_ref,
        "release_manifest_path": args.release_manifest,
    }
    if args.native:
        result = build_qdrant_index_native(**common)
    else:
        result = build_qdrant_index(**common)
    if args.publish_alias:
        if args.native:
            parser.error("--publish-alias requires a dense collection")
        if result.get("status") not in {"ok", "skipped"} or args.dry_run:
            parser.error("--publish-alias requires a successful non-dry-run build")
        from qdrant_client import QdrantClient
        from app.core.config import get_settings
        client = QdrantClient(url=get_settings().qdrant_url)
        try:
            result["publication"] = switch_collection_alias(
                client, args.publish_alias, args.collection,
            )
        finally:
            client.close()
    print(result)


if __name__ == "__main__":
    main()
